# IMPORTS
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.checkpoint import checkpoint
import math




@dataclass
class ModelConfig:
    """
    Model configuration settings for the model.

    ### General Settings
    - vocab_size (int): Total number of tokens in the vocabulary.
    - hidden_size (int): The main embedding dimension used across all layers.
    - num_layers (int): The number of sequential Transformer blocks.
    - initializer_range (float): Scale for weight initialization to ensure stable gradients.
    - tie_word_embeddings (bool): Whether to reuse embedding weights for the final output projection.
    - max_seq_len (int): The maximum context window size for the model.
    - max_batch_size (int): The maximum number of sequences to process at once.

    ### Normalization
    - rms_norm_eps (float): A tiny value added during RMSNorm to prevent division by zero.

    ### RoPE & YaRN (Positional Encoding)
    - rope_theta (float): The base constant for rotary frequency calculations.
    - rope_type (str): The method for context scaling (e.g., 'yarn' for long-context support).
    - beta_slow (float): Controls the low-frequency boundary for YaRN interpolation.
    - beta_fast (float): Controls the high-frequency boundary for YaRN interpolation.
    - factor (float): The scaling factor for stretching the context window (e.g., 2.0 doubles the length).
    - mscale (Optional[int]): Scaling factor for attention logits to prevent entropy collapse in long sequences.
    - original_max_seq_len (int): The starting context length the model was originally 
        trained on (e.g., 4096) before applying context extension techniques like YaRN.

    ### MLA (Multi-Head Latent Attention)
    - num_attention_heads (int): Number of parallel attention heads.
    - kv_lora_rank (int): The compressed dimension for KV vectors, used to minimize KV-cache memory usage.
    - qk_nope_dim (int): The dimension of the query/key vectors that remain "Non-Positionally Encoded."
    - qk_rope_dim (int): The dimension of the query/key vectors where RoPE is applied.

    ### MoE (Mixture of Experts)
    - num_experts (int): The total pool of expert sub-networks in each layer.
    - num_experts_per_token (int): The "Top-K" value; how many experts are active for a single token.
    - moe_intermediate_size (int): The hidden expansion dimension within each individual expert's FFN.
    """
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 14
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    max_seq_len: int = 512
    max_batch_size: int = 1
    
    # RMSNorm
    rms_norm_eps: float = 1e-6
    
    # RoPE & YaRN 
    rope_theta: float = 10000.0
    rope_type: str = "default"
    beta_slow: float = 1.0
    beta_fast: float = 32.0
    factor: float = 1.0
    mscale: Optional[float] = None
    original_max_seq_len: int = 512

    # MLA
    num_attention_heads: int = 8
    kv_lora_rank: int = 128
    qk_nope_dim: int = 64
    qk_rope_dim: int = 64

    # MoE
    num_experts: int = 16
    num_experts_per_token: int = 2
    moe_intermediate_size: int = 1024





class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm scales the input by the inverse of the root mean square of its elements,
    providing training stability similar to LayerNorm but with lower computational overhead
    as it omits the mean-centering (re-centering) step.

    Args:
        config (ModelConfig): Configuration object containing 'hidden_size' (dim) 
                             and 'rms_norm_eps' (epsilon).

    Attributes:
        weight (nn.Parameter): Learnable scaling parameter (gamma) initialized to 1.0.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) with YaRN scaling support.

    Args:
        config (ModelConfig): Config containing rope dimensions, theta, 
            and scaling factors for long-context extension.

    Returns:
        torch.Tensor: The input tensor with rotary position embeddings applied.
    """
    def __init__(self, config):
        super().__init__()

        self.dim = config.qk_rope_dim
        self.max_seq_len = getattr(config, "max_seq_len", 1024)
        self.theta = getattr(config, "rope_theta", 10000.0)

        self.rope_type = getattr(config, "rope_type", "default") 

  
        self.factor = float(getattr(config, "factor", 1.0))
        self.original_max_seq_len = int(
            getattr(config, "original_max_seq_len", self.max_seq_len)
        )
        self.beta_slow = float(getattr(config, "beta_slow", 1.0))
        self.beta_fast = float(getattr(config, "beta_fast", 32.0))


        self.attention_factor = getattr(config, "mscale", None)

        if self.dim % 2 != 0:
            raise ValueError("qk_rope_dim must be even")

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        if self.rope_type == "yarn":
            inv_freq = self._build_yarn_inv_freq(inv_freq)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer(
            "freqs_cis",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False,
        )

    def _build_yarn_inv_freq(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """
        Computes YaRN-scaled frequencies for context window extension.

        Args:
            inv_freq (torch.Tensor): Original inverse frequencies.

        Returns:
            torch.Tensor: Scaled frequencies based on the YaRN interpolation method.
        """
        
        s = max(self.factor, 1.0)

        if s == 1.0:
            return inv_freq

        wavelengths = 2.0 * math.pi / inv_freq
        r = self.original_max_seq_len / wavelengths

        ramp = ((r - self.beta_slow) / (self.beta_fast - self.beta_slow)).clamp(0.0, 1.0)

        inv_freq_scaled = inv_freq / s
        inv_freq_yarn = (1.0 - ramp) * inv_freq_scaled + ramp * inv_freq
        return inv_freq_yarn

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        squeeze_head = False

        if x.dim() == 3:
            x = x.unsqueeze(2)  
            squeeze_head = True
        elif x.dim() != 4:
            raise ValueError("RoPE expects x with shape [B, T, D] or [B, T, H, D]")

        dtype = x.dtype
        device = x.device
        bsz, seq_len, n_heads, dim = x.shape

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. "
                "Increase max_seq_len or extend the cache."
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        else:
            position_ids = torch.as_tensor(position_ids, device=device, dtype=torch.long)

        if position_ids.dim() == 0:
            position_ids = position_ids.view(1)

        if position_ids.dim() == 1:
            freqs = self.freqs_cis.index_select(0, position_ids).unsqueeze(0).unsqueeze(2)
        elif position_ids.dim() == 2:
            if position_ids.shape != (bsz, seq_len):
                raise ValueError("position_ids shape must match [B, T]")
            freqs = self.freqs_cis[position_ids].unsqueeze(2)
        else:
            raise ValueError("position_ids must have shape [T] or [B, T]")

        x_complex = torch.view_as_complex(
            x.float().reshape(bsz, seq_len, n_heads, dim // 2, 2)
        )
        y = x_complex * freqs
        y = torch.view_as_real(y).flatten(-2).to(dtype)

        if squeeze_head:
            y = y.squeeze(2)

        return y
class MLA(nn.Module):
    """
    Implements Multi-Head Latent Attention (MLA) with Low-Rank Compression.

    This module reduces the KV cache footprint by projecting hidden states into a 
    latent space before up-projecting into query, key, and value components. It 
    uses a split-stream approach for Rotary Positional Embeddings (RoPE).

    Args:
        config (ModelConfig): Configuration object containing 'hidden_size', 
                             'kv_lora_rank', 'qk_nope_dim', and 'qk_rope_dim'.

    Returns:
        torch.Tensor: The attention output tensor of shape [batch, seq_len, hidden_size].
    """


    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.attention_factor = config.mscale
        self.max_seq_len = config.max_seq_len
        self.original_max_seq_len = config.original_max_seq_len
        self.factor=config.factor

        self.head_dim = self.qk_nope_dim + self.qk_rope_dim

        self.w_in = nn.Linear(self.hidden_size, 2 * self.kv_lora_rank, bias=False)


        self.w_uq_qr = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_dim + self.num_heads * self.qk_rope_dim,
            bias=False,
        )


        self.w_uk = nn.Parameter(
            torch.empty(self.num_heads, self.qk_nope_dim, self.kv_lora_rank)
        )


        self.w_uv = nn.Parameter(
            torch.empty(self.num_heads, self.kv_lora_rank, self.head_dim)
        )

        self.w_kr = nn.Linear(self.hidden_size, self.qk_rope_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = RoPE(config)

        base_scale = (self.qk_nope_dim + self.qk_rope_dim) ** -0.5

        use_scaled_attention = self.max_seq_len > self.original_max_seq_len and self.attention_factor is not None
        self.softmax_scale = base_scale * self.attention_factor if use_scaled_attention else base_scale

        nn.init.xavier_uniform_(self.w_uk)
        nn.init.xavier_uniform_(self.w_uv)


    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        c_in = self.w_in(x)
        c_kv, c_q = c_in.split(self.kv_lora_rank, dim=-1)


        q_proj = self.w_uq_qr(c_q).reshape(
            bsz, seq_len, self.num_heads, self.qk_nope_dim + self.qk_rope_dim
        )
        q_nope, q_rope = q_proj.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_rope = self.rope(q_rope, position_ids)


        q_lat = torch.einsum("bthd,hdr->bthr", q_nope, self.w_uk)


        k_rope = self.w_kr(x).unsqueeze(2)  # [B, T, 1, rope_dim]
        k_rope = self.rope(k_rope, position_ids)
        k_rope = k_rope.expand(-1, -1, self.num_heads, -1)

        k_lat = c_kv.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        v = c_kv.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        q = torch.cat([q_lat, q_rope], dim=-1).transpose(1, 2)  # [B, H, T, r + rope]
        k = torch.cat([k_lat, k_rope], dim=-1).transpose(1, 2)   # [B, H, T, r + rope]
        v = v.transpose(1, 2)                                    # [B, H, T, r]

        attn_lat = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.softmax_scale,
        )

        attn_out = torch.einsum("bhtk,hkd->bhtd", attn_lat, self.w_uv)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)

        return self.w_o(attn_out)



class MoE(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) layer with shared and routed experts.

    This module routes tokens to the top-K specialized experts while concurrently 
    processing them through a shared expert to capture common knowledge. It 
    includes load balancing (auxiliary loss) and stability (z-loss) mechanisms.

    Args:
        config (ModelConfig): Configuration containing 'num_experts', 
                             'num_experts_per_token', and 'moe_intermediate_size'.

    Returns:
        tuple: A tuple containing:
            - output (torch.Tensor): Combined output from shared and routed experts.
            - aux_loss (torch.Tensor): Load balancing loss for the router.
            - z_loss (torch.Tensor): Stability loss to prevent logit explosion.
            - router_logits (torch.Tensor): Raw logits from the expert selection.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)


        self.w13 = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate_size)
        )
        self.shared_w13 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.shared_w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.xavier_uniform_(self.w13)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.shared_w13.weight)
        nn.init.xavier_uniform_(self.shared_w2.weight)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        N = x_flat.shape[0]
        E = self.num_experts
        K = self.num_experts_per_token
        I = self.intermediate_size

        gate_up_shared = self.shared_w13(x_flat)
        gate_s, up_s = gate_up_shared.chunk(2, dim=-1)
        shared_out = self.shared_w2(F.silu(gate_s) * up_s)
        
        router_logits = self.router(x_flat).float()


        topk_logits, topk_indices = torch.topk(router_logits, K, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)


        router_probs = F.softmax(router_logits, dim=-1)
        importance = router_probs.mean(dim=0)  
        load = (topk_indices.reshape(-1).bincount(minlength=E).float())
        load = load / load.sum()

        aux_loss = (importance * load).sum() * E


        z_loss =  torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)


        expert_ids = topk_indices.reshape(-1)
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)
        gates = topk_weights.reshape(-1)


        sort_perm = torch.argsort(expert_ids)
        expert_ids = expert_ids[sort_perm]
        token_ids = token_ids[sort_perm]
        gates = gates[sort_perm]

 
        counts = torch.bincount(expert_ids, minlength=E)
        max_count = counts.max()

        starts = torch.repeat_interleave(
            torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]]),
            counts
        )

        slot_ids = torch.arange(expert_ids.numel(), device=x.device) - starts

        packed_inputs = x_flat.new_zeros((E, max_count, hidden))
        packed_inputs[expert_ids, slot_ids] = x_flat[token_ids]


        proj = torch.matmul(packed_inputs, self.w13.transpose(1, 2)) 
        gate, up = proj.chunk(2, dim=-1)
        expert_hidden = F.silu(gate) * up


        packed_outputs = torch.matmul(expert_hidden, self.w2.transpose(1, 2))



        valid_outputs = packed_outputs[expert_ids, slot_ids]

        routed_out = x_flat.new_zeros((N, hidden))
        routed_out.index_add_(
            0,
            token_ids,
            valid_outputs * gates.unsqueeze(-1)
        )
        
        output = (shared_out + routed_out).view(bsz, seq_len, hidden)
    
        return output, aux_loss, z_loss, router_logits

class TransformerBlock(nn.Module):
    """
    A Transformer layer using MLA (attention) and MoE (experts).

    Args:
        config (ModelConfig): Model hyperparameters.

    Returns:
        x (torch.Tensor): Updated hidden states.
        aux_loss (torch.Tensor): Load balancing loss for the router.
        z_loss (torch.Tensor): Stability loss to prevent logit explosion.
        router_logits (torch.Tensor): Raw logits from the expert selection.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)

        self.attn = MLA(config)
        self.moe = MoE(config)

        self.ls1 = nn.Parameter(torch.ones(1) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        h = x + self.ls1 * self.attn(self.attn_norm(x), position_ids)
        moe_out, aux_loss, z_loss, router_logits = self.moe(self.ffn_norm(h))
        x = h + self.ls2 * moe_out
        return x, aux_loss, z_loss, router_logits

class Transformer(nn.Module):
    """
    Full MoE-MLA Transformer model.

    Assembles the token embeddings, a stack of MoE-enabled Transformer blocks, 
    and the final language modeling head. Implements weight tying between 
    embeddings and the LM head, and supports memory-efficient gradient checkpointing.

    Args:
        config (ModelConfig): Full architectural configuration including 'vocab_size', 
                             'hidden_size', 'num_layers', and MoE-specific settings.

    Returns:
        dict: A dictionary containing:
            - 'logits' (torch.Tensor): Final prediction scores for the vocabulary 
              across the sequence [batch_size, seq_len, vocab_size].
            - 'aux_loss' (torch.Tensor): Accumulated load-balancing loss summed 
              from all internal MoE layers.
            - 'z_loss' (torch.Tensor): Accumulated router stability loss summed 
              from all internal MoE layers.
            - 'router_logits' (torch.Tensor): The raw router scores from the 
              very last layer in the stack.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Allows the trainer to enable checkpointing dynamically."""
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        x = self.embed_tokens(input_ids)

     
        
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_z_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        last_router_logits = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, aux_loss, z_loss,router_logits = checkpoint(layer, x, position_ids, use_reentrant=False)
            else:
                x, aux_loss, z_loss,router_logits = layer(x, position_ids)

            
            last_router_logits = router_logits
            total_aux_loss += aux_loss
            total_z_loss += z_loss

        x=self.norm(x)
        
        logits=self.lm_head(x)
        
        return {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "z_loss": total_z_loss,
            "router_logits": last_router_logits
        }