# IMPORTS
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.utils.checkpoint import checkpoint




@dataclass
class ModelConfig:
    """
    Configuration class for a DeepSeek-style MoE model with MLA.

    Attributes:
        vocab_size (int): Total number of tokens in the vocabulary.
        hidden_size (int): Dimensionality of the model layers.
        num_layers (int): Total number of transformer blocks.
        initializer_range (float): Standard deviation for weight initialization.
        tie_word_embeddings (bool): Whether to share weights between input and output embeddings.
        max_seq_len (int): Maximum sequence length (context window).

        rms_norm_eps (float): Small constant added to denominator in RMSNorm for stability.
        rope_theta (float): The base period for Rotary Positional Embeddings (RoPE).

        # MLA (Multi-head Latent Attention)
        num_attention_heads (int): Number of attention heads.
        kv_lora_rank (int): The rank of the compressed KV latent vector.
        qk_nope_dim (int): Dimension of the Non-rope part of Query/Key projections.
        qk_rope_dim (int): Dimension of the Rotary-enabled part of Query/Key projections.

        # MoE (Mixture of Experts)
        num_experts (int): Total number of available experts in each MoE layer.
        num_experts_per_token (int): Number of experts activated for each token (Top-K).
        moe_intermediate_size (int): The hidden dimension size inside each individual expert.
    """
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 12
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    max_seq_len: int = 1024

    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    # MLA
    num_attention_heads: int = 8
    kv_lora_rank: int = 128
    qk_nope_dim: int = 32
    qk_rope_dim: int = 32

    # MoE
    num_experts: int = 8
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
    Rotary Positional Embedding (RoPE) using complex number rotation.

    Applies a relative position encoding by rotating segments of the query and key
    embeddings in complex space, allowing the model to capture relative distances
    between tokens effectively.

    Args:
        config (ModelConfig): Config containing 'qk_rope_dim', 'max_seq_len', 
                             and 'rope_theta'.

    Attributes:
        freqs_cis (torch.Tensor): Precomputed complex exponential (cos + i*sin) 
                                 cache for all possible positions up to max_seq_len.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.qk_rope_dim
        self.max_seq_len = config.max_seq_len
        self.theta = config.rope_theta
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        dtype = x.dtype
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        freqs_cis = self.freqs_cis[:seq_len].view(1, seq_len, 1, -1)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(3).to(dtype)

class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) with KV Compression.

    MLA reduces the KV cache size by projecting Keys and Values into a low-rank 
    latent space (c_kv). It also separates Query/Key components into 
    content-based (nope) and position-based (rope) vectors.

    Args:
        config (ModelConfig): Configuration object containing hidden_size, 
                             lora_rank, and attention dimensions.

    Attributes:
        w_dkv (nn.Linear): Down-projection for Keys and Values into latent space.
        w_dq (nn.Linear): Down-projection for Queries into latent space.
        rope (RoPE): Rotary Positional Embedding module for the 'rope' dimensions.
        scale (float): Scaling factor for the attention scores (1/sqrt(head_dim)).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim

        self.w_dkv = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.w_dq = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.w_ukv = nn.Linear(self.kv_lora_rank,self.num_heads * self.head_dim * 2,  bias=False )
        self.w_uq_qr = nn.Linear(self.kv_lora_rank,self.num_heads * self.head_dim + self.num_heads * self.qk_rope_dim,bias=False)
        self.w_kr = nn.Linear(self.hidden_size, self.qk_rope_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rope = RoPE(config)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape


        c_kv = self.w_dkv(x)
        kv = self.w_ukv(c_kv)  
        kv = kv.view(bsz, seq_len, self.num_heads, 2 * self.head_dim)
        k_nope, v = kv.split(self.head_dim, dim=-1)


        c_q = self.w_dq(x)
        q_proj = self.w_uq_qr(c_q)  
        q_proj = q_proj.view(bsz, seq_len, self.num_heads, self.head_dim + self.qk_rope_dim)
        q_nope, q_rope = q_proj.split([self.head_dim, self.qk_rope_dim], dim=-1)

        k_rope = self.w_kr(x).view(bsz, seq_len, 1, self.qk_rope_dim)

        q_rope = self.rope(q_rope, position_ids)
        k_rope = self.rope(k_rope, position_ids)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k_rope = k_rope.repeat(1, 1, self.num_heads, 1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        def sdpa_func(q_in, k_in, v_in):
            return F.scaled_dot_product_attention(
                q_in, k_in, v_in,
                is_causal=True
            )

        attn_output = sdpa_func(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        return self.w_o(attn_output)

class Expert(nn.Module):
    """
    A single Mixture of Experts (MoE) feed-forward block using SwiGLU activation.

    This block implements the Gated Linear Unit variant where the hidden 
    representation is the element-wise product of a SiLU-activated linear 
    projection (w1) and a gated linear projection (w3).

    Args:
        config (ModelConfig): Config object with 'hidden_size' and 
                             'moe_intermediate_size'.

    Attributes:
        w1 (nn.Linear): The "gate" projection.
        w2 (nn.Linear): The "down" projection back to hidden_size.
        w3 (nn.Linear): The "up" projection.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    """
    Sparsely Gated Mixture of Experts (MoE) Layer.

    This module implements a dynamic routing mechanism that distributes input tokens 
    across a set of specialized expert networks. By selecting only the top-k experts 
    per token, it increases model capacity without a linear increase in computation.

    Args:
        config (ModelConfig): A configuration object that must have:
            - num_experts (int): The total pool of expert networks.
            - num_experts_per_token (int): How many experts each token is routed to.
            - hidden_size (int): The input and output dimensionality of the tokens.

    Returns:
        output (torch.Tensor): The aggregated result of expert processing, weighted 
            by router probabilities. Shape: (batch_size, seq_len, hidden_size).
        aux_loss (torch.Tensor): A load-balancing loss that penalizes over-reliance 
            on a small subset of experts.
        z_loss (torch.Tensor): A stability loss that discourages the router from 
            producing extremely high-magnitude logits.
        router_logits (torch.Tensor): The raw scores for each expert before softmax.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)  
        N = x_flat.shape[0]


        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)


        z_loss = 1e-3 * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)


        topk_weights, topk_indices = torch.topk(
            router_probs, self.num_experts_per_token, dim=-1
        )

        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)


        importance = router_probs.mean(dim=0)        
        load = router_probs.sum(dim=0)               

        load = load / (load.sum() + 1e-6)

        aux_loss = torch.sum(importance * load) * self.num_experts

        token_indices = torch.arange(N, device=x.device).unsqueeze(1).expand_as(topk_indices)

        flat_experts = topk_indices.reshape(-1)
        flat_tokens = token_indices.reshape(-1)
        flat_gates = topk_weights.reshape(-1)

        sorted_idx = torch.argsort(flat_experts)
        inverse_idx = torch.argsort(sorted_idx)

        flat_experts = flat_experts[sorted_idx]
        flat_tokens = flat_tokens[sorted_idx]
        flat_gates = flat_gates[sorted_idx]

        sorted_inputs = x_flat[flat_tokens]


        expert_counts = torch.bincount(
            flat_experts,
            minlength=self.num_experts
        )

        sorted_outputs = torch.empty_like(sorted_inputs)

        start = 0
        for expert_idx in range(self.num_experts):
            count = expert_counts[expert_idx]

            if count <= 0:
                continue

            end = start + count

            expert_in = sorted_inputs[start:end]
            expert_out = self.experts[expert_idx](expert_in)


            sorted_outputs[start:end] = expert_out.to(sorted_outputs.dtype)

            start = end

        unsorted_outputs = sorted_outputs[inverse_idx]

        unsorted_outputs = unsorted_outputs.view(
            N, self.num_experts_per_token, hidden
        )

        output_flat = (unsorted_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        output = output_flat.view(bsz, seq_len, hidden)

        return output, aux_loss, z_loss, router_logits

class TransformerBlock(nn.Module):
    """
    A single Transformer layer combining Multi-Head Latent Attention (MLA) and MoE.

    This block implements a pre-norm residual architecture. It utilizes Layer Scale 
    (ls1, ls2) to weigh the residual branches, which helps stabilize deep 
    network initialization.

    Args:
        config (ModelConfig): Configuration object for hidden dimensions and MoE settings.

    Returns:
        tuple: A tuple containing:
            - x (torch.Tensor): The processed hidden states [batch_size, seq_len, hidden_size].
            - aux_loss (torch.Tensor): Load balancing loss from the MoE router.
            - z_loss (torch.Tensor): Router stability loss (z-loss).
            - router_logits (torch.Tensor): The raw scores for expert selection.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config)
        self.mla = MLA(config)
        self.ls1 = nn.Parameter(torch.ones(1) * 0.1)
        self.norm2 = RMSNorm(config)
        self.moe = MoE(config)
        self.ls2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):

        residual = x
        x = self.norm1(x)
        x = self.mla(x, position_ids)
        x = residual + self.ls1 * x

        residual = x
        x = self.norm2(x)
        moe_out, aux_loss, z_loss,router_logits = self.moe(x)
        x = residual + self.ls2 * moe_out
        return x, aux_loss, z_loss,router_logits

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

     
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)
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