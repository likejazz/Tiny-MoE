import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint
import math
from training.training_configs import ModelConfig
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False



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
        self.max_seq_len = config.max_seq_len
        self.theta = config.rope_theta

        self.rope_type = config.rope_type

  
        self.factor = float(getattr(config, "factor", 1.0))
        self.original_max_seq_len = int(
            getattr(config, "original_max_seq_len", self.max_seq_len)
        )
        self.beta_slow = float(getattr(config, "beta_slow", 1.0))
        self.beta_fast = float(getattr(config, "beta_fast", 32.0))



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
    Multi-head Latent Attention (MLA) module.

    Implements DeepSeek-style MLA with RoPE, YaRN context extension,
    FlashAttention support, and low-rank KV compression for efficient
    attention.

    Args:
        config: Model configuration containing MLA hyperparameters.

    Attributes:
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
        kv_lora_rank: Rank of the compressed KV representation.
        qk_nope_dim: Dimension of the non-positional query/key component.
        qk_rope_dim: Dimension of the rotary query/key component.
        max_seq_len: Maximum supported sequence length.
        original_max_seq_len: Original training context length.
        factor: YaRN context extension factor.
        attention_factor: YaRN attention scaling factor.
        attn_impl: Attention backend.
        flash_available: Whether FlashAttention is available.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.max_seq_len = config.max_seq_len
        self.original_max_seq_len = config.original_max_seq_len
        self.factor=config.factor
        self.attention_factor = 0.1 * math.log(self.factor) + 1
        self.attn_impl=config.attn_impl
        self.flash_available = FLASH_AVAILABLE

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
        # Batch_Size = B
        # Seq_len = T
        # kv_lora_rank = Dc
        # qk_nope_dim = Dn
        # qk_rope_dim = Dr
        # head_dim = Dv
        # num_heads = H
        bsz, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        c_in = self.w_in(x) # [B,T,D] @ [D,2*Dc] -> [B,T,2*Dc]
        c_kv, c_q = c_in.split(self.kv_lora_rank, dim=-1)  #c_kv [B,T,Dc] c_q  [B,T,Dc]

        #[B,T,Dc] @ [D,H*(Dn+Dr)] - > [B,T,H(Dn+Dr)]
        q_proj = self.w_uq_qr(c_q).reshape(
            bsz, seq_len, self.num_heads, self.qk_nope_dim + self.qk_rope_dim
        ) # [B,T,H,Dn+Dr]
        q_nope, q_rope = q_proj.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1) #[B,T,H,Dn]
        q_rope = self.rope(q_rope, position_ids) # [B,T,H,Dr]


        q_nope = q_nope.transpose(1, 2)  #[B,H,T,Dn]
        q_rope = q_rope.transpose(1, 2)  #[B,H,T,Dr]

        # [B,T,Dc] -> Unsqueeze - > [B,1,T,Dc]
        # [H,Dr,Dc] - > Transpose - > [H,Dc,Dr]
        # [B,1,T,Dc] @ [H,Dc,Dr] - > [B,H,T,Dr]
        k_nope = c_kv.unsqueeze(1) @ self.w_uk.transpose(-1, -2)
        # [B,T,D] @ [D,Dr] - >[B,T,Dr] -> unsqueeze -> [B,T,1,Dr] - > squeeze - > [B,T,Dr]
        k_rope = self.rope(self.w_kr(x).unsqueeze(2), position_ids).squeeze(2)  
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  #[B,1,T,Dr] - > [B,H,T,Dr]       

        #[B,T,Dc] - > Unsqueeze -> [B,1,T,Dc] @ [H,Dc,Dv] - > [B,H,T,Dv]
        v = c_kv.unsqueeze(1) @ self.w_uv


        q = torch.cat([q_nope, q_rope], dim=-1)  #[B,H,T,Dn] + [B,H,T,Dr] - > [B,H,T,Dn+Dr]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B,H,T,Dn+Dr]
 
        
        if self.attn_impl == "sdpa":
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                scale=self.softmax_scale,
            )  # [B,H,T,Dv]

        elif self.attn_impl == "flash_attn" and self.flash_available:
            attn_out = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
                softmax_scale=self.softmax_scale,
            ).transpose(1, 2)  #[B,T,H,Dv] - > transpose - > [B,H,T,Dv]

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)  # [B,T,H*Dv]
        return self.w_o(attn_out) # [B,T,H*Dv] @ [H*Dv,D] - > [B,T,D]


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
            - router_metrics (dict): Dictionary containing all router related metrics  such as
            expert load, load ratio, load standard deviation, minimum/maximum
            expert load, routing confidence, entropy, experts utilization.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.capacity_factor = config.capacity_factor

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)


        self.w13 = nn.Parameter(
            torch.empty(self.num_experts,self.hidden_size,2 * self.intermediate_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts,self.intermediate_size, self.hidden_size,)
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

        gate_up_shared = self.shared_w13(x_flat) #[N,D] @ [D,2I] - > [N,2I] 
        gate_s, up_s = gate_up_shared.chunk(2, dim=-1) # gate_S [N,I] up_s [N,I]
        shared_out = self.shared_w2(F.silu(gate_s) * up_s) # [N,I] @ [N,I] - > [N,I] @ [I,D] - > [N,D]
        
        router_logits = self.router(x_flat).to(torch.float32)  # [N,D] @ [D,E] - > [N,E]

        log_z = torch.logsumexp(router_logits, dim=-1)              # [N]  
        router_probs = torch.exp(router_logits - log_z.unsqueeze(-1)) # [N,E] 
        importance = router_probs.mean(dim=0)                         # [E]           
        
        topk_logits, topk_indices = torch.topk(
            router_logits, K, dim=-1, sorted=False
        )         #[N,K]                                              
        topk_weights = F.softmax(topk_logits, dim=-1)    # [N,K]       
        
        expert_ids = topk_indices.reshape(-1) #[N*K]       
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1) #[N*K]
        gates = topk_weights.reshape(-1).to(x_flat.dtype) #[N*K]      
        
        expert_ids, sort_perm = torch.sort(expert_ids)  #[N*K]
        token_ids = token_ids[sort_perm] #[N*K]
        gates = gates[sort_perm] #[N*K]
        
        counts = torch.bincount(expert_ids, minlength=E)   #[E]    
        expert_starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])  #[E]
        starts = expert_starts[expert_ids]  #[N*K]             
        

        load = counts.float() / (counts.sum() + 1e-9) #[E]
        aux_loss = E * torch.dot(importance, load) 
        z_loss = torch.mean(log_z ** 2)
        

        capacity = max(int((N * K / E) * self.capacity_factor),1)
        
        slot_ids = torch.arange(expert_ids.numel(), device=x.device) - starts #[N*K]
        

        valid_mask = slot_ids < capacity #[N*K]
        
   
        expert_ids = expert_ids[valid_mask]#[N*K]
        slot_ids = slot_ids[valid_mask]#[N*K]
        token_ids = token_ids[valid_mask]#[N*K]
        gates = gates[valid_mask]#[N*K]
        

        drop_counts = torch.bincount(expert_ids, minlength=E).float()
        load = drop_counts / (drop_counts.sum() + 1e-9)
        
        confidence = router_probs.max(dim=-1).values.mean()
        
  
        entropy = -(load * (load + 1e-9).log()).sum()
        utilization = torch.exp(entropy) / E
        

        router_metrics = {
            "load_std": load.std().detach().reshape(1),
            "load_max": load.max().detach().reshape(1),
            "load_min": load.min().detach().reshape(1),
            "load_ratio": (load.max() / (load.min() + 1e-9)).detach().reshape(1),
            "entropy": entropy.detach().reshape(1),
            "utilization": utilization.detach().reshape(1),
            "confidence": confidence.detach().reshape(1),
        }

        packed_inputs = x_flat.new_zeros((E, capacity, hidden)) #[E,C,D]
        packed_inputs[expert_ids, slot_ids] = x_flat[token_ids] #[E,C,D]
        

        proj = torch.matmul(packed_inputs, self.w13) # [E,C,D] @ [E,D,2I] -> [E,C,2I]
        gate, up = proj.chunk(2, dim=-1) #[E,C,I]
        packed_outputs = torch.matmul(F.silu(gate) * up, self.w2)  # [E,C,I] @ [E,I,D] -> [E,C,D]

        valid_outputs = packed_outputs[expert_ids, slot_ids] #[M,D] (M is number of valid tokens)
        

        routed_out = x_flat.new_zeros((N, hidden)) #[N,D]
        routed_out.index_add_(
            0,
            token_ids,
            valid_outputs * gates.unsqueeze(-1)
        ) #[N,D]
        
        # [N,D] + [N,D] -> [N,D] -> view -> [B,T,D]
        output = (shared_out + routed_out).view(bsz, seq_len, hidden)
        return output, aux_loss, z_loss, router_metrics


class TransformerBlock(nn.Module):
    """
    A Transformer layer using MLA (attention) and MoE (experts).

    Args:
        config (ModelConfig): Model hyperparameters.

    Returns:
        x (torch.Tensor): Updated hidden states.
        aux_loss (torch.Tensor): Load balancing loss for the router.
        z_loss (torch.Tensor): Stability loss to prevent logit explosion.
        router_metrics (dict): Dictionary containing all router related metrics 
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
        moe_out, aux_loss, z_loss, router_metrics = self.moe(self.ffn_norm(h))
        x = h + self.ls2 * moe_out
        return x, aux_loss, z_loss, router_metrics

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
            - 'router_metrics (dict):Dictionary containing all router related metrics 
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
        all_layer_metrics = []
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, aux_loss, z_loss, router_metrics = checkpoint(layer, x, position_ids, use_reentrant=False)
            else:
                x, aux_loss, z_loss, router_metrics = layer(x, position_ids)
                
            if router_metrics is not None:
                all_layer_metrics.append(router_metrics)

            total_aux_loss += aux_loss
            total_z_loss += z_loss

        x = self.norm(x)
        logits = self.lm_head(x)
        
        avg_router_metrics = {}
        if len(all_layer_metrics) > 0:
            for key in all_layer_metrics[0].keys():
                avg_router_metrics[f"router/{key}"] = torch.stack(
                    [m[key] for m in all_layer_metrics]
                ).mean()
        

        return {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "z_loss": total_z_loss,
            "router_metrics": avg_router_metrics
        }