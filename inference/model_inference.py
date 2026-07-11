import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.inference_configs import ModelConfig
from typing import Optional
import math
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
    def __init__(self, config:ModelConfig):
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
    Inference Multi-head Latent Attention (MLA) module.

    Implements DeepSeek-style MLA for autoregressive inference with support
    for KV cache compression, optional weight absorption, RoPE, YaRN, and
    FlashAttention.

    Args:
        config: Model configuration containing inference settings.

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
        absorb_weights: Whether to precompute absorbed weights for faster inference.
    """

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_dim = config.qk_nope_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.factor=config.factor
        self.mscale = 0.1 * config.mscale * math.log(self.factor) + 1.0
        self.max_seq_len = config.max_seq_len
        self.original_max_seq_len = config.original_max_seq_len
        self.attn_impl=config.attn_impl
        self.flash_available = FLASH_AVAILABLE
        self.absorb_weights = config.absorb_weights
        self.head_dim = self.qk_nope_dim + self.qk_rope_dim
        self.max_batch_size = config.max_batch_size


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


        use_scaled_attention = False
        if self.max_seq_len > self.original_max_seq_len and self.factor > 1.0:
            use_scaled_attention = True

        self.softmax_scale = base_scale * (self.mscale ** 2) if use_scaled_attention else base_scale
        
        self.register_buffer(
            "kv_cache",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank),
            persistent=False
        )
        self.register_buffer(
            "pe_cache",
            torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_dim),
            persistent=False
        )
        if self.absorb_weights:
            self.register_buffer(
                "w_q_absorbed",torch.empty(self.num_heads,self.kv_lora_rank, self.kv_lora_rank),persistent=False
            )
            self.register_buffer(
                "w_out_absorbed",torch.empty(self.hidden_size,self.num_heads,self.kv_lora_rank),persistent=False
            )

        self._weights_absorbed = False
        self.cache_len = 0

        nn.init.xavier_uniform_(self.w_uk)
        nn.init.xavier_uniform_(self.w_uv)

    def reset_cache(self):
        self.cache_len = 0

    def _absorb_weights(self):
        """
        Precompute absorbed projection weights for faster inference.

        Returns:
            None.
        """
        if self._weights_absorbed:
            return
        if self.absorb_weights:
            with torch.inference_mode():
                w_uq_ur = self.w_uq_qr.weight.contiguous() #[H(Dn+Dr), Dc]
                w_uq_ur = w_uq_ur.view(self.num_heads,self.qk_nope_dim + self.qk_rope_dim, self.kv_lora_rank) #[H,Dn+Dr,Dc]

                w_nope = w_uq_ur[:,:self.qk_nope_dim, :] #[H,Dn,Dc]
                #[H,Dc,Dn] @ [H,Dn,Dc] - > [H,Dc,Dc]
                w_q_absorbed = torch.matmul(w_nope.transpose(-1,-2), self.w_uk)

                W_o = self.w_o.weight.contiguous()
                W_o = W_o.view(self.hidden_size,self.num_heads,self.head_dim) #[D,H,Dv]
                W_uv = self.w_uv.transpose(-1, -2) #[H,Dv,Dc]
                #[D,H,Dv] @ [H,Dv,Dc] - > [D,H,Dc]
                w_out_absorbed = torch.einsum("dhv,hvc->dhc",W_o,W_uv)   
                
                self.w_q_absorbed.copy_(w_q_absorbed)
                self.w_out_absorbed.copy_(w_out_absorbed)
                assert w_q_absorbed.shape == (
                    self.num_heads,
                    self.kv_lora_rank,
                    self.kv_lora_rank,
                )

                assert w_out_absorbed.shape == (
                    self.hidden_size,
                    self.num_heads,
                    self.kv_lora_rank,
                )
                
                del self.w_uk
                del self.w_uv

                self._weights_absorbed = True
            
    @torch.inference_mode()
    def forward_absorbed(self, x, attention_mask=None, position_ids: Optional[torch.Tensor] = None):
        """
        Run the forward pass using absorbed projection weights.

        Args:
            x: Input hidden states.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Attention output.
        """
        if not self._weights_absorbed:
            self._absorb_weights()
        bsz, seq_len, _ = x.shape
        start = self.cache_len
        end = start + seq_len

        if position_ids is None:
            position_ids = torch.arange(seq_len,device=x.device).unsqueeze(0).expand(bsz,-1)

        # [B,T,D] @ [D,2*Dc] - > [B,T,2*Dc]
        c_in = self.w_in(x)
        c_kv, c_q = c_in.split(self.kv_lora_rank, dim=-1) #[B,T,Dc]

        # [B,T,Dc] @ [Dc,H(Dn+Dr)] - > [B,T,H(Dn+Dr)] - > [B,T,H,Dn+Dr]
        q_proj = self.w_uq_qr(c_q).reshape(
            bsz, seq_len, self.num_heads, self.qk_nope_dim + self.qk_rope_dim
        )

        q_rope = q_proj[:, :, :, self.qk_nope_dim:]         # [B, T, H, Dr]
        q_rope = self.rope(q_rope, position_ids)            # [B, T, H, Dr]
        q_rope = q_rope.transpose(1, 2)                     # [B, H, T, Dr]


        self.kv_cache[:bsz, start:end] = c_kv
        #[B,T,D] @ [D,Dr] - > [B,T,Dr] - > unsqueeze [B,T,1,Dr] - > squeeze -> [B,T,Dr]
        k_rope_cur = self.rope(self.w_kr(x).unsqueeze(2), position_ids).squeeze(2)  
        self.pe_cache[:bsz, start:end] = k_rope_cur
        self.cache_len = end


        past_c_kv = self.kv_cache[:bsz, :end]  
        past_pe = self.pe_cache[:bsz, :end]  

        # c_q
        # [b, q, d]
        # [B, T, Dc]

        # w_q_absorbed
        # [h, d, c]
        # [H, Dc, Dc]

        # past_c_kv
        # [b, k, c]
        # [B, T_cache, Dc]

        # ↓

        # q_nope
        # [b, h, q, k]
        # [B, H, T, T_cache]
        scores_nope = torch.einsum(
            "bqd,hdc,bkc->bhqk",
            c_q,
            self.w_q_absorbed,
            past_c_kv,
        )
        #[B,H,T,Dr] @ [B,T_cache,Dr] - > [B,H,T,T_cache]
        scores_rope = torch.einsum(
            "bhqd,bkd->bhqk",
            q_rope,
            past_pe,
        )
        scores = (scores_nope + scores_rope) * self.softmax_scale

        if attention_mask is not None:
            scores += attention_mask

        attn = torch.softmax(scores, dim=-1)
        #[B,H,T,T_cache] @ [B,T_cache,Dc] - > [B,H,T,Dc]
        latent_context = torch.einsum(
            "bhqk,bkd->bhqd",
            attn,
            past_c_kv,
        )
        #[B,H,T,Dc] @ [D,H,Dc] - > [B,T,D]
        output = torch.einsum(
            "bhqc,dhc->bqd",
            latent_context,
            self.w_out_absorbed,
        )
        return output
    
    @torch.inference_mode()
    def forward_normal(self, x, attention_mask=None, position_ids: Optional[torch.Tensor] = None):
        """
        Run the standard MLA forward pass.

        Args:
            x: Input hidden states.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Attention output.
        """
        bsz, seq_len, _ = x.shape
        start = self.cache_len
        end = start + seq_len

        if position_ids is None:
            position_ids = torch.arange(seq_len,device=x.device).unsqueeze(0).expand(bsz,-1)

        # [B,T,D] @ [D,2*Dc] - > [B,T,2*Dc]
        c_in = self.w_in(x)
        c_kv, c_q = c_in.split(self.kv_lora_rank, dim=-1) #[B,T,Dc]

        # [B,T,Dc] @ [Dc,H(Dn+Dr)] - > [B,T,H(Dn+Dr)] - > [B,T,H,Dn+Dr]
        q_proj = self.w_uq_qr(c_q).reshape(
            bsz, seq_len, self.num_heads, self.qk_nope_dim + self.qk_rope_dim
        )
        q_nope, q_rope = q_proj.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)#[B,T,H,Dn],[B,T,H,Dr]
        q_rope = self.rope(q_rope, position_ids) #[B,T,H,Dr]


        q_nope = q_nope.transpose(1, 2)  #[B,H,T,Dn]
        q_rope = q_rope.transpose(1, 2)  #[B,H,T,Dr]


        self.kv_cache[:bsz, start:end] = c_kv
        #[B,T,D] @ [D,Dr] - > [B,T,Dr] - > unsqueeze [B,T,1,Dr] - > squeeze -> [B,T,Dr]
        k_rope_cur = self.rope(self.w_kr(x).unsqueeze(2), position_ids).squeeze(2)  
        self.pe_cache[:bsz, start:end] = k_rope_cur
        self.cache_len = end


        past_c_kv = self.kv_cache[:bsz, :end]  
        past_pe = self.pe_cache[:bsz, :end]  


        # [B,H,T,Dn] @ [H,Dn,Dc] -> [B,H,T,Dc]
        q_lat = q_nope @ self.w_uk

        #[B,T,Dc] -> Unsqueeze - > [B,1,T,Dc] - > Expand - > [B,H,T,Dc]
        #[B,T,Dr] - > Unsqueeze - > [B,1,T,Dr] - > Expand - > [B,H,T,Dr]
        #[B,T,Dc] - > Unsqueeze - > [B,1,T,Dc] - > Expand - > [B,H,T,Dc]
        k_lat  = past_c_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  
        k_rope = past_pe.unsqueeze(1).expand(-1, self.num_heads, -1, -1)    
        v  = past_c_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  
        #[B,H,T,Dc] + [B,H,T,Dr] - > [B,H,T,Dc+Dr]
        #[B,H,T,Dc] + [B,H,T,Dr] - > [B,H,T,Dc+Dr]
        q = torch.cat([q_lat,  q_rope], dim=-1)  
        k = torch.cat([k_lat,  k_rope], dim=-1)  


        if self.attn_impl == "sdpa":
            attn_lat = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=(seq_len > 1 and attention_mask is None),
                scale=self.softmax_scale,
            )  #Q @ K^T/sqrt(D) * V - > [B,H,T,Dc+Dr] - > [B,H,Dc+Dr,T] - > [ B,H,T,T] @ [B,H,T,Dc] - > [B,H,T,Dc]

        elif self.attn_impl == "flash_attn" and self.flash_available:
            attn_lat = flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=(seq_len > 1 and attention_mask is None),
                softmax_scale=self.softmax_scale,
            ).transpose(1, 2)  # [B,T,H,Dc] - > [B,H,T,Dc]

        #[B,H,T,Dc] @ [H,Dc,Dv] - > [B,H,T,Dv]
        attn_out = attn_lat @ self.w_uv
        #[B,H,T,Dv] - > Transpose - >  [B,T,H,Dv] - > [B,T,H*Dv]
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)  
        return self.w_o(attn_out) # [B,T,H*Dv] @ [H*Dv,D] - > [B,T,D]


    def forward(self, x, attention_mask=None, position_ids: Optional[torch.Tensor] = None):
        """
        Run the MLA forward pass.

        Dispatches to either the standard or absorbed implementation.

        Args:
            x: Input hidden states.
            attention_mask: Optional attention mask.
            position_ids: Optional position IDs.

        Returns:
            Attention output.
        """
        if self.absorb_weights:
            return self.forward_absorbed(x, attention_mask, position_ids)
        else:
            return self.forward_normal(x, attention_mask, position_ids)

    
class MoE(nn.Module):
    """
    Implements a Mixture-of-Experts (MoE) layer with shared and routed experts.

    This module routes tokens to the top-K specialized experts while concurrently 
    processing them through a shared expert to capture common knowledge.

    Args:
        config (ModelConfig): Configuration containing 'num_experts', 
                             'num_experts_per_token', and 'moe_intermediate_size'.

    Returns:
        output (torch.Tensor): Combined output from shared and routed experts.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)


        self.w13 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_size)
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
    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden) #[B*T,D] OR [N,D]
        N = x_flat.shape[0]
        E = self.num_experts
        K = self.num_experts_per_token


        gate_up_shared = self.shared_w13(x_flat) #[N,D] @ [D,2I] - > [N,2I] 
        gate_s, up_s = gate_up_shared.chunk(2, dim=-1) # gate_S [N,I] up_s [N,I]
        shared_out = self.shared_w2(F.silu(gate_s) * up_s) # [N,I] @ [N,I] - > [N,I] @ [I,D] - > [N,D]


        router_logits = self.router(x_flat).to(torch.float32)   # [N,D] @ [D,E] - > [N,E]
        topk_logits, topk_indices = torch.topk(router_logits, K, dim=-1)  #[N,K]
        topk_weights = F.softmax(topk_logits, dim=-1)  # [N,K]  

        expert_ids = topk_indices.reshape(-1)#[N*K]
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)#[N*K]
        gates = topk_weights.reshape(-1).to(x_flat.dtype) #[N*K]

        expert_ids, sort_perm = torch.sort(expert_ids)  #[N*K]
        token_ids = token_ids[sort_perm]#[N*K]
        gates = gates[sort_perm]#[N*K]

        counts = torch.bincount(expert_ids, minlength=E)#[E]
        max_count = counts.max()

        valid_mask = (
            torch.arange(max_count, device=x.device).unsqueeze(0) < counts.unsqueeze(1)
        )

        packed_inputs = x_flat.new_zeros((E, max_count, hidden))#[E,C,D]
        packed_inputs[valid_mask] = x_flat[token_ids]#[E,C,D]

        proj = torch.matmul(packed_inputs, self.w13)# [E,C,D] @ [E,D,2I] -> [E,C,2I]
        gate, up = proj.chunk(2, dim=-1)#[E,C,I]
        packed_outputs = torch.matmul(F.silu(gate) * up, self.w2)  # [E,C,I] @ [E,I,D] -> [E,C,D]

        valid_outputs = packed_outputs[valid_mask]#[M,D] (M is number of valid tokens)

        routed_out = x_flat.new_zeros((N, hidden)) #[N,D]
        routed_out.index_add_(0, token_ids, valid_outputs * gates.unsqueeze(-1))
        # [N,D] + [N,D] -> [N,D] -> view -> [B,T,D]
        output = (shared_out + routed_out).view(bsz, seq_len, hidden)
        return output
    
class TransformerBlock(nn.Module):
    """
    A single Transformer layer optimized for inference.

    This block integrates Multi-Head Latent Attention (MLA) and a Mixture-of-Experts 
    (MoE) layer with residual connections and RMSNorm. It is decorated with 
    inference_mode to reduce memory overhead during generation.

    Args:
        config (ModelConfig): Configuration object for layer dimensions and MoE settings.

    Returns:
        torch.Tensor: The processed hidden states of shape [batch, seq_len, hidden_size].
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config)
        self.ffn_norm = RMSNorm(config)

        self.attn = MLA(config)
        self.moe = MoE(config)

        self.ls1 = nn.Parameter(torch.ones(1) * 0.1)
        self.ls2 = nn.Parameter(torch.ones(1) * 0.1)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, position_ids=None):
        h = x + self.ls1 * self.attn(self.attn_norm(x), position_ids)
        x = h + self.ls2 * self.moe(self.ffn_norm(h))
        return x


class Transformer(nn.Module):
    """
    The core Transformer model architecture optimized for inference and generation.

    This class coordinates the embedding layer, a stack of TransformerBlocks (MLA + MoE), 
    and the final language modeling head. it supports KV-cache management and 
    optional tied embeddings.

    Args:
        config (ModelConfig): Configuration object containing model dimensions, 
                             layer counts, and vocabulary size.

    Returns:
        torch.Tensor: Logits for the next-token prediction. If 'return_last_only' 
                      is True, returns only the logits for the final position.
    """

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def reset_cache(self):
        for layers in self.layers:
            if hasattr(layers.attn, "reset_cache"):
                layers.attn.reset_cache()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor,  position_ids=None,return_last_only: bool = False):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        x = self.embed_tokens(input_ids)

        for layers in self.layers:
            x = layers(x,position_ids)

        x = self.norm(x)
        if return_last_only:
            x = x[:, -1:, :]

        logits=self.lm_head(x)
        return logits
    
    