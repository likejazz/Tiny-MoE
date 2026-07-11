from dataclasses import dataclass
from typing import Optional,Literal
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@dataclass
class ModelConfig:
    """
    Configuration for the Mixture-of-Experts (MoE) language model.

    Core
    ----
    vocab_size              : Number of tokens in the tokenizer vocabulary.
    hidden_size             : Embedding and hidden dimension used throughout the model.
    num_layers              : Number of transformer decoder layers.
    initializer_range       : Standard deviation for weight initialization.
    tie_word_embeddings     : Share input embedding and output projection weights.
    max_seq_len             : Maximum supported sequence length.
    max_batch_size          : Maximum batch size for static KV cache allocation.

    RMSNorm
    -------
    rms_norm_eps            : Epsilon added for numerical stability in RMSNorm.

    RoPE & YaRN
    -----------
    rope_theta              : Base frequency used by Rotary Positional Embeddings.
    rope_type               : Positional encoding variant ("default" or "yarn").
    beta_slow               : Lower interpolation boundary for YaRN.
    beta_fast               : Upper interpolation boundary for YaRN.
    factor                  : Context extension factor used by YaRN.
    original_max_seq_len    : Original context length before YaRN extension.
    mscale                  : Attention magnitude scaling factor for YaRN.

    MLA (Multi-head Latent Attention)
    ---------------------------------
    num_attention_heads     : Number of attention heads.
    kv_lora_rank            : Rank of the compressed latent key/value representation.
    qk_nope_dim             : Per-head dimension of the non-positional query/key path.
    qk_rope_dim             : Per-head dimension of the rotary query/key path.
    attn_impl               : Attention backend ("sdpa" or "flash_attn").
    absorb_weights          : Precompute absorbed MLA weights for faster inference.

    MoE (Mixture of Experts)
    ------------------------
    num_experts             : Number of experts in each MoE layer.
    num_experts_per_token   : Number of experts selected for each token.
    moe_intermediate_size   : Hidden dimension of each expert feed-forward network.
    capacity_factor         : Expert capacity multiplier used during routing.
    """

    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 14
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    max_seq_len: int = 2048
    max_batch_size: int = 1
    
    # RMSNorm
    rms_norm_eps: float = 1e-6
    
    # RoPE & YaRN 
    rope_theta: float = 10000.0
    rope_type: str = "yarn"
    beta_slow: float = 1.0
    beta_fast: float = 32.0
    factor: float = 4.0
    original_max_seq_len: int = 512
    mscale: float = 1.0

    # MLA
    num_attention_heads: int = 8
    kv_lora_rank: int = 96
    qk_nope_dim: int = 48
    qk_rope_dim: int = 16
    attn_impl: Literal["sdpa", "flash_attn"] = "sdpa"
    absorb_weights: bool = False

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    moe_intermediate_size: int = 1024
    capacity_factor: float = 1.25

@dataclass
class GenerationConfig:
    """
    Configuration for text generation during inference.

    Generation
    ----------
    max_new_tokens         : Maximum number of new tokens to generate.
    temperature            : Sampling temperature; lower values produce more deterministic outputs.
    top_k                  : Sample from the top-k most likely tokens.
    top_p                  : Sample from the smallest set of tokens whose cumulative probability exceeds p.
    repetition_penalty     : Penalty applied to discourage repeated tokens.
    eos_token_id           : Token that terminates generation.
    pad_token_id           : Token used for padding sequences.
    do_sample              : Whether to use probabilistic sampling instead of greedy decoding.
    use_yarn               : Enable YaRN context extension during generation.
    """
    max_new_tokens: int = 50
    temperature: float = 0.5
    top_k: Optional[int] = 30
    top_p: Optional[float] = 0.85
    repetition_penalty: float = 1.25
    eos_token_id: Optional[int] = tokenizer.eos_token_id
    pad_token_id: Optional[int] = tokenizer.eos_token_id
    do_sample: bool = True
    no_repeat_ngram_size: int = 2
    model_variant : Literal["base","base-yarn","fine-tuned"] = "fine-tuned"