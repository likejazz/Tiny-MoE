from dataclasses import dataclass
from typing import Tuple, Literal
@dataclass
class ModelConfig:
    """
    Model architecture configuration for a MoE language model with MLA attention.
    
    Core
    ----
    vocab_size              : Number of tokens in the vocabulary.
    hidden_size             : Embedding and hidden state dimension across all layers.
    num_layers              : Number of transformer decoder layers.
    initializer_range       : Std dev for weight initialization.
    tie_word_embeddings     : Shares weights between input embeddings and output projection.
    max_seq_len             : Maximum sequence length the model can process.
    max_batch_size          : Maximum batch size for static KV-cache allocation.
    
    RMSNorm
    -------
    rms_norm_eps            : Small epsilon added for numerical stability in RMSNorm.
    
    RoPE & YaRN
    -----------
    rope_theta              : Base frequency controlling how fast RoPE rotations decay.
    rope_type               : Positional encoding variant — "yarn" enables context extension.
    beta_slow               : YaRN low-frequency interpolation boundary.
    beta_fast               : YaRN high-frequency interpolation boundary.
    factor                  : YaRN scale factor for extending beyond the training context.
    mscale                  : YaRN magnitude scaling; inferred automatically if None.
    original_max_seq_len    : The sequence length the model was originally trained on.
    
    MLA (Multi-head Latent Attention)
    ----------------------------------
    num_attention_heads     : Number of query heads.
    kv_lora_rank            : Rank of the low-rank KV compression — smaller means less KV cache memory.
    qk_nope_dim             : Per-head dimension for the non-positional (NoPE) query/key path.
    qk_rope_dim             : Per-head dimension for the rotary-encoded query/key path.
    attn_impl               : Attention backend — "flash_attn" is faster and more memory efficient.
    
    MoE (Mixture of Experts)
    ------------------------
    num_experts             : Total number of experts in each MoE FFN layer.
    num_experts_per_token   : How many experts each token is routed to.
    moe_intermediate_size   : Hidden dimension inside each expert's FFN.
    """
    vocab_size: int = 32000
    hidden_size: int = 512
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
    original_max_seq_len: int = 512

    # MLA
    num_attention_heads: int = 8
    kv_lora_rank: int = 96
    qk_nope_dim: int = 48
    qk_rope_dim: int = 16
    attn_impl: Literal["sdpa", "flash_attn"] = "sdpa"

    # MoE
    num_experts: int = 8
    num_experts_per_token: int = 2
    moe_intermediate_size: int = 1024
    capacity_factor: float = 1.25


class TrainConfig:
    """
    Training Configuration.
    
    Optimization
    ------------
    lr                  : How fast the model learns. Too high = unstable, too low = slow.
    weight_decay        : Gently penalizes large weights to prevent overfitting.
    betas               : How much the optimizer trusts recent vs. past gradients (β₁, β₂).
    grad_clip           : Prevents exploding gradients by capping their max norm.
    num_train_steps     : How long to train for.
    warmup_steps        : Eases the LR from 0 up slowly before the main schedule kicks in.
    optimizer_type      : Which optimizer to use — "8bitAdamW" saves a lot of GPU memory.
    
    Batch / Throughput
    ------------------
    micro_batch_size    : How many samples are processed at once on a single device.
    grad_accum_steps    : Simulates a larger batch by accumulating gradients before updating.
    max_seq_len         : Longest sequence the model sees during training.
    
    Precision
    ---------
    mixed_precision     : Use "bf16" on modern GPUs for stability, "fp16" otherwise.
    
    Memory
    ------
    activation_checkpointing : Trades a bit of compute to use significantly less memory.
    compile_model            : Fuses ops for faster training.Can be unstable, use with caution.
    compile_mode             : How hard torch.compile tries to optimize — "max-autotune" is slowest to start but fastest to run.
    
    MoE Stability
    -------------
    router_aux_loss_coef : Nudges the router to spread tokens evenly across experts.
    router_z_loss_coef   : Stops the router from becoming overconfident, keeps logits small.
    capacity_factor      : How many extra tokens each expert can take before dropping the rest.
    top_k                : How many experts each token gets sent to.
    
    Logging & Checkpointing
    -----------------------
    save_interval   : How often to save a checkpoint (in eval intervals).
    num_eval_steps  : How many batches to run when evaluating.
    eval_interval   : How often to evaluate (in training steps).
    log_interval    : How often to print metrics.
    out_dir         : Where checkpoints get saved.
    
    DeepSpeed (ZeRO)
    ----------------
    deepspeed_enabled           : Turns on DeepSpeed for distributed training.
    wall_clock_breakdown        : Prints DeepSpeed timing internals for profiling.
    zero_stage                  : Controls how aggressively ZeRO shards model state across GPUs.
    overlap_comm                : Hides communication latency by overlapping it with computation.
    contiguous_gradients        : Packs gradients together to reduce memory fragmentation.
    reduce_bucket_size          : Controls gradient reduction bucket size.
    allgather_bucket_size       : Controls parameter all-gather bucket size.
    allgather_partitions        : Rebuilds sharded parameters using all-gather operations.
    reduce_scatter              : Uses reduce-scatter for gradient synchronization.
    offload_optimizer           : Moves optimizer state and updates to CPU memory.
    offload_param               : Moves model parameters to CPU memory.
    partition_activations       : Splits activation checkpoints across GPUs to save memory.
    cpu_checkpointing           : Stores activation checkpoints in CPU memory.
    contiguous_memory_optimization : Keeps checkpoint memory contiguous and defragmented.
    num_checkpoints             : Sets how many activation checkpoints are tracked.
    loss_scale_window           : Controls fp16 dynamic loss scale adjustment timing.
    moe_enabled                 : Enables Mixture-of-Experts training.
    ep_size                     : Sets the expert parallel group size.
    moe_param_group             : Separates MoE parameters into dedicated optimizer groups.
    use_residual                : Adds residual connections to MoE outputs.      : Prints DeepSpeed timing internals — handy for finding bottlenecks.
    """
    # Optimization
    lr: float = 2.5e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    num_train_steps: int = 30000
    warmup_steps: int = 2800
    optimizer_type: Literal["AdamW", "PagedAdamW", "8bitAdamW"] = "8bitAdamW"
    
    # Batch / Throughput
    micro_batch_size: int = 64
    grad_accum_steps: int = 4
    max_seq_len: int = 512
    
    # Precision / Performance
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    
    # Memory
    activation_checkpointing: bool = True
    compile_model: bool = True          
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    
    # MoE Stability
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    
    # Logging
    save_interval: int = 600
    num_eval_steps: int = 50
    eval_interval: int = 1000
    log_interval: int = 5
    out_dir: str = "/kaggle/working/checkpoints"
    
    # DeepSpeed
    deepspeed_enabled: bool = True
    wall_clock_breakdown: bool = False
    zero_stage: int = 2
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 100_000_000
    allgather_bucket_size: int = 100_000_000
    allgather_partitions: bool = True
    reduce_scatter: bool = True
    offload_optimizer: bool = False
    offload_param: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    num_checkpoints: int = 8
    loss_scale_window: int = 1000
    moe_enabled: bool = True
    ep_size: int = 2
    moe_param_group: bool = True
    use_residual: bool = True