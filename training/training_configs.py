from dataclasses import dataclass
from typing import Tuple, Literal
@dataclass
class ModelConfig:
    """
    Configuration for the model.

    This dataclass defines all architectural hyperparameters used to
    construct the model, including the transformer backbone, normalization,
    positional encoding, Multi-head Latent Attention (MLA), and Mixture-of-
    Experts (MoE) components.

    Core
    ----
    vocab_size              : Number of tokens in the tokenizer vocabulary.
    hidden_size             : Embedding dimension and hidden size used throughout the model.
    num_layers              : Number of transformer decoder layers.
    initializer_range       : Standard deviation used for weight initialization.
    tie_word_embeddings     : Whether to share the input embedding and output projection weights.
    max_seq_len             : Maximum sequence length supported by the model.

    RMSNorm
    -------
    rms_norm_eps            : Small constant added for numerical stability in RMSNorm.

    RoPE & YaRN
    -----------
    rope_theta              : Base frequency used by Rotary Positional Embeddings (RoPE).
    rope_type               : Positional encoding variant ("default" or "yarn").
    beta_slow               : Lower interpolation boundary used by YaRN.
    beta_fast               : Upper interpolation boundary used by YaRN.
    factor                  : Context extension factor applied by YaRN.
    original_max_seq_len    : Original training context length before YaRN extension.
    mscale                  : Magnitude scaling factor applied to attention during YaRN.

    MLA (Multi-head Latent Attention)
    ---------------------------------
    num_attention_heads     : Number of attention heads.
    kv_lora_rank            : Rank of the compressed latent key/value representation.
    qk_nope_dim             : Per-head dimension of the non-positional query/key component.
    qk_rope_dim             : Per-head dimension of the rotary-encoded query/key component.
    attn_impl               : Attention implementation ("sdpa" or "flash_attn").

    MoE (Mixture of Experts)
    ------------------------
    num_experts             : Number of feed-forward experts in each MoE layer.
    num_experts_per_token   : Number of experts selected for each token.
    moe_intermediate_size   : Hidden dimension of each expert's feed-forward network.
    capacity_factor         : Capacity multiplier controlling the maximum tokens assigned to each expert.
    """
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 14
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    max_seq_len: int = 512
    
    # RMSNorm
    rms_norm_eps: float = 1e-6
    
    # RoPE & YaRN 
    rope_theta: float = 10000.0
    rope_type: str = "default"
    beta_slow: float = 1.0
    beta_fast: float = 32.0
    factor: float = 1.0
    original_max_seq_len: int = 512
    mscale: float = 1.0

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
    Training configuration for model optimization and runtime behavior.

    This dataclass defines the optimizer, batching strategy, mixed precision,
    memory optimizations, logging, checkpointing, and DeepSpeed settings used
    during training.

    Optimization
    ------------
    lr                          : Initial learning rate.
    weight_decay                : Weight decay coefficient for regularization.
    betas                       : Beta coefficients used by the optimizer.
    grad_clip                   : Maximum gradient norm for clipping.
    num_train_steps             : Total number of training steps.
    warmup_steps                : Number of learning rate warmup steps.
    optimizer_type              : Optimizer implementation to use.

    Batch / Throughput
    ------------------
    micro_batch_size            : Number of samples processed per device before gradient accumulation.
    grad_accum_steps            : Number of gradient accumulation steps.
    max_seq_len                 : Sequence length used during training.

    Precision / Performance
    -----------------------
    mixed_precision             : Mixed precision mode ("no", "fp16", or "bf16").

    Memory
    ------
    activation_checkpointing    : Enable activation checkpointing to reduce memory usage.
    compile_model               : Compile the model with PyTorch 2.x.
    compile_mode                : Torch compilation optimization mode.

    MoE Stability
    -------------
    router_aux_loss_coef        : Auxiliary load-balancing loss coefficient.
    router_z_loss_coef          : Router z-loss coefficient for stabilizing routing logits.

    Logging
    -------
    save_interval               : Steps between checkpoint saves.
    num_eval_steps              : Number of evaluation batches.
    eval_interval               : Steps between evaluations.
    log_interval                : Steps between logging metrics.
    out_dir                     : Directory for checkpoints and outputs.

    DeepSpeed
    ---------
    deepspeed_enabled           : Enable DeepSpeed training.
    wall_clock_breakdown        : Collect DeepSpeed timing statistics.
    zero_stage                  : ZeRO optimization stage.
    overlap_comm                : Overlap communication with computation.
    contiguous_gradients        : Store gradients contiguously in memory.
    reduce_bucket_size          : Bucket size used for gradient reduction.
    allgather_bucket_size       : Bucket size used during parameter all-gather.
    allgather_partitions        : Enable partitioned all-gather operations.
    reduce_scatter              : Enable reduce-scatter communication.
    offload_optimizer           : Offload optimizer states to CPU.
    offload_param               : Offload model parameters to CPU.
    partition_activations       : Partition activations to reduce memory usage.
    cpu_checkpointing           : Store activation checkpoints in CPU memory.
    contiguous_memory_optimization
                                : Optimize activation memory layout.
    num_checkpoints             : Number of activation checkpoints.
    loss_scale_window           : Dynamic loss scaling adjustment window.
    moe_enabled                 : Enable DeepSpeed Mixture-of-Experts support.
    ep_size                     : Expert parallelism size.
    moe_param_group             : Place MoE parameters into separate optimizer groups.
    use_residual                : Enable residual MoE routing if supported.

    DataLoader
    ---------
    drop_last                  : Drop the final incomplete batch for consistent batch sizes.
    num_workers                : Number of subprocesses used for loading data.
    prefetch_factor            : Number of batches prefetched by each worker.
    persistent_workers         : Keep DataLoader workers alive between epochs.
    pin_memory                 : Pin CPU memory to accelerate host-to-device transfers.
    
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
    compile_mode: Literal["default", "reduce-overhead", "max-autotune","max-autotune-no-cudagraphs"] = "max-autotune-no-cudagraphs"
    
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

    # DataLoader
    drop_last: bool = True
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory:bool = True