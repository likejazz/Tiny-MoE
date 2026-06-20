import os
import math
import torch
import torch.nn.functional as F
from accelerate import Accelerator,DeepSpeedPlugin
from typing import Dict,Any
from TrainingConfigs import TrainConfig
from torch.utils.data import DataLoader
import shutil
def build_deepspeed_config(cfg: TrainConfig) -> Dict[str, Any]:


    zero_config: Dict[str, Any] = {
        "stage": cfg.zero_stage,
        "overlap_comm": cfg.overlap_comm,
        "contiguous_gradients": cfg.contiguous_gradients,
        "reduce_bucket_size": cfg.reduce_bucket_size,
    }


    if cfg.zero_stage == 1:
        zero_config.update({
            "reduce_scatter": True,
        })

    elif cfg.zero_stage == 2:
        zero_config.update({
            "allgather_partitions": True,
            "allgather_bucket_size": cfg.reduce_bucket_size,
        })

    elif cfg.zero_stage == 3:
        zero_config.update({
            "stage3_prefetch_bucket_size": cfg.reduce_bucket_size // 2,
            "stage3_param_persistence_threshold": 1_000_000,
        })

        if cfg.offload_param:
            zero_config["offload_param"] = {"device": "cpu", "pin_memory": True}


    if cfg.zero_stage >= 2 and cfg.offload_optimizer:
        zero_config["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    ds_config: Dict[str, Any] = {
        "train_micro_batch_size_per_gpu": cfg.micro_batch_size,
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "gradient_clipping": cfg.grad_clip,
        "wall_clock_breakdown": cfg.wall_clock_breakdown,
        "zero_optimization": zero_config,
        "activation_checkpointing": {
            "partition_activations": cfg.partition_activations,
            "cpu_checkpointing": cfg.cpu_checkpointing,
            "contiguous_memory_optimization": cfg.contiguous_memory_optimization,
            "number_checkpoints": cfg.num_checkpoints,
        },
    }

    if cfg.moe_enabled:
        ds_config["moe"] = {
            "enabled": True,
            "ep_size": cfg.ep_size,
            "moe_param_group": cfg.moe_param_group,
            "use_residual": cfg.use_residual,
        }

    if cfg.mixed_precision == "fp16":
        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": cfg.loss_scale_window,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        }
    elif cfg.mixed_precision == "bf16":
        ds_config["bf16"] = {"enabled": True}
    else:
        ds_config["fp16"] = {"enabled": False}
        ds_config["bf16"] = {"enabled": False}

    return ds_config

def has_checkpoint(path):
    return (
        os.path.exists(os.path.join(path, "model.safetensors")) or
        os.path.exists(os.path.join(path, "pytorch_model.bin"))
    )

def prepare_checkpoint_from_dataset(cfg: TrainConfig):
    """
    Finds and copies the latest model checkpoint from a read-only dataset to the local working directory.

    Args:
        cfg (TrainConfig): Configuration object containing 'out_dir'.

    Returns:
        str or None: The local path to the copied checkpoint, or None if no checkpoint exists.
    """

    SRC_BASE = "/kaggle/input/datasets/abdelrhmanebied/model-checkpoint/checkpoints/regular"
    DST_BASE = os.path.join(cfg.out_dir, "regular")
    os.makedirs(DST_BASE, exist_ok=True)

    if not os.path.exists(SRC_BASE):
        print("No dataset checkpoint found, starting fresh.")
        return None

    subdirs = [d for d in os.listdir(SRC_BASE) if d.startswith("step_")]
    if not subdirs:
        print("No checkpoints inside dataset.")
        return None

    latest = max(subdirs, key=lambda x: int(x.split("_")[-1]))

    SRC_PATH = os.path.join(SRC_BASE, latest)
    DST_PATH = os.path.join(DST_BASE, latest)

    if not os.path.exists(DST_PATH):
        print(f"Copying {latest} → working dir...")
        shutil.copytree(SRC_PATH, DST_PATH, dirs_exist_ok=True)
    else:
        print(f"{latest} already exists in working dir")

    print(f"Checkpoint ready at: {DST_PATH}")
    return DST_PATH

def get_lr(step, cfg: TrainConfig):
    """
    Calculates the learning rate for a specific training step.

    Args:
        step (int): The current training step/iteration.
        cfg (TrainConfig): Configuration object containing 'lr', 'warmup_steps', 
                          and 'num_train_steps'.

    Returns:
        float: The calculated learning rate for the given step.
    """

    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / (cfg.num_train_steps - cfg.warmup_steps)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * progress))


def compute_loss(outputs, targets, cfg: TrainConfig):
    """
    Computes the cross-entropy loss with MoE stabilization penalties.

    Args:
        outputs (dict): Model output dictionary containing 'logits' [B, T, V], 
                        and optional 'aux_loss' and 'z_loss' tensors.
        targets (torch.Tensor): Ground truth token IDs of shape [B, T].
        cfg (TrainConfig): Config object with 'router_aux_loss_coef' and 
                          'router_z_loss_coef' scaling factors.

    Returns:
        torch.Tensor: The total combined scalar loss.
    """
    logits = outputs["logits"]
    aux_loss = outputs.get("aux_loss", 0.0)
    z_loss = outputs.get("z_loss", 0.0)
    

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    loss = loss + cfg.router_aux_loss_coef * aux_loss \
                 + cfg.router_z_loss_coef * z_loss

    return loss



def evaluate(model, val_dataloader: DataLoader, cfg: TrainConfig, accelerator):
    """
    Evaluates the model on a validation set to compute loss and perplexity.

    Args:
        model (nn.Module): The model to evaluate.
        val_dataloader (DataLoader): Iterator providing validation batches.
        cfg (TrainConfig): Configuration for evaluation steps and loss coefficients.
        accelerator (Accelerator): HF Accelerator for distributed reduction and device management.

    Returns:
        tuple (float, float): A tuple containing the global average (validation loss, perplexity).
    """

    model.eval()
    model = accelerator.unwrap_model(model)
    data_iter = iter(val_dataloader)
    
    device = accelerator.device
    local_loss_sum = torch.tensor(0.0, device=device)
    local_count = torch.tensor(0.0, device=device)
    
    with torch.inference_mode():
   
        for step, batch in enumerate(data_iter):
            if step >= cfg.num_eval_steps:
                break

    
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch.get("labels", input_ids).to(device, non_blocking=True)

     
            if "position_ids" in batch:
                position_ids = batch["position_ids"].to(device, non_blocking=True)
            else:
 
                batch_size, seq_len = input_ids.shape
                position_ids = torch.arange(seq_len, device=accelerator.device).unsqueeze(0).expand(batch_size, -1)\
                
            outputs = model(input_ids, position_ids=position_ids)
            loss = compute_loss(outputs, labels, cfg)
            
            batch_size = input_ids.size(0)
            local_loss_sum += loss.detach() * batch_size
            local_count += batch_size

    global_loss_sum = accelerator.reduce(local_loss_sum, reduction="sum")
    global_count = accelerator.reduce(local_count, reduction="sum")

    val_loss = (global_loss_sum / global_count).item()
    val_ppl = math.exp(min(val_loss, 20))

    model.train()
    return val_loss, val_ppl
    
def build_accelerator(cfg: TrainConfig) -> Accelerator:
    ds_plugin = None

    if cfg.deepspeed_enabled:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=build_deepspeed_config(cfg))

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb",
        deepspeed_plugin=ds_plugin,
    )
    return accelerator