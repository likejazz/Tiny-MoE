#  Imports
import os
import time
import math
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from accelerate import Accelerator,DistributedDataParallelKwargs
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from model import Transformer, ModelConfig
from data import PackedStreamingDataset
from transformers import AutoTokenizer

#  Authentication
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import wandb
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") 
login(token=hf_token)

wandb_key = user_secrets.get_secret("WANDB_API_KEY")

    



"""
Hardware & Backend Initialization:
- Disables NCCL P2P/IB to prevent hangs in specific distributed environments (like Kaggle/Colab).
- Optimizes CUDA memory fragmentation using expandable segments.
- Enables Flash Attention and high-precision matmuls for faster training.
"""
# Networking Workarounds (fixes 'NCCL timeout' or 'P2P' errors)
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Memory Management (prevents OOM by managing fragmented chunks)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"
# Backend & Logging Stability
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"
# Attention Kernel Selection (Flash Attention > Memory Efficient > Math)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)   
# Tensor Core Optimization (Balances speed and numerical precision)
torch.set_float32_matmul_precision('high')

class TrainConfig:
    """
    Configuration for MoE model training, optimization, and hardware performance.

    Attributes:
        lr (float): Peak learning rate for the cosine schedule.
        weight_decay (float): L2 regularization coefficient.
        betas (Tuple[float, float]): Adam optimizer momentum coefficients.
        grad_clip (float): Maximum norm for gradient clipping to prevent explosions.
        num_train_steps (int): Total number of iterations to train for.
        warmup_steps (int): Number of steps for the initial linear LR ramp-up.
        
        micro_batch_size (int): Number of samples per GPU per forward pass.
        grad_accum_steps (int): Number of steps to accumulate gradients before updating.
        max_seq_len (int): Maximum context window size for input sequences.
        
        mixed_precision (str): Type of precision (e.g., 'fp16', 'bf16', 'no').
        compile_model (bool): If True, uses torch.compile for faster execution.
        compile_type (str): Optimization mode for the compiler (e.g., 'reduce-overhead').
        
        activation_checkpointing (bool): Saves VRAM by recomputing activations during backward.
        
        router_aux_loss_coef (float): Penalty strength for expert load imbalance.
        router_z_loss_coef (float): Penalty strength to keep router logits stable.
        capacity_factor (float): Buffer ratio for tokens assigned to an expert.
        top_k (int): Number of experts activated per token.
        
        save_interval (int): How often (in steps) to save model checkpoints.
        save_best_model (bool): If True, tracks and saves the state with the lowest loss.
        log_interval (int): How often (in steps) to log metrics to the console/WandB.
        out_dir (str): Path to the directory where checkpoints are stored.
    """
    # Optimization
    lr = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    grad_clip = 1.0
    num_train_steps = 7630
    warmup_steps = 760

    # Batch / Throughput
    micro_batch_size = 16
    grad_accum_steps = 8
    max_seq_len = 1024

    # Precision / Performance
    mixed_precision = "fp16"
    compile_model = True
    compile_type = "default"

    # Memory
    activation_checkpointing = False

    # MoE Stability
    router_aux_loss_coef = 0.01
    router_z_loss_coef = 1e-4
    capacity_factor = 1.1
    top_k = 2

    # Logging
    save_interval=100
    save_best_model = False
    log_interval = 10
    out_dir = "/kaggle/working/checkpoints"
    
# Helper Functions 
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
    
    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        ignore_index=-100
    )

    loss = loss + cfg.router_aux_loss_coef * aux_loss \
                 + cfg.router_z_loss_coef * z_loss

    return loss




def train(model,dataloader:DataLoader,cfg: TrainConfig):
    """
    Main training loop utilizing HF Accelerator for distributed MoE training.

    Handles mixed precision, gradient accumulation, model compilation, 
    checkpointing (regular and best), and telemetry logging to WandB.

    Args:
        model (torch.nn.Module): The transformer model to train (typically an MoE).
        dataloader (DataLoader): PyTorch DataLoader providing batches of tokenized data.
        cfg (TrainConfig): Configuration object containing hyperparameters and 
                          environment settings.

    Returns:
        None: The function manages state internally and saves checkpoints to disk.
    """
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    
    device = accelerator.device
    if accelerator.is_main_process:
     wandb.login(key=wandb_key)
        
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="Tiny-MoE",
            config=vars(cfg)
        )
        
    accelerator.wait_for_everyone()

   

    if cfg.activation_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            accelerator.print("Gradient Checkpointing enabled.")
        else:
            accelerator.print("Warning: Model doesn't support gradient checkpointing.")

    
    optimizer =torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    betas=cfg.betas,
    weight_decay=cfg.weight_decay,
    fused=True
    )

    
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )
    
    if cfg.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model,mode=cfg.compile_type)

    best_loss=float("inf")
    best_step=0
    tokens_seen=0
    model.train()
    step = 0
    checkpoint_dir = os.path.join(cfg.out_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    regular_ckpt_dir = os.path.join(cfg.out_dir, "regular")
    os.makedirs(regular_ckpt_dir, exist_ok=True)
    
    resume_path=checkpoint_dir if os.path.exists(checkpoint_dir) else regular_ckpt_dir
    if os.path.exists(resume_path) and os.path.exists(os.path.join(resume_path, "pytorch_model.bin")):
        accelerator.print(f"Resuming from best checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        
        
        meta_path = os.path.join(resume_path, "metadata.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            step = meta["step"]
            best_loss = meta["loss"]
            tokens_seen = meta["tokens_seen"]
            accelerator.print(f"Resumed at step {step} | best_loss {best_loss:.4f}")
    else:
        accelerator.print("No checkpoint found — starting from scratch.")
    
    progress_bar = tqdm(total=cfg.num_train_steps,initial=step, disable=not accelerator.is_main_process)
    
    
    data_iter = iter(dataloader)

    while step < cfg.num_train_steps:
        if cfg.compile_type == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        step_start=time.perf_counter()
        with accelerator.accumulate(model):
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            local_tokens = batch["input_ids"].numel()
            input_ids = batch["input_ids"].to(device,non_blocking=True)
            
            labels = batch.get("labels", input_ids).to(device,non_blocking=True)
            pos_base = batch.get("position_ids", torch.arange(cfg.max_seq_len, device=device))
            position_ids = pos_base.view(1, -1).expand(input_ids.shape[0], -1)
            
            
            outputs = model(input_ids,position_ids=position_ids)
            loss = compute_loss(outputs, labels, cfg)
            

            
            accelerator.backward(loss)


            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                
                token_tensor = torch.tensor(local_tokens, dtype=torch.long, device=device)
                global_step_tokens = accelerator.reduce(token_tensor, reduction="sum").item()
                tokens_seen += global_step_tokens
                
                lr = get_lr(step, cfg)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                    
                current_loss=accelerator.gather(loss).mean().item()
                step_time = time.perf_counter() - step_start

                is_best=current_loss<best_loss
                if is_best:
                    best_loss=current_loss
                    best_step=step
                
                
                if cfg.save_best_model and is_best:
                    
                    accelerator.print(f"New best loss {best_loss:.4f} at step {step} → saving...")
                    accelerator.save_state(checkpoint_dir)
                    
                    if accelerator.is_main_process:
                        torch.save({
                            "step": step,
                            "loss": best_loss,
                            "tokens_seen": tokens_seen,  
                        }, os.path.join(checkpoint_dir, "metadata.pt"))
                        
                if step % cfg.save_interval == 0:
                    if is_best:

                        accelerator.print(f"Step {step}: Loss improved to {best_loss:.4f}. Saving checkpoint...")
                        
                        
                        accelerator.wait_for_everyone()
                        
                        
                        save_path = os.path.join(regular_ckpt_dir, f"step_{step}")
                        accelerator.save_state(save_path, safe_serialization=True)
                        
                        
                        if accelerator.is_main_process:
                            torch.save({
                                "step": step,
                                "loss": best_loss,
                                "tokens_seen": tokens_seen, 
                            }, os.path.join(save_path, "metadata.pt"))
                    else:
                        accelerator.print(f"ℹ️ Step {step}: ({current_loss:.4f}) did not beat {best_loss:.4f}  No save.")
                        

                
                if accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}", 
                        "sec/step": f"{step_time:.2f}"
                    }) 
                
                
                if step % cfg.log_interval == 0:
                    accelerator.print(
                        f"Step {step} | Loss: {current_loss:.4f} | Tokens: {tokens_seen:,} | LR: {lr:.2e}")
                    
                    log_data = {
                        "loss": current_loss,
                        "best_loss": best_loss,
                        "lr": lr,
                        "tokens_seen":tokens_seen,
                        "step": step,
                        "best_step": best_step,
                        "step_time": step_time,
                        "peak_Vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }

                    
                    if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                        raw_aux = outputs.aux_loss
                        log_data["aux_loss"] = accelerator.gather(raw_aux).mean().item() if isinstance(raw_aux, torch.Tensor) else raw_aux

                    if hasattr(outputs, 'z_loss') and outputs.z_loss is not None:
                        raw_z = outputs.z_loss
                        log_data["z_loss"] = accelerator.gather(raw_z).mean().item() if isinstance(raw_z, torch.Tensor) else raw_z

                    
                    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:

                        logits = outputs.router_logits
                        

                        chosen_experts = torch.argmax(logits, dim=-1)
                        

                        unique_experts_used = torch.unique(chosen_experts)
                        
    
                        utilization = len(unique_experts_used) / config.num_experts
   
                        avg_util = accelerator.reduce(torch.tensor(utilization, device=device), reduction="mean").item()
                        log_data["router/utilization_pct"] = avg_util
                    
    
                    accelerator.log(log_data, step=step)

    
                    
        
                    
    
    progress_bar.close()
    accelerator.wait_for_everyone() 
    accelerator.end_training()
    accelerator.print("Training complete.")
    
    
    
if __name__ == "__main__":
    cfg = TrainConfig()
    config = ModelConfig()
    model = Transformer(config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = PackedStreamingDataset(config, tokenizer)
    dataloader = DataLoader(dataset, 
                            batch_size=cfg.micro_batch_size,
                            drop_last=True,
                            num_workers=2,
                            prefetch_factor=2,
                           persistent_workers=True,
                            pin_memory=True)
    train(model, dataloader, cfg)
