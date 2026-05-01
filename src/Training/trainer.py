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
import bitsandbytes as bnb
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from model import Transformer, ModelConfig
from data import Training_Streaming_Dataset,Eval_Streaming_Dataset
from transformers import AutoTokenizer
import shutil


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
        optimizer_type (str): Sets the Optimizer type (PagedAdamW | AdamW |8bit Adam W) default for AdamW Paged For PagedAdamW else 8bit AdamW.
        
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
    lr = 1e-3
    weight_decay = 0.01
    betas = (0.9, 0.95)
    grad_clip = 1.0
    num_train_steps = 45776
    warmup_steps = 2500
    optimizer_type='8bitAdamW'

    # Batch / Throughput
    micro_batch_size = 32
    grad_accum_steps =4
    max_seq_len = 512

    # Precision / Performance
    mixed_precision = "fp16"
    compile_model = True
    compile_type = "default"

    # Memory
    activation_checkpointing = True

    # MoE Stability
    router_aux_loss_coef = 0.002
    router_z_loss_coef = 1e-4
    capacity_factor = 1.1
    top_k = 2

    # Logging
    save_interval=700
    num_eval_steps = 50
    eval_interval = 1000
    save_best_model = False
    log_interval = 20
    out_dir = "/kaggle/working/checkpoints"
# Helper Functions 
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
    
    shift_targets = targets[:, 1:]

    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        shift_targets.reshape(-1),
        ignore_index=-100
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
    model= accelerator.unwrap_model(model)
 
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
    
def train(model,train_dataloader:DataLoader,val_dataloader:DataLoader,cfg: TrainConfig):
    """
    Executes a distributed training loop with support for MoE-specific logging, 
    mixed precision, and automated checkpoint management.

    This function handles the end-to-end training process including optimizer 
    initialization (supporting 8-bit variants), model compilation, gradient 
    accumulation, and periodic evaluation. It also features a robust resume 
    mechanism that automatically detects the latest or best checkpoint.

    Args:
        model (torch.nn.Module): The transformer model to be trained.
        train_dataloader (DataLoader): Iterable yielding training batches.
        val_dataloader (DataLoader): Iterable yielding validation batches.
        cfg (TrainConfig): Configuration object containing hyperparameters such as 
            learning rate, weight decay, optimization type, checkpoint intervals, 
            and MoE-specific settings.

    Side Effects:
        - Initializes a WandB run and logs metrics (loss, perplexity, router load, etc.).
        - Saves model weights and optimizer states to `cfg.out_dir`.
        - Prints training progress to the console via the main process accelerator.
        - Enables gradient checkpointing and torch compilation on the model if configured.

    Note:
        The function specifically calculates MoE metrics if the model outputs 
        `router_logits`, including expert load balancing, entropy, and utilization 
        to monitor for expert collapse.
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
            project_name="Tiny-MoE-2",
            config=vars(cfg)
        )
        
    accelerator.wait_for_everyone()

   

    if cfg.activation_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            accelerator.print("Gradient Checkpointing enabled.")
        else:
            accelerator.print("Warning: Model doesn't support gradient checkpointing.")

    if cfg.optimizer_type == "default":
        optimizer =torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        fused=True
        )
        print('Using AdamW')
    elif cfg.optimizer_type== "Paged":
        optimizer =bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay
        )
        print('Using PagedAdamW')
    else:
        optimizer =bnb.optim.AdamW8bit(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay
        )
        print('Using 8bitAdamW')


    base_model = model 
    if cfg.compile_model:
        accelerator.print("Compiling model...")
        accelerator.print(f"Compiling Method is {cfg.compile_type}")
        model = torch.compile(base_model,mode=cfg.compile_type,dynamic=True, fullgraph=False)

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )


    eval_model=base_model
    prepare_checkpoint_from_dataset(cfg)
    best_loss=float("inf")
    best_step=0
    tokens_seen=0
    model.train()
    step = 0
    checkpoint_dir = os.path.join(cfg.out_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    regular_ckpt_dir = os.path.join(cfg.out_dir, "regular")
    os.makedirs(regular_ckpt_dir, exist_ok=True)
    
    def has_checkpoint(path):
        return (
            os.path.exists(os.path.join(path, "model.safetensors")) or
            os.path.exists(os.path.join(path, "pytorch_model.bin"))
        )

    resume_path = None
    

    if has_checkpoint(checkpoint_dir):
        resume_path = checkpoint_dir
    

    elif os.path.exists(regular_ckpt_dir):
        subdirs = [
            os.path.join(regular_ckpt_dir, d)
            for d in os.listdir(regular_ckpt_dir)
            if d.startswith("step_")
        ]
    
        if subdirs:
            latest = max(subdirs, key=lambda x: int(x.split("_")[-1]))
            if has_checkpoint(latest):
                resume_path = latest
    

    if resume_path:
        accelerator.print(f"Resuming from: {resume_path}")
        accelerator.load_state(resume_path)
    
        meta_path = os.path.join(resume_path, "metadata.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
    
            step = meta["step"]
            tokens_seen = meta["tokens_seen"]
    
            best_step = meta.get("best_step", step)
            best_loss = meta.get("best_loss", meta.get("loss", float("inf")))
    
            accelerator.print(
                f"Resumed at step {step} | best_loss {best_loss:.4f}"
            )
    else:
        accelerator.print("No checkpoint found — starting from scratch.")
    
    progress_bar = tqdm(total=cfg.num_train_steps,initial=step, disable=not accelerator.is_main_process)
    
    
    data_iter = iter(train_dataloader)

    while step < cfg.num_train_steps:
        if cfg.compile_type == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        step_start=time.perf_counter()
        with accelerator.accumulate(model):
            
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)
                
            local_tokens = batch["input_ids"].numel()
            input_ids = batch["input_ids"].to(device,non_blocking=True)
            
            labels = batch.get("labels", input_ids).to(device,non_blocking=True)

            curr_bsz, curr_seq_len = input_ids.shape 


            pos_base = torch.arange(curr_seq_len, device=device)

            position_ids = pos_base.unsqueeze(0).expand(curr_bsz, -1) 

            
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
                perplexity = math.exp(min(current_loss, 20))
                step_time = time.perf_counter() - step_start
                tps = global_step_tokens / step_time
                accelerator.wait_for_everyone()
                if step % cfg.eval_interval == 0:
                    accelerator.print("Evaluating Now...")
                    accelerator.wait_for_everyone()
                    val_loss, val_ppl = evaluate(eval_model, val_dataloader, cfg, accelerator)
                    accelerator.wait_for_everyone()
                    
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
                            "best_step": best_step,
                            "loss": current_loss,
                            "best_loss": best_loss,
                            "tokens_seen": tokens_seen,  
                        }, os.path.join(checkpoint_dir, "metadata.pt"))
                        
                if step % cfg.save_interval == 0:


                    accelerator.print(f"Saving checkpoint at step {step}...")
                    
                    
                    accelerator.wait_for_everyone()
                    
                    
                    save_path = os.path.join(regular_ckpt_dir, f"step_{step}")
                    accelerator.save_state(save_path, safe_serialization=True)
                    
                    
                    if accelerator.is_main_process:
                        torch.save({
                            "step": step,
                            "best_step": best_step,
                            "loss": current_loss,
                            "best_loss" : best_loss,
                            "tokens_seen": tokens_seen, 
                        }, os.path.join(save_path, "metadata.pt"))
                        accelerator.print(f" Checkpoint saved at {save_path}")
                
                        

                
                if accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "loss": f"{current_loss:.4f}", 
                        "sec/step": f"{step_time:.2f}",
                        "Perplex" : f"{perplexity:.4f}",
                    }) 
                
                
                if step % cfg.log_interval == 0:
                    accelerator.print(
                        f"Step {step} | Loss: {current_loss:.4f} | Tokens: {tokens_seen:,} | LR: {lr:.2e} | Perplexity : {perplexity:.4f}"
                    )
                
                    log_data = {
                        "loss": current_loss,
                        "best_loss": best_loss,
                        "lr": lr,
                        "perplexity":perplexity,
                        "tokens_seen": tokens_seen,
                        "step": step,
                        "best_step": best_step,
                        "step_time": step_time,
                        "tps": tps,
                        "peak_Vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                    }
                
                    if step % cfg.eval_interval == 0:
                        log_data["val_loss"] = val_loss
                        log_data["val_ppl"] = val_ppl
                    if "aux_loss" in outputs and outputs["aux_loss"] is not None:
                        raw_aux = outputs["aux_loss"].detach()
                        log_data["aux_loss"] = accelerator.gather(raw_aux).mean().item()
                    
                    if "z_loss" in outputs and outputs["z_loss"] is not None:
                        raw_z = outputs["z_loss"].detach()
                        log_data["z_loss"] = accelerator.gather(raw_z).mean().item()
                    
                    if "router_logits" in outputs and outputs["router_logits"] is not None:
                        logits = accelerator.gather(outputs["router_logits"]).detach()
                        
                        real_model = model.module if hasattr(model, "module") else model
                        
                        num_experts = real_model.config.num_experts
                        top_k = real_model.config.num_experts_per_token
                        topk_indices = torch.topk(
                            logits, top_k, dim=-1
                        ).indices
                
                        chosen_experts = topk_indices.reshape(-1)


                        counts = torch.bincount(
                            chosen_experts, minlength=num_experts
                        ).float()
                
                        load = counts / (counts.sum() + 1e-9)
                        
                
                        log_data["router/load_std"] = load.std().item()
                        log_data["router/load_max"] = load.max().item()
                        log_data["router/load_min"] = load.min().item()
                        log_data["router/load_ratio"] = (load.max() / (load.min() + 1e-9)).item()
                

                        entropy = -(load * (load + 1e-9).log()).sum()
                        log_data["router/entropy"] = entropy.item()
                        utilization=torch.exp(entropy)/num_experts
                        log_data["router/utilization"] = utilization.item()
                        probs = torch.softmax(logits, dim=-1)
                        confidence = probs.max(dim=-1).values.mean()
                        log_data["router/confidence"] = confidence.item()
                        
                    if accelerator.is_main_process:
                     accelerator.log(log_data, step=step)
                    torch.cuda.reset_peak_memory_stats(device)

    
                    
        
                    
    
    progress_bar.close()
    accelerator.wait_for_everyone() 
    accelerator.end_training()
    accelerator.print("Training complete.")
    
    
    
if __name__ == "__main__":
    cfg = TrainConfig()
    config = ModelConfig()
    model = Transformer(config)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    train_dataset = Training_Streaming_Dataset(config, tokenizer)
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=cfg.micro_batch_size,
                            drop_last=True,
                            num_workers=2,
                            prefetch_factor=2,
                           persistent_workers=True,
                            pin_memory=True)
    val_dataset = Eval_Streaming_Dataset(config, 
                                         tokenizer,
                                         eval_samples=1200)
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=cfg.micro_batch_size,
                        drop_last=True,
                        num_workers=0,
                        pin_memory=True)
                    
    train(model, train_dataloader, val_dataloader, cfg)
