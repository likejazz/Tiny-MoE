import os
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator,DeepSpeedPlugin
from tqdm import tqdm
import bitsandbytes as bnb
from model import Transformer
from data import Training_Streaming_Dataset,Eval_Streaming_Dataset
from transformers import AutoTokenizer
from safetensors.torch import save_file
from helpers import build_deepspeed_config,has_checkpoint,prepare_checkpoint_from_dataset,get_lr,compute_loss,evaluate,build_accelerator
from TrainingConfigs import TrainConfig,ModelConfig



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

    accelerator = build_accelerator(cfg)
    device = accelerator.device
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="Tiny-MoE-200M",
            config=vars(cfg)
        )
        
    accelerator.wait_for_everyone()

   

    if cfg.activation_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            accelerator.print("Gradient Checkpointing enabled.")
        else:
            accelerator.print("Warning: Model doesn't support gradient checkpointing.")

    if cfg.optimizer_type == "AdamW":
        optimizer =torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        fused=True
        )
        accelerator.print('Using AdamW')
    elif cfg.optimizer_type == "Paged":
        optimizer =bnb.optim.PagedAdamW8bit(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay
        )
        accelerator.print('Using PagedAdamW')
    elif cfg.optimizer_type == "8bitAdamW":
        optimizer =bnb.optim.AdamW8bit(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay
        )
        accelerator.print('Using 8bitAdamW')


    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    
    prepare_checkpoint_from_dataset(cfg)
    accelerator.wait_for_everyone()     
    best_loss=float("inf")
    best_step=0
    tokens_seen=0

    step = 0
    checkpoint_dir = os.path.join(cfg.out_dir, "best")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    
    regular_ckpt_dir = os.path.join(cfg.out_dir, "regular")
    os.makedirs(regular_ckpt_dir, exist_ok=True)
    

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
    

    if resume_path is not None:
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
                f"Resumed at step {step} | best_loss {best_loss:.4f} | best_step {best_step} | tokens_seen {tokens_seen}"
            )
    else:
        accelerator.print("No checkpoint found — starting from scratch.")
        
    if cfg.compile_model:
        accelerator.print("Compiling Now...")
        accelerator.print(f"Compiling mode is {cfg.compile_mode}")
        model=torch.compile(model=model,mode=cfg.compile_mode,dynamic=True,fullgraph=False)
        
    model.train()
    progress_bar = tqdm(total=cfg.num_train_steps,initial=step, disable=not accelerator.is_main_process)
    data_iter = iter(train_dataloader) 

    while step < cfg.num_train_steps:
        step_start=time.perf_counter()
        
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
            
        with accelerator.accumulate(model):
            local_tokens = batch["input_ids"].numel()
            input_ids = batch["input_ids"].to(device,non_blocking=True)
            labels = batch.get("labels", input_ids).to(device,non_blocking=True)
            position_ids = batch["position_ids"].to(device, non_blocking=True)
            
            if cfg.compile_mode in ["reduce-overhead", "max-autotune"]:
                torch.compiler.cudagraph_mark_step_begin()
                
            outputs = model(input_ids,position_ids=position_ids)
            loss = compute_loss(outputs, labels, cfg)
            

            
            accelerator.backward(loss)


            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                
                token_tensor = torch.tensor(local_tokens, dtype=torch.long, device=device)
                global_step_tokens = accelerator.reduce(token_tensor, reduction="sum").item()
                tokens_seen += global_step_tokens*cfg.grad_accum_steps
                
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
                    val_loss, val_ppl = evaluate(model, val_dataloader, cfg, accelerator)
                    accelerator.wait_for_everyone()
                    
                is_best=current_loss<best_loss
                if is_best:
                    best_loss=current_loss
                    best_step=step
                        
                if step % cfg.save_interval == 0:
                    accelerator.print(f"Saving checkpoint at step {step}...")
                    accelerator.wait_for_everyone()
                    
                    save_path = os.path.join(regular_ckpt_dir, f"step_{step}")
                    if accelerator.is_main_process:
                        os.makedirs(save_path, exist_ok=True)
                    
  
                    accelerator.save_state(save_path, safe_serialization=True)
                    

                    state_dict = accelerator.get_state_dict(model, unwrap=True)
                    
                    if accelerator.is_main_process:
                        torch.save({
                            "step": step,
                            "best_step": best_step,
                            "loss": current_loss,
                            "best_loss" : best_loss,
                            "tokens_seen": tokens_seen, 
                        }, os.path.join(save_path, "metadata.pt"))
                        
                        clean_state_dict = {}
                        pointer_map = {}
                        
                        for key, tensor in state_dict.items():
                            clean_key = key.replace("_orig_mod.", "").replace("module.", "")
                            
                            if not tensor.is_contiguous():
                                tensor = tensor.contiguous()
                                    
                            ptr = tensor.data_ptr()
                            
                            if ptr in pointer_map:
                                clean_state_dict[clean_key] = tensor.clone()
                                accelerator.print(f"Untying Cloned shared storage for key: {clean_key}")
                            else:
                                pointer_map[ptr] = clean_key
                                clean_state_dict[clean_key] = tensor
                            
                        save_file(clean_state_dict, os.path.join(save_path, "model.safetensors"))
                        accelerator.print(f"checkpoint successfully saved at {save_path}")
                        
                    accelerator.wait_for_everyone()

                
                        
                
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
                        "grad_norm": grad_norm,
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
                    
                    if "router_metrics" in outputs and outputs["router_metrics"] is not None:
                        for metric_name, metric_tensor in outputs["router_metrics"].items():
                            gathered_metric = accelerator.gather(metric_tensor)
                            log_data[metric_name] = gathered_metric.mean().item()
                    if accelerator.is_main_process:
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token == tokenizer.eos_token
    train_dataset = Training_Streaming_Dataset(config, tokenizer,seed = 800813,shuffle_buffer_size = 50_000)
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=cfg.micro_batch_size,
                            drop_last=True,
                            num_workers=2,
                            prefetch_factor=2,
                            persistent_workers=True,
                            pin_memory=True)
    val_dataset = Eval_Streaming_Dataset(config, 
                                         tokenizer,
                                         eval_samples=5000)
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=cfg.micro_batch_size,
                        drop_last=True,
                        num_workers=0,
                        pin_memory=True)
                    
    train(model, train_dataloader, val_dataloader, cfg)
