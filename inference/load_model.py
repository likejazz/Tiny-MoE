from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from inference.inference_configs import GenerationConfig, ModelConfig 
from inference.model_inference import Transformer

def load_and_prepare_model(cfg: GenerationConfig, device: torch.device):
    """
    Downloads the appropriate weights from Hugging Face, configures 
    the model settings based on the run mode, and returns the ready model.
    """
    repo_id = "AbdelrhmanEbied/Tiny-MoE"
    

    config = ModelConfig()
    

    if cfg.model_variant == "base-yarn":
        print("\nDownloading YaRN model...")
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="base/yarn/model.safetensors"
        )
        state_dict = load_file(weights_path)
        print("YaRN weights downloaded successfully.")
        

        config.factor = 4
        config.rope_type = "yarn"
        config.max_seq_len = 2048
    elif cfg.model_variant == "base":
        print("\nDownloading base model...")
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="base/model.safetensors"
        )
        state_dict = load_file(weights_path)
        print("Base weights downloaded successfully.")
        config.factor = 1
        config.rope_type = "default"
        config.max_seq_len = 512
    elif cfg.model_variant =="fine-tuned":       
        print("\nDownloading FineTuned model...")
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename="fine-tuned/model.safetensors"
        )
        state_dict = load_file(weights_path)
        print("YaRN weights downloaded successfully.")
        

        config.factor = 4
        config.rope_type = "yarn"
        config.max_seq_len = 2048
        
    model = Transformer(config) 
    

    model.load_state_dict(state_dict, strict=False)
    

    model = model.to(device)
    model.eval() 
    
    return model