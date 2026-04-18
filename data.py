from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") 

from huggingface_hub import login
login(token=hf_token)
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

class PackedStreamingDataset(IterableDataset):
    """
    An efficient streaming dataset that interleaves multiple sources and packs tokens.

    This class handles the 'token packing' problem by maintaining a buffer of 
    tokenized text, yielding full sequences of 'max_seq_len' to ensure 
    zero-padding training.

    Args:
        config (ModelConfig): Configuration object containing 'max_seq_len'.
        tokenizer (PreTrainedTokenizer): The tokenizer to convert text to IDs.
        total_tokens (int): The hard limit for the number of tokens to yield.

    Yields:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): Packed token IDs of length max_seq_len.
            - 'position_ids' (torch.Tensor): Sequential indices from 0 to max_seq_len-1.
            - 'labels' (torch.Tensor): Same as input_ids (for causal language modeling).
    """
    def __init__(self, config, tokenizer, total_tokens: int = 20_000_000_00):
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len   
        self.total_tokens = total_tokens
        
        # 60% Web
        ds_web = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        
        # 25% Code 
        ds_code = load_dataset(
            "HuggingFaceTB/cosmopedia",
            name="web_samples_v1",
            split="train",
            streaming=True
        )
        
        # 15% Math
        ds_math = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True
        )
        

        self.dataset = interleave_datasets(
            [ds_web, ds_code, ds_math],
            probabilities=[0.60, 0.25, 0.15],
            stopping_strategy="first_exhausted"
        )
        
        self.token_buffer = []
        self.tokens_yielded = 0

    def __iter__(self):
        for example in self.dataset:
            
            text = (example.get("text") or 
                    example.get("content") or 
                    str(example))
            
            
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False
            )["input_ids"]
            
            self.token_buffer.extend(tokens)
            
            if len(self.token_buffer) > (self.max_seq_len * 80):
                while len(self.token_buffer) >= self.max_seq_len:
                    chunk = self.token_buffer[:self.max_seq_len]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    position_ids = torch.arange(self.max_seq_len, dtype=torch.long)
                    
                    yield {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}
                    
                    self.token_buffer = self.token_buffer[self.max_seq_len:]
                    self.tokens_yielded += self.max_seq_len
            
            
            
            while len(self.token_buffer) >= self.max_seq_len:
                chunk = self.token_buffer[:self.max_seq_len]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                position_ids = torch.arange(self.max_seq_len, dtype=torch.long)
                
                yield {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}
                
                self.token_buffer = self.token_buffer[self.max_seq_len:]
                self.tokens_yielded += self.max_seq_len
                
                if self.tokens_yielded >= self.total_tokens:
                    return