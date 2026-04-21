from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") 

from huggingface_hub import login
login(token=hf_token)
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset,Dataset
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from itertools import islice


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

class PackedStreamingDataset(IterableDataset):
    """
    An efficient streaming dataset that interleaves multiple sources and packs tokens.

    This class mixes data from three major sources (Web, Code, and Math) using 
    specified probabilities. It solves the 'padding' problem by concatenating 
    tokenized text into a continuous stream and carving out perfectly sized 
    chunks of 'max_seq_len'.

    Args:
        config (ModelConfig): Configuration containing 'max_seq_len'.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process raw text.
        total_tokens (int): The maximum number of tokens to process.
        split (str): Either "train" or "eval"; determines if the dataset 
                    shards for distributed training or skips/takes samples.
        eval_samples (int): The number of samples reserved for the evaluation split.

    Yields:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): A full chunk of tokens of length max_seq_len.
            - 'position_ids' (torch.Tensor): Sequential indices [0, ..., max_seq_len-1].
            - 'labels' (torch.Tensor): Identical to input_ids for next-token prediction.
    """


    def __init__(self, config, tokenizer, total_tokens: int,split : str, eval_samples : int ):
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len   
        self.total_tokens = total_tokens
        self.split=split
        
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
        

        raw_mixed = interleave_datasets(
            [ds_web, ds_code, ds_math],
            probabilities=[0.60, 0.25, 0.15],
            stopping_strategy="first_exhausted"
        )
        if split=="eval":
            self.dataset = raw_mixed.take(eval_samples)
        else:
            self.dataset = raw_mixed.skip(eval_samples).shuffle(seed=3)
        self.token_buffer = []
        self.tokens_yielded = 0

    def __iter__(self):
        if self.split == "train":
            accelerator = Accelerator()
            
            iterator = self.dataset.shard(
                num_shards=accelerator.num_processes, 
                index=accelerator.process_index
            )
        else:
          iterator=self.dataset
        for example in iterator:

            
            text = (example.get("text") or 
                    example.get("content") or 
                    str(example))
            
            
            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False
            )["input_ids"]
            tokens.append(self.tokenizer.eos_token_id)
            
            self.token_buffer.extend(tokens)
            
            while len(self.token_buffer) >= self.max_seq_len:
                chunk = self.token_buffer[:self.max_seq_len]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                position_ids = torch.arange(self.max_seq_len, dtype=torch.long)
                
                yield {"input_ids": input_ids, "position_ids": position_ids, "labels": input_ids}
                
                self.token_buffer = self.token_buffer[self.max_seq_len:]
                self.tokens_yielded += self.max_seq_len
            



