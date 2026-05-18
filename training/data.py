from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") 

from huggingface_hub import login
login(token=hf_token)
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset,get_worker_info
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator



tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
seed=3

class Training_Streaming_Dataset(IterableDataset):
    """
    An efficient streaming dataset that interleaves multiple sources and packs tokens.

    This class mixes data from three major sources (Web, Code, and Math) and 
    shards that stream across distributed workers. It concatenates tokenized 
    text into a continuous stream and yields chunks of 'max_seq_len'.

    Args:
        config (ModelConfig): Configuration object containing 'max_seq_len'.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process raw text 
                                         and provide the 'eos_token_id'.

    Yields:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): A full chunk of tokens of length max_seq_len.
            - 'position_ids' (torch.Tensor): Sequential indices [0, ..., max_seq_len-1].
            - 'labels' (torch.Tensor): Identical to input_ids for next-token prediction.
    """


    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len


        ds_web = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        ds_cosmopedia_web = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            name="cosmopedia-v2",
            split="train",
            streaming=True,
        )
        ds_math = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
        )
        
        ds_wiki = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )

        raw_mixed = interleave_datasets(
            [ds_web, ds_cosmopedia_web, ds_math,ds_wiki],
            probabilities=[0.57, 0.23, 0.14, 0.06],
            stopping_strategy="all_exhausted",
            seed=seed,
        )


        raw_mixed = raw_mixed.shuffle(seed=seed)
        self.traindataset=raw_mixed

    

    def __iter__(self):
        accelerator = Accelerator()
        process_index = accelerator.process_index
        num_processes = accelerator.num_processes

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers


        global_worker_id = process_index * num_workers + worker_id
        total_shards = num_processes * num_workers

        iterator = self.traindataset.shard(
            num_shards=total_shards,
            index=global_worker_id,
        )

        token_buffer = []
        for example in iterator:
            text = (
                example.get("text")
                or example.get("content")
                or example.get("article")
                or str(example)
            )

            tokens = self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            tokens.append(self.tokenizer.eos_token_id)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_seq_len + 1:
                full_chunk = token_buffer[: self.max_seq_len + 1]
                input_ids = torch.tensor(full_chunk[:-1], dtype=torch.long)
                labels = torch.tensor(full_chunk[1:], dtype=torch.long)
                position_ids = torch.arange(self.max_seq_len, dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "labels": labels,
                }

                token_buffer = token_buffer[self.max_seq_len:]

class Eval_Streaming_Dataset(IterableDataset):
    """
    A streaming dataset for model evaluation using Wikitext-103.

    This class loads the validation split of Wikitext, takes a fixed number 
    of samples, and packs them into continuous chunks of 'max_seq_len'.

    Args:
        config (ModelConfig): Configuration containing 'max_seq_len'.
        tokenizer (PreTrainedTokenizer): The tokenizer used to process raw text.
        eval_samples (int): The specific number of raw samples to take from 
                           the validation stream.

    Yields:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): A full chunk of tokens of length max_seq_len.
            - 'position_ids' (torch.Tensor): Sequential indices [0, ..., max_seq_len-1].
            - 'labels' (torch.Tensor): Identical to input_ids for next-token prediction.
    """

    def __init__ (self, config, tokenizer,eval_samples : int):
            super().__init__()
            self.tokenizer = tokenizer
            self.max_seq_len = config.max_seq_len
            self.eval_samples=eval_samples
        
            ds_val = load_dataset(
                "allenai/c4",
                "en",
                split="validation",
                streaming=True,
            )
            self.evaldataset=ds_val
    def __iter__(self):
        

            iterator = self.evaldataset.take(self.eval_samples)


            token_buffer = []
            for example in iterator:
                text = (
                    example.get("text")
                    or example.get("content")
                    or example.get("article")
                    or str(example)
                )

                tokens = self.tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                )["input_ids"]

                tokens.append(self.tokenizer.eos_token_id)
                token_buffer.extend(tokens)

                while len(token_buffer) >= self.max_seq_len + 1:
                    full_chunk = token_buffer[: self.max_seq_len + 1]
                    input_ids = torch.tensor(full_chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(full_chunk[1:], dtype=torch.long)
                    position_ids = torch.arange(self.max_seq_len, dtype=torch.long)

                    yield {
                        "input_ids": input_ids,
                        "position_ids": position_ids,
                        "labels": labels,
                    }

                    token_buffer = token_buffer[self.max_seq_len:]
            
        
