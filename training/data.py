from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN") 

from huggingface_hub import login
login(token=hf_token)
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from torch.utils.data import IterableDataset, get_worker_info
import torch
import torch.distributed as dist




tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

class Training_Streaming_Dataset(IterableDataset):
    """
    Streaming pretraining dataset that mixes multiple corpora, shards them across
    distributed workers, and packs contiguous token windows for next-token LM.

    The dataset:
    - streams from several Hugging Face sources,
    - interleaves them with fixed sampling probabilities,
    - applies buffer-based shuffling for approximate randomness,
    - shards the stream across ranks and DataLoader workers,
    - concatenates tokenized text into a continuous token buffer,
    - emits fixed-length training chunks of size `max_seq_len`.

    Parameters
    ----------
    config : ModelConfig
        Configuration object containing at least `max_seq_len`.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to convert raw text into token IDs.
    seed : int
        Base seed for interleaving and shuffling.
    shuffle_buffer_size : int, optional
        Buffer size used by the streaming shuffle. Larger values improve mixing
        at the cost of more RAM. Default is 50_000.

    Yields
    ------
    dict
        Dictionary with:
        - input_ids: torch.LongTensor of shape [max_seq_len]
        - position_ids: torch.LongTensor of shape [max_seq_len]
        - labels: torch.LongTensor of shape [max_seq_len]
    """

    def __init__(self, config, tokenizer, seed: int, shuffle_buffer_size: int = 50_000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

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

        mixed = interleave_datasets(
            [ds_web, ds_cosmopedia_web, ds_math, ds_wiki],
            probabilities=[0.57, 0.23, 0.14, 0.06],
            stopping_strategy="all_exhausted",
            seed=seed,
        )

        self.stream = mixed.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )

    def __iter__(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id
        total_shards = world_size * num_workers

        iterator = self.stream.shard(
            num_shards=total_shards,
            index=global_worker_id,
        )

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id must be defined for streaming packing.")

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

            tokens.append(eos_id)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_seq_len + 1:
                window = token_buffer[: self.max_seq_len + 1]

                yield {
                    "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                    "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
                    "labels": torch.tensor(window[1:], dtype=torch.long),
                }

                token_buffer = token_buffer[self.max_seq_len:]


class Eval_Streaming_Dataset(IterableDataset):
    """
    Deterministic streaming evaluation dataset that reads a fixed number of raw
    examples, shards them across distributed workers, and packs contiguous token
    windows for next-token LM evaluation.

    The dataset:
    - streams from the validation split of C4,
    - takes a fixed number of raw examples,
    - shards that fixed window across ranks and DataLoader workers,
    - concatenates tokenized text into a continuous buffer,
    - yields packed sequences of length `max_seq_len`.

    Parameters
    ----------
    config : ModelConfig
        Configuration object containing at least `max_seq_len`.
    tokenizer : PreTrainedTokenizer
        Tokenizer used to convert raw text into token IDs.
    eval_samples : int
        Number of raw validation examples to consume before packing.

    Yields
    ------
    dict
        Dictionary with:
        - input_ids: torch.LongTensor of shape [max_seq_len]
        - position_ids: torch.LongTensor of shape [max_seq_len]
        - labels: torch.LongTensor of shape [max_seq_len]
    """

    def __init__(self, config, tokenizer, eval_samples: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.eval_samples = eval_samples

        self.stream = load_dataset(
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )

    def __iter__(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id
        total_shards = world_size * num_workers

        iterator = self.stream.take(self.eval_samples).shard(
            num_shards=total_shards,
            index=global_worker_id,
        )

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id must be defined for streaming packing.")

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

            tokens.append(eos_id)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.max_seq_len + 1:
                window = token_buffer[: self.max_seq_len + 1]

                yield {
                    "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                    "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
                    "labels": torch.tensor(window[1:], dtype=torch.long),
                }

                token_buffer = token_buffer[self.max_seq_len:]