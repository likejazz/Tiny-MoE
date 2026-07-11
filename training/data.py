from datasets import load_dataset, interleave_datasets,Dataset
from torch.utils.data import IterableDataset, get_worker_info
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any, Iterator


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



        mixed = interleave_datasets(
            [ds_web, ds_cosmopedia_web, ds_math],
            probabilities=[0.60, 0.25, 0.15],
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






class BaseFinetuningDataset:
    def __init__(self, tokenizer, config, dataset_name: str,split = "train",text_field = "conversations",max_sample_length = 1800):
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_len
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.max_sample_length = max_sample_length

    def convert(self, sample):
        """
        Convert a conversation from the system/human/gpt format to the
        system/user/assistant format.

        Args:
            sample: Input conversation sample.

        Returns:
            Converted conversation.
        """
        messages = []

        role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
        }

        for msg in sample[self.text_field]:
            role = role_map.get(msg.get("from"))
            if role is not None:
                messages.append(
                    {
                        "role": role,
                        "content": msg["value"],
                    }
                )

        return {"messages": messages}

    def preprocess(self, example):
        """
        Tokenize, process, and mask a conversation for training.

        Args:
            example: Input conversation.

        Returns:
            Preprocessed training sample.
        """
        tokens = []
        loss_mask = []

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]

            if role == "system":
                text = f"### System:\n{content}\n\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.extend(ids)
                loss_mask.extend([0] * len(ids))

            elif role == "user":
                text = f"### User:\n{content}\n\n"
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                tokens.extend(ids)
                loss_mask.extend([0] * len(ids))

            elif role == "assistant":
                prefix = "### Assistant:\n"
                prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
                tokens.extend(prefix_ids)
                loss_mask.extend([0] * len(prefix_ids))

                response = content + (self.tokenizer.eos_token or "")
                response_ids = self.tokenizer.encode(response, add_special_tokens=False)
                tokens.extend(response_ids)
                loss_mask.extend([1] * len(response_ids))

        tokens = tokens[: self.max_seq_len + 1]
        loss_mask = loss_mask[: self.max_seq_len + 1]

        if len(tokens) < 2:
            return None

        input_ids = tokens[:-1]
        labels = [
            token if keep else -100
            for token, keep in zip(tokens[1:], loss_mask[1:])
        ]
        position_ids = list(range(len(input_ids)))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }


class NoStreamingFinetuningDataset(BaseFinetuningDataset):
    def load(self):
        dataset = load_dataset(
            self.dataset_name,
            split= self.split,
        )

        dataset = dataset.map(
            self.convert,
            remove_columns=dataset.column_names,
        )

        dataset = dataset.map(
            self.preprocess,
            remove_columns=dataset.column_names,
        )
        if self.max_sample_length is not None:
            dataset = dataset.filter(
                lambda example: len(example["input_ids"]) <= self.max_sample_length
            )

        return dataset
    


class StreamingFinetuningDataset(
    BaseFinetuningDataset,
    IterableDataset,
):
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

        stream = load_dataset(
            self.dataset_name,
            split="train",
            streaming=True,
        )

        stream = stream.shard(
            num_shards=total_shards,
            index=global_worker_id,
        )

        for sample in stream:
            sample = self.convert(sample)
            sample = self.preprocess(sample)

            if len(sample["input_ids"]) > self.max_sample_length:
                continue

            if sample is not None:
                yield sample


class FinetuningDataset:

    @staticmethod
    def load(
        tokenizer,
        config,
        dataset_name,
        streaming=False,
        max_sample_length=1800,
    ):
        """
        Load and preprocess a dataset.

        Uses the streaming or non-streaming implementation depending on
        the `streaming` flag.

        Args:
            tokenizer: Tokenizer used for preprocessing.
            config: Training or preprocessing configuration.
            dataset_name: Name of the dataset to load.
            streaming: Whether to use dataset streaming.
            max_sample_length: Maximum allowed sample length.

        Returns:
            A processed dataset instance.
        """
        if streaming:
            return StreamingFinetuningDataset(
                tokenizer,
                config,
                dataset_name,
                max_sample_length=max_sample_length, 
            )

        return NoStreamingFinetuningDataset(
            tokenizer,
            config,
            dataset_name,
            max_sample_length=max_sample_length,
        ).load()