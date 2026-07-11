import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Set, List
from inference.inference_configs import GenerationConfig



def _prepare_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()
    return input_ids


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - [B, V]
      - [B, T, V]
    Returns:
      - [B, V]
    """
    if logits.dim() == 3:
        return logits[:, -1, :]
    if logits.dim() == 2:
        return logits
    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")


def _apply_repetition_penalty(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    penalty: float,
    ignore_token_ids: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Apply a repetition penalty to previously generated tokens.

    Args:
        logits: Input logits.
        prev_tokens: Previously generated token IDs.
        penalty: Repetition penalty factor.
        ignore_token_ids: Token IDs excluded from the penalty.

    Returns:
        Adjusted logits.
    """
    if penalty is None or penalty <= 1.0:
        return logits
    if prev_tokens.numel() == 0:
        return logits

    out = logits.clone()

    for b in range(out.size(0)):
        tokens = prev_tokens[b]

        if ignore_token_ids is not None:
            keep = torch.ones_like(tokens, dtype=torch.bool)
            for tid in ignore_token_ids:
                keep &= tokens != tid
            tokens = tokens[keep]

        if tokens.numel() == 0:
            continue

        tokens = torch.unique(tokens)
        token_logits = out[b, tokens]

 
        token_logits = torch.where(
            token_logits < 0,
            token_logits * penalty,
            token_logits / penalty,
        )
        out[b, tokens] = token_logits

    return out


def _apply_top_k(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    Apply top-k filtering to logits.

    Args:
        logits: Input logits.
        k: Number of highest-probability tokens to keep.

    Returns:
        Filtered logits.
    """
    if k is None or k <= 0:
        return logits

    vocab_size = logits.size(-1)
    k = min(k, vocab_size)
    if k >= vocab_size:
        return logits

    kth_values = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_values, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """
    Apply top-p (nucleus) filtering to logits.

    Args:
        logits: Input logits.
        p: Cumulative probability threshold.

    Returns:
        Filtered logits.
    """
    if p is None or p <= 0.0 or p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = sorted_probs.cumsum(dim=-1)

    remove_mask = cum_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    remove_vocab = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_idx,
        src=remove_mask,
    )
    return logits.masked_fill(remove_vocab, float("-inf"))


def _sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample the next token from the logits distribution.

    Args:
        logits: Input logits.

    Returns:
        Sampled token IDs.
    """
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_no_repeat_ngram(
    logits: torch.Tensor,
    prev_tokens: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    """
    Hard-block tokens that would create a repeated n-gram.

    Args:
        logits: [B, V] token logits.
        prev_tokens: [B, T] previously generated token IDs.
        ngram_size: Size of the n-gram to block.

    Returns:
        Modified logits with blocked tokens set to -inf.
    """
    if ngram_size is None or ngram_size <= 1:
        return logits

    if prev_tokens is None:
        return logits

    if prev_tokens.dim() == 1:
        prev_tokens = prev_tokens.unsqueeze(0)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    logits = logits.clone()
    batch_size, seq_len = prev_tokens.shape

    if seq_len < ngram_size - 1:
        return logits

    for b in range(batch_size):
        history = prev_tokens[b].tolist()

        if len(history) < ngram_size - 1:
            continue


        banned: Dict[Tuple[int, ...], Set[int]] = {}

        for i in range(len(history) - ngram_size + 1):
            prefix = tuple(history[i : i + ngram_size - 1])
            next_token = history[i + ngram_size - 1]
            banned.setdefault(prefix, set()).add(next_token)

        current_prefix = tuple(history[-(ngram_size - 1):])
        blocked_tokens = banned.get(current_prefix)

        if blocked_tokens:
            blocked_idx = torch.tensor(
                list(blocked_tokens),
                device=logits.device,
                dtype=torch.long,
            )
            logits[b, blocked_idx] = float("-inf")

    return logits

def filter_logits(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    prev_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter logits and select the next token.
    """
    logits = _normalize_logits(logits)

    if (
        cfg.repetition_penalty is not None
        and cfg.repetition_penalty > 1.0
        and prev_tokens is not None
    ):
        ignore_ids = tuple(
            x for x in (cfg.eos_token_id, cfg.pad_token_id) if x is not None
        )
        logits = _apply_repetition_penalty(
            logits,
            prev_tokens=prev_tokens,
            penalty=cfg.repetition_penalty,
            ignore_token_ids=ignore_ids if len(ignore_ids) > 0 else None,
        )

    if (
        getattr(cfg, "no_repeat_ngram_size", None) is not None
        and cfg.no_repeat_ngram_size > 1
        and prev_tokens is not None
    ):
        logits = _apply_no_repeat_ngram(
            logits,
            prev_tokens=prev_tokens,
            ngram_size=cfg.no_repeat_ngram_size,
        )

    if cfg.temperature is not None and cfg.temperature != 1.0:
        if cfg.temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / cfg.temperature

    vocab_size = logits.size(-1)

    if cfg.top_k is not None and cfg.top_k > 0:
        logits = _apply_top_k(logits, min(cfg.top_k, vocab_size))

    if cfg.top_p is not None and cfg.top_p < 1.0:
        logits = _apply_top_p(logits, cfg.top_p)

    if not cfg.do_sample:
        return logits, logits.argmax(dim=-1)

    return logits, _sample(logits)