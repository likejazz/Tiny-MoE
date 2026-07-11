from typing import Generator,Optional
from inference.inference_configs import GenerationConfig
import torch
from transformers import AutoTokenizer
from inference.sampler import _normalize_logits,_prepare_input_ids,filter_logits
from inference.load_model import load_and_prepare_model






tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_inference_prompt(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    parts = []

    if system_prompt is not None:
        parts.append(f"### System:\n{system_prompt}\n\n")

    parts.append(f"### User:\n{user_prompt}\n\n")
    parts.append("### Assistant:\n")

    return "".join(parts)


@torch.inference_mode()
def _generate_tokens(
    model,
    input_ids: torch.Tensor,
    cfg: GenerationConfig,
) -> Generator[torch.Tensor, None, torch.Tensor]:
    """
    Yields next-token tensors of shape [B].
    Returns the final generated tensor through StopIteration.value.
    """
    model.eval()

    if hasattr(model, "reset_cache"):
        model.reset_cache()

    input_ids = _prepare_input_ids(input_ids)
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    total_len = prompt_len + cfg.max_new_tokens
    out = torch.empty((batch_size, total_len), dtype=torch.long, device=device)
    out[:, :prompt_len] = input_ids

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if cfg.eos_token_id is not None:
        finished |= (input_ids == cfg.eos_token_id).any(dim=1)

    cur_len = prompt_len

    logits = model(input_ids, return_last_only=True)

    while cur_len < total_len and not finished.all():
        next_logits = _normalize_logits(logits)
        prev_tokens = out[:, :cur_len]

        next_logits, next_token = filter_logits(
            next_logits,
            cfg,
            prev_tokens=prev_tokens,
        )

        if next_token.dim() == 2 and next_token.size(1) == 1:
            next_token = next_token.squeeze(1)
        elif next_token.dim() != 1:
            raise ValueError(f"Unexpected next_token shape: {tuple(next_token.shape)}")

        if cfg.eos_token_id is not None and cfg.pad_token_id is not None:
            next_token = next_token.masked_fill(finished, cfg.pad_token_id)

        out[:, cur_len] = next_token
        cur_len += 1

        if cfg.eos_token_id is not None:
            finished |= (next_token == cfg.eos_token_id)

        yield next_token

        if cur_len < total_len and not finished.all():
            logits = model(next_token.unsqueeze(1), return_last_only=True)

    return out[:, :cur_len]


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_prompt(
    model,
    tokenizer,
    cfg: GenerationConfig,
    device: str = "cuda",
    system_prompt: Optional[str] = None,
) -> None:
    model.eval()

    while True:
        user_prompt = input("\nPrompt: ").strip()
        if user_prompt.lower() in {"exit", "quit"}:
            break
        if cfg.model_variant == "fine-tuned":
            text = build_inference_prompt(user_prompt, system_prompt=system_prompt)
        else:
            text = user_prompt

        input_ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)

        generated_ids = input_ids.clone()
        prev_text = _decode_tokens(tokenizer, generated_ids[0].tolist())

        print("\nModel: ", end="", flush=True)

        with torch.inference_mode():
            for next_token in _generate_tokens(model, input_ids, cfg):
                if next_token.dim() == 1:
                    next_token = next_token.unsqueeze(1)

                token_id = int(next_token[0, 0].item())

                if cfg.eos_token_id is not None and token_id == cfg.eos_token_id:
                    break

                generated_ids = torch.cat([generated_ids, next_token.to(device)], dim=1)

                new_text = _decode_tokens(tokenizer, generated_ids[0].tolist())

                if new_text.startswith(prev_text):
                    suffix = new_text[len(prev_text):]
                else:
                    lcp = 0
                    limit = min(len(prev_text), len(new_text))
                    while lcp < limit and prev_text[lcp] == new_text[lcp]:
                        lcp += 1
                    suffix = new_text[lcp:]

                if suffix:
                    print(suffix, end="", flush=True)
                    prev_text = new_text

        print()





def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GenerationConfig(
        temperature=1,
        top_p=None,
        top_k=None,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        max_new_tokens=50,   
        do_sample=False,
        model_variant="base"
    )
    model = load_and_prepare_model(cfg,device)
    system_prompt = "You Are a helpful Ai Assistant."
    run_prompt(model, tokenizer, cfg, device=device,system_prompt=system_prompt)


if __name__ == "__main__":
    main()
