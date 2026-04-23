import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoConfig

from src.Training.model_train import Transformer

AutoConfig.register("deepseek_custom", Transformer.config_class)
AutoModelForCausalLM.register(Transformer.config_class, Transformer)

def get_optimal_settings():
    if not torch.cuda.is_available():
        return "cpu", "sdpa", torch.float32

    major, _ = torch.cuda.get_device_capability()
    is_ampere_or_newer = major >= 8
    

    ATTN_IMPL = "sdpa" 
    DTYPE = torch.bfloat16 if is_ampere_or_newer else torch.float16
        
    return "cuda", ATTN_IMPL, DTYPE


MODEL_PATH = "./model"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50
DEVICE, ATTN_IMPL, DTYPE = get_optimal_settings()

print(f"Using Device: {DEVICE} | Attention: {ATTN_IMPL} | Dtype: {DTYPE}")





tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


if DEVICE == "cuda":
    print("Running On Gpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=DTYPE,
        attn_implementation=ATTN_IMPL
    )

else:
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=DTYPE,
    attn_implementation=ATTN_IMPL,
    )
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )



@torch.inference_mode()
def generate(prompt,):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":


    

    prompt = "Explain black holes in simple terms:"
    start = time.perf_counter()
    result = generate(prompt)
    end = time.perf_counter()
    
    print(f"\n--- Generated in {end - start:.4f}s ---")
    print(result)