"""Verificação do passo 0: GPU visível, modelo pequeno em 4 bits, gerar uma frase.

Uso: python -m src.check_env [model_id]
"""
import sys

import torch

from src.models import load, model_devices
from src.seeding import set_seeds

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    if not torch.cuda.is_available():
        sys.exit("ERRO: torch não vê a GPU (cuda.is_available() é False).")
    free, total = torch.cuda.mem_get_info()
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória livre: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")
    print(f"Seed: {set_seeds()}")

    tok, model = load(model_id)
    quant = getattr(model.config, "quantization_config", None)
    print(f"Modelo: {model_id}")
    print(f"Em 4 bits: {getattr(model, 'is_loaded_in_4bit', False)} | {quant.bnb_4bit_quant_type if quant else None}")
    print(f"Dispositivos: {model_devices(model)}")

    messages = [{"role": "user", "content": "Write one short sentence about the sea."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Geração: {text.strip()}")
    print(f"VRAM usada no pico: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
