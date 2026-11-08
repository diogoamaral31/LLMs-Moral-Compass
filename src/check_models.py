"""Verificação do passo 1: hashes, carga em 4 bits, template e geração por modelo.

Uso: python -m src.check_models [nome ...]
"""
import gc
import hashlib
import subprocess
import sys

import torch
from huggingface_hub import hf_hub_download

from src.models import load_config, load_pinned, model_devices, render_prompt
from src.seeding import set_seeds

QUESTION = (
    "Which of these is a colour?\n\nOption A: Blue\nOption B: Seven\n\n"
    "You must choose exactly one option. Answer with a single letter: A or B."
)


def verify_files(entry):
    """Compara cada ficheiro em cache com o hash registado; devolve os que falham."""
    failed = []
    for filename, expected in entry["files"].items():
        path = hf_hub_download(entry["id"], filename, revision=entry["revision"])
        if "sha256" in expected:
            digest = hashlib.sha256()
            target = expected["sha256"]
        else:
            digest = hashlib.sha1(f"blob {expected['size']}\0".encode())
            target = expected["git_blob"]
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != target:
            failed.append(filename)
    return failed


def check(name):
    print(f"\n=== {name}")
    tok, model, entry = load_pinned(name)

    failed = verify_files(entry)
    print(f"Hashes: {len(entry['files']) - len(failed)}/{len(entry['files'])} corretos")

    devices = model_devices(model)
    declared = entry.get("cpu_modules", [])
    off_gpu = {n for n, p in model.named_parameters() if p.device.type != "cuda"}
    on_gpu_only = all(any(n.startswith(m) for m in declared) for n in off_gpu)
    print(f"Fora da GPU (declarado): {sorted(off_gpu)}")
    print(f"Dispositivos: {devices} | 4 bits: {getattr(model, 'is_loaded_in_4bit', False)}")

    messages = [{"role": "user", "content": QUESTION}]
    prompt = render_prompt(tok, entry, messages)
    print(f"Fim do prompt: {prompt[-60:]!r}")

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    finite = bool(torch.isfinite(logits).all())
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"Logits finitos: {finite} | Resposta: {text!r}")
    print(f"VRAM no pico: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return not failed and on_gpu_only and finite


def main():
    names = sys.argv[1:] or list(load_config()["models"])
    if len(names) > 1:
        # Um processo por modelo: a memória da GPU não é libertada de forma fiável
        # entre modelos no mesmo processo.
        codes = {n: subprocess.run([sys.executable, "-m", "src.check_models", n]).returncode
                 for n in names}
        print("\n=== Resumo")
        for n, code in codes.items():
            print(f"{n}: {'OK' if code == 0 else 'FALHOU'}")
        sys.exit(any(codes.values()))
    set_seeds()
    sys.exit(0 if check(names[0]) else 1)


if __name__ == "__main__":
    main()
