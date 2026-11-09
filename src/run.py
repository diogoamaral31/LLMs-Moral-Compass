"""Orquestração retomável das corridas (passos 5 e 12).

Corre o grid (modelo x item x língua x condição x ordem), salta o que já está em
--out e grava em blocos (ficheiro substituído de forma atómica). Interromper com
Ctrl+C e voltar a correr o mesmo comando retoma de onde parou.

Uso: python -m src.run [--models nome ...] [--out caminho.parquet] [--block 40] [--limit N]
Predefinição: piloto, 2 modelos, data/results/pilot.parquet.
"""
import argparse
import datetime as dt
import gc
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.models import choice_probs, load_config, load_pinned, quant_label
from src.prompts import ORDERS, available_items, build, master_items, option_probs, template_version
from src.seeding import set_seeds

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "results" / "pilot.parquet"
PILOT_MODELS = ["qwen3-8b", "aya-expanse-8b"]
PILOT_CONDITIONS = [("en", "B0"), ("en", "paraphrase"), ("pt-PT", "B0")]
KEY = ["model", "item_id", "language", "condition", "order"]


def grid(model):
    return [
        {"model": model, "item_id": item_id, "language": lang, "condition": cond, "order": order}
        for lang, cond in PILOT_CONDITIONS
        for item_id in available_items(lang, cond)
        for order in ORDERS
    ]


def load_done(out):
    if not out.exists():
        return pd.DataFrame(), set()
    df = pd.read_parquet(out)
    return df, set(map(tuple, df[KEY].itertuples(index=False, name=None)))


def save(df, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)


def write_environment(out):
    """Versões e hardware desta corrida, ao lado dos resultados."""
    import transformers
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    info = {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "transformers": str(transformers.__version__),
        "pandas": str(pd.__version__), "gpu_and_driver": smi, "quant": quant_label(),
        "template_version": template_version(), "seed": set_seeds(),
    }
    path = out.with_name(out.stem + "_env.yaml")
    log = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else []
    path.write_text(yaml.safe_dump(log + [info], sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_model(model, out, block, limit):
    existing, done = load_done(out)
    todo = [g for g in grid(model) if tuple(g[k] for k in KEY) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{model}: {len(todo)} pedidos por fazer ({len(done)} já gravados no ficheiro)", flush=True)
    if not todo:
        return
    seed = set_seeds()
    tok, mdl, entry = load_pinned(model)
    run_id = f"{dt.datetime.now():%Y%m%dT%H%M%S}-{model}"
    quant = quant_label()
    types = {i: r["type"] for i, r in master_items().items()}
    rows, t_start = [], time.perf_counter()

    def flush():
        nonlocal existing, rows
        if rows:
            existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            save(existing, out)
            rows = []

    try:
        for n, g in enumerate(todo, 1):
            messages = build(g["item_id"], g["language"], g["condition"], g["order"])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = choice_probs(tok, mdl, entry, messages)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            rows.append({
                "run_id": run_id, "model": model, "model_id": entry["id"], "revision": entry["revision"],
                "quant": quant, "template_version": template_version(),
                "item_id": g["item_id"], "type": types[g["item_id"]], "language": g["language"],
                "condition": g["condition"], "order": g["order"],
                "p_A": r["p_A"], "p_B": r["p_B"], "p_other": r["p_other"],
                **option_probs(r["p_A"], r["p_B"], g["order"]),
                "top_token": r["top_token"], "top_prob": r["top_prob"], "n_tokens": r["n_tokens"],
                "seed": seed, "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "latency_s": latency,
            })
            if len(rows) >= block:
                flush()
                elapsed = time.perf_counter() - t_start
                print(f"  {n}/{len(todo)} | {elapsed / n:.3f} s/pedido (com carregamento excluído do total)", flush=True)
    finally:
        flush()  # também em Ctrl+C ou erro
    print(f"{model}: concluído em {time.perf_counter() - t_start:.0f} s", flush=True)
    del mdl
    gc.collect()
    torch.cuda.empty_cache()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=PILOT_MODELS, choices=list(load_config()["models"]))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="máx. de pedidos por modelo (teste)")
    args = parser.parse_args()

    if len(args.models) > 1:
        # Um processo por modelo (a memória da GPU não é libertada de forma fiável).
        extra = ["--out", str(args.out), "--block", str(args.block), "--limit", str(args.limit)]
        codes = {m: subprocess.run([sys.executable, "-m", "src.run", "--models", m, *extra]).returncode
                 for m in args.models}
        for m, c in codes.items():
            print(f"{m}: {'OK' if c == 0 else 'FALHOU'}")
        sys.exit(any(codes.values()))
    write_environment(args.out)
    run_model(args.models[0], args.out, args.block, args.limit)


if __name__ == "__main__":
    main()
