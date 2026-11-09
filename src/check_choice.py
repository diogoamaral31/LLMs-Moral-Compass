"""Verificações do passo 4, uma vez por modelo, sobre os itens em inglês (B0, duas ordens).

1. Tokenização: as etiquetas são tokens únicos e o prompt acaba onde a resposta começa.
2. Greedy: o 1.º token gerado em greedy coincide com o argmax do vocabulário e,
   quando é uma etiqueta, com o argmax entre A e B.
3. Amostragem: em alguns itens, respostas amostradas com temperatura 1 (sem top-k/top-p)
   dão frequências de A/B próximas de P(A)/P(B).

Grava o detalhe em data/results/checks/<modelo>_choice.csv e o resumo em
data/results/checks/<modelo>_choice.yaml.

Uso: python -m src.check_choice [nome ...] [--samples N] [--sample-items K]
"""
import argparse
import gc
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.models import (choice_probs, encode_prompt, label_token_ids, load_config, load_pinned,
                        quant_label, render_prompt)
from src.prompts import ORDERS, available_items, build, master_items, option_probs, template_version
from src.seeding import set_seeds

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "results" / "checks"
LABELS = ("A", "B")
# Primeira etiqueta isolada na resposta: "A", "**A**", "A.", "Option A", "(B)".
ANSWER_RE = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")


def parse_answer(text):
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None


@torch.no_grad()
def sample_answers(tok, model, entry, messages, n, chunk=5):
    """n respostas amostradas da distribuição completa, em blocos (8 GB de VRAM não chegam para 30 de uma vez)."""
    enc = encode_prompt(tok, entry, messages, model.device)
    start = enc["input_ids"].shape[1]
    answers = []
    for k in range(0, n, chunk):
        out = model.generate(
            **enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, min_p=None,
            max_new_tokens=8, num_return_sequences=min(chunk, n - k),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        answers += [tok.decode(seq[start:], skip_special_tokens=True).strip() for seq in out]
        del out
    return answers


def check(name, n_samples, n_sample_items):
    print(f"\n=== {name}")
    seed = set_seeds()
    tok, model, entry = load_pinned(name)
    summary = {"model": name, "model_id": entry["id"], "revision": entry["revision"],
               "quant": quant_label(), "template_version": template_version(), "seed": seed}
    ok = True

    # 1. Tokenização
    tokens = {}
    for label in LABELS:
        ids = label_token_ids(tok, label)
        tokens[label] = {"main": ids[0], "variants": [tok.convert_ids_to_tokens(i) for i in ids]}
        print(f"Etiqueta {label}: principal {ids[0]} {tok.convert_ids_to_tokens(ids[0])!r}, "
              f"variantes {tokens[label]['variants'][1:]}")
    summary["label_tokens"] = tokens
    prompt = render_prompt(tok, entry, build(next(iter(master_items()))))
    print(f"Fim do prompt: {prompt[-50:]!r}")
    summary["prompt_tail"] = prompt[-50:]

    # 2. Greedy vs argmax, em todos os itens e nas duas ordens
    rows = []
    for item_id in available_items():
        for order in ORDERS:
            messages = build(item_id, order=order)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = choice_probs(tok, model, entry, messages, LABELS)
            torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            enc = encode_prompt(tok, entry, messages, model.device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=1, top_k=None, top_p=None,
                                 temperature=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            greedy = tok.decode(out[0, -1:])
            label_greedy = greedy.strip() if greedy.strip() in LABELS else None
            label_argmax = "A" if r["p_A"] >= r["p_B"] else "B"
            rows.append({
                "item_id": item_id, "type": master_items()[item_id]["type"], "order": order,
                **r, **option_probs(r["p_A"], r["p_B"], order),
                "greedy_token": greedy, "greedy_is_top": greedy == r["top_token"],
                "greedy_matches_ab": None if label_greedy is None else label_greedy == label_argmax,
                "latency_s": latency,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("Sem itens em inglês para verificar.")
        return False
    controls = df[df["type"] == "control"]
    variant_mass = (df["p_A"] - df["p_A_main"] + df["p_B"] - df["p_B_main"]).max()
    greedy_label = df["greedy_matches_ab"].dropna()
    summary.update({
        "requests": len(df),
        "greedy_is_top": float(df["greedy_is_top"].mean()),
        "greedy_is_label": float(len(greedy_label) / len(df)),
        "greedy_matches_ab_argmax": float(greedy_label.mean()) if len(greedy_label) else None,
        "p_other_mean": float(df["p_other"].mean()),
        "p_other_max": float(df["p_other"].max()),
        "variant_mass_max": float(variant_mass),
        "control_accuracy": float((controls["p_v1"] > 0.5).mean()) if len(controls) else None,
        "position_bias_mean_p_A": float(df["p_A"].div(df["p_A"] + df["p_B"]).mean()),
        "latency_s_median": float(df["latency_s"].median()),
        "prompt_tokens_median": float(df["n_tokens"].median()),
    })
    print(f"Greedy = argmax do vocabulário: {summary['greedy_is_top']:.0%} | "
          f"greedy é etiqueta: {summary['greedy_is_label']:.0%} | "
          f"greedy = argmax(A,B): {summary['greedy_matches_ab_argmax']}")
    print(f"p_other média {summary['p_other_mean']:.4f}, máx. {summary['p_other_max']:.4f} | "
          f"massa máx. em variantes {variant_mass:.2e}")
    others = df.loc[df["p_other"] > 0.1, ["item_id", "order", "p_other", "top_token"]]
    if len(others):
        print(f"Pedidos com p_other > 0.1:\n{others.to_string(index=False)}")
    print(f"Controlos certos (p_v1 > 0.5, por ordem): {summary['control_accuracy']:.0%} | "
          f"P(A) médio entre A/B (viés de posição): {summary['position_bias_mean_p_A']:.2f}")
    print(f"Latência mediana: {summary['latency_s_median']:.3f} s/pedido "
          f"({summary['prompt_tokens_median']:.0f} tokens)")
    ok &= summary["greedy_is_top"] == 1.0 and greedy_label.all()

    # 3. Amostragem vs probabilidades, nos itens mais incertos (onde a comparação informa)
    sample_rows = []
    if n_samples and n_sample_items:
        df["uncertainty"] = (df["p_A"] / (df["p_A"] + df["p_B"]) - 0.5).abs()
        picks = df.sort_values("uncertainty").head(n_sample_items)
        for _, row in picks.iterrows():
            answers = sample_answers(tok, model, entry, build(row["item_id"], order=row["order"]), n_samples)
            parsed = [parse_answer(a) for a in answers]
            n_a, n_b = parsed.count("A"), parsed.count("B")
            sample_rows.append({
                "item_id": row["item_id"], "order": row["order"],
                "p_A": row["p_A"], "p_B": row["p_B"], "p_other": row["p_other"],
                "freq_A": n_a / n_samples, "freq_B": n_b / n_samples,
                "freq_other": 1 - (n_a + n_b) / n_samples,
                "examples": " | ".join(answers[:3]),
            })
        s = pd.DataFrame(sample_rows)
        s["abs_diff_A"] = (s["freq_A"] - s["p_A"]).abs()
        # Erro-padrão binomial com n amostras; acima de ~3 EP é suspeito.
        se = (s["p_A"] * (1 - s["p_A"]) / n_samples).clip(lower=1e-4) ** 0.5
        s["z"] = (s["freq_A"] - s["p_A"]) / se
        summary["sampling"] = {
            "items": len(s), "samples_per_item": n_samples,
            "mean_abs_diff_A": float(s["abs_diff_A"].mean()),
            "max_abs_z": float(s["z"].abs().max()),
        }
        print(f"Amostragem ({n_samples} por item):")
        print(s[["item_id", "order", "p_A", "freq_A", "p_other", "freq_other", "z"]]
              .round(3).to_string(index=False))
        sample_rows = s.to_dict("records")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["uncertainty"], errors="ignore").to_csv(OUT_DIR / f"{name}_choice.csv", index=False)
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(OUT_DIR / f"{name}_sampling.csv", index=False)
    summary["passed"] = bool(ok)
    (OUT_DIR / f"{name}_choice.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Resultado: {'OK' if ok else 'FALHOU'} -> {OUT_DIR / f'{name}_choice.yaml'}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # tokens como "ĠA" falham em cp1252
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample-items", type=int, default=6)
    args = parser.parse_args()
    names = args.names or list(load_config()["models"])
    if len(names) > 1:
        # Um processo por modelo (a memória da GPU não é libertada de forma fiável).
        extra = ["--samples", str(args.samples), "--sample-items", str(args.sample_items)]
        codes = {n: subprocess.run([sys.executable, "-m", "src.check_choice", n, *extra]).returncode
                 for n in names}
        print("\n=== Resumo")
        for n, code in codes.items():
            print(f"{n}: {'OK' if code == 0 else 'FALHOU'}")
        sys.exit(any(codes.values()))
    sys.exit(0 if check(names[0], args.samples, args.sample_items) else 1)


if __name__ == "__main__":
    main()
