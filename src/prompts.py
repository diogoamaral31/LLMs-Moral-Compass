"""Templates e condições do prompt (passos 4 e 11).

Textos por língua e condição:
- en, B0:          data/dilemmas.csv (scenario_en, option_1_en, option_2_en)
- outra língua:    data/translations/<língua>.csv (item_id, scenario, option_1, option_2, notes)
- paraphrase:      data/paraphrases/<língua>.csv (item_id, scenario, notes); as opções
                   são as da mesma língua em B0, só o cenário é reformulado.

Ordem de apresentação (o ficheiro guarda sempre opção 1 = value_1):
- "AB": opção 1 aparece como A, opção 2 como B
- "BA": opção 2 aparece como A, opção 1 como B

Uso: python -m src.prompts [item_id] [língua] [condição] [ordem]
     Sem argumentos, valida todos os ficheiros de texto e mostra um exemplo.
"""
import csv
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from src.schema import read_items

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_FILE = ROOT / "configs" / "prompt.yaml"
TRANSLATIONS_DIR = ROOT / "data" / "translations"
PARAPHRASES_DIR = ROOT / "data" / "paraphrases"

BASE_LANGUAGE = "en"
ORDERS = ("AB", "BA")
CONDITIONS = ("B0", "paraphrase")  # pressão entra no passo 11
TRANSLATION_COLUMNS = ["item_id", "scenario", "option_1", "option_2", "notes"]
PARAPHRASE_COLUMNS = ["item_id", "scenario", "notes"]


@lru_cache(maxsize=None)
def load_template():
    return yaml.safe_load(TEMPLATE_FILE.read_text(encoding="utf-8"))


def template_version():
    return load_template()["version"]


@lru_cache(maxsize=None)
def master_items():
    """Itens mestre por item_id, com os campos já sem espaços nas pontas."""
    _, rows = read_items()
    return {r["item_id"].strip(): {k: v.strip() for k, v in r.items()} for r in rows}


def _read_table(path, columns):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f"{path}: colunas {reader.fieldnames}, esperado {columns}")
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in reader]
    return {r["item_id"]: r for r in rows}


@lru_cache(maxsize=None)
def texts(language=BASE_LANGUAGE, condition="B0"):
    """Devolve {item_id: {"scenario", "option_1", "option_2"}} para a língua e condição."""
    if language not in load_template()["languages"]:
        raise ValueError(f"língua sem template em {TEMPLATE_FILE.name}: {language}")
    if condition not in CONDITIONS:
        raise ValueError(f"condição desconhecida: {condition}")

    if condition == "paraphrase":
        base = texts(language, "B0")
        para = _read_table(PARAPHRASES_DIR / f"{language}.csv", PARAPHRASE_COLUMNS)
        return {i: {**base[i], "scenario": r["scenario"]} for i, r in para.items() if i in base}

    if language == BASE_LANGUAGE:
        return {
            i: {"scenario": r["scenario_en"], "option_1": r["option_1_en"], "option_2": r["option_2_en"]}
            for i, r in master_items().items()
        }
    table = _read_table(TRANSLATIONS_DIR / f"{language}.csv", TRANSLATION_COLUMNS)
    return {i: {k: r[k] for k in ("scenario", "option_1", "option_2")} for i, r in table.items()}


def available_items(language=BASE_LANGUAGE, condition="B0"):
    """item_ids que existem no ficheiro mestre e têm texto nesta língua e condição."""
    return [i for i in master_items() if i in texts(language, condition)]


def build(item_id, language=BASE_LANGUAGE, condition="B0", order="AB"):
    """Mensagens de chat para um item. Um item por pedido, sem histórico."""
    if order not in ORDERS:
        raise ValueError(f"ordem desconhecida: {order}")
    frame = load_template()["languages"][language]
    t = texts(language, condition)[item_id]
    option_a, option_b = (t["option_1"], t["option_2"]) if order == "AB" else (t["option_2"], t["option_1"])
    label_a, label_b = load_template()["labels"]
    user = (
        f"{t['scenario']}\n\n"
        f"{frame['option']} {label_a}: {option_a}\n"
        f"{frame['option']} {label_b}: {option_b}\n\n"
        f"{frame['instruction']}"
    )
    messages = [{"role": "user", "content": user}]
    if frame.get("system"):
        messages.insert(0, {"role": "system", "content": frame["system"]})
    return messages


def option_probs(p_A, p_B, order):
    """Converte P(A)/P(B) em P(opção 1)/P(opção 2) e p_v1 = p1 / (p1 + p2).

    Para controlos, p_v1 é a probabilidade da resposta correta entre as duas.
    """
    p_opt1, p_opt2 = (p_A, p_B) if order == "AB" else (p_B, p_A)
    total = p_opt1 + p_opt2
    return {"p_opt1": p_opt1, "p_opt2": p_opt2, "p_v1": p_opt1 / total if total > 0 else float("nan")}


def validate():
    """Verifica os ficheiros de tradução e paráfrase existentes; devolve a lista de erros."""
    errors = []
    master = master_items()
    languages = list(load_template()["languages"])
    for language in languages:
        frame = load_template()["languages"][language]
        for key in ("option", "instruction"):
            if not frame.get(key):
                errors.append(f"template {language}: {key} vazio")
        for condition in CONDITIONS:
            path = (PARAPHRASES_DIR if condition == "paraphrase" else TRANSLATIONS_DIR) / f"{language}.csv"
            if language == BASE_LANGUAGE and condition == "B0":
                continue
            if not path.exists():
                continue
            try:
                table = texts(language, condition)
            except (ValueError, KeyError) as e:
                errors.append(str(e))
                continue
            where = f"{path.relative_to(ROOT)}"
            raw = _read_table(path, PARAPHRASE_COLUMNS if condition == "paraphrase" else TRANSLATION_COLUMNS)
            for i in raw:
                if i not in master:
                    errors.append(f"{where}: {i} não existe em dilemmas.csv")
            for i in master:
                if i not in raw:
                    errors.append(f"{where}: falta {i}")
            for i, t in table.items():
                for k, v in t.items():
                    if not v:
                        errors.append(f"{where}: {i} com {k} vazio")
                if t["option_1"] and t["option_1"] == t["option_2"]:
                    errors.append(f"{where}: {i} com as duas opções iguais")
            if condition == "paraphrase" and language == BASE_LANGUAGE:
                for i in table:
                    if i in master and table[i]["scenario"] == master[i]["scenario_en"]:
                        errors.append(f"{where}: {i} igual ao original")
    return errors


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # tokens como "ĠA" falham em cp1252
    args = sys.argv[1:]
    if args:
        item_id, language, condition, order = args + [None, BASE_LANGUAGE, "B0", "AB"][len(args):]
        for m in build(item_id, language, condition, order):
            print(f"--- {m['role']}\n{m['content']}")
        return

    errors = validate()
    for e in errors:
        print(e)
    print(f"Template {template_version()}")
    for language in load_template()["languages"]:
        for condition in CONDITIONS:
            try:
                n = len(available_items(language, condition))
            except FileNotFoundError:
                n = 0
            print(f"  {language:6s} {condition:10s} {n:3d} itens")
    first = next(iter(master_items()))
    print(f"\nExemplo ({first}, en, B0, BA):")
    for m in build(first, order="BA"):
        print(f"--- {m['role']}\n{m['content']}")
    if errors:
        sys.exit(f"{len(errors)} erro(s)")


if __name__ == "__main__":
    main()
