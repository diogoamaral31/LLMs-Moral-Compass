"""Corrige os CSV do UniMoral em data/multilang e grava cópias em data/multilang_clean.

Os ficheiros originais não são alterados. Correções:
1. Linhas de ficheiros espanhóis com o cenário e as ações em russo (erro de
   exportação): o texto é substituído pela versão espanhola do mesmo Scenario_id.
2. Scenario_id com várias versões ligeiramente diferentes do mesmo texto: fica
   a versão mais frequente nesse ficheiro.
3. Scenario guardado como lista Python em texto ("['...']"): passa a texto simples.
   Possible_actions passa a lista JSON em todas as línguas.
4. Colunas de índice deixadas pelo pandas ('' e 'Unnamed: 0') são removidas.

O token <unk> (palavras perdidas na tradução automática) não é recuperável:
as linhas afetadas ficam assinaladas na coluna has_unk.

Uso: python -m src.clean_unimoral
"""
import ast
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "multilang"
OUT_DIR = ROOT / "data" / "multilang_clean"
INDEX_COLUMNS = {"", "Unnamed: 0"}
SCRIPTS = {"English": "LATIN", "Spanish": "LATIN", "Russian": "CYRILLIC",
           "Arabic": "ARABIC", "Hindi": "DEVANAGARI", "Chinese": "CJK"}

csv.field_size_limit(10**8)


def dominant_script(text):
    counts = Counter()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            counts[next((s for s in set(SCRIPTS.values()) if name.startswith(s)), "OTHER")] += 1
    return counts.most_common(1)[0][0] if counts else "NONE"


def parse_list(value):
    value = value.strip()
    if value.startswith("["):
        return [str(x).strip() for x in ast.literal_eval(value)]
    return [value]


def clean_file(path):
    lang = path.name.split("_")[0]
    expected = SCRIPTS[lang]
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = [c for c in reader.fieldnames if c not in INDEX_COLUMNS]
        rows = list(reader)

    # Versão de referência por Scenario_id: a mais frequente no alfabeto certo.
    versions = defaultdict(Counter)
    for r in rows:
        if dominant_script(r["Scenario"]) == expected:
            versions[r["Scenario_id"]][(r["Scenario"], r["Possible_actions"])] += 1
    reference = {sid: c.most_common(1)[0][0] for sid, c in versions.items()}

    out, stats = [], Counter()
    for r in rows:
        sid = r["Scenario_id"]
        notes = []
        scenario, actions = r["Scenario"], r["Possible_actions"]
        if dominant_script(scenario) != expected:
            if sid not in reference:
                raise ValueError(f"{path.name}: {sid} sem versão em {lang}")
            notes.append(f"text in {dominant_script(scenario).lower()} script replaced")
            stats["wrong_script"] += 1
            scenario, actions = reference[sid]
        elif (scenario, actions) != reference[sid]:
            notes.append("variant text replaced by the most frequent version")
            stats["variant"] += 1
            scenario, actions = reference[sid]

        scenario_parts = parse_list(scenario)
        action_list = parse_list(actions)
        if len(scenario_parts) != 1 or len(action_list) != 2:
            raise ValueError(f"{path.name}: {sid} com formato inesperado")

        row = {c: r[c] for c in columns}
        row["Scenario"] = scenario_parts[0]
        row["Possible_actions"] = json.dumps(action_list, ensure_ascii=False)
        row["has_unk"] = int("<unk>" in scenario or "<unk>" in actions)
        row["cleaning_note"] = "; ".join(notes)
        stats["unk"] += row["has_unk"]
        out.append(row)

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / path.name, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns + ["has_unk", "cleaning_note"])
        writer.writeheader()
        writer.writerows(out)
    return len(out), stats


def main():
    files = sorted(p for p in SRC_DIR.glob("*.csv") if p.name.split("_")[0] in SCRIPTS)
    print(f"{'ficheiro':32s} {'linhas':>6s} {'alfabeto':>8s} {'variante':>8s} {'<unk>':>6s}")
    for path in files:
        n, s = clean_file(path)
        print(f"{path.name:32s} {n:6d} {s['wrong_script']:8d} {s['variant']:8d} {s['unk']:6d}")
    print(f"\nCópias corrigidas em {OUT_DIR}")


if __name__ == "__main__":
    main()
