"""Esquema de data/dilemmas.csv e respetiva validação (passo 2).

Uso: python -m src.schema [caminho.csv]
"""
import csv
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "dilemmas.csv"

COLUMNS = [
    "item_id",
    "type",
    "domain",
    "scenario_en",
    "option_1_en",
    "option_2_en",
    "value_1",
    "value_2",
    "label_annotator2_v1",
    "label_annotator2_v2",
    "notes",
]
TYPES = {"dilemma", "control"}
REQUIRED = ["item_id", "type", "scenario_en", "option_1_en", "option_2_en"]


def read_items(path=DEFAULT_PATH):
    """Lê o CSV como texto, sem conversões, e devolve (colunas, linhas)."""
    # utf-8-sig tolera o BOM que o Excel acrescenta ao gravar.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def validate(path=DEFAULT_PATH):
    """Devolve a lista de erros; lista vazia significa ficheiro válido."""
    columns, rows = read_items(path)
    if columns != COLUMNS:
        return [f"colunas inesperadas (o separador tem de ser a vírgula): {columns}"]

    errors = []
    seen = set()
    # A linha 1 é o cabeçalho, por isso os itens começam na linha 2.
    for line, row in enumerate(rows, start=2):
        if None in row or None in row.values():
            errors.append(f"linha {line}: número de colunas errado")
            continue
        row = {k: v.strip() for k, v in row.items()}
        where = f"linha {line} ({row['item_id'] or 'sem item_id'})"

        for col in REQUIRED:
            if not row[col]:
                errors.append(f"{where}: {col} vazio")
        if row["item_id"] and row["item_id"] in seen:
            errors.append(f"{where}: item_id repetido")
        seen.add(row["item_id"])
        if row["type"] and row["type"] not in TYPES:
            errors.append(f"{where}: type inválido '{row['type']}'")
        if row["option_1_en"] and row["option_1_en"] == row["option_2_en"]:
            errors.append(f"{where}: as duas opções são iguais")

        if row["type"] == "dilemma":
            for col in ("domain", "value_1", "value_2"):
                if not row[col]:
                    errors.append(f"{where}: {col} vazio num dilema")
            if row["value_1"] and row["value_1"] == row["value_2"]:
                errors.append(f"{where}: value_1 e value_2 são iguais")
        elif row["type"] == "control":
            if row["value_1"] or row["value_2"]:
                errors.append(f"{where}: um controlo não pode ter value_1/value_2")
    return errors


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    errors = validate(path)
    for error in errors:
        print(error)
    if errors:
        sys.exit(f"{path}: {len(errors)} erro(s)")
    _, rows = read_items(path)
    n_dilemmas = sum(r["type"].strip() == "dilemma" for r in rows)
    print(f"{path}: válido, {n_dilemmas} dilemas e {len(rows) - n_dilemmas} controlos")


if __name__ == "__main__":
    main()
