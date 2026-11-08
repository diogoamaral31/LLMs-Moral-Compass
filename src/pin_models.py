"""Regista em configs/models.yaml os hashes dos ficheiros de cada revisão fixada.

Usa só metadados do Hugging Face, por isso não descarrega os pesos.
Uso: python -m src.pin_models
"""
from fnmatch import fnmatch

import yaml
from huggingface_hub import HfApi

from src.models import CONFIG_FILE, load_config

PATTERNS = ["*.safetensors", "*.json", "*.jinja", "*.model", "*.txt"]


def file_hashes(repo_id, revision):
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files = {}
    for sibling in info.siblings:
        if not any(fnmatch(sibling.rfilename, p) for p in PATTERNS):
            continue
        if sibling.lfs:
            files[sibling.rfilename] = {"sha256": sibling.lfs.sha256, "size": sibling.size}
        else:
            files[sibling.rfilename] = {"git_blob": sibling.blob_id, "size": sibling.size}
    return files


def main():
    config = load_config()
    for name, entry in config["models"].items():
        entry["files"] = file_hashes(entry["id"], entry["revision"])
        print(f"{name}: {len(entry['files'])} ficheiros")
    CONFIG_FILE.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
