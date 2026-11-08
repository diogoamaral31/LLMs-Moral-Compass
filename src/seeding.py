"""Único sítio onde as seeds são fixadas. O valor vive em configs/seeds.yaml."""
import random
from pathlib import Path

import numpy as np
import torch
import yaml

SEEDS_FILE = Path(__file__).resolve().parents[1] / "configs" / "seeds.yaml"


def set_seeds(seed=None):
    """Fixa random, numpy e torch (CPU e GPU) e devolve a seed usada."""
    if seed is None:
        seed = yaml.safe_load(SEEDS_FILE.read_text(encoding="utf-8"))["global_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
