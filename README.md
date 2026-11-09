# LLMs Moral Compass

Do LLMs have a strong moral compass? Measuring how local LLMs choose between conflicting values, and how far those choices drift across languages and under contextual pressure.

## Requirements

- Windows 11, Python 3.11
- NVIDIA GPU with at least 8 GB of VRAM (developed on an RTX 4060 Laptop, driver 532.27, CUDA 12.1)

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` installs PyTorch from the CUDA 11.8 index. That build is the newest that runs on this driver.

## Check the environment

```powershell
.venv\Scripts\python.exe -m src.check_env
```

This checks that the GPU is visible, loads a small model in 4 bits (NF4) and generates one sentence. The first run downloads the model.

## Prompt and choice probabilities

The prompt template is frozen in `configs/prompt.yaml` (its `version` is recorded with every request). Labels stay `A`/`B` in every language, and each item runs in two orders: `AB` (option 1 shown as A) and `BA` (option 1 shown as B).

```powershell
# Validate translations and paraphrases, and print an example prompt
.venv\Scripts\python.exe -m src.prompts
# Print one prompt: item, language, condition (B0 or paraphrase), order
.venv\Scripts\python.exe -m src.prompts D08 pt-PT B0 BA
# Step 4 checks per model: label tokens, greedy vs argmax, sampling vs probabilities
.venv\Scripts\python.exe -m src.check_choice
```

`src.check_choice` writes its results to `data/results/checks/`. In code, one request is:

```python
from src.models import load_pinned, choice_probs
from src.prompts import build, option_probs

tok, model, entry = load_pinned("qwen3-8b")
r = choice_probs(tok, model, entry, build("D01", "en", "B0", "BA"))
# r: p_A, p_B, p_other, p_A_main, p_B_main, top_token, top_prob, n_tokens
option_probs(r["p_A"], r["p_B"], "BA")  # p_opt1, p_opt2, p_v1
```

`p_A` adds up every single token that decodes to `A` once whitespace is stripped (for example `A`, `ĠA`, `▁A`). `p_other` is the probability left outside A and B. It covers refusals ("This…") as well as answers that start with formatting such as `**`. `top_token` tells the two apart.

## Layout

| Path | Contents |
|---|---|
| `data/dilemmas.csv` | Master items in English (added in step 2) |
| `data/translations/` | One file per language, `<language>.csv` |
| `data/paraphrases/` | Paraphrased scenarios, `<language>.csv` |
| `data/results/` | Raw outputs |
| `src/` | Model loading, prompts, run orchestration, analysis |
| `notebooks/` | Figures and tables |
| `configs/` | Models, versions, seeds |
| `PREREGISTRATION.md` | Hypotheses and thresholds, committed before the main run |
