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

## Layout

| Path | Contents |
|---|---|
| `data/dilemmas.csv` | Master items in English (added in step 2) |
| `data/translations/` | One file or folder per language |
| `data/results/` | Raw outputs |
| `src/` | Model loading, prompts, run orchestration, analysis |
| `notebooks/` | Figures and tables |
| `configs/` | Models, versions, seeds |
| `PREREGISTRATION.md` | Hypotheses and thresholds, committed before the main run |
