yy年度に卒業修了予定者のレポジトリテンプレート（Copilotと作成）。年度はOverleafと同じ。例：2026年3月卒業の最上叡智産の場合（topicは3語程度）

25-mogami-eichi-scene-flow

以下テンプレ。ここまで消す。

# <project title>

## Repo name (must)
- Format: `yy-family-given-topic[-purpose]`
- All lowercase, words separated by hyphens `-`
- `yy` = graduation/completion fiscal year (2 digits; same rule as Overleaf)
- Examples:
  - `25-mogami-eichi-scene-flow`
  - `25-mogami-eichi-scene-flow-code`
  - `25-mogami-eichi-scene-flow-data`

## Overview
- Author: <family given>
- FY (yy): <25>
- Topic keywords: <2-4 words>
- Upstream (if any): <URL or "none"> (reference repo/code URL / 参考URL)

## Environment (must)
- OS: <Ubuntu 22.04 / Windows 11 / etc.>
- Python: <e.g., 3.11.7>
- Key libs: <e.g., numpy==..., torch==..., opencv-python==...> (major libraries only / `pip freeze` から主要なものを抜粋)
- GPU/CUDA (if any): <e.g., CUDA 12.x>

## Setup
### Option A: pip
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Option B: conda (if you use environment.yml)
```bash
conda env create -f environment.yml
conda activate <env-name>
```

> **Note:** `data/` and `outputs/` are gitignored, so local data and experiment results can be placed there and will normally stay out of commits.
> `data/` と `outputs/` は `.gitignore` 済みなので、手元のデータや実験結果はそこに置いてよく、通常のコミットには入りません。

## Data policy (must)
### Public datasets
- Do NOT copy dataset files into this repository.
- Provide the exact URL (paper/official/DOI) and required subset description.

### Private / custom datasets
- Share via limited-access URL (GitHub/OneDrive/etc.).
- List required files explicitly (file path, name, size).
- If manual steps exist, describe them and share only the necessary files.

## How to run (reproducibility) (must)
1. Setup environment (above)
2. Get data (URLs above)
3. Run:
```bash
python <your_entrypoint>.py --config <config>
```
4. Expected outputs:
- <output path>
- <example filenames>

## Reproducibility check (must)
- Confirm the experiment can be reproduced on another machine.
- If any manual operation exists, write it below.

## Manual steps (if any)
- <GUI clicks / parameter edits / file edits, etc.>
