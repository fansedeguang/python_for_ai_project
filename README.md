# python_for_ai_project
Topic A: Text Classification Pipeline Forensics  
Text Classification Pipeline Forensics: Diagnose a noisy disaster-tweet classifier, reproduce a reference baseline, and explain which pipeline changes are genuinely trustworthy.

**`setup_dirs.py`** script that creates the required folder structure automatically.

# Disaster Tweets Pipeline Forensics – Topic A

This repository implements a full diagnostic pipeline for the Kaggle Disaster Tweets classification task, following the course assignment specifications. All experiments are CPU‑only and use the fixed `train/dev/heldout` split provided in `data/split_indices.json`.

## Project Structure

```
.
├── data/                         # Raw data (download automatically)
│   ├── train.csv                 # Full labeled dataset (from GitHub mirror)
│   └── split_indices.json        # Fixed split indices (provided)
├── configs/                      # Reference contracts
│   └── project_contract.json     # Baseline metrics & tolerance
├── pipeline/                     # Reusable modules (no ticket‑specific logic)
│   ├── __init__.py
│   ├── loader.py                 # Load CSV + apply fixed split
│   ├── preprocess.py             # Text cleaning functions
│   ├── features.py               # TF‑IDF + shallow features
│   ├── models.py                 # Model training (LR, NB, SVM, etc.)
│   ├── evaluate.py               # Metrics, confusion deltas, PR curves
│   └── utils.py                  # Seed, version assertions, helpers
├── experiments/                  # Scripts for each ticket
│   ├── run_baseline.py           # Ticket 1
│   ├── run_normalization.py      # Ticket 2
│   ├── run_shortcut_audit.py     # Ticket 3
│   ├── run_threshold_tuning.py   # Ticket 4
│   ├── run_data_audit.py         # Ticket 5
│   └── run_final_heldout.py      # Final evaluation (after freezing)
├── tickets/                      # Qualitative analysis (Markdown)
│   ├── ticket-1-baseline.md
│   ├── ticket-2-normalization.md
│   ├── ticket-3-shortcuts.md
│   ├── ticket-4-decision-rule.md
│   └── ticket-5-data-quality.md
├── predictions/                  # Prediction exports
│   ├── heldout_predictions.csv   # Required final artifact
│   └── dev_predictions/          # Optional intermediate predictions
├── results/                      # Machine‑checkable summaries
│   ├── summary.csv               # Must include fixed_fp/fixed_fn etc.
│   ├── threshold_sweep.csv       # Precision/Recall/F1 sweep
│   ├── data_quality_audit.csv    # Disposition column required
│   └── figures/                  # Optional plots
├── logs/                         # Experiment logs & AI chat history
│   └── chat.md                   # AI usage declaration
├── tests/                        # Validation scripts (schema checks)
│   ├── test_split_integrity.py
│   ├── test_artifact_schema.py
│   └── test_reproducibility.py
├── requirements.txt              # Locked dependencies
├── download_data.py              # Script to fetch train.csv from mirror
├── setup_dirs.py                 # Creates all necessary folders
├── Makefile                      # Optional: `make run_all`, `make validate`
└── report.pdf                    # Final project report
```

## Environment Setup

We recommend using a **virtual environment** to isolate dependencies.

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 2. Install required packages

All package versions are pinned in `requirements.txt` for reproducibility.

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

The core packages are:

- Python 3.12.7
- scikit-learn 1.9.0
- pandas 3.0.4
- numpy 2.4.6
- (optional) PyTorch 2.7.0 (CPU) – if you add a neural baseline

## Data Download

The assignment uses only the **labeled `train.csv`** from the official mirror.  
Run the provided download script:

```bash
python download_data.py
```

This will create the `data/` folder and fetch the file.  
The script automatically skips download if the file already exists.

> **Note:** `test.csv` and `sample_submission.csv` are **not used** in this project.

## Create Folder Structure

If any of the above directories are missing, run:

```bash
python setup_dirs.py
```

This script creates all required folders (`data/`, `pipeline/`, `experiments/`, etc.) and a few placeholder files to keep the structure valid.

## Reproducing Experiments

All experiments read the fixed split from `data/split_indices.json`.  
The baseline is implemented in `experiments/run_baseline.py`.  
To run all tickets in sequence:

```bash
python experiments/run_baseline.py
python experiments/run_normalization.py --variants all
python experiments/run_shortcut_audit.py
python experiments/run_threshold_tuning.py
python experiments/run_data_audit.py
# After all tuning is frozen:
python experiments/run_final_heldout.py
```

You can also use the provided `Makefile` (if created) for one‑click execution.

## Validation Before Submission

Before packaging your final submission, run the schema validators:

```bash
python tests/test_artifact_schema.py
python tests/test_split_integrity.py
python tests/test_reproducibility.py
```

These checks ensure that:

- `heldout_predictions.csv` contains only held‑out IDs.
- `results/summary.csv` has the required columns (`fixed_fp`, `fixed_fn`, etc.).
- `results/data_quality_audit.csv` uses only valid `disposition` values (`fix`, `keep_but_flag`, `ambiguous`, `reject_false_positive`).

## AI Usage Declaration

We used AI assistants (e.g., ChatGPT) to generate code skeletons and regular expressions. All AI‑generated code was manually verified on at least 50 samples; the validation logs and reasoning are recorded in `logs/chat.md`. All external sources (datasets, libraries, tools) are properly cited in the final `report.pdf`.

## Contact

For any issues, please refer to the course forum or contact the teaching staff.
```

---



## How to Use

1. **Place** `README.md`, `download_data.py`, `setup_dirs.py`, and `requirements.txt` in your project root.
2. Run the setup to create the folder skeleton:
   ```bash
   python setup_dirs.py
   ```
3. Download the data:
   ```bash
   python download_data.py
   ```
4. Create and activate a virtual environment, then install dependencies as described in the README.

These two files complete the initialisation phase of your project. You can now begin implementing the pipeline modules and ticket experiments.