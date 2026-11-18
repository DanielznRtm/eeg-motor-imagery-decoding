# EEG Motor Imagery Decoding

Decoding imagined movements (left hand, right hand, feet, tongue) from 22-channel
EEG, comparing a classical CSP + LDA baseline, EEGNet and a convolutional-recurrent
model on BCI Competition IV dataset 2a.

> **Status: work in progress.** Phase 1 (setup + exploratory analysis) is done.
> Results tables will be added as each model is trained; every number in this
> README is produced by a script in this repository and saved under `results/`.

## Protocol

- **Data:** BCI Competition IV 2a (MOABB `BNCI2014_001`), 9 subjects, 2 sessions
  of 288 trials each, 4 balanced classes.
- **Evaluation:** within-subject, train on session 1 (`0train`), test on
  session 2 (`1test`), as in the competition. Metrics: accuracy and Cohen's kappa
  per subject, mean ± std across subjects, confusion matrices.
- **No leakage:** normalization statistics, early stopping and hyper-parameters
  use only session 1 (with a stratified 20% validation split); session 2 is used
  once, for the final evaluation. The exploratory analysis also uses session 1 only.

## Setup

Python 3.11.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the data (~750 MB, stored by MNE in `~/mne_data`) and write a trial-count
summary to `results/data_summary.csv`:

```bash
python -m src.data
```

Run the tests (`-m "not data"` skips the ones that need the dataset):

```bash
pytest
```

## Exploratory analysis

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb): raw signals, PSD per class,
ERD/ERS time-frequency maps at C3/Cz/C4, mu-band topographies and per-subject
lateralization. Re-run it with:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## License

MIT
