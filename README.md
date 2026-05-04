# CAV-HDV Lane Merging Safety Analysis

> **Paper:** Empirical Analysis of HDV Merging Safety in CAV-Adjacent Highway Zones: A Variational Bayesian and Offline Reinforcement Learning Framework  
> **Author:** Md Sifat Bin Siraj  
> **Dataset:** TGSIM I-395, Washington D.C.

---

## Overview

This repository contains the complete analysis pipeline for studying **HDV (Human-Driven Vehicle) merge safety** in highway zones where **CAVs (Connected & Automated Vehicles)** are present. The study uses real naturalistic trajectory data — no simulator.

---

## Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | `data_check` | Raw data validation and quality check |
| 2 | `data_clean` | Cleaning, renaming, parquet cache |
| 3 | `safety` | TTC, PET, gap, CAV proximity metrics |
| 4 | `c1` | Variational Bayesian IDM estimation |
| 5 | `c2` | Attention-based LSTM Seq-to-Seq model |
| 6 | `c3` | CQL offline RL policy |
| 7 | `simulation` | Python IDM simulation validation |
| 8 | `figures` | All paper figures (300 DPI) |
| 9 | `tables` | All paper tables (CSV) |
| 10 | `policy_analysis` | Entropy, KL divergence, Q-gap analysis |

---

## Key Results

- **2,155** HDV merge events analyzed
- **52.2%** of merges had PET < 2s (unsafe)
- CAV speed ≥ 15 m/s reduces critical merge rate: **60.7% → 33.3%**
- ANOVA: F = 39.28, p < 0.001 (PET across CAV speed bins)
- C2 best validation MSE: **0.0706** (vs GRU: 0.1202, LR: 0.1158)
- CQL action match: **93.4%** | Naive baseline: **93.8%**

---

## Installation

```bash
pip install numpy pandas scipy matplotlib torch scikit-learn pyarrow
```

---

## How to Run

**Full pipeline:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --output results/
```

**Resume after interruption:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --resume c3
```

**Single step only:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --only figures
```

---

## Dataset

**TGSIM I-395** — publicly available, CC0 license  
🔗 [Download here] https://data.transportation.gov/Automobiles/Third-Generation-Simulation-Data-TGSIM-I-395-Traje/97n2-kuqi/about_data

Place the CSV file at: `data/TGSIM_I395.csv`

---

## Output Structure

results/
├── figures/          # Fig 1–8 (300 DPI PNG)
├── tables/           # CSV tables
├── checkpoints/
│   ├── c2/           # LSTM model weights
│   └── c3/           # CQL policy weights
└── data/processed/   # Parquet cache files

---

## Citation

If you use this code, please cite our paper once published.

---

## License

Code: MIT | Dataset: CC0 (TGSIM I-395)
