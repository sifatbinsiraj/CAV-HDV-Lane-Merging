# Simulation-Free Offline RL for CAV-HDV Highway Merge Safety

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Under%20Review-orange)](.)

> **"Simulation-Free Offline Reinforcement Learning for CAV-HDV Highway Merge Safety Using Naturalistic Trajectory Data"**

---

## Overview

This repository contains the complete analysis pipeline for our paper on CAV-HDV highway merge safety. We develop a **three-component, simulation-free framework** trained exclusively on real naturalistic trajectory data from the TGSIM I-395 dataset:

| Component | Method | Key Result |
|---|---|---|
| **C1** — Behavioral Estimation | Variational Bayesian IDM | Bimodal headway: 0.89s (aggressive) / 2.11s (conservative) |
| **C2** — Response Prediction | Attention-based LSTM Seq2Seq | Val MSE: 0.0706 (vs mean baseline 0.1163) |
| **C3** — Policy Learning | CQL Offline RL | 93.4% action match; soft entropy 0.9825 bits |

**Key finding:** CAVs traveling at ≥15 m/s reduce critical merge event rates by ~27 percentage points (60.7% → 33.3%), supporting a practical speed advisory for mixed-traffic corridors.

---

## Dataset

**TGSIM I-395** — Third-Generation Simulation Data, Washington D.C.
- 4.32M trajectory records
- 21 CAV units, ~2 hours peak-hour data
- 2,155 HDV merge events analyzed
- **Publicly available (CC0 license):**
  https://data.transportation.gov/Automobiles/Third-Generation-Simulation-Data-TGSIM-I-395-Traje/97n2-kuqi

---

## Installation

```bash
git clone https://github.com/[author]/cav-merging-safety.git
cd cav-merging-safety
pip install -r requirements.txt
```

---

## Quick Start

**Full pipeline:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --output results/
```

**Resume after interruption:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --resume c3
```

**Single step:**
```bash
python complete_analysis.py --input data/TGSIM_I395.csv --only policy_analysis
```

---

## Pipeline Steps

| Step | Name | Description |
|---|---|---|
| 1 | `data_check` | Raw data validation and quality report |
| 2 | `data_clean` | Cleaning, column renaming, parquet cache |
| 3 | `safety` | TTC, PET, gap acceptance, CAV proximity + ANOVA |
| 4 | `c1` | Variational Bayesian IDM per-vehicle estimation |
| 5 | `c2` | Attention LSTM Seq2Seq + baseline comparison |
| 6 | `c3` | CQL offline RL policy (no simulator) |
| 7 | `simulation` | IDM-based policy validation |
| 8 | `figures` | All paper figures at 300 DPI |
| 9 | `tables` | All paper tables as CSV |
| 10 | `policy_analysis` | Entropy, KL, Q-gap, bootstrap CIs |

---

## Key Results

### Safety Analysis (Paper Table 8)

| CAV Speed (m/s) | n | Critical Rate | 95% CI |
|---|---|---|---|
| 0–5 | 28 | 60.7% | 42.9%–78.6% |
| 5–10 | 133 | 47.4% | 39.1%–55.6% |
| 10–15 | 77 | 50.6% | 39.0%–62.3% |
| **15+** | **42** | **33.3%** | **19.0%–47.6%** |

ANOVA: F = 39.28, p < 0.001 | Kruskal-Wallis: H = 37.90, p < 0.001

> **Note:** Speed-safety relationship is *associational*, not causal. Unobserved confounders (traffic density, time-of-day) are not controlled.

### C2 Model Comparison (Paper Table 5)

| Model | Val MSE |
|---|---|
| Mean response baseline | 0.1163 |
| Linear Regression | 0.1158 |
| GRU (2-layer) | 0.1202 |
| MLP (2-layer) | 0.1621 |
| **LSTM+Attention (proposed)** | **0.0706** |

### Policy Comparison (Paper Table 11)

| Metric | Naive | CQL |
|---|---|---|
| Action match (%) | 93.8 (92.7–94.8) | 93.4 (92.3–94.5) |
| Policy entropy (bits) | 0.00 | 0.98 |
| Effective actions | 1.00 | 1.98 |
| Non-Maintain actions | 0 (0%) | 7 (0.36%) |
| Q-gap (critical states) | — | 0.39 |
| Q-gap (Maintain states) | — | 2.49 |

> CIs overlap → no statistically significant difference in action match. CQL adds Q-value structure and policy diversity that the naive baseline cannot provide.

---

## Output Structure

```
results/
├── data/
│   └── processed/
│       ├── tgsim_clean.parquet
│       ├── safety_metrics.parquet
│       └── c1_bayesian_params.parquet
├── checkpoints/
│   ├── c2/  (c2_final.pt, training_log.csv)
│   └── c3/  (c3_final.pt, transitions.parquet, training_log.csv)
├── figures/
│   ├── fig1_trajectory_map.png
│   ├── fig2_safety_metrics.png
│   └── ...
├── tables/
│   ├── table8_cav_speed.csv
│   ├── table11_policy_comparison.csv
│   ├── speed_bin_bootstrap_ci.csv
│   └── ...
└── PROGRESS.json
```

---

## Citation

If you use this code or data in your research, please cite:



## License

This code is released under the MIT License. The TGSIM dataset is publicly available under CC0 (see dataset link above).
