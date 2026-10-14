# ⚡ Microgrid Dispatch Classifier & SHAP Explainability Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Dispatch_Classifier-EB5424)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-brightgreen)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A self-contained, reproducible machine learning pipeline that trains an XGBoost dispatch classifier on public microgrid and battery degradation data, evaluates accuracy via a stratified hold-out test set, and generates exact TreeSHAP local and global explainability artifacts.

---

## 1. Quickstart (One-Command Execution)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python run.py
```

Execution takes **~15 to 20 seconds**. It will:
1. Load public forecast records and NASA battery degradation time-series from `./data/`.
2. Engineer 34 operational features (net surplus, forecast uncertainties, cyclical time encodings, battery states).
3. Assign physical expert policies (`CONSERVATIVE`, `MODERATE`, `AGGRESSIVE`) and balance class distributions.
4. Train an XGBoost tree ensemble with stratified 80/20 splitting and early stopping.
5. Compute exact TreeSHAP attributions and save 13 visual explanation artifacts to `./outputs/`.
6. Run live operational scenarios and print decision rationales with top-5 SHAP drivers to the console.

---

## 2. Directory Layout

```text
dispatch_classifier/
│
├── run.py                 # Single-command pipeline entrypoint
├── requirements.txt       # Minimal dependencies
├── README.md              # Pipeline documentation
│
├── data/                  # Public source datasets
│   ├── bihar_predictions.csv         # 48h solar, wind, and village load forecast quantiles
│   └── nasa_2yr_all_batteries.csv    # NASA lithium-ion battery cycling & impedance records
│
├── outputs/               # Generated automatically by run.py
│   ├── dispatch_engine.ubj           # Trained XGBoost model binary
│   ├── confusion_matrix.png          # Test set evaluation heatmap
│   ├── shap_beeswarm_*.png           # Global beeswarm plots per class
│   ├── shap_waterfall_*.png          # Local waterfall plots per class
│   ├── shap_feature_importance.png   # Global mean |SHAP| ranking
│   ├── shap_force_*.html             # Interactive D3 force plots
│   └── per_sample_explanations.csv   # Attributions and top drivers dataset
│
└── src/
    ├── __init__.py
    ├── constants.py           # Feature definitions, labels, and operator descriptions
    ├── data_loader.py         # Ingestion and temporal alignment of public CSVs
    ├── feature_engineering.py # Computation of 34 physical microgrid indicators
    ├── label_generator.py     # Deterministic physical policies & augmentation
    ├── train.py               # XGBoost training, early stopping, and metric evaluation
    ├── explainability.py      # TreeSHAP computation & visualization generator
    └── inference.py           # Live scenario simulator & plain-language reporting
```

---

## 3. Physical Dispatch Policies

The classifier selects between three operational strategies:

| Class | Mode | Strategy | Trigger Conditions |
| :---: | :--- | :--- | :--- |
| **0** | **CONSERVATIVE** | Deficit protection & equipment preservation | Battery $SOH < 73\%$, $SOC < 22\%$, cell temp $> 50^\circ\text{C}$, or critical deficit with no grid. |
| **1** | **MODERATE** | Balanced nominal operation | Routine operation; solar/wind balances demand with modest battery cycling. |
| **2** | **AGGRESSIVE** | Renewable maximization | Solar $> 4\text{ kW}$ covering $>15\%$ of load, battery $>65\%$ healthy ($SOH \ge 80\%$), or large surplus. |

---

## 4. Evaluation & Performance

* **Hold-out Strategy**: 80% Training / 20% Held-out Test Set with `stratify=y` to preserve exact class proportions.
* **Accuracy**: ~**99.4%** on unseen test data.
* **Evaluation Matrix**: Saved to `outputs/confusion_matrix.png`.

> **Note on ~99% Accuracy (Policy Distillation):**  
> Ground-truth labels are generated using deterministic physical microgrid operating rules. Because all boundary inputs (e.g. `soh_pct`, `soc_pct`, `renewable_cover`, `cell_temp_c`) are columns in feature matrix $X$, gradient boosted decision trees (XGBoost) easily find the exact axis-aligned split thresholds. This demonstrates successful **surrogate policy modeling** (distilling complex multi-rule systems into sub-millisecond executable machine learning).

---

## 5. Explainable AI with TreeSHAP

Rather than treating the model as a black box, the system uses **TreeSHAP** to calculate exact Shapley values from cooperative game theory:

1. **Global Explanations**: `shap_beeswarm_*.png` shows how features across the entire operational history push decisions toward or away from each dispatch mode.
2. **Local Waterfall Plots**: `shap_waterfall_*.png` shows the step-by-step feature attributions that drove a specific operational decision.
3. **No Hallucinations**: Top drivers are formatted into plain-language sentences using deterministic templates based on exact feature weights without generative language models.
