"""
Explainability Module
Computes exact TreeSHAP attributions, beeswarms, waterfalls, and force plots.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from .constants import FEATURE_COLS, FEATURE_LABELS, CLASS_NAMES, OUT_DIR, OPERATOR_DESC


def compute_shap_explanations(model, X_test):
    """
    Computes exact TreeSHAP attributions and generates all artifact plots.
    """
    print("\n   -> Initializing SHAP TreeExplainer...")
    # Use Booster or model directly
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(os.path.join(OUT_DIR, "dispatch_engine.ubj"))
        explainer = shap.TreeExplainer(bst)

    sample_size = min(len(X_test), 2000)
    X_sample = X_test.iloc[:sample_size].copy()
    feature_display_names = np.array([FEATURE_LABELS.get(c, c) for c in FEATURE_COLS])

    print(f"   -> Computing Shapley values for {sample_size:,} operational instances...")
    shap_values = explainer(X_sample)

    # 1. Global Beeswarm plots per class
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(9, 7.5))
        plt.sca(ax)
        shap.summary_plot(
            shap_values[:, :, cls_idx].values,
            X_sample,
            feature_names=feature_display_names,
            show=False,
            max_display=15,
            plot_type="dot",
            color_bar_label="Feature value",
        )
        plt.title(f"SHAP Global Feature Attributions - Class: {cls_name}", fontsize=12, pad=10)
        plt.tight_layout()
        fname = os.path.join(OUT_DIR, f"shap_beeswarm_{cls_name.lower()}.png")
        plt.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"   -> Saved: {fname}")

    # 2. Overall Mean Absolute Feature Importance
    fig, ax = plt.subplots(figsize=(8, 6.5))
    plt.sca(ax)
    shap.summary_plot(
        shap_values.values,
        X_sample,
        feature_names=feature_display_names,
        class_names=CLASS_NAMES,
        show=False,
        max_display=15,
        plot_type="bar",
    )
    plt.title("Overall SHAP Feature Importance (Mean |SHAP Value|)", fontsize=12, pad=10)
    plt.tight_layout()
    bar_path = os.path.join(OUT_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"   -> Saved: {bar_path}")

    # 3. Local Waterfall plots for one sample of each class
    if hasattr(model, "predict"):
        preds = model.predict(X_sample)
    else:
        import xgboost as xgb
        preds = np.argmax(model.predict(xgb.DMatrix(X_sample)), axis=1)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        matches = np.where(preds == cls_idx)[0]
        if len(matches) == 0:
            continue
        sample_idx = matches[0]

        single_exp = shap.Explanation(
            values=shap_values.values[sample_idx, :, cls_idx],
            base_values=shap_values.base_values[sample_idx, cls_idx] if hasattr(shap_values.base_values, "__len__") else explainer.expected_value[cls_idx],
            data=X_sample.iloc[sample_idx].values,
            feature_names=feature_display_names,
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.sca(ax)
        shap.plots.waterfall(single_exp, max_display=10, show=False)
        plt.title(f"Local SHAP Waterfall Explanation - Sample Decision: {cls_name}", fontsize=11, pad=10)
        plt.tight_layout()
        wf_path = os.path.join(OUT_DIR, f"shap_waterfall_{cls_name.lower()}.png")
        plt.savefig(wf_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"   -> Saved: {wf_path}")

        # Interactive HTML force plot
        try:
            base_val = float(single_exp.base_values)
            force_plot = shap.force_plot(
                base_val,
                single_exp.values,
                X_sample.iloc[sample_idx].round(2),
                feature_names=feature_display_names,
            )
            html_path = os.path.join(OUT_DIR, f"shap_force_{cls_name.lower()}.html")
            shap.save_html(html_path, force_plot)
            print(f"   -> Saved: {html_path}")
        except Exception:
            pass

    # 4. Generate per-sample explanation CSV
    rows = []
    for i in range(min(50, len(X_sample))):
        pred_cls = int(preds[i])
        sv_pred = shap_values.values[i, :, pred_cls]
        top3_idx = np.argsort(np.abs(sv_pred))[-3:][::-1]

        drivers_str = "; ".join([
            f"{OPERATOR_DESC.get(FEATURE_COLS[fi], FEATURE_COLS[fi])} ({sv_pred[fi]:+.3f})"
            for fi in top3_idx
        ])

        rows.append({
            "sample_index": i,
            "predicted_mode": CLASS_NAMES[pred_cls],
            "solar_q50_kw": X_sample.iloc[i]["solar_power_q50"],
            "load_q50_kw": X_sample.iloc[i]["load_demand_q50"],
            "soh_pct": X_sample.iloc[i]["soh_pct"],
            "soc_pct": X_sample.iloc[i]["soc_pct"],
            "top_drivers": drivers_str,
        })

    csv_path = os.path.join(OUT_DIR, "per_sample_explanations.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"   -> Saved: {csv_path}")

    return explainer
