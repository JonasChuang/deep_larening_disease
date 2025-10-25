#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explain a trained GRU time-series classifier (N, T, F) with SHAP.
Outputs:
  - shap_global_features.png   (Feature importance aggregated over time & samples)
  - shap_time_importance.png   (Time-step importance aggregated over samples & features)
  - shap_heatmap_case0.png     (Per-case time × feature SHAP heatmap)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from Hyperkalemia import Hyperkalemia
# SHAP
import shap

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    """Make results more reproducible (to the extent possible)."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def sample_background(X_train: np.ndarray, size: int = 128, seed: int = 42) -> np.ndarray:
    """Pick a small background set for SHAP (balances speed & stability)."""
    size = min(size, len(X_train))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=size, replace=False)
    return X_train[idx]

def pick_explain_set(X: np.ndarray, k: int = 64) -> np.ndarray:
    """Pick the first-k samples to explain (adjust as needed)."""
    return X[: min(k, len(X))]

def save_barh(values: np.ndarray, labels: list, top: int, outfile: str, title: str, xlabel: str):
    order = np.argsort(values)[::-1][:top]
    vals = values[order][::-1]
    labs = [labels[i] for i in order][::-1]

    plt.figure(figsize=(6, 0.35 * len(vals) + 2))
    plt.barh(range(len(vals)), vals)
    plt.yticks(range(len(vals)), labs)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def save_line(x, y, outfile: str, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(7, 3))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def save_heatmap(matrix_2d: np.ndarray, feat_names: list, outfile: str, title: str):
    # matrix_2d shape: (T, F)
    vabs = np.max(np.abs(matrix_2d)) if np.any(matrix_2d) else 1.0
    vmin, vmax = -vabs, vabs
    plt.figure(figsize=(9, 0.35 * len(feat_names) + 2))
    plt.imshow(matrix_2d.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label="SHAP value (+ pushes risk up)")
    plt.yticks(range(len(feat_names)), feat_names)
    plt.xlabel("Time index (hours since t0)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# -----------------------------
# SHAP core
# -----------------------------
def compute_shap_values(model, X_background: np.ndarray, X_explain: np.ndarray):
    """
    Return:
        shap_values: np.ndarray, shape (k, T, F)
        base_values: float or array-like (expected value)
    Strategy:
        Try modern shap.Explainer; if it fails, fallback to DeepExplainer.
    """
    # Prefer the modern API (works for tf/keras models on many setups)
    try:
        explainer = shap.Explainer(model, X_background)
        explanation = explainer(X_explain)
        shap_values = explanation.values
        base_values = explanation.base_values
        return shap_values, base_values
    except Exception as e1:
        print("[Info] shap.Explainer failed, fallback to DeepExplainer. Reason:", repr(e1))
        try:
            deep_explainer = shap.DeepExplainer(model, X_background)
            sv_list = deep_explainer.shap_values(X_explain)
            shap_values = sv_list[0] if isinstance(sv_list, list) else sv_list
            base_values = getattr(deep_explainer, "expected_value", 0.0)
            return shap_values, base_values
        except Exception as e2:
            print("[Error] SHAP computation failed with DeepExplainer as well:", repr(e2))
            raise

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        OUTDIR=Hyperkalemia.OUTDIR+"//gru_hk_model.keras"
        parser = argparse.ArgumentParser(description="Explain GRU model with SHAP")
        # parser.add_argument("--model_path", default="./artifacts_hypona_seq/gru_hypona_model.keras",
        #                     help="Path to trained Keras model (.keras/.h5)")

        parser.add_argument("--model_path", default=OUTDIR,
                            help="Path to trained Keras model (.keras/.h5)")
        
        parser.add_argument("--X_train", default=None, help="Numpy .npy path for train tensor (N,T,F)")
        parser.add_argument("--X_test",  default=None, help="Numpy .npy path for test tensor (N,T,F)")
        parser.add_argument("--features", default=None, help="Numpy .npy or .txt for feature names (F,)")
        parser.add_argument("--outdir", default="./artifacts_hypona_seq", help="Output directory")
        parser.add_argument("--bg_size", type=int, default=128, help="Background sample size for SHAP")
        parser.add_argument("--k", type=int, default=64, help="Number of samples to explain")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        args = parser.parse_args()

        set_seed(args.seed)
        ensure_outdir(args.outdir)

        # ---- Load model
        print("[Load] model:", args.model_path)
        model = keras.models.load_model(args.model_path, compile=False)

        # ---- Load data
        np.save("X_train.npy", X_train)   # shape (N_train, T, F)
        np.save("X_test.npy", X_test)

        if args.X_train is None or args.X_test is None:
            raise ValueError("Please provide --X_train and --X_test .npy files of shape (N,T,F).")
        X_train = np.load(args.X_train)
        X_test  = np.load(args.X_test)

        # ---- Feature names
        if args.features is None:
            feat_names = [f"feat_{i}" for i in range(X_train.shape[2])]
        else:
            if args.features.endswith(".npy"):
                feat_names = list(np.load(args.features))
            else:
                # assume a newline-delimited text file
                with open(args.features, "r", encoding="utf-8") as f:
                    feat_names = [ln.strip() for ln in f if ln.strip()]
            if len(feat_names) != X_train.shape[2]:
                raise ValueError(f"Feature name length mismatch: got {len(feat_names)}, need {X_train.shape[2]}.")

        # ---- Background & explain sets
        X_bg = sample_background(X_train, size=args.bg_size, seed=args.seed)
        X_explain = pick_explain_set(X_test, k=args.k)

        print("[Info] X_bg:", X_bg.shape, "X_explain:", X_explain.shape)

        # ---- Compute SHAP
        shap_values, base_values = compute_shap_values(model, X_bg, X_explain)
        # shap_values shape should be (k, T, F)
        if shap_values.ndim != 3:
            raise RuntimeError(f"Unexpected SHAP shape {shap_values.shape}, expected (k, T, F).")

        print("[OK] SHAP values shape:", shap_values.shape)

        # ---- 3 Plots
        # 1) Feature importance (aggregate over samples & time)
        feat_importance = np.mean(np.abs(shap_values), axis=(0, 1))  # (F,)
        save_barh(
            values=feat_importance,
            labels=feat_names,
            top=min(15, len(feat_names)),
            outfile=os.path.join(args.outdir, "shap_global_features.png"),
            title="Top features by mean |SHAP|",
            xlabel="Mean |SHAP| (global importance)"
        )

        # 2) Time importance (aggregate over samples & features)
        T = shap_values.shape[1]
        time_importance = np.mean(np.abs(shap_values), axis=(0, 2))  # (T,)
        save_line(
            x=list(range(T)),
            y=time_importance,
            outfile=os.path.join(args.outdir, "shap_time_importance.png"),
            title="Time-step importance by mean |SHAP|",
            xlabel="Time index (hours since t0)",
            ylabel="Mean |SHAP|"
        )

        # 3) One-case heatmap (time × feature)
        case_idx = 0
        case_matrix = shap_values[case_idx]  # (T, F)
        save_heatmap(
            matrix_2d=case_matrix,
            feat_names=feat_names,
            outfile=os.path.join(args.outdir, "shap_heatmap_case0.png"),
            title=f"Case #{case_idx}: SHAP heatmap (time × feature)"
        )

        print("[Done] Figures saved to:", os.path.abspath(args.outdir))
    except Exception as Error:
        print(str(Error))

