"""
Ckd 檢驗報告 × 死亡率預測（deep Learning, My Sql→keras）
End-to-end pipeline: MIMIC-IV (MySQL) CKD cohort lab features -> Deep Learning mortality prediction
- Reads a prepared feature table `ckd_lab_features_24h` (one row per hadm_id)
- Target: 30-day mortality (mort_30d) or in-hospital mortality (in_hosp_mort)
- Preprocess: median impute + standardize
- Model: Keras MLP (BN + Dropout), class-balanced training, early stopping
- Calibration: Platt scaling (logistic regression on validation probs)
- Evaluation: AUC, AUPRC, F1 (best PR-threshold), Brier, confusion matrix
- Saves: model (.keras), preprocessing (.joblib), calibrator (.joblib), and test predictions CSV

Before running:
  pip install pandas SQLAlchemy PyMySQL scikit-learn tensorflow xgboost shap joblib
Set DB credentials below (USER, PWD, HOST, DB)
"""

""""
RANDOM_STATE:
資料科學 / 機器學習 / 深度學習程式碼裡看到，用來設定隨機數生成器（random number generator, RNG）的種子值
很多演算法（例如 train/test split、隨機森林、神經網路權重初始化）都會用到「隨機數」。

如果不固定種子 → 每次程式跑出來的結果都會不同（資料分割不同、模型權重初始化不同）。

如果設定固定種子（例如 RANDOM_STATE = 42）→ 每次執行都會得到相同的隨機序列，結果可重現（reproducible）
42 很常出現（因為 Douglas Adams 的《銀河便車指南》中「宇宙最終答案」是 42 🤓）

其實任何整數都可以（例如 0, 1, 2025...）

"""
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import keras
from keras import layers

# ============== USER CONFIG ==============
USER = os.getenv("MIMIC_MYSQL_USER", "user")
PWD  = os.getenv("MIMIC_MYSQL_PWD",  "password")
HOST = os.getenv("MIMIC_MYSQL_HOST", "127.0.0.1")
PORT = int(os.getenv("MIMIC_MYSQL_PORT", "3306"))
DB   = os.getenv("MIMIC_MYSQL_DB",   "mimiciv")
TABLE = os.getenv("CKD_FEATURE_TABLE", "ckd_lab_features_24h")
TARGET = os.getenv("CKD_TARGET", "mort_30d")  # or "in_hosp_mort"
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE  = float(os.getenv("VAL_SIZE",  "0.2"))  # portion of train for validation
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
OUTDIR = os.getenv("OUTDIR", "./artifacts_ckd_dl")

# ============== UTILS ==============

def set_reproducible(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def db_engine() -> str:
    return f"mysql+pymysql://{USER}:{PWD}@{HOST}:{PORT}/{DB}"


def load_table(engine_str: str, table: str) -> pd.DataFrame:
    eng = create_engine(engine_str)
    return pd.read_sql(f"SELECT * FROM {table}", eng)


def split_train_val_test(X: pd.DataFrame, y: pd.Series,
                         test_size: float, val_size: float,
                         random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # further split train->train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_mlp(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="mortality_prob")(x)

    model = keras.Model(inputs, outputs, name="ckd_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def best_pr_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1s))
    # precision_recall_curve returns thresholds len = n_points-1
    if i == 0 or i > len(thr):
        return 0.5
    return float(thr[i-1])


def evaluate_block(y_true: np.ndarray, y_prob: np.ndarray, label: str = "TEST") -> dict:
    thr = best_pr_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "threshold": thr,
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "CM": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n[{label}] thr={out['threshold']:.3f}  AUC={out['AUC']:.4f}  AUPRC={out['AUPRC']:.4f}  F1={out['F1']:.4f}  Brier={out['Brier']:.4f}")
    print("CM:", np.array(out["CM"]))
    return out


def compute_class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    set_reproducible(RANDOM_STATE)
    ensure_outdir(OUTDIR)

    print("\n=== Loading feature table ===")
    df = load_table(db_engine(), TABLE)

    id_cols = [c for c in ["subject_id", "hadm_id", "t0", "stay_id", "intime"] if c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in table {TABLE}")

    # Drop all-null columns
    feature_cols = [c for c in df.columns if c not in id_cols + [TARGET]]
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).values

    # Split
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_train_val_test(X, y, TEST_SIZE, VAL_SIZE, RANDOM_STATE)

    # Preprocess (fit on TRAIN only)
    prep = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler(with_mean=True)),
    ])
    X_tr_np = prep.fit_transform(X_tr)
    X_val_np = prep.transform(X_val)
    X_te_np = prep.transform(X_te)

    # Model
    model = build_mlp(input_dim=X_tr_np.shape[1])
    cw = compute_class_weights(y_tr)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=5, factor=0.5, min_lr=1e-5),
    ]

    print("\n=== Training MLP (class-weighted) ===")
    history = model.fit(
        X_tr_np, y_tr,
        validation_data=(X_val_np, y_val),
        epochs=200,
        batch_size=256,
        callbacks=callbacks,
        class_weight=cw,
        verbose=2,
    )

    print("\n=== Validation performance (raw) ===")
    val_prob = model.predict(X_val_np, batch_size=1024).ravel()
    val_metrics = evaluate_block(y_val, val_prob, label="VAL (raw)")

    print("\n=== Test performance (raw) ===")
    te_prob = model.predict(X_te_np, batch_size=1024).ravel()
    te_metrics = evaluate_block(y_te, te_prob, label="TEST (raw)")

    # ---- Calibration: Platt scaling on validation set ----
    print("\n=== Calibration (Platt on VAL) ===")
    platt = LogisticRegression(max_iter=1000)
    platt.fit(val_prob.reshape(-1,1), y_val)
    te_prob_cal = platt.predict_proba(te_prob.reshape(-1,1))[:,1]
    te_metrics_cal = evaluate_block(y_te, te_prob_cal, label="TEST (calibrated)")

    # ---- Save artifacts ----
    print("\n=== Saving artifacts ===")
    model_path = os.path.join(OUTDIR, "ckd_mlp.keras")
    prep_path  = os.path.join(OUTDIR, "ckd_preprocess.joblib")
    platt_path = os.path.join(OUTDIR, "ckd_platt.joblib")
    report_path= os.path.join(OUTDIR, "ckd_metrics.json")

    model.save(model_path)
    joblib.dump(prep,  prep_path)
    joblib.dump(platt, platt_path)

    metrics_all = {
        "val_raw": val_metrics,
        "test_raw": te_metrics,
        "test_calibrated": te_metrics_cal,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "feature_cols": feature_cols,
        "target": TARGET,
        "class_weights": cw,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)

    # ---- Save test predictions for audit ----
    te_df = X_te.copy()
    te_df[TARGET] = y_te
    te_df["prob_raw"] = te_prob
    te_df["prob_cal"] = te_prob_cal
    if id_cols:
        # join back IDs from original df by index
        te_ids = df.loc[X_te.index, id_cols]
        te_out = pd.concat([te_ids.reset_index(drop=True), te_df.reset_index(drop=True)], axis=1)
    else:
        te_out = te_df.reset_index(drop=True)

    pred_path = os.path.join(OUTDIR, "ckd_test_predictions.csv")
    te_out.to_csv(pred_path, index=False)

    print("\nDone. Artifacts saved in:", os.path.abspath(OUTDIR))
    print("- Model:", model_path)
    print("- Preprocess:", prep_path)
    print("- Calibrator:", platt_path)
    print("- Metrics JSON:", report_path)
    print("- Test predictions CSV:", pred_path)
