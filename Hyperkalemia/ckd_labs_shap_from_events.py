#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從事件長表 (stay_id, charttime, label, valuenum) 產出表格特徵 → 訓練 XGBoost → SHAP 解釋
輸出到 --outdir：
  - metrics.json  (AUC/AUPRC/Brier + 陽性率)
  - features.csv  (每個 stay 的特徵表，含 y)
  - shap_importance_bar.png
  - shap_beeswarm.png
必要套件：pandas numpy scikit-learn xgboost shap matplotlib
"""

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# -------------------------
# 1) 讀取與前處理
# -------------------------
def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 標準欄位檢查
    need = {"stay_id","charttime","label","valuenum"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺少欄位：{miss}")
    # 基本清理
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["valuenum"]  = pd.to_numeric(df["valuenum"], errors="coerce")
    df = df.dropna(subset=["stay_id","charttime","label","valuenum"])
    # label 一律大寫（避免同義不同大小寫）
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    # 顯式轉型
    df["stay_id"] = df["stay_id"].astype(int)
    return df

# -------------------------
# 2) 由事件聚合成表格特徵
#    每個 stay_id × 每個檢驗 → {mean, min, max, last}
# -------------------------
def events_to_features(df: pd.DataFrame) -> pd.DataFrame:
    # 先做當次時間排序，以便取得 "last"
    df = df.sort_values(["stay_id","label","charttime"])
    # 針對同 (stay_id, label) 取 mean/min/max
    agg = df.groupby(["stay_id","label"])["valuenum"].agg(
        val_mean="mean", val_min="min", val_max="max", val_last="last"
    ).reset_index()
    # 寬表：欄名 = {LABEL}_{agg}
    wide = agg.pivot(index="stay_id", columns="label", values=["val_mean","val_min","val_max","val_last"])
    # 攤平欄名
    wide.columns = [f"{lvl2}_{lvl1.split('_')[1]}" for lvl1,lvl2 in wide.columns]
    wide = wide.reset_index()
    # 依需要可以只留常見腎臟/電解質（若不想全保留）
    return wide

# -------------------------
# 3) 產生/讀入目標 y
# -------------------------
def build_auto_target(events: pd.DataFrame, mode: str = "hypona130") -> pd.DataFrame:
    """
    回傳 (stay_id, y)
    hypona130 : 若任一 SODIUM <= 130 → y=1
    hyperk55  : 若任一 POTASSIUM >= 5.5 → y=1
    """
    mode = mode.lower()
    grp = events.groupby("stay_id")
    if mode == "hypona130":
        y = grp.apply(lambda g: int(((g["label"]=="SODIUM") & (g["valuenum"]<=130)).any()))
    elif mode == "hyperk55":
        y = grp.apply(lambda g: int(((g["label"]=="POTASSIUM") & (g["valuenum"]>=5.5)).any()))
    else:
        raise ValueError("未知 auto_target，請用 hypona130 或 hyperk55")
    return y.reset_index().rename(columns={0:"y"})

def load_target_csv(target_csv: str) -> pd.DataFrame:
    tdf = pd.read_csv(target_csv)
    need = {"stay_id","y"}
    miss = need - set(tdf.columns)
    if miss:
        raise ValueError(f"target_csv 缺少欄位：{miss}")
    tdf = tdf[["stay_id","y"]].copy()
    tdf["stay_id"] = tdf["stay_id"].astype(int)
    tdf["y"] = (pd.to_numeric(tdf["y"], errors="coerce")>0).astype(int)
    return tdf

# -------------------------
# 4) 訓練 / 評估 / SHAP
# -------------------------
def suggest_scale_pos_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - y.sum()), 1)
    return neg / pos

def train_xgb(Xtr, ytr, Xva, yva, seed=42):
    spw = suggest_scale_pos_weight(ytr)
    model = XGBClassifier(
        n_estimators=600, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_weight=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed, scale_pos_weight=spw, n_jobs=0
    )
    model.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xva,yva)], verbose=False)
    return model

def eval_block(y_true, p_prob):
    return dict(
        AUC   = float(roc_auc_score(y_true, p_prob)),
        AUPRC = float(average_precision_score(y_true, p_prob)),
        Brier = float(brier_score_loss(y_true, p_prob)),
        PosRate = float(np.mean(y_true))
    )

def shap_plots(model, X_for_shap: pd.DataFrame, outdir: str, max_display=20):
    explainer = shap.TreeExplainer(model)
    expl = explainer(X_for_shap)
    # 全域重要度（bar）
    mean_abs = np.mean(np.abs(expl.values), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    vals = mean_abs[order][::-1]
    labs = list(X_for_shap.columns[order][::-1])

    plt.figure(figsize=(6, 0.4*len(labs)+2))
    plt.barh(range(len(labs)), vals)
    plt.yticks(range(len(labs)), labs)
    plt.xlabel("Mean |SHAP| (global importance)")
    plt.title("Top Features (SHAP)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "shap_importance_bar.png"), dpi=150); plt.close()

    # beeswarm
    plt.figure(figsize=(7,5))
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "shap_beeswarm.png"), dpi=150); plt.close()

# -------------------------
# 5) 主流程
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--events_csv", required=True,default=, help="事件長表 CSV：stay_id,charttime,label,valuenum")
    ap.add_argument("--target_csv", default=None, help="(可選) 外部標籤 CSV：stay_id,y；指定時會覆蓋 auto_target")
    ap.add_argument("--auto_target", default="hypona130", choices=["hypona130","hyperk55"], help="自動標籤規則")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size",  type=float, default=0.2)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--outdir",    default="./CSV")
    ap.add_argument("--max_display", type=int, default=20)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 讀事件
    ev = load_events("Hyperkalemia/CSV/fetch_labevents.csv")

    # 產生特徵寬表
    feats = events_to_features(ev)           # (stay_id, FEATURE_agg, ...)
    feats = feats.sort_values("stay_id").reset_index(drop=True)

    # 產生/讀入 y
    if args.target_csv:
        ydf = load_target_csv(args.target_csv)
    else:
        ydf = build_auto_target(ev, mode=args.auto_target)
    # 合併 y
    data = feats.merge(ydf, on="stay_id", how="inner")
    # 刪除全為 NaN 的欄位
    feature_cols = [c for c in data.columns if c not in ("stay_id","y")]
    all_nan = [c for c in feature_cols if data[c].isna().all()]
    if all_nan:
        data = data.drop(columns=all_nan)
        feature_cols = [c for c in feature_cols if c not in all_nan]
    # 缺值保留為 NaN（XGBoost 可處理）；如需插補可自行加上

    # 存特徵表以便審閱/再用
    data.to_csv(os.path.join(args.outdir, "features.csv"), index=False)

    # 切資料（stratify by y）
    X = data[feature_cols]
    y = data["y"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    rel_val = args.val_size / (1 - args.test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=rel_val, random_state=args.seed, stratify=y_tr)

    # 訓練
    model = train_xgb(X_tr, y_tr, X_va, y_va, seed=args.seed)

    # 推論與評估
    p_tr = model.predict_proba(X_tr)[:,1]
    p_va = model.predict_proba(X_va)[:,1]
    p_te = model.predict_proba(X_te)[:,1]

    metrics = {
        "train": eval_block(y_tr, p_tr),
        "val"  : eval_block(y_va, p_va),
        "test" : eval_block(y_te, p_te)
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # SHAP（用測試集做解釋）
    shap_plots(model, X_te, args.outdir, max_display=min(args.max_display, len(feature_cols)))
    print("Saved to:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
