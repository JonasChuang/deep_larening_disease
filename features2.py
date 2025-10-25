# -*- coding: utf-8 -*-
"""
Hyperkalemia feature selection (RFECV) + SHAP explanation
- Input : labs.csv  (columns: stay_id, charttime, label, valuenum)
- Output: selected_features.txt, rfecv_feature_ranking.csv, shap_summary_bar.png, shap_beeswarm.png
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import shap


def load_and_pivot(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    wide = (
        df.pivot_table(index='stay_id', columns='label', values='valuenum', aggfunc='mean')
          .reset_index()
    )
    return wide


def find_potassium_col(wide: pd.DataFrame, lab_name) -> str:
    # if lab_name in wide.columns:
    #     return lab_name
    # if lab_name=="POTASSIUM":
    #     pat = re.compile(r'\b(potassium|k\+?|serum[_\s-]*k)\b', flags=re.I)
    # if lab_name=="SODIUM":
    #     pat = re.compile(r'\b(sodium|k\+?|serum[_\s-]*k)\b', flags=re.I)
    # for c in wide.columns:
    #     if pat.search(str(c)):
    #         return c
    _PATTERNS = {
        "POTASSIUM": re.compile(
            r"\b(potassium|k\+?|serum[_\s-]*k|k[_-]?meq|k\([\w]+\))\b", re.I),
        "SODIUM": re.compile(
            r"\b(sodium|na\+?|serum[_\s-]*na|na[_-]?meq|na\([\w]+\))\b", re.I),
    }
    cols_ci = {str(c).lower(): c for c in wide.columns}
    if lab_name.lower() in cols_ci:
        return cols_ci[lab_name.lower()]

    # 2) 用白名單 regex 尋找別名
    pat = _PATTERNS.get(lab_name.upper())
    if pat is None:
        raise ValueError(f"未定義的檢驗名稱：{lab_name}")

    # 先優先常見精簡名（K、NA、K+、NA+）
    preferred_order = ["k", "k+", "na", "na+"]
    for p in preferred_order:
        for c in wide.columns:
            if str(c).lower() == p:
                return c
    # 再做一般 regex 搜尋（第一個吻合者）
    for c in wide.columns:
        if pat.search(str(c)):
            return c

    # 3) 找不到 → 給出友善訊息
    hint = "（例如：POTASSIUM, K, K+, SERUM_K）" if lab_name.upper() == "POTASSIUM" \
        else "（例如：SODIUM, NA, NA+, SERUM_NA）"
            
    
    raise ValueError(f"找不到 {lab_name} 欄位 {hint}")


def drop_potassium_like(X: pd.DataFrame,lab_name) -> pd.DataFrame:
    """
    SODIUM 要抓的是 Na 相關命名：Sodium, NA, NA+, serum_na, na_meq, na(...) 等，而不是 K。

    用 \b（單字邊界）避免把 dna、transferrin saturation 之類字串的「na」誤判。

    re.X（verbose）讓樣式可讀性更好，之後要擴充（例如 plasma_na、na_mmol）很方便。

    用 elif／字典映射，避免後面條件覆蓋前面的 regex。
    """
    LEAK_PATTERNS = {
    "POTASSIUM": re.compile(
        r"""
        \bpotassium\b
        | \bK\+?\b
        | \bserum[_\s-]*k\b
        | \bk[_-]?meq\b
        | \bk\([\w]+\)
        """, re.I | re.X
    ),
    "SODIUM": re.compile(
        r"""
        \bsodium\b
        | \bNA\+?\b
        | \bserum[_\s-]*na\b
        | \bna[_-]?meq\b
        | \bna\([\w]+\)
        """, re.I | re.X
    ),
    }
    pat = LEAK_PATTERNS.get(lab_name.upper())
    if pat is None:
        # 未定義就不動資料
        return X
    leak_cols = [c for c in X.columns if pat.search(str(c))]
    if leak_cols:
        print("🔒 移除疑似洩漏欄位：", leak_cols)
        X = X.drop(columns=leak_cols)
    
    # # 依院內命名可再擴充
    # if lab_name=="POTASSIUM":
    #     leak_regex = re.compile(r'(potassium|^k$|\bk\+$|\bk\b|serum[_\s-]*k|k_meq|k\(\w+\))', flags=re.I)
    # if lab_name=="SODIUM":
    #     leak_regex = re.compile(r'(sodium|^k$|\bk\+$|\bk\b|serum[_\s-]*k|k_meq|k\(\w+\))', flags=re.I)

    # leak_cols = [c for c in X.columns if leak_regex.search(str(c))]
    # if leak_cols:
    #     print("🔒 移除疑似洩漏欄位：", leak_cols)
    #     X = X.drop(columns=leak_cols)
    return X


def do_rfecv_get_features(X: pd.DataFrame, y: pd.Series,
                          min_features_to_select: int = 3,
                          random_state: int = 42):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        class_weight='balanced', random_state=random_state, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rfecv = RFECV(estimator=rf, step=1, cv=cv, scoring='roc_auc',
                  n_jobs=-1, min_features_to_select=min_features_to_select)
    rfecv.fit(X, y)

    # 排名表
    feature_ranks = pd.DataFrame({
        'feature': X.columns,
        'ranking': rfecv.ranking_,
        'selected': rfecv.support_
    }).sort_values(by=['selected', 'ranking'], ascending=[False, True])

    # CV 曲線
    mean_scores = rfecv.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 5))
    plt.title('RFECV Cross-Validation Score')
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross-validation ROC-AUC')
    plt.plot(range(1, len(mean_scores) + 1), mean_scores, marker='o')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rfecv_cv_curve.png', dpi=160)
    plt.close()

    selected_cols = feature_ranks.loc[feature_ranks['selected'], 'feature'].tolist()
    print(f"✅ RFECV 最佳特徵數：{rfecv.n_features_}")
    print("✅ 被選特徵：", selected_cols)

    feature_ranks.to_csv('rfecv_feature_ranking.csv', index=False)
    with open('selected_features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(selected_cols))

    return selected_cols, feature_ranks


def train_xgb_and_plot_shap(X: pd.DataFrame, y: pd.Series, selected_cols: list,
                            random_state: int = 42):
    X = X[selected_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    # CV AUC（報告數字）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=4,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric='auc', random_state=random_state, n_jobs=-1
    )
    cv_auc = cross_val_score(xgb_model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1).mean()
    print(f"📈 XGBoost（選中特徵）5-fold CV AUC = {cv_auc:.4f}")

    # 拆 train/test 產生 SHAP 圖（避免用整批資料）
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    xgb_model.fit(X_tr, y_tr)
    y_prob = xgb_model.predict_proba(X_te)[:, 1]
    print(f"🧪 Holdout ROC-AUC = {roc_auc_score(y_te, y_prob):.4f}")

    # ROC 圖（可選）
    RocCurveDisplay.from_predictions(y_te, y_prob)
    plt.tight_layout()
    plt.savefig('roc_curve_holdout.png', dpi=160)
    plt.close()

    # === SHAP ===
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_te)  # 對 test set 解釋

    # Bar（全局平均影響力）
    shap.summary_plot(shap_values, X_te, plot_type='bar', show=False, max_display=min(20, X_te.shape[1]))
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=160)
    plt.close()

    # Beeswarm（方向 + 影響大小）
    shap.summary_plot(shap_values, X_te, show=False, max_display=min(20, X_te.shape[1]))
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=160)
    plt.close()

    return cv_auc, xgb_model


def main(csv_path: str,LAB_NAME:str, k_threshold: float = 5.5):

    LAB=["POTASSIUM","SODIUM"]
    # 1) 讀取與 pivot
    wide = load_and_pivot(csv_path)

    # 2) 標籤（高血鉀）
    k_col = find_potassium_col(wide,LAB_NAME)

    if LAB_NAME == "POTASSIUM":

        wide['target_hyperk'] = np.where(wide[k_col] > 5.5, 1, 0) #高血鉀 特徵
    if LAB_NAME == "SODIUM":

        wide['target_hyperk'] = np.where(wide[k_col] < 135, 1, 0) #高血鉀 特徵

    # 3) 準備特徵（移除 id / label）
    X = wide.drop(columns=['stay_id', 'target_hyperk'])
    y = wide['target_hyperk'].astype(int)

    # 4) 防資料洩漏：移除血鉀相關欄位
    X = drop_potassium_like(X,LAB_NAME)

    # 5) RFECV 取得最佳特徵
    selected_cols, feature_ranks = do_rfecv_get_features(X.fillna(X.median(numeric_only=True)), y)

    # 6) 用選中特徵訓練 XGBoost 並產生 SHAP 圖
    cv_auc, model = train_xgb_and_plot_shap(X, y, selected_cols)

    print("\n📁 已輸出檔案：")
    print("  - rfecv_cv_curve.png")
    print("  - rfecv_feature_ranking.csv")
    print("  - selected_features.txt")
    print("  - roc_curve_holdout.png")
    print("  - shap_summary_bar.png")
    print("  - shap_beeswarm.png")


if __name__ == "__main__":
    # 換成你的 CSV 路徑
    LAB=["POTASSIUM","SODIUM"]
    main("D:\\project\\deep_larening_disease\\Hyperkalemia\\CSV\\all_lab.csv","SODIUM")
