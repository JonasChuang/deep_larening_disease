# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import xgboost as xgb
import shap,re
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple, List, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
def boruta_feature_selection(CSV_PATH,LAB_NAME):#Boruta
    #LAB_NAME:要預測的檢驗名稱
    """
    使用 Boruta 方法從實驗室檢驗數據中選擇重要特徵，以預測高血鉀（K > 5.5 mEq/L）。
    輸入資料假設為 CSV 格式，包含欄位：stay_id, charttime, label, valuenum。
    """
    # === 1. 讀取原始資料 ===
    df = pd.read_csv(CSV_PATH)   # 你的CSV，例如 stay_id, charttime, label, valuenum

    # === 2. 整理資料 ===
    # 將每個 stay_id 各檢驗項目轉為欄位（Pivot）
    df_pivot = (
        df.pivot_table(index='stay_id', columns='label', values='valuenum', aggfunc='mean')
        .reset_index()
    )

    # === 3. 建立標籤變數 y ===
    # 定義高血鉀（K > 5.5 mEq/L）為 1，否則 0
    df_pivot['target_hyperk'] = np.where(df_pivot[LAB_NAME] > 5.5, 1, 0)

    # 若部分資料沒有血鉀值，先移除
    df_pivot = df_pivot.dropna(subset=['target_hyperk'])

    # === 4. 特徵矩陣與標籤 ===
    X = df_pivot.drop(columns=['stay_id', 'target_hyperk'])
    y = df_pivot['target_hyperk'].astype(int)

    leak_regex = re.compile(r'(potassium|^k$|\bk\+$|\bk\b|serum[_\s-]*k)', flags=re.IGNORECASE)
    leak_cols = [c for c in X.columns if leak_regex.search(str(c))]
    if leak_cols:
        print("🔒 移除疑似洩漏的血鉀相關欄位：", leak_cols)
        X = X.drop(columns=leak_cols)


    # 處理缺值
    X = X.fillna(X.median())

    # === 5. 建立隨機森林 + Boruta 特徵選擇器 ===
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight='balanced',
        max_depth=7,
        random_state=42
    )

    boruta_selector = BorutaPy(
        rf,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )

    boruta_selector.fit(X.values, y.values)

    # === 6. 結果輸出 ===
    feature_ranks = pd.DataFrame({
        'feature': X.columns,
        'rank': boruta_selector.ranking_,
        'selected': boruta_selector.support_
    }).sort_values(by='rank')

    print("\n✅ Boruta 選出的重要特徵：")
    print(feature_ranks[feature_ranks['selected'] == True])

    # === 7. 儲存結果 ===
    feature_ranks.to_csv("boruta_feature_ranking.csv", index=False)
    print("\n📁 已輸出：boruta_feature_ranking.csv")

def rfecv_feature_selection(CSV_PATH,LAB_NAME):#RFECV
    df = pd.read_csv(CSV_PATH)   # 你的CSV檔案名稱

    # === 2. Pivot，每位病人各檢驗平均值 ===
    df_pivot = (
        df.pivot_table(index='stay_id', columns='label', values='valuenum', aggfunc='mean')
        .reset_index()
    )

    # === 3. 定義高血鉀標籤 ===
    # 若資料中有 POTASSIUM 欄位，設定 K > 5.5 為高血鉀
    if 'POTASSIUM' not in df_pivot.columns:
        raise ValueError("❗ 資料中沒有 'POTASSIUM' 欄位，無法建立高血鉀標籤。請確認 CSV 是否包含血鉀 (K) 檢驗。")

    df_pivot['target_hyperk'] = np.where(df_pivot['POTASSIUM'] > 5.5, 1, 0)
    df_pivot = df_pivot.dropna(subset=['target_hyperk'])

    # === 4. 特徵矩陣 X、標籤 y ===
    X = df_pivot.drop(columns=['stay_id', 'target_hyperk'])
    y = df_pivot['target_hyperk'].astype(int)

    # 5) 關鍵：排除任何「疑似血鉀」的特徵欄位，避免資料洩漏
    #    規則：名稱含 potassium / k / k+（大小寫不敏感）
    leak_regex = re.compile(r'(potassium|^k$|\bk\+$|\bk\b|serum[_\s-]*k)', flags=re.IGNORECASE)
    leak_cols = [c for c in X.columns if leak_regex.search(str(c))]
    if leak_cols:
        print("🔒 移除疑似洩漏的血鉀相關欄位：", leak_cols)
        X = X.drop(columns=leak_cols)



    # 填補缺值
    X = X.fillna(X.median())

    # === 5. RFECV 特徵選擇 ===
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rfecv = RFECV(
        estimator=rf,
        step=1,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    rfecv.fit(X, y)

    # === 6. 結果輸出 ===
    feature_ranks = pd.DataFrame({
        'feature': X.columns,
        'ranking': rfecv.ranking_,
        'selected': rfecv.support_
    }).sort_values(by='ranking')

    print("\n✅ RFECV 選出的重要特徵：")
    print(feature_ranks[feature_ranks['selected'] == True])

    mean_test_scores = np.mean(rfecv.cv_results_['mean_test_score'], axis=0) \
    if isinstance(rfecv.cv_results_['mean_test_score'], list) else rfecv.cv_results_['mean_test_score']




    # === 7. 視覺化 CV 結果 ===
    plt.figure(figsize=(8, 5))
    plt.title('RFECV Cross-Validation Score')
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross-validation ROC-AUC')
    #plt.plot(range(1, len(rfecv.cv_results_) + 1), rfecv.cv_results_, marker='o')
    plt.plot(range(1, len(mean_test_scores) + 1), mean_test_scores, marker='o')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === 8. 儲存結果 ===
    feature_ranks.to_csv("rfecv_feature_ranking.csv", index=False)
    print("\n📁 已輸出：rfecv_feature_ranking.csv")
def xgb_feature_importance(CSV_PATH,LAB_NAME):
    df = pd.read_csv(CSV_PATH)   # 你的CSV資料

    # === 2. Pivot：將各項檢驗轉為欄位 ===
    df_pivot = (
        df.pivot_table(index='stay_id', columns='label', values='valuenum', aggfunc='mean')
        .reset_index()
    )

    # === 3. 建立高血鉀標籤 ===
    if 'POTASSIUM' not in df_pivot.columns:
        raise ValueError("❗ 找不到 'POTASSIUM' 欄位，請確認資料中是否有血鉀 (K) 檢驗")
    #血鉀建立標籤（>5.5 → 高血鉀=1）
    df_pivot['target_hyperk'] = np.where(df_pivot['POTASSIUM'] > 5.5, 1, 0)
    df_pivot = df_pivot.dropna(subset=['target_hyperk'])

    # === 4. 特徵與標籤 ===
    X = df_pivot.drop(columns=['stay_id', 'target_hyperk'])
    y = df_pivot['target_hyperk'].astype(int)


    # 5) 關鍵：排除任何「疑似血鉀」的特徵欄位，避免資料洩漏
    #    規則：名稱含 potassium / k / k+（大小寫不敏感）
    leak_regex = re.compile(r'(potassium|^k$|\bk\+$|\bk\b|serum[_\s-]*k)', flags=re.IGNORECASE)
    leak_cols = [c for c in X.columns if leak_regex.search(str(c))]
    if leak_cols:
        print("🔒 移除疑似洩漏的血鉀相關欄位：", leak_cols)
        X = X.drop(columns=leak_cols)

    # 缺值填補
    X = X.fillna(X.median())

    # === 5. 訓練/測試切分 ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # === 6. 建立 XGBoost 模型 ===
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc'
    )

    model.fit(X_train, y_train)

    # === 7. 評估 ===
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\n✅ ROC-AUC = {auc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    # === 8. 特徵重要度 ===
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n📊 Top 10 重要特徵：")
    print(importance.head(10))

    # === 9. 可視化 ===
    plt.figure(figsize=(8,6))
    xgb.plot_importance(model, max_num_features=15, importance_type='gain')
    plt.title("XGBoost Feature Importance (Gain)")
    plt.show()

    # === 10. SHAP 解釋 ===
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15)
    shap.summary_plot(shap_values, X_test, max_display=15)

    # === 11. 儲存結果 ===
    importance.to_csv("xgboost_feature_importance.csv", index=False)
    print("\n📁 已輸出：xgboost_feature_importance.csv")

def leakage_audit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str = "binary",                 # "binary" 或 "reg"
    id_col: Optional[str] = None,         # 若提供：檢查 train/test 洩漏（同 ID 同一個樣本出現在兩邊）
    split_col: Optional[str] = None,      # 若提供：指定 "train"/"test" 或 0/1 來分割
    time_col: Optional[str] = None,       # 若提供：檢查時間洩漏（特徵時間是否晚於 outcome 時間）
    outcome_time_col: Optional[str] = None,
    high_corr_threshold: float = 0.9,     # 多重共線性的特徵-特徵高相關門檻
    high_target_corr_threshold: float = 0.9,  # 特徵-標籤的過高相關門檻（binary）
    quasi_const_threshold: float = 0.99,  # 近常數（最多類別佔比）
    missing_threshold: float = 0.6,       # 缺值過高門檻
    top_k_univariate_auc: int = 20,       # 輸出單變數 AUC 的前幾名
    suspicious_name_patterns: Optional[List[str]] = None,  # 疑似洩漏的欄位命名關鍵字
    random_state: int = 42
) -> Dict[str, Any]:
    """
    檢查 資料洩漏 風險，並產生報告。
    回傳一個 dict，包含多張 DataFrame 報告與簡短結論。
    適用於分類（binary）與回歸（reg）；回歸時不計算 AUC，改用 |corr| 與 MI。
    """

    # ---------- 0) 準備 ----------
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # 型別偵測
    is_binary = (task == "binary" or (task != "reg" and type_of_target(y) in ["binary"]))
    rng = np.random.RandomState(random_state)

    if suspicious_name_patterns is None:
        # 可依任務調整：這裡加上常見可能洩漏名稱（你在高血鉀研究可加入 'K','POTASS'）
        suspicious_name_patterns = [
            r"target", r"label", r"outcome", r"y_", r"_y$", r"ground.?truth", r"leak",
            r"future", r"post.*event", r"after.*(event|outcome)", r"result", r"answer",
            r"potassium", r"\bk\b", r"k\+"
        ]

    # 將類別型轉成可處理的數值版本（供相關係數與 MI 使用）
    X_num = X.copy()
    encoders = {}
    for c in X_num.columns:
        if X_num[c].dtype == 'O' or str(X_num[c].dtype).startswith('category'):
            le = LabelEncoder()
            X_num[c] = le.fit_transform(X_num[c].astype(str))
            encoders[c] = le

    # ---------- 1) 常數/近常數/缺值/重複欄位 ----------
    const_mask = X.nunique(dropna=False) <= 1
    const_features = X.columns[const_mask].tolist()

    # 近常數：單一頻率最高的比例 >= quasi_const_threshold
    quasi_const = []
    for c in X.columns:
        vc = X[c].value_counts(dropna=False, normalize=True)
        if not vc.empty and vc.iloc[0] >= quasi_const_threshold and c not in const_features:
            quasi_const.append(c)

    missing_ratio = X.isna().mean().sort_values(ascending=False)
    high_missing = missing_ratio[missing_ratio >= missing_threshold].index.tolist()

    # 重複欄位（完全一樣）
    duplicated_cols = []
    seen = {}
    for c in X.columns:
        sig = (tuple(pd.util.hash_pandas_object(X[c], index=False)))
        if sig in seen:
            duplicated_cols.append((seen[sig], c))
        else:
            seen[sig] = sig  # 用 hash 簽章辨識

    df_basic = pd.DataFrame({
        "feature": X.columns,
        "is_constant": [c in const_features for c in X.columns],
        "is_quasi_constant": [c in quasi_const for c in X.columns],
        "missing_ratio": [missing_ratio.get(c, 0.0) for c in X.columns],
        "is_high_missing": [c in high_missing for c in X.columns],
        "suspicious_name": [bool(re.search("|".join(suspicious_name_patterns), str(c), flags=re.I)) for c in X.columns]
    }).sort_values(["is_constant","is_quasi_constant","is_high_missing","missing_ratio"], ascending=[False, False, False, False])

    # ---------- 2) 特徵名稱可疑（可能洩漏） ----------
    suspicious_by_name = df_basic[df_basic["suspicious_name"]].sort_values("missing_ratio", ascending=False)

    # ---------- 3) 特徵-標籤 過強關聯（可能洩漏） ----------
    leakage_target_rows = []

    if is_binary:
        # Pearson |corr| & 單變數 ROC-AUC & Mutual Information
        # 注意：AUC 以「數值高→陽性」直覺計算，必要時會 1 - AUC 做最大化
        for c in X_num.columns:
            col = X_num[c].astype(float)
            if col.isna().all():
                continue
            # corr
            try:
                corr = np.corrcoef(np.nan_to_num(col), y.astype(float))[0,1]
                corr = float(corr)
            except Exception:
                corr = np.nan
            # univariate AUC
            try:
                v = pd.Series(col).fillna(col.median())
                auc = roc_auc_score(y, v)
                auc = max(auc, 1 - auc)  # 方向不管，取最大可分性
            except Exception:
                auc = np.nan
            # mutual information
            try:
                mi = mutual_info_classif(v.values.reshape(-1,1), y.values, discrete_features=False, random_state=rng)
                mi = float(mi[0])
            except Exception:
                mi = np.nan

            leakage_target_rows.append({"feature": c, "abs_corr_with_y": abs(corr) if pd.notna(corr) else np.nan,
                                        "univariate_auc": auc, "mutual_info": mi})
        df_target = pd.DataFrame(leakage_target_rows).sort_values(["univariate_auc","abs_corr_with_y","mutual_info"], ascending=False)
        suspicious_by_target = df_target[
            (df_target["univariate_auc"] >= 0.99) |
            (df_target["abs_corr_with_y"] >= high_target_corr_threshold)
        ]
    else:
        # 回歸：以 |corr| 與 MI 評估洩漏
        for c in X_num.columns:
            col = X_num[c].astype(float)
            if col.isna().all():
                continue
            try:
                corr = np.corrcoef(np.nan_to_num(col), y.astype(float))[0,1]
                corr = float(corr)
            except Exception:
                corr = np.nan
            try:
                # 以等頻分箱近似分類 MI
                y_disc = pd.qcut(y, q=min(10, max(2, int(np.sqrt(len(y))))), duplicates="drop")
                le_y = LabelEncoder().fit_transform(y_disc.astype(str))
                mi = mutual_info_classif(col.fillna(col.median()).values.reshape(-1,1), le_y, discrete_features=False, random_state=rng)
                mi = float(mi[0])
            except Exception:
                mi = np.nan
            leakage_target_rows.append({"feature": c, "abs_corr_with_y": abs(corr) if pd.notna(corr) else np.nan,
                                        "mutual_info": mi})
        df_target = pd.DataFrame(leakage_target_rows).sort_values(["abs_corr_with_y","mutual_info"], ascending=False)
        suspicious_by_target = df_target[(df_target["abs_corr_with_y"] >= 0.99)]

    # 只列出單變數 AUC/指標前幾名（便於快速審視）
    top_univariate = df_target.head(top_k_univariate_auc).copy()

    # ---------- 4) 多重共線性：高相關對 & VIF ----------
    corr_pairs = []
    corr_mat = X_num.corr(numeric_only=True).abs()
    # 取上三角
    for i, ci in enumerate(corr_mat.columns):
        for j, cj in enumerate(corr_mat.columns):
            if j <= i: 
                continue
            val = corr_mat.iloc[i, j]
            if pd.notna(val) and val >= high_corr_threshold:
                corr_pairs.append((ci, cj, float(val)))
    df_high_corr_pairs = pd.DataFrame(corr_pairs, columns=["feature_a","feature_b","abs_corr"]).sort_values("abs_corr", ascending=False)

    # VIF（需要 statsmodels）
    df_vif = None
    if variance_inflation_factor is not None and X_num.shape[1] >= 2:
        # 移除常數與缺值太多的欄位再計算 VIF
        keep_cols = [c for c in X_num.columns if (c not in const_features) and (c not in high_missing)]
        X_vif = X_num[keep_cols].fillna(X_num[keep_cols].median())
        try:
            vif_vals = []
            for k, col in enumerate(X_vif.columns):
                vif_vals.append({"feature": col, "VIF": float(variance_inflation_factor(X_vif.values, k))})
            df_vif = pd.DataFrame(vif_vals).sort_values("VIF", ascending=False)
        except Exception:
            df_vif = None

    # ---------- 5) train/test 洩漏（可選） ----------
    df_split_leak = None
    if split_col is not None and split_col in X.columns:
        split_vals = X[split_col]
        if id_col is not None and id_col in X.columns:
            # 同一個 id 同時出現在 train 與 test
            ids_train = set(X.loc[split_vals.astype(str).str.lower().isin(["train","0","false","tr","t","train_set"]) | (split_vals==0), id_col])
            ids_test  = set(X.loc[split_vals.astype(str).str.lower().isin(["test","1","true","te","tst","val","validation"]) | (split_vals==1), id_col])
            overlap = ids_train.intersection(ids_test)
            df_split_leak = pd.DataFrame({"id_in_both_train_and_test": sorted(list(overlap))})
        else:
            # 若沒有 id，只檢查是否有重複的完整樣本（特徵向量）出現在兩邊（成本較高）
            try:
                X_all = X.drop(columns=[split_col])
                key = pd.util.hash_pandas_object(X_all, index=False)
                df_tmp = pd.DataFrame({"hash": key, "split": split_vals})
                dup_hash = df_tmp.groupby("hash")["split"].nunique()
                leaks = dup_hash[dup_hash > 1].index
                df_split_leak = df_tmp[df_tmp["hash"].isin(leaks)].copy()
            except Exception:
                pass

    # ---------- 6) 時間洩漏（可選） ----------
    df_time_leak = None
    if (time_col is not None and time_col in X.columns) and (outcome_time_col is not None):
        try:
            t_feat = pd.to_datetime(X[time_col], errors="coerce")
            t_out = pd.to_datetime(outcome_time_col, errors="coerce") if not isinstance(outcome_time_col, str) else pd.to_datetime(X[outcome_time_col], errors="coerce")
            time_leak_mask = t_feat > t_out
            df_time_leak = pd.DataFrame({
                "idx": np.where(time_leak_mask.fillna(False))[0],
                "feature_time": t_feat[time_leak_mask],
                "outcome_time": t_out[time_leak_mask]
            })
        except Exception:
            pass

    # ---------- 7) 結論摘要 ----------
    notes = []
    if len(suspicious_by_name) > 0:
        notes.append("⚠️ 發現疑似由欄位名稱判定的洩漏特徵（如 'target', 'label', 'potassium/k' 等）。")
    if len(suspicious_by_target) > 0:
        if is_binary:
            notes.append("⚠️ 發現單變數 AUC 或 |corr| 與標籤過高的特徵，請檢查是否含答案或後見資訊（data leakage）。")
        else:
            notes.append("⚠️ 發現與標籤 |corr| 過高的特徵，可能含答案或後見資訊（data leakage）。")
    if df_high_corr_pairs.shape[0] > 0:
        notes.append("ℹ️ 偵測到多組高相關（>|{:.2f}|）特徵對，建議做降維或擇一保留。".format(high_corr_threshold))
    if df_vif is not None and (df_vif["VIF"] > 10).any():
        notes.append("ℹ️ VIF>10 的特徵存在，顯示多重共線性嚴重。")
    if df_split_leak is not None and len(df_split_leak) > 0:
        notes.append("❗ train/test 可能有重疊樣本或相同 ID 出現在兩邊。")
    if df_time_leak is not None and len(df_time_leak) > 0:
        notes.append("❗ 特徵時間晚於 outcome 時間，疑似時間洩漏。")
    if not notes:
        notes.append("✅ 未發現明顯的洩漏或嚴重共線性（請仍以 SHAP/專家審閱複核）。")

    # ---------- 8) 輸出包 ----------
    out = {
        "basic_quality": df_basic.reset_index(drop=True),
        "suspicious_by_name": suspicious_by_name.reset_index(drop=True),
        "target_association": df_target.reset_index(drop=True),
        "top_univariate": top_univariate.reset_index(drop=True),
        "suspicious_by_target": suspicious_by_target.reset_index(drop=True),
        "high_corr_pairs": df_high_corr_pairs.reset_index(drop=True),
        "vif": None if df_vif is None else df_vif.reset_index(drop=True),
        "train_test_overlap": None if df_split_leak is None else df_split_leak.reset_index(drop=True),
        "time_leak_cases": None if df_time_leak is None else df_time_leak.reset_index(drop=True),
        "duplicated_columns": pd.DataFrame(duplicated_cols, columns=["col_a","col_b"]),
        "notes": notes
    }
    return out