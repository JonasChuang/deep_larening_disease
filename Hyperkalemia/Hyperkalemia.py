"""
0–24h 每小時序列抽取（排除 K）→ GRU 訓練
MIMIC-IV｜0–24h 每小時序列（排除 K）→ GRU → 預測 24–48h 高血鉀
- 資料來源：chartevents（vitals）、labevents（labs，不含Potassium）、outputevents（尿量）
- 對齊基準：ICU intime（可改成入院 admittime，請見註解）
- 標籤：在 24–48h 內是否首次出現 K >= K_CUTOFF（預設 5.5 mmol/L）
- 模型：GRU(64) + Dropout，early stopping 以 val AUPRC 監控
- 指標：AUC, AUPRC, F1@PR最佳閾值, Brier
輸出：./artifacts_hk_seq/ 下存模型與指標
"""
from main import ENG
import os, json, joblib, numpy as np, pandas as pd, shap
from typing import Dict, Tuple, List
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, precision_recall_curve,roc_curve
import tensorflow as tf
from keras import layers
import keras
import matplotlib.pyplot as plt
#from tool import sqlalchmy_raw_sql

OUTDIR = os.getenv("OUTDIR", "./artifacts_hk_seq")

# 時窗
SEQ_HOURS = int(os.getenv("SEQ_HOURS", "24"))     # 特徵序列長度（0–24h）
LBL_FROM  = int(os.getenv("LBL_FROM_H", "24"))    # 標籤視窗起點（相對 t0 小時）
LBL_TO    = int(os.getenv("LBL_TO_H", "48"))      # 標籤視窗終點
K_CUTOFF  = float(os.getenv("K_CUTOFF", "5.5"))   # 高血鉀閾值
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE  = float(os.getenv("VAL_SIZE",  "0.2"))
#TEST_SIZE 和 VAL_SIZE 決定了 資料集的切分比例（訓練 / 驗證 / 測試）。建議數值要看總樣本數量 和 研究目的
# 一般經驗法則
#標準比例:
# Train 60% / Val 20% / Test 20%
# 常見於中等規模數據
# 訓練有足夠數據，驗證集能調參，測試集獨立檢驗泛化
# 大數據情境（>10萬筆樣本）:
# Train 80% / Val 10% / Test 10%
# 訓練資料足夠 → 測試/驗證不需太大
# 節省運算資源，加快實驗
# 小數據情境（幾千筆甚至更少）:
# Train 70% / Val 15% / Test 15%
# 減少測試集比例，避免浪費太多資料
# 有時甚至用 交叉驗證 (cross-validation) 取代固定 val/test

# 醫療研究常見考量
# Test（外部驗證）要保護好：通常 15–20% 的資料會單獨保留，不能用來調參。
# Val（模型調參用）：10–20% 是常見範圍。
# Train：剩下的全部




QUICK_TEST = os.getenv("QUICK_TEST", "false").lower() == "true"

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ======== 特徵定義 ========
VITAL_LABELS = [
    'HEART RATE',#220045
    'RESPIRATORY RATE', #220210
    'SPO2',
    'TEMPERATURE CELSIUS', #223762
    'NON INVASIVE BLOOD PRESSURE SYSTOLIC', # 220179
    'NON INVASIVE BLOOD PRESSURE DIASTOLIC', #220180
    'MEAN ARTERIAL PRESSURE (NIBP)'
]
LAB_LABELS = [  # 排除 POTASSIUM，避免洩漏
    'SODIUM', 'BICARBONATE', 'CHLORIDE', 'CREATININE', 'UREA NITROGEN', 'GLUCOSE'
]
FEATURE_MAP = {
    'HEART RATE':'hr','RESPIRATORY RATE':'rr','SPO2':'spo2','TEMPERATURE CELSIUS':'temp',
    'NON INVASIVE BLOOD PRESSURE SYSTOLIC':'sbp','NON INVASIVE BLOOD PRESSURE DIASTOLIC':'dbp',
    'MEAN ARTERIAL PRESSURE (NIBP)':'map',
    'SODIUM':'sodium','BICARBONATE':'bicarb','CHLORIDE':'chloride',
    'CREATININE':'creatinine','UREA NITROGEN':'bun','GLUCOSE':'glucose',
    'URINE_OUTPUT':'urine'
}

def run_df(sql: str, params: Dict=None) -> pd.DataFrame:
    try:
        print(sql)
        return pd.read_sql(text(sql), ENG, params=params)
    except Exception as ERROR:
        print(str(ERROR))
        print(sql)

def make_cohort():
    # 以 ICU 入室為對齊（若改成入院，請把 icustays 改 admissions，t0=admittime）
    # intime 入ICU 時間

    SQL=""" --大母體SQL
        select
            `i`.`subject_id` as `subject_id`,
            `i`.`hadm_id` as `hadm_id`,
            `i`.`stay_id` as `stay_id`,
            `i`.`intime` as `t0`,
            (`i`.`intime` + interval 24 hour) as `t_end`,
            (`i`.`intime` + interval 24 hour) as `lbl_from`,
            (`i`.`intime` + interval 48 hour) as `lbl_to`
        from
            (`icustays` `i`
        join `patients` on
            ((`patients`.`subject_id` = `i`.`subject_id`)))
        where
            (`i`.`subject_id` in (
            select
                `p`.`subject_id`
            from
                `patients` `p`
            where
                (`p`.`anchor_year_group` in ('2020 - 2022', '2014 - 2016')
                and ((p.anchor_age + YEAR(i.intime) - p.anchor_year) >= 18)
                ))
                
                and exists(
                select
                    1
                from
                    `diagnoses_icd` `icd`
                where
                    ((`icd`.`hadm_id` = `i`.`hadm_id`)
                        and (`icd`.`subject_id` = `i`.`subject_id`)
                            and (((`icd`.`icd_version` = 10)
                                and (replace(`icd`.`icd_code`, '.', '') like 'N18%'))
                                or ((`icd`.`icd_version` = 9)
                                    and (replace(`icd`.`icd_code`, '.', '') like '585%'))))));

    """
    
    #大母體
    stays = pd.read_sql_query(text("SELECT * FROM icu_adm_view"), ENG)
    stays.to_csv("Hyperkalemia/CSV/icu_adm_view.csv", index=False, encoding="utf-8-sig")

    #fetch_chartevents
    SQL = f"""
    SELECT ce.stay_id, ce.charttime, UPPER(di.label) AS label, ce.valuenum
    FROM chartevents ce
    JOIN d_items di ON di.itemid = ce.itemid
    JOIN icu_adm_view s ON s.stay_id = ce.stay_id
    WHERE ce.charttime >= s.t0 AND ce.charttime < s.t_end
    AND ce.itemid IN ('220045','220179','220180','220210','223762')
    AND ce.valuenum IS NOT NULL
    AND ce.valuenum REGEXP '^[0-9]+$' AND CAST(ce.valuenum AS UNSIGNED) > 1
    
    """


    
    fetch_chartevents=pd.read_sql(text(SQL), ENG)
    fetch_chartevents.to_csv("Hyperkalemia/CSV/fetch_chartevents.csv", index=False, encoding="utf-8-sig")
    print("chartevents OK")
    #檢驗
    SQL = f"""
    SELECT s.stay_id, le.charttime, UPPER(dl.label) AS label, le.valuenum
    FROM labevents le
    JOIN d_labitems dl ON dl.itemid = le.itemid
    JOIN icu_adm_view s ON s.hadm_id = le.hadm_id
    WHERE le.valuenum IS NOT NULL
        AND le.charttime >= s.t0 AND le.charttime < s.t_end
        AND UPPER(dl.label) IN ('SODIUM', 'BICARBONATE', 'CHLORIDE', 'CREATININE', 'UREA NITROGEN', 'GLUCOSE')
        AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
        -- AND le.valuenum REGEXP '^[0-9]+$' AND CAST(le.valuenum AS UNSIGNED) > 1

    """
    fetch_labevents = pd.read_sql(text(SQL), ENG)
    fetch_labevents.to_csv("Hyperkalemia/CSV/fetch_labevents.csv", index=False, encoding="utf-8-sig")
    print("labevents OK")

    SQL = f"""
    SELECT oe.stay_id, oe.charttime, UPPER(di.label) AS URINE_OUTPUT, oe.value AS valuenum
    FROM outputevents oe
    JOIN d_items di ON di.itemid = oe.itemid
    JOIN icu_adm_view s ON s.stay_id = oe.stay_id
    WHERE oe.valueuom IS NOT NULL
        AND oe.charttime >= s.t0 AND oe.charttime < s.t_end
        AND UPPER(di.label) LIKE '%URINE%'
        AND oe.value REGEXP '^[0-9]+(\\.[0-9]+)?$'
        -- AND oe.value REGEXP '^[0-9]+$' AND CAST(oe.value AS UNSIGNED) > 1
 
    """
    URINE = pd.read_sql(text(SQL), ENG)
    URINE.to_csv("Hyperkalemia/CSV/URINE.csv", index=False, encoding="utf-8-sig")

    print("URINE OK")
    
    SQL = """
      -- 高血鉀 母體
       SELECT s.stay_id,
             MAX(CASE WHEN le.valuenum >= '5.5' THEN 1 ELSE 0 END) AS y_hk
      FROM icu_adm_view s
      JOIN labevents le FORCE INDEX (idx_hadm_item_time_val) ON le.hadm_id = s.hadm_id
      JOIN d_labitems dl ON dl.itemid = le.itemid
      WHERE  le.charttime >= s.lbl_from AND le.charttime < s.lbl_to
       AND UPPER(dl.label) LIKE '%POTASSIUM%'
       AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
     -- and labevents in(50833,50971,52610)
     
       
       GROUP BY s.stay_id
      
    """
    POTASSIUM = pd.read_sql(text(SQL), ENG)
    POTASSIUM.to_csv("Hyperkalemia/CSV/POTASSIUM.csv", index=False, encoding="utf-8-sig")
    print("POTASSIUM OK")

    SQL = """
      -- 低鈉母體
       SELECT s.stay_id,
             MAX(CASE WHEN le.valuenum <= 130 THEN 1 ELSE 0 END) AS y_hypona
      FROM icu_adm_view s
      JOIN labevents le FORCE INDEX (idx_hadm_item_time_val) ON le.hadm_id = s.hadm_id
      JOIN d_labitems dl ON dl.itemid = le.itemid
      WHERE  le.charttime >= s.lbl_from AND le.charttime < s.lbl_to
       AND UPPER(dl.label) LIKE '%SODIUM%'
       AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
  
       GROUP BY s.stay_id
      
    """
    low_sodium = pd.read_sql(text(SQL), ENG)
    low_sodium.to_csv("Hyperkalemia/CSV/low_sodium.csv", index=False, encoding="utf-8-sig")

    
    return 0

        

def to_hourly(df_long: pd.DataFrame, t0_df: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["stay_id","time_index","label","value"])
    df = df_long.merge(t0_df[["stay_id","t0"]], on="stay_id", how="left")
    df["offset_h"] = (pd.to_datetime(df["charttime"]) - pd.to_datetime(df["t0"])).dt.total_seconds()/3600.0
    df = df[(df["offset_h"] >= 0) & (df["offset_h"] < SEQ_HOURS)].copy()
    df["time_index"] = df["offset_h"].round().astype(int)
    df.sort_values(["stay_id","label","charttime"], inplace=True)
    df = df.groupby(["stay_id","label","time_index"], as_index=False).tail(1)
    return df[["stay_id","label","time_index","valuenum"]].rename(columns={"valuenum":"value"})

def build_tensor(char_h: pd.DataFrame, lab_h: pd.DataFrame, ur_h: pd.DataFrame, stays: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    try:
        """
        （chartevents、labevents、outputevents → 每筆事件一列），轉換成 三維張量 (N × T × F)，供深度學習模型（GRU/CNN/Transformer）輸入
        
        """

        #frames:把來自 生命徵象 char_h、化驗 lab_h、尿量 ur_h 的資料合併在一起
        frames = [x for x in [char_h, lab_h, ur_h] if x is not None and not x.empty]
        if not frames:
            raise RuntimeError("No hourly features extracted.")
        long = pd.concat(frames, ignore_index=True)

        long["feature"] = long["label"].map(FEATURE_MAP)
        long = long.dropna(subset=["feature"])

        """
        long.pivot_table:
        每個病人 (stay_id)、每個小時 (time_index)，對應一組特徵值。

        用 pivot_table 把「長表」變成「寬表」，每一欄是一個 feature，例如 hr, rr, sbp, cr, bun...。

        aggfunc="last"：如果同一小時有多筆，就取最後一筆
        """
        wide = long.pivot_table(index=["stay_id","time_index"], columns="feature", values="value", aggfunc="last").reset_index()

        # 填滿 0..SEQ_HOURS-1 ,保每個病人都有從 0 到 SEQ_HOURS-1 的時間索引
        grid = stays[["stay_id"]].drop_duplicates().assign(key=1)\
            .merge(pd.DataFrame({"time_index":np.arange(SEQ_HOURS),"key":1}), on="key").drop("key",axis=1)
        wide = grid.merge(wide, on=["stay_id","time_index"], how="left").sort_values(["stay_id","time_index"])

        #缺值處理
        feat_cols = [c for c in wide.columns if c not in ["stay_id","time_index"]]
        # 每個 stay forward fill，再以整體中位數填補
        def _ff(g): g[feat_cols]=g[feat_cols].ffill(); return g
        wide = wide.groupby("stay_id", as_index=False).apply(_ff).reset_index(drop=True)
        med = wide[feat_cols].median()
        wide[feat_cols] = wide[feat_cols].fillna(med)

        # 標準化（fit 全體或只 fit 訓練都可；簡化先 fit 全體）
        #把所有特徵縮放成 平均值 0、標準差 1
        scaler = StandardScaler()
        wide[feat_cols] = scaler.fit_transform(wide[feat_cols])
        joblib.dump(scaler, os.path.join(OUTDIR, "scaler.joblib"))

        order = wide[["stay_id"]].drop_duplicates().values.ravel()
        #N:病人數、T:時間長度、F:特徵數量（hr、rr、sbp、bun、cr...）
        N, T, F = len(order), SEQ_HOURS, len(feat_cols)
        X = np.zeros((N, T, F), dtype=np.float32)
        for i, sid in enumerate(order):
            blk = wide[wide["stay_id"]==sid].sort_values("time_index")[feat_cols].values
            X[i,:,:] = blk[:T]#X[i,:,:] → 病人 i 的 (T × F) 時序矩陣
        return X, feat_cols, pd.DataFrame({"stay_id":order})
    except Exception as Error:
        print(str(Error))

def build_tensor2(char_h: pd.DataFrame, lab_h: pd.DataFrame, ur_h: pd.DataFrame, stays: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    將 long format（char/lab/output 每筆事件一列）轉為三維張量 (N × T × F)
    並處理：label→feature 映射、時間網格補齊、缺值處理、標準化。
    """

    # 1) 合併來源
    frames = [x for x in [char_h, lab_h, ur_h] if x is not None and not x.empty]
    if not frames:
        raise RuntimeError("No hourly features extracted.")
    long = pd.concat(frames, ignore_index=True)

    # 2) 安全數值轉換（避免 '103'、'nan'、'' 汙染）
    #    如果來源欄叫 'value'，確保它是數字；無法轉的直接變 NaN
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    # 3) label → 統一的 feature 名稱；丟掉 map 不到的
    long["feature"] = long["label"].map(FEATURE_MAP)
    long = long.dropna(subset=["feature"])

    # 4) 長轉寬（同一小時多筆取最後一筆）
    wide = (long
            .pivot_table(index=["stay_id","time_index"],
                         columns="feature", values="value", aggfunc="last")
            .reset_index())

    # 5) 補滿 0..SEQ_HOURS-1 的時間格
    grid = (stays[["stay_id"]].drop_duplicates().assign(key=1)
            .merge(pd.DataFrame({"time_index": np.arange(SEQ_HOURS), "key": 1}), on="key")
            .drop("key", axis=1))
    wide = (grid.merge(wide, on=["stay_id","time_index"], how="left")
                 .sort_values(["stay_id","time_index"]))

    # 6) 缺值處理：先 per-stay 前向填補，再用整體中位數補
    feat_cols = [c for c in wide.columns if c not in ["stay_id","time_index"]]

    # 再保險一次：把可能殘留的字串數字轉成 float
    wide[feat_cols] = wide[feat_cols].apply(pd.to_numeric, errors="coerce")

    def _ff(g):
        g[feat_cols] = g[feat_cols].ffill()
        return g
    wide = wide.groupby("stay_id", as_index=False).apply(_ff).reset_index(drop=True)

    # 刪除「整欄都是 NaN」的特徵（避免 median 出問題）
    all_nan_cols = [c for c in feat_cols if wide[c].isna().all()]
    if all_nan_cols:
        wide = wide.drop(columns=all_nan_cols)
        feat_cols = [c for c in feat_cols if c not in all_nan_cols]

    # 這時再算中位數並補
    med = wide[feat_cols].median(numeric_only=True)
    wide[feat_cols] = wide[feat_cols].fillna(med)

    # 7) 標準化（先轉 float32，再 scale）
    from sklearn.preprocessing import StandardScaler
    wide[feat_cols] = wide[feat_cols].astype("float32")
    scaler = StandardScaler()
    wide[feat_cols] = scaler.fit_transform(wide[feat_cols])
    joblib.dump(scaler, os.path.join(OUTDIR, "scaler.joblib"))

    # 8) 組 N×T×F 張量
    order = wide[["stay_id"]].drop_duplicates().values.ravel()
    N, T, F = len(order), SEQ_HOURS, len(feat_cols)
    X = np.zeros((N, T, F), dtype=np.float32)
    for i, sid in enumerate(order):
        blk = (wide[wide["stay_id"] == sid]
               .sort_values("time_index")[feat_cols]
               .values)
        X[i, :, :] = blk[:T]

    return X, feat_cols, pd.DataFrame({"stay_id": order})

def focal_loss(gamma=2., alpha=0.25):
    """
    意思是把損失函數 換成 Focal Loss，並設定兩個超參數：gamma=2.0、alpha=0.25
    「讓模型忽略容易的樣本，把注意力放在少數、困難的正樣本」，在不平衡資料（像 CKD 死亡、敗血症、低鈉）特別有效

    1. Focal Loss 的背景

    在 類別嚴重不平衡（例如敗血症、低鈉、腎衰竭發生率只有 5%）的情況下，傳統 Binary Cross-Entropy (BCE) 會被大量負樣本主導，導致模型傾向預測「全是陰性」。

    Focal Loss 是由 Lin et al., 2017 (RetinaNet) 提出的，用來解決分類不平衡。

    FL(pt)=−α(1−pt)γ log(pt)
    
    實際效果:

    在醫療數據上，使用 focal loss 常會，AUC 提升一點點或差不多；AUPRC 提升顯著（因為更關注正樣本）；

    收斂速度比 BCE 慢，但泛化到 test set 更穩。

    與 Binary Cross-Entropy 比較

    BCE（不平衡時）：

    假設資料 95% 陰性，模型就算永遠預測 0，也能得到很低的 loss。

    Focal Loss：

    會降低「容易分對的負樣本」在 loss 中的比重。

    會加強「難分的正樣本」的影響。

    適合醫療預測（罕見事件）。
    
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(w * tf.pow(1 - pt, gamma) * tf.math.log(pt))
    return loss


def build_gru(input_shape) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="seq")
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(64, return_sequences=False)(x)#找特徵
  
    x = layers.Dropout(0.3)(x)#斷神經元
    out = layers.Dense(1, activation="sigmoid")(x)#訓練
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  #loss="binary_crossentropy",
                  loss=focal_loss(gamma=2.0, alpha=0.25),
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="auprc")])
    return model

def build_lstm(input_shape) -> keras.Model:
    inp = keras.Input(shape=input_shape, name="seq")
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(64, return_sequences=False)(x)#找特徵
    x = layers.Dropout(0.3)(x)#斷神經元
    out = layers.Dense(1, activation="sigmoid")(x)#訓練
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="auprc")])
    return model

def best_pr_thr(y, p):
    """找到 Precision–Recall 曲線下，能讓 F1-score 最大化的 threshold
    y：真實標籤 (0/1)

    p：模型輸出的機率 (0~1)

    """
    #precision_recall_curve:回傳 不同 threshold 下的 precision, recall, threshold 值。
    prec, rec, thr = precision_recall_curve(y, p)

    f1 = (2*prec*rec)/(prec+rec+1e-12)#用 precision 和 recall 計算 F1-score
    i = int(np.nanargmax(f1))#找到 F1-score 最大的索引位置

    #取出對應的最佳 threshold。
    #注意這裡用 thr[i-1]，因為 precision_recall_curve 的 threshold 長度比 prec/rec 少 1。
    #如果算不到就回傳 0.5 當預設值。
    #✅：best_pr_thr 挑一個「最佳 F1 的臨界值」
    return (thr[max(i-1,0)] if i < len(thr) else 0.5)

def eval_block(y, p, tag):
    #用最佳 threshold 評估模型在某個資料集上的表現，然後把機率轉成 0/1 預測
    thr = best_pr_thr(y, p)
    yhat = (p >= thr).astype(int)
    res = {
        "threshold": float(thr),
        "AUC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "F1": float(f1_score(y, yhat)),
        "Brier": float(brier_score_loss(y, p))#Brier score（校準程度)
    }
    print(f"\n[{tag}] thr={res['threshold']:.3f}  AUC={res['AUC']:.4f}  AUPRC={res['AUPRC']:.4f}  F1={res['F1']:.4f}  Brier={res['Brier']:.4f}")

    #列印ROC 曲線
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"{tag} ROC (AUC={res['AUC']:.3f})")
    plt.plot([0,1],[0,1],'k--')  # 參考線
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {tag}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return res

def plot_eval_radar(res_list, tags):#雷達圖
    metrics = ["AUC","AUPRC","F1","Brier"]
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close circle

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for res, tag in zip(res_list, tags):
        values = [res[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, marker="o", label=tag)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0,1)
    plt.legend(loc="upper right")
    plt.title("Model Evaluation Metrics")
    plt.show()

def plot_eval_metrics(res, tag="Validation"):#長條圖
    metrics = ["AUC","AUPRC","F1","Brier"]
    values = [res[m] for m in metrics]

    plt.figure(figsize=(6,4))
    bars = plt.bar(metrics, values, color=["skyblue","lightgreen","orange","lightcoral"])
    plt.title(f"{tag} Metrics (thr={res['threshold']:.3f})")
    plt.ylim(0,1)
    plt.ylabel("Score")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", va="bottom")

    plt.show()


def shap_show(model,Xtr,Xte,feat_cols):

    bg_n = min(128, len(Xtr))
    X_bg = Xtr[np.random.default_rng(42).choice(len(Xtr), size=bg_n, replace=False)]

    # 取一批要解釋的樣本（這裡用測試集前 64 筆示範）
    k = min(64, len(Xte))
    X_explain = Xte[:k]

    # 建立 explainer（新式 API 優先；不行就退回 DeepExplainer）
    try:
        explainer = shap.Explainer(model, X_bg)   # 需 shap 新版
        expl = explainer(X_explain)               # shap.Explanation
        sv = expl.values                          # (k, T, F)
        base = expl.base_values                   # (k,) 或標量
    except Exception as Error:
        print(str(Error))
        explainer = shap.DeepExplainer(model, X_bg)   # 舊版 API
        sv_list = explainer.shap_values(X_explain)    # list/ndarray
        sv = sv_list[0] if isinstance(sv_list, list) else sv_list  # (k, T, F)
        base = getattr(explainer, "expected_value", 0.0)
    print("SHAP values shape:", sv.shape) 

    # 各特徵的平均 |SHAP|（跨樣本與時間）
    imp_feat = np.mean(np.abs(sv), axis=(0,1))   # -> (F,)
    rank = np.argsort(imp_feat)[::-1]
    top = min(15, len(feat_cols))

    plt.figure(figsize=(6, 0.35*top + 2))
    plt.barh(range(top), imp_feat[rank[:top]][::-1])
    plt.yticks(range(top), [feat_cols[i] for i in rank[:top]][::-1])
    plt.xlabel("Mean |SHAP| (global importance)")
    plt.title("Top features (aggregated over time)")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "shap_global_features.png"), dpi=150)
    plt.close()

    T = sv.shape[1]
    imp_time = np.mean(np.abs(sv), axis=(0,2))   # -> (T,)

    plt.figure(figsize=(7,3))
    plt.plot(range(T), imp_time, marker="o")
    plt.xlabel("Hour since ICU intime (t)")
    plt.ylabel("Mean |SHAP|")
    plt.title("When does the model care most?")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "shap_time_importance.png"), dpi=150)
    plt.close()

    case_idx = 0
    M = sv[case_idx]   # (T, F) 這個病人的 SHAP 矩陣
    vmin, vmax = -np.max(np.abs(M)), np.max(np.abs(M))

    plt.figure(figsize=(9, 0.35*len(feat_cols) + 2))
    plt.imshow(M.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(label="SHAP value (+ drives risk up)")
    plt.yticks(range(len(feat_cols)), feat_cols)
    plt.xlabel("Hour"); plt.ylabel("Feature")
    plt.title(f"Case #{case_idx}: SHAP heatmap (time × feature)")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "shap_heatmap_case0.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("==> 建 cohort/時窗、抓特徵事件")
    #data=make_cohort()
    
    df_char=pd.read_csv("Hyperkalemia/CSV/fetch_chartevents.csv")
    df_lab = pd.read_csv("Hyperkalemia/CSV/fetch_labevents.csv")
    df_ur = pd.read_csv("Hyperkalemia/CSV/URINE.csv")
    stays = pd.read_csv("Hyperkalemia/CSV/icu_adm_view.csv")

    print("==> 轉每小時網格（0–24h）")
    char_h = to_hourly(df_char, stays)
    lab_h  = to_hourly(df_lab,  stays)
    ur_h   = to_hourly(df_ur,   stays)

    X, feat_cols, sid_df = build_tensor2(char_h, lab_h, ur_h, stays)

    print("==> 取標籤（24–48h 高血鉀）")
 
    #ydf = pd.read_csv("Hyperkalemia/CSV/POTASSIUM.csv")#高血鉀
    ydf = pd.read_csv("Hyperkalemia/CSV/low_sodium.csv")#低鈉
    y = sid_df.merge(ydf, on="stay_id", how="left")["y_hk"].fillna(0).astype(int).values

    # 切 Train/Val/Test（by stay）
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    n_test = int(len(idx)*TEST_SIZE)
    n_val  = int((len(idx)-n_test)*VAL_SIZE)
    te_idx = idx[:n_test]; va_idx = idx[n_test:n_test+n_val]; tr_idx = idx[n_test+n_val:]
    #驗證集 Xva、測試集 Xte
    Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te_idx]
    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    print("==> 訓練 GRU")
    model = build_gru((X.shape[1], X.shape[2]))
    # 類別權重（不平衡）
    pos = max(ytr.sum(), 1); neg = max(len(ytr)-pos, 1)
    cw = {0: 0.5*len(ytr)/neg, 1: 0.5*len(ytr)/pos}
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=12, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auprc", mode="max", patience=5, factor=0.5, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva),
               epochs=80,#epoch（訓練週期）是 深度學習訓練裡的一個基本單位
               batch_size=128, #batch_size=128 時，大概也就 1–2k 個樣本，每次參數更新時，送進模型的樣本數量。它是一個「訓練速度 vs 模型效果」之間的平衡點
               class_weight=cw,
               callbacks=cbs,
               verbose=2)

    
    va_prob = model.predict(Xva, batch_size=256).ravel()
    te_prob = model.predict(Xte, batch_size=256).ravel()# 模型預測
    #model.predict: 驗證集每個樣本的 預測機率(0~1)

    print("==> 評估")
    val_metrics = eval_block(yva, va_prob, "VAL")
    test_metrics= eval_block(yte, te_prob, "TEST")# 真實標籤
    plot_eval_metrics(val_metrics, tag="VAL")#驗證集
    plot_eval_metrics(test_metrics, tag="TEST")#測試集

    plot_eval_radar([val_metrics, test_metrics], ["VAL","TEST"])

    #shap_show(model,Xtr,Xte,feat_cols)

    print("==> 輸出")
    mdir = OUTDIR
    model.save(os.path.join(mdir, "gru_hk_model.keras"))
    with open(os.path.join(mdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val":val_metrics, "test":test_metrics, "features":feat_cols,
                   "seq_hours": SEQ_HOURS, "label_window":[LBL_FROM, LBL_TO],
                   "k_cutoff": K_CUTOFF}, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(mdir, "test_prob.npy"), te_prob)#模型預測值
    np.save(os.path.join(mdir, "test_y.npy"),   yte)#這是測試集的 真實標籤 (ground truth)，通常是 0 或 1
    # te_prob:
    # 測試集每個樣本的 預測機率 (通常是 sigmoid 輸出的值，介於 0~1)。
    # 存成 test_prob.npy，方便後續做 ROC、PR、Calibration、SHAP 分析。

    # yte:
    # 測試集的 真實標籤 (ground truth)，通常是 0/1。
    # 存成 test_y.npy，對應 te_prob，用來計算 AUC、AUPRC、F1 等評估指標。

    print("完成，Artifacts 於：", os.path.abspath(mdir))

    




# 訓練深度學習模型（例如用 MIMIC ICU 病人做 CKD / 低鈉 / 敗血症預測）分成三個集合：
# 1️⃣ 訓練集 (Training set)
# 用來 訓練模型參數（例如 GRU 裡的權重）。
# 模型會一邊看訓練集、一邊更新權重，讓 loss 下降。

# 2️⃣ 驗證集 (Validation set)
# 不參與訓練，只在 每個 epoch 結束後用來檢查模型表現。
# 功能：
# 幫助你調整 超參數（learning rate、batch size、dropout…）。
# 早停 (early stopping)：如果驗證集 loss 開始上升，代表模型 overfitting，要停下來。
# 選擇最佳模型 checkpoint。
# ✅ 重點：驗證集模擬「模型在沒看過的新資料」的表現，但它仍然參與調參。

# 3️⃣ 測試集 (Test set)
# 完全獨立，最後才用。
# 不會用來訓練，也不會用來調參。
# 目的是：模擬「真實世界未來病人」的狀況，評估模型的 最終泛化能力。
# 只在你模型確定好（hyperparameters fixed）後，才會對測試集做評估並報告 AUC / AUPRC / F1。

# 🔎 舉例（MIMIC-IV CKD ICU 低鈉預測）

# 有 10,000 個 ICU stay。
# 可能切法
# 訓練集 (Train)：70% → 用來學習權重。
# 驗證集 (Validation)：15% → 用來調參、early stopping。
# 測試集 (Test)：15% → 只用一次，最後報告結果。