"""
Ckd 檢驗報告 × 死亡率預測（deep Learning, My Sql→keras）
MIMIC-IV 腎臟疾病（CKD + ICU）重症預測：MySQL → Python → GRU 深度學習

任務（可擇一或同時）：
  - y_mort: 院內死亡（admissions.deathtime != NULL）
  - y_rrt : 住院期間是否接受腎臟替代治療（RRT；依 procedures_icd 代碼）

特徵（時序，預設 ICU 入室後 0–48 小時，1 小時一點）：
  - 生命徵象（chartevents）：HR, SBP, DBP, MAP, RR, SpO2, Temp
  - 檢驗（labevents）：Creatinine, BUN, Sodium, Potassium, Bicarbonate
  - 尿量（outputevents）：每小時尿量

模型：GRU(64) → Dropout → Dense(sigmoid)
指標：AUC, AUPRC（並回報 F1@PR 最佳閾值）, Brier

依賴：
  pip install pandas SQLAlchemy PyMySQL scikit-learn tensorflow joblib numpy

使用方式：
  1) 設定下方 MySQL 連線與資料庫/表名（或用環境變數）。
  2) 直接執行本檔：python kidney_severity_mimic_dl.py
  3) 產物輸出於 ./artifacts_kidney_seq
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, precision_recall_curve
import tensorflow as tf
import keras
from keras import layers

# ===================== 使用者設定 =====================
USER = os.getenv("MIMIC_MYSQL_USER", "user")
PWD  = os.getenv("MIMIC_MYSQL_PWD",  "password")
HOST = os.getenv("MIMIC_MYSQL_HOST", "127.0.0.1")
PORT = int(os.getenv("MIMIC_MYSQL_PORT", "3306"))
DB   = os.getenv("MIMIC_MYSQL_DB",   "mimiciv")
OUTDIR = os.getenv("OUTDIR", "./artifacts_kidney_seq")
T_HOURS = int(os.getenv("T_HOURS", "48"))  # 序列長度（小時）
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE  = float(os.getenv("VAL_SIZE",  "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
QUICK_TEST = os.getenv("QUICK_TEST", "false").lower() == "true"  # 若為 true 會限制樣本加速

# 欲納入的 label 名稱（都用 UPPER 後比對）
VITAL_LABELS = [
    'HEART RATE',
    'RESPIRATORY RATE',
    'SPO2',
    'TEMPERATURE CELSIUS',
    'NON INVASIVE BLOOD PRESSURE SYSTOLIC',
    'NON INVASIVE BLOOD PRESSURE DIASTOLIC',
    'MEAN ARTERIAL PRESSURE (NIBP)'
]
LAB_LABELS = [
    'CREATININE', 'UREA NITROGEN', 'SODIUM', 'POTASSIUM', 'BICARBONATE'
]

# ICD 手術碼（RRT）：ICD-9 39.95（血液透析）；ICD-10-PCS 5A1D*（機械性腎臟替代）
RRT_ICD9 = ['3995', '5498']  # 覆蓋 hemodialysis (39.95) 與 peritoneal dialysis(54.98)
RRT_ICD10_PREFIX = ['5A1D']

np.random.seed(RANDOM_STATE)
TF_SEED = RANDOM_STATE
try:
    tf.random.set_seed(TF_SEED)
except Exception:
    pass

# ===================== DB 連線 =====================
ENG = create_engine(f"mysql+pymysql://{USER}:{PWD}@{HOST}:{PORT}/{DB}")

# ===================== SQL 協助函式 =====================

def run_sql(sql: str, params: Dict = None):
    with ENG.begin() as conn:
        return conn.execute(text(sql), params or {})

def read_sql_df(sql: str, params: Dict = None) -> pd.DataFrame:
    return pd.read_sql(text(sql), ENG, params=params)

# ===================== 步驟 1：建立 CKD + ICU cohort =====================

def build_cohort_temp_tables():
    # CKD 住院
    run_sql("DROP TEMPORARY TABLE IF EXISTS tmp_ckd_hadm;")
    run_sql(
        """
        CREATE TEMPORARY TABLE tmp_ckd_hadm AS
        SELECT DISTINCT d.subject_id, d.hadm_id
        FROM diagnoses_icd d
        WHERE (d.icd_version=10 AND REPLACE(d.icd_code,'.','') LIKE 'N18%')
           OR (d.icd_version=9  AND REPLACE(d.icd_code,'.','') LIKE '585%');
        """
    )
    # 對應 ICU stay（限定有 ICU，且取得 intime 作為序列起點）
    run_sql("DROP TEMPORARY TABLE IF EXISTS tmp_ckd_stays;")
    run_sql(
        f"""
        CREATE TEMPORARY TABLE tmp_ckd_stays AS
        SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime
        FROM icustays i
        JOIN tmp_ckd_hadm c USING(subject_id, hadm_id);
        """
    )

# ===================== 步驟 2：抓取事件資料（0–T_HOURS） =====================

FEATURE_MAP = {
    # 映射成統一欄名（便於最後 pivot）
    'HEART RATE': 'hr',
    'RESPIRATORY RATE': 'rr',
    'SPO2': 'spo2',
    'TEMPERATURE CELSIUS': 'temp',
    'NON INVASIVE BLOOD PRESSURE SYSTOLIC': 'sbp',
    'NON INVASIVE BLOOD PRESSURE DIASTOLIC': 'dbp',
    'MEAN ARTERIAL PRESSURE (NIBP)': 'map',
    'CREATININE': 'creatinine',
    'UREA NITROGEN': 'bun',
    'SODIUM': 'sodium',
    'POTASSIUM': 'potassium',
    'BICARBONATE': 'bicarbonate',
    'URINE_OUTPUT': 'urine'
}


def fetch_chartevents() -> pd.DataFrame:
    labels = ",".join([f"'{l}'" for l in VITAL_LABELS])
    limit = " LIMIT 500000" if QUICK_TEST else ""
    sql = f"""
        SELECT ce.stay_id, ce.charttime, UPPER(di.label) AS label, ce.valuenum
        FROM chartevents ce
        JOIN d_items di ON di.itemid = ce.itemid
        JOIN tmp_ckd_stays s ON s.stay_id = ce.stay_id
        WHERE ce.valuenum IS NOT NULL
          AND ce.charttime >= s.intime
          AND ce.charttime <  DATE_ADD(s.intime, INTERVAL {T_HOURS} HOUR)
          AND UPPER(di.label) IN ({labels})
        {limit}
    """
    df = read_sql_df(sql)
    return df


def fetch_labevents() -> pd.DataFrame:
    labels = ",".join([f"'{l}'" for l in LAB_LABELS])
    limit = " LIMIT 200000" if QUICK_TEST else ""
    sql = f"""
        SELECT s.stay_id, le.charttime, UPPER(dl.label) AS label, le.valuenum
        FROM labevents le
        JOIN d_labitems dl ON dl.itemid = le.itemid
        JOIN tmp_ckd_stays s ON s.hadm_id = le.hadm_id
        WHERE le.valuenum IS NOT NULL
          AND le.charttime >= s.intime
          AND le.charttime <  DATE_ADD(s.intime, INTERVAL {T_HOURS} HOUR)
          AND UPPER(dl.label) IN ({labels})
        {limit}
    """
    df = read_sql_df(sql)
    return df


def fetch_urine() -> pd.DataFrame:
    limit = " LIMIT 300000" if QUICK_TEST else ""
    sql = f"""
        SELECT oe.stay_id, oe.charttime, UPPER(di.label) AS label, oe.valuenum
        FROM outputevents oe
        JOIN d_items di ON di.itemid = oe.itemid
        JOIN tmp_ckd_stays s ON s.stay_id = oe.stay_id
        WHERE oe.valuenum IS NOT NULL
          AND oe.charttime >= s.intime
          AND oe.charttime <  DATE_ADD(s.intime, INTERVAL {T_HOURS} HOUR)
          AND UPPER(di.label) LIKE '%URINE%'
        {limit}
    """
    df = read_sql_df(sql)
    if not df.empty:
        df['label'] = 'URINE_OUTPUT'
    return df

# 標籤：院內死亡 & 住院期間 RRT（procedures_icd）

def fetch_labels() -> pd.DataFrame:
    sql = f"""
        SELECT s.stay_id, s.subject_id, s.hadm_id, s.intime,
               (a.deathtime IS NOT NULL) AS y_mort
        FROM tmp_ckd_stays s
        JOIN admissions a USING(subject_id, hadm_id);
    """
    lab = read_sql_df(sql)
    # RRT（手術碼）
    rrt9 = ",".join([f"'{c}'" for c in RRT_ICD9])
    like10 = " OR ".join([f"REPLACE(p.icd_code,'.','') LIKE '{pfx}%'" for pfx in RRT_ICD10_PREFIX])
    sql_rrt = f"""
        SELECT DISTINCT subject_id, hadm_id, 1 AS y_rrt
        FROM procedures_icd p
        WHERE (p.icd_version=9  AND REPLACE(p.icd_code,'.','') IN ({rrt9}))
           OR (p.icd_version=10 AND ({like10}))
    """
    rrt = read_sql_df(sql_rrt)
    lab = lab.merge(rrt, on=['subject_id','hadm_id'], how='left')
    lab['y_rrt'] = lab['y_rrt'].fillna(0).astype(int)
    return lab

# ===================== 步驟 3：建構每小時序列 =====================

def to_hourly(df_long: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=['stay_id','time_index','feature','value'])
    # 合併起點時間
    df = df_long.merge(stays[['stay_id','intime']], on='stay_id', how='left')
    # 對齊到小時（以 ICU intime 為 0h）
    df['offset_h'] = (pd.to_datetime(df['charttime']) - pd.to_datetime(df['intime'])).dt.total_seconds()/3600.0
    df = df[(df['offset_h'] >= 0) & (df['offset_h'] < T_HOURS)].copy()
    df['time_index'] = df['offset_h'].round().astype(int)
    # 取同一小時的最後一筆（也可改 mean）
    df = df.sort_values(['stay_id','label','charttime']).groupby(['stay_id','label','time_index'], as_index=False).tail(1)
    df = df[['stay_id','label','time_index','valuenum']].rename(columns={'valuenum':'value'})
    return df


def build_tensor(char_hourly: pd.DataFrame, lab_hourly: pd.DataFrame, urine_hourly: pd.DataFrame,
                 stays: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    feat_frames = []
    for df in [char_hourly, lab_hourly, urine_hourly]:
        if df is None or df.empty:
            continue
        feat_frames.append(df)
    if not feat_frames:
        raise RuntimeError("No features extracted.")
    all_long = pd.concat(feat_frames, axis=0, ignore_index=True)
    # 映射統一欄名
    all_long['feature'] = all_long['label'].map(FEATURE_MAP)
    all_long = all_long.dropna(subset=['feature'])
    # 建 wide 表：每 stay × time × feature
    wide = all_long.pivot_table(index=['stay_id','time_index'], columns='feature', values='value', aggfunc='last')
    wide = wide.reset_index()
    # 為每個 stay 補滿 0..T_HOURS-1 的時間點
    stays_idx = stays[['stay_id']].drop_duplicates()
    grid = stays_idx.assign(key=1).merge(pd.DataFrame({'time_index':np.arange(T_HOURS), 'key':[1]*T_HOURS}), on='key').drop('key',axis=1)
    wide = grid.merge(wide, on=['stay_id','time_index'], how='left').sort_values(['stay_id','time_index'])
    # 依 stay 做 forward-fill，再填中位數
    feature_cols = [c for c in wide.columns if c not in ['stay_id','time_index']]
    def ffill_group(g):
        g[feature_cols] = g[feature_cols].ffill()
        return g
    wide = wide.groupby('stay_id', as_index=False).apply(ffill_group).reset_index(drop=True)
    # 中位數填補
    med = wide[feature_cols].median()
    wide[feature_cols] = wide[feature_cols].fillna(med)
    # 標準化（fit on all; 可改成 train-only 再回頭 transform）
    scaler = StandardScaler()
    scaled = scaler.fit_transform(wide[feature_cols])
    joblib.dump(scaler, os.path.join(OUTDIR, 'scaler.joblib'))
    # 組 tensor：N × T × F
    stays_order = wide[['stay_id']].drop_duplicates().values.ravel()
    N = len(stays_order)
    F = len(feature_cols)
    T = T_HOURS
    X = np.zeros((N, T, F), dtype=np.float32)
    for i, sid in enumerate(stays_order):
        blk = wide[wide['stay_id']==sid].sort_values('time_index')
        X[i,:,:] = blk[feature_cols].values[:T]
    return X, feature_cols, pd.DataFrame({'stay_id':stays_order})

# ===================== 步驟 4：模型與訓練 =====================

def build_gru(input_shape: Tuple[int,int]) -> keras.Model:
    inp = keras.Input(shape=input_shape, name='seq')
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc'), keras.metrics.AUC(curve='PR', name='auprc')])
    return model


def best_pr_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2*prec*rec)/(prec+rec+1e-12)
    i = int(np.nanargmax(f1s))
    if i == 0 or i > len(thr):
        return 0.5
    return float(thr[i-1])


def eval_metrics(y_true, y_prob, tag: str):
    thr = best_pr_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    res = {
        'threshold': thr,
        'AUC': float(roc_auc_score(y_true, y_prob)),
        'AUPRC': float(average_precision_score(y_true, y_prob)),
        'F1': float(f1_score(y_true, y_pred)),
        'Brier': float(brier_score_loss(y_true, y_prob))
    }
    print(f"\n[{tag}] thr={thr:.3f}  AUC={res['AUC']:.4f}  AUPRC={res['AUPRC']:.4f}  F1={res['F1']:.4f}  Brier={res['Brier']:.4f}")
    return res

# ===================== 主流程 =====================
if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)
    print("\n==> 建立 cohort...")
    build_cohort_temp_tables()#  1：建立 CKD + ICU cohort

    # 取得 stays 與標籤
    stays = read_sql_df("SELECT * FROM tmp_ckd_stays")
    if QUICK_TEST:
        stays = stays.sample(n=min(200, len(stays)), random_state=RANDOM_STATE)
        run_sql("DROP TEMPORARY TABLE IF EXISTS tmp_ckd_stays;")
        stays.to_sql('tmp_ckd_stays', ENG, if_exists='replace', index=False)  # 非 TEMP，但便於 demo

    labels = fetch_labels()
    labels = labels.merge(stays[['stay_id']], on='stay_id', how='inner')

    print("==> 抓取事件（chartevents/labevents/outputevents）...")
    df_char = fetch_chartevents()
    df_lab  = fetch_labevents()
    df_ur   = fetch_urine()

    print("==> 轉成每小時序列...")
    char_h = to_hourly(df_char, stays)
    lab_h  = to_hourly(df_lab,  stays)
    ur_h   = to_hourly(df_ur,   stays)

    X, feature_cols, sid_df = build_tensor(char_h, lab_h, ur_h, stays)

    # 對齊標籤
    Y_mort = sid_df.merge(labels[['stay_id','y_mort']], on='stay_id', how='left')['y_mort'].fillna(0).astype(int).values
    Y_rrt  = sid_df.merge(labels[['stay_id','y_rrt']],  on='stay_id', how='left')['y_rrt'].fillna(0).astype(int).values

    # Train/Val/Test 切分（按 stay 級）
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(sid_df))
    rng.shuffle(idx)
    n_test = int(len(idx)*TEST_SIZE)
    n_val  = int((len(idx)-n_test)*VAL_SIZE)
    te_idx = idx[:n_test]
    va_idx = idx[n_test:n_test+n_val]
    tr_idx = idx[n_test+n_val:]

    def run_task(Y, task_name):
        print(f"\n==== 任務：{task_name} ====")
        Xtr, Xva, Xte = X[tr_idx], X[va_idx], X[te_idx]
        ytr, yva, yte = Y[tr_idx], Y[va_idx], Y[te_idx]
        model = build_gru((X.shape[1], X.shape[2]))
        # 類別權重（避免不平衡）
        pos = max(ytr.sum(), 1)
        neg = max(len(ytr)-pos, 1)
        cw = {0: 0.5*len(ytr)/neg, 1: 0.5*len(ytr)/pos}
        cb = [
            keras.callbacks.EarlyStopping(monitor='val_auprc', mode='max', patience=12, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_auprc', mode='max', patience=5, factor=0.5, min_lr=1e-5)
        ]
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=80, batch_size=128, class_weight=cw, callbacks=cb, verbose=2)
        va_prob = model.predict(Xva, batch_size=256).ravel()
        te_prob = model.predict(Xte, batch_size=256).ravel()
        va_metrics = eval_metrics(yva, va_prob, tag=f'{task_name}-VAL')
        te_metrics = eval_metrics(yte, te_prob, tag=f'{task_name}-TEST')
        # 存檔
        mdir = os.path.join(OUTDIR, f'model_{task_name}')
        os.makedirs(mdir, exist_ok=True)
        model.save(os.path.join(mdir, 'gru_model.keras'))
        with open(os.path.join(mdir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({'val': va_metrics, 'test': te_metrics, 'features': feature_cols, 'T_hours': T_HOURS}, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(mdir, 'test_prob.npy'), te_prob)
        np.save(os.path.join(mdir, 'test_y.npy'),   yte)
        print(f"Artifacts saved at: {mdir}")

    # 跑兩個任務：院內死亡 / RRT
    run_task(Y_mort, 'mortality')
    run_task(Y_rrt,  'rrt')

    print("\n完成！輸出位於:", os.path.abspath(OUTDIR))
