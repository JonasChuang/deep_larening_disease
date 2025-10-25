from main import ENG
import os, json, joblib, numpy as np, pandas as pd
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

def make_cohort():
    # 以 ICU 入室為對齊（若改成入院，請把 icustays 改 admissions，t0=admittime）
    # intime 入ICU 時間
    
    #大母體
    stays = pd.read_sql_query(text("SELECT * FROM icu_adm_view"), ENG)
    stays.to_csv("Hyperkalemia/CSV/icu_adm_view.csv", index=False, encoding="utf-8-sig")
       
        


        
    
    #fetch_chartevents
    SQL = f"""
    /*
     'HEART RATE':220045
     ,'RESPIRATORY RATE':220210
     ,'SPO2','TEMPERATURE CELSIUS':223762,
        'NON INVASIVE BLOOD PRESSURE SYSTOLIC',:220179
        'NON INVASIVE BLOOD PRESSURE DIASTOLIC':220180,
       
    */
    -- 常見 itemid：心跳、呼吸率、NIBP、體溫等（這裡示範用 d_items.label 過濾）
    SELECT ce.stay_id, ce.charttime, UPPER(di.label) AS label, ce.valuenum
    FROM icu_adm_view s
    JOIN chartevents ce ON ce.stay_id = s.stay_id
    JOIN d_items di ON di.itemid = ce.itemid
    WHERE ce.charttime >= s.t0 AND ce.charttime < s.t_end
    AND ce.itemid IN ('220045','220179','220180','220210','223762')
    AND ce.valuenum REGEXP '^[0-9]+$' AND CAST(ce.valuenum AS UNSIGNED) > 1
    AND UPPER(di.label) IN (
        'HEART RATE','RESPIRATORY RATE','SPO2','TEMPERATURE CELSIUS',
        'NON INVASIVE BLOOD PRESSURE SYSTOLIC',
        'NON INVASIVE BLOOD PRESSURE DIASTOLIC',
      
    );
    
    """


    
    fetch_chartevents=pd.read_sql(text(SQL), ENG)
    fetch_chartevents.to_csv("Hyperkalemia/CSV/fetch_chartevents.csv", index=False, encoding="utf-8-sig")
    
    print("chartevents OK")
    #檢驗
    SQL = f"""
    SELECT s.stay_id, le.charttime, UPPER(dl.label) AS label, le.valuenum
    FROM icu_adm_view s
    JOIN labevents le ON le.hadm_id = s.hadm_id
    JOIN d_labitems dl ON dl.itemid = le.itemid
    WHERE le.valuenum IS NOT NULL
    AND le.charttime >= s.t0 AND le.charttime < s.t_end
    AND UPPER(dl.label) IN ('POTASSIUM','CHLORIDE','BICARBONATE',
                            'CREATININE','UREA NITROGEN','GLUCOSE','LACTATE');
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
      -- 低鈉
       SELECT s.stay_id,
             MAX(CASE WHEN le.valuenum >= '5.5' THEN 1 ELSE 0 END) AS y_hk
      FROM icu_adm_view s
      JOIN labevents le FORCE INDEX (idx_hadm_item_time_val) ON le.hadm_id = s.hadm_id
      JOIN d_labitems dl ON dl.itemid = le.itemid
      WHERE  le.charttime >= s.lbl_from AND le.charttime < s.lbl_to
       AND UPPER(dl.label) LIKE '%SODIUM%'
       AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
     -- and labevents in(50833,50971,52610)
     
       
       GROUP BY s.stay_id
      
    """
    low_sodium = pd.read_sql(text(SQL), ENG)
    low_sodium.to_csv("Hyperkalemia/CSV/low_sodium.csv", index=False, encoding="utf-8-sig")
    print("low_sodium OK")



    



    return 0