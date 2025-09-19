"""
(B) 時序加值：GRU（每小時一點，0–24h）

用 chartevents/labevents（不含鉀）+（可選）尿量，做 0–24h × 每小時的張量，再預測 24–48h 是否發生高血鉀。思路與你之前 CKD/GRU 腳本相同，只需把特徵清單換掉、保持排除 K 值即可。

4) 模型與實務重點

避免洩漏：特徵嚴禁包含標註窗（24–48h）內的鉀值；早期偵測任務也不要同時使用當次 K 值。

臨床可解釋：鉀會受 腎功能（Cr/BUN）、酸鹼（HCO₃⁻）、用藥（ACEi/ARB/K-sparing diuretics、KCl）、尿量 影響；可逐步納入。

不平衡：高血鉀比例通常不高，請回報 AUPRC，並用 class weight / 調閾值。

校正：臨床上要看機率可信度，建議加 Platt/Isotonic 校正。

外部驗證：可用不同年份或 eICU 交叉檢查泛化。
"""

# pip install pandas SQLAlchemy PyMySQL scikit-learn tensorflow joblib
import pandas as pd, numpy as np, tensorflow as tf
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, precision_recall_curve
import keras
from keras import layers

engine = create_engine("mysql+pymysql://user:pwd@host:3306/mimiciv")
df = pd.read_sql("SELECT * FROM hk_features_24to48", engine)

y = df["y_hk_24_48"].astype(int).values
X = df.drop(columns=["subject_id","hadm_id","admittime","y_hk_24_48"])
# 丟掉全空欄
X = X.loc[:, X.notna().any(0)]

# Train/Test
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preprocess
prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=True))])
Xtr_np = prep.fit_transform(Xtr); Xte_np = prep.transform(Xte)

# MLP
inp = keras.Input(shape=(Xtr_np.shape[1],))
x = layers.Dense(128, activation="relu")(inp); x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x);   x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="auprc")])

cb = [keras.callbacks.EarlyStopping(monitor="val_auprc", mode="max", patience=10, restore_best_weights=True)]
model.fit(Xtr_np, ytr, validation_split=0.2, epochs=200, batch_size=256, callbacks=cb, verbose=2)

proba = model.predict(Xte_np, batch_size=1024).ravel()
prec, rec, thr = precision_recall_curve(yte, proba); f1 = (2*prec*rec/(prec+rec+1e-12)); t = thr[max(f1.argmax()-1,0)]
pred = (proba >= t).astype(int)
print("AUC=", roc_auc_score(yte, proba), "AUPRC=", average_precision_score(yte, proba),
      "F1=", f1.max(), "Brier=", brier_score_loss(yte, proba))
