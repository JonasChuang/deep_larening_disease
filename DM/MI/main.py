import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1) 讀取你做好的特徵表 (一列=一個 stay/hadm，含 mort_30d 或 in_hosp_mort)
engine = create_engine("mysql+pymysql://user:pwd@host:3306/mimiciv")
df = pd.read_sql("SELECT * FROM mi_features_24h", engine)

y = df["mort_30d"]  # 或 in_hosp_mort
X = df.drop(columns=["mort_30d","in_hosp_mort","subject_id","hadm_id","stay_id"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2) 基線：邏輯斯回歸（含插補/標準化）
lr = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler(with_mean=False)),
    ("clf", LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1))
])

# 3) XGBoost（處理非線性、缺失）
xgb = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
    eval_metric="logloss", n_jobs=-1, tree_method="hist"
)

for name, model in [("LR", lr), ("XGB", xgb)]:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.predict_proba(X_test)[:,1]
    pred  = (proba >= 0.5).astype(int)
    print(name,
          "AUC=", roc_auc_score(y_test, proba),
          "AUPRC=", average_precision_score(y_test, proba),
          "F1=", f1_score(y_test, pred),
          "Brier=", brier_score_loss(y_test, proba))