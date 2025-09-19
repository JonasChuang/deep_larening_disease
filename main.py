

import os
from sqlalchemy import create_engine
from Hyperkalemia import Hyperkalemia,shap_view  #高血鉀 預測

# ======== 使用者設定（可用環境變數覆寫） ========
USER = os.getenv("MIMIC_MYSQL_USER", "user")
PWD  = os.getenv("MIMIC_MYSQL_PWD",  "password")
HOST = os.getenv("MIMIC_MYSQL_HOST", "10.2.163.201")
PORT = int(os.getenv("MIMIC_MYSQL_PORT", "3306"))
DB   = os.getenv("MIMIC_MYSQL_DB",   "mimiciv")
OUTDIR = os.getenv("OUTDIR", "./artifacts_hk_seq")

ENG = create_engine("mysql+pymysql://test:test@10.2.163.201:3306/mimic3_1")# 連接設定
if __name__ == "__main__":
    #shap_view.main()
    Hyperkalemia.main()
    print("OK")