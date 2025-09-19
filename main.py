

import os
from sqlalchemy import create_engine
from Hyperkalemia import Hyperkalemia  #高血鉀 預測

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