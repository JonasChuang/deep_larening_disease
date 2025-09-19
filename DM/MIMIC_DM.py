from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === MySQL 連線設定 ===
# user = "root"
# password = "your_password"
# host = "localhost"
# database = "mimiciv"
# engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
import pymysql
pymysql.install_as_MySQLdb()
engine = create_engine("mysql+pymysql://test:test@10.2.163.201:3306/mimic3_1")# 連接設定


def aa():

    '''
    比較不同年齡組、ICU 類型、存活 vs 死亡，並計算死亡率與 ICU 停留天數中位數，最後用 森林圖與熱力圖 展示結果
    篩選糖尿病患者（ICD-9 250.xx 或 ICD-10 E10/E11/E13）
    併發症判斷（ICD 診斷碼）
    連接 ICU 類型、年齡、出院狀態
    年齡分組（<50, 50–64, 65–79, ≥80）
    ICU 類型分類（SICU、MICU、CCU 等）
    存活 vs 死亡

    計算：
    併發症發生率
    院內死亡率
    ICU 停留天數中位數
  
    森林圖（Forest Plot） → 顯示各併發症的死亡風險（相對比率或死亡率）

    熱力圖（Heatmap） → 年齡分組 × ICU 類型 × 死亡率
    
    
    '''
    try:
        # === SQL 查詢糖尿病患者 ICU 資料 ===
        sql = """
        WITH diabetes_icustays AS (
            SELECT DISTINCT
                icu.subject_id, icu.hadm_id, icu.stay_id,
                icu.first_careunit AS icu_type,
                TIMESTAMPDIFF(YEAR, p.dod, a.admittime) AS age,
                (a.deathtime IS NOT NULL) AS died,
                TIMESTAMPDIFF(HOUR, icu.intime, icu.outtime)/24 AS icu_los
            FROM mimic3_1.icustays icu
            JOIN mimic3_1.admissions a ON icu.hadm_id = a.hadm_id
            JOIN mimic3_1.patients p ON icu.subject_id = p.subject_id
            JOIN mimic3_1.diagnoses_icd d ON icu.hadm_id = d.hadm_id
            WHERE (
                (d.icd_version = 9 AND d.icd_code LIKE '250%%')
                OR (d.icd_version = 10 AND (
                    d.icd_code LIKE 'E10%%'
                        OR d.icd_code LIKE 'E11%%'
                        OR d.icd_code LIKE 'E13%%'
                ))
            )
        )
        SELECT
            di.*, diag.icd_code, diag.icd_version
        FROM diabetes_icustays di
        JOIN mimic3_1.diagnoses_icd diag ON di.hadm_id = diag.hadm_id

        """

        df = pd.read_sql(sql, engine)

        # === 定義併發症對應 ICD 前綴 ===
        complications = {
            "Sepsis": ["99591", "A41"],
            "Acute Kidney Injury": ["584", "N17"],
            "Respiratory Failure": ["51881", "J96"],
            "Myocardial Infarction": ["410", "I21"],
            "Arrhythmia": ["427", "I47"],
            "DKA/HHS": ["2501", "E101"],
            "Hypoglycemia": ["2512", "E162"],
            "Fungal Infection": ["112", "B37"],
            "Thromboembolism": ["4151", "I26"]
        }

        # === 標記併發症 ===
        for comp, codes in complications.items():
            df[comp] = df.apply(
                lambda x: any(x["icd_code"].startswith(code) for code in codes if str(x["icd_code"]).startswith(code)),
                axis=1
            )

        # === 聚合到患者層級 ===
        group_cols = ["subject_id", "hadm_id", "icu_type", "age", "died", "icu_los"]
        df_grouped = df.groupby(group_cols).max().reset_index()

        # === 年齡分組 ===
        bins = [0, 50, 65, 80, 200]
        labels = ["<50", "50-64", "65-79", "≥80"]
        df_grouped["age_group"] = pd.cut(df_grouped["age"], bins=bins, labels=labels, right=False)

        # === 計算統計 ===
        stats = []
        for comp in complications.keys():
            temp = df_grouped[df_grouped[comp] == True]
            total_patients = len(df_grouped)
            comp_patients = len(temp)
            mortality = temp["died"].mean() * 100
            median_los = temp["icu_los"].median()
            stats.append([comp, comp_patients, comp_patients/total_patients*100, mortality, median_los])

        df_stats = pd.DataFrame(stats, columns=["Complication", "PatientCount", "PctPatients", "MortalityRate", "MedianICULOS"])

        # === 匯出 Excel ===
        df_stats.to_excel("diabetes_icu_complications_analysis.xlsx", index=False)

        # === 森林圖（死亡率） ===
        plt.figure(figsize=(8,6))
        plt.errorbar(df_stats["MortalityRate"], df_stats["Complication"], xerr=0, fmt='o')
        plt.xlabel("Mortality Rate (%)")
        plt.title("Mortality Rate by Complication (Diabetes ICU Patients)")
        plt.grid(True)
        plt.savefig("forest_plot_mortality.png", dpi=300)
        plt.show()

        # === 熱力圖（年齡 × ICU 類型 × 死亡率） ===
        heatmap_data = df_grouped.groupby(["age_group", "icu_type"])["died"].mean().unstack() * 100
        plt.figure(figsize=(10,6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="Reds")
        plt.title("ICU Mortality Rate (%) by Age Group and ICU Type")
        plt.savefig("heatmap_mortality.png", dpi=300)
        plt.show()
        return 0
    except Exception as Error:
        print("CSV_TO_DB 錯誤!!!!!!! \n"+str(Error))
if __name__ == '__main__':
    aa()