import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.formula.api import ols
from lifelines import CoxPHFitter
def a1():#顯示 95% 信賴區間
    # 假資料（或用 pd.read_csv 載入你自己的）
    data = {
        'Age': [50, 62, 45, 58, 39, 67, 52, 43, 60, 55],
        'BMI': [28.2, 31.5, 24.3, 35.1, 22.8, 29.4, 30.2, 27.9, 33.0, 26.5],
        'HbA1c': [7.2, 8.1, 6.5, 9.3, 6.0, 7.6, 8.0, 7.0, 8.5, 6.8],
        'Group': ['A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'B', 'A']
    }
    df = pd.DataFrame(data)

    # 畫圖
    plt.figure(figsize=(10, 6))
    sns.set(style='whitegrid')

    # 散佈圖 + 回歸線（依 group 分色）
    sns.lmplot(
        data=df,
        x='BMI',
        y='HbA1c',
        hue='Group',
        height=6,
        aspect=1.5,
        markers=['o', 's'],
        palette='Set2',
        ci=95  # 顯示 95% 信賴區間
    )

    plt.title('BMI vs HbA1c with Regression Line (by Group)')
    plt.xlabel('BMI')
    plt.ylabel('HbA1c (%)')
    plt.tight_layout()
    plt.show()
def a2():
   
    # 假資料
    data = {
        'PatientID': ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'],
        'BMI': [28.2, 31.5, 24.3, 35.1, 22.8, 29.4, 30.2, 27.9, 33.0, 26.5],
        'HbA1c': [7.2, 8.1, 6.5, 9.3, 6.0, 7.6, 8.0, 7.0, 8.5, 6.8]
    }
    df = pd.DataFrame(data)

    # 設定高風險條件
    high_risk = df[(df['BMI'] > 30) | (df['HbA1c'] > 8)]

    # 繪圖
    plt.figure(figsize=(10, 6))
    #sns.set(style='whitegrid')
    sns.set_theme(style="whitegrid", palette="Set2")
    # 基本散佈圖與回歸線
    sns.regplot(data=df, x='BMI', y='HbA1c', scatter=True, ci=95, color='skyblue')

    # 標註高風險個案
    for i, row in high_risk.iterrows():
        plt.text(row['BMI'] + 0.2, row['HbA1c'] + 0.1, row['PatientID'], color='red', fontsize=10)

    # 高風險區背景
    plt.axvspan(30, df['BMI'].max() + 1, color='red', alpha=0.05)
    plt.axhspan(8, df['HbA1c'].max() + 1, color='red', alpha=0.05)
    plt.text(30.5, 8.2, 'High Risk Zone', color='red', fontsize=12)

    # 標題與標籤
    plt.title('BMI vs HbA1c — Highlighting High-Risk Patients')
    plt.xlabel('BMI')
    plt.ylabel('HbA1c (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def a3(flg):#T檢定分析
    
    if flg =="":
        # 假資料
        data = {
            'Group': ['Intervention'] * 10 + ['Control'] * 10,
            'HbA1c': [7.2, 7.0, 6.8, 6.9, 7.1, 7.0, 6.7, 6.8, 7.0, 6.9,
                    7.8, 8.0, 7.5, 7.6, 7.7, 7.9, 7.4, 7.5, 7.6, 7.8]
        }
        df = pd.DataFrame(data)

        # T 檢定
        t_stat, p_val = ttest_ind(
            df[df['Group'] == 'Intervention']['HbA1c'],
            df[df['Group'] == 'Control']['HbA1c'],
            equal_var=False
        )

        # 畫圖
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="whitegrid", palette="Set2")

        # 箱形圖 + 點圖（顯示個別觀察）
        sns.boxplot(data=df, x='Group', y='HbA1c', palette='Set2')
        sns.stripplot(data=df, x='Group', y='HbA1c', color='black', alpha=0.6, jitter=0.1)

        # 標示平均值
        group_means = df.groupby('Group')['HbA1c'].mean()
        for i, mean in enumerate(group_means):
            plt.text(i, mean + 0.05, f"Mean: {mean:.2f}", ha='center', color='blue')

        # 顯示 p-value
        plt.title(f'HbA1c Comparison by Group (p = {p_val:.4f})', fontsize=14)
        plt.xlabel('Group')
        plt.ylabel('HbA1c (%)')
        plt.tight_layout()
        plt.show()
    if flg =="CSV":
        df = pd.read_csv('HbA1c.csv')  # 請替換為實際檔名

        # 去除遺失值
        df = df.dropna(subset=['group', 'HbA1c'])

        # 檢查欄位格式
        df['group'] = df['group'].astype(str)
        df['HbA1c'] = pd.to_numeric(df['HbA1c'], errors='coerce')

       
        plt.figure(figsize=(12, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='group', y='HbA1c', data=df)
        plt.title("HbA1c 各組分佈（Boxplot）")

        # Barplot（平均 ± 標準差）
        plt.subplot(1, 2, 2)
        sns.barplot(x='group', y='HbA1c', data=df, errorbar='sd')
        plt.title("HbA1c 各組平均 ± 標準差")

        plt.tight_layout()
        plt.show()
         # 選擇兩組資料（以 Control 與 MedicationA 為例）
        group1 = df[df['group'] == 'Control']['HbA1c']
        group2 = df[df['group'] == 'MedicationA']['HbA1c']

        # T 檢定分析
        t_stat, p_val = ttest_ind(group1, group2, equal_var=True)

        print("🎯 T 檢定結果")
        print(f"t 統計量 = {t_stat:.4f}")
        print(f"p 值 = {p_val:.4f}")
        if p_val < 0.05:
            print("→ HbA1c 在兩組之間有顯著差異（p < 0.05）")
        else:
            print("→ HbA1c 在兩組之間沒有顯著差異（p ≥ 0.05）")


def a4():
    # 假資料
    data = {
        'Group': ['Intervention'] * 10 + ['Control'] * 10,
        'HbA1c': [7.2, 7.0, 6.8, 6.9, 7.1, 7.0, 6.7, 6.8, 7.0, 6.9,
                7.8, 8.0, 7.5, 7.6, 7.7, 7.9, 7.4, 7.5, 7.6, 7.8],
        'BMI': [27.1, 26.5, 25.0, 26.0, 25.5, 26.3, 25.9, 25.6, 26.1, 25.7,
                29.0, 29.5, 28.8, 29.2, 29.1, 28.9, 29.3, 29.4, 28.7, 29.6],
        'FPG': [120, 115, 110, 118, 117, 116, 112, 114, 119, 113,
                135, 138, 136, 140, 137, 139, 134, 133, 132, 136]
    }
    df = pd.DataFrame(data)

    # 資料轉長格式（方便一次畫出多變數）
    df_melted = pd.melt(df, id_vars='Group', value_vars=['HbA1c', 'BMI', 'FPG'],
                        var_name='Indicator', value_name='Value')

    # 畫圖
    plt.figure(figsize=(10, 6))
    #sns.set(style="whitegrid")
    sns.set_theme(style="whitegrid", palette="Set2")
    sns.boxplot(data=df_melted, x='Indicator', y='Value', hue='Group', palette='Set2')

    plt.title('Comparison of HbA1c, BMI, and FPG by Group')
    plt.ylabel('Value')
    plt.xlabel('Indicator')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

def a5():#Logistic Regression
    ''''
    這裡提供一個完整的 Python 邏輯斯回歸（Logistic Regression）範例，
    針對糖尿病病患的檢驗數據進行預測（例如：預測是否為高風險個案，或血糖是否控制良好）。
    解釋：
    HighRisk = 1 表示 HbA1c 超過 7（血糖控制不良）

    可以解釋哪些變數（如 FPG）對高風險影響最大

    使用 confusion_matrix 和 classification_report 檢查準確率、召回率等指標
    '''
    #ROC 曲線	比較真陽性率 (TPR) 與 假陽性率 (FPR) 的關係
    #AUC 值	模型辨識能力的指標（1.0 為完美，0.5 為亂猜）

    # 假資料
    data = {
        'HbA1c': [6.5, 7.8, 6.8, 8.2, 7.0, 8.5, 6.6, 6.7, 7.1, 7.6],
        'FPG': [110, 145, 120, 160, 125, 170, 105, 115, 130, 150],
        'BMI': [24, 30, 26, 32, 27, 33, 25, 26, 28, 31],
        'Age': [55, 62, 58, 65, 60, 68, 53, 59, 61, 64]
    }
    df = pd.DataFrame(data)

    # 標籤：HbA1c > 7.0 為高風險
    df['HighRisk'] = (df['HbA1c'] > 7.0).astype(int)

    # 特徵與標籤
    X = df[['FPG', 'BMI', 'Age']]
    y = df['HighRisk']

    # 分割資料
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 建模
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 預測機率
    y_prob = model.predict_proba(X_test)[:, 1]  # 取預測為 1 的機率

    # 計算 ROC 曲線與 AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # 畫 ROC 圖
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for High Risk HbA1c Prediction')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def a6():#Logistic 預測機率視覺化圖（條狀分佈）
    ''''
    這裡提供一個完整的 Python 邏輯斯回歸（Logistic Regression）範例，
    針對糖尿病病患的檢驗數據進行預測（例如：預測是否為高風險個案，或血糖是否控制良好）。
    解釋：
    HighRisk = 1 表示 HbA1c 超過 7（血糖控制不良）

    可以解釋哪些變數（如 FPG）對高風險影響最大

    使用 confusion_matrix 和 classification_report 檢查準確率、召回率等指標
    '''
    #ROC 曲線	比較真陽性率 (TPR) 與 假陽性率 (FPR) 的關係
    #AUC 值	模型辨識能力的指標（1.0 為完美，0.5 為亂猜）

    # 假資料
    data = {
    'HbA1c': [6.5, 7.8, 6.8, 8.2, 7.0, 8.5, 6.6, 6.7, 7.1, 7.6],
    'FPG': [110, 145, 120, 160, 125, 170, 105, 115, 130, 150],
    'BMI': [24, 30, 26, 32, 27, 33, 25, 26, 28, 31],
    'Age': [55, 62, 58, 65, 60, 68, 53, 59, 61, 64]
    }
    df = pd.DataFrame(data)

    # 定義標籤：HbA1c > 7 為高風險
    df['HighRisk'] = (df['HbA1c'] > 7.0).astype(int)

    # 特徵與標籤
    X = df[['FPG', 'BMI', 'Age']]
    y = df['HighRisk']

    # 訓練模型
    model = LogisticRegression()
    model.fit(X, y)

    # 預測機率
    df['PredProb'] = model.predict_proba(X)[:, 1]

    # 按機率排序（從低到高）
    df_sorted = df.sort_values(by='PredProb').reset_index()

    # 畫圖：每位病人的預測機率
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df_sorted.index, y='PredProb', hue='HighRisk', data=df_sorted, dodge=False, palette={0: 'skyblue', 1: 'tomato'})

    plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Patient Index (Sorted by Predicted Probability)')
    plt.ylabel('Predicted Probability of High Risk')
    plt.title('Predicted Probability from Logistic Regression')
    plt.legend(title='Actual Risk (HighRisk)')
    plt.tight_layout()
    plt.show()
def a7():#邏輯斯回歸 + 預測機率 + 信賴區間
   

    # 建立資料
    data = {
        'HbA1c': [6.5, 7.8, 6.8, 8.2, 7.0, 8.5, 6.6, 6.7, 7.1, 7.6],
        'FPG':   [110, 145, 120, 160, 125, 170, 105, 115, 130, 150],
        'BMI':   [24, 30, 26, 32, 27, 33, 25, 26, 28, 31],
        'Age':   [55, 62, 58, 65, 60, 68, 53, 59, 61, 64]
    }
    df = pd.DataFrame(data)

    # 建立標籤：HbA1c > 7 為高風險
    df['HighRisk'] = (df['HbA1c'] > 7.0).astype(int)

    # 自變數與截距
    X = sm.add_constant(df[['FPG', 'BMI']])  # 建議移除 'Age' 以避免 singular matrix
    y = df['HighRisk']

    # 建立邏輯斯模型
    model = sm.Logit(y, X)
    result = model.fit()  # 注意：不能用 fit_regularized()，否則無法預測 CI

    # 預測機率與信賴區間
    pred = result.get_prediction(X)
    pred_df = pred.summary_frame(alpha=0.05)  # 包含 mean, mean_ci_lower, mean_ci_upper

    # 整併結果
    df['PredProb'] = pred_df['mean']
    df['CI_lower'] = pred_df['mean_ci_lower']
    df['CI_upper'] = pred_df['mean_ci_upper']

    # 按預測機率排序
    df_sorted = df.sort_values('PredProb').reset_index()

    # 繪圖：預測機率 + 信賴區間
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted.index, df_sorted['PredProb'], label='Predicted Probability', color='blue', marker='o')
    plt.fill_between(df_sorted.index, df_sorted['CI_lower'], df_sorted['CI_upper'],
                    color='skyblue', alpha=0.3, label='95% Confidence Interval')
    plt.axhline(0.5, linestyle='--', color='gray', label='Threshold = 0.5')
    plt.xlabel('Patient Index (Sorted)')
    plt.ylabel('Probability of High HbA1c')
    plt.title('Logistic Regression Predicted Probability with 95% CI')
    plt.legend()
    plt.tight_layout()
    plt.show()

def a7(flg):#邏糖尿病照護管理 檢驗 ANOVA 分析 統計圖

    if flg == "data":
        data = {
        'group': ['Control'] * 10 + ['MedicationA'] * 10 + ['MedicationB'] * 10,
        'HbA1c': [5.9, 6.1, 5.8, 6.0, 6.2, 5.7, 5.9, 6.0, 5.8, 6.1,
                6.5, 6.7, 7.0, 6.8, 7.2, 7.1, 6.6, 6.8, 7.0, 6.9,
                7.4, 7.6, 7.2, 7.5, 7.1, 7.3, 7.0, 7.4, 7.6, 7.5]
        }
        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='group', y='HbA1c', data=df)
        plt.title('HbA1c 分組 Boxplot')

        # 平均值圖
        plt.subplot(1, 2, 2)
        sns.barplot(x='group', y='HbA1c', data=df, errorbar='sd')
        plt.title('HbA1c 分組平均 ± 標準差')

        plt.tight_layout()
        plt.show()
        model = ols('HbA1c ~ C(group)', data=df).fit()

        # 執行 ANOVA
        anova_result = sm.stats.anova_lm(model, typ=2)

        print(anova_result)
        p_value = anova_result['PR(>F)'][0]
        if p_value < 0.05:
            print("→ 結論：各組 HbA1c 數值有顯著差異（p < 0.05）")
        else:
            print("→ 結論：各組 HbA1c 數值無顯著差異（p ≥ 0.05）")
    if flg == "CSV":
        df = pd.read_csv('HBA1C.csv')  # ⬅️ CSV 檔需有 "group" 和 "HbA1c" 欄位

        # 確認資料
        print(df.head())
        plt.figure(figsize=(12, 5))

        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='group', y='HbA1c', data=df)
        plt.title('HbA1c 各組 Boxplot')

        # Barplot：平均 ± 標準差（新版寫法）
        plt.subplot(1, 2, 2)
        sns.barplot(x='group', y='HbA1c', data=df, errorbar='sd')
        plt.title('HbA1c 各組平均 ± 標準差')

        plt.tight_layout()
        plt.show()
        # 建立線性模型（OLS）
        model = ols('HbA1c ~ C(group)', data=df).fit()

        # 執行 ANOVA 檢定
        anova_result = sm.stats.anova_lm(model, typ=2)
        print("\nANOVA 檢定結果：")
        print(anova_result)

        # 判斷顯著性
        p_value = anova_result['PR(>F)'][0]
        if p_value < 0.05:
            print("→ 各組 HbA1c 平均值有顯著差異（p < 0.05）")
        else:
            print("→ 各組 HbA1c 平均值無顯著差異（p ≥ 0.05）")

def a8(flg):#糖尿病人得到腦中?
    ''''
    結果解釋重點
    exp(coef)：Hazard Ratio（HR）風險比

    1：增加風險

    <1：降低風險

    p 值：若 < 0.05 表示該變數對事件（如中風）有顯著影響

    confidence interval：風險比的可信區間

    可以放入的變數示例：
    變數名稱	說明
    age	年齡
    sex	性別（0=女, 1=男）
    diabetes	是否為糖尿病患者
    BP	血壓
    hba1c	HbA1c 數值
    stroke_time	發生中風的時間

    '''
    
    
    df = pd.read_csv("DM/COX_CVA.csv")  # 你的 CSV 檔案
    df.columns = df.columns.str.strip().str.lower()

    # 重新命名必要欄位
    df = df.rename(columns={
        "追蹤時間": "time",
        "中風事件": "event",
        "年齡": "age",
        "性別": "sex",
        "糖尿病": "diabetes"
    })

    df = df[["time", "event", "age", "sex", "diabetes"]]
    # 確保欄位沒缺值
    df = df.dropna()
   
    df = df.astype(float)# 確保欄位類型正確

    # 建立 Cox 比例風險模型
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')

    # 印出 Cox 回歸結果
    cph.print_summary()  # 包含 HR, CI, p-value 等

    # 畫出各變數的 hazard ratio ?
    cph.plot()
    plt.title("Cox 回歸：各變數對中風風險的影響")
    plt.tight_layout()
    plt.show()
    return 0

def a9():#糖尿病人得到腦中?
    ''''
    結糖尿病患 得胰臟癌的 COX 迴歸分析圖

    '''
    
    
    df = pd.read_csv("DM/COX_CVA2.csv")  # 你的 CSV 檔案


    # 重新命名欄位（視情況而定）
    df = df.rename(columns={
        "followup_days": "time",
        "cancer_event": "event"
    })

    # 保留 Cox 分析用變數（避免像 patient_id 這種文字欄位干擾）
    df = df[["time", "event", "age", "sex", "diabetes"]]

    # 處理 sex 為數值（例如 M=1, F=0）
    df["sex"] = df["sex"].map({"M": 1, "F": 0})

    # 建立 Cox 模型
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")

    # 顯示分析摘要
    cph.print_summary()
    cph.plot()
    plt.title("Cox Hazard Ratio：糖尿病與胰臟癌風險分析")
    plt.tight_layout()
    plt.show()

    return 0
def a10():#糖尿病人得到腦中?
    ''''
    結糖尿病患 得胰臟癌的 COX 迴歸分析圖

    '''
    
    
    df = pd.read_csv("DM/COX_CVA3.csv")  # 請換成你自己的檔名

    # 確保正確欄位命名與格式
    df = df.rename(columns={
        "followup_days": "time",
        "MI_event": "event"
    })

    # 若性別是文字也轉數字（如有）
    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = df["sex"].map({"M": 1, "F": 0})

    # 只選取 Cox 變數
    df = df[["time", "event", "HbA1c", "cholesterol"]].dropna()

    # 模型建立
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")

    # 印出結果
    cph.print_summary()
    cph.plot()
    plt.title("HbA1c 與 Cholesterol 對心肌梗塞風險的影響")
    plt.tight_layout()
    plt.show()

    return 0

if __name__ == '__main__':
    a10()

