--MI的病人
WITH mi_dx AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM diagnoses_icd
  WHERE (icd_version = 10 AND (icd_code LIKE 'I21%' OR icd_code LIKE 'I22%'))
     OR (icd_version = 9  AND  icd_code LIKE '410%')
)
SELECT a.subject_id, a.hadm_id, i.stay_id, i.first_careunit, i.intime, i.outtime
FROM admissions a
LEFT JOIN icustays i USING (subject_id, hadm_id)
JOIN mi_dx USING (subject_id, hadm_id);


--院內死亡與 30 天死亡
SELECT
  a.subject_id, a.hadm_id,
  CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END AS in_hosp_mort,
  CASE
    WHEN a.deathtime IS NOT NULL
         AND TIMESTAMPDIFF(DAY, a.admittime, a.deathtime) <= 30 THEN 1
    WHEN p.dod IS NOT NULL
         AND TIMESTAMPDIFF(DAY, a.admittime, p.dod) <= 30 THEN 1
    ELSE 0
  END AS mort_30d
FROM admissions a
LEFT JOIN patients p USING (subject_id);


/*特徵萃取（前 24 小時）
常用族群：年齡/性別、生命徵象（心率、血壓、呼吸、體溫、SpO₂）、實驗室（BUN、Cr、Na/K/HCO₃⁻、乳酸、血紅素、WBC、血小板、PT/INR、Glucose）
、治療/處置（升壓藥、機械通氣、IABP、PCI/CABG 程序碼）、共病（可計算 Charlson/Elixhauser）。
MIMIC 官方的 vitalsign 衍生查詢 可直接把 chartevents 轉成「每次量測一列」的整齊表，便於在前 24h 做 min/max/mean/last 匯總；Charlson 計算 SQL 也已提供。
GitHub
+1

示例：以 ICU 入室起算 24h 的實驗室匯總（MySQL）
*/
WITH first24 AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
         TIMESTAMPADD(HOUR, 24, i.intime) AS t24
  FROM icustays i
),
labs AS (
  SELECT f.stay_id, dl.label,
         AVG(le.valuenum) AS lab_mean,
         MIN(le.valuenum) AS lab_min,
         MAX(le.valuenum) AS lab_max
  FROM labevents le
  JOIN first24 f
    ON le.subject_id = f.subject_id
   AND le.hadm_id    = f.hadm_id
   AND le.charttime >= f.intime
   AND le.charttime <  f.t24
  JOIN d_labitems dl ON dl.itemid = le.itemid
  WHERE le.valuenum IS NOT NULL
    AND dl.label IN ('BUN','Creatinine','Sodium','Potassium','Bicarbonate','Lactate',
                     'Glucose','Hemoglobin','WBC','Platelet','PT','INR')
  GROUP BY f.stay_id, dl.label
)
SELECT * FROM labs;