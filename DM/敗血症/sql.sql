WITH sepsis_dx AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM mimic3_1.diagnoses_icd d
  WHERE
    (d.icd_version = 9  AND (
       REPLACE(d.icd_code,'.','') LIKE '038%'   OR
       REPLACE(d.icd_code,'.','') = '99591'     OR
       REPLACE(d.icd_code,'.','') = '99592'     OR
       REPLACE(d.icd_code,'.','') = '78552'
    ))
    OR
    (d.icd_version = 10 AND (
       REPLACE(d.icd_code,'.','') LIKE 'A40%'   OR
       REPLACE(d.icd_code,'.','') LIKE 'A41%'   OR
       REPLACE(d.icd_code,'.','') = 'R6520'     OR
       REPLACE(d.icd_code,'.','') = 'R6521'     OR
       REPLACE(d.icd_code,'.','') = 'R572'
    ))
)
SELECT
  a.subject_id, a.hadm_id,
  a.admittime, a.dischtime, a.deathtime,
  a.admission_type, a.insurance, a.language, a.marital_status
FROM mimic3_1.admissions a
JOIN sepsis_dx s USING(subject_id, hadm_id)
ORDER BY a.subject_id, a.admittime;

-- （可選）只取每位病人的「第一次敗血症住院」
WITH sepsis_hadm AS (
  SELECT
    a.subject_id, a.hadm_id, a.admittime,
    ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS rn
  FROM mimic3_1.admissions a
  JOIN (
    SELECT DISTINCT subject_id, hadm_id
    FROM mimic3_1.diagnoses_icd d
    WHERE
      (d.icd_version = 9  AND (
         REPLACE(d.icd_code,'.','') LIKE '038%' OR
         REPLACE(d.icd_code,'.','') IN ('99591','99592','78552')
      ))
      OR
      (d.icd_version = 10 AND (
         REPLACE(d.icd_code,'.','') LIKE 'A40%' OR
         REPLACE(d.icd_code,'.','') LIKE 'A41%' OR
         REPLACE(d.icd_code,'.','') IN ('R6520','R6521','R572')
      ))
  ) s USING(subject_id, hadm_id)
)
SELECT * FROM sepsis_hadm WHERE rn = 1;

/*
方法 B：Sepsis-3「疑似感染（Suspected Infection）」住院（不含 SOFA）
Sepsis-3 建議以「抗生素 + 培養」的時間窗定義「疑似感染」：

若 先培養，則 72 小時內要有抗生素

若 先用抗生素，則 24 小時內要有培養
下例用 prescriptions.starttime 當抗生素時間、microbiologyevents.charttime 當採檢時間。抗生素以常見關鍵字比對（可依你環境補齊）。


*/

WITH abx AS (
  SELECT DISTINCT subject_id, hadm_id, starttime AS abx_time, drug
  FROM prescriptions
  WHERE UPPER(drug) REGEXP
    'PIPERACILLIN|TAZOBACTAM|MEROPENEM|IMIPENEM|ERTAPENEM|CEFTRIAXONE|CEFTAZIDIME|CEFEPIME|'
    'CEFAZOLIN|CEFOXITIN|CEFUROXIME|CIPROFLOXACIN|LEVOFLOXACIN|MOXIFLOXACIN|GENTAMICIN|'
    'AMIKACIN|VANCOMYCIN|LINEZOLID|DAPTOMYCIN|AZITHROMYCIN|CLARITHROMYCIN|'
    'AMPICILLIN|AMOXICILLIN|AMOXICILLIN/CLAVULANATE|AMPICILLIN/SULBACTAM|'
    'PENICILLIN|FLUCONAZOLE|CASPOFUNGIN|ANIDULAFUNGIN|MICAFUNGIN'
),

-- 2) 取（血液）培養時間（如要放寬，可拿掉 BLOOD 條件）
cult AS (
  SELECT subject_id, hadm_id, charttime AS cult_time, spec_type_desc
  FROM mimic3_1.microbiologyevents
  WHERE UPPER(spec_type_desc) LIKE '%BLOOD%'
),

-- 3) 疑似感染時間窗：cult→abx ≤72h，或 abx→cult ≤24h
suspected_infection AS (
  SELECT
    COALESCE(a.subject_id, c.subject_id) AS subject_id,
    COALESCE(a.hadm_id, c.hadm_id)       AS hadm_id,
    LEAST(a.abx_time, c.cult_time)       AS soi_time
  FROM abx a
  JOIN cult c
    ON a.subject_id = c.subject_id AND a.hadm_id = c.hadm_id
   AND (
        (c.cult_time BETWEEN a.abx_time AND DATE_ADD(a.abx_time, INTERVAL 24 HOUR)) OR
        (a.abx_time BETWEEN c.cult_time AND DATE_ADD(c.cult_time, INTERVAL 72 HOUR))
       )
)

-- 4) 以住院為單位輸出「疑似感染住院」
SELECT
  a.subject_id, a.hadm_id, a.admittime, a.dischtime,
  MIN(s.soi_time) AS first_soi_time
FROM mimic3_1.admissions a
JOIN suspected_infection s USING(subject_id, hadm_id)
GROUP BY a.subject_id, a.hadm_id, a.admittime, a.dischtime
ORDER BY a.subject_id, a.admittime;