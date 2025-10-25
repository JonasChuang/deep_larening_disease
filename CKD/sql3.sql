/*ckd_adm_view:CKD 住院的病人 2397 人*/
/*INSERT INTO   mimic3_1.ckd_admissions
(subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, `language`, marital_status, race, edregtime, edouttime, hospital_expire_flag, icd1, icd2, icd3)
*/
SELECT
  a.subject_id, a.hadm_id,
  a.admittime, a.dischtime, a.deathtime,
  admission_type, admit_provider_id, admission_location, discharge_location, insurance, `language`, marital_status, race, edregtime, edouttime, hospital_expire_flag,
 ( select CASE WHEN diagnoses_icd.icd_code REGEXP '^250|^E10|^E11|^E13' THEN 'Diabetes'
        WHEN diagnoses_icd.icd_code REGEXP '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70' THEN 'Cardiovascular'
        WHEN diagnoses_icd.icd_code REGEXP '^2777|^272|^E88\.81|^E78|^278|^E66' THEN 'Metabolic'
        ELSE diagnoses_icd.icd_code END from diagnoses_icd  where diagnoses_icd.subject_id=a.subject_id and diagnoses_icd.hadm_id = a.hadm_id and seq_num=1 ) as ICD_1,
  ( select  CASE WHEN diagnoses_icd.icd_code REGEXP '^250|^E10|^E11|^E13' THEN 'Diabetes'
        WHEN diagnoses_icd.icd_code REGEXP '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70' THEN 'Cardiovascular'
        WHEN diagnoses_icd.icd_code REGEXP '^2777|^272|^E88\.81|^E78|^278|^E66' THEN 'Metabolic'
        ELSE diagnoses_icd.icd_code END from diagnoses_icd  where diagnoses_icd.subject_id=a.subject_id and diagnoses_icd.hadm_id = a.hadm_id and seq_num=2 ) as ICD_2,
  ( select  CASE WHEN diagnoses_icd.icd_code REGEXP '^250|^E10|^E11|^E13' THEN 'Diabetes'
        WHEN diagnoses_icd.icd_code REGEXP '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70' THEN 'Cardiovascular'
        WHEN diagnoses_icd.icd_code REGEXP '^2777|^272|^E88\.81|^E78|^278|^E66' THEN 'Metabolic'
        ELSE diagnoses_icd.icd_code END from diagnoses_icd  where diagnoses_icd.subject_id=a.subject_id and diagnoses_icd.hadm_id = a.hadm_id and seq_num=3 ) as ICD_3,

        (patients.anchor_age + YEAR(a.admittime) - patients.anchor_year) as age_years,
gender
FROM admissions a
INNER join patients  on(patients.subject_id =a.subject_id  )
where a.subject_id in(
    -- 2014 ~2022 年病患
SELECT p.subject_id
FROM patients p where anchor_year_group in('2020 - 2022','2014 - 2016')
)
and (patients.anchor_age + YEAR(a.admittime) - patients.anchor_year) >= 18

-- N18 CKD 診斷
and exists(

select * from diagnoses_icd where  diagnoses_icd.subject_id=a.subject_id and diagnoses_icd.hadm_id = a.hadm_id 
 and seq_num in(1,2,3)
and  diagnoses_icd.icd_version = 10 
and (diagnoses_icd.icd_code LIKE 'N18%' -- CKD
or diagnoses_icd.icd_code LIKE 'N17%' -- AKI
or  diagnoses_icd.icd_code LIKE 'E09%'or diagnoses_icd.icd_code LIKE 'E10%'or diagnoses_icd.icd_code LIKE 'E11%'or diagnoses_icd.icd_code LIKE 'E12%'or diagnoses_icd.icd_code LIKE 'E13%' -- DM
or diagnoses_icd.icd_code LIKE 'I21%'or diagnoses_icd.icd_code LIKE 'I22%'  -- AMI
)

/*
50868	Anion Gap
50882	Bicarbonate
50902	Chloride
50912	Creatinine
50971	Potassium
50983	Sodium
51006	Urea Nitrogen
51221	Hematocrit
51265	Platelet Count
*/

)

--檢驗平均值




select
    a.itemid,
    a.lab_name                          AS 實驗室項目,
    a.fluid,
    a.category,
    COUNT(a.lab_name  ) as 筆數 ,
     round(MIN(a.valuenum)  ,2)                     AS 最小值,
     round( MAX(a.valuenum) ,2)                        AS 最大值,
    round( AVG(a.valuenum),2)                     AS 平均值,
    
   
     (    SELECT 
   round(AVG(valuenum),2) AS median_value
FROM (
  SELECT 
    valuenum,
    ROW_NUMBER() OVER (ORDER BY valuenum) AS row_num,
    COUNT(*) OVER () AS total_rows
  FROM z_ckd_lab where itemid=a.itemid
) AS ordered
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
      round(STDDEV(a.valuenum) ,2)                  AS 標準差
FROM
    z_ckd_lab a
    GROUP by a.lab_name ,a.itemid,a.fluid,a.category
    -- order by a.lab_name asc
    UNION
    select  '' as itemid ,
    'EGFR(入院)'                          AS 實驗室項目,
    '' as fluid,
    '' as category,
    COUNT(a.admit_egfr  ) as 筆數 ,
    MIN(a.admit_egfr)                     AS 最小值,
    MAX(a.admit_egfr)                     AS 最大值,
    round( AVG(a.admit_egfr),2)                     AS 平均值,
    
  
          (    SELECT 
  AVG(admit_egfr) AS median_value
FROM (
  SELECT 
    admit_egfr,
    ROW_NUMBER() OVER (ORDER BY admit_egfr) AS row_num,
    COUNT(*) OVER () AS total_rows
  FROM z_ckd_adm where z_ckd_adm.admit_egfr is not null
) AS ordered
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
        round(STDDEV(a.admit_egfr) ,2)                  AS 標準差
    from z_ckd_adm a
    where a.admit_egfr is not null
  UNION
    select  '' as itemid ,
    'EGFR(最新)'                          AS 實驗室項目,
    '' as fluid,
    '' as category,
    COUNT(a.EGFR  ) as 筆數 ,
    MIN(a.EGFR)                     AS 最小值,
    MAX(a.EGFR)                     AS 最大值,
    round( AVG(a.EGFR),2)                     AS 平均值,
    
  
          (    SELECT 
  AVG(EGFR) AS median_value
FROM (
  SELECT 
    EGFR,
    ROW_NUMBER() OVER (ORDER BY EGFR) AS row_num,
    COUNT(*) OVER () AS total_rows
  FROM z_ckd_adm where z_ckd_adm.EGFR is not null
) AS ordered
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
        round(STDDEV(a.EGFR) ,2)                  AS 標準差
    from z_ckd_adm a
    where a.EGFR is not null
    UNION
    select  '' as itemid ,
    'age_years'                          AS 實驗室項目,
    '' as fluid,
    '' as category,
    COUNT(a.age_years  ) as 筆數 ,
    MIN(a.age_years)                     AS 最小值,
    MAX(a.age_years)                     AS 最大值,
    round( AVG(a.age_years),2)                     AS 平均值,
    
  
          (    SELECT 
  AVG(age_years) AS median_value
FROM (
  SELECT 
    age_years,
    ROW_NUMBER() OVER (ORDER BY age_years) AS row_num,
    COUNT(*) OVER () AS total_rows
  FROM z_ckd_adm where z_ckd_adm.age_years is not null
) AS ordered
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
        round(STDDEV(a.age_years) ,2)                  AS 標準差
    from z_ckd_adm a
    where a.age_years is not null
    UNION
     select  '' as itemid ,
    'bmi'                          AS 實驗室項目,
    '' as fluid,
    '' as category,
    COUNT(a.bmi  ) as 筆數 ,
    MIN(a.bmi)                     AS 最小值,
    MAX(a.bmi)                     AS 最大值,
    round( AVG(a.bmi),2)                     AS 平均值,
    
  
          (    SELECT 
   round(AVG(bmi),2) AS median_value
FROM (
  SELECT 
    bmi,
    ROW_NUMBER() OVER (ORDER BY bmi) AS row_num,
    COUNT(*) OVER () AS total_rows
  FROM z_ckd_adm where z_ckd_adm.bmi is not null
) AS ordered
WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
        round(STDDEV(a.bmi) ,2)                  AS 標準差
    from z_ckd_adm a
    where a.bmi is not null
    
    

    
    

-- 48 小時內上升 ≥0.3 mg/dL 的配對（s0 → s1）
SELECT
  lab1.subject_id,
  lab1.hadm_id,
  lab1.charttime as lab1_rpttime,
  
 lab1.valuenum AS lab1_rpt,
 lab2.charttime as lab2_rpttime,
  
 lab2.valuenum AS lab2_rpt
 
FROM z_ckd_lab lab1
JOIN z_ckd_lab lab2 ON lab2.hadm_id   = lab1.hadm_id
   
   AND lab2.charttime > lab1.charttime
   AND lab2.charttime <= lab1.charttime + INTERVAL 48 HOUR
   AND (lab2.valuenum - lab1.valuenum) >= 0.3
   AND lab1.valuenum BETWEEN 0.1 AND 20    -- 合理值防呆（可調）
   AND lab2.valuenum BETWEEN 0.1 AND 20

-- 高血鉀
SELECT 
hadm_id
,case when max( valuenum) >5.5 then 1 else 0 end as y_hk

FROM z_ckd_lab zcl where itemid in('50971','52610','50822','52452')
group by hadm_id