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

--檢驗

select * from mimic3_1.d_labitems dl 
where itemid in(
'50912','51006','50971','50983','50902','50882','50868','51221','51265'
)

--CKD 病人的檢驗
select
charttime ,storetime , d_labitems.label ,value ,valuenum ,valueuom ,ref_range_lower ,ref_range_upper,flag,comments
-- a. 
from mimic3_1.labevents a

join ckd_admissions ca USING(subject_id, hadm_id  )
inner  join mimic3_1.d_labitems on( d_labitems.itemid =a.itemid )
where a.itemid in(
'50912','51006','50971','50983','50902','50882','50868','51221','51265'
)
and exists(
select * from ckd_adm_view where ckd_adm_view.subject_id =a.subject_id and ckd_adm_view.hadm_id =a.hadm_id

)
--心跳

where chartevents.itemid in(

select d_items.itemid from d_items WHERE UPPER(label) = 'HEART RATE'
)
 AND chartevents.valuenum IS NOT null
LIMIT 100;



