-- CKD 住院 hadm_id（先用前面 ICD 過濾）
WITH ckd_hadm AS (
  SELECT DISTINCT hadm_id
  FROM diagnoses_icd
  WHERE (icd_version=10 AND REPLACE(icd_code,'.','') LIKE 'N18%')
     OR (icd_version=9  AND REPLACE(icd_code,'.','') LIKE '585%')
)
SELECT m.subject_id, m.hadm_id, m.charttime, m.spec_type_desc,
       m.org_name, m.ab_name, m.interpretation, m.dilution_value
FROM microbiologyevents m
JOIN ckd_hadm c USING(hadm_id)
WHERE m.org_name IS NOT NULL
ORDER BY m.hadm_id, m.charttime
LIMIT 100;

-- 看「血液培養」陽性結果
SELECT subject_id, hadm_id, charttime, spec_type_desc,
       org_name, ab_name, interpretation
FROM microbiologyevents
WHERE UPPER(spec_type_desc) LIKE '%BLOOD%'
  AND org_name IS NOT NULL
ORDER BY charttime
LIMIT 50;

-- 看某種菌的抗藥性（例：E. coli）
SELECT ab_name, interpretation, COUNT(*) AS n
FROM microbiologyevents
WHERE UPPER(org_name) LIKE '%COLI%'
GROUP BY ab_name, interpretation
ORDER BY ab_name;
