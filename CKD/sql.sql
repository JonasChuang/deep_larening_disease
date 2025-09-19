/* ============================================================
   MIMIC-IV｜CKD 住院 × 入院後 0–24h：身高/體重/心跳 特徵表
   需求：MySQL 8+
   說明：
     - CKD 定義：ICD-10 N18.*；ICD-9 585.*
     - 時窗：以 admissions.admittime 起算 0–24 小時
     - 來源：chartevents（透過 d_items 的 label 過濾）
     - 產出表：ckd_vitals_24h（每住院 hadm_id 一列）
   ============================================================ */

USE mimic3_1;  -- ← 改成你的 DB 名稱

/* 0)（建議）索引加速：首次執行一次即可 */
CREATE INDEX IF NOT EXISTS idx_dx_icd     ON diagnoses_icd (icd_version, icd_code, subject_id, hadm_id);
CREATE INDEX IF NOT EXISTS idx_chartevt   ON chartevents (hadm_id, stay_id, charttime, itemid, valuenum);
CREATE INDEX IF NOT EXISTS idx_d_items    ON d_items (itemid, label);
CREATE INDEX IF NOT EXISTS idx_admissions ON admissions (hadm_id, admittime, dischtime);

/* 1) 取得 CKD 住院 cohort（hadm 層級） */
DROP TEMPORARY TABLE IF EXISTS cohort_ckd;
CREATE TEMPORARY TABLE cohort_ckd AS
SELECT DISTINCT d.subject_id, d.hadm_id
FROM diagnoses_icd d
WHERE
  (d.icd_version = 10 AND REPLACE(d.icd_code,'.','') LIKE 'N18%')
  OR
  (d.icd_version = 9  AND REPLACE(d.icd_code,'.','') LIKE '585%');

/* （可選）只保留成人（≥18 歲）——取消下列註解以啟用
DROP TEMPORARY TABLE IF EXISTS cohort_ckd_adult;
CREATE TEMPORARY TABLE cohort_ckd_adult AS
SELECT c.subject_id, c.hadm_id
FROM cohort_ckd c
JOIN admissions a USING(subject_id, hadm_id)
JOIN patients  p USING(subject_id)
WHERE (p.anchor_age + YEAR(a.admittime) - p.anchor_year) >= 18;

DROP TEMPORARY TABLE IF EXISTS cohort_ckd;
RENAME TABLE cohort_ckd_adult TO cohort_ckd;
*/

/* 2) 入院 0–24h 時窗 */
DROP TEMPORARY TABLE IF EXISTS first24_hosp;
CREATE TEMPORARY TABLE first24_hosp AS
SELECT a.subject_id, a.hadm_id,
       a.admittime AS t0,
       TIMESTAMPADD(HOUR, 24, a.admittime) AS t24
FROM admissions a
JOIN cohort_ckd  c USING(subject_id, hadm_id);

/* 3) 擷取 chartevents 中的 身高/體重/心跳（0–24h） */
DROP TEMPORARY TABLE IF EXISTS ce_24h;
CREATE TEMPORARY TABLE ce_24h AS
SELECT
  f.hadm_id,
  ce.stay_id,
  ce.charttime,
  UPPER(di.label) AS label,
  ce.valuenum,
  ce.valueuom
FROM chartevents ce
JOIN d_items di    ON di.itemid = ce.itemid
JOIN first24_hosp f ON f.hadm_id = ce.hadm_id
WHERE ce.valuenum IS NOT NULL
  AND ce.charttime >= f.t0
  AND ce.charttime <  f.t24
  AND UPPER(di.label) IN (
        'HEART RATE',           -- 心跳
        'HEIGHT',               -- 身高
        'WEIGHT',               -- 體重（泛稱）
        'ADMISSION WEIGHT',     -- 入院體重
        'DAILY WEIGHT'          -- 每日體重
      );

/* 4) 針對各變數做彙總
      - Heart Rate：avg/min/max
      - Height：取「最新值」（通常身高恆定；也可改用 MAX/MIN 檢查異常）
      - Weight：提供三種口徑（mean/min/max）＋「最早值」（近似入院體重）
*/

/* 4.1 心跳 HR（每 hadm 彙總） */
DROP TEMPORARY TABLE IF EXISTS hr_agg;
CREATE TEMPORARY TABLE hr_agg AS
SELECT
  hadm_id,
  AVG(valuenum) AS hr_mean_0_24h,
  MIN(valuenum) AS hr_min_0_24h,
  MAX(valuenum) AS hr_max_0_24h
FROM ce_24h
WHERE label = 'HEART RATE'
GROUP BY hadm_id;

/* 4.2 身高 Height：取 0–24h 內「最後一次」紀錄（若有多筆） */
DROP TEMPORARY TABLE IF EXISTS height_last;
CREATE TEMPORARY TABLE height_last AS
WITH ranked AS (
  SELECT
    hadm_id,
    charttime,
    valuenum AS height_val,
    ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime DESC) AS rn
  FROM ce_24h
  WHERE label = 'HEIGHT'
)
SELECT hadm_id, height_val AS height_cm
FROM ranked
WHERE rn = 1;

/* 4.3 體重 Weight：提供 mean/min/max + 「最早值」（近似 Admission Weight） */
DROP TEMPORARY TABLE IF EXISTS weight_agg;
CREATE TEMPORARY TABLE weight_agg AS
SELECT
  hadm_id,
  AVG(valuenum) AS weight_mean_0_24h,
  MIN(valuenum) AS weight_min_0_24h,
  MAX(valuenum) AS weight_max_0_24h
FROM ce_24h
WHERE label IN ('WEIGHT','ADMISSION WEIGHT','DAILY WEIGHT')
GROUP BY hadm_id;

DROP TEMPORARY TABLE IF EXISTS weight_first;
CREATE TEMPORARY TABLE weight_first AS
WITH ranked AS (
  SELECT
    hadm_id,
    charttime,
    valuenum AS weight_first_val,
    ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime ASC) AS rn
  FROM ce_24h
  WHERE label IN ('WEIGHT','ADMISSION WEIGHT','DAILY WEIGHT')
)
SELECT hadm_id, weight_first_val
FROM ranked
WHERE rn = 1;

/* 5) 產出最終特徵表（每 hadm 一列） */
DROP TABLE IF EXISTS ckd_vitals_24h;
CREATE TABLE ckd_vitals_24h AS
SELECT
  f.subject_id,
  f.hadm_id,
  f.t0  AS admittime,
  /* Height */
  h.height_cm,
  /* Weight */
  w.weight_first_val AS weight_first_0_24h,   -- 入院後最早體重（近似 admission weight）
  wa.weight_mean_0_24h,
  wa.weight_min_0_24h,
  wa.weight_max_0_24h,
  /* Heart Rate */
  hr.hr_mean_0_24h,
  hr.hr_min_0_24h,
  hr.hr_max_0_24h
FROM first24_hosp f
LEFT JOIN height_last h  ON h.hadm_id = f.hadm_id
LEFT JOIN weight_agg  wa ON wa.hadm_id = f.hadm_id
LEFT JOIN weight_first w ON w.hadm_id = f.hadm_id
LEFT JOIN hr_agg      hr ON hr.hadm_id = f.hadm_id;

/* 6)（可選）檢查缺值情況 */
SELECT
  SUM(height_cm            IS NULL) AS n_miss_height,
  SUM(weight_first_0_24h   IS NULL) AS n_miss_weight_first,
  SUM(weight_mean_0_24h    IS NULL) AS n_miss_weight_mean,
  SUM(hr_mean_0_24h        IS NULL) AS n_miss_hr_mean
FROM ckd_vitals_24h;

/* 7)（可選）加上死亡標籤，方便後續模型使用
ALTER TABLE ckd_vitals_24h
  ADD COLUMN in_hosp_mort TINYINT,
  ADD COLUMN mort_30d     TINYINT;

UPDATE ckd_vitals_24h x
JOIN admissions a USING(hadm_id)
LEFT JOIN patients  p USING(subject_id)
SET
  x.in_hosp_mort = CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END,
  x.mort_30d = CASE
      WHEN a.deathtime IS NOT NULL AND TIMESTAMPDIFF(DAY, a.admittime, a.deathtime) <= 30 THEN 1
      WHEN p.dod      IS NOT NULL AND TIMESTAMPDIFF(DAY, a.admittime, p.dod)      <= 30 THEN 1
      ELSE 0 END;
*/
