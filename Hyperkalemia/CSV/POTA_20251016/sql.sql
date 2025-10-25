select subject_id,hadm_id, admittime as t0, dischtime as t_end

from z_ckd_adm zca where potassium ='1'

--
--


select hadm_id as stay_id 


,charttime,lab_name,valuenum
from z_ckd_lab zcl 
where exists(
select *

from z_ckd_adm  where potassium ='1' and z_ckd_adm.hadm_id=zcl.hadm_id


)
---
--POTASSIUM 檢驗
---
  SELECT s.hadm_id,
                    MAX(CASE WHEN le.valuenum >= '5.5' THEN 1 ELSE 0 END) AS y_hk
            FROM z_ckd_adm s
            JOIN z_ckd_lab le ON le.hadm_id = s.hadm_id
         
            WHERE   le.charttime >= s.admittime
                    AND le.charttime <  DATE_ADD(s.admittime, INTERVAL 24 HOUR)
            -- AND UPPER(dl.label) LIKE '%POTASSIUM%'
            and le.itemid in('50971','52610','50822','52452')
            AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
            -- and labevents in(50833,50971,52610)
            
            
            GROUP BY s.hadm_id

 -- 全部檢驗
SELECT s.hadm_id, le.charttime, UPPER(dl.label) AS label, le.valuenum
FROM z_ckd_lab le
JOIN d_labitems dl ON dl.itemid = le.itemid
JOIN z_ckd_adm s  ON s.hadm_id = le.hadm_id 
WHERE le.valuenum IS NOT NULL
AND  le.charttime >= s.admittime AND le.charttime < s.dischtime
-- AND UPPER(dl.label) IN ('SODIUM', 'BICARBONATE', 'CHLORIDE', 'CREATININE', 'UREA NITROGEN', 'GLUCOSE')
-- AND le.itemid IN ('50920', '51006', '50912', '50862', '50811', '50808', '50970', '51007', '50907', '50998', '50905', '50983', '50971', '50852', '50931', '51002')


AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'


---
---低血鈉
  SELECT s.hadm_id,
                    MAX(CASE WHEN le.valuenum <= '135' THEN 1 ELSE 0 END) AS y_hk
            FROM z_ckd_adm s
            JOIN z_ckd_lab le ON le.hadm_id = s.hadm_id
         
            WHERE   le.charttime >= s.admittime
                    AND le.charttime <  DATE_ADD(s.admittime, INTERVAL 24 HOUR)
            -- AND UPPER(dl.label) LIKE '%POTASSIUM%'
            and le.itemid in ('50983')
            AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
            -- and labevents in(50833,50971,52610)
            
            
            GROUP BY s.hadm_id