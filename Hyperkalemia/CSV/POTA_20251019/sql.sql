                                  -- 檢驗    
         SELECT a.hadm_id,
                    MAX(CASE WHEN le.valuenum >= '5.5' THEN 1 ELSE 0 END) AS y_hk
            FROM z_ckd_adm a
            JOIN z_ckd_lab le ON le.hadm_id = a.hadm_id
         
            WHERE   le.charttime >= a.admittime
                    AND le.charttime <  DATE_ADD(a.admittime, INTERVAL 24 HOUR)
            -- AND UPPER(dl.label) LIKE '%POTASSIUM%'
            and le.itemid in('50971','52610','50822','52452')
            AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
             and not EXISTS( -- 排除AKI診斷
                select
                        1
                    from
                        `diagnoses_icd`
                    where
                        ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                            and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                                and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5,6,7))
                                    and (((`diagnoses_icd`.`icd_version` = 10)
                                        and (replace(`diagnoses_icd`.`icd_code`, '.', '') like 'N17%'))
                                        or ((`diagnoses_icd`.`icd_version` = 9)
                                            and (replace(`diagnoses_icd`.`icd_code`, '.', '') like '584%')))))
            
                                            
            -- and labevents in(50833,50971,52610)
            
            
            GROUP BY a.hadm_id
            
            
            
            
            
        
                                            
                                            
--發生高血鉀 前的檢驗


SELECT s.hadm_id, le.charttime, UPPER(dl.label) AS label, le.valuenum
FROM z_ckd_lab le
JOIN d_labitems dl ON dl.itemid = le.itemid
JOIN z_ckd_adm s  ON s.hadm_id = le.hadm_id 
WHERE le.valuenum IS NOT NULL
AND  le.charttime >= s.admittime AND le.charttime < s.dischtime
-- AND UPPER(dl.label) IN ('SODIUM', 'BICARBONATE', 'CHLORIDE', 'CREATININE', 'UREA NITROGEN', 'GLUCOSE')
-- AND le.itemid IN ('50920', '51006', '50912', '50862', '50811', '50808', '50970', '51007', '50907', '50998', '50905', '50983', '50971', '50852', '50931', '51002')

AND le.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
 and s.aki='0'
 and EXISTS(
 
 select * from z_ckd_lab lab2 where lab2.hadm_id =le.hadm_id 
 and lab2.itemid in('50971','52610','50822','52452') and lab2.valuenum >= '5.5' 
 and le.charttime <lab2.charttime
 )
 
 
 
 