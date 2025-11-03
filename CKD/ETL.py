import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine, text
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict
#from main import ENG
ENG = create_engine("mysql+pymysql://test:test@10.2.163.201:3306/mimic3_1")# 連接設定

def adm_ckd():#轉檔
    try:

        sql_map={   #z_k_adm:高血鉀住院
                    "z_k_adm":"""
                            exists(
                            select
                                1
                            from
                                `diagnoses_icd`
                            where
                                ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                                    and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                                        and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5,6,7))
                                            and (((`diagnoses_icd`.`icd_version` = 10)
                                                and (replace(`diagnoses_icd`.`icd_code`, '.', '') like 'E875%'))
                                                or ((`diagnoses_icd`.`icd_version` = 9)
                                                    and (replace(`diagnoses_icd`.`icd_code`, '.', '') like 'E875%')))))


                            """,
                    # z_ckd_adm:慢性腎臟病住院
                    "z_ckd_adm":"""   exists(
                                        select
                                            1
                                        from
                                            `diagnoses_icd`
                                        where
                                            ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                                                and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                                                    and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5,6,7))
                                                        and (((`diagnoses_icd`.`icd_version` = 10)
                                                            and (replace(`diagnoses_icd`.`icd_code`, '.', '') like 'N18%'))
                                                            or ((`diagnoses_icd`.`icd_version` = 9)
                                                                and (replace(`diagnoses_icd`.`icd_code`, '.', '') like '585%')))))

                                                            """,
                    #z_sodium_adm:低血鈉住院
                    "z_sodium_adm":"""     exists(
                    select
                        1
                    from
                        `diagnoses_icd`
                    where
                        ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                            and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                                and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5,6,7))
                                    and (((`diagnoses_icd`.`icd_version` = 10)
                                        and (replace(`diagnoses_icd`.`icd_code`, '.', '') like 'E871%'))
                                        or ((`diagnoses_icd`.`icd_version` = 9)
                                            and (replace(`diagnoses_icd`.`icd_code`, '.', '') like '2762%')))))
                                              """,
                     #z_aki_adm:AKI住院
                    "z_aki_adm":"""     exists(
                            select
                                1
                            from
                                `diagnoses_icd`
                            where
                                ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                                    and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                                        and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5,6,7))
                                            and (((`diagnoses_icd`.`icd_version` = 10)
                                                and substr(`diagnoses_icd`.`icd_code`,1,4) in('N170','N171','N172','N178') )
                                               )))
                                              """


        }
        
     
        with ENG.connect() as connection: 
            for table_name, condition in sql_map.items():
                print(f"處理表格:{table_name}")
                connection.execute(text("DELETE FROM "+table_name))
                
                SQL = f"""

                    INSERT INTO mimic3_1.{table_name}
                    (subject_id, hadm_id, admittime, dischtime,deathtime, admission_type, admit_provider_id, admission_location
                    , insurance, marital_status, race, edregtime, edouttime, hospital_expire_flag, icd_1, icd_2, icd_3
                , age_years, gender, aki, potassium, low_sodium,body_weight,body_height)

                
                    
                        
                select
                `a`.`subject_id` as `subject_id`,
                `a`.`hadm_id` as `hadm_id`,
                `a`.`admittime` as `admittime`,
                `a`.`dischtime` as `dischtime`,
                `a`.`deathtime` as `deathtime`,
                `a`.`admission_type` as `admission_type`,
                `a`.`admit_provider_id` as `admit_provider_id`,
                `a`.`admission_location` as `admission_location`,
                -- `a`.`discharge_location` as `discharge_location`,
                `a`.`insurance` as `insurance`,
                -- `a`.`language` as `language`,
                `a`.`marital_status` as `marital_status`,
                `a`.`race` as `race`,
                `a`.`edregtime` as `edregtime`,
                `a`.`edouttime` as `edouttime`,
                `a`.`hospital_expire_flag` as `hospital_expire_flag`,
                (
                select
                    (case
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^250|^E10|^E11|^E13') then 'Diabetes'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70') then 'Cardiovascular'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^2777|^272|^E88.81|^E78|^278|^E66') then 'Metabolic'
                        else `diagnoses_icd`.`icd_code`
                    end)
                from
                    `diagnoses_icd`
                where
                    ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                        and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                            and (`diagnoses_icd`.`seq_num` = 1))) as `ICD_1`,
                (
                select
                    (case
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^250|^E10|^E11|^E13') then 'Diabetes'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70') then 'Cardiovascular'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^2777|^272|^E88.81|^E78|^278|^E66') then 'Metabolic'
                        else `diagnoses_icd`.`icd_code`
                    end)
                from
                    `diagnoses_icd`
                where
                    ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                        and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                            and (`diagnoses_icd`.`seq_num` = 2))) as `ICD_2`,
                (
                select
                    (case
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^250|^E10|^E11|^E13') then 'Diabetes'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^401|^405|^I10|^I15|^410|^411|^414|^428|^4292|^440|^4439|^I21|^I25|^I50|^I70') then 'Cardiovascular'
                        when regexp_like(`diagnoses_icd`.`icd_code`, '^2777|^272|^E88.81|^E78|^278|^E66') then 'Metabolic'
                        else `diagnoses_icd`.`icd_code`
                    end)
                from
                    `diagnoses_icd`
                where
                    ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                        and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                            and (`diagnoses_icd`.`seq_num` = 3))) as `ICD_3`,
                ((`patients`.`anchor_age` + year(`a`.`admittime`)) - `patients`.`anchor_year`) as `age_years`,
                `patients`.`gender` as `gender`,
                (
                select
                    (case
                        when (count(0) > 0) then '1'
                        else 0
                    end)
                from
                    `diagnoses_icd`
                where
                    ((`diagnoses_icd`.`subject_id` = `a`.`subject_id`)
                    and (`diagnoses_icd`.`seq_num` in (1, 2, 3, 4, 5))
                        and (`diagnoses_icd`.`hadm_id` = `a`.`hadm_id`)
                            and ((`diagnoses_icd`.`icd_code` like 'N170%')
                                or (`diagnoses_icd`.`icd_code` like 'N179')
                                    or (`diagnoses_icd`.`icd_code` like '5849')))) as `AKI`,
                (
                select
                    (case
                        when (count(0) > 0) then '1'
                        else 0
                    end)
                from
                    `labevents` `lab`
                where
                    ((`lab`.`subject_id` = `a`.`subject_id`)
                        and (`lab`.`hadm_id` = `a`.`hadm_id`)
                        AND lab.charttime >= a.admittime
                        AND lab.charttime <  DATE_ADD(a.admittime, INTERVAL 24 HOUR)
                            and (`lab`.`itemid` in ('50971','52610','50822','52452'))
                                and regexp_like(`lab`.`valuenum`, '^[0-9]+(\\.[0-9]+)?$')
                                    and (`lab`.`valuenum` >= '5.5'))) as `POTASSIUM`,
                (
                select
                    (case
                        when (count(0) > 0) then '1'
                        else 0
                    end)
                from
                    `labevents` `lab`
                where
                    ((`lab`.`subject_id` = `a`.`subject_id`)
                        and (`lab`.`hadm_id` = `a`.`hadm_id`)
                        AND lab.charttime >= a.admittime
                        AND lab.charttime <  DATE_ADD(a.admittime, INTERVAL 24 HOUR)
                            and `lab`.`itemid` in ('50983','52623')
                                and regexp_like(`lab`.`valuenum`, '^[0-9]+(\\.[0-9]+)?$')
                                    and (`lab`.`valuenum` <= '135'))) as `low_sodium`

                ,(select
                    ce.valuenum 
                        
                    FROM mimic3_1.chartevents ce
                    -- JOIN mimic3_1.d_items di ON di.itemid = ce.itemid
                    where ce.subject_id=a.subject_id and ce.hadm_id =a.hadm_id  and  ce.itemid in('224639','226512')
                        AND ce.valuenum IS NOT NULL
                        -- 合理範圍過濾，避免錄入錯誤
                    
                        LIMIT 1
                        
                        ) as body_weight,
                        (
                        select
                    ce.valuenum 
                        
                    FROM mimic3_1.chartevents ce
                    -- JOIN mimic3_1.d_items di ON di.itemid = ce.itemid
                    where ce.subject_id=a.subject_id and ce.hadm_id =a.hadm_id  and  ce.itemid in('226730')
                        AND ce.valuenum IS NOT NULL
                        -- 合理範圍過濾，避免錄入錯誤
                    
                        LIMIT 1
                        
                        )as body_height
                from
                    `admissions` `a`
                join `patients` on
                    ((`patients`.`subject_id` = `a`.`subject_id` AND`patients`.`anchor_year_group` in ('2020 - 2022', '2017 - 2019') ))
                where 
                {condition}
                
                """
                connection.execute(text(SQL))
                print(f"{table_name} 轉檔完成")
                SQL=f"""  select a.subject_id,a.hadm_id from {table_name}  a where exists(
                        select * from labevents b where  b.subject_id= a.subject_id 
                        and `b`.`hadm_id` = `a`.`hadm_id` AND b.itemid = 50912 
                        AND b.valuenum IS NOT null
                        )
                """
                rtn=connection.execute(text(SQL))
                results2 = rtn.mappings().all()

                SQL=f"""-- 更新BMI
                update {table_name} set 
                bmi = ROUND(body_weight / POW(body_height / 100, 2), 2)
                WHERE body_height IS NOT NULL AND body_weight IS NOT null
                """

                connection.execute(text(SQL))

                for I in results2:
                    SQL=f"""

                    WITH
                    egfr  AS (
                    SELECT
                        a.subject_id, a.hadm_id ,
                        ( select b.valuenum 
                    
                    from labevents b    
                    where b.subject_id= a.subject_id and `b`.`hadm_id` = `a`.`hadm_id` AND itemid = 50912 AND b.valuenum IS NOT null
                    ORDER by b.charttime DESC LIMIT 1
                    ) as lab
                        
                        ,a.gender, a.age_years,
                        /* κ, α 依性別 */
                        CASE WHEN a.gender='F' THEN 0.7 ELSE 0.9 END AS kappa,
                        CASE WHEN a.gender='F' THEN -0.241 ELSE -0.302 END AS alpha
                    FROM {table_name} a  where a.subject_id='{I["subject_id"]}' AND hadm_id='{I["hadm_id"]}'
                    )
                    
                    SELECT
                        subject_id, hadm_id,  lab, gender, age_years,
                        ROUND(
                        142
                        * POW(LEAST(lab / kappa, 1), alpha)
                        * POW(GREATEST(lab / kappa, 1), -1.200)
                        * POW(0.9938, age_years)
                        * CASE WHEN gender='F' THEN 1.012 ELSE 1.000 END
                        ,1) AS egfr_data
                    FROM egfr

                    """
                    rtn=connection.execute(text(SQL))
                    egfr_data = rtn.mappings().all()

                    SQL=f"""

                    WITH
                    egfr2  AS (
                    SELECT
                        a.subject_id, a.hadm_id ,
                        ( select b.valuenum 
                    
                    from labevents b    
                    where b.subject_id= a.subject_id and `b`.`hadm_id` = `a`.`hadm_id` AND itemid = 50912 AND b.valuenum IS NOT null
                    ORDER by b.charttime ASC LIMIT 1
                    ) as lab
                        
                        ,a.gender, a.age_years,
                        /* κ, α 依性別 */
                        CASE WHEN a.gender='F' THEN 0.7 ELSE 0.9 END AS kappa,
                        CASE WHEN a.gender='F' THEN -0.241 ELSE -0.302 END AS alpha
                    FROM {table_name} a  where a.subject_id='{I["subject_id"]}' AND hadm_id='{I["hadm_id"]}'
                    )
                    
                    SELECT
                        subject_id, hadm_id,  lab, gender, age_years,
                        ROUND(
                        142
                        * POW(LEAST(lab / kappa, 1), alpha)
                        * POW(GREATEST(lab / kappa, 1), -1.200)
                        * POW(0.9938, age_years)
                        * CASE WHEN gender='F' THEN 1.012 ELSE 1.000 END
                        ,1) AS admit_egfr
                    FROM egfr2

                    """
                    rtn=connection.execute(text(SQL))
                    admit_egfr = rtn.mappings().all()


                    if egfr_data.__len__():

                        SQL=f"""
                        UPDATE {table_name} SET
                        EGFR='{egfr_data[0]["egfr_data"]}'
                        ,admit_egfr='{admit_egfr[0]["admit_egfr"]}'
                        WHERE subject_id='{I["subject_id"]}' AND hadm_id='{I["hadm_id"]}'
                        """
                        connection.execute(text(SQL))

                        print(I)
            connection.commit()
            


            #print(table_name+",完成")
        print("轉檔完成")
        return 0
    except Exception as Error:
        print("CSV_TO_DB 錯誤!!!!!!! \n"+str(Error))


def CKD_LAB():#轉檔
    try:
        
        SQL_MAP={  "z_ckd_adm":"z_ckd_lab","z_k_adm":"z_k_lab" ,"z_sodium_adm":"z_sodium_lab","z_aki_adm":"z_aki_lab"   }
        
        for adm_table,lab_table in SQL_MAP.items():
            print(f"處理表格:{lab_table}")
            with ENG.connect() as connection: 
                connection.execute(text("DELETE FROM "+lab_table))

            SQL = f"""
            SELECT  
            a.itemid,
            c.label AS lab_name
            ,a.valuenum,a.subject_id,a.hadm_id,a.charttime,c.fluid,c.category
            ,ref_range_lower ,ref_range_upper ,flag ,comments
            FROM labevents a
            JOIN {adm_table} b ON (b.subject_id = a.subject_id  AND b.hadm_id    = a.hadm_id)
            join d_labitems   c on (c.itemid = a.itemid) 
            WHERE a.itemid IN ('50920','51006','50912','50862','50811','50808','50970','51007','50907','50904','51102'
                                ,'50998','50905','50852','50931','51002','51992','51069','52703'
                                ,'50971','52610','50822','52452' -- Potassium 高血鉀
                                ,'50983','52623','52455' -- 低血鈉
                                
                                )
                                
                AND a.valuenum REGEXP '^[0-9]+(\\.[0-9]+)?$'
            
            """

            reader=pd.read_sql(text(SQL), ENG,chunksize=100000  )
                
            for chunk in tqdm(reader):
                
                chunk.to_sql(lab_table, con=ENG, if_exists='append', index=False, method='multi')
            SQL = f"""
            

            SELECT ce.itemid,ce.subject_id,ce.hadm_id, ce.charttime
            , case ce.itemid when 220179 THEN'SYSTOLIC(SBP)'
            when 220180 THEN'DIASTOLIC(DBP)' else UPPER(di.label) END
            as lab_name
            
            , ce.valuenum
            FROM chartevents ce
            JOIN d_items di ON di.itemid = ce.itemid
            JOIN {adm_table}   s ON s.hadm_id = ce.hadm_id -- and s.AKI ='1'
            WHERE ce.charttime >= s.admittime  AND ce.charttime < s.dischtime 
            AND ce.itemid IN ('220045','220179','220180','220210','223762')
            AND ce.valuenum IS NOT NULL
            AND ce.valuenum REGEXP '^[0-9]+$' AND CAST(ce.valuenum AS UNSIGNED) > 1
            
            
            """

            reader=pd.read_sql(text(SQL), ENG,chunksize=100000  )
                
            for chunk in tqdm(reader):
                
                chunk.to_sql(lab_table, con=ENG, if_exists='append', index=False, method='multi')
                    
        
        
        
        print("轉檔完成")
        return 0
    except Exception as Error:
        print("CSV_TO_DB 錯誤!!!!!!! \n"+str(Error))

def CKD_TO_STAT_EXCEL():#統計轉EXCEL
    try:
        SQL_MAP={  "z_ckd_adm":"z_ckd_lab","z_k_adm":"z_k_lab" ,"z_sodium_adm":"z_sodium_lab" ,"z_aki_adm":"z_aki_lab"   }
        
        # 根據表格名稱設定工作表名稱
        sheet_names = {
            "z_ckd_adm": "慢性腎臟病統計",
            "z_k_adm": "高血鉀統計", 
            "z_sodium_adm": "低血鈉統計"
        }
        
        # 創建 Excel 檔案路徑
        excel_file = "EXCEL/統計資料匯總.xlsx"
        
        # 收集所有統計資料
        all_data = {}
        
        for adm_table,lab_table in SQL_MAP.items():
            SQL = f"""
            
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
            FROM {lab_table} where itemid=a.itemid
            ) AS ordered
            WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
                round(STDDEV(a.valuenum) ,2)                  AS 標準差
            FROM
                {lab_table} a
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
            FROM {adm_table} where {adm_table}.admit_egfr is not null
            ) AS ordered
            WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
                    round(STDDEV(a.admit_egfr) ,2)                  AS 標準差
                from {adm_table} a
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
            FROM {adm_table} where {adm_table}.EGFR is not null
            ) AS ordered
            WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
                    round(STDDEV(a.EGFR) ,2)                  AS 標準差
                from {adm_table} a
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
            FROM {adm_table} where {adm_table}.age_years is not null
            ) AS ordered
            WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
                    round(STDDEV(a.age_years) ,2)                  AS 標準差
                from {adm_table} a
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
            FROM {adm_table} where {adm_table}.bmi is not null
            ) AS ordered
            WHERE row_num IN (FLOOR((total_rows + 1) / 2), CEIL((total_rows + 1) / 2))) as 中位數,
                    round(STDDEV(a.bmi) ,2)                  AS 標準差
                from {adm_table} a
                where a.bmi is not null
        

            """
            
            print(f"處理表格:{adm_table}")
            stat_data = pd.read_sql(text(SQL), ENG)
            sheet_name = sheet_names.get(adm_table, adm_table)
            all_data[sheet_name] = stat_data
            
        # 一次性寫入所有工作表到同一個 Excel 檔案
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for sheet_name, data in all_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"統計資料已儲存到工作表: {sheet_name}")
        except ImportError:
            # 如果沒有 openpyxl，使用 xlsxwriter
            print("警告: 未安裝 openpyxl，使用 xlsxwriter 引擎")
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                for sheet_name, data in all_data.items():
                    data.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"統計資料已儲存到工作表: {sheet_name}")
        
        print(f"所有統計資料已匯總到 {excel_file}")
        return 0
    except Exception as Error:
        print("CKD_TO_STAT_EXCEL 錯誤!!!!!!! \n"+str(Error))

# def split_per_label_groupwise(df, label_col="label", group_col="hadm_id",
#                               train_p=0.8, val_p=0.1, test_p=0.1, random_state=42):
#     """針對每個 label 分別做 group-aware 的 8:1:1 切分；最後再合併。"""
#     assert abs(train_p + val_p + test_p - 1.0) < 1e-6
#     parts = defaultdict(list)

#     # 逐個 label 切分
#     for lab, d in df.groupby(label_col, sort=False):
#         groups = d[group_col].dropna().astype(str)
#         uniq_groups = groups.drop_duplicates()

#         if len(uniq_groups) <= 3:
#             # 群組太少：退而求其次用列為單位切分（仍維持比例）
#             n = len(d)
#             train_end = int(n * train_p)
#             val_end = train_end + int(n * val_p)
#             d_shuf = d.sample(frac=1, random_state=random_state)
#             parts['train'].append(d_shuf.iloc[:train_end])
#             parts['val'].append(d_shuf.iloc[train_end:val_end])
#             parts['test'].append(d_shuf.iloc[val_end:])
#             continue

#         # 第一步：切出 80% 的群組做 train
#         gss1 = GroupShuffleSplit(n_splits=1, train_size=train_p, random_state=random_state)
#         tr_idx, rest_idx = next(gss1.split(d, groups=groups))
#         d_train = d.iloc[tr_idx]
#         d_rest  = d.iloc[rest_idx]
#         rest_groups = d_rest[group_col].astype(str)

#         # 第二步：在剩下的 20% 中等比分成 10% val、10% test
#         gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
#         val_idx, test_idx = next(gss2.split(d_rest, groups=rest_groups))
#         d_val  = d_rest.iloc[val_idx]
#         d_test = d_rest.iloc[test_idx]

#         parts['train'].append(d_train)
#         parts['val'].append(d_val)
#         parts['test'].append(d_test)

#     # 合併各 label 的切分
#     df_train = pd.concat(parts['train'], ignore_index=True)
#     df_val   = pd.concat(parts['val'],   ignore_index=True)
#     df_test  = pd.concat(parts['test'],  ignore_index=True)

#     # 檢查比例（整體與各 label）
#     def _check(name, d):
#         by_lab = d[label_col].value_counts().sort_index()
#         return name, len(d), (len(d)/len(df)).round(3), by_lab.to_dict()

#     print(_check("train", df_train))
#     print(_check("val",   df_val))
#     print(_check("test",  df_test))

#     return df_train, df_val, df_test
# df = pd.read_csv("Hyperkalemia/CSV/all_lab.csv")
# df_train, df_val, df_test = split_per_label_groupwise(df)

# # 輸出檔案（可選）
# out_dir = f"Hyperkalemia/CSV/"
# df_train.to_csv(f"{out_dir}/labevents_train.csv", index=False)
# df_val.to_csv(  f"{out_dir}/labevents_val.csv",   index=False)
# df_test.to_csv( f"{out_dir}/labevents_test.csv",  index=False)

if __name__ == "__main__":
    #CKD_TO_CSV()
    
    adm_ckd()
    CKD_LAB()
    CKD_TO_STAT_EXCEL()

