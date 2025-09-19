from sqlalchemy import  text
def sqlalchmy_raw_sql(sqlslc_Session , SQL:str):
    """ 
    注意
    1.SELECT 回傳 dictionary,欄位轉大寫、NULL 轉 空白、回傳資料表欄位原始格式、NUMBER、DATE、NVARCHAR。
    2.也可使用UPDATE指令，但建議在外層直接使用execute(text(SQL))，效能比較好。
    3.日期型態 記得轉TO_CHAR。
    
        
    """
    try:
        
        my_data = sqlslc_Session.execute(text(SQL))
        if my_data.returns_rows == True :
            my_data = my_data.mappings().all()
            #欄位轉大寫、NULL 轉 空白
            #為了效能，不使用 str.rstrip() ，有需要了話自行在SQL 下 rstrip
            ROW_DATA = [{key.upper(): ('' if value is None else str(value)) for ( key, value) in row.items()} for row in my_data]
            
            return ROW_DATA
        else:
            rowcount=my_data.rowcount # UPDATE 的筆數
            return rowcount

    except Exception as Error:
        REC_OBJ = {
            "LOG_LEVEL": "999",
            "ERROR_MSG": str(Error),
            "FUNCTION_NAME": "sqlalchmy_raw_sql",
            "REC_DESC": "",
            "MSG": SQL
        }
        print(REC_OBJ)
        raise