import psycopg2 as psycopg2
import pandas as pd

try:
    connection = psycopg2.connect(
        # 외부 서버는
        host='localhost',
        database='postgres',
        # 외부는
        # user = 'ryu'
        user='postgres',
        password='1234')
    print("connect success")
except:
    print("I am unable to connect to the database")

# connection.cursor() : 데이터베이스 와 통신할 수 있는 cursor 생성
cur = connection.cursor()

# select * from 레시피 명
cur.execute("select * from rcp_search_count;")

table_nm = pd.DataFrame(cur)
table_nm.columns = [desc[0] for desc in cur.description]  # 컬럼명 가져오고 싶을때 사용
