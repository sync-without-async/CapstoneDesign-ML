import mysql.connector as db

from tqdm import tqdm

DATABASE_HOST = '121.155.78.249'
DATABASE_PORT = 3306
DATABASE_USER = 'hallymuser'
DATABASE_PASSWORD = 'hallym@udontknowme12'
DATABASE = 'rehabdb'

connector = db.connect(
    user=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT,
    database=DATABASE
)
cursor = connector.cursor()

query = "SELECT NOW();"
cursor.execute(query)
result = cursor.fetchall()
print(result)

