import mysql.connector as db

from tqdm import tqdm

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

