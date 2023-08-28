import mysql.connector as db
import json

from tqdm import tqdm

with open("../secret_key.json", "r") as f:
    secret_key = json.load(f)["database"]

DATABASE_USER = secret_key["user"]
DATABASE_PASSWORD = secret_key["password"]
DATABASE_HOST = secret_key["host"]
DATABASE_PORT = secret_key["port"]
DATABASE = secret_key["database"]

connector = db.connect(
    user=DATABASE_USER,
    password=DATABASE_PASSWORD,
    host=DATABASE_HOST,
    port=DATABASE_PORT,
    database=DATABASE
)
cursor = connector.cursor()
table_name = "program_video"

query = f"SELECT * FROM {table_name};"
cursor.execute(query)
result = cursor.fetchall()
print(result)
