import mysql.connector as db
import json 
import time

from datetime import datetime
from tqdm import tqdm

def database_connector(database_secret_path: str = "database_secret.json") -> tuple:
    with open(database_secret_path, "r") as f:
        database_secret = json.load(f)['database']

    connector = db.connect(
        user=database_secret["user"],
        password=database_secret["password"],
        host=database_secret["host"],
        port=database_secret["port"],
        database=database_secret["database"]
    )
    cursor = connector.cursor()

    return connector, cursor

def database_query(
        connector: db.MySQLConnection,
        cursor: db.cursor.MySQLCursor,
        query: str,
        verbose: bool = False
    ):
    cursor.execute(query)
    result = cursor.fetchall()
    if verbose:
        print(result)
    return result
