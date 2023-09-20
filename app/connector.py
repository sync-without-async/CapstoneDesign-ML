import mysql.connector as db
import polars as pl
import numpy as np
import json 

def database_connector(database_secret_path: str = "database_secret.json") -> tuple:
    """Connect to the database and return the connector and the cursor.
    
    Args:
        database_secret_path (str, optional): The path of the database secret file. Defaults to "database_secret.json". 
    
    Returns:
        tuple: The connector and the cursor."""
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

def database_query(connector: db.MySQLConnection, cursor: db.cursor.MySQLCursor, query: str, verbose: bool = False) -> pl.DataFrame:
    """Query the database and return the result as a polars dataframe.
    Before using this function, you should connect to the database using database_connector function.
    Connector function will return the connector and the cursor. You should pass the connector and the cursor to this function.

    Example:
        >>> connector, cursor = database_connector(database_secret_path="secret_key.json")
        >>> table_name = "video"
        >>> query = f"SELECT * FROM {table_name};"
        >>> result = database_query(connector, cursor, query, verbose=False)
        >>> if result.shape[0] == 0:    return {"error": "No video found."}

    Args:
        connector (db.MySQLConnection): The connector that is returned from database_connector function.
        cursor (db.cursor.MySQLCursor): The cursor that is returned from database_connector function.
        query (str): The query that you want to execute.
        verbose (bool, optional): If True, print the result. Defaults to False.

    Returns:
        pl.DataFrame: The result of the query."""
    cursor.execute(query)
    result = cursor.fetchall()
    if verbose:
        print(result)
    
    result = np.array(result)
    return result
