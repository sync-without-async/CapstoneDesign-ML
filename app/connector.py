import mysql.connector as db
import polars as pl
import logging
import json 

logging.basicConfig(level=logging.INFO, format='[CONNECTOR_MODULE]%(asctime)s %(levelname)s %(message)s')

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
    return result

def database_select_using_pk(
        table: pl.DataFrame = None,
        pk: int = None,
        verbose: bool = False
    ) -> pl.DataFrame:
    """Select a row using primary key. This function will return the result as a polars dataframe.
    Using polars dataframe, you can easily convert the result to a list or a numpy array. Also, you can easily convert the result to a json object.
    But, the pk column should be the first column of the table. If it is not, you should change the column order.
    
    Example:
        >>> connector, cursor = database_connector(database_secret_path="secret_key.json")
        >>> table_name = "video"
        >>> query = f"SELECT * FROM {table_name};"
        >>> result = database_query(connector, cursor, query, verbose=False)
        >>> if result.shape[0] == 0:    return {"error": "No video found."}
        >>> result = database_select_using_pk(result, pk=1, verbose=True)
        >>> result = result.to_numpy().tolist()[0]

    Args:
        table (pl.DataFrame): The table that is returned from database_query function.
        pk (str or int): The primary key of the row that you want to select.
        verbose (bool, optional): If True, print the result. Defaults to False.

    Returns:
        pl.DataFrame: The result of the query."""
    if table is None: raise ValueError(f"table argument is required. Excepted: pl.DataFrame, but got {table}")
    if pk is None: raise ValueError(f"pk argument is required. Excepted: str or int, but got {pk}")

    if verbose: logging.info(f"Selecting row using primary key...")
    result = table.filter(
        table[:, 0] == pk
    )
    if verbose: logging.info(f"Selected row using primary key.")

    if len(result) == 0:
        logging.error(f"No row found using primary key: {pk}")
        return None

    return result

def insert_summary_database(
        connector: db.MySQLConnection = None, 
        cursor: db.MySQLConnection.cursor = None, 
        target_table_name: str = None,
        target_database: str = None,
        target_columns: str = None,
        target_values: str = None,
        target_room_number: int = None,
        verbose: bool = False
    ) -> bool:
    """THIS FUNCTION ONLY FOR INSERTING SUMMARY TO DATABASE. DO NOT USE THIS FUNCTION FOR OTHER PURPOSES.
    Insert summary to database. This function will return True if the summary is inserted successfully. Otherwise, it will return False.
    Before using this function, you should connect to the database using database_connector function.
    Connector function will return the connector and the cursor. You should pass the connector and the cursor to this function.

    Example:
        >>> connector, cursor = database_connector(database_secret_path="secret_key.json")
        >>> table_name = "video"
        >>> query = f"SELECT * FROM {table_name};"
        >>> result = database_query(connector, cursor, query, verbose=False)
        >>> if result.shape[0] == 0:    return {"error": "No video found."}
        >>> result = database_select_using_pk(result, pk=1, verbose=True)
        >>> result = result.to_numpy().tolist()[0]

    Args:
        connector (db.MySQLConnection): The connector that is returned from database_connector function.
        cursor (db.cursor.MySQLCursor): The cursor that is returned from database_connector function.
        target_table_name (str): The table name that you want to insert the summary.
        target_database (str): The database name that you want to insert the summary.
        target_columns (str): The column name that you want to insert the summary.
        target_values (str): The summary that you want to insert.
        target_room_number (int): The room number of the summary that you want to insert.
        verbose (bool, optional): If True, print the result. Defaults to False.

    Returns:
        bool: True if the summary is inserted successfully. Otherwise, False."""
    if connector is None: raise ValueError(f"connector argument is required. Excepted: db.MySQLConnection, but got {connector}")
    if cursor is None: raise ValueError(f"cursor argument is required. Excepted: db.MySQLConnection.cursor, but got {cursor}")
    if target_values is None:  raise ValueError(f"target_values argument is required. Excepted: list, but got {target_values}")
    if target_room_number is None:  raise ValueError(f"target_room_number argument is required. Excepted: int, but got {target_room_number}")
    if target_database is None:
        target_database = "rehabdb"
        logging.info(f"target_database argument is not provided. Using default value: {target_database}")

    if target_table_name is None: 
        target_table_name = "audio"
        logging.info(f"target_table_name argument is not provided. Using default value: {target_table_name}")

    if target_columns is None: 
        target_columns = "summary"
        logging.info(f"target_columns argument is not provided. Using default value: {target_columns}")


    query_template = f"UPDATE {target_table_name} SET {target_columns} = %s WHERE ano = %s;"
    if verbose: logging.info(f"Executing query: {query_template}")

    if verbose: logging.info(f"Inserting summary to database...")
    try:
        cursor.execute(query_template, (target_values, target_room_number))
        connector.commit()
        if verbose: logging.info(f"Summary inserted successfully.")
        return True

    except Exception as e:
        logging.error(f"Error occured while inserting summary to database. Error: {e}")
        return False