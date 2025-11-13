from psycopg2 import pool
from psycopg2.extensions import connection
from contextlib import contextmanager
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# Global variable to hold the connection pool instance, making it a singleton.
_connection_pool = None

def create_connection_pool():
    """
    Create a PostgreSQL connection pool.
    
    Args:
        min_connections (int): Minimum number of connections in the pool
        max_connections (int): Maximum number of connections in the pool
        **kwargs: Connection parameters (dbname, user, password, host, port)
        
    Returns:
        SimpleConnectionPool: PostgreSQL connection pool
    """
    global _connection_pool
    # Return the existing pool if it's already created.
    if _connection_pool:
        return _connection_pool

    try:
        connection_details = {
            'dbname': 'invoice_management',
            "user": os.environ.get('POSTGRES_USER'),
            'password': os.environ.get('POSTGRES_PASSWORD'),
            'host': os.environ.get('POSTGRES_HOST'),
            'port': '4322'
        }
        _connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=100,
            dbname=connection_details.get('dbname'),
            user=connection_details.get('user'),
            password=connection_details.get('password'),
            host=connection_details.get('host', 'localhost'),
            port=connection_details.get('port') # Use port from connection_details for consistency
        )
        print("Connection pool created successfully")
        return _connection_pool
    except Exception as error:
        print(f"Error creating connection pool: {error}")
        raise
@contextmanager
def get_cursor(db:pool.SimpleConnectionPool):
    # Use the correct public type hint for the connection object.
    con: connection = db.getconn()
    try:
        cursor = con.cursor()
        yield cursor
        con.commit()
    except Exception as error:
        print("Exception occurred in get_cursor, rolling back transaction.", error)
        # Rollback the transaction on any exception within the 'with' block.
        con.rollback()
        raise

async def execute_query(query, params=None):
    """
    Execute a SQL query using a connection from the pool.
    
    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters (optional)
        
    Returns:
        list: Query results
    """
    
    try:
        # Get the singleton connection pool instance.
        conn_pool = create_connection_pool()
        with get_cursor(conn_pool) as cursor:
            print(f"Executing query: {query} with params: {params}")
            cursor.execute(query, params)
            # Check if the cursor has results to fetch (e.g., from a RETURNING clause)
            if cursor.description:
                row = cursor.fetchone()
                if row:
                    return {"id": row[0], "result": row}
            # Return None if no result was returned (e.g., for a simple UPDATE/INSERT)
            return {"id": None, "result": None}
    except Exception as error:
        print(f"Error executing query: {error}")
        return None
async def fetch_query_results(query, params=None):
    """
    Execute a SELECT query and fetch all results using a connection from the pool.
    
    Args:
        query (str): SQL SELECT query to execute
        params (tuple): Query parameters (optional)
        
    Returns:
        list: Query results as a list of tuples
    """
    try:
        # Get the singleton connection pool instance.
        conn_pool = create_connection_pool()
        with get_cursor(conn_pool) as cursor:
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            # conn_pool.putconn(cursor.connection)
            return results
    except Exception as error:
        print(f"Error fetching query results: {error}")
        raise