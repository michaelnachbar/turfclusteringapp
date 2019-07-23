
import json

import MySQLdb
import pymysql
import sqlalchemy

import pandas as pd

from django.conf import settings


def new_connection():
    "Return a new database connection."
    from django.db import connection
    connection = connection.copy()
    if not connection.connection:
        connection.connect()
    
    return connection.connection

def new_sqlalchemy_connection():
    "Return a new sqlalchemy connection."
    from django.conf import settings
    
    default_sqlalchemy_dialects = 'mysql oracle postgresql sqlite mssql'.split(' ')
    
    driver = settings.DATABASES['default'].get('DRIVER', None)
    dialect = settings.DATABASES['default'].get('DIALECT', None)
    if not dialect:
        engine = settings.DATABASES['default'].get('ENGINE', None)
        if not engine:
            raise RuntimeError("Must specify database ENGINE in settings or DIALECT.")
        engine = engine.split('.')[-1]
        for dialect in default_sqlalchemy_dialects:
            if engine.startswith(dialect):
                break
        else:
            dialect = None
            raise RuntimeError("Set DATABASES['default'].DIALECT in settings, you're probably using a custom engine ({}).".format(engine))
    schema = '+'.join(filter(None, (dialect, driver)))
    
    return sqlalchemy.create_engine(schema+'://', creator=new_connection)

def execute_mysql(statement):
    "Helper to execute SQL with the default database connection."
    conn = new_connection()
    c=conn.cursor()
    return c.execute(statement)

def simple_query(query):
    "Helper to return rows from executed SQL with the default database connection."
    return execute_mysql(query).fetchall()

def write_mysql_data(df,table_name,region,if_exists='append',better_append=False,chunksize=None,dtype=None):
    df["region"] = region
    con = new_sqlalchemy_connection()
    if not better_append:
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,chunksize=chunksize,dtype=dtype)
    else:
        max_id = read_mysql_data("SELECT MAX(id) FROM {table_name}".format(table_name=table_name))
        id_range = range(max_id+1,max_id + len(df) + 1)
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,index_label=id_range) 

def read_mysql_data(query,alchemy=True):
    if alchemy:
        con = new_sqlalchemy_connection()
    else:
        con = new_connection()
    return pd.read_sql(query,con)

def make_json_columns(df,stand_alone_columns,json_columns):
    df = df.copy()
    df['json_col'] = df[json_columns].apply(lambda x: x.to_dict(), axis=1)
    print 'added json'
    df = df.loc[:,stand_alone_columns + ('json_col')]
    return df

def write_json_data(df,json_columns,region):
    print 'starting function'
    for i in df.itertuples():
        try:
            vj = voter_json()
            vj.region = region
            vj.address = i.address
            temp_dict = {j: getattr(i,j) for j in json_columns}
            vj.json_data = json.dumps(temp_dict)
            vj.save()
        except Exception as e:
            print e
            print i
            1/0
            #continue
