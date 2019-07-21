
import json

import MySQLdb
import pymysql
import sqlalchemy

import pandas as pd

from django.conf import settings


def make_mysql_connection(skip_db = False):
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    database_name = settings.DATABASES['default']['NAME']
    if skip_db:
        return MySQLdb.connect(user=user,password=password)
    return MySQLdb.connect(user=user,password=password,database_name=database_name)

def make_sqlalchemy_connection():
    user = settings.DATABASES['default']['USER']
    password = settings.DATABASES['default']['PASSWORD']
    host = settings.DATABASES['default']['HOST'] or 'localhost'
    dbname = settings.DATABASES['default']['NAME']
    return sqlalchemy.create_engine('mysql+pymysql://{user}:{password}@{host}/{name}'.format(user=user,password=password,host=host,name=dbname))

def make_mysqldb_connection():
    x = MySQLdb.connect(host='localhost', \
                    user=settings.DATABASES['default']['USER'], \
                    passwd=settings.DATABASES['default']['PASSWORD'], \
                    db="canvas_cutting")
    x = MySQLdb.connect(host="159.89.185.59", \
                    user="username", \
                    passwd="password", \
                    db="canvas_cutting")
    return x

def execute_mysql(statement):
    conn = make_mysql_connection(True)
    c=conn.cursor()
    c.execute(statement)

def simple_query(query):
    conn = make_mysql_connection(True)
    c=conn.cursor()
    c.execute(query)
    return c.fetchall()

def write_mysql_data(df,table_name,region,if_exists='append',better_append=False,chunksize=None,dtype=None):
    df["region"] = region
    con = make_sqlalchemy_connection()
    if not better_append:
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,chunksize=chunksize,dtype=dtype)
    else:
        max_id = read_mysql_data("SELECT MAX(id) FROM {table_name}".format(table_name=table_name))
        id_range = range(max_id+1,max_id + len(df) + 1)
        df.to_sql(con=con, name=table_name, if_exists=if_exists,index=False,index_label=id_range) 

def read_mysql_data(query,alchemy=True):
    if alchemy:
        con = make_sqlalchemy_connection()
    else:
        con = make_mysqldb_connection()
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
