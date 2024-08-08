import time
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
import os
from fst import write_fst
from globals import pathProject, skipdaily

# Environment -------------------------------------------------------------
start_time = time.time()

## LOGIN TO WRDS
wrds_user = input('wrds username: ')
wrds_password = input('wrds password: ')

host = 'wrds-pgdata.wharton.upenn.edu'
port = 9737
dbname = 'wrds'
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=wrds_user,
        password=wrds_password,
        sslmode='require'
    )
    print("Connection to WRDS database established successfully.")
except Exception as e:
    print(f"An error occurred while connecting to the database: {e}")

num_rows_to_pull = -1 # Set to -1 for all rows and to some positive value for testing
yearmax_crspd = datetime.now().year

connection_string = f"postgresql://{wrds_user}:{wrds_password}@{host}:{port}/{dbname}?sslmode=require"
engine = create_engine(connection_string)
query = """
SELECT 
    a.permno, a.permco, a.date, a.ret, a.retx, a.vol, a.shrout, a.prc, a.cfacshr, a.bidlo, a.askhi,
    b.shrcd, b.exchcd, b.siccd, b.ticker, b.shrcls,  -- from identifying info table
    c.dlstcd, c.dlret                                -- from delistings table
FROM crsp.msf AS a
LEFT JOIN crsp.msenames AS b
ON a.permno = b.permno
AND b.namedt <= a.date
AND a.date <= b.nameendt
LEFT JOIN crsp.msedelist AS c
ON a.permno = c.permno
AND date_trunc('month', a.date) = date_trunc('month', c.dlstdt)
"""
try:
    with engine.connect() as conn:
        m_crsp = pd.read_sql_query(query, conn, chunksize=num_rows_to_pull)
        m_crsp = pd.concat(m_crsp, ignore_index=True)
        print(m_crsp.head()) 
except Exception as e:
    print(f"An error occurred: {e}")

write_fst(m_crsp, pathProject+'Portfolios/Data/Intermediate/m_crsp_raw.fst')

# CRSP daily --------------------------------------------------------------
if not skipdaily:
    for year in list(range(1926, yearmax_crspd + 1)):
        print(f"downloading daily crsp for a year")
        query=query = f"""
        SELECT a.permno, a.date, a.ret, a.shrout, a.prc, a.cfacshr
        FROM crsp.dsf AS a
        WHERE date >= '{year}-01-01'
        AND date <= '{year}-12-31'
        """
        try:
            with engine.connect() as conn:
                # Execute the query and fetch all rows
                temp_d_crsp = pd.read_sql_query(query, conn)
                print(temp_d_crsp.head())
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
        
        if year==1926:
            d_crsp = temp_d_crsp
        else:
            d_crsp= pd.concat([d_crsp, temp_d_crsp], ignore_index=True)
    write_fst(d_crsp, pathProject+'Portfolios/Data/Intermediate/d_crsp_raw.fst')

    end_time=time.time()
    print(f'total time taken for file 10: {end_time-start_time}')