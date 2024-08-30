from globals import pathDataIntermediate, pathPredictors

import os
import pandas as pd
from fst import read_fst
import numpy as np

def read_data():
    ### READ DATA

    crspret = read_fst(os.path.join(pathDataIntermediate, 'crspmret.fst'))
    crspinfo = read_fst(os.path.join(pathDataIntermediate, 'crspminfo.fst'))

    return crspret, crspinfo

def make_STreversal(crspret):
    ### MAKE STreversal
    strev_csv_path = os.path.join(pathPredictors, 'STreversal.csv')
    if not os.path.exists(strev_csv_path):
        temp = crspret[['permno', 'date', 'ret']].copy()
        temp['STreversal'] = temp['ret'].fillna(0)
        temp['yyyymm'] = temp['date'].dt.year * 100 + temp['date'].dt.month
        temp = temp.dropna(subset=['STreversal'])
        temp = temp[['permno', 'yyyymm', 'STreversal']]
        
        temp.to_csv(strev_csv_path, index=False)
def make_price(crspinfo):
    ### MAKE Price
    price_csv_path = os.path.join(pathPredictors, 'Price.csv')
    if not os.path.exists(price_csv_path):
        temp = crspinfo[['permno', 'yyyymm', 'prc']].copy()
        temp['Price'] = np.log(np.abs(temp['prc']))
        temp = temp.dropna(subset=['Price'])
        temp = temp[['permno', 'yyyymm', 'Price']]
        
        temp.to_csv(price_csv_path, index=False)
def make_size(crspinfo):
    ### MAKE Size
    size_csv_path = os.path.join(pathPredictors, 'Size.csv')
    if not os.path.exists(size_csv_path):
        temp = crspinfo[['permno', 'yyyymm', 'me']].copy()
        temp['Size'] = np.log(temp['me'])
        temp = temp.dropna(subset=['Size'])
        temp = temp[['permno', 'yyyymm', 'Size']]
        
        temp.to_csv(size_csv_path, index=False)



