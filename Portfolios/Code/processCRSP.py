import os
import pandas as pd
import numpy as np

from globals import pathProject, skipdaily

def process():
    # ==== MONTHLY CRSP SETUP ====

    # Read the FST file
    crspm = pd.read_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/m_crsp_raw.csv'))
    print("created crspm csv file")

    # Incorporate delisting return
    print('incorporating delisting return')
    crspm['dlret'] = np.where(
        crspm['dlret'].isna() & crspm['dlstcd'].isin([500] + list(range(520, 585))) & crspm['exchcd'].isin([1, 2]),
        -0.35,
        crspm['dlret']
    )
    crspm['dlret'] = np.where(
        crspm['dlret'].isna() & crspm['dlstcd'].isin([500] + list(range(520, 585))) & (crspm['exchcd'] == 3),
        -0.55,
        crspm['dlret']
    )
    crspm['dlret'] = np.where(
        (crspm['dlret'] < -1) & (~crspm['dlret'].isna()),
        -1,
        crspm['dlret']
    )
    crspm['dlret'] = crspm['dlret'].fillna(0)
    crspm['ret'] = (1 + crspm['ret']) * (1 + crspm['dlret']) - 1
    crspm['ret'] = np.where(
        crspm['ret'].isna() & (crspm['dlret'] != 0),
        crspm['dlret'],
        crspm['ret']
    )

    # Convert ret to percentage and format other fields
    print("converting return to percentage and formating")
    crspm['ret'] = 100 * crspm['ret']
    crspm['date'] = pd.to_datetime(crspm['date'])
    crspm['me'] = abs(crspm['prc']) * crspm['shrout']
    crspm['yyyymm'] = crspm['date'].dt.year * 100 + crspm['date'].dt.month

    # Keep around me and melag for sanity
    templag = crspm[['permno', 'yyyymm', 'me']].copy()
    templag['yyyymm'] = templag['yyyymm'] + 1
    templag['yyyymm'] = np.where(templag['yyyymm'] % 100 == 13, templag['yyyymm'] + 100 - 12, templag['yyyymm'])
    templag = templag.rename(columns={'me': 'melag'})

    print("subsetting into two smaller datasets for cleanliness")
    # Subset into two smaller datasets for cleanliness
    crspmret = crspm[['permno', 'date', 'yyyymm', 'ret']].dropna(subset=['ret'])
    crspmret = crspmret.merge(templag, on=['permno', 'yyyymm'], how='left').sort_values(by=['permno', 'yyyymm'])

    crspminfo = crspm[['permno', 'yyyymm', 'prc', 'exchcd', 'me', 'shrcd']].sort_values(by=['permno', 'yyyymm'])

    print("Adding info for easy me quantile screens")
    # Add info for easy me quantile screens
    tempcut = crspminfo[crspminfo['exchcd'] == 1].groupby('yyyymm').agg(
        me_nyse10=('me', lambda x: x.quantile(0.1)),
        me_nyse20=('me', lambda x: x.quantile(0.2))
    ).reset_index()
    crspminfo = crspminfo.merge(tempcut, on='yyyymm', how='left')

    print("creating crspmret and crspinfo")
    # Write to disk
    crspmret.to_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspmret.csv'))
    crspminfo.to_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspminfo.csv'))

    # Clean up
    del crspm, crspmret, crspminfo, templag, tempcut

    # ==== DAILY CRSP SETUP ====

    if not skipdaily:
        # Read daily CRSP data
        crspdret = pd.read_csv(
            os.path.join(pathProject, 'Portfolios/Data/Intermediate/d_crsp_raw.csv')
        )
        crspdret = crspdret.filter(items=['permno', 'date', 'ret'])

        # Drop NA and reformat
        crspdret = crspdret.dropna(subset=['ret'])
        crspdret['ret'] = 100 * crspdret['ret']
        crspdret['date'] = pd.to_datetime(crspdret['date'])
        crspdret['yyyymm'] = crspdret['date'].dt.year * 100 + crspdret['date'].dt.month
        crspdret = crspdret.sort_values(by=['permno', 'date'])

        # Calculate passive within-month gains
        crspdret['passgain'] = crspdret.groupby(['permno', 'yyyymm'])['ret'].shift(1, fill_value=0)
        crspdret['passgain'] = crspdret.groupby(['permno', 'yyyymm'])['passgain'].apply(lambda x: (1 + x / 100).cumprod())

        # Merge on last month's lagged me
        templag = pd.read_csv(
            os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspminfo.csv')
        )
        templag = templag.filter(items=['permno', 'yyyymm', 'me'])
        templag['yyyymm'] = templag['yyyymm'] + 1
        templag['yyyymm'] = np.where(templag['yyyymm'] % 100 == 13, templag['yyyymm'] + 100 - 12, templag['yyyymm'])
        templag = templag.rename(columns={'me': 'melag'})
        
        crspdret = crspdret.merge(templag, on=['permno', 'yyyymm'], how='left')

        # Write to disk
        crspdret.to_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspdret.csv'))
