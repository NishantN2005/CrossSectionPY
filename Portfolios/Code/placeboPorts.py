import pandas as pd
import os
from openpyxl import Workbook

from globals import pathProject, pathDataPortfolios

def read_data():
# ==== ENVIRONMENT AND DATA ====
    crspret = pd.read_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspmret.csv'))

    crspinfo = pd.read_csv(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspminfo.csv'))

    return crspret, crspinfo

def select_signals(alldocumentation, crspret, crspinfo, ifquickrun, loop_over_strategies, writestandard, sumportmonth):
    ######################################################################
    ### SELECT SIGNALS
    ######################################################################
    print('----alldocumentation----')
    print(alldocumentation)
    strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == "Placebo"]
    strategylist0 = ifquickrun(strategylist0)

    #####################################################################
    ### COMPUTE PORTFOLIOS
    #####################################################################
    portmonth = loop_over_strategies(strategylist0,crspret,crspinfo)
    ## EXPORT
    print('------portmonth-----')
    print(portmonth)

    writestandard(portmonth,pathDataPortfolios,"PlaceboPortsFull.csv")

    # SUMMARY STATS BY SIGNAL -------------------------------------------------
    sumbase = sumportmonth(
        portmonth,
        alldocumentation,
        groupme=["signalname", "port", "samptype"],
        Nstocksmin=20
    )
    strategy_subset = strategylist0[[
        'sweight', 'q_cut', 'q_filt', 'portperiod', 'startmonth', 'filterstr'
    ] + strategylist0.columns.tolist()]
    sumbase = sumbase.merge(strategy_subset, on='signalname', how='left')

    ## export
    ls_insamp_only = sumbase[
    (sumbase['samptype'] == 'insamp') & (sumbase['port'] == 'LS')
    ].sort_values(by='tstat')

    # Define the full dataset
    full = sumbase

    # Define the file path
    file_path = f"{pathDataPortfolios}/PlaceboSummary.xlsx"

    # Export the DataFrames to an Excel file with multiple sheets
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        ls_insamp_only.to_excel(writer, sheet_name='ls_insamp_only', index=False)
        full.to_excel(writer, sheet_name='full', index=False)

    # FEEDBACK ON ERRORS -------------------------------------------------
    tempsum = sumportmonth(
        portmonth,
        alldocumentation,
        groupme=["signalname", "port", "samptype"],
        Nstocksmin=20
    )
    print("The following ports are computed succesfully")

    filtered_tempsum = tempsum[
        (tempsum['port'] == "LS") & 
        (tempsum['samptype'] == "insamp")
    ].sort_values(by='tstat')
    print(filtered_tempsum)

    print("The following ports failed to compute")

    filtered_tempsum = tempsum[
        tempsum['port'].isna()
    ].sort_values(by='tstat')
    print(filtered_tempsum)







