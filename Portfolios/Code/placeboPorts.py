from fst import read_fst
import pandas as pd
import os

from globals import pathProject, pathDataPortfolios

def read_data():
# ==== ENVIRONMENT AND DATA ====
    crspret = read_fst(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspmret.fst')).to_pandas()
    crspret=pd.DataFrame(crspret)

    crspinfo = read_fst(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspminfo.fst'))
    crspinfo=pd.DataFrame(crspinfo)

    return crspret, crspinfo

def select_signals(alldocumentation, crspret, crspinfo, ifquickrun, loop_over_strategies, writestandard, sumportmonth):
    ######################################################################
    ### SELECT SIGNALS
    ######################################################################
    strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == "Placebo"]
    strategylist0 = ifquickrun()

    #####################################################################
    ### COMPUTE PORTFOLIOS
    #####################################################################
    portmonth = loop_over_strategies(strategylist0,crspret,crspinfo)
    ## EXPORT
    writestandard(portmonth,pathDataPortfolios,"PlaceboPortsFull.csv")

    # SUMMARY STATS BY SIGNAL -------------------------------------------------
    sumbase = sumportmonth(
        portmonth,
        groupme=["signalname", "port", "samptype"],
        Nstocksmin=20
    )
    strategy_subset = strategylist0[[
        'sweight', 'q_cut', 'q_filt', 'portperiod', 'startmonth', 'filterstr'
    ] + strategylist0.columns.tolist()]
    sumbase = sumbase.merge(strategy_subset, on='signalname', how='left')

    ## export
    ls_insamp_only = sumbase[
        (sumbase['samptype'] == "insamp") & 
        (sumbase['port'] == "LS")
    ].sort_values(by='tstat')

    sheets = {
        'ls_insamp_only': ls_insamp_only,
        'full': sumbase
    }

    output_path = f"{pathDataPortfolios}/PlaceboSummary.xlsx"

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for sheet_name, data in sheets.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    # FEEDBACK ON ERRORS -------------------------------------------------
    tempsum = sumportmonth(
        portmonth,
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







