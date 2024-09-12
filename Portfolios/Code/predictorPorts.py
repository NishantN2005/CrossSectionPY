import os
import pandas as pd
from openpyxl import load_workbook

from globals import pathProject,pathPredictors,pathDataPortfolios

def read_data():
    # ENVIRONMENT AND DATA ====
    crspinfo = pd.read_csv(f"{pathProject}/Portfolios/Data/Intermediate/crspminfo.csv")
    crspret = pd.read_csv(f"{pathProject}/Portfolios/Data/Intermediate/crspmret.csv")

    return crspinfo, crspret

def process(alldocumentation,crspret,crspinfo, ifquickrun, loop_over_strategies,checkport, writestandard, sum_port_month):
    # SELECT SIGNALS AND CHECK FOR CSVS ====

    strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == "Predictor"]
    strategylist0 = ifquickrun(strategylist0)

    csv_files = os.listdir(pathPredictors)
    csvlist = pd.DataFrame({
        'signalname': [filename[:-4] for filename in csv_files],
        'in_csv': 1
    })

    missing = pd.merge(strategylist0[['signalname']], csvlist, on='signalname', how='left')
    missing = missing[missing['in_csv'].isna()]

    if missing.shape[0] > 0:
        print('Warning: the following predictor signal CSVs are missing:')
        print(missing['signalname'].tolist())

        temp = input('Press Enter to continue, type "quit" to quit: ')
        if temp == 'quit':
            print('Erroring out')
            raise SystemExit()

    # BASE PORTS ====
    port = loop_over_strategies(strategylist0,crspret,crspinfo)

    # Feedback
    checkport(port)

    # EXPORT
    writestandard(port, pathDataPortfolios, "PredictorPortsFull.csv")

    # OUTPUT WIDE
    portlswide = port[port['port'] == "LS"]
    portlswide = portlswide.pivot(index='date', columns='signalname', values='ret').reset_index()
    portlswide = portlswide.sort_values(by='date')

    writestandard(portlswide, pathDataPortfolios, "PredictorLSretWide.csv")

    # SUMMARY STATS BY SIGNAL ====

    # Reread in case you want to edit the summary later
    port = pd.read_csv(f"{pathDataPortfolios}/PredictorPortsFull.csv")

    sumbase = sum_port_month(
        port,
        alldocumentation,
        groupme=['signalname', 'port', 'samptype'],
        Nstocksmin=1
    ).merge(alldocumentation, on='signalname', how='left')

    sumshort = sumbase[(sumbase['samptype'] == "insamp") & (sumbase['port'] == "LS")]
    sumshort = sumshort.sort_values(by='signalname').drop(columns=['signallag'])

    # Export summary stats
    with pd.ExcelWriter(f"{pathDataPortfolios}/PredictorSummary.xlsx", engine='openpyxl') as writer:
        sumshort.to_excel(writer, sheet_name='short', index=False)
        sumbase.to_excel(writer, sheet_name='full', index=False)

    # FEEDBACK ON ERRORS ====

    print("The following ports are computed successfully")
    print(sumbase[(sumbase['port'] == "LS") & (sumbase['samptype'] == "insamp")].sort_values(by='tstat'))

    print("The following ports failed to compute")
    failed_ports = sumbase[sumbase['port'].isna()].sort_values(by='tstat')
    print(failed_ports[['signalname', 'port']])

    if failed_ports.shape[0] == 0:
        print('All portfolios successfully computed')
