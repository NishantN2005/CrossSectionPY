import pandas as pd
from fst import read_fst
import os
from globals import pathProject, pathPredictors, pathDataPortfolios


# ENVIRONMENT AND DATA ====
crspinfo = read_fst(f'{pathProject}/Portfolios/Data/Intermediate/crspminfo.fst')
crspret = read_fst(f'{pathProject}/Portfolios/Data/Intermediate/crspmret.fst')


# SELECT SIGNALS AND CHECK FOR CSVS ====
strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == 'Predictor']
strategylist0 = ifquickrun()

csvlist = pd.DataFrame({'signalname': [f[:-4] for f in os.listdir(pathPredictors)], 'in_csv': 1})

missing = pd.merge(strategylist0[['signalname']], csvlist, on='signalname', how='left')
missing = missing[missing['in_csv'].isna()]

# BASE PORTS ====
port = loop_over_strategies(strategylist0)

# feedback
checkport(port)

## EXPORT
port.to_csv(f'{pathDataPortfolios}/PredictorPortsFull.csv', index=False)

sumbase = sumportmonth(port, groupme=['signalname', 'port', 'samptype'], Nstocksmin=1)
sumbase = pd.merge(sumbase, alldocumentation, on='signalname', how='left')


with pd.ExcelWriter(f'{pathDataPortfolios}/PredictorSummary.xlsx') as writer:
    sumshort.to_excel(writer, sheet_name='short', index=False)
    sumbase.to_excel(writer, sheet_name='full', index=False)

print("The following ports are computed successfully")
print(sumbase[(sumbase['port'] == 'LS') & (sumbase['samptype'] == 'insamp')].sort_values(by='tstat'))

print("The following ports failed to compute")
print(sumbase[sumbase['port'].isna()].sort_values(by='tstat')[['signalname', 'port']])

if sumbase[sumbase['port'].isna()].shape[0] == 0:
    print('20_PredictorPorts.py: all portfolios successfully computed')



