# Note: daily portfolios currently (2021 04) do not aggregate up to monthly
# because the daily portfolios recalculate stock weights (equal or value-weighting) 
# every day while the monthly portfolios recalculate stock weights every month.

# Note: monthly portfolios are not screened at all for minimum number of stocks
# and instead, we store Nstocks, and then screen for Nstocks when we do summary stats
# However, to keep the daily portfolios data of a manageable size
# we do not store Nlong and Nshort, and instead impose the screen at the portfolio 
# level. 
from datetime import datetime
from fst import read_fst
import pandas as pd
import numpy as np
import os
import re
import statsmodels.api as sm
from statsmodels.formula.api import ols

from globals import pathProject, pathDataPortfolios
# takes about 1.5 hours per implementation, or about 10 hours total


### ENVIRONMENT AND DATA ####
start_time = datetime.now()

# minimum number of stocks in a portfolio
# for now set to 1 (2021 04), matching baseline
# setting to 20 removes IO_ShortInterest portfolios
Nstocksmin = 1


### load crsp returns
crspinfo = read_fst(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspminfo.fst')).to_pandas()
crspinfo=pd.DataFrame(crspinfo)

crspnet = read_fst(os.path.join(pathProject, 'Portfolios/Data/Intermediate/crspdret.fst')).to_pandas()
crspnet = pd.DataFrame(crspinfo)

### SET UP PATHS
#no other files use these paths

pathDataDaily  = os.path.join(pathProject, 'Portfolios/Data/DailyPortfolios/')
pathDataDailyBase   = os.path.join(pathDataDaily, 'Predictor/')
pathDataDailyBaseVW   = os.path.join(pathDataDaily, 'PredictorVW/')
pathDataDailyDecile = os.path.join(pathDataDaily, 'CtsPredictorDecile/')
pathDataDailyDecileVW  = os.path.join(pathDataDaily, 'CtsPredictorDecileVW/')
pathDataDailyQuintile = os.path.join(pathDataDaily, 'CtsPredictorQuintile/')
pathDataDailyQuintileVW  = os.path.join(pathDataDaily, 'CtsPredictorQuintileVW/')

os.makedirs(pathDataDaily, exist_ok=True)
os.makedirs(pathDataDailyBase, exist_ok=True)
os.makedirs(pathDataDailyBaseVW, exist_ok=True)
os.makedirs(pathDataDailyDecile, exist_ok=True)
os.makedirs(pathDataDailyDecileVW, exist_ok=True)
os.makedirs(pathDataDailyQuintile, exist_ok=True)
os.makedirs(pathDataDailyQuintileVW, exist_ok=True)


### SELECT SIGNALS
strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == 'Predictor']
strategylist0 = ifquickrun()

strategylistcts = strategylist0['Cat.Form'=='continuous']

### BASELINE ####
## BASELINE
print('dailyPredictorPorts.py: predictor baseline strats')

port = loop_over_strategies(
    strategylist0
  , saveportcsv = True
  , saveportpath = pathDataDailyBase
  , saveportNmin = Nstocksmin
  , passive_gain = True
)

## BASELINE
print('dailyPredictorPorts.py: predictor baseline VW')
strategylist0_mutated = strategylist0.assign(sweight='VW')
port = loop_over_strategies(
    strategylist0_mutated,
    saveportcsv=True,
    saveportpath=pathDataDailyBaseVW,
    saveportNmin=Nstocksmin,
    passive_gain=True
)

print('baseline time',datetime.now())


### DECILES ####
## FORCE DECILES
print ('dailyPredictorPorts.py: predictor force decile strats')
strategylistcts_mutated = strategylistcts.assign(q_cut=0.1)
port = loop_over_strategies(
    strategylistcts_mutated,
    saveportcsv=True,
    saveportpath=pathDataDailyDecile,
    saveportNmin=Nstocksmin,
    passive_gain=True
)

## FORCE DECILES AND VW
print ('dailyPredictorPorts.py: predictor force decile and VW strats')
strategylistcts_mutated = strategylistcts.assign(q_cut=0.1, sweight='VW')

port = loop_over_strategies(
    strategylistcts_mutated,
    saveportcsv=True,
    saveportpath=pathDataDailyDecileVW,
    saveportNmin=Nstocksmin,
    passive_gain=True
)

print('deciles time',datetime.now())

### QUINTILES ####
### QUINTILES ####
print ('dailyPredictorPorts.py: predictor force quint strats')
strategylistcts_mutated = strategylistcts.assign(q_cut=0.2)

port = loop_over_strategies(
    strategylistcts_mutated,
    saveportcsv=True,
    saveportpath=pathDataDailyQuintile,
    saveportNmin=Nstocksmin,
    passive_gain=True
)

## FORCE QUINTILES AND VW
print ('dailyPredictorPorts.py: predictor force quint and VW strats')
strategylistcts_mutated = strategylistcts.assign(q_cut=0.2, sweight='VW')

port = loop_over_strategies(
    strategylistcts_mutated,
    saveportcsv=True,
    saveportpath=pathDataDailyQuintileVW,
    saveportNmin=Nstocksmin,
    passive_gain=True
)
print('50_DailyPredictorPorts.R done!')
end_time = datetime.now()

print('start time, end time = ')

print(start_time)
print(end_time)

# CHECK CSVS ####
# this creates DailyPortSummary.xlsx
### FUNCTION FOR CHECKING A WHOLE FOLDER OF DAILY RETURNS

def checkdir(dircurr, csvlist):
    # Initialize an empty DataFrame for sumsignal
    sumsignal = pd.DataFrame()

    # Loop over each signal name in the csvlist
    for signalcurr in csvlist['signalname']:
        # Read the CSV file
        retd = pd.read_csv(os.path.join(pathDataDaily, dircurr, f'{signalcurr}_ret.csv'))
        
        # Reshape the data from wide to long format
        retd = retd.melt(id_vars=['date'], var_name='port', value_name='ret')
        
        # Filter out rows where ret is NA
        retd = retd.dropna(subset=['ret'])
        
        # Summarize the data
        tempstat = retd.groupby('port').agg(
            nobs_years=('ret', lambda x: len(x)/250),
            rbar_monthly=('ret', lambda x: x.mean()*20)
        ).reset_index()
        
        # Add the signal name to the summary
        tempstat['signalname'] = signalcurr
        
        # Append the summary to the sumsignal DataFrame
        sumsignal = pd.concat([sumsignal, tempstat], ignore_index=True)
    
    # Summarize the sumsignal DataFrame
    sumdir = sumsignal.groupby('port').agg(
        n_distinct_signalname=('signalname', pd.Series.nunique),
        mean_nobs_years=('nobs_years', 'mean'),
        mean_rbar_monthly=('rbar_monthly', 'mean')
    ).reset_index()
    
    # Add the implementation (directory) name
    sumdir['implementation'] = dircurr
    
    return sumdir
print(f"Checking on Daily Port stats, {datetime.now()} ")
dirlist = [d for d in os.listdir(pathDataDaily) if os.path.isdir(os.path.join(pathDataDaily, d))]
dirlist = dirlist[dirlist != '']

### check for completeness of daily portfolio csvs and summary stats
sumdaily = pd.DataFrame()

for dircurr in dirlist:
    print(f"checking on, {dircurr}")
    file_list = os.listdir(os.path.join(pathDataDaily, dircurr))
    csvlist = pd.DataFrame({
    'signalname': [re.sub('_ret.csv$', '', file) for file in file_list],
    'incsv': 1
    })

    ## check for mismatches in signal lists
    if dircurr[:3] == 'Cts':
        doclist = strategylistcts[['signalname']].assign(indoc=1)
    else:
        doclist = strategylist0[['signalname']].assign(indoc=1)
    
    mismatch = pd.merge(doclist, csvlist, on='signalname', how='outer')
    mismatch = mismatch[(mismatch['indoc'].isna()) | (mismatch['incsv'].isna())]
    if mismatch.shape[0] > 0:
        print(f'Warning: mismatch between signal docs and csvs for {pathDataDaily}{dircurr}/')
        print(mismatch)

    sumdir = checkdir(dircurr, csvlist)

    # Print the summary information
    print(f'Summary of {dircurr}')
    print(sumdir)

### check timing of daily predictor ports (base) with monthly returns
portmonthly = pd.read_csv(f"{pathDataPortfolios}/PredictorPortsFull.csv")

portmonthly = portmonthly.assign(
    datem=pd.to_datetime(portmonthly['date']),  # Convert date to datetime
    retm=portmonthly['ret']
).loc[:, ['signalname', 'port', 'datem', 'retm']]

dircurr = 'Predictor'

csv_files = os.listdir(os.path.join(pathDataDaily, dircurr))
csvlist = pd.DataFrame({
    'signalname': [re.sub(r'_ret\.csv$', '', file) for file in csv_files],
    'incsv': 1
})

signallist = csvlist['signalname']
print('checking daily vs monthly return timing')

reg_retm_retmagg=pd.DataFrame()

def fit_model(group):
            # Define the model using the formula
            model = ols('retm ~ retm_agg', data=group).fit()
            return model

for signalcurr in signallist:
    print(signalcurr)

    #read daily
    temp = pd.read_csv(f"{pathDataDaily}/{dircurr}/{signalcurr}_ret.csv")

    if temp.shape[0] > 0:
        # aggregate to monthly  
        datd = temp.melt(id_vars=['date'], var_name='port', value_name='ret')
        datd['port'] = datd['port'].str[4:6]
        datd = datd.dropna(subset=['ret'])
        datd['date'] = pd.to_datetime(datd['date'])
        datd['datem'] = (datd['date'] + pd.offsets.MonthEnd(0)).dt.floor('D')
        datd_summary = datd.groupby(['datem', 'port']).agg(
            retm_agg=lambda x: 100 * (np.prod(1 + x / 100) - 1)
        ).reset_index()

        datboth = portmonthly[portmonthly['signalname'] == signalcurr]
        datboth = datboth.merge(datd_summary, on=['port', 'datem'], how='left')
        datboth = datboth.dropna(subset=['retm_agg'])

        # remove port if too few observations
        temp = datboth.groupby('port').agg(
            nobs=('retm', lambda x: x.notna().sum())
        ).reset_index()
        temp = temp[temp['nobs'] > 10]
        datboth = datboth[datboth['port'].isin(temp['port'])]

        # regress monthly on daily aggregated to monthly
        models = datboth.groupby('port').apply(fit_model)

        # Initialize lists to store the results
        signalnames = []
        ports = []
        intercepts = []
        slopes = []
        rsqs = []

        # Iterate over each fitted model
        for port, model in models.items():
            signalnames.append(signalcurr)  # Assuming signalcurr is defined
            ports.append(port)
            intercepts.append(model.params['Intercept'])
            slopes.append(model.params['retm_agg'])
            rsqs.append(model.rsquared)

        # Create the DataFrame
        reg_curr = pd.DataFrame({
            'signalname': signalnames,
            'port': ports,
            'intercept': intercepts,
            'slope': slopes,
            'rsq': rsqs
        })
    else:
        # here the return file is empty, probably because not enough stocks in the portfolio
        reg_curr.loc[:, :] = pd.NA
        reg_curr = reg_curr.iloc[[0]].assign(signalname=signalcurr)
    #append
    reg_retm_retmagg = pd.concat([reg_retm_retmagg, reg_curr], ignore_index=True)

# summarize regressions
reg_sum = reg_retm_retmagg.groupby('port').apply(
    lambda x: x.dropna(subset=['slope'])
)
reg_sum = reg_sum.groupby('port').agg(
    slope_10th_quantile=('slope', lambda x: x.quantile(0.1)),
    slope_50th_median=('slope', lambda x: x.quantile(0.5)),
    rsq_10th_quantile=('rsq', lambda x: x.quantile(0.1)),
    rsq_50th_median=('rsq', lambda x: x.quantile(0.5))
).reset_index()


### output
data_to_write = {
    'sumstats': sumdaily.sort_values(by=['implementation', 'port']).reset_index(drop=True),
    'timingcheck': reg_sum
}
with pd.ExcelWriter(f"{pathDataDaily}/DailyPortSummary.xlsx", engine='openpyxl') as writer:
    for sheet_name, df in data_to_write.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Done: Checking on Daily Port stats, {datetime.now()} ")




