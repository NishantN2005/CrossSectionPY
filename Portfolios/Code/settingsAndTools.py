import os
import numpy as np
import pandas as pd
import re
import warnings
import time
import csv
#paths
from globals import pathProject, pathResults,pathDataPortfolios, pathDataIntermediate, pathPredictors, pathPlacebos, pathtemp

#settings
from globals import quickrun, quickrunlist

def create_folders():
    ## Create folders if they don't exist
    # Portfolios/ paths
    if not os.path.exists(pathResults):
        os.mkdir(pathResults)
    if not os.path.exists(f'{pathProject}Portfolios/Data'):
        os.mkdir(f'{pathProject}Portfolios/Data')
    if not os.path.exists(pathDataPortfolios):
        os.mkdir(pathDataPortfolios)
    if not os.path.exists(pathDataIntermediate):
        os.mkdir(pathDataIntermediate)


    # Signals/Data/ paths
    if not os.path.exists(f'{pathProject}Signals/Data'):
        os.mkdir(pathProject+'Signals/Data/')
    if not os.path.exists(pathPredictors):
        os.mkdir(pathPredictors)
    if not os.path.exists(pathPlacebos):
        os.mkdir(pathPlacebos)
    if not os.path.exists(pathtemp):
        os.mkdir(pathtemp)

def read_documentation():
    # little function for converting string NA into numeric NA
    def as_num(x, na_strings = np.array(["NA",'None','none'])):
        if not all(isinstance(item, str) for item in x):
            raise ValueError("Input must be a list of strings")
        
        na = [item in na_strings for item in x]
        x = [0 if is_na else item for item, is_na in zip(x, na)]
        x = np.array(x, dtype=float)
        x[na] = np.nan

        return x
    
    #load signal header
    all_documentation = pd.read_csv(f"{pathProject}SignalDoc.csv")
    all_documentation = all_documentation.rename(columns={'Acronym': 'signalname'})
    all_documentation = all_documentation.assign(Cat_Data=lambda df: pd.Categorical(
            df['Cat.Data'], 
            categories=['Accounting', 'Analyst', 'Event', 'Options', 'Price', 'Trading', '13F', 'Other'],
            ordered=True
        ))
    all_documentation = all_documentation.assign(Cat_Economic=lambda df: df['Cat.Economic'].str.title())
    all_documentation = all_documentation.rename(columns={'Stock Weight': 'sweight',
            'LS Quantile': 'q_cut',
            'Quantile Filter': 'q_filt',
            'Portfolio Period': 'portperiod',
            'Start Month': 'startmonth',
            'Filter': 'filterstr'
        })
    # Modify filterstr column
    all_documentation = all_documentation.assign(filterstr=lambda df: df['filterstr'].replace(['NA', 'None', 'none'], pd.NA))
    # Drop columns starting with 'Note'
    all_documentation = all_documentation.drop(columns=[col for col in all_documentation.columns if col.startswith('Note')])
    # Arrange by signalname
    all_documentation = all_documentation.sort_values(by='signalname')
    return all_documentation
    
def check_signals(docs = read_documentation(), path_proj = pathProject):
    # Classification in SignalDoc
    prds_predictor = docs.loc[docs['Cat.Signal']=='Predictor']['signalname']
    prds_placebo = docs.loc[docs['Cat.Signal']=='Placebo']['signalname']
    
    # Created signals
    fls_predictors = os.listdir(os.path.join(path_proj, 'Signals/Data/Predictors/'))
    fls_placebos = os.listdir(os.path.join(path_proj, 'Signals/Data/Placebos/'))

    # Predictor in Data/Predictor?
    predNotInData = []
    for p in prds_predictor:
        if sum(bool(re.search(p, fl, re.IGNORECASE)) for fl in fls_predictors):
            predNotInData.append(p)

    # Placebo in Data/Placebo?
    placeboNotInData = []
    for p in prds_placebo:
        if sum(bool(re.search(p, fls_placebos, re.IGNORECASE)) for fl in fls_placebos):
            placeboNotInData.append(p)
    
    # Output warnings
    if predNotInData:
        warnings.warn('The following Predictors in SignalDoc have not been created in Data/Predictors:')
        print(predNotInData)

    if placeboNotInData: 
        warnings.warn('The following Placebos in SignalDoc have not been created in Data/Placebos:')
        print(placeboNotInData)
    
    if not predNotInData and not placeboNotInData: 
        print('All predictors and placebos were created.')

### FUNCTION FOR STANDARD CSV EXPORT
def write_standard(df, path, filename):
    file_path = f'{path}{filename}'

    df.to_csv(file_path, 
                sep=",", 
                header=True, 
                index=False, 
                doublequote=True, 
                quoting=pd.io.common.csv.QUOTE_NONE)
    
### FUNCTION FOR SUMMARIZING PORTMONTH DATASET
def sum_port_month(portret, alldocumentation, groupme = ['signalname', 'samptype', 'port'], Nstocksmin=20):
    # Left join portret with alldocumentation
    temp = portret.merge(
        alldocumentation[['signalname', 'SampleStartYear', 'SampleEndYear', 'Year']],
        on='signalname',
        how='left'
    )

    # Add samptype column using conditions
    temp['samptype'] = np.select(
    [
        (temp['date'].dt.year >= temp['SampleStartYear']) & (temp['date'].dt.year <= temp['SampleEndYear']),
        (temp['date'].dt.year > temp['SampleEndYear']) & (temp['date'].dt.year <= temp['Year']),
        (temp['date'].dt.year > temp['Year'])
    ],
    ['insamp', 'between', 'postpub'],
    default=pd.NA
    )

    # Drop unnecessary columns
    temp = temp.drop(columns=['SampleStartYear', 'SampleEndYear', 'Year'])

    # Add Ncheck column
    temp['Ncheck'] = np.where(temp['port'] != 'LS', temp['Nlong'], np.minimum(temp['Nlong'], temp['Nshort']).astype(int))

    # Filter rows based on Nstocksmin condition
    temp = temp[temp['Ncheck'] >= Nstocksmin]

    # Group by specified columns and summarize
    tempsum = temp.groupby(groupme).agg(
        tstat=('ret', lambda x: round(np.mean(x) / np.std(x) * np.sqrt(len(x)), 2)),
        rbar=('ret', lambda x: round(np.mean(x), 2)),
        vol=('ret', lambda x: round(np.std(x), 2)),
        T=('ret', 'size'),
        Nlong=('Nlong', lambda x: round(np.mean(x), 1)),
        Nshort=('Nshort', lambda x: round(np.mean(x), 1)),
        signallag=('signallag', lambda x: round(np.mean(x), 3))
    ).reset_index()

    # Sort the DataFrame
    tempsum = tempsum.sort_values(by=['samptype', 'signalname', 'port'])

    return tempsum

### CHECK PORTFOLIOS ###
def check_port(port, groupme = ['signalname', 'port']):
    summary_df = sum_port_month(port, read_documentation(), Nstocksmin=1)

    # Filter for 'insamp' samptype and print
    insamp_summary = summary_df[summary_df['samptype'] == 'insamp']
    print(insamp_summary)

    # Mutate port to add Nok column
    port['Nok'] = np.where(port['Nlong'] >= 20, 'Nlong>=20', 'Nlong<20')

        # Group, summarize, and pivot
    port_summary = (
        port.groupby(['signalname', 'port', 'Nok'])
        .size()
        .reset_index(name='nportmonths')
        .pivot(index=['signalname', 'port'], columns='Nok', values='nportmonths')
        .fillna(0)  # Fill NaN with 0 for missing values
    )

    # Add prefix to column names
    port_summary.columns = ['t w/ ' + col for col in port_summary.columns]

    # Reset index to make it a regular DataFrame
    port_summary = port_summary.reset_index()

    # Print the port summary
    print(port_summary)

### FUNCTION FOR QUICK TESTING ALL SCRIPTS
def if_quick_run(strategy_list):
    if quickrun:
        print('running quickly')
        strategy_list = strategy_list[
        strategy_list['signalname'].isin(quickrunlist)
        ]

    return strategy_list

#############################################################################

def loop_over_strategies(strategy_list, 
                            saveportcsv=False,
                            saveportpath = np.nan,
                            saveportNmin = 1,
                            passive_gain = False 
                            ):
    Nstrat = len(strategy_list) #number of rows

    allport=[]

    for i in range(Nstrat):
        print(f'{i}/{Nstrat}:{strategy_list['signalname'][i]}')

        # Select specific columns for the i-th row and print
        print(strategy_list.loc[i, ['signalname', 'Cat.Form', 'q_cut', 'sweight', 'portperiod', 'q_filt', 'filterstr']])


        start_time = time.time()

        try:
            tempport = signalname_to_ports(
            signalname=strategy_list.iloc[i]['signalname'],
            Cat_Form=strategy_list.iloc[i]['Cat.Form'],
            q_cut=strategy_list.iloc[i]['q_cut'],
            sweight=strategy_list.iloc[i]['sweight'],
            Sign=strategy_list.iloc[i]['Sign'],
            startmonth=strategy_list.iloc[i]['startmonth'],
            portperiod=strategy_list.iloc[i]['portperiod'],
            q_filt=strategy_list.iloc[i]['q_filt'],
            filterstr=strategy_list.iloc[i]['filterstr'],
            passive_gain=passive_gain
        )
        except Exception as e:
            print('Error in signalname_to_ports, returning df with NaN')
            if allport:
                num_columns = allport[0].shape[1]
            else:
                num_columns=5
            tempport = pd.DataFrame(np.nan, index=[0], columns=range(num_columns))
        
        # Check if tempport is valid
        if pd.isna(tempport.iloc[0, 0]):
            if allport:
                tempport.columns = allport[0].columns
            tempport['signalname'] = strategy_list.iloc[0]['signalname']

        #write csv file
        if saveportcsv and saveportpath:
            print(f"Saving wide port to {saveportpath}")

            tempwide = tempport[tempport['Nlong'] >= saveportNmin][['port', 'date', 'ret']]
            tempwide = tempwide.pivot(index='date', columns='port', values='ret').reset_index()

            write_standard(tempwide, saveportpath, f"{strategy_list.iloc[i]['signalname']}_ret.csv")

        # Append the processed port to allport
        allport.append(tempport)

        end_time = time.time()
        print(f"Elapsed Time: {end_time - start_time} seconds")

        del tempport

        allport_df = pd.concat(allport, ignore_index=True)

        return allport_df









        
        



