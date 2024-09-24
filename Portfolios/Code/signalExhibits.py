from settingsAndTools import read_documentation
import pandas as pd
import numpy as np
import os
from globals import pathProject, pathResults,pathPredictors, pathDataIntermediate
from settingsAndTools import check_signals
import itertools
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def corSpearman(x, y):
        return np.corrcoef(x, y)[0, 1]

def calculate_correlation(i, loopList, pathDataIntermediate):
    # Reading data using fst
    cols_to_use = columns_to_read = loopList.iloc[i].tolist()
    print(i, cols_to_use)
    file_path = os.path.join(pathDataIntermediate, 'temp.csv')
    tempSignals = pd.read_csv(file_path, usecols=cols_to_use)

    # Filter complete cases (remove rows with any NaN values)
    tempSignals = tempSignals.dropna().values  # Converts to numpy matrix

    # Spearman correlation
    correlation = corSpearman(tempSignals[:, 0], tempSignals[:, 1])

    return correlation

def parallel_correlation(loop_list):
    # Parallel processing using multiprocessing.Pool
    with Pool() as pool:
        results = pool.starmap(calculate_correlation, [(i, loop_list, pathDataIntermediate) for i in range(len(loop_list))])
    
    return results
# Figure N: Dataset Coverage -------------------
def process():
    # first count for each paper
    count_us = read_documentation()
    print("-------count_us------")
    print(count_us.columns)
    count_us=(
           count_us[count_us['Predictability in OP']!='9_drop']
           .assign(bench=lambda x: x['Cat.Signal']=='Predictor')
           .groupby('Predictability in OP')
           .agg(bench=('bench','sum'), extended=('Cat.Signal', 'size'))
           .reset_index()
           )


    count_mp = (
        pd.read_csv(f"{pathProject}Comparison_to_MetaReplications.csv")
        .query("metastudy == 'MP'")
        .assign(covered=lambda df: df['ourname'] != '_missing_')  # Create new 'covered' column
        .groupby('Predictability.in.OP')  # Group by 'Predictability_in_OP'
        .agg(n=('metastudy', 'size'), covered=('covered', 'sum'))  # Summarize
        .assign(pctcov=lambda df: df['covered'] / df['n'] * 100)  # Calculate percentage coverage
        .reset_index()
        )
    
    count_ghz=(
        pd.read_csv(f"{pathProject}Comparison_to_MetaReplications.csv")
        .query("metastudy == 'GHZ'")  # Filter rows where metastudy is 'GHZ'
        .assign(covered=lambda df: df['ourname'] != '_missing_')  # Create new 'covered' column
        .groupby('Predictability.in.OP')  # Group by 'Predictability_in_OP'
        .agg(n=('metastudy', 'size'), covered=('covered', 'sum'))  # Summarize by counting and summing
        .assign(pctcov=lambda df: df['covered'] / df['n'] * 100)  # Calculate percentage coverage
        .reset_index()  # Reset index to make 'Predictability_in_OP' a column again
    )
    print("-----count_ghz------")
    print(count_ghz.columns)

    # for HXZ, we create a special category for alternative holding periods
    df_hxz = pd.read_csv(f"{pathProject}Comparison_to_MetaReplications.csv")

    # Filter for metastudy == 'HXZ' and create 'covered' column
    df_hxz = df_hxz[df_hxz['metastudy'] == 'HXZ'].copy()
    df_hxz['covered'] = df_hxz['ourname'] != '_missing_'

    # Group by 'ourname' and create 'holdalt' column using row_number equivalent
    df_hxz['holdalt'] = df_hxz.groupby('ourname').cumcount() + 1    

    # Update 'Predictability.in.OP' based on the value of 'holdalt'
    df_hxz['Predictability.in.OP'] = df_hxz.apply(
        lambda row: row['Predictability.in.OP'] if row['holdalt'] == 1 else 'z0_altholdper', axis=1
    )

    # Group by 'Predictability.in.OP' and summarize
    count_hxz = df_hxz.groupby('Predictability.in.OP').agg(
        n=('Predictability.in.OP', 'size'),
        covered=('covered', 'sum')
    ).reset_index()

    # Calculate the percentage coverage
    count_hxz['pctcov'] = (count_hxz['covered'] / count_hxz['n']) * 100

    # HLZ has its own csv since it's so different (not replication)
    # coverage then needs to be more judgmental

    df_hlz = pd.read_csv(f"{pathProject}Comparison_to_HLZ.csv")

    # Create 'covered' column
    df_hlz['covered'] = df_hlz['Coverage'] != 'zz missing'
    # Select relevant columns
    df_hlz = df_hlz[['Risk factor', 'Predictability.in.OP', 'covered', 'Coverage']] 
    # Group by 'Predictability.in.OP' and summarize
    count_hlz = df_hlz.groupby('Predictability.in.OP').agg(
        n=('Predictability.in.OP', 'size'),
        covered=('covered', 'sum')
    ).reset_index()

    # Calculate the percentage coverage
    count_hlz['pctcov'] = (count_hlz['covered'] / count_hlz['n']) * 100

    print("-----count_us-----")
    mapping={"Predictability in OP":'Predictability.in.OP' }
    count_us.rename(columns=mapping, inplace = True)
    #merge
    tab_n=(
        count_us
        .merge(count_mp[['Predictability.in.OP', 'n']].rename(columns={'n': 'mp'}), on='Predictability.in.OP', how='outer')
        .merge(count_ghz[['Predictability.in.OP', 'n']].rename(columns={'n': 'ghz'}), on='Predictability.in.OP', how='outer')
        .merge(count_hlz[['Predictability.in.OP', 'n']].rename(columns={'n': 'hlz'}), on='Predictability.in.OP', how='outer')
        .merge(count_hxz[['Predictability.in.OP', 'n']].rename(columns={'n': 'hxz'}), on='Predictability.in.OP', how='outer')
        )
    # Replace NaN values with 0
    tab_n = tab_n.fillna(0)

    # Sort the DataFrame by 'Predictability.in.OP'
    tab_n = tab_n.sort_values(by='Predictability.in.OP')

    
    # Assuming count_mp, count_ghz, count_hlz, and count_hxz are already defined

    # Perform the full joins
    tab_pctcov = (
        count_mp[['Predictability.in.OP', 'pctcov']].rename(columns={'pctcov': 'mp'})
        .merge(count_ghz[['Predictability.in.OP', 'pctcov']].rename(columns={'pctcov': 'ghz'}), on='Predictability.in.OP', how='outer')
        .merge(count_hlz[['Predictability.in.OP', 'pctcov']].rename(columns={'pctcov': 'hlz'}), on='Predictability.in.OP', how='outer')
        .merge(count_hxz[['Predictability.in.OP', 'pctcov']].rename(columns={'pctcov': 'hxz'}), on='Predictability.in.OP', how='outer')
    )

    # Replace NaN values with 0
    tab_pctcov = tab_pctcov.fillna(0)


    # Write to Excel
    with pd.ExcelWriter(os.path.join(pathResults, 'coverage.xlsx')) as writer:
        tab_n.to_excel(writer, sheet_name='n', index=False)
        tab_pctcov.to_excel(writer, sheet_name='pctcov', index=False)

    
    # Figure 1 (stock): Correlations (stock level) ----------------------------
    # Check which signals have been created

    check_signals()

    all_documentation=read_documentation()
    # Focus on Predictors

    prds=all_documentation.loc[all_documentation['Cat.Signal']=='Predictor','signalname'].tolist()
    signs=all_documentation.loc[all_documentation['Cat.Signal']=='Predictor', 'Sign'].tolist()

    print("---------pathPredictors----------")
    print(pathPredictors)

    # Create table with all Predictors
    signals=pd.read_csv(f"{pathPredictors}{prds[0]}.csv")[['permno', 'yyyymm']]

    print('---------------signals----------------')
    print(signals.columns)
    print(signals.head())
    for i in range(len(prds)):
        print(i, prds[i])
        if os.path.exists(f"{pathPredictors}{prds[i]}.csv"):
            tempin = pd.read_csv(f"{pathPredictors}{prds[i]}.csv")
            tempin.iloc[:,2]=signs[i]*tempin.iloc[:, 2]

            signals = pd.merge(signals, tempin, how='outer')

        else:
            print(f"{prds[i]}, does not exist in Data/Predictors folder")

    # Create loop list (there's probably an easier way to get unique pairs)

    prds = signals.drop(columns=['permno', 'yyyymm']).columns.tolist()
    
    product_list = list(itertools.product(range(1, len(prds) + 1), range(1, len(prds) + 1)))

    temp1 = pd.DataFrame(product_list, columns=['Var1', 'Var2'])
    temp1 = temp1[temp1['Var1'] > temp1['Var2']]

    loop_list = pd.DataFrame({
    'Var1': [prds[i - 1] for i in temp1['Var1']], 
    'Var2': [prds[i - 1] for i in temp1['Var2']]   
    })

    # Fig 1a: Pairwise rank correlation of signals

    # Save data to go easier on memory
    signals.to_csv(os.path.join(pathDataIntermediate, 'temp.csv'))
    del signals
    

    
    
    temp = parallel_correlation(loop_list)

    # Convert temp to a pandas DataFrame
    df = pd.DataFrame({'rho': temp})

    #create plot
    plt.figure(figsize=(8, 6))
    sns.histplot(df['rho'], bins=30, kde=False)

    plt.xlabel('Pairwise rank correlation')
    plt.ylabel('Count')

    sns.set_theme(style="whitegrid")

    plt.savefig(f"{pathResults}fig1aStock_pairwisecorrelations.pdf", 
                format='pdf', 
                bbox_inches='tight')
    

    global allRhos 
    allRhos= pd.DataFrame({
    'rho': temp,
    'series': 'Pairwise'
    })

    vars=['Size', 'BM', 'Mom12m','GP','AssetGrowth']
    
    for vv in vars:
        print(vv)
        #Create grid
        product_list = list(itertools.product(prds, vv))
        loopList = pd.DataFrame(product_list, columns=['Var1', 'Var2'])
        loopList = loopList[loopList['Var1'] != loopList['Var2']]

        # Get the number of available cores
        cores = multiprocessing.cpu_count()

        with multiprocessing.Pool(cores - 1) as pool:
            # Compute the correlations in parallel
            rhos = pool.map(process_pair, range(len(loopList)))
            rhos = np.array(rhos)

        #Create histogram
        df = pd.DataFrame({'rho': rhos})
        plt.figure(figsize=(10, 6))
        sns.histplot(df['rho'], bins=25, kde=False)
        plt.xlim(-1, 1)
        plt.xlabel('Pairwise rank correlation', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        sns.set_theme(style="whitegrid")

        plt.savefig(f'{pathResults}fig1SignalCorrelationWith_{vv}.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        
        allRhos = pd.concat([allRhos, pd.DataFrame({
            'rho': rhos,
            'series': vv
            })], ignore_index=True)
    # Plot all correlations on the same axis with faceting by 'series'
    if len(allRhos.index)!=0:
        plt.figure(figsize=(10, 8))
        g = sns.FacetGrid(allRhos, col="series", col_wrap=3, sharey=False, height=4)
        g.map(sns.histplot, "rho", bins=25)
        g.set(xlim=(-1, 1))

        # Add labels
        g.set_axis_labels("Correlation coefficient", "Count")

        # Apply a minimal theme
        sns.set_theme(style="whitegrid")

        # Save the plot as a PDF file
        plt.savefig(os.path.join(pathResults, 'fig1Stock_jointly.pdf'), format='pdf', bbox_inches='tight')

        # Optionally, close the plot if you're done with it
        plt.close()

        # Save the DataFrame as a binary file
        with open(os.path.join(pathResults, 'rhoStockLevel.pkl'), 'wb') as file:
            pickle.dump(allRhos, file)


            

        
        

        



