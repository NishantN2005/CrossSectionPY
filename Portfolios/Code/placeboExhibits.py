import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tabulate import tabulate

from globals import pathDataPortfolios, pathResults,pathProject

# Predictor t-stat in extended dataset ------------------------------------

def relevent_set(readdocumentation):
    # Define relevant set
    docnew = readdocumentation()
    docnew = docnew[docnew['Predictability.in.OP'] != '9_drop']
    docnew['Category'] = pd.Categorical(
        docnew['Predictability.in.OP'],
        categories=["indirect", "4_not", "3_maybe", "2_likely", "1_clear"],
        ordered=True
    )
    docnew['Category'] = docnew['Category'].cat.rename_categories({
        "indirect": "Indirect Evidence",
        "4_not": "Not Predictor",
        "3_maybe": "maybe",
        "2_likely": "Likely Predictor",
        "1_clear": "Clear Predictor"
    })

    # Add stats
    stats = pd.read_excel(f"{pathDataPortfolios}/PredictorSummary.xlsx", sheet_name='short')
    stats['success'] = (stats['tstat'].round(2) >= 1.96).astype(int)
    stats = stats[['signalname', 'success', 'tstat', 'rbar']]

    placebo_stats = pd.read_excel(f"{pathDataPortfolios}/PlaceboSummary.xlsx", sheet_name='ls_insamp_only')
    placebo_stats['success'] = (placebo_stats['tstat'].round(2) >= 1.96).astype(int)
    placebo_stats = placebo_stats[['signalname', 'success', 'tstat', 'rbar']]

    stats = pd.concat([stats, placebo_stats], ignore_index=True)

    
    statsFull = pd.read_excel(f"{pathDataPortfolios}/PredictorSummary.xlsx", sheet_name='full')
    statsFull = statsFull[(statsFull['samptype'] == 'postpub') & (statsFull['port'] == 'LS')]
    statsFull = statsFull[['signalname', 'tstat']].rename(columns={'tstat': 't-stat PS'})

    placebo_statsFull = pd.read_excel(f"{pathDataPortfolios}/PlaceboSummary.xlsx", sheet_name='full')
    placebo_statsFull = placebo_statsFull[(placebo_statsFull['samptype'] == 'postpub') & (placebo_statsFull['port'] == 'LS')]
    placebo_statsFull = placebo_statsFull[['signalname', 'tstat']].rename(columns={'tstat': 't-stat PS'})

    statsFull = pd.concat([statsFull, placebo_statsFull], ignore_index=True)

    # Merge data
    df_merge = pd.merge(
        docnew, 
        stats[['signalname', 'tstat', 'rbar']], 
        on='signalname', 
        how='left'
    )
    df_merge = pd.merge(df_merge, statsFull, on='signalname', how='left')

    df_merge = df_merge.assign(
        ref=lambda x: x['Authors'] + ' (' + x['Year'].astype(str) + ')',
        Predictor=lambda x: x['LongDescription'],
        sample=lambda x: x['SampleStartYear'].astype(str) + '-' + x['SampleEndYear'].astype(str),
        **{
            'Mean Return': df_merge['rbar'].round(2),
            't-stat IS': df_merge['tstat'].round(2),
            'Evidence': df_merge['Evidence.Summary'],
            'Category': df_merge['Category']
        }
    )

    df_merge['ref'] = df_merge['ref'].replace('NA (NA)', '')

    df_merge = df_merge.sort_values(by='ref').reset_index(drop=True)

    df_merge = df_merge[[
        'ref', 'Predictor', 'signalname', 'sample', 'Mean Return', 
        't-stat IS', 'Evidence', 'Category'
    ]]

    df_plot = df_merge.assign(tstat=df_merge['t-stat IS'].abs())[["Category", "tstat"]]

    plt.figure(figsize=(10, 8))
    sns.stripplot(x='tstat', y='Category', data=df_plot, jitter=True, size=5, linewidth=0.5)
    plt.axvline(x=1.96, color='black', linestyle='--')
    plt.xlabel('t-statistic')
    plt.ylabel('')

    plt.style.use('seaborn-whitegrid')

    output_filename = f"{pathResults}/fig2b_reprate_PredictorPlacebo_Jitter.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    # Create Latex output table 2: Placebos
    temp = df_merge[
        df_merge['Category'].isin(['Not Predictor', 'Indirect Evidence'])
    ].sort_values(by='ref')[
        ['ref', 'Predictor', 'signalname', 'Category', 'Mean Return', 't-stat IS', 'Evidence']
    ]
    outputtable2 = tabulate(temp, headers='keys', tablefmt='grid')


    outputtable2 = tabulate(temp.values, tablefmt="latex", stralign="center")
    output_path = f"{pathResults}/bigSignalTablePlacebos.tex"
    with open(output_path, "w") as f:
        f.write(outputtable2)

def mclean_plontiff_figs(alldocumentation):
    # McLean and Pontiff style graphs -----------------------------------------
    # (placed after placebo creation because we classify a few of MP's predictors as placebos)

    # stats
    stats_short = pd.read_excel(
        f"{pathDataPortfolios}/PredictorSummary.xlsx", 
        sheet_name='short'
    )[['signalname', 'tstat', 'rbar']]
    stats_ls_insamp = pd.read_excel(
        f"{pathDataPortfolios}/PlaceboSummary.xlsx", 
        sheet_name='ls_insamp_only'
    )[['signalname', 'tstat', 'rbar']]
    stats = pd.concat([stats_short, stats_ls_insamp], ignore_index=True)

    #statsFull
    stats_full_predictor = pd.read_excel(
        f"{pathDataPortfolios}/PredictorSummary.xlsx", 
        sheet_name='full'
    )
    stats_full_predictor = stats_full_predictor[
        (stats_full_predictor['samptype'] == 'postpub') & 
        (stats_full_predictor['port'] == 'LS')
    ][['signalname', 'tstat', 'rbar']]
    stats_full_placebo = pd.read_excel(
        f"{pathDataPortfolios}/PlaceboSummary.xlsx", 
        sheet_name='full'
    )
    stats_full_placebo = stats_full_placebo[
        (stats_full_placebo['samptype'] == 'postpub') & 
        (stats_full_placebo['port'] == 'LS')
    ][['signalname', 'tstat', 'rbar']]

    statsFull = pd.concat([stats_full_predictor, stats_full_placebo], ignore_index=True)

    mpSignals = pd.read_csv(f"{pathProject}/Comparison_to_MetaReplications.csv")
    mpSignals = mpSignals[
        (mpSignals['metastudy'] == 'MP') & 
        (mpSignals['ourname'] != '_missing_')
    ]

    # Merge data
    alldocumentation['inMP'] = alldocumentation['signalname'].isin(mpSignals['ourname'])

    df_merge = alldocumentation[
        (alldocumentation['Cat.Signal'] == 'Predictor') | (alldocumentation['inMP'])
    ]
    df_merge = df_merge.merge(stats, on="signalname", how="left")
    df_merge = df_merge.merge(
        statsFull[['signalname', 'tstat', 'rbar']].rename(columns={'tstat': 'tstatPS', 'rbar': 'rbarPS'}), 
        on="signalname", 
        how="left"
    )
    cols_to_modify = ['tstat', 'tstatPS', 'rbar', 'rbarPS']
    df_merge[cols_to_modify] = df_merge[cols_to_modify].applymap(lambda x: abs(x) if x < 0 else x)

    df_merge = df_merge.assign(
        DeclineTstat=df_merge['tstat'] - df_merge['tstatPS'],
        DeclineRBar=df_merge['rbar'] - df_merge['rbarPS'],
        Category=pd.Categorical(
            df_merge['Predictability.in.OP'],
            categories=["indirect", "4_not", "3_maybe", "2_likely", "1_clear"],
            ordered=True
        ).rename_categories({
            "indirect": "no evidence", "4_not": "not", "3_maybe": "maybe", "2_likely": "likely", "1_clear": "clear"
        }),
        CatPredPlacebo=df_merge['Cat.Signal']
    )[['signalname', 'tstat', 'tstatPS', 'DeclineTstat', 'rbar', 'rbarPS', 'DeclineRBar', 'Category', 'CatPredPlacebo', 'inMP']]

    df_merge = df_merge[
        (df_merge['signalname'] != 'IO_ShortInterest') &
        (df_merge['Category'].isin(['clear', 'likely']))
    ]

    # In-sample return
    df_merge['inMPStr'] = np.where(df_merge['inMP'], 'in MP (2016)', 'not in MP (2016)')
    df_merge['inMPStr'] = np.where(df_merge['inMP'], 'in MP (2016)', 'not in MP (2016)')

    fig, axs = plt.subplots(nrow=2, ncol=1, figsize=(10, 16))

    # First subplot: Plotting the "plotret"
    sns.regplot(x='DeclineRBar', y='rbar', data=df_merge, scatter=False, color='black', ax=axs[0])
    sns.scatterplot(x='DeclineRBar', y='rbar', hue='CatPredPlacebo', style='inMPStr', 
                    data=df_merge, s=100, palette=['none', 'black'], 
                    edgecolor='black', markers=['o', '^'], legend=False, ax=axs[0])
    axs[0].plot([-1, 2], [-1, 2], linestyle='dotted', color='black')
    axs[0].axhline(0, color='black', linestyle='-')
    axs[0].axvline(0, color='black', linestyle='-')
    axs[0].set_xlim(-1.0, 2)
    axs[0].set_ylim(0, 2.5)
    axs[0].set_xlabel('Decline in return post-publication')
    axs[0].set_ylabel('In-Sample return')
    axs[0].legend(title='', loc='upper left', bbox_to_anchor=(0, 1))

    # Second subplot: Plotting the "plott"
    sns.regplot(x='DeclineRBar', y='tstat', data=df_merge, scatter=False, color='black', ax=axs[1])
    sns.scatterplot(x='DeclineRBar', y='tstat', hue='CatPredPlacebo', style='inMPStr', 
                    data=df_merge, s=100, palette=['none', 'black'], 
                    edgecolor='black', markers=['o', '^'], legend=False, ax=axs[1])
    axs[1].axhline(0, color='black', linestyle='-')
    axs[1].axvline(0, color='black', linestyle='-')
    axs[1].set_xlim(-1.0, 2)
    axs[1].set_ylim(0, 14)
    axs[1].set_yticks(np.arange(0, 15, 2))
    axs[1].set_xlabel('Decline in return post-publication')
    axs[1].set_ylabel('In-Sample t-statistic')
    axs[1].legend(title='', loc='upper left', bbox_to_anchor=(0, 1))

    # Adjust layout to avoid overlap
    plt.tight_layout()

    output_filename = f"{pathResults}/fig5_MP_both.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')


    # manual inspection 
    result = df_merge[df_merge['inMP']].loc[:, ['signalname', 'tstat', 'Category']].sort_values(by='tstat')
    result = df_merge[df_merge['inMP']].agg({
        'rbar': ['mean', 'std'],
        'tstat': lambda x: (x > 1.5).sum()
    })
    result.columns = ['mean_rbar', 'sd_rbar', 'sum_tstat_greater_1_5']

def replication_rate(readdocumentation):
    # Replication rate vis-a-vis other studies --------------------------------
    mpSignals = pd.read_csv(f"{pathProject}/Comparison_to_MetaReplications.csv")

    mpSignals = mpSignals[(mpSignals['metastudy'] == 'MP') & (mpSignals['ourname'] != '_missing_')]

    hxzSignals = pd.read_csv(f"{pathProject}/Comparison_to_MetaReplications.csv")

    hxzSignals = hxzSignals[(hxzSignals['metastudy'] == 'HXZ') & (hxzSignals['ourname'] != '_missing_')]


    stats_short = pd.read_excel(f"{pathDataPortfolios}/PredictorSummary.xlsx", sheet_name='short')
    stats_ls_insamp = pd.read_excel(f"{pathDataPortfolios}/PlaceboSummary.xlsx", sheet_name='ls_insamp_only')
    stats_short = stats_short[['signalname', 'tstat', 'rbar']]
    stats_ls_insamp = stats_ls_insamp[['signalname', 'tstat', 'rbar']]
    stats = pd.concat([stats_short, stats_ls_insamp], ignore_index=True)
    documentation = readdocumentation()[['signalname', 'Cat.Signal', 'Predictability.in.OP']]

    documentation['Cat.Signal'] = documentation['Cat.Signal'].replace({
        'Predictor': 'Clear or Likely',
        'Placebo': 'Indirect or Not'
    })
    stats = stats.merge(documentation, on='signalname', how='left')


    df_tmp = stats.assign(
        tstat=stats['tstat'].abs(),  # Take the absolute value of tstat
        PredOP=pd.Categorical(
            stats['Predictability.in.OP'], 
            categories=['1_clear', '2_likely', '3_maybe', 'indirect', '4_not'],
            ordered=True
        ).rename_categories({
            '1_clear': 'Clear Predictor', 
            '2_likely': 'Likely Predictor', 
            '3_maybe': 'Indirect Signal', 
            'indirect': 'Indirect Signal', 
            '4_not': 'Not Predictor'
        }),
        inMP=stats['signalname'].isin(mpSignals['ourname']),  # Flag for whether in MP
        inHXZ=stats['signalname'].isin(hxzSignals['ourname'])  # Flag for whether in HXZ
    )[['signalname', 'tstat', 'PredOP', 'Cat.Signal', 'inMP', 'inHXZ']]

    # Our study
    df_tmp['PredOP'] = pd.Categorical(df_tmp['PredOP'], 
                                    categories=reversed(df_tmp['PredOP'].cat.categories),
                                    ordered=True)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PredOP', y='tstat', hue='Cat.Signal', style='Cat.Signal', data=df_tmp,
        markers={'Clear or Likely': 'o', 'Indirect or Not': 'D'}, s=100
    )
    plt.axhline(y=1.96, color='black', linestyle='--')

    plt.gca().invert_yaxis()
    plt.ylabel('t-statistic')
    plt.xlabel('')
    plt.style.use('seaborn-whitegrid')

    plt.legend(title='', loc='lower right', bbox_to_anchor=(0.8, 0.1))

    output_filename = f"{pathResults}/fig_reprate_ourstudy.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    # HXZ
    df_filtered = df_tmp[df_tmp['inHXZ']]
    df_filtered['PredOP'] = pd.Categorical(df_filtered['PredOP'], 
                                        categories=reversed(df_filtered['PredOP'].cat.categories),
                                        ordered=True)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PredOP', y='tstat', hue='Cat.Signal', style='Cat.Signal', data=df_filtered,
        markers={'Clear or Likely': 'o', 'Indirect or Not': 'D'}, s=150, edgecolor='black'
    )
    plt.axhline(y=1.96, color='black', linestyle='--')
    plt.gca().invert_yaxis()
    plt.ylabel('t-statistic')
    plt.xlabel('')
    plt.style.use('seaborn-whitegrid')
    plt.legend(title='', loc='upper left', bbox_to_anchor=(0.8, 0.15), frameon=False)

    output_filename = f"{pathResults}/fig_reprate_HXZ.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    # MP
    df_filtered = df_tmp[df_tmp['inMP']]
    df_filtered['PredOP'] = pd.Categorical(df_filtered['PredOP'], 
                                        categories=reversed(df_filtered['PredOP'].cat.categories),
                                        ordered=True)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PredOP', y='tstat', hue='Cat.Signal', style='Cat.Signal', data=df_filtered,
        markers={'Clear or Likely': 'o', 'Indirect or Not': 'D'}, s=150, edgecolor='black'
    )
    plt.axhline(y=1.96, color='black', linestyle='--')

    plt.gca().invert_yaxis()

    plt.ylabel('t-statistic')
    plt.xlabel('')
    plt.style.use('seaborn-whitegrid')
    plt.legend(title='', loc='upper left', bbox_to_anchor=(0.8, 0.2), frameon=False)

    output_filename = f"{pathResults}/fig_reprate_MP.pdf"
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')

    # manual counts
    grouped_summary = df_tmp[df_tmp['inHXZ']].groupby('PredOP').agg(
        tstat_below_1_96=('tstat', lambda x: (x < 1.96).sum()),
        total_count=('tstat', 'size')
    )
    summary = df_tmp[df_tmp['inHXZ']].agg(
        fail=('tstat', lambda x: (x < 1.96).sum()),
        total=('tstat', 'size')
    )
    summary['fail_proportion'] = summary['fail'] / summary['total']
    filtered_arranged = df_tmp[(df_tmp['inHXZ']) & (df_tmp['PredOP'] == 'Clear')].sort_values(by='tstat')

    summary = df_tmp[df_tmp['inMP']].agg(
        below_1_5=('tstat', lambda x: (x < 1.5).sum() / len(x)),
        below_1_96=('tstat', lambda x: (x < 1.96).sum() / len(x))
    )

    summary = df_tmp[(df_tmp['inMP']) & (df_tmp['PredOP'] == 'Clear')].agg(
        below_1_5=('tstat', lambda x: (x < 1.5).sum() / len(x)),
        below_1_96=('tstat', lambda x: (x < 1.96).sum() / len(x))
    )

