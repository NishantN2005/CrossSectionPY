import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec


from globals import pathDataPortfolios, pathResults

def process(sum_port_month):
    ### LOAD PORT-MONTH RETURNS AND SUMMARIZE ----

    ## summarize alt holding period 
    holdperlist = ['1','3','6','12']
    sumholdper = pd.DataFrame()

    for i in range(len(holdperlist)):
        tempport = pd.read_csv(f"{pathDataPortfolios}PredictorAltPorts_HoldPer_{holdperlist[i]}.csv")
        tempport = tempport[tempport['port'] == 'LS']

        tempsum = sum_port_month(tempport, ['signalname', 'samptype', 'port'], Nstocksmin=20)
        tempsum = pd.DataFrame(tempsum).assign(portperiod=holdperlist[i])
        tempsum = tempsum[tempsum['samptype'] == 'insamp']

        sumholdper = pd.concat([sumholdper, tempsum], ignore_index=True)
    # add baseline
    tempport = pd.read_csv(f"{pathDataPortfolios}PredictorPortsFull.csv")
    tempport = tempport[tempport['port'] == 'LS']

    tempsum = sum_port_month(tempport, ['signalname', 'samptype', 'port'], Nstocksmin=20)
    tempsum = pd.DataFrame(tempsum)  # Convert to DataFrame if sumportmonth doesn't return one
    tempsum['portperiod'] = 'base'
    tempsum = tempsum[tempsum['samptype'] == 'insamp']

    sumholdper = pd.concat([sumholdper, tempsum], ignore_index=True)


    ## summarize alt liq screens
    csvlist = [
        'PredictorAltPorts_LiqScreen_ME_gt_NYSE20pct.csv',
        'PredictorAltPorts_LiqScreen_NYSEonly.csv',
        'PredictorAltPorts_LiqScreen_Price_gt_5.csv',
        'PredictorAltPorts_LiqScreen_VWforce.csv'
    ]
    screenlist = ['me','nyse','price','vwforce']
    sumliqscreen = pd.DataFrame()

    for i in range(len(csvlist)):
        tempport = pd.read_csv(f"{pathDataPortfolios}/{csvlist[i]}")
        tempport = tempport[tempport['port'] == 'LS']

        tempsum = sum_port_month(tempport, ['signalname', 'samptype', 'port'], Nstocksmin=20)
        tempsum['screen'] = screenlist[i]
        tempsum = tempsum[tempsum['samptype'] == 'insamp']

        sumliqscreen = pd.concat([sumliqscreen, tempsum], ignore_index=True)

    # add baseline
    tempport = pd.read_csv(f"{pathDataPortfolios}/PredictorPortsFull.csv")
    tempport = tempport[tempport['port'] == 'LS']

    tempsum = sum_port_month(tempport, group_by_columns=['signalname', 'samptype', 'port'], Nstocksmin=20)
    tempsum = pd.DataFrame(tempsum)
    tempsum['screen'] = 'none'
    tempsum = tempsum[tempsum['samptype'] == 'insamp']

    sumliqscreen = pd.concat([sumliqscreen, tempsum], ignore_index=True)

    ## Summarize decile sorts
    portDeciles = pd.read_csv(f'{pathDataPortfolios}/PredictorAltPorts_Deciles.csv')

    sumDeciles = sum_port_month(portDeciles, ['signalname', 'samptype', 'port'], Nstocksmin=20)
    sumDeciles = pd.DataFrame(sumDeciles)

    # Figures: Monotonicity  -------------------------------------------------
    filtered_df = sumDeciles[(sumDeciles['port'] != 'LS') & (sumDeciles['samptype'] == 'insamp') & (sumDeciles['Nlong'] > 100)]

    # Apply the mutate equivalent in Python
    filtered_df['Increase'] = filtered_df.groupby('signalname')['rbar'].apply(lambda x: ['Increase' if (i != '01' and x.iloc[n] >= x.iloc[n-1]) else 'No increase' for n, i in enumerate(x.index)])

    # Plotting using seaborn or matplotlib
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='port', y='rbar', data=filtered_df, showfliers=False, color='lightgray', width=0.5)
    sns.stripplot(x='port', y='rbar', data=filtered_df, hue='Increase', jitter=True, dodge=True, marker='o', size=8)

    plt.xlabel('Decile Portfolio')
    plt.ylabel('Mean Return (% monthly, in-sample)')
    plt.legend(title='', loc='upper left', bbox_to_anchor=(0.2, 0.8))

    # Save
    plt.savefig(f'{pathResults}/fig_mono.pdf', format='pdf', dpi=300, bbox_inches='tight')

    df = sumholdper[sumholdper['port'] == 'LS']

    xlevels = ['base', '1', '3', '6', '12']
    xlabels = ['Original Papers', '1 month', '3 months', '6 months', '12 months']

    df['portperiod'] = pd.Categorical(df['portperiod'], categories=xlevels, ordered=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Jitter plot
    sns.stripplot(x='portperiod', y='rbar', data=df, jitter=True, size=5, color='black')
    sns.boxplot(x='portperiod', y='rbar', data=df, whis=[0, 100], width=0.6, showfliers=False)
    plt.xlabel('Rebalancing frequency')
    plt.ylabel('Mean Return (% monthly, in-sample)')
    plt.ylim(-0.5, 2)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Mean Return by Rebalancing Frequency')

    # Save the plot
    plt.savefig(f"{pathResults}/fig4_holding_period_boxplot_meanJitter.pdf", format='pdf', bbox_inches='tight')


    # Figures: Liquidity screens -----------------------------------------------

    df = sumliqscreen
    xlevels = ['none', 'price', 'nyse', 'me','vwforce']
    xlabels = ['Original Papers', 'Price>5', 'NYSE only', 'ME > NYSE 20 pct','VW force']

    # Jitter and boxplots
    df['screen'] = pd.Categorical(df['screen'], categories=xlevels, ordered=True)
    df['screen'].cat.rename_categories(xlabels, inplace=True)

    df_plot = df[['screen', 'rbar']].rename(columns={'rbar': 'Return'})
    df_melted = df_plot.melt(id_vars='screen', var_name='key', value_name='value')
    filename = f"{pathResults}fig4_liquidity_boxplot_meanJitter.pdf"

    with PdfPages(filename) as pdf:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='screen', y='value', data=df_melted, color='lightgray', fliersize=0)
        sns.stripplot(x='screen', y='value', data=df_melted, jitter=0.2, color='black', size=5)

        # Customize the plot
        plt.xlabel('Liquidity screen')
        plt.ylabel('Mean Return (% monthly, in-sample)')
        plt.ylim(-0.5, 2)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save to the specified PDF file
        pdf.savefig()
        plt.close()

    # Figures: Deciles --------------------------------------------------------

    filtered_df = sumDeciles[(sumDeciles['port'] != 'LS') & 
                            (sumDeciles['samptype'] == 'insamp') & 
                            (sumDeciles['rbar'] < 10)]

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='port', y='rbar', data=filtered_df, color='lightgray', fliersize=0)
    sns.stripplot(x='port', y='rbar', data=filtered_df, jitter=0.2, color='black', size=5)

    # Customize the plot
    plt.xlabel('Decile Portfolio')
    plt.ylabel('Mean Return in-sample (ppt per month)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot as a PDF
    filename = f"{pathResults}fig_Decile_boxplot_meanJitter.pdf"
    with PdfPages(filename) as pdf:
        pdf.savefig()
        plt.close()


    #### CHECK VW FOR QUINT AND DEC ####
    # not used in paper, but good to check
    df1 = pd.read_csv(f"{pathDataPortfolios}PredictorAltPorts_Deciles.csv")
    df1['q_cut'] = 0.1
    df1['sweight'] = 'EW'

    df2 = pd.read_csv(f"{pathDataPortfolios}PredictorAltPorts_DecilesVW.csv")
    df2['q_cut'] = 0.1
    df2['sweight'] = 'VW'

    df3 = pd.read_csv(f"{pathDataPortfolios}PredictorAltPorts_Quintiles.csv")
    df3['q_cut'] = 0.2
    df3['sweight'] = 'EW'

    df4 = pd.read_csv(f"{pathDataPortfolios}PredictorAltPorts_QuintilesVW.csv")
    df4['q_cut'] = 0.2
    df4['sweight'] = 'VW'

    # Combine all DataFrames
    all_data = pd.concat([df1, df2, df3, df4], ignore_index=True)

    sumall = sum_port_month(all_data, ['samptype', 'signalname', 'q_cut', 'sweight', 'port'])

    filtered_df = sumall[(sumall['port'] != 'LS') & (sumall['samptype'] == 'insamp')]
    sumimp = filtered_df.groupby(['q_cut', 'sweight', 'port']).agg(rbar=('rbar', 'mean')).reset_index()

    # Create the first plot (p1)
    sumimp_filtered_01 = sumimp[sumimp['q_cut'] == 0.1]
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    sns.lineplot(data=sumimp_filtered_01, x='port', y='rbar', hue='sweight', marker='o', ax=ax1, palette='tab10')
    ax1.set_title('Simple Check on Forced Quantile Implementations')
    ax1.set_xlabel('Port')
    ax1.set_ylabel('Rbar')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create the second plot (p2)
    sumimp_filtered_02 = sumimp[sumimp['q_cut'] == 0.2]
    ax2 = fig.add_subplot(gs[1])
    sns.lineplot(data=sumimp_filtered_02, x='port', y='rbar', hue='sweight', marker='o', ax=ax2, palette='tab10')
    ax2.set_xlabel('Port')
    ax2.set_ylabel('Rbar')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout
    plt.tight_layout()

    # Save the combined plot to a PDF
    filename = f"{pathResults}xfig_altquant_check.pdf"
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
        plt.close(fig)