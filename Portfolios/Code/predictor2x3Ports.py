from fst import read_fst
import pandas as pd
import numpy as np


from globals import pathProject, pathDataPortfolios

# ENVIRONMENT AND DATA ====
crspinfo = read_fst(f"{pathProject}Portfolios/Data/Intermediate/crspminfo.fst").to_pandas()
crspret = read_fst(f"{pathProject}'Portfolios/Data/Intermediate/crspmret.fst").to_pandas()

# SELECT SIGNALS 
strategylist0 = alldocumentation[
    (alldocumentation['Cat.Signal'] == "Predictor") &
    (alldocumentation['Cat.Form'] == 'continuous')
]
strategylist = ifquickrun()

# FUNCTION FOR CONVERTING SIGNALNAME TO 2X3 PORTS ====
# analogous to signalname_to_ports in portfolioFunction.py

def signalname_to_2x3(signalname):
    # Import signal and sign
    sign_value = strategylist['Sign'].iloc[s]

    # Assuming import_signal is a defined function in Python
    signal = import_signal(signalname, None, sign_value)

    # ASSIGN TO 2X3 PORTFOLIOS 
    # Keep value of signal corresponding to June.
    signaljune = signal[signal['yyyymm'] % 100 == 6]

    # For NYSE subset, compute signal quantiles for high and low
    # as well as median ME. FF93 is unclear about the shrcd screen
    # here, but WRDS does it

    filtered_signaljune = signaljune[(signaljune['exchcd'] == 1) & (signaljune['shrcd'].isin([10, 11]))]

    # Group by 'yyyymm' and calculate the desired quantiles
    nysebreaks = filtered_signaljune.groupby('yyyymm').agg(
        qsignal_l=('signal', lambda x: x.quantile(0.3)),
        qsignal_h=('signal', lambda x: x.quantile(0.7)),
        qme_mid=('me', lambda x: x.quantile(0.5))
    ).reset_index()

    filtered_signaljune = signaljune[(signaljune['exchcd'].isin([1, 2, 3])) & 
                                 (signaljune['shrcd'].isin([10, 11]))]

    # Perform a left join with nysebreaks on the 'yyyymm' column
    merged_df = filtered_signaljune.merge(nysebreaks, on='yyyymm', how='left')

    # Create new columns based on conditions
    merged_df['q_signal'] = pd.cut(
        merged_df['signal'], 
        bins=[-float('inf'), merged_df['qsignal_l'], merged_df['qsignal_h'], float('inf')],
        labels=['L', 'M', 'H']
    )

    merged_df['q_me'] = pd.cut(
        merged_df['me'], 
        bins=[-float('inf'), merged_df['qme_mid'], float('inf')],
        labels=['S', 'B']
    )

    merged_df['port6'] = merged_df['q_me'] + merged_df['q_signal']

    # Select the desired columns
    port6 = merged_df[['permno', 'yyyymm', 'port6', 'signal']]

    # FIND MONTHLY FACTOR RETURNS 
    # Find VW returns, signal lag, and number of firms
    # for a given portfolio

    port6ret = crspret[['permno', 'date', 'yyyymm', 'ret', 'melag']].merge(port6, on=['permno', 'yyyymm'], how='left')

    # Group by permno, sort by permno and date, fill missing values, and create lagged columns
    port6ret = port6ret.groupby('permno').apply(lambda x: x.sort_values('date')).reset_index(drop=True)
    port6ret['port6'] = port6ret.groupby('permno')['port6'].ffill()
    port6ret['signal'] = port6ret.groupby('permno')['signal'].ffill()
    port6ret['port6_lag'] = port6ret.groupby('permno')['port6'].shift()
    port6ret['signal_lag'] = port6ret.groupby('permno')['signal'].shift()

    # Filter out rows where melag is NaN
    port6ret = port6ret.dropna(subset=['melag'])

    # Group by port6_lag and date, and calculate value-weighted returns and signal lag
    port6ret_grouped = port6ret.groupby(['port6_lag', 'date']).agg(
        ret_vw=('ret', lambda x: pd.Series(x).multiply(port6ret.loc[x.index, 'melag']).sum() / port6ret.loc[x.index, 'melag'].sum()),
        signallag=('signal_lag', lambda x: pd.Series(x).multiply(port6ret.loc[x.index, 'melag']).sum() / port6ret.loc[x.index, 'melag'].sum()),
        n_firms=('permno', 'count')
    ).reset_index()

    # Rename columns and mutate new ones
    port6ret_grouped.rename(columns={
        'port6_lag': 'port',
        'ret_vw': 'ret',
        'n_firms': 'Nlong'
    }, inplace=True)

    port6ret_grouped['signalname'] = signalname
    port6ret_grouped['Nshort'] = 0

    # Filter for specific ports and select relevant columns
    port6ret = port6ret_grouped[port6ret_grouped['port'].isin(["SL", "SM", "SH", "BL", "BM", "BH"])][
        ['signalname', 'port', 'date', 'ret', 'signallag', 'Nlong', 'Nshort']
    ]

    # Equal-weight extreme portfolios to make FF1993-style factor
    portls_ret = port6ret[['port', 'date', 'ret']]

    # Pivot the DataFrame to get wider format
    portls_ret_wide = portls_ret.pivot(index='date', columns='port', values='ret').reset_index()

    # Calculate the new 'ret' column
    portls_ret_wide['ret'] = 0.5 * (portls_ret_wide['SH'] + portls_ret_wide['BH']) - 0.5 * (portls_ret_wide['SL'] + portls_ret_wide['BL'])

    # Select the final columns
    portls_ret = portls_ret_wide[['date', 'ret']]


    # Get number of firms in long-short stocks
    portls_N = port6ret[['port', 'date', 'Nlong']]

    # Pivot the DataFrame to get wider format
    portls_N_wide = portls_N.pivot(index='date', columns='port', values='Nlong').reset_index()

    # Calculate the new 'Nlong' and 'Nshort' columns
    portls_N_wide['Nlong'] = portls_N_wide['SH'] + portls_N_wide['BH']
    portls_N_wide['Nshort'] = portls_N_wide['SL'] + portls_N_wide['BL']

    # Select the final columns
    portls_N = portls_N_wide[['date', 'Nlong', 'Nshort']]

    # merge returns with number of firms and fill in LS info
    portls = pd.merge(portls_ret, portls_N, on='date')

    # Mutate the DataFrame by adding and modifying columns
    portls['signalname'] = signalname
    portls['port'] = "LS"
    portls['signallag'] = None
    portls['Nshort'] = portls.apply(lambda row: None if pd.isna(row['ret']) else row['Nshort'], axis=1)
    portls['Nlong'] = portls.apply(lambda row: None if pd.isna(row['ret']) else row['Nlong'], axis=1)

    # Select the final columns
    portls = portls[['signalname', 'port', 'date', 'ret', 'signallag', 'Nlong', 'Nshort']]

    port = pd.concat([port6ret, portls], ignore_index=True)

    # Convert 'port' to a categorical type with specified levels
    port['port'] = pd.Categorical(
        port['port'], 
        categories=["SL", "SM", "SH", "BL", "BM", "BH", "LS"],
        ordered=True
    )

    # Sort the DataFrame by 'port' and 'date'
    port = port.sort_values(by=['port', 'date']).reset_index(drop=True)

# LOOP OVER SIGNALS ====
num_signals = len(strategylist)
allport=[]

for s in range(num_signals):
    signalname = strategylist['signalname'].iloc[s]

    print(f"Processing Signal No. {s} ===> {signalname}")

    try:
        tempport = signalname_to_2x3(
            signalname=strategylist['signalname'].iloc[s]
        )
    except:
        print("Error in signalname_to_2x3, returning df with NA")

        # Assuming allport[s-1] exists and has been defined before
        ncols = allport[s-1].shape[1] if len(allport) > 0 and s > 0 else 0
        tempport = pd.DataFrame(np.nan, index=[0], columns=range(ncols))
    
    # add column names if signalname_to_2x3 failed
    if pd.isna(tempport.iloc[0, 0]):
        # Set column names to match those of the first element in allport
        tempport.columns = allport[0].columns
        
        # Set the signalname to the one from the first strategy
        tempport['signalname'] = strategylist['signalname'].iloc[0]
    allport[s] = tempport

    port = pd.concat(allport, ignore_index=True)

    # WRITE TO DISK  ====
    output_path = f"{pathDataPortfolios}/PredictorAltPorts_FF93style.csv"
    port.to_csv(output_path, index=False)

