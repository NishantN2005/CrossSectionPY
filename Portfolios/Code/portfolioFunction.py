from globals import pathPredictors, pathCRSPPredictors, pathPlacebos, pathtemp, verbose
import os
import pandas as pd
import numpy as np

#### FUNCTION FOR TURNING SIGNAL CSV TO K PORTFOLIOS

def import_signal(signal_name, filterstr, sign, crspinfo):
    """
    Load signal data from a CSV file, apply filters, and adjust signal values.

    Parameters:
    - signalname: str, name of the signal file (without '.csv')
    - filterstr: str, filter condition to apply to the signal data
    - Sign: int, factor to adjust the signal (e.g., 1 or -1)

    Returns:
    - DataFrame: Filtered and adjusted signal data with CRSP information joined.
    """
    # this is useful to have outside of signalname_to_longports in case
    # you just want to look at the signals

    ### LOAD SIGNAL DATA AND APPLY FILTERS
    if os.path.exists(f"{pathPredictors}{signal_name}.csv"):
        csv_name = f"{pathPredictors}{signal_name}.csv"
    elif os.path.exists(f"{pathCRSPPredictors}{signal_name}.csv"):
        csv_name = f"{pathCRSPPredictors}{signal_name}.csv"
    elif os.path.exists(f"{pathPlacebos}{signal_name}.csv"):
        csv_name = f"{pathPlacebos}{signal_name}.csv"
    elif os.path.exists(f"{pathtemp}{signal_name}.csv"):
        csv_name = f"{pathtemp}{signal_name}.csv"
    else:
        print('Error: signalname CSV not found')
        raise FileNotFoundError('Error: signalname CSV not found')
    

    # Load the signal data
    signal = pd.read_csv(csv_name)
    signal.rename(columns={signal_name: 'signal'}, inplace=True)
    signal = signal.dropna(subset=['signal'])

   # Add CRSP information (assuming crspinfo is a DataFrame in your workspace)
   #----------crspinfo is defined in another file in og----------
    signal = signal.merge(crspinfo, on=['permno', 'yyyymm'], how='left')

    # Apply filters and adjust signal values
    if str(filterstr) != 'nan':
        signal = signal.query(str(filterstr))

    signal['signal'] *= sign

    return signal

def signalname_to_ports(signal_name, crspret,crspinfo, cat_form=np.nan, q_cut=np.nan, sweight=np.nan,sign=np.nan, long_port_name=np.nan,
                        short_port_name=np.nan, start_month=np.nan, port_period=np.nan, q_filt=np.nan, filterstr=np.nan, 
                        passive_gain=False):
    """
    Process signal data to create K portfolios.
    
    Parameters:
    - signalname: str, the name of the signal
    - Cat_Form: str, category form ('continuous', 'discrete', 'custom')
    - q_cut: float, quantile cut for portfolio creation
    - sweight: str, weighting strategy ('EW' for equal weight, 'VW' for value weight)
    - Sign: int, signal adjustment factor
    - longportname: str or list, name of long portfolio
    - shortportname: str or list, name of short portfolio
    - startmonth: int, start month for rebalancing
    - portperiod: int, period of portfolio rebalancing
    - q_filt: str, filter for quantile selection (e.g., 'NYSE')
    - filterstr: str, filter condition applied to signals
    - passive_gain: bool, if True, apply passive gain adjustments

    Returns:
    - DataFrame: Final portfolio data
    """
    #Settings and checks

    if pd.isnull(sweight):
        sweight='EW'
    if pd.isnull(sign):
        sign=1
    if pd.isnull(long_port_name):
        long_port_name = 'max'
    if pd.isnull(short_port_name):
        short_port_name = 'min'
    if pd.isnull(start_month):
        start_month=6
    if pd.isnull(port_period):
        port_period = 1
    if pd.isnull(q_cut):
        q_cut=0.2
    if pd.isnull(cat_form):
        cat_form = 'continuous'
    if crspret.empty:
        print('Error: crspret not in workspace')
        raise ValueError('Please load Intermediate/crspmret.csv or crspdret.csv as crspret')
    
    if crspinfo.empty:
        print('Error: crspinfo not in workspace.')
        raise ValueError('Please load Intermediate/crspminfo.csv as crspinfo')

    if passive_gain and ('passgain' not in crspret.columns):
        print('Error: passive_gain = T but crspret does not have a passgain column')
        raise ValueError('Please check the correct crspret is loaded')

    if sweight == 'VW' and ('melag' not in crspret.columns):
        print('Error: sweight == VW but crspret does not have melag column')
        raise ValueError('Please check crsp processing')
    

    # function for sorting portfolios if signal is not already a port assignment
    def single_sort(q_filt, q_cut):
        ## create breakpoints
        # subset to firms used for breakpoints, right now only exclude based on q_filt

        #------honestly have no idea how signal is defined here--------------
        tempbreak= signal = import_signal(signal_name, filterstr, sign, crspinfo)
        if q_filt is not None and q_filt=='NYSE':
            tempbreak = tempbreak[tempbreak['exchcd'] == 1]

        ## create breakpoints
        # turn q_cut into a list carefully
        if q_cut <= 1/3:
            print("I AM IN Q_CUT<=1/3")
            print(q_cut)
            plist = list(np.arange(q_cut, 1 - 2 * q_cut, q_cut)) + [1 - q_cut]
        else:
            print("I AM IN ELSE")
            print(q_cut)
            plist = list(np.unique([q_cut, 1 - q_cut]))


        # find breakpoints 
        temp=[]
        # Loop over plist to calculate quantiles and store them in temp
        print('----------quantiles here----------')
        print(plist)
        for pi, prob in enumerate(plist, start=1):
            print(f"pi: {pi}, prob: {prob}")
            # Calculate quantiles and store in a DataFrame
            temp_df = tempbreak.groupby('yyyymm').apply(
                lambda x: pd.Series({'breakpoint': x['signal'].quantile(prob)})
            ).reset_index()

            # Add the 'breaki' column
            temp_df['breaki'] = pi

            # Append the result to the temp list
            temp.append(temp_df)

        breaklist = pd.concat(temp, ignore_index=True)
        breaklist_wide = breaklist.pivot(
            index='yyyymm',   # The column to keep as rows
            columns='breaki',  # The column to pivot (create new columns)
            values='breakpoint'  # The values to populate in the new columns
        )
        breaklist_wide.columns = ['break' + str(col) for col in breaklist_wide.columns]
        breaklist = breaklist_wide.reset_index()

        if len(plist) > 1:
            print("PLIST 162 HERE:", plist)
            print("BREAKLIST_WIDE")
            print(len(breaklist_wide))
            print(breaklist_wide.columns)
            idgood = breaklist_wide.iloc[:, len(plist)-1] - breaklist_wide.iloc[:, 1] > 0
            breaklist_wide = breaklist_wide[idgood]
        
        ## assign to portfolios
        # the extreme portfolios get the 'benefit of the doubt' in the inequalities

        # initialize
        signal = pd.merge(signal, breaklist_wide, on='yyyymm', how='left')
        signal['port'] = pd.NA

        # assign lowest signal
        signal['port'] = np.where(signal['signal'] <= signal['break1'], 1, signal['port'])

        #assign middle
        if len(plist) >= 2:
            for porti in range(2, len(plist) + 1):  # Loop through break2 to breakN
                breakstr = f'break{porti}'  # Construct the column name (e.g., 'break2')
                
                # Create a boolean mask where port is NA and signal is less than the breakpoint
                id_mask = signal['port'].isna() & (signal['signal'] < signal[breakstr])
                
                # Update the port column for the rows where the mask is True
                signal.loc[id_mask, 'port'] = porti
        
        # assign highest signal
        breakstr = f'break{len(plist)}'
        id_mask = signal['port'].isna() & (signal['signal'] >= signal[breakstr])
        signal.loc[id_mask, 'port'] = len(plist) + 1
        signal = signal.drop(columns=[col for col in signal.columns if col.startswith('break')])

        return signal
    
    # MAIN WORK ====

    # Import signal ====
    if verbose:
        print("loading signal data and applying filters")
    
    signal = import_signal(signal_name,filterstr,sign,crspinfo)

    # Assign stocks to portfolios ====
    # (if necessary)
    if verbose:
        print("assigning stocks to portfolios")
    
    if cat_form=='continuous':
        signal = single_sort(q_filt,q_cut)
    elif cat_form=='discrete':
        # for custom categorical portfolios (e.g. Gov Index, PS, MS)
        # by default we go long "largest" cat and short "smallest" cat 

        support = sorted(signal['signal'].unique())
        signal['port'] = pd.NA 
        for i in range(len(support)):
            signal.loc[signal['signal'] == support[i], 'port'] = i + 1

    elif cat_form=='custom':
        signal['port'] = signal['signal']

    # == Portfolio Returns (stock weighting happens here) ====
    # make all na except  "rebalancing months", which is actually signal updating months
    # and then fill na with stale data

    # find months that portfolio assignements are updated
    rebmonths = (start_month + np.arange(0, 13) * port_period) % 12
    rebmonths[rebmonths == 0] = 12
    rebmonths = np.unique(rebmonths)


    signal['port'] = np.where(
    (signal['yyyymm'] % 100).isin(rebmonths),
    signal['port'],
    pd.NA
    )
    signal = signal.sort_values(by=['permno', 'yyyymm'])
    signal['port'] = signal.groupby('permno')['port'].fillna(method='ffill')
    signal = signal.dropna(subset=['port'])


    ### CREATE PORTFOLIOS
    ### ASSIGN TO PORTFOLIOS AND SIGN
    signallag = signal[['permno', 'yyyymm', 'signal', 'port']].copy()
    signallag['yyyymm'] += 1
    signallag['yyyymm'] = np.where(
        signallag['yyyymm'] % 100 == 13,
        signallag['yyyymm'] + 100 - 12,
        signallag['yyyymm']
    )
    signallag.rename(columns={'signal': 'signallag'}, inplace=True)

    if verbose:
        print("joining lagged signals onto crsp returns")
    
    # scoping implies crspret in the big loop is not affected
    crspret = crspret.merge(
    signallag[['permno', 'yyyymm', 'signallag', 'port']],
    on=['permno', 'yyyymm'],
    how='left'
    )

    ## stock weights
    # equal vs value-weighting
    if sweight=='VW':
        crspret['weight']=crspret['melag']
    else:
        crspret['weight']=1
    # adjustments for passive gains 
    # (this is only used for the daily ports right now)
    if passive_gain:
        crspret['weight'] = crspret['weight'] * crspret['passgain']
    if verbose:
        print("calculating portfolio returns")
    
    filtered_crspret = crspret.dropna(subset=['port', 'ret', 'weight'])
    port = filtered_crspret.groupby(['port', 'date']).apply(
        lambda x: pd.Series({
            'ret': np.average(x['ret'], weights=x['weight']),
            'signallag': np.average(x['signallag'], weights=x['weight']),
            'Nlong': len(x)
        })
    ).reset_index()
    port['Nshort'] = 0
    port = port[['port', 'date', 'ret', 'signallag', 'Nlong', 'Nshort']]
    port = port.reset_index(drop=True)

    if verbose:
        print("end of signalname_to_ports")
    
    # Add long-short portfolio  ====     
    if long_port_name == 'max':
        long_port_name = max(port['port'])
    if (short_port_name == 'min'):
        short_port_name = min(port['port'])

    # equal-weight long portfolios

    if type(long_port_name)==int or type(long_port_name)==float:
        long = port[port['port'] == long_port_name]
    elif type(long_port_name)==list:
        long = port[port['port'].isin(list(long_port_name))]
    long = long.groupby('date').agg(
        retL=('ret', 'mean'),
        Nlong=('Nlong', 'sum')
    ).reset_index()

    if type(short_port_name)==int or type(short_port_name)==float:
        short = port[port['port'] == short_port_name]
    elif type(short_port_name)==list:
        short = port[port['port'].isin(short_port_name)]
    short = short.groupby('date').agg(
        retS=('ret', lambda x: -x.mean()),
        Nshort=('Nlong', 'sum')
    ).reset_index()


    longshort = pd.merge(long, short, on='date', how='inner')
    longshort['ret'] = longshort['retL'] + longshort['retS']
    longshort['port'] = 'LS'
    longshort['signallag'] = pd.NA
    longshort = longshort[['port', 'date', 'ret', 'signallag', 'Nlong', 'Nshort']]

    # convert port to 2-character string for ease of use
    port['port'] = port['port'].apply(lambda x: f'{int(x):02d}')

    # bind long with longshort
    port = pd.concat([port, longshort], ignore_index=True)

    port['signalname']=signal_name

    port = port[['signalname'] + [col for col in port.columns if col != 'signalname']]
    port = port.sort_values(by=['signalname', 'port', 'date'])

    return port










