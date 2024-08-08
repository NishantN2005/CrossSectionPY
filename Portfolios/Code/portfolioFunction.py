from globals import pathPredictors, pathCRSPPredictors, pathPlacebos, pathtemp
import os
import pandas as pd
import numpy as np

#### FUNCTION FOR TURNING SIGNAL CSV TO K PORTFOLIOS

def import_signal(signal_name, filterstr, sign):
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
    paths = [
        f"{pathPredictors}{signal_name}.csv",
        f"{pathCRSPPredictors}{signal_name}.csv",
        f"{pathPlacebos}{signal_name}.csv",
        f"{pathtemp}{signal_name}.csv"
    ]
    # Find the existing CSV file
    csv_name = next((path for path in paths if os.path.exists(path)), None)
    if not csv_name:
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
    if filterstr:
        signal = signal.query(filterstr)

    signal['signal'] *= sign

    return signal

def signal_name_to_port(signal_name, cat_form='continuous', q_cut=0.2, sweight='EW', long_port_name='max',
                        short_port_name='min', start_month=6, port_period=1, q_filt=None, filterstr=None, 
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
    if not 'crspret' in globals():
        print('Error: crspret not in workspace')
        raise ValueError('Please load Intermediate/crspmret.fst or crspdret.fst as crspret')
    
    if not 'crspinfo' in globals():
        print('Error: crspinfo not in workspace.')
        raise ValueError('Please load Intermediate/crspminfo.fst as crspinfo')

    if passive_gain and 'passgain' not in crspret.columns:
        print('Error: passive_gain = T but crspret does not have a passgain column')
        raise ValueError('Please check the correct crspret is loaded')

    if sweight == 'VW' and 'melag' not in crspret.columns:
        print('Error: sweight == VW but crspret does not have melag column')
        raise ValueError('Please check crsp processing')
    

    # function for sorting portfolios if signal is not already a port assignment
    def single_sort(q_filt, q_cut):
        ## create breakpoints
        # subset to firms used for breakpoints, right now only exclude based on q_filt

        #------honestly have no idea how signal is defined here--------------
        tempbreak=signal
        if q_filt is not None and q_filt=='NYSE':
            tempbreak = tempbreak[tempbreak['exchcd'] == 1]

        ## create breakpoints
        # turn q_cut into a list carefully
        if q_cut <= 1/3:
            plist = list(np.arange(q_cut, 1 - 2 * q_cut, q_cut)) + [1 - q_cut]
        else:
            plist = list(np.unique([q_cut, 1 - q_cut]))

        # find breakpoints 
        

