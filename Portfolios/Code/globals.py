import os

#PROJECT SETTINGS
global quickrun
quickrun = True # use True if you want to run quickly for testing

global quickrunlist 
quickrunlist = ['Accruals','AM', 'AMq'] # list of signals to use for quickrun

global skipdaily
skipdaily = True # use True to skip daily CRSP which is very slow

global verbose
verbose = False # use True if you want lots of feedback


# ENTER PROJECT PATH HERE (i.e. this should be the path to your local repo folder & location of SignalDoc.csv)
global pathProject 
pathProject = os.getcwd() + "/"

#### PATHS
global pathPredictors 
pathPredictors  = pathProject+'Signals/Data/Predictors/'

global pathPlacebos 
pathPlacebos=pathProject+'Signals/Data/Placebos/'

global pathCRSPPredictors
pathCRSPPredictors= pathProject+'Signals/Data/CRSPPredictors/'

global pathtemp 
pathtemp= pathProject+'Signals/Data/temp/'

global pathCode 
pathCode= pathProject+'Portfolios/Code/'

global pathDataIntermediate 
pathDataIntermediate = pathProject+'Portfolios/Data/Intermediate/'

global pathDataPortfolios 
pathDataPortfolios = pathProject+'Portfolios/Data/Portfolios/'

global pathDataSummary 
pathDataSummary = pathProject+'Portfolios/Data/Summary/'

global pathResults 
pathResults = pathProject+'Results/'



