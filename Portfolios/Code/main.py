import os
import globals
import settingsAndTools
import portfolioFunction

if __name__=="__main__":
    # ENTER PROJECT PATH HERE (i.e. this should be the path to your local repo folder & location of SignalDoc.csv)
    pathProject = os.getcwd() + "/"
    

    # Check whether project path is set correctly
    if not os.path.exists(pathProject+'Portfolios'):
        raise Exception('Project path is not defined correctly')
    
    # setwd to folder with all R scripts for convenience
    os.chdir(pathProject+'Portfolios/Code/')

    #check/create folder paths
    settingsAndTools.create_folders()
    settingsAndTools.read_documentation()

    #TODO
    #source('01_PortfolioFunction.R', echo=T)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # PREPARE INTERMEDIATE DATA ####

    print('master: 10_DownloadCRSP.py')
    try:
        exec(open('10_DownloadCRSP.py').read())
    except Exception as e:
        print(f"An error occurred: {e}")



