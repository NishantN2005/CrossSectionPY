import os
import settingsAndTools
import processCRSP
import time
import downloadCRSP
import createCRSPPredictors
import predictorPorts
import predictorExhibits
import predictorAltPorts
import checkPredictorExhibits
import predictor2x3Ports
import placeboPorts
import placeboExhibits
import signalExhibits
import dailyPredictorPortfolios

from globals import quickrun, skipdaily

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
    alldocumentation = settingsAndTools.read_documentation()


    #TODO
    #source('01_PortfolioFunction.R', echo=T)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # PREPARE INTERMEDIATE DATA ####

    #downloadCRSP.py file
    print('main: downloadCRSP.py')
    try:
        start_time = time.time()

        wrds_user, wrds_password=downloadCRSP.wrds_login()
        downloadCRSP.download_data()

        end_time=time.time()
        print(f'total time taken for file 10: {end_time-start_time}')
    except Exception as e:
        print(f"An error occurred: {e}")

    #processCrsp.py file
    print('main: processCrsp.py')
    try:
        processCRSP.process()
    except Exception as e: 
        print(f"An error occured: {e}")

    #createCRSPPredictors.py file
    print('main: createCRSPPredictors.py')
    try:
        crspret, crspinfo = createCRSPPredictors.read_data()
        createCRSPPredictors.make_STreversal(crspret)
        createCRSPPredictors.make_price(crspinfo)
        createCRSPPredictors.make_size(crspinfo)
    except Exception as e:
        print(f"an error occured: {e}")
    
    # ==== CREATE BASELINE PORTFOLIOS ====
    print('main: predictorPorts.py')
    try:
        crspret, crspinfo = predictorPorts.read_data()
        predictorPorts.process(
            alldocumentation,
            crspret,
            crspinfo,
            settingsAndTools.if_quick_run,
            settingsAndTools.loop_over_strategies,
            settingsAndTools.check_port,
            settingsAndTools.write_standard,
            settingsAndTools.sum_port_month
            )
    except Exception as e:
        print(f'An error has occured: {e}')

    if not quickrun:
        print("main: predictorExhibits")
        try:
            ff = predictorExhibits.download_file()
            predictorExhibits.figure_1(ff)
            predictorExhibits.figure_2(alldocumentation)
            predictorExhibits.scatter_fig(alldocumentation)
            predictorExhibits.mclean_pontiff_fig(alldocumentation)
            predictorExhibits.summary_table(settingsAndTools.read_documentation())
        except Exception as e:
            print(f'An error has occured: {e}')

    # CREATE ALTERNATIVE PORTFOLIOS AND PLACEBOS ####
    print('main: predictorAltPorts')
    try:
        crspinfo, crspret = predictorAltPorts.read_data()
        strategylist0 = predictorAltPorts.select_signals(
            alldocumentation,
            settingsAndTools.if_quick_run
            )
        predictorAltPorts.process_strategies(
            strategylist0,
            crspinfo,
            crspret,
            settingsAndTools.loop_over_strategies, 
            settingsAndTools.check_port,
            settingsAndTools.write_standard
            )
    except Exception as e:
        print(f'An error has occured: {e}')

    if not quickrun:
        print('checkPredictorExhibits')
        try:
            checkPredictorExhibits.process(settingsAndTools.sum_port_month)
        except Exception as e:
            print(f'An exception has occured: {e}')
    
    print("main: predictor2x3Ports")
    try:
        crspinfo, crspret = predictor2x3Ports.read_data()
        strategylist = predictor2x3Ports.select_signals(alldocumentation, settingsAndTools.if_quick_run)
        predictor2x3Ports.loop_over_signals(strategylist, crspret, crspinfo)
    except Exception as e:
            print(f'An exception has occured: {e}')

    print("main: placeboPorts")
    try:
        crspret, crspinfo = placeboPorts.read_data()
        placeboPorts.select_signals(
            alldocumentation,
            crspret,
            crspinfo,
            settingsAndTools.if_quick_run, 
            settingsAndTools.loop_over_strategies,
            settingsAndTools.write_standard,
            settingsAndTools.sum_port_month
            )
    except Exception as e:
        print(f'An error has occured: {e}')
    
    if not quickrun:
        print("placeboExhibits")
        try:
            placeboExhibits.relevent_set(settingsAndTools.read_documentation)
            placeboExhibits.mclean_plontiff_figs(alldocumentation)
            placeboExhibits.replication_rate(settingsAndTools.read_documentation)
        except Exception as e:
            print(f'An error has occured: {e}')


    # EXTRA STUFF ####
    # this can be run at the end since it takes a long time and isn't necessary for other results
    print("main: signalExhibits")
    try:
        signalExhibits.process()
    except Exception as e:
        print(f'An error has occured: {e}')

    if not skipdaily:
        print("dailyPredictorPorts")
        try:

            dailyPredictorPortfolios.process(
                alldocumentation,
                settingsAndTools.if_quick_run, 
                settingsAndTools.loop_over_strategies
                )
        except Exception as e:
            print(f'An error has occured: {e}')
    

    
    



