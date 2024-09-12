import pandas as pd

from globals import pathProject, pathDataPortfolios

def read_data():
    # ==== ENVIRONMENT AND DATA ====
    crspinfo = pd.read_csv(f"{pathProject}/Portfolios/Data/Intermediate/crspminfo.csv")

    crspret = pd.read_csv(f"{pathProject}/Portfolios/Data/Intermediate/crspmret.csv")

    return crspinfo, crspret

def select_signals(alldocumentation, ifquickrun):
    # SELECT SIGNALS 
    strategylist0 = alldocumentation[alldocumentation['Cat.Signal'] == 'Predictor']
    strategylist0 = ifquickrun(strategylist0) 

    return strategylist0

def process_strategies(strategylist0,crspinfo,crspret, loop_over_strategies, checkport, writestandard):
    #### ALT HOLDING PERIODS ####
    holdPerList = [1,3,6,12]
    for i in range(len(holdPerList)):
        print(f'Running port period = {holdPerList[i]}==========================')

        strategylist0['portperiod'] = holdPerList[i]
        port = loop_over_strategies(strategylist0,crspret,crspinfo)
        
        checkport(port)

        writestandard(
        port,
        pathDataPortfolios,
        f"PredictorAltPorts_HoldPer_{holdPerList[i]}.csv"
        )

    #### ALT LIQUIDITY ADJUSTMENTS ####
    print("CheckLiq: ME > NYSE 20 pct =========================================")

    ## ME > NYSE 20th pct
    # create ME screen
    # customscreen is used on the signal df, which is then lagged, so no look ahead here
    port = loop_over_strategies(
        strategylist0.assign(filterstr="me > me_nyse20"),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(
    port,
    pathDataPortfolios, "PredictorAltPorts_LiqScreen_ME_gt_NYSE20pct.csv"
    )

    ## Price > 5
    print("CheckLiq: Price > 5  =========================================")
    port = loop_over_strategies(
        strategylist0.assign(filterstr="abs(prc) > 5"),
        crspret,
        crspinfo
    )
    writestandard(
    port,
    pathDataPortfolios, "PredictorAltPorts_LiqScreen_Price_gt_5.csv"
    )

    ## NYSE only
    print("CheckLiq: NYSE only =========================================")
    port = loop_over_strategies(
        strategylist0.assign(filterstr="exchcd==1"),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(
    port,
    pathDataPortfolios, "PredictorAltPorts_LiqScreen_NYSEonly.csv"
    )

    ## VW
    print("CheckLiq: VW force =========================================")
    port = loop_over_strategies(
        strategylist0.assign(sweight="VW"),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(
    port,
    pathDataPortfolios, "PredictorAltPorts_LiqScreen_VWforce.csv"
    )


    #### ALT QUANTILES ####
    ## DECILE SORTS
    print("----------strategylist0----------")
    print(strategylist0)
    print('-------columns--------')
    print(strategylist0.columns)
    strategylistcts = strategylist0[strategylist0['Cat.Form'] == 'continuous']

    # OP stock weighting
    port = loop_over_strategies(
        strategylistcts.assign(q_cut=0.1),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(port, pathDataPortfolios, "PredictorAltPorts_Deciles.csv")

    # force value weighting
    port = loop_over_strategies(
        strategylistcts.assign(q_cut=0.1, sweight='VW'),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(port, pathDataPortfolios, "PredictorAltPorts_DecilesVW.csv")


    ## QUINTILE SORTS
    # OP stock weighting

    port = loop_over_strategies(
        strategylistcts.assign(q_cut=0.2),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(port, pathDataPortfolios, "PredictorAltPorts_Quintiles.csv")

    # force value weighting
    port = loop_over_strategies(
        strategylistcts.assign(q_cut=0.2, sweight='VW'),
        crspret,
        crspinfo
    )
    checkport(port)
    writestandard(port, pathDataPortfolios, "PredictorAltPorts_QuintilesVW.csv")

