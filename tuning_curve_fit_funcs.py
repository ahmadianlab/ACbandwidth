import numpy as np
from scipy.special import erf
import time
import tqdm

#-------------------------------------------------------------------------------
def carandini_form(x, mExp, sigmaD, sigmaS, RD, RS):
#the Carandini form (from Ayaz et al. (2013)) for Bandwidth tuning curve, inspired by the divisive normalization model
    return (RD*(erf(x/(np.sqrt(2)*sigmaD)))**mExp)/(1+RS*(erf(x/(np.sqrt(2)*sigmaS)))**mExp)

#-------------------------------------------------------------------------------
def carandini_form_fixed_m(x, sigmaD, sigmaS, RD, RS):
#the Carandini form (from Ayaz et al. (2013)) for Bandwidth tuning curve, inspired by the divisive normalization model
    mExp = 4
    return (RD*(erf(x/(np.sqrt(2)*sigmaD)))**mExp)/(1+RS*(erf(x/(np.sqrt(2)*sigmaS)))**mExp)


#-------------------------------------------------------------------------------
def diff_gauss_form(x, mExp, sigmaD, sigmaS, RD, RS):
#the difference of Gaussians form for Bandwidth tuning curve, inspired by "subtractive normalization"
    return RD* (erf(x/(np.sqrt(2)*sigmaD)))**mExp - RS* erf(x/(np.sqrt(2)*sigmaS)))**mExp

#-------------------------------------------------------------------------------
def diff_gauss_form_fixed_m(x, sigmaD, sigmaS, RD, RS):
#the difference of Gaussians form for Bandwidth tuning curve, inspired by "subtractive normalization"
    mExp = 4
    return RD* (erf(x/(np.sqrt(2)*sigmaD)))**mExp - RS* erf(x/(np.sqrt(2)*sigmaS)))**mExp


#-------------------------------------------------------------------------------
def carandini_fit(stimuli, responses, RFscale = None, mFixed=False):
#find best fit of Carandini form (Ayaz et al. 2013) to bandwidth tuning curve of one cell using least squares
    from scipy.optimize import curve_fit

    if RFscale is None:
        RFscale = stimuli[2] # = stimuli[np.argmax(responses)] # = summation field size
    MaxResp = np.max(np.abs(responses))

    #set up Lower and Upper bounds on Parameters
    maxM = 10
    maxsigD = np.max(stimuli) #3* np.max(stimuli) #max(xs)/2
    maxsigS = 2* np.max(stimuli) #4* np.max(stimuli)
    maxRD = 10* MaxResp
    maxRS = 20 #100
    Upper = np.asarray([maxM ,maxsigD ,maxsigS , maxRD, maxRS])
    #Lower = np.asarray([1, RFscale/10., RFscale/10.,   0.001*maxRD, 0])
    Lower= np.asarray([1 , RFscale/100., RFscale/100., -maxRD ,  0])

    #set up different initialization sets for parameters
    Nparsets = 8
    initpars = np.asarray([[2.5, RFscale, 2*RFscale, MaxResp, 1]]).T * np.ones((5,Nparsets)) #uses broadcasting. This gives error (shape mismatch for broadcasting): initpars = np.ones((5,Nparsets)) * np.asarray([2.5, RFscale, 2*RFscale, MaxResp, 1])
    #                      [m  , sigmaD , sigmaS   , RD    , RS ]
    p = 0
    #initpars[3,p] = responses[1]/(erf(stimuli[1]/np.sqrt(2)/initpars[1,p]))**initpars[0,p] # RD
    initpars[3,p] *= 2
    p += 1
    initpars[3,p] = -initpars[3,p-1] # RD
    p += 1
    initpars[0,p] = 2.0 # m
    initpars[4,p] = 0.01 # RS
    p += 1
    initpars[:,p] = initpars[:,p-1]
    initpars[3,p] = -initpars[3,p-1] # RD
    p += 1
    initpars[0,p] = 2.0 # m
    initpars[3,p] *= 4 # RD
    initpars[4,p] = 5 # RS
    p += 1
    initpars[:,p] = initpars[:,p-1]
    initpars[3,p] = -initpars[3,p-1] # RD
    p += 1
    initpars[0,p] = 3.0 # m
    initpars[1,p] = 4*RFscale # sigmaD
    initpars[2,p] = RFscale # sigmaS
    #initpars[3,p] = responses[1]/(erf(stimuli[1]/np.sqrt(2)/initpars[1,p]))**initpars[0,p] # RD
    initpars[4,p] = 5 # RS

    curve_form = carandini_form
    if mFixed:
        Upper = Upper[1:]
        Lower = Lower[1:]
        initpars = initpars[1:,:]
        curve_form = carandini_form_fixed_m

    fitParams = None
    fitResponses = None
    SSE = np.inf
    for p in range(Nparsets):
        #print("paramset {}".format(p))
        try:
            #fitParams0, ParamsCov0 = curve_fit(carandini_form, stimuli, responses, p0=initpars[:,p], bounds = (Lower,Upper), maxfev=10000)
            fitParams0 = curve_fit(curve_form, stimuli, responses, p0=initpars[:,p], bounds = (Lower,Upper), maxfev=10000)[0]
            fitResponses0 = curve_form(stimuli, *fitParams0)
            SqErr = np.sum((responses - fitResponses0)**2)
            if SqErr < SSE:
                fitParams = fitParams0
                fitResponses = fitResponses0
                SSE = SqErr

        except RuntimeError:
            print("Could not fit {} curve to tuning data with the {}-th initialization set.".format('Carandini',p))

    return fitParams, SSE #, fitResponses


#-------------------------------------------------------------------------------
def diff_of_gauss_fit(stimuli, responses, RFscale = None, mFixed=False):
#find best fit of difference of gaussians model (really diff of erf's) to bandwidth tuning curve of one cell using least squares
    from scipy.optimize import curve_fit

    if RFscale is None:
        RFscale = stimuli[2] # = stimuli[np.argmax(responses)] # = summation field size
    MaxResp = np.max(np.abs(responses))
    MaxNegResp = np.max(-responses)

    #set up Lower and Upper bounds on Parameters
    maxM = 10
    maxsigD = 3*np.max(stimuli) #3* np.max(stimuli) #max(xs)/2
    maxsigS = 3*np.max(stimuli) #4* np.max(stimuli)
    maxRD = 2* MaxResp #10* MaxResp
    Upper = np.asarray([maxM ,maxsigD ,maxsigS , maxRD, maxRD])
    #Lower = np.asarray([1, RFscale/10., RFscale/10.,   0.001*maxRD, 0])
    Lower= np.asarray([1 , RFscale/100., RFscale/100., 0 ,  0])
    #                 [m  , sigmaD     , sigmaS      , RD     , RS ]

    #set up different initialization sets for parameters
    Nparsets = 5
    initpars = np.asarray([[2.5, RFscale, 2*RFscale, MaxResp, MaxNegResp]]).T * np.ones((5,Nparsets)) #uses broadcasting. This gives error (shape mismatch for broadcasting): initpars = np.ones((5,Nparsets)) * np.asarray([2.5, RFscale, 2*RFscale, MaxResp, 1])
    #                      [m  , sigmaD , sigmaS   , RD    , RS ]
    p = 0
    # #initpars[3,p] = responses[1]/(erf(stimuli[1]/np.sqrt(2)/initpars[1,p]))**initpars[0,p] # RD
    initpars[3,p] *= .5
    p += 1
    initpars[0,p] = 2.0 # m
    initpars[4,p] = 0.01 # RS
    p += 1
    initpars[0,p] = 2.0 # m
    initpars[3,p] *= 2 # RD
    initpars[4,p] *= 2 # RS
    p += 1
    initpars[0,p] = 3.0 # m
    initpars[1,p] = 4*RFscale # sigmaD
    initpars[2,p] = RFscale # sigmaS
    #initpars[3,p] = responses[1]/(erf(stimuli[1]/np.sqrt(2)/initpars[1,p]))**initpars[0,p] # RD

    curve_form = diff_gauss_form
    if mFixed:
        Upper = Upper[1:]
        Lower = Lower[1:]
        initpars = initpars[1:,:]
        curve_form = diff_gauss_form_fixed_m

    fitParams = None
    fitResponses = None
    SSE = np.inf
    for p in range(Nparsets):
        #print("paramset {}".format(p))
        try:
            #fitParams0, ParamsCov0 = curve_fit(carandini_form, stimuli, responses, p0=initpars[:,p], bounds = (Lower,Upper), maxfev=10000)
            fitParams0 = curve_fit(curve_form, stimuli, responses, p0=initpars[:,p], bounds = (Lower,Upper), maxfev=10000)[0]
            fitResponses0 = curve_form(stimuli, *fitParams0)
            SqErr = np.sum((responses - fitResponses0)**2)
            if SqErr < SSE:
                fitParams = fitParams0
                fitResponses = fitResponses0
                SSE = SqErr

        except RuntimeError:
            print("Could not fit {} curve to tuning data with the {}-th initialization set.".format('Carandini',p))

    return fitParams, SSE #, fitResponses

#-------------------------------------------------------------------------------
def extract_stats_from_fit(stimuli, responses, test_stims, mFixed=False, function_class_fit=carandini_fit):
# calculate and package tuning curve features using the best fit Carandini-form (default)
# or the best fit difference-of-Gaussians-form (if function_class_fit = diff_of_gauss_fit)

    #print('goh')
    fitParams, SSE = function_class_fit(stimuli, responses, mFixed=mFixed)
    #add some

    Params = {}
    if not mFixed:
        Params['m'] = fitParams[0]
    Params['sigmaD'] = fitParams[1-1*mFixed]
    Params['sigmaS'] = fitParams[2-1*mFixed]
    Params['RD'] = fitParams[3-1*mFixed]
    Params['RS'] = fitParams[4-1*mFixed]

    #calculate goodness of fit
    FitGoodness = {}
    FitGoodness['SSE'] = SSE
    SStotal = np.sum((responses-np.mean(responses))**2)
    FitGoodness['R2'] = 1-(SSE/SStotal)

    curve_form = carandini_form
    if mFixed:
        curve_form = carandini_form_fixed_m

    Stats = {}
    test_resps = curve_form(test_stims, *fitParams)
    Stats['maxResp'] = np.max(test_resps)
    Stats['maxNegResp'] = -np.max(-test_resps)
    Stats['maxAbsResp'] = np.max(np.abs(test_resps))
    Stats['prefBW'] = test_stims[np.argmax(test_resps)]
    Stats['AbsPrefBW'] = test_stims[np.argmax(np.abs(test_resps))]
    Stats['wnResp'] = test_resps[-1]
    Stats['SuppInd'] = 1 - Stats['wnResp']/Stats['maxResp']

    return Params, FitGoodness, Stats, test_resps


#-------------------------------------------------------------------------------
def FitAllCells(Datafile, Celltype, Response_type, BLisZeroBW = True, WNoctave = 6, NtestBWs = 50, mFixed=False, function_class_fit=carandini_fit):
#fit all cells in a data-file with Carandini form (default) or the difference-of-Gaussians-form (if function_class_fit = diff_of_gauss_fit)
    data = np.load(Datafile)
    EvokedRates = data[Celltype+Response_type+'Responses']
    BaselineRates = data[Celltype+Response_type+'BaselineSpikeRates']
    Bandwidths = data['stimulusBandwidth']
    data.close()

    Bandwidths[-1] = WNoctave #white noise BW set to WNoctave octaves
    test_BWs = np.linspace(Bandwidths[0],Bandwidths[-1],NtestBWs) #bandwidths used for interpolation post-fit

    rawTCs = EvokedRates#.copy()
    if BLisZeroBW:
        rawTCs[0,:] = BaselineRates #replace pure tone responses with baseline (=0 bandwidth = no stimulus)
    rawTCs = rawTCs - rawTCs[0,:] #remove the BW=0 from all, uses broadcasting

    StatsList = []
    ParamsList = []
    GOFlist = []
    fitTCs = np.zeros((len(test_BWs),rawTCs.shape[1]))
    start_time = time.time()
    for cel in tqdm.tqdm(range(rawTCs.shape[1])): #loop over all pyr cells
        #print("cell number {}".format(cel))
        Params, FitGoodness, Stats, testResponses = extract_stats_from_fit(Bandwidths, rawTCs[:,cel], test_BWs, mFixed=mFixed, function_class_fit=function_class_fit)
        StatsList.append(Stats)
        ParamsList.append(Params)
        GOFlist.append(FitGoodness)
        fitTCs[:,cel] = testResponses
    end_time = time.time()
    print("\n Elapsed time was %g seconds" % (end_time - start_time))

    return StatsList, ParamsList, GOFlist, fitTCs, rawTCs, test_BWs, Bandwidths, BaselineRates

#-------------------------------------------------------------------------------
def makeFeaturesDict(GOFlist,StatsList,ParamsList,rawTCs,fitTCs,Bandwidths,test_BWs,Baselines,BLisZeroBW,Celltype,Response_type):
#package the fit-parameters and TC-features of all cells of a given type into a "features dictionary"
    TCfeatures = {'CarandiniParams': {},'MiscParams': {},'GoF': {},\
                  'fitTCs': fitTCs, 'rawTCs': rawTCs,'test_BWs': test_BWs, 'Bandwidths': Bandwidths,\
                  'Baselines': Baselines, 'BLisZeroBW': BLisZeroBW, 'Ncells': rawTCs.shape[1],\
                  'Celltype': Celltype,'Response_type': Response_type}

    #package goodness-of-fit parameters
    TCfeatures['GoF']['R2'] = np.asarray([stats['R2'] for stats in GOFlist])
    TCfeatures['GoF']['SSE'] = np.asarray([stats['SSE'] for stats in GOFlist])

    #package Miscellaneous tuning curve parameters
    TCfeatures['MiscParams']['Baseline'] = TCfeatures['Baselines']
    if BLisZeroBW:
        TCfeatures['MiscParams']['peakResp'] = np.asarray([stats['maxResp'] for stats in StatsList])
        TCfeatures['MiscParams']['peakNegResp'] = np.asarray([stats['maxNegResp'] for stats in StatsList])
        TCfeatures['MiscParams']['peakAbsResp'] = np.asarray([stats['maxAbsResp'] for stats in StatsList])
        TCfeatures['MiscParams']['peakRate'] = TCfeatures['MiscParams']['peakResp'] + TCfeatures['MiscParams']['Baseline']
        TCfeatures['MiscParams']['prefBW'] = np.asarray([stats['prefBW'] for stats in StatsList])
        TCfeatures['MiscParams']['AbsPrefBW'] = np.asarray([stats['AbsPrefBW'] for stats in StatsList])
        TCfeatures['MiscParams']['wnResp'] = np.asarray([stats['wnResp'] for stats in StatsList])
        TCfeatures['MiscParams']['wnRate'] = TCfeatures['MiscParams']['wnResp'] + TCfeatures['MiscParams']['Baseline']
        TCfeatures['MiscParams']['SuppInd_noBL'] = np.asarray([stats['SuppInd'] for stats in StatsList])
        TCfeatures['MiscParams']['SuppInd_wBL'] = 1 - TCfeatures['MiscParams']['wnRate']/TCfeatures['MiscParams']['peakRate']
    else:
        TCfeatures['MiscParams']['peakRate'] = np.asarray([stats['maxResp'] for stats in StatsList])
        TCfeatures['MiscParams']['peakResp'] = TCfeatures['MiscParams']['peakRate'] - TCfeatures['MiscParams']['Baseline']
        TCfeatures['MiscParams']['prefBW'] = np.asarray([stats['prefBW'] for stats in StatsList])
        TCfeatures['MiscParams']['AbsPrefBW'] = TCfeatures['test_BWs'][np.argmax(np.abs(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)] #uses broadcasting
        TCfeatures['MiscParams']['peakAbsResp'] = np.max(np.abs(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)
        TCfeatures['MiscParams']['peakNegResp'] = -np.max(-(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)
        TCfeatures['MiscParams']['wnRate'] = np.asarray([stats['wnResp'] for stats in StatsList])
        TCfeatures['MiscParams']['wnResp'] = TCfeatures['MiscParams']['wnRate'] - TCfeatures['BLisZeroBW']*TCfeatures['MiscParams']['Baseline']
        TCfeatures['MiscParams']['SuppInd_noBL'] = 1 - TCfeatures['MiscParams']['wnResp']/TCfeatures['MiscParams']['peakResp']
        TCfeatures['MiscParams']['SuppInd_wBL'] = np.asarray([stats['SuppInd'] for stats in StatsList])
    TCfeatures['MiscParams']['peakSignedResp'] = (TCfeatures['MiscParams']['peakResp'] > np.abs(TCfeatures['MiscParams']['peakNegResp']))*TCfeatures['MiscParams']['peakResp'] + (TCfeatures['MiscParams']['peakResp'] < np.abs(TCfeatures['MiscParams']['peakNegResp']))*TCfeatures['MiscParams']['peakNegResp']

    # #package Miscellaneous tuning curve parameters
    # TCfeatures['MiscParams']['Baseline'] = TCfeatures['Baselines']
    # for keyname in StatsList[0]:
    #     TCfeatures['MiscParams'][keyname] = np.asarray([stats[keyname] for stats in StatsList])
    # if BLisZeroBW:
    #     TCfeatures['MiscParams']['peakRate'] = TCfeatures['MiscParams']['peakResp'] + TCfeatures['MiscParams']['Baseline']
    #     TCfeatures['MiscParams']['wnRate'] = TCfeatures['MiscParams']['wnResp'] + TCfeatures['MiscParams']['Baseline']
    #     TCfeatures['MiscParams']['SuppInd_wBL'] = 1 - TCfeatures['MiscParams']['wnRate']/TCfeatures['MiscParams']['peakRate']
    # else:
    #     TCfeatures['MiscParams']['peakResp'] = TCfeatures['MiscParams']['peakRate'] - TCfeatures['MiscParams']['Baseline']
    #     TCfeatures['MiscParams']['AbsPrefBW'] = TCfeatures['test_BWs'][np.argmax(np.abs(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)] #uses broadcasting
    #     TCfeatures['MiscParams']['peakAbsResp'] = np.max(np.abs(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)
    #     TCfeatures['MiscParams']['peakNegResp'] = -np.max(-(TCfeatures['fitTCs']-TCfeatures['Baselines']),axis=0)
    #     TCfeatures['MiscParams']['wnResp'] = TCfeatures['MiscParams']['wnRate'] - TCfeatures['BLisZeroBW']*TCfeatures['MiscParams']['Baseline']
    #     TCfeatures['MiscParams']['SuppInd_noBL'] = 1 - TCfeatures['MiscParams']['wnResp']/TCfeatures['MiscParams']['peakResp']
    # TCfeatures['MiscParams']['peakSignedResp'] = (TCfeatures['MiscParams']['peakResp'] > np.abs(TCfeatures['MiscParams']['peakNegResp']))*TCfeatures['MiscParams']['peakResp'] + (TCfeatures['MiscParams']['peakResp'] < np.abs(TCfeatures['MiscParams']['peakNegResp']))*TCfeatures['MiscParams']['peakNegResp']


    #package Carandini parameters
    for keyname in ParamsList[0]:
        TCfeatures['CarandiniParams'][keyname] = np.asarray([stats[keyname] for stats in ParamsList])
    if BLisZeroBW:
        TCfeatures['CarandiniParams']['R0'] = TCfeatures['MiscParams']['Baseline']


    #create plotting labels for goodness-of-fit parameters
    GoFNames = list(TCfeatures['GoF'].keys())
    GoFNames.sort()
    TCfeatures['GoFNames'] = GoFNames
    #print(" , ".join(TCparamNames))
    GoFLabels = dict(zip(GoFNames, GoFNames))
    GoFLabels['R2'] = '$R^2$'
    TCfeatures['GoFLabels'] = GoFLabels

    #create plotting labels for miscellaneous TC parameters
    TCparamNames = list(TCfeatures['MiscParams'].keys())
    TCparamNames.sort()
    TCfeatures['MiscParamsNames'] = TCparamNames
    #print(" , ".join(TCparamNames))
    MiscParamsLabels = {}
    MiscParamsLabels['AbsPrefBW'] = 'BW of max abs response (Oct.)'
    MiscParamsLabels['Baseline'] = 'Baseline rate (Hz)'
    MiscParamsLabels['SuppInd_noBL'] = 'S.I. w BL removed'
    MiscParamsLabels['SuppInd_wBL'] = 'S.I. of rates'
    MiscParamsLabels['peakAbsResp'] = 'max abs response (Hz)'
    MiscParamsLabels['peakNegResp'] = 'max negative response (Hz)'
    MiscParamsLabels['peakRate'] = 'max rate (Hz)'
    MiscParamsLabels['peakResp'] = 'max response (Hz)'
    MiscParamsLabels['peakSignedResp'] = 'signed max response (Hz)'
    MiscParamsLabels['prefBW'] = 'preferred bandwidth (Oct.)'
    MiscParamsLabels['wnRate'] = 'Rate at WN (Hz)'
    MiscParamsLabels['wnResp'] = 'Response to WN (Hz)'
    TCfeatures['MiscParamsLabels'] = MiscParamsLabels

    #create plotting labels for Carandini parameters
    TCparamNames = list(TCfeatures['CarandiniParams'].keys())
    TCparamNames.sort()
    TCfeatures['CarandiniParamsNames'] = TCparamNames
    #print(" , ".join(TCparamNames))
    CarandiniParamsLabels = dict(zip(TCparamNames, TCparamNames))
    CarandiniParamsLabels['m'] = '$m$'
    CarandiniParamsLabels['R0'] = '$R_0$ (Hz)'
    CarandiniParamsLabels['RD'] = '$R_D$ (Hz)'
    CarandiniParamsLabels['RS'] = '$R_S$'
    CarandiniParamsLabels['sigmaD'] = '$\sigma_D$ (Oct.)'
    CarandiniParamsLabels['sigmaS'] = '$\sigma_S$ (Oct.)'
    TCfeatures['CarandiniParamsLabels'] = CarandiniParamsLabels


    return TCfeatures
