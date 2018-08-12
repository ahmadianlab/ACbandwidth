# Wrote this script to run in nearly identical notebooks
# for different cell types, by first assigning a number to nb_cell
# and then running:
# %run -i celltype_identified_TCfits_nb

if nb_cell == 0:
    print('importing modules\n')
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import tqdm

    import tuning_curve_fit_funcs as tcfits
    import tuning_curve_feature_plotting as tcfplot
    from importlib import reload
    reload(tcfits)
    reload(tcfplot)

#-------------------------------------------------------------------------------
elif nb_cell == 1:
    #load experimental data
    PROJ_DIR = '/Users/yashar/Google-Drive/Documents/Work/Projects/'
    DATA_DIR = PROJ_DIR + 'AuditoryContextualModul/Matlab/Lakunina-Jaramillo-Data/'
    filename = 'photoidentified_cells_responsesTCs_new.npz'
    data = np.load(DATA_DIR + filename)
    print("\n".join(data.keys())) #:gives the list of arrays in the file
    data.close()

#-------------------------------------------------------------------------------
elif nb_cell == 2:
    #fit TC's
    TCfeatures = tcfits.fit_all_cells(DATA_DIR + filename, Celltype, Response_type,
            BLisZeroBW, WNoctave,  mFixed=mExp, function_class_fit=curve_fit_class)

#-------------------------------------------------------------------------------
elif nb_cell == 3:
    #plot R^2 histogram
    test_BWs = TCfeatures['test_BWs']
    Bandwidths = TCfeatures['Bandwidths']

    R2 = TCfeatures['GoF']['R2']
    R2Label = TCfeatures['GoFLabels']['R2']
    #fig = plt.figure(figsize=(10,5))
    n, bins, patches = plt.hist(R2, 15, facecolor='g',alpha=0.7)
    plt.axvline(x=np.median(R2),c='r',linestyle='--')
    plt.axvline(x=np.mean(R2),c='g',linestyle='--')
    plt.xlabel(R2Label)
    plt.ylabel(CelltypeLabel+ ' Cell Count')
    plt.text(np.median(R2)*.7, 7, 'median('+R2Label+') = %1.2f' % np.median(R2))
    plt.text(np.mean(R2)*.7, 6, 'mean('+R2Label+') = %1.2f' % np.mean(R2))
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)
    #plt.show()

#-------------------------------------------------------------------------------
elif nb_cell == 4:
    #visualize all fits, from worse to best
    import pylab as pl
    from IPython import display

    R2inds = np.argsort(R2)
    ii=0
    for cel in R2inds[:]:
        pl.clf()
        baseline = TCfeatures['Baselines'][cel]
        pl.plot(test_BWs,baseline + TCfeatures['fitTCs'][:,cel],'r-',label='fit curve ('+R2Label+' = %1.2f)' % R2[cel])
        pl.plot(Bandwidths, baseline + TCfeatures['rawTCs'][:,cel],'bo-',label='data')
        pl.plot(Bandwidths, baseline*(1+0*Bandwidths),'k--',label='baseline')
        ys = np.asarray(pl.gca().get_ylim())
    #    pl.text(np.mean(R2)*.7, 30, 'mean('+R2Label+') = %1.2f' % np.mean(R2))
        pl.plot(TCfeatures['MiscParams']['prefBW'][cel] + 0*ys, baseline + np.linspace(-.5,.5,2)*np.diff(ys)/2,'g-',label='pref BW')
        pl.plot(np.linspace(0,1,2)*TCfeatures['CarandiniParams']['sigmaD'][cel], 0*ys + baseline + np.diff(ys)/20,'m-')
        pl.plot(np.linspace(0,1,2)*TCfeatures['CarandiniParams']['sigmaS'][cel], 0*ys + baseline - np.diff(ys)/20,'y-')
        pl.text(0.4*TCfeatures['CarandiniParams']['sigmaD'][cel], baseline + np.diff(ys)*1.5/20, '$\sigma_D$',color='m')
        pl.text(0.4*TCfeatures['CarandiniParams']['sigmaS'][cel], baseline - np.diff(ys)*1.7/20, '$\sigma_S$',color='y')

        pars = (TCfeatures['CarandiniParams']['RD'][cel], TCfeatures['CarandiniParams']['RS'][cel],TCfeatures['CarandiniParams']['sigmaD'][cel],TCfeatures['CarandiniParams']['sigmaS'][cel])
        pl.text(2,baseline + np.diff(ys)*2/20,'$R_D$ = %2.1f' % pars[0])
        pl.text(2,baseline - np.diff(ys)*2/20,'$R_S$ = %2.1f' % pars[1])
        pl.xlabel('Bandwidth (Octave)')
        pl.ylabel('Rate (Hz)')
        ii+=1
        pl.title('Cell %i of %i,   %i%%' % (ii,len(R2),100 * ii//len(R2)))
        pl.legend()
        display.display(pl.gcf())
        display.clear_output(wait=True)
        time.sleep(pause)
    pl.close()

#-------------------------------------------------------------------------------
elif nb_cell == 5:
    #Save all fit plots, from worse to best
    PLOTS_DIR = DATA_DIR + 'Plots/TCplots/'+CelltypeLabel+'cells/'
    if curve_fit_class == "carandini_fit":
        curve_type = 'cell_TC_carandini_fit_'
    elif curve_fit_class == "diff_of_gauss_fit":
        curve_type = 'cell_TC_gauss_fit_'

    R2inds = np.argsort(R2)
    ii = 1
    for cel in R2inds[:]:
        fig, ax = plt.subplots()
        baseline = TCfeatures['Baselines'][cel]
        ax.plot(test_BWs,baseline + TCfeatures['fitTCs'][:,cel],'r-',
                        label='fit curve ('+R2Label+' = %1.2f)' % R2[cel])
        ax.plot(Bandwidths, baseline + TCfeatures['rawTCs'][:,cel],'bo-',label='data')
        ax.plot(Bandwidths, baseline*(1+0*Bandwidths),'k--',label='baseline')
        ys = np.asarray(pl.gca().get_ylim())
    #    ax.text(np.mean(R2)*.7, 30, 'mean('+R2Label+') = %1.2f' % np.mean(R2))
        ax.plot(TCfeatures['MiscParams']['prefBW'][cel] + 0*ys,
                baseline + np.linspace(-.5,.5,2)*np.diff(ys)/2,'g-',label='pref BW')
        ax.plot(np.linspace(0,1,2)*TCfeatures['CarandiniParams']['sigmaD'][cel],
                                        0*ys + baseline + np.diff(ys)/20,'m-')
        ax.plot(np.linspace(0,1,2)*TCfeatures['CarandiniParams']['sigmaS'][cel],
                                        0*ys + baseline - np.diff(ys)/20,'y-')
        ax.text(0.4*TCfeatures['CarandiniParams']['sigmaD'][cel],
                            baseline + np.diff(ys)*1.5/20, '$\sigma_D$',color='m')
        ax.text(0.4*TCfeatures['CarandiniParams']['sigmaS'][cel],
                            baseline - np.diff(ys)*1.7/20, '$\sigma_S$',color='y')

        pars = (TCfeatures['MiscParams']['SuppInd_wBL'][cel],
                TCfeatures['CarandiniParams']['RD'][cel],
                TCfeatures['CarandiniParams']['RS'][cel],
                TCfeatures['CarandiniParams']['sigmaD'][cel],
                TCfeatures['CarandiniParams']['sigmaS'][cel])
    #     ax.text(2,baseline + np.diff(ys)*2/20,'$R_D$ = %2.1f' % pars[0])
    #     ax.text(2,baseline - np.diff(ys)*2/20,'$R_S$ = %2.1f' % pars[1])
    #    ax.set_title('Cell %i of %i,   %i%%' % (ii,len(R2),100 * ii//len(R2)))
        ax.set_title('$SI$ = %2.1f, $R_D$ = %2.1f, $R_S$ = %2.1f,\
                                $\sigma_D$ = %2.1f, $\sigma_S$ = %2.1f' % pars)
        ax.set_xlabel('Bandwidth (Octave)')
        ax.set_ylabel('Rate (Hz)')
        ax.legend()

        fig.savefig(PLOTS_DIR +CelltypeLabel+curve_type+str(ii)+'.png')
        plt.close(fig)
        ii +=1

#-------------------------------------------------------------------------------
elif nb_cell == 6:
    #check correlations of TC features with R^2 of fit
    TCfeaturesToPlot = TCfeatures['MiscParams'].copy()#copying dictionary
    TCparamNames = sorted(TCfeaturesToPlot.keys()) #the .keys()
    TCparamNames.insert(0,'R2')#'R2' is first name in list
    print(" , ".join(TCparamNames))
    TCfeaturesToPlot['R2'] = TCfeatures['GoF']['R2'] #adding R^2
    ParamsLabels = TCfeatures['MiscParamsLabels'].copy()#copying dictionary
    ParamsLabels['R2'] = TCfeatures['GoFLabels']['R2']

    betterOrder = [10,1,4,3,2,7,5,9,11,12,8,6]
    ylist = range(1,12+1)
    pairs = [[0, betterOrder[k-1]] for k in ylist]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,
                             figsize=(13,10),cols=4,wspace=.4,linecolor='r')

    del(TCfeaturesToPlot,ParamsLabels,TCparamNames)

#-------------------------------------------------------------------------------
elif nb_cell == 7:
    #check correlations of fit-parameters with R^2 of fit
    TCfeaturesToPlot = TCfeatures['CarandiniParams'].copy()#copying dictionary
    TCparamNames = sorted(TCfeaturesToPlot.keys())
    TCparamNames.insert(0,'R2')#'R2' is first name in list
    print(" , ".join(TCparamNames))
    TCfeaturesToPlot['R2'] = TCfeatures['GoF']['R2'] #adding R^2
    ParamsLabels = TCfeatures['CarandiniParamsLabels'].copy()#copying dictionary
    ParamsLabels['R2'] = TCfeatures['GoFLabels']['R2']

    #betterOrder = [10,1,4,3,2,7,5,9,11,12,8,6]
    ylist = range(1,6)
    betterOrder = ylist
    pairs = [[0, betterOrder[k-1]] for k in ylist]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,figsize=(13,8),cols=3,wspace=.3,linecolor='r')

    del(TCfeaturesToPlot,ParamsLabels,TCparamNames)

#-------------------------------------------------------------------------------
elif nb_cell == 8:
    #plot TC features histogram
    TCfeaturesToPlot = TCfeatures['MiscParams']
    ParamsLabels = TCfeatures['MiscParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot.keys())
    print(" , ".join(TCparamNames))

    betterOrder = [10,1,4,3,2,7,5,9,11,12,8,6]
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,nbins=15)
    print("Ncells = %1.0f" % TCfeatures['Ncells'])

#-------------------------------------------------------------------------------
elif nb_cell == 9:
    print("feature histograms from Raw TC's")
    tcfplot.makeHistograms(TCfeatures['RawMiscParams'], TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,nbins=15)


#-------------------------------------------------------------------------------
elif nb_cell == 10:
    n_ftr = 12
    pairs = list(zip(range(n_ftr),range(n_ftr)))
    betterOrder = [10,1,4,3,2,7,5,9,11,12,8,6]
    pairs = [[betterOrder[i]-1 for i in p] for p in pairs]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,
                           figsize=(15,15), cols=4, hspace=.3, wspace=.3,
                           yTCfeatures=TCfeatures['RawMiscParams'], axis_equal=True)
    print("\n\n Comparing raw (y-axis) vs. fit-curve-extracted TC features")

#-------------------------------------------------------------------------------
elif nb_cell == 11:
    # Correlations of preferred BW with other quantities:
    pairs = [(5,1),(6,1),(8,1),(9,1),(10,1),(3,1),(11,1)]
    pairs = [[betterOrder[i-1]-1 for i in p] for p in pairs]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,uniformYs=True,figsize=(15,10))

#-------------------------------------------------------------------------------
elif nb_cell == 12:
    # Correlations of suppression index with other quantities:
    pairs = [(5,3),(6,3),(8,3),(9,3),(10,3),(1,3),(11,3)]
    pairs = [[betterOrder[i-1]-1 for i in p] for p in pairs]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,uniformYs=True,figsize=(15,10))

#-------------------------------------------------------------------------------
elif nb_cell == 13:
    # Correlations of rates and responses (not interesting)
    inds = range(5,13) #for rates
    pairs = [(j,i) for i in inds for j in inds]
    pairs = [[betterOrder[i-1]-1 for i in p] for p in pairs]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,
                     cols=len(inds), figsize=(27/1.5,23/1.5), hspace=.4,
                     wspace=.4, uniformYs=True, symm=True)

#-------------------------------------------------------------------------------
elif nb_cell == 14:
    # histograms of difference of Gaussians parameters
    TCfeaturesToPlot = TCfeatures['CarandiniParams']
    ParamsLabels = TCfeatures['CarandiniParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot)
    print(" , ".join(TCparamNames))

    betterOrder = [1,4,2,3,5,6] #list(range(1,7))
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,figsize=(8,10),nbins=15)
    print("Ncells = %1.0f" % TCfeatures['Ncells'])

#-------------------------------------------------------------------------------
elif nb_cell == 15:
    # cross-correlations:
    inds = range(6)
    pairs = [(j,i) for i in inds for j in inds]
    betterOrder = [1,2,3,5,6,4]
    pairs = [[betterOrder[i]-1 for i in p] for p in pairs]
    tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,
                cols=len(inds), figsize=(14,14), hspace=.4, wspace=.4, symm=True)


#-------------------------------------------------------------------------------
elif nb_cell == 16:
    # cross-correlation of TC features and curve-family parameters
    betterOrder_x = [2,3,5,6]
    betterOrder_y = [10,1,4,8,6,12]
    pairs = [[i-1, j-1] for j in betterOrder_y for i in betterOrder_x ]
    sig_vec = tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels, pairs,
            cols=len(betterOrder_x),figsize=(14,18),hspace=.3,wspace=.2,uniformYs=True,
            uniformXs=True, yTCfeatures=TCfeatures['MiscParams'],
            yTCparamsNames=sorted(TCfeatures['MiscParams']),
            yTCparamsLabels = TCfeatures['MiscParamsLabels'])

#-------------------------------------------------------------------------------
elif nb_cell == 17:
    # the significant cross-correl's
    pairs1 = [pairs[i] for i in np.nonzero(sig_vec)[0]]
    sigs = tcfplot.makeScatterPlots(TCfeaturesToPlot, TCparamNames, ParamsLabels,
                     pairs1,cols=len(betterOrder_x),figsize=(14,6),wspace=.5,
                     yTCfeatures=TCfeatures['MiscParams'],
                     yTCparamsNames=sorted(TCfeatures['MiscParams']),
                     yTCparamsLabels=TCfeatures['MiscParamsLabels'])


#-------------------------------------------------------------------------------
elif nb_cell == 18:
    # Limiting to positively responsive cells (pie chart)
    cellinds = np.nonzero(TCfeatures['RawMiscParams']['peakResp']>0)[0] #== np.nonzero(TCfeatures['RawMiscParams']['prefBW']>0)[0])
    n1 = len(cellinds)
    Ncells = TCfeatures['Ncells']
    perc = 100*n1//Ncells
    #print(('\n\n %i%% (or %i/%i)   of ' % (perc,len(cellinds),TCfeatures['Ncells']))+CelltypeLabel+' cells have a positive response.')
    print(('\n\n %i/%i  of ' % (len(cellinds),TCfeatures['Ncells']))+CelltypeLabel+' cells have a positive response.')
    colors  = ['lightcoral', 'lightskyblue']
    labels = ['positive response','suppression only']
    plt.pie([n1,Ncells-n1], colors=colors, startangle=90,labels=labels, autopct='%1.0f%%');
    #plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.show()

#-------------------------------------------------------------------------------
elif nb_cell == 19:
    # features histograms after limiting to +response cells.
    TCfeaturesToPlot = TCfeatures['MiscParams']
    ParamsLabels = TCfeatures['MiscParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot)

    betterOrder = [10,4,2,7,5,9,11,12]
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,nbins=15,cell_inds=cellinds)
    print("Ncells = %1.0f" % n1)


#-------------------------------------------------------------------------------
elif nb_cell == 20:
    # curve parameters after limiting to +response cells.
    TCfeaturesToPlot = TCfeatures['CarandiniParams']
    ParamsLabels = TCfeatures['CarandiniParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot)

    betterOrder = [2,3,5,6] #list(range(1,7))
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,figsize=(8,10),nbins=15,cell_inds=cellinds)
    print("Ncells = %1.0f" % n1)

#-------------------------------------------------------------------------------
elif nb_cell == 21:
    # pie chart for SI>0 cells
    cellinds = np.nonzero(TCfeatures['MiscParams']['SuppInd_wBL']>0)[0] #== np.nonzero(TCfeatures['RawMiscParams']['prefBW']>0)[0])
    n1 = len(cellinds)
    Ncells = TCfeatures['Ncells']
    perc = 100*n1//Ncells
    #print(('\n\n %i%% (or %i/%i)   of ' % (perc,len(cellinds),TCfeatures['Ncells']))+CelltypeLabel+' cells have a positive response.')
    print(('\n\n %i/%i  of ' % (len(cellinds),TCfeatures['Ncells']))+CelltypeLabel+' cells have a suppression index > 0.')
    colors  = ['lightcoral', 'lightskyblue']
    labels = ['SI > 0','SI = 0']
    plt.pie([n1,Ncells-n1], colors=colors, startangle=90,labels=labels, autopct='%1.0f%%');
    #plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.show()

#-------------------------------------------------------------------------------
elif nb_cell == 22:
    # features histograms after limiting to SI>0 cells.
    TCfeaturesToPlot = TCfeatures['MiscParams']
    ParamsLabels = TCfeatures['MiscParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot)

    betterOrder = [10,4,2,7,5,9,11,12]
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,nbins=15,cell_inds=cellinds)
    print("Ncells = %1.0f" % n1)


#-------------------------------------------------------------------------------
elif nb_cell == 23:
    # curve parameters after limiting to SI>0 cells.
    TCfeaturesToPlot = TCfeatures['CarandiniParams']
    ParamsLabels = TCfeatures['CarandiniParamsLabels']
    TCparamNames = sorted(TCfeaturesToPlot)

    betterOrder = [2,3,5,6] #list(range(1,7))
    tcfplot.makeHistograms(TCfeaturesToPlot, TCparamNames,ParamsLabels, CelltypeLabel,betterOrder,figsize=(8,10),nbins=15,cell_inds=cellinds)
    print("Ncells = %1.0f" % n1)











# ==============================================================================
# Cute alternative for generating the whole notebook at once:
# The drawback is that
def notebook_generator(Celltype, CelltypeLabel, Response_type='sustained',
                                    mExp=None, BLisZeroBW=True, WNoctave=6):
    """ Use it to generates cells in a notebook
    Usage: in your jupyter nb run
    nb_cell = celltype_identified_TCfits_nb.notebook_generator(Celltype,
                      CelltypeLabel, Response_type, mExp, BLisZeroBW, WNoctave)
    #then in each cell run:
    next(nb_cell)
    """
#-------------------------------------------------------------------------------
    print('importing modules\n')
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import tqdm

    import tuning_curve_fit_funcs as tcfits
    import tuning_curve_feature_plotting as tcfplot
    from importlib import reload
    reload(tcfits)
    reload(tcfplot)

    yield
#-------------------------------------------------------------------------------
    PROJ_DIR = '/Users/yashar/Google-Drive/Documents/Work/Projects/'
    DATA_DIR = PROJ_DIR + 'AuditoryContextualModul/Matlab/Lakunina-Jaramillo-Data/'
    filename = 'photoidentified_cells_responsesTCs_new.npz'
    data = np.load(DATA_DIR + filename)
    print("\n".join(data.keys())) #:gives the list of arrays in the file
    data.close()

    yield
#-------------------------------------------------------------------------------
    TCfeatures = tcfits.fit_all_cells(DATA_DIR + filename, Celltype, Response_type, BLisZeroBW,
                             WNoctave,  mFixed=mExp, function_class_fit= "diff_of_gauss_fit")

    yield
#-------------------------------------------------------------------------------



# # dictionary comprehension:
# N_nb_cells = 10
# nb_cell_dict = dict((cell, 'nb_cell_'+str(cell)) for cell in range(N_nb_cells))
# # in python 3 can do:    nb_cell_dict = {cell:'nb_cell_'+str(cell) for cell in range(N_nb_cells)}
