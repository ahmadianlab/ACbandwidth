import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#-------------------------------------------------------------------------------
def makeHistograms(TCfeatures, TCparamNames, ParamLabels, CelltypeLabel,betterOrder,\
                    cell_inds=None,nbins=30,cols=2,figsize=(10,17),hspace=.5):
    rows = 1 + (len(TCparamNames) // cols)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=hspace)
    for sp in range(len(betterOrder)):#
        ax = fig.add_subplot(rows,cols,sp+1) #plt.subplot(rows,cols,sp+1)
        paramName = TCparamNames[betterOrder[sp]-1]
        xs = TCfeatures[paramName]
        if cell_inds is not None:
            xs = xs[cell_inds]
        #n, bins, patches = plt.hist(xs[np.isfinite(xs)], bins = nbins, facecolor='b',alpha = 0.6)
        ax.hist(xs[np.isfinite(xs)], bins = nbins, facecolor='b',alpha = 0.6)
        ax.axvline(x=np.median(xs),c='r',linestyle='--')
        ax.axvline(x=np.mean(xs),c='g',linestyle='--')
        if sp%cols==0:
            ax.set_ylabel(CelltypeLabel+ ' Cell Count') #plt.ylabel(...)
        ax.set_xlabel(ParamLabels[paramName])  #plt.xlabel(...)
        xpos = ax.get_xlim()[0] + np.diff(ax.get_xlim())*.3 #plt.gca().get_xlim instead of ax.get_xlim
        relypos = 0.6
        dy = 0.13
        ax.text(xpos, ax.get_ylim()[1]*(relypos+dy), 'median = %1.2f' % np.median(xs),color='red')#plt.gca().get_ylim instead of ax.get_ylim
        ax.text(xpos, ax.get_ylim()[1]*relypos, 'mean = %1.2f' % np.mean(xs),color='green')
        ax.text(xpos, ax.get_ylim()[1]*(relypos-dy), 'SD = %1.2f' % np.std(xs))
        #ax.axis('tight') #plt.axis('tight')
        #ax.grid(True)
    #fig.show()   ##plt.show()

#-------------------------------------------------------------------------------
def makeScatterPlots(TCfeatures, TCparamNames, ParamLabels, pairs,\
        uniformYs=False, uniformXs = False, symm=False,\
        cols=3,figsize=(15,8),hspace=.3,wspace=None,linecolor='k',\
        cell_inds=None, yTCfeatures=None, yTCparamsNames=None, yTCparamsLabels=None):

    if yTCfeatures is not None:
        if yTCparamsNames is None:
            yTCparamsNames = TCparamNames
            yTCparamsLabels = ParamLabels
        else:
            assert yTCparamsLabels is not None #if yTCparamsNames is not none, then yTCparamsLabels cannot be None either

    sig_vec = np.zeros((len(pairs),))
    rows = 1 + (len(pairs) // cols)
    plt.figure(figsize=figsize)
    if wspace is None:
        plt.subplots_adjust(hspace=hspace)
    else:
        plt.subplots_adjust(hspace=hspace,wspace=wspace)
    for sp in range(len(pairs)):
        col = 1 + (sp % cols)
        row = 1 + (sp // cols)
        if col>row or (not symm):
            plt.subplot(rows,cols,sp+1)
            x_paramName = TCparamNames[pairs[sp][0]]
            xs = TCfeatures[x_paramName]
            if cell_inds is not None:
                xs = xs[cell_inds]
            xslabel = ParamLabels[x_paramName]
            if yTCfeatures is None:
                y_paramName = TCparamNames[pairs[sp][1]]
                ys = TCfeatures[y_paramName]
                if cell_inds is not None:
                    ys = ys[cell_inds]
                yslabel = ParamLabels[y_paramName]
            else:
                y_paramName = yTCparamsNames[pairs[sp][1]]
                ys = yTCfeatures[y_paramName]
                yslabel = yTCparamsLabels[y_paramName]
            #    a,b,r,p = linregress(xs,ys)     #according to help, linregress should work like next line, but it doesn't:
            linobj = linregress(xs,ys)
            a, b, r, p = linobj.slope, linobj.intercept, linobj.rvalue, linobj.pvalue
            if p<0.05:
                sig_vec[sp] = 1
                c_scat = 'b'
                c_line = linecolor #'k'
            else:
                c_scat = 'k'
                c_line = 'b'
            plt.scatter(xs,ys,c=c_scat,s = 4.5**2,alpha = .4) #s= (plt.rcParams['lines.markersize']/2) ** 2)
            xs1 = np.asarray(plt.gca().get_xlim())
            ys1 = np.asarray(plt.gca().get_ylim())
            plt.plot(xs1,a*xs1 + b, c_line)
            plt.scatter(np.mean(xs),np.mean(ys),c='#22bb22',s = 6**2,alpha = 1)
            plt.scatter(np.median(xs),np.median(ys),c='r',s = 5**2,alpha = 1)
            plt.text(xs1[0] + .5 * np.diff(xs1),ys1[0] + .6 * np.diff(ys1),'r = %1.2f\np = %1.3f' % (r,p))
            if (not uniformXs and not (symm and col-row != 1) ) or (uniformXs and row == 1 + ((len(pairs)-1) // cols)):
                plt.xlabel(xslabel)
            if (symm and col-row==1) or (not symm and (not uniformYs or sp%cols==0)):
                plt.ylabel(yslabel)
    return sig_vec
