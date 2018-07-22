import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#-------------------------------------------------------------------------------
def makeHistograms(TCfeatures, TCparamNames, ParamLabels, CelltypeLabel,betterOrder,nbins=30,cols=2,figsize=(10,17),hspace=.5):
    rows = 1 + (len(TCparamNames) // cols)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    for sp in range(len(TCparamNames)):
        plt.subplot(rows,cols,sp+1)
        paramName = TCparamNames[betterOrder[sp]-1]
        xs = TCfeatures[paramName]
        #n, bins, patches = plt.hist(xs[np.isfinite(xs)], bins = nbins, facecolor='b',alpha = 0.6)
        plt.hist(xs[np.isfinite(xs)], bins = nbins, facecolor='b',alpha = 0.6)
        plt.axvline(x=np.median(xs),c='r',linestyle='--')
        plt.axvline(x=np.mean(xs),c='g',linestyle='--')
        if sp%cols==0:
            plt.ylabel(CelltypeLabel+ ' Cell Count')
        plt.xlabel(ParamLabels[paramName])
        xpos = plt.gca().get_xlim()[0] + np.diff(plt.gca().get_xlim())*.3
        relypos = 0.6
        dy = 0.13
        plt.text(xpos, plt.gca().get_ylim()[1]*(relypos+dy), 'median = %1.2f' % np.median(xs),color='red')
        plt.text(xpos, plt.gca().get_ylim()[1]*relypos, 'mean = %1.2f' % np.mean(xs),color='green')
        plt.text(xpos, plt.gca().get_ylim()[1]*(relypos-dy), 'SD = %1.2f' % np.std(xs))
        #plt.axis([40, 160, 0, 0.03])
        #plt.grid(True)
        #plt.show()

#-------------------------------------------------------------------------------
def makeScatterPlots(TCfeatures, TCparamNames, ParamLabels, pairs,uniformYs=False,cols=3,figsize=(15,8),hspace=.3,wspace=None,symm=False,linecolor='k'):
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
            y_paramName = TCparamNames[pairs[sp][1]]
            ys = TCfeatures[y_paramName]
            #    a,b,r,p = linregress(xs,ys)     #according to help, linregress should work like next line, but it doesn't:
            linobj = linregress(xs,ys)
            a, b, r, p = linobj.slope, linobj.intercept, linobj.rvalue, linobj.pvalue
            if p<0.05:
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
            if not (symm and col-row != 1):
                plt.xlabel(ParamLabels[x_paramName])
            if (symm and col-row==1) or (not symm and (not uniformYs or sp%cols==0)):
                plt.ylabel(ParamLabels[y_paramName])
