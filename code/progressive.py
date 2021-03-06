from feature_extraction import FeatureExtractor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import use, cm
use('Agg')

import numpy as np
import matplotlib.pyplot as pl

def running_predictor(rgr, x, y, plotname, Npasses=1, window=0.1, use_predicted=False,
                      run_with_noise=True, mag_band=2, start=19., fs=5, a=0.2, labels=None):
    """
    Start at high S/N and predict downward.
    """
    if start is None:
        start = np.min(x[:, mag_band])

    for i in range(Npasses):
        
        # make a pass
        yold = y.copy()
        current = start
        while True:
            hiind = np.where((x[:, mag_band] >= current) & (x[:, mag_band] < current + window))
            loind = np.where((x[:, mag_band] >= current + window) &
                             (x[:, mag_band] < current + 2 * window))
            if loind[0].size == 0:
                break
            if use_predicted:
                rgr.fit(x[hiind], y[hiind])
            else:
                rgr.fit(x[hiind], yold[hiind])
            try:
                y[loind] = rgr.predict(x[loind])
            except:
                pass
            current += window
            print i, current

        f = pl.figure(figsize=(2 * fs, fs))
        ind = (x[:, mag_band] >= start + window) & (x[:, mag_band] < current + 2 * window)
        if labels is not None:
            gidx = (labels == 0) & ind
            sidx = (labels == 1) & ind
            sNold = len(yold[sidx & (yold < 0.1)])
            gNold = len(yold[gidx & (yold > 0.1)]) 
            sN = len(y[sidx & (y < 0.1)])
            gN = len(y[gidx & (y > 0.1)]) 
        pl.subplot(121)
        #pl.plot(x[ind, mag_band], yold[ind, mag_band], 'ok', alpha=a)
        pl.plot(x[ind, mag_band], yold[ind], '.k', alpha=a)
        pl.plot(x[gidx, mag_band], yold[gidx], '.b')
        pl.plot(x[sidx, mag_band], yold[sidx], '.g')
        pl.ylim(-0.1, 0.5)
        pl.title('Original, Nstar %d, Ngal %d (0.1)' % (sNold, gNold))
        pl.subplot(122)
        #pl.plot(x[ind, mag_band], y[ind, mag_band], 'ok', alpha=a)
        pl.plot(x[ind, mag_band], y[ind], '.k', alpha=a)
        pl.plot(x[gidx, mag_band], y[gidx], '.b')
        pl.plot(x[sidx, mag_band], y[sidx], '.g')
        pl.title('Predicted, Nstar %d, Ngal %d (0.1)' % (sN, gN))
        pl.ylim(-0.1, 0.5)
        f.savefig(plotname + '_%d.png' % i)
        print plotname + '_%d.png' % i
        yold = y.copy()

def prep(n, labelcol):
    """
    Load the data
    """
    f = pf.open(n)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    try:
        labels = data.field(labelcol)
    except:
        labels = np.zeros(data.field(0).size) - 99

    ylim = np.inf
    featurenames = ['cmodelmag', 'psffwhm', 'petror50', 'petror90']
    targetnames = ['psfmag', 'cmodelmag']
    filters = ['u', 'g', 'r', 'i', 'z']

    x = FeatureExtractor(data, featurenames, filters, color_band='r', scale_kind=None,
                         mag_range=None)
    data = data[x.idx]
    labels = labels[x.idx]
    y = FeatureExtractor(data, targetnames, filters, color_band=None, scale_kind=None,
                         mag_range=None)

    # taylor to target, set for psf - model                                                      
    y.features[:, :5] = y.features[:, :5] - y.features[:, 5:10]
    y.features[:, 5:10] = np.sqrt(y.features[:, 10:15] ** 2. + y.features[:, 15:20] ** 2.)
    y.features = y.features[:, :10]

    # restrict y range                                                                           
    ylim = 10.
    ind = y.features[:, 2] < ylim
    x.features = x.features[ind]
    y.features = y.features[ind]
    labels = labels[ind]
    y.Ndata = y.features.shape[0]

    return x, y, labels

if __name__ == '__main__':

    import pyfits as pf

    seed = 1234
    np.random.seed(seed)

    n = '../data/bright_bwmatches.fits'
    xl, yl, labels = prep(n, 'bwtype')    
    n = '../data/sdss30k_rfadely.fit'
    x, y, l = prep(n, 'bwtype')

    x.features = np.vstack((x.features, xl.features))
    y.features = np.vstack((y.features, yl.features))
    labels = np.append(l, labels)

    # specify scikit regressor                                                                   
    rname = 'RF'
    if rname == 'KNN':
        rgr = KNeighborsRegressor(n_neighbors=5)
    if rname == 'RF':
        rgr = RandomForestRegressor(n_estimators=64)

    running_predictor(rgr, x.features, y.features[:, 2], '../plots/progressive_%s' % rname,
                      Npasses=1, labels=labels,
                      window=0.1, use_predicted=False, run_with_noise=False, start=16)
