from feature_extraction import FeatureExtractor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import use, cm
use('Agg')

import numpy as np
import matplotlib.pyplot as pl

def plot_results(xaxis, data, predictions, plotname, xnames, yname, rname, fs=5, colors=None,
                 a=0.35, s=0.8, yrng=(0.025, 0.975), crang=(0.025, 0.975), yerrs=None):
    """
    Plot the results from regression.
    """
    title = 'N:%d, Regression: %s, Features: %s' % (data.size, rname, ', '.join(xnames))
    if colors is None:
        colors = 'k'
        kwargs = {'alpha':a, 'marker':'o', 'color':'k'}
    else:
        mn = np.sort(colors)[np.ceil(crang[0] * colors.size).astype(np.int)]
        mx = np.sort(colors)[np.floor(crang[1] * colors.size).astype(np.int)]
        kwargs = {'alpha':a, 'marker':'o', 'c':colors, 'cmap':cm.jet, 'vmin':mn, 'vmax':mx}

    residuals = data - predictions
    fractional = residuals / data
    xlabel = xaxis.keys()[0]
    x = xaxis[xlabel]

    f = pl.figure(figsize=(2 * fs, fs))
    pl.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125, hspace=0.5)
    pl.suptitle(title, fontsize=1000. / len(title))

    # the target
    pl.subplot(221)
    pl.scatter(x, data, **kwargs)
    pl.xlabel(xlabel)
    pl.ylabel(yname)
    tmp = np.sort(data)
    mn = tmp[np.ceil(data.size * yrng[0]).astype(np.int)]
    mx = tmp[np.floor(data.size * yrng[1]).astype(np.int)]
    pl.ylim(mn, mx)
    pl.title('Data')

    # predictions
    pl.subplot(222)
    pl.scatter(x, predictions, **kwargs)
    pl.xlabel(xlabel)
    pl.ylabel(yname)
    tmp = np.sort(predictions)
    pl.ylim(mn, mx)
    pl.title('Predictions')

    # residuals
    pl.subplot(223)
    pl.scatter(x, residuals, **kwargs)
    pl.xlabel(xlabel)
    pl.ylabel('Residuals')
    tmp = np.sort(residuals)
    pl.ylim(tmp[np.ceil(data.size * yrng[0]).astype(np.int)],
            tmp[np.floor(data.size * yrng[1]).astype(np.int)])
    pl.title('residuals, med. abs. err:%0.2e' % np.median(np.abs(residuals)))

    if yerrs is None:
        # fractional residuals
        pl.subplot(224)
        pl.scatter(x, fractional, **kwargs)
        pl.xlabel(xlabel)
        pl.ylabel('Fractional Residuals')
        tmp = np.sort(fractional)
        pl.ylim(tmp[np.ceil(data.size * yrng[0]).astype(np.int)],
                tmp[np.floor(data.size * yrng[1]).astype(np.int)])
        pl.title('residuals, med. abs. frac. err:%0.2e' % np.median(np.abs(fractional)))
    else:
        # chi
        chi2 = (residuals / yerrs) ** 2.
        pl.subplot(224)
        pl.scatter(x, chi2, **kwargs)
        pl.xlabel(xlabel)
        pl.ylabel('Fractional Residuals')
        tmp = np.sort(chi2)
        pl.ylim(0, 10)
        pl.title('$\chi^2$, med. $\chi^2$:%0.2e' % np.median(np.abs(chi2)))

    f.savefig(plotname)

if __name__ == '__main__':

    import pyfits as pf

    seed = 12345
    np.random.seed(seed)

    n = '../data/mr10k_short_rfadely.fit'
    f = pf.open(n)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    targetnames = ['psfmag', 'cmodelmag']
    featurenames = ['cmodelmag', 'psffwhm', 'petroR50']
    filters = ['u', 'g', 'r', 'i', 'z']

    # extract x and y
    x = FeatureExtractor(data, featurenames, filters, color_band='r', scale_kind=None,
                         mag_range=None)
    #proj, lats = x.run_hmf(9)
    #x.features[:, :10] = proj
    data = data[x.idx]
    y = FeatureExtractor(data, targetnames, filters, color_band=None, scale_kind=None,
                         mag_range=None)
    x.features = x.features[y.idx, :15]

    yerrs = np.sqrt(y.features[:, 10:15] ** 2. + y.features[:, 15:20] ** 2.)
    y.features = y.features[:, :5] - y.features[:, 5:10]

    #yerrs = y.features[:, 5:]
    #y.features = y.features[:, :5]
    ind = y.features[:, 2] < 1.0
    x.features = x.features[ind]
    y.features = y.features[ind]
    y.Ndata = y.features.shape[0]

    # split the data
    np.random.seed(seed)
    ind = np.random.permutation(y.Ndata)
    Ntrain = np.round(y.Ndata / 2.).astype(np.int)
    xtrain = x.features[ind[:Ntrain]]
    ytrain = y.features[ind[:Ntrain]]
    xtest = x.features[ind[Ntrain:]]
    ytest = y.features[ind[Ntrain:]]
    yerrs = yerrs[ind[Ntrain:]]

    # run
    rgr = KNeighborsRegressor()
    rgr.fit(xtrain, ytrain)
    predictions = rgr.predict(xtest)

    plot_results({'r mag':xtest[:, 2]}, ytest[:, 2], predictions[:, 2],
                 '../plots/psfminusmodel_mfp_knn_1.0.png',
                 featurenames, 'psfmag - modelmag', 'KNN', yerrs=yerrs[:, 2], colors=xtest[:, 1])
