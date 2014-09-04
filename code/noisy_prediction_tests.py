from hmf import HMF
from feature_extraction import FeatureExtractor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import use, cm
use('Agg')

import numpy as np
import matplotlib.pyplot as pl

def plot_results(x, y, yerr, nx, ny, predictions, plotname, xnames, xlabel, ylabel, rname, Ntrain,
                 noise, fs=5, colors=None, a=0.35, s=0.8, yrng=(0.025, 0.975),
                 crang=(0.025, 0.975),
                 noiseax='X and Y'):
    """
    Plot the results from regression.
    """
    title = 'N_train:%d, Ntest:%d, Regression: %s, Features: %s' % (Ntrain, x.size, rname,
                                                                    ', '.join(xnames))
    if colors is None:
        colors = 'k'
        kwargs = {'alpha':a, 'marker':'o', 'color':'k'}
    else:
        mn = np.sort(colors)[np.ceil(crang[0] * colors.size).astype(np.int)]
        mx = np.sort(colors)[np.floor(crang[1] * colors.size).astype(np.int)]
        kwargs = {'alpha':a, 'marker':'o', 'c':colors, 'cmap':cm.jet, 'vmin':mn, 'vmax':mx}

    residuals = y - predictions 
    fractional = residuals / y
    chi = (residuals / yerr)
    nr = y - ny
    nf = nr / y
    nc = (nr / yerr)
    
    f = pl.figure(figsize=(3 * fs, 3 * fs))
    pl.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125)
    pl.suptitle(title, fontsize=1200. / len(title))

    xs = [x, nx, x, x, x, x, x, x, x]
    ys = [y, ny, predictions, residuals, fractional, chi, nr, nf, nc]
    ylabels = [ylabel, ylabel, ylabel, 'Predictions vs Original', '', '', 'Noisy vs Original', '', '']
    tmp = np.sort(chi**2)
    tmp = tmp[0.025*tmp.size:0.975*tmp.size]
    titles = ['Original', noiseax + ' Noisified by %0.0f%%' % (noise * 100.), 'Predications',
              'Residuals, median:%0.2e' % np.median(residuals),
              'Frac. Residuals, median:%0.2e' % np.median(fractional),
              '$\chi^2$, median:%0.2e' % np.median(tmp ** 2. ),
              'Residuals, median:%0.2e' % np.median(nr),
              'Frac. Residuals, median:%0.2e' % np.median(nf),
              '$\chi^2$, median:%0.2e' % np.median(nc ** 2.)]
    ylims = [(-0.1, 0.5), (-0.1, 0.5), (-0.1, 0.5), (-0.1, 0.1), (-10, 10), (-2, 2), 
             (-0.1, 0.1), (-10, 10), (-2, 2)]
    for i in range(len(xs)):
        pl.subplot(3, 3, i + 1)
        pl.scatter(xs[i], ys[i], **kwargs)
        pl.xlabel(xlabel)
        pl.ylabel(ylabels[i])
        tmp = np.sort(ys[i])
        mn = tmp[np.ceil(y.size * yrng[0]).astype(np.int)]
        mx = tmp[np.floor(y.size * yrng[1]).astype(np.int)]
        #pl.ylim(mn, mx)
        pl.ylim(ylims[i][0], ylims[i][1])
        pl.title(titles[i])

    f.savefig(plotname)

    #bins = 64
    #x = np.linspace(-5, 5, 5000)
    #f = pl.figure(figsize=(2 * fs, fs))
    #pl.subplot(121)
    #pl.hist(chi2, bins, color='#FF9900', alpha=a, label='Predictions', normed=True)
    #pl.plot(x, np.exp(-x ** 2.) / np.sqrt(2. * np.pi), 'k')
    #pl.xlabel('$\chi$')
    #pl.subplot(122)
    #pl.hist(nc, bins, color='g', alpha=a, label='Noisy Data', normed=True)
    #pl.plot(x, np.exp(-x ** 2.) / np.sqrt(2. * np.pi), 'k')
    #pl.xlabel('$\chi$')
    #pl.savefig('../plots/foo.png')

def make_noisy_predictions(regressor, x, y, noisify='xy', noise_fraction=0.1, run_with_noise=False,
                           data_fraction=0.5):
    """
    Noisify the specified data and make predictions.
    """
    xd = x.features.shape[1] / 2
    yd = y.features.shape[1] / 2
    x_noise, y_noise = 0., 0.
    x_newerrs = x.features[:, xd:]
    y_newerrs = y.features[:, yd:]
    if 'x' in noisify: 
        x_noise, x_newerrs = x.noisify(noise_fraction)
    if 'y' in noisify: 
        y_noise, y_newerrs = y.noisify(noise_fraction)

    noisy_x = x.features[:, :xd] + x_noise
    noisy_y = y.features[:, :yd] + y_noise

    #from sklearn.decomposition import PCA, KernelPCA
    #m = KernelPCA()
    #proj = m.fit_transform(noisy_x)
    #noisy_x = np.hstack((noisy_x, proj))

    if run_with_noise:
        noisy_x = np.hstack((noisy_x, x_newerrs))

    # split the data
    ind = np.random.permutation(y_noise.shape[0])
    Ntrain = np.round(y_noise.shape[0] * data_fraction).astype(np.int)
    train_ind = ind[:Ntrain]
    test_ind = ind[Ntrain:]

    # run
    regressor.fit(noisy_x[train_ind], noisy_y[train_ind])
    predictions = regressor.predict(noisy_x[test_ind])

    return noisy_x, noisy_y, predictions, train_ind, test_ind

if __name__ == '__main__':

    import pyfits as pf

    seed = 1234
    np.random.seed(seed)

    n = '../data/sdss30k_rfadely.fit'
    f = pf.open(n)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)
    
    ylim = np.inf
    featurenames = ['cmodelmag', 'psffwhm', 'petror50', 'petror90']
    targetnames = ['psfmag', 'cmodelmag']
    filters = ['u', 'g', 'r', 'i', 'z']

    x = FeatureExtractor(data, featurenames, filters, color_band='r', scale_kind=None,
                         mag_range=None)
    data = data[x.idx]
    y = FeatureExtractor(data, targetnames, filters, color_band=None, scale_kind=None,
                         mag_range=None)

    # taylor to target, set for psf - model
    y.features[:, :5] = y.features[:, :5] - y.features[:, 5:10]
    y.features[:, 5:10] = np.sqrt(y.features[:, 10:15] ** 2. + y.features[:, 15:20] ** 2.)
    y.features = y.features[:, :10]

    # restrict x range
    xlim = (19.5, 20.5)
    ind = (x.features[:, 2] > xlim[0]) & (x.features[:, 2] < xlim[1])
    x.features = x.features[ind]
    y.features = y.features[ind]
    y.Ndata = y.features.shape[0]

    # restrict y range
    ylim = 0.5
    ind = y.features[:, 2] < ylim
    x.features = x.features[ind]
    y.features = y.features[ind]
    y.Ndata = y.features.shape[0]

    # specify scikit regressor
    rname = 'RF'
    if rname == 'KNN':
        rgr = KNeighborsRegressor(n_neighbors=8)
    if rname == 'RF':
        rgr = RandomForestRegressor(n_estimators=128)

    nf = 0.5
    nx, ny, pre, trn, test = make_noisy_predictions(rgr, x, y, noisify='xy',
                                                    noise_fraction=nf,
                                                    run_with_noise=True, data_fraction=0.5)

    xlabel = 'r mag'
    ylabel = 'r psfmag - modelmag'
    plotname = '../plots/foo.png'
    plot_results(x.features[test, 2], y.features[test, 2], y.features[test, 7], nx[test, 2],
                 ny[test, 2], pre[:, 2], plotname, featurenames, xlabel, ylabel, rname, trn.size, nf,
                 colors=x.features[test, 1])
