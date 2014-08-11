from prep import prepare_data
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import use, cm
use('Agg')

import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

def evaluate(x, y, regressor, seed, Nfolds, single=False):
    """
    Evaluate the RMSE of the prediction for the given regressor, using Nfolds.
    """
    Ntrain = np.round(1. / Nfolds * photo.shape[0]).astype(np.int)
    np.random.seed(seed)
    rmses = np.zeros(Nfolds)
    for i in range(Nfolds):
        inds = np.random.permutation(x.shape[0])
        x_train, y_train = x[inds[:Ntrain]], y[inds[:Ntrain]]
        x_test, y_test = x[inds[Ntrain:]], y[inds[Ntrain:]]
        if single:
            pre = np.zeros(y_test.shape)
            for i in range(morph.shape[1]):
                rgr = regressor.fit(x_train, y_train[:, i])
                pre[:, i] = regressor.predict(x_test)
        else:
            rgr = regressor.fit(x_train, y_train)
            pre = regressor.predict(x_test)
        rmses[i] = np.sum(np.sqrt((y_test - pre) ** 2.))
    return np.mean(rmses)

def determine_parms(photo, morph, regressor, seed, kind, logfilename, Nfolds=10):
    """
    Use CV to determine parameters
    """
    f = open(logfilename, 'w')
    if kind == 'SVR':
        f.write('# c, gamma, best_rmse, current_rmse\n')
        # made up ranges to explore
        cs = np.exp(np.linspace(-2, 6, 9))
        gs = np.exp(np.linspace(-5, 4, 10))
        best_rmse = np.inf
        for c in cs:
            for g in gs:
                regressor.set_params(C=c, gamma=g)
                rmse = evaluate(photo, morph, regressor, seed, Nfolds, single=True)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_c = c
                    best_g = g
                print '%0.4e, %0.4e, %0.4e, %0.4e\n' % (c, g, best_rmse, rmse)
                f.write('%0.4e, %0.4e, %0.4e, %0.4e\n' % (c, g, best_rmse, rmse))
    f.close()

def plot_r_predictions(photo, morph, scalers, regressor, seed, title, plotname, single=False,
                       Nfolds=2, lims=(0.075, 0.925)):
    """
    Plot the predicted values of morphology as a fn of magnitude.
    """
    # predict on the Nfold of data
    np.random.seed(seed)
    Ntrain = np.round(1. / Nfolds * photo.shape[0]).astype(np.int)
    inds = np.random.permutation(photo.shape[0])
    x_train, y_train = photo[inds[:Ntrain]], morph[inds[:Ntrain]]
    x_test, y_test = photo[inds[Ntrain:]], morph[inds[Ntrain:]]
    if single:
        pre = np.zeros(y_test.shape)
        for i in range(morph.shape[1]):
            rgr = regressor.fit(x_train, y_train[:, i])
            pre[:, i] = regressor.predict(x_test)
    else:
        rgr = regressor.fit(x_train, y_train)
        pre = regressor.predict(x_test)

    photo = x_test
    morph = y_test

    photo = scalers[0].inverse_transform(photo)
    morph = scalers[1].inverse_transform(morph)
    pre = scalers[1].inverse_transform(pre)

    # plot
    a, s = 0.35, 0.85
    fs = 10
    Nm = morph.shape[1] / 5
    f = pl.figure(figsize=(Nm * fs, fs))
    for i in range(Nm):
        pl.subplot(3, Nm, i * Nm + 1)
        pl.scatter(photo[:, 2], morph[:, i * Nm + 2], c=photo[:, i], cmap=cm.jet, alpha=a,
                   marker='o')
        pl.colorbar(shrink=s)
        pl.ylabel('Observed PetroR50')
        pl.title(title)
        pl.xlim(14, 22)
        tmp = np.sort(morph[:, i * Nm + 2])
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.subplot(3, Nm, i * Nm + 2)
        pl.scatter(photo[:, 2], pre[:, i * Nm + 2], c=photo[:, i], cmap=cm.jet, alpha=a,
                   marker='o')
        pl.colorbar(shrink=s)
        pl.xlim(14, 22)
        tmp = np.sort(pre[:, i * Nm + 2])
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.ylabel('Predicted PetroR50')
        pl.subplot(3, Nm, i * Nm + 3)
        pl.scatter(photo[:, 2], morph[:, i * Nm + 2] - pre[:, i * Nm + 2], c=photo[:, i],
                   cmap=cm.jet, alpha=a, marker='o')
        tmp = np.sort(morph[:, i * Nm + 2] - pre[:, i * Nm + 2])
        pl.colorbar(shrink=s)
        pl.xlim(14, 22)
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.ylabel('Residuals')
        pl.xlabel('r mag')
    f.savefig(plotname)

if __name__ == '__main__':

    seed = 12345

    f = '../data/mr10k_fluxes_rfadely.fit'
    f = pf.open(f)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    photo, morph, scalers, ind = prepare_data(data, ['modelmags'], ['p50'], add_fwhms=True,
                                              color_range=(0.025, 0.975), scale='Whiten')

    determine_parms(photo, morph, SVR(), seed, 'SVR', 'svr_grid.txt')


    assert 0
    plot_r_predictions(photo, morph, scalers, SVR(), seed,
                       'modelmags, psffwhm', '../plots/p50_mf_r_svr.png', single=True)
    plot_r_predictions(photo, morph, scalers, LinearRegression(), seed,
                       'modelmags, psffwhm', '../plots/p50_mf_r_linear.png')
