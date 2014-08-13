from prep import prepare_data
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, Lars, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
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

    if scalers is not None:
        photo = scalers[0].inverse_transform(photo)
        morph = scalers[1].inverse_transform(morph)
        pre = scalers[1].inverse_transform(pre)

    # plot
    a, s = 0.35, 0.85
    fs = 5
    Nm = morph.shape[1] / 5
    f = pl.figure(figsize=(2 * Nm * fs, fs))
    pl.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125)
    for i in range(Nm):
        residuals = morph[:, i * Nm + 2] - pre[:, i * Nm + 2]
        fracres = residuals / morph[:, i* Nm + 2]
        pl.suptitle(title, fontsize=(1000. / len(title)))
        pl.subplot(2, Nm * 2, i * Nm + 1)
        pl.scatter(photo[:, 2], morph[:, i * Nm + 2], c=photo[:, i], cmap=cm.jet, alpha=a,
                   marker='o')
        pl.colorbar(shrink=s)
        pl.ylabel('Observed PetroR50')
        pl.title('N:%d, Med. RMSE:%0.2e' % (photo.shape[0], np.median(np.sqrt(residuals ** 2.))))
        pl.xlim(photo[:, 2].min(), np.minimum(photo[:, 2].max(), 23))
        tmp = np.sort(morph[:, i * Nm + 2])
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.subplot(2, Nm * 2, i * Nm + 2)
        pl.scatter(photo[:, 2], pre[:, i * Nm + 2], c=photo[:, i], cmap=cm.jet, alpha=a,
                   marker='o')
        pl.colorbar(shrink=s)
        pl.title('N:%d, Med. FRMSE:%0.2e' % (photo.shape[0], np.median(np.sqrt((fracres) ** 2.))))
        pl.xlim(photo[:, 2].min(), np.minimum(photo[:, 2].max(), 23))
        tmp = np.sort(pre[:, i * Nm + 2])
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.ylabel('Predicted PetroR50')
        pl.subplot(2, Nm * 2, i * Nm + 3)
        pl.scatter(photo[:, 2], morph[:, i * Nm + 2] - pre[:, i * Nm + 2], c=photo[:, i],
                   cmap=cm.jet, alpha=a, marker='o')
        tmp = np.sort(morph[:, i * Nm + 2] - pre[:, i * Nm + 2])
        pl.colorbar(shrink=s)
        pl.xlim(photo[:, 2].min(), np.minimum(photo[:, 2].max(), 23))
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.ylabel('Residuals')
        pl.xlabel('r mag')
        pl.subplot(2, Nm * 2, i * Nm + 4)
        pl.scatter(photo[:, 2], (morph[:, i * Nm + 2] - pre[:, i * Nm + 2]) / morph[:, i* Nm + 2],
                   c=photo[:, i],
                   cmap=cm.jet, alpha=a, marker='o')
        tmp = np.sort(morph[:, i * Nm + 2] - pre[:, i * Nm + 2])
        pl.colorbar(shrink=s)
        pl.xlim(photo[:, 2].min(), np.minimum(photo[:, 2].max(), 23))
        pl.ylim(tmp[lims[0] * tmp.size], tmp[lims[1] * tmp.size])
        pl.ylabel('Fractional Residuals')
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

    features = ['modelmags', 'psffwhms']
    targets = ['p50']
    mrng = np.array([15, 22])
    photo, morph, scalers, ind = prepare_data(data, features, ['p50'],
                                              color_range=(0.025, 0.975), scale=None,
                                              r_modelmag_range=mrng)
    
    #determine_parms(photo, morph, SVR(), seed, 'SVR', 'svr_grid.txt')

    rgrname = 'SVM'
    rgr = SVR()
    title = 'Regression: %s, Features: (%s)' % (rgrname, ', '.join(features))
    plotname = '../plots/p50_mf_r_%s_%0.2f-%0.2f.png' % (rgrname, mrng[0], mrng[1])
    plot_r_predictions(photo, morph, scalers, rgr, seed, title, plotname, single=True)
