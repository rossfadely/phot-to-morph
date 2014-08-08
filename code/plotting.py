from prep import prepare_data
from matplotlib import use
use('Agg')

import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

def plot_morph(mags, morph, kind, labels, plotname, axlab, lims=(0, 0.95)):
    """
    Simple scatter plot of morphological parameters as a fn of magnitude.
    """
    assert morph.shape[1] % 5 == 0, 'All ugriz bands are not present'
    f = pl.figure(figsize=(25,5))
    for i in range(5):
        pl.subplot(1, 5, i + 1)
        tmp = np.sort(morph[:, i])
        ind = kind == 6 # stars
        pl.plot(morph[ind, i], mags[ind, i], 'g.', alpha=0.3, mec='none')
        ind = kind == 3 # gals
        pl.plot(morph[ind, i], mags[ind, i], 'b.', alpha=0.3, mec='none')
        pl.title(labels[i])
        pl.xlim(tmp[np.ceil(lims[0] * tmp.size).astype(np.int)],
                tmp[np.floor(lims[1] * tmp.size).astype(np.int)])
        pl.xlabel(axlab)
        pl.ylabel('magnitude')
    f.savefig(plotname)
    pl.close()

if __name__ == '__main__':

    f = '../data/mr10k_cmodel_rfadely.fit'
    f = pf.open(f)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    f = pl.figure()
    ind = data.field('type') == 3
    x = data.field('psfmag_r')[ind]
    y = x - data.field('cmodelmag_r')[ind]
    pl.plot(x, y, 'b.', alpha=0.3, mec='none')
    ind = data.field('type') == 6
    x = data.field('psfmag_r')[ind]
    y = x - data.field('cmodelmag_r')[ind]
    pl.plot(x, y, 'g.', alpha=0.3, mec='none')
    pl.plot([14, 24], [0.145, 0.145], 'k', alpha=0.2)
    pl.xlabel('r psfmag')
    pl.ylabel('r psfmag - r cmodelmag')
    pl.ylim(-0.2, 0.4)
    f.savefig('../plots/foo.png')

    assert 0
    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    photo, morph, scalers, ind = prepare_data(data)

    photo = scalers[0].inverse_transform(photo)
    morph = scalers[1].inverse_transform(morph)

    labels = ['u', 'g', 'r', 'i', 'z']
    kind = data.field('type')[ind]

    mags = photo[:, :5]
    mags[:, :2] += mags[:, 2, None]
    mags[:, 3:] += mags[:, 2, None]
    plot_morph(mags, morph[:, -5:], kind, labels, '../plots/petro50_1.png', 'PetroR50')
    
    f = pl.figure()
    ind = kind == 6
    pl.plot(mags[ind, 2], photo[ind, 7], 'g.', alpha=0.3, mec='none')
    ind = kind == 3
    pl.plot(mags[ind, 2], photo[ind, 7], 'b.', alpha=0.3, mec='none')
    pl.ylabel('r psf mag - r model mag')    
    pl.xlabel('r psf mag')
    f.savefig('../plots/rband_psfminus_model.png')

    f = pl.figure()
    ind = kind == 6
    pl.plot(mags[ind, 3], photo[ind, 8] + mags[ind, 2] - mags[ind, 3], 'g.', alpha=0.3, mec='none')
    ind = kind == 3
    pl.plot(mags[ind, 3], photo[ind, 8] + mags[ind, 2] - mags[ind, 3], 'b.', alpha=0.3, mec='none')
    pl.ylabel('i psf mag - i model mag')    
    pl.xlabel('i psf mag')
    f.savefig('../plots/iband_psfminus_model.png')

    f = pl.figure()
    ind = kind == 6
    pl.plot(mags[ind, 1], photo[ind, 6] + mags[ind, 2] - mags[ind, 1], 'g.', alpha=0.3, mec='none')
    ind = kind == 3
    pl.plot(mags[ind, 1], photo[ind, 6] + mags[ind, 2] - mags[ind, 1], 'b.', alpha=0.3, mec='none')
    pl.ylabel('g psf mag - g model mag')    
    pl.xlabel('g psf mag')
    f.savefig('../plots/gband_psfminus_model.png')
