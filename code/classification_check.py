from matplotlib import use
use('Agg')

import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

def flux_summed_mags(mags):
    """
    Compute mags from sum of fluxes.
    """
    bs = np.array([1.4, 0.9, 1.2, 1.8, 7.4]) * 1.e-10
    # u band
    mags[:, 0] -= 0.04
    # z band
    mags[:, -1] += 0.02
    
    fluxes = mags * -0.4 * np.log(10.0)
    fluxes -= np.log(bs)[:, None]
    fluxes *= 2. * bs[:, None]
    fluxes = np.sinh(fluxes)

    fluxes = np.sum(fluxes, axis=0)

    return -2.5 * np.log10(fluxes)

def check_plot(data, modelmags, psfmags, name):
    """
    Simple scatter plot to verify sdss class
    """
    f = pl.figure()
    ind = data.field('type') == 3
    pl.plot(data.field('psfmag_r')[ind], (psfmags - modelmags)[ind], 'b.', alpha=0.3, mec='none')
    ind = data.field('type') == 6
    pl.plot(data.field('psfmag_r')[ind], (psfmags - modelmags)[ind], 'g.', alpha=0.3, mec='none')
    pl.ylim(-0.2, 0.3)
    f.savefig(name)

if __name__ == '__main__':

    f = '../data/mr10k_fluxes_rfadely.fit'
    f = pf.open(f)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    cmodelmags = np.array([data.field('cmodelmag_u'),
                           data.field('cmodelmag_g'),
                           data.field('cmodelmag_r'),
                           data.field('cmodelmag_i'),
                           data.field('cmodelmag_z')])
    cmodelmags = flux_summed_mags(cmodelmags)

    psfmags = np.array([data.field('psfmag_u'),
                           data.field('psfmag_g'),
                           data.field('psfmag_r'),
                           data.field('psfmag_i'),
                           data.field('psfmag_z')])
    psfmags = flux_summed_mags(psfmags)


    check_plot(data, cmodelmags, psfmags, '../plots/foo.png')
