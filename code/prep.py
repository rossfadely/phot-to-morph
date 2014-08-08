import numpy as np
import pyfits as pf

from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, scale=True, chop_missing=True):
    """
    Take SDSS data and construct colors, make any cuts, and whiten (maybe).
    """
    fn = ['u', 'g', 'r', 'i', 'z']
    # assign and extinction correct
    e1s = np.zeros((len(data.field(0)), len(fn)))
    e2s = np.zeros((len(data.field(0)), len(fn)))
    p50s = np.zeros((len(data.field(0)), len(fn)))
    psfmags = np.zeros((len(data.field(0)), len(fn)))
    modelmags = np.zeros((len(data.field(0)), len(fn)))
    for i, f in enumerate(fn):
        psfmags[:, i] = data.field('psfmag_' + f) - data.field('extinction_' + f)
        modelmags[:, i] = data.field(f) - data.field('extinction_' + f)
        #modelmags[:, i] = data.field('cmodelmag_' + f) - data.field('extinction_' + f)

        e1s[:, i] = data.field('me1_' + f)
        e2s[:, i] = data.field('me2_' + f)
        p50s[:, i] = data.field('petroR50_' + f)

    photo = np.hstack((modelmags, psfmags))
    # for now, just the ellipticities, and 0.5 petro radius
    morph = np.hstack((e1s, e2s, p50s))

    # get rid of samples with missing data if desired
    ind = None
    if chop_missing:
        ind = np.ones(morph.shape[0], dtype=np.bool)
        for i in range(morph.shape[0]):
            if np.any(morph[i, :] == -9999.) | np.any(photo == -9999.):
                ind[i] = False
        morph = morph[ind]
        photo = photo[ind]

    # subtract r model mag from everything but r model mag
    photo[:, :2] -= photo[:, 2, None]
    photo[:, 3:] -= photo[:, 2, None]

    # instantiate scaler
    scalers = None
    if scale:
        photo_scaler = MinMaxScaler()
        morph_scaler = MinMaxScaler()
        scalers = (photo_scaler, morph_scaler)

    # apply scaling
    if scalers is not None:
        photo = photo_scaler.fit_transform(photo)
        morph = morph_scaler.fit_transform(morph)

    return photo, morph, scalers, ind


if __name__ == '__main__':

    f = '../data/mr10k_1_rfadely.fit'
    f = pf.open(f)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    prepare_data(data)
