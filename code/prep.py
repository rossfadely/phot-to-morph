import numpy as np
import pyfits as pf

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def prepare_data(data, photo_features, morph_targets, scale='MinMax', chop_missing=True,
                 color_range=(0.025, 0.925), r_modelmag_range=None):
    """
    Take SDSS data and construct colors, make any cuts, and whiten (maybe).
    """
    fn = ['u', 'g', 'r', 'i', 'z']

    # assign and extinction correct
    e1s = np.zeros((len(data.field(0)), len(fn)))
    e2s = np.zeros((len(data.field(0)), len(fn)))
    p50s = np.zeros((len(data.field(0)), len(fn)))
    psfmags = np.zeros((len(data.field(0)), len(fn)))
    psffwhms = np.zeros((len(data.field(0)), len(fn)))
    modelmags = np.zeros((len(data.field(0)), len(fn)))
    for i, f in enumerate(fn):
        psfmags[:, i] = data.field('psfmag_' + f) - data.field('extinction_' + f)
        psffwhms[:, i] = data.field('psffwhm_' + f)
        modelmags[:, i] = data.field(f) - data.field('extinction_' + f)

        e1s[:, i] = data.field('me1_' + f)
        e2s[:, i] = data.field('me2_' + f)
        p50s[:, i] = data.field('petroR50_' + f)

    # cut on magnitude range
    if r_modelmag_range is not None:
        ind = (modelmags[:, 2] > r_modelmag_range[0]) & (modelmags[:, 2] < r_modelmag_range[1])
        psfmags = psfmags[ind]
        psffwhms = psffwhms[ind]
        modelmags = modelmags[ind]
        e1s = e1s[ind]
        e2s = e2s[ind]
        p50s = p50s[ind]

    # subtract a filter
    color_filt = 2
    ind = np.delete(range(5), color_filt)
    psfmags[:, ind] -= psfmags[:, color_filt, None]
    modelmags[:, ind] -= modelmags[:, color_filt, None]

    # build initial feature matrix
    for i, f in enumerate(photo_features):
        if f == 'psfmags':
            new = psfmags
        elif f == 'modelmags':
            new = modelmags
        if i == 0:
            photo = new
        else:
            photo = np.hstack((photo, new))
    for i, f in enumerate(morph_targets):
        if f == 'p50':
            new = p50s
        elif f == 'e1':
            new = e1s
        elif f == 'e2':
            new = e2s
        if i == 0:
            morph = new
        else:
            morph = np.hstack((morph, new))

    # restrict color range
    mn = 12
    if color_range is not None:
        for i in range(photo.shape[1]):
            if photo[:, i].min() > mn:
                pass
            else:
                bnd = np.sort(photo[:, i])[np.int(np.ceil(color_range[0] * photo.shape[0]))]
                ind = photo[:, i] < bnd
                photo[ind, i] = bnd
                bnd = np.sort(photo[:, i])[np.int(np.floor(color_range[1] * photo.shape[0]))]
                ind = photo[:, i] > bnd
                photo[ind, i] = bnd

    # tag on psffwhms
    if photo_features[-1] == 'psffwhms':
        photo = np.hstack((photo, psffwhms))

    # get rid of samples with missing data if desired
    ind = None
    if chop_missing:
        ind = np.ones(morph.shape[0], dtype=np.bool)
        for i in range(morph.shape[0]):
            if np.any(morph[i, :] == -9999.) | np.any(photo == -9999.):
                ind[i] = False
        morph = morph[ind]
        photo = photo[ind]

    # instantiate scaler
    scalers = None
    if scale == 'MinMax':
        photo_scaler = MinMaxScaler()
        morph_scaler = MinMaxScaler()
    if scale == 'Whiten':
        photo_scaler = StandardScaler()
        morph_scaler = StandardScaler()
        
    # apply scaling
    if scale is not None:
        scalers = (photo_scaler, morph_scaler)
        photo = photo_scaler.fit_transform(photo)
        morph = morph_scaler.fit_transform(morph)

    return photo, morph, scalers, ind


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

    photo, morph, scalers, ind = prepare_data(data)
