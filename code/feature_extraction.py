import numpy as np

from hmf import HMF
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureExtractor(object):
    """
    Feature extraction from a SDSS pyfits data structure.
    """
    def __init__(self, data, feature_names, filters, add_errs=True, color_band='r',
                 scale_kind='MinMax', mag_range=None, cut_missing=True):
        """
        :data: pyfits table data structure
        :feature_name: list of strings giving features to be extracted
        """
        self.Ndata = len(data.field(0))
        self.Nfeatures = len(feature_names) * len(filters) 
        
        self.idx = np.arange(self.Ndata, dtype=np.int)
        self.scaler = None
        self.filters = filters
        self.feature_names = feature_names

        # build initial array of features
        ind = 0
        self.features = np.zeros((self.Ndata, self.Nfeatures))
        for i, n in enumerate(self.feature_names):
            for f in filters:
                self.features[:, ind] = data.field(n + '_' + f).copy()
                if 'mag' in n:
                    self.features[:, ind] -= data.field('extinction_' + f)
                ind += 1

        if add_errs:
            self.features = np.hstack((self.features, np.zeros_like(self.features)))
            for i, n in enumerate(self.feature_names):
              for f in filters:
                  try:
                      self.features[:, ind] = data.field(n + 'err_' + f)
                  except:
                      self.features[:, ind] = 1.0
                  ind += 1
            self.Nfeatures *= 2

        if color_band is not None:
            self.features = self.colors(data, filters, self.features, color_band, add_errs)

        # restrict magnitude range if necessary
        if mag_range is not None:
            ind = 0
            limit_band = mag_range.keys()[0]
            for i, n in enumerate(self.feature_names):
                for f in filters:
                    if ('mag' in n) & (f == limit_band):
                        idx = np.where((self.features[:, ind] > mag_range[limit_band][0]) &
                                       (self.features[:, ind] < mag_range[limit_band][1]))[0]
                        self.features = self.features[idx]
                        self.idx = idx
                        done = True
                        break
                    ind += 1
                if done:
                    break
            self.Ndata = self.features.shape[0]

        # trim missing data if desired
        if cut_missing:
            ind = np.ones(self.Ndata, dtype=np.bool)
            for i in range(self.Ndata):
                if np.any(self.features[i, :] == -9999.):
                    ind[i] = False
            self.features = self.features[ind]
            self.idx = self.idx[ind]
            self.Ndata = self.features.shape[0]

        if scale_kind is not None:
            self.scale(scale_kind)

    def colors(self, data, filters, features, color_band, add_errs):
        """
        Compute colors and new errors.
        """
        # find features which are magnitudes, subtract off a band
        ind = 0
        for i, n in enumerate(self.feature_names):
            if 'mag' in n:
                mag = data.field(n + '_' + color_band).copy()
                mag -= data.field('extinction_' + color_band)
            for f in filters:
                if f == color_band:
                    pass
                elif 'mag' in n:
                    features[:, ind] -= mag
                ind += 1

        # add errs in quadrature
        if add_errs:
            for i, n in enumerate(self.feature_names):
                if 'mag' in n:
                    err = data.field(n + 'err_' + color_band)
                for f in filters:
                    if f == color_band:
                        pass
                    elif 'mag' in n:
                        features[:, ind] = np.sqrt(features[:, ind] ** 2. + err ** 2.)
                    ind += 1

        return features

    def scale(self, kind):
        """
        Scale the data using sklearn.
        """
        if kind == 'MinMax':
            self.scaler = MinMaxScaler()
        if kind == 'Whiten':
            self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def run_hmf(self, K, data_ind='all', err_ind='all'):
        """
        Run HMF on the data
        """
        if data_ind == 'all':
            D = self.features.shape[1] / 2
            data_ind = np.arange(D, dtype=np.int)
            err_ind = np.arange(D, 2 * D, dtype=np.int)
        else:
            assert isinstance(data_ind, np.ndarray), 'Need numpy integer array of indicies for data'
            assert isinstance(err_ind, np.ndarray), 'Need numpy integer array of indicies for errs'
        assert K < len(data_ind), 'K is not less than the number of features'
        h = HMF(self.features[:, data_ind], K, 1. / self.features[:, err_ind] ** 2.)
        return h.projections, h.latents.T

    def noisify(self, fraction):
        """
        Create and return gaussian noise.
        """
        D = self.features.shape[1] / 2
        new_errs = (1. + fraction) * self.features[:, -D:]
        diff = np.sqrt(new_errs ** 2. - self.features[:, -D:] ** 2.)
        noise = np.random.randn(self.features.shape[0], D) * diff
        return noise, new_errs

if __name__ == '__main__':

    import pyfits as pf

    n = '../data/mr10k_short_rfadely.fit'
    f = pf.open(n)
    data = f[1].data
    names = f[1].columns.names
    f.close()

    print '\n\n\nThese are the column names:\n%s\n\n' % names

    Ng = np.where(data.field('type') == 3)[0].size
    Ns = np.where(data.field('type') == 6)[0].size
    print '\nSDSS says there are %d galaxies and %d stars.\n\n' % (Ng, Ns)

    names = ['cmodelmag', 'psfmag']
    filters = ['u', 'g', 'r', 'i', 'z']

    f = FeatureExtractor(data, names, filters, color_band=None, scale_kind=None,
                         mag_range=None, cut_missing=False)
