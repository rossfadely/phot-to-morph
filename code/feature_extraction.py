import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class FeatureExtractor(object):
    """
    Feature extraction from a SDSS pyfits data structure.
    """
    def __init__(self, data, feature_names, filters, add_errs=True, color_band='r'):
        """
        :data: pyfits table data structure
        :feature_name: list of strings giving features to be extracted
        """
        self.Ndata = len(data.field(0))
        self.Nfeatures = len(feature_names) * len(filters) 

        self.scaler = None
        self.filters = filters
        self.feature_names = feature_names

        # build initial array of features
        ind = 0
        self.features = np.zeros((self.Ndata, self.Nfeatures))
        for i, n in enumerate(self.feature_names):
            for f in filters:
                self.features[:, ind] = data.field(n + '_' + f)
                if 'mag' in n:
                    self.features[:, ind] -= data.field('extinction_' + f)
                ind += 1

        if add_errs:
            self.features = np.hstack((self.features, np.zeros_like(self.features)))
            for i, n in enumerate(self.feature_names):
              for f in filters:
                    self.features[:, ind] = data.field(n + 'err_' + f)
                    ind += 1
            self.Nfeatures *= 2

        self.features = self.colors(color_band, add_errs)

    def colors(color_band, add_errs):
        if color_band is None:
            return self.features

        # find features which are magnitudes, subtract off a band
        ind = 0
        for i, n in enumerate(self.feature_names):
            if 'mag' in n:
                mag = data.field(n + '_' + color_band)
                mag -= data.field('extinction_' + color_band)
            for f in filters:
                if f == color_band:
                    pass
                elif 'mag' in n:
                    self.features[:, ind] -= mag
                ind += 1

        # add errs in quadrature
        if add_errs:
            for i, n in enumerate(self.feature_names):
                for f in filters:
                    if f == color_band:
                        pass
                    elif 'mag' in n:
                        self.features[:, ind] = np.sqrt(self.features[:, ind] ** 2. +
                                                        data.field(n + 'err_' + f) ** 2.)
                    ind += 1
        return self.features

    def scale(self, kind):
        """
        Scale the data using sklearn.
        """
        if kind == 'MinMax':
            self.scaler = MinMaxScaler()
        if kind == 'Whiten':
            self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform()

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

    names = ['cmodelmag']
    filters = ['u', 'g', 'r', 'i', 'z']

    f = FeatureExtractor(data, names, filters)

