import numpy as np
from sklearn.utils import extmath


class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [
            i for i in range(self.num_factors) \
                if i not in self.latent_factor_indices]

    @property
    def num_latent_factors(self):
        return len(self.latent_factor_indices)

    def sample_latent_factors(self, num, random_state):
        """Sample a batch of the latent factors."""
        factors = np.zeros(
            shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num, random_state)
        return factors

    def sample_all_factors(self, latent_factors, random_state):
        """Samples the remaining factors based on the latent factors."""
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(
            shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        # Complete all the other factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples, random_state)
        return all_factors

    def _sample_factor(self, i, num, random_state):
        return random_state.randint(self.factor_sizes[i], size=num)


class StateSpaceAtomIndex(object):
    """Index mapping from features to positions of state space atoms."""

    def __init__(self, factor_sizes, features):
        """
        Creates the StateSpaceAtomIndex.

        Args:
            factor_sizes: List of integers with the number of distinct values for each
            of the factors.
            features: Numpy matrix where each row contains a different factor
            configuration. The matrix needs to cover the whole state space.
        """
        self.factor_sizes = factor_sizes
        num_total_atoms = np.prod(self.factor_sizes)
        self.factor_bases = num_total_atoms / np.cumprod(self.factor_sizes)
        feature_state_space_index = self._features_to_state_space_index(features)
        if np.unique(feature_state_space_index).size != num_total_atoms:
            raise ValueError("Features matrix does not cover the whole state space.")
        lookup_table = np.zeros(num_total_atoms, dtype=np.int64)
        lookup_table[feature_state_space_index] = np.arange(num_total_atoms)
        self.state_space_to_save_space_index = lookup_table

    def features_to_index(self, features):
        """
        Returns the indices in the input space for given factor configurations.

        Args:
            features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the input space should be
            returned.
        """
        state_space_index = self._features_to_state_space_index(features)
        return self.state_space_to_save_space_index[state_space_index]

    def _features_to_state_space_index(self, features):
        """
        Returns the indices in the atom space for given factor configurations.

        Args:
            features: Numpy matrix where each row contains a different factor
            configuration for which the indices in the atom space should be
            returned.
        """
        if (np.any(features > np.expand_dims(self.factor_sizes, 0)) or np.any(features < 0)):
            raise ValueError("Feature indices have to be within [0, factor_size-1]!")
        return np.array(np.dot(features, self.factor_bases), dtype=np.int64)


class GroundTruthData(object):
    """Abstract class for data sets that are two-step generative models."""

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        raise NotImplementedError()

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]


from ..utils import get_attribute
from .. import dataset as dt
import torchvision as tv
import torch


class Spectra(GroundTruthData):
    def __init__(self, config, latent_factor_indices=[0, 1, 2]):
        self.config = config
        self._set_config_params(self.config)
        self._set_transform()
        self.init_latent_factors(latent_factor_indices)
        self.loaded = False

    def init_latent_factors(self, latent_factor_indices):
        raise NotImplementedError('Dataset specific implementation')

    def load(self):
        raise NotImplementedError('Dataset specific implementation')

    def _set_config_params(self, config):
        self.input_columns = ['spectrum']
        self.types = [torch.float32] * len(self.input_columns)
        self.n_samples = get_attribute(self.config, 'n_samples')
        self.random_select = get_attribute(self.config, 
            'random_select', default=False, required=False)
        self.n_peaks = get_attribute(self.config, 'max_num_peaks')
        self.max_mz = get_attribute(self.config, 'max_mz')
        self.min_intensity = get_attribute(self.config, 'min_intensity')
        self.normalize_intensity = get_attribute(self.config, 
            'normalize_intensity', default=True, required=False)
        self.normalize_mass = get_attribute(self.config, 
            'normalize_mass', default=True, required=False)
        self.rescale_intensity = get_attribute(self.config, 
            'rescale_intensity', default=False, required=False)

    def set_transform(self, transform):
        self.transform = transform

    def _set_transform(self):
        self.transform = tv.transforms.Compose([
            dt.SplitSpectrum(),
            dt.TopNPeaks(n=self.n_peaks),
            dt.FilterPeaks(max_mz=self.max_mz, min_intensity=self.min_intensity),
            dt.Normalize(
                intensity=self.normalize_intensity, 
                mass=self.normalize_mass, 
                rescale_intensity=self.rescale_intensity),
            dt.ToMZIntConcatAlt(max_num_peaks=self.n_peaks)
        ])

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = self.index.features_to_index(all_factors)
        return self.spectrum[indices].astype(np.float32)

    def _load_data(self, dataset_name):
        self.loader, self.metadata = dt.load_data(
            dataset_name, self.transform, self.n_samples, int(1e7), 
            self.random_select, torch.device('cpu'), self.input_columns, 
            self.types, split=False, filename=('%s_dis.csv' % dataset_name))
        self.spectrum, self.ids = next(iter(self.loader))
        self.spectrum = self.spectrum.cpu().detach().numpy()


class MoNA(Spectra):
    """
    MoNA data set.

    The ground-truth factors of variation are:
    0 - ionization mode (2 different values)
    1 - collision energy (20 different values)
    2 - instrument type (6 different values)
    """
    def __init__(self, config, latent_factor_indices=[0, 1, 2]):
        super().__init__(config, latent_factor_indices)

    def init_latent_factors(self, latent_factor_indices):
        self.factor_sizes = [2, 20, 6]
        features = extmath.cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = latent_factor_indices
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(
            self.factor_sizes, self.latent_factor_indices)

    def load(self):
        self._load_data('MoNA')
        self.data_shape = self.spectrum[0].shape
        self.loaded = True


class HMDB(Spectra):
    """
    HMDB data set.

    The ground-truth factors of variation are:
    0 - ionization mode (2 different values)
    1 - collision energy (3 different values)
    """
    def __init__(self, config, latent_factor_indices=[0, 1]):
        super().__init__(config, latent_factor_indices)

    def init_latent_factors(self, latent_factor_indices):
        self.factor_sizes = [2, 3]
        features = extmath.cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = latent_factor_indices
        self.num_total_factors = features.shape[1]
        self.index = StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = SplitDiscreteStateSpace(
            self.factor_sizes, self.latent_factor_indices)

    def load(self):
        self._load_data('HMDB')
        self.data_shape = self.spectrum[0].shape
        self.loaded = True
