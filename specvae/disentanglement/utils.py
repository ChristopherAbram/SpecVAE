import numpy as np
import sklearn


def obtain_representation(observations, representation_function, batch_size):
    """
    Obtain representations from observations.

    Args:
        observations: Observations for which we compute the representation.
        representation_function: Function that takes observation as input and
            outputs a representation.
        batch_size: Batch size to compute the representation.

    Returns:
        representations: Codes (num_codes, num_points)-Numpy array.
    """
    representations = None
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        if i == 0:
            representations = representation_function(current_observations)
        else:
            representations = np.vstack(
                (representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations)


def generate_batch_factor_code(
    ground_truth_data, representation_function,
    num_points, random_state, batch_size):
    """
    Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and
            outputs a representation.
        num_points: Number of points to sample.
        random_state: Numpy random state used for randomness.
        batch_size: Batchsize to sample points.

    Returns:
        representations: Codes (num_codes, num_points)-np array.
        factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
            ground_truth_data.sample(num_points_iter, random_state)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations)
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack(
                (representations, representation_function(current_observations)))
        i += num_points_iter
    return np.transpose(representations), np.transpose(factors)


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


def _histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def make_discretizer(target, num_bins=20, discretizer_fn=_histogram_discretize):
    """Wrapper that creates discretizers."""
    return discretizer_fn(target, num_bins)
