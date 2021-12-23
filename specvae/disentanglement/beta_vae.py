import numpy as np
import logging
from sklearn import linear_model


def compute_beta_vae(
    ground_truth_data, representation_function,
    random_state=np.random.RandomState(123), artifact_dir=None,
    batch_size=128, num_train=100, num_eval=100):
    """
    Computes the BetaVAE disentanglement metric using scikit-learn.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observations as input and
            outputs a dim_representation sized representation for each observation.
        random_state: Numpy random state used for randomness.
        artifact_dir: Optional path to directory where artifacts can be saved.
        batch_size: Number of points to be used to compute the training_sample.
        num_train: Number of points used for training.
        num_eval: Number of points used for evaluation.

    Returns:
        Dictionary with scores:
            train_accuracy: Accuracy on training set.
            eval_accuracy: Accuracy on evaluation set.
    """
    logging.info("Generating training set.")
    train_points, train_labels = _generate_training_batch(
        ground_truth_data, representation_function, batch_size, num_train,
        random_state)

    logging.info("Training sklearn model.")
    model = linear_model.LogisticRegression(random_state=random_state)
    model.fit(train_points, train_labels)

    logging.info("Evaluate training set accuracy.")
    train_accuracy = model.score(train_points, train_labels)
    train_accuracy = np.mean(model.predict(train_points) == train_labels)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    logging.info("Generating evaluation set.")
    eval_points, eval_labels = _generate_training_batch(
        ground_truth_data, representation_function, batch_size, num_eval,
        random_state)

    logging.info("Evaluate evaluation set accuracy.")
    eval_accuracy = model.score(eval_points, eval_labels)
    logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
    scores_dict = {}
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    return scores_dict


def _generate_training_batch(
    ground_truth_data, representation_function,
    batch_size, num_points, random_state):
    """
    Sample a set of training samples based on a batch of ground-truth data.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observations as input and
            outputs a dim_representation sized representation for each observation.
        batch_size: Number of points to be used to compute the training_sample.
        num_points: Number of points to be sampled for training set.
        random_state: Numpy random state used for randomness.

    Returns:
        points: (num_points, dim_representation)-sized numpy array with training set
            features.
        labels: (num_points)-sized numpy array with training set labels.
    """
    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64)
    for i in range(num_points):
        labels[i], feature_vector = _generate_training_sample(
            ground_truth_data, representation_function, batch_size, random_state)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def _generate_training_sample(
    ground_truth_data, representation_function, batch_size, random_state):
    """
    Sample a single training sample based on a mini-batch of ground-truth data.

    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observation as input and
            outputs a representation.
        batch_size: Number of points to be used to compute the training_sample
        random_state: Numpy random state used for randomness.

    Returns:
        index: Index of coordinate to be used.
        feature_vector: Feature vector of training sample.
    """
    # Select random coordinate to keep fixed.
    index = random_state.randint(ground_truth_data.num_factors)
    # Sample two mini batches of latent variables.
    factors1 = ground_truth_data.sample_factors(batch_size, random_state)
    factors2 = ground_truth_data.sample_factors(batch_size, random_state)
    # Ensure sampled coordinate is the same across pairs of samples.
    factors2[:, index] = factors1[:, index]
    # Transform latent variables to observation space.
    observation1 = ground_truth_data.sample_observations_from_factors(
        factors1, random_state)
    observation2 = ground_truth_data.sample_observations_from_factors(
        factors2, random_state)
    # Compute representations based on the observations.
    representation1 = representation_function(observation1)
    representation2 = representation_function(observation2)
    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1 - representation2), axis=0)
    return index, feature_vector
