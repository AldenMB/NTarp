from functools import partial
import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.random_projection import GaussianRandomProjection
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.model_selection import train_test_split
from .separability import w, threshold, distribution_of_w_assuming_gaussian_data


class NTarp(ClusterMixin, BaseEstimator):
    """
    This is an implementation of the nTARP clustering method.
    This method selects a hyperplane $X\cdot direction = threshold$
    to split the data set.

    Parameters
    ----------
    n : int, default=50
        the parameter n from the nTARP method, which indicates
        how many trial projections to perform. 50 is a good value
        for many cases, though some may require a higher value.
        This is essentially a fitting parameter; increasing n will
        lead to higher-quality clusterings and higher runtimes, but
        will also increase the chance of overfitting.

    reserve_fraction : float, default=0.5
        How many of the input values should be saved for validating the resulting model?
        We take the first 1-reserve_fraction of the data to train on, and use the remaining
        input values to test the result.

    p_threshold : float, default=0.05
        How special should the W score on the reserved data be? A lower p_threshold
        will make NTarp more conservative, only allowing cluster assignments which are
        highly unlikely assuming a gaussian null hypothesis.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    direction_ : ndarray of shape (X.shape[-1],)
        the direction used in defining the separating hyperplane.

    threshold_ : float
        the threshold used in definint the separating hyperplane.

    p_ : float
        the p-value computed in the validation step

    Notes
    -----
    50 is a good value of n for many cases, though some may require a
    higher value. This is essentially a fitting parameter; increasing
    n will lead to higher-quality clusterings and higher runtimes,
    but will also increase the chance of overfitting.

    References
    ----------
    [1] https://arxiv.org/pdf/1806.05297.pdf the original
    description of the method
    [2] https://arxiv.org/abs/2008.09579 a more theoretical
    development of its structure.
    """

    def __init__(
        self, n=50, *, reserve_fraction=0.5, p_threshold=0.05, random_state=None
    ):
        self.n = n
        self.p_threshold = 0.05
        self.split = partial(
            train_test_split, test_size=reserve_fraction, random_state=random_state
        )
        self.projector = GaussianRandomProjection(n, random_state=random_state)
        self.direction_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Compute nTARP clustering.


        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        """
        # We are not using the projector to reduce to dimension n,
        # but rather to one dimension n times. These are numerically
        # equivalent, but it means the projector will give an
        # inappropriate warning if n_features is less than n.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DataDimensionalityWarning)
            self.projector.fit(X)
        return self.refit(X)

    def refit(self, X, y=None):
        """Re-choose a direction and threshold based on
        the same directions, but using new data. This can be used
        to validate the choice of direction and threshold.
        """
        train, test = self.split(X)

        # direction and threshold from the train data
        projection = self.projector.transform(train)
        direction_index = w(projection.T).argmin()
        self.direction_ = self.projector.components_[direction_index]
        self.threshold_ = threshold(train, self.direction_)

        # a simple way to guarantee we do not split any data in this case
        self.p_ = self.gaussian_p(test)
        if self.p_ > self.p_threshold:
            self.threshold_ = np.inf

        # needed for fit_predict:
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        """Predict the cluster assignment of each sample in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            boolean indicating each sample's cluster assignment.
        """
        return np.inner(X, self.direction_) > self.threshold_

    def score(self, X):
        """
        By convention .score should be bigger-is-better, while
        nTARP works by minimizing the parameter w. Hence its complement
        1-w is a good choice for score, since it is always positive.
        This is analogous to the R^2 value used in regression, in that
        it also represents a fraction of explained variance.
        """
        return 1 - w(self.predict(X))

    def gaussian_p(self, X):
        """
        Under the assumption that X was drawn from a Gaussian
        distribution, this gives the p-value of the observed w score
        based on a limiting argument given in [2].
        """
        alternative_w = w(self.predict(X))
        null_model = distribution_of_w_assuming_gaussian_data(X.shape[-1])
        return null_model.cdf(alternative_w)
