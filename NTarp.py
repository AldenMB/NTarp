import numpy as np
from scipy.stats import norm
import warnings
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.random_projection import GaussianRandomProjection
from sklearn.exceptions import DataDimensionalityWarning


class NTarp(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, n=50, *, random_state=None):
        self.n = n
        self.projector = GaussianRandomProjection(n, random_state=random_state)
        self.direction_ = None
        self.threshold_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DataDimensionalityWarning)
            self.projector.fit(X)
        return self.refit(X)

    def refit(self, X, y=None):
        projection = self.projector.transform(X)
        direction_index = w(projection.T).argmin()
        self.direction_ = self.projector.components_[direction_index]
        self.threshold_ = threshold(X, self.direction_)
        #needed for fit_predict:
        self.labels_ = self.predict(X)
        return self

    def predict(self, X):
        return np.inner(X, self.direction_) > self.threshold_

    def score(self, X):
        return 1 - w(self.predict(X))

    def gaussian_p(self, X):
        alternative_w = w(self.predict(X))
        null_model = distribution_of_w_assuming_gaussian_data(X.shape[-1])
        return null_model.cdf(alternative_w)


def w(x, axis=-1, threshold=None):
    """
    Compute the fraction of variance not explained by a simple thresholding.
    x is a numpy array holding the data on which W should be computed.
    axis only works with its default value currently.
    threshold should only be used when x is flat.
    """
    sorted_x = np.sort(x, axis=axis)
    varlist = explained_variance_list(sorted_x, axis=axis)
    if threshold is None:
        best_variance = varlist.max(axis=axis)
    else:
        below_threshold_count = (sorted_x < threshold).sum()
        if below_threshold_count in (0, len(sorted_x)):
            best_variance = 0
        else:
            best_variance = varlist[below_threshold_count - 1]
    return 1 - best_variance / sorted_x.var(axis=axis)


def explained_variance_list(x, axis=-1):
    """
    Find the amount of variance explained by thresholding at each of the possible
    positions along x.
    NOTE: x needs to be sorted.
    """
    N = x.shape[axis]
    truncated = np.take(x, range(N - 1), axis=axis)
    s1 = np.cumsum(truncated, axis=axis)
    s2 = np.expand_dims(x.sum(axis=axis), axis=axis) - s1
    n1 = np.arange(1, N)
    n2 = N - n1
    # I really would have liked to put it this way, because it is more readable:
    # return (s1*n2-s2*n1)**2 /(n1*n2*N*N)
    # however, this ends up as a significant bottleneck for memory.
    # a better solution is the following in-place computation.
    s1 *= n2
    s2 *= n1
    s1 -= s2
    np.square(s1, out=s1)
    s1 /= 1.0 * n1 * n2 * N * N
    return s1


def threshold(data, direction):
    """
    Find a suitable threshold value which maximizes explained variance of the data projected onto direction.
    NOTE: the chosen hyperplane would be described mathematically as $ x \dot direction = threshold $.
    """
    projected_data = np.inner(data, direction)
    sorted_x = np.sort(projected_data)
    best_sep_index = explained_variance_list(sorted_x).argmax()
    return (sorted_x[best_sep_index] + sorted_x[best_sep_index + 1]) / 2


def distribution_of_w_assuming_gaussian_data(n):
    """
    Create a scipy frozen distribution object which approximates the distribution of w assuming
    that n points are drawn from a gaussian distribution. This approximation is asymptotically
    correct for n large, but should not be used when n is small.
    """
    if n < 5:
        raise ValueError(
            f"this approximation is invalid for n<5, but was called with n={n}"
        )
    if n < 10:
        warnings.warn(
            "calling function distribution_of_w_assuming_gaussian_data with n={}. This provides a poor approximation when n<10.".format(
                n
            )
        )
    k2 = (8 * (np.pi - 3)) / np.pi**2
    s2 = 1 - (2 / np.pi)
    return norm(
        loc=s2 - 1 / n, scale=np.sqrt(k2 / n - 0.4 / n**1.9)
    )  # these constants were determined empirically to work well even for low n.


def validity_test_approximate(validation_data, threshold, direction):
    projected = np.inner(validation_data, direction)
    alt_w = w(projected, threshold=threshold)
    null_model = distribution_of_w_assuming_gaussian_data(projected.shape[-1])
    return {"p": null_model.cdf(alt_w), "w": alt_w}
