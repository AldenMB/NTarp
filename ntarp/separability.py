# revision June 22, 2020, Alden Bradford
import numpy as np
from scipy.stats import norm
import warnings


def w(x, threshold=None):
    """
    Compute the fraction of variance not explained by a simple thresholding.
    x is a numpy array holding the data on which W should be computed.
    axis only works with its default value currently.
    threshold should only be used when x is flat.
    """
    sorted_x = np.sort(x, axis=-1)
    varlist = explained_variance_list(sorted_x)
    if threshold is None:
        best_variance = varlist.max(axis=-1)
    else:
        below_threshold_count = (sorted_x < threshold).sum()
        if below_threshold_count in (0, len(sorted_x)):
            best_variance = 0
        else:
            best_variance = varlist[below_threshold_count - 1]
    return 1 - best_variance / sorted_x.var(axis=-1)


def explained_variance_list(x):
    """
    Find the amount of variance explained by thresholding at each of the possible
    positions along x.
    NOTE: x needs to be sorted.
    """
    N = x.shape[-1]
    s1 = np.cumsum(x[..., :-1], axis=-1)
    s2 = np.cumsum(x[..., :0:-1], axis=-1)[..., ::-1]
    n1 = np.arange(1, N)
    n2 = np.arange(N-1, 0, -1)
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


def separate(data, direction, threshold_given=None):
    """
    Give a boolean mask separating data by a computed threshold along direction. To facilitate comparison,
    the first entry of the mask will always be True.
    
    If the optional parameter threshold_given is provided, it is used instead of computing the best threshold.
    """
    projected_data = np.inner(data, direction)
    if threshold_given is None:
        threshold_given = threshold(data, direction)
    answer = projected_data <= threshold_given
    if answer[0]:
        return answer
    else:
        return ~answer


def threshold(data, direction):
    """
    Find a suitable threshold value which maximizes explained variance of the data projected onto direction.
    NOTE: the chosen hyperplane would be described mathematically as $ x \dot direction = threshold $.
    """
    projected_data = np.inner(data, direction)
    sorted_x = np.sort(projected_data)
    best_sep_index = explained_variance_list(sorted_x).argmax()
    return (sorted_x[best_sep_index] + sorted_x[best_sep_index + 1]) / 2


def n_tarp(data, n, times=1):
    """
    Perform n-TARP clustering on data, where the last axis of data is taken to be
    the coordinates of each data point. Because all the relevant information about the generated clusters can
    be concluded from just the direction of projection and the data, only the computed direction vectors are returned.
    
    The optional parameter times is given as a vectorized way of performing n-tarp repeatedly.
    This is good for testing purposes, but if a better projection is all that is desired then it would be better to
    simply increase n.
    """
    directions = np.random.randn(times, n, data.shape[-1])
    projected = np.inner(directions, data)
    ws = w(projected)
    best_dir_index = np.argmin(ws, axis=-1)[..., np.newaxis, np.newaxis]
    return np.take_along_axis(directions, best_dir_index, axis=1).squeeze()


_validity_test_memo = {}


def validity_test(validation_data, threshold, direction, null_model_size=100_000):
    memo_key = (null_model_size, validation_data.shape[0])
    if memo_key not in _validity_test_memo:
        null_data = np.random.randn(null_model_size, validation_data.shape[0])
        _validity_test_memo[memo_key] = w(null_data)
    null_w = _validity_test_memo[memo_key]
    projected = np.inner(validation_data, direction)
    alt_w = w(projected, threshold=threshold)
    p = (null_w < alt_w).mean() + 1 / null_model_size
    return {"p": p, "w": alt_w}


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
    k2 = (8 * (np.pi - 3)) / np.pi ** 2
    s2 = 1 - (2 / np.pi)
    return norm(
        loc=s2 - 1 / n, scale=np.sqrt(k2 / n - 0.4 / n ** 1.9)
    )  # these constants were determined empirically to work well even for low n.


def validity_test_approximate(validation_data, threshold, direction):
    projected = np.inner(validation_data, direction)
    alt_w = w(projected, threshold=threshold)
    null_model = distribution_of_w_assuming_gaussian_data(projected.shape[-1])
    return {"p": null_model.cdf(alt_w), "w": alt_w}


def chunk_masks(data, train_size, validation_size, test_size):
    n = data.shape[0]
    shuffle = np.random.permutation(n)
    return (
        shuffle[:train_size],
        shuffle[train_size : train_size + validation_size],
        shuffle[
            train_size + validation_size : train_size + validation_size + test_size
        ],
    )


def n_tarp_validation(data, masks, n=10, p_cutoff=0.05):
    train, val, test = (data[mask] for mask in masks)
    direction = n_tarp(train, n)
    thresh = threshold(train, direction)
    p = validity_test_approximate(val, thresh, direction)["p"]
    return {
        "success": p < p_cutoff,
        "direction": direction,
        "threshold": thresh,
        "p_val": p,
    }
