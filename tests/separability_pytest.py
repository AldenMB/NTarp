from ntarp import separability as sep
import numpy as np
from hypothesis import given, assume, note
from hypothesis.extra.numpy import arrays, floating_dtypes
from hypothesis import strategies as st


@given(
    arrays(
        shape=st.integers(min_value=2, max_value=100),
        elements=st.floats(min_value=-1e10, max_value=1e10),
        dtype=floating_dtypes(sizes=(64,)),
    )
)
def test_explained_variance(X):
    X.sort()
    n = len(X)
    unexplained_variance = np.fromiter(
        (X[:i].var() * i + X[i:].var() * (n - i) for i in range(1, n)), float, n - 1,
    )
    unexplained_variance /= n
    explained_variance = sep.explained_variance_list(X)
    total_variance = unexplained_variance + explained_variance
    note(f"explained variance: {explained_variance}")
    note(f"unexplained variance: {unexplained_variance}")
    note(f"total variance: {total_variance}")
    assert (explained_variance >= 0).all()
    assert np.allclose(X.var(), total_variance, rtol=1e-5, atol=1e-5)
