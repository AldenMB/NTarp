from ntarp import separability as sep
import numpy as np
from hypothesis import given, assume, note, settings
from hypothesis.extra.numpy import arrays, floating_dtypes, array_shapes
from hypothesis import strategies as st
from ckmeans_1d_dp import ckmeans


one_by_hundred = arrays(
    shape=st.integers(min_value=3, max_value=100),
    elements=st.floats(min_value=-1e10, max_value=1e10),
    dtype=floating_dtypes(sizes=(64,)),
    unique=True,
)


@given(one_by_hundred)
def test_explained_variance(X):
    X.sort()
    n = len(X)
    unexplained_variance = np.fromiter(
        (X[:i].var() * i + X[i:].var() * (n - i) for i in range(1, n)),
        float,
        n - 1,
    )
    unexplained_variance /= n
    explained_variance = sep.explained_variance_list(X)
    total_variance = unexplained_variance + explained_variance
    note(f"explained variance: {explained_variance}")
    note(f"unexplained variance: {unexplained_variance}")
    note(f"total variance: {total_variance}")
    assert (explained_variance >= 0).all()
    assert np.allclose(X.var(), total_variance, rtol=1e-5, atol=1e-5)


@given(one_by_hundred)
def test_explained_variance_positive(X):
    explained_variance = sep.explained_variance_list(X)
    assert (explained_variance >= 0).all()


@settings(max_examples = 2000)
@given(
    st.one_of(
        one_by_hundred,
        arrays(
            shape=array_shapes(min_dims=2, min_side=3),
            elements=st.floats(min_value=-1e10, max_value=1e10),
            dtype=floating_dtypes(sizes=(64,)),
            unique=True,
        ),
    )
)
def test_agree_with_ckmeans(X):
    w = sep.w(X)
    ck = ckmeans(X, k=2)
    ckw = ck.tot_withinss / ck.totss

    note(f"withinss from separability: {w}")
    note(f"withinss from ckmeans: {ckw}")
    
    note(f"reproduce x as np.frombuffer({X.tobytes()}).reshape({X.shape})") 
    assert np.allclose(w, ckw, equal_nan=True, rtol=1e-5, atol=1e-5)
