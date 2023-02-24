import unittest
from ntarp import separability as sep
import numpy as np


class TestWFormula(unittest.TestCase):
    def test_explained_variance_against_unexplained(self):
        np.random.seed(1234)
        for X in [
            np.arange(10, dtype=float),
            np.random.randn(100),
            np.zeros(10),
            np.array([1], dtype=float),
            np.array([], dtype=float),
        ]:
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
            self.assertTrue(np.allclose(total_variance, X.var()))

    def test_out_of_bounds_threshold(self):
        np.random.seed(1234)
        X = np.random.random((20,))
        self.assertEqual(sep.w(X, threshold=2.5), 1)
        self.assertEqual(sep.w(X, threshold=-0.5), 1)


if __name__ == "__main__":
    unittest.main()
