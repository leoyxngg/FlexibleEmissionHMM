import numpy as np

def as_sequence(X, n_features=None):
    """
    Validate the shape of input is (T, D), for 1D data, it converts (T,) np array
    to (T, 1)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("Each sequence must be a nonempty finite array of shape (T, D)")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {X.shape[1]}")
    return X
