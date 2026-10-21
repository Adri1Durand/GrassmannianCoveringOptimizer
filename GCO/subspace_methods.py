"""Subspace methods: they propose the subspace S_k (an (n-1) x (p-1) orthonormal frame of the hyperplane orthogonal to the anchor).

A method is a function  method(gco, ctx) -> S  or  (S, n_fixed)  (n_fixed: number of leading columns GCO should keep if it has to
restore admissibility). It may evaluate the black box through ctx.evaluate: those points are counted in the budget, added to the DOE
and logged like the points of the suboptimizer. Add a method with GCO.register_subspace_method(name, method).
"""
import warnings

import numpy as np


class XYView:
    """Minimal read-only (X, y) holder with the Xy() accessor of DOE."""
    def __init__(self, X, y):
        self._X, self._y = X, y

    def Xy(self):
        return self._X, self._y


class SubspaceContext:
    """What a subspace method receives besides the GCO instance.

    doe_perp   : evaluated points in the coordinates of the hyperplane orthogonal to the anchor (doe_perp.Xy() -> (X_perp, y)).
    x_best     : anchor (unit box [-1, 1]^n).
    W, h_k     : visited extended variables and exclusion radius (GCO applies the admissibility test afterwards).
    budget_left: evaluations still available.
    evaluate(X): evaluates the black box at the rows of X (unit box) and returns their values (fewer rows if the budget runs out).
    """
    def __init__(self, gco, doe_perp, x_best, W, h_k):
        self.gco, self.doe_perp, self.x_best, self.W, self.h_k = gco, doe_perp, x_best, W, h_k

    @property
    def budget_left(self):
        return self.gco._budget_left - len(self.gco._sub_F)

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X = X[: max(self.budget_left, 0)]
        F = np.array([float(self.gco.fun(x)) for x in X])
        if len(F):
            self.gco._sub_X.extend(list(X))
            self.gco._sub_F.extend(list(F))
            self.gco._update_doe(X, F)
        return F


# ---------------------------------------------------------------------------------------------------------------- helpers
def pls_frame(gco, X_perp, y):
    """Orthonormal (n-1) x (p-1) frame from the PLS directions (x_rotations_) of (X_perp, y), completed with random vectors if rank-deficient."""
    from sklearn.cross_decomposition import PLSRegression
    import scipy.linalg
    n_components = min(gco.p - 1, X_perp.shape[0], X_perp.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = PLSRegression(n_components).fit(X_perp, y).x_rotations_
    S = scipy.linalg.orth(S)
    for _ in range(max((gco.p - 1) - S.shape[1], 0)):
        v = np.random.randn(S.shape[0])
        v -= S @ (S.T @ v)
        v /= np.linalg.norm(v)
        S = np.hstack([S, v.reshape(-1, 1)])
    return scipy.linalg.orth(S)


def local_data(gco, doe_perp, x_best, factor):
    """All the points while the DOE holds at most K = ceil(factor*(n-1)) points (at least p), then only the K points closest to
    x_best (Euclidean distance in the full space). factor <= 0 keeps every point."""
    X_perp, y = doe_perp.Xy()
    y = np.asarray(y)
    K = max(int(np.ceil(factor * (gco.n - 1))), gco.p)
    if factor <= 0 or len(y) <= K:
        return X_perp, y
    d2 = np.sum((gco.doe.Xy()[0] - x_best) ** 2, axis=1)
    idx = np.argpartition(d2, K - 1)[:K]
    return X_perp[idx], y[idx]


def fd_gradient(gco, ctx):
    """Forward-difference gradient at the anchor: n evaluations through ctx.evaluate (None if the budget is too short or g is unusable).
    f(x_best) is already known (it is in the DOE)."""
    n, h = gco.n, gco.opt["grad_fd_step"]
    if ctx.budget_left < n:
        return None
    x = ctx.x_best
    X, y = gco.doe.Xy()
    f0 = float(y[int(np.argmin(np.sum((X - x) ** 2, axis=1)))])
    steps = np.where(x + h <= 1.0, h, -h)
    F = ctx.evaluate(x[None, :] + np.diag(steps))
    if len(F) < n:
        return None
    g = (F - f0) / steps
    if not np.all(np.isfinite(g)) or not np.any(g):
        return None
    return g


# ---------------------------------------------------------------------------------------------------------------- methods
def random_method(gco, ctx):
    """Random subspace (orthonormalised by ExtendedVariable)."""
    return np.random.randn(gco.n - 1, gco.p - 1)


def random_orth_method(gco, ctx):
    """Random subspace, returned orthonormalised (same distribution; used by the former subspace_selection='mixed')."""
    return np.linalg.qr(np.random.randn(gco.n - 1, gco.p - 1))[0]


def pls_method(gco, ctx):
    """PLS on every evaluated point (directions of strongest covariance with f)."""
    from sklearn.cross_decomposition import PLSRegression
    import scipy.linalg
    X_perp, y = ctx.doe_perp.Xy()
    n_components = min(gco.p - 1, X_perp.shape[0], X_perp.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = PLSRegression(n_components).fit(X_perp, y).x_rotations_
    S = scipy.linalg.orth(S)
    n_cols = S.shape[1]
    if n_cols < gco.p - 1:  # complete a rank-deficient basis with random directions
        rand = np.random.randn(S.shape[0], (gco.p - 1) - n_cols)
        for i in range(rand.shape[1]):
            v = rand[:, i]
            v -= S @ (S.T @ v)
            v /= np.linalg.norm(v)
            S = np.hstack([S, v.reshape(-1, 1)])
    return scipy.linalg.orth(S)


def local_pls_method(gco, ctx):
    """PLS on the points closest to the anchor (see local_data and the option local_pls_factor)."""
    X_perp, y = local_data(gco, ctx.doe_perp, ctx.x_best, gco.opt["local_pls_factor"])
    return pls_frame(gco, X_perp, y)


def make_gradient_method(base):
    """First column = steepest-descent direction from a forward-difference gradient (every grad_every-th call), other columns = base."""
    def method(gco, ctx):
        gco._grad_count = getattr(gco, "_grad_count", 0) + 1
        S_base = base(gco, ctx)
        if (gco._grad_count - 1) % max(gco.opt["grad_every"], 1) != 0:
            return S_base
        g = fd_gradient(gco, ctx)
        if g is None:
            return S_base
        c = gco.anchor_basis.T @ (-g)  # descent direction in the hyperplane orthogonal to the anchor
        nc = np.linalg.norm(c)
        if nc == 0.0:
            return S_base
        c = c / nc
        Q = np.linalg.qr(S_base - np.outer(c, c @ S_base))[0][:, : gco.p - 2]
        return np.linalg.qr(np.column_stack([c, Q]))[0], 1
    return method


SUBSPACE_METHODS = {
    "random": random_method,
    "PLS": pls_method,
    "Local_PLS": local_pls_method,
    "random_orth": random_orth_method,
    "grad_PLS": make_gradient_method(pls_method),
    "grad_random": make_gradient_method(random_orth_method),
}
