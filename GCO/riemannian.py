"""Variants of the exclusion radius and of the admissible projection with pymanopt (options 'riemannian').
Slow (pymanopt Nelder-Mead with restarts) and not recommended: the 'gradient' methods of exclusion.py are faster and more accurate."""
import warnings

import numpy as np

from .utils.extended_var import ExtendedVariable, GrassmannMetric


def _pymanopt():
    try:
        import pymanopt
        from pymanopt.manifolds import Grassmann
        from pymanopt.optimizers import NelderMead
    except ImportError:
        raise ImportError("pymanopt is required for the 'riemannian' methods. Install it with: pip install pymanopt")
    return pymanopt, Grassmann, NelderMead


def exclusion_radius_riemannian(gco, x_best, W, tau=0.05, n_restarts=5):
    """max_S softmin_w d((x_best, S), w), softmin = -tau log sum exp(-d/tau), by pymanopt (Nelder-Mead, restarts)."""
    pymanopt, Grassmann, NelderMead = _pymanopt()
    manifold = Grassmann(gco.n - 1, gco.p - 1)

    @pymanopt.function.numpy(manifold)
    def cost(S):
        omega = ExtendedVariable(x=x_best, S=S)
        dists = np.array([omega.distance(w, r=gco.ratio, metric=GrassmannMetric.CHORDAL) for w in W])
        return tau * np.log(np.sum(np.exp(-dists / tau)))  # = -softmin (pymanopt minimizes)

    optimizer = NelderMead(verbosity=0)
    best = -np.inf
    for _ in range(n_restarts):
        result = optimizer.run(pymanopt.Problem(manifold=manifold, cost=cost), initial_point=None)
        best = max(best, -result.cost)
    return best


def project_into_admissible_riemannian(gco, omega, W, h_k, tau=0.05, n_restarts=5):
    """min_S d(S, S_k)^2 + K max(0, h_k - softmin_w d((x_best, S), w))^2 by pymanopt; Monte Carlo projection if nothing admissible."""
    pymanopt, Grassmann, NelderMead = _pymanopt()
    manifold = Grassmann(gco.n - 1, gco.p - 1)
    x_best = omega.x

    @pymanopt.function.numpy(manifold)
    def cost(S):
        candidate = ExtendedVariable(x=x_best, S=S)
        d_to_Sk = candidate.distance(omega, r=gco.ratio, metric=GrassmannMetric.CHORDAL)
        dists = np.array([candidate.distance(w, r=gco.ratio, metric=GrassmannMetric.CHORDAL) for w in W])
        softmin = -tau * np.log(np.sum(np.exp(-dists / tau)))
        return d_to_Sk ** 2 + gco.K_PENALTY * max(0.0, h_k - softmin) ** 2

    optimizer = NelderMead(verbosity=0)
    best_S, best_cost = None, np.inf
    for _ in range(n_restarts):
        result = optimizer.run(pymanopt.Problem(manifold=manifold, cost=cost), initial_point=None)
        if result.cost < best_cost and gco._is_admissible(ExtendedVariable(x=x_best, S=result.point), W, h_k):
            best_cost, best_S = result.cost, result.point
    if best_S is None:
        warnings.warn("Riemannian projection did not find an admissible solution. Falling back to Monte Carlo projection.", RuntimeWarning)
        return gco._project_into_admissible_mc(omega, W, h_k)
    return ExtendedVariable(x=x_best, S=best_S)
