"""Suboptimizers: they minimise the latent function in the p-dimensional subspace.

Contract: suboptimizer(fun, sigma_k, initial_doe, budget=None, lb=None, ub=None, cache_name=None, ...) -> (Zk, Fk, n_eval)
  fun          latent function (R^p -> R), each call is one black-box evaluation
  sigma_k      minimal step (frame) size: the call stops when its step falls below sigma_k
  initial_doe  already evaluated points of the subspace (DOE with Xy()), latent coordinates
  lb, ub       latent bounds;  budget: maximal number of evaluations (None = no cap)
  returns the evaluated latent points, their values and their number, in evaluation order.
"""
import itertools
import os

import numpy as np
import PyNomad

from .utils.doe_class import DOE
from .utils.nomad_io import write_nomad_cache, NomadCacheFormat, PointKey


def suboptimizer(fun, sigma_k):
    """Placeholder showing the role of a suboptimizer (see the module docstring for the actual signature)."""
    raise NotImplementedError("suboptimizer is a placeholder function. Implement it with your favorite optimization method.")


def doe_to_rounded_map(doe: DOE, decimals: int) -> dict[PointKey, float]:
    """{rounded point: best y} from a DOE (if several points round to the same key, keep the minimum)."""
    X, y = doe.Xy()
    out: dict[PointKey, float] = {}
    for si, yi in zip(X, y):
        key = tuple(np.round(si, decimals=decimals).tolist())
        val = float(yi)
        if key not in out or val < out[key]:
            out[key] = val
    return out


def mads_suboptimizer(fun, sigma_k, initial_doe, budget=None, ub=None, lb=None, stat_file=None, cache_name=None):
    """MADS (PyNomad) from the best point of initial_doe, MIN_FRAME_SIZE = sigma_k; initial_doe is given to NOMAD as its cache
    (those points are not re-evaluated)."""
    Zk, Fk = [], []

    def bb(eval_point):
        z = np.array([eval_point.get_coord(i) for i in range(eval_point.size())])
        fval = fun(z)
        eval_point.setBBO(f"{fval:.17g}".encode("UTF-8"))
        Zk.append(z)
        Fk.append(fval)
        return 1

    cache_path = "cache_file.txt" if cache_name is None else f"cache_{cache_name}.txt"
    if os.path.exists(cache_path):
        os.remove(cache_path)
    write_nomad_cache(doe_s=doe_to_rounded_map(doe=initial_doe, decimals=12), cache_path=cache_path,
                      fmt=NomadCacheFormat(bb_output_type="OBJ", cache_hits=0), sort_items=True)

    params = [
        "BB_OUTPUT_TYPE OBJ",
        "DISPLAY_DEGREE 0",
        "DISPLAY_ALL_EVAL false",
        "DISPLAY_STATS BBE OBJ MESH_SIZE POLL_SIZE",
        "USE_CACHE_FILE_FOR_RERUN true",
        f"CACHE_FILE {cache_path}",
    ]
    if budget is not None:
        params.append(f"MAX_BB_EVAL {budget}")
    if sigma_k is not None:
        params.append("MIN_FRAME_SIZE ( " + " ".join(f"{sigma_k:.17g}" for _ in range(len(ub))) + " )")
    if stat_file is not None:
        params.append(f"STATS_FILE {stat_file} BBE OBJ MESH_SIZE POLL_SIZE")

    X, y = initial_doe.Xy()
    x0 = X[np.argmin(y)].tolist()
    PyNomad.optimize(bb, x0, lb, ub, params)
    return Zk, Fk, len(Fk)


_trego_calls = itertools.count()  # successive calls get different (but reproducible) seeds


def trego_suboptimizer(fun, sigma_k, initial_doe, budget=None, ub=None, lb=None, stat_file=None, cache_name=None,
                       seed=None, G=1, L=4, beta=0.5, tsigma_0=0.05, refit_every=3, max_train=100):
    """TREGO (utils/trego.py) with a minimal trust-region size derived from sigma_k. initial_doe warm-starts the GP (not counted,
    not returned). No evaluation cap if budget is None and sigma_k is given; budget 5p if both are None. stat_file and cache_name
    are ignored. seed None: a call counter, so that successive calls differ but runs are reproducible."""
    try:  # imported here: utils/trego.py is optional (the rest of GCO does not need it)
        from .utils.trego import trego
    except ImportError as e:
        raise ImportError("trego_suboptimizer needs GCO/utils/trego.py, which is not available in this installation.") from e
    if seed is None:
        seed = next(_trego_calls)
    return trego(fun, initial_doe, lb, ub, tsigma_min=sigma_k, budget=budget, seed=seed,
                 G=G, L=L, beta=beta, tsigma_0=tsigma_0, refit_every=refit_every, max_train=max_train)
