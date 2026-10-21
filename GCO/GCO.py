"""GCO - Grassmannian Covering Optimizer.

Derivative-free minimisation of a black box f on a box [lb, ub] in R^n by successive searches in p-dimensional subspaces anchored at the
best point. At iteration k:
  1. anchor x_best (unit box [-1, 1]^n) and basis B of the hyperplane orthogonal to it;
  2. a subspace method proposes S_k ((n-1) x (p-1), subspace_methods.py); the subspace is V_k = span{x_best, B S_k};
  3. exclusion (exclusion.py): (x_best, S_k) must be at distance >= gamma h_k of every visited (x_w, S_w), otherwise it is projected
     onto the admissible set;
  4. the suboptimizer (suboptimizers.py) minimises the latent function f_Z on V_k (latent.py) down to the step size sigma_k;
  5. success: new anchor, sigma_k / beta; failure: sigma_k * beta (sigma_k <= C_SIGMA h_k). Stop on the budget or when sigma_k < 1e-9.

Files: options.py (all options and their defaults), subspace_methods.py, exclusion.py, latent.py, suboptimizers.py, riemannian.py.
Example: python -m GCO.GCO  (from the repository root).
"""
import os
import time

import numpy as np

from .options import OPTIONS_SPEC, init_options
from .subspace_methods import SUBSPACE_METHODS, SubspaceContext, XYView
from .exclusion import ExclusionMixin
from .latent import LatentMixin
from .suboptimizers import suboptimizer, mads_suboptimizer, trego_suboptimizer, doe_to_rounded_map  # noqa: F401 (public API)
from .utils.extended_var import ExtendedVariable
from .utils.doe_class import DOE
from .utils.domain import BoxScaler

_XYView = XYView  # former name


class GCO(ExclusionMixin, LatentMixin):
    OPTIONS_SPEC = OPTIONS_SPEC

    def __init__(self, fun, vars_prop, opt):
        """
        fun       : objective, f(x) for x a 1-d array of the original space.
        vars_prop : {"dim": n, "lower_bound": lb, "upper_bound": ub}.
        opt       : options (see options.py); "subspace_dimension" is mandatory.
        """
        self.opt = self._init_options(opt)
        if self.opt["subspace_selection"] == "mixed":  # former name: Local_PLS alternated with random
            self.opt.update(subspace_selection="Local_PLS", local_pls_factor=self.opt["mixed_local_factor"],
                            mixed_period=self.opt["mixed_random_period"], mixed_method="random_orth")
        for key in ("subspace_selection", "mixed_method"):
            if self.opt[key] not in SUBSPACE_METHODS:
                raise ValueError(f"Option '{key}' must be one of {sorted(SUBSPACE_METHODS)}, got '{self.opt[key]}'.")

        # log file
        self.logfile_path = self.opt["file_path"]
        self.output_columns = self.opt["output_columns"]
        self._eval_times = []  # time stamp of each black-box call
        self._t0 = time.perf_counter()
        self._init_logfile()

        # problem, in the unit box [-1, 1]^n
        self.n = int(vars_prop["dim"])
        self.scaler = BoxScaler(np.asarray(vars_prop["lower_bound"], dtype=float).reshape(-1),
                                np.asarray(vars_prop["upper_bound"], dtype=float).reshape(-1))
        self.fun = self._scale_function(fun)
        self.p = int(self.opt["subspace_dimension"])
        if self.opt["cache_projection_threshold"] is not None:
            self.opt["cache_projection_threshold"] *= np.sqrt(self.n)

        self.gamma = self.opt["gamma"]
        self.beta = self.opt["contraction"]
        self._update_discontinuity_point()

    @classmethod
    def _init_options(cls, user_opt):
        return init_options(user_opt, cls.OPTIONS_SPEC)

    @staticmethod
    def register_subspace_method(name, method):
        """Add a subspace method: method(gco, ctx) -> S ((n-1) x (p-1)) or (S, n_fixed), see subspace_methods.SubspaceContext."""
        SUBSPACE_METHODS[name] = method

    # ======================================================================================================== main loop
    def run_optim(self, X0, F0, suboptimizer, budget):
        """Minimise from the evaluated points (X0, F0) (original space; they count in the budget). Returns (x_best, f_best),
        x_best in the unit box."""
        X0 = np.clip(self.scaler.to_unit(X0), -1.0, 1.0)
        x_best, f_best = X0[np.argmin(F0)], float(np.min(F0))
        self.doe = DOE(self.n)
        self.doe.add(X0, F0)
        self.update_anchor = True
        self._init_scales()

        self._t0 = time.perf_counter()  # origin of the TIME column
        self._eval_times = []
        f_ref = np.inf
        for i, (x_i, f_i) in enumerate(zip(X0, F0)):  # log the initial DOE
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(bbe=i + 1, f_val=f_i, x=x_i, elapsed=0.0)

        tot_eval, sigma_k, W = len(F0), self.sigma_0, []
        cache_name = self.opt["file_path"].replace(".txt", "")
        while tot_eval < budget:
            # 1) admissible anchored subspace; the subspace method may evaluate points (counted here)
            self._budget_left = budget - tot_eval
            self._sub_X, self._sub_F = [], []
            omega_k, h_k = self.anchored_subspace(W, x_best)
            if self._sub_F:
                X_s, F_s = np.array(self._sub_X), np.array(self._sub_F)
                self.save_data(X_s, F_s, f_best, tot_eval)
                tot_eval += len(F_s)
            else:
                X_s, F_s = np.empty((0, self.n)), np.empty(0)
            if tot_eval >= budget:
                if len(F_s) and F_s.min() < f_best:
                    x_best, f_best = X_s[np.argmin(F_s)], float(F_s.min())
                break
            W.append(omega_k)

            # 2) suboptimizer in the latent space
            Zk, Fk, n_eval = suboptimizer(fun=self.latent_fun, sigma_k=sigma_k, initial_doe=self.latent_doe,
                                          lb=self.latent_lb, ub=self.latent_ub, cache_name=cache_name)
            Zk, Fk = np.array(Zk, dtype=float).reshape(-1, self.p), np.array(Fk, dtype=float)
            Xk = np.clip(Zk @ self.curent_subspace.T, -1.0, 1.0)
            self.save_data(Xk, Fk, min(f_best, F_s.min()) if len(F_s) else f_best, tot_eval)
            tot_eval += n_eval

            # 3) best point and step size (the points of the subspace method count like those of the suboptimizer)
            X_all, F_all = (np.vstack([X_s, Xk]), np.concatenate([F_s, Fk])) if len(F_s) else (Xk, Fk)
            if len(F_all) and np.min(F_all) < f_best:
                x_best, f_best = X_all[np.argmin(F_all)], np.min(F_all)
                self.update_anchor = True
                sigma_k = min(sigma_k / self.beta, self.C_SIGMA * h_k)
            else:
                self.update_anchor = False
                sigma_k = min(sigma_k * self.beta, self.C_SIGMA * h_k)
            self._update_doe(Xk, Fk)  # (the points of the subspace method are already in the DOE)
            if sigma_k < self.opt["min_termination_scale_parameter"]:
                print(f"Termination : sigma_k = {sigma_k} < {self.opt['min_termination_scale_parameter']}")
                break
        return x_best, f_best

    def _init_scales(self):
        """sigma_0 = 30 % of the expected distance between two points of [-1,1]^p, h_0 = twice the expected chordal distance in
        Gr(p-1, n-1), C_SIGMA = sigma_0 / h_0 (cap sigma_k <= C_SIGMA h_k), r = weight of the anchor part in the distance."""
        n, p = self.n, self.p
        self.sigma_0 = np.sqrt(2 * p / 3) * 0.3
        self.h_0 = np.sqrt((p - 1) ** 2 - (p - 1) / (n - 1)) * 2
        self.C_SIGMA = self.sigma_0 / self.h_0
        self.ratio = 2 * n * (n - 1) / (2 * n * (n - 1) + 3 * (p - 1) * (n - 2))

    # ======================================================================================================== anchored subspace
    def anchored_subspace(self, W, x_best):
        """Admissible subspace V_k at x_best, then latent function, latent DOE and latent bounds for it. Returns (omega_k, h_k)."""
        h_k = self.compute_exclusion_radius(x_best, W)
        if self.update_anchor:
            self._get_anchor_basis(x_best)
        X, y = self.doe.Xy()
        doe_perp = XYView(X @ self.anchor_basis, y)  # evaluated points in the coordinates of the hyperplane orthogonal to the anchor
        omega_k = self.get_extended_variable(doe_perp, x_best, W, h_k)
        V_k = self.augmentation_isomorphism(omega_k)  # Gr(p-1, n-1) -> Gr(p, n)
        self.curent_subspace = V_k
        self._update_latent_function(V_k)
        self._update_latent_doe(V_k)
        self._update_latent_bound(V_k)
        return omega_k, h_k

    def get_extended_variable(self, doe_perp, x_best, W, h_k):
        """The subspace method proposes S_k (every mixed_period-th call: mixed_method), then admissibility is enforced."""
        ctx = SubspaceContext(self, doe_perp, x_best, W, h_k)
        self._mix_count = getattr(self, "_mix_count", 0) + 1
        k = self.opt["mixed_period"]
        name = self.opt["mixed_method"] if k > 0 and self._mix_count % k == 0 else self.opt["subspace_selection"]
        self.last_selection = name
        out = SUBSPACE_METHODS[name](self, ctx)
        S_k, n_fixed = out if isinstance(out, tuple) else (out, 0)
        omega = ExtendedVariable(x_best, S_k)
        if not self._is_admissible(omega, W, h_k):
            if n_fixed > 0:  # keep the pinned columns (e.g. the gradient) and redraw the others
                om = self._admissible_with_fixed(np.linalg.qr(S_k)[0], n_fixed, x_best, W, h_k)
                if om is not None:
                    return om
            omega = self._project_into_admissible(omega, W, h_k)
        return omega

    def augmentation_isomorphism(self, omega):
        """Orthonormal basis (n x p) of V = span{x_best, B S}."""
        x = self.anchore
        V_raw = np.column_stack([x / np.linalg.norm(x), self.anchor_basis @ omega.S])
        return np.linalg.qr(V_raw)[0]

    def _get_anchor_basis(self, x):
        """Basis B (n x (n-1)) of the hyperplane orthogonal to the anchor direction u = x / ||x||: the fixed basis C of the hyperplane
        orthogonal to the random 'discontinuity point' a, mapped by the Householder reflection that sends a to u."""
        a, C = self.discontinuity_point, self.discontinuity_basis
        if np.linalg.norm(x) < 1e-8:
            u = np.random.randn(self.n)
            u = u / np.linalg.norm(u)
        else:
            u = x / np.linalg.norm(x)
        self.anchore = u
        self.anchor_basis = (np.eye(self.n) - 2 * np.outer(a - u, a - u) / np.linalg.norm(a - u) ** 2) @ C

    def _update_discontinuity_point(self):
        """Random unit vector a and orthonormal basis of the hyperplane orthogonal to it."""
        self.discontinuity_point = np.random.uniform(-1, 1, self.n)
        self.discontinuity_point /= np.linalg.norm(self.discontinuity_point)
        self.discontinuity_basis = self._orthogonal_complement_basis(self.discontinuity_point)
        return self

    @staticmethod
    def _orthogonal_complement_basis(v):
        """Orthonormal basis (n x (n-1)) of the hyperplane orthogonal to v."""
        v = v.flatten()
        v = v / np.linalg.norm(v)
        Q = np.linalg.qr(np.column_stack([v.reshape(-1, 1), np.eye(len(v))]), mode='complete')[0]
        return Q[:, 1:]  # the first column of Q is +-v

    # ======================================================================================================== evaluations and log
    def _scale_function(self, fun):
        """f on the unit box (time-stamped calls)."""
        def scaled_fun(z):
            val = fun(self.scaler.from_unit(z))
            self._eval_times.append(time.perf_counter())
            return val
        return scaled_fun

    def _update_doe(self, Xk, Fk):
        self.doe.add(Xk, y=Fk)
        return self

    def _init_logfile(self):
        if os.path.exists(self.logfile_path):
            os.remove(self.logfile_path)
        with open(self.logfile_path, "w") as f:
            f.write("  ".join(self.output_columns) + "  \n")

    def save_data(self, Xk, Fk, f_ref, tot_eval):
        """Log every improvement of f_ref among the evaluations (Xk, Fk), numbered from tot_eval + 1 (one entry per black-box call)."""
        times = self._eval_times[-len(Fk):] if len(Fk) else []
        for i, (x_i, f_i) in enumerate(zip(Xk, Fk)):
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(bbe=tot_eval + i + 1, f_val=f_i, x=x_i,
                                    elapsed=times[i] - self._t0 if i < len(times) else float("nan"))

    def _write_logfile(self, bbe, f_val, x, elapsed=float("nan")):
        fields = {"BBE": f"{bbe}", "OBJ": f"{f_val:.17e}", "TIME": f"{elapsed:.6f}"}
        with open(self.logfile_path, "a") as f:
            f.write("  ".join(fields[c] for c in self.output_columns) + " \n")


if __name__ == "__main__":
    # Example: shifted Rosenbrock in dimension 30 on [-10, 10]^30, budget 100 (n + 1) evaluations, default options
    # (grad_PLS alternated with random one call out of 4, MADS in the subspaces). Run from the repository root:  python -m GCO.GCO
    n, p = 30, 4
    lb, ub = np.full(n, -10.0), np.full(n, 10.0)
    shift = np.random.default_rng(0).uniform(-5.0, 5.0, n)  # the minimum (f = 0) is at x = 1 + shift, not at the centre of the box

    def rosenbrock(x):
        y = np.asarray(x) - shift
        return float(np.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2))

    np.random.seed(0)  # GCO uses numpy's global random generator
    doe = DOE(n, lb=lb, ub=ub, seed=0)
    doe.init_from_method(N_points=2 * n, method="LHS", fun=rosenbrock)  # initial design (counted in the budget)
    X0, F0 = doe.Xy()

    gco = GCO(rosenbrock, vars_prop={"dim": n, "lower_bound": lb, "upper_bound": ub},
              opt={"subspace_dimension": p, "file_path": "GCO_example_log.txt", "output_columns": ["BBE", "OBJ", "TIME"]})
    x_best, f_best = gco.run_optim(X0=X0, F0=F0, suboptimizer=mads_suboptimizer, budget=100 * (n + 1))

    print(f"best initial value : {np.min(F0):.6g}")
    print(f"best value found   : {f_best:.6g}   (minimum 0 at x = 1 + shift)")
    print(f"best point         : {np.round(gco.scaler.from_unit(x_best)[:5], 4)} ...")
    print(f"history of the improvements (BBE, OBJ, TIME): {gco.logfile_path}")
