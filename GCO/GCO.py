import itertools
import os
import time

import numpy as np
from .utils.extended_var import *
from .utils.doe_class import DOE
from .utils.domain import BoxScaler
from .utils.trego import trego
#from SSO_algo.utils.subspace_class import Subspace
from .utils.nomad_io import write_nomad_cache, NomadCacheFormat,PointKey

import PyNomad
import warnings


class _XYView :
    """Minimal read-only (X, y) holder with the Xy() accessor of DOE."""
    def __init__(self, X, y) :
        self._X, self._y = X, y

    def Xy(self) :
        return self._X, self._y


class GCO :
    OPTIONS_SPEC = {
        "file_path": {
            "type": str,
            "default": "log_GCO.txt",
            "desc": "Path to the logfile where optimization successes are recorded.",
        },
        "output_columns": {
            "type": list,
            "default": ["BBE", "OBJ"],
            "choices": {"BBE", "OBJ", "TIME"},
            "desc": (
                "Columns written in the logfile, in order. Allowed entries: "
                "'BBE' (cumulative number of black-box evaluations), "
                "'OBJ' (objective value), "
                "'TIME' (wall-clock seconds elapsed since the start of the "
                "optimization when the black-box evaluation giving OBJ finished). "
                "Example: ['BBE', 'OBJ', 'TIME']."
            ),
        },
        "subspace_dimension": {
            "type": int,
            "default": None,  # Mandatory: no default
            "desc": (
                "Dimension p of the anchored subspace. "
                "Must satisfy 1 < p < n (problem dimension)."
            ),
        },
        "min_termination_scale_parameter": {
            "type": float,
            "default": 1e-9,
            "desc": (
                "Minimum scale parameter for the suboptimizer. "
                "Optimization stops when the suboptimizer's scale parameter falls below this value."
            ),
        },
        "max_cache_point": {
            "type": int,
            "default": 900,  # marge de sécurité sous la limite de 1000 lignes de NOMAD
            "desc": (
                "Maximum number of points written to the NOMAD cache file. "
                "If the DOE history exceeds this number, points are selected "
                "via 'cache_selection_method'."
            ),
        },

        "cache_selection_method": {
            "type": str,
            "default": "closest_subspace",
            "choices": {"random", "closest_subspace"},
            "desc": (
                "Method used to select which points are kept in the cache when "
                "the DOE history exceeds max_cache_point. "
                "'random': uniform random subsample. "
                "'closest_subspace': keep points with smallest projection residual "
                "||A A^T x - x|| onto the current subspace."
            ),
        },
        "cache_projection_threshold": {
            "type": float,
            "default": 0.0025,
            "desc": (
                "Projection residual coefficient for points kept in the latent DOE. "
                "The effective threshold is this value times sqrt(n), so a value "
                "of 0.0025 means 0.0025*sqrt(n). Only points satisfying "
                "||V_k V_k^T x - x|| <= the effective threshold are retained. "
                "None disables this filter."
            ),
        },

        "subspace_selection": {
            "type": str,
            "default": "PLS",
            "choices": {"random", "PLS", "Local_PLS", "mixed"},
            "desc": "Method used to select the subspace matrix S_k on Gr(p-1, n-1).",
        },
        "local_pls_factor": {
            "type": float,
            "default": 2.0,
            "desc": ("subspace_selection='Local_PLS': while the DOE holds at most K = ceil(factor*(n-1)) points the PLS uses all of them; beyond, "
                     "only the K points closest to x_best. 0 = always all points (global PLS)."),
        },
        "mixed_random_period": {
            "type": int,
            "default": 2,
            "desc": "subspace_selection='mixed': every period-th iteration uses a random subspace, the others use PLS (0 = never random).",
        },
        "mixed_local_factor": {
            "type": float,
            "default": 0.0,
            "desc": ("subspace_selection='mixed': once the DOE holds more than K = ceil(factor*(n-1)) points, the PLS iterations use "
                     "only the K points closest to x_best (local PLS); before that, all points (global PLS). 0 = always global."),
        },
        "admissible_projection_method": {
            "type": str,
            "default": "monte_carlo",
            "choices": {"monte_carlo", "riemannian"},
            "desc": (
                "Method used to project an inadmissible extended variable "
                "onto the admissible set. "
                "'monte_carlo' samples uniformly on the Grassmannian via Haar measure. "
                "'riemannian' uses pymanopt to solve the constrained problem."
            ),
        },
        "exclusion_radius_method": {
            "type": str,
            "default": "gradient",
            "choices": {"monte_carlo", "riemannian", "gradient"},
            "desc": (
                "Method used to estimate the exclusion radius h_k of the visited "
                "extended variable list W. "
                "'monte_carlo' samples uniformly on the Grassmannian via Haar measure. "
                "'riemannian' uses pymanopt to maximize the min-distance objective. "
                "'gradient' keeps the best of a few random candidates and improves it by projected gradient ascent on the Grassmannian."
            ),
        },
        "W_prune_by_anchor": {
            "type": bool,
            "default": True,
            "desc": "Discard from the history W, in the distance checks, the extended variables whose anchor is too far from the current one to be closer than the threshold (exact, see _restrict_W).",
        },
        "exclusion_radius_gradient_init": {
            "type": int,
            "default": 20,
            "desc": "'gradient' exclusion radius: number of random candidates among which the ascent starts (best one).",
        },
        "exclusion_radius_gradient_max_iter": {
            "type": int,
            "default": 15,
            "desc": "'gradient' exclusion radius: maximum number of ascent steps (also stops when the gain of a step is below 0.1 %).",
        },
        "admissible_projection_n_samples": {
            "type": int,
            "default": 5000,
            "desc": "Number of Haar-distributed candidates drawn by the Monte Carlo admissible projection.",
        },
        "exclusion_radius_n_samples": {
            "type": int,
            "default": 500,
            "desc": "Number of Haar-distributed samples used by the Monte Carlo exclusion radius estimate.",
        },
        "contraction" : {
            "type" : float,
            "default": 0.5
        },
        "gamma" : {
            "type" : float,
            "default" : 0.9
        },
        "ratio_percent" : {
            "type" : float,
            "default" : 0.5
        },
    }

    @classmethod
    def _init_options(cls, user_opt: dict) -> dict:
        """
        Validate and merge user options with defaults from OPTIONS_SPEC.

        Parameters
        ----------
        user_opt : dict
            User-provided options. Unknown keys raise a ValueError.

        Returns
        -------
        dict
            Fully resolved options dict with all keys from OPTIONS_SPEC.

        Raises
        ------
        ValueError
            If an unknown key is passed, a mandatory option is missing,
            a value has the wrong type, or a value is not in the allowed choices.
        """
        # deprecated names (covering radius -> exclusion radius)
        user_opt = {k.replace("covering_radius", "exclusion_radius"): v for k, v in user_opt.items()}
        # --- Check for unknown keys ---
        unknown = set(user_opt) - set(cls.OPTIONS_SPEC)
        if unknown:
            raise ValueError(
                f"Unknown option(s): {sorted(unknown)}. "
                f"Valid options are: {sorted(cls.OPTIONS_SPEC)}."
            )

        opt = {}
        for key, spec in cls.OPTIONS_SPEC.items():
            if key in user_opt:
                value = user_opt[key]
            else:
                value = spec["default"]

            # --- Check mandatory options ---
            if value is None and spec["default"] is None:
                raise ValueError(
                    f"Option '{key}' is mandatory and has no default. "
                    f"Description: {spec['desc']}"
                )

            # --- Type checking ---
            if value is not None:
                expected_type = spec["type"]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Option '{key}' expects type {expected_type.__name__}, "
                        f"got {type(value).__name__}."
                    )

            # --- Choices checking ---
            if "choices" in spec and isinstance(value, list):
                bad = [v for v in value if v not in spec["choices"]]
                if bad or not value:
                    raise ValueError(
                        f"Option '{key}' must be a non-empty list with entries in "
                        f"{sorted(spec['choices'])}, got {value}."
                    )
                value = list(value)
            elif "choices" in spec and value not in spec["choices"]:
                raise ValueError(
                    f"Option '{key}' must be one of {sorted(spec['choices'])}, "
                    f"got '{value}'."
                )

            opt[key] = value

        return opt


    def __init__(self, fun, vars_prop, opt) :
        """
        Initialize the optimization case.

        Parameters
        ----------
        fun: callable
            Default: objective function obj = fun(x). x 1d array
            The grouped_eval flag in optim_settings can be used to pass a
            problem function [obj, con_1, con_2, ...] = fun(x). It is used when not only
            the objective has multiple fidelities but also the constraints. Attention,
            that implies that constraints can't be evaluated separetely from objective anymore.
            For MOO : fun : ndarray[1,n_var] -> ndarray[1,n_obj], bool(Fail)
        var_prop: dict with variables properies 

        """
        
        self.opt = self._init_options(opt)

        # Saving properties :
        self.logfile_path = self.opt["file_path"]
        self.output_columns = self.opt["output_columns"]
        self._eval_times = []  # timestamp of each black-box call
        self._t0 = time.perf_counter()
        self.n = int(vars_prop["dim"])

        self._init_logfile()

        # Problem properties :
        user_LB = np.asarray(vars_prop["lower_bound"], dtype=float).reshape(-1)
        user_UB = np.asarray(vars_prop["upper_bound"], dtype=float).reshape(-1)
        self.scaler = BoxScaler(user_LB, user_UB)
        self.fun = self._scale_function(fun)
        

        self.p = int(self.opt["subspace_dimension"])

        if self.opt["cache_projection_threshold"] is not None:
            self.opt["cache_projection_threshold"] *= np.sqrt(self.n)

        # Constants of CGO :
        
        self.K_PENALTY = 100000.0 # Penalty constant for the latent function outside the zonotope
        self.gamma = self.opt.get("gamma", 0.95)

        # self.C_SIGMA = 0.1 # Link between exclusion radius h_k and the min step size sigma_k of the suboptimizer : sigma_k = C_SIGMA * h_k
        self.beta = self.opt["contraction"] # Contraction parameter
        
        # Draw a random discontinuity point in the unit shere.
        self._update_discontinuity_point()


    def run_optim(self, X0, F0, suboptimizer, budget) :
        """Run the optimization."""
        X0 = self.scaler.to_unit(X0) # scale the initial point to the unit box [-1, 1]^n
        X0 = np.clip(X0, -1.0, 1.0)
        # Get the best point and value among the initial points
        x_best = X0[np.argmin(F0)]
        f_best = np.min(F0)
        self.doe = DOE(self.n)
        self.doe.add(X0,F0)

        self.update_anchor = True # flag to update the subspace at each iteration

        # Initilize scales parameters
        Expected_dist_X = np.sqrt(2*self.p/3)
        Expected_dist_Gr = np.sqrt((self.p-1)**2-(self.p-1)/(self.n-1))
        self.sigma_0 = Expected_dist_X * 0.3 # 30% of Expected disance in [-1,1]^p
        self.h_0 = Expected_dist_Gr * 2 # Expected chordal distance in Gr(p-1,R^n-1)
        self.C_SIGMA = self.sigma_0/self.h_0
        #self.ratio = (Expected_dist_Gr**2/Expected_dist_X**2) * self.opt.get("ratio_percent", 0.3333)
        n,p = self.n,self.p
        self.ratio = 2*n*(n-1) / ( 2*n*(n-1) + 3*(p-1)*(n-2) )

        self._t0 = time.perf_counter()  # reference for the TIME column
        self._eval_times = []
        tot_eval, k, h_k, sigma_k = 0, 0, self.h_0, self.sigma_0
        W = [] # list of extended variables
        h = []


        f_best = float(np.min(F0))

        # Log initial DOE
        f_ref = np.inf
        for i, (x_i, f_i) in enumerate(zip(X0, F0)):
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(bbe=i + 1, f_val=f_i, x=x_i, elapsed=0.0)
        
        tot_eval += len(F0)

        while tot_eval < budget :
            k += 1 # subspace number 
            #print(f"Subspace n°{k}")

            # Get the realisable anchored subspace V_k
            omega_k, h_k = self.anchored_subspace(W, x_best)

            # Update the list of extended variables, and global exclusion radius
            W.append(omega_k) 
            h.append(h_k)
            #sigma_k = self.C_SIGMA * h_k

            # Run suboptimizer in associated latent space
            #print(f"Start sub-optimizer with min_framesize {sigma_k}")
            cache_name = self.opt["file_path"]
            cache_name = cache_name.replace(".txt", "")
            Zk, Fk, n_eval = suboptimizer(fun = self.latent_fun,
                                          sigma_k = sigma_k,
                                          initial_doe = self.latent_doe,
                                          lb = self.latent_lb,
                                          ub = self.latent_ub,
                                          cache_name = cache_name) # Optimize in the Latent space 
            Zk = np.array(Zk)   # shape (n_points, k)
            # print(f"Shape Zk : {Zk.shape}")
            Fk = np.array(Fk)   # shape (n_points,)
            # print(f"Zk {Zk}")
            #Xk = Zk @ self.curent_subspace.T  # shape (n_points, n)
            Xk = np.clip(Zk @ self.curent_subspace.T, -1.0, 1.0)


            # Log AVANT mise à jour de f_best
            self.save_data(Xk, Fk, f_best, tot_eval)
            
            
            f_k = np.min(Fk)
            tot_eval += n_eval
            # Update the best point and value if f_k < f_best
            if f_k < f_best:
                x_best = Xk[np.argmin(Fk)]
                f_best = f_k
                self.update_anchor = True # update the subspace at the next iteration
                # Success Inrease sigma_k
                sigma_k = min ( sigma_k /self.beta ,self.C_SIGMA * h_k ) 
            else :
                self.update_anchor = False # keep the same subspace at the next iteration
                # Failure Decrease sigma_k
                sigma_k = min( sigma_k*self.beta , self.C_SIGMA*h_k)
            self._update_doe(Xk, Fk) # Update the DOE with the new evaluated points and values
            if sigma_k < self.opt["min_termination_scale_parameter"] :
                print(f"Termination : sigma_k = {sigma_k} < {self.opt['min_termination_scale_parameter']}")
                break

        return x_best, f_best

    # --- Optimization tools --- #
    def anchored_subspace(self, W, x_best) :
        """Compute an admissible subspace V_k.
        Udpate the latent function with the new subspace V_k.
        """
        h_k = self.compute_exclusion_radius(x_best, W)
        #print(f"Current exclusion radius {h_k}")
        if self.update_anchor :
            self._get_anchor_basis(x_best) #basis of x_best perpendicular space
        B = self.anchor_basis
        X, y = self.doe.Xy() 
        # the main DOE is already duplicate-free, so the perpendicular DOE only needs to expose Xy() for the subspace selection
        # (rebuilding a deduplicated DOE of ~1000 points at every iteration was a large part of the bookkeeping cost)
        doe_perp = _XYView(X @ B, y)
        omega_k = self.get_extended_variable(doe_perp, x_best, W, h_k)
        V_k = self.augmentation_isomorphism(omega_k) # From Gr(p-1,n-1) to Gr(p,n)

        self.curent_subspace = V_k
        self._update_latent_function(V_k)   # met à jour self.A et self.latent_fun
        self._update_latent_doe(V_k)        # met à jour self.latent_doe (avec sélection cache)
        self._update_latent_bound(V_k)      # met à jour self.latent_lb / self.latent_ub

        return omega_k, h_k

    # --- Anchored subspace tools --- #
    def get_extended_variable(self, doe_perp, x_best, W, h_k) :
        """Get the subspace matrix S_k."""
        if self.opt["subspace_selection"] == "random" :
            # Draw a random subspace matrix S_k of n-1 rows and p columns
            S_k = np.random.randn(self.n-1, self.p-1)
        elif self.opt["subspace_selection"] == "PLS":
            from sklearn.cross_decomposition import PLSRegression as pls
            import scipy
            import warnings

            X_perp, y = doe_perp.Xy()
            n_components = min(self.p - 1, X_perp.shape[0], X_perp.shape[1])
            _pls = pls(n_components)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                S_k = _pls.fit(X_perp, y).x_rotations_
            S_k = scipy.linalg.orth(S_k)

            # orth may reduce rank if PLS columns are degenerate — complete the basis
            n_cols = S_k.shape[1]
            if n_cols < self.p - 1:
                deficit = (self.p - 1) - n_cols
                rand = np.random.randn(S_k.shape[0], deficit)
                for i in range(deficit):
                    v = rand[:, i]
                    v -= S_k @ (S_k.T @ v)
                    v /= np.linalg.norm(v)
                    S_k = np.hstack([S_k, v.reshape(-1, 1)])
            S_k = scipy.linalg.orth(S_k)  # ensure orthonormality

        elif self.opt["subspace_selection"] == "Local_PLS" :
            X_perp, y = self._local_data(doe_perp, x_best, self.opt["local_pls_factor"])
            S_k = self._pls_frame(X_perp, y)

        elif self.opt["subspace_selection"] == "mixed":
            self._sel_count = getattr(self, "_sel_count", 0) + 1
            period = self.opt["mixed_random_period"]
            if period > 0 and self._sel_count % period == 0:
                S_k = np.linalg.qr(np.random.randn(self.n - 1, self.p - 1))[0]
                self.last_selection = "random"
            else:
                X_perp, y = self._local_data(doe_perp, x_best, self.opt["mixed_local_factor"])
                self.last_selection = "local_PLS" if len(y) < len(doe_perp.Xy()[1]) else "global_PLS"
                S_k = self._pls_frame(X_perp, y)
        else :
            raise NotImplementedError("Subspace selection method not implemented.")
        omega = ExtendedVariable(x_best,S_k)
        if not self._is_admissible(omega, W, h_k) :
            omega = self._project_into_admissible(omega, W, h_k)
        return omega

    def _local_data(self, doe_perp, x_best, factor):
        """
        Data of a local PLS: all the points while the DOE holds at most K = ceil(factor*(n-1)) points (at least p), then only the K
        points closest to x_best (Euclidean distance in the full space). factor <= 0 keeps every point (global PLS).
        """
        X_perp, y = doe_perp.Xy()
        y = np.asarray(y)
        K = max(int(np.ceil(factor * (self.n - 1))), self.p)
        if factor <= 0 or len(y) <= K:
            return X_perp, y
        d2 = np.sum((self.doe.Xy()[0] - x_best) ** 2, axis=1)
        idx = np.argpartition(d2, K - 1)[:K]
        return X_perp[idx], y[idx]

    def _pls_frame(self, X_perp, y):
        """Orthonormal (n-1)x(p-1) frame from the PLS x_rotations of (X_perp, y); rank deficiency completed with random vectors."""
        from sklearn.cross_decomposition import PLSRegression as pls
        import scipy
        import warnings
        n_components = min(self.p - 1, X_perp.shape[0], X_perp.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S_k = pls(n_components).fit(X_perp, y).x_rotations_
        S_k = scipy.linalg.orth(S_k)
        deficit = (self.p - 1) - S_k.shape[1]
        for _ in range(max(deficit, 0)):
            v = np.random.randn(S_k.shape[0])
            v -= S_k @ (S_k.T @ v)
            v /= np.linalg.norm(v)
            S_k = np.hstack([S_k, v.reshape(-1, 1)])
        return scipy.linalg.orth(S_k)

    def _W_arrays(self, W):
        """Stack the visited extended variables: anchors (k, n) and concatenated subspaces (n-1, k*(p-1))."""
        Xw = np.array([w.x for w in W])
        Sw = np.concatenate([w.S for w in W], axis=1)
        return Xw, Sw

    def _restrict_W(self, x, W_arrays, bound):
        """
        Keep only the visited extended variables that can be closer than `bound` to (x, S), whatever S.
        Since d_Omega^2 = r ||x - x_w||^2 + (1-r) d_G^2 >= r ||x - x_w||^2, a w with r ||x - x_w||^2 >= bound^2 is never
        closer than `bound`: dropping it changes neither the admissibility test (bound = gamma h_k) nor a min distance that
        is known to be <= bound. Exact (no approximation); it only shortens the history the distances are computed on.
        """
        Xw, Sw = W_arrays
        if not self.opt.get("W_prune_by_anchor", True) or Xw.shape[0] == 0:
            return W_arrays
        keep = self.ratio * np.sum((Xw - x[None, :]) ** 2, axis=1) < bound * bound
        if keep.mean() > 0.6:  # too little to gain: gathering the columns would cost more than it saves
            return W_arrays
        q = self.p - 1
        cols = (np.flatnonzero(keep)[:, None] * q + np.arange(q)[None, :]).ravel()
        return Xw[keep], Sw[:, cols]

    def _radius_bound(self, x, W_arrays):
        """Upper bound of min_w d_Omega((x, S), w) valid for every S: the w nearest to x in x, with d_G^2 <= p-1."""
        Xw, _ = W_arrays
        r = self.ratio
        dx2 = np.min(np.sum((Xw - x[None, :]) ** 2, axis=1))
        return np.sqrt(r * dx2 + (1.0 - r) * (self.p - 1)) * (1.0 + 1e-9) + 1e-12

    def _min_distance_to_W(self, x, S_batch, W_arrays, chunk_bytes=150e6):
        """
        min_{w in W} d_Omega((x, S), w) for every S of S_batch (shape (B, n-1, p-1), orthonormal columns).
        Same quantity as ExtendedVariable.distance (chordal) but computed with one GEMM per chunk:
            d_G^2(S, S_w) = (p-1) - ||S^T S_w||_F^2,   d_Omega^2 = r ||x - x_w||^2 + (1-r) d_G^2.
        """
        Xw, Sw = W_arrays
        B, _, q = S_batch.shape
        k = Xw.shape[0]
        if k == 0:  # no extended variable can be closer than the bound used to discard them
            return np.full(B, np.inf)
        r = self.ratio
        rdx2 = r * np.sum((Xw - x[None, :]) ** 2, axis=1)  # (k,)
        out = np.empty(B)
        step = max(1, int(chunk_bytes // (8 * q * q * k)))
        for i in range(0, B, step):
            Sb = S_batch[i:i + step]
            b = Sb.shape[0]
            G = Sb.transpose(1, 0, 2).reshape(Sb.shape[1], b * q).T @ Sw  # (b*q, k*q)
            G4 = G.reshape(b, q, k, q)
            fro2 = np.einsum('bikj,bikj->bk', G4, G4)
            d2 = rdx2[None, :] + (1.0 - r) * np.maximum(q - fro2, 0.0)
            out[i:i + b] = np.sqrt(d2.min(axis=1))
        return out

    def _sample_grassmann(self, n_samples):
        """n_samples orthonormal (n-1, p-1) frames, Haar distributed (Gaussian matrices + QR)."""
        # a Generator seeded from the global stream (reproducible with np.random.seed) is ~2x faster than legacy randn (measured: 2.1x on 5000 candidates)
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Z = rng.standard_normal((n_samples, self.n - 1, self.p - 1))
        Q, _ = np.linalg.qr(Z)
        return Q

    def _is_admissible(self, omega , W, h_k) :
        """Check if the subspace V_k is belong to restricted anchored space."""
        if len(W) == 0 :
            return True
        W_arrays = self._restrict_W(omega.x, self._W_arrays(W), self.gamma * h_k)
        d = self._min_distance_to_W(omega.x, omega.S[None, :, :], W_arrays)[0]
        return not (d < self.gamma * h_k)

    def _orthobasis(self, X):
        """Compute an orthonormal basis of the subspace spanned by the columns of X."""
        # Sécuriser : forcer X en 2D
        X = np.atleast_2d(X)
        if X.shape[0] < X.shape[1]:  # si shape (1, n) au lieu de (n, 1)
            X = X.T

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        rank = np.sum(S > 1e-10)
        return U[:, :rank]

    def _orthogonal_complement_basis(self, v):
        """Compute an orthonormal basis of the hyperplane orthogonal to v."""
        v = v.flatten()
        n = len(v)
        v = v / np.linalg.norm(v)
        # QR decomposition on the full matrix [v | I_n]
        M = np.column_stack([v.reshape(-1, 1), np.eye(n)])
        Q, _ = np.linalg.qr(M, mode='complete')  # Q is (n, n)
        # First column of Q aligns with v, the rest span the orthogonal complement
        return Q[:, 1:]  # shape (n, n-1)


    def _get_anchor_basis(self,x):
        a = self.discontinuity_point
        C = self.discontinuity_basis
        if np.linalg.norm(x) < 1e-8 :
            u = np.random.randn(self.n)
            u = u / np.linalg.norm(u)
        else :
            u = x/np.linalg.norm(x)
        self.anchore = u 
        # self.anchor_basis = (np.eye(self.n) - 2 * ((a-u) @ (a-u).T) / np.linalg.norm(a-u)**2) @ C
        self.anchor_basis = (np.eye(self.n) - 2 * np.outer(a-u, a-u) / np.linalg.norm(a-u)**2) @ C
        #return self

    def _project_into_admissible(self, omega: ExtendedVariable, W: list, h_k: float) -> ExtendedVariable:
        """
        Find the closest admissible extended variable to omega on Gr(p-1, n-1).

        Solves:
            S* = argmin_{S in Gr(p-1, n-1)} d_G(S, S_k)
                s.t. min_{w in W} d_Omega((x_best, S), w) >= h_k

        Parameters
        ----------
        omega : ExtendedVariable
            Current (inadmissible) extended variable with anchor x_best and subspace S_k.
        W : list of ExtendedVariable
            List of already visited extended variables.
        h_k : float
            Exclusion radius lower bound (admissibility threshold).

        Returns
        -------
        ExtendedVariable
            Admissible extended variable closest to omega on the Grassmannian.
        """
        method = self.opt.get("admissible_projection_method", "monte_carlo")

        if method == "monte_carlo":
            return self._project_into_admissible_mc(omega, W, h_k)
        elif method == "riemannian":
            return self._project_into_admissible_riemannian(omega, W, h_k)
        else:
            raise ValueError(
                f"Unknown admissible projection method '{method}'. "
                "Choose 'monte_carlo' or 'riemannian'."
            )

    def _project_into_admissible_mc(self, omega: ExtendedVariable, W: list, h_k: float) -> ExtendedVariable:
        """
        Monte Carlo search for the closest admissible subspace to omega.
        Samples uniformly on Gr(p-1, n-1) via Haar measure and retains
        the admissible sample closest to S_k in chordal distance.

        Parameters
        ----------
        omega : ExtendedVariable
        W : list of ExtendedVariable
        h_k : float

        Returns
        -------
        ExtendedVariable
        """
        n_samples = self.opt.get("admissible_projection_n_samples", 5000)
        thr = self.gamma * h_k
        W_arrays = self._restrict_W(omega.x, self._W_arrays(W), thr)
        q = self.p - 1
        r = self.ratio

        best_S = None
        best_dist = np.inf
        chunk = 500
        for i in range(0, n_samples, chunk):
            Sc = self._sample_grassmann(min(chunk, n_samples - i))
            dmin = self._min_distance_to_W(omega.x, Sc, W_arrays)
            adm = ~(dmin < thr)
            if not adm.any():
                continue
            Sa = Sc[adm]
            M = np.matmul(Sa.transpose(0, 2, 1), omega.S)
            d_to_Sk = np.sqrt((1.0 - r) * np.maximum(q - np.einsum('bij,bij->b', M, M), 0.0))
            j = int(np.argmin(d_to_Sk))
            if d_to_Sk[j] < best_dist:  # strict: first minimum wins, like the sequential loop
                best_dist = d_to_Sk[j]
                best_S = Sa[j]

        if best_S is None:
            raise RuntimeError(
                f"No admissible subspace found after {n_samples} Monte Carlo samples. "
                "Consider increasing 'admissible_projection_n_samples' or "
                "checking the exclusion radius computation."
            )

        return ExtendedVariable(x=omega.x, S=best_S)

    def _project_into_admissible_riemannian(self, omega: ExtendedVariable, W: list, h_k: float) -> ExtendedVariable:
        """
        Riemannian optimization to find the closest admissible subspace to omega.

        Minimizes d_G(S, S_k) subject to min_{w in W} d_Omega((x_best, S), w) >= h_k.
        The constraint is enforced via a penalty term:
            cost(S) = d_G(S, S_k)^2 + K * max(0, h_k - softmin_w d_Omega((x_best,S), w))^2

        Parameters
        ----------
        omega : ExtendedVariable
        W : list of ExtendedVariable
        h_k : float

        Returns
        -------
        ExtendedVariable
        """
        try:
            import pymanopt
            from pymanopt.manifolds import Grassmann
            from pymanopt.optimizers import SteepestDescent,TrustRegions, NelderMead
        except ImportError:
            raise ImportError(
                "pymanopt is required for the 'riemannian' admissible projection method. "
                "Install it with: pip install pymanopt"
            )

        tau = self.opt.get("exclusion_radius_softmin_tau", 0.05)
        n_restarts = self.opt.get("admissible_projection_n_restarts", 5)

        manifold = Grassmann(self.n - 1, self.p - 1) # Anchored subspace isomorphic to Gr(p-1,R^{n-1})
        x_best = omega.x

        @pymanopt.function.numpy(manifold)
        def cost(S):
            omega_candidate = ExtendedVariable(x=x_best, S=S)

            # Grassmann distance to S_k (objective)
            d_to_Sk = omega.distance(omega,r=self.ratio,metric=GrassmannMetric.CHORDAL)
            # Softmin over distances to W (constraint approximation)
            dists = np.array([
                omega_candidate.distance(w, r=self.ratio, metric=GrassmannMetric.CHORDAL)
                for w in W
            ])
            softmin = -tau * np.log(np.sum(np.exp(-dists / tau)))

            # Penalized cost
            violation = max(0.0, h_k - softmin)
            return d_to_Sk ** 2 + self.K_PENALTY * violation ** 2
        
            # Extrem barrier 
            if violation > 0 :
                return np.inf
            else :
                return d_to_Sk

        optimizer = NelderMead(verbosity=0)

        best_S = None
        best_cost = np.inf

        for _ in range(n_restarts):
            problem = pymanopt.Problem(manifold=manifold, cost=cost)
            result = optimizer.run(problem, initial_point=None)

            if result.cost < best_cost:
                # Verify hard admissibility of the solution
                omega_result = ExtendedVariable(x=x_best, S=result.point)
                if self._is_admissible(omega_result, W, h_k):
                    best_cost = result.cost
                    best_S = result.point

        if best_S is None:
            # Fallback to Monte Carlo if riemannian failed to find admissible solution
            warnings.warn(
                "Riemannian projection did not find an admissible solution. "
                "Falling back to Monte Carlo projection.",
                RuntimeWarning,
            )
            return self._project_into_admissible_mc(omega, W, h_k)

        return ExtendedVariable(x=x_best, S=best_S)
 
    def augmentation_isomorphism(self, omega) :
        """
        Returns a basis of the p-dimensional anchored subspace of R^n, 
        from the subspace of interest perpendicular to x_best.
        """
        x = self.anchore
        S = omega.S
        B = self.anchor_basis
        V_raw = np.column_stack([x / np.linalg.norm(x), B @ S])  # shape (n, p)
        V_k, _ = np.linalg.qr(V_raw)  # orthonormalize columns
        return V_k

    def _update_discontinuity_point(self) :
        """Draw a random discontinuity point in the unit shere. 
        And uptate the orthobasis of the discontinuity point."""
        self.discontinuity_point = np.random.uniform(-1, 1, self.n)
        self.discontinuity_point /= np.linalg.norm(self.discontinuity_point)
        #self.discontinuity_basis = self._orthobasis(self.discontinuity_point)
        self.discontinuity_basis = self._orthogonal_complement_basis(self.discontinuity_point)
        return self

    def compute_exclusion_radius(self, x_best: np.ndarray, W: list) -> float:
        """
        Estimate the exclusion radius h_k of the current extended variable list W,
        at anchor point x_best.

        The exclusion radius is defined as:
            h_k = sup_{S in Gr(p-1, n-1)} min_{w in W} d_Omega((x_best, S), w)

        Parameters
        ----------
        x_best : ndarray of shape (n,)
            Current best point (anchor).
        W : list of ExtendedVariable
            List of already visited extended variables.

        Returns
        -------
        float
            Estimated exclusion radius h_k.
        """
        if len(W) == 0:
            return self.h_0#self.opt.get("initial_exclusion_radius", 1.0)

        method = self.opt.get("exclusion_radius_method", "gradient")

        if method == "monte_carlo":
            return self._exclusion_radius_mc(x_best, W)
        elif method == "riemannian":
            return self._exclusion_radius_riemannian(x_best, W)
        elif method == "gradient":
            return self._exclusion_radius_gradient(x_best, W)
        else:
            raise ValueError(
                f"Unknown exclusion radius method '{method}'. "
                "Choose 'monte_carlo' or 'riemannian'."
            )

    def _exclusion_radius_mc(self, x_best: np.ndarray, W: list) -> float:
        """
        Monte Carlo estimate of the exclusion radius by uniform sampling
        over Gr(p-1, n-1) via Haar measure (QR decomposition of Gaussian matrices).

        Parameters
        ----------
        x_best : ndarray of shape (n,)
        W : list of ExtendedVariable

        Returns
        -------
        float
        """
        n_samples = self.opt.get("exclusion_radius_n_samples", 500)
        W_arrays = self._W_arrays(W)
        W_arrays = self._restrict_W(x_best, W_arrays, self._radius_bound(x_best, W_arrays))
        S_sample = self._sample_grassmann(n_samples)
        return float(np.max(self._min_distance_to_W(x_best, S_sample, W_arrays)))

    def _exclusion_radius_gradient(self, x_best: np.ndarray, W: list) -> float:
        """
        Estimate h_k = max_S min_w d((x_best, S), w) by projected gradient ascent on the Grassmannian.

        Start: best of `exclusion_radius_gradient_init` Haar candidates. The min over w is smoothed by a soft-min over ALL the
        (remaining) w: ascent direction = sum over w of the gradients of d_w with weights exp(-(d_w - d_min)/tau) / d_w,
        tau = 0.05 d_min, where
            grad_S d_w = -(1-r) S_w (S_w^T S) / d_w,
        projected on the tangent space (I - S S^T) and followed by a QR retraction; adaptive step (doubled on success,
        halved on failure), accepted only if the true min distance increases. Like Monte Carlo it returns a lower bound of h_k.
        The history is first restricted to the w that can matter (see _restrict_W).
        """
        r = self.ratio
        q = self.p - 1
        W_arrays = self._W_arrays(W)
        Xw, Sw = self._restrict_W(x_best, W_arrays, self._radius_bound(x_best, W_arrays))
        k = Xw.shape[0]
        rdx2 = r * np.sum((Xw - x_best[None, :]) ** 2, axis=1)

        def dists(S):
            SwS = Sw.T @ S  # (k q, q), also used for the gradient
            M = SwS.reshape(k, q, q)
            fro2 = np.einsum('kij,kij->k', M, M)
            return np.sqrt(np.maximum(rdx2 + (1.0 - r) * np.maximum(q - fro2, 0.0), 0.0)), SwS

        cand = self._sample_grassmann(self.opt.get("exclusion_radius_gradient_init", 20))
        dmin_c = self._min_distance_to_W(x_best, cand, (Xw, Sw))
        S = cand[int(np.argmax(dmin_c))]
        d, SwS = dists(S)
        best = d.min()
        step = 0.3
        for _ in range(self.opt.get("exclusion_radius_gradient_max_iter", 30)):
            dm = d.min()
            tau = 0.05 * max(dm, 1e-8)
            wgt = np.exp(-(d - dm) / tau) / np.maximum(d, 1e-8)
            D = -(1.0 - r) * Sw @ (np.repeat(wgt, q)[:, None] * SwS)
            D -= S @ (S.T @ D)
            nD = np.linalg.norm(D)
            if nD < 1e-14:
                break
            ok = False
            for _ in range(6):  # backtracking
                S_new = np.linalg.qr(S + (step / nD) * D)[0]
                d_new, SwS_new = dists(S_new)
                if d_new.min() > best:
                    ok = True
                    break
                step *= 0.5
            if not ok:
                break
            gain = d_new.min() - best
            S, d, SwS, best = S_new, d_new, SwS_new, d_new.min()
            step *= 2.0
            if gain < 1e-3 * best:
                break
        return float(best)

    def _exclusion_radius_riemannian(self, x_best: np.ndarray, W: list) -> float:
        """
        Riemannian optimization estimate of the exclusion radius using pymanopt.
        Maximizes min_{w in W} d_Omega((x_best, S), w) over S in Gr(p-1, n-1).

        Since pymanopt minimizes, we minimize the negative of the objective.
        The non-smooth min is approximated by a smooth log-sum-exp lower bound:
            softmin(d_1, ..., d_m) = -tau * log( sum_i exp(-d_i / tau) )

        Parameters
        ----------
        x_best : ndarray of shape (n,)
        W : list of ExtendedVariable

        Returns
        -------
        float
        """
        try:
            import pymanopt
            from pymanopt.manifolds import Grassmann
            from pymanopt.optimizers import SteepestDescent, TrustRegions, NelderMead
        except ImportError:
            raise ImportError(
                "pymanopt is required for the 'riemannian' exclusion radius method. "
                "Install it with: pip install pymanopt"
            )

        tau = self.opt.get("exclusion_radius_softmin_tau", 0.05)
        n_restarts = self.opt.get("exclusion_radius_n_restarts", 5)

        manifold = Grassmann(self.n - 1, self.p - 1)

        @pymanopt.function.numpy(manifold)
        def cost(S):
            omega = ExtendedVariable(x=x_best, S=S)
            dists = np.array([
                omega.distance(w, r=self.ratio, metric=GrassmannMetric.CHORDAL)
                for w in W
            ])
            # Smooth approximation of min via log-sum-exp (softmin)
            softmin = -tau * np.log(np.sum(np.exp(-dists / tau)))
            return -softmin  # negate to maximize

        optimizer = NelderMead(verbosity=0)

        best_val = -np.inf
        for _ in range(n_restarts):
            problem = pymanopt.Problem(manifold=manifold, cost=cost)
            # print(f"initial_simplex : {initial_simplex}")
            result = optimizer.run(problem, initial_point=None)

            val = -result.cost  # back to maximization value
            if val > best_val:
                best_val = val

        return best_val

    # --- Latent space tools : cost function / doe --- #
    def _gamma_W(self, s: np.ndarray) -> np.ndarray:
        """
        Euclidean projection of As onto the box [-1, 1]^n.

        Parameters
        ----------
        s : ndarray of shape (p,)

        Returns
        -------
        x : ndarray of shape (n,)
        """
        return np.clip(self.A @ s, -1.0, 1.0)

    def _gamma_Z(self, s: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Constrained projection onto the zonotope Z:
            min_{x in [-1,1]^n}  ||x - As||^2
            s.t.                  A^T x = s

        Parameters
        ----------
        s : ndarray of shape (p,)

        Returns
        -------
        x       : ndarray of shape (n,)  — best feasible point found
        success : bool — True iff s lies in the zonotope Z
        """
        s = np.asarray(s, dtype=float).reshape(-1)
        x0 = self.A @ s  # unconstrained optimum; already satisfies A^T x0 = s (A orthonormal)

        # If x0 is already in the box, we are done.
        if np.all(x0 >= -1.0) and np.all(x0 <= 1.0):
            return x0, True

        from scipy.optimize import minimize

        result = minimize(
            fun=lambda x: np.dot(x - x0, x - x0),
            x0=np.clip(x0, -1.0, 1.0),
            method="SLSQP",
            jac=lambda x: 2.0 * (x - x0),
            bounds=[(-1.0, 1.0)] * self.n,
            constraints={"type": "eq", "fun": lambda x: self.A.T @ x - s,
                        "jac": lambda x: self.A.T},
            options={"ftol": 1e-12, "maxiter": 500},
        )

        return result.x, result.success

    def _update_latent_function(self, A: np.ndarray) -> "GCO":
        """
        Update the embedding matrix and the latent cost function f_Z.

        The latent function is defined on the subspace box as:

            f_Z(s) = f(gamma_Z(s))           if s in Z  (zonotope)
                = f(gamma_W(s))
                    + K * ||A^T gamma_W(s) - s||   otherwise

        Parameters
        ----------
        A : ndarray of shape (n, p), orthonormal columns — defines the subspace.
        """
        self.A = np.asarray(A, dtype=float)
        if self.A.shape != (self.n, self.p):
            raise ValueError(
                f"A must have shape ({self.n}, {self.p}), got {self.A.shape}."
            )

        def fun_Z(s: np.ndarray) -> float:
            s = np.asarray(s, dtype=float).reshape(-1)
            x_Z, in_zonotope = self._gamma_Z(s)
            if in_zonotope:
                return float(self.fun(x_Z))
            # Consistent PB
            x_W = self._gamma_W(s)
            penalty = self.K_PENALTY * np.linalg.norm(self.A.T @ x_W - s)**2
            return float(self.fun(x_W)) + penalty
            # EB
            return np.inf

        self.latent_fun = fun_Z
        return self

    def _update_latent_doe(self, V_k) :
        """Update the latent DOE with the new subspace V_k."""
        X, y = self.doe.Xy()
        self.latent_doe = DOE(dim=V_k.shape[1])

        X_keep, y_keep = self._select_cache_points(X, y, V_k)
        self.latent_doe.add(X_keep @ V_k, y=y_keep)
        return self

    @staticmethod
    def _projection_residuals(X: np.ndarray, V_k: np.ndarray) -> np.ndarray:
        """||x - V_k V_k^T x|| for each row x of X, computed in O(N n p) without the n-dim projections
        (Pythagoras: ||x||^2 - ||V_k^T x||^2, V_k having orthonormal columns)."""
        XV = X @ V_k
        sq = np.einsum('ij,ij->i', X, X) - np.einsum('ij,ij->i', XV, XV)
        return np.sqrt(np.maximum(sq, 0.0))

    def _select_cache_points(self, X: np.ndarray, y: np.ndarray, V_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduce the DOE history to at most max_cache_point points, to respect
        NOMAD's cache file line limit, using the method specified in
        self.opt['cache_selection_method'].

        Parameters
        ----------
        X : ndarray of shape (n_points, n)
            Historical points in the original (unit box) space.
        y : ndarray of shape (n_points,)
            Corresponding objective values.
        V_k : ndarray of shape (n, p)
            Current subspace basis (orthonormal columns).

        Returns
        -------
        X_keep : ndarray of shape (min(n_points, max_cache_point), n)
        y_keep : ndarray of shape (min(n_points, max_cache_point),)
        """
        max_points = self.opt.get("max_cache_point", 900)
        threshold = self.opt.get("cache_projection_threshold")

        if threshold is not None and threshold < 0:
            raise ValueError("cache_projection_threshold must be non-negative or None.")

        n_points = X.shape[0]
        residuals = None

        if threshold is not None:
            residuals = GCO._projection_residuals(X, V_k)
            eligible = residuals <= threshold
            X = X[eligible]
            y = y[eligible]
            residuals = residuals[eligible] # keep residuals aligned with the filtered X, y
            n_points = X.shape[0]

            if n_points == 0:
                raise ValueError(
                    "cache_projection_threshold filtered out every DOE point. "
                    "Increase the threshold or disable the filter."
                )

        if n_points <= max_points:
            return X, y

        method = self.opt.get("cache_selection_method", "closest_subspace")

        if method == "random":
            rng = np.random.default_rng()
            idx = rng.choice(n_points, size=max_points, replace=False)

        elif method == "closest_subspace":
            # Residual of orthogonal projection onto span(V_k): ||V_k V_k^T x - x||
            if residuals is None:
                residuals = GCO._projection_residuals(X, V_k)
            idx = np.argsort(residuals)[:max_points]

        else:
            raise ValueError(f"Unknown cache_selection_method: {method}")

        return X[idx], y[idx]
    
    # --- Other tools --- #
    def _scale_function(self,fun) :
        """Scale the function to the unit box [-1, 1]^n."""
        def scaled_fun(z) :
            x = self.scaler.from_unit(z)
            val = fun(x)
            self._eval_times.append(time.perf_counter())
            return val
        return scaled_fun
    
    def _update_doe(self, Xk, Fk) :
        """Update the DOE with the new evaluated points and values."""
        self.doe.add(Xk, y=Fk)
        return self
    
    def _init_logfile(self) -> None:
        """Initialize the logfile with a header."""
        if os.path.exists(self.logfile_path):
            os.remove(self.logfile_path)

        with open(self.logfile_path, "w") as f:
            f.write("  ".join(self.output_columns) + "  \n")

    def save_data(self,Xk,Fk,f_ref,tot_eval) :
        # --- Log all intra-iteration successes ---
        # Each entry of Fk corresponds to exactly one black-box call, in order.
        times = self._eval_times[-len(Fk):] if len(Fk) else []
        for i, (x_i, f_i) in enumerate(zip(Xk, Fk)):
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(
                    bbe=tot_eval + i + 1,
                    f_val=f_i,
                    x=x_i,
                    elapsed=times[i] - self._t0 if i < len(times) else float("nan"),
                )

    def _write_logfile(self, bbe: int, f_val: float, x: np.ndarray, elapsed: float = float("nan")) -> None:
        """
        Append a success entry to the logfile.

        Parameters
        ----------
        bbe   : int     — cumulative number of black-box evaluations at the time of success.
        f_val : float   — objective value at the success point.
        x     : ndarray of shape (n,) — point in the **original** (unscaled) space.
        elapsed : float — seconds since the start of the optimization (TIME column).
        """
        #print(f"Succes coordinates in unit space :  {x}")
        x_original = self.scaler.from_unit(x)
        x_str = "  ".join(f"{xi:.17e}" for xi in x_original)
        # with open(self.logfile_path, "a") as f:
        #     f.write(f"{bbe}  {f_val:.17e}  {x_str}\n")
        fields = {
            "BBE": f"{bbe}",
            "OBJ": f"{f_val:.17e}",
            "TIME": f"{elapsed:.6f}",
        }
        with open(self.logfile_path, "a") as f:
            f.write("  ".join(fields[c] for c in self.output_columns) + " \n")

    def _update_latent_bound(self, V):
        rad = np.sum(np.abs(V), axis=0)
        self.latent_lb = -rad
        self.latent_ub = +rad

def suboptimizer(fun, sigma_k) :
    """Optimize the function fun, with a min step size of sigma_k.
    Return array of evaluated points Xk, array of evaluated values Fk, and number of evaluations n_eval.
    """
    raise NotImplementedError("suboptimizer is a placeholder function. Implement it with your favorite optimization method.")

# --- Example suboptimizers --- #
def objective (point):
    return sum(point.get_coord(index) ** 2 for index in range(point.size()))

def mads_suboptimizer(fun, sigma_k, initial_doe, budget=None, ub=None, lb=None, stat_file= None, cache_name = None) :
    """Optimize the function fun with MADS, with a min step size of sigma_k.
    Return array of evaluated points Xk, array of evaluated values Fk, and number of evaluations n_eval.
    """
    Zk = []
    Fk = []
    n_eval = 0
    def bb(eval_point) :
        nonlocal n_eval
        z_value = np.array([eval_point.get_coord(i) for i in range(eval_point.size())])
        fval = fun(z_value)
        eval_point.setBBO(f"{fval:.17g}".encode("UTF-8"))
        Zk.append(z_value)
        Fk.append(fval)
        n_eval += 1
        return 1

    if cache_name == None :
        cache_path = "cache_file.txt"
    else :
        cache_path = f"cache_{cache_name}.txt"
    if os.path.exists(cache_path):
        os.remove(cache_path)

    write_nomad_cache(doe_s=doe_to_rounded_map(doe=initial_doe,decimals=12), cache_path=cache_path,
            fmt=NomadCacheFormat(bb_output_type="OBJ", cache_hits=0),
            sort_items=True)
    
    params = [
            "BB_OUTPUT_TYPE OBJ",                 # simple: only objective
            "DISPLAY_DEGREE 0",
            "DISPLAY_ALL_EVAL false",
            "DISPLAY_STATS BBE OBJ MESH_SIZE POLL_SIZE",
            "USE_CACHE_FILE_FOR_RERUN true",
            f"CACHE_FILE {str(cache_path)}",
        ]
    if budget is not None :
        params.append(f"MAX_BB_EVAL {budget}")
    if sigma_k is not None:
        min_frame = " ".join(f"{sigma_k:.17g}" for _ in range(len(ub)))
        params.append(f"MIN_FRAME_SIZE ( {min_frame} )")
    if stat_file is not None :
        params.append(f"STATS_FILE {str(stat_file)} BBE OBJ MESH_SIZE POLL_SIZE")


    # Meilleur point du DOE comme X0
    X,y = initial_doe.Xy()
    best_idx = np.argmin(y)
    x0 = X[best_idx].tolist()
    _ = PyNomad.optimize(bb,x0,lb,ub,params)
    return Zk, Fk, n_eval

def doe_to_rounded_map(doe: DOE, decimals: int) -> dict[PointKey, float]:
    """
    Build a mapping {(rounded_point): best_y} from a DOE.
    If multiple points collide to the same rounded key, keep the minimum y.
    """
    X, y = doe.Xy()
    out: dict[PointKey, float] = {}

    for si, yi in zip(X, y):
        key = tuple(np.round(si, decimals=decimals).tolist())
        val = float(yi)
        if key not in out or val < out[key]:
            out[key] = val
    return out

_trego_calls = itertools.count()  # successive calls get different (but reproducible) seeds

def trego_suboptimizer(fun, sigma_k, initial_doe, budget=None, ub=None, lb=None, stat_file=None, cache_name=None,
                       seed=None, G=1, L=4, beta=0.5, tsigma_0=0.05, refit_every=3, max_train=100) :
    """Optimize the function fun with TREGO (see utils/trego.py), with a min trust-region size derived from sigma_k.
    The initial_doe is used to warm start the GP but is not counted nor returned.
    Like MADS, there is no evaluation cap if budget is None and sigma_k is given (the run stops when the
    trust-region size tsigma falls below sigma_k); if both are None the budget is 5*p. An explicit budget is always honoured.
    tsigma_0 is the initial trust-region size (float, or callable of the normalised minimum size).
    stat_file and cache_name are accepted and ignored.
    If seed is None, a call counter is used so that successive calls differ but runs are reproducible.
    Return array of evaluated points Xk, array of evaluated values Fk, and number of evaluations n_eval.
    """
    if seed is None :
        seed = next(_trego_calls)
    return trego(fun, initial_doe, lb, ub, tsigma_min=sigma_k, budget=budget, seed=seed,
                 G=G, L=L, beta=beta, tsigma_0=tsigma_0, refit_every=refit_every, max_train=max_train)

if __name__ == "__main__" :
    import sys
    from pathlib import Path
    dossier_parent = Path(__file__).resolve().parent.parent
    sys.path.append(str(dossier_parent))
    #from Folder_of_test.test_function import *
    from scipy.stats import qmc

    n = 400
    lb = np.array([-10] * n)
    ub = np.array([10] * n)

    var = {"dim" : n, "lower_bound" : lb, "upper_bound" : ub }
    seed = 0

    n = 400
    lb = np.array([-10] * n)
    ub = np.array([10] * n)

    var = {"dim" : n, "lower_bound" : lb, "upper_bound" : ub }
    seed = 0

    pb_name = "rosenbrock"

    def rosenbrock(x):
        """Rosenbrock function in n dimensions."""
        x = np.asarray(x)
        return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    _gco = GCO(rosenbrock,vars_prop=var,opt={"file_path": f"GCO(MADS,PLS,p10)_{pb_name}.txt", "subspace_dimension" : 10, "subspace_selection" : "PLS"})

    initial_doe = DOE(n,lb=lb,ub=ub)
    initial_doe.init_from_method(N_points=n//2, method="LHS",fun=rosenbrock)
    X,F = initial_doe.Xy()

    x_best, f_best = _gco.run_optim(X0=X,F0=F,suboptimizer = mads_suboptimizer, budget=5000)
    print(x_best,f_best)


    X_mads,F_mads,_ = mads_suboptimizer(rosenbrock,None,initial_doe=initial_doe,budget=1000*n,ub=ub,lb=lb,stat_file=f"Nomad_rossenbrock_dim_{n}.txt")
    f_mads = min(F_mads)
    print(f"min mads = {f_mads}")

    initial_doe = DOE(n,lb=lb,ub=ub)
    initial_doe.init_from_method(N_points=n//2, method="LHS",fun=rosenbrock)
    X,F = initial_doe.Xy()

    x_best, f_best = _gco.run_optim(X0=X,F0=F,suboptimizer = mads_suboptimizer, budget=5000)
    print(x_best,f_best)


    X_mads,F_mads,_ = mads_suboptimizer(rosenbrock,None,initial_doe=initial_doe,budget=5000,ub=ub,lb=lb,stat_file=f"Nomad_rossenbrock_dim_{n}.txt")
    f_mads = min(F_mads)
    print(f"min mads = {f_mads}")
