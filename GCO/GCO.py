import os

import numpy as np
from utils.extended_var import *
from utils.doe_class import DOE
from utils.domain import BoxScaler
#from SSO_algo.utils.subspace_class import Subspace
from utils.nomad_io import write_nomad_cache, NomadCacheFormat,PointKey

import PyNomad
import warnings


class GCO :
    OPTIONS_SPEC = {
        "file_path": {
            "type": str,
            "default": "log_GCO.txt",
            "desc": "Path to the logfile where optimization successes are recorded.",
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
        "subspace_selection": {
            "type": str,
            "default": "PLS",
            "choices": {"random", "PLS", "Local_PLS"},
            "desc": "Method used to select the subspace matrix S_k on Gr(p-1, n-1).",
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
        "covering_radius_method": {
            "type": str,
            "default": "monte_carlo",
            "choices": {"monte_carlo", "riemannian"},
            "desc": (
                "Method used to estimate the covering radius h_k of the visited "
                "extended variable list W. "
                "'monte_carlo' samples uniformly on the Grassmannian via Haar measure. "
                "'riemannian' uses pymanopt to maximize the min-distance objective."
            ),
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
            if "choices" in spec and value not in spec["choices"]:
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
        self.n = int(vars_prop["dim"])

        self._init_logfile()

        # Problem properties :
        user_LB = np.asarray(vars_prop["lower_bound"], dtype=float).reshape(-1)
        user_UB = np.asarray(vars_prop["upper_bound"], dtype=float).reshape(-1)
        self.scaler = BoxScaler(user_LB, user_UB)
        self.fun = self._scale_function(fun)
        

        self.p = int(self.opt["subspace_dimension"])

        # Constants of CGO :
        
        self.K_PENALTY = 100000.0 # Penalty constant for the latent function outside the zonotope
        self.gamma = self.opt.get("gamma", 0.95)

        # self.C_SIGMA = 0.1 # Link between covering radius h_k and the min step size sigma_k of the suboptimizer : sigma_k = C_SIGMA * h_k
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

        tot_eval, k, h_k, sigma_k = 0, 0, self.h_0, self.sigma_0
        W = [] # list of extended variables
        h = []


        f_best = float(np.min(F0))

        # Log initial DOE
        f_ref = np.inf
        for i, (x_i, f_i) in enumerate(zip(X0, F0)):
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(bbe=i + 1, f_val=f_i, x=x_i)
        
        tot_eval += len(F0)

        while tot_eval < budget :
            k += 1 # subspace number 
            #print(f"Subspace n°{k}")

            # Get the realisable anchored subspace V_k
            omega_k, h_k = self.anchored_subspace(W, x_best)

            # Update the list of extended variables, and global covering radius
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
        h_k = self.compute_covering_radius(x_best, W)
        #print(f"Current covering radius {h_k}")
        if self.update_anchor :
            self._get_anchor_basis(x_best) #basis of x_best perpendicular space
        B = self.anchor_basis
        X, y = self.doe.Xy() 
        doe_perp = DOE(dim=B.shape[1])
        doe_perp.add(X@B, y=y)
        omega_k = self.get_extended_variable(doe_perp, x_best, W, h_k)
        V_k = self.augmentation_isomorphism(omega_k) # From Gr(p-1,n-1) to Gr(p,n)

        self.curent_subspace = V_k
        self._update_latent_function(V_k)
        self._update_latent_doe(V_k)
        self._update_latent_bound(V_k) 

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
            from sklearn.cross_decomposition import PLSRegression as pls
            import scipy

            # Choose the K closest points to x_best
            K = min(10 * (self.n - 1), len(doe_perp.Xy()[0]))
            X_full, y_full = self.doe.Xy()

            # Sort by distance to x_best
            dists = np.linalg.norm(X_full - x_best, axis=1)
            idx = np.argsort(dists)[:K]
            X_local = X_full[idx]
            y_local = y_full[idx]

            # Project into the anchor perpendicular space
            X_perp_local = X_local @ self.anchor_basis

            _pls = pls(n_components=self.p - 1)
            S_k = _pls.fit(X_perp_local, y_local).x_rotations_
            S_k = scipy.linalg.orth(S_k)

        else :
            raise NotImplementedError("Subspace selection method not implemented.")
        omega = ExtendedVariable(x_best,S_k)
        if not self._is_admissible(omega, W, h_k) :
            omega = self._project_into_admissible(omega, W, h_k)
        return omega

    def _is_admissible(self, omega , W, h_k) :
        """Check if the subspace V_k is belong to restricted anchored space."""
        #print(f"Relaxed exclusion radius : {self.gamma* h_k}")
        for w in W :
            #print(f"Actual distance to this neighboor {omega.distance(w, r=self.ratio, metric=GrassmannMetric.CHORDAL)}")
            if omega.distance(w, r=self.ratio, metric=GrassmannMetric.CHORDAL) < self.gamma * h_k :
                return False
        return True

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
            Covering radius lower bound (admissibility threshold).

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

        best_S = None
        best_dist = np.inf

        for _ in range(n_samples):
            S_raw = np.random.randn(self.n - 1, self.p - 1)
            Q, _ = np.linalg.qr(S_raw)
            S_candidate = Q[:, : self.p - 1]

            omega_candidate = ExtendedVariable(x=omega.x, S=S_candidate)

            if self._is_admissible(omega_candidate, W, h_k):
                #dist_to_Sk = omega_candidate._grassmann_distance(omega, metric=GrassmannMetric.CHORDAL)
                dist_to_Sk = omega_candidate.distance(omega,r=self.ratio, metric = GrassmannMetric.CHORDAL)
                if dist_to_Sk < best_dist:
                    best_dist = dist_to_Sk
                    best_S = S_candidate

        if best_S is None:
            raise RuntimeError(
                f"No admissible subspace found after {n_samples} Monte Carlo samples. "
                "Consider increasing 'admissible_projection_n_samples' or "
                "checking the covering radius computation."
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

        tau = self.opt.get("covering_radius_softmin_tau", 0.05)
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

    def compute_covering_radius(self, x_best: np.ndarray, W: list) -> float:
        """
        Estimate the covering radius h_k of the current extended variable list W,
        at anchor point x_best.

        The covering radius is defined as:
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
            Estimated covering radius h_k.
        """
        if len(W) == 0:
            return self.h_0#self.opt.get("initial_covering_radius", 1.0)

        method = self.opt.get("covering_radius_method", "monte_carlo")

        if method == "monte_carlo":
            return self._covering_radius_mc(x_best, W)
        elif method == "riemannian":
            return self._covering_radius_riemannian(x_best, W)
        else:
            raise ValueError(
                f"Unknown covering radius method '{method}'. "
                "Choose 'monte_carlo' or 'riemannian'."
            )

    def _covering_radius_mc(self, x_best: np.ndarray, W: list) -> float:
        """
        Monte Carlo estimate of the covering radius by uniform sampling
        over Gr(p-1, n-1) via Haar measure (QR decomposition of Gaussian matrices).

        Parameters
        ----------
        x_best : ndarray of shape (n,)
        W : list of ExtendedVariable

        Returns
        -------
        float
        """
        n_samples = self.opt.get("covering_radius_n_samples", 500)
        max_min_dist = -np.inf

        for _ in range(n_samples):
            S_raw = np.random.randn(self.n - 1, self.p - 1)
            Q, _ = np.linalg.qr(S_raw)
            S_sample = Q[:, : self.p - 1]

            omega_sample = ExtendedVariable(x=x_best, S=S_sample)
            min_dist = min(
                omega_sample.distance(w, r=self.ratio, metric=GrassmannMetric.CHORDAL)
                for w in W
            )
            if min_dist > max_min_dist:
                max_min_dist = min_dist

        return max_min_dist

    def _covering_radius_riemannian(self, x_best: np.ndarray, W: list) -> float:
        """
        Riemannian optimization estimate of the covering radius using pymanopt.
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
                "pymanopt is required for the 'riemannian' covering radius method. "
                "Install it with: pip install pymanopt"
            )

        tau = self.opt.get("covering_radius_softmin_tau", 0.05)
        n_restarts = self.opt.get("covering_radius_n_restarts", 5)

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
        self.latent_doe.add(X@V_k, y=y)
        return self
    
    # --- Other tools --- #
    def _scale_function(self,fun) :
        """Scale the function to the unit box [-1, 1]^n."""
        def scaled_fun(z) :
            x = self.scaler.from_unit(z)
            return fun(x)
        return scaled_fun
    
    def _update_doe(self, Xk, Fk) :
        """Update the DOE with the new evaluated points and values."""
        self.doe.add(Xk, y=Fk)
        return self
    
    def _init_logfile(self) -> None:
        """Initialize the logfile with a header."""
        if os.path.exists(self.logfile_path):
            os.remove(self.logfile_path)

        x_header = "  ".join(f"X{i}" for i in range(self.n))
        with open(self.logfile_path, "w") as f:
            f.write(f"BBE  OBJ  \n")

    def save_data(self,Xk,Fk,f_ref,tot_eval) :
        # --- Log all intra-iteration successes ---
        for i, (x_i, f_i) in enumerate(zip(Xk, Fk)):
            if f_i < f_ref:
                f_ref = f_i
                self._write_logfile(
                    bbe=tot_eval + i + 1,
                    f_val=f_i,
                    x=x_i,
                )

    def _write_logfile(self, bbe: int, f_val: float, x: np.ndarray) -> None:
        """
        Append a success entry to the logfile.

        Parameters
        ----------
        bbe   : int     — cumulative number of black-box evaluations at the time of success.
        f_val : float   — objective value at the success point.
        x     : ndarray of shape (n,) — point in the **original** (unscaled) space.
        """
        #print(f"Succes coordinates in unit space :  {x}")
        x_original = self.scaler.from_unit(x)
        x_str = "  ".join(f"{xi:.17e}" for xi in x_original)
        # with open(self.logfile_path, "a") as f:
        #     f.write(f"{bbe}  {f_val:.17e}  {x_str}\n")
        with open(self.logfile_path, "a") as f:
            f.write(f"{bbe}  {f_val:.17e} \n")

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
    result = PyNomad.optimize(bb,x0,lb,ub,params)

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

def trego_suboptimizer(fun, sigma_k) :
    """Optimize the function fun with Trego, with a min step size of sigma_k.
    Return array of evaluated points Xk, array of evaluated values Fk, and number of evaluations n_eval.
    """

if __name__ == "__main__" :
    import sys
    from pathlib import Path
    dossier_parent = Path(__file__).resolve().parent.parent
    sys.path.append(str(dossier_parent))
    from Folder_of_test.test_function import *
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
