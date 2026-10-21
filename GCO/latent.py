"""Latent problem of GCO in the p-dimensional subspace V = span(A) (A: n x p, orthonormal columns).

A latent point s in R^p is evaluated at the point of the box [-1,1]^n that represents it:
    f_Z(s) = f(gamma_Z(s))                                  if s is in the zonotope Z = {A^T x : x in [-1,1]^n},
    f_Z(s) = f(clip(A s)) + K ||A^T clip(A s) - s||^2        otherwise (penalty),
with gamma_Z(s) = argmin ||x - A s||^2  s.t.  A^T x = s, -1 <= x <= 1.
The suboptimizer also receives the already evaluated points close to V (latent DOE) and the bounding box of Z.
"""
import numpy as np

from .utils.doe_class import DOE


class LatentMixin:
    """Methods of GCO building the latent function, its DOE and its bounds (uses self.n, self.p, self.fun, self.opt, self.doe)."""

    K_PENALTY = 100000.0  # penalty constant outside the zonotope

    # ------------------------------------------------------------------------------------------------ latent function
    def _update_latent_function(self, A):
        """Set the subspace A (n x p) and build self.latent_fun = f_Z."""
        self.A = np.asarray(A, dtype=float)
        if self.A.shape != (self.n, self.p):
            raise ValueError(f"A must have shape ({self.n}, {self.p}), got {self.A.shape}.")

        def fun_Z(s):
            s = np.asarray(s, dtype=float).reshape(-1)
            x_Z, in_zonotope = self._gamma_Z(s)
            if in_zonotope:
                return float(self.fun(x_Z))
            x_W = self._gamma_W(s)
            penalty = self.K_PENALTY * np.linalg.norm(self.A.T @ x_W - s) ** 2
            return float(self.fun(x_W)) + penalty

        self.latent_fun = fun_Z
        return self

    def _gamma_W(self, s):
        """Euclidean projection of A s onto the box [-1, 1]^n."""
        return np.clip(self.A @ s, -1.0, 1.0)

    def _gamma_Z(self, s):
        """(gamma_Z(s), True) if s is in the zonotope, else (a point of the box, False). Dual Newton by default, SLSQP as fallback."""
        s = np.asarray(s, dtype=float).reshape(-1)
        x0 = self.A @ s  # unconstrained optimum, already satisfies A^T x0 = s
        if np.all(x0 >= -1.0) and np.all(x0 <= 1.0):
            return x0, True

        if self.opt.get("zonotope_projection", "newton") == "newton":
            res = self._gamma_Z_newton(s)
            if res is not None:
                return res
            self.newton_stats["fallbacks"] += 1

        from scipy.optimize import minimize
        result = minimize(
            fun=lambda x: np.dot(x - x0, x - x0),
            x0=np.clip(x0, -1.0, 1.0),
            method="SLSQP",
            jac=lambda x: 2.0 * (x - x0),
            bounds=[(-1.0, 1.0)] * self.n,
            constraints={"type": "eq", "fun": lambda x: self.A.T @ x - s, "jac": lambda x: self.A.T},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        return result.x, result.success

    def _gamma_Z_newton(self, s, max_iter=60):
        """
        Dual problem in dimension p: x(lam) = clip(A (s + lam)), with lam solving g(lam) = A^T clip(A (s + lam)) - s = 0, the gradient of
        the convex psi(lam) = 1/2 ||lam||^2 - 1/2 dist^2(A (s + lam), [-1,1]^n). Semismooth Newton (Hessian A_F^T A_F + mu ||g|| I, F = free
        coordinates), Armijo line search. s outside Z is detected by the certificate u^T s > ||A u||_1, u in {lam, -g, lam - g}; then
        (clip(A s), False). If Newton does not conclude: LP membership test, then a second, more regularised pass.
        Returns (x, in_zonotope), or None (the caller then uses SLSQP).
        """
        A = self.A
        if not hasattr(self, "newton_stats"):
            self.newton_stats = {"calls": 0, "newton_inside": 0, "exterior": 0, "lp_exterior": 0, "second_pass": 0, "maxit_hits": 0,
                                 "fallbacks": 0, "iters": 0}
        st = self.newton_stats
        st["calls"] += 1
        res = self._dual_newton(s, max_iter, 1e-3, certify=True)
        if res is not None:
            return res
        from scipy.optimize import linprog
        n = A.shape[0]
        lp = linprog(np.zeros(n), A_eq=A.T, b_eq=s, bounds=[(-1.0, 1.0)] * n, method="highs")
        if lp.status == 2:  # infeasible: s is outside the zonotope
            st["lp_exterior"] += 1
            return np.clip(A @ s, -1.0, 1.0), False
        st["second_pass"] += 1
        res = self._dual_newton(s, 4 * max_iter, 1e-1, certify=False)
        if res is None:
            st["maxit_hits"] += 1
        return res

    def _dual_newton(self, s, max_iter, mu, certify):
        A = self.A
        st = self.newton_stats
        p = A.shape[1]
        tol = 1e-11 * (1.0 + np.linalg.norm(s))
        lam = np.zeros(p)

        def psi(l):
            y = A @ (s + l)
            r = y - np.clip(y, -1.0, 1.0)
            return 0.5 * float(l @ l) - 0.5 * float(r @ r)

        for it in range(max_iter):
            y = A @ (s + lam)
            c = np.clip(y, -1.0, 1.0)
            g = A.T @ c - s
            ng = np.linalg.norm(g)
            if ng <= tol:
                st["newton_inside"] += 1
                st["iters"] += it
                return c, True
            if certify:
                for u in (lam, -g, lam - g):  # exact certificate that s is outside Z
                    us = float(u @ s)
                    if us > np.abs(A @ u).sum() + 1e-12 * (1.0 + abs(us)):
                        st["exterior"] += 1
                        st["iters"] += it
                        return np.clip(A @ s, -1.0, 1.0), False
            AF = A[np.abs(y) < 1.0]
            H = AF.T @ AF + (mu * ng) * np.eye(p)
            try:
                d = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                d = -np.linalg.lstsq(H, g, rcond=None)[0]
            slope = float(g @ d)
            if not slope < 0.0:
                d, slope = -g, -float(g @ g)
            f0 = psi(lam)
            slack = 1e-14 * (1.0 + abs(f0))  # rounding tolerance of the line search
            t, ok = 1.0, False
            for _ in range(40):
                if psi(lam + t * d) <= f0 + 1e-4 * t * slope + slack:
                    ok = True
                    break
                t *= 0.5
            if not ok:
                break
            lam = lam + t * d
        return None

    # ------------------------------------------------------------------------------------------------ latent DOE and bounds
    def _update_latent_doe(self, V_k):
        """Latent DOE = evaluated points close to V_k (see _select_cache_points), in latent coordinates."""
        X, y = self.doe.Xy()
        self.latent_doe = DOE(dim=V_k.shape[1])
        X_keep, y_keep = self._select_cache_points(X, y, V_k)
        self.latent_doe.add(X_keep @ V_k, y=y_keep)
        return self

    def _update_latent_bound(self, V):
        """Bounding box of the zonotope: |s_j| <= sum_i |V_ij|."""
        rad = np.sum(np.abs(V), axis=0)
        self.latent_lb = -rad
        self.latent_ub = +rad

    @staticmethod
    def _projection_residuals(X, V_k):
        """||x - V_k V_k^T x|| for each row x of X (Pythagoras: ||x||^2 - ||V_k^T x||^2)."""
        XV = X @ V_k
        sq = np.einsum('ij,ij->i', X, X) - np.einsum('ij,ij->i', XV, XV)
        return np.sqrt(np.maximum(sq, 0.0))

    def _select_cache_points(self, X, y, V_k):
        """Points given to the suboptimizer: those with ||x - V_k V_k^T x|| <= cache_projection_threshold (if set), at most
        max_cache_point of them (closest to V_k, or random, see cache_selection_method)."""
        max_points = self.opt.get("max_cache_point", 900)
        threshold = self.opt.get("cache_projection_threshold")
        if threshold is not None and threshold < 0:
            raise ValueError("cache_projection_threshold must be non-negative or None.")

        n_points = X.shape[0]
        residuals = None
        if threshold is not None:
            residuals = LatentMixin._projection_residuals(X, V_k)
            eligible = residuals <= threshold
            X, y, residuals = X[eligible], y[eligible], residuals[eligible]
            n_points = X.shape[0]
            if n_points == 0:
                raise ValueError("cache_projection_threshold filtered out every DOE point. Increase the threshold or disable the filter.")

        if n_points <= max_points:
            return X, y

        method = self.opt.get("cache_selection_method", "closest_subspace")
        if method == "random":
            idx = np.random.default_rng().choice(n_points, size=max_points, replace=False)
        elif method == "closest_subspace":
            if residuals is None:
                residuals = LatentMixin._projection_residuals(X, V_k)
            idx = np.argsort(residuals)[:max_points]
        else:
            raise ValueError(f"Unknown cache_selection_method: {method}")
        return X[idx], y[idx]
