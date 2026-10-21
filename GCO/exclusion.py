"""Exclusion geometry of GCO: distances to the visited extended variables W, exclusion radius h_k, admissibility test and
projection of an inadmissible subspace onto the admissible set.

Extended variable w = (x_w, S_w): anchor x_w (unit box) and orthonormal frame S_w ((n-1) x q, q = p-1). Distance:
    d(w1, w2)^2 = r ||x1 - x2||^2 + (1 - r) d_G(S1, S2)^2,   d_G(S1, S2)^2 = q - ||S1^T S2||_F^2   (chordal distance).
Exclusion radius: h_k = max_S min_{w in W} d((x_best, S), w).   Admissible: min_{w in W} d((x_best, S), w) >= gamma h_k.
"""
import numpy as np

from .utils.extended_var import ExtendedVariable


class ExclusionMixin:
    """Methods of GCO about W, h_k and admissibility (uses self.n, self.p, self.ratio, self.gamma, self.opt, self.h_0)."""

    # ------------------------------------------------------------------------------------------------ distances to W
    def _W_arrays(self, W):
        """Stack the visited extended variables: anchors (k, n) and concatenated frames (n-1, k*q)."""
        Xw = np.array([w.x for w in W])
        Sw = np.concatenate([w.S for w in W], axis=1)
        return Xw, Sw

    def _restrict_W(self, x, W_arrays, bound):
        """Keep only the w that can be closer than `bound` to (x, S), whatever S. Since d^2 >= r ||x - x_w||^2, a w with
        r ||x - x_w||^2 >= bound^2 is never closer than `bound`: exact pruning (no approximation)."""
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
        """Upper bound of min_w d((x, S), w) valid for every S (the w nearest to x in x, with d_G^2 <= q)."""
        Xw, _ = W_arrays
        r = self.ratio
        dx2 = np.min(np.sum((Xw - x[None, :]) ** 2, axis=1))
        return np.sqrt(r * dx2 + (1.0 - r) * (self.p - 1)) * (1.0 + 1e-9) + 1e-12

    def _min_distance_to_W(self, x, S_batch, W_arrays, chunk_bytes=150e6):
        """min_{w in W} d((x, S), w) for every frame S of S_batch (shape (B, n-1, q)), one matrix product per chunk."""
        Xw, Sw = W_arrays
        B, _, q = S_batch.shape
        k = Xw.shape[0]
        if k == 0:
            return np.full(B, np.inf)
        r = self.ratio
        rdx2 = r * np.sum((Xw - x[None, :]) ** 2, axis=1)
        out = np.empty(B)
        step = max(1, int(chunk_bytes // (8 * q * q * k)))
        for i in range(0, B, step):
            Sb = S_batch[i:i + step]
            b = Sb.shape[0]
            G = Sb.transpose(1, 0, 2).reshape(Sb.shape[1], b * q).T @ Sw  # (b*q, k*q): all the S^T S_w at once
            G4 = G.reshape(b, q, k, q)
            fro2 = np.einsum('bikj,bikj->bk', G4, G4)
            d2 = rdx2[None, :] + (1.0 - r) * np.maximum(q - fro2, 0.0)
            out[i:i + b] = np.sqrt(d2.min(axis=1))
        return out

    def _sample_grassmann(self, n_samples):
        """n_samples random (n-1) x q orthonormal frames, uniform (Haar) on the Grassmannian (Gaussian matrices + QR).
        The Generator is seeded from the global numpy stream, so np.random.seed keeps runs reproducible."""
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        Z = rng.standard_normal((n_samples, self.n - 1, self.p - 1))
        return np.linalg.qr(Z)[0]

    def _is_admissible(self, omega, W, h_k):
        """min_{w in W} d(omega, w) >= gamma h_k."""
        if len(W) == 0:
            return True
        W_arrays = self._restrict_W(omega.x, self._W_arrays(W), self.gamma * h_k)
        d = self._min_distance_to_W(omega.x, omega.S[None, :, :], W_arrays)[0]
        return not (d < self.gamma * h_k)

    # ------------------------------------------------------------------------------------------------ exclusion radius
    def compute_exclusion_radius(self, x_best, W):
        """h_k = max_S min_{w in W} d((x_best, S), w), estimated by the method of the option exclusion_radius_method (h_0 if W is empty).
        The maximiser S* is kept (self._radius_argmax) as an always-admissible fallback for the projection."""
        if len(W) == 0:
            return self.h_0
        method = self.opt.get("exclusion_radius_method", "gradient")
        if method == "gradient":
            return self._exclusion_radius_gradient(x_best, W)
        if method == "monte_carlo":
            return self._exclusion_radius_mc(x_best, W)
        if method == "riemannian":
            from .riemannian import exclusion_radius_riemannian
            return exclusion_radius_riemannian(self, x_best, W)
        raise ValueError(f"Unknown exclusion radius method '{method}'.")

    def _exclusion_radius_mc(self, x_best, W):
        """Best of exclusion_radius_n_samples random subspaces (a lower bound of h_k)."""
        W_arrays = self._W_arrays(W)
        W_arrays = self._restrict_W(x_best, W_arrays, self._radius_bound(x_best, W_arrays))
        S_sample = self._sample_grassmann(self.opt.get("exclusion_radius_n_samples", 500))
        dmin = self._min_distance_to_W(x_best, S_sample, W_arrays)
        j = int(np.argmax(dmin))
        self._radius_argmax = (x_best.copy(), S_sample[j].copy())
        return float(dmin[j])

    def _exclusion_radius_gradient(self, x_best, W):
        """Projected gradient ascent on the Grassmannian (a lower bound of h_k, tighter than Monte Carlo).

        Start: best of exclusion_radius_gradient_init random subspaces. The min over w is smoothed by a soft-min: direction
        D = sum_w exp(-(d_w - d_min)/tau) grad_S d_w with grad_S d_w = -(1-r) S_w (S_w^T S) / d_w and tau = 0.05 d_min, projected on
        the tangent space (D <- D - S S^T D), QR retraction, adaptive step accepted only if the true min distance increases.
        """
        r = self.ratio
        q = self.p - 1
        W_arrays = self._W_arrays(W)
        Xw, Sw = self._restrict_W(x_best, W_arrays, self._radius_bound(x_best, W_arrays))
        k = Xw.shape[0]
        rdx2 = r * np.sum((Xw - x_best[None, :]) ** 2, axis=1)

        def dists(S):
            SwS = Sw.T @ S  # (k q, q), reused for the gradient
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
        self._radius_argmax = (x_best.copy(), S.copy())
        return float(best)

    # ------------------------------------------------------------------------------------------------ admissible subspace
    def _admissible_with_fixed(self, S, n_fixed, x_best, W, h_k):
        """First admissible [C, R] among fixed_columns_draws random completions R orthogonal to the pinned columns C = S[:, :n_fixed]."""
        C = S[:, :n_fixed]
        rest = self.p - 1 - n_fixed
        for _ in range(self.opt["fixed_columns_draws"]):
            R = np.random.randn(self.n - 1, rest)
            R = R - C @ (C.T @ R)
            om = ExtendedVariable(x_best, np.linalg.qr(np.column_stack([C, np.linalg.qr(R)[0]]))[0])
            if self._is_admissible(om, W, h_k):
                return om
        return None

    def _project_into_admissible(self, omega, W, h_k):
        """Admissible extended variable close to omega, by the method of the option admissible_projection_method."""
        method = self.opt.get("admissible_projection_method", "gradient")
        if method == "gradient":
            return self._project_into_admissible_gradient(omega, W, h_k)
        if method == "monte_carlo":
            return self._project_into_admissible_mc(omega, W, h_k)
        if method == "riemannian":
            from .riemannian import project_into_admissible_riemannian
            return project_into_admissible_riemannian(self, omega, W, h_k)
        raise ValueError(f"Unknown admissible projection method '{method}'.")

    def _radius_maximiser(self, omega, W, h_k):
        """Fallback: the maximiser S* of the exclusion radius at the same anchor has min_w d = h_k >= gamma h_k, so it is admissible.
        Returns it (re-checked), or None if it is not available."""
        rm = getattr(self, "_radius_argmax", None)
        if rm is None or not np.array_equal(rm[0], omega.x):
            return None
        om = ExtendedVariable(x=omega.x, S=rm[1])
        if self._is_admissible(om, W, h_k):
            self.n_radius_fallbacks = getattr(self, "n_radius_fallbacks", 0) + 1
            return om
        return None

    def _project_into_admissible_gradient(self, omega, W, h_k):
        """
        1) Ascent: for every w with d_w < gamma h_k, grad_S d_w = -(1-r) S_w (S_w^T S) / d_w; direction D = sum_w (gamma h_k - d_w) grad_S d_w,
           projected on the tangent space, QR retraction, adaptive step on the penalty sum_w (gamma h_k - d_w)_+^2; stops at the first
           admissible S.
        2) Bisection on t in [0, 1] along S(t) = qr((1 - t) S_k + t S): smallest admissible t, to stay close to the proposed S_k.
        If the ascent fails: Monte Carlo projection (which itself falls back on S*).
        """
        thr = self.gamma * h_k
        r = self.ratio
        x = omega.x
        S0 = np.linalg.qr(omega.S)[0]
        S = S0.copy()
        q = S.shape[1]
        Xw, Sw = self._restrict_W(x, self._W_arrays(W), thr)
        k = Xw.shape[0]
        if k == 0:
            return ExtendedVariable(x=x, S=S0)
        rdx2 = r * np.sum((Xw - x[None, :]) ** 2, axis=1)

        def dists(S):
            M = (Sw.T @ S).reshape(k, q, q)
            return np.sqrt(np.maximum(rdx2 + (1.0 - r) * np.maximum(q - np.einsum('kij,kij->k', M, M), 0.0), 0.0))

        d = dists(S)
        pen = np.sum(np.maximum(thr - d, 0.0) ** 2)
        step = 0.5
        for _ in range(self.opt["gradient_projection_max_iter"]):
            if d.min() >= thr:
                break
            viol = np.flatnonzero(d < thr)
            coef = (thr - d[viol]) / np.maximum(d[viol], 1e-8)
            cols = (viol[:, None] * q + np.arange(q)[None, :]).ravel()
            Swv = Sw[:, cols]
            D = -(1.0 - r) * (Swv * np.repeat(coef, q)[None, :]) @ (Swv.T @ S)
            D -= S @ (S.T @ D)
            nD = np.linalg.norm(D)
            if nD < 1e-14:
                break
            ok = False
            while step >= 1e-8:  # backtracking on the penalty
                S_new = np.linalg.qr(S + (step / nD) * D)[0]
                d_new = dists(S_new)
                pen_new = np.sum(np.maximum(thr - d_new, 0.0) ** 2)
                if pen_new < pen:
                    ok = True
                    break
                step *= 0.5
            if not ok:
                break
            S, d, pen = S_new, d_new, pen_new
            step *= 2.0
        if d.min() < thr:
            return self._project_into_admissible_mc(omega, W, h_k)
        lo, hi, S_hi = 0.0, 1.0, S
        for _ in range(14):  # bisection back towards S_k
            t = 0.5 * (lo + hi)
            S_t = np.linalg.qr((1.0 - t) * S0 + t * S)[0]
            if dists(S_t).min() >= thr:
                hi, S_hi = t, S_t
            else:
                lo = t
        return ExtendedVariable(x=x, S=S_hi)

    def _project_into_admissible_mc(self, omega, W, h_k):
        """Closest admissible subspace to omega (chordal distance) among admissible_projection_n_samples random subspaces;
        S* if none is admissible."""
        n_samples = self.opt.get("admissible_projection_n_samples", 5000)
        thr = self.gamma * h_k
        W_arrays = self._restrict_W(omega.x, self._W_arrays(W), thr)
        q = self.p - 1
        r = self.ratio
        best_S, best_dist = None, np.inf
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
            if d_to_Sk[j] < best_dist:  # strict: the first minimum wins
                best_dist = d_to_Sk[j]
                best_S = Sa[j]
        if best_S is None:
            om = self._radius_maximiser(omega, W, h_k)
            if om is not None:
                return om
            raise RuntimeError(f"No admissible subspace found after {n_samples} Monte Carlo samples. "
                               "Consider increasing 'admissible_projection_n_samples' or checking the exclusion radius computation.")
        return ExtendedVariable(x=omega.x, S=best_S)
