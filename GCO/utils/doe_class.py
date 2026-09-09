from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Iterable, Mapping
import numpy as np
from dataclasses import dataclass

Objective = Callable[[np.ndarray], float]

@dataclass
class DOE:
    dim: int
    lb: Optional[np.ndarray] = None
    ub: Optional[np.ndarray] = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._meta: list[dict[str, Any]] = []

        if self.lb is not None:
            self.lb = np.asarray(self.lb, dtype=float).reshape(-1)
            if self.lb.shape != (self.dim,):
                raise ValueError("lb doit être de taille (dim,).")
        if self.ub is not None:
            self.ub = np.asarray(self.ub, dtype=float).reshape(-1)
            if self.ub.shape != (self.dim,):
                raise ValueError("ub doit être de taille (dim,).")
        if self.lb is not None and self.ub is not None:
            if not np.all(self.lb < self.ub):
                raise ValueError("Il faut lb < ub composante par composante.")

    def _add_eval(self, x: np.ndarray, y: float, meta: Optional[dict[str, Any]] = None) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape != (self.dim,):
            raise ValueError(f"x doit être de shape ({self.dim},).")
        self._X.append(x)
        self._y.append(float(y))
        self._meta.append(meta or {})

    def add(self, x, y=None, fun: Optional[Objective] = None, meta=None,
            tol: float = 1e-10, max_size: Optional[int] = 1000) -> None:
        X = np.asarray(x, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.dim:
            raise ValueError(f"x doit avoir {self.dim} colonnes.")

        if y is None:
            if fun is None:
                raise ValueError("Fournir y ou fun.")
            for xi in X:
                if not self._is_duplicate(xi, tol):
                    self._add_eval(xi, fun(xi), meta=meta)
        else:
            yy = np.asarray(y, dtype=float).reshape(-1)
            if yy.size == 1 and X.shape[0] > 1:
                yy = np.full((X.shape[0],), float(yy[0]))
            if yy.shape != (X.shape[0],):
                raise ValueError("y doit être scalaire ou de taille (m,).")
            for xi, yi in zip(X, yy):
                if not self._is_duplicate(xi, tol):
                    self._add_eval(xi, yi, meta=meta)

        if max_size is not None:
            self._prune_by_density(max_size)

    def _is_duplicate(self, xi: np.ndarray, tol: float) -> bool:
        """
        Vérifie si xi est déjà présent dans le DOE (à une tolérance tol près).
        Utilise une recherche vectorisée (rapide même sans KD-tree).
        """
        if not self._X:
            return False
        X_arr = np.asarray(self._X, dtype=float)
        dists = np.linalg.norm(X_arr - xi[None, :], axis=1)
        return bool(np.any(dists < tol))
    
    def _prune_by_density(self, max_size: int) -> None:
        """
        Retire des points en excès avec une probabilité inversement
        proportionnelle à la distance au plus proche voisin. Après chaque
        suppression, seuls les points dont le plus proche voisin a été
        supprimé voient leur distance/probabilité recalculée (les autres
        restent valides car leur NN n'a pas changé).
        Le meilleur point courant (min y) est toujours protégé.
        """
        from scipy.spatial import cKDTree

        n = len(self._X)
        if n <= max_size:
            return
        n_to_remove = n - max_size

        X_arr = np.asarray(self._X, dtype=float)
        y_arr = np.asarray(self._y, dtype=float)
        best_idx = int(np.argmin(y_arr))

        alive = np.ones(n, dtype=bool)  # masque des points encore présents
        eps = 1e-12

        def compute_nn(mask_idx):
            """Recalcule (index du NN, distance au NN) pour les indices donnés,
            en cherchant parmi tous les points encore vivants."""
            alive_idx = np.where(alive)[0]
            tree = cKDTree(X_arr[alive_idx])
            # k=2 pour exclure soi-même
            dists, idx_local = tree.query(X_arr[mask_idx], k=2)
            nn_local = idx_local[:, 1]
            nn_dist = dists[:, 1]
            nn_global = alive_idx[nn_local]
            return nn_global, nn_dist

        alive_idx0 = np.where(alive)[0]
        nn_of = np.full(n, -1, dtype=int)      # nn_of[i] = index global du NN de i
        nn_dist = np.full(n, np.inf)           # distance au NN

        nn_g, nn_d = compute_nn(alive_idx0)
        nn_of[alive_idx0] = nn_g
        nn_dist[alive_idx0] = nn_d

        removed_count = 0
        while removed_count < n_to_remove:
            alive_idx = np.where(alive)[0]

            weights = 1.0 / (nn_dist[alive_idx] + eps)
            # protège le meilleur point
            weights[alive_idx == best_idx] = 0.0

            total_w = weights.sum()
            if total_w <= 0:
                # sécurité : ne devrait arriver que si un seul point reste protégé
                break

            probs = weights / total_w
            chosen_local = np.random.choice(len(alive_idx), p=probs)
            chosen_global = alive_idx[chosen_local]

            alive[chosen_global] = False
            removed_count += 1

            # Recalcul uniquement pour les points dont le NN vient d'être supprimé
            affected = np.where((nn_of == chosen_global) & alive)[0]
            if affected.size > 0 and alive.sum() > 1:
                nn_g, nn_d = compute_nn(affected)
                nn_of[affected] = nn_g
                nn_dist[affected] = nn_d

        keep_idx = np.where(alive)[0]
        self._X = [self._X[i] for i in keep_idx]
        self._y = [self._y[i] for i in keep_idx]
        if hasattr(self, "_meta") and self._meta is not None:
            self._meta = [self._meta[i] for i in keep_idx]

    def ingest(self, doe_like: Any, meta=None) -> None:
        if isinstance(doe_like, Mapping):
            X = doe_like.get("x", None)
            y = doe_like.get("y", None)
            if X is None or y is None:
                raise ValueError("Dict DOE attendu avec clés 'x' et 'y'.")
            self.add(X, y, meta=meta)
            return

        if isinstance(doe_like, tuple) and len(doe_like) == 2:
            X, y = doe_like
            self.add(X, y, meta=meta)
            return

        raise TypeError("Format DOE non reconnu pour ingest().")

    #@property
    def init_from_method(self, N_points: int, method: str, fun: Objective, meta=None) -> None:
        if self.lb is None or self.ub is None:
            raise ValueError("Il faut lb et ub pour init_from_method().")

        method = method.lower()
        if method == "uniform":
            X = self.rng.uniform(self.lb, self.ub, size=(N_points, self.dim))
        elif method == "lhs":
            X01 = self._lhs_unit(N_points)
            X = self.lb + (self.ub - self.lb) * X01
        else:
            raise ValueError("method doit être 'uniform' ou 'lhs'.")

        self.add(X, fun=fun, meta=(meta or {"source": method}))

    def _lhs_unit(self, n: int) -> np.ndarray:
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.rng)
            return np.asarray(sampler.random(n=n), dtype=float)
        except Exception:
            cut = np.linspace(0.0, 1.0, n + 1)
            u = self.rng.random((n, self.dim))
            X = np.empty((n, self.dim), dtype=float)
            for j in range(self.dim):
                a = cut[:n]
                b = cut[1:]
                pts = a + (b - a) * u[:, j]
                self.rng.shuffle(pts)
                X[:, j] = pts
            return X

    def Xy(self):
        X = np.asarray(self._X, dtype=float) if self._X else np.empty((0, self.dim), dtype=float)
        y = np.asarray(self._y, dtype=float) if self._y else np.empty((0,), dtype=float)
        return X, y

    def best(self):
        if not self._y:
            raise ValueError("DOE vide.")
        i = int(np.argmin(self._y))
        return self._X[i].copy(), float(self._y[i])

    def __len__(self):
        return len(self._y)