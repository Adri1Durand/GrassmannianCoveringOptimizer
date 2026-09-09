"""
extended_variable.py
--------------------
Defines the ExtendedVariable class used in the GCO algorithm.

An extended variable omega = (x, S) encodes a point x in the original space
together with a matrix S representing an anchored subspace in the perpendicular
space of the anchor basis at x.

References
----------
See the GCO algorithm description.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class GrassmannMetric(Enum):
    """Available metrics on the Grassmann manifold."""
    CHORDAL = "chordal"
    CANONICAL = "canonical"


@dataclass
class ExtendedVariable:
    """
    Extended variable used in the GCO subspace optimization algorithm.

    An extended variable omega = (x, S) where:
      - x is a point in the original (scaled) search space R^n.
      - S is an (n-1) x (p-1) orthonormal matrix whose columns span a
        (p-1)-dimensional subspace of R^(n-1), i.e. a point on Gr(p-1, n-1).

    The Grassmann point is defined in the perpendicular space of the anchor
    basis at x, following the map Phi^{-1}_x.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Point in the original search space.
    S : ndarray of shape (n-1, p-1)
        Matrix spanning the anchored subspace. Will be orthonormalized
        automatically via QR decomposition.
    metric : GrassmannMetric
        Grassmann distance to use. Default is chordal.
    """

    x: np.ndarray
    S: np.ndarray
    #metric: GrassmannMetric = GrassmannMetric.CHORDAL

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.S = np.asarray(self.S, dtype=float)

        if self.S.ndim != 2:
            raise ValueError(f"S must be a 2D matrix, got shape {self.S.shape}.")

        n_minus_1, p_minus_1 = self.S.shape

        if self.x.shape[0] != n_minus_1 + 1:
            raise ValueError(
                f"Inconsistent dimensions: x has dimension {self.x.shape[0]} "
                f"but S has {n_minus_1} rows, expected n-1 = {self.x.shape[0] - 1}."
            )
        if p_minus_1 >= n_minus_1:
            raise ValueError(
                f"S must have fewer columns than rows: got shape {self.S.shape}."
            )
        if p_minus_1 < 1:
            raise ValueError(
                f"S must have at least 1 column, got {p_minus_1}."
            )

        # Orthonormalize columns of S via QR
        Q, _ = np.linalg.qr(self.S)
        self.S = Q[:, :p_minus_1]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Dimension of the original search space."""
        return self.x.shape[0]

    @property
    def p(self) -> int:
        """Dimension of the latent subspace (including the anchor direction)."""
        return self.S.shape[1] + 1

    
    # ------------------------------------------------------------------
    # Grassmann distances (static methods)
    # ------------------------------------------------------------------

    @staticmethod
    def _grassmann_chordal(S1: np.ndarray, S2: np.ndarray) -> float:
        """
        Chordal distance between two Grassmann points represented by
        orthonormal matrices S1 and S2.

        d_chordal(S1, S2) = || S1 S1^T - S2 S2^T ||_F / sqrt(2)

        Parameters
        ----------
        S1 : ndarray of shape (n-1, p-1)
        S2 : ndarray of shape (n-1, p-1)

        Returns
        -------
        float
        """
        P1 = S1 @ S1.T
        P2 = S2 @ S2.T
        return np.linalg.norm(P1 - P2, ord='fro') / np.sqrt(2)

    @staticmethod
    def _grassmann_canonical(S1: np.ndarray, S2: np.ndarray) -> float:
        """
        Canonical (geodesic) distance between two Grassmann points represented
        by orthonormal matrices S1 and S2.

        d_canonical(S1, S2) = || theta ||_2

        where theta are the principal angles between the two subspaces.

        Parameters
        ----------
        S1 : ndarray of shape (n-1, p-1)
        S2 : ndarray of shape (n-1, p-1)

        Returns
        -------
        float
        """
        M = S1.T @ S2
        sigma = np.linalg.svd(M, compute_uv=False)
        sigma = np.clip(sigma, -1.0, 1.0)
        angles = np.arccos(sigma)
        return np.linalg.norm(angles)

    def _grassmann_distance(self, other: "ExtendedVariable", metric: GrassmannMetric) -> float:
        if metric == GrassmannMetric.CHORDAL:
            return self._grassmann_chordal(self.S, other.S)
        elif metric == GrassmannMetric.CANONICAL:
            return self._grassmann_canonical(self.S, other.S)
        else:
            raise ValueError(f"Unknown metric: {metric}")


    # ------------------------------------------------------------------
    # Extended distance
    # ------------------------------------------------------------------

    def distance(self, other: "ExtendedVariable", r: float, metric: GrassmannMetric = GrassmannMetric.CHORDAL) -> float:
        """
        Extended distance between two extended variables.

        d_Omega(w1, w2) = sqrt(r ||x1 - x2||^2 + (1-r) d_G(S1, S2)^2)

        Parameters
        ----------
        other : ExtendedVariable
        r : float
            Scaling constant balancing x-distance and Grassmann distance.
        metric : GrassmannMetric
            Grassmann metric to use. Default is chordal.

        Returns
        -------
        float
        """
        if self.n != other.n or self.p != other.p:
            raise ValueError(
                f"Cannot compute distance between ExtendedVariables with "
                f"different dimensions: ({self.n}, {self.p}) vs ({other.n}, {other.p})."
            )
        if r <= 0:
            raise ValueError(f"r must be strictly positive, got r={r}.")

        dx = np.linalg.norm(self.x - other.x)
        dG = self._grassmann_distance(other, metric=metric)
        return np.sqrt(r * dx ** 2 + (1-r)* dG ** 2)


    def distance_all(self, other: "ExtendedVariable", r: float, metric: GrassmannMetric = GrassmannMetric.CHORDAL) -> float:
        """
        Extended distance between two extended variables.

        d_Omega(w1, w2) = sqrt( r * ||x1 - x2||^2 + (r-1) d_G(S1, S2)^2)

        Parameters
        ----------
        other : ExtendedVariable
        r : float
            Scaling constant balancing x-distance and Grassmann distance.
        metric : GrassmannMetric
            Grassmann metric to use. Default is chordal.

        Returns
        -------
        float
        """
        if self.n != other.n or self.p != other.p:
            raise ValueError(
                f"Cannot compute distance between ExtendedVariables with "
                f"different dimensions: ({self.n}, {self.p}) vs ({other.n}, {other.p})."
            )
        if r <= 0:
            raise ValueError(f"r must be strictly positive, got r={r}.")

        dx = np.linalg.norm(self.x - other.x)
        dG = self._grassmann_distance(other, metric=metric)
        dO = np.sqrt(r*dx ** 2 + (1-r)*dG ** 2)
        print(f"Grassmannian distance : {dG}")
        print(f"Euclidian distance : {dx}")
        print(f"Extended distance {dO}")
        return dO

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExtendedVariable(n={self.n}, p={self.p}, "
            #f"metric={self.metric.value},\n"
            f"  x={self.x},\n"
            f"  S shape={self.S.shape})"
        )

if __name__ == "__main__":
    import numpy as np
    #from extended_variable import ExtendedVariable, GrassmannMetric

    # ------------------------------------------------------------------
    # Example: two extended variables in R^5, with p=3 (subspace dim)
    # n=5, so S has shape (n-1, p-1) = (4, 2)
    # ------------------------------------------------------------------

    n = 5
    p = 3

    # First extended variable
    x1 = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    S1_raw = np.random.randn(n - 1, p - 1)  # will be orthonormalized automatically
    omega1 = ExtendedVariable(x=x1, S=S1_raw)

    # Second extended variable
    x2 = np.array([0.4, 0.2, 0.9, 0.1, 0.6])
    S2_raw = np.random.randn(n - 1, p - 1)
    omega2 = ExtendedVariable(x=x2, S=S2_raw)

    print("omega1:", omega1)
    print()
    print("omega2:", omega2)
    print()

    # ------------------------------------------------------------------
    # Compute extended distance with a given r
    # ------------------------------------------------------------------
    r = 1.0
    d = omega1.distance(omega2, r=r, metric=GrassmannMetric.CHORDAL)
    print(f"Extended distance d_Omega(omega1, omega2) with r={r}: {d:.6f} (chordal)")

    d = omega1.distance(omega2, r=r, metric=GrassmannMetric.CANONICAL)
    print(f"Extended distance d_Omega(omega1, omega2) with r={r}: {d:.6f} (canonic)")
