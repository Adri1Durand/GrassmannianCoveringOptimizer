# utils/domain.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class BoxScaler:
    """
    Affine map between user box [lb, ub] and normalized box [-1, 1]^n.

    x in [lb, ub]  <->  z in [-1, 1]^n
    """
    lb: np.ndarray
    ub: np.ndarray

    def __post_init__(self) -> None:
        lb = np.asarray(self.lb, dtype=float).reshape(-1)
        ub = np.asarray(self.ub, dtype=float).reshape(-1)
        if lb.shape != ub.shape:
            raise ValueError("lb and ub must have the same shape.")
        if lb.size == 0:
            raise ValueError("Empty bounds.")
        if not np.all(lb < ub):
            raise ValueError("Require lb < ub componentwise.")

        object.__setattr__(self, "lb", lb)
        object.__setattr__(self, "ub", ub)
        object.__setattr__(self, "_width", ub - lb)

    @property
    def dim(self) -> int:
        return int(self.lb.size)

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 2.0 * (x - self.lb) / self._width - 1.0

    def from_unit(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.lb + 0.5 * (z + 1.0) * self._width

    def clip_to_user(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(x, self.lb, self.ub)

    def clip_to_unit(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.clip(z, -1.0, 1.0)
