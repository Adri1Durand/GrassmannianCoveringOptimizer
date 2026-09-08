"""
GCO_subspace_dim_study.py
=========================
Study the optimal subspace dimension p* as a function of problem dimension n
for the GCO optimizer. For each function and each n, we identify p*(n) as the
subspace dimension yielding the best mean f_best over N_RUNS independent runs.

Early stopping: if mean_f_best(p_i) > mean_f_best(p_{i-1}), we stop exploring
larger subspace dimensions.

Results are saved as:
    - results/data/subspace_study_<function_name>.csv
    - results/figures/subspace_study_<function_name>.png
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from GCO import GCO, mads_suboptimizer
from utils.doe_class import DOE

dossier_parent = Path(__file__).resolve().parent.parent
sys.path.append(str(dossier_parent))
from Folder_of_test.test_function import *

# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------

N_RUNS = 4

N_VALUES = [10, 15, 23, 34, 51, 76, 114, 170, 256, 384]#, 577, 865]
P_VALUES = [3, 4, 6, 8, 10, 13, 17, 22, 28]

# N_VALUES = [10,15]
# P_VALUES = [3,4]

# FUNCTION_LIST = [rosenbrock, ackley]  # <-- add / remove functions here

FUNCTION_LIST = [rosenbrock, ackley, rastrigin, griewank, schwefel]  # <-- add / remove functions here
 
RESULTS_DATA_DIR = os.path.join("results", "data")
RESULTS_FIG_DIR = os.path.join("results", "figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget(n: int) -> int:
    return min(5000, 100 * n)


def _valid_p_values(n: int) -> list:
    """Return P_VALUES filtered to p < n - 1."""
    return [p for p in P_VALUES if p < n - 1 and p < _budget(n)] 


def _run_single(fun, n: int, p: int, seed: int) -> float:
    """Run GCO once and return f_best."""
    lb = np.full(n, -10.0)
    ub = np.full(n, 10.0)
    var = {"dim": n, "lower_bound": lb, "upper_bound": ub}

    rng = np.random.default_rng(seed)
    doe = DOE(n, lb=lb, ub=ub, seed=int(rng.integers(0, 2**31)))
    doe.init_from_method(N_points=n // 2, method="LHS", fun=fun)
    X, F = doe.Xy()

    gco = GCO(
        fun,
        vars_prop=var,
        opt={"subspace_dimension": p, "subspace_selection": "PLS"},
    )
    _, f_best = gco.run_optim(
        X0=X, F0=F, suboptimizer=mads_suboptimizer, budget=_budget(n)
    )
    return float(f_best)


def _mean_f_best(fun, n: int, p: int) -> float:
    """Return mean f_best over N_RUNS independent runs."""
    values = [_run_single(fun, n, p, seed) for seed in range(N_RUNS)]
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Core study
# ---------------------------------------------------------------------------


def study_function(fun) -> pd.DataFrame:
    """
    For each n in N_VALUES, find p*(n) with early stopping.

    Early stop condition: if mean_f_best(p_i) > mean_f_best(p_{i-1})
                                               AND
                             mean_f_best(p_i) > mean_f_best(p_{i-2})
    then stop and do not evaluate p_{i+1}.

    Returns a DataFrame with columns: n, p, mean_f_best, p_star.
    """
    records = []

    for n in N_VALUES:
        valid_ps = _valid_p_values(n)
        if len(valid_ps) == 0:
            print(f"  n={n}: no valid p, skipping.")
            continue

        print(f"\n  n={n} | budget={_budget(n)} | valid p={valid_ps}")

        mean_history = []  # list of (p, mean_val) in evaluation order

        for idx, p in enumerate(valid_ps):
            mean_val = _mean_f_best(fun, n, p)
            mean_history.append((p, mean_val))
            print(f"    p={p:3d} | mean_f_best={mean_val:.6e}")

            # Early stopping: need at least 3 evaluations
            if len(mean_history) >= 3:
                p_i_val   = mean_history[-1][1]
                p_im1_val = mean_history[-2][1]
                p_im2_val = mean_history[-3][1]

                if p_i_val > p_im1_val and p_i_val > p_im2_val:
                    print(
                        f"    Early stop at p={p}: "
                        f"mean_f_best worse than both p_{mean_history[-2][0]} "
                        f"and p_{mean_history[-3][0]}."
                    )
                    break

        # p*(n): p with the lowest mean_f_best among evaluated ones
        best_p, best_val = min(mean_history, key=lambda x: x[1])
        print(f"  => p*(n={n}) = {best_p} | mean_f_best = {best_val:.6e}")

        for p, mean_val in mean_history:
            records.append(
                {
                    "n": n,
                    "p": p,
                    "mean_f_best": mean_val,
                    "p_star": (p == best_p),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_data(df: pd.DataFrame, fun_name: str) -> None:
    os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DATA_DIR, f"subspace_study_{fun_name}.csv")
    df.to_csv(path, index=False)
    print(f"  Data saved to {path}")


def save_figure(df: pd.DataFrame, fun_name: str) -> None:
    os.makedirs(RESULTS_FIG_DIR, exist_ok=True)

    df_star = df[df["p_star"]].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        df_star["n"],
        df_star["p"],
        marker="o",
        linewidth=1.5,
        color="steelblue",
        label=r"$p^*(n)$",
    )

    ax.set_xlabel("Problem dimension $n$", fontsize=13)
    ax.set_ylabel("Optimal subspace dimension $p^*$", fontsize=13)
    ax.set_title(
        f"Optimal subspace dimension vs problem dimension\n({fun_name})",
        fontsize=13,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(df_star["n"].tolist())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    fig.tight_layout()
    path = os.path.join(RESULTS_FIG_DIR, f"subspace_study_{fun_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    for fun in FUNCTION_LIST:
        fun_name = fun.__name__
        print(f"\n{'='*60}")
        print(f"Function: {fun_name}")
        print(f"{'='*60}")

        df = study_function(fun)
        save_data(df, fun_name)
        save_figure(df, fun_name)

    print("\nDone.")


if __name__ == "__main__":
    main()