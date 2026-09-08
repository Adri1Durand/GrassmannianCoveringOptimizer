import os
import sys
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

from GCO import GCO, mads_suboptimizer
from utils.doe_class import DOE

dossier_parent = Path(__file__).resolve().parent.parent
sys.path.append(str(dossier_parent))
from Folder_of_test.test_function import *


# =============================================================================
# Problem setup
# =============================================================================
n = 200
lb = np.array([-10] * n)
ub = np.array([10] * n)

var = {"dim": n, "lower_bound": lb, "upper_bound": ub}
function = ackley

initial_doe = DOE(n, lb=lb, ub=ub)
initial_doe.init_from_method(N_points=n // 2, method="LHS", fun=function)
X, F = initial_doe.Xy()
print(f"Start value : {min(F)}")
# =============================================================================
# Parameter grid
# =============================================================================
ratio_list       = [1.0, 0.5, 2.0]
gamma_list       = [0.99, 0.95, 0.9, 0.8]
contraction_list = [0.9, 0.3]

# =============================================================================
# Grid search
# =============================================================================
# Raw results storage: raw_results[ratio][gamma][contraction] = f_best
raw_results: dict = {}

for ratio, gamma, contraction in product(ratio_list, gamma_list, contraction_list):

    _gco = GCO(
        function,
        vars_prop=var,
        opt={
            "subspace_dimension": 8,
            "subspace_selection": "PLS",
            "ratio_percent": ratio,
            "gamma":       gamma,
            "contraction": contraction,
        },
    )
    x_best, f_best = _gco.run_optim(
        X0=X, F0=F, suboptimizer=mads_suboptimizer, budget=2000)
    print(f"Done for {ratio,gamma,contraction} we get f = {f_best}")
    raw_results.setdefault(ratio, {}).setdefault(gamma, {})[contraction] = f_best


# =============================================================================
# Build MultiIndex DataFrame  (rows = contraction, columns = (ratio, gamma))
# =============================================================================
def build_results_dataframe(
    raw: dict,
    ratio_list: list,
    gamma_list: list,
    contraction_list: list,
) -> pd.DataFrame:
    """
    Convert the nested results dictionary into a pandas DataFrame whose
    structure mirrors the target table:

        columns : MultiIndex  (ratio  [level-0],  gamma  [level-1])
        index   : contraction values

    Parameters
    ----------
    raw : dict
        Nested dict raw[ratio][gamma][contraction] = f_best.
    ratio_list, gamma_list, contraction_list : list
        Ordered parameter lists used during the grid search.

    Returns
    -------
    pd.DataFrame
    """
    col_index = pd.MultiIndex.from_product(
        [ratio_list, gamma_list], names=["ratio", "gamma"]
    )
    df = pd.DataFrame(index=contraction_list, columns=col_index, dtype=float)
    df.index.name = "contraction"

    for ratio in ratio_list:
        for gamma in gamma_list:
            for contraction in contraction_list:
                df.loc[contraction, (ratio, gamma)] = raw[ratio][gamma][contraction]

    return df


results_df = build_results_dataframe(
    raw_results, ratio_list, gamma_list, contraction_list
)

# =============================================================================
# Save results
# =============================================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "grid_search_results.csv")
results_df.to_csv(CSV_PATH)

print(results_df.to_string())
print(f"\nResults saved to '{CSV_PATH}'.")
