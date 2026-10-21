"""Options of GCO: specification (type, default, allowed values, description) and validation."""

OPTIONS_SPEC = {
    # ------------------------------------------------------------------ outputs
    "file_path": {
        "type": str,
        "default": "log_GCO.txt",
        "desc": "Path to the logfile where optimization successes are recorded.",
    },
    "output_columns": {
        "type": list,
        "default": ["BBE", "OBJ"],
        "choices": {"BBE", "OBJ", "TIME"},
        "desc": ("Columns written in the logfile, in order: 'BBE' (cumulative number of black-box evaluations), 'OBJ' (objective value), "
                 "'TIME' (wall-clock seconds since the start of the optimization). Example: ['BBE', 'OBJ', 'TIME']."),
    },
    # ------------------------------------------------------------------ main parameters
    "subspace_dimension": {
        "type": int,
        "default": None,  # mandatory
        "desc": "Dimension p of the anchored subspaces, 1 < p < n.",
    },
    "contraction": {
        "type": float,
        "default": 0.5,
        "desc": "Factor beta of the step-size update: sigma_k / beta after a successful call, sigma_k * beta after a failed one.",
    },
    "gamma": {
        "type": float,
        "default": 0.9,
        "desc": "Admissibility threshold: a subspace is admissible if its distance to every visited extended variable is >= gamma * h_k.",
    },
    "min_termination_scale_parameter": {
        "type": float,
        "default": 1e-9,
        "desc": "GCO stops when sigma_k (minimal frame size given to the suboptimizer) falls below this value.",
    },
    "ratio_percent": {
        "type": float,
        "default": 0.5,
        "desc": "Unused (kept so that existing scripts passing it still run).",
    },
    # ------------------------------------------------------------------ choice of the subspace
    "subspace_selection": {
        "type": str,
        "default": "grad_PLS",
        # valid names = keys of SUBSPACE_METHODS (subspace_methods.py), checked in GCO.__init__
        "desc": ("Method proposing the subspace (see subspace_methods.py): 'grad_PLS', 'PLS', 'Local_PLS', 'random', 'grad_random', "
                 "or a method added with GCO.register_subspace_method. random / PLS / Local_PLS evaluate no point, grad_PLS / grad_random "
                 "evaluate n points (forward differences, counted in the budget) every grad_every-th call. "
                 "'mixed' is the former name of Local_PLS alternated with random (see mixed_random_period)."),
    },
    "mixed_period": {
        "type": int,
        "default": 4,
        "desc": ("Alternation: every mixed_period-th call (k, 2k, 3k, ...) uses mixed_method instead of subspace_selection (0 = never). "
                 "Default: grad_PLS, grad_PLS, grad_PLS, random, ..."),
    },
    "mixed_method": {
        "type": str,
        "default": "random",
        "desc": "Second subspace method of the alternation.",
    },
    "grad_every": {
        "type": int,
        "default": 1,
        "desc": "grad_PLS / grad_random: the gradient is estimated every grad_every-th call; the other calls use PLS / random alone.",
    },
    "grad_fd_step": {
        "type": float,
        "default": 1e-6,
        "desc": "grad_PLS / grad_random: forward-difference step in unit-box coordinates (backward step at the upper bound).",
    },
    "fixed_columns_draws": {
        "type": int,
        "default": 20,
        "desc": ("When a method pins its first columns (the gradient of grad_PLS) and the subspace is not admissible: number of random "
                 "completions orthogonal to the pinned columns tried before the admissible projection."),
    },
    "local_pls_factor": {
        "type": float,
        "default": 2.0,
        "desc": ("Local_PLS: while the DOE holds at most K = ceil(factor*(n-1)) points the PLS uses all of them, then only the K points "
                 "closest to x_best. 0 = always all points (global PLS)."),
    },
    "mixed_random_period": {
        "type": int,
        "default": 2,
        "desc": ("Former option, only with subspace_selection='mixed': becomes subspace_selection='Local_PLS' "
                 "(local_pls_factor = mixed_local_factor), mixed_period = mixed_random_period, mixed_method='random_orth'."),
    },
    "mixed_local_factor": {
        "type": float,
        "default": 0.0,
        "desc": "Former option, only with subspace_selection='mixed' (see mixed_random_period). 0 = global PLS.",
    },
    # ------------------------------------------------------------------ exclusion radius and admissibility
    "exclusion_radius_method": {
        "type": str,
        "default": "gradient",
        "choices": {"gradient", "monte_carlo", "riemannian"},
        "desc": ("Estimate of the exclusion radius h_k = max_S min_w d((x_best, S), w): 'gradient' = projected gradient ascent on the "
                 "Grassmannian from the best of a few random candidates; 'monte_carlo' = best of random (Haar) subspaces; "
                 "'riemannian' = pymanopt (slow)."),
    },
    "exclusion_radius_gradient_init": {
        "type": int,
        "default": 20,
        "desc": "'gradient' exclusion radius: number of random candidates among which the ascent starts.",
    },
    "exclusion_radius_gradient_max_iter": {
        "type": int,
        "default": 15,
        "desc": "'gradient' exclusion radius: maximum number of ascent steps (also stops when a step gains less than 0.1 %).",
    },
    "exclusion_radius_n_samples": {
        "type": int,
        "default": 500,
        "desc": "'monte_carlo' exclusion radius: number of random subspaces.",
    },
    "admissible_projection_method": {
        "type": str,
        "default": "gradient",
        "choices": {"gradient", "monte_carlo", "riemannian"},
        "desc": ("Projection of an inadmissible subspace onto the admissible set: 'gradient' = moves S_k away from the violating extended "
                 "variables by projected gradient ascent, then back towards S_k by bisection; 'monte_carlo' = closest admissible among "
                 "random subspaces; 'riemannian' = pymanopt (slow). If nothing admissible is found, the maximiser S* of the exclusion "
                 "radius (admissible by construction) is used."),
    },
    "gradient_projection_max_iter": {
        "type": int,
        "default": 300,
        "desc": "'gradient' projection: maximum number of ascent steps (then Monte Carlo, then S*).",
    },
    "admissible_projection_n_samples": {
        "type": int,
        "default": 5000,
        "desc": "'monte_carlo' projection: number of random subspaces.",
    },
    "W_prune_by_anchor": {
        "type": bool,
        "default": True,
        "desc": ("Skip, in the distance computations, the visited extended variables whose anchor is too far from the current one to "
                 "be closer than the threshold (exact, see ExclusionMixin._restrict_W)."),
    },
    # ------------------------------------------------------------------ latent function
    "zonotope_projection": {
        "type": str,
        "default": "newton",
        "choices": {"newton", "slsqp"},
        "desc": ("Projection gamma_Z of a latent point onto the zonotope {x in [-1,1]^n : A^T x = s}: 'newton' = semismooth Newton "
                 "on the dual problem in dimension p (falls back on SLSQP if it does not conclude); 'slsqp' = scipy SLSQP in dimension n."),
    },
    "max_cache_point": {
        "type": int,
        "default": 900,  # below NOMAD's limit of 1000 lines
        "desc": "Maximum number of evaluated points given to the suboptimizer (NOMAD cache).",
    },
    "cache_selection_method": {
        "type": str,
        "default": "closest_subspace",
        "choices": {"random", "closest_subspace"},
        "desc": "Points kept when there are more than max_cache_point: 'closest_subspace' (smallest ||x - V V^T x||) or 'random'.",
    },
    "cache_projection_threshold": {
        "type": float,
        "default": 0.0025,
        "desc": ("Only the evaluated points with ||x - V V^T x|| <= threshold * sqrt(n) are given to the suboptimizer. "
                 "None disables the filter."),
    },
}


def init_options(user_opt, spec=OPTIONS_SPEC):
    """Validate the user options and complete them with the defaults of `spec`.
    Raises ValueError for an unknown option, a missing mandatory option, a wrong type or a value outside the allowed choices."""
    user_opt = {k.replace("covering_radius", "exclusion_radius"): v for k, v in user_opt.items()}  # former names
    unknown = set(user_opt) - set(spec)
    if unknown:
        raise ValueError(f"Unknown option(s): {sorted(unknown)}. Valid options are: {sorted(spec)}.")

    opt = {}
    for key, s in spec.items():
        value = user_opt.get(key, s["default"])
        if value is None and s["default"] is None:
            raise ValueError(f"Option '{key}' is mandatory and has no default. Description: {s['desc']}")
        if value is not None and not isinstance(value, s["type"]):
            raise ValueError(f"Option '{key}' expects type {s['type'].__name__}, got {type(value).__name__}.")
        if "choices" in s:
            if isinstance(value, list):
                if not value or any(v not in s["choices"] for v in value):
                    raise ValueError(f"Option '{key}' must be a non-empty list with entries in {sorted(s['choices'])}, got {value}.")
                value = list(value)
            elif value not in s["choices"]:
                raise ValueError(f"Option '{key}' must be one of {sorted(s['choices'])}, got '{value}'.")
        opt[key] = value
    return opt
