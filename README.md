# GCO : Grassmannian Covering Optimizer

Optimisation sans dérivées (DFO) de fonctions boîte noire en grande dimension `n`, par recherches successives dans des sous-espaces de petite
dimension `p` ancrés au meilleur point courant. Chaque sous-espace est optimisé par un sous-optimiseur (MADS/PyNomad, TREGO ou un autre), et
GCO impose que les sous-espaces successifs soient **suffisamment différents** les uns des autres (condition d'exclusion sur la Grassmannienne).

## 1. Utilisation

```python
import numpy as np
from GCO import GCO, mads_suboptimizer
from GCO.utils.doe_class import DOE

n = 100
lb, ub = np.full(n, -10.0), np.full(n, 10.0)
f = lambda x: float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

gco = GCO(f, vars_prop={"dim": n, "lower_bound": lb, "upper_bound": ub},
          opt={"subspace_dimension": 10, "file_path": "log.txt"})  # défauts : grad_PLS, mixed_period=4, projection par gradient
doe = DOE(n, lb=lb, ub=ub)
doe.init_from_method(N_points=n // 2, method="LHS", fun=f)
X0, F0 = doe.Xy()
x_best, f_best = gco.run_optim(X0=X0, F0=F0, suboptimizer=mads_suboptimizer, budget=5000)
```

`budget` est le nombre total d'évaluations de la boîte noire (le DOE initial compte). Le fichier `file_path` enregistre chaque amélioration
(colonnes `BBE`, `OBJ`, `TIME` au choix, option `output_columns`). Avec PyNomad en dimension ≥ 50, ajouter le paramètre NOMAD
`EVAL_QUEUE_SORT DIR_LAST_SUCCESS` (déjà utilisé par les scripts de test, pas dans `mads_suboptimizer`).

Exemple complet exécutable (Rosenbrock décalé, n = 30) : `python -m GCO.GCO` depuis la racine du dépôt (voir la fin de `GCO/GCO.py`).

**Organisation du code** (`GCO/`) :

| Fichier | Contenu |
|---|---|
| `GCO.py` | classe `GCO` : boucle principale `run_optim`, sous-espace ancré, base de l'ancre, journal ; exemple dans `__main__` |
| `options.py` | toutes les options (type, défaut, description) et leur validation |
| `subspace_methods.py` | méthodes de sous-espace (PLS, random, Local_PLS, grad_PLS, …), contexte `SubspaceContext`, registre |
| `exclusion.py` | distances aux variables étendues visitées `W`, rayon d'exclusion, admissibilité, projection sur l'admissible |
| `latent.py` | fonction latente, projection sur la zonotope (Newton dual / SLSQP), DOE et bornes latentes |
| `suboptimizers.py` | `mads_suboptimizer`, `trego_suboptimizer` |
| `riemannian.py` | variantes pymanopt (lentes, déconseillées) |
| `utils/` | DOE, variable étendue, mise à l'échelle de la boîte, fichiers de cache NOMAD, TREGO |

## 2. Principe de l'algorithme

Le domaine est ramené à la boîte `[-1, 1]^n`. À chaque itération `k` :

1. **Ancre.** `x_best` est le meilleur point évalué. On construit une base `B` (n × (n−1)) de l'hyperplan orthogonal à `x_best`.
2. **Sous-espace.** On choisit `S_k` dans la Grassmannienne `Gr(p−1, n−1)` (matrice (n−1)×(p−1) à colonnes orthonormées) par une
   **méthode de réduction de dimension** (section 3). Le sous-espace de travail de dimension `p` est `V_k = span{x_best, B S_k}`.
3. **Variable étendue.** `ω_k = (x_best, S_k)`. La liste `W` contient toutes les variables étendues déjà visitées.
   La distance entre deux variables étendues est
   `d(ω, ω')² = r‖x − x'‖² + (1 − r)·d_G(S, S')²`, avec la distance chordale `d_G² = (p−1) − ‖SᵀS'‖_F²`
   et `r = 2n(n−1) / (2n(n−1) + 3(p−1)(n−2))`.
4. **Rayon d'exclusion** `h_k = max_S min_{w∈W} d((x_best, S), w)` : la distance du sous-espace le plus isolé à l'historique `W`
   (section 4). `h_0` est une valeur initiale tant que `W` est vide.
5. **Admissibilité.** `ω_k` est admissible si `min_w d(ω_k, w) ≥ γ·h_k` (γ = `gamma`, 0,9 par défaut). Sinon il est **projeté** sur l'admissible :
   le sous-espace admissible le plus proche de `S_k` (section 5).
6. **Fonction latente.** On optimise `f_Z(s) = f(γ_Z(s))` pour `s ∈ R^p` borné, où `γ_Z` projette sur la zonotope `{x ∈ [-1,1]^n : Vᵀx = s}`
   (SLSQP, seulement si `V s` sort de la boîte ; sinon pénalité en dehors). Le sous-optimiseur part du meilleur point du DOE restreint à `V_k`
   (les points déjà évalués proches de `V_k` sont réutilisés, seuil `cache_projection_threshold`, au plus `max_cache_point` points).
7. **Pas.** Le sous-optimiseur reçoit la taille de maille minimale `σ_k`. Succès (meilleur point amélioré) : `σ_k ← min(σ_k / β, C·h_k)` et l'ancre change.
   Échec : `σ_k ← min(σ_k·β, C·h_k)` et l'ancre est conservée. β = `contraction` (0,5 par défaut).
8. **Arrêt.** Budget épuisé, ou `σ_k < min_termination_scale_parameter` (1e-9). Le budget du sous-optimiseur est le budget restant : c'est `σ_k` qui
   arrête normalement l'algorithme, comme dans la version de référence.

Le mécanisme d'exclusion oblige les sous-espaces successifs à rester éloignés de ceux déjà visités : la Grassmannienne est explorée de manière
de plus en plus fine à mesure que `h_k` diminue.

## 3. Méthodes de réduction de dimension (`subspace_selection`)

Elles proposent `S_k` avant projection sur l'admissible. Les données sont les points évalués `X` projetés dans l'hyperplan perpendiculaire à l'ancre
(`X Bᵀ`) et leurs valeurs `y`. GCO combine donc deux objets qui peuvent **tous deux évaluer la boîte noire** : le sous-optimiseur et la méthode de
sous-espace. Une méthode de sous-espace peut évaluer un nombre fini de nouveaux points pour construire `S_k` ; ces points sont comptés dans le budget,
ajoutés au DOE, journalisés et pris en compte pour le meilleur point, exactement comme ceux du sous-optimiseur.

| Valeur | Principe |
|---|---|
| `"random"` | `S_k` tiré uniformément (mesure de Haar : QR d'une matrice gaussienne). Aucune information sur `f`. |
| `"PLS"` | Régression PLS (moindres carrés partiels) de `y` sur `X`, avec `p−1` composantes ; les directions `x_rotations_` (orthonormalisées) sont celles de plus forte covariance avec `f`. Utilise **tous** les points du DOE (au plus 1000). La base est complétée aléatoirement si le rang est déficient. |
| `"Local_PLS"` | Même PLS, mais sur les points proches de l'ancre. Tant que le DOE a au plus `K = ⌈local_pls_factor·(n−1)⌉` points (au moins `p`), tous sont utilisés ; au-delà, seulement les `K` plus proches de `x_best` (distance euclidienne en espace complet). `local_pls_factor = 2` par défaut ; `0` revient à la PLS globale. |
| `"grad_PLS"` (défaut) | Gradient par différences finies avant en `x_best` (**n évaluations**, pas `grad_fd_step` = 1e-6 en coordonnées de la boîte unité, pas arrière à la borne supérieure ; `f(x_best)` est déjà connu). La première colonne de `S_k` est `Bᵀ(−g)` normalisé ; les `p−2` autres sont les directions PLS rendues orthogonales. Un appel sur `grad_every` seulement (défaut 1), les autres appels = PLS seule. Si le sous-espace n'est pas admissible, GCO garde la colonne gradient et tire jusqu'à `fixed_columns_draws` (20) complétions aléatoires orthogonales ; sinon projection habituelle (la colonne est alors perdue pour cet appel). |
| `"grad_random"` | Idem avec une complétion aléatoire au lieu de PLS. |

Nombre de points évalués par la méthode : 0 pour `random`, `PLS`, `Local_PLS` ; `n` (un appel sur `grad_every`) pour `grad_PLS` et `grad_random`.
Un gradient coûte donc ≈ 1 % du budget `100(n+1)`.

Pourquoi `grad_PLS` par défaut (avec `mixed_period = 4`, projection par gradient) : sur les problèmes S2MPJ décalés (tests internes, 2 graines),
c'est la meilleure variante de GCO seul mesurée. Par rapport à mixed4 + PLS : −0,58 décade d'écart final (n = 62..191, meilleur sur 88 % des
instances, p < 0,001) et −0,24 décade (n = 10..50) ; à 100(n+1) évaluations sur n = 62..191, τ = 1e-1/1e-3/1e-5/1e-7 résolus 0,85/0,29/0,21/0,09
contre 0,62/0,12/0/0. Coût : les sondes prennent en médiane 19 % du budget (n = 62..191) et 12 % (n = 10..50). Pour une boîte noire chère,
`grad_every = 5` donne le même résultat médian (différence appariée 0,00 décade) avec 4 fois moins de sondes (5 % / 3 %), mais perd sur quelques
problèmes intérieurs lisses (PENALTY2, LINVERSE). Pour retrouver l'ancien comportement : `subspace_selection = "PLS"`.

Ajouter une méthode : `GCO.register_subspace_method(nom, methode)` avec `methode(gco, ctx)` qui renvoie `S` (`(n−1) × (p−1)`) ou `(S, n_fixed)`
pour demander de conserver les `n_fixed` premières colonnes lors du rétablissement de l'admissibilité. `ctx` (classe `SubspaceContext`) fournit
`doe_perp`, `x_best`, `W`, `h_k`, `budget_left` et `evaluate(X)` (évalue les lignes de `X`, en coordonnées de la boîte unité, et renvoie leurs valeurs ;
moins de lignes si le budget est épuisé). Exemple :

```python
def deux_points(gco, ctx):
    F = ctx.evaluate(ctx.x_best + 0.01 * np.random.randn(2, gco.n))   # 2 évaluations comptées
    return np.random.randn(gco.n - 1, gco.p - 1)
GCO.register_subspace_method("deux_points", deux_points)
```

**Alternance de deux méthodes (option de GCO, pas une méthode de sous-espace).** `mixed_period = k` (0 = pas d'alternance, **défaut 4**) : les
appels `k, 2k, 3k, …` utilisent `mixed_method` (défaut `"random"`), les autres `subspace_selection`. Le défaut donne donc grad_PLS, grad_PLS, grad_PLS, random, … Exemple : `subspace_selection="grad_PLS"`,
`mixed_period=4` donne grad_PLS, grad_PLS, grad_PLS, random, … (seuls les appels grad_PLS évaluent des points). Ancienne écriture, toujours
acceptée et équivalente au bit près : `subspace_selection="mixed"` avec `mixed_random_period = k` et `mixed_local_factor`, traduite en
`subspace_selection="Local_PLS"` (`local_pls_factor = mixed_local_factor`, 0 = PLS globale), `mixed_period = k`, `mixed_method="random_orth"`.

Remarque : un sous-espace aléatoire de dimension supérieure à la dimension active d'une fonction à structure bas rang exacte contient déjà
l'optimum (argument de type REMBO). La PLS n'apporte alors rien de plus ; son intérêt dépend de la fonction. Dans les tests réalisés en
développement, PLS, PLS locale et aléatoire étaient équivalentes sur les fonctions de bas rang et classiques essayées (Rosenbrock, Rastrigin, Ackley, ellipsoïdes).

## 4. Rayon d'exclusion (`exclusion_radius_method`)

| Valeur | Principe |
|---|---|
| `"gradient"` (défaut) | Montée de gradient riemannienne sur `Gr(p−1, n−1)`. Départ : meilleur de `exclusion_radius_gradient_init` (20) candidats de Haar. Le min sur `W` est lissé par un soft-min `f_τ(S) = −τ ln Σ_w exp(−d_w(S)/τ)`, `τ = 0,05·d_min`. Gradient : `∇_S f_τ = Σ_w softmin_w · ∇_S d_w` avec `∇_S d_w = −(1−r)·S_w(S_wᵀS)/d_w`, projeté sur le plan tangent (`D ← D − S(SᵀD)`), rétraction QR, pas adaptatif (doublé sur succès, divisé par 2 sinon, accepté seulement si le vrai min augmente). Arrêt : `exclusion_radius_gradient_max_iter` itérations, gain relatif < 1e-3 ou gradient nul. Renvoie une **borne inférieure** de `h_k`. |
| `"monte_carlo"` | Max sur `exclusion_radius_n_samples` (500) sous-espaces de Haar de la distance min à `W`. Borne inférieure également. |
| `"riemannian"` | Optimisation pymanopt ; très lente et peu fiable, déconseillée. |

## 5. Projection sur l'admissible (`admissible_projection_method`)

- `"gradient"` (défaut) : partant de `S_k`, montée de gradient sur la Grassmannienne pour s'éloigner des `w` qui violent l'exclusion
  (direction `Σ_w (γh_k − d_w) ∇_S d_w`, `∇_S d_w = −(1−r) S_w(S_wᵀS)/d_w`, projetée sur le plan tangent, rétraction QR, pas adaptatif sur la
  pénalité `Σ_w (γh_k − d_w)_+²`), arrêt au premier sous-espace admissible, puis bissection sur le segment `S_k → S` pour revenir au plus près de
  `S_k`. En tests unitaires : toujours admissible, 10 à 100× plus proche de `S_k` que le Monte Carlo, 1-2 ms au lieu de 0,02-1,7 s.
- `"monte_carlo"` : `admissible_projection_n_samples` (5000) candidats de Haar, on garde l'admissible le plus proche de `S_k` (distance chordale).
- `"riemannian"` : pymanopt (lent).

Repli commun : si aucun candidat admissible n'est trouvé, GCO utilise le sous-espace `S*` qui réalise le rayon d'exclusion (calculé à la même
ancre) : sa distance minimale à `W` vaut `h_k ≥ γh_k`, il est donc admissible par construction (plus d'erreur « No admissible subspace found »).

Pourquoi ces défauts (projection par gradient + `mixed_period = 4`) : avec PLS seule, la projection par gradient, qui reste proche du sous-espace
PLS, dégradait les runs sur n = 62..191 (écart final +0,2 décade, τ = 1e-1 résolu 53 % contre 74 %) ; l'alternance avec un sous-espace aléatoire
un appel sur 4 réduit cette perte à +0,07 / +0,10 décade (médiane appariée, deux implémentations, p = 0,08 et 0,02 : une petite perte reste
possible sur n = 62..191) et l'annule sur n = 10..50 (+0,01, p = 0,5), pour un temps 5 à 30 % plus court que PLS + Monte Carlo
(tests internes, problèmes S2MPJ décalés, 2 graines). Avec grad_PLS au lieu de PLS, le gain est net (section 3).

## 6. Accélérations de calcul

Elles ne changent pas l'algorithme :
- Distances chordales par `‖SSᵀ − S'S'ᵀ‖²_F/2 = (p−1) − ‖SᵀS'‖²_F` (aucune matrice n×n) et un produit matriciel par bloc de candidats pour toutes les
  variables de `W` (20 à 28× plus rapide pour une projection à 5000 candidats).
- Tirage de Haar par lot (une QR batchée, `numpy.random.Generator` initialisé depuis le flux global : `np.random.seed` reste valable).
- DOE : détection de doublons par une matrice de distances unique, élagage (plafond 1000 points) par matrice de distances calculée une fois.
- **Élagage de `W` par l'ancre** (`W_prune_by_anchor`, vrai par défaut) : comme `d ≥ √r·‖x − x_w‖`, une variable `w` trop loin en `x` ne peut pas être sous
  le seuil et est écartée du calcul de distance (exact ; actif seulement si il retire plus de 40 % de `W`, donc surtout pour `n` grand).
- **Projection sur la zonotope par Newton dual** (`zonotope_projection="newton"`, défaut) : quand `V s` sort de la boîte `[-1,1]^n`, la projection
  `γ_Z(s) = argmin ‖x − Vs‖² s.c. Vᵀx = s, x ∈ [-1,1]^n` était résolue par SLSQP en dimension `n` à chaque évaluation. Elle s'écrit
  `x(λ) = clip(V(s+λ))` avec `λ ∈ R^p` tel que `g(λ) = Vᵀclip(V(s+λ)) − s = 0`, gradient d'une fonction convexe linéaire par morceaux ;
  un Newton semi-lisse (Hessienne généralisée `V_FᵀV_F`, `F` = coordonnées libres, recherche linéaire d'Armijo) la résout en dimension `p`.
  Un certificat exact (`uᵀs > ‖Vu‖₁` pour `u ∈ {λ, −g}`) détecte les points hors zonotope sans optimiser ; on renvoie alors `(clip(Vs), False)` comme
  avant. Repli sur SLSQP si Newton ne converge pas en 60 itérations (`newton_stats` compte les appels, replis, etc.). `"slsqp"` restaure l'ancien comportement.
- Résidus de projection du cache calculés par Pythagore (`‖x‖² − ‖Vᵀx‖²`).

Mesuré sur 36 configurations (n = 10, 20, 50 ; PyNomad ; PLS et aléatoire ; 10 runs chacune), l'ancien code et le nouveau convergent de façon équivalente
et le temps total est réduit de 59 % en moyenne (t_nouveau = 0,41·t_ancien).

## 7. Options principales

| Option | Défaut | Rôle |
|---|---|---|
| `subspace_dimension` | obligatoire | dimension `p` des sous-espaces (1 < p < n) |
| `subspace_selection` | `"grad_PLS"` | `"random"`, `"PLS"`, `"Local_PLS"`, `"grad_PLS"`, `"grad_random"` (ou une méthode enregistrée) ; `"mixed"` = ancienne écriture |
| `mixed_period`, `mixed_method` | 4, `"random"` | alternance : un appel sur `mixed_period` utilise `mixed_method` (section 3) |
| `grad_every`, `grad_fd_step`, `fixed_columns_draws` | 1, 1e-6, 20 | méthodes à gradient (section 3) |
| `local_pls_factor` | 2.0 | taille `K = ⌈f·(n−1)⌉` de la PLS locale |
| `mixed_random_period`, `mixed_local_factor` | 2, 0.0 | uniquement avec l'ancienne écriture `"mixed"` (section 3) |
| `gamma` | 0.9 | seuil d'admissibilité `γ·h_k` |
| `contraction` | 0.5 | facteur `β` de mise à jour de `σ_k` |
| `exclusion_radius_method` | `"gradient"` | `"gradient"`, `"monte_carlo"`, `"riemannian"` |
| `exclusion_radius_n_samples` | 500 | échantillons du rayon Monte Carlo |
| `exclusion_radius_gradient_init` / `_max_iter` | 20 / 15 | candidats de départ / itérations de la montée de gradient |
| `admissible_projection_method` | `"gradient"` | `"gradient"`, `"monte_carlo"`, `"riemannian"` (section 5) |
| `admissible_projection_n_samples` | 5000 | candidats de la projection Monte Carlo |
| `zonotope_projection` | `"newton"` | `"newton"` ou `"slsqp"` : projection sur la zonotope (section 6) |
| `W_prune_by_anchor` | True | élagage exact de `W` (section 6) |
| `min_termination_scale_parameter` | 1e-9 | arrêt quand `σ_k` passe en dessous |
| `max_cache_point`, `cache_selection_method`, `cache_projection_threshold` | 900, `"closest_subspace"`, 0.0025 | réutilisation des points déjà évalués dans le sous-espace |
| `file_path`, `output_columns` | `"log_GCO.txt"`, `["BBE","OBJ"]` | fichier de log |

Les anciens noms `covering_radius_*` restent acceptés (alias de `exclusion_radius_*`).

## 8. Sous-optimiseurs

Un sous-optimiseur est une fonction `suboptimizer(fun, sigma_k, initial_doe, lb, ub, cache_name=...)` qui renvoie `(Zk, Fk, n_eval)` : points évalués dans
l'espace latent (dimension `p`), valeurs, nombre d'évaluations. Fournis : `mads_suboptimizer` (PyNomad, `MIN_FRAME_SIZE = σ_k`) et `trego_suboptimizer`.
