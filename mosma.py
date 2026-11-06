"""Multi-Objective Slime Mould Algorithm (MOSMA) for whole-lot portfolios.

Objectives (both minimised):  f1 = -(net expected return),  f2 = volatility.

Design:
* Genotype  : continuous weight vector (simplex, capped by `weight_limit`).
* Phenotype : the whole-lot portfolio obtained with `utils.weights_to_lots`.
  Objectives are evaluated on the *lots actually bought* (leftover cash earns
  nothing, fees on the rebalancing from `k_prev` are deducted), so the search
  really optimises discrete, feasible portfolios instead of rounding a
  continuous optimum afterwards.
* Ranking   : NSGA-II style (non-dominated fronts, then crowding distance).
  The SMA weights W and the probability p are computed from that rank.
* Archive   : external archive of non-dominated lot portfolios, truncated with
  the crowding distance.
* Final set : NSGA-III style niching on the archive with *uniformly spaced*
  reference directions, to give the user an evenly spread front.

Reference: Li et al. (2020) SMA; Houssein et al. (2021) MOSMA;
Deb et al. (2002) NSGA-II; Deb & Jain (2014) NSGA-III.
"""
import numpy as np
from utils import weights_to_lots, LOT_SIZE


# ---------------------------------------------------------------------------
# Constraint handling
# ---------------------------------------------------------------------------
def repair_weights(weights, weight_limit):
    """Project onto {w >= 0, sum w = 1, w_i <= cap} (iterative capping)."""
    n = len(weights)
    w = np.clip(np.nan_to_num(weights), 0, None)
    if w.sum() <= 0:
        return np.ones(n) / n
    cap = max(weight_limit, 1.0 / n)
    w = w / w.sum()
    capped = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = w > cap + 1e-12
        if not over.any():
            break
        w[over] = cap
        capped |= over
        free = ~capped
        if not free.any():
            break
        remaining = 1.0 - w[capped].sum()
        s = w[free].sum()
        w[free] = remaining / free.sum() if s <= 1e-12 else w[free] / s * remaining
    return w / w.sum()


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def evaluate_portfolio(weights, mu, cov, prices, k_prev, wealth, buy_fee, sell_fee, weight_limit):
    """Return (objectives, lots, realised weights over total wealth)."""
    lots, cash, tc = weights_to_lots(weights, wealth, prices, k_prev, buy_fee, sell_fee, weight_limit)
    w_real = lots * LOT_SIZE * prices / wealth          # cash has weight 1 - sum(w_real)
    ret = float(mu @ w_real) - tc / wealth              # one-off rebalancing cost
    risk = float(np.sqrt(max(w_real @ cov @ w_real, 0.0)))
    return np.array([-ret, risk]), lots, w_real


# ---------------------------------------------------------------------------
# NSGA-II tools
# ---------------------------------------------------------------------------
def non_dominated_fronts(F):
    """Fast non-dominated sort. Returns a list of index arrays (front 0 first)."""
    n = len(F)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dominates = le & lt                                  # dominates[i, j]: i dominates j
    dom_count = dominates.sum(axis=0).astype(int)
    assigned = np.zeros(n, dtype=bool)
    fronts = []
    while not assigned.all():
        current = np.where(~assigned & (dom_count == 0))[0]
        if current.size == 0:
            break
        fronts.append(current)
        assigned[current] = True
        dom_count -= dominates[current].sum(axis=0)
    return fronts


def crowding_distance(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k])
        span = F[order[-1], k] - F[order[0], k]
        d[order[0]] = d[order[-1]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return d


def nsga2_order(F):
    """Indices sorted best -> worst by (front rank, -crowding distance)."""
    order = []
    for front in non_dominated_fronts(F):
        cd = crowding_distance(F[front])
        order.extend(front[np.argsort(-cd)])
    return np.array(order)


def hypervolume_2d(F, ref):
    """Exact 2-D hypervolume of the points of F dominating `ref` (minimisation)."""
    pts = F[np.all(F < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_f2 = 0.0, ref[1]
    for f1, f2 in pts:
        if f2 < best_f2:
            hv += (ref[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return hv


def update_archive(arch_W, arch_F, new_W, new_F, max_size):
    W = np.vstack([arch_W, new_W]) if len(arch_W) else new_W
    F = np.vstack([arch_F, new_F]) if len(arch_F) else new_F
    _, uniq = np.unique(np.round(F, 10), axis=0, return_index=True)
    W, F = W[uniq], F[uniq]
    front = non_dominated_fronts(F)[0]
    W, F = W[front], F[front]
    while len(F) > max_size:                              # drop most crowded point
        cd = crowding_distance(F)
        drop = np.argmin(cd)
        W, F = np.delete(W, drop, axis=0), np.delete(F, drop, axis=0)
    return W, F


def nsga3_niching(W, F, num_ref_points=50):
    """Keep, for each uniformly spaced reference direction, the closest point."""
    if len(F) <= 2:
        return W, F
    z_min, z_max = F.min(axis=0), F.max(axis=0)
    Fn = (F - z_min) / np.where(z_max - z_min == 0, 1e-12, z_max - z_min)
    t = np.linspace(0, 1, num_ref_points)
    refs = np.stack([1 - t, t], axis=1)
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    proj = Fn @ refs.T                                    # (n_points, n_refs)
    dist = np.sqrt(np.maximum((Fn ** 2).sum(axis=1, keepdims=True) - proj ** 2, 0))
    keep = np.unique(np.argmin(dist, axis=0))
    return W[keep], F[keep]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def execute_mosma(n_assets, expected_returns, cov_matrix, weight_limit,
                  current_prices, k_prev, total_wealth, alpha_buy, alpha_sell,
                  pop_size=150, max_iterations=150,
                  stop_criterion="Max Iterations", patience=15, use_repair=True,
                  z=0.03, archive_size=200, num_ref_points=50, seed=None,
                  progress=None):
    """Return a list of weight vectors (genotypes) spanning the Pareto front.

    `progress(fraction, text)` is an optional callback (e.g. a Streamlit bar).
    """
    rng = np.random.default_rng(seed)
    mu, cov = np.asarray(expected_returns), np.asarray(cov_matrix)
    prices, k_prev = np.asarray(current_prices, float), np.asarray(k_prev, float)
    cap = max(weight_limit, 1.0 / n_assets)

    def fix(w):
        if use_repair:
            return repair_weights(w, weight_limit)
        w = np.clip(w, 0, None)
        return w / w.sum() if w.sum() > 0 else np.ones(n_assets) / n_assets

    def evaluate(P):
        F = np.empty((len(P), 2))
        for i, w in enumerate(P):
            F[i] = evaluate_portfolio(w, mu, cov, prices, k_prev, total_wealth,
                                      alpha_buy, alpha_sell, weight_limit)[0]
        return F

    pop = np.array([fix(rng.dirichlet(np.ones(n_assets))) for _ in range(pop_size)])
    arch_W, arch_F = np.empty((0, n_assets)), np.empty((0, 2))
    hv_ref, best_hv, stall = None, -np.inf, 0

    for it in range(max_iterations):
        if progress and it % 5 == 0:
            progress(it / max_iterations, f"MOSMA iteration {it}/{max_iterations}")

        pop = np.array([fix(w) for w in pop])
        F = evaluate(pop)
        arch_W, arch_F = update_archive(arch_W, arch_F, pop, F, archive_size)

        if stop_criterion == "Hypervolume Plateau":
            if hv_ref is None:   # fixed reference point = nadir of the first population + 10 %
                span = np.ptp(F, axis=0) + 1e-12
                hv_ref = F.max(axis=0) + 0.1 * span
            hv = hypervolume_2d(arch_F, hv_ref)
            if hv > best_hv * (1 + 1e-3) + 1e-15:
                best_hv, stall = hv, 0
            else:
                stall += 1
            if stall >= patience:
                break

        # --- SMA weights from the NSGA-II rank ------------------------------
        order = nsga2_order(F)
        s = np.empty(pop_size)
        s[order] = np.arange(pop_size) / max(pop_size - 1, 1)   # 0 = best, 1 = worst
        log_term = np.log10(s + 1)
        r = rng.random(pop_size)
        good = np.zeros(pop_size, dtype=bool)
        good[order[: pop_size // 2]] = True
        Wsma = np.where(good, 1 + r * log_term, 1 - r * log_term)

        frac = (it + 1) / max_iterations
        a = np.arctanh(np.clip(1 - frac, 0, 1 - 1e-12))
        b = 1 - frac
        vb = rng.uniform(-a, a, (pop_size, 1))
        vc = rng.uniform(-b, b, (pop_size, 1))
        p = np.tanh(s)[:, None]                                  # SMA: p = tanh|S(i) - DF|

        X_best = arch_W[rng.integers(len(arch_W), size=pop_size)]  # elite from the archive
        A = rng.integers(pop_size, size=(pop_size, n_assets))
        B = rng.integers(pop_size, size=(pop_size, n_assets))
        cols = np.arange(n_assets)
        XA, XB = pop[A, cols], pop[B, cols]

        approach = X_best + vb * (Wsma[:, None] * XA - XB)
        contract = vc * pop
        new_pop = np.where(rng.random((pop_size, n_assets)) < p, approach, contract)
        restart = rng.random(pop_size) < z
        new_pop[restart] = rng.random((restart.sum(), n_assets)) * cap
        pop = new_pop

    if progress:
        progress(1.0, "MOSMA done")
    if len(arch_W) == 0:
        return []
    W, _ = nsga3_niching(arch_W, arch_F, num_ref_points)
    return list(W)
