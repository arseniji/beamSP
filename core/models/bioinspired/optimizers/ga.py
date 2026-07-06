import numpy as np


def ga(objective, bounds, seed, pop_size=80, generations=300, elite_frac=0.05,
       tournament=3, blx_alpha=0.5, mutation_rate=0.2):
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds], float)
    high = np.array([b[1] for b in bounds], float)
    span = high - low
    dim = len(bounds)
    n_elite = max(1, int(pop_size * elite_frac))

    pop = rng.uniform(low, high, size=(pop_size, dim))
    fit = np.array([objective(ind) for ind in pop])

    for gen in range(generations):
        sigma = span * (0.2 * (1.0 - gen / generations) + 0.01)
        children = [pop[i] for i in np.argsort(fit)[:n_elite]]
        while len(children) < pop_size:
            p1 = _tournament(pop, fit, rng, tournament)
            p2 = _tournament(pop, fit, rng, tournament)
            child = _blx(p1, p2, rng, blx_alpha)
            child = _mutate(child, sigma, rng, mutation_rate)
            children.append(np.clip(child, low, high))
        pop = np.array(children)
        fit = np.array([objective(ind) for ind in pop])

    return pop[np.argmin(fit)]


def _tournament(pop, fit, rng, k):
    idx = rng.integers(len(pop), size=k)
    return pop[idx[np.argmin(fit[idx])]]


def _blx(p1, p2, rng, alpha):
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    d = hi - lo
    return rng.uniform(lo - alpha * d, hi + alpha * d)


def _mutate(child, sigma, rng, rate):
    child = child.copy()
    mask = rng.random(child.size) < rate
    child[mask] += rng.normal(0.0, 1.0, child.size)[mask] * sigma[mask]
    return child
