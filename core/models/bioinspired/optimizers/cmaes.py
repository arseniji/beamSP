import numpy as np


def cmaes(objective, bounds, seed, maxiter=300):
    import cma as _cma

    low = np.array([b[0] for b in bounds], float)
    high = np.array([b[1] for b in bounds], float)
    x0 = (low + high) / 2.0
    sigma0 = float(np.mean(high - low) / 4.0)

    es = _cma.CMAEvolutionStrategy(
        x0.tolist(), sigma0,
        {"bounds": [low.tolist(), high.tolist()], "seed": seed,
         "verbose": -9, "maxiter": maxiter})
    es.optimize(objective)
    return np.array(es.result.xbest)
