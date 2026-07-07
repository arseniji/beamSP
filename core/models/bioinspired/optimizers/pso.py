import numpy as np


def pso(objective, bounds, seed, n_particles=60, iters=300, w_hi=0.9, w_lo=0.4, c1=1.5, c2=1.5):
    rng = np.random.default_rng(seed)
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    dim = len(bounds)
    span = high - low
    v_max = 0.2 * span

    pos = rng.uniform(low, high, size=(n_particles, dim))
    vel = rng.uniform(-v_max, v_max, size=(n_particles, dim))
    pbest = pos.copy()
    pbest_fit = np.array([objective(p) for p in pos])
    g = int(np.argmin(pbest_fit))
    gbest = pbest[g].copy()

    for it in range(iters):
        w = w_hi - (w_hi - w_lo) * it / iters
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, low, high)
        fit = np.array([objective(p) for p in pos])
        improved = fit < pbest_fit
        pbest[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]
        g = int(np.argmin(pbest_fit))
        if pbest_fit[g] < objective(gbest):
            gbest = pbest[g].copy()

    return gbest
