from functools import partial

from core.models.bioinspired.optimizers.ga import ga
from core.models.bioinspired.optimizers.de import de
from core.models.bioinspired.optimizers.pso import pso

OPTIMIZERS = {
    "ga": partial(ga, pop_size=150, generations=2000),
    "de": partial(de, maxiter=100, popsize=20),
    "pso": partial(pso, n_particles=80, iters=600),
}
