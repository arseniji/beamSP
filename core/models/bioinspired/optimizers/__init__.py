from functools import partial

from core.models.bioinspired.optimizers.ga import ga
from core.models.bioinspired.optimizers.de import de

OPTIMIZERS = {
    "ga": partial(ga, pop_size=150, generations=2000),
    "de": partial(de, maxiter=100, popsize=20),
}
