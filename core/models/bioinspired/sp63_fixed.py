from config import settings as cfg

from core.formula.power_law import PowerLawForm
from core.models.bioinspired.formula_search import FormulaSearchModel
from core.models.bioinspired.optimizers import OPTIMIZERS

import core.models.registry as registry

fixed = {
    cfg.FEATURES.index("a_h0"): 0.0,
    cfg.FEATURES.index("H"): 1.0,
    cfg.FEATURES.index("s"): 1.0,
    cfg.FEATURES.index("R"): 1.0,
}


class FixedBioDESP63(FormulaSearchModel):
    name = "bio_de_sp63_fixed"

    def __init__(self):
        super().__init__(
            optimizer=OPTIMIZERS["de"],
            form=PowerLawForm(fixed_exponents=fixed),
            seed=cfg.SEED,
        )


registry._REGISTRY["bio_de_sp63_fixed"] = FixedBioDESP63

from entrypoint._common import run_method

if __name__ == "__main__":
    run_method(
        title="bio_de СП63: a/h0=0, H=s=R=1, E свободен",
        model_names=["bio_de_sp63_fixed"],
        argv=["--target", "SP63", "--export"],
    )
