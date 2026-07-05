from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.baseline.scaled_linear import ScaledLinearModel


class RidgeModel(ScaledLinearModel):
    name = "ridge"
    group = "linear"

    def __init__(self, alpha=100.0):
        self._m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
