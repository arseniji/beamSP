from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.baseline.scaled_linear import ScaledLinearModel


class LassoModel(ScaledLinearModel):
    name = "lasso"
    group = "linear"

    def __init__(self, alpha=3.0):
        self._m = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))
