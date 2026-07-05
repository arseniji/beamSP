from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.baseline.scaled_linear import ScaledLinearModel


class LinearModel(ScaledLinearModel):
    name = "linear"
    group = "linear"

    def __init__(self):
        self._m = make_pipeline(StandardScaler(), LinearRegression())
