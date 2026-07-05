from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.baseline.scaled_linear import ScaledLinearModel


class ElasticNetModel(ScaledLinearModel):
    name = "elasticnet"
    group = "linear"

    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self._m = make_pipeline(StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))