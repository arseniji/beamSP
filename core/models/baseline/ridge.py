from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.base import BaseModel

class RidgeModel(BaseModel):
    name = "ridge"
    group = "linear"

    def __init__(self, alpha=1.0):
        self._m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X, y):
        self._m.fit(X, y); return self

    def predict(self, X):
        return self._m.predict(X)