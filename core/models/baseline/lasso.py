import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.base import BaseModel


class LassoModel(BaseModel):
    name = "lasso"
    group = "linear"

    def __init__(self, alpha=3.0):
        self._m = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=10000))

    def fit(self, X, y):
        self._m.fit(X, y); return self

    def predict(self, X):
        return self._m.predict(X)
