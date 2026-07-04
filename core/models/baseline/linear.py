import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core.models.base import BaseModel


class LinearModel(BaseModel):
    name = "linear"
    group = "linear"

    def __init__(self):
        self._m = make_pipeline(StandardScaler(), LinearRegression())

    def fit(self, X, y):
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)
