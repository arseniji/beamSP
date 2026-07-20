from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.models.base import BaseModel


class MLPModel(BaseModel):
    name = "mlp"
    group = "neural"

    def __init__(self, hidden=(32, 16), alpha=0.5, seed=1337):
        self._hidden = hidden
        self._alpha = alpha
        self._seed = seed
        self._m = None

    def set_seed(self, seed):
        self._seed = seed
        return self

    def fit(self, X, y):
        net = MLPRegressor(
            hidden_layer_sizes=self._hidden, activation="relu", alpha=self._alpha,
            max_iter=3000, early_stopping=False, random_state=self._seed)
        pipe = make_pipeline(StandardScaler(), net)
        self._m = TransformedTargetRegressor(
            regressor=pipe, transformer=StandardScaler())
        self._m.fit(X, y); return self

    def predict(self, X):
        return self._m.predict(X)