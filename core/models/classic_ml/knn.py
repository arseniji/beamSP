from core.models.base import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class KNNModel(BaseModel):
    name = "knn"
    group = "neighbors"

    def __init__(self, n_neighbors=55, weights="distance"):
        self._m = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights))

    def fit(self, X, y):
        self._m.fit(X, y); return self

    def predict(self, X):
        return self._m.predict(X)
