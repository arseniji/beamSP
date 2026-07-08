from core.models.base import BaseModel
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class SVRModel(BaseModel):
    name = "svr"
    group = "kernel"

    def __init__(self, C=300.0, epsilon=0.1, gamma=0.025):
        self._m = make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma))

    def fit(self, X, y):
        self._m.fit(X, y); return self

    def predict(self, X):
        return self._m.predict(X)