import numpy as np
from core.models.base import BaseModel

class ScaledLinearModel(BaseModel):
    _m = None

    def fit(self, X, y):
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return self._m.predict(X)

    def formula(self, feature_names, target="Qдв", eps=1e-8):
        scaler = self._m.steps[0][1]
        est = self._m.steps[-1][1]
        raw = est.coef_ / scaler.scale_
        b0 = float(est.intercept_ - np.sum(est.coef_ * scaler.mean_ / scaler.scale_))

        terms = [f"{b0:.4g}"]
        for name, c in zip(feature_names, raw):
            if abs(c) > eps:
                terms.append(f"{c:+.4g}*{name}")
        return f"{target} = " + " ".join(terms)
