import numpy as np


class PowerLawForm:
    log_a_bounds = (-20.0, 20.0)
    power_bounds = (-5.0, 5.0)

    def usable_features(self, X):
        return np.where(X.min(axis=0) > 0)[0]

    def param_bounds(self, n_features):
        return [self.log_a_bounds] + [self.power_bounds] * n_features

    def objective(self, Xu, y):
        logX = np.log(Xu)
        log_y = np.log(np.clip(y, 1e-9, None))

        def mse(params):
            log_pred = params[0] + logX @ params[1:]
            return float(np.mean((log_pred - log_y) ** 2))

        return mse

    def predict(self, params, Xu):
        log_q = params[0] + np.log(Xu) @ params[1:]
        return np.exp(np.clip(log_q, -50, 50))

    def render(self, params, names, target="Qдв"):
        a = np.exp(params[0])
        terms = [f"{name}^{p:.3f}" for name, p in zip(names, params[1:])]
        return f"{target} = {a:.4g} * " + " * ".join(terms)
