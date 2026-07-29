import numpy as np


class PowerLawForm:
    log_a_bounds = (-20.0, 20.0)
    power_bounds = (-5.0, 5.0)

    def __init__(self, fixed_exponents=None):
        self._fixed_exponents = dict(fixed_exponents or {})
        self._used_idx = None
        self._free_mask = None
        self._fixed_vals = None

    def usable_features(self, X):
        used_idx = np.where(X.min(axis=0) > 0)[0]
        self._used_idx = used_idx
        self._free_mask = np.array([i not in self._fixed_exponents for i in used_idx])
        self._fixed_vals = np.array(
            [self._fixed_exponents.get(i, 0.0) for i in used_idx], dtype=float)
        return used_idx

    def _expand(self, free_params):
        powers = self._fixed_vals.copy()
        powers[self._free_mask] = free_params
        return powers

    def param_bounds(self, n_features):
        n_free = int(self._free_mask.sum())
        return [self.log_a_bounds] + [self.power_bounds] * n_free

    def objective(self, Xu, y):
        logX = np.log(Xu)
        log_y = np.log(np.clip(y, 1e-9, None))

        def mse(params):
            powers = self._expand(params[1:])
            log_pred = params[0] + logX @ powers
            return float(np.mean((log_pred - log_y) ** 2))

        return mse

    def predict(self, params, Xu):
        powers = self._expand(params[1:])
        log_q = params[0] + np.log(Xu) @ powers
        return np.exp(np.clip(log_q, -50, 50))

    def render(self, params, names, target="Qдв"):
        a = np.exp(params[0])
        powers = self._expand(params[1:])
        terms = [f"{name}^{p:.3f}" for name, p in zip(names, powers)]
        return f"{target} = {a:.4g} * " + " * ".join(terms)