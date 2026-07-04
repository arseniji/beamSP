from core.models.baseline.linear import LinearModel
from core.models.baseline.ridge import RidgeModel
from core.models.baseline.lasso import LassoModel

_REGISTRY = {
    LinearModel.name: LinearModel,
    RidgeModel.name: RidgeModel,
    LassoModel.name: LassoModel,
}

def get_model(name):
    if name not in _REGISTRY:
        raise KeyError(f"Модель '{name}' не зарегистрирована. "
                       f"Доступны: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def available_models():
    return list(_REGISTRY)