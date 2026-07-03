from .linear import LinearModel

_REGISTRY = {
    LinearModel.name: LinearModel,
}

def get_model(name):
    if name not in _REGISTRY:
        raise KeyError(f"Модель '{name}' не зарегистрирована. "
                       f"Доступны: {list(_REGISTRY)}")
    return _REGISTRY[name]()