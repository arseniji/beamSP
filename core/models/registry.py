from core.models.baseline.elastic_net import ElasticNetModel
from core.models.baseline.linear import LinearModel
from core.models.baseline.ridge import RidgeModel
from core.models.baseline.lasso import LassoModel
from core.models.classic_ml.gbr import GBRModel
from core.models.classic_ml.gpr import GPRModel
from core.models.classic_ml.knn import KNNModel
from core.models.classic_ml.svr import SVRModel
from core.models.bioinspired.formula_search import bio_models
from core.models.sym_regression.baesian_symbolic_regression import BayesianSymbolicRegressionModel
from core.models.sym_regression.symbolic_regression import SymbolicRegressionModel

_BASELINE = [LinearModel, RidgeModel, LassoModel, ElasticNetModel]
_CLASSIC = [GBRModel, SVRModel, KNNModel, GPRModel]
_FORMULA = [SymbolicRegressionModel, BayesianSymbolicRegressionModel]

_REGISTRY = {cls.name: cls for cls in (*_BASELINE, *_CLASSIC, *_FORMULA)}
_REGISTRY.update(bio_models())


def model_class(name):
    if name not in _REGISTRY:
        raise KeyError(f"Модель '{name}' не зарегистрирована. "
                       f"Доступны: {list(_REGISTRY)}")
    return _REGISTRY[name]


def get_model(name):
    return model_class(name)()


def available_models():
    return list(_REGISTRY)