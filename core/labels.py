GROUP_LABELS = {
    "linear": "Линейные",
    "formula": "Степенная формула (биоинспир.)",
}

TARGET_LABELS = {
    "SP63": "СП 63.13330",
    "RUK78": "Руководство 1978",
}

MODEL_LABELS = {
    "linear": "Линейная регрессия",
    "ridge": "Ridge-регрессия",
    "lasso": "Lasso-регрессия",
    "elasticnet": "ElasticNet-регрессия",
    "bio_ga": "Степенная: генетический алгоритм",
    "bio_de": "Степенная: дифференциальная эволюция",
    "bio_pso": "Степенная: рой частиц (PSO)",
    "bio_cmaes": "Степенная: CMA-ES",
    "gbr": "Градиентный бустинг",
    "svr": "Метод опорных векторов (SVR)",
    "knn": "Метод ближайших соседей (KNN)",
    "gpr": "Гауссовский процесс (GPR)",
    "symreg": "Символьная регрессия (gplearn)",
    "bayes_symreg": "Байесовская символьная регрессия (ARD)",
    "pinn": "Физически-информированная нейросеть (PINN)",
    "mlp": "Многослойный перцептрон (MLP)",
}

Q_UNIT = "кН"

def method_label(name):
    return MODEL_LABELS.get(name, name)


def target_label(name):
    return TARGET_LABELS.get(name, name)
