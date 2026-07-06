from entrypoint._common import run_method

if __name__ == "__main__":
    run_method(
        title="Базовая линия: все линейные методы",
        model_names=["linear", "ridge", "lasso", "elasticnet"],
    )
