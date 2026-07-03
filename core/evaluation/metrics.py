import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = y_true.size
    abs_err = np.abs(y_true - y_pred)

    ratio = np.divide(y_true, y_pred, out=np.full_like(y_true, np.nan), where=y_pred != 0)
    ratio = ratio[np.isfinite(ratio)]

    rel = np.divide(abs_err, np.abs(y_true), out=np.full_like(y_true, np.nan), where=y_true != 0)
    rel = rel[np.isfinite(rel)]

    mape = float(np.mean(rel) * 100) if rel.size else float("nan")

    within15 = (float(np.mean((ratio >= 0.85) & (ratio <= 1.15)) * 100) if ratio.size else float("nan"))

    return {
        "Qexp/Qpred_mean": float(np.mean(ratio)) if ratio.size else float("nan"),
        "CV": float(np.std(ratio) / np.mean(ratio)) if ratio.size and np.mean(ratio) != 0 else float("nan"),
        "within15": within15,
        "MAPE": mape,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedianAE": float(np.median(abs_err)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MaxError": float(np.max(abs_err)) if n else float("nan"),
        "pct_negative": float(np.mean(y_pred < 0) * 100) if n else float("nan"),
        "pct_conservative": float(np.mean(y_pred <= y_true) * 100) if n else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
    }
