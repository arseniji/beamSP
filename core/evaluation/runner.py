import numpy as np
import pandas as pd
from core.data.loader import load_dataset, profile_ids
from core.data.synth import synthesize
from core.evaluation.run_result import RunResult
from core.models.registry import get_model
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from sklearn.metrics import r2_score


def _overfit_metrics(model, X, y, real, logo_r2):
    try:
        model.fit(X, y)
        pred_train = model.predict(X[real])
        r2_train = float(r2_score(y[real], pred_train))
    except Exception:
        return float("nan"), float("nan")
    return r2_train, r2_train - logo_r2


def _worst_fold_rmse(y, preds, groups, real):
    worst = 0.0
    for g in np.unique(groups):
        mask = real & (groups == g)
        if mask.sum() == 0:
            continue
        rmse = float(np.sqrt(np.mean((y[mask] - preds[mask]) ** 2)))
        worst = max(worst, rmse)
    return worst

def prepare(df, feature_cols, synth_cfg, seed):
    if synth_cfg.get("enabled"):
        df = synthesize(df, synth_cfg["samples_per_profile"],
                        synth_cfg["noise"], feature_cols, seed)
    else:
        df = df.copy().reset_index(drop=True)
        df["is_synth"] = 0
        df["profile_id"] = profile_ids(df, feature_cols)
    groups = df["profile_id"].to_numpy()
    return df, groups


def run(model_names, feature_cols, targets, data_path, synth_cfg, seed=42,
        formula_names=None, progress=None):
    formula_names = formula_names or []
    df0 = load_dataset(data_path)
    result = RunResult()
    total = len(targets) * (len(model_names) + len(formula_names))
    done = 0

    for tkey, tcol in targets.items():
        df, groups = prepare(df0, feature_cols, synth_cfg, seed)
        X = df[feature_cols].to_numpy(float)
        y = df[tcol].to_numpy(float)
        is_synth = df["is_synth"].to_numpy(bool)

        rows = []
        preds_by_model = {}
        formulas = {}
        for name in model_names:
            preds = leave_one_group_out(lambda n=name: get_model(n),
                                        X, y, groups, is_synth)
            real = ~is_synth & np.isfinite(preds)
            m = compute_metrics(y[real], preds[real])
            m["RMSE_worst"] = _worst_fold_rmse(y, preds, groups, real)
            model = get_model(name)
            r2_train, overfit = _overfit_metrics(model, X, y, real, m["R2"])
            m["R2_train"] = r2_train
            m["overfit"] = overfit
            if hasattr(model, "formula"):
                formulas[name] = model.formula(feature_cols)

            m["model"] = name
            rows.append(m)
            preds_by_model[name] = {"y_true": y[real], "y_pred": preds[real]}
            done += 1
            if progress:
                progress(done, total, f"{name} / {tkey}")
        col_order = ["Qexp/Qpred_mean", "CV", "within15",
                     "MAPE", "MAE", "MedianAE", "RMSE", "RMSE_worst", "MaxError",
                     "pct_negative", "pct_conservative",
                     "R2", "R2_train", "overfit"]
        if rows:
            table = pd.DataFrame(rows).set_index("model")
            table = table[[c for c in col_order if c in table.columns]]
            result.metrics[tkey] = table.sort_values("RMSE")
        else:
            result.metrics[tkey] = pd.DataFrame()
        result.predictions[tkey] = preds_by_model
        result.formulas[tkey] = formulas

    return result
