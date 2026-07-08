import argparse
import itertools

import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import method_label, target_label
from core.models.registry import model_class


def _num(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def parse_grid(pairs):
    grid = {}
    for pair in pairs:
        name, _, vals = pair.partition("=")
        grid[name] = [_num(v) for v in vals.split(",")]
    return grid


def _fmt(params):
    return " ".join(f"{k}={v}" for k, v in params.items()) or "(дефолт)"


def _logo(cls, params, target_col, synth_cfg, seed):
    df, groups = prepare(load_dataset(cfg.DATA_PATH), cfg.FEATURES, synth_cfg, seed)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = df["is_synth"].to_numpy(bool)
    preds = leave_one_group_out(lambda: cls(**params), X, y, groups, is_synth)
    real = ~is_synth & np.isfinite(preds)
    m = compute_metrics(y[real], preds[real])
    return m["R2"], m["RMSE"]


def _r2_train(cls, params, target_col, synth_cfg, seed):
    df, _ = prepare(load_dataset(cfg.DATA_PATH), cfg.FEATURES, synth_cfg, seed)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    real = ~df["is_synth"].to_numpy(bool)
    model = cls(**params).fit(X, y)
    from sklearn.metrics import r2_score
    return float(r2_score(y[real], model.predict(X[real])))


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров модели (LOGO)")
    p.add_argument("--model", required=True, help="имя модели из реестра, напр. gbr")
    p.add_argument("--grid", nargs="*", default=[], help="параметры: name=v1,v2,… (перебор перекрёстно)")
    p.add_argument("--target", choices=list(cfg.TARGETS) + ["both"], default="both")
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--seed", type=int, default=cfg.SEED)
    args = p.parse_args(argv)

    cls = model_class(args.model)
    grid = parse_grid(args.grid)
    keys = list(grid)
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))] or [{}]
    targets = list(cfg.TARGETS) if args.target == "both" else [args.target]

    synth_cfg = dict(cfg.SYNTH)
    synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
    if args.no_synth:
        synth_cfg["enabled"] = False
    if args.samples is not None:
        synth_cfg["samples_per_profile"] = args.samples

    print(f"Подбор параметров '{method_label(args.model)}' (LOGO): "
          f"конфигов {len(configs)}, seed={args.seed}")

    for tkey in targets:
        rows = []
        for params in configs:
            r2, rmse = _logo(cls, params, cfg.TARGETS[tkey], synth_cfg, args.seed)
            r2tr = _r2_train(cls, params, cfg.TARGETS[tkey], synth_cfg, args.seed)
            rows.append((params, r2, rmse, r2tr - r2))
        best = max(rows, key=lambda r: r[1])
        print(f"\n=== {target_label(tkey)} ===")
        print(f"{'параметры':<44} {'R²':>7} {'RMSE':>7} {'overfit':>8}")
        for params, r2, rmse, overfit in rows:
            tag = " лучшее" if params is best[0] else ""
            print(f"{_fmt(params):<44} {r2:>7.3f} {rmse:>7.2f} {overfit:>8.3f}{tag}")
        print(f"Лучшее: {_fmt(best[0])} → R²={best[1]:.3f}, overfit={best[3]:.3f}")


if __name__ == "__main__":
    main()
