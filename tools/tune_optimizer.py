import argparse
import itertools
from functools import partial

import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import target_label
from core.models.bioinspired import optimizers as opt_mod
from core.models.bioinspired.formula_search import FormulaSearchModel


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


def _logo(optimizer_fn, params, target_col, synth_cfg, seed):
    df, groups = prepare(load_dataset(cfg.DATA_PATH), cfg.FEATURES, synth_cfg, seed)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = df["is_synth"].to_numpy(bool)
    preds = leave_one_group_out(
        lambda: FormulaSearchModel(partial(optimizer_fn, **params), seed=seed),
        X, y, groups, is_synth)
    real = ~is_synth & np.isfinite(preds)
    return compute_metrics(y[real], preds[real])


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров оптимизатора (LOGO)")
    p.add_argument("--optimizer", default="de", help="имя из optimizers.py: ga|de|…")
    p.add_argument("--grid", nargs="*", default=[], help="параметры: name=v1,v2,… (перебор перекрёстно)")
    p.add_argument("--target", choices=list(cfg.TARGETS) + ["both"], default="both")
    p.add_argument("--repeats", type=int, default=1, help="усреднить по N зёрнам")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    args = p.parse_args(argv)

    optimizer_fn = getattr(opt_mod, args.optimizer)
    grid = parse_grid(args.grid)
    keys = list(grid)
    configs = [dict(zip(keys, combo))
               for combo in itertools.product(*(grid[k] for k in keys))] or [{}]
    seeds = [args.seed + i for i in range(args.repeats)]
    targets = list(cfg.TARGETS) if args.target == "both" else [args.target]

    synth_cfg = dict(cfg.SYNTH)
    synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
    if args.no_synth:
        synth_cfg["enabled"] = False
    if args.samples is not None:
        synth_cfg["samples_per_profile"] = args.samples

    print(f"Подбор параметров '{args.optimizer}' (схема LOGO): "
          f"конфигов {len(configs)}, seeds={seeds}")

    for tkey in targets:
        rows = []
        for params in configs:
            ms = [_logo(optimizer_fn, params, cfg.TARGETS[tkey], synth_cfg, sd)
                  for sd in seeds]
            rows.append((params,
                         float(np.mean([m["R2"] for m in ms])),
                         float(np.mean([m["RMSE"] for m in ms])),
                         float(np.mean([m["Qexp/Qpred_mean"] for m in ms]))))
        best = max(rows, key=lambda r: r[1])
        print(f"\n=== {target_label(tkey)} ===")
        print(f"{'параметры':<34} {'R²':>7} {'RMSE':>7} {'Qэ/Qп':>7}")
        for params, r2, rmse, ratio in rows:
            tag = " лучшее" if params is best[0] else ""
            print(f"{_fmt(params):<34} {r2:>7.3f} {rmse:>7.2f} {ratio:>7.3f}{tag}")
        print(f"Лучшее: {_fmt(best[0])} → R²={best[1]:.3f}")


if __name__ == "__main__":
    main()
