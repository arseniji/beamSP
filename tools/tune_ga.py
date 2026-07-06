import argparse
from functools import partial

import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import target_label
from core.models.bioinspired.optimizers import ga
from core.models.bioinspired.formula_search import FormulaSearchModel


def _ga_model(pop, gens, seed):
    return FormulaSearchModel(partial(ga, pop_size=pop, generations=gens), seed=seed)


def _logo_metrics(pop, gens, target_col, synth_cfg, seed):
    df, groups = prepare(load_dataset(cfg.DATA_PATH), cfg.FEATURES, synth_cfg, seed)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = df["is_synth"].to_numpy(bool)
    preds = leave_one_group_out(
        lambda: _ga_model(pop, gens, seed),
        X, y, groups, is_synth)
    real = ~is_synth & np.isfinite(preds)
    return compute_metrics(y[real], preds[real])


def sweep(configs, target_col, synth_cfg, seeds):
    rows = []
    for pop, gens in configs:
        r2, rmse, ratio = [], [], []
        for sd in seeds:
            m = _logo_metrics(pop, gens, target_col, synth_cfg, sd)
            r2.append(m["R2"]); rmse.append(m["RMSE"]); ratio.append(m["Qexp/Qpred_mean"])
        rows.append((pop, gens, float(np.mean(r2)), float(np.mean(rmse)),
                     float(np.mean(ratio))))
    return rows


def _print_table(tkey, rows, n_seeds):
    best = max(rows, key=lambda r: r[2])
    print(f"\n=== {target_label(tkey)} ===")
    print(f"{'pop':>5} {'gens':>6}  {'R²':>7}  {'RMSE':>7}  {'Qэ/Qп':>7}")
    for pop, gens, r2, rmse, ratio in rows:
        mark = " лучшее" if (pop, gens) == (best[0], best[1]) else ""
        print(f"{pop:>5} {gens:>6}  {r2:>7.3f}  {rmse:>7.2f}  {ratio:>7.3f}{mark}")
    note = f" (среднее по {n_seeds} зёрнам)" if n_seeds > 1 else ""
    print(f"Лучшее: pop={best[0]}, gens={best[1]} → R²={best[2]:.3f}{note}")
    return best


def _print_formula(pop, gens, target_col, synth_cfg, seed):
    df = load_dataset(cfg.DATA_PATH)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    model = _ga_model(pop, gens, seed).fit(X, y)
    print("  формула:", model.formula(cfg.FEATURES))


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор гиперпараметров ГА (bio_ga)")
    p.add_argument("--pop", default="40,80,150", help="размеры популяции через запятую")
    p.add_argument("--gens", default="200,500,1000", help="числа поколений через запятую")
    p.add_argument("--target", choices=list(cfg.TARGETS) + ["both"], default="both")
    p.add_argument("--repeats", type=int, default=1, help="усреднить по N зёрнам")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--no-synth", action="store_true")
    p.add_argument("--samples", type=int)
    args = p.parse_args(argv)

    pops = [int(x) for x in args.pop.split(",")]
    gens = [int(x) for x in args.gens.split(",")]
    configs = [(pp, gg) for pp in pops for gg in gens]
    seeds = [args.seed + i for i in range(args.repeats)]
    targets = list(cfg.TARGETS) if args.target == "both" else [args.target]

    synth_cfg = dict(cfg.SYNTH)
    synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
    if args.no_synth:
        synth_cfg["enabled"] = False
    if args.samples is not None:
        synth_cfg["samples_per_profile"] = args.samples

    print("Подбор гиперпараметров ГА (степенная формула, схема LOGO)")
    print(f"seeds={seeds}, синтез={'вкл' if synth_cfg['enabled'] else 'выкл'}")

    for tkey in targets:
        rows = sweep(configs, cfg.TARGETS[tkey], synth_cfg, seeds)
        best = _print_table(tkey, rows, len(seeds))
        _print_formula(best[0], best[1], cfg.TARGETS[tkey], synth_cfg, args.seed)


if __name__ == "__main__":
    main()
