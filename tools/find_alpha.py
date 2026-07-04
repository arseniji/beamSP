import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np

from config import settings as cfg
from core.data.loader import load_dataset
from core.evaluation.runner import prepare
from core.evaluation.validation import leave_one_group_out
from core.evaluation.metrics import compute_metrics
from core.labels import target_label
from core.models.baseline.ridge import RidgeModel

DEFAULT_ALPHAS = [0, 0.1, 1, 3, 10, 30, 50, 100, 150, 200, 300, 1000]


def sweep(target_col, alphas, synth_cfg, seed):
    df, groups = prepare(load_dataset(cfg.DATA_PATH), cfg.FEATURES, synth_cfg, seed)
    X = df[cfg.FEATURES].to_numpy(float)
    y = df[target_col].to_numpy(float)
    is_synth = df["is_synth"].to_numpy(bool)

    out = []
    for a in alphas:
        preds = leave_one_group_out(lambda a=a: RidgeModel(alpha=a),
                                    X, y, groups, is_synth)
        real = ~is_synth & np.isfinite(preds)
        m = compute_metrics(y[real], preds[real])
        out.append((a, m["R2"], m["RMSE"]))
    return out


def _print_table(tkey, rows):
    best = max(rows, key=lambda r: r[1])
    print(f"\n=== {target_label(tkey)} ===")
    print(f"{'alpha':>8}  {'R²':>7}  {'RMSE':>7}")
    for a, r2, rmse in rows:
        mark = " лучшее" if (a, r2, rmse) == best else ""
        print(f"{a:>8}  {r2:>7.3f}  {rmse:>7.2f}{mark}")
    print(f"Лучшая alpha по R²: {best[0]} (R²={best[1]:.3f}, RMSE={best[2]:.2f})")
    return best


def build_figure(results, alphas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tkeys = list(results)
    fig, axes = plt.subplots(1, len(tkeys), figsize=(6 * len(tkeys), 4.5), squeeze=False)
    x = np.arange(len(alphas))
    labels = [f"{a:g}" for a in alphas]
    for ax, tkey in zip(axes[0], tkeys):
        rows = results[tkey]
        r2 = [r[1] for r in rows]
        rmse = [r[2] for r in rows]
        best_i = max(range(len(rows)), key=lambda i: rows[i][1])

        ax.plot(x, r2, "o-", color="#2166ac", label="R²")
        ax.set_ylabel("R²", color="#2166ac")
        ax.tick_params(axis="y", labelcolor="#2166ac")

        ax2 = ax.twinx()
        ax2.plot(x, rmse, "s--", color="#e08a1e", label="RMSE")
        ax2.set_ylabel("RMSE, кН", color="#e08a1e")
        ax2.tick_params(axis="y", labelcolor="#e08a1e")

        ax.axvline(best_i, color="#2a9d3f", ls=":", lw=1.5)
        ax.annotate(f"α = {alphas[best_i]:g}", xy=(best_i, r2[best_i]),
                    xytext=(4, -12), textcoords="offset points",
                    color="#2a9d3f", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
        ax.set_xlabel("alpha")
        ax.set_title(target_label(tkey))
        ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Подбор alpha для Ridge (LOGO)")
    p.add_argument("--target", choices=list(cfg.TARGETS) + ["both"], default="both")
    p.add_argument("--alphas", help="список через запятую, напр. 0,10,50,100")
    p.add_argument("--no-synth", action="store_true", help="отключить синтез")
    p.add_argument("--samples", type=int, help="образцов синтеза на профиль")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument("--plot", action="store_true", help="сохранить график")
    p.add_argument("--out", type=Path, help="путь для графика (подразумевает --plot)")
    args = p.parse_args(argv)

    alphas = ([float(x) for x in args.alphas.split(",")]
              if args.alphas else DEFAULT_ALPHAS)
    targets = (list(cfg.TARGETS) if args.target == "both" else [args.target])

    synth_cfg = dict(cfg.SYNTH)
    synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
    if args.no_synth:
        synth_cfg["enabled"] = False
    if args.samples is not None:
        synth_cfg["samples_per_profile"] = args.samples

    print("Подбор alpha для Ridge (схема Leave-One-Group-Out)")
    print(f"seed={args.seed}, синтез={'вкл' if synth_cfg['enabled'] else 'выкл'}")

    results = {}
    best = {}
    for tkey in targets:
        rows = sweep(cfg.TARGETS[tkey], alphas, synth_cfg, args.seed)
        results[tkey] = rows
        best[tkey] = _print_table(tkey, rows)[0]

    if len(best) > 1:
        print("\nИтог по целям:")
        for tkey, a in best.items():
            print(f"  {target_label(tkey)}: alpha = {a}")

    if args.plot or args.out:
        out = args.out or (cfg.RESULTS_DIR / "alpha_sweep.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig = build_figure(results, alphas)
        fig.savefig(out, dpi=150)
        print(f"\nГрафик сохранён: {out}")

    return best

if __name__ == "__main__":
    main()
