import argparse

from config import settings as cfg
from core.evaluation.runner import run
from core.reporting import print_report, export_results


def run_method(title, model_names=None, argv=None):

    p = argparse.ArgumentParser(description=title)
    p.add_argument("--target", choices=list(cfg.TARGETS) + ["both"], default="both")
    p.add_argument("--no-synth", action="store_true", help="отключить синтез")
    p.add_argument("--samples", type=int, help="образцов синтеза на профиль")
    p.add_argument("--export", action="store_true", help="сохранить в results/")
    args = p.parse_args(argv)

    print("=" * 70)
    print(title)
    print("=" * 70)

    targets = (dict(cfg.TARGETS) if args.target == "both"
               else {args.target: cfg.TARGETS[args.target]})
    synth_cfg = dict(cfg.SYNTH)
    if args.no_synth:
        synth_cfg["enabled"] = False
    if args.samples is not None:
        synth_cfg["samples_per_profile"] = args.samples

    def progress(done, total, label):
        print(f"[{done}/{total}] {label}")

    result = run(
        model_names=model_names or [], feature_cols=cfg.FEATURES, targets=targets,
        data_path=cfg.DATA_PATH, synth_cfg=synth_cfg, seed=cfg.SEED,
        progress=progress,
    )
    print_report(result)

    if args.export:
        saved = export_results(result, cfg.RESULTS_DIR)
        print("\nСохранено в results/:")
        for name in saved:
            print(f"{name}")
    return result
