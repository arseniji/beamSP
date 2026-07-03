import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import settings as cfg
from core.evaluation.runner import run
from core.labels import method_label, target_label
from core.models.registry import available_models
from core.reporting import print_report, export_results


def _prompt(text, default):
    raw = input(f"{text} [{default}]: ").strip()
    return raw or str(default)


def _ask_yes_no(text, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{text} ({d}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "д", "да", "1")


def _ask_int(text, default):
    while True:
        raw = _prompt(text, default)
        try:
            return int(raw)
        except ValueError:
            print("  нужно целое число, попробуй ещё раз")


def _choose_target():
    keys = list(cfg.TARGETS)
    print("\nЦелевая величина:")
    for i, k in enumerate(keys, 1):
        print(f"  {i}) {target_label(k)}")
    print(f"  {len(keys) + 1}) обе")
    raw = _prompt("выбор", len(keys) + 1)
    try:
        idx = int(raw)
    except ValueError:
        idx = len(keys) + 1
    if 1 <= idx <= len(keys):
        k = keys[idx - 1]
        return {k: cfg.TARGETS[k]}
    return dict(cfg.TARGETS)


def _choose_models():
    names = available_models()
    print("\nМетоды (через запятую, пусто — все):")
    for i, n in enumerate(names, 1):
        print(f"  {i}) {method_label(n) or n}")
    raw = input("выбор: ").strip()
    if not raw:
        return names
    chosen = []
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit() and 1 <= int(tok) <= len(names):
            chosen.append(names[int(tok) - 1])
        elif tok in names:
            chosen.append(tok)
        else:
            print(f"  пропущено неизвестное: {tok!r}")
    return chosen or names


def _configure():
    targets = _choose_target()
    model_names = _choose_models()

    synth_cfg = dict(cfg.SYNTH)
    synth_cfg["noise"] = dict(cfg.SYNTH["noise"])
    synth_cfg["enabled"] = _ask_yes_no("\nВключить синтез образцов?",
                                       cfg.SYNTH["enabled"])
    if synth_cfg["enabled"]:
        synth_cfg["samples_per_profile"] = _ask_int(
            "  образцов синтеза на профиль",
            cfg.SYNTH["samples_per_profile"])

    do_export = _ask_yes_no("\nСохранить результаты в results/?", False)
    return targets, model_names, synth_cfg, do_export


def _run_once():
    targets, model_names, synth_cfg, do_export = _configure()

    print("\n" + "=" * 70)
    print("Запуск:", ", ".join(method_label(m) or m for m in model_names),
          "/", ", ".join(target_label(t) for t in targets))
    print("=" * 70)

    def progress(done, total, label):
        print(f"[{done}/{total}] {label}")

    result = run(
        model_names=model_names, feature_cols=cfg.FEATURES, targets=targets,
        data_path=cfg.DATA_PATH, synth_cfg=synth_cfg, seed=cfg.SEED,
        progress=progress,
    )
    print_report(result)

    if do_export:
        saved = export_results(result, cfg.RESULTS_DIR)
        print(f"\nСохранено в {cfg.RESULTS_DIR}:")
        for name in saved:
            print(f"  {name}")


def main():
    print("=" * 70)
    print("Вклад двутавра Qдв")
    print("=" * 70)
    try:
        while True:
            _run_once()
            if not _ask_yes_no("\nЗапустить ещё раз с другими параметрами?", False):
                break
    except (KeyboardInterrupt, EOFError):
        print("\nПрервано.")
        return
    if getattr(sys, "frozen", False):
        input("\nГотово. Enter — выход.")


if __name__ == "__main__":
    main()
