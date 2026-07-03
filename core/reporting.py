from datetime import datetime
from core.labels import method_label, target_label, Q_UNIT


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def print_report(result):
    for tkey, table in result.metrics.items():
        print(f"\n=== {target_label(tkey)} ===")
        if table.empty:
            print("  (нет моделей — только методы вывода формулы)")
        else:
            disp = table.copy()
            disp.index = [method_label(m) for m in disp.index]
            print(disp.round(4).to_string())

        formulas = result.formulas.get(tkey, {})
        if formulas:
            print("\n  Выведенные формулы:")
            for name, expr in formulas.items():
                print(f"  [{method_label(name)}] {expr}")


def build_comparison_figure(result, tkey, scatter_model=None):
    plt = _pyplot()
    preds = result.predictions.get(tkey, {})
    metrics_df = result.metrics.get(tkey)
    fig = plt.figure(figsize=(11, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    tlabel = target_label(tkey)

    if scatter_model is None and metrics_df is not None and not metrics_df.empty:
        scatter_model = metrics_df["RMSE"].idxmin()

    if scatter_model in preds:
        yt = preds[scatter_model]["y_true"]
        yp = preds[scatter_model]["y_pred"]
        ax1.scatter(yt, yp, s=30)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax1.plot([lo, hi], [lo, hi], "--", color="gray", label="идеал")
        ax1.set_xlabel(f"Qэксп, {Q_UNIT}")
        ax1.set_ylabel(f"Qпред, {Q_UNIT}")
        ax1.set_title(f"{method_label(scatter_model)}\n{tlabel}", fontsize=10)
        ax1.legend(fontsize=8)

    if metrics_df is not None and not metrics_df.empty and "RMSE" in metrics_df:
        df = metrics_df.sort_values("RMSE", ascending=False)
        labels = [method_label(m) for m in df.index]
        bars = ax2.barh(range(len(labels)), df["RMSE"].to_numpy())
        bars[-1].set_color("#2a9d3f")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=8)
        ax2.set_xlabel(f"RMSE, {Q_UNIT} (меньше — лучше)")
        ax2.set_title(f"Сравнение методов / {tlabel}", fontsize=10)

    fig.tight_layout()
    return fig

def export_results(result, out_dir, stamp=None):
    plt = _pyplot()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = []

    for tkey, df in result.metrics.items():
        if df.empty:
            continue
        path = out_dir / f"metrics_{tkey}_{stamp}.csv"
        df.to_csv(path, encoding="utf-8-sig")
        saved.append(path.name)
        fig = build_comparison_figure(result, tkey)
        fig_path = out_dir / f"plots_{tkey}_{stamp}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        saved.append(fig_path.name)
    return saved
