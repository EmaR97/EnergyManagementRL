import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from ...utility import get_logger

logger = get_logger(__name__)

DAY = 288


def plot_gap_fills(before, after, config, date_suffix=""):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(
        config["data_paths"].get("simulation_inputs", "../data/simulation_inputs"),
        "gap_plots",
    )
    if date_suffix:
        plots_dir = os.path.join(plots_dir, date_suffix)
    os.makedirs(plots_dir, exist_ok=True)

    any_nan = before.isna().any(axis=1)
    cols = [
        "production_power_kw",
        "load_power_kw",
        "GRID_VOLTAGE",
        "SOC",
        "stored_power_kw",
        "grid_power_kw",
    ]
    titles = ["Production (kW)", "Load (kW)", "Voltage (V)", "SOC (%)", "Stored (kW)", "Grid (kW)"]
    window_days = 3
    width = max(1, len(str(len(before))))

    pos = 0
    n = len(before)
    while pos < n:
        if not any_nan.iloc[pos]:
            pos += 1
            continue

        gap_start = pos
        while pos < n and any_nan.iloc[pos]:
            pos += 1
        gap_end = pos

        if gap_end - gap_start <= 1:
            continue

        w0 = max(0, gap_start - DAY * window_days)
        w1 = min(n, gap_end + DAY * window_days)
        idx = before.index[w0:w1]

        fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
        fig.suptitle(
            f"Gap  {before.index[gap_start]:%Y-%m-%d %H:%M}  \u2192  {before.index[min(gap_end, n) - 1]:%Y-%m-%d %H:%M}  ({gap_end - gap_start} rows)",
            fontsize=13,
        )

        for (ax_grp, col, title) in zip(axes.flatten(), cols, titles):
            if col not in before.columns:
                ax_grp.set_visible(False)
                continue
            ax_grp.plot(
                idx,
                before[col].iloc[w0:w1].values,
                color="#e74c3c",
                alpha=0.5,
                linewidth=0.8,
                label="Before fill",
            )
            ax_grp.plot(
                idx,
                after[col].iloc[w0:w1].values,
                color="#2980b9",
                alpha=0.8,
                linewidth=1,
                label="After fill",
            )
            ax_grp.axvspan(
                before.index[gap_start],
                before.index[min(gap_end, n) - 1],
                alpha=0.12,
                color="gray",
            )
            ax_grp.set_ylabel(title, fontsize=9)
            ax_grp.legend(fontsize=7, loc="upper right")
            ax_grp.grid(True, alpha=0.25)

        axes[-1, -1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        plt.tight_layout()

        fname = (
            f"{gap_end - gap_start:0{width}d}_"
            f"gap_{before.index[gap_start]:%Y%m%d_%H%M}_"
            f"{before.index[min(gap_end, n) - 1]:%Y%m%d_%H%M}.png"
        )
        fig.savefig(os.path.join(plots_dir, fname), dpi=150)
        plt.close(fig)

    n_plots = len([f for f in os.listdir(plots_dir) if f.endswith(".png")])
    logger.info(f"Saved {n_plots} gap-plot(s) to {plots_dir}")
