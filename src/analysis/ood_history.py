"""Read back the OOD probe curve a training run recorded, and find where it peaked.

The question this exists to answer: the training NLL keeps improving for hours, but does
OOD prediction keep improving with it, or does it peak early and drift? The probe
(`src.analysis.ood_probe`) writes a score every `--ood_probe_every` epochs; this reads
those back and puts the peak epoch of each metric next to the NLL at that moment, so
"still improving" and "peaked at epoch 2000 and decayed" are distinguishable.

Two sources, because they carry different things:

  `from_checkpoint` - `ood_history` in any checkpoint written by a run with a probe. Full
      fidelity, including the per-combo breakdown, but the file is ~7.6 GB and has to be
      loaded whole (`weights_only=True` cannot read the config), so it is freed as soon
      as the history is out. One at a time on a 16 GB allocation.

  `from_log` - the `OOD[...]` lines in the run's log. Lossy for older runs (a single
      probe combo, no IQR) but it is what exists for runs that predate `ood_history`
      being stored, and it carries the NLL from the surrounding `Epoch` lines, which the
      checkpoint's `metrics_loss` also has but keyed differently.

`var_ratio` is scored against a target of 1.0 rather than maximized: it is the ratio of
predicted to real variance, so both 0.5 and 2.0 are failures in opposite directions.
"""

import gc
import os
import re

import numpy as np
import pandas as pd

# metric -> how "best" is defined. `target` means closest to that value.
_DIRECTION = {
    "mmd": "min",
    "r2_top50": "max",
    "cos2_top50": "max",
    "accuracy": "max",
    "var_ratio": 1.0,
}

_EPOCH_RE = re.compile(r"Epoch (\d+):.*?NLL: ([-\d.eE+]+)")
_PROBE_RE = re.compile(
    r"OOD\[(?P<combo>.*?)\] MMD: (?P<mmd>[\d.eE+-]+) \| "
    r"R2_top50: (?P<r2_top50>[-\d.eE+]+) \| var_ratio: (?P<var_ratio>[\d.eE+-]+) \| "
    r"acc: (?P<accuracy>[\d.]+)%"
    r"(?: \(n=(?P<n_combos>\d+), MMD IQR (?P<iqr_lo>[\d.eE+-]+)-(?P<iqr_hi>[\d.eE+-]+)\))?"
)
_PER_COMBO_RE = re.compile(
    r"^\s+(?P<name>\S+): MMD (?P<mmd>[\d.eE+-]+) \| R2 (?P<r2_top50>[-\d.eE+]+) \| "
    r"var_ratio (?P<var_ratio>[\d.eE+-]+) \| acc (?P<accuracy>[\d.]+)%"
)


def from_checkpoint(path, device="cpu"):
    """`(history, nll)` from a checkpoint's `ood_history` and `metrics_loss`.

    `nll` is a DataFrame indexed by epoch, or None when the checkpoint predates the
    probe. The checkpoint is freed before returning - it is large enough that holding two
    of them is an OOM on a 16 GB allocation.
    """
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # `is not None` rather than truthiness throughout: metrics_loss comes back holding
    # numpy arrays, and `if array:` on more than one element raises rather than being
    # falsy - so `x or []` and `if metrics.get(...)` both blow up on a real checkpoint.
    stored = checkpoint.get("ood_history")
    history = list(stored) if stored is not None else []

    nll = nll_frame(checkpoint.get("metrics_loss"))

    del checkpoint
    gc.collect()
    return history, nll


def nll_frame(metrics_loss):
    """The NLL curve out of a run's `metrics_loss`, indexed by epoch, or None.

    Split out so a notebook that already has `metrics_loss` in memory can hand it to
    `analyze` instead of loading the checkpoint a second time.
    """
    metrics = metrics_loss if metrics_loss is not None else {}
    epochs = np.asarray(metrics.get("epoch", []))
    nlls = np.asarray(metrics.get("NLL", []))
    if not (epochs.size and nlls.size):
        return None
    n = min(epochs.size, nlls.size)
    return pd.DataFrame({"epoch": epochs[:n], "NLL": nlls[:n]}).set_index("epoch")


def from_log(path):
    """`(history, nll)` parsed from a training log's `OOD[...]` and `Epoch` lines.

    Probes are logged after the epoch they follow, so each one is attributed to the most
    recent `Epoch` line - the same pairing the training loop wrote them in.
    """
    history, nll_rows = [], []
    epoch = nll = None
    with open(path, errors="ignore") as handle:
        for line in handle:
            match = _EPOCH_RE.search(line)
            if match:
                epoch, nll = int(match.group(1)), float(match.group(2))
                nll_rows.append({"epoch": epoch, "NLL": nll})
                continue
            match = _PER_COMBO_RE.match(line.split(" - INFO - ")[-1])
            if match and history:
                entry = match.groupdict()
                # popped before the comprehension, not inside the subscript: an
                # assignment evaluates its right-hand side first, so `per[entry.pop(...)]`
                # would still have "name" in `entry` and try float("combo_1")
                name = entry.pop("name")
                per = history[-1].setdefault("per_combo", {})
                per[name] = {
                    k: float(v) / (100.0 if k == "accuracy" else 1.0)
                    for k, v in entry.items()
                }
                continue
            match = _PROBE_RE.search(line)
            if match:
                found = match.groupdict()
                record = {
                    "epoch": epoch,
                    "combo": found["combo"],
                    "accuracy": float(found["accuracy"]) / 100.0,
                }
                for key in ("mmd", "r2_top50", "var_ratio"):
                    record[key] = float(found[key])
                if found["n_combos"]:
                    record["n_combos"] = int(found["n_combos"])
                    record["mmd_iqr"] = [float(found["iqr_lo"]), float(found["iqr_hi"])]
                history.append(record)
    nll_frame = pd.DataFrame(nll_rows).drop_duplicates("epoch").set_index("epoch")
    return history, (nll_frame if len(nll_frame) else None)


def to_frame(history):
    """One row per probe: epoch plus every aggregate metric it recorded."""
    rows = []
    for entry in history:
        row = {k: v for k, v in entry.items() if k not in ("per_combo", "mmd_iqr", "r2_top50_iqr")}
        if "mmd_iqr" in entry:
            row["mmd_iqr_lo"], row["mmd_iqr_hi"] = entry["mmd_iqr"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.set_index("epoch").sort_index() if len(frame) else frame


def per_combo_frame(history):
    """Long form: one row per (epoch, combo). Empty for a single-combo probe, which has
    no breakdown to give."""
    rows = []
    for entry in history:
        for name, metrics in (entry.get("per_combo") or {}).items():
            rows.append({"epoch": entry["epoch"], "combo": name, **metrics})
    frame = pd.DataFrame(rows)
    return frame.sort_values(["combo", "epoch"]) if len(frame) else frame


def _best_index(series):
    """Index of the best value, by this metric's own definition of best."""
    direction = _DIRECTION.get(series.name, "max")
    if direction == "min":
        return series.idxmin()
    if direction == "max":
        return series.idxmax()
    return (series - direction).abs().idxmin()  # closest to target


def peak_summary(history, nll=None):
    """Where each metric peaked, and what the NLL was doing at that point.

    `peak_frac` is the peak's position in the run (1.0 = the last probe). That is the
    number the "does it peak early?" question actually turns on: a metric peaking at 0.35
    while the NLL is still falling at 1.0 is the divergence worth acting on, and one
    peaking at 0.99 is not.
    """
    frame = to_frame(history)
    if not len(frame):
        return pd.DataFrame()
    epochs = frame.index.to_numpy()
    span = epochs.max() - epochs.min()
    rows = []
    for metric in _DIRECTION:
        if metric not in frame:
            continue
        series = frame[metric].dropna()
        if series.empty:
            continue
        peak_epoch = _best_index(series)
        row = {
            "metric": metric,
            "best": series.loc[peak_epoch],
            "peak_epoch": peak_epoch,
            "peak_frac": (peak_epoch - epochs.min()) / span if span else 1.0,
            "final": series.iloc[-1],
        }
        row["decay_from_peak"] = row["best"] - row["final"]
        if nll is not None and len(nll):
            nearest = nll.index.to_numpy()
            row["NLL_at_peak"] = float(
                nll["NLL"].iloc[int(np.abs(nearest - peak_epoch).argmin())]
            )
            row["NLL_final"] = float(nll["NLL"].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def combo_peak_summary(history):
    """Per-combo peak epoch for each metric - whether the ten combos turn together or the
    aggregate peak is one combo's noise."""
    frame = per_combo_frame(history)
    if not len(frame):
        return pd.DataFrame()
    rows = []
    for combo, group in frame.groupby("combo"):
        group = group.set_index("epoch")
        row = {"combo": combo}
        for metric in _DIRECTION:
            if metric in group:
                row[f"{metric}_peak"] = _best_index(group[metric].dropna())
        rows.append(row)
    return pd.DataFrame(rows).set_index("combo").sort_index()


def analyze(source, nll=None, plot_path=None, verbose=True):
    """Entry point: takes a checkpoint path, a log path, or a history list.

    `nll` supplies the NLL curve when `source` is a bare history list - pass
    `nll_frame(metrics_loss)` from a notebook that already has it, rather than reloading
    the checkpoint. Ignored when `source` is a path, which carries its own.

    Returns a dict with the frames, and prints the summary when `verbose`.
    """
    if isinstance(source, (str, os.PathLike)):
        loader = from_log if str(source).endswith(".log") else from_checkpoint
        history, nll = loader(source)
    else:
        history = list(source) if source is not None else []

    result = {
        "history": history,
        "nll": nll,
        "frame": to_frame(history),
        "per_combo": per_combo_frame(history),
        "peaks": peak_summary(history, nll),
        "combo_peaks": combo_peak_summary(history),
    }

    if verbose:
        frame = result["frame"]
        if not len(frame):
            print("No OOD probe records found - was the run trained with a probe?")
            return result
        label = history[0].get("combo", "?")
        print(f"{len(frame)} probes over epochs {frame.index.min()}-{frame.index.max()} "
              f"on {label}")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(result["peaks"].round(4))
            if len(result["combo_peaks"]):
                print("\nper-combo peak epochs:")
                print(result["combo_peaks"])
        if nll is not None and len(nll):
            print(f"\nNLL {nll['NLL'].iloc[0]:.1f} -> {nll['NLL'].iloc[-1]:.1f} "
                  f"(best {nll['NLL'].min():.1f} @ epoch {nll['NLL'].idxmin()})")

    if plot_path:
        plot(history, nll, plot_path)
        result["plot"] = plot_path
    return result


def plot(history, nll=None, path=None):
    """Each metric against epoch, with the NLL on a twin axis - the divergence this
    module exists to show is only visible with the two on the same x."""
    import matplotlib.pyplot as plt

    frame = to_frame(history)
    metrics = [m for m in _DIRECTION if m in frame and frame[m].notna().any()]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.6 * len(metrics)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics):
        ax.plot(frame.index, frame[metric], color="steelblue", marker="o", ms=3, lw=1.2)
        if metric == "mmd" and "mmd_iqr_lo" in frame:
            ax.fill_between(frame.index, frame["mmd_iqr_lo"], frame["mmd_iqr_hi"],
                            color="steelblue", alpha=0.18, lw=0)
        peak = _best_index(frame[metric].dropna())
        ax.axvline(peak, color="crimson", ls="--", lw=1)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        if metric == "var_ratio":
            ax.axhline(1.0, color="gray", ls=":", lw=1)
        if nll is not None and len(nll):
            twin = ax.twinx()
            twin.plot(nll.index, nll["NLL"], color="darkorange", lw=1, alpha=0.6)
            twin.set_ylabel("NLL", color="darkorange")

    axes[-1].set_xlabel("epoch")
    axes[0].set_title("OOD probe vs NLL (red = peak, orange = NLL)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
