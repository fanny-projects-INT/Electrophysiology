from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget, QApplication, QScrollArea, QLabel,
    QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QPushButton, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


# -----------------------------
# Defaults (edit as you like)
# -----------------------------
DEFAULT_FS = 30000
DEFAULT_DTYPE = np.int16
DEFAULT_NUM_CHANNELS_BIN = 385  # 384 + sync
DEFAULT_PRE = 20
DEFAULT_POST = 40

# Auto-filter thresholds (IBL metrics file)
# ✅ If you set a threshold to NaN (or None), that criterion is DISABLED.
DEFAULT_MC_THRESH = 0.0
DEFAULT_NC_THRESH = 100.0
DEFAULT_AMP_THRESH_UV = np.nan  # default: disabled unless you set it


@dataclass
class QCParams:
    fs: int = DEFAULT_FS
    dtype: object = DEFAULT_DTYPE
    num_channels_bin: int = DEFAULT_NUM_CHANNELS_BIN
    pre: int = DEFAULT_PRE
    post: int = DEFAULT_POST

    # ✅ 100 clusters by default (no need to pass params in run)
    page_size: int = 100

    # ✅ set to np.nan to disable a criterion
    mc_thresh: float = DEFAULT_MC_THRESH         # max_confidence >=
    nc_thresh: float = DEFAULT_NC_THRESH         # noise_cutoff <
    amp_thresh_uv: float = DEFAULT_AMP_THRESH_UV # amp_median >= (µV)

    # Waveform display robustness / speed
    baseline_correct: bool = True
    max_wf_display: int = 100       # number of waveforms drawn per cluster
    max_bestchan_probe: int = 25    # spikes used to pick best channel

    # robust y-lims (envelope percentiles across waveforms, per timepoint)
    wf_env_low: float = 5.0
    wf_env_high: float = 95.0

    # progressive UI build to avoid freezes
    build_chunk: int = 6
    build_interval_ms: int = 0

    def __post_init__(self):
        try:
            self.dtype = np.dtype(self.dtype)
        except Exception as e:
            raise TypeError(
                f"QCParams.dtype invalid: {self.dtype!r}. Expected: np.int16, 'int16', etc."
            ) from e

        if self.mc_thresh is None:
            self.mc_thresh = np.nan
        if self.nc_thresh is None:
            self.nc_thresh = np.nan
        if self.amp_thresh_uv is None:
            self.amp_thresh_uv = np.nan


# -----------------------------
# Small helpers
# -----------------------------
def fmt(x, nd=2) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def fmt_uv(x) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.1f} µV"
    except Exception:
        return "NA"


def _is_enabled(x) -> bool:
    """Threshold enabled if it's a real number (not None/NaN)."""
    try:
        return not np.isnan(float(x))
    except Exception:
        return False


def find_metrics_file(alf_folder: Path) -> Path:
    candidates = [
        alf_folder / "cluster_metrics.csv",
        alf_folder / "clusters.metrics.csv",
        alf_folder / "clusters.metrics.tsv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing metrics file in {alf_folder}. Tried: {[c.name for c in candidates]}"
    )


def read_metrics(metrics_path: Path) -> tuple[pd.DataFrame, str]:
    delim = "\t" if metrics_path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(metrics_path, sep=delim)

    if "cluster_id" in df.columns:
        cid_col = "cluster_id"
    elif "cluster_ids" in df.columns:
        cid_col = "cluster_ids"
    else:
        raise KeyError(
            f"Missing cluster_id column in {metrics_path.name}. Columns={list(df.columns)}"
        )

    needed = ["max_confidence", "noise_cutoff", "amp_median"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing} in {metrics_path.name}. Columns={list(df.columns)}"
        )

    df[cid_col] = df[cid_col].astype(int)
    return df, cid_col


def labels_csv_path(alf_folder: Path) -> Path:
    return alf_folder / "clusters.labels.csv"


def _normalize_label_to_good_bad(lbl: str) -> str:
    """
    Normalize to {good, bad}.
    - "good" -> good
    - "bad"/"noise"/"mua" -> bad
    - other -> "" (ignored)
    """
    if lbl is None:
        return ""
    s = str(lbl).strip().lower()
    if s == "good":
        return "good"
    if s in {"bad", "noise", "mua"}:
        return "bad"
    return ""


def load_manual_labels(alf_folder: Path) -> dict[int, str]:
    p = labels_csv_path(alf_folder)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}

    if "cluster_id" not in df.columns or "label" not in df.columns:
        return {}

    out: dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            cid = int(r["cluster_id"])
            lab = _normalize_label_to_good_bad(r["label"])
            if lab:
                out[cid] = lab
        except Exception:
            pass
    return out


def export_all_cluster_labels_csv(alf_folder: Path, labels: dict[int, str], source: str = "auto_or_manual"):
    """
    Write clusters.labels.csv with ALL clusters.
    labels: dict {cluster_id -> "good"/"bad"}
    """
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for cid in sorted(labels.keys()):
        rows.append(
            {
                "cluster_id": int(cid),
                "label": _normalize_label_to_good_bad(labels[cid]) or "bad",
                "source": source,
                "timestamp": now,
            }
        )
    pd.DataFrame(rows).to_csv(labels_csv_path(alf_folder), index=False)


def ensure_labels_csv_complete(alf_folder: Path, labels: dict[int, str]) -> None:
    """
    Do NOT overwrite user choices.
    - If file does not exist: create it with all clusters.
    - If exists but missing clusters: rewrite a complete file using current 'labels'
      (already includes previous manual choices + defaults for missing).
    - If exists and complete: do nothing.
    """
    p = labels_csv_path(alf_folder)
    if not p.exists():
        export_all_cluster_labels_csv(alf_folder, labels, source="auto_init")
        return

    try:
        df = pd.read_csv(p)
        if "cluster_id" not in df.columns:
            export_all_cluster_labels_csv(alf_folder, labels, source="rebuild")
            return
        existing = set(int(x) for x in df["cluster_id"].values.tolist())
    except Exception:
        export_all_cluster_labels_csv(alf_folder, labels, source="rebuild")
        return

    needed = set(int(c) for c in labels.keys())
    if existing != needed:
        export_all_cluster_labels_csv(alf_folder, labels, source="complete_fill")


def export_good_clusters_npy(alf_folder: Path, labels: dict[int, str]):
    good = np.array(sorted([cid for cid, lab in labels.items() if lab == "good"]), dtype=int)
    np.save(alf_folder / "good_clusters.npy", good)


def status_dot(label: str) -> QLabel:
    dot = QLabel("●")
    dot.setAlignment(Qt.AlignCenter)
    dot.setFixedWidth(20)
    if label == "good":
        dot.setStyleSheet("font-size:20px; color:#2ecc71;")
    else:
        dot.setStyleSheet("font-size:20px; color:#e74c3c;")
    return dot


def compute_auto_pass_from_metrics(metrics_map: dict[int, pd.Series], params: QCParams) -> list[int]:
    """
    Recompute auto_pass from already-loaded metrics_map using current thresholds.
    Criteria can be disabled with NaN.
    """
    out: list[int] = []
    mc_on = _is_enabled(params.mc_thresh)
    nc_on = _is_enabled(params.nc_thresh)
    amp_on = _is_enabled(params.amp_thresh_uv)

    for cid, row in metrics_map.items():
        try:
            mc = float(row["max_confidence"])
            nc = float(row["noise_cutoff"])
            amp = float(row["amp_median"])
        except Exception:
            continue

        if mc_on and not (mc >= float(params.mc_thresh)):
            continue
        if nc_on and not (nc < float(params.nc_thresh)):
            continue
        if amp_on and not (amp >= float(params.amp_thresh_uv)):
            continue

        out.append(int(cid))

    return sorted(set(out))


# -----------------------------
# Probe context loader
# -----------------------------
def load_probe_ctx(base_folder: Path, probe: str, params: QCParams) -> dict:
    rec_folder = base_folder / "Rec" / probe
    alf_folder = base_folder / "alf" / probe

    if not rec_folder.exists():
        raise FileNotFoundError(f"Missing {rec_folder}")
    if not alf_folder.exists():
        raise FileNotFoundError(f"Missing {alf_folder}")

    probe_idx = int(probe.replace("probe", ""))  # probe00->0, probe01->1
    bin_file = rec_folder / f"disabled_g0_t0.imec{probe_idx}.ap.bin"

    spike_clusters_file = alf_folder / "spikes.clusters.npy"
    spike_times_file = alf_folder / "spikes.times.npy"
    spike_depths_file = alf_folder / "spikes.depths.npy"
    spike_samples_file = alf_folder / "spikes.samples.npy"  # optional

    for p in [bin_file, spike_clusters_file, spike_times_file, spike_depths_file]:
        if not p.exists():
            raise FileNotFoundError(f"[{probe}] Missing file: {p}")

    spike_clusters = np.load(spike_clusters_file).astype(int)
    spike_times = np.load(spike_times_file)
    spike_depths = np.load(spike_depths_file).astype(float)
    spike_samples = np.load(spike_samples_file).astype(int) if spike_samples_file.exists() else None

    all_clusters = np.unique(spike_clusters).astype(int)
    all_clusters_sorted = sorted([int(c) for c in all_clusters])

    metrics_path = find_metrics_file(alf_folder)
    mdf, cid_col = read_metrics(metrics_path)
    mdf = mdf[mdf[cid_col].isin(all_clusters)].copy()

    # metrics map
    mdf = mdf.set_index(cid_col)
    metrics_map = {int(cid): mdf.loc[cid] for cid in mdf.index}

    # auto-pass from metrics_map (criteria can be disabled with NaN)
    auto_pass = compute_auto_pass_from_metrics(metrics_map, params)

    # load bin (memmap)
    data = np.memmap(bin_file, dtype=params.dtype, mode="r")
    if data.size % params.num_channels_bin != 0:
        raise RuntimeError(
            f"[{probe}] data.size={data.size} not divisible by num_channels_bin={params.num_channels_bin}"
        )
    total_samples = data.size // params.num_channels_bin
    data = data.reshape((total_samples, params.num_channels_bin))
    sync_chan = params.num_channels_bin - 1

    # manual labels override (if clusters.labels.csv exists)
    manual_labels = load_manual_labels(alf_folder)

    auto_pass_set = set(auto_pass)
    labels_good_bad: dict[int, str] = {
        cid: ("good" if cid in auto_pass_set else "bad") for cid in all_clusters_sorted
    }
    labels_good_bad.update(manual_labels)

    return dict(
        probe=probe,
        base_folder=base_folder,
        rec_folder=rec_folder,
        alf_folder=alf_folder,
        bin_file=bin_file,

        spike_times=spike_times,
        spike_clusters=spike_clusters,
        spike_samples=spike_samples,
        spike_depths=spike_depths,
        all_clusters=all_clusters_sorted,

        metrics_path=metrics_path,
        metrics_map=metrics_map,

        auto_pass=auto_pass,
        labels=labels_good_bad,  # cid -> good/bad

        data=data,
        total_samples=total_samples,
        sync_chan=sync_chan,

        # perf: cache waveforms per cluster
        waveform_cache={},  # cid -> pack dict
    )


# -----------------------------
# Waveform cache pack (one cluster)
# -----------------------------
def get_cluster_wfpack(ctx: dict, cid: int, params: QCParams):
    cache = ctx["waveform_cache"]
    if cid in cache:
        return cache[cid]

    fs = params.fs
    pre = params.pre
    post = params.post
    window = pre + post

    spike_clusters = ctx["spike_clusters"]
    spike_times = ctx["spike_times"]
    spike_samples = ctx["spike_samples"]
    spike_depths = ctx["spike_depths"]
    data = ctx["data"]
    total_samples = ctx["total_samples"]
    sync_chan = ctx["sync_chan"]

    idx = np.where(spike_clusters == cid)[0]
    if idx.size == 0:
        return None

    # spread sampling if huge cluster
    if idx.size > params.max_wf_display:
        pick = np.linspace(0, idx.size - 1, params.max_wf_display).astype(int)
        idx_small = idx[pick]
    else:
        idx_small = idx

    d = spike_depths[idx]
    d = d[np.isfinite(d)]
    cluster_depth = float(np.median(d)) if d.size else np.nan

    if spike_samples is not None:
        st_small = spike_samples[idx_small]
    else:
        st_small = (spike_times[idx_small] * fs).astype(int)

    # best channel (quick)
    tmp = []
    for s in st_small[: params.max_bestchan_probe]:
        if pre < s < total_samples - post:
            tmp.append(data[s - pre:s + post, :])
    if not tmp:
        return None
    tmp = np.asarray(tmp)  # (n, window, ch)
    ptp = tmp.ptp(axis=1).mean(axis=0)
    ptp[sync_chan] = -np.inf
    best_chan = int(np.argmax(ptp))

    # waveforms on best_chan
    wfs = []
    for s in st_small:
        if pre < s < total_samples - post:
            wfs.append(data[s - pre:s + post, best_chan])
    if not wfs:
        return None
    wfs = np.asarray(wfs)  # (n, window)

    if params.baseline_correct and pre > 2:
        baseline = np.median(wfs[:, :pre], axis=1, keepdims=True)
        wfs = wfs - baseline

    wf_avg = wfs.mean(axis=0)

    # robust y-lims from envelope percentiles across waveforms per timepoint
    ylim = None
    try:
        low = np.percentile(wfs, params.wf_env_low, axis=0)
        high = np.percentile(wfs, params.wf_env_high, axis=0)
        lo, hi = float(np.min(low)), float(np.max(high))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.10 * (hi - lo)
            ylim = (lo - pad, hi + pad)
    except Exception:
        pass

    t_ms = np.arange(window) / fs * 1000

    pack = dict(
        idx=idx,
        cluster_depth=cluster_depth,
        best_chan=best_chan,
        wfs=wfs,
        wf_avg=wf_avg,
        t_ms=t_ms,
        ylim=ylim,
    )
    cache[cid] = pack
    return pack


# -----------------------------
# Waveform block (one cluster)
# -----------------------------
def make_cluster_widget(ctx: dict, cid: int, params: QCParams, on_label_change):
    pack = get_cluster_wfpack(ctx, int(cid), params)
    if pack is None:
        return None

    wfs = pack["wfs"]
    wf_avg = pack["wf_avg"]
    t_ms = pack["t_ms"]
    ylim = pack["ylim"]

    fs = params.fs
    pre = params.pre
    post = params.post
    window = pre + post

    spike_clusters = ctx["spike_clusters"]
    spike_depths = ctx["spike_depths"]
    probe = ctx["probe"]

    block = QWidget()
    layout = QHBoxLayout(block)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(20)

    # plot (vectorized)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(t_ms, wfs.T, alpha=0.10, color="#999999", linewidth=0.8)
    ax.plot(t_ms, wf_avg, linewidth=1.8, color="#e74c3c")
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(f"{probe} — Cluster {cid}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.6)

    canvas = FigureCanvasQTAgg(fig)
    plt.close(fig)  # ✅ critical: prevent matplotlib figure accumulation (lag/crash)

    # metrics
    row = ctx["metrics_map"].get(int(cid), None)
    mc = row["max_confidence"] if row is not None else None
    nc = row["noise_cutoff"] if row is not None else None
    amp = row["amp_median"] if row is not None else None

    labels = ctx["labels"]
    current = labels.get(int(cid), "bad")

    # header row: dot + GOOD/BAD
    top_row = QWidget()
    top_l = QHBoxLayout(top_row)
    top_l.setContentsMargins(0, 0, 0, 0)
    top_l.setSpacing(8)
    dot = status_dot(current)
    state_txt = QLabel(f"<b>{current.upper()}</b>")
    state_txt.setStyleSheet("font-size:14px;")
    top_l.addWidget(dot)
    top_l.addWidget(state_txt)
    top_l.addStretch(1)

    # cluster stats
    idx = pack["idx"]
    d = spike_depths[idx]
    d = d[np.isfinite(d)]
    cluster_depth = float(np.median(d)) if d.size else np.nan

    info = QLabel(
        f"""
<b style="font-size:15px;">Cluster {cid}</b><br>
<span style="font-size:14px;">
N spikes: {idx.size}<br>
Depth: {fmt(cluster_depth,1)} µm<br><br>
max_confidence: {fmt(mc,1)}<br>
noise_cutoff: {fmt(nc,2)}<br>
amp_median: {fmt_uv(amp)}<br>
</span>
"""
    )
    info.setAlignment(Qt.AlignTop)
    info.setStyleSheet("padding:6px;")

    # toggle button
    btn_toggle = QPushButton()
    btn_toggle.setMinimumWidth(190)
    btn_toggle.setStyleSheet("font-size:14px; padding:6px 10px;")

    def refresh_button_and_dot():
        cur = labels.get(int(cid), "bad")
        btn_toggle.setText("Mark as BAD" if cur == "good" else "Mark as GOOD")
        dot.setStyleSheet(
            "font-size:20px; color:#2ecc71;" if cur == "good"
            else "font-size:20px; color:#e74c3c;"
        )
        state_txt.setText(f"<b>{cur.upper()}</b>")

    def toggle_label():
        cur = labels.get(int(cid), "bad")
        new = "bad" if cur == "good" else "good"
        labels[int(cid)] = new

        export_all_cluster_labels_csv(ctx["alf_folder"], labels, source="manual_toggle")
        export_good_clusters_npy(ctx["alf_folder"], labels)

        refresh_button_and_dot()
        on_label_change()

    btn_toggle.clicked.connect(toggle_label)
    refresh_button_and_dot()

    right = QWidget()
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(8)
    right_layout.addWidget(top_row)
    right_layout.addWidget(info)
    right_layout.addWidget(btn_toggle)

    layout.addWidget(canvas)
    layout.addWidget(right)

    return block


# -----------------------------
# Per-probe tab with scroll + grid 2 columns, and default 100 clusters
# -----------------------------
class ProbeTab(QWidget):
    def __init__(self, ctx: dict, params: QCParams):
        super().__init__()
        self.ctx = ctx
        self.params = params

        self.filter_mode = "all"  # all | good | bad
        self.page = 0

        # progressive build state
        self._pending_cids: list[int] = []
        self._timer: QTimer | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        self.header = QLabel("")
        self.header.setStyleSheet("padding:6px; font-size:16px;")
        outer.addWidget(self.header)

        controls = QWidget()
        c = QHBoxLayout(controls)
        c.setContentsMargins(0, 0, 0, 0)
        c.setSpacing(10)

        self.combo = QComboBox()
        self.combo.addItems(["all", "good", "bad"])
        self.combo.setStyleSheet("font-size:14px; padding:4px;")
        self.combo.currentTextChanged.connect(self._on_filter_changed)

        self.btn_reset = QPushButton("Reset to Auto")
        self.btn_reset.setStyleSheet("font-size:14px; padding:6px 10px;")
        self.btn_reset.clicked.connect(self._reset_to_auto)

        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        for b in (self.btn_prev, self.btn_next):
            b.setStyleSheet("font-size:14px; padding:6px 10px;")
        self.btn_prev.clicked.connect(self._prev)
        self.btn_next.clicked.connect(self._next)

        self.page_lbl = QLabel("")
        self.page_lbl.setStyleSheet("padding:6px; font-size:14px;")

        c.addWidget(QLabel("Show:"))
        c.addWidget(self.combo)
        c.addWidget(self.btn_reset)
        c.addStretch(1)
        c.addWidget(self.btn_prev)
        c.addWidget(self.btn_next)
        c.addWidget(self.page_lbl)

        outer.addWidget(controls)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setContentsMargins(10, 10, 10, 10)
        self.grid.setHorizontalSpacing(20)
        self.grid.setVerticalSpacing(20)
        self.scroll.setWidget(self.container)

        self.refresh()

    def _reset_to_auto(self):
        metrics_map = self.ctx["metrics_map"]
        all_clusters = self.ctx["all_clusters"]

        auto_pass = compute_auto_pass_from_metrics(metrics_map, self.params)
        auto_set = set(auto_pass)

        labels = self.ctx["labels"]
        for cid in all_clusters:
            labels[int(cid)] = "good" if int(cid) in auto_set else "bad"

        self.ctx["auto_pass"] = auto_pass

        export_all_cluster_labels_csv(self.ctx["alf_folder"], labels, source="reset_auto")
        export_good_clusters_npy(self.ctx["alf_folder"], labels)

        self.refresh()

    def _get_filtered_cluster_list(self) -> list[int]:
        clusters = self.ctx["all_clusters"]
        labels = self.ctx["labels"]
        if self.filter_mode == "all":
            return clusters
        return [cid for cid in clusters if labels.get(int(cid), "bad") == self.filter_mode]

    def _on_filter_changed(self, txt: str):
        self.filter_mode = txt
        self.page = 0
        self.refresh()

    def _prev(self):
        if self.page > 0:
            self.page -= 1
            self.refresh()

    def _next(self):
        clusters = self._get_filtered_cluster_list()
        max_page = max(0, (len(clusters) - 1) // self.params.page_size)
        if self.page < max_page:
            self.page += 1
            self.refresh()

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def refresh(self):
        # stop previous progressive build
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        self.setUpdatesEnabled(False)
        try:
            self._clear_grid()

            probe = self.ctx["probe"]
            labels = self.ctx["labels"]
            all_clusters = self.ctx["all_clusters"]
            auto_pass = set(self.ctx["auto_pass"])

            n_good = sum(1 for cid in all_clusters if labels.get(cid, "bad") == "good")
            n_bad = len(all_clusters) - n_good
            n_auto_good = len(auto_pass)

            crit = []
            crit.append(f"mc={'ON' if _is_enabled(self.params.mc_thresh) else 'OFF'}")
            crit.append(f"nc={'ON' if _is_enabled(self.params.nc_thresh) else 'OFF'}")
            crit.append(f"amp={'ON' if _is_enabled(self.params.amp_thresh_uv) else 'OFF'}")
            crit_txt = ", ".join(crit)

            self.header.setText(
                f"<b>{probe}</b> — Clusters: {len(all_clusters)} | Good={n_good}, Bad={n_bad} | "
                f"Auto-pass: {n_auto_good}<br>"
                f"Criteria: {crit_txt} (metrics: {self.ctx['metrics_path'].name})"
            )

            clusters = self._get_filtered_cluster_list()
            if not clusters:
                self.page_lbl.setText("Page 0/0")
                empty = QLabel("No clusters to display for this filter.")
                empty.setStyleSheet("padding:10px; font-size:14px;")
                self.grid.addWidget(empty, 0, 0)
                return

            max_page = (len(clusters) - 1) // self.params.page_size
            self.page = max(0, min(self.page, max_page))

            start = self.page * self.params.page_size
            end = min(len(clusters), start + self.params.page_size)
            page_clusters = clusters[start:end]

            self.page_lbl.setText(f"Page {self.page+1}/{max_page+1} — showing {len(page_clusters)} clusters")

            # reset scroll to top when changing page/filter
            self.scroll.verticalScrollBar().setValue(0)

            self._pending_cids = list(page_clusters)

        finally:
            self.setUpdatesEnabled(True)

        def on_label_change():
            # simple: rebuild current view
            self.refresh()

        # progressive populate to avoid freezing UI
        self._timer = QTimer(self)
        self._timer.setInterval(self.params.build_interval_ms)

        def add_chunk():
            if not self._pending_cids:
                if self._timer is not None:
                    self._timer.stop()
                    self._timer = None
                return

            self.setUpdatesEnabled(False)
            try:
                n = self.params.build_chunk
                for _ in range(n):
                    if not self._pending_cids:
                        break
                    cid = int(self._pending_cids.pop(0))
                    w = make_cluster_widget(self.ctx, cid, self.params, on_label_change)
                    if w is None:
                        continue
                    i = self.grid.count()
                    row = i // 2
                    col = i % 2
                    self.grid.addWidget(w, row, col)
            finally:
                self.setUpdatesEnabled(True)

        self._timer.timeout.connect(add_chunk)
        self._timer.start()


# -----------------------------
# Top-level Dashboard with probe tabs
# -----------------------------
class Dashboard(QWidget):
    def __init__(self, base_folder: Path, probe_contexts: dict[str, dict], params: QCParams):
        super().__init__()
        self.setWindowTitle(f"QC Labeler — {base_folder.name}")
        self.resize(1700, 1050)

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        for probe, ctx in probe_contexts.items():
            tabs.addTab(ProbeTab(ctx, params), probe)


def run_qc_labeler(session_id: str, data_root: Path, params: QCParams | None = None):
    """
    Auto-label:
      - good = auto-pass
      - bad  = others
    UI lets you toggle good/bad.
    Writes:
      alf/probeXX/clusters.labels.csv  (complete; NOT overwritten if already complete)
      alf/probeXX/good_clusters.npy
    """
    if params is None:
        params = QCParams()  # ✅ default page_size=100 here

    base_folder = data_root / session_id
    rec_root = base_folder / "Rec"

    if not (rec_root / "probe00").exists():
        raise FileNotFoundError(f"Missing Rec/probe00 in {base_folder}")

    probes = ["probe00"]
    if (rec_root / "probe01").exists():
        probes.append("probe01")

    probe_contexts = {p: load_probe_ctx(base_folder, p, params) for p in probes}

    # Do NOT overwrite at startup:
    for _, ctx in probe_contexts.items():
        ensure_labels_csv_complete(ctx["alf_folder"], ctx["labels"])
        export_good_clusters_npy(ctx["alf_folder"], ctx["labels"])

    app = QApplication(sys.argv)
    w = Dashboard(base_folder, probe_contexts, params)
    w.show()
    sys.exit(app.exec_())


# -----------------------------
# Example entry point
# -----------------------------
if __name__ == "__main__":
    SESSION_ID = "VF071_2025_12_18"
    DATA_ROOT = Path(r"D:\Data_Mice")
    run_qc_labeler(SESSION_ID, DATA_ROOT)
