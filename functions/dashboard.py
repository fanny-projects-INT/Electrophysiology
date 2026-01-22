from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


# ============================================================
# Config defaults (can be overridden from run script)
# ============================================================
DEFAULTS: Dict[str, Any] = {
    "LABELS_NAME": "clusters.labels.csv",
    "OUT_DIRNAME": "qc_dashboard_png",

    # NEW: if set (Path or str), all PNGs go there
    # If None -> old behavior (per session / OUT_DIRNAME)
    "OUT_ROOT": None,

    # canvas
    "OUT_W_PX": 1400,
    "OUT_H_PX": 800,
    "OUT_DPI": 150,

    # params
    "DEPTH_BINS": 70,
    "DEPTH_SMOOTH_SIGMA": 2.5,

    "TIME_BINS": 320,
    "TIME_SMOOTH_SIGMA": 2.0,

    "HEAT_TBINS": 180,
    "HEAT_DBINS": 140,
    "HEAT_MAX_POINTS": 700_000,
    "HEAT_CLIP_PERCENTILE": 99.6,
    "HEAT_TIME_UNIT": "min",  # "s" or "min"

    "FR_BINS": 40,
    "FR_SMOOTH_SIGMA": 1.3,

    # style
    "BLUE": "#2A6FDB",
    "TITLE_SIZE": 10.5,
    "FONT_SIZE": 10,
}


# ============================================================
# Style
# ============================================================
def apply_style(cfg: Dict[str, Any]) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E6E6E6",
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "font.size": cfg["FONT_SIZE"],
        "axes.titleweight": "normal",
        "axes.titlesize": cfg["TITLE_SIZE"],
        "axes.labelsize": cfg["FONT_SIZE"],
    })


# ============================================================
# IO
# ============================================================
def load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        return None


def read_labels(labels_csv: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(labels_csv)
    except Exception:
        return None

    if "label" not in df.columns or "cluster_id" not in df.columns:
        return None

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def find_probes_for_session(root: Path, session: str, labels_name: str) -> List[Path]:
    session_folder = root / session
    if not session_folder.exists():
        return []
    csvs = list(session_folder.rglob(labels_name))
    return sorted({c.parent for c in csvs})


def load_probe_data(alf_probe: Path, labels_name: str):
    labels = read_labels(alf_probe / labels_name)
    if labels is None:
        return None

    clusters = load_npy(alf_probe / "spikes.clusters.npy")
    depths   = load_npy(alf_probe / "spikes.depths.npy")
    times    = load_npy(alf_probe / "spikes.times.npy")

    if clusters is None or depths is None or times is None:
        return None

    clusters = np.asarray(clusters).astype(int, copy=False)
    depths = np.asarray(depths).astype(float, copy=False)
    times = np.asarray(times).astype(float, copy=False)

    if not (clusters.shape[0] == depths.shape[0] == times.shape[0]):
        return None

    return labels, clusters, depths, times


# ============================================================
# Helpers / metrics
# ============================================================
def gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y
    half = int(np.ceil(3 * sigma))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    ypad = np.pad(y, (half, half), mode="reflect")
    return np.convolve(ypad, k, mode="valid")


def style_axes(ax):
    ax.grid(alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def good_cluster_ids(labels: pd.DataFrame) -> np.ndarray:
    return labels.loc[labels["label"] == "good", "cluster_id"].to_numpy(dtype=int)


def compute_good_unit_depths(labels: pd.DataFrame, clusters: np.ndarray, depths: np.ndarray) -> np.ndarray:
    gids = good_cluster_ids(labels)
    out = []
    for cid in gids:
        d = depths[clusters == cid]
        d = d[np.isfinite(d)]
        if d.size:
            out.append(float(np.median(d)))
    return np.asarray(out, dtype=float)


def depth_density(good_depths: np.ndarray, bins: int, smooth_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(good_depths, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = counts.astype(float)
    if dens.max() > 0:
        dens /= dens.max()
    dens = gaussian_smooth(dens, sigma=smooth_sigma)
    if dens.max() > 0:
        dens /= dens.max()
    return centers, dens


def good_spike_times(times: np.ndarray, clusters: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    gids = good_cluster_ids(labels)
    if gids.size == 0:
        return np.array([], dtype=float)
    m = np.isin(clusters, gids)
    t = times[m]
    t = t[np.isfinite(t)]
    return t.astype(float)


def spike_density_over_time(spike_times: np.ndarray, n_bins: int, smooth_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    if spike_times.size == 0:
        return np.array([]), np.array([])
    tmax = float(np.nanmax(spike_times))
    edges = np.linspace(0.0, tmax, n_bins + 1)
    counts, _ = np.histogram(spike_times, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = gaussian_smooth(counts.astype(float), sigma=smooth_sigma)
    return centers, y


def get_good_spikes_time_depth(times: np.ndarray, clusters: np.ndarray, depths: np.ndarray, labels: pd.DataFrame):
    gids = good_cluster_ids(labels)
    if gids.size == 0:
        return np.array([]), np.array([])
    m = np.isin(clusters, gids)
    t = times[m]
    d = depths[m]
    ok = np.isfinite(t) & np.isfinite(d)
    return t[ok].astype(float), d[ok].astype(float)


def compute_firing_rates_good_units(times: np.ndarray, clusters: np.ndarray, labels: pd.DataFrame) -> np.ndarray:
    gids = good_cluster_ids(labels)
    if gids.size == 0:
        return np.array([], dtype=float)

    rates = []
    for cid in gids:
        t = times[clusters == cid]
        t = t[np.isfinite(t)]
        if t.size < 2:
            continue
        duration = float(t.max() - t.min())
        if duration <= 0:
            continue
        rates.append(float(t.size / duration))
    return np.asarray(rates, dtype=float)


# ============================================================
# 1 dashboard per probe
# ============================================================
def make_dashboard_png(
    session: str,
    alf_probe: Path,
    out_png: Path,
    cfg: Dict[str, Any],
) -> bool:
    loaded = load_probe_data(alf_probe, cfg["LABELS_NAME"])
    if loaded is None:
        return False

    labels, clusters, depths, times = loaded
    blue = cfg["BLUE"]

    n_good = int((labels["label"] == "good").sum())
    duration_s = float(np.nanmax(times)) if times.size else np.nan
    duration_min = duration_s / 60.0 if np.isfinite(duration_s) else np.nan

    good_depths = compute_good_unit_depths(labels, clusters, depths)
    t_good = good_spike_times(times, clusters, labels)
    t_cent, t_count = spike_density_over_time(t_good, cfg["TIME_BINS"], cfg["TIME_SMOOTH_SIGMA"])
    ht, hd = get_good_spikes_time_depth(times, clusters, depths, labels)

    fr = compute_firing_rates_good_units(times, clusters, labels)
    fr = fr[np.isfinite(fr) & (fr > 0)]

    fig_w_in = cfg["OUT_W_PX"] / cfg["OUT_DPI"]
    fig_h_in = cfg["OUT_H_PX"] / cfg["OUT_DPI"]
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=cfg["OUT_DPI"], constrained_layout=True)

    # Layout tuned: top row taller (FR less cramped) + spike density a bit lower
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[0.42, 0.92, 1.20],
        width_ratios=[0.75, 1.10, 1.10],
        hspace=0.18,
        wspace=0.18,
    )

    # ---- info (auto-fit box around text)
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis("off")

    info_text = (
        f"mouse_ID : {session}\n"
        f"probe     : {alf_probe.name}\n"
        f"good units: {n_good}\n"
        f"duration  : {duration_min:.1f} min"
        if np.isfinite(duration_min) else
        f"mouse_ID : {session}\nprobe     : {alf_probe.name}\ngood units: {n_good}\nduration  : NA"
    )

    ax_info.text(
        0.0, 0.95, info_text,
        ha="left", va="top",
        fontsize=cfg["FONT_SIZE"],
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#F7F7F7",
            edgecolor="#E6E6E6",
            linewidth=0.8
        )
    )

    # ---- firing rate
    ax_fr = fig.add_subplot(gs[0, 1:])
    if fr.size >= 2:
        bins = np.logspace(np.log10(fr.min()), np.log10(fr.max()), cfg["FR_BINS"])
        hist, edges = np.histogram(fr, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        hist_s = gaussian_smooth(hist, sigma=cfg["FR_SMOOTH_SIGMA"])

        ax_fr.plot(centers, hist_s, color=blue, lw=1.9)
        ax_fr.fill_between(centers, 0, hist_s, color=blue, alpha=0.15)
        ax_fr.set_xscale("log")

        ax_fr.set_title("Firing rate distribution (good units)", pad=4)
        ax_fr.set_xlabel("Firing rate (Hz)")
        ax_fr.set_ylabel("Density")
        style_axes(ax_fr)
    else:
        ax_fr.axis("off")

    # ---- depth density
    ax_depth = fig.add_subplot(gs[1:, 0])
    if good_depths.size:
        y, x = depth_density(good_depths, cfg["DEPTH_BINS"], cfg["DEPTH_SMOOTH_SIGMA"])
        ax_depth.fill_betweenx(y, 0, x, color=blue, alpha=0.15)
        ax_depth.plot(x, y, color=blue, lw=2.0)
        ax_depth.set_ylim(y.max(), y.min())
        ax_depth.set_xlim(0, 1.0)

    ax_depth.set_title("Good unit density along depth", pad=4)
    ax_depth.set_xlabel("Norm. density")
    ax_depth.set_ylabel("Depth (µm)")
    style_axes(ax_depth)

    # ---- spike density over time
    ax_time = fig.add_subplot(gs[1, 1:])
    if t_cent.size:
        ax_time.plot(t_cent, t_count, color=blue, lw=1.9)
        ax_time.fill_between(t_cent, 0, t_count, color=blue, alpha=0.12)

    ax_time.set_title("Good spike density over time", pad=4)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Count / bin")
    style_axes(ax_time)

    # ---- heatmap
    ax_hm = fig.add_subplot(gs[2, 1:])
    if ht.size:
        if ht.size > cfg["HEAT_MAX_POINTS"]:
            rng = np.random.default_rng(0)
            idx = rng.choice(ht.size, size=cfg["HEAT_MAX_POINTS"], replace=False)
            ht2 = ht[idx]
            hd2 = hd[idx]
        else:
            ht2, hd2 = ht, hd

        if cfg["HEAT_TIME_UNIT"] == "min":
            ht2 = ht2 / 60.0
            xlabel = "Time (min)"
        else:
            xlabel = "Time (s)"

        t_edges = np.linspace(0.0, float(np.max(ht2)), cfg["HEAT_TBINS"] + 1)
        d_edges = np.linspace(float(np.min(hd2)), float(np.max(hd2)), cfg["HEAT_DBINS"] + 1)

        H, _, _ = np.histogram2d(ht2, hd2, bins=[t_edges, d_edges])
        Z = H.T

        pos = Z[Z > 0]
        if pos.size:
            vmax = float(np.percentile(pos, cfg["HEAT_CLIP_PERCENTILE"]))
            vmax = max(vmax, 1.0)

            norm = LogNorm(vmin=1, vmax=vmax)
            cmap = mpl.cm.magma.copy()
            cmap.set_bad("black")

            im = ax_hm.imshow(
                Z,
                aspect="auto",
                origin="lower",
                extent=[t_edges[0], t_edges[-1], d_edges[0], d_edges[-1]],
                cmap=cmap,
                norm=norm,
                interpolation="bilinear",
            )
            ax_hm.set_ylim(d_edges[-1], d_edges[0])

            ax_hm.set_title("Good spikes (time × depth)", pad=4)
            ax_hm.set_xlabel(xlabel)
            ax_hm.set_ylabel("Depth (µm)")
            ax_hm.spines["top"].set_visible(False)
            ax_hm.spines["right"].set_visible(False)

            cbar = fig.colorbar(im, ax=ax_hm, fraction=0.030, pad=0.018)
            cbar.outline.set_visible(False)
            cbar.set_label("Count (log)", rotation=90)
        else:
            ax_hm.axis("off")
    else:
        ax_hm.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=cfg["OUT_DPI"])
    plt.close(fig)
    return True


# ============================================================
# Run helpers
# ============================================================
def list_sessions_with_labels(root: Path, labels_name: str) -> List[str]:
    sessions = set()
    for csv in root.rglob(labels_name):
        try:
            rel = csv.relative_to(root)
            sessions.add(rel.parts[0])  # session folder directly under root
        except Exception:
            pass
    return sorted(sessions)


def run_dashboards(
    data_root: Path,
    run_mode: List[str] | str,
    cfg: Dict[str, Any],
) -> None:
    apply_style(cfg)

    if isinstance(run_mode, str) and run_mode.upper() == "ALL":
        sessions = list_sessions_with_labels(data_root, cfg["LABELS_NAME"])
        print(f"[INFO] Mode ALL: found {len(sessions)} session(s) with labels.")
    else:
        sessions = list(run_mode)

    # output dir: either global OUT_ROOT or per-session OUT_DIRNAME
    out_root = cfg.get("OUT_ROOT", None)
    out_root_path = Path(out_root) if out_root else None
    if out_root_path:
        out_root_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] OUT_ROOT = {out_root_path}")

    for session in sessions:
        probes = find_probes_for_session(data_root, session, cfg["LABELS_NAME"])
        if not probes:
            print(f"[WARN] No probes found for session: {session}")
            continue

        if out_root_path:
            out_dir = out_root_path
        else:
            out_dir = data_root / session / cfg["OUT_DIRNAME"]
            out_dir.mkdir(parents=True, exist_ok=True)

        for alf_probe in probes:
            out_png = out_dir / f"{session}_{alf_probe.name}_dashboard.png"
            ok = make_dashboard_png(session, alf_probe, out_png, cfg)
            if ok:
                print(f"[OK] {out_png}")
            else:
                print(f"[SKIP] {session} / {alf_probe.name}")
