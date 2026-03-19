from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_good_cluster_ids(
    alf_probe: Path,
    labels_name: str = "clusters.labels.csv",
) -> np.ndarray:
    labels = pd.read_csv(alf_probe / labels_name)

    if ("label" not in labels.columns) or ("cluster_id" not in labels.columns):
        raise ValueError(f"{labels_name} must contain columns: cluster_id, label")

    labels["label"] = labels["label"].astype(str).str.lower().str.strip()
    labels["cluster_id"] = pd.to_numeric(labels["cluster_id"], errors="coerce")
    labels = labels.dropna(subset=["cluster_id"])
    labels["cluster_id"] = labels["cluster_id"].astype(int)

    return labels.loc[labels["label"] == "good", "cluster_id"].to_numpy(dtype=int)


def load_channel_region_segments(
    channel_locations_json: Path,
    depth_max: float = 4000.0,
) -> List[Tuple[float, float, str]]:
    with open(channel_locations_json, "r", encoding="utf-8") as f:
        ch = json.load(f)

    rows = []
    for v in ch.values():
        if isinstance(v, dict) and ("axial" in v) and ("brain_region" in v):
            try:
                rows.append((float(v["axial"]), str(v["brain_region"])))
            except Exception:
                pass

    rows.sort(key=lambda x: x[0])

    if not rows:
        raise ValueError("No valid entries found in channel_locations.json")

    ax = np.array([r[0] for r in rows], dtype=float)
    regs = [r[1] for r in rows]
    mids = (ax[:-1] + ax[1:]) / 2 if len(ax) > 1 else np.array([], dtype=float)

    segments: List[Tuple[float, float, str]] = []
    start = float(ax[0])
    cur = regs[0]

    for i in range(1, len(ax)):
        if regs[i] != cur:
            end = float(mids[i - 1]) if mids.size else float(ax[i])
            if end > start:
                segments.append((start, end, cur))
            start = end
            cur = regs[i]

    end_last = float(ax[-1])
    if end_last > start:
        segments.append((start, end_last, cur))

    clipped = []
    for a, b, r in segments:
        aa = max(0.0, min(depth_max, a))
        bb = max(0.0, min(depth_max, b))
        if bb > aa:
            clipped.append((aa, bb, r))

    if not clipped:
        raise ValueError("No valid region segments after clipping")

    return clipped


def region_at_depth(
    depth_um: float,
    segments: List[Tuple[float, float, str]],
) -> str:
    for a, b, r in segments:
        if a <= depth_um <= b:
            return r
    return "UNK"


def assign_cluster_regions(
    clusters: np.ndarray,
    depths: np.ndarray,
    segments: List[Tuple[float, float, str]],
    cluster_ids: Optional[np.ndarray] = None,
) -> Dict[int, str]:
    if cluster_ids is None:
        cluster_ids = np.unique(clusters)

    region_by_cluster: Dict[int, str] = {}

    for cid in cluster_ids:
        d = depths[clusters == cid]
        d = np.asarray(d, dtype=float)
        d = d[np.isfinite(d)]

        if d.size == 0:
            region_by_cluster[int(cid)] = "UNK"
            continue

        med_depth = float(np.median(d))
        region_by_cluster[int(cid)] = region_at_depth(med_depth, segments)

    return region_by_cluster


def region_matches_any_pattern(region_name: str, patterns: List[str]) -> bool:
    r = str(region_name).strip()
    for p in patterns:
        p = str(p).strip()
        if r == p or r.startswith(p):
            return True
    return False


def invert_group_definition(
    region_groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for group_name, patterns in region_groups.items():
        out[str(group_name)] = [str(x) for x in patterns]
    return out


def assign_cluster_group(
    region_by_cluster: Dict[int, str],
    region_groups: Dict[str, List[str]],
) -> Dict[int, str]:
    cluster_group: Dict[int, str] = {}

    for cid, region_name in region_by_cluster.items():
        assigned = "OTHER"
        for group_name, patterns in region_groups.items():
            if region_matches_any_pattern(region_name, patterns):
                assigned = group_name
                break
        cluster_group[int(cid)] = assigned

    return cluster_group


def gaussian_smooth_1d(x: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return x.copy()

    half = int(np.ceil(3 * sigma_bins))
    grid = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel /= kernel.sum()

    xpad = np.pad(x, (half, half), mode="reflect")
    y = np.convolve(xpad, kernel, mode="valid")
    return y


def zscore_1d(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def plot_spike_density_for_region_groups(
    alf_probe: Path,
    channel_locations_json: Optional[Path] = None,
    *,
    labels_name: str = "clusters.labels.csv",
    depth_max: float = 4000.0,
    use_good_only: bool = True,
    region_groups: Optional[Dict[str, List[str]]] = None,
    selected_groups: Optional[List[str]] = None,
    bin_size_s: float = 0.001,
    smooth_sigma_bins: float = 0.5,
    time_unit: str = "ms",          # "min", "s", "ms"
    normalize_by_n_clusters: bool = False,
    zscore_curves: bool = True,
    figsize: Tuple[float, float] = (13, 5),
    dpi: int = 150,
    title: Optional[str] = None,
    view_start_s: float = 0.0,
    view_width_s: Optional[float] = 0.1,   # 100 ms au départ
    interactive_nav: bool = True,
    pan_step_frac: float = 0.5,
    zoom_factor: float = 0.5,
    min_zoom_bins: int = 1,                # autorise zoom très fort
    draw_style: str = "step",              # "step" ou "line"
):
    alf_probe = Path(alf_probe)

    if channel_locations_json is None:
        channel_locations_json = alf_probe / "channel_locations.json"
    channel_locations_json = Path(channel_locations_json)

    if region_groups is None:
        region_groups = {
            "MO": ["MOs", "MOp"],
            "OFC": ["ORB", "FRP"],
            "CP_STR": ["CP", "STR"],
        }

    region_groups = invert_group_definition(region_groups)

    if selected_groups is None:
        selected_groups = list(region_groups.keys())

    spikes_clusters = np.load(alf_probe / "spikes.clusters.npy", mmap_mode="r").astype(int, copy=False)
    spikes_depths = np.load(alf_probe / "spikes.depths.npy", mmap_mode="r").astype(float, copy=False)
    spikes_times = np.load(alf_probe / "spikes.times.npy", mmap_mode="r").astype(float, copy=False)

    if not (spikes_clusters.shape[0] == spikes_depths.shape[0] == spikes_times.shape[0]):
        raise ValueError("spikes arrays must have same length")

    ok = np.isfinite(spikes_clusters) & np.isfinite(spikes_depths) & np.isfinite(spikes_times)
    spikes_clusters = spikes_clusters[ok]
    spikes_depths = np.clip(spikes_depths[ok], 0.0, depth_max)
    spikes_times = spikes_times[ok]

    if use_good_only:
        good_ids = load_good_cluster_ids(alf_probe, labels_name=labels_name)
        keep = np.isin(spikes_clusters, good_ids)
        spikes_clusters = spikes_clusters[keep]
        spikes_depths = spikes_depths[keep]
        spikes_times = spikes_times[keep]

    cluster_ids_for_assignment = np.unique(spikes_clusters)

    segments = load_channel_region_segments(channel_locations_json, depth_max=depth_max)

    region_by_cluster = assign_cluster_regions(
        clusters=spikes_clusters,
        depths=spikes_depths,
        segments=segments,
        cluster_ids=cluster_ids_for_assignment,
    )

    group_by_cluster = assign_cluster_group(
        region_by_cluster=region_by_cluster,
        region_groups=region_groups,
    )

    tmax = float(np.max(spikes_times)) if spikes_times.size else 1.0
    tmax = max(tmax, 1e-6)

    edges = np.arange(0.0, tmax + bin_size_s, bin_size_s, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([0.0, bin_size_s], dtype=np.float64)

    centers_s = 0.5 * (edges[:-1] + edges[1:])

    curves: Dict[str, np.ndarray] = {}
    cluster_count_by_group: Dict[str, int] = {}
    raw_region_count_by_group: Dict[str, Dict[str, int]] = {}

    for group_name in selected_groups:
        group_cluster_ids = np.array(
            [cid for cid, g in group_by_cluster.items() if g == group_name],
            dtype=int
        )

        if group_cluster_ids.size == 0:
            rate = np.zeros(edges.size - 1, dtype=np.float32)
            curves[group_name] = rate
            cluster_count_by_group[group_name] = 0
            raw_region_count_by_group[group_name] = {}
            continue

        mask = np.isin(spikes_clusters, group_cluster_ids)
        counts, _ = np.histogram(spikes_times[mask], bins=edges)
        rate = counts.astype(np.float32) / float(bin_size_s)

        if normalize_by_n_clusters:
            n_clu = max(1, len(np.unique(group_cluster_ids)))
            rate = rate / n_clu

        if smooth_sigma_bins > 0:
            rate = gaussian_smooth_1d(rate, smooth_sigma_bins).astype(np.float32)

        if zscore_curves:
            rate = zscore_1d(rate).astype(np.float32)

        curves[group_name] = rate
        cluster_count_by_group[group_name] = int(len(np.unique(group_cluster_ids)))

        detail: Dict[str, int] = {}
        for cid in np.unique(group_cluster_ids):
            rr = region_by_cluster.get(int(cid), "UNK")
            detail[rr] = detail.get(rr, 0) + 1
        raw_region_count_by_group[group_name] = dict(sorted(detail.items()))

    if str(time_unit).lower() == "min":
        x = centers_s / 60.0
        xlabel = "Time (min)"
        total_duration_x = tmax / 60.0
        x0_init = view_start_s / 60.0
        width_init = None if view_width_s is None else view_width_s / 60.0
    elif str(time_unit).lower() == "ms":
        x = centers_s * 1000.0
        xlabel = "Time (ms)"
        total_duration_x = tmax * 1000.0
        x0_init = view_start_s * 1000.0
        width_init = None if view_width_s is None else view_width_s * 1000.0
    else:
        x = centers_s
        xlabel = "Time (s)"
        total_duration_x = tmax
        x0_init = view_start_s
        width_init = view_width_s

    ylabel = "Spike density (z-score)" if zscore_curves else "Spike density (spikes/s)"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for group_name in selected_groups:
        y = curves[group_name]
        n_clu = cluster_count_by_group[group_name]
        if draw_style == "step":
            ax.step(x, y, where="mid", lw=0.9, label=f"{group_name} (n={n_clu})")
        else:
            ax.plot(x, y, lw=0.9, label=f"{group_name} (n={n_clu})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        suffix = "good only" if use_good_only else "all clusters"
        title = f"{alf_probe.name} - spike density over time ({suffix})"
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    if width_init is not None:
        x0 = max(0.0, x0_init)
        x1 = min(total_duration_x, x0 + width_init)
        if x1 <= x0:
            x1 = min(total_duration_x, x0 + max(width_init, 1e-12))
        ax.set_xlim(x0, x1)

    def _update_y_limits():
        xlim = ax.get_xlim()
        ymin = np.inf
        ymax = -np.inf

        for group_name in selected_groups:
            y = curves[group_name]
            m = (x >= xlim[0]) & (x <= xlim[1])
            if np.any(m):
                yy = y[m]
                if yy.size:
                    ymin = min(ymin, float(np.min(yy)))
                    ymax = max(ymax, float(np.max(yy)))

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = -1.0, 1.0
        elif ymax <= ymin:
            ymax = ymin + 1.0

        pad = 0.08 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)

    _update_y_limits()

    if interactive_nav:
        help_text = "←/→ pan   ↑/↓ zoom   molette zoom   Home/End   r reset"
        fig.text(0.5, 0.01, help_text, ha="center", va="bottom", fontsize=9, alpha=0.8)

        full_x0 = 0.0
        full_x1 = total_duration_x
        initial_xlim = ax.get_xlim()
        bin_dx = float(x[1] - x[0]) if len(x) > 1 else 1e-9
        min_w = max(min_zoom_bins * bin_dx, 1e-12)

        def _apply_new_xlim(new_x0: float, new_x1: float):
            new_x0 = max(full_x0, new_x0)
            new_x1 = min(full_x1, new_x1)

            if (new_x1 - new_x0) < min_w:
                center = 0.5 * (new_x0 + new_x1)
                new_x0 = max(full_x0, center - 0.5 * min_w)
                new_x1 = min(full_x1, center + 0.5 * min_w)

            ax.set_xlim(new_x0, new_x1)
            _update_y_limits()
            fig.canvas.draw_idle()

        def _on_key(event):
            cur_x0, cur_x1 = ax.get_xlim()
            cur_w = cur_x1 - cur_x0
            if cur_w <= 0:
                return

            if event.key == "right":
                step = pan_step_frac * cur_w
                _apply_new_xlim(cur_x0 + step, cur_x1 + step)

            elif event.key == "left":
                step = pan_step_frac * cur_w
                _apply_new_xlim(cur_x0 - step, cur_x1 - step)

            elif event.key == "up":
                center = 0.5 * (cur_x0 + cur_x1)
                new_w = max(min_w, cur_w * zoom_factor)
                _apply_new_xlim(center - 0.5 * new_w, center + 0.5 * new_w)

            elif event.key == "down":
                center = 0.5 * (cur_x0 + cur_x1)
                new_w = min(full_x1 - full_x0, cur_w / zoom_factor)
                _apply_new_xlim(center - 0.5 * new_w, center + 0.5 * new_w)

            elif event.key == "home":
                _apply_new_xlim(full_x0, full_x0 + cur_w)

            elif event.key == "end":
                _apply_new_xlim(full_x1 - cur_w, full_x1)

            elif event.key == "r":
                ax.set_xlim(initial_xlim)
                _update_y_limits()
                fig.canvas.draw_idle()

        def _on_scroll(event):
            if event.inaxes != ax or event.xdata is None:
                return

            cur_x0, cur_x1 = ax.get_xlim()
            cur_w = cur_x1 - cur_x0
            if cur_w <= 0:
                return

            center = float(event.xdata)

            if event.button == "up":
                new_w = max(min_w, cur_w * zoom_factor)
            elif event.button == "down":
                new_w = min(full_x1 - full_x0, cur_w / zoom_factor)
            else:
                return

            rel = (center - cur_x0) / cur_w if cur_w > 0 else 0.5
            new_x0 = center - rel * new_w
            new_x1 = new_x0 + new_w
            _apply_new_xlim(new_x0, new_x1)

        fig.canvas.mpl_connect("key_press_event", _on_key)
        fig.canvas.mpl_connect("scroll_event", _on_scroll)

    out = {
        "region_by_cluster": region_by_cluster,
        "group_by_cluster": group_by_cluster,
        "cluster_count_by_group": cluster_count_by_group,
        "raw_region_count_by_group": raw_region_count_by_group,
        "curves": curves,
        "x": x,
        "x_seconds": centers_s,
        "bin_edges_seconds": edges,
    }

    return fig, ax, out