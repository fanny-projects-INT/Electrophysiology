# dashboard.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def plot_good_spikes_heatmap_with_regions(
    alf_probe: Path,
    channel_locations_json: Optional[Path] = None,
    labels_name: str = "clusters.labels.csv",
    *,
    # ------------------------------------------------------------
    # DEPTH CONVENTION
    # ------------------------------------------------------------
    depth_max: float = 4000.0,

    # ------------------------------------------------------------
    # HEATMAP (high resolution)
    # ------------------------------------------------------------
    heat_time_unit: str = "min",
    heat_tbins: int = 700,
    heat_dbins: int = 900,
    heat_max_points: int = 1_500_000,
    heat_clip_percentile: float = 99.6,
    heat_add_pseudocount: bool = True,
    heat_time_max: Optional[float] = None,
    heat_cmap: str = "inferno",
    heat_smooth_sigma_t: float = 0.7,
    heat_smooth_sigma_d: float = 0.7,
    heat_interpolation: str = "bilinear",

    # ------------------------------------------------------------
    # RAIL HEATMAP (in negative time, BEFORE t=0)
    # ------------------------------------------------------------
    hm_rail_time_frac: float = 0.05,
    rail_alpha: float = 0.92,

    # ------------------------------------------------------------
    # DENSITY (good units)
    # ------------------------------------------------------------
    depth_bins: int = 140,
    depth_smooth_sigma: float = 1.25,
    density_line_lw: float = 2.4,
    density_fill_alpha: float = 0.28,

    # Density rail (strictly on [-den_rail_width, 0])
    den_rail_width: float = 0.28,
    den_rail_alpha: float = 0.92,

    # ------------------------------------------------------------
    # REGIONS / LABELS / SEPARATORS
    # ------------------------------------------------------------
    show_separators: bool = True,
    separator_lw: float = 1.1,
    separator_alpha: float = 0.9,
    separator_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),

    region_label: bool = True,
    region_label_fontsize: int = 8,
    region_label_min_height_um: float = 120.0,

    # ------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------
    figsize: Tuple[float, float] = (13.6, 7.0),
    dpi: int = 150,
    title: Optional[str] = None,
    density_title: str = "Good unit density",

    # ------------------------------------------------------------
    # CLEAN DISPLAY: counts left of rail + title in rail
    # ------------------------------------------------------------
    show_good_unit_counts: bool = True,
    count_fontsize: int = 9,
    count_pad: float = 0.16,
    count_text_x_nudge: float = -0.02,
    show_density_title_in_rail: bool = True,
    density_title_fontsize: int = 9,

    # ------------------------------------------------------------
    # Colorbar sizing (heatmap)
    # ------------------------------------------------------------
    cbar_fraction: float = 0.075,
    cbar_pad: float = 0.02,
) -> Tuple[plt.Figure, Dict[str, plt.Axes]]:

    alf_probe = Path(alf_probe)
    depth_max = float(depth_max)

    # ============================================================
    # Load labels
    # ============================================================
    labels = pd.read_csv(alf_probe / labels_name)
    if ("label" not in labels.columns) or ("cluster_id" not in labels.columns):
        raise ValueError(f"{labels_name} must contain columns: cluster_id, label")
    labels["label"] = labels["label"].astype(str).str.lower().str.strip()
    labels["cluster_id"] = pd.to_numeric(labels["cluster_id"], errors="coerce")
    labels = labels.dropna(subset=["cluster_id"])
    labels["cluster_id"] = labels["cluster_id"].astype(int)
    good_ids = labels.loc[labels["label"] == "good", "cluster_id"].to_numpy(dtype=int)

    n_good = int(good_ids.size)

    # ============================================================
    # Load spikes
    # ============================================================
    clusters = np.load(alf_probe / "spikes.clusters.npy", mmap_mode="r").astype(int, copy=False)
    depths   = np.load(alf_probe / "spikes.depths.npy",   mmap_mode="r").astype(float, copy=False)
    times    = np.load(alf_probe / "spikes.times.npy",    mmap_mode="r").astype(float, copy=False)

    if not (clusters.shape[0] == depths.shape[0] == times.shape[0]):
        raise ValueError("spikes arrays must have same length")

    # ============================================================
    # GOOD spikes ONLY
    # ============================================================
    if good_ids.size:
        m = np.isin(clusters, good_ids)
        t = times[m].astype(float)
        d = depths[m].astype(float)
        ok = np.isfinite(t) & np.isfinite(d)
        t, d = t[ok], d[ok]
    else:
        t = np.array([], dtype=float)
        d = np.array([], dtype=float)

    d = np.clip(d, 0.0, depth_max)

    # Downsample
    if t.size > int(heat_max_points):
        rng = np.random.default_rng(0)
        idx = rng.choice(t.size, size=int(heat_max_points), replace=False)
        t, d = t[idx], d[idx]

    # Time unit
    if str(heat_time_unit).lower() == "min":
        t = t / 60.0
        xlabel = "Time (min)"
    else:
        xlabel = "Time (s)"

    tmax = float(heat_time_max) if (heat_time_max is not None and np.isfinite(float(heat_time_max))) else (float(np.max(t)) if t.size else 1.0)
    tmax = max(tmax, 1e-6)

    # ============================================================
    # Heatmap histogram
    # ============================================================
    H, t_edges, d_edges = np.histogram2d(
        t, d,
        bins=[int(heat_tbins), int(heat_dbins)],
        range=[[0.0, tmax], [0.0, depth_max]],
    )
    Z = H.T
    if bool(heat_add_pseudocount):
        Z = Z + 1.0

    # Smooth (bin-space gaussian, no scipy)
    def _gauss_kernel(sigma: float) -> np.ndarray:
        if sigma <= 0:
            return np.array([1.0], dtype=float)
        half = int(np.ceil(3 * sigma))
        x = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        return k

    def _conv1d_reflect(A: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
        if k.size == 1:
            return A
        half = (k.size - 1) // 2
        pad = [(0, 0)] * A.ndim
        pad[axis] = (half, half)
        Ap = np.pad(A, pad, mode="reflect")
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, Ap)

    Z = _conv1d_reflect(Z, _gauss_kernel(float(heat_smooth_sigma_d)), axis=0)
    Z = _conv1d_reflect(Z, _gauss_kernel(float(heat_smooth_sigma_t)), axis=1)

    # Norm
    pos = Z[np.isfinite(Z)]
    vmax = float(np.percentile(pos, float(heat_clip_percentile))) if pos.size else 2.0
    vmax = max(vmax, 2.0)
    norm = LogNorm(vmin=1.0, vmax=vmax)

    # ============================================================
    # Regions -> segments + colors
    # ============================================================
    segments: List[Tuple[float, float, str]] = []
    region_colors: Dict[str, Tuple[float, float, float]] = {}

    if channel_locations_json is not None and Path(channel_locations_json).exists():
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

        if rows:
            ax = np.array([r[0] for r in rows], dtype=float)
            regs = [r[1] for r in rows]
            mids = (ax[:-1] + ax[1:]) / 2 if len(ax) > 1 else np.array([], dtype=float)

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

            # clip
            seg2 = []
            for a, b, r in segments:
                aa = max(0.0, min(depth_max, a))
                bb = max(0.0, min(depth_max, b))
                if bb > aa:
                    seg2.append((aa, bb, r))
            segments = seg2

            uniq = sorted({r for _, _, r in segments})
            cmap_regions = mpl.cm.tab20
            for i, r in enumerate(uniq):
                region_colors[r] = cmap_regions(i % cmap_regions.N)[:3]

    # ============================================================
    # Density curve (good units)
    # ============================================================
    good_unit_depths = []
    for cid in good_ids:
        dd = depths[clusters == cid]
        dd = np.asarray(dd, dtype=float)
        dd = dd[np.isfinite(dd)]
        if dd.size:
            good_unit_depths.append(float(np.median(dd)))
    good_unit_depths = np.clip(np.asarray(good_unit_depths, dtype=float), 0.0, depth_max)

    counts, edges = np.histogram(good_unit_depths, bins=int(depth_bins), range=(0.0, depth_max))
    y_centers = 0.5 * (edges[:-1] + edges[1:])
    dens = counts.astype(float)
    if dens.max() > 0:
        dens /= dens.max()

    # smoothing
    if depth_smooth_sigma > 0:
        sigma = float(depth_smooth_sigma)
        half = int(np.ceil(3 * sigma))
        x = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        dens = np.convolve(np.pad(dens, (half, half), mode="reflect"), k, mode="valid")
        if dens.max() > 0:
            dens /= dens.max()

    # ============================================================
    # Count GOOD units per region
    # ============================================================
    good_units_per_region: Dict[str, int] = {}
    if segments and good_unit_depths.size:
        def _region_at_depth(y0: float) -> str:
            for a, b, r in segments:
                if (y0 >= a) and (y0 <= b):
                    return r
            return "UNK"

        for y0 in good_unit_depths:
            rr = _region_at_depth(float(y0))
            good_units_per_region[rr] = good_units_per_region.get(rr, 0) + 1

    # ============================================================
    # Fill holes in region assignment along y_centers (NO holes)
    # ============================================================
    def region_per_depth(y: np.ndarray) -> np.ndarray:
        lab = np.array([None] * y.size, dtype=object)
        for a, b, r in segments:
            m = (y >= a) & (y <= b)
            lab[m] = r
        s = pd.Series(lab, dtype="object").ffill().bfill().fillna("UNK")
        return s.to_numpy(dtype=object)

    lab_y = region_per_depth(y_centers)

    # boundaries from changes (for separators)
    change = np.flatnonzero(lab_y[1:] != lab_y[:-1])
    bounds = [0.5 * (y_centers[i] + y_centers[i + 1]) for i in change]

    # ============================================================
    # FIGURE
    # ============================================================
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.30], wspace=0.08)

    ax_hm = fig.add_subplot(gs[0, 0])
    ax_den = fig.add_subplot(gs[0, 1], sharey=ax_hm)

    base_title = title if title else alf_probe.name
    ax_hm.set_title(f"{base_title} - Good units: {n_good}", pad=8)
    ax_den.set_title("", pad=8)

    # ----------------------------
    # Heatmap
    # ----------------------------
    cmap = mpl.cm.get_cmap(heat_cmap)
    im = ax_hm.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[0.0, tmax, 0.0, depth_max],
        cmap=cmap,
        norm=norm,
        interpolation=str(heat_interpolation),
        zorder=1,
    )

    rail_t = float(hm_rail_time_frac) * tmax
    ax_hm.set_xlim(-rail_t, tmax)

    ax_hm.set_xlabel(xlabel)
    ax_hm.set_ylabel("Depth from tip (µm)")

    # remove spines
    for sp in ["top", "right", "left", "bottom"]:
        ax_hm.spines[sp].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=float(cbar_fraction), pad=float(cbar_pad))
    cbar.outline.set_visible(False)
    cbar.set_label("Good spike count (log)")

    # Heatmap rail (negative time)
    if segments:
        for a, b, r in segments:
            col = region_colors.get(r, (0.7, 0.7, 0.7))
            ax_hm.add_patch(
                Rectangle(
                    (-rail_t, a),
                    rail_t,
                    (b - a),
                    facecolor=col,
                    edgecolor="none",
                    alpha=rail_alpha,
                    zorder=4,
                )
            )

        if show_separators:
            for y in bounds:
                ax_hm.axhline(y, color=separator_color, lw=separator_lw, alpha=separator_alpha, zorder=6)

        if region_label:
            for a, b, r in segments:
                if (b - a) < float(region_label_min_height_um):
                    continue
                col = region_colors.get(r, (0.7, 0.7, 0.7))
                ax_hm.text(
                    0.01, 0.5 * (a + b), r,
                    transform=ax_hm.get_yaxis_transform(),
                    ha="left", va="center",
                    fontsize=region_label_fontsize,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc=(*col, 0.92), ec="none"),
                    zorder=7,
                )

    # ----------------------------
    # Density panel
    # ----------------------------
    ax_den.set_xlabel("Norm.")
    extra_left = float(count_pad) if show_good_unit_counts else 0.0
    ax_den.set_xlim(-float(den_rail_width) - extra_left, 1.02)
    ax_den.yaxis.set_visible(False)

    for sp in ["top", "right", "left", "bottom"]:
        ax_den.spines[sp].set_visible(False)

    # separators ONLY on rail
    xlim0, xlim1 = ax_den.get_xlim()
    span = (xlim1 - xlim0) if (xlim1 > xlim0) else 1.0
    rail_x0 = -float(den_rail_width)
    rail_x1 = 0.0
    rail_frac0 = float(np.clip((rail_x0 - xlim0) / span, 0.0, 1.0))
    rail_frac1 = float(np.clip((rail_x1 - xlim0) / span, 0.0, 1.0))

    # ----------------------------
    # Rail blocks + labels + counts
    # ----------------------------
    start = 0
    while start < y_centers.size:
        rr = lab_y[start]
        end = start + 1
        while end < y_centers.size and lab_y[end] == rr:
            end += 1

        col = region_colors.get(rr, (0.7, 0.7, 0.7))

        y0 = edges[start]
        y1 = edges[end] if end < edges.size else edges[-1]
        ymid = 0.5 * (y0 + y1)

        ax_den.fill_betweenx([y0, y1], -float(den_rail_width), 0.0, color=col, alpha=den_rail_alpha, linewidth=0, zorder=1)

        if region_label and (y1 - y0) >= float(region_label_min_height_um):
            ax_den.text(
                -float(den_rail_width) + 0.02, ymid, rr,
                ha="left", va="center",
                fontsize=region_label_fontsize,
                color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=(*col, 0.92), ec="none"),
                zorder=4,
            )

        if show_good_unit_counts and (y1 - y0) >= float(region_label_min_height_um):
            n_rr = int(good_units_per_region.get(rr, 0))
            x_count = -float(den_rail_width) - extra_left + 0.01 + float(count_text_x_nudge)
            ax_den.text(
                x_count, ymid, f"{n_rr}",
                ha="left", va="center",
                fontsize=int(count_fontsize),
                color="white",
                bbox=dict(boxstyle="round,pad=0.12", fc=(0, 0, 0, 0.45), ec="none"),
                zorder=5,
            )

        start = end

    if show_separators:
        for y in bounds:
            ax_den.axhline(
                y,
                xmin=rail_frac0,
                xmax=rail_frac1,
                color=separator_color,
                lw=separator_lw,
                alpha=separator_alpha,
                zorder=2,
            )

    # title inside rail
    if show_density_title_in_rail:
        ax_den.text(
            -float(den_rail_width) + 0.02,
            depth_max + 0.03 * depth_max,
            density_title,
            ha="left", va="top",
            fontsize=int(density_title_fontsize),
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc=(0, 0, 0, 0.35), ec="none"),
            zorder=8,
        )
    else:
        ax_den.set_title(density_title, pad=8)

    # ============================================================
    # NO-HOLES DENSITY:
    # 1) draw continuous curve once (guarantees continuity)
    # 2) draw colored segments with 1-point overlap at boundaries
    # ============================================================

    # continuous "base" curve (very subtle) -> ensures no visible gaps
    ax_den.plot(
        dens,
        y_centers,
        lw=float(density_line_lw),
        color=(0, 0, 0, 0.22),
        zorder=3,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    # colored fill + colored curve by region, WITH overlap
    start = 0
    n = y_centers.size
    while start < n:
        rr = lab_y[start]
        end = start + 1
        while end < n and lab_y[end] == rr:
            end += 1

        # overlap by 1 point to avoid micro-gaps at boundaries
        end_plot = min(end + 1, n) if end < n else n

        col = region_colors.get(rr, (0.15, 0.15, 0.15))

        ax_den.fill_betweenx(
            y_centers[start:end_plot],
            0.0,
            dens[start:end_plot],
            color=col,
            alpha=float(density_fill_alpha),
            linewidth=0,
            zorder=4,
        )

        ax_den.plot(
            dens[start:end_plot],
            y_centers[start:end_plot],
            color=col,
            lw=float(density_line_lw),
            zorder=6,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

        start = end

    return fig, {"heatmap": ax_hm, "density": ax_den}