import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'affichage notebook
import matplotlib.pyplot as plt

from pathlib import Path
import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QScrollArea,
    QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


def run_dashboard(
    session_id: str,
    data_root: Path,
    mc_thresh: float = 0,
    nc_thresh: float = 100,
    fs: int = 30000,
    dtype=np.int16,
    pre: int = 20,
    post: int = 40,
    num_channels_bin: int = 385,
):
    """
    Ouvre un dashboard PyQt pour visualiser les clusters (ALF) et waveforms (bin).
    Compatible 1 ou 2 probes via:
      Rec/probe00 (obligatoire) + Rec/probe01 (optionnel)
      alf/probe00 / alf/probe01

    mc_thresh / nc_thresh : tes filtres (max_confidence, noise_cutoff)
    """
    base_folder = data_root / session_id
    mouse_name = base_folder.name

    rec_root = base_folder / "Rec"
    alf_root = base_folder / "alf"

    if not (rec_root / "probe00").exists():
        raise FileNotFoundError(f"Missing folder: {rec_root / 'probe00'}")

    present_probes = ["probe00"]
    if (rec_root / "probe01").exists():
        present_probes.append("probe01")

    window = pre + post
    sync_chan = num_channels_bin - 1

    # ---------------- HELPERS ----------------
    def fmt(x, nd=2):
        if x is None:
            return "NA"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "NA"

    def fmt_uv(x):
        if x is None:
            return "NA"
        try:
            return f"{float(x):.1f} µV"
        except Exception:
            return "NA"

    # ---------------- DATA LOADER ----------------
    def load_probe_data(probe: str) -> dict:
        rec_folder = rec_root / probe
        alf_folder = alf_root / probe

        probe_idx = int(probe.replace("probe", ""))  # probe00->0, probe01->1
        bin_file = rec_folder / f"disabled_g0_t0.imec{probe_idx}.ap.bin"

        spike_times_file    = alf_folder / "spikes.times.npy"
        spike_clusters_file = alf_folder / "spikes.clusters.npy"
        spike_samples_file  = alf_folder / "spikes.samples.npy"
        spike_depths_file   = alf_folder / "spikes.depths.npy"

        metrics_candidates = [
            alf_folder / "cluster_metrics.csv",
            alf_folder / "clusters.metrics.csv",
            alf_folder / "clusters.metrics.tsv",
        ]
        cluster_metrics_file = next((p for p in metrics_candidates if p.exists()), None)

        required = [bin_file, spike_times_file, spike_clusters_file, spike_depths_file]
        for p in required:
            if not p.exists():
                raise FileNotFoundError(f"[{mouse_name}/{probe}] Missing file: {p}")

        if cluster_metrics_file is None:
            raise FileNotFoundError(
                f"[{mouse_name}/{probe}] Missing metrics file in ALF. Tried: "
                + ", ".join([str(p.name) for p in metrics_candidates])
            )

        spike_times    = np.load(spike_times_file)
        spike_clusters = np.load(spike_clusters_file).astype(int)
        spike_depths   = np.load(spike_depths_file).astype(float)
        all_clusters   = np.unique(spike_clusters)

        if spike_samples_file.exists():
            spike_samples = np.load(spike_samples_file).astype(int)
        else:
            spike_samples = None

        data = np.memmap(bin_file, dtype=dtype, mode="r")
        if data.size % num_channels_bin != 0:
            raise RuntimeError(
                f"[{mouse_name}/{probe}] Incohérence: data.size={data.size} n'est pas divisible par num_channels_bin={num_channels_bin}."
            )
        total_samples = data.size // num_channels_bin
        data = data.reshape((total_samples, num_channels_bin))

        delimiter = "\t" if cluster_metrics_file.suffix.lower() == ".tsv" else ","
        metrics = np.genfromtxt(
            cluster_metrics_file,
            delimiter=delimiter,
            names=True,
            dtype=None,
            encoding="utf-8"
        )

        cols = metrics.dtype.names
        if cols is None:
            raise RuntimeError(f"[{mouse_name}/{probe}] Impossible de lire l'en-tête de {cluster_metrics_file.name}")

        cluster_id_col = "cluster_id" if "cluster_id" in cols else ("cluster_ids" if "cluster_ids" in cols else None)
        if cluster_id_col is None:
            raise KeyError(f"[{mouse_name}/{probe}] Colonne cluster_id/cluster_ids absente. Colonnes: {list(cols)}")

        needed = ["max_confidence", "noise_cutoff", "amp_median"]
        missing = [c for c in needed if c not in cols]
        if missing:
            raise KeyError(
                f"[{mouse_name}/{probe}] Colonnes manquantes dans {cluster_metrics_file.name}: {missing}. Colonnes: {list(cols)}"
            )

        ibl_map = {}
        for i in range(len(metrics)):
            cid = int(metrics[cluster_id_col][i])
            ibl_map[cid] = {c: metrics[c][i] for c in cols}

        clusters_selected = []
        for cid, row in ibl_map.items():
            try:
                mc = float(row["max_confidence"])
                nc = float(row["noise_cutoff"])
            except Exception:
                continue
            if (mc >= mc_thresh) and (nc < nc_thresh):
                clusters_selected.append(int(cid))

        all_set = set(all_clusters.tolist())
        clusters_selected = [cid for cid in clusters_selected if cid in all_set]
        clusters_selected.sort()

        return dict(
            probe=probe,
            bin_file=bin_file,
            cluster_metrics_file=cluster_metrics_file,
            spike_times=spike_times,
            spike_clusters=spike_clusters,
            spike_samples=spike_samples,
            spike_depths=spike_depths,
            all_clusters=all_clusters,
            clusters_selected=clusters_selected,
            ibl_map=ibl_map,
            data=data,
            total_samples=total_samples,
        )

    def get_metrics_for_cluster(ctx: dict, cid: int):
        row = ctx["ibl_map"].get(int(cid), {})
        mc = row.get("max_confidence", None)
        nc = row.get("noise_cutoff", None)
        amp = row.get("amp_median", None)
        return mc, nc, amp

    # ---------------- UI BLOCKS ----------------
    def make_cluster_block(ctx: dict, cid: int):
        spike_clusters = ctx["spike_clusters"]
        spike_times    = ctx["spike_times"]
        spike_samples  = ctx["spike_samples"]
        spike_depths   = ctx["spike_depths"]
        data           = ctx["data"]
        total_samples  = ctx["total_samples"]
        probe          = ctx["probe"]

        block = QWidget()
        block.setMinimumHeight(340)

        layout = QHBoxLayout(block)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(35)

        idx = np.where(spike_clusters == cid)[0]
        if idx.size == 0:
            return None

        idx_small = idx[:120]

        d = spike_depths[idx]
        d = d[np.isfinite(d)]
        cluster_depth = float(np.median(d)) if d.size else np.nan

        if spike_samples is not None:
            st_small = spike_samples[idx_small]
        else:
            st_small = (spike_times[idx_small] * fs).astype(int)

        tmp = []
        for s in st_small[:40]:
            if pre < s < total_samples - post:
                tmp.append(data[s - pre:s + post, :])
        if len(tmp) == 0:
            return None

        tmp = np.array(tmp)
        ptp = tmp.ptp(axis=1).mean(axis=0)

        ptp[sync_chan] = -np.inf
        best_chan = int(np.argmax(ptp))

        wfs = []
        for s in st_small:
            if pre < s < total_samples - post:
                wfs.append(data[s - pre:s + post, best_chan])
        if len(wfs) == 0:
            return None

        wfs = np.array(wfs)
        wf_avg = wfs.mean(axis=0)

        t_ms = np.arange(window) / fs * 1000
        fig, ax = plt.subplots(figsize=(6, 4.8))
        for wf in wfs:
            ax.plot(t_ms, wf, color="black", alpha=0.10)
        ax.plot(t_ms, wf_avg, color="red", linewidth=1.6)

        ax.set_title(f"{probe} — Cluster {cid}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.8)

        canvas = FigureCanvasQTAgg(fig)

        mc, nc, amp = get_metrics_for_cluster(ctx, cid)

        info = QLabel(
            f"""
<b>{probe} — Cluster {cid}</b><br><br>
N spikes : {idx.size}<br>
Depth : {fmt(cluster_depth, nd=1)} µm<br><br>
max_confidence : {fmt(mc, nd=1)} (≥ {mc_thresh:g})<br>
noise_cutoff : {fmt(nc, nd=2)} (&lt; {nc_thresh:g})<br>
amp_median : {fmt_uv(amp)}<br>
"""
        )
        info.setAlignment(Qt.AlignTop)
        info.setStyleSheet("font-size:12px; padding:10px;")

        layout.addWidget(canvas)
        layout.addWidget(info)
        return block

    class ProbeTab(QWidget):
        def __init__(self, ctx: dict):
            super().__init__()
            main_layout = QVBoxLayout(self)

            header = QLabel(
                f"<h2>{mouse_name} — {ctx['probe']}</h2>"
                f"Clusters totaux : {len(ctx['all_clusters'])} — "
                f"Affichés : {len(ctx['clusters_selected'])}<br>"
                f"Filtres : max_confidence ≥ {mc_thresh:g} ; noise_cutoff &lt; {nc_thresh:g}<br>"
                f"QC source: {ctx['cluster_metrics_file']}"
            )
            header.setStyleSheet("padding: 14px; font-size: 15px;")
            main_layout.addWidget(header)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            main_layout.addWidget(scroll)

            container = QWidget()
            scroll.setWidget(container)

            grid = QGridLayout(container)
            grid.setContentsMargins(15, 15, 15, 15)
            grid.setHorizontalSpacing(45)
            grid.setVerticalSpacing(50)

            col = 0
            row = 0
            for cid in ctx["clusters_selected"]:
                block = make_cluster_block(ctx, int(cid))
                if block is None:
                    continue

                grid.addWidget(block, row, col)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1

    class Dashboard(QWidget):
        def __init__(self, probe_contexts: dict):
            super().__init__()
            self.setWindowTitle(f"Cluster Dashboard – {mouse_name}")
            self.resize(1600, 1050)

            layout = QVBoxLayout(self)
            tabs = QTabWidget()
            layout.addWidget(tabs)

            for probe, ctx in probe_contexts.items():
                tabs.addTab(ProbeTab(ctx), probe)

    # charge toutes les probes présentes
    probe_contexts = {p: load_probe_data(p) for p in present_probes}

    app = QApplication(sys.argv)
    w = Dashboard(probe_contexts)
    w.show()
    sys.exit(app.exec_())
