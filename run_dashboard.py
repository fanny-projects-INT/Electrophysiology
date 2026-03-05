from pathlib import Path
import matplotlib.pyplot as plt

from functions.dashboard import plot_good_spikes_heatmap_with_regions

# =========================
# PARAMÈTRES
# =========================

DATA_ROOT = Path(r"F:\Data_Mice")
SESSION_ID = "VF069_2025_12_03"
PROBE = "probe00"

depth_max = 4000          # 0 = tip, 4000 = surface
heat_time_unit = "min"

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    alf_probe = DATA_ROOT / SESSION_ID / "alf" / PROBE
    channel_locations = alf_probe / "channel_locations.json"

    fig, axes = plot_good_spikes_heatmap_with_regions(
        alf_probe=alf_probe,
        channel_locations_json=channel_locations,
        title=f"{SESSION_ID} - {PROBE}",
        depth_max=depth_max,
        heat_time_unit=heat_time_unit,
    )

    plt.show()