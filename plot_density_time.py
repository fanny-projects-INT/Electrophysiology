# plot_density_time.py
from pathlib import Path
import matplotlib.pyplot as plt

from functions.density_time import plot_spike_density_for_region_groups


# =========================
# PARAMÈTRES GÉNÉRAUX
# =========================

DATA_ROOT = Path(r"F:\Data_Mice")
SESSION_ID = "VF066_2025_12_05"
PROBE = "probe01"

depth_max = 4000.0

# "s" conseillé pour explorer finement
time_unit = "s"

# =========================
# GROUPES DE RÉGIONS
# =========================
# Matching par préfixe :
# - "MOs" match -> MOs, MOs1, MOs2...
# - "MOp" match -> MOp...
# - "ORB" match -> ORB, ORBl, ORBm, ORBvl...
# - "STR" match -> STR, STRd, STRv...
# - "CP" match -> CP

REGION_GROUPS = {
    "MO": ["MOs", "MOp"],
    "OFC": ["ORB", "FRP"],
    "CP_STR": ["CP", "STR"],
}

SELECTED_GROUPS = ["MO", "OFC", "CP_STR"]


# =========================
# RÉGLAGES TEMPORELS
# =========================

# Taille des bins pour calculer la densité
BIN_SIZE_S = 1.0

# 0.0 = pas de lissage
# 0.2 ou 0.3 = lissage très léger
SMOOTH_SIGMA_BINS = 0.0

# Fenêtre affichée au départ
VIEW_START_S = 0.0
VIEW_WIDTH_S = 20.0

# Navigation clavier
INTERACTIVE_NAV = True

# Défilement = 50% de la fenêtre actuelle
PAN_STEP_FRAC = 0.5

# Zoom in/out
ZOOM_FACTOR = 0.6


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    alf_probe = DATA_ROOT / SESSION_ID / "alf" / PROBE
    channel_locations = alf_probe / "channel_locations.json"

    fig, ax, out = plot_spike_density_for_region_groups(
        alf_probe=alf_probe,
        channel_locations_json=channel_locations,

        labels_name="clusters.labels.csv",
        depth_max=depth_max,
        use_good_only=True,

        region_groups=REGION_GROUPS,
        selected_groups=SELECTED_GROUPS,

        bin_size_s=BIN_SIZE_S,
        smooth_sigma_bins=SMOOTH_SIGMA_BINS,
        time_unit=time_unit,
        normalize_by_n_clusters=False,

        view_start_s=VIEW_START_S,
        view_width_s=VIEW_WIDTH_S,
        interactive_nav=INTERACTIVE_NAV,
        pan_step_frac=PAN_STEP_FRAC,
        zoom_factor=ZOOM_FACTOR,

        title=f"{SESSION_ID} - {PROBE} - spike density by target regions",
    )

    # =========================
    # RÉSUMÉ CONSOLE
    # =========================
    print("\n=== Cluster count by group ===")
    for k, v in out["cluster_count_by_group"].items():
        print(f"{k}: {v}")

    print("\n=== Raw region composition by group ===")
    for group_name, detail in out["raw_region_count_by_group"].items():
        print(f"\n{group_name}")
        if not detail:
            print("  (no clusters)")
        else:
            for reg, n in detail.items():
                print(f"  {reg}: {n}")

    print("\n=== Controls ===")
    print("Left / Right : pan")
    print("Up / Down    : zoom in / out")
    print("Home / End   : go to start / end")
    print("r            : reset view")

    plt.show()