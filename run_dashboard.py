from pathlib import Path
from functions import dashboard as db


# ============================================================
# EDIT HERE
# ============================================================
DATA_ROOT = Path(r"E:\Aurelien\Data_Mice")

# Global output folder (all PNGs saved here)
OUT_ROOT = Path(r"E:\Aurelien\Data_Mice\Dashbords")

# Choose:
# RUN_MODE = "ALL"
# or RUN_MODE = ["VF071_2025_12_18", "VF069_2025_12_06"]
RUN_MODE = "ALL"
# RUN_MODE = ["VF071_2025_12_18"]


cfg = db.DEFAULTS.copy()
cfg.update({
    "OUT_ROOT": OUT_ROOT,        # <- all PNGs saved here

    # canvas
    "OUT_W_PX": 1400,
    "OUT_H_PX": 800,
    "OUT_DPI": 150,

    # style
    "TITLE_SIZE": 10.5,
    "FONT_SIZE": 10,

    # keep time in minutes for heatmap
    "HEAT_TIME_UNIT": "min",
})


if __name__ == "__main__":
    db.run_dashboards(DATA_ROOT, RUN_MODE, cfg)
