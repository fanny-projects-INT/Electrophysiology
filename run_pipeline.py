from pathlib import Path

from functions.paths import build_paths
from functions.load import load_recordings
from functions.sort import run_kilosort4
from functions.alf import export_alf

from spikeinterface.sorters import get_default_sorter_params


DATA_ROOT = Path(r"D:\Data_Mice")
DB_PATH   = DATA_ROOT / "full_db_all_rigs.feather"

MOUSE_LIST = [
    "VF071_2025_12_17",
]

# Kilosort params 
KS_PARAMS = get_default_sorter_params("kilosort4")


def main():
    # build sessions
    sessions = [build_paths(name, data_root=DATA_ROOT, db_path=DB_PATH) for name in MOUSE_LIST]

    for sess in sessions:
        # 1) load
        load_recordings(sess)

        # 2) sort
        run_kilosort4(sess, params=KS_PARAMS, remove_existing_folder=True)

        # 3) export ALF
        export_alf(sess, stop_on_error=False)


if __name__ == "__main__":
    main()
