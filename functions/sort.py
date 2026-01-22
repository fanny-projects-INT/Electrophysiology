from pathlib import Path
from spikeinterface.sorters import run_sorter, get_default_sorter_params


def run_kilosort4(sess: dict, params: dict | None = None, remove_existing_folder: bool = True) -> dict:
    """
    Lance Kilosort4 sur toutes les probes présentes dans sess["recordings"].
    Remplit:
      sess["sortings"][probe]
      sess["sorting"] (compat = probe00)
      P["recording_json"], P["sorting_json"] par probe
    """
    if params is None:
        params = get_default_sorter_params("kilosort4")

    print(f"\n=== Kilosort4: {sess['session_name']} ===")

    # compat: si pas sess["recordings"], fallback sur sess["recording"]
    recordings = sess.get("recordings", None)
    if recordings is None:
        rec0 = sess.get("recording", None)
        if rec0 is None:
            print("  [SKIP] No recording found in sess['recordings'] or sess['recording'].")
            return sess
        recordings = {"probe00": rec0}

    sess["sortings"] = {}

    for probe, recording in recordings.items():
        print(f"  -> running on {probe}")

        P = sess["probes"].get(probe, None)
        if P is None:
            print(f"  [SKIP] Missing sess['probes'][{probe}] paths.")
            continue

        ks_folder = Path(P["ks_folder"])
        ks_folder.mkdir(parents=True, exist_ok=True)

        try:
            sorting = run_sorter(
                sorter_name="kilosort4",
                recording=recording,
                folder=ks_folder,
                remove_existing_folder=remove_existing_folder,
                verbose=True,
                **params
            )

            # save dumps (SpikeInterface JSON dumps)
            recording.dump(ks_folder / "recording.json")
            sorting.dump(ks_folder / "sorting.json")

            # store per-probe
            sess["sortings"][probe] = sorting
            P["recording_json"] = ks_folder / "recording.json"
            P["sorting_json"]   = ks_folder / "sorting.json"

            print(f"  [OK] {probe}: Sorting + dumps saved.")

        except Exception as e:
            print(f"  [ERROR] {sess['session_name']} {probe} failed: {type(e).__name__}: {e}")
            sess.setdefault("error_sorting", {})
            sess["error_sorting"][probe] = f"{type(e).__name__}: {e}"

    # compat: anciennes clés -> probe00
    if "probe00" in sess.get("sortings", {}):
        sess["sorting"] = sess["sortings"]["probe00"]
        sess["recording_json"] = sess["probes"]["probe00"].get("recording_json", sess["probes"]["probe00"]["ks_folder"] / "recording.json")
        sess["sorting_json"]   = sess["probes"]["probe00"].get("sorting_json",   sess["probes"]["probe00"]["ks_folder"] / "sorting.json")

    return sess
