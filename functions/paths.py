from pathlib import Path


def build_paths(session_name: str, data_root: Path, db_path: Path | None = None) -> dict:
    base_folder = data_root / session_name

    rec_root = base_folder / "Rec"
    ks_root  = base_folder / "KS"
    alf_root = base_folder / "alf"
    shift_path = base_folder / "shift.txt"

    # probes présentes (toujours probe00, probe01 optionnelle)
    if not (rec_root / "probe00").is_dir():
        raise FileNotFoundError(f"[{session_name}] Missing folder: {rec_root / 'probe00'}")

    present_probes = ["probe00"] + (["probe01"] if (rec_root / "probe01").is_dir() else [])
    multi_probe = (len(present_probes) == 2)

    # dict par probe
    probes = {}
    for p in present_probes:
        probe_idx = int(p.replace("probe", ""))  # probe00 -> 0 ; probe01 -> 1

        rec_folder = rec_root / p
        ks_folder  = ks_root  / p
        alf_folder = alf_root / p

        ks_folder.mkdir(parents=True, exist_ok=True)
        alf_folder.mkdir(parents=True, exist_ok=True)

        probes[p] = {
            "probe": p,
            "probe_idx": probe_idx,
            "stream_name": f"imec{probe_idx}.ap",   # ✅ clé pour read_spikeglx
            "rec_folder": rec_folder,
            "ks_folder": ks_folder,
            "alf_folder": alf_folder,
            "recording_json": ks_folder / "recording.json",
            "sorting_json":   ks_folder / "sorting.json",
            "output_folder":  ks_folder,
        }

    # compat: anciennes clés -> probe00 (pour ne rien casser)
    p0 = probes["probe00"]

    return {
        "session_name": session_name,
        "base_folder": base_folder,

        # compat (pipeline existant)
        "rec_folder": p0["rec_folder"],
        "ks_folder": p0["ks_folder"],
        "alf_folder": p0["alf_folder"],
        "shift_path": shift_path,
        "recording_json": p0["recording_json"],
        "sorting_json": p0["sorting_json"],
        "output_folder": p0["output_folder"],

        # optionnel
        "db_path": db_path,

        # multi-probe
        "present_probes": present_probes,
        "multi_probe": multi_probe,
        "probes": probes,
    }
