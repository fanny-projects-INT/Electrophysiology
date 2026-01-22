from pathlib import Path
from atlaselectrophysiology.extract_files import extract_data


def export_alf(sess: dict, stop_on_error: bool = False) -> dict:
    """
    Exporte les résultats Kilosort au format ALF (IBL) pour chaque probe.
    Utilise KS/probeXX[/sorter_output] + Rec/probeXX -> alf/probeXX

    stop_on_error:
      - False: continue même si une probe échoue
      - True : raise au premier problème
    """
    session = sess["session_name"]

    probes = sess.get("probes", None)
    if probes is None:
        probes = {"probe00": {"rec_folder": sess["rec_folder"], "ks_folder": sess["ks_folder"], "alf_folder": sess["alf_folder"]}}

    for probe, P in probes.items():
        rec_folder = Path(P["rec_folder"])
        ks_folder  = Path(P["ks_folder"])
        alf_folder = Path(P["alf_folder"])

        tag = f"{session}/{probe}"

        # --- checks de base ---
        if not rec_folder.exists():
            msg = f"[{tag}] ❌ rec_folder introuvable: {rec_folder}"
            if stop_on_error:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        sorter_output = ks_folder / "sorter_output"
        ks_input = sorter_output if sorter_output.exists() else ks_folder

        if not ks_input.exists():
            msg = f"[{tag}] ❌ ks_folder introuvable: {ks_input}"
            if stop_on_error:
                raise FileNotFoundError(msg)
            print(msg)
            continue

        alf_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{tag}] Export ALF…")
        print(f"  ks_input   : {ks_input}")
        print(f"  rec_folder : {rec_folder}")
        print(f"  alf_folder : {alf_folder}")

        try:
            extract_data(ks_input, rec_folder, alf_folder)
            print(f"[{tag}] ✅ Export ALF terminé")
        except Exception as e:
            msg = f"[{tag}] ❌ Export ALF échoué: {type(e).__name__}: {e}"
            if stop_on_error:
                raise
            print(msg)

    return sess
