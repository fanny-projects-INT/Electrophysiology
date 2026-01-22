from spikeinterface.extractors import read_spikeglx


def load_recordings(sess: dict) -> dict:
    """
    Charge les recordings SpikeGLX pour toutes les probes présentes.
    Remplit:
      sess["recordings"][probe]  (multi-probe)
      sess["recording"]          (compat = probe00)
    """
    print(f"\n=== Loading recording: {sess['session_name']} ===")

    sess["recordings"] = {}

    for probe, P in sess["probes"].items():
        stream_name = P.get("stream_name", None)
        if stream_name is None:
            # fallback (au cas où)
            probe_idx = int(probe.replace("probe", ""))
            stream_name = f"imec{probe_idx}.ap"

        print(f"  -> loading {probe} ({stream_name})")

        rec = read_spikeglx(
            P["rec_folder"],
            stream_name=stream_name
        )

        sess["recordings"][probe] = rec

    # compat mono-probe
    sess["recording"] = sess["recordings"]["probe00"]
    return sess
