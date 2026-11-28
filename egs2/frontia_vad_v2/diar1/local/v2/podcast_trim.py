#!/usr/bin/env python3
"""Find where a podcast episode's conversation ends, to cut the outro after it.

    python3 local/v2/podcast_trim.py --root ${VAD_CORPORA} --corpus ipse_dixit \
        --out local/v2/downloads/ipse_dixit.trim.tsv

Some podcasts close every episode with a clip of someone else's recording (Ipse
Dixit: an old song or radio sketch, 1-3 min) that the show's licence does not
cover. This transcribes the last --tail seconds of each episode with OWSM-CTC
in 30 s windows (hop --hop s) and takes the last window whose text has a
sign-off phrase ("thanks for having me", ...). The conversation is kept up to
that window's start, so a few seconds of speech may be lost, but none of the
outro is kept.

An episode is flagged, and kept only up to --fallback seconds before its end,
when no window has a sign-off or the sign-off is less than --min_outro seconds
from the end (the outro is then probably a song whose words matched).

Output columns: relative_path, duration (s), keep_end (s), method (signoff or
fallback), the matched text. The adapter keeps [0, keep_end) of each file.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

SIGNOFF = re.compile(
    r"(for having me|having me on|thanks? (you )?(so|very) much|thanks again|"
    r"thank you|appreciate (it|you|your time|that)|my pleasure|the pleasure|"
    r"look forward|best of luck|good luck|take care|thanks for (coming|joining|being))",
    re.I,
)
SR = 16000
WIN = 30


def load(path):
    """Mono 16 kHz float32 (libsndfile reads mp3)."""
    import soundfile as sf
    import torchaudio.functional as AF

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    return wav.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", help="default: downloads/<corpus>.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="<eng>")
    ap.add_argument("--tail", type=float, default=420.0)
    ap.add_argument("--hop", type=float, default=10.0)
    ap.add_argument("--min_outro", type=float, default=20.0)
    ap.add_argument("--fallback", type=float, default=240.0)
    ap.add_argument("--model_tag", default="espnet/owsm_ctc_v4_1B")
    ap.add_argument("--revision", default="6075bf3d42e6037d1ef7abd7c5558ade886e6eaa")
    a = ap.parse_args()

    from align import absolute_config, model_files

    from espnet2.bin.s2t_inference import Speech2Text

    manifest = Path(
        a.manifest or Path(__file__).parent / "downloads" / f"{a.corpus}.tsv"
    )
    rels = [
        line.split("\t")[1]
        for line in manifest.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    root, config, model = model_files(a.model_tag, a.revision)
    out = Path(a.out)
    conf = absolute_config(root, config, out.with_suffix(".owsm_config.yaml"))
    s2t = Speech2Text(
        s2t_train_config=str(conf),
        s2t_model_file=str(model),
        device="cuda" if torch.cuda.is_available() else "cpu",
        lang_sym=a.lang,
        task_sym="<asr>",
    )
    rows, flagged = [], 0
    for n, rel in enumerate(rels):
        wav = load(Path(a.root) / a.corpus / rel)
        dur = len(wav) / SR
        start = max(0.0, dur - a.tail)
        last = None
        t = start
        while t < dur - 1:
            seg = wav[int(t * SR) : int((t + WIN) * SR)]
            seg = np.pad(seg, (0, WIN * SR - len(seg)))
            with torch.no_grad():
                text = s2t.best_path(seg)[0][0]
            text = re.sub(r"<[^>]+>", "", text).strip()
            m = SIGNOFF.search(text)
            if m:
                last = (t, text[max(0, m.start() - 40) : m.end() + 40])
            t += a.hop
        if last is not None and dur - last[0] >= a.min_outro:
            keep, method, note = last[0], "signoff", last[1]
        else:
            keep, method = max(0.0, dur - a.fallback), "fallback"
            note = "no sign-off" if last is None else f"sign-off too late: {last[1]}"
            flagged += 1
        note = note.replace("\t", " ").replace("\n", " ")
        rows.append(f"{rel}\t{dur:.2f}\t{keep:.2f}\t{method}\t{note}\n")
        print(
            f"[{n + 1}/{len(rels)}] {rel} {dur:.0f} s keep {keep:.0f} s {method}",
            flush=True,
        )
    out.write_text(
        "# relative_path\tduration\tkeep_end\tmethod\tmatched_text\n"
        f"# podcast_trim.py --corpus {a.corpus} --tail {a.tail} --hop {a.hop} "
        f"--min_outro {a.min_outro} --fallback {a.fallback} ({a.model_tag}@{a.revision[:8]})\n"
        + "".join(rows)
    )
    print(f"{len(rows)} episodes, {flagged} flagged (fallback)")


if __name__ == "__main__":
    main()
