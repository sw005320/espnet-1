#!/usr/bin/env python3
"""Find the parts of podcast episodes that the show's licence may not cover.

    python3 local/v2/podcast_trim.py --root ${VAD_CORPORA} --corpus ipse_dixit \
        --mode outro --out local/v2/downloads/ipse_dixit.trim.tsv
    python3 local/v2/podcast_trim.py --root ${VAD_CORPORA} --corpus astronomy_cast \
        --mode ads --out local/v2/downloads/astronomy_cast.trim.tsv

Both modes transcribe with OWSM-CTC in 30 s windows and write the intervals of
each file to keep.

--mode outro: some podcasts close every episode with a clip of someone else's
recording (Ipse Dixit: an old song or radio sketch, 1-3 min). The last --tail
seconds are transcribed (hop --hop s) and the last window whose text has a
sign-off phrase ("thanks for having me", ...) ends the kept part, at that
window's start: a few seconds of speech may be lost, none of the outro is kept.
An episode is flagged, and kept only up to --fallback seconds before its end,
when no window has a sign-off or the sign-off is less than --min_outro seconds
from the end (the outro is then probably a song whose words matched).

--mode ads: some podcasts have host-read sponsor messages (Astronomy Cast in its
sponsored years). The whole episode is transcribed (hop --hop s); every window
whose text has a sponsor phrase ("brought to you by", "promo code", "dot com
slash", ...) is dropped whole, and drops less than --merge_gap seconds apart
are joined. A window is 30 s, so each drop has up to 30 s of margin around the
phrase. A phrase within 30 characters of a match of --own (the show's Patreon,
its own website) belongs to the show's own announcements and is kept.

Output columns: relative_path, duration (s), keep (start-end intervals in
seconds, joined by ";"), method, matched text. The adapter keeps only the listed
intervals of each file.
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
SPONSOR = re.compile(
    r"(brought to you by|sponsored by|is sponsored|our sponsors?\b|promo code|"
    r"offer code|use (the )?code|coupon|\.com ?slash|dot com slash|percent off|"
    r"dollars off|free trial|for more information visit|thanks to our sponsor)",
    re.I,
)
SR = 16000
WIN = 30


def sponsor(text, own):
    """The first sponsor phrase in text with no match of `own` within 30 characters.

    `own` matches the show's own announcements (its Patreon, its website), which
    use the same phrases as sponsor messages but are part of the show.
    """
    for m in SPONSOR.finditer(text):
        if not own.search(text[max(0, m.start() - 30) : m.end() + 30]):
            return m
    return None


def load(path):
    """Mono 16 kHz float32 (libsndfile reads mp3)."""
    import soundfile as sf
    import torchaudio.functional as AF

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    return wav.numpy()


def windows(s2t, wav, start, hop):
    """(start time, text) of 30 s windows from `start` to the end, `hop` s apart."""
    dur = len(wav) / SR
    t = start
    while t < dur - 1:
        # int() of both ends can give WIN * SR + 1 samples
        seg = wav[int(t * SR) :][: WIN * SR]
        seg = np.pad(seg, (0, WIN * SR - len(seg)))
        with torch.no_grad():
            text = s2t.best_path(seg)[0][0]
        yield t, re.sub(r"<[^>]+>", "", text).strip()
        t += hop


def snippet(text, m):
    return text[max(0, m.start() - 40) : m.end() + 40].replace("\t", " ")


def outro(s2t, wav, a):
    dur = len(wav) / SR
    last = None
    for t, text in windows(s2t, wav, max(0.0, dur - a.tail), a.hop):
        m = SIGNOFF.search(text)
        if m:
            last = (t, snippet(text, m))
    if last is not None and dur - last[0] >= a.min_outro:
        return [(0.0, last[0])], "signoff", last[1]
    note = "no sign-off" if last is None else f"sign-off too late: {last[1]}"
    return [(0.0, max(0.0, dur - a.fallback))], "fallback", note


def ads(s2t, wav, a):
    dur = len(wav) / SR
    drops, notes = [], []
    for t, text in windows(s2t, wav, 0.0, a.hop):
        m = sponsor(text, a.own)
        if not m:
            continue
        s, e = t, min(t + WIN, dur)
        if drops and s - drops[-1][1] < a.merge_gap:
            drops[-1] = (drops[-1][0], e)
        else:
            drops.append((s, e))
        notes.append(f"{t:.0f}s: {snippet(text, m)}")
    keep, prev = [], 0.0
    for s, e in drops:
        if s > prev:
            keep.append((prev, s))
        prev = max(prev, e)
    if prev < dur:
        keep.append((prev, dur))
    return keep, ("ads" if drops else "clean"), " | ".join(notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mode", choices=("outro", "ads"), required=True)
    ap.add_argument("--manifest", help="default: downloads/<corpus>.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="<eng>")
    ap.add_argument("--hop", type=float, default=None, help="outro 10, ads 15")
    ap.add_argument("--tail", type=float, default=420.0)
    ap.add_argument("--min_outro", type=float, default=20.0)
    ap.add_argument("--fallback", type=float, default=240.0)
    ap.add_argument("--merge_gap", type=float, default=30.0)
    ap.add_argument(
        "--own",
        default=r"patreon|brought to you by you\b",
        help="regex of the show's own announcements; sponsor phrases near it are kept",
    )
    ap.add_argument("--model_tag", default="espnet/owsm_ctc_v4_1B")
    ap.add_argument("--revision", default="6075bf3d42e6037d1ef7abd7c5558ade886e6eaa")
    a = ap.parse_args()
    if a.hop is None:
        a.hop = 10.0 if a.mode == "outro" else 15.0
    own_pattern = a.own
    a.own = re.compile(a.own, re.I)

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
    run = outro if a.mode == "outro" else ads
    rows, counts = [], {}
    for n, rel in enumerate(rels):
        wav = load(Path(a.root) / a.corpus / rel)
        dur = len(wav) / SR
        keep, method, note = run(s2t, wav, a)
        counts[method] = counts.get(method, 0) + 1
        kept = sum(e - s for s, e in keep)
        spans = ";".join(f"{s:.2f}-{e:.2f}" for s, e in keep)
        rows.append(f"{rel}\t{dur:.2f}\t{spans}\t{method}\t{note}\n")
        print(
            f"[{n + 1}/{len(rels)}] {rel} {dur:.0f} s, kept {kept:.0f} s, {method}",
            flush=True,
        )
    opts = f"--mode {a.mode} --hop {a.hop}"
    opts += (
        f" --tail {a.tail} --min_outro {a.min_outro} --fallback {a.fallback}"
        if a.mode == "outro"
        else f" --merge_gap {a.merge_gap} --own '{own_pattern}'"
    )
    out.write_text(
        "# relative_path\tduration\tkeep\tmethod\tmatched_text\n"
        f"# podcast_trim.py --corpus {a.corpus} {opts} "
        f"({a.model_tag}@{a.revision[:8]})\n" + "".join(rows)
    )
    print(f"{len(rows)} episodes: {counts}")


if __name__ == "__main__":
    main()
