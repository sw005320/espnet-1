#!/usr/bin/env python3
"""Score aligned words against the LibriSpeech MFA alignments.

Two measures over the utterances present in both:

1. Word boundary error: for words matched by position (same transcript, so
   the i-th word of the hypothesis is the i-th non-silent MFA word), the
   absolute start and end differences: mean, median, p90, and the share
   within 20 and 50 ms.
2. Label-rule DetER: speech labels made from the aligned words by a rule
   (gaps shorter than ``min_pause`` filled, ``end_pad`` seconds added to the
   end of every speech run, nothing added to starts), scored on a 10 ms grid
   against the MFA words over the whole utterance:
   DetER = (miss + false alarm) / reference speech.

Usage:
    score_boundaries.py --mfa_zip librispeech_alignments.zip \
        --words_ctm exp/.../dev_clean/words.ctm --out exp/.../dev_clean/score
"""

import argparse
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import humanfriendly

from espnet2.align.scoring import (
    boundary_errors,
    common_utterances,
    deter_table,
    summarize,
)

MFA_SILENCE = {"", "sil", "sp", "spn", "<eps>"}


def read_textgrid_words(text: str):
    """[(start, end, word)] of the non-silent intervals of the "words" tier."""
    tiers = re.split(r"\n\s*item \[\d+\]:", text)
    for tier in tiers:
        if re.search(r'name = "words"', tier):
            out = []
            for m in re.finditer(
                r'xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"', tier
            ):
                word = m.group(3).strip()
                if word.lower() not in MFA_SILENCE:
                    out.append((float(m.group(1)), float(m.group(2)), word))
            return out
    return []


def read_mfa(zip_path: Path, uids):
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if not n.endswith(".TextGrid"):
                continue
            uid = n.rsplit("/", 1)[-1][: -len(".TextGrid")]
            if uid in uids:
                out[uid] = read_textgrid_words(z.read(n).decode("utf-8", "replace"))
    return out


def read_ctm(path: Path):
    out = defaultdict(list)
    for line in path.open():
        uid, _, start, dur, word = line.split()[:5]
        s = float(start)
        out[uid].append((s, s + float(dur), word))
    for v in out.values():
        v.sort()
    return out


def read_durations(utt2dur=None, utt2num_samples=None, fs="16k"):
    """Utterance durations in seconds, from utt2dur or utt2num_samples."""
    if utt2dur is not None:
        return {k: float(v) for k, v in (ln.split()[:2] for ln in utt2dur.open())}
    if utt2num_samples is not None:
        rate = humanfriendly.parse_size(fs)
        return {
            k: int(v) / rate
            for k, v in (ln.split()[:2] for ln in utt2num_samples.open())
        }
    raise SystemExit("give --utt2dur or --utt2num_samples (with --fs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mfa_zip", type=Path, required=True)
    ap.add_argument("--words_ctm", type=Path, required=True)
    ap.add_argument("--utt2dur", type=Path, help="utterance durations (seconds)")
    ap.add_argument("--utt2num_samples", type=Path, help="utterance lengths (samples)")
    ap.add_argument("--fs", default="16k", help="sampling rate of utt2num_samples")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min_pauses", default="0.2,0.3,0.5")
    ap.add_argument("--end_pads", default="0,0.05,0.1,0.2,0.3")
    a = ap.parse_args()

    hyp = read_ctm(a.words_ctm)
    ref = read_mfa(a.mfa_zip, set(hyp))
    durs = read_durations(a.utt2dur, a.utt2num_samples, a.fs)
    try:
        utts = common_utterances(hyp, ref, durs)
    except ValueError as e:
        raise SystemExit(f"{a.words_ctm}: {e}")
    a.out.mkdir(parents=True, exist_ok=True)

    ds, de, mismatched = boundary_errors(hyp, ref, utts)
    lines = [
        f"utterances scored: {len(utts)} (word-count mismatch skipped: "
        f"{mismatched}); words: {len(ds)}",
        f"word start error: {summarize(ds)}" if ds else "word start error: n/a",
        f"word end error:   {summarize(de)}" if de else "word end error: n/a",
        "",
        "label-rule DetER % (miss / FA), start pad 0, scored within the audio",
        "min_pause end_pad DetER miss FA",
    ]
    rows = deter_table(
        hyp,
        ref,
        durs,
        utts,
        [float(x) for x in a.min_pauses.split(",")],
        [float(x) for x in a.end_pads.split(",")],
    )
    for mp, ep, d, m, f in rows:
        if d is None:
            lines.append(f"{mp:.2f} {ep:.2f} n/a (no reference speech)")
        else:
            lines.append(f"{mp:.2f} {ep:.2f} {d:.2f} {m:.2f} {f:.2f}")
    text = "\n".join(lines) + "\n"
    (a.out / "boundaries.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
