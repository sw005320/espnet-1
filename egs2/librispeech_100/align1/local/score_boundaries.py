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
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

MFA_SILENCE = {"", "sil", "sp", "spn", "<eps>"}
STEP = 0.01


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


def rule_labels(words, min_pause: float, end_pad: float):
    """Speech runs of a word list under the labelling rule."""
    runs = []
    for s, e, _ in words:
        if runs and s - runs[-1][1] < min_pause:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    return [(s, e + end_pad) for s, e in runs]


def frames(runs, n):
    v = [False] * n
    for s, e in runs:
        for i in range(max(0, int(round(s / STEP))), min(n, int(round(e / STEP)))):
            v[i] = True
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mfa_zip", type=Path, required=True)
    ap.add_argument("--words_ctm", type=Path, required=True)
    ap.add_argument("--utt2dur", type=Path, help="utterance durations (seconds)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min_pauses", default="0.2,0.3,0.5")
    ap.add_argument("--end_pads", default="0,0.05,0.1,0.2,0.3")
    a = ap.parse_args()

    hyp = read_ctm(a.words_ctm)
    ref = read_mfa(a.mfa_zip, set(hyp))
    durs = {}
    if a.utt2dur:
        durs = {k: float(v) for k, v in (line.split()[:2] for line in a.utt2dur.open())}
    common = sorted(u for u in hyp if u in ref and ref[u])
    a.out.mkdir(parents=True, exist_ok=True)

    ds, de, mismatched = [], [], 0
    for u in common:
        h, r = hyp[u], ref[u]
        if len(h) != len(r):
            mismatched += 1
            continue
        for (hs, he, _), (rs, re_, _) in zip(h, r):
            ds.append(abs(hs - rs))
            de.append(abs(he - re_))

    def summ(xs):
        xs = sorted(xs)
        return (
            f"mean {1000 * statistics.mean(xs):.1f} ms, median "
            f"{1000 * statistics.median(xs):.1f} ms, p90 "
            f"{1000 * xs[int(0.9 * (len(xs) - 1))]:.1f} ms, <=20 ms "
            f"{100 * sum(x <= 0.02 for x in xs) / len(xs):.1f} %, <=50 ms "
            f"{100 * sum(x <= 0.05 for x in xs) / len(xs):.1f} %"
        )

    lines = [
        f"utterances scored: {len(common)} (word-count mismatch skipped: "
        f"{mismatched}); words: {len(ds)}",
        f"word start error: {summ(ds)}" if ds else "word start error: n/a",
        f"word end error:   {summ(de)}" if de else "word end error: n/a",
        "",
        "label-rule DetER % (miss / FA), start pad 0",
        "min_pause end_pad DetER miss FA",
    ]
    for mp in (float(x) for x in a.min_pauses.split(",")):
        for ep in (float(x) for x in a.end_pads.split(",")):
            miss = fa = speech = 0
            for u in common:
                end = max(
                    durs.get(u, 0.0),
                    max(e for _, e, _ in ref[u]),
                    max(e for _, e, _ in hyp[u]) + ep,
                )
                n = int(round(end / STEP)) + 1
                rv = frames([(s, e) for s, e, _ in ref[u]], n)
                hv = frames(rule_labels(hyp[u], mp, ep), n)
                speech += sum(rv)
                miss += sum(r and not h for r, h in zip(rv, hv))
                fa += sum(h and not r for r, h in zip(rv, hv))
            lines.append(
                f"{mp:.2f} {ep:.2f} {100 * (miss + fa) / speech:.2f} "
                f"{100 * miss / speech:.2f} {100 * fa / speech:.2f}"
            )
    text = "\n".join(lines) + "\n"
    (a.out / "boundaries.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
