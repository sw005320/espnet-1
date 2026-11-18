"""Scoring of word alignments: boundary errors and label-rule DetER."""

import statistics
from typing import Dict, List, Optional, Sequence, Tuple

Word = Tuple[float, float, str]  # (start s, end s, word)
STEP = 0.01  # seconds per frame of the DetER grid


def rule_labels(
    words: Sequence[Word], min_pause: float, end_pad: float
) -> List[Tuple[float, float]]:
    """Speech runs of a word list under a labelling rule.

    Gaps shorter than ``min_pause`` are filled and ``end_pad`` seconds are
    added to the end of every speech run; starts are not padded.
    """
    runs: List[List[float]] = []
    for s, e, _ in sorted(words):
        if runs and s - runs[-1][1] < min_pause:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    return [(s, e + end_pad) for s, e in runs]


def _frames(runs, n: int) -> List[bool]:
    v = [False] * n
    for s, e in runs:
        for i in range(max(0, int(round(s / STEP))), min(n, int(round(e / STEP)))):
            v[i] = True
    return v


def deter_counts(
    ref: Sequence[Word],
    hyp: Sequence[Word],
    duration: float,
    min_pause: float,
    end_pad: float,
) -> Tuple[int, int, int]:
    """(miss, false alarm, reference speech) frame counts of one utterance.

    Only the audio is scored: the grid covers [0, duration), and reference
    and hypothesis runs are cut at the end of the audio.
    """
    n = int(round(duration / STEP))
    rv = _frames([(s, e) for s, e, _ in ref], n)
    hv = _frames(rule_labels(hyp, min_pause, end_pad), n)
    miss = sum(r and not h for r, h in zip(rv, hv))
    fa = sum(h and not r for r, h in zip(rv, hv))
    return miss, fa, sum(rv)


def common_utterances(
    hyp: Dict[str, Sequence[Word]],
    ref: Dict[str, Sequence[Word]],
    durations: Dict[str, float],
) -> List[str]:
    """Utterances with a hypothesis, a non-empty reference and a duration.

    Raises:
        ValueError: when there are none, naming the counts and example ids of
            each side, so that a wrong split or file is easy to see.
    """
    common = sorted(u for u in hyp if ref.get(u) and u in durations)
    if not common:

        def ex(ids):
            return ", ".join(sorted(ids)[:3]) or "-"

        raise ValueError(
            "no utterance has a hypothesis, a reference and a duration: "
            f"hypothesis {len(hyp)} (e.g. {ex(hyp)}), reference {len(ref)} "
            f"(e.g. {ex(ref)}), durations {len(durations)} (e.g. {ex(durations)})"
        )
    return common


def boundary_errors(
    hyp: Dict[str, Sequence[Word]],
    ref: Dict[str, Sequence[Word]],
    utts: Sequence[str],
) -> Tuple[List[float], List[float], int]:
    """Absolute start and end errors of position-matched words.

    Also returns the number of utterances skipped because their word counts
    differ.
    """
    ds, de, mismatched = [], [], 0
    for u in utts:
        h, r = sorted(hyp[u]), sorted(ref[u])
        if len(h) != len(r):
            mismatched += 1
            continue
        for (hs, he, _), (rs, re_, _) in zip(h, r):
            ds.append(abs(hs - rs))
            de.append(abs(he - re_))
    return ds, de, mismatched


def summarize(xs: Sequence[float]) -> str:
    """Mean, median, p90 and shares within 20 / 50 ms of absolute errors."""
    xs = sorted(xs)
    return (
        f"mean {1000 * statistics.mean(xs):.1f} ms, median "
        f"{1000 * statistics.median(xs):.1f} ms, p90 "
        f"{1000 * xs[int(0.9 * (len(xs) - 1))]:.1f} ms, <=20 ms "
        f"{100 * sum(x <= 0.02 for x in xs) / len(xs):.1f} %, <=50 ms "
        f"{100 * sum(x <= 0.05 for x in xs) / len(xs):.1f} %"
    )


def deter_table(
    hyp: Dict[str, Sequence[Word]],
    ref: Dict[str, Sequence[Word]],
    durations: Dict[str, float],
    utts: Sequence[str],
    min_pauses: Sequence[float],
    end_pads: Sequence[float],
) -> List[Tuple[float, float, Optional[float], Optional[float], Optional[float]]]:
    """(min_pause, end_pad, DetER %, miss %, FA %) for every rule.

    The percentages are None when the reference has no speech frame.
    """
    rows = []
    for mp in min_pauses:
        for ep in end_pads:
            miss = fa = speech = 0
            for u in utts:
                m, f, sp = deter_counts(ref[u], hyp[u], durations[u], mp, ep)
                miss, fa, speech = miss + m, fa + f, speech + sp
            if speech == 0:
                rows.append((mp, ep, None, None, None))
            else:
                rows.append(
                    (
                        mp,
                        ep,
                        100 * (miss + fa) / speech,
                        100 * miss / speech,
                        100 * fa / speech,
                    )
                )
    return rows
