import itertools

import torch

from espnet2.align.ctc_align import ctc_viterbi


def _collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


def _brute(log_probs, tokens):
    T, V = log_probs.shape
    best, best_path = float("-inf"), None
    for path in itertools.product(range(V), repeat=T):
        if _collapse(path) != list(tokens):
            continue
        score = float(sum(log_probs[t, p] for t, p in enumerate(path)))
        if score > best:
            best, best_path = score, path
    return best_path


def _spans(path, n):
    """Frames of each token occurrence in a frame-label path."""
    spans, k, prev = [[-1, -1] for _ in range(n)], -1, 0
    for t, p in enumerate(path):
        if p != 0 and (p != prev or prev == 0):
            k += 1
            spans[k][0] = t
        if p != 0:
            spans[k][1] = t + 1
        prev = p
    return [tuple(s) for s in spans]


def test_matches_brute_force():
    torch.manual_seed(0)
    for tokens in ([1, 2], [1, 1], [2, 1, 2]):
        for T in (3, 4, 5):
            lp = torch.randn(T, 3).log_softmax(-1)
            ref = _brute(lp, tokens)
            got = ctc_viterbi(lp, tokens)
            if ref is None:
                assert got == []
            else:
                assert got == _spans(ref, len(tokens))


def test_repeated_token_needs_blank():
    lp = torch.randn(2, 3).log_softmax(-1)
    assert ctc_viterbi(lp, [1, 1]) == []  # "1 _ 1" needs 3 frames
