"""CTC Viterbi alignment: the frames each token is emitted at."""

from typing import List, Sequence, Tuple

import torch


@torch.no_grad()
def ctc_viterbi(
    log_probs: torch.Tensor, tokens: Sequence[int], blank: int = 0
) -> List[Tuple[int, int]]:
    """Best CTC path of one utterance.

    Args:
        log_probs: (T, V) log-probabilities of one utterance (valid frames only).
        tokens: the target token ids (no blanks).
        blank: the blank id.

    Returns:
        For every token, (first frame, end frame) of the frames the best path
        emits it at (end exclusive); empty when no path fits in T frames.
    """
    T = log_probs.size(0)
    L = len(tokens)
    if L == 0:
        return []
    ext = [blank]
    for t in tokens:
        ext += [int(t), blank]
    S = len(ext)
    lp = log_probs[:, ext].float()  # (T, S)
    neg = torch.tensor(float("-inf"))
    # transitions into s: from s, s-1, and s-2 when ext[s] is a token that
    # differs from ext[s-2]
    skip = torch.zeros(S, dtype=torch.bool)
    for s in range(2, S):
        skip[s] = ext[s] != blank and ext[s] != ext[s - 2]
    delta = torch.full((S,), float("-inf"))
    delta[0] = lp[0, 0]
    if S > 1:
        delta[1] = lp[0, 1]
    back = torch.zeros(T, S, dtype=torch.long)
    for t in range(1, T):
        stay = delta
        step = torch.cat([neg.view(1), delta[:-1]])
        jump = torch.cat([neg.expand(2), delta[:-2]])
        jump = torch.where(skip, jump, neg.expand(S))
        cand = torch.stack([stay, step, jump])  # (3, S)
        best, arg = cand.max(dim=0)
        back[t] = torch.arange(S) - arg
        delta = best + lp[t]
    ends = [S - 1, S - 2] if S > 1 else [S - 1]
    s = max(ends, key=lambda i: float(delta[i]))
    if not torch.isfinite(delta[s]):
        return []
    path = [s]
    for t in range(T - 1, 0, -1):
        s = int(back[t, s])
        path.append(s)
    path = path[::-1]
    spans: List[List[int]] = [[-1, -1] for _ in range(L)]
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if spans[k][0] < 0:
                spans[k][0] = t
            spans[k][1] = t + 1
    return [(a, b) for a, b in spans]
