"""HMM state graphs over unit sequences, full-sum forward and Viterbi.

An utterance is a sequence of *segments*. A segment is a set of alternative
units, of which the path passes through exactly one, or through none when the
segment is skippable. A word's characters are one segment each (one unit, not
skippable); a pause between two words is one skippable segment whose
alternatives are the pause units.

Each unit is expanded into a left-to-right chain of ``num_states(unit)``
states with self-loops. A path starts in an entry state of the first segment
it does not skip and ends in the last state of a unit of the last segment it
does not skip; see ``build_graph`` for the transition probabilities.

The emission score of a state at frame t is the network's log-probability of
the state's output class (``unit * max_states + position``) at t, optionally
minus a log state prior.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union

import torch


@dataclass
class Segment:
    """A set of alternative units, optionally skippable.

    A skippable segment is entered with probability ``prob`` (shared equally
    by its units) and skipped otherwise; a mandatory segment is always
    entered, its units sharing probability 1.
    """

    units: Tuple[int, ...]
    skippable: bool = False
    prob: float = 0.5


@dataclass
class StateGraph:
    """Left-to-right HMM over one utterance.

    Attributes:
        state_class: output class of each state.
        state_unit: unit of each state.
        state_pos: position of each state within its unit (0-based).
        preds: per state, the (predecessor state, log transition prob) pairs,
            the self-loop included.
        init: (state, log prob) pairs of the states a path may start in.
        finals: states a path may end in.
        min_frames: the fewest frames any complete path needs.
    """

    state_class: List[int]
    state_unit: List[int]
    state_pos: List[int]
    preds: List[List[Tuple[int, float]]]
    init: List[Tuple[int, float]]
    finals: List[int]
    min_frames: int


def build_graph(
    segments: Sequence[Segment],
    num_states: Callable[[int], int],
    max_states: int,
    loop_prob: Union[float, Callable[[int], float]] = 0.5,
) -> StateGraph:
    """Expand a segment sequence into an HMM state graph.

    A state loops with its unit's loop probability. The last state of a unit
    moves on with the remaining probability, which is split over the entry
    states of the following segments: a skippable segment takes its ``prob``
    share and passes the rest on to the segment after it. The initial
    distribution is split the same way from the first segment. Inner states
    of a unit move to the next state of the unit.

    Args:
        segments: the utterance's segments in time order.
        num_states: number of HMM states of a unit (1 <= n <= max_states).
        max_states: states reserved per unit in the output classes.
        loop_prob: self-loop probability of every state, or of a unit's states.
    """
    loop_of = loop_prob if callable(loop_prob) else (lambda u: loop_prob)
    state_class: List[int] = []
    state_unit: List[int] = []
    state_pos: List[int] = []
    # (first state, last state) of each unit, per segment
    seg_units: List[List[Tuple[int, int]]] = []
    for seg in segments:
        if len(seg.units) == 0:
            raise ValueError("a segment needs at least one unit")
        if seg.skippable and not 0.0 < seg.prob < 1.0:
            raise ValueError(f"skippable segment prob must be in (0, 1): {seg.prob}")
        spans = []
        for u in seg.units:
            n = num_states(u)
            if not 1 <= n <= max_states:
                raise ValueError(f"unit {u}: {n} states, max_states {max_states}")
            lp = loop_of(u)
            if not 0.0 < lp < 1.0:
                raise ValueError(f"loop_prob must be in (0, 1), got {lp}")
            first = len(state_class)
            for k in range(n):
                state_class.append(u * max_states + k)
                state_unit.append(u)
                state_pos.append(k)
            spans.append((first, first + n - 1))
        seg_units.append(spans)
    if all(seg.skippable for seg in segments):
        raise ValueError("every segment is skippable; the path could be empty")

    # entry[k]: (entry state, prob) of a path that next enters a unit at or
    # after segment k; ends[k]: whether a path may end before segment k
    K = len(segments)
    entry: List[List[Tuple[int, float]]] = [[] for _ in range(K + 1)]
    ends = [True] * (K + 1)
    for k in range(K - 1, -1, -1):
        seg, spans = segments[k], seg_units[k]
        share = seg.prob if seg.skippable else 1.0
        here = [(first, share / len(spans)) for first, _ in spans]
        rest = (
            [(s, (1.0 - share) * p) for s, p in entry[k + 1]] if seg.skippable else []
        )
        entry[k] = here + rest
        ends[k] = seg.skippable and ends[k + 1]

    preds: List[List[Tuple[int, float]]] = [
        [(s, math.log(loop_of(state_unit[s])))] for s in range(len(state_class))
    ]
    finals: List[int] = []
    for k, spans in enumerate(seg_units):
        for first, last in spans:
            move = 1.0 - loop_of(state_unit[first])
            for s in range(first, last):
                preds[s + 1].append((s, math.log(move)))
            for t, p in entry[k + 1]:
                preds[t].append((last, math.log(move * p)))
            if ends[k + 1]:
                finals.append(last)

    # fewest frames: shortest unit of every mandatory segment
    shortest = sum(
        min(last - first + 1 for first, last in spans)
        for seg, spans in zip(segments, seg_units)
        if not seg.skippable
    )
    return StateGraph(
        state_class=state_class,
        state_unit=state_unit,
        state_pos=state_pos,
        preds=preds,
        init=[(s, math.log(p)) for s, p in entry[0]],
        finals=sorted(finals),
        min_frames=shortest,
    )


@dataclass
class PaddedGraphs:
    """A batch of state graphs as padded tensors (padding has -inf scores)."""

    state_class: torch.Tensor  # (B, S) long
    pred_idx: torch.Tensor  # (B, S, K) long
    pred_logp: torch.Tensor  # (B, S, K) float, -inf where unused
    init_logp: torch.Tensor  # (B, S) float, -inf where not initial
    final_mask: torch.Tensor  # (B, S) bool
    num_states: torch.Tensor  # (B,) long


def pad_graphs(graphs: Sequence[StateGraph], device=None) -> PaddedGraphs:
    """Stack state graphs into padded tensors."""
    B = len(graphs)
    S = max(len(g.state_class) for g in graphs)
    K = max(len(p) for g in graphs for p in g.preds)
    neg_inf = float("-inf")
    state_class = torch.zeros(B, S, dtype=torch.long)
    pred_idx = torch.zeros(B, S, K, dtype=torch.long)
    pred_logp = torch.full((B, S, K), neg_inf)
    init_logp = torch.full((B, S), neg_inf)
    final_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, g in enumerate(graphs):
        n = len(g.state_class)
        state_class[b, :n] = torch.tensor(g.state_class)
        for s, plist in enumerate(g.preds):
            for k, (p, lp) in enumerate(plist):
                pred_idx[b, s, k] = p
                pred_logp[b, s, k] = lp
        for s, lp in g.init:
            init_logp[b, s] = lp
        final_mask[b, g.finals] = True
    num_states = torch.tensor([len(g.state_class) for g in graphs])
    return PaddedGraphs(
        state_class=state_class.to(device),
        pred_idx=pred_idx.to(device),
        pred_logp=pred_logp.to(device),
        init_logp=init_logp.to(device),
        final_mask=final_mask.to(device),
        num_states=num_states.to(device),
    )


def state_emissions(log_probs: torch.Tensor, graphs: PaddedGraphs) -> torch.Tensor:
    """Gather per-state emission scores (B, T, S) from class scores (B, T, C)."""
    idx = graphs.state_class.unsqueeze(1).expand(-1, log_probs.size(1), -1)
    return log_probs.gather(2, idx)


def hmm_forward(
    emit: torch.Tensor, lengths: torch.Tensor, graphs: PaddedGraphs
) -> torch.Tensor:
    """Log-likelihood of each utterance, summed over all state paths.

    Args:
        emit: (B, T, S) per-state emission scores.
        lengths: (B,) number of valid frames.
        graphs: the padded state graphs.

    Returns:
        (B,) log-likelihoods; -inf when no path fits in the frames.
    """
    B, T, S = emit.shape
    # Impossible scores are held at a finite floor instead of -inf: a
    # logsumexp whose inputs are all -inf has NaN gradients (0 * NaN), which
    # would reach the network through padded and unreachable states.
    pred_logp = graphs.pred_logp.to(emit.dtype).clamp(min=_FLOOR)
    alpha = (graphs.init_logp.to(emit.dtype) + emit[:, 0]).clamp(min=_FLOOR)
    final = torch.full((B,), _FLOOR, dtype=emit.dtype, device=emit.device)
    final = _take_final(final, alpha, graphs.final_mask, lengths == 1)
    batch = torch.arange(B, device=emit.device)[:, None, None]
    for t in range(1, T):
        prev = alpha[batch, graphs.pred_idx]  # (B, S, K)
        alpha = torch.logsumexp(prev + pred_logp, dim=2) + emit[:, t]
        alpha = alpha.clamp(min=_FLOOR)
        final = _take_final(final, alpha, graphs.final_mask, lengths == t + 1)
    # an utterance whose best final score is at the floor has no complete path
    return final.masked_fill(final < _FLOOR / 2, float("-inf"))


# a score floor far below any real path score, safe to add twice in float32
_FLOOR = -1.0e30


def _take_final(final, alpha, final_mask, ends_here):
    if not bool(ends_here.any()):
        return final
    score = torch.logsumexp(alpha.masked_fill(~final_mask, _FLOOR), dim=1)
    return torch.where(ends_here, score, final)


@torch.no_grad()
def hmm_viterbi(
    emit: torch.Tensor, lengths: torch.Tensor, graphs: PaddedGraphs
) -> List[List[int]]:
    """Best state path of each utterance (one state index per valid frame).

    An utterance with no complete path gets an empty list.
    """
    B, T, S = emit.shape
    pred_logp = graphs.pred_logp.to(emit.dtype)
    delta = graphs.init_logp.to(emit.dtype) + emit[:, 0]
    last = delta.new_full((B, S), float("-inf"))
    last = torch.where((lengths == 1).unsqueeze(1), delta, last)
    batch = torch.arange(B, device=emit.device)[:, None, None]
    backptr = []
    for t in range(1, T):
        cand = delta[batch, graphs.pred_idx] + pred_logp  # (B, S, K)
        best, arg = cand.max(dim=2)
        backptr.append(graphs.pred_idx.gather(2, arg.unsqueeze(2)).squeeze(2))
        delta = best + emit[:, t]
        last = torch.where((lengths == t + 1).unsqueeze(1), delta, last)
    end = last.masked_fill(~graphs.final_mask, float("-inf"))
    paths = []
    for b in range(B):
        if not bool(torch.isfinite(end[b]).any()):
            paths.append([])
            continue
        s = int(end[b].argmax())
        path = [s]
        for t in range(int(lengths[b]) - 1, 0, -1):
            s = int(backptr[t - 1][b, s])
            path.append(s)
        paths.append(path[::-1])
    return paths
