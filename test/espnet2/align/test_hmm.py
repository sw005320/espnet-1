import itertools
import math

import pytest
import torch

from espnet2.align.hmm import (
    Segment,
    build_graph,
    hmm_forward,
    hmm_viterbi,
    pad_graphs,
    state_emissions,
)


def _paths(graph, T):
    """Every state path of length T, with its score.

    A path starts in an initial state and ends in a final state; its score
    adds the initial, transition and final log-probabilities.
    """
    preds = {s: dict(p) for s, p in enumerate(graph.preds)}
    init = dict(graph.init)
    finals = dict(graph.finals)
    n = len(graph.state_class)
    for seq in itertools.product(range(n), repeat=T):
        if seq[0] not in init or seq[-1] not in finals:
            continue
        score = init[seq[0]] + finals[seq[-1]]
        ok = True
        for a, b in zip(seq, seq[1:]):
            if a not in preds[b]:
                ok = False
                break
            score += preds[b][a]
        if ok:
            yield seq, score


def _brute(graph, emit, T):
    """log sum over all paths of transition + emission scores, and the best path."""
    total = []
    best = (float("-inf"), None)
    for seq, trans in _paths(graph, T):
        s = trans + sum(emit[t, st] for t, st in enumerate(seq))
        total.append(s)
        if float(s) > best[0]:
            best = (float(s), list(seq))
    if not total:
        return torch.tensor(float("-inf")), None
    return torch.logsumexp(torch.stack([torch.as_tensor(x) for x in total]), 0), best[1]


CASES = [
    # (segments, states per unit)
    ([Segment((1,)), Segment((2,))], 1),
    ([Segment((1,)), Segment((2,))], 2),
    ([Segment((1,)), Segment((2,))], 3),
    ([Segment((0,), True), Segment((1,)), Segment((0, 3), True), Segment((2,))], 1),
    ([Segment((0,), True), Segment((1,)), Segment((0, 3), True), Segment((2,))], 2),
    ([Segment((1,)), Segment((0,), True), Segment((2,)), Segment((0,), True)], 2),
]


def _num_states(S):
    return lambda u: S


@pytest.mark.parametrize("segments,S", CASES)
@pytest.mark.parametrize("T", [2, 4, 6])
def test_forward_matches_brute_force(segments, S, T):
    torch.manual_seed(T * 10 + S)
    max_states = 3
    C = 4 * max_states
    g = build_graph(segments, _num_states(S), max_states, loop_prob=0.4)
    logits = torch.randn(1, T, C, dtype=torch.float64, requires_grad=True)
    log_probs = logits.log_softmax(-1)
    pg = pad_graphs([g])
    emit = state_emissions(log_probs, pg)
    ll = hmm_forward(emit, torch.tensor([T]), pg)
    ref, best = _brute(g, emit[0], T)
    if best is None:
        assert torch.isinf(ll).all()
        assert T < g.min_frames
        return
    assert T >= g.min_frames
    torch.testing.assert_close(ll[0], ref)
    (gf,) = torch.autograd.grad(ll[0], logits, retain_graph=True)
    (gr,) = torch.autograd.grad(ref, logits)
    torch.testing.assert_close(gf, gr)
    path = hmm_viterbi(emit.detach(), torch.tensor([T]), pg)[0]
    assert path == best


def test_batch_and_padding_match_single():
    torch.manual_seed(0)
    max_states, C = 2, 8
    segs = [
        [Segment((1,)), Segment((0,), True), Segment((2,))],
        [Segment((3,)), Segment((2,)), Segment((1,))],
    ]
    graphs = [build_graph(s, _num_states(2), max_states) for s in segs]
    lengths = torch.tensor([7, 9])
    log_probs = torch.randn(2, 9, C, dtype=torch.float64).log_softmax(-1)
    pg = pad_graphs(graphs)
    batched = hmm_forward(state_emissions(log_probs, pg), lengths, pg)
    for b, g in enumerate(graphs):
        p1 = pad_graphs([g])
        L = int(lengths[b])
        single = hmm_forward(
            state_emissions(log_probs[b : b + 1, :L], p1), lengths[b : b + 1], p1
        )
        torch.testing.assert_close(batched[b], single[0])
    paths = hmm_viterbi(state_emissions(log_probs, pg), lengths, pg)
    assert [len(p) for p in paths] == [7, 9]


def test_viterbi_respects_topology():
    torch.manual_seed(1)
    g = build_graph(
        [Segment((1,)), Segment((0,), True), Segment((2,))], _num_states(3), 3
    )
    pg = pad_graphs([g])
    T = 12
    emit = state_emissions(torch.randn(1, T, 9).log_softmax(-1), pg)
    path = hmm_viterbi(emit, torch.tensor([T]), pg)[0]
    allowed = {s: {p for p, _ in g.preds[s]} for s in range(len(g.state_class))}
    assert path[0] in dict(g.init)
    assert path[-1] in dict(g.finals)
    for a, b in zip(path, path[1:]):
        assert a in allowed[b]
    # every unit is visited for at least its number of states
    runs = {}
    for s in path:
        runs[g.state_unit[s]] = runs.get(g.state_unit[s], 0) + 1
    assert runs[1] >= 3 and runs[2] >= 3


def test_too_short_gives_minus_inf():
    g = build_graph([Segment((1,)), Segment((2,))], _num_states(3), 3)
    assert g.min_frames == 6
    pg = pad_graphs([g])
    emit = state_emissions(torch.randn(1, 5, 9).log_softmax(-1), pg)
    assert torch.isinf(hmm_forward(emit, torch.tensor([5]), pg)).all()
    assert hmm_viterbi(emit, torch.tensor([5]), pg) == [[]]


def test_all_skippable_is_rejected():
    with pytest.raises(ValueError):
        build_graph([Segment((0,), True)], _num_states(1), 1)


def test_transition_probs():
    # out-going probability (loop, moves and the end probability of a final
    # state) is 1 for every state; skips pass on (1 - prob)
    segs = [
        Segment((0,), True, 0.5),
        Segment((1,)),
        Segment((0, 3), True, 0.4),
        Segment((2,)),
    ]
    g = build_graph(segs, lambda u: 1, 1, loop_prob=lambda u: 0.75 if u == 0 else 0.5)
    out = {s: 0.0 for s in range(len(g.state_class))}
    for s, plist in enumerate(g.preds):
        for p, lp in plist:
            out[p] += math.exp(lp)
    for s, lp in g.finals:
        out[s] += math.exp(lp)
    for total in out.values():
        assert total == pytest.approx(1.0)
    assert sum(math.exp(lp) for _, lp in g.init) == pytest.approx(1.0)
    # state of unit 1 -> pause units get 0.4 / 2 each, unit 2 gets 0.6
    s1 = g.state_unit.index(1)
    moves = sorted(
        (g.state_unit[t], math.exp(lp))
        for t, plist in enumerate(g.preds)
        for p, lp in plist
        if p == s1 and t != s1
    )
    assert moves == [
        (0, pytest.approx(0.5 * 0.2)),
        (2, pytest.approx(0.5 * 0.6)),
        (3, pytest.approx(0.5 * 0.2)),
    ]


def test_gradients_finite_with_padding_and_unreachable_states():
    # graphs of different sizes (padded states) and an utterance too short
    # for its HMM must not give NaN gradients to the network
    torch.manual_seed(2)
    graphs = [
        build_graph([Segment((1,)), Segment((2,))], _num_states(3), 3),
        build_graph(
            [Segment((1,)), Segment((0,), True), Segment((2,)), Segment((3,))],
            _num_states(3),
            3,
        ),
    ]
    pg = pad_graphs(graphs)
    logits = torch.randn(2, 8, 12, requires_grad=True)
    lengths = torch.tensor([8, 5])  # the second needs 9 frames
    ll = hmm_forward(state_emissions(logits.log_softmax(-1), pg), lengths, pg)
    assert torch.isfinite(ll[0]) and torch.isinf(ll[1])
    ll.masked_fill(~torch.isfinite(ll), 0.0).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() == 0


def test_final_transitions_count():
    # a mandatory 1-state unit then an optional 1-state pause entered with
    # probability 0.9, self-loops 0.5, two frames, equal emissions:
    # no pause = 0.5 (loop) * 0.5 * 0.1 (leave, skip the pause) = 0.025,
    # pause = 0.5 * 0.9 (enter the pause) * 0.5 (leave it) = 0.225
    g = build_graph([Segment((1,)), Segment((0,), True, 0.9)], lambda u: 1, 1, 0.5)
    pg = pad_graphs([g])
    emit = torch.zeros(1, 2, len(g.state_class), dtype=torch.float64)
    path = hmm_viterbi(emit, torch.tensor([2]), pg)[0]
    assert [g.state_unit[s] for s in path] == [1, 0]
    ll = hmm_forward(emit, torch.tensor([2]), pg)
    torch.testing.assert_close(
        ll[0], torch.tensor(math.log(0.025 + 0.225), dtype=torch.float64)
    )
