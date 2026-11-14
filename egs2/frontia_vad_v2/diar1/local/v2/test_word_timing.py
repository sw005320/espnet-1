"""Word spans of CTC segmentation: segments vs the frames of the tokens."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

cs = pytest.importorskip("ctc_segmentation")

from ctc_batch import token_word_spans  # noqa: E402

FRAME = 0.08


def _align(spikes, words, frames=40):
    """CTC segmentation of synthetic posteriors.

    Blank everywhere except one spike frame per token; spikes = [(frame, token id)].
    """
    config = cs.CtcSegmentationParameters(char_list=["<blank>", "a", "b", "c", "d"])
    config.index_duration = FRAME
    config.blank = 0
    gt, ubi = cs.prepare_token_list(config, [np.array(w) for w in words])
    lpz = np.full((frames, 5), -20.0)
    lpz[:, 0] = -0.01
    for t, k in spikes:
        lpz[t, :] = -20.0
        lpz[t, k] = -0.01
    timings, probs, _ = cs.ctc_segmentation(config, lpz, gt)
    segments = cs.determine_utterance_segments(
        config, ubi, probs, timings, ["w"] * len(words)
    )
    return timings, ubi, segments, config


def test_token_spans_cover_the_frames_of_each_word():
    # word 0 = tokens a b at frames 5 and 7; word 1 = c at 20; word 2 = d a at 30, 32
    spikes = [(5, 1), (7, 2), (20, 3), (30, 4), (32, 1)]
    timings, ubi, _, config = _align(spikes, [[1, 2], [3], [4, 1]])
    spans = token_word_spans(timings, ubi, config.index_duration_in_seconds)
    expected = [(5, 8), (20, 21), (30, 33)]  # first token frame .. last token frame + 1
    for (s, e), (fs, fe) in zip(spans, expected):
        assert s == pytest.approx(fs * FRAME)
        assert e == pytest.approx(fe * FRAME)


def test_segmentation_starts_words_early_and_closes_pauses():
    """Segments start up to 0.5 s before the first token.

    determine_utterance_segments places them so; this made the word labels of
    v2's stage 4 start early.
    """
    spikes = [(5, 1), (7, 2), (20, 3), (30, 4), (32, 1)]
    timings, ubi, segments, config = _align(spikes, [[1, 2], [3], [4, 1]])
    spans = token_word_spans(timings, ubi, config.index_duration_in_seconds)
    word1_start = segments[1][0]
    assert word1_start == pytest.approx(20 * FRAME - 0.5)
    assert spans[1][0] - word1_start == pytest.approx(0.5)
    # the token spans keep the pause between words 0 and 1 (0.64 s .. 1.60 s)
    assert spans[1][0] - spans[0][1] == pytest.approx(12 * FRAME)


def test_a_word_without_tokens_is_an_error():
    with pytest.raises(ValueError):
        token_word_spans(np.zeros(4), [1, 2], FRAME)
