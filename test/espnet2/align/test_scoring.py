import pytest

from espnet2.align.scoring import (
    boundary_errors,
    common_utterances,
    deter_counts,
    deter_table,
    rule_labels,
)


def test_end_pad_does_not_score_past_the_audio():
    # 1 s of audio, reference and hypothesis both cover it all: the 0.3 s end
    # pad falls outside the audio and must not count as false alarm
    ref = [(0.0, 1.0, "a")]
    hyp = [(0.0, 1.0, "a")]
    miss, fa, speech = deter_counts(ref, hyp, duration=1.0, min_pause=0.5, end_pad=0.3)
    assert (miss, fa, speech) == (0, 0, 100)


def test_rule_fills_short_gaps_and_pads_ends():
    words = [(0.0, 0.5, "a"), (0.6, 1.0, "b"), (2.0, 2.5, "c")]
    assert rule_labels(words, 0.3, 0.1) == [(0.0, 1.1), (2.0, 2.6)]


def test_counts_miss_and_false_alarm():
    ref = [(0.0, 0.5, "a"), (1.0, 1.5, "b")]
    hyp = [(0.0, 0.5, "a"), (0.6, 1.0, "x")]
    miss, fa, speech = deter_counts(ref, hyp, 2.0, min_pause=0.0, end_pad=0.0)
    assert (miss, fa, speech) == (50, 40, 100)


def test_no_common_utterance_is_a_clear_error():
    with pytest.raises(ValueError, match="hypothesis 1 .*reference 1 "):
        common_utterances(
            {"dev-1": [(0.0, 1.0, "a")]},
            {"test-9": [(0.0, 1.0, "a")]},
            {"dev-1": 1.0},
        )


def test_no_reference_speech_gives_none():
    rows = deter_table(
        {"u": [(0.0, 1.0, "a")]},
        {"u": [(5.0, 6.0, "a")]},  # all of it after the audio end
        {"u": 2.0},
        ["u"],
        [0.5],
        [0.3],
    )
    assert rows == [(0.5, 0.3, None, None, None)]


def test_boundary_errors_skip_word_count_mismatch():
    ds, de, skipped = boundary_errors(
        {"u": [(0.1, 0.5, "a"), (0.6, 1.0, "b")], "v": [(0.0, 1.0, "ab")]},
        {"u": [(0.0, 0.5, "a"), (0.6, 1.2, "b")], "v": [(0, 0.5, "a"), (0.5, 1, "b")]},
        ["u", "v"],
    )
    assert skipped == 1
    assert [round(x, 3) for x in ds] == [0.1, 0.0]
    assert [round(x, 3) for x in de] == [0.0, 0.2]
