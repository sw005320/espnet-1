"""Labels of transcript clips inside a pack (align.py --clip_mode)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align import clip_speech, pack_speech, read_clips  # noqa: E402


def test_pack_of_three_clips():
    # clips at 0-4 s, 4-7 s and 7-12 s of the pack; word spans inside each clip
    clips = [
        (
            0.0,
            4.0,
            [(0.5, 1.0), (1.3, 2.0), (3.5, 3.9)],
        ),  # 0.3 s pause, last word near the edge
        (4.0, 7.0, [(0.1, 0.8)]),  # first word near the clip start
        (7.0, 12.0, [(1.0, 2.0), (3.0, 4.0)]),  # 1.0 s pause
    ]
    out = pack_speech(clips, min_pause=0.5, pad_start=0.05, pad_end=0.4)
    assert out == pytest.approx(
        [
            (0.45, 2.4),  # 0.3 s pause joined, then padded
            (3.45, 4.0),  # end pad clipped at the clip edge (3.9 + 0.4 > 4.0)
            (4.05, 5.2),  # start pad stays inside the clip: 4.0 + 0.1 - 0.05
            (7.95, 9.4),  # the 1.0 s pause of clip 3 is kept
            (9.95, 11.4),
        ]
    )


def test_no_merge_across_clip_edges():
    # the last word of clip 1 ends 0.1 s before the edge and the first word of
    # clip 2 starts 0.1 s after it: a 0.2 s gap, below min_pause, but in two clips
    clips = [(0.0, 2.0, [(0.5, 1.9)]), (2.0, 4.0, [(0.1, 1.0)])]
    out = pack_speech(clips, min_pause=0.5, pad_start=0.0, pad_end=0.0)
    assert out == pytest.approx([(0.5, 1.9), (2.1, 3.0)])


def test_inter_clip_silence_is_kept():
    clips = [(0.0, 3.0, [(1.0, 1.5)]), (3.0, 6.0, [(1.0, 1.5)])]
    out = pack_speech(clips, min_pause=10.0, pad_start=0.1, pad_end=0.4)
    assert out == pytest.approx([(0.9, 1.9), (3.9, 4.9)])


def test_clip_speech_never_leaves_the_clip():
    assert clip_speech([(0.0, 1.0)], 1.2, 0.5, 0.3, 0.5) == pytest.approx([(0.0, 1.2)])


def test_read_clips(tmp_path):
    (tmp_path / "clips").write_text(
        "p+2\ta\t0\t16000\tone two\n"
        "p+2\tb\t16000\t40000\tthree\n"
        "q\tq\t0\t500000\tfour\n"
    )
    assert read_clips(tmp_path / "clips") == {
        "p+2": [("a", 0, 16000, "one two"), ("b", 16000, 40000, "three")],
        "q": [("q", 0, 500000, "four")],
    }


def test_vad_dump_writes_the_clips_of_a_pack(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("omniio")
    from vad_dump import VadDumpWriter

    fs = 16000
    with VadDumpWriter(tmp_path, "c", "read", "A", "CC BY 4.0", pack_seconds=5) as w:
        for k, sec in enumerate((2.0, 1.5, 2.0, 6.0)):
            x = np.zeros(int(sec * fs), dtype=np.float32)
            w.add(f"u{k}", x, fs, text=f"word {k}", lang_sym="<eng>")
    clips = read_clips(tmp_path / "clips")
    # u0..u2 (5.5 s) form one pack; u3 is longer than pack_seconds: its own clip
    assert clips["c-u0+3"] == [
        ("c-u0", 0, 32000, "word 0"),
        ("c-u1", 32000, 56000, "word 1"),
        ("c-u2", 56000, 88000, "word 2"),
    ]
    assert clips["c-u3"] == [("c-u3", 0, 96000, "word 3")]
