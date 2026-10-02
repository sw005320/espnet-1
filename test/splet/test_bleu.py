"""Tests for the sacrebleu-backed corpus metrics."""

import json
import subprocess
import sys

import pytest

from splet.bin.scorer import main
from splet.corpus_metrics import bleu_setup, chrf_setup, sacrebleu_scoring, ter_setup
from splet.scorer_shared import corpus_scoring, load_corpus_modules

pytest.importorskip("sacrebleu")

REFS = [
    "The cat sat on the mat.",
    "Hello, world!",
    "It is raining again today, isn't it?",
    "We'll meet at the station at 5 p.m.",
]
HYPS = [
    "the cat sat on a mat",
    "Hello world!",
    "It rains again today, isn't it?",
    "We will meet at the station at 5 pm.",
]


def _write(path, lines):
    path.write_text(
        "".join(f"utt{i} {line}\n" for i, line in enumerate(lines)), encoding="utf-8"
    )


def _sacrebleu_cli(tmp_path, *flags):
    """Run the sacrebleu command the way st.sh stage 13 does."""
    ref = tmp_path / "ref.trn.detok"
    hyp = tmp_path / "hyp.trn.detok"
    ref.write_text("\n".join(REFS) + "\n", encoding="utf-8")
    hyp.write_text("\n".join(HYPS) + "\n", encoding="utf-8")
    out = subprocess.run(
        [sys.executable, "-m", "sacrebleu", *flags, str(ref), "-i", str(hyp)]
        + ["-m", "bleu", "chrf", "ter", "-w", "6"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {entry["name"]: entry for entry in json.loads(out)}


def _splet_score(tmp_path, capsys, config):
    ref = tmp_path / "ref.txt"
    hyp = tmp_path / "hyp.txt"
    _write(ref, REFS)
    _write(hyp, HYPS)
    conf = tmp_path / "score.yaml"
    conf.write_text(config, encoding="utf-8")
    argv = ["--pred", str(hyp), "--gt", str(ref), "--score_config", str(conf)]
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_matches_st_sh_case_sensitive_pass(tmp_path, capsys):
    """st.sh: sacrebleu ref -i hyp -m bleu chrf ter."""
    expected = _sacrebleu_cli(tmp_path)
    summary = _splet_score(
        tmp_path, capsys, "- name: bleu\n- name: chrf\n- name: ter\n"
    )

    for key, name in (("bleu", "BLEU"), ("chrf", "chrF2"), ("ter", "TER")):
        assert summary[key] == pytest.approx(expected[name]["score"], abs=1e-6)
        assert summary[f"{key}_signature"] == expected[name]["signature"]
    assert summary["num_utterances"] == 4


def test_matches_st_sh_lowercased_sacrebleu_call(tmp_path, capsys):
    """st.sh: sacrebleu -lc ref -i hyp -m bleu chrf ter.

    -lc lowercases BLEU only, so only the bleu entry takes lowercase: true.
    The remove_punctuation.pl step st.sh runs before this call is not tested.
    """
    expected = _sacrebleu_cli(tmp_path, "-lc")
    summary = _splet_score(
        tmp_path,
        capsys,
        "- name: bleu\n  lowercase: true\n- name: chrf\n- name: ter\n",
    )

    for key, name in (("bleu", "BLEU"), ("chrf", "chrF2"), ("ter", "TER")):
        assert summary[key] == pytest.approx(expected[name]["score"], abs=1e-6)
        assert summary[f"{key}_signature"] == expected[name]["signature"]
    assert "case:lc" in summary["bleu_signature"]


def test_corpus_bleu_is_not_the_mean_of_sentence_bleus():
    scorer = bleu_setup()
    corpus = sacrebleu_scoring(scorer, HYPS, REFS)["bleu"]
    sentences = [
        sacrebleu_scoring(scorer, [h], [r])["bleu"] for h, r in zip(HYPS, REFS)
    ]
    assert corpus != pytest.approx(sum(sentences) / len(sentences), abs=0.1)


def test_bleu_reports_brevity_penalty_and_lengths():
    result = sacrebleu_scoring(bleu_setup(), ["a b c"], ["a b c d e f"])
    assert result["bleu_hyp_len"] == 3
    assert result["bleu_ref_len"] == 6
    assert result["bleu_ratio"] == pytest.approx(0.5)
    assert result["bleu_bp"] < 1.0


def test_normalization_is_applied_to_both_sides():
    normalize = [{"name": "lowercase"}, {"name": "remove_punctuation", "keep": "'"}]
    hyps, refs = ["hello WORLD it's fine"], ["Hello, world! It's fine."]
    assert sacrebleu_scoring(bleu_setup(), hyps, refs)["bleu"] < 100.0
    for setup, name, perfect in (
        (bleu_setup, "bleu", 100.0),
        (chrf_setup, "chrf", 100.0),
        (ter_setup, "ter", 0.0),
    ):
        result = sacrebleu_scoring(setup(normalize=normalize), hyps, refs)
        assert result[name] == pytest.approx(perfect)


def test_utterances_are_matched_by_key_not_by_order():
    modules = load_corpus_modules([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in reversed(list(enumerate(HYPS)))}
    in_order = sacrebleu_scoring(bleu_setup(), HYPS, REFS)["bleu"]
    assert corpus_scoring(pred, modules, gt)["bleu"] == pytest.approx(in_order)


def test_hypothesis_without_reference_is_an_error():
    modules = load_corpus_modules([{"name": "bleu"}])
    with pytest.raises(KeyError, match="utt9"):
        corpus_scoring({"utt0": "a", "utt9": "b"}, modules, {"utt0": "a"})


def test_reference_without_hypothesis_is_warned_about(caplog):
    modules = load_corpus_modules([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {"utt0": HYPS[0]}
    with caplog.at_level("WARNING"):
        corpus_scoring(pred, modules, gt)
    assert "3 of 4 references have no hypothesis" in caplog.text
    assert "utt1" in caplog.text


def test_matching_keys_give_no_warning(caplog):
    modules = load_corpus_modules([{"name": "bleu"}])
    gt = {f"utt{i}": text for i, text in enumerate(REFS)}
    pred = {f"utt{i}": text for i, text in enumerate(HYPS)}
    with caplog.at_level("WARNING"):
        corpus_scoring(pred, modules, gt)
    assert caplog.text == ""


def test_bleu_needs_references():
    with pytest.raises(ValueError, match="references"):
        sacrebleu_scoring(bleu_setup(), ["a"], None)


def test_corpus_and_utterance_metrics_in_one_config(tmp_path, capsys):
    summary = _splet_score(tmp_path, capsys, "- name: wer\n- name: bleu\n")
    assert "wer" in summary
    assert "bleu" in summary


def test_list_metrics_shows_the_corpus_tier(capsys):
    assert main(["--list_metrics"]) == 0
    printed = capsys.readouterr().out
    for name in ("bleu", "chrf", "ter"):
        assert f"{name}\tcorpus" in printed
