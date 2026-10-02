#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""BLEU, chrF and TER, computed by sacrebleu.

These are what ``egs2/TEMPLATE/st1/st.sh`` reports: stage 13 runs
``sacrebleu -m bleu chrf ter`` over the detokenized hypotheses. They are
called here rather than reimplemented, and every result carries the
signature sacrebleu prints next to its score, e.g.
``nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0``. Two BLEU
numbers are only comparable when their signatures match, so a score written
out without one cannot be checked against anything.

Keyword arguments in the score config go straight to the sacrebleu metric
class (``sacrebleu.metrics.BLEU``, ``CHRF`` or ``TER``) and so use its names,
not the command line's::

    - name: bleu
      tokenize: zh        # sacrebleu --tokenize zh
      lowercase: true     # sacrebleu -lc

Note that ``-lc`` on the command line only lowercases BLEU. chrF stays case
sensitive and TER is case insensitive unless ``case_sensitive: true`` is
given, so st.sh's ``sacrebleu -lc`` call is ``lowercase: true`` on ``bleu``
alone. st.sh also runs ``remove_punctuation.pl`` before that call. The
``remove_punctuation`` normalizer with ``keep: "'"`` agrees with it on ASCII
text but also removes non-ASCII symbols (``°``, ``€``, ``©``) that the Perl
script keeps, so that pass is not reproduced exactly.

Scores are on sacrebleu's 0-100 scale, not the 0-1 scale of the error rates
in :mod:`splet.utterance_metrics`, so that they read the same as the numbers
in RESULTS.md and in papers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from splet.normalizers import build_normalizer

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    BLEU = CHRF = TER = None


def _setup(name: str, metric_class, normalize, kwargs) -> Dict[str, Any]:
    """Build the sacrebleu metric object and the normalizer shared by all three."""
    if metric_class is None:
        raise ImportError(f"{name} requires sacrebleu>=2.0: pip install sacrebleu")
    return {
        "name": name,
        "metric": metric_class(**kwargs),
        "normalizer": build_normalizer(normalize),
    }


def bleu_setup(normalize: Optional[list] = None, **kwargs) -> Dict[str, Any]:
    """Prepare corpus BLEU.

    Args:
        normalize: Normalization pipeline config, applied to both sides
            before sacrebleu sees them.
        **kwargs: Passed to ``sacrebleu.metrics.BLEU``.

    Returns:
        The scorer state passed back into :func:`sacrebleu_scoring`.
    """
    return _setup("bleu", BLEU, normalize, kwargs)


def chrf_setup(normalize: Optional[list] = None, **kwargs) -> Dict[str, Any]:
    """Prepare corpus chrF. ``word_order: 2`` gives chrF++.

    Args:
        normalize: Normalization pipeline config, applied to both sides.
        **kwargs: Passed to ``sacrebleu.metrics.CHRF``.

    Returns:
        The scorer state passed back into :func:`sacrebleu_scoring`.
    """
    return _setup("chrf", CHRF, normalize, kwargs)


def ter_setup(normalize: Optional[list] = None, **kwargs) -> Dict[str, Any]:
    """Prepare corpus TER (translation edit rate; lower is better).

    Args:
        normalize: Normalization pipeline config, applied to both sides.
        **kwargs: Passed to ``sacrebleu.metrics.TER``.

    Returns:
        The scorer state passed back into :func:`sacrebleu_scoring`.
    """
    return _setup("ter", TER, normalize, kwargs)


def sacrebleu_scoring(
    scorer: Dict[str, Any],
    pred_texts: List[str],
    gt_texts: Optional[List[str]],
) -> Dict[str, Any]:
    """Score a whole corpus with one sacrebleu metric.

    Args:
        scorer: State from :func:`bleu_setup`, :func:`chrf_setup` or
            :func:`ter_setup`.
        pred_texts: Hypotheses, one per utterance.
        gt_texts: References, in the same order as ``pred_texts``.

    Returns:
        The score and its sacrebleu signature. BLEU also reports the brevity
        penalty and the two lengths it was computed from, which is what
        st.sh's result file shows after the score.

    Raises:
        ValueError: If there are no references, or no utterances to score.
    """
    name = scorer["name"]
    if gt_texts is None:
        raise ValueError(f"{name} needs references (--gt)")
    if not pred_texts:
        raise ValueError(f"{name} has nothing to score")

    normalizer = scorer["normalizer"]
    metric = scorer["metric"]
    score = metric.corpus_score(
        [normalizer(text) for text in pred_texts],
        [[normalizer(text) for text in gt_texts]],
    )

    result: Dict[str, Any] = {
        name: score.score,
        f"{name}_signature": str(metric.get_signature()),
    }
    if name == "bleu":
        result.update(
            {
                "bleu_bp": score.bp,
                "bleu_ratio": score.ratio,
                "bleu_hyp_len": score.sys_len,
                "bleu_ref_len": score.ref_len,
            }
        )
    return result
