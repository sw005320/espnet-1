#!/usr/bin/env python3
"""Forced alignment with a trained neural HMM aligner (the align task).

Writes, per utterance, CTM lines ``utt 1 start duration name`` to

    <output_dir>/units.ctm   every HMM unit the best path passes through
    <output_dir>/words.ctm   words (runs of characters between word
                             boundaries and pauses)
    <output_dir>/pauses.ctm  pause units
    <output_dir>/words_fwdctc.ctm, words_bwdctc.ctm
                             words on the forward / time-reversed CTC head's
                             best path (when the model has that head)
    <output_dir>/words_refined.ctm
                             HMM words moved towards the CTC estimates

Frame t covers [t * frame_shift + frame_offset, (t + 1) * frame_shift +
frame_offset) seconds.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.legacy.utils.cli_utils import get_commandline_args
from espnet2.tasks.align import AlignTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none

Span = Tuple[str, int, int]


def words_and_pauses(
    spans: Sequence[Span],
    word_lengths: Sequence[int],
    boundary: str,
    pause_units: Sequence[str],
) -> Tuple[List[Span], List[Span]]:
    """Group unit spans into the transcript's words, and list the pauses.

    The word-boundary unit is skippable, so words are taken from the
    transcript: the units on the path other than the boundary and pause units
    are consumed in order, ``word_lengths[i]`` of them for word i.
    """
    units = [x for x in spans if x[0] != boundary and x[0] not in pause_units]
    pauses = [x for x in spans if x[0] in pause_units]
    words: List[Span] = []
    i = 0
    for n in word_lengths:
        group = units[i : i + n]
        i += n
        if not group:
            break
        words.append(("".join(x[0] for x in group), group[0][1], group[-1][2]))
    return words, pauses


@typechecked
def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: str,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    align_train_config: Optional[str],
    align_model_file: Optional[str],
    allow_variable_data_keys: bool,
    prior_scale: float,
    frame_shift: float,
    frame_offset: float,
):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    device = "cuda" if ngpu >= 1 else "cpu"
    torch.manual_seed(seed)

    model, train_args = AlignTask.build_model_from_file(
        align_train_config, align_model_file, device
    )
    model.to(dtype=getattr(torch, dtype)).eval()

    loader = AlignTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=AlignTask.build_preprocess_fn(train_args, False),
        collate_fn=AlignTask.build_collate_fn(train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=False,
    )

    boundary = (
        model.token_list[model.boundary_id] if model.boundary_id is not None else ""
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def sec(frame: int) -> float:
        return frame * frame_shift + frame_offset

    names = [
        "units",
        "words",
        "pauses",
        "words_fwdctc",
        "words_bwdctc",
        "words_refined",
    ]
    files = {n: (out / f"{n}.ctm").open("w") for n in names}

    def write(name, key, items):
        for item_name, start, end in items:
            files[name].write(
                f"{key} 1 {sec(start):.3f} "
                f"{(end - start) * frame_shift:.3f} {item_name}\n"
            )

    try:
        for keys, batch in loader:
            batch = to_device(batch, device)
            aligned = model.align(
                batch["speech"],
                batch["speech_lengths"],
                batch["text"],
                batch["text_lengths"],
                prior_scale=prior_scale,
                with_ctc=True,
            )
            texts = batch["text"].tolist()
            lens = batch["text_lengths"].tolist()
            for key, (spans, fwd, bwd), ids, n in zip(keys, aligned, texts, lens):
                if not spans:
                    logging.warning(f"{key}: no alignment (audio too short)")
                    continue
                tokens = [model.token_list[i] for i in ids[:n]]
                positions = model.word_positions(ids[:n])
                words, pauses = words_and_pauses(
                    spans, [len(p) for p in positions], boundary, model.pause_units
                )
                write("units", key, spans)
                write("words", key, words)
                write("pauses", key, pauses)
                wf = ctc_words(tokens, fwd, positions) if fwd else None
                wb = ctc_words(tokens, bwd, positions) if bwd else None
                if wf:
                    write("words_fwdctc", key, wf)
                if wb:
                    write("words_bwdctc", key, wb)
                write("words_refined", key, refine(words, wf, wb))
    finally:
        for f in files.values():
            f.close()


def ctc_words(
    tokens: Sequence[str],
    spans: Sequence[Tuple[int, int]],
    word_positions: Sequence[Sequence[int]],
) -> List[Span]:
    """Words from per-token CTC spans.

    A word spans from the first frame of its first token to the end frame of
    its last token; ``word_positions`` lists each word's token positions.
    """
    return [
        ("".join(tokens[i] for i in pos), spans[pos[0]][0], spans[pos[-1]][1])
        for pos in word_positions
    ]


def refine(
    words: Sequence[Span],
    fwd: Optional[Sequence[Span]],
    bwd: Optional[Sequence[Span]],
) -> List[Span]:
    """HMM word spans moved towards the CTC estimates.

    A forward CTC head emits a token in the late part of its frames and a
    time-reversed head in the early part. The start is therefore averaged
    with the word start on the time-reversed path, and the end with the word
    end on the forward path. A word is left as it is when the word counts
    differ.
    """
    out = []
    for i, (name, s, e) in enumerate(words):
        ns, ne = s, e
        if bwd is not None and len(bwd) == len(words):
            ns = round((s + bwd[i][1]) / 2)
        if fwd is not None and len(fwd) == len(words):
            ne = round((e + fwd[i][2]) / 2)
        out.append((name, ns, max(ne, ns + 1)))
    return out


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Forced alignment with a neural HMM aligner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", default="float32", choices=["float16", "float32", "float64"]
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--align_train_config", type=str_or_none)
    group.add_argument("--align_model_file", type=str_or_none)

    group = parser.add_argument_group("Alignment related")
    group.add_argument(
        "--prior_scale",
        type=float,
        default=0.0,
        help="Weight of the log state prior subtracted from the emission scores",
    )
    group.add_argument(
        "--frame_shift",
        type=float,
        default=0.01,
        help="Seconds per output frame",
    )
    group.add_argument(
        "--frame_offset",
        type=float,
        default=0.0,
        help="Seconds added to every frame time (e.g. a subsampling delay)",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
