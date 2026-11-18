#!/usr/bin/env python3
"""Train a neural HMM forced aligner (the align task)."""

from espnet2.tasks.align import AlignTask


def get_parser():
    parser = AlignTask.get_parser()
    return parser


def main(cmd=None):
    r"""Align training.

    Example:

        % python align_train.py --print_config --optim adadelta \
                > conf/train_align.yaml
        % python align_train.py --config conf/train_align.yaml
    """
    AlignTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
