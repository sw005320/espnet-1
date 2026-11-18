"""Forced-alignment task: a neural HMM aligner trained with a full-sum loss."""

import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import typechecked

from espnet2.align.espnet_model import ESPnetAlignModel
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.asr import (
    encoder_choices,
    frontend_choices,
    normalize_choices,
    specaug_choices,
)
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.types import int_or_none, str2bool, str_or_none

model_choices = ClassChoices(
    "model",
    classes=dict(espnet=ESPnetAlignModel),
    type_check=ESPnetAlignModel,
    default="espnet",
)


class AlignTask(AbsTask):
    num_optimizers: int = 1

    class_choices_list = [
        frontend_choices,
        specaug_choices,
        normalize_choices,
        model_choices,
        encoder_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Task related")
        required = parser.get_default("required")
        required += ["token_list"]
        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token (index 0 is the CTC blank)",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            choices=[
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
            help="The initialization method",
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        group = parser.add_argument_group("Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="char",
            choices=["char", "bpe", "word", "phn"],
            help="The text is tokenized at this level; the HMM units are these tokens",
        )
        group.add_argument("--bpemodel", type=str_or_none, default=None)
        group.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument("--cleaner", type=str_or_none, default=None)
        group.add_argument("--g2p", type=str_or_none, choices=g2p_choices, default=None)
        cls.add_augmentation_arguments(parser)

        for class_choices in cls.class_choices_list:
            # --<name> and --<name>_conf, e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        if not args.use_preprocessor:
            return None
        return CommonPreprocessor(
            train=train,
            token_type=args.token_type,
            token_list=args.token_list,
            bpemodel=args.bpemodel,
            non_linguistic_symbols=args.non_linguistic_symbols,
            text_cleaner=args.cleaner,
            g2p_type=args.g2p,
            rir_scp=getattr(args, "rir_scp", None),
            rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
            noise_scp=getattr(args, "noise_scp", None),
            noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
            noise_db_range=getattr(args, "noise_db_range", "13_15"),
            short_noise_thres=getattr(args, "short_noise_thres", 0.5),
        )

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        # alignment needs the transcript at inference too
        return ("speech", "text")

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        return ()

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetAlignModel:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        if args.input_size is None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        specaug = None
        if args.specaug is not None:
            specaug = specaug_choices.get_class(args.specaug)(**args.specaug_conf)
        normalize = None
        if args.normalize is not None:
            normalize = normalize_choices.get_class(args.normalize)(
                **args.normalize_conf
            )
        encoder = encoder_choices.get_class(args.encoder)(
            input_size=input_size, **args.encoder_conf
        )

        model_class = model_choices.get_class(getattr(args, "model", "espnet"))
        model = model_class(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
            **args.model_conf,
        )
        if args.init is not None:
            initialize(model, args.init)
        return model
