import pytest
import torch

from espnet2.tasks.align import AlignTask

TOKENS = ["<blank>", "<unk>", "<space>", "a", "b", "c"]


def _plain(tokens, **conf):
    from espnet2.align.espnet_model import ESPnetAlignModel
    from espnet2.asr.encoder.transformer_encoder import TransformerEncoder

    enc = TransformerEncoder(
        4,
        output_size=8,
        attention_heads=2,
        num_blocks=1,
        linear_units=8,
        input_layer="linear",
    )
    return ESPnetAlignModel(len(tokens), tokens, None, None, None, enc, **conf)


def _build(**model_conf):
    parser = AlignTask.get_parser()
    args = parser.parse_args(["--token_list", "x", "--output_dir", "out"])
    args.token_list = TOKENS
    args.frontend = "default"
    args.frontend_conf = {"n_fft": 512, "hop_length": 160, "win_length": 400}
    args.specaug = None
    args.normalize = None
    args.encoder = "e_branchformer"
    args.encoder_conf = dict(
        output_size=16,
        attention_heads=2,
        num_blocks=1,
        linear_units=32,
        cgmlp_linear_units=32,
        cgmlp_conv_kernel=3,
        merge_conv_kernel=3,
        input_layer=model_conf.pop("input_layer", "conv2d1"),
    )
    args.model_conf = model_conf
    return AlignTask.build_model(args)


@pytest.mark.parametrize(
    "conf",
    [
        dict(num_states=1),
        dict(num_states=2, fwd_ctc_weight=0.3, bwd_ctc_weight=0.3),
        dict(num_states=3, prior_scale=0.3),
        dict(num_states=2, input_layer="conv2d", upsample="deconv", upsample_factor=4),
        dict(num_states=2, input_layer="conv2d", upsample="repeat", upsample_factor=4),
    ],
)
def test_forward_backward_and_align(conf):
    torch.manual_seed(0)
    model = _build(**conf)
    speech = torch.randn(2, 8000)
    speech_lengths = torch.tensor([8000, 6000])
    text = torch.tensor([[3, 4, 2, 5], [5, 3, -1, -1]])
    text_lengths = torch.tensor([4, 2])
    model.train()
    loss, stats, weight = model(speech, speech_lengths, text, text_lengths)
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    model.eval()
    out = model.align(speech, speech_lengths, text, text_lengths, with_ctc=True)
    for (spans, fwd, bwd), n_tok in zip(out, (4, 2)):
        units = [u for u, _, _ in spans if u not in ("<sil>", "<space>")]
        assert len(units) == n_tok - (1 if n_tok == 4 else 0)
        # spans tile the frames without gaps
        assert spans[0][1] == 0
        for (_, _, e), (_, s, _) in zip(spans, spans[1:]):
            assert e == s
        if fwd is not None:
            assert len(fwd) == n_tok and len(bwd) == n_tok


def test_segments_put_optional_pause_between_words():
    model = _build(num_states=2)
    segs = model.segments([3, 4, 2, 5])
    pause = model.pause_ids[0]
    assert [s.units for s in segs] == [
        (pause,),
        (3,),
        (4,),
        (2, pause),
        (5,),
        (pause,),
    ]
    assert [s.skippable for s in segs] == [True, False, False, True, False, True]


def test_word_tokens_each_make_a_word():
    model = _plain(["<blank>", "<unk>", "HELLO", "WORLD"], token_type="word")
    assert model.words([2, 3]) == [[2], [3]]
    pause = model.pause_ids[0]
    # a pause slot between the two words (no <space> token in word mode)
    assert [s.units for s in model.segments([2, 3])] == [
        (pause,),
        (2,),
        (pause,),
        (3,),
        (pause,),
    ]


def test_bpe_word_marker_starts_a_word():
    tokens = ["<blank>", "<unk>", "\u2581he", "llo", "\u2581wor", "ld"]
    model = _plain(tokens, token_type="bpe")
    assert model.words([2, 3, 4, 5]) == [[2, 3], [4, 5]]
    assert model.word_positions([2, 3, 4, 5]) == [[0, 1], [2, 3]]


def test_char_words_split_at_space():
    model = _plain(TOKENS, token_type="char")
    assert model.words([3, 4, 2, 5]) == [[3, 4], [5]]
    assert model.word_positions([3, 4, 2, 5]) == [[0, 1], [3]]


def test_token_type_without_word_boundaries_is_rejected():
    with pytest.raises(ValueError, match="phn"):
        _plain(TOKENS, token_type="phn")
