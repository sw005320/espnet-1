"""Neural HMM forced aligner trained with a full-sum loss.

The network predicts, for every frame, a distribution over HMM states: each
unit (a token of ``token_list`` or an extra pause unit) owns ``max_states``
output classes, one per state position. Training maximises the likelihood of
the transcript summed over all state paths of its left-to-right HMM (no blank,
no alignment needed: a flat start). Forward and time-reversed CTC heads on the
same encoder can be trained alongside; they are not part of the alignment.
"""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from typeguard import typechecked

from espnet2.align.ctc_align import ctc_viterbi
from espnet2.align.hmm import (
    PaddedGraphs,
    Segment,
    StateGraph,
    build_graph,
    hmm_forward,
    hmm_viterbi,
    pad_graphs,
    state_emissions,
)
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetAlignModel(AbsESPnetModel):
    """Character HMM aligner with optional bidirectional CTC heads.

    Args:
        vocab_size: size of ``token_list``.
        token_list: the tokens; index 0 is the CTC blank and is never an HMM unit.
        frontend, specaug, normalize, encoder: as in the ASR task.
        num_states: HMM states per token unit.
        pause_units: names of the extra pause units (appended after the tokens).
        pause_num_states: HMM states per pause unit. One state (no minimum
            duration beyond a frame) as in RWTH full-sum HMMs.
        pause_prob: probability of entering the optional pause slot between
            words and at the edges (Kaldi's silence probability, 0.5).
        pause_loop_prob: self-loop probability of pause states (defaults to
            ``loop_prob``).
        token_type: how the transcript was tokenized, which sets where words
            begin: "char" (words are separated by ``word_boundary``), "word"
            (every token is a word) or "bpe" (a piece starting with the
            SentencePiece word marker "\u2581" begins a word). Other types
            carry no word boundaries and are rejected.
        word_boundary: token that separates words in the text (char token_type
            writes a space as ``<space>``).
        boundary_is_unit: the word-boundary token is itself an HMM unit,
            skippable, an alternative to the pause units between words.
        edge_pause: pause units may also start and end the utterance.
        loop_prob: self-loop probability of every state.
        upsample: how encoder frames are brought back to feature frames when
            the encoder subsamples: "none", "repeat" or "deconv".
        upsample_factor: the encoder's time subsampling factor.
        hmm_weight, fwd_ctc_weight, bwd_ctc_weight: loss weights.
        prior_scale: weight of the log state prior subtracted from the
            emission scores in training (0 disables it).
        prior_momentum: momentum of the running state prior.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Sequence[str],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        num_states: int = 2,
        pause_units: Sequence[str] = ("<sil>",),
        pause_num_states: int = 1,
        pause_prob: float = 0.5,
        pause_loop_prob: Optional[float] = None,
        token_type: str = "char",
        word_boundary: str = "<space>",
        boundary_is_unit: bool = True,
        edge_pause: bool = True,
        loop_prob: float = 0.5,
        upsample: str = "none",
        upsample_factor: int = 1,
        hmm_weight: float = 1.0,
        fwd_ctc_weight: float = 0.0,
        bwd_ctc_weight: float = 0.0,
        prior_scale: float = 0.0,
        prior_momentum: float = 0.99,
        ignore_id: int = -1,
    ):
        super().__init__()
        if upsample not in ("none", "repeat", "deconv"):
            raise ValueError(f"unknown upsample: {upsample}")
        self.vocab_size = vocab_size
        self.token_list = list(token_list)
        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.num_states = num_states
        self.pause_units = list(pause_units)
        self.pause_num_states = pause_num_states
        self.pause_prob = pause_prob
        self.pause_loop_prob = loop_prob if pause_loop_prob is None else pause_loop_prob
        self.max_states = max(num_states, self.pause_num_states)
        if token_type not in ("char", "word", "bpe"):
            raise ValueError(
                f"token_type {token_type!r} is not supported: the aligner needs "
                "word boundaries, given by char (<space>), word or bpe tokens"
            )
        self.token_type = token_type
        self.boundary_id = (
            self.token_list.index(word_boundary)
            if word_boundary in self.token_list
            else None
        )
        self.boundary_is_unit = boundary_is_unit and self.boundary_id is not None
        self.edge_pause = edge_pause
        self.loop_prob = loop_prob
        self.hmm_weight = hmm_weight
        self.fwd_ctc_weight = fwd_ctc_weight
        self.bwd_ctc_weight = bwd_ctc_weight
        self.prior_scale = prior_scale
        self.prior_momentum = prior_momentum
        self.ignore_id = ignore_id

        # unit ids: tokens keep their token id; pause units follow the tokens
        self.pause_ids = [vocab_size + i for i in range(len(self.pause_units))]
        self.num_units = vocab_size + len(self.pause_units)
        self.num_classes = self.num_units * self.max_states

        eout = encoder.output_size()
        self.upsample = upsample
        self.upsample_factor = upsample_factor
        if upsample == "deconv" and upsample_factor > 1:
            self.upsampler = torch.nn.ConvTranspose1d(
                eout, eout, kernel_size=upsample_factor, stride=upsample_factor
            )
        else:
            self.upsampler = None
        self.state_out = torch.nn.Linear(eout, self.num_classes)
        self.register_buffer("log_prior", torch.full((self.num_classes,), 0.0))

        self.fwd_ctc = CTC(vocab_size, eout) if fwd_ctc_weight > 0 else None
        self.bwd_ctc = CTC(vocab_size, eout) if bwd_ctc_weight > 0 else None

    # ------------------------------------------------------------------ graph
    def unit_num_states(self, unit: int) -> int:
        """Number of HMM states of a unit."""
        return self.pause_num_states if unit >= self.vocab_size else self.num_states

    def unit_loop_prob(self, unit: int) -> float:
        """Self-loop probability of a unit's states."""
        return self.pause_loop_prob if unit >= self.vocab_size else self.loop_prob

    def unit_name(self, unit: int) -> str:
        """Name of a unit (a token or a pause unit)."""
        if unit >= self.vocab_size:
            return self.pause_units[unit - self.vocab_size]
        return self.token_list[unit]

    def word_positions(self, token_ids: Sequence[int]) -> List[List[int]]:
        """The transcript's words, as lists of positions in ``token_ids``.

        Word-boundary tokens are not part of any word.
        """
        words: List[List[int]] = []
        for i, t in enumerate(token_ids):
            t = int(t)
            if t == self.boundary_id:
                if words and words[-1]:
                    words.append([])
                continue
            if self.token_type == "word":
                starts = True
            elif self.token_type == "bpe":
                starts = self.token_list[t].startswith("\u2581")
            else:
                starts = False
            if not words or (starts and words[-1]):
                words.append([])
            words[-1].append(i)
        return [w for w in words if w]

    def words(self, token_ids: Sequence[int]) -> List[List[int]]:
        """The transcript's words as lists of unit (token) ids."""
        return [[int(token_ids[i]) for i in w] for w in self.word_positions(token_ids)]

    def segments(self, token_ids: Sequence[int]) -> List[Segment]:
        """Segments of a transcript: one per unit, a pause slot between words."""
        between = tuple(self.pause_ids)
        if self.boundary_is_unit:
            between = (self.boundary_id,) + between
        words = self.words(token_ids)
        segs: List[Segment] = []
        if self.edge_pause and self.pause_ids:
            segs.append(Segment(tuple(self.pause_ids), True, self.pause_prob))
        for i, w in enumerate(words):
            if i > 0 and between:
                segs.append(Segment(between, True, self.pause_prob))
            segs.extend(Segment((u,)) for u in w)
        if self.edge_pause and self.pause_ids:
            segs.append(Segment(tuple(self.pause_ids), True, self.pause_prob))
        return segs

    def build_graphs(
        self, text: torch.Tensor, text_lengths: torch.Tensor
    ) -> List[StateGraph]:
        """State graphs of a padded batch of transcripts."""
        graphs = []
        for ids, n in zip(text.tolist(), text_lengths.tolist()):
            graphs.append(
                build_graph(
                    self.segments(ids[:n]),
                    self.unit_num_states,
                    self.max_states,
                    loop_prob=self.unit_loop_prob,
                )
            )
        return graphs

    # ---------------------------------------------------------------- network
    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend, normalisation and encoder.

        When ``upsample`` is set, the encoder frames are brought back to the
        feature frame rate.
        """
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)
        enc, enc_lengths, _ = self.encoder(feats, feats_lengths)
        if self.upsample != "none" and self.upsample_factor > 1:
            r = self.upsample_factor
            if self.upsampler is not None:
                enc = self.upsampler(enc.transpose(1, 2)).transpose(1, 2)
            else:
                enc = enc.repeat_interleave(r, dim=1)
            enc_lengths = torch.minimum(enc_lengths * r, feats_lengths)
            enc = enc[:, : int(enc_lengths.max())]
        return enc, enc_lengths

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech = speech[:, : speech_lengths.max()]
        if self.frontend is not None:
            return self.frontend(speech, speech_lengths)
        return speech, speech_lengths

    def state_log_probs(self, enc: torch.Tensor) -> torch.Tensor:
        """(B, T, C) log-probabilities over the HMM state classes."""
        return self.state_out(enc).log_softmax(dim=-1)

    def emissions(
        self, log_probs: torch.Tensor, graphs: PaddedGraphs, prior_scale: float
    ) -> torch.Tensor:
        """Per-state emission scores, with the scaled log prior removed."""
        if prior_scale > 0:
            log_probs = log_probs - prior_scale * self.log_prior
        return state_emissions(log_probs, graphs)

    @torch.no_grad()
    def _update_prior(self, log_probs: torch.Tensor, lengths: torch.Tensor):
        mask = (
            torch.arange(log_probs.size(1), device=lengths.device)[None, :]
            < lengths[:, None]
        )
        post = log_probs.exp()[mask].mean(dim=0)
        prior = self.log_prior.exp()
        if bool((self.log_prior == 0).all()):
            prior = post  # first batch
        else:
            m = self.prior_momentum
            prior = m * prior + (1 - m) * post
        self.log_prior.copy_(prior.clamp(min=1e-12).log())

    # ------------------------------------------------------------------- loss
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = speech.size(0)
        text = text[:, : int(text_lengths.max())]
        enc, enc_lengths = self.encode(speech, speech_lengths)
        stats: Dict[str, Optional[torch.Tensor]] = {}
        loss = enc.new_zeros(())

        log_probs = self.state_log_probs(enc)
        if self.training and self.prior_scale > 0:
            self._update_prior(log_probs.detach(), enc_lengths)
        graphs = self.build_graphs(text, text_lengths)
        pg = pad_graphs(graphs, device=enc.device)
        emit = self.emissions(log_probs, pg, self.prior_scale if self.training else 0.0)
        ll = hmm_forward(emit.float(), enc_lengths, pg)
        ok = torch.isfinite(ll)
        if not bool(ok.all()):
            logging.warning(f"{int((~ok).sum())} utterance(s) too short for their HMM")
        frames = enc_lengths.clamp(min=1).float()
        loss_hmm = -(ll.masked_fill(~ok, 0.0)).sum() / batch_size
        stats["loss_hmm"] = loss_hmm.detach()
        stats["loss_hmm_per_frame"] = (
            -(ll.masked_fill(~ok, 0.0)).sum() / frames[ok].sum().clamp(min=1)
        ).detach()
        loss = loss + self.hmm_weight * loss_hmm

        if self.fwd_ctc is not None:
            loss_f = self.fwd_ctc(enc, enc_lengths, text, text_lengths)
            stats["loss_fwd_ctc"] = loss_f.detach()
            loss = loss + self.fwd_ctc_weight * loss_f
        if self.bwd_ctc is not None:
            enc_r = _reverse_padded(enc, enc_lengths)
            text_r = _reverse_padded(text, text_lengths, pad=self.ignore_id)
            loss_b = self.bwd_ctc(enc_r, enc_lengths, text_r, text_lengths)
            stats["loss_bwd_ctc"] = loss_b.detach()
            loss = loss + self.bwd_ctc_weight * loss_b

        stats["loss"] = loss.detach()
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    # -------------------------------------------------------------- alignment
    @torch.no_grad()
    def align(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        prior_scale: float = 0.0,
        with_ctc: bool = False,
    ):
        """Viterbi alignment of a batch.

        Returns, per utterance, the (unit name, first frame, end frame) of
        every unit the best HMM path passes through, in time order. With
        ``with_ctc`` it returns ``(spans, fwd, bwd)`` per utterance, where
        ``fwd`` and ``bwd`` are the (first frame, end frame) of every
        transcript token on the best forward and time-reversed CTC paths
        (``None`` without that head).
        """
        enc, enc_lengths = self.encode(speech, speech_lengths)
        log_probs = self.state_log_probs(enc)
        graphs = self.build_graphs(text, text_lengths)
        pg = pad_graphs(graphs, device=enc.device)
        emit = self.emissions(log_probs, pg, prior_scale)
        paths = hmm_viterbi(emit.float(), enc_lengths, pg)
        out = []
        for g, path in zip(graphs, paths):
            spans: List[Tuple[str, int, int]] = []
            prev = None
            for t, s in enumerate(path):
                # every unit occurrence has its own states, so a span starts
                # exactly when the path enters the first state of a unit
                if prev is None or (s != prev and g.state_pos[s] == 0):
                    spans.append((self.unit_name(g.state_unit[s]), t, t + 1))
                else:
                    name, start, _ = spans[-1]
                    spans[-1] = (name, start, t + 1)
                prev = s
            out.append(spans)
        if not with_ctc:
            return out
        fwd = self._ctc_spans(self.fwd_ctc, enc, enc_lengths, text, text_lengths, False)
        bwd = self._ctc_spans(self.bwd_ctc, enc, enc_lengths, text, text_lengths, True)
        return list(zip(out, fwd, bwd))

    def _ctc_spans(self, ctc, enc, enc_lengths, text, text_lengths, reverse):
        if ctc is None:
            return [None] * enc.size(0)
        out = []
        for b in range(enc.size(0)):
            T, L = int(enc_lengths[b]), int(text_lengths[b])
            x = enc[b : b + 1, :T]
            tokens = text[b, :L].tolist()
            if reverse:
                x = x.flip(1)
                tokens = tokens[::-1]
            lp = ctc.log_softmax(x)[0]
            spans = ctc_viterbi(lp, tokens, blank=0)
            if reverse and spans:
                spans = [(T - e, T - s) for s, e in spans[::-1]]
            out.append(spans)
        return out


def _reverse_padded(x: torch.Tensor, lengths: torch.Tensor, pad=0.0) -> torch.Tensor:
    """Reverse each sequence along time within its length, keeping padding."""
    out = x.clone()
    for b, n in enumerate(lengths.tolist()):
        out[b, :n] = x[b, :n].flip(0)
        if n < x.size(1):
            out[b, n:] = pad
    return out
