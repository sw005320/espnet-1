# Forced alignment (align1)

This is the template of the `align1` recipe for ESPnet2. It trains a neural
HMM forced aligner and aligns transcribed speech at the word, character and
pause level.

## Model

- An ESPnet encoder (e.g. E-Branchformer) predicts, for every output frame,
  a distribution over HMM states. Each unit (a character token, the
  word-boundary token `<space>`, or the extra pause unit `<sil>`) owns
  `num_states` left-to-right states with self-loops. There is no blank.
- The transcript becomes a state graph:
  - an optional `<sil>` at both edges;
  - the characters of each word;
  - between two words, an optional slot holding `<space>` or `<sil>`
    (Kaldi-style optional silence, entered with probability `pause_prob`).
- Training maximises the likelihood of the transcript summed over all state
  paths (full-sum forward-backward in log space,
  `espnet2/align/hmm.py`), from a flat start: no external alignment.
- Optional forward and time-reversed CTC heads are trained on the same
  encoder. Their best paths give a second, independent word-boundary
  estimate.
- Alignment is the Viterbi state path. The output frame rate is the encoder's
  (use `input_layer: conv2d1` for 10 ms frames), or the feature rate when the
  model upsamples a subsampled encoder (`upsample: repeat|deconv`).

## Recipe flow

1. Data preparation (`local/data.sh`): Kaldi-style data directories with
   `wav.scp` and `text`.
2. Format `wav.scp` into `dump/raw`.
3. Remove too short/long utterances from the training and validation sets.
4. Character token list (`<blank>` at index 0, then `<unk>` and the characters,
   with spaces written as `<space>`).
5. Collect statistics for global mean-variance normalisation.
6. Training (`espnet2/bin/align_train.py`).
7. Alignment (`espnet2/bin/align_inference.py`): per data set, CTM files of
   - `units` (every unit on the path),
   - `words`,
   - `pauses`,
   - `words_fwdctc` and `words_bwdctc` (the CTC heads' word spans),
   - `words_refined` (HMM words moved towards the CTC estimates).
8. Scoring with `local/score.sh <align_dir> <data_dir>` when the recipe has
   one (e.g. word-boundary error against reference alignments).

## Recipes

- `egs2/mini_an4/align1`: a tiny run for CI.
- `egs2/librispeech_100/align1`: LibriSpeech 100 h, scored against the
  LibriSpeech MFA word alignments (set `LIBRISPEECH_MFA` in `db.sh`).
