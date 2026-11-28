# frontia_vad_v1: voice activity detection as single-output diarization

Voice activity detection trained with `diar.sh --num_spk 1`: the model has one
sigmoid output (speech / non-speech) per 8 ms frame, and the label of a frame is
speech when anyone in the recording speaks. Two models:

- **Causal** (`conf/train.yaml`): every block is causal (no STFT centring, global
  mean/variance normalisation, a causal TCN encoder, labels aggregated over the
  same frames). The look-ahead is one STFT window (32 ms); about 1 s of past
  context. 184k parameters.
- **Dual-mode** (`conf/tuning/train_dualmode.yaml`): the same encoder with a
  look-ahead chosen at run time, one set of weights. Online: look-ahead 0
  (`conf/decode.yaml`); offline: 124 frames = 0.99 s (`conf/tuning/decode_offline.yaml`).

Training reads 10 s chunks straight from the Omni-IO archives that stage 2 dumps:
the chunk iterator with `chunk_random_access` reads only each chunk's part of the
archive entry. The feature statistics of stage 4 are collected over whole
recordings, one at a time with two loader workers (`--stats_args "--batch_size 1 --num_workers 2"`: each worker holds whole recordings); inference runs on
whole recordings.
Scoring is md-eval with no collar. `local/show_vad_result.py` picks one threshold
and median filter on AMI Mix-Headset dev and reports every set at that setting,
next to each set's own best (`RESULTS_vad.md` in the experiment directory).

## Data

Set the corpus paths in `db.sh`. `local/data.sh --corpora <list>` selects the
corpora; run.sh passes all of them. Labels are the union of all speakers'
segments of a recording, written as one pseudo speaker.

Training (and validation: `data/train`, `data/dev`):

| corpus | `db.sh` | licence | microphones | splits |
|---|---|---|---|---|
| AMI | `AMI` | CC BY 4.0 | every wav: Headset-*, Lapel-*, Array1-0*, Array2-0*, Mix-Headset | full-corpus-ASR partition (`../../ami/asr1/local/split_*.orig`) |
| CHiME-6 (OpenSLR SLR150) | `CHIME6` | CC BY-SA 4.0 | every channel of the six arrays and of the binaural in-ear microphones | train / dev / eval |
| NOTSOFAR-1 (240825.1) | `NOTSOFAR1` | CC BY 4.0 | close-talk, every channel of the 7-channel arrays, single-channel devices | train / dev / eval |
| ICSI | `ICSI` | CC BY 4.0 | every chan*.sph and Mix-Headset | lhotse partition (`local/icsi_split_*.txt`) |

CHiME-6 was used as released on OpenSLR (SLR150), under CC BY-SA 4.0
(`LICENSE.txt` of the corpus). It has since been re-released under CC BY 4.0 on
Hugging Face (`chimechallenge/chime6`), but the v1 models were trained on the
CC BY-SA release.

AMI and ICSI are prepared from every audio file of a meeting, so a copy that is
still being downloaded would give fewer channels without an error.
`local/prep_ami.py` and `local/prep_icsi.py` therefore stop when a file listed in
`local/ami_audio_files.txt` or `local/icsi_audio_files.txt` (a complete copy) is
missing; `--allow_missing` only warns.

Noise for on-the-fly augmentation (`data/noise.scp`, SNR -5 to 20 dB):

| corpus | `db.sh` | licence | part used |
|---|---|---|---|
| MUSAN | `MUSAN` | CC licences (some `music/` tracks CC BY-SA) / public domain (per-file LICENSE) | `noise/`, `music/` (`speech/` is left out: babble added to a non-speech region would be labelled non-speech) |
| CHiME-3 background noise | `CHIME3_BACKGROUND` | CC BY 4.0 | all channels |

Evaluation only (test sets; the dev/eval splits of the training corpora are test
sets too). Multi-channel recordings are scored on their first channel.

| corpus | `db.sh` | licence | sets | reference |
|---|---|---|---|---|
| DiPCo | `DIPCO` | CDLA-Permissive-1.0 | dev, eval | every array channel and close-talk |
| AMI Mix-Headset, BUT setup | `AMI`, `AMI_DIAR_SETUP` | AMI: CC BY 4.0; setup: Apache-2.0 | dev, eval (BUT dev, test) | `only_words` RTTMs, as in Bredin and Laurent, Interspeech 2021 |
| AliMeeting (OpenSLR SLR119) | `ALIMEETING` | CC BY-SA 4.0 | dev (Eval_Ali), eval (Test_Ali) | far-field TextGrids |
| AISHELL-4 (OpenSLR SLR111) | `AISHELL4` | CC BY-SA 4.0 | eval (test) | RTTMs of the release |
| VoxConverse | `VOXCONVERSE` | CC BY 4.0 | dev, eval (test) | RTTMs of `joonson/voxconverse` |
| Earnings-21 | `EARNINGS21` | CC BY-SA 4.0 | eval | RTTMs of `revdotcom/speech-datasets` |
| AVA-Speech | `AVA_SPEECH` | labels CC BY 4.0; audio of YouTube movies | eval (15:00-30:00 of each movie) | every label except NO_SPEECH is speech |
| TEN VAD testset | `TEN_VAD` | Apache-2.0 with the additional conditions of the ten-vad repository | eval | frame labels (`.scv`) |

Archives that unpack into many small files (AliMeeting, AISHELL-4) can stay
packed: `local/data.sh` extracts only the members it needs into `downloads/`.

### Training noise

The training audio was augmented by adding recorded noise to it. That noise
audio is not distributed with the models trained by this recipe; it was used
only as additive background noise during training. All of MUSAN `music/` was
used, and some of those tracks are licensed CC BY-SA.

### ShareAlike material and the model licence

The v1 training data therefore include material under CC BY-SA 4.0: CHiME-6 as
released on OpenSLR, and some MUSAN `music/` tracks. Whether trained weights
inherit the ShareAlike term is unsettled, so the v1 models
(`espnet/frontia_vad_v1_online`, `espnet/frontia_vad_v1_offline`) are released
under CC BY-SA 4.0, and that cannot change without retraining.

The v2 series (`egs2/frontia_vad_v2`) removes the ShareAlike material. It uses
the CC BY 4.0 release of CHiME-6. Its noise and music lists are filtered per
file to CC0, CC BY or public domain, without vocals. The final training lists
are counted per licence before training. Models from v2.2 (clean lists) on are
released under CC BY 4.0, e.g. `espnet/frontia_vad_v2_3`.

## Running

```bash
./run.sh                                        # causal model, stages 1-7
./run.sh --diar_config conf/tuning/train_dualmode.yaml --diar_tag train_dualmode \
    --inference_model valid.acc.ave_2best.pth
python3 local/show_vad_result.py exp/diar_train --tune_set ami_mh_dev
```

Offline scores of the dual-mode model: see the comment at the top of `run.sh`.
Inference of a set runs in `--inference_nj` jobs. With `--gpu_inference true`,
`run.pl` runs one job per GPU; `--inference_max_jobs_run 8` runs 8 at once (one
2.5 h recording needs about 2 GB of GPU memory with this model).

The dual-mode model needs the TCN encoder with a run-time look-ahead and the
batch augmentation in the model (`wav_augment`).

<!-- Generated by local/show_vad_result.py (scripts/utils/show_diar_result.sh gives the per-setting DER) -->
# RESULTS
## Environments
- date: `Thu Nov 26 05:32:54 JST 2026`
- python version: `3.12.8 | Intel Corporation | (main, Mar 31 2025, 23:28:18) [GCC 14.2.0]`
- espnet2 version: `espnet2 202611`
- pytorch version: `pytorch 2.13.0+cu129`
- Git hash: `c8c465032647916c87943498e0925dc8e7c11207`
  - Commit date: `Mon Nov 23 18:43:57 2026 -0500`
- scoring: md-eval with a UEM of each whole recording

These results are from one run of this recipe from scratch at the commit above:
`./run.sh` (stages 1-7), then the dual-mode commands of "Running" on the same
dumps, one GPU for training.

The pretrained models below were trained before the AMI and ICSI downloads were
complete: their data had fewer AMI array and headset channels and fewer ICSI
headset channels (the same meetings; ami_train 1,743 recordings instead of
2,652, ami_eval 176 instead of 320, icsi_eval 15 instead of 36). Their scores
on the AMI and ICSI sets and on `dev` therefore differ from the tables here.

Detection error (%) = (missed + false alarm speech) / reference speech, md-eval,
collar 0, over each whole recording (`local/score.sh`: a UEM from
0 to the end of the recording, so false alarms before the first and after the
last speech count). With one pseudo speaker there is no speaker confusion, so
the diarization error rate is the detection error. Stage 7 scores every set at
thresholds 0.3-0.9 and median filters of 1 and 11 frames; `local/show_vad_result.py`
picks the setting on AMI Mix-Headset dev and reports every set at it ("tuned"),
next to each set's own best setting, which is tuned on that set.

## Causal model: `conf/train.yaml`

- config: `conf/train.yaml`, 10 s windows, batch 64, learning rate 5e-4, seed 1
- inference model: `valid.acc.best.pth` (epoch 18), whole recordings, `conf/decode.yaml`
- pretrained model: https://huggingface.co/espnet/frontia_vad_v1_online
- model licence: CC BY-SA 4.0 (the training data include material under CC BY-SA:
  CHiME-6 and some MUSAN `music/` tracks)
- tuned on ami_mh_dev: threshold 0.7, median filter 11
- DetER over the 21 sets at the tuned setting: 9.34 hour-weighted, 12.58 macro average

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 8.50 | 3.67 | 4.83 | 8.04 | 0.6, 11 |
| alimeeting_dev | 3.9 | 7.53 | 5.06 | 2.46 | 7.46 | 0.6, 11 |
| alimeeting_eval | 9.9 | 7.14 | 4.67 | 2.46 | 7.14 | 0.7, 11 |
| ami_dev | 147.7 | 10.54 | 2.74 | 7.80 | 9.32 | 0.5, 11 |
| ami_eval | 152.4 | 12.13 | 2.42 | 9.71 | 9.77 | 0.5, 11 |
| ami_mh_dev | 7.6 | 10.16 | 6.44 | 3.71 | 10.16 | 0.7, 11 |
| ami_mh_eval | 7.3 | 10.71 | 5.36 | 5.35 | 10.71 | 0.7, 11 |
| ava_speech_eval | 19.9 | 49.81 | 43.39 | 6.42 | 32.96 | 0.9, 11 |
| chime6_dev | 117.9 | 10.12 | 1.74 | 8.37 | 4.62 | 0.3, 11 |
| chime6_eval | 122.7 | 14.85 | 5.24 | 9.60 | 14.83 | 0.6, 11 |
| dev | 411.3 | 8.66 | 2.27 | 6.39 | 7.20 | 0.5, 11 |
| dipco_dev | 98.1 | 10.96 | 4.23 | 6.73 | 8.56 | 0.4, 11 |
| dipco_eval | 92.0 | 10.23 | 5.32 | 4.91 | 9.54 | 0.6, 11 |
| earnings21_eval | 32.8 | 17.57 | 10.00 | 7.57 | 17.57 | 0.7, 11 |
| icsi_dev | 22.0 | 12.26 | 5.60 | 6.66 | 12.26 | 0.7, 11 |
| icsi_eval | 28.6 | 17.10 | 3.78 | 13.31 | 14.31 | 0.5, 11 |
| notsofar_dev | 123.7 | 4.37 | 1.62 | 2.76 | 4.17 | 0.6, 11 |
| notsofar_eval | 398.5 | 4.47 | 2.71 | 1.75 | 4.47 | 0.7, 11 |
| ten_vad_eval | 0.1 | 23.53 | 22.37 | 1.16 | 15.95 | 0.9, 11 |
| voxconverse_dev | 18.9 | 5.05 | 3.86 | 1.19 | 5.05 | 0.7, 11 |
| voxconverse_eval | 39.0 | 8.45 | 7.32 | 1.14 | 8.21 | 0.8, 11 |
# VAD results: diar_dualmode_la0

The false alarm on AVA-Speech (movies with music and effects) is far above every
other set; its own best threshold is the highest tried.

## Dual-mode model: `conf/tuning/train_dualmode.yaml`

- config: `conf/tuning/train_dualmode.yaml`, 10 s windows, batch 64, learning rate 5e-4, seed 1
- inference model: `valid.acc.ave_2best.pth` (average of the 2 best checkpoints,
  `keep_nbest_models: [2, 5]`), whole recordings
- pretrained model (offline use): https://huggingface.co/espnet/frontia_vad_v1_offline
- model licence: CC BY-SA 4.0 (the training data include material under CC BY-SA:
  CHiME-6 and some MUSAN `music/` tracks)
- tuned on ami_mh_dev: threshold 0.7 (online and offline), median filter 11

DetER (%) over the 21 sets at the tuned setting, hour-weighted / macro average:

| inference model | online | offline |
|---|---|---|
| `valid.acc.ave_2best.pth` | 9.10 / 12.31 | 7.20 / 10.41 |

The tables below are of `valid.acc.ave_2best.pth`.

### Online (look-ahead 0, `conf/decode.yaml`)

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 8.48 | 4.06 | 4.42 | 8.19 | 0.6, 11 |
| alimeeting_dev | 3.9 | 7.43 | 5.21 | 2.22 | 7.38 | 0.6, 11 |
| alimeeting_eval | 9.9 | 7.11 | 4.93 | 2.18 | 7.11 | 0.7, 11 |
| ami_dev | 147.7 | 9.93 | 3.22 | 6.72 | 9.20 | 0.6, 11 |
| ami_eval | 152.4 | 11.42 | 2.73 | 8.69 | 9.72 | 0.5, 11 |
| ami_mh_dev | 7.6 | 10.19 | 6.96 | 3.23 | 10.19 | 0.7, 11 |
| ami_mh_eval | 7.3 | 10.63 | 5.72 | 4.91 | 10.63 | 0.7, 11 |
| ava_speech_eval | 19.9 | 50.02 | 43.98 | 6.04 | 32.22 | 0.9, 11 |
| chime6_dev | 117.9 | 10.10 | 1.82 | 8.28 | 4.71 | 0.3, 11 |
| chime6_eval | 122.7 | 14.85 | 5.19 | 9.66 | 14.83 | 0.6, 11 |
| dev | 411.3 | 8.45 | 2.50 | 5.95 | 7.19 | 0.5, 11 |
| dipco_dev | 98.1 | 10.81 | 4.19 | 6.62 | 8.55 | 0.4, 11 |
| dipco_eval | 92.0 | 10.11 | 5.40 | 4.71 | 9.50 | 0.6, 11 |
| earnings21_eval | 32.8 | 14.08 | 10.76 | 3.32 | 13.31 | 0.8, 11 |
| icsi_dev | 22.0 | 12.36 | 5.77 | 6.59 | 12.36 | 0.7, 11 |
| icsi_eval | 28.6 | 16.99 | 3.95 | 13.04 | 14.60 | 0.4, 11 |
| notsofar_dev | 123.7 | 4.41 | 1.70 | 2.71 | 4.23 | 0.6, 11 |
| notsofar_eval | 398.5 | 4.39 | 2.58 | 1.81 | 4.39 | 0.7, 11 |
| ten_vad_eval | 0.1 | 23.29 | 21.75 | 1.54 | 16.66 | 0.9, 11 |
| voxconverse_dev | 18.9 | 5.04 | 3.97 | 1.07 | 5.04 | 0.7, 11 |
| voxconverse_eval | 39.0 | 8.50 | 7.43 | 1.07 | 8.35 | 0.8, 11 |
# VAD results: diar_dualmode_la124

### Offline (look-ahead 124 frames = 0.99 s, `conf/tuning/decode_offline.yaml`)

| set | speech h | DetER (tuned) | FA | Miss | DetER (set's own best) | its threshold, median |
|---|---|---|---|---|---|---|
| aishell4_eval | 11.5 | 6.34 | 3.05 | 3.29 | 5.95 | 0.6, 11 |
| alimeeting_dev | 3.9 | 6.08 | 4.84 | 1.24 | 6.08 | 0.7, 11 |
| alimeeting_eval | 9.9 | 5.51 | 4.09 | 1.42 | 5.51 | 0.7, 11 |
| ami_dev | 147.7 | 7.31 | 2.41 | 4.90 | 6.81 | 0.5, 11 |
| ami_eval | 152.4 | 8.45 | 1.94 | 6.51 | 7.24 | 0.5, 11 |
| ami_mh_dev | 7.6 | 8.25 | 6.13 | 2.12 | 8.25 | 0.7, 11 |
| ami_mh_eval | 7.3 | 8.22 | 4.89 | 3.33 | 8.22 | 0.7, 11 |
| ava_speech_eval | 19.9 | 41.99 | 36.46 | 5.52 | 28.93 | 0.9, 11 |
| chime6_dev | 117.9 | 7.64 | 1.64 | 6.00 | 4.20 | 0.3, 11 |
| chime6_eval | 122.7 | 12.91 | 5.14 | 7.77 | 12.85 | 0.6, 11 |
| dev | 411.3 | 6.26 | 1.95 | 4.31 | 5.46 | 0.5, 11 |
| dipco_dev | 98.1 | 9.66 | 3.60 | 6.06 | 8.04 | 0.5, 11 |
| dipco_eval | 92.0 | 9.32 | 4.79 | 4.52 | 9.06 | 0.6, 11 |
| earnings21_eval | 32.8 | 12.87 | 8.70 | 4.16 | 12.59 | 0.8, 11 |
| icsi_dev | 22.0 | 10.67 | 5.02 | 5.64 | 10.67 | 0.7, 11 |
| icsi_eval | 28.6 | 15.42 | 3.54 | 11.88 | 13.55 | 0.4, 11 |
| notsofar_dev | 123.7 | 2.92 | 1.15 | 1.77 | 2.80 | 0.6, 11 |
| notsofar_eval | 398.5 | 3.08 | 1.84 | 1.24 | 3.08 | 0.7, 11 |
| ten_vad_eval | 0.1 | 22.74 | 21.75 | 0.99 | 15.29 | 0.9, 11 |
| voxconverse_dev | 18.9 | 4.78 | 3.63 | 1.15 | 4.78 | 0.7, 11 |
| voxconverse_eval | 39.0 | 8.17 | 7.03 | 1.14 | 7.93 | 0.8, 11 |
