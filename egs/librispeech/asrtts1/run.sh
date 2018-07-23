#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=           # will be deprecated, please use ngpu
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
seed=1         # random seed number
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN
fs=16000       # sampling frequency
fmax=""        # maximum frequency
fmin=""        # minimum frequency
n_mels=80      # number of mel basis
n_fft=512      # number of fft points
n_shift=160    # number of shift points
win_length=400 # number of samples in analysis window

# ASR network archtecture
# encoder related
etype=blstmp   # encoder architecture type
elayers=6
eunits=320
eprojs=512 # ASR default 320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
adim=320
atype=location
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5

# TTS network archtecture
# encoder related
tts_embed_dim=512
tts_elayers=1
tts_eunits=512
tts_econv_layers=3 # if set 0, no conv layer is used
tts_econv_chans=512
tts_econv_filts=5
# decoder related
tts_dlayers=2
tts_dunits=1024
tts_prenet_layers=2  # if set 0, no prenet is used
tts_prenet_units=256
tts_postnet_layers=5 # if set 0, no postnet is used
tts_postnet_chans=512
tts_postnet_filts=5
# attention related
tts_adim=128
tts_aconv_chans=32
tts_aconv_filts=15      # resulting in filter_size = aconv_filts * 2 + 1
tts_cumulate_att_w=true # whether to cumulate attetion weight
tts_use_batch_norm=true # whether to use batch normalization in conv layer
tts_use_concate=true    # whether to concatenate encoder embedding with decoder lstm outputs
tts_use_residual=false  # whether to use residual connection in encoder convolution
tts_use_masking=true    # whether to mask the padded part in loss calculation
tts_bce_pos_weight=1.0  # weight for positive samples of stop token in cross-entropy calculation

tts_dropout=0.5
tts_zoneout=0.1

# common configurations

# minibatch related
batchsize=50
batch_sort_key="" # empty or input or output (if empty, shuffled batch will be used)
maxlen_in=400  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adam
lr=1e-3
eps=1e-6
weight_decay=0.0
epochs=20

# rnnlm related
lm_weight=0.3

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z ${gpu} ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ ${gpu} -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960_tts
train_dev=dev_tts
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)_tts
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank_tts
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
	local/make_fbank.sh --cmd "${train_cmd}" --nj 32 \
			    --fs ${fs} --fmax "${fmax}" --fmin "${fmin}" --win_length ${win_length} \
			    --n_mels ${n_mels} --n_fft ${n_fft} --n_shift ${n_shift} \
			    data/${x}_tts exp/make_fbank/${x} ${fbankdir}
    done

    utils/combine_data.sh data/train_960_tts data/train_clean_100_tts data/train_clean_360_tts data/train_other_500_tts
    utils/combine_data.sh data/dev_tts data/dev_clean_tts data/dev_other_tts
    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    utils/copy_data_dir.sh data/${train_set} data/${train_set}_org
    utils/copy_data_dir.sh data/${train_dev} data/${train_dev}_org
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    # make sure that we only use the pair data for global normalization
    compute-cmvn-stats scp:data/train_clean_100_tts/feats.scp data/train_clean_100_tts/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,18}/${USER}/espnet-data/egs/librispeech/asrtts1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,18}/${USER}/espnet-data/egs/librispeech/asrtts1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi

    dump.sh --cmd "$train_cmd" --nj 80 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/train_clean_100_tts/cmvn.ark exp/dump_feats/train_tts ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/train_clean_100_tts/cmvn.ark exp/dump_feats/dev_tts ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}_tts/feats.scp data/train_clean_100_tts/cmvn.ark exp/dump_feats/recog_tts/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/train_clean_100_tts_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/train_clean_100_tts/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # add mode
    # train_clean_100 for pair data, train_clean_360 for audio only, train_other_500 and/or train_clean_360
    awk '{print $1 " p"}' data/train_clean_100_tts/utt2spk >  data/${train_set}/utt2mode.scp
    awk '{print $1 " a"}' data/train_clean_360_tts/utt2spk >> data/${train_set}/utt2mode.scp
    awk '{print $1 " t"}' data/train_other_500_tts/utt2spk >> data/${train_set}/utt2mode.scp
    # dev set has pair data
    awk '{print $1 " p"}' data/${train_dev}/utt2spk >  data/${train_dev}/utt2mode.scp


    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --scps data/${train_set}/utt2mode.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --scps data/${train_dev}/utt2mode.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask}_tts ${dict} > ${feat_recog_dir}/data.json
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
lmexpdir=exp/train_rnnlm_2layer_bs256
mkdir -p ${lmexpdir}
#if [ ${stage} -le 3 ]; then
if false; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train
    mkdir -p ${lmdatadir}
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 data/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --epoch 60 \
        --batchsize 256 \
        --dict ${dict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_adim_${adim}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
    expdir=${expdir}_taco2_enc${tts_embed_dim}
    if [ ${tts_econv_layers} -gt 0 ];then
        expdir=${expdir}-${tts_econv_layers}x${tts_econv_filts}x${tts_econv_chans}
    fi
    expdir=${expdir}-${tts_elayers}x${tts_eunits}_dec${tts_dlayers}x${tts_dunits}
    if [ ${tts_prenet_layers} -gt 0 ];then
        expdir=${expdir}_pre${tts_prenet_layers}x${tts_prenet_units}
    fi
    if [ ${tts_postnet_layers} -gt 0 ];then
        expdir=${expdir}_post${tts_postnet_layers}x${tts_postnet_filts}x${tts_postnet_chans}
    fi
    expdir=${expdir}_att${tts_adim}-${tts_aconv_filts}x${tts_aconv_chans}
    if ${tts_cumulate_att_w};then
        expdir=${expdir}_cm
    fi
    if ${tts_use_batch_norm};then
        expdir=${expdir}_bn
    fi
    if ${tts_use_residual};then
        expdir=${expdir}_rs
    fi
    if ${tts_use_concate};then
        expdir=${expdir}_cc
    fi
    if ${tts_use_masking};then
        expdir=${expdir}_msk_pw${tts_bce_pos_weight}
    fi
    expdir=${expdir}_do${tts_dropout}_zo${tts_zoneout}_lr${lr}_ep${eps}_wd${weight_decay}_bs$((batchsize*ngpu))
    if [ ! -z ${batch_sort_key} ];then
        expdir=${expdir}_sort_by_${batch_sort_key}_mli${maxlen_in}_mlo${maxlen_out}
    fi
    expdir=${expdir}_sd${seed}
else
    expdir=exp/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asrtts_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --tts-embed_dim ${tts_embed_dim} \
        --tts-elayers ${tts_elayers} \
        --tts-eunits ${tts_eunits} \
        --tts-econv_layers ${tts_econv_layers} \
        --tts-econv_chans ${tts_econv_chans} \
        --tts-econv_filts ${tts_econv_filts} \
        --tts-dlayers ${tts_dlayers} \
        --tts-dunits ${tts_dunits} \
        --tts-prenet_layers ${tts_prenet_layers} \
        --tts-prenet_units ${tts_prenet_units} \
        --tts-postnet_layers ${tts_postnet_layers} \
        --tts-postnet_chans ${tts_postnet_chans} \
        --tts-postnet_filts ${tts_postnet_filts} \
        --tts-adim ${tts_adim} \
        --tts-aconv-chans ${tts_aconv_chans} \
        --tts-aconv-filts ${tts_aconv_filts} \
        --tts-cumulate_att_w ${tts_cumulate_att_w} \
        --tts-use_batch_norm ${tts_use_batch_norm} \
        --tts-use_concate ${tts_use_concate} \
        --tts-use_residual ${tts_use_residual} \
        --tts-use_masking ${tts_use_masking} \
        --tts-bce_pos_weight ${tts_bce_pos_weight} \
        --tts-dropout-rate ${tts_dropout} \
        --tts-zoneout-rate ${tts_zoneout} \
        --lr ${lr} \
        --eps ${eps} \
        --weight-decay ${weight_decay} \
        --batch_sort_key ${batch_sort_key} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_rnnlm${lm_weight}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json 

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --rnnlm ${lmexpdir}/rnnlm.model.best \
            --lm-weight ${lm_weight} \
            &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

