#!/usr/bin/env bash

# Forced-alignment recipe: train a neural HMM aligner (espnet2/tasks/align.py)
# with a full-sum loss on transcribed speech, then align any data set.
#
#  1. Data preparation (local/data.sh)
#  2. Format wav.scp -> ${dumpdir}/raw
#  3. Remove long/short utterances (train/valid)
#  4. Character token list
#  5. Collect statistics
#  6. Training
#  7. Alignment (CTM of units, words, pauses)
#  8. Scoring (local/score.sh <align_dir> <data_dir>, if the recipe has one)

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    # arithmetic expansion drops the padding BSD wc adds
    echo "$((a))"
}
# Data type of a dump dir's wav.scp for the data loader
audio_type() {
    local fmt
    fmt="$(cat "$1/audio_format" 2>/dev/null || echo "${audio_format}")"
    if [[ "${fmt}" == *ark* ]]; then
        echo kaldi_ark
    else
        echo sound
    fi
}
SECONDS=0

# General configuration
stage=1               # Processes starts from the specified stage.
stop_stage=10000      # Processes is stopped at the specified stage.
ngpu=1                # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1           # The number of nodes.
nj=32                 # The number of parallel jobs.
inference_nj=32       # The number of parallel jobs in alignment.
gpu_inference=false   # Whether to align on gpu.
dumpdir=dump          # Directory to dump features.
expdir=exp            # Directory to save experiments.
python=python3        # Specify python to execute espnet commands.

# Data preparation related
local_data_opts=      # The options given to local/data.sh.

# Feature extraction related
audio_format=flac     # Audio format: wav, flac (raw features only).
fs=16k                # Sampling rate.
min_wav_duration=0.1  # Minimum duration in second.
max_wav_duration=20   # Maximum duration in second.

# Tokenization related
oov="<unk>"           # Out of vocabulary symbol.
blank="<blank>"       # CTC blank symbol (index 0; not an HMM unit).
nlsyms_txt=none       # Non-linguistic symbol list if existing.
cleaner=none          # Text cleaner.

# Aligner model related
align_tag=            # Suffix to the result dir for aligner model training.
align_exp=            # Specify the directory path for the aligner experiment.
align_stats_dir=      # Specify the directory path for the statistics.
align_config=         # Config for aligner model training.
align_args=           # Arguments for aligner model training.
feats_normalize=global_mvn # Normalization layer type.
num_splits_align=1    # Number of splitting for the training corpus.

# Alignment related
inference_config=     # Config for alignment (align_inference.py options).
inference_args=       # Arguments for alignment.
inference_model=      # Aligner model file in ${align_exp}; empty: the averaged model of the
                      # first best_model_criterion in ${align_exp}/config.yaml.
align_out_tag=        # Suffix to the alignment output dir.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=            # Name of training set.
valid_set=            # Name of validation set used for monitoring/tuning network training.
test_sets=            # Names of data sets to align, separated by spaces.
lang=noinfo           # The language type of corpus.

help_message=$(cat << EOF
Usage: $0 --train_set "<train_set_name>" --valid_set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in alignment (default="${inference_nj}").
    --gpu_inference  # Whether to align on gpu (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --audio_format     # Audio format (default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --oov         # Out of vocabulary symbol (default="${oov}").
    --blank       # CTC blank symbol (default="${blank}").
    --nlsyms_txt  # Non-linguistic symbol list (default="${nlsyms_txt}").
    --cleaner     # Text cleaner (default="${cleaner}").

    # Aligner model related
    --align_tag        # Suffix to the result dir (default="${align_tag}").
    --align_exp        # Experiment directory (default="${align_exp}").
    --align_stats_dir  # Statistics directory (default="${align_stats_dir}").
    --align_config     # Config for training (default="${align_config}").
    --align_args       # Arguments for training (default="${align_args}").
    --feats_normalize  # Normalization layer type (default="${feats_normalize}").
    --num_splits_align # Number of splitting for the training corpus (default="${num_splits_align}").

    # Alignment related
    --inference_config # Config for alignment (default="${inference_config}").
    --inference_args   # Arguments for alignment (default="${inference_args}").
    --inference_model  # Model file for alignment; empty: the averaged model of the first
                       # best_model_criterion of the training config (default="${inference_model}").
    --align_out_tag    # Suffix to the alignment output dir (default="${align_out_tag}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set  # Name of training set (required).
    --valid_set  # Name of validation set (required).
    --test_sets  # Names of data sets to align (required).
    --lang       # The language type of corpus (default="${lang}").
EOF
)

log "$0 $*"
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; }
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; }
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; }

data_feats=${dumpdir}/raw
token_listdir=data/${lang}_token_list/char
token_list="${token_listdir}/tokens.txt"

if [ -z "${align_tag}" ]; then
    if [ -n "${align_config}" ]; then
        align_tag="$(basename "${align_config}" .yaml)_raw"
    else
        align_tag="train_raw"
    fi
    if [ -n "${align_args}" ]; then
        align_tag+="$(echo "${align_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi
[ -z "${align_stats_dir}" ] && align_stats_dir="${expdir}/align_stats_raw_${lang}_char"
[ -z "${align_exp}" ] && align_exp="${expdir}/align_${align_tag}"

# The model file and the output tag of alignment are set once training has
# written ${align_exp}/config.yaml (stages 7 and 8).
set_inference_model() {
    if [ -z "${inference_model}" ]; then
        inference_model=$(${python} -c "
import sys, yaml
with open(sys.argv[1]) as f:
    phase, key, _ = yaml.safe_load(f)['best_model_criterion'][0]
print(f'{phase}.{key}.ave.pth')
" "${align_exp}/config.yaml")
    fi
    if [ -z "${align_out_tag}" ]; then
        if [ -n "${inference_config}" ]; then
            align_out_tag="$(basename "${inference_config}" .yaml)"
        else
            align_out_tag=align
        fi
        if [ -n "${inference_args}" ]; then
            align_out_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
        fi
        align_out_tag+="_model_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
}

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Format wav.scp: data/ -> ${data_feats}"
    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
            _suf="/org"
        else
            _suf=""
        fi
        utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"
        rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
        _opts=
        if [ -e data/"${dset}"/segments ]; then
            _opts+="--segments data/${dset}/segments "
        fi
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
            "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
        echo raw > "${data_feats}${_suf}/${dset}/feats_type"
        echo "${audio_format}" > "${data_feats}${_suf}/${dset}/audio_format"
        echo "${fs}" > "${data_feats}${_suf}/${dset}/sample_rate"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"
    _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
    _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
    _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")
    for dset in "${train_set}" "${valid_set}"; do
        utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/org/${dset}/audio_format" "${data_feats}/${dset}/"
        if [ -f "${data_feats}/org/${dset}/sample_rate" ]; then
            cp "${data_feats}/org/${dset}/sample_rate" "${data_feats}/${dset}/"
        fi
        <"${data_feats}/org/${dset}/utt2num_samples" \
            awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
        <"${data_feats}/org/${dset}/wav.scp" \
            utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples" \
            >"${data_feats}/${dset}/wav.scp"
        # Remove empty transcripts
        <"${data_feats}/org/${dset}/text" \
            awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"
        utils/fix_data_dir.sh "${data_feats}/${dset}"
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Generate character token_list from ${data_feats}/${train_set}/text"
    mkdir -p "${token_listdir}"
    # index 0 is the CTC blank; the HMM units are the other tokens plus the
    # model's pause units
    ${python} -m espnet2.bin.tokenize_text \
        --token_type char \
        --input "${data_feats}/${train_set}/text" --output "${token_list}" \
        --non_linguistic_symbols "${nlsyms_txt}" \
        --field 2- \
        --cleaner "${cleaner}" \
        --write_vocabulary true \
        --add_symbol "${blank}:0" \
        --add_symbol "${oov}:1"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    _train_dir="${data_feats}/${train_set}"
    _valid_dir="${data_feats}/${valid_set}"
    log "Stage 5: Aligner collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

    _opts=
    if [ -n "${align_config}" ]; then
        _opts+="--config ${align_config} "
    fi
    _opts+="--frontend_conf fs=${fs} "
    _type=$(audio_type "${_train_dir}")

    _logdir="${align_stats_dir}/logdir"
    mkdir -p "${_logdir}"
    _nj=$(min "${nj}" "$(<${_train_dir}/wav.scp wc -l)" "$(<${_valid_dir}/wav.scp wc -l)")
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    utils/split_scp.pl "${_train_dir}/wav.scp" ${split_scps}
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    utils/split_scp.pl "${_valid_dir}/wav.scp" ${split_scps}

    log "Generate '${align_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
    mkdir -p "${align_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${align_stats_dir}/run.sh"; chmod +x "${align_stats_dir}/run.sh"

    log "Aligner collect-stats started... log: '${_logdir}/stats.*.log'"
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        ${python} -m espnet2.bin.align_train \
            --collect_stats true \
            --use_preprocessor true \
            --token_type char \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --train_data_path_and_name_and_type "${_train_dir}/wav.scp,speech,${_type}" \
            --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
            --valid_data_path_and_name_and_type "${_valid_dir}/wav.scp,speech,${_type}" \
            --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${align_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    if [ "${feats_normalize}" != global_mvn ]; then
        _opts+="--skip_sum_stats"
    fi
    ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${align_stats_dir}"

    for dset in train valid; do
        <"${align_stats_dir}/${dset}/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${align_stats_dir}/${dset}/text_shape.char"
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    _train_dir="${data_feats}/${train_set}"
    _valid_dir="${data_feats}/${valid_set}"
    log "Stage 6: Aligner training: train_set=${_train_dir}, valid_set=${_valid_dir}"

    _opts=
    if [ -n "${align_config}" ]; then
        _opts+="--config ${align_config} "
    fi
    _opts+="--frontend_conf fs=${fs} "
    _type=$(audio_type "${_train_dir}")
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--normalize=global_mvn --normalize_conf stats_file=${align_stats_dir}/train/feats_stats.npz "
    fi

    if [ "${num_splits_align}" -gt 1 ]; then
        _split_dir="${align_stats_dir}/splits${num_splits_align}"
        if [ ! -f "${_split_dir}/.done" ]; then
            rm -f "${_split_dir}/.done"
            ${python} -m espnet2.bin.split_scps \
                --scps \
                    "${_train_dir}/wav.scp" \
                    "${_train_dir}/text" \
                    "${align_stats_dir}/train/speech_shape" \
                    "${align_stats_dir}/train/text_shape.char" \
                --num_splits "${num_splits_align}" \
                --output_dir "${_split_dir}"
            touch "${_split_dir}/.done"
        else
            log "${_split_dir}/.done exists. Spliting is skipped"
        fi
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/wav.scp,speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
        _opts+="--train_shape_file ${_split_dir}/speech_shape "
        _opts+="--train_shape_file ${_split_dir}/text_shape.char "
        _opts+="--multiple_iterator true "
    else
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/wav.scp,speech,${_type} "
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
        _opts+="--train_shape_file ${align_stats_dir}/train/speech_shape "
        _opts+="--train_shape_file ${align_stats_dir}/train/text_shape.char "
    fi

    log "Generate '${align_exp}/run.sh'. You can resume the process from stage 6 using this script"
    mkdir -p "${align_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${align_exp}/run.sh"; chmod +x "${align_exp}/run.sh"

    log "Aligner training started... log: '${align_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        jobname="$(basename ${align_exp})"
    else
        jobname="${align_exp}/train.log"
    fi
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${align_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${align_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m espnet2.bin.align_train \
            --use_preprocessor true \
            --token_type char \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --valid_data_path_and_name_and_type "${_valid_dir}/wav.scp,speech,${_type}" \
            --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
            --valid_shape_file "${align_stats_dir}/valid/speech_shape" \
            --valid_shape_file "${align_stats_dir}/valid/text_shape.char" \
            --resume true \
            --fold_length 80000 \
            --fold_length 150 \
            --output_dir "${align_exp}" \
            ${_opts} ${align_args}
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    set_inference_model
    log "Stage 7: Alignment: training_dir=${align_exp}, model=${inference_model}"
    if [ ! -e "${align_exp}/${inference_model}" ]; then
        log "Error: ${align_exp}/${inference_model} does not exist"
        exit 1
    fi
    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi
    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi

    log "Generate '${align_exp}/run_align.sh'. You can resume the process from stage 7 using this script"
    mkdir -p "${align_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${align_exp}/run_align.sh"; chmod +x "${align_exp}/run_align.sh"

    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${align_exp}/${align_out_tag}/${dset}"
        _logdir="${_dir}/logdir"
        mkdir -p "${_logdir}"

        _nj=$(min "${inference_nj}" "$(<${_data}/wav.scp wc -l)")
        key_file=${_data}/wav.scp
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        utils/split_scp.pl "${key_file}" ${split_scps}

        log "Alignment started... log: '${_logdir}/align_inference.*.log'"
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/align_inference.JOB.log \
            ${python} -m espnet2.bin.align_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/wav.scp,speech,$(audio_type "${_data}")" \
                --data_path_and_name_and_type "${_data}/text,text,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --align_train_config "${align_exp}"/config.yaml \
                --align_model_file "${align_exp}"/"${inference_model}" \
                --output_dir "${_logdir}"/output.JOB \
                ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/align_inference.*.log) ; exit 1; }

        for f in units words pauses words_fwdctc words_bwdctc words_refined; do
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/${f}.ctm"
            done | sort -k1,1 -k3,3n >"${_dir}/${f}.ctm"
        done
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    set_inference_model
    log "Stage 8: Scoring"
    if [ -x local/score.sh ]; then
        for dset in ${test_sets}; do
            ALIGN_SAMPLE_RATE="${fs}" local/score.sh \
                "${align_exp}/${align_out_tag}/${dset}" "${data_feats}/${dset}"
        done
    else
        log "local/score.sh does not exist; skip scoring"
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
