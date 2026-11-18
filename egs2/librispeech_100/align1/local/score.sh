#!/usr/bin/env bash
# Score one alignment directory against the LibriSpeech MFA word alignments.
# Usage: local/score.sh <align_dir> <data_dir>
set -e
set -u
set -o pipefail

. ./db.sh

if [ $# -ne 2 ]; then
    echo "Usage: $0 <align_dir> <data_dir>"
    exit 1
fi
align_dir=$1
data_dir=$2

if [ -z "${LIBRISPEECH_MFA:-}" ]; then
    echo "$0: set LIBRISPEECH_MFA in db.sh to the MFA alignment zip" \
        "(librispeech_alignments.zip with <utt>.TextGrid files); skip scoring"
    exit 0
fi

opts=
if [ -f "${data_dir}/utt2dur" ]; then
    opts="--utt2dur ${data_dir}/utt2dur"
fi
python3 local/score_boundaries.py \
    --mfa_zip "${LIBRISPEECH_MFA}" \
    --words_ctm "${align_dir}/words.ctm" \
    --out "${align_dir}/score" ${opts}
