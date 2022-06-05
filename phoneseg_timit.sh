#!/bin/bash

set -e
set -x

RVERB="-v --dry-run"
RVERB=""
CPC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SAVE_DIR="$(
python - "$@" << END
if 1:
  import argparse
  import os.path
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('load', type=str,
                        help="Path to the checkpoint to evaluate.")
  parser.add_argument('--pathCheckpoint')
  parser.add_argument('--tolerance', type=int, default=2)
  parser.add_argument('--strict', action='store_true')
  args, _ = parser.parse_known_args()

  checkpoint_dir = os.path.dirname(args.load)
  checkpoint_no = args.load.split('_')[-1][:-3]
  expLabel = f"{checkpoint_no}_tolerance{args.tolerance}"
  if args.strict:
    expLabel += '_strict'
  pathCheckpoint = f"{checkpoint_dir}/phoneSegTIMIT_{expLabel}"
  print(pathCheckpoint)
END
)"
mkdir -p ${SAVE_DIR}
echo $0 "$@" >> ${SAVE_DIR}/out.txt
exec python -u cpc/eval/phone_segmentation.py \
    --pathDB /pio/scratch/1/i323106/data/TIMIT/test/ \
    --load "$@" \
    --pathPhone /pio/scratch/1/i323106/data/TIMIT/converted_aligned_phones.txt \
    --file_extension .WAV \
    --pathCheckpoint $SAVE_DIR \
    2>&1 | tee -ai ${SAVE_DIR}/out.txt
