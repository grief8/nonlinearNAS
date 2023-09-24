#!/bin/bash
# shellcheck disable=SC2120
function run_proxylessnas() {
  model=$1
  lossType=$2
  wid=$3
  strategy=$5
  batch=$6
  extra_args=$7
  dataset=tiny
  echo start "${model}" "${lossType}" "$wid" "${strategy}"
  dir=./checkpoints/replace/"${model}"/"${strategy}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python main.py  \
  --net "${model}" \
  --dataset "${dataset}" \
  --data_path /data/lifabing/tiny-imagenet-200 \
  --grad_reg_loss_type "${lossType}" \
  --worker_id "$wid" \
  --pretrained \
  --epochs 150 \
  --train_batch_size "${batch}" \
  --checkpoint_path "${dir}"/arch_path.pt \
  --exported_arch_path "${dir}"/checkpoint2.json \
  --train_mode "$4" \
  --strategy "$strategy" \
  "$extra_args" \
  --branches 4
}
#for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
#do
#  run $constraint "$1" "$2" &
#done;
#run_proxylessnas "$1" add#linear 0  "$2" "$3" &
#run_proxylessnas "$1" mul#log 0 "$2" "$3"
run_proxylessnas "$1" add#linear 0,1 "$2" "$3" "$4" "$5"
