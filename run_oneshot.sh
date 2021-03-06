#!/bin/bash
function run() {
  constraint=$1
  wid=$(echo "scale=0;  ($1*10)%8/1" | bc)
#  if test "$wid" -eq 3
#  then
#      wid=7
#  fi
  model=$2
  lossType=$3
  echo start "${model}" "${lossType}" "$constraint" "$wid"
  dir=./checkpoints/oneshot/"${model}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python search.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id "$wid" \
  --epochs 100 \
  --batch-size 1024 \
  --loss-type "${lossType}" \
  --constraint "$constraint" \
  --arc-checkpoint "${dir}"/contraints-"$constraint".json \
  --model-path "${dir}"/contraints-"$constraint".onnx
  #  retrain
  python train.py \
  --net "${model}" \
  --gpu \
  --pretrained \
  --worker-id "$wid" \
  --batch-size 1024 \
  --loss-type "${lossType}" \
  --arc-checkpoint "${dir}"/contraints-"$constraint".json
}

# shellcheck disable=SC2120
function run_proxylessnas() {
  model=$1
  lossType=$2
  wid=$3
  strategy=$5
  echo start "${model}" "${lossType}" "$wid" "${strategy}"
  dir=./checkpoints/oneshot/"${model}"/"${strategy}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python main.py  \
  --net "${model}" \
  --grad_reg_loss_type "${lossType}" \
  --worker_id "$wid" \
  --epochs 120 \
  --train_batch_size 256 \
  --checkpoint_path "${dir}"/arch_path.pt \
  --exported_arch_path "${dir}"/checkpoint.json \
  --train_mode "$4" \
  --strategy "$strategy"
}
#for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
#do
#  run $constraint "$1" "$2" &
#done;
run_proxylessnas "$1" add#linear 0,1,2,3  "$2" "$3" &
run_proxylessnas "$1" mul#log 4,5,6,7 "$2" "$3"
