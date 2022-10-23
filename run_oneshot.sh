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
  --dataset cifar100 \
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
  strategy=$4
  echo start "${model}" "${lossType}" "$wid" "${strategy}"
  dir=./checkpoints/channel/"${model}"/"${strategy}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python main.py  \
  --net "${model}" \
  --dataset cifar100 \
  --data_path ~/data/ \
  --grad_reg_loss_type "${lossType}" \
  --pretrained \
  --worker_id "$wid" \
  --epochs 300 \
  --train_batch_size 2048 \
  --checkpoint_path "${dir}"/checkpoint.pth \
  --strategy "$strategy"
}
#for constraint in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
#do
#  run $constraint "$1" "$2" &
#done;
#run_proxylessnas "$1" add#linear 0  "$2" &
#run_proxylessnas "$1" mul#log 0 "$2"
run_proxylessnas "$1" snl 0 "$2"
