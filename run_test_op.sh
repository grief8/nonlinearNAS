#!/bin/bash

# shellcheck disable=SC2120
function run_proxylessnas() {
  model=$1
  lossType=$2
  wid=$3
  strategy=$5
  batch=$6
  extra_args=$7
  echo start "${model}" "${lossType}" "$wid" "${strategy}"
  dir=./checkpoints/concat/"${model}"/"${strategy}"/"${lossType}"
  #  search
  mkdir -p "${dir}"
  python test_op.py  \
  --net "${model}" \
  --dataset cifar100 \
  --data_path ~/data/ \
  --grad_reg_loss_type "${lossType}" \
  --worker_id "$wid" \
  --pretrained \
  --epochs 150 \
  --train_batch_size "${batch}" \
  --checkpoint_path "${dir}"/arch_path.pt \
  --exported_arch_path "${dir}"/checkpoint2.json \
  --train_mode "$4" \
  --strategy "$strategy" \
  --kd_teacher_path ~/projects/nonlinearNAS/checkpoints/teacher/cifar_resnet152.pth \
  --branches 4
}
run_proxylessnas "$1" add#linear 0,1,2,3,4,5,6,7 "$2" "$3" "$4"
