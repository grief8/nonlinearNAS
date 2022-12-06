#!/bin/bash
for md in 'searchresnet18' 'searchresnet34' 'searchresnet101' 'searchresnet152';
do
  # shellcheck disable=SC2034
  for st in 'latency' 'throughput';
  do
    for lt in 'add#linear' 'mul#log';
    do
      python baker.py \
      --net "${md}" \
      --dataset "$1" \
      --grad_reg_loss_type "${lt}" \
      --strategy "${st}"
    done
  done
done