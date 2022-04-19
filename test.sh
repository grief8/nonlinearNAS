#!/bin/bash
for md in 'mobilenet' 'resnet18' 'vgg16' 'resnet50';
do
  # shellcheck disable=SC2034
  for st in 'latency' 'throughput';
  do
    for lt in 'add#linear' 'mul#log';
    do
      python baker.py \
      --net "${md}" \
      --grad_reg_loss_type "${lt}" \
      --strategy "${st}"
    done
  done
done