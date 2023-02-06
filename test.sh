#!/bin/bash
for md in 'searchcifarsupermodel16' 'searchcifarsupermodel22' 'searchcifarsupermodel26' 'searchcifarsupermodel50' 'searchcifarsupermodel101';
do
  # shellcheck disable=SC2034
  for st in 'latency' 'throughput';
  do
    for lt in 'add#linear' 'mul#log';
    do
      echo baker.py \
      --net "${md}" \
      --dataset "$1" \
      --grad_reg_loss_type "${lt}" \
      --strategy "${st}" \
      --choice "v0.7"
      
      python baker.py \
      --net "${md}" \
      --dataset "$1" \
      --grad_reg_loss_type "${lt}" \
      --strategy "${st}" \
      --choice "v0.7"
    done
  done
done