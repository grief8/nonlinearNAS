#!/bin/bash
model=$1
batch=$2
./run_oneshot_cifar100.sh "${model}" search latency "${batch}" --pretrained;
./run_oneshot_cifar100.sh "${model}" retrain latency "${batch}" --clamp;
./run_oneshot_cifar100.sh "${model}" retrain latency "${batch}" --pretrained;
./run_oneshot_cifar100.sh "${model}" retrain latency "${batch}" --std_pruning;