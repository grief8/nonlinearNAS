#!/bin/bash
model=$1
count=$2
batch=$3
./run_oneshot_cifar100.sh ${model} search ${count} ${batch};
for idx in 1 2 6 4 8 -1;
    do 
    ./run_oneshot_avg.sh ${model} retrain ${count} ${batch} layer4_block${idx};
done
echo "-----------------------"
echo "Begin layer"
echo "-----------------------"
for idx in 1 2 6 4 8 -1;
    do 
    ./run_oneshot_avg.sh ${model} retrain ${count} ${batch} layer4_block${idx};
done
