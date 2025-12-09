#!/bin/bash

# 循环数据集
for D in FD001 FD002 FD003 FD004; do
    # 循环窗口大小
    for W in 48 96; do
        echo "==============================================="
        echo "Running dataset $D with window_size $W"
        echo "==============================================="

        python MainCondition.py \
            --epoch 70 \
            --dataset $D \
            --lr 2e-3 \
            --state sample \
            --model_name DiffWave \
            --T 500 \
            --window_size $W \
            --sample_type ddpm \
            --input_size 14

        echo
    done
done

echo "All runs finished!"
