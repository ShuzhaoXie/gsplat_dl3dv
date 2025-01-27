#!/bin/bash

split=$1
scene=$2
time=$(date "+%Y-%m-%d_%H:%M:%S")

CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default \
    --data_dir ~/storage/DL3DV/960P-unzip/$split/$scene --data_factor 4 \
    --result_dir ./results/$split/$scene/$time \
    --dataset_type DL3DV \
    --init_type random