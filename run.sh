#!/bin/bash

split=$1
scene=$2
cid=$3
time=$(date "+%Y-%m-%d_%H:%M:%S")

CUDA_VISIBLE_DEVICES=$cid python simple_trainer.py default \
    --data_dir DL3DV/960P-unzip/$split/$scene --data_factor 4 \
    --result_dir DL3DV/pretrained/$split/$scene/$time \
    --dataset_type DL3DV \
    --init_type random
