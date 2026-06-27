#!/bin/bash

RUNNAME=llama2_chat_7B_CAREPO
######################## logging ##########################
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_PATH=$(readlink -f "$0")
LOG_PATH=$(dirname "$SCRIPT_PATH")/../log
mkdir -p ${LOG_PATH}
LOG_FILE="${LOG_PATH}/$(basename "$SCRIPT_PATH" ".sh").${TIMESTAMP}.log"
exec > >(stdbuf -oL tee -a "${LOG_FILE}") 2>&1

RESULTS_PATH=$(dirname "$SCRIPT_PATH")/../results
mkdir -p ${RESULTS_PATH}
cd ${RESULTS_PATH}
mkdir -p ${RUNNAME}
######################## logging ##########################


MODEL=llama2_chat_7B
HEADS=-1
SHOT=100
SEED=42

#train
for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_fold $FOLD \
        --n_shot $SHOT --do_sample --use_dual_dirs \
        --K $HEADS --tune_alpha --run_name $RUNNAME \
        --train_only
done

#test
for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_fold $FOLD \
        --n_shot $SHOT --do_sample --use_dual_dirs \
        --K $HEADS --tune_alpha --run_name $RUNNAME \
        --scan_checkpoints
done
