#!/bin/bash

RUNNAME=other_model_BASELINE
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


for MODEL in gpt-j-6b phi-1_5 llama_7B gemma-1.1-7b-it Phi-3.5-mini-instruct llama2_7B; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL \
        --num_fold 1 \
        --run_name $RUNNAME 
done

for MODEL in glm-4-9b-chat Qwen2.5-7B-Instruct Qwen2-7B-Instruct Ministral-8B-Instruct Phi-4-mini-instruct llama31_inst_8B; do
    MODEL_TYPE=type-2 CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL \
        --num_fold 1 \
        --run_name $RUNNAME 
done
