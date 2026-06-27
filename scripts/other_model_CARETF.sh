#!/bin/bash

RUNNAME=other_model_CARETF
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



SEED=42
for MODEL in gpt-j-6b phi-1_5 llama_7B; do
for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_fold $FOLD \
        --n_shot 50 --do_sample --adaptive \
        --K $HEADS --alpha $ALPHA --temperature $TEMP --use_normalized_center_of_mass \
        --run_name $RUNNAME 
done
done

for MODEL in gemma-1.1-7b-it Phi-3.5-mini-instruct llama2_7B; do
for FOLD in 0 1; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_fold $FOLD \
        --n_shot 100 --do_sample --adaptive \
        --K $HEADS --alpha $ALPHA --temperature $TEMP --use_normalized_center_of_mass \
        --run_name $RUNNAME 
done
done


for MODEL in glm-4-9b-chat Qwen2.5-7B-Instruct Qwen2-7B-Instruct Ministral-8B-Instruct Phi-4-mini-instruct llama31_inst_8B; do
for FOLD in 0 1; do
    MODEL_TYPE=type-2 CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_fold $FOLD \
        --n_shot 100 --do_sample --adaptive \
        --K $HEADS --alpha $ALPHA --temperature $TEMP --use_normalized_center_of_mass \
        --run_name $RUNNAME 
done
done
