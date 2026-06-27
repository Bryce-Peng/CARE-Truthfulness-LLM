#!/bin/bash

RUNNAME=llama2_chat_7B_OOD
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
MODEL=llama2_chat_7B

nq_open=../standard_datasets/iti_nq_open_val.csv # 3610
arc_c=../standard_datasets/ai2_arc_c_test.csv    # 1172
trivia_qa=../standard_datasets/iti_trivia_qa_val.csv # 3610
openbookqa=../standard_datasets/openbookqa_test.csv # 500
halueval=../standard_datasets/HaluEval_qa_1000.csv # 1000


#for dataset in $nq_open $arc_c $trivia_qa $openbookqa; do
for dataset in $halueval $openbookqa; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_name $RUNNAME \
        --num_fold 1 --test_dataset $dataset --only_mc
done

# CARE-TF for OOD test
HEADS=0.55
ALPHA=10
TEMP=100

#for dataset in $nq_open $arc_c $trivia_qa $openbookqa; do
for dataset in $halueval $openbookqa; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_name $RUNNAME \
        --n_shot $SHOT --do_sample --adaptive \
        --K $HEADS --alpha $ALPHA --temperature $TEMP --use_normalized_center_of_mass \
        --num_fold 0 --test_dataset $dataset --only_mc 
done

# CARE-PO for OOD test

SHOT=100
HEADS=-1

#train
CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_name $RUNNAME \
    --n_shot $SHOT --do_sample --use_dual_dirs \
    --K $HEADS --tune_alpha \
    --num_fold 0 --train_only

#test
#for dataset in $nq_open $arc_c $trivia_qa $openbookqa; do
for dataset in $halueval $openbookqa; do
    CUDA_VISIBLE_DEVICES=0 python ../care/run_pipeline.py $MODEL --seed $SEED --run_name $RUNNAME \
        --n_shot $SHOT --do_sample --use_dual_dirs \
        --K $HEADS --tune_alpha \
        --num_fold 0 --test_dataset $dataset --only_mc \
        --scan_checkpoints
done

