import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import re
from tqdm import tqdm
import argparse
from utils import *
from data_processor import *
from feature_extractor import *
from intervention_operator import *
from evaluate import *
import torch
from transformers import AutoConfig
from care_trainer import tune_alpha, wrap_llama_with_checkpoint, inference_mode
import gc


def main(args):
    TUNE_PATH = validate_save_path(f'./{args.run_name}/tunes')
    model_config = AutoConfig.from_pretrained(HF_NAMES[args.model_name], 
                                              do_sample=False,
                                              local_files_only=True,)
    torch_dtype = getattr(model_config, 'torch_dtype', torch.float16)
    if torch_dtype != torch.bfloat16 or not cuda_supports_bfloat16():
        torch_dtype = torch.float16 

    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads 
    num_to_intervene = args.K if args.K < 1 and args.K > 0 else int(args.K)
    
    setseed(args.seed)

    base_key, data_key, act_key, vene_key = get_config_key(args) 
    
    dataset, all_folds_train_indices, all_folds_val_indices, all_test_files = data_load_split_pipeline(args, 
                                                                                                       SPLITS_PATH, 
                                                                                                       data_key,
                                                                                                       DATASET_CONFIGS[args.dataset_name])
    if args.use_prefix:
        all_folds_prefix = make_many_shot_prefix(args, 
                                                SPLITS_PATH, 
                                                data_key, 
                                                dataset, 
                                                all_folds_train_indices, 
                                                all_folds_val_indices,)
    all_folds_interventions, all_folds_directions, all_folds_top_heads, tuning_activations = data2vene_pipeline(args, 
                                                                                            dataset, 
                                                                                            all_folds_train_indices, 
                                                                                            all_folds_val_indices, 
                                                                                            num_layers,
                                                                                            num_heads,
                                                                                            num_to_intervene,
                                                                                            INNER_STATES_PATH, 
                                                                                            act_key, 
                                                                                            vene_key,
                                                                                            device='cuda')
    
    for i in tqdm(range(args.num_fold), total=args.num_fold, desc=f'k-fold validation'):
        if args.run_fold >= 0 and args.run_fold < args.num_fold and i != args.run_fold:
            continue
        if args.scan_checkpoints:
            output_dir = TUNE_PATH / f'{vene_key}_{i}_of_{args.num_fold}'
            checkpoint_dirs = sorted(
                                    [d for d in os.listdir(output_dir) 
                                    if os.path.isdir(os.path.join(output_dir, d)) 
                                    and re.match(r"checkpoint-\d+", d)],
                                    key=lambda x: int(x.split("-")[-1])
                                )
            for checkpoint_dir in checkpoint_dirs[-1:]:# last checkpoint
                checkpoint_path = os.path.join(output_dir, checkpoint_dir, 'model.safetensors')
                torch.cuda.empty_cache()
                model = MODELCLASS[args.model_name].from_pretrained(HF_NAMES[args.model_name],
                                                        config=model_config, 
                                                        low_cpu_mem_usage=True, 
                                                        torch_dtype=torch_dtype, 
                                                        device_map="auto", 
                                                        local_files_only=True,)
                check_device(model)
                model = wrap_llama_with_checkpoint(model, 
                                                   checkpoint_path=checkpoint_path, 
                                                   adaptive=args.adaptive, 
                                                   act_fn=args.act_fn, 
                                                   use_dual_dirs=args.use_dual_dirs)
                inference_mode(model)
                model.eval()
                run_key = vene_key+checkpoint_dir
                if args.test_dataset is not None and os.path.isfile(args.test_dataset):
                    test_split_file = args.test_dataset
                    test_key = os.path.basename(args.test_dataset)
                    run_key = run_key + test_key
                    print(f'Test on other data {test_key}')
                else:
                    test_split_file = all_test_files[i]
                curr_fold_results = one_fold_evaluate_pipeline(args,
                                                       model, 
                                                       num_heads, 
                                                       all_folds_interventions[i],
                                                       i,
                                                       test_split_file=test_split_file,
                                                       vene_key=run_key,
                                                       many_shot_prefix=all_folds_prefix[i] if args.use_prefix else None)
                del model 
                torch.cuda.empty_cache()
                gc.collect()
            continue

        if args.tune_alpha and args.num_fold > 1:
            train_set_idxs = all_folds_train_indices[i]
            val_set_idxs = all_folds_val_indices[i]

            train_dataset = dataset.select(train_set_idxs)
            val_dataset = dataset.select(val_set_idxs)
            tuning_dataset = {'train': train_dataset,'val': val_dataset}
            torch.cuda.empty_cache()
            tune_alpha(args, 
                               tuning_dataset, 
                               vene_key, 
                               i, 
                               all_folds_top_heads[i], 
                               all_folds_directions[i],
                               tuning_activations=tuning_activations,
                               )
            if args.train_only:
                gc.collect()
                torch.cuda.empty_cache()
                continue
            inference_mode(model)
            check_device(model)
        else:
            torch.cuda.empty_cache()
            model = MODELCLASS[args.model_name].from_pretrained(HF_NAMES[args.model_name],
                                                    config=model_config, 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch_dtype, 
                                                    device_map="auto", 
                                                    local_files_only=True,)
            check_device(model)
        model.eval()

        run_key = vene_key
        if args.test_dataset is not None and os.path.isfile(args.test_dataset):
            test_split_file = args.test_dataset
            test_key = os.path.basename(args.test_dataset)
            run_key = run_key + test_key
            print(f'Test on other data {test_key}')
        else:
            test_split_file = all_test_files[i]

        curr_fold_results = one_fold_evaluate_pipeline(args,
                                                       model, 
                                                       num_heads, 
                                                       all_folds_interventions[i],
                                                       i,
                                                       test_split_file=test_split_file,
                                                       vene_key=run_key,
                                                       many_shot_prefix=all_folds_prefix[i] if args.use_prefix else None)
        del model 
        torch.cuda.empty_cache()
        gc.collect()

    if args.num_fold == 0:
        i = 0
        if args.scan_checkpoints:
            output_dir = TUNE_PATH / f'{vene_key}_{0}_of_{args.num_fold}'
            checkpoint_dir = sorted(
                                    [d for d in os.listdir(output_dir) 
                                    if os.path.isdir(os.path.join(output_dir, d)) 
                                    and re.match(r"checkpoint-\d+", d)],
                                    key=lambda x: int(x.split("-")[-1])
                                )[-1]
            checkpoint_path = os.path.join(output_dir, checkpoint_dir, 'model.safetensors')
            torch.cuda.empty_cache()
            model = MODELCLASS[args.model_name].from_pretrained(HF_NAMES[args.model_name],
                                                    config=model_config, 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch_dtype, 
                                                    device_map="auto", 
                                                    local_files_only=True,)
            check_device(model)
            model = wrap_llama_with_checkpoint(model, 
                                                checkpoint_path=checkpoint_path, 
                                                adaptive=args.adaptive, 
                                                act_fn=args.act_fn, 
                                                use_dual_dirs=args.use_dual_dirs)
            inference_mode(model)
            model.eval()
            run_key = vene_key+checkpoint_dir
            if args.test_dataset is not None and os.path.isfile(args.test_dataset):
                test_split_file = args.test_dataset
                test_key = os.path.basename(args.test_dataset)
                run_key = run_key + test_key
                print(f'Test on other data {test_key}')
            curr_fold_results = one_fold_evaluate_pipeline(args,
                                                    model, 
                                                    num_heads, 
                                                    all_folds_interventions[0],
                                                    i,
                                                    test_split_file=test_split_file,
                                                    vene_key=run_key,
                                                    many_shot_prefix=all_folds_prefix[i] if args.use_prefix else None)
            return

        torch.cuda.empty_cache()
        if args.tune_alpha:
            train_set_idxs = all_folds_train_indices[i]
            val_set_idxs = all_folds_val_indices[i]

            train_dataset = dataset.select(train_set_idxs)
            val_dataset = dataset.select(val_set_idxs)
            tuning_dataset = {'train': train_dataset,'val': val_dataset}
            tune_alpha(args, 
                               tuning_dataset, 
                               vene_key, 
                               i, 
                               all_folds_top_heads[i], 
                               all_folds_directions[i],
                               tuning_activations=tuning_activations,
                               )
            if args.train_only:
                return
            inference_mode(model)
            check_device(model)
        else:
            model = MODELCLASS[args.model_name].from_pretrained(HF_NAMES[args.model_name],
                                                    config=model_config, 
                                                    low_cpu_mem_usage=True, 
                                                    torch_dtype=torch_dtype, 
                                                    device_map="auto", 
                                                    local_files_only=True,)
            check_device(model)
        model.eval()

        run_key = vene_key
        if args.test_dataset is not None and os.path.isfile(args.test_dataset):
            test_split_file = args.test_dataset
            test_key = os.path.basename(args.test_dataset)
            run_key = run_key + test_key
            print(f'Test on other data {test_key}')

        curr_fold_results = one_fold_evaluate_pipeline(args,
                                                       model, 
                                                       num_heads, 
                                                       all_folds_interventions[i],
                                                       i,
                                                       test_split_file=test_split_file,
                                                       vene_key=run_key,
                                                       many_shot_prefix=all_folds_prefix[i] if args.use_prefix else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, default='llama2_chat_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--seed', type=int, default=42, help='seed')

    # data
    parser.add_argument('--dataset_name', type=str, default='tqa_gh_v1', choices=DATASET_CONFIGS.keys())
    parser.add_argument('--mc_key', type=str, default='mc2_targets')
    parser.add_argument('--feature_name', type=str, default='head_out')
    parser.add_argument('--data_ratio', type=float, help='ratio of total data set size to be utilized', default=1.0)
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--test_ratio', type=float, default=1.0)
    parser.add_argument('--test_dataset', type=str, required=False)

    # extract
    parser.add_argument('--n_shot', type=int, help='n shot', default=0)
    parser.add_argument('--sample_times', type=int, help='sample n tokens from 1 sequence', default=1)
    parser.add_argument('--last_token', action='store_true', help='sample activations from last tokens', default=False)
    parser.add_argument('--do_sample',action='store_true', help='whether do sample when groupping demos', default=False)
    
    # operator
    parser.add_argument('--K', type=float, default=0, help='number of top heads to intervene on; if <1, works as the threshold of probes accuracy')
    parser.add_argument('--alpha', type=float, default=0, help='alpha, intervention strength')
    parser.add_argument('--temperature', type=float, default=1.0, help='sigmoid temperature for adaption')
    parser.add_argument('--within_layers', type=str, default=None, required=False)
    parser.add_argument('--edit_module', type=str, default='head_out')

    parser.add_argument('--use_normalized_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--use_split_data', action='store_true', help='use train-val-splits for com and probing, other than all the fold', default=False)
    
    parser.add_argument('--probe_class', type=str, default='ncomc') # lr, ncomc

    # evaluate
    parser.add_argument('--judge_name', type=str, default="allenai/truthfulqa-info-judge-llama2-7B", required=False)
    parser.add_argument('--info_name', type=str, default="allenai/truthfulqa-info-judge-llama2-7B", required=False)
    parser.add_argument('--only_mc', action='store_true', default=False)

    # tuning alpha
    parser.add_argument('--tune_alpha', action='store_true', default=False)
    parser.add_argument('--train_only', action='store_true', default=False)
    parser.add_argument('--scan_checkpoints', action='store_true', default=False,help='')
    parser.add_argument('--lr',type=float,default=1e-0)
    parser.add_argument('--min_lr_rate',type=float,default=0.5)
    parser.add_argument('--warmup_ratio',type=float,default=0.1)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--train_batch',type=int,default=8)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=4)
    parser.add_argument('--num_epoch',type=int,default=3)
    parser.add_argument('--eval_batch',type=int,default=16)
    parser.add_argument('--dpo_beta', type=float, default=0.2,required=False,help='The hyperparameter of beta value for DPO')
    parser.add_argument('--l1_lambda', type=float, default=0, help='l1 regularization lambda for alpha',required=False)
    parser.add_argument('--l2_lambda', type=float, default=0, help='l2 regularization lambda for temperature',required=False)
    parser.add_argument('--use_dual_dirs', action='store_true', default=False,help='')
    parser.add_argument('--adaptive', action='store_true', default=False,help='')
    parser.add_argument('--save_strategy',type=str,default='last',required=False,help='The strategy to save the model: best: only save the best model; no: do not save the model')
    parser.add_argument('--lr_scheduler_type',type=str,default='cosine_with_min_lr',required=False, help='')
    parser.add_argument('--act_fn',type=str,default='relu',required=False,help='')
    parser.add_argument('--apply_chat_template', action='store_true', default=False,help='Using llama2 chat template in the prompt; False by default')

    # others
    parser.add_argument('--use_prefix', action='store_true', help='fewshot samples', default=False)
    parser.add_argument('--run_name', type=str, default='default_run')
    parser.add_argument('--run_fold', type=int, default='-1', help='run a specific fold')
    
    args = parser.parse_args()

    main(args)
