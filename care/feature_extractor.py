import torch
from baukit import TraceDict
from tqdm import tqdm
import numpy as np
import os
import pickle
from utils import validate_save_path, QA_SEP, DEMO_SEP, HF_NAMES, MODELCLASS, cuda_supports_bfloat16
from transformers import AutoTokenizer, AutoConfig
from data_processor import data_to_prompt_tqa_hf, make_many_shot_prompt, data_to_tuning_prompt_tqa_hf
from numpy.random import RandomState

def get_llama_module_activations(model, input_ids, device, sample_pos=-1, target='self_attn.head_out', position_ids=None): 
    decoders='model.layers'
    model_type = getattr(model.config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'
        target = 'attn.head_out'
        
    MODULES = [f"{decoders}.{i}.{target}" for i in range(model.config.num_hidden_layers)]

    sample_pos = torch.tensor(sample_pos)
    if sample_pos.ndim == 0:
        sample_pos = sample_pos.unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        input_ids = input_ids.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)
        with TraceDict(model, MODULES) as ret:
            output = model(input_ids, position_ids=position_ids)
        module_activations = [ret[module].output.squeeze()[sample_pos].cpu() for module in MODULES]
        module_activations = torch.stack(module_activations, dim = 0).detach() # layer, seqlen, dim

    return module_activations 


def find_sublist(main_list, sub_list):
    sub_len = len(sub_list)
    for i in range(len(main_list) - sub_len + 1):
        if main_list[i:i+sub_len] == sub_list:
            return i
    return -1


def get_special_offset(tokenizer):
    full_ec = tokenizer.encode('just a test text', add_special_tokens=True)
    bare_ec = tokenizer.encode('just a test text', add_special_tokens=False)
    offset = find_sublist(full_ec, bare_ec)
    return offset


def find_last_answer_span(prompts, grouped_indices, tokenizer, chat=False, grouped_prompts=None):
    offset = get_special_offset(tokenizer)
    print(f'tokenizer offset is {offset}')

    all_answer_spans = []
    for i, indices in enumerate(tqdm(grouped_indices, desc=f'find_last_answer_span')):
        num_prompt = len(indices)
        answer_spans = []
        grouped_prompt = grouped_prompts[i]
        for j in range(num_prompt):
            if j < num_prompt - 1:
                answer_spans.append((0,0))
                continue
            if not chat:
                temp_prompt = DEMO_SEP.join([prompts[k] for k in indices[:j]]) # 0:j=1 -> []~[]
                last_qa = prompts[indices[j]]
                last_qa_split = last_qa.split(QA_SEP)
                assert len(last_qa_split) == 2, f'len(last_qa_split) is {len(last_qa_split)}: {last_qa}'
                last_q = last_qa_split[0]
                temp_prefix_prompt = ''.join([last_q, QA_SEP]) if j == 0 else ''.join([temp_prompt, DEMO_SEP, last_q, QA_SEP]) 
                temp_full_prompt = last_qa if j == 0 else ''.join([temp_prompt, DEMO_SEP, last_qa])
                start_idx = len(tokenizer.encode(temp_prefix_prompt, add_special_tokens=False)) + offset # first token in answer span
                end_idx = len(tokenizer.encode(temp_full_prompt, add_special_tokens=False)) + offset # last position in answer span, [s:e] is the answer slice
            else:
                temp_prompt = grouped_prompt[:2*j]
                last_qa = grouped_prompt[2*j:2*j+2]
                temp_prefix_prompt = grouped_prompt[:2*j+1]
                temp_full_prompt = grouped_prompt[:2*j+2]
                # for llama-2, 'add_generation_prompt' and 'add_special_tokens' is not effective
                prefix_tokens = tokenizer.apply_chat_template(temp_prefix_prompt, tokenize=True) 
                full_tokens = tokenizer.apply_chat_template(temp_full_prompt, tokenize=True)
                start_idx = len(prefix_tokens) + 2 # +2 for " A:"
                end_idx = len(full_tokens) - 1 # -1 for </s>
            
            answer_spans.append((start_idx, end_idx))
        all_answer_spans.append(answer_spans)

    return all_answer_spans


def get_multi_shot_activations(prompts, 
                               grouped_labels, 
                               model, 
                               tokenizer, 
                               seed, 
                               device, 
                               grouped_indices, 
                               raw_prompts, 
                               tag='',
                               chat=False,
                               sample_times=1,
                               last_token=False,):
    
    all_activations =[]
    tokenids = []
    sample_positions = []
    all_labels = []
    print("Getting activations")
    activation_extract_fn = get_llama_module_activations 
    
    all_answer_spans = find_last_answer_span(raw_prompts, grouped_indices, tokenizer, chat, grouped_prompts=prompts)

    token_counts = []

    np_rng = RandomState(seed=seed)
    
    for idx, prompt in enumerate(tqdm(prompts, total=len(prompts), desc=f'{tag} get_multi_shot_activations')):
    
        sample_count = sample_times 
        sample_range = np.arange(*all_answer_spans[idx][-1]) 

        actual_sample_count = min(sample_count, len(sample_range))
        sample_pos = np_rng.choice(sample_range, actual_sample_count, replace=False) if not last_token else sample_range[-actual_sample_count:]
        sample_pos = np.sort(sample_pos)
        sample_positions.append(sample_pos)

        label = grouped_labels[idx]
        all_labels.extend([label] * actual_sample_count)

        if not chat:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        else:
            input_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                return_tensors='pt',
            )
        token_counts.append(input_ids.shape[1])
        tokenids.append(input_ids[0][sample_pos].clone().numpy())

        params = {
        'model': model,
        'input_ids': input_ids,
        'device': device,
        'sample_pos': sample_pos,
        }
        
        activation = activation_extract_fn(**params) # l s h*d
        all_activations.append(activation.permute(1,0,2).clone())
        del activation
        torch.cuda.empty_cache()

    all_activations = torch.cat(all_activations, dim=0)  
    all_labels = torch.tensor(all_labels)
    print(f"The activation set shapes as {all_activations.shape}")
    meta_info = {
                "tokenids": tokenids,
                "all_answer_spans": all_answer_spans,
                "sample_positions": sample_positions,
                "token_counts":token_counts,
            }
    print(f'token_counts is {token_counts}')
    return all_activations, all_labels, meta_info



def get_single_prompt_lt_activations(prompts, model, tokenizer, seed, device, chat=False):
    all_activations =[]
    print("Getting activations")
    activation_extract_fn = get_llama_module_activations
    
    for idx, prompt in enumerate(tqdm(prompts, total=len(prompts), desc=f'get_single_prompt_lt_activations')):

        if not chat:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        else:
            input_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                return_tensors='pt',
            )
        
        sample_pos = -1
        activation = activation_extract_fn(model, input_ids, device, sample_pos) # l s h*d

        all_activations.append(activation.permute(1,0,2).clone())

        del activation
        torch.cuda.empty_cache()

    all_activations = torch.cat(all_activations, dim=0)  
    print(f"The activation set shapes as {all_activations.shape}")
    return all_activations


def act_files(save_path, fold, num_fold, split):
    name_key = f'fold_{fold}_of_{num_fold}_{split}'
    all_activations_file = save_path / f'{name_key}_activations.pt'
    meta_info_file = save_path / f'{name_key}_meta_info.pkl'
    labels_file = save_path / f'{name_key}_labels.pt'  
    all_single_activations_file = save_path / f'{name_key}_single_activations.pt'
    return name_key, all_activations_file, meta_info_file, labels_file, all_single_activations_file


def check_all_file_exist(fold_ids, num_fold, path):
    for i in fold_ids:
        for split in ['all', 'train', 'val']:
            name_key, all_activations_file, meta_info_file, labels_file, all_single_activations_file = act_files(path, 
                                                                                                                 i, 
                                                                                                                 num_fold, 
                                                                                                                 split)

            if not os.path.exists(all_activations_file) or not os.path.exists(meta_info_file) or not os.path.exists(labels_file):
                return False
            if not os.path.exists(all_single_activations_file):
                return False
    return True


def activation_extract_pipeline(args, 
                                dataset, 
                                all_folds_train_indices, 
                                all_folds_val_indices, 
                                inner_states_path, 
                                act_key, 
                                device='cuda'):
    num_fold = args.num_fold
    seed = args.seed
    n_shot = args.n_shot

    save_path = validate_save_path(inner_states_path / act_key)

    if num_fold > 1:
        fold_ids = range(num_fold) 
    elif num_fold == 0:
        fold_ids = [0]
    else: # num_fold == 1
        fold_ids = []
    
    # create model
    model_name = HF_NAMES[args.model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    model_config = AutoConfig.from_pretrained(
        model_name,
        do_sample=False,
        local_files_only=True,
    )
    torch_dtype = getattr(model_config, 'torch_dtype', torch.float16)
    if torch_dtype != torch.bfloat16 or not cuda_supports_bfloat16():
        torch_dtype = torch.float16 
    torch.cuda.empty_cache()
    model = None
    
       
    for i in fold_ids:
        print(f'Processing activation extracting fold {i}')
        train_set_idxs = all_folds_train_indices[i]
        val_set_idxs = all_folds_val_indices[i]

        train_dataset = dataset.select(train_set_idxs)
        val_dataset = dataset.select(val_set_idxs)
        
        print(f'processing train dataset')
        train_prompts, train_labels = data_to_prompt_tqa_hf(train_dataset, chat=args.apply_chat_template, mc_key=args.mc_key)
        print(f'processing val dataset')
        val_prompts, val_labels = data_to_prompt_tqa_hf(val_dataset, chat=args.apply_chat_template, mc_key=args.mc_key)

        all_prompts = train_prompts + val_prompts
        all_labels = train_labels + val_labels

        working_list = [('all', all_prompts, all_labels), 
                        ('train', train_prompts, train_labels), 
                        ('val', val_prompts, val_labels)] if args.use_split_data else [('all', all_prompts, all_labels)]
        for split, raw_prompts, raw_labels in working_list:
            name_key, all_activations_file, meta_info_file, labels_file, all_single_activations_file = act_files(save_path, 
                                                                                                                 i, 
                                                                                                                 num_fold, 
                                                                                                                 split)
            
        
            grouped_prompts, shot_nums, grouped_indices, grouped_labels = make_many_shot_prompt(raw_prompts, 
                                                                                                raw_labels,
                                                                                                n_shot, 
                                                                                                seed, 
                                                                                                args.do_sample, 
                                                                                                save_path, 
                                                                                                name_key,
                                                                                                chat=args.apply_chat_template)
            # feature
            print(f'getting many_shot activations ... ', flush=True)
            if not os.path.exists(all_activations_file) or not os.path.exists(meta_info_file) or not os.path.exists(labels_file):
                
                if model is None:
                    model = MODELCLASS[args.model_name].from_pretrained(model_name, 
                                            config=model_config, 
                                            low_cpu_mem_usage = True, 
                                            torch_dtype=torch_dtype, 
                                            device_map="auto", 
                                            local_files_only=True)
                all_activations, act_labels, meta_info = get_multi_shot_activations(grouped_prompts, 
                                                                                    grouped_labels,
                                                                                    model, 
                                                                                    tokenizer, 
                                                                                    seed, 
                                                                                    device, 
                                                                                    grouped_indices, 
                                                                                    raw_prompts, 
                                                                                    tag=f'{act_key}_{name_key}',
                                                                                    chat=args.apply_chat_template,
                                                                                    sample_times=args.sample_times,
                                                                                    last_token=args.last_token)
                torch.save(all_activations, all_activations_file, )
                torch.save(act_labels, labels_file, )
                with open(meta_info_file, "wb") as f:
                    pickle.dump(meta_info, f)
            else:
                meta_info = None
                act_labels = None
                all_activations = None
            
            
    chatflag = '_chat' if args.apply_chat_template else ''
    tuning_activations_file = inner_states_path / f"{args.model_name}_{args.dataset_name}{chatflag}_{args.feature_name}_tuning_activations.pt"
    
    if num_fold != 1:
        if not os.path.exists(tuning_activations_file):
            tuning_prompts = data_to_tuning_prompt_tqa_hf(dataset, seed, chat=args.apply_chat_template, mc_key=args.mc_key)
            if model is None:
                    model = MODELCLASS[args.model_name].from_pretrained(model_name, 
                                            config=model_config, 
                                            low_cpu_mem_usage = True, 
                                            torch_dtype=torch_dtype, 
                                            device_map="auto", 
                                            local_files_only=True)
            tuning_activations = get_single_prompt_lt_activations(tuning_prompts, 
                                                                    model, 
                                                                    tokenizer, 
                                                                    seed, 
                                                                    device,
                                                                    chat=args.apply_chat_template)
            torch.save(tuning_activations, tuning_activations_file, )
        else:
            tuning_activations = torch.load(tuning_activations_file)
    else:
        tuning_activations = None

            
    def activations_generator():
        if num_fold > 1:
            fold_ids = range(num_fold) 
        elif num_fold == 0:# all for intervene, no for test
            fold_ids = [0]
        else: # num_fold == 1 or < 0  all for test, no intervene
            fold_ids = []
        
        for i in fold_ids:
            final_acts = []
            final_labels = []
            final_metas = []
            
            for split in ['all', 'train', 'val']:
                if not args.use_split_data and split != 'all':
                    final_acts.append([])
                    final_metas.append([])
                    final_labels.append([])
                    continue

                _, all_activations_file, meta_info_file, labels_file, all_single_activations_file = act_files(save_path, 
                                                                                                              i, 
                                                                                                              num_fold, 
                                                                                                              split)
                all_activations = torch.load(all_activations_file)

                final_acts.append(all_activations)

                with open(meta_info_file, "rb") as f:
                    meta_info = pickle.load(f)
                final_metas.append(meta_info)

                labels = torch.load(labels_file)
                final_labels.append(labels)
            yield tuple(final_acts), tuple(final_labels), tuple(final_metas) # (all, train, val)
    del model
    torch.cuda.empty_cache()

    return activations_generator, tuning_activations