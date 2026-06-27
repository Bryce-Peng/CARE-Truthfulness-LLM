from datasets import load_dataset, Dataset, Features, Value, Sequence
import random
import json
import os
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Union, List, Tuple
from utils import save_json, load_json, get_shuffled, np_choice, validate_save_path, QA_SEP, DEMO_SEP
import pickle

def load_tqa_hf(mc_path, gen_path, remoter_version=None):  
    column_map = {
        'type': 'Type',
        'category': 'Category',
        'question': 'Question',
        'best_answer': 'Best Answer',
        'best_incorrect_answer': 'Best Incorrect Answer', 
        'correct_answers': 'Correct Answers',
        'incorrect_answers': 'Incorrect Answers',
        'source': 'Source'
    }
    dataset_gen = load_dataset(gen_path, "generation", revision=remoter_version)['validation']
    df = dataset_gen.to_pandas()
    df["question"] = df["question"].str.strip() 
    df = df.rename(columns=column_map, errors='ignore')
    
    dataset = load_dataset(mc_path, "multiple_choice", revision=remoter_version)['validation']
    golden_q_order = list(dataset["question"])
    
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)})).reset_index(drop=True)
    assert list(dataset['question']) == list(df["Question"])
    return dataset, df


def load_tqa_gh(mc_path, gen_path):
    features = Features({
        "question": Value("string"),
        "mc0_targets": {
            "choices": Sequence(Value("string")),
            "labels": Sequence(Value("int32"))
        },
        "mc1_targets": {
            "choices": Sequence(Value("string")),
            "labels": Sequence(Value("int32"))
        },
        "mc2_targets": {
            "choices": Sequence(Value("string")),
            "labels": Sequence(Value("int32"))
        }
    })
    
    with open(mc_path, 'r') as f:  
        raw_data = json.load(f)
    
    processed = []
    for item in raw_data:
        new_item = {"question": item["question"]}
        
        for mc_key in ["mc0_targets", "mc1_targets", "mc2_targets"]:
            if mc_key in item:
                sorted_items = list(item[mc_key].items())
                choices = [k for k, v in sorted_items]
                labels = [v for k, v in sorted_items]
                new_item[mc_key] = {
                    "choices": choices,
                    "labels": labels
                }
            else:
                new_item[mc_key] = {"choices": [], "labels": []}
        
        processed.append(new_item)

    dataset = Dataset.from_list(processed, features=features)
    golden_q_order = list(dataset["question"])

    df = pd.read_csv(gen_path)
    df["Question"] = df["Question"].str.strip() 
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)})).reset_index(drop=True)

    assert list(dataset['question']) == list(df["Question"])

    return dataset, df


def load_standard_qa(filepath):
    features = Features({
        "question": Value("string"),
        "mc2_targets": {
            "choices": Sequence(Value("string")),
            "labels": Sequence(Value("int32"))
        }
    })
    
    df = pd.read_csv(filepath)
    
    processed = []
    for _, item in df.iterrows():
        new_item = {"question": item["Question"]}
        
        for mc_key in ["mc2_targets"]:
            correct_choices = item["Correct Answers"].split("<MULTIPLE_ANS_SEP>")
            incorrect_choices = item["Incorrect Answers"].split("<MULTIPLE_ANS_SEP>")
            choices = correct_choices + incorrect_choices
            labels = [1 for c in correct_choices] + [0 for c in incorrect_choices]
            new_item[mc_key] = {
                "choices": choices,
                "labels": labels
            }
        
        processed.append(new_item)

    dataset = Dataset.from_list(processed, features=features)

    dataset_questions = list(dataset['question'])
    df_questions = list(df["Question"])
    
    if dataset_questions != df_questions:
        differences = find_question_order_diff(dataset_questions, df_questions)
        for diff in differences:
            print(f"Index {diff['index']}:")
            print(f"  dataset: {diff['dataset_question']}")
            print(f"  df     : {diff['df_question']}")
        raise ValueError("Orders differ")

    return dataset, df

def find_question_order_diff(dataset_questions, df_questions):
    differences = []
    
    for idx, (dq, dfq) in enumerate(zip(dataset_questions, df_questions)):
        if dq != dfq:
            differences.append({
                "index": idx,
                "dataset_question": dq,
                "df_question": dfq
            })
    
    return differences


def format_tqa(question, choice, connector=QA_SEP, chat=False):
    if not chat:
        return f"Q: {question}{connector} {choice}"
    else:
        conversation = [{"role": "user", "content": f"Q: {question}"},
                        {"role": "assistant", "content": f"A: {choice}"}]
        return conversation


def data_to_prompt_tqa_hf(dataset, chat=False, mc_key='mc2_targets'):
    
    prompts = []
    labels = []
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i][mc_key]['choices']
        choice_labels = dataset[i][mc_key]['labels']

        assert len(choices) == len(choice_labels), (len(choices), len(choice_labels))

        for j in range(len(choices)): 
            choice = choices[j]
            label = choice_labels[j]
            prompt = format_tqa(question, choice, chat=chat)
            prompts.append(prompt)
            labels.append(label)

    all_class_label = sorted(set(labels))
    for class_label in all_class_label:
        sample_count = len([label for label in labels if label == class_label])
        label_ratio = sample_count / len(labels)
        print(f"Label [{class_label}] samples: {sample_count} ({label_ratio:.2%})")

    return prompts, labels


def data_to_tuning_prompt_tqa_hf(dataset, seed, chat=False, mc_key='mc2_targets'):
    prompts = []
    rng = random.Random(seed)
    
    for i in range(len(dataset)):
        question = dataset[i]['question']
        choices = dataset[i][mc_key]['choices']
        rand_idx = rng.randint(0, len(dataset) - 1)
        rand_question = dataset[rand_idx]['question']

        for j in range(len(choices)): 
            choice = choices[j]
            prompt = f"Q: {question} A: {choice} Q: {rand_question}" if not chat else [
                    {"role": "user", "content": f"Q: {question}"},
                    {"role": "assistant", "content": f"A: {choice}"},
                    {"role": "user", "content": f"Q: {rand_question}"},
                ]
            prompts.append(prompt)

    return prompts


def convert_tqa_to_preference_dataset(dataset, key='mc2_targets'):
    datasets = {'train':{},'val':{}}
    model_type = os.getenv("MODEL_TYPE", "default")
    if "type-2" == model_type:
        prompt_template = "Q: {question} \nA:"
    else:
        prompt_template = "Q: {question}\nA:"

    for split in datasets:
        
        preference_dataset = {'prompt': [],'chosen':[], 'rejected': []}
        for entry in dataset[split]:
            prompt = prompt_template.format(question=entry['question'])
            choices = entry[key]['choices']
            labels = entry[key]['labels']

            entry_chosen = []
            entry_rejected = []
            for i in range(len(choices)):
                label = labels[i]
                if label ==1:
                    entry_chosen.append(entry[key]['choices'][i])
                else:
                    entry_rejected.append(entry[key]['choices'][i])
        
            if len(entry_chosen)!=len(entry_rejected):
                entry_chosen = entry_chosen[:min(len(entry_rejected),len(entry_chosen))]
                entry_rejected = entry_rejected[:len(entry_chosen)]

            prompts = [prompt for _ in range(len(entry_chosen))]
            preference_dataset['prompt'].extend(prompts)
            preference_dataset['chosen'].extend(entry_chosen)
            preference_dataset['rejected'].extend(entry_rejected)

        datasets[split] = Dataset.from_dict(preference_dataset)
    return datasets
    

script_dir = os.path.dirname(os.path.realpath(__file__))
TQA_PATH = f"{script_dir}/../TruthfulQA"
STD_PATH = f"{script_dir}/../standard_datasets"

DATASET_CONFIGS = {
    'tqa_hf':{
        'args' : {
            'mc_path': f"truthfulqa/truthful_qa",
            'gen_path': f"truthfulqa/truthful_qa",
            },
        'load_fn' : load_tqa_hf,
    },
    'tqa_gh_v0':{
        'args' : {
            'mc_path': f"{TQA_PATH}/data/v0/mc_task.json",
            'gen_path': f"{TQA_PATH}/data/v0/TruthfulQA.csv",
            },
        'load_fn' : load_tqa_gh,
    },
    'tqa_gh_v1':{
        'args' : {
            'mc_path': f"{TQA_PATH}/data/v1/mc_task.json",
            'gen_path': f"{TQA_PATH}/data/v1/TruthfulQA.csv",
            },
        'load_fn' : load_tqa_gh,
    },
    'tqa_gh_v2':{
        'args' : {
            'mc_path': f"{TQA_PATH}/data/mc_task.json",
            'gen_path': f"{TQA_PATH}/TruthfulQA.csv",
            },
        'load_fn' : load_tqa_gh,
    },
    'ai2_arc_c_test':{
        'args' : {
            'filepath': f"{STD_PATH}/ai2_arc_c_test.csv",
            },
        'load_fn' : load_standard_qa,
    },
    'iti_nq_open_val':{
        'args' : {
            'filepath': f"{STD_PATH}/iti_nq_open_val.csv",
            },
        'load_fn' : load_standard_qa,
    },
    'iti_trivia_qa_val':{
        'args' : {
            'filepath': f"{STD_PATH}/iti_trivia_qa_val.csv",
            },
        'load_fn' : load_standard_qa,
    },
    'openbookqa_test':{
        'args' : {
            'filepath': f"{STD_PATH}/openbookqa_test.csv",
            },
        'load_fn' : load_standard_qa,
    },
    'halueval':{
        'args' : {
            'filepath': f"{STD_PATH}/HaluEval_qa_1000.csv",
            },
        'load_fn' : load_standard_qa,
    },
}


def k_fold_split(
                num_fold: int,
                dataframe: pd.DataFrame,
                data_ratio: float,
                seed: int,
                data_key: str,
                save_path: Union[str, Path],
            ) -> List[np.ndarray]:
    
    save_path = validate_save_path(save_path)
    fold_indices_file = save_path / f"{data_key}_{num_fold}_fold_indices.yaml"
    
    if Path(fold_indices_file).exists():
        print(f"load existing indices for k-fold split: {fold_indices_file.name}")
        with open(fold_indices_file, 'r') as f:
            fold_indices = [np.array(indices) for indices in yaml.safe_load(f)]
    else:
        total_idxs = np_choice(len(dataframe), int(data_ratio * len(dataframe)), replace=False, seed=seed)
        num_fold = num_fold if num_fold !=0 else 1 
        fold_indices = np.array_split(total_idxs, num_fold)

        Path(fold_indices_file).parent.mkdir(parents=True, exist_ok=True)
        with open(fold_indices_file, 'w') as f:
            yaml.dump([indices.tolist() for indices in fold_indices], f)

    return fold_indices


def train_val_test_split_all_folds( num_fold: int,
                                    dataframe: pd.DataFrame,
                                    fold_indices: List[np.ndarray],
                                    val_ratio: float,
                                    seed: int,
                                    data_key: str,
                                    save_path: Union[str, Path]
                                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    save_path = validate_save_path(save_path)

    all_train_indices = []
    all_val_indices = []
    all_test_files = []

    if num_fold == 1:
        test_set_indices = fold_indices[0]
        test_file = save_path / f"{data_key}_fold_{0}_of_{num_fold}_test_split.csv"
        test_split = dataframe.iloc[test_set_indices]
        if not Path(test_file).exists():
            test_split.to_csv(test_file, index=False)
        all_test_files.append(test_file)
        print(f'Only 1-fold, which is all for testing.')
        return [], [], all_test_files

    if num_fold == 0:
        # pick a val set using numpy
        train_val_indices = np.concatenate([fold_indices[0]])
        train_set_indices = np_choice(train_val_indices, size=int(len(train_val_indices)*(1-val_ratio)), replace=False, seed=seed)
        val_set_indices = np.array([x for x in train_val_indices if x not in train_set_indices])

        train_file = save_path / f"{data_key}_fold_{0}_of_{num_fold}_train_split.csv"
        val_file = save_path / f"{data_key}_fold_{0}_of_{num_fold}_val_split.csv"

        train_split = dataframe.iloc[train_set_indices]
        if not Path(train_file).exists():
            train_split.to_csv(train_file, index=False)

        val_split = dataframe.iloc[val_set_indices]
        if not Path(val_file).exists():
            val_split.to_csv(val_file, index=False)


        all_train_indices.append(train_set_indices)
        all_val_indices.append(val_set_indices)
        return all_train_indices, all_val_indices, []
        

    for i in range(num_fold):
        
        print(f"Processing splits fold {i}")
        test_set_indices = fold_indices[i]
        test_file = save_path / f"{data_key}_fold_{i}_of_{num_fold}_test_split.csv"
        train_file = save_path / f"{data_key}_fold_{i}_of_{num_fold}_train_split.csv"
        val_file = save_path / f"{data_key}_fold_{i}_of_{num_fold}_val_split.csv"
        test_split = dataframe.iloc[test_set_indices]
        if not Path(test_file).exists():
            test_split.to_csv(test_file, index=False)

        # pick a val set using numpy
        train_val_indices = np.concatenate([fold_indices[j] for j in range(num_fold) if j != i])
        train_set_indices = np_choice(train_val_indices, size=int(len(train_val_indices)*(1-val_ratio)), replace=False, seed=seed)
        val_set_indices = np.array([x for x in train_val_indices if x not in train_set_indices])

        train_split = dataframe.iloc[train_set_indices]
        if not Path(train_file).exists():
            train_split.to_csv(train_file, index=False)

        val_split = dataframe.iloc[val_set_indices]
        if not Path(val_file).exists():
            val_split.to_csv(val_file, index=False)

        all_train_indices.append(train_set_indices)
        all_val_indices.append(val_set_indices)
        all_test_files.append(test_file)
        
    return all_train_indices, all_val_indices, all_test_files


def sampled_group(indices, n, seed):
    i = 0
    indices = get_shuffled(indices, seed)
    all_grouped_indices = []
    rng = random.Random(seed)

    while i < len(indices):
        remaining = indices[:i] + indices[i+1:]
        grouped_indices = rng.sample(remaining, n) # non-replace
        grouped_indices = grouped_indices + [indices[i]]
        all_grouped_indices.append(grouped_indices)
        i += 1
    return all_grouped_indices


def simple_group(indices, n, seed):
    i = 0
    indices = get_shuffled(indices, seed)
    all_grouped_indices = []
    while i < len(indices):
        remaining = len(indices) - i
        if remaining < n:
            grouped_indices = indices[i:]
            all_grouped_indices.append(grouped_indices)
            print(f"grouped the last {remaining} samples!")
            break
        else:
            grouped_indices = indices[i:i+n+1]
            all_grouped_indices.append(grouped_indices)
            i += n + 1
    return all_grouped_indices


def make_many_shot_prompt(prompts, labels, n_shot, seed, do_sample, save_path, name_key, chat=False):
    save_path = validate_save_path(save_path)
    grouped_indices_file = save_path / f'{name_key}_grouped_indices.json'
    grouped_labels_file = save_path / f'{name_key}_grouped_labels.json'
    if Path(grouped_indices_file).exists():
        all_grouped_indices = load_json(grouped_indices_file)
        grouped_labels = load_json(grouped_labels_file)
        all_grouped_indices = [[int(idx) for idx in group] for group in all_grouped_indices]
    else:
        group_fn = sampled_group if do_sample else simple_group
        all_indices = list(range(len(prompts)))
        all_class_label = sorted(set(labels))
        all_grouped_indices = []
        grouped_labels = []
        for class_label in all_class_label:
            indices = [i for i, l in zip(all_indices, labels) if l == class_label]
            grouped_indices = group_fn(indices, n_shot, seed)
            all_grouped_indices.extend(grouped_indices)
            grouped_labels.extend([class_label] * len(grouped_indices))

        save_json(all_grouped_indices, grouped_indices_file)
        save_json(grouped_labels, grouped_labels_file)

    shot_nums = [len(group) for group in all_grouped_indices]
    if not chat:
        grouped_prompts = [DEMO_SEP.join([prompts[idx] for idx in group]) 
                           for group in all_grouped_indices]
    else:
        grouped_prompts = [
            [message for idx in group for message in prompts[idx]]
            for group in all_grouped_indices
        ]
    return grouped_prompts, shot_nums, all_grouped_indices, grouped_labels


def make_many_shot_prefix(args, save_path, data_key, dataset, all_folds_train_indices, all_folds_val_indices):
    mc_key = args.mc_key
    num_fold = args.num_fold
    save_path = validate_save_path(save_path)
    all_folds_prefix_file = save_path / f'{data_key}_{num_fold}_folds_prefix.pkl'
    if all_folds_prefix_file.exists():
        with open(all_folds_prefix_file, 'rb') as f:
            return pickle.load(f)
    else:
        all_folds_prefix = []
        if num_fold > 1:
            fold_ids = range(num_fold) 
        elif num_fold == 0:
            fold_ids = [0]
        for i in fold_ids:
            print(f'Processing activation extracting fold {i}')
            train_set_idxs = all_folds_train_indices[i]
            val_set_idxs = all_folds_val_indices[i]
            dev_set_idxs = np.concatenate([train_set_idxs,val_set_idxs])
            demo_idxs = np_choice(dev_set_idxs, size=args.n_shot, replace=False, seed=args.seed)
            demo_dataset = dataset.select(demo_idxs)

            prompts = []
            for i in range(len(demo_dataset)):
                question = demo_dataset[i]['question']
                choices = demo_dataset[i][mc_key]['choices']
                choice_labels = demo_dataset[i][mc_key]['labels']

                assert len(choices) == len(choice_labels), (len(choices), len(choice_labels))

                for j in range(len(choices)): 
                    choice = choices[j]
                    label = choice_labels[j]
                    if label == 1:
                        prompt = format_tqa(question, choice, chat=args.apply_chat_template) # best answer is the first correct answer
                        prompts.append(prompt)
                        break
            if not args.apply_chat_template:
                many_shot_prefix = DEMO_SEP.join(prompts) + DEMO_SEP
            else:
                many_shot_prefix = [message for message in prompts]
            all_folds_prefix.append(many_shot_prefix)
        with open(all_folds_prefix_file, "wb") as f:
            pickle.dump(all_folds_prefix, f)
    return all_folds_prefix


def data_load_split_pipeline(args, 
                             splits_path, 
                             data_key, 
                             dataset_config):
    num_fold = args.num_fold
    data_ratio = args.data_ratio
    val_ratio = args.val_ratio
    seed = args.seed

    load_fn = dataset_config['load_fn']
    dataset, dataframe = load_fn(**dataset_config['args'])

    fold_indices = k_fold_split(num_fold,
                                dataframe,
                                data_ratio,
                                seed,
                                data_key,
                                splits_path,)
    
    all_folds_train_indices, all_folds_val_indices, all_test_files = train_val_test_split_all_folds(
                                    num_fold,
                                    dataframe,
                                    fold_indices,
                                    val_ratio,
                                    seed,
                                    data_key,
                                    splits_path)

    return dataset, all_folds_train_indices, all_folds_val_indices, all_test_files