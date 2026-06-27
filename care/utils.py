from numpy.random import RandomState
import random
import json
import os
from pathlib import Path
from typing import Union, Any
import torch
import numpy as np
from models import (LlamaForCausalLM, 
                    GlmForCausalLM, 
                    GemmaForCausalLM, 
                    GPTJForCausalLM, 
                    MistralForCausalLM, 
                    Phi3ForCausalLM,
                    PhiForCausalLM,
                    Qwen2ForCausalLM)


class SeparatorConfig:
    def __init__(self, model_type: str = "default"):
        self._model_type = model_type.lower()
        self._separators = self._load_separators()

    def _load_separators(self) -> dict:
        return {
            "default": {"QA_SEP": "\nA:", "DEMO_SEP": "\n\n"},
            "type-2": {"QA_SEP": " \nA:", "DEMO_SEP": " \n\n"},
        }

    @property
    def QA_SEP(self) -> str:
        return self._separators.get(self._model_type, self._separators["default"])["QA_SEP"]

    @property
    def DEMO_SEP(self) -> str:
        return self._separators.get(self._model_type, self._separators["default"])["DEMO_SEP"]

model_type = os.getenv("MODEL_TYPE", "default")
print(f'use {model_type} tokenizer!')
separator_config = SeparatorConfig(model_type)

QA_SEP = separator_config.QA_SEP  # Separator For Q and A in QA pairs
DEMO_SEP = separator_config.DEMO_SEP  # Separator For demonstrations in many shot prompts


HF_NAMES = {
    'llama_7B': 'huggyllama/llama-7b',
    'alpaca_7B': 'PKU-Alignment/alpaca-7b-reproduced', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    'llama31_8B': 'meta-llama/Llama-3.1-8B', 
    'llama31_inst_8B': 'meta-llama/Llama-3.1-8B-Instruct', 
    'glm-4-9b-chat': "THUDM/glm-4-9b-chat-hf",
    'gpt-j-6b': "EleutherAI/gpt-j-6b",  
    'Qwen2.5-7B-Instruct': "Qwen/Qwen2.5-7B-Instruct", 
    'Qwen2-7B-Instruct': "Qwen/Qwen2-7B-Instruct", 
    'Ministral-8B-Instruct': "mistralai/Ministral-8B-Instruct-2410", 
    'Phi-4-mini-instruct': "microsoft/Phi-4-mini-instruct", 
    'Phi-3.5-mini-instruct': "microsoft/Phi-3.5-mini-instruct",  
    'phi-1_5': "microsoft/phi-1_5", 
    'gemma-1.1-7b-it': "google/gemma-1.1-7b-it", 
}

MODELCLASS = {
    'llama_7B': LlamaForCausalLM,
    'alpaca_7B': LlamaForCausalLM,
    'vicuna_7B': LlamaForCausalLM,
    'llama2_7B': LlamaForCausalLM,
    'llama2_chat_7B': LlamaForCausalLM,
    'llama2_chat_13B': LlamaForCausalLM,
    'llama2_chat_70B': LlamaForCausalLM,
    'llama31_8B': LlamaForCausalLM,
    'llama31_inst_8B': LlamaForCausalLM,
    'glm-4-9b-chat': GlmForCausalLM,
    'gpt-j-6b': GPTJForCausalLM,
    'Qwen2.5-7B-Instruct': Qwen2ForCausalLM,
    'Qwen2-7B-Instruct': Qwen2ForCausalLM,
    'Ministral-8B-Instruct': MistralForCausalLM,
    'Phi-4-mini-instruct': Phi3ForCausalLM,
    'Phi-3.5-mini-instruct': Phi3ForCausalLM,
    'phi-1_5': PhiForCausalLM,
    'gemma-1.1-7b-it': GemmaForCausalLM,
}

def cuda_supports_bfloat16():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 8


def save_json(data: list, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename: str) -> list:
    with open(filename, 'r') as f:
        return json.load(f)


def sampled_int(start, end, seed):
    rng = random.Random(seed)
    return rng.randint(start, end) # including both end points


def sample_elements(pos_choices, count, seed):
    rng = random.Random(seed)

    return rng.sample(pos_choices, count) 


def get_shuffled(data, seed):
    rng = random.Random(seed)
    rng.shuffle(data)
    return data


def np_choice(array, size, replace=False, seed=42, ):
    data_rng = RandomState(seed=seed)
    return data_rng.choice(array, size, replace=replace)


def validate_save_path(save_path: Union[str, Path]) -> Path:
    path = Path(save_path).absolute()
    
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f"path exists but not a dir: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
        print(f"mkdir: {path}")
    
    return path

def check_device(model):
    for name, param in model.named_parameters():
        if param.device.type == 'meta' or param.device.type == 'cpu':
            print(f"Check device: Layer {name} is on {param.device.type} device!")
            return
    print(f"Check device: ok! no params on meta or cpu")


def get_config_key(args):
    base_key = f'{args.model_name}_{args.dataset_name}_seed_{args.seed}'
    if args.mc_key != "mc2_targets":
        base_key += f'_{args.mc_key[:3]}'

    data_key = f'{base_key}'
    if args.data_ratio != float(1.0):
        data_key += "_ratio_{:.2f}".format(args.data_ratio)
    if args.val_ratio != float(0.2):
        data_key += "_val_{:.2f}".format(args.val_ratio)

    act_key = f'{data_key}'
    if args.feature_name != 'head_out':
        act_key += args.feature_name
    if args.n_shot > 0:
        act_key += f"_{args.n_shot}_shot{f'_sample' if args.do_sample else ''}"
    if args.apply_chat_template:
        act_key += f'_chat'
    if args.sample_times > 1:
        act_key += f'_{args.sample_times}tks'
    if args.last_token:
        act_key += f'_lt'

    vene_key = f'{act_key}'
    if args.K!=0:
        vene_key += f'_top_{args.K}'
    elif args.use_prefix:
        vene_key += f'_fsp'
    else:
        vene_key += '_no_intervene'
    if args.edit_module != 'head_out':
        vene_key += f'_{args.edit_module}'
    if args.tune_alpha:
        vene_key += f'_tune'
        vene_key += f'_{args.dpo_beta:.2f}'
    elif args.alpha != 0:
        vene_key += f'_A_{args.alpha}'
    if args.use_random_dir:
        vene_key += '_rand'
    elif args.use_normalized_center_of_mass:
        vene_key += '_ncom'

    if args.probe_class is not None:
        vene_key += f'_{args.probe_class}'
    if args.adaptive:
        vene_key += f'_adp'
        if args.temperature != float(1.0):
            temperature = "{:.1f}".format(args.temperature)
            vene_key += f'_T_{temperature}'
    if args.use_dual_dirs:
        vene_key += f'_dual'

    return base_key, data_key, act_key, vene_key


def setseed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_path(run_name):
    answer_path = validate_save_path(f'./{run_name}/answer_dump')
    summary_path = validate_save_path(f'./{run_name}/summary_dump')
    return answer_path, summary_path


INNER_STATES_PATH = validate_save_path(f'../inner_states')
SPLITS_PATH = validate_save_path(f'../datasets/splits')
