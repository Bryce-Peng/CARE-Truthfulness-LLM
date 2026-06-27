import sys
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from probe import CenterOfMeanClassifier
from tqdm import tqdm
from transformers import AutoConfig
import torch
from utils import HF_NAMES
from feature_extractor import activation_extract_pipeline
import math


def get_com_directions(num_layers, num_heads, activations, labels): 

    ndif_directions = []
    pos_directions = []
    opposite_neg_directions = []
    
    usable_labels = labels
    with torch.no_grad():
        for layer in range(num_layers): 
            usable_head_wise_activations = activations[:,layer,:] # b, h*d
            true_mass_mean = torch.mean(usable_head_wise_activations[usable_labels == 1], dim=0) # hd
            false_mass_mean = torch.mean(usable_head_wise_activations[usable_labels == 0], dim=0)
            ndif_directions.append((true_mass_mean / torch.norm(true_mass_mean, p=2))  - (false_mass_mean / torch.norm(false_mass_mean, p=2)))
            pos_directions.append(true_mass_mean)
            opposite_neg_directions.append(-false_mass_mean)
        pos_directions = torch.stack(pos_directions)
        opposite_neg_directions = torch.stack(opposite_neg_directions)
        ndif_directions = torch.stack(ndif_directions) # l, h*d
        pos_directions = pos_directions.reshape(num_layers*num_heads, -1)
        opposite_neg_directions = opposite_neg_directions.reshape(num_layers*num_heads, -1)
        ndif_directions = ndif_directions.reshape(num_layers*num_heads, -1)

    return pos_directions, opposite_neg_directions, ndif_directions


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def get_model(probe_type, **kwargs):
    model_map = {
        'lr': LogisticRegression,
        'ncomc': CenterOfMeanClassifier,
    }
    model_class = model_map.get(probe_type)
    if model_class is None:
        raise ValueError(f"Unknown algorithm: {probe_type}")
    if probe_type == 'ncomc':
        return model_class(normalize=True, **kwargs)
    
    return model_class(**kwargs)

def train_probes(seed, 
                 train_activations, 
                 train_labels, 
                 val_activations, 
                 val_labels, 
                 num_layers, 
                 num_heads, 
                 probe_class, 
                 ):
    
    all_head_accs = []
    probes = []

    all_X_train = train_activations.reshape(*train_activations.shape[:2], num_heads, -1)
    all_X_val = val_activations.reshape(*val_activations.shape[:2], num_heads, -1)

    y_train = train_labels
    y_val = val_labels

    for layer in tqdm(range(num_layers), desc=f'train probes'): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:].float().numpy()
            X_val = all_X_val[:,layer,head,:].float().numpy()
            
            clf = get_model(probe_class, 
                            random_state=seed, 
                            max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np


def get_top_heads(train_activations, 
                  train_labels, 
                  val_activations, 
                  val_labels, 
                  num_layers, 
                  num_heads, 
                  seed, 
                  num_to_intervene, 
                  within_layers=None, 
                  use_random_dir=False, 
                  probe_class=None, 
                  ):

    probes, all_head_accs_np = train_probes(seed, 
                                            train_activations, 
                                            train_labels, 
                                            val_activations, 
                                            val_labels, 
                                            num_layers, 
                                            num_heads, 
                                            probe_class, 
                                            )
    all_head_scores = all_head_accs_np.copy()

    np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    print("All Head Accuracies:")
    print(np.around(all_head_accs_np, 4))
    
    top_heads = []
    if num_to_intervene == -1: # -1 => all
        num_to_intervene = num_heads*num_layers
    
    top_accs = np.argsort(all_head_scores)

    top_accs = top_accs[::-1]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, 
                                       num_heads*num_layers, 
                                       replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs]
    if within_layers is not None:
        top_heads = [(l, h) for l, h in top_heads if l in within_layers]
        num_to_intervene = num_heads * len(within_layers) if num_to_intervene == -1 else num_to_intervene

    if num_to_intervene > 0 and num_to_intervene < 1:
        
        num_to_intervene = int(num_heads * num_layers * num_to_intervene)

    assert num_to_intervene > 0, f"num_to_intervene is {num_to_intervene} must > 0"
    top_heads = top_heads[:num_to_intervene]
    
    print(f"Selecting {num_to_intervene} Heads and Corresponding Accuracies:")
    top_accs_val = [f"{all_head_accs_np[layer, head]:.4f}" for (layer, head) in top_heads]
    print(f'top_heads={[(int(a), int(b)) for (a, b) in top_heads]}')
    print(f'top_accs_val={top_accs_val}')
    
    return top_heads, probes


def get_interventions_dict(args,
                           top_heads, 
                           probes, 
                           tuning_activations, 
                           num_heads, 
                           use_center_of_mass, 
                           use_random_dir, 
                           com_directions, 
                           edit_module="head_out", 
                           adaptive=False, 
                           context_clfs=None,
                           ): 
    head_dim = tuning_activations.shape[-1] # b l h d
    model_config = AutoConfig.from_pretrained(HF_NAMES[args.model_name], 
                                              do_sample=False,
                                              local_files_only=True,)
    decoders='model.layers'
    model_type = getattr(model_config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'

    interventions = {}
    layer_repr = {}
    mha_layers = list(set([layer for layer, head in top_heads if head != -1]))
    module = f'self_attn.{edit_module}' if model_type != 'gptj' else f'attn.{edit_module}'
    for layer, head in top_heads:
        interventions[f"{decoders}.{layer}.{module}"] = []
        layer_repr[f"{layer}.directions"] = []

    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)] 
        elif use_random_dir: 
            direction = torch.randn(head_dim)
        else: 
            direction = torch.from_numpy(probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_).to(tuning_activations.dtype)
    
        layer_repr[f"{layer}.directions"].append(direction)

    for layer in mha_layers: 
        layer_direction = torch.cat(layer_repr[f"{layer}.directions"]) # size = (128*heads,)
        norm_coef = math.sqrt(len(layer_repr[f"{layer}.directions"])) / torch.norm(layer_direction, p=2)
        layer_repr[f"{layer}.norm_coef"] = norm_coef

    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = torch.randn(head_dim)
        else: 
            direction = torch.from_numpy(probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_).to(tuning_activations.dtype)

        direction = direction * layer_repr[f"{layer}.norm_coef"]
        
        activations = tuning_activations[:,layer,head,:]# batch x dim
        proj_vals = activations @ direction.T / torch.norm(direction, p=2)
        proj_val_std = torch.clip(torch.std(proj_vals),min=1e-8)
        if adaptive:
            activations = tuning_activations[:,layer,head,:]# batch x dim
            ca_direction = torch.from_numpy(context_clfs[layer_head_to_flattened_idx(layer, head, num_heads)].coef_).to(dtype=activations.dtype)
            ca_proj_vals = activations @ ca_direction.T / torch.norm(ca_direction, p=2)
            ca_proj_val_std = torch.clip(torch.std(ca_proj_vals),min=1e-8)
            interventions[f"{decoders}.{layer}.{module}"].append((head, 
                                                                    direction.squeeze(), 
                                                                    proj_val_std, 
                                                                    context_clfs[layer_head_to_flattened_idx(layer, head, num_heads)],
                                                                    ca_proj_val_std))
        else:
            interventions[f"{decoders}.{layer}.{module}"].append((head, 
                                                                    direction.squeeze(), 
                                                                    proj_val_std))
    for layer, head in top_heads: 
        interventions[f"{decoders}.{layer}.{module}"] = sorted(interventions[f"{decoders}.{layer}.{module}"], 
                                                                    key = lambda x: x[0])
            
    
    print('Show <proj_val_std>:')
    all_proj = []
    for k, v in interventions.items():
        if isinstance(v, list):
            proj_val_std_list = [float(tri[2]) for tri in v]
            all_proj += proj_val_std_list
            print(f"{k}: {[round(std, 5) for std in proj_val_std_list]}")
    print("mean of proj_val_std: ", np.mean(all_proj))

    if adaptive:
        print('Show <ca_proj_val_std>:')
        all_ca_proj = []
        for k, v in interventions.items():
            if isinstance(v, list):
                ca_proj_val_std_list = [float(tri[4]) for tri in v]
                all_ca_proj += ca_proj_val_std_list
                print(f"{k}: {[round(std, 5) for std in ca_proj_val_std_list]}")
        print("mean of ca_proj_val_std: ", np.mean(all_ca_proj))

    for layer in mha_layers: 
        head_interventions = []
        if adaptive:
            layer_dir = torch.zeros((num_heads,head_dim))
            layer_adaptive_dir = torch.zeros((num_heads,head_dim))
            for head, dir, std, clf, ca_std in interventions[f"{decoders}.{layer}.{module}"]:
                denominator = std * args.temperature + 1e-8
                adaptive_dir = - F.normalize(dir, p=2, dim=-1)  / denominator
                direction = args.alpha * std * dir
                layer_dir[head, :] = direction
                layer_adaptive_dir[head, :] = adaptive_dir
            interventions[f"{decoders}.{layer}.{module}"] = (layer_dir.unsqueeze(0).unsqueeze(0), 
                                                                layer_adaptive_dir.unsqueeze(0).unsqueeze(0))
        else:
            for head, dir, std in interventions[f"{decoders}.{layer}.{module}"]:
                direction = args.alpha * std * dir
                head_interventions.append((head, direction))
            interventions[f"{decoders}.{layer}.{module}"] = head_interventions

    return interventions


def print_distribution(labels, tag):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_ratios = label_counts / len(labels)
    for label, count, ratio in zip(unique_labels, label_counts, label_ratios):
        print(f"{tag} label {label}: count={count}, ratio={ratio:.2%}")


def operator_pipeline(args,
                      fold,
                      act_key,
                      num_layers, 
                      num_heads, 
                      activations, 
                      labels, 
                      num_to_intervene, 
                      tuning_activations, 
                      device,
                      ):
    
    all_acts, train_acts, val_acts = activations
    all_labels, train_labels, val_labels = labels
    
    for l, tag in zip(labels, ('all activations', 'train activations', 'val activations')):
        print_distribution(l, tag)

    pos_directions, opposite_neg_directions, ndif_directions = get_com_directions(num_layers, 
                                                                                                  num_heads, 
                                                                                                  all_acts, 
                                                                                                  all_labels)
    
    use_center_of_mass = True
    if args.use_normalized_center_of_mass:
        com_directions = ndif_directions
    else:
        use_center_of_mass = False
        com_directions = None

    if args.probe_class == 'ncomc': 
        train_acts = all_acts
        train_labels = all_labels
        val_acts = all_acts
        val_labels = all_labels

    if not args.tune_alpha:
        top_heads, probes = get_top_heads(train_acts, 
                                        train_labels, 
                                        val_acts, 
                                        val_labels, 
                                        num_layers, 
                                        num_heads, 
                                        args.seed, 
                                        num_to_intervene, 
                                        within_layers=args.within_layers, 
                                        use_random_dir=args.use_random_dir, 
                                        probe_class=args.probe_class, 
                                        )
    if args.tune_alpha:
        interventions = {}
    else:
        interventions = get_interventions_dict(args,
                                               top_heads, 
                                                probes, 
                                                tuning_activations, 
                                                num_heads, 
                                                use_center_of_mass, 
                                                args.use_random_dir, 
                                                com_directions, 
                                                edit_module=args.edit_module, 
                                                adaptive=args.adaptive, 
                                                context_clfs=probes,
                                                )
        
    if args.tune_alpha:
        top_heads = [(l,h) for l in range(num_layers) for h in range(num_heads)]
        return interventions, (pos_directions, opposite_neg_directions), (top_heads, top_heads)
    return interventions, com_directions, top_heads


def data2vene_pipeline(args, 
                       dataset, 
                       all_folds_train_indices, 
                       all_folds_val_indices, 
                       num_layers,
                       num_heads,
                       num_to_intervene,
                       inner_states_path, 
                       act_key, 
                       vene_key,
                       device='cuda'):
    if args.num_fold == 1:
        return [{}], [{}], [{}], [{}]
    if args.num_fold != 0 and num_to_intervene==0:
        return [{}]*args.num_fold, [{}]*args.num_fold, [{}]*args.num_fold, [{}]*args.num_fold
    
    activations_generator, tuning_activations = activation_extract_pipeline(args,
                                                                            dataset, 
                                                                            all_folds_train_indices, 
                                                                            all_folds_val_indices, 
                                                                            inner_states_path, 
                                                                            act_key,
                                                                            device)
    if tuning_activations is not None:
        tuning_activations = tuning_activations.reshape(*tuning_activations.shape[:2], num_heads, -1) # b l (h d) -> b l h d

    all_folds_interventions = []
    all_folds_directions = []
    all_folds_top_heads = []
    for i, (activations, labels, meta_infos) in enumerate(tqdm(activations_generator(), 
                                                            total=args.num_fold, 
                                                            desc=f'k-fold interventions')):
        print(f'Processing operator fold {i}')
        interventions, com_directions, top_heads = operator_pipeline(args,
                                                                    i,
                                                                    act_key,
                                                                    num_layers, 
                                                                    num_heads, 
                                                                    activations, 
                                                                    labels, 
                                                                    num_to_intervene, 
                                                                    tuning_activations,
                                                                    device 
                                                                    )
        all_folds_interventions.append(interventions)
        all_folds_directions.append(com_directions)
        all_folds_top_heads.append(top_heads)

    return all_folds_interventions, all_folds_directions, all_folds_top_heads, tuning_activations