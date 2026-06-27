import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, set_seed, logging, PreTrainedModel
import torch.nn.functional as F
from trl import DPOTrainer, DPOConfig
import wandb
import numpy as np
import random
from data_processor import convert_tqa_to_preference_dataset
from utils import validate_save_path, HF_NAMES, check_device, MODELCLASS, cuda_supports_bfloat16
from contextlib import contextmanager
from collections import OrderedDict
from typing import Any, Union
import math
import gc


class CARESingleDirLayer(nn.Module):
    def __init__(self, config, fvec=None, freeze_mask=None, adaptive=False, adaptive_vec=None, act_fn='relu'):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.active = True
        self.adaptive = adaptive
        self.act_fn = act_fn
        self.inference_mode = False
        self.init_meta = False

        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_heads)])
        if self.act_fn=='relu':
            self.act_layer = nn.ReLU()
        
        if self.adaptive:
            if adaptive_vec is None:
                adaptive_vec = torch.zeros(self.num_heads * self.head_dim)
            if adaptive_vec.shape == (self.num_heads * self.head_dim,):
                adaptive_vec = adaptive_vec.reshape(self.num_heads, self.head_dim).contiguous()
            self.register_buffer('adaptive_vec', adaptive_vec, persistent=True) 
            self.temperature = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_heads)])

        if freeze_mask is None:
            freeze_mask = torch.ones(self.num_heads, dtype=torch.bool)
        self.register_buffer('freeze_mask', freeze_mask, persistent=True) 

        if fvec is None:
            fvec = torch.zeros(self.num_heads * self.head_dim)  
        if fvec.shape == (self.num_heads * self.head_dim,):
            fvec = fvec.reshape(self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(0).contiguous()
        self.register_buffer('fvec', fvec, persistent=True)  
        
        self._init_weights()

    def _init_weights(self):
        for i, param in enumerate(self.alpha):
            if self.act_fn == 'relu':
                if self.freeze_mask[i]:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.uniform_(param, 0.01, 0.5) 

        if self.adaptive:
            for i, param in enumerate(self.temperature):
                nn.init.normal_(param, mean=math.log(math.e - 1), std=0.01) 

        if not self.freeze_mask.all():
            self.enable()
        else:
            self.disable()

    def disable(self):
        self.active = False 
        for p in self.parameters():
            p.requires_grad_(False) 
    
    def enable(self):
        self.active = True
        for i in range(self.num_heads):
            self.alpha[i].requires_grad = not self.freeze_mask[i]
            if self.adaptive:
                self.temperature[i].requires_grad = not self.freeze_mask[i]
            #    self.beta[i].requires_grad = not self.freeze_mask[i]

    def to_inference_mode(self):
        layer_scale = self.act_layer(torch.cat([a for a in self.alpha])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
        if not layer_scale.any():
            self.active = False
        self.delta = layer_scale * self.fvec # [bsz, qlen, num_heads, 1] * [1, 1, num_heads, head_dim]
        if self.adaptive:
            self.layer_temp = F.softplus(torch.cat([t for t in self.temperature])).unsqueeze(0).unsqueeze(0).contiguous()  # [1, ,1, num_heads]
        
        self.inference_mode = True

    def to_train_mode(self):
        self.inference_mode = False

    def forward(self, attn_out):
        if not self.active:
            return attn_out
        
        if not self.init_meta:
            self.to(device=attn_out.device, dtype=attn_out.dtype)
            if self.inference_mode:
                self.delta = self.delta.to(device=attn_out.device, dtype=attn_out.dtype)
                if self.adaptive:
                    self.layer_temp = self.layer_temp.to(device=attn_out.device, dtype=attn_out.dtype)
            self.init_meta = True

        if self.inference_mode:
            delta = self.delta
            if self.adaptive:
                layer_temp = self.layer_temp
        else:
            layer_scale = self.act_layer(torch.cat([a for a in self.alpha])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
            if not layer_scale.any():
                self.active = False
                return attn_out
            
            delta = layer_scale * self.fvec # [bsz, qlen, num_heads, 1] * [1, 1, num_heads, head_dim]
            if self.adaptive:
                layer_temp = F.softplus(torch.cat([t for t in self.temperature])).unsqueeze(0).unsqueeze(0).contiguous()  # [1, ,1, num_heads]
        
        bsz, qlen, hidden_size = attn_out.shape
        if self.adaptive:
            attn_out_4d = attn_out.view(bsz, qlen, self.num_heads, self.head_dim)
            similarity = torch.einsum("blhd,hd->blh", attn_out_4d, self.adaptive_vec)
            adaptive_scale = torch.sigmoid(-similarity * layer_temp)  # [bsz, qlen, num_heads]

            delta = adaptive_scale.unsqueeze(-1) * delta # [bsz, qlen, num_heads, 1] * [1, 1, num_heads, head_dim]

        return attn_out + delta.reshape(*delta.shape[:2], hidden_size).contiguous()


class CAREDualDirLayer(nn.Module):
    def __init__(self, config, fvec=None, fvec2=None, freeze_mask=None, adaptive=False, adaptive_vec=None, act_fn='relu'):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
        self.active = True
        self.adaptive = adaptive
        self.act_fn = act_fn
        self.inference_mode = False
        self.init_meta = False

        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_heads)])
        self.beta = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_heads)])
        if self.act_fn=='relu':
            self.act_layer = nn.ReLU()
        
        if freeze_mask is None:
            freeze_mask = torch.ones(self.num_heads, dtype=torch.bool)
        self.register_buffer('freeze_mask', freeze_mask, persistent=True) 

        if self.adaptive:
            if adaptive_vec is None:
                adaptive_vec = torch.zeros(self.num_heads * self.head_dim)
            if adaptive_vec.shape == (self.num_heads * self.head_dim,):
                adaptive_vec = adaptive_vec.reshape(self.num_heads, self.head_dim).contiguous()
            self.register_buffer('adaptive_vec', adaptive_vec, persistent=True) 
            self.temperature = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.num_heads)])

        if fvec is None:
            fvec = torch.ones(self.num_heads * self.head_dim)  
        if fvec.shape == (self.num_heads * self.head_dim,):
            fvec = fvec.reshape(self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(0).contiguous()
        self.register_buffer('fvec', fvec, persistent=True)  

        if fvec2 is None:
            fvec2 = torch.ones(self.num_heads * self.head_dim)  
        if fvec2.shape == (self.num_heads * self.head_dim,):
            fvec2 = fvec2.reshape(self.num_heads, self.head_dim).unsqueeze(0).unsqueeze(0).contiguous()
        self.register_buffer('fvec2', fvec2, persistent=True)  
        self._init_weights()

    def _init_weights(self):
        for i, param in enumerate(self.alpha):
            if self.act_fn == 'relu':
                if self.freeze_mask[i]:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.uniform_(param, 0.01, 0.5) 

        for i, param in enumerate(self.beta):
            if self.act_fn == 'relu':
                if self.freeze_mask[i]:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.uniform_(param, 0.01, 0.5) 

        if self.adaptive:
            for i, param in enumerate(self.temperature):
                nn.init.normal_(param, mean=math.log(math.e - 1), std=0.01) 

        if not self.freeze_mask.all():
            self.enable()
        else:
            self.disable()

    def disable(self):
        self.active = False 
        for p in self.parameters():
            p.requires_grad_(False) 
    
    def enable(self):
        self.active = True
        for i in range(self.num_heads):
            self.alpha[i].requires_grad = not self.freeze_mask[i]
            self.beta[i].requires_grad = not self.freeze_mask[i]
            if self.adaptive:
                self.temperature[i].requires_grad = not self.freeze_mask[i]
         
    def to_inference_mode(self):
        layer_scale = self.act_layer(torch.cat([a for a in self.alpha])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
        layer_scale2 = self.act_layer(torch.cat([a for a in self.beta])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
        if not layer_scale.any() and not layer_scale2.any():
            self.active = False
        self.delta = layer_scale * self.fvec + layer_scale2 * self.fvec2 # [bsz, qlen, num_heads, 1] * [1, 1, num_heads, head_dim]
        if self.adaptive:
            self.layer_temp = F.softplus(torch.cat([t for t in self.temperature])).unsqueeze(0).unsqueeze(0).contiguous()  # [1, ,1, num_heads]
        
        self.inference_mode = True

    def to_train_mode(self):
        self.inference_mode = False

    def forward(self, attn_out):
        if not self.active:
            return attn_out
        
        if not self.init_meta:
            self.to(device=attn_out.device, dtype=attn_out.dtype)
            if self.inference_mode:
                self.delta = self.delta.to(device=attn_out.device, dtype=attn_out.dtype)
                if self.adaptive:
                    self.layer_temp = self.layer_temp.to(device=attn_out.device, dtype=attn_out.dtype)
            self.init_meta = True

        if self.inference_mode:
            delta = self.delta
            if self.adaptive:
                layer_temp = self.layer_temp
        else:
            layer_scale = self.act_layer(torch.cat([a for a in self.alpha])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
            layer_scale2 = self.act_layer(torch.cat([a for a in self.beta])).unsqueeze(0).unsqueeze(0).unsqueeze(-1).contiguous()  # [1, ,1, num_heads, 1]
            if not layer_scale.any() and not layer_scale2.any():
                self.active = False
                return attn_out
            
            delta = layer_scale * self.fvec + layer_scale2 * self.fvec2 # [1, 1, num_heads, 1] * [1, 1, num_heads, head_dim]
            if self.adaptive:
                layer_temp = F.softplus(torch.cat([t for t in self.temperature])).unsqueeze(0).unsqueeze(0).contiguous()  # [1, ,1, num_heads]
                self.adaptive_vec.data = F.normalize(delta, p=2, dim=-1).squeeze(0).squeeze(0)

        bsz, qlen, hidden_size = attn_out.shape
        if self.adaptive:
            
            attn_out_4d = attn_out.view(bsz, qlen, self.num_heads, self.head_dim)
            
            similarity = torch.einsum("blhd,hd->blh", attn_out_4d, self.adaptive_vec)
            adaptive_scale = torch.sigmoid(-similarity * layer_temp)  # [bsz, qlen, num_heads]

            delta = adaptive_scale.unsqueeze(-1) * delta # [bsz, qlen, num_heads, 1] * [1, 1, num_heads, head_dim]
            return attn_out + delta.reshape(bsz, qlen, hidden_size).contiguous()

        return attn_out + delta.reshape(*delta.shape[:2], hidden_size).contiguous() 

def create_freeze_mask(config, unfreeze_heads_list=None):
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    
    freeze_masks = torch.ones((num_layers, num_heads), dtype=torch.bool)
    
    if unfreeze_heads_list is not None:
        for l, h in unfreeze_heads_list:
            freeze_masks[l, h] = False
    
    return freeze_masks


def fvec_loader(config, directions, proj_stds=None):
    if directions is not None:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        assert directions.shape == (config.num_hidden_layers, config.num_attention_heads, head_dim), directions.shape
        directions = F.normalize(directions, p=2, dim=-1)  
        directions = directions.reshape(-1, head_dim * config.num_attention_heads)
        
        if proj_stds is not None:
            directions = directions * proj_stds.repeat_interleave(head_dim, dim=1) # l h -> l h*d
    return directions

def adaptive_vec_loader(config, directions, proj_stds=None):
    if directions is not None:
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        assert directions.shape == (config.num_hidden_layers, config.num_attention_heads, head_dim), directions.shape

        directions = F.normalize(directions, p=2, dim=-1)

        if proj_stds is not None:
            directions = directions / proj_stds.unsqueeze(-1)

        directions = directions.reshape(-1, head_dim * config.num_attention_heads)
        
    return directions

def load_alpha(model, checkpoint_path):
    config = model.config
    num_heads = getattr(config, 'num_attention_heads')
    hidden_size = getattr(config, 'hidden_size')
    head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        checkpoint = {key: f.get_tensor(key) for key in f.keys()}

    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    for k, v in checkpoint.items():
        if 'adaptive_vec' in k and v.ndim!=2:
            checkpoint[k] = v.reshape(num_heads, head_dim)
        if 'fvec' in k and v.ndim==1:
            checkpoint[k] = v.reshape(num_heads, head_dim).unsqueeze(0).unsqueeze(0)
        if 'adaptive_vec2' in k and v.ndim!=2:
            checkpoint[k] = v.reshape(num_heads, head_dim)
        if 'fvec2' in k and v.ndim==1:
            checkpoint[k] = v.reshape(num_heads, head_dim).unsqueeze(0).unsqueeze(0)

    model.load_state_dict(checkpoint, strict=False)
    return model

def get_submodule(module, attr_path):
        '''nested sub module'''
        attrs = attr_path.strip('.').split('.')
        for attr in attrs:
            module = getattr(module, attr)
        return module

def set_submodule(module, attr_path, value):
   
    attrs = attr_path.strip('.').split('.')
    for attr in attrs[:-1]:
        module = getattr(module, attr)
    setattr(module, attrs[-1], value)

def wrap_llama_with_alpha(model, 
                          unfreeze_heads_list=None, 
                          alpha_vecs=None, 
                          beta_vecs=None, 
                          checkpoint_path=None, 
                          adaptive=False, 
                          act_fn='relu',
                          tuning_activations=None,
                          inject_to='self_attn.o_proj',
                          tag='alpha_injection'):
    
    decoders='model.layers'
    model_type = getattr(model.config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'
        inject_to = 'attn.out_proj'
    elif model_type == 'phi':
        inject_to = 'self_attn.dense'

    def inject_before_module(layer, inject_to, injection, tag='alpha_injection'):
        original_module = get_submodule(layer, inject_to)
        set_submodule(
            layer, 
            inject_to, 
            nn.Sequential(
                OrderedDict([
                    (tag, injection),
                    ("original", original_module),
                ])
            ))
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    if alpha_vecs is not None and beta_vecs is not None:
        alpha_vecs = torch.from_numpy(alpha_vecs) if isinstance(alpha_vecs, np.ndarray) else alpha_vecs # l h*d
        beta_vecs = torch.from_numpy(beta_vecs) if isinstance(beta_vecs, np.ndarray) else beta_vecs # l h*d
        directions = F.normalize(alpha_vecs, p=2, dim=-1) + F.normalize(beta_vecs, p=2, dim=-1) 
        directions = directions.reshape(num_layers, num_heads, -1) # l h d

        alpha_vecs = alpha_vecs.reshape(num_layers, num_heads, -1) # l h d
        beta_vecs = beta_vecs.reshape(num_layers, num_heads, -1) # l h d
        ca_directions = F.normalize(alpha_vecs, p=2, dim=-1) + F.normalize(beta_vecs, p=2, dim=-1)
    else:
        directions = None
        ca_directions = None

    proj_stds = None
    ca_proj_stds = None
    if tuning_activations is not None and directions is not None and ca_directions is not None:
        activations = torch.from_numpy(tuning_activations) if isinstance(tuning_activations, np.ndarray) else tuning_activations # b l h d
        unit_directions = F.normalize(directions, p=2, dim=-1)
        proj_vals = torch.einsum("blhd,lhd->blh", activations, unit_directions)
        proj_stds = torch.std(proj_vals, dim=0).clip(min=1e-8) # l h

        unit_ca_directions = F.normalize(ca_directions, p=2, dim=-1)
        ca_proj_vals = torch.einsum("blhd,lhd->blh", activations, unit_ca_directions)
        ca_proj_stds = torch.std(ca_proj_vals, dim=0).clip(min=1e-8) # l h

    adaptive_vecs = adaptive_vec_loader(model.config, directions, proj_stds=proj_stds)
    freeze_masks = create_freeze_mask(model.config, unfreeze_heads_list)
    fvecs = fvec_loader(model.config, directions, proj_stds=proj_stds)

    
    for layer_idx, layer in enumerate(get_submodule(model, decoders)):
        
        freeze_mask = freeze_masks[layer_idx] if freeze_masks is not None else None

        fvec = fvecs[layer_idx] if fvecs is not None else None
        adaptive_vec = adaptive_vecs[layer_idx] if adaptive_vecs is not None else None
        
        injection = CARESingleDirLayer(
            model.config, 
            fvec=fvec.clone() if fvec is not None else None,
            freeze_mask=freeze_mask.clone() if freeze_mask is not None else None,
            adaptive=adaptive,
            adaptive_vec=adaptive_vec.clone() if adaptive_vec is not None else None,
            act_fn=act_fn
        )
        inject_before_module(layer, inject_to, injection) 
        
    if checkpoint_path is not None:
        model = load_alpha(model, checkpoint_path)
        for layer in get_submodule(model, decoders):
            injected_module = get_submodule(layer, inject_to)
            if hasattr(injected_module, tag):
                alpha_layer = getattr(injected_module, tag)
                if not alpha_layer.freeze_mask.all():
                    alpha_layer.enable()
                else:
                    alpha_layer.disable()

    for name, param in model.named_parameters():
        if tag not in name:
            param.requires_grad = False

    return model


def wrap_llama_with_alpha_beta(model, 
                          unfreeze_heads_list=None, 
                          alpha_vecs=None, 
                          beta_vecs=None, 
                          checkpoint_path=None, 
                          adaptive=False, 
                          act_fn='relu',
                          tuning_activations=None,
                          inject_to='self_attn.o_proj',
                          tag='care_injection'):
    
    decoders='model.layers'
    model_type = getattr(model.config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'
        inject_to = 'attn.out_proj'
    elif model_type == 'phi':
        inject_to = 'self_attn.dense'

    def inject_before_module(layer, inject_to, injection, tag='care_injection'):
        original_module = get_submodule(layer, inject_to)
        set_submodule(
            layer, 
            inject_to, 
            nn.Sequential(
                OrderedDict([
                    (tag, injection),
                    ("original", original_module),
                ])
            ))
        
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    if tuning_activations is not None and alpha_vecs is not None:
        alpha_vecs = torch.from_numpy(alpha_vecs) if isinstance(alpha_vecs, np.ndarray) else alpha_vecs # l h*d
        directions = F.normalize(alpha_vecs, p=2, dim=-1) 
        directions = directions.reshape(num_layers, num_heads, -1) # l h d
        activations = torch.from_numpy(tuning_activations) if isinstance(tuning_activations, np.ndarray) else tuning_activations # b l h d
        unit_directions = F.normalize(directions, p=2, dim=-1)
        proj_vals = torch.einsum("blhd,lhd->blh", activations, unit_directions)
        proj_stds = torch.std(proj_vals, dim=0).clip(min=1e-8) # l h
    else:
        directions = None
        proj_stds = None
    adaptive_vecs = adaptive_vec_loader(model.config, directions, proj_stds=proj_stds)
    fvecs = fvec_loader(model.config, directions, proj_stds=proj_stds)

    if tuning_activations is not None and beta_vecs is not None:
        beta_vecs = torch.from_numpy(beta_vecs) if isinstance(beta_vecs, np.ndarray) else beta_vecs # l h*d
        directions2 = F.normalize(beta_vecs, p=2, dim=-1) 
        directions2 = directions2.reshape(num_layers, num_heads, -1) # l h d
        activations = torch.from_numpy(tuning_activations) if isinstance(tuning_activations, np.ndarray) else tuning_activations # b l h d
        unit_directions2 = F.normalize(directions2, p=2, dim=-1)
        proj_vals2 = torch.einsum("blhd,lhd->blh", activations, unit_directions2)
        proj_stds2 = torch.std(proj_vals2, dim=0).clip(min=1e-8) # l h
    else:
        directions2 = None
        proj_stds2 = None
    fvecs2 = fvec_loader(model.config, directions2, proj_stds=proj_stds2)

    freeze_masks = create_freeze_mask(model.config, unfreeze_heads_list)

    for layer_idx, layer in enumerate(get_submodule(model, decoders)):
        
        freeze_mask = freeze_masks[layer_idx] if freeze_masks is not None else None

        fvec = fvecs[layer_idx] if fvecs is not None else None
        fvec2 = fvecs2[layer_idx] if fvecs2 is not None else None
        adaptive_vec = adaptive_vecs[layer_idx] if adaptive_vecs is not None else None
        
        injection = CAREDualDirLayer(
            model.config, 
            fvec=fvec.clone() if fvec is not None else None,
            fvec2=fvec2.clone() if fvec2 is not None else None,
            freeze_mask=freeze_mask.clone() if freeze_mask is not None else None,
            adaptive=adaptive,
            adaptive_vec=adaptive_vec.clone() if adaptive_vec is not None else None,
            act_fn=act_fn
        )
        inject_before_module(layer, inject_to, injection) 
        
    if checkpoint_path is not None:
        model = load_alpha(model, checkpoint_path)
        for layer in get_submodule(model, decoders):
            injected_module = get_submodule(layer, inject_to)
            if hasattr(injected_module, tag):
                alpha_layer = getattr(injected_module, tag)
                if not alpha_layer.freeze_mask.all():
                    alpha_layer.enable()
                else:
                    alpha_layer.disable()

    for name, param in model.named_parameters():
        if tag not in name:
            param.requires_grad = False

    return model

def inference_mode(model, inject_to='self_attn.o_proj', decoders='model.layers'):
    config = model.config
    model_type = getattr(config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'
        inject_to = 'attn.out_proj'
    elif model_type == 'phi':
        inject_to = 'self_attn.dense'

    for layer in get_submodule(model, decoders):
        injected_module = get_submodule(layer, inject_to)
        for key in ['alpha_injection', 'beta_injection', 'care_injection']:
            if hasattr(injected_module, key):
                inject_layer = getattr(injected_module, key)
                inject_layer.to_inference_mode()


def train_mode(model, inject_to='self_attn.o_proj', decoders='model.layers'):
    config = model.config
    model_type = getattr(config, 'model_type')
    if model_type == 'gptj':
        decoders = 'transformer.h'
        inject_to = 'attn.out_proj'
    elif model_type == 'phi':
        inject_to = 'self_attn.dense'

    for layer in get_submodule(model, decoders):
        injected_module = get_submodule(layer, inject_to)
        for key in ['alpha_injection', 'beta_injection', 'care_injection']:
            if hasattr(injected_module, key):
                inject_layer = getattr(injected_module, key)
                inject_layer.to_train_mode()


def wrap_llama_with_checkpoint(model, checkpoint_path, adaptive, act_fn, use_dual_dirs):
    wrap_fn = wrap_llama_with_alpha_beta if use_dual_dirs else wrap_llama_with_alpha
    return wrap_fn(model, checkpoint_path=checkpoint_path, adaptive=adaptive, act_fn=act_fn)
    

class CARETrainer(DPOTrainer):
    def _save(self, output_dir=None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        inject_state = {
            k: v.cpu() 
            for k, v in state_dict.items()  
            if "_injection" in k
        }
        
        super()._save(output_dir, state_dict=inject_state)

    @contextmanager
    def disable_alpha(self):
        decoders = 'model.layers'
        inject_to = 'self_attn.o_proj'
        model_type = getattr(self.model.config, 'model_type')
        if model_type == 'gptj':
            decoders = 'transformer.h'
            inject_to = 'attn.out_proj'
        elif model_type == 'phi':
            inject_to = 'self_attn.dense'

        original_states = []
        try:
            inject_layers = []
            for layer in get_submodule(self.model, decoders):
                injected_module = get_submodule(layer, inject_to)
                for key in ['alpha_injection', 'beta_injection', 'care_injection']:
                    if hasattr(injected_module, key):
                        inject_layer = getattr(injected_module, key)
                        inject_layers.append(inject_layer)
            
            original_states = [lyr.active for lyr in inject_layers]
            
            for lyr in inject_layers:
                lyr.disable()
                lyr.active = False
            
            yield
        finally:
            for lyr, act in zip(inject_layers, original_states):
                lyr.active = act
                if act:
                    lyr.enable()

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        with self.disable_alpha():
            chosen_logps, rejected_logps = super().compute_ref_log_probs(batch)
        return chosen_logps, rejected_logps
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        if return_outputs:
            loss, metrics = loss

        if self.l1_lambda != 0.0:
            named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

            alpha_params = []
            for name, param in named_parameters:
                if '_injection.alpha.' in name or '_injection.beta.' in name:
                    alpha = F.relu(param).sum().to(loss.device)
                    if alpha > 0 :
                        alpha_params.append(alpha)
            if alpha_params and self.l1_lambda != 0.0:
                loss = loss + self.l1_lambda * sum(alpha_params)

        if return_outputs:
            return loss, metrics

        return loss

def tune_alpha(args, dataset, vene_key, fold_no, unfreeze_heads_list=None, directions=None, tuning_activations=None):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    set_seed(seed)

    TUNE_PATH = validate_save_path(f'./{args.run_name}/tunes')
    output_dir = TUNE_PATH / f'{vene_key}_{fold_no}_of_{args.num_fold}'
    
    task_map = {
        'truthfulqa': {
            'dataloader': convert_tqa_to_preference_dataset,
            'trainer': CARETrainer
        },
    }
   
    wandb.init(mode="disabled")
    

    logging.set_verbosity_error()
    
    model_name = HF_NAMES[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    model_config = AutoConfig.from_pretrained(HF_NAMES[args.model_name], 
                                            do_sample=False,
                                            local_files_only=True,)
    torch_dtype = getattr(model_config, 'torch_dtype', torch.float32)
    if torch_dtype != torch.bfloat16 or not cuda_supports_bfloat16():
        torch_dtype = torch.float32 
    bf16 = torch_dtype == torch.bfloat16
    model = MODELCLASS[args.model_name].from_pretrained(HF_NAMES[args.model_name],
                                            config=model_config, 
                                            low_cpu_mem_usage=True, 
                                            #torch_dtype=torch.float16, 
                                            torch_dtype=torch_dtype,
                                            device_map="auto", 
                                            local_files_only=True,)
    check_device(model)
    alpha_vecs, beta_vecs = directions
    alpha_heads, beta_heads = unfreeze_heads_list
    if args.use_dual_dirs:
        model = wrap_llama_with_alpha_beta(model, 
                                        unfreeze_heads_list=alpha_heads, 
                                        alpha_vecs=alpha_vecs, 
                                        beta_vecs=beta_vecs,
                                        adaptive=args.adaptive,
                                        act_fn=args.act_fn,
                                        tuning_activations=tuning_activations)
    else:
        model = wrap_llama_with_alpha(model, 
                                        unfreeze_heads_list=alpha_heads, 
                                        alpha_vecs=alpha_vecs, 
                                        beta_vecs=beta_vecs,
                                        adaptive=args.adaptive,
                                        act_fn=args.act_fn,
                                        tuning_activations=tuning_activations)

    
    if args.save_strategy == 'best':
        save_strategy = 'epoch'
        load_best_model_at_end = True
        save_total_limit = 10
    elif args.save_strategy == 'no':
        save_strategy = 'no'
        load_best_model_at_end = False
        save_total_limit = None
    elif args.save_strategy == 'last':
        save_strategy = 'epoch'
        load_best_model_at_end = False
        save_total_limit = 10
    else:
        raise ValueError(f'Save strategy {args.save_strategy} is not supported')
    training_args = DPOConfig(
        output_dir=output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs = {'min_lr_rate': args.min_lr_rate},
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epoch,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit = save_total_limit,
        report_to='wandb',
        logging_strategy="steps",
        logging_steps=5, 
        seed = args.seed,
        do_train = True,
        do_eval = True,
        bf16=bf16,
        max_prompt_length=128,
        max_completion_length=128,
        padding_value=0,
        beta = args.dpo_beta,
        reference_free = False,
        loss_type = 'sigmoid',
        weight_decay = args.weight_decay,
        precompute_ref_log_probs=True
    )
    torch.autograd.set_detect_anomaly(True)
    task = 'truthfulqa' if 'tqa' in args.dataset_name else ''
        
    datasets = task_map[task]['dataloader'](dataset, key=args.mc_key)
    for key in ['train','val']:
        print(f"Number of {key} samples: {len(datasets[key])}")
    trainer = task_map[task]['trainer']

    train_dataset = datasets['train']
    eval_dataset = datasets['val']
    

    if task == 'truthfulqa':
        trainer = trainer(
            model=model,
            ref_model=None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            args=training_args,
        )
        trainer.l1_lambda = args.l1_lambda
    trainer.train()

    model.zero_grad(set_to_none=True)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    gc.collect()
    torch.cuda.empty_cache()


