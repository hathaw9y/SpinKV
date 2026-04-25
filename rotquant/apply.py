import torch
import torch.nn as nn

from hadamard_utils import random_hadamard_matrix

from .fusion import fuse_norms
from .rotation import (
    absorb_R_input, absorb_R_output, absorb_R_into_embedding,
    patch_online_rotate, patch_linear_bfp,
)
from .attention.llama import patch_llama_attention
from .attention.opt import patch_opt_attention


def add_model_type(model) -> None:
    name = model.config._name_or_path.lower()
    if 'llama-2' in name:
        model.model_type = 'llama2'
    elif 'opt' in name:
        model.model_type = 'opt'
    else:
        raise ValueError(f"Unsupported Model: {model.config._name_or_path}")


def _apply_llama_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    head_dim = hidden // model.config.num_attention_heads

    R_res = random_hadamard_matrix(hidden, device=device)
    R_mlp = random_hadamard_matrix(intermediate, device=device)
    R_head = random_hadamard_matrix(head_dim, device=device)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj,
                     mlp.gate_proj, mlp.up_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.o_proj, R_res)
        absorb_R_output(mlp.down_proj, R_res)

        if not hook.offline:
            absorb_R_input(mlp.down_proj, R_mlp)
            patch_online_rotate(mlp.down_proj, R_mlp, hook)

        patch_llama_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_mlp, R_head
    torch.cuda.empty_cache()


def _apply_opt_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    ffn_dim = model.config.ffn_dim
    head_dim = hidden // model.config.num_attention_heads

    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    R_res = random_hadamard_matrix(hidden, device=device)
    R_ffn = random_hadamard_matrix(ffn_dim, device=device)
    R_head = random_hadamard_matrix(head_dim, device=device)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        attn = layer.self_attn
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.out_proj, R_res)
        absorb_R_input(layer.fc1, R_res)
        absorb_R_output(layer.fc2, R_res)

        if not hook.offline:
            absorb_R_input(layer.fc2, R_ffn)
            patch_online_rotate(layer.fc2, R_ffn, hook)

        patch_opt_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_ffn, R_head
    torch.cuda.empty_cache()


def _patch_attention_only(model, hook) -> None:
    """rotation 없이 attention patch만 적용 (R_head=None)."""
    if model.model_type == 'llama2':
        for layer_idx, layer in enumerate(model.model.layers):
            patch_llama_attention(layer.self_attn, None, layer_idx, hook)
    elif model.model_type == 'opt':
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            patch_opt_attention(layer.self_attn, None, layer_idx, hook)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def apply_rotate(model, device, hook, rotate: bool = True) -> None:
    """
    rotate=True : fuse_norms + Hadamard 회전 + attention patch
    rotate=False: fuse_norms + attention patch (cq/collect를 raw 도메인에서 사용)
    """
    add_model_type(model)
    fuse_norms(model)
    if not rotate:
        _patch_attention_only(model, hook)
        if hook.bfp:
            _patch_linear_bfp(model, hook)
        return

    if model.model_type == 'llama2':
        _apply_llama_rotate(model, device, hook)
    elif model.model_type == 'opt':
        _apply_opt_rotate(model, device, hook)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")

    if hook.bfp:
        _patch_linear_bfp(model, hook)


def _patch_linear_bfp(model, hook) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            patch_linear_bfp(module, hook)
