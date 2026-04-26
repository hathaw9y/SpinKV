import os
import types
import torch
import torch.nn as nn

from .fusion import fuse_norms
from .rotation import (
    absorb_R_input, absorb_R_output, absorb_R_into_embedding,
    patch_online_rotate, patch_linear_bfp, apply_linear_weight_bfp,
)
from .attention.llama import patch_llama_attention
from .attention.opt import patch_opt_attention


def _random_hadamard_matrix(size: int, device):
    from hadamard_utils import random_hadamard_matrix
    return random_hadamard_matrix(size, device=device)


def add_model_type(model) -> None:
    name = model.config._name_or_path.lower()
    if 'llama-2' in name:
        model.model_type = 'llama2'
    elif 'opt' in name:
        model.model_type = 'opt'
    else:
        raise ValueError(f"Unsupported Model: {model.config._name_or_path}")


def _orth_path(hook, kind: str) -> str:
    group_size = getattr(hook, 'orth_group_size', None)
    if group_size is not None:
        path = os.path.join(hook.orth_dir, hook.model_dir, f"{kind}_raw_gs{group_size}.pt")
        if os.path.exists(path):
            return path
        print(f"orthogonal matrix not found for group_size={group_size}: {path}")
    fallback = os.path.join(hook.orth_dir, hook.model_dir, f"{kind}_raw.pt")
    print(f"fallback orthogonal matrix: {fallback}")
    return fallback


def _load_orthogonal(hook, kind: str, device) -> torch.Tensor:
    path = _orth_path(hook, kind)
    if not os.path.exists(path):
        raise FileNotFoundError(f"orthogonal matrix not found: {path}")
    print(f"Load orthogonal matrix: {path}")
    return torch.load(path, map_location=device)


def _patch_online_input_rotate(linear: nn.Linear, R: torch.Tensor, hook) -> None:
    """Hadamard intermediate path처럼 weight 흡수와 runtime input rotate를 함께 적용."""
    absorb_R_input(linear, R)
    patch_online_rotate(linear, R, hook)


def _basis_change(x: torch.Tensor, R_from: torch.Tensor, R_to: torch.Tensor) -> torch.Tensor:
    return (x @ R_from.to(x.dtype).T) @ R_to.to(x.dtype)


def _apply_llama_hadamard_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    head_dim = hidden // model.config.num_attention_heads

    R_res = _random_hadamard_matrix(hidden, device=device)
    R_mlp = _random_hadamard_matrix(intermediate, device=device)
    R_head = _qk_rotation(model, device, hook)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj,
                     mlp.gate_proj, mlp.up_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.o_proj, R_res)
        absorb_R_output(mlp.down_proj, R_res)

        if not hook.offline:
            _patch_online_input_rotate(mlp.down_proj, R_mlp, hook)

        patch_llama_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_mlp, R_head
    torch.cuda.empty_cache()


def _apply_opt_hadamard_rotate(model, device, hook) -> None:
    hidden = model.config.hidden_size
    ffn_dim = model.config.ffn_dim
    head_dim = hidden // model.config.num_attention_heads

    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    R_res = _random_hadamard_matrix(hidden, device=device)
    R_ffn = _random_hadamard_matrix(ffn_dim, device=device)
    R_head = _qk_rotation(model, device, hook)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        attn = layer.self_attn
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.out_proj, R_res)
        absorb_R_input(layer.fc1, R_res)
        absorb_R_output(layer.fc2, R_res)

        if not hook.offline:
            _patch_online_input_rotate(layer.fc2, R_ffn, hook)

        patch_opt_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_ffn, R_head
    torch.cuda.empty_cache()


def _apply_llama_orthogonal_rotate(model, device, hook) -> None:
    R_attn = _load_orthogonal(hook, 'self_attn_input', device)
    R_mlp = _load_orthogonal(hook, 'mlp_input', device)
    R_lm = _load_orthogonal(hook, 'lm_head_input', device)
    R_head = _qk_rotation(model, device, hook)

    for layer_idx, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        R_a = R_attn[layer_idx]
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_a)

        R_m = R_mlp[layer_idx]
        for proj in [mlp.gate_proj, mlp.up_proj]:
            absorb_R_input(proj, R_m)

        R_next = R_attn[layer_idx + 1] if layer_idx + 1 < R_attn.shape[0] else R_lm
        absorb_R_output(attn.o_proj, R_m)
        absorb_R_output(mlp.down_proj, R_next)

        patch_llama_attention(attn, R_head, layer_idx, hook)
        _patch_llama_decoder_layer(layer, R_a, R_m, R_next)

    absorb_R_input(model.lm_head, R_lm)
    absorb_R_into_embedding(model, R_attn[0])

    del R_attn, R_mlp, R_lm, R_head
    torch.cuda.empty_cache()


def _apply_opt_orthogonal_rotate(model, device, hook) -> None:
    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    R_attn = _load_orthogonal(hook, 'self_attn_input', device)
    R_mlp = _load_orthogonal(hook, 'mlp_input', device)
    R_lm = _load_orthogonal(hook, 'lm_head_input', device)
    R_head = _qk_rotation(model, device, hook)

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        attn = layer.self_attn
        R_a = R_attn[layer_idx]
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_a)

        R_m = R_mlp[layer_idx]
        absorb_R_input(layer.fc1, R_m)
        R_next = R_attn[layer_idx + 1] if layer_idx + 1 < R_attn.shape[0] else R_lm
        absorb_R_output(attn.out_proj, R_m)
        absorb_R_output(layer.fc2, R_next)

        patch_opt_attention(attn, R_head, layer_idx, hook)
        _patch_opt_decoder_layer(layer, R_a, R_m, R_next)

    absorb_R_input(model.lm_head, R_lm)
    absorb_R_into_embedding(model, R_attn[0])

    del R_attn, R_mlp, R_lm, R_head
    torch.cuda.empty_cache()


def _patch_llama_decoder_layer(layer, R_attn, R_mlp, R_next) -> None:
    layer._spinkv_R_attn = R_attn
    layer._spinkv_R_mlp = R_mlp
    layer._spinkv_R_next = R_next

    def forward_fn(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False, **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        residual = _basis_change(residual, R_attn, R_mlp)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        residual = _basis_change(residual, R_mlp, R_next)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    layer.forward = types.MethodType(forward_fn, layer)


def _patch_opt_decoder_layer(layer, R_attn, R_mlp, R_next) -> None:
    layer._spinkv_R_attn = R_attn
    layer._spinkv_R_mlp = R_mlp
    layer._spinkv_R_next = R_next

    def forward_fn(
        self, hidden_states, attention_mask=None, layer_head_mask=None,
        output_attentions=False, use_cache=False, past_key_value=None,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training,
        )
        residual = _basis_change(residual, R_attn, R_mlp)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training,
        )
        residual = _basis_change(residual, R_mlp, R_next)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    layer.forward = types.MethodType(forward_fn, layer)


def _qk_rotation(model, device, hook):
    if hook.qk_rotate is None:
        return None
    if hook.qk_rotate != 'hadamard':
        raise ValueError(f"Unsupported qk_rotate: {hook.qk_rotate}")
    if model.model_type == 'llama2':
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    elif model.model_type == 'opt':
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")
    return _random_hadamard_matrix(head_dim, device=device)


def _patch_attention_only(model, device, hook) -> None:
    """rotation 없이 attention patch만 적용 (R_head=None)."""
    R_head = _qk_rotation(model, device, hook)
    if model.model_type == 'llama2':
        for layer_idx, layer in enumerate(model.model.layers):
            patch_llama_attention(layer.self_attn, R_head, layer_idx, hook)
    elif model.model_type == 'opt':
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            patch_opt_attention(layer.self_attn, R_head, layer_idx, hook)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def _tag_linear_bfp_categories(model) -> None:
    if model.model_type == 'llama2':
        for layer in model.model.layers:
            attn, mlp = layer.self_attn, layer.mlp
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj._spinkv_bfp_category = 'qkv'
            attn.o_proj._spinkv_bfp_category = 'o'
            for proj in (mlp.gate_proj, mlp.up_proj):
                proj._spinkv_bfp_category = 'up_gate'
            mlp.down_proj._spinkv_bfp_category = 'down'
    elif model.model_type == 'opt':
        for layer in model.model.decoder.layers:
            attn = layer.self_attn
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                proj._spinkv_bfp_category = 'qkv'
            attn.out_proj._spinkv_bfp_category = 'o'
            layer.fc1._spinkv_bfp_category = 'up_gate'
            layer.fc2._spinkv_bfp_category = 'down'
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")


def apply_rotate(model, device, hook, rotate: str | None = "hadamard") -> None:
    """
    rotate='hadamard'   : fuse_norms + Hadamard 회전 + attention patch
    rotate='orthogonal' : fuse_norms + learned orthogonal 회전 + attention patch
    rotate=None         : fuse_norms + attention patch (cq/collect를 raw 도메인에서 사용)
    """
    add_model_type(model)
    fuse_norms(model)
    _tag_linear_bfp_categories(model)
    if rotate is None:
        _patch_attention_only(model, device, hook)
        if getattr(hook, 'weight_bfp', False):
            _apply_linear_weight_bfp(model, hook)
        if getattr(hook, 'bfp', False):
            _patch_linear_bfp(model, hook)
        return

    if rotate == 'hadamard':
        if model.model_type == 'llama2':
            _apply_llama_hadamard_rotate(model, device, hook)
        elif model.model_type == 'opt':
            _apply_opt_hadamard_rotate(model, device, hook)
        else:
            raise ValueError(f"Unsupported model_type: {model.model_type}")
    elif rotate == 'orthogonal':
        if model.model_type == 'llama2':
            _apply_llama_orthogonal_rotate(model, device, hook)
        elif model.model_type == 'opt':
            _apply_opt_orthogonal_rotate(model, device, hook)
        else:
            raise ValueError(f"Unsupported model_type: {model.model_type}")
    else:
        raise ValueError(f"Unsupported rotate: {rotate}")

    if getattr(hook, 'weight_bfp', False):
        _apply_linear_weight_bfp(model, hook)

    if getattr(hook, 'bfp', False):
        _patch_linear_bfp(model, hook)


def _patch_linear_bfp(model, hook) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            patch_linear_bfp(module, hook)


def _apply_linear_weight_bfp(model, hook) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            apply_linear_weight_bfp(module, hook)
