import types
import torch
import torch.nn as nn

from utils import bfp_quantize_activation, bfp_quantize_weight_transpose


def _bfp_bits_for_linear(linear: nn.Linear, hook) -> int:
    category = getattr(linear, '_spinkv_bfp_category', None)
    override = {
        'qkv': getattr(hook, 'bfp_qkv_bits', None),
        'o': getattr(hook, 'bfp_o_bits', None),
        'up_gate': getattr(hook, 'bfp_up_gate_bits', None),
        'down': getattr(hook, 'bfp_down_bits', None),
    }.get(category, None)
    return getattr(hook, 'bfp_bits', 8) if override is None else override


# ---------------- absorb R into weights ----------------
def absorb_R_input(linear: nn.Linear, R: torch.Tensor) -> None:
    """입력 측 회전: W <- W @ R"""
    dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W @ R.double()).to(dtype)


def absorb_R_output(linear: nn.Linear, R: torch.Tensor) -> None:
    """출력 측 회전: W <- R^T @ W, b <- b @ R"""
    dtype = linear.weight.dtype
    W = linear.weight.data.double()                       # [out, in]
    linear.weight.data = (R.double().T @ W).to(dtype)
    if linear.bias is not None:
        b = linear.bias.data.double()
        linear.bias.data = (b @ R.double()).to(dtype)


def absorb_R_into_embedding(model, R: torch.Tensor) -> None:
    if model.model_type == 'llama2':
        embs = [model.model.embed_tokens]
    elif model.model_type == 'opt':
        embs = [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")

    for emb in embs:
        dtype = emb.weight.dtype
        W = emb.weight.data.double()
        emb.weight.data = (W @ R.double()).to(dtype)


# ---------------- online rotation ----------------
def patch_online_rotate(linear: nn.Linear, R: torch.Tensor, hook=None) -> None:
    """forward 시점에 입력을 R로 회전시키는 monkey patch."""
    R_local = R

    def forward_fn(self, x):
        x = x @ R_local.to(x.dtype)
        if hook is not None and hook.bfp:
            x = bfp_quantize_activation(
                x, hook.bfp_block_size, _bfp_bits_for_linear(self, hook),
            )
        return torch.nn.functional.linear(x, self.weight, self.bias)

    linear._spinkv_online_rotate = True
    linear.forward = types.MethodType(forward_fn, linear)


def patch_linear_bfp(linear: nn.Linear, hook) -> None:
    if getattr(linear, '_spinkv_online_rotate', False):
        return

    org_forward = linear.forward

    def forward_fn(self, x):
        if hook.bfp:
            x = bfp_quantize_activation(
                x, hook.bfp_block_size, _bfp_bits_for_linear(self, hook),
            )
        return org_forward(x)

    linear.forward = types.MethodType(forward_fn, linear)


@torch.no_grad()
def apply_linear_weight_bfp(linear: nn.Linear, hook) -> None:
    linear.weight.data = bfp_quantize_weight_transpose(
        linear.weight.data,
        getattr(hook, 'weight_bfp_block_size', 128),
        getattr(hook, 'weight_bfp_bits', 8),
    )
