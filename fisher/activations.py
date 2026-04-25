import os
import torch
from typing import NamedTuple, Optional


class Activations(NamedTuple):
    """
    Loaded KV activations + gradients.

    k, v, k_g, v_g : (n_layers, n_heads, T, head_dim)
    k_ropes        : same shape, OR None for OPT (no post-RoPE key activations).
    """
    k: torch.Tensor
    k_g: torch.Tensor
    v: torch.Tensor
    v_g: torch.Tensor
    k_ropes: Optional[torch.Tensor]


def _merge_batch_seq(x: torch.Tensor) -> torch.Tensor:
    """(L, B, H, S, D) -> (L, H, B*S, D)"""
    l, b, h, s, d = x.shape
    return x.transpose(1, 2).reshape(l, h, -1, d)
 
 
def _is_llama(model_name: str) -> bool:
    return 'llama-2' in model_name.lower()


def load_activations(model_name: str,
                     folder_name: str = 'activations') -> Activations:
    """
    LLaMA: saved as (k, k_ropes_grad, v, v_g, k_ropes)  -- k_ropes_grad shared between k and k_ropes
    OPT:   saved as (k, k_g, v, v_g)                    -- no k_ropes
    """
    file_path = os.path.join(folder_name, model_name)
    payload = torch.load(file_path)

    if _is_llama(model_name):
        k, k_g, v, v_g, k_ropes = payload
        k_ropes = _merge_batch_seq(k_ropes)
    else:
        k, k_g, v, v_g = payload
        k_ropes = None

    return Activations(
        k=_merge_batch_seq(k),
        k_g=_merge_batch_seq(k_g),
        v=_merge_batch_seq(v),
        v_g=_merge_batch_seq(v_g),
        k_ropes=k_ropes,
    )