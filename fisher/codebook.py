import torch
from tqdm import tqdm

from .kmeans import batched_weighted_kmeans


def _fisher_weights(grad_g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """grad_g: (n_groups, T, n_channel) -> (n_groups, T) normalized along T."""
    w = (grad_g ** 2).sum(dim=-1)
    return w / (w.sum(dim=-1, keepdim=True) + eps)


def fisher_codebook_batched(x: torch.Tensor, x_grads: torch.Tensor,
                             n_channel: int = 4, n_cluster: int = 256,
                             device: str = 'cuda') -> torch.Tensor:
    """
    Per-(layer, head) Fisher-weighted product-quantization codebook learning.

    Args:
        x:       (n_layers, n_heads, T, head_dim)
        x_grads: same shape as x

    Returns:
        codebooks: (n_layers, n_heads, n_groups, n_cluster, n_channel)
    """
    n_layers, n_heads, _, head_dim = x.shape
    n_groups = head_dim // n_channel
    org_dtype = x.dtype

    cb = []
    for layer_idx in tqdm(range(n_layers), desc="Layer"):
        data_l = x[layer_idx].float().to(device)
        grad_l = x_grads[layer_idx].float().to(device)

        l_cb = []
        for head_idx in range(n_heads):
            data = data_l[head_idx]   # (T, head_dim)
            grad = grad_l[head_idx]

            # (T, head_dim) -> (n_groups, T, n_channel)
            data_g = data.reshape(-1, n_groups, n_channel).permute(1, 0, 2).contiguous()
            grad_g = grad.reshape(-1, n_groups, n_channel).permute(1, 0, 2).contiguous()

            w = _fisher_weights(grad_g)
            centers = batched_weighted_kmeans(
                data_g, w, n_clusters=n_cluster, use_kmeanspp=True,
            )                          # (n_groups, n_cluster, n_channel)
            l_cb.append(centers.cpu().to(org_dtype))

        cb.append(l_cb)

    return torch.stack([torch.stack(h) for h in cb])