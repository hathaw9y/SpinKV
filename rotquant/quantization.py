import torch


def vq_quantize(kv: torch.Tensor, codebook: torch.Tensor, n_channel: int = 4) -> torch.Tensor:
    """
    Vector Quantization for KV cache.

    Args:
        kv:       (B, H, T, D)
        codebook: (..., K, C)  with D = G * C, G = D // n_channel
        n_channel: sub-vector dimension C

    Returns:
        decoded:  (B, H, T, D), same dtype as kv
    """
    B, H, T, D = kv.shape
    G = D // n_channel
    K, C = codebook.shape[-2], codebook.shape[-1]

    # (B, H, T, D) -> (H*G, B*T, C)
    x = kv.reshape(B, H, T, G, C).permute(1, 3, 0, 2, 4).reshape(H * G, B * T, C)
    cb = codebook.reshape(H * G, K, C)

    dist = torch.cdist(x, cb, compute_mode='use_mm_for_euclid_dist')  # (H*G, B*T, K)
    idx = dist.argmin(dim=-1)                                          # (H*G, B*T)

    decoded = cb.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))        # (H*G, B*T, C)
    decoded = decoded.reshape(H, G, B, T, C).permute(2, 0, 3, 1, 4).reshape(B, H, T, D)
    return decoded.to(kv.dtype)