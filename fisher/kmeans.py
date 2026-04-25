import torch


def kmeanspp_init(data: torch.Tensor, weights: torch.Tensor, K: int,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    Weighted k-means++ initialization.

    Args:
        data:    (B, N, D)
        weights: (B, N)
        K:       number of clusters

    Returns:
        centers: (B, K, D)
    """
    B, N, D = data.shape
    centers = torch.empty(B, K, D, device=data.device, dtype=data.dtype)

    def _gather_points(idx_b1: torch.Tensor) -> torch.Tensor:
        """idx_b1: (B, 1) -> (B, D) gathered points."""
        return torch.gather(data, 1, idx_b1.unsqueeze(-1).expand(-1, -1, D)).squeeze(1)

    # 첫 center: weights에 비례한 sampling
    idx = torch.multinomial(weights, 1)              # (B, 1)
    centers[:, 0] = _gather_points(idx)

    min_dist_sq = torch.cdist(data, centers[:, :1]).squeeze(-1) ** 2  # (B, N)

    for k in range(1, K):
        prob = min_dist_sq * weights
        prob_sum = prob.sum(dim=-1, keepdim=True)
        degenerate = prob_sum.squeeze(-1) < eps      # (B,)

        # degenerate batch는 weights uniform fallback
        safe_prob = torch.where(
            degenerate.unsqueeze(-1),
            weights,
            prob / (prob_sum + eps),
        )
        idx = torch.multinomial(safe_prob, 1)
        new_center = _gather_points(idx)             # (B, D)
        centers[:, k] = new_center

        new_dist_sq = ((data - new_center.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, N)
        min_dist_sq = torch.minimum(min_dist_sq, new_dist_sq)

    return centers


def batched_weighted_kmeans(data: torch.Tensor, weights: torch.Tensor,
                             n_clusters: int = 256, n_iter: int = 100,
                             tol: float = 1e-5, eps: float = 1e-8,
                             use_kmeanspp: bool = True) -> torch.Tensor:
    """
    Batched weighted k-means (Lloyd).

    Args:
        data:       (B, N, D)
        weights:    (B, N)

    Returns:
        centers:    (B, K, D)
    """
    B, N, D = data.shape
    K = n_clusters

    if use_kmeanspp:
        centers = kmeanspp_init(data, weights, K, eps=eps)
    else:
        idx = torch.randint(0, N, (B, K), device=data.device)
        centers = torch.gather(data, 1, idx.unsqueeze(-1).expand(-1, -1, D)).clone()

    w = weights.unsqueeze(-1)  # (B, N, 1)

    for _ in range(n_iter):
        dist = torch.cdist(data, centers)            # (B, N, K)
        assign = dist.argmin(dim=-1)                 # (B, N)

        one_hot = torch.nn.functional.one_hot(assign, K).to(data.dtype)
        weighted = one_hot * w                       # (B, N, K)

        num = torch.einsum('bnk,bnd->bkd', weighted, data)  # (B, K, D)
        den = weighted.sum(dim=1).unsqueeze(-1)             # (B, K, 1)

        new_centers = num / (den + eps)

        # 비어있는 cluster는 이전 center 유지
        empty = den.squeeze(-1) < eps                # (B, K)
        new_centers = torch.where(empty.unsqueeze(-1), centers, new_centers)

        shift = (new_centers - centers).abs().max()
        centers = new_centers
        if shift < tol:
            break

    return centers