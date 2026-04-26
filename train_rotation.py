import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import transformers

from utils import convert2fp16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--act_dir", type=str, default="activations")
    p.add_argument("--out_dir", type=str, default="orthogonal_matrices")
    p.add_argument("--group_size", type=int, default=128,
                   help="BFP group size")
    p.add_argument("--mant_bits", type=int, default=8,
                   help="BFP mantissa bit 수")
    p.add_argument("--loss", type=str, default="bfp_mse",
                   choices=["bfp_mse", "bfp_relative_mse", "group_variance"],
                   help="orthogonal matrix 학습 loss")
    p.add_argument("--block_size", type=int, default=None,
                   help="deprecated alias of --group_size")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--atol", type=float, default=1e-8)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--max_samples", type=int, default=32768)
    return p.parse_args()


def _model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _act_path(act_root: str, kind: str, rotated: bool) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    return os.path.join(act_root, f"{kind}_{rot_tag}.pt")


def _matrix_path(out_root: str, kind: str, rotated: bool, group_size: int) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    return os.path.join(out_root, f"{kind}_{rot_tag}_gs{group_size}.pt")


def _flatten_samples(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    if max_samples > 0 and x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]
    return x


def _check_group_size(Y: torch.Tensor, group_size: int) -> None:
    D = Y.shape[-1]
    if D % group_size != 0:
        raise ValueError(f"hidden dim {D} is not divisible by group_size={group_size}")


def bfp_group_variance_loss(Y: torch.Tensor, group_size: int) -> torch.Tensor:
    """BFP 공유 exponent 단위(group) 안에서 |activation| 분산을 낮춘다."""
    _check_group_size(Y, group_size)

    num_groups = Y.shape[-1] // group_size
    Y_abs = Y.abs()
    Y_group = Y_abs.reshape(*Y_abs.shape[:-1], num_groups, group_size)
    return Y_group.var(dim=-1, unbiased=False).mean()


def bfp_reconstruction_loss(
    Y: torch.Tensor, group_size: int, mant_bits: int,
    relative: bool = False, eps: float = 1e-6,
) -> torch.Tensor:
    """convert2fp16으로 만든 BFP target과의 reconstruction error를 직접 낮춘다."""
    _check_group_size(Y, group_size)

    with torch.no_grad():
        Y_bfp, _, _ = convert2fp16(
            Y.detach(), block_size=group_size, mbits=mant_bits,
        )
        Y_bfp = Y_bfp.to(dtype=Y.dtype, device=Y.device)

    err = Y - Y_bfp
    if relative:
        err = err / Y.detach().abs().clamp_min(eps)
    return err.pow(2).mean()


def rotation_loss(Y: torch.Tensor, group_size: int, mant_bits: int,
                  loss_type: str) -> torch.Tensor:
    if loss_type == "bfp_mse":
        return bfp_reconstruction_loss(Y, group_size, mant_bits, relative=False)
    if loss_type == "bfp_relative_mse":
        return bfp_reconstruction_loss(Y, group_size, mant_bits, relative=True)
    if loss_type == "group_variance":
        return bfp_group_variance_loss(Y, group_size)
    raise ValueError(f"Unsupported loss: {loss_type}")


def train_orthogonal_matrix(
    X, group_size, mant_bits=8, loss_type="bfp_mse",
    num_steps=1000, lr=1e-2, tol=1e-4, atol=1e-8, patience=100,
    verbose=True,
):
    D = X.shape[-1]
    assert D % group_size == 0

    device = X.device
    X_fp32 = X.float()

    A = nn.Parameter(torch.randn(D, D, device=device, dtype=torch.float32) * 0.01)
    I = torch.eye(D, device=device, dtype=torch.float32)
    optimizer = optim.Adam([A], lr=lr)

    with torch.no_grad():
        init_loss = rotation_loss(X_fp32, group_size, mant_bits, loss_type).item()
    if verbose:
        print(f"    identity {loss_type} loss {init_loss:.6f}")

    best_loss, bad = float('inf'), 0
    best_Q = None
    for step in range(num_steps):
        optimizer.zero_grad()
        S_mat = A - A.T
        Q = torch.linalg.solve(I + S_mat, I - S_mat)

        Y = X_fp32 @ Q
        loss = rotation_loss(Y, group_size, mant_bits, loss_type)

        loss.backward()
        optimizer.step()

        cur = loss.item()
        # 개선 조건: relative AND absolute 둘 다 의미있는 감소일 때만 "개선"
        threshold = best_loss - max(best_loss * tol, atol)
        if best_loss == float('inf') or cur < threshold:
            best_loss, bad = cur, 0
            best_Q = Q.detach().clone()
        else:
            bad += 1
        if bad >= patience:
            if verbose:
                print(f"    early stop @ step {step} | loss {cur:.6f}")
            break
        if verbose and (step % 100 == 0):
            print(f"    step {step:5d} | loss {cur:.6f}")

    if best_Q is not None:
        return best_Q, best_loss

    with torch.no_grad():
        S_mat = A - A.T
        Q = torch.linalg.solve(I + S_mat, I - S_mat)
    return Q.detach(), best_loss


def _train_one_matrix(x: torch.Tensor, args) -> tuple[torch.Tensor, float]:
    x = _flatten_samples(x, args.max_samples).to(args.device)
    Q, loss = train_orthogonal_matrix(
        x,
        group_size=args.group_size,
        mant_bits=args.mant_bits,
        loss_type=args.loss,
        num_steps=args.num_steps,
        lr=args.lr,
        tol=args.tol,
        atol=args.atol,
        patience=args.patience,
    )
    return Q.cpu(), loss


def _train_layerwise(x: torch.Tensor, args, kind: str) -> torch.Tensor:
    matrices = []
    for layer_idx in range(x.shape[0]):
        print(f"  {kind} layer {layer_idx}")
        Q, loss = _train_one_matrix(x[layer_idx], args)
        print(f"    best loss {loss:.6f}")
        matrices.append(Q)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.stack(matrices, dim=0)


def _train_and_save(act_root: str, out_root: str, kind: str,
                    rotated: bool, args) -> None:
    path = _act_path(act_root, kind, rotated)
    if not os.path.exists(path):
        return

    print(f"\n=== {kind} rotated={rotated} ===")
    x = torch.load(path, map_location="cpu")
    if x.shape[-1] % args.group_size != 0:
        raise ValueError(
            f"{kind} hidden dim {x.shape[-1]} is not divisible by group_size={args.group_size}"
        )

    if kind == 'lm_head_input':
        Q, loss = _train_one_matrix(x, args)
        print(f"  best loss {loss:.6f}")
        matrices = Q
    else:
        matrices = _train_layerwise(x, args, kind)

    out_path = _matrix_path(out_root, kind, rotated, args.group_size)
    torch.save(matrices, out_path)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    if args.block_size is not None:
        args.group_size = args.block_size
    transformers.set_seed(args.seed)

    model_dir = _model_dir_name(args.model)
    act_root = os.path.join(args.act_dir, model_dir)
    out_root = os.path.join(args.out_dir, model_dir)
    os.makedirs(out_root, exist_ok=True)

    for rotated in (True, False):
        for kind in ('self_attn_input', 'mlp_input', 'lm_head_input'):
            _train_and_save(act_root, out_root, kind, rotated, args)


if __name__ == '__main__':
    main()
