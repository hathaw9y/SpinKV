import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import transformers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--act_dir", type=str, default="activations")
    p.add_argument("--out_dir", type=str, default="orthogonal_matrices")
    p.add_argument("--block_size", type=int, default=128)
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


def _matrix_path(out_root: str, kind: str, rotated: bool) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    return os.path.join(out_root, f"{kind}_{rot_tag}.pt")


def _flatten_samples(x: torch.Tensor, max_samples: int) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    if max_samples > 0 and x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0])[:max_samples]
        x = x[idx]
    return x


def train_orthogonal_matrix(
    X, block_size, num_steps=1000, lr=1e-2, tol=1e-4, atol=1e-8,
    patience=100, verbose=True,
):
    D = X.shape[-1]
    assert D % block_size == 0
    num_blocks = D // block_size

    device = X.device
    X_fp32 = X.float()

    A = nn.Parameter(torch.randn(D, D, device=device, dtype=torch.float32) * 0.01)
    I = torch.eye(D, device=device, dtype=torch.float32)
    optimizer = optim.Adam([A], lr=lr)

    best_loss, bad = float('inf'), 0
    for step in range(num_steps):
        optimizer.zero_grad()
        S_mat = A - A.T
        Q = torch.linalg.solve(I + S_mat, I - S_mat)

        Y = X_fp32 @ Q
        Y_abs = Y.abs()
        Y_blocks = Y_abs.reshape(*Y_abs.shape[:-1], num_blocks, block_size)
        loss = Y_blocks.var(dim=-1, unbiased=False).mean()

        loss.backward()
        optimizer.step()

        cur = loss.item()
        # 개선 조건: relative AND absolute 둘 다 의미있는 감소일 때만 "개선"
        threshold = best_loss - max(best_loss * tol, atol)
        if best_loss == float('inf') or cur < threshold:
            best_loss, bad = cur, 0
        else:
            bad += 1
        if bad >= patience:
            if verbose:
                print(f"    early stop @ step {step} | loss {cur:.6f}")
            break
        if verbose and (step % 100 == 0):
            print(f"    step {step:5d} | loss {cur:.6f}")

    with torch.no_grad():
        S_mat = A - A.T
        Q = torch.linalg.solve(I + S_mat, I - S_mat)
    return Q.detach(), best_loss


def _train_one_matrix(x: torch.Tensor, args) -> tuple[torch.Tensor, float]:
    x = _flatten_samples(x, args.max_samples).to(args.device)
    Q, loss = train_orthogonal_matrix(
        x,
        block_size=args.block_size,
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
    if x.shape[-1] % args.block_size != 0:
        raise ValueError(
            f"{kind} hidden dim {x.shape[-1]} is not divisible by block_size={args.block_size}"
        )

    if kind == 'lm_head_input':
        Q, loss = _train_one_matrix(x, args)
        print(f"  best loss {loss:.6f}")
        matrices = Q
    else:
        matrices = _train_layerwise(x, args, kind)

    out_path = _matrix_path(out_root, kind, rotated)
    torch.save(matrices, out_path)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
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
