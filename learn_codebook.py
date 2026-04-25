import os
import argparse
import torch
import transformers

from fisher import fisher_codebook_batched
from fisher.activations import _merge_batch_seq, _is_llama


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_channel", type=int, default=4)
    p.add_argument("--n_cluster", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--act_dir", type=str, default="activations")
    p.add_argument("--out_dir", type=str, default="codebooks")
    p.add_argument("--mant", action="store_true",
                   help="activation mantissa로 codebook 학습")
    p.add_argument("--mant_bits", type=int, default=8)
    p.add_argument("--mant_block_size", type=int, default=128)
    return p.parse_args()


def _model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _act_path(act_root: str, kind: str, rotated: bool) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    return os.path.join(act_root, f"{kind}_{rot_tag}.pt")


def _load(act_root: str, kind: str, rotated: bool) -> torch.Tensor:
    """run_ppl.py가 저장한 activation을 (L, H, B*S, D)로 변환해 로드."""
    x = torch.load(_act_path(act_root, kind, rotated))
    return _merge_batch_seq(x)


def convert2fp16(x: torch.Tensor, block_size: int = 128,
                 mbits: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    flat = x.reshape(*x.shape[:-1], -1, block_size).half()

    int_bits = flat.view(torch.int16)
    elem_exp = ((int_bits >> 10) & 0x1F)
    shared_exp = elem_exp.max(dim=-1, keepdim=True).values

    shift = (shared_exp - elem_exp).clamp(min=0, max=10)

    mantissa = (int_bits & 0x03FF) | 0x0400
    mantissa_shifted = mantissa >> shift

    # 각 원소별 첫번째 1 다음 mbits-1개 비트만 남기고 나머지 0
    truncate_bits = 11 - mbits + 1
    round_bit = (mantissa_shifted >> (truncate_bits - 1)) & 1
    mantissa_truncated = ((mantissa_shifted >> truncate_bits) + round_bit)

    sign = ((int_bits >> 15) & 0x1).half()
    mantissa_signed = mantissa_truncated.to(torch.int16) * (1 - 2 * sign)

    restored = (1 - 2 * sign).half() \
            * (mantissa_truncated << truncate_bits).half() / 1024.0 \
            * (2.0 ** (shared_exp - 15)).half()

    real_exp = (2 ** (shared_exp - 15).half()).expand(*shared_exp.shape[:-1], block_size)

    return restored.reshape(shape), mantissa_signed.reshape(shape), real_exp.reshape(shape)


def _cb_path(out_root: str, kind: str, rotated: bool,
             n_channel: int, n_cluster: int, mant: bool = False) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    if mant:
        kind = f"{kind}_mant"
    return os.path.join(out_root, f"{kind}_{rot_tag}_c{n_channel}_k{n_cluster}.pt")


def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    model_dir = _model_dir_name(args.model)
    act_root = os.path.join(args.act_dir, model_dir)
    out_root = os.path.join(args.out_dir, model_dir)
    os.makedirs(out_root, exist_ok=True)

    is_llama = _is_llama(args.model)

    def _learn_and_save(act_kind: str, grad_kind: str, cb_kind: str, rotated: bool):
        x = _load(act_root, act_kind, rotated)
        # gradient는 항상 rotated 도메인의 것을 Fisher weight로 사용
        x_g = _load(act_root, grad_kind, rotated=True)
        cb = fisher_codebook_batched(
            x, x_g,
            n_channel=args.n_channel,
            n_cluster=args.n_cluster,
            device=args.device,
        )
        path = _cb_path(out_root, cb_kind, rotated, args.n_channel, args.n_cluster)
        torch.save(cb, path)
        print(f"Saved: {path}")

    def _learn_mant_and_save(act_kind: str, grad_kind: str, cb_kind: str, rotated: bool):
        x = _load(act_root, act_kind, rotated)
        _, x_mant, _ = convert2fp16(
            x, block_size=args.mant_block_size, mbits=args.mant_bits,
        )
        # gradient는 항상 rotated 도메인의 것을 Fisher weight로 사용
        x_g = _load(act_root, grad_kind, rotated=True)
        cb = fisher_codebook_batched(
            x_mant.half(), x_g,
            n_channel=args.n_channel,
            n_cluster=args.n_cluster,
            device=args.device,
        )
        path = _cb_path(out_root, cb_kind, rotated, args.n_channel, args.n_cluster,
                        mant=True)
        torch.save(cb, path)
        print(f"Saved: {path}")

    if is_llama:
        targets = [
            ('k',      'k_rope_grad', 'k'),
            ('v',      'v_grad',      'v'),
            ('k_rope', 'k_rope_grad', 'k_rope'),
        ]
    else:
        targets = [
            ('k', 'k_grad', 'k'),
            ('v', 'v_grad', 'v'),
        ]

    for rotated in (True, False):
        print(f"\n=== rotated={rotated} ===")
        for act_kind, grad_kind, cb_kind in targets:
            if args.mant:
                _learn_mant_and_save(act_kind, grad_kind, cb_kind, rotated)
            else:
                _learn_and_save(act_kind, grad_kind, cb_kind, rotated)


if __name__ == '__main__':
    main()
