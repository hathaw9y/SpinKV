import os
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils import eval_ppl_wikitext, collect_kv_wikitext
from rotquant import Hook, apply_rotate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id (e.g. meta-llama/Llama-2-7b-hf, facebook/opt-1.3b, facebook/opt-6.7b)")
    p.add_argument("--ppl_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_rotate", action="store_true",
                   help="모델 회전 없이 실행 (collect/cq는 attention patch만으로 동작)")
    p.add_argument("--collect", action="store_true", help="Collecting KV")
    p.add_argument("--cq", action="store_true", help="Coupled Quantization KV")
    p.add_argument("--pre_rope", action="store_true", help="Quantization Pre KV")
    p.add_argument("--n_channel", type=int, default=4)
    p.add_argument("--n_cluster", type=int, default=256,
                   help="codebook 파일 검색용 cluster 수")
    p.add_argument("--mant", action="store_true",
                   help="mantissa codebook으로 CQ 실행")
    p.add_argument("--mant_bits", type=int, default=8)
    p.add_argument("--mant_block_size", type=int, default=128)
    p.add_argument("--act_dir", type=str, default="activations")
    p.add_argument("--cb_dir", type=str, default="codebooks")
    return p.parse_args()


def _disable_init():
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def _load_model(model_id: str, device: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    if hasattr(model.config, 'do_layer_norm_before') and not model.config.do_layer_norm_before:
        raise ValueError("post-LN OPT (e.g. opt-350m) is not supported. Use opt-1.3b or larger.")
    return model, tokenizer


def _model_dir_name(model_id: str) -> str:
    """meta-llama/Llama-2-7b-hf -> meta-llama_Llama-2-7b-hf"""
    return model_id.replace("/", "_")


def _cb_filename(kind: str, rotated: bool, n_channel: int, n_cluster: int,
                 mant: bool = False) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    if mant:
        kind = f"{kind}_mant"
    return f"{kind}_{rot_tag}_c{n_channel}_k{n_cluster}.pt"


def _act_filename(kind: str, rotated: bool) -> str:
    rot_tag = 'rot' if rotated else 'raw'
    return f"{kind}_{rot_tag}.pt"


def _build_hook(args, model_dir: str) -> Hook:
    if args.collect and args.cq:
        raise ValueError("activate only one of --collect / --cq")

    hook = Hook()
    hook.collect = args.collect
    hook.cq = args.cq
    hook.mant = args.mant
    hook.mant_bits = args.mant_bits
    hook.mant_block_size = args.mant_block_size

    if hook.cq:
        rotated = not args.no_rotate
        cb_root = os.path.join(args.cb_dir, model_dir)

        def _cb(kind):
            return os.path.join(
                cb_root,
                _cb_filename(kind, rotated, args.n_channel, args.n_cluster, args.mant),
            )

        hook.v_cb = torch.load(_cb('v'))
        hook.channel = args.n_channel
        hook.pre_rope = args.pre_rope

        if args.pre_rope:
            # pre-RoPE quantization: k codebook만 필요
            hook.k_cb = torch.load(_cb('k'))
            hook.kr_cb = None
        else:
            # post-RoPE quantization (LLaMA) or rotation 직후 (OPT)
            # OPT는 k_rope 파일이 없으니 k_cb로 fallback
            kr_path = _cb('k_rope')
            if os.path.exists(kr_path):
                hook.kr_cb = torch.load(kr_path)
                hook.k_cb = None
            else:
                # OPT 경로: opt.py는 k_cb를 사용
                hook.k_cb = torch.load(_cb('k'))
                hook.kr_cb = None
    return hook


def _stack_by_layer(d: dict) -> torch.Tensor:
    """{layer_idx: [tensor, ...]} -> stacked numpy -> torch tensor (B=1 squeeze)."""
    return torch.from_numpy(np.array([d[k] for k in sorted(d)])).squeeze(2)


def _save_one(tensor: torch.Tensor, out_dir: str, kind: str, rotated: bool) -> None:
    path = os.path.join(out_dir, _act_filename(kind, rotated))
    torch.save(tensor, path)
    print(f"  saved: {path}")


def _save_activations(model, hook: Hook, model_dir: str, rotated: bool,
                      act_dir: str) -> None:
    out_dir = os.path.join(act_dir, model_dir)
    os.makedirs(out_dir, exist_ok=True)

    # rotated tensors
    _save_one(_stack_by_layer(hook.k),      out_dir, 'k',      rotated)
    _save_one(_stack_by_layer(hook.v),      out_dir, 'v',      rotated)
    _save_one(_stack_by_layer(hook.v_grad), out_dir, 'v_grad', rotated)

    if model.model_type == 'llama2':
        _save_one(_stack_by_layer(hook.k_ropes),      out_dir, 'k_rope',      rotated)
        _save_one(_stack_by_layer(hook.k_ropes_grad), out_dir, 'k_rope_grad', rotated)
    else:
        _save_one(_stack_by_layer(hook.k_grad), out_dir, 'k_grad', rotated)

    # raw tensors
    _save_one(_stack_by_layer(hook.k_raw), out_dir, 'k', rotated=False)
    _save_one(_stack_by_layer(hook.v_raw), out_dir, 'v', rotated=False)
    if model.model_type == 'llama2':
        _save_one(_stack_by_layer(hook.k_ropes_raw), out_dir, 'k_rope', rotated=False)


def main():
    args = parse_args()
    set_seed(args.seed)
    _disable_init()

    model, tokenizer = _load_model(args.model, args.device)
    model_dir = _model_dir_name(args.model)

    hook = _build_hook(args, model_dir)

    rotate = not args.no_rotate
    print(f"Apply rotate={rotate}")
    apply_rotate(model, args.device, hook, rotate=rotate)

    if args.collect:
        print("Collect Activation from Wikitext2")
        collect_kv_wikitext(model, tokenizer, hook)
        _save_activations(model, hook, model_dir, rotated=rotate, act_dir=args.act_dir)
    else:
        print("\n--- PPL evaluation on WikiText-2 ---")
        ppl = eval_ppl_wikitext(model, tokenizer, seq_len=args.ppl_seq_len, device=args.device)
        print(f"\n[{args.model}] PPL: {ppl:.4f}")


if __name__ == '__main__':
    main()
