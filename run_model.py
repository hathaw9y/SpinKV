import os
import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from utils import eval_ppl_wikitext, collect_qkv_wikitext, collect_act_wikitext
from rotquant import Hook, apply_rotate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id (e.g. meta-llama/Llama-2-7b-hf, facebook/opt-1.3b, facebook/opt-6.7b)")
    p.add_argument("--ppl_seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--rotate", type=str, choices=["hadamard", "orthogonal"], default=None,
                   help="rotation 방식 선택")
    p.add_argument("--qk_rotate", type=str, choices=["hadamard"], default=None,
                   help="RoPE 이후 Q/K rotation 방식")
    p.add_argument("--offline", action="store_true",
                   help="MLP/FFN online rotation 비활성화")
    p.add_argument("--collect_qkv", action="store_true", help="Collecting QKV")
    p.add_argument("--collect_act", action="store_true", help="Collecting normalized activations")
    p.add_argument("--cq", action="store_true", help="Coupled Quantization KV")
    p.add_argument("--pre_rope", action="store_true", help="Quantization Pre KV")
    p.add_argument("--n_channel", type=int, default=4)
    p.add_argument("--n_cluster", type=int, default=256,
                   help="codebook 파일 검색용 cluster 수")
    p.add_argument("--mant", action="store_true",
                   help="mantissa codebook으로 CQ 실행")
    p.add_argument("--mant_bits", type=int, default=8)
    p.add_argument("--mant_block_size", type=int, default=128)
    p.add_argument("--bfp", action="store_true",
                   help="linear input activation과 attention QK matmul 입력에 bfp 적용")
    p.add_argument("--bfp_bits", type=int, default=8)
    p.add_argument("--bfp_block_size", type=int, default=128)
    p.add_argument("--weight_bfp", action="store_true",
                   help="linear weight를 W.T 기준으로 bfp 적용")
    p.add_argument("--weight_bfp_bits", type=int, default=8,
                   help="weight BFP mantissa bit 수 (--bfp_bits와 독립)")
    p.add_argument("--weight_bfp_block_size", type=int, default=128)
    p.add_argument("--act_dir", type=str, default="activations")
    p.add_argument("--cb_dir", type=str, default="codebooks")
    p.add_argument("--orth_dir", type=str, default="orthogonal_matrices")
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
    if args.collect_qkv and args.cq:
        raise ValueError("activate only one of --collect_qkv / --cq")
    if args.collect_qkv and args.collect_act:
        raise ValueError("activate only one of --collect_qkv / --collect_act")

    hook = Hook()
    hook.collect = args.collect_qkv
    hook.cq = args.cq
    hook.mant = args.mant
    hook.mant_bits = args.mant_bits
    hook.mant_block_size = args.mant_block_size
    hook.bfp = args.bfp
    hook.bfp_bits = args.bfp_bits
    hook.bfp_block_size = args.bfp_block_size
    hook.weight_bfp = args.weight_bfp
    hook.weight_bfp_bits = args.weight_bfp_bits
    hook.weight_bfp_block_size = args.weight_bfp_block_size
    hook.offline = args.offline
    hook.qk_rotate = args.qk_rotate
    hook.orth_dir = args.orth_dir
    hook.orth_group_size = args.bfp_block_size
    hook.model_dir = model_dir

    if hook.cq:
        rotated = args.rotate is not None
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
    return torch.stack([
        torch.stack([x.detach().cpu() for x in d[k]], dim=0)
        for k in sorted(d)
    ], dim=0).squeeze(2)


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
        _save_one(_stack_by_layer(hook.q_ropes),      out_dir, 'q_rope',      rotated)
        _save_one(_stack_by_layer(hook.k_ropes),      out_dir, 'k_rope',      rotated)
    _save_one(_stack_by_layer(hook.k_grad), out_dir, 'k_grad', rotated)


def _stack_list(xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(xs, dim=0)


def _stack_act_by_layer(d: dict) -> torch.Tensor:
    return torch.stack([torch.stack(d[k], dim=0) for k in sorted(d)], dim=0)


def _save_act_activations(hook: Hook, model_dir: str, rotated: bool,
                          act_dir: str) -> None:
    out_dir = os.path.join(act_dir, model_dir)
    os.makedirs(out_dir, exist_ok=True)

    _save_one(_stack_act_by_layer(hook.self_attn_input), out_dir, 'self_attn_input', rotated)
    _save_one(_stack_act_by_layer(hook.mlp_input), out_dir, 'mlp_input', rotated)
    _save_one(_stack_list(hook.lm_head_input), out_dir, 'lm_head_input', rotated)


def _register_act_hooks(model, hook: Hook) -> list:
    handles = []

    def _first_tensor(inputs, kwargs):
        if len(inputs) > 0:
            return inputs[0]
        for key in ('hidden_states', 'input', 'inputs'):
            if key in kwargs:
                return kwargs[key]
        for value in kwargs.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("forward pre-hook could not find tensor input")

    def _save_input_list(xs):
        def _hook(_module, inputs, kwargs):
            xs.append(_first_tensor(inputs, kwargs).detach().cpu())
        return _hook

    def _save_input_layer(d, layer_idx):
        def _hook(_module, inputs, kwargs):
            d[layer_idx].append(_first_tensor(inputs, kwargs).detach().cpu())
        return _hook

    if model.model_type == 'llama2':
        for layer_idx, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                _save_input_layer(hook.self_attn_input, layer_idx), with_kwargs=True,
            ))
            handles.append(layer.mlp.register_forward_pre_hook(
                _save_input_layer(hook.mlp_input, layer_idx), with_kwargs=True,
            ))
        handles.append(model.lm_head.register_forward_pre_hook(
            _save_input_list(hook.lm_head_input), with_kwargs=True,
        ))
    elif model.model_type == 'opt':
        for layer_idx, layer in enumerate(model.model.decoder.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                _save_input_layer(hook.self_attn_input, layer_idx), with_kwargs=True,
            ))
            handles.append(layer.fc1.register_forward_pre_hook(
                _save_input_layer(hook.mlp_input, layer_idx), with_kwargs=True,
            ))
        handles.append(model.lm_head.register_forward_pre_hook(
            _save_input_list(hook.lm_head_input), with_kwargs=True,
        ))
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")

    return handles


def main():
    args = parse_args()
    set_seed(args.seed)
    _disable_init()

    model, tokenizer = _load_model(args.model, args.device)
    model_dir = _model_dir_name(args.model)

    hook = _build_hook(args, model_dir)

    rotated = args.rotate is not None
    print(f"Apply rotate={args.rotate or 'none'}")
    apply_rotate(model, args.device, hook, rotate=args.rotate)

    if args.collect_qkv:
        print("Collect QKV from Wikitext2")
        collect_qkv_wikitext(model, tokenizer, hook)
        _save_activations(model, hook, model_dir, rotated=rotated, act_dir=args.act_dir)
    elif args.collect_act:
        print("Collect Normalized Activation from Wikitext2")
        handles = _register_act_hooks(model, hook)
        collect_act_wikitext(model, tokenizer, device=args.device)
        for handle in handles:
            handle.remove()
        _save_act_activations(hook, model_dir, rotated=rotated, act_dir=args.act_dir)
    else:
        print("\n--- PPL evaluation on WikiText-2 ---")
        ppl = eval_ppl_wikitext(model, tokenizer, seq_len=args.ppl_seq_len, device=args.device)
        print(f"\n[{args.model}] PPL: {ppl:.4f}")


if __name__ == '__main__':
    main()
