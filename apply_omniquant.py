import argparse
import importlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed


class _PrintLogger:
    def info(self, msg):
        print(msg)


class _SpinKVLM:
    def __init__(self, model, tokenizer, device, seqlen):
        self.model = model
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self.seqlen = seqlen
        self.batch_size_per_gpu = 1

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--variant", type=str,
                   choices=["base", "hadamard", "orthogonal", "all"],
                   default="all")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--omniquant_path", type=str, default=os.environ.get("OMNIQUANT_PATH", "OmniQuant"),
                   help="OpenGVLab/OmniQuant repo path")
    p.add_argument("--orth_dir", type=str, default="orthogonal_matrices")
    p.add_argument("--output_dir", type=str, default="omniquant_logs")
    p.add_argument("--save_dir", type=str, default=None,
                   help="directory for fake-quantized HF checkpoints")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_ppl", action="store_true")
    p.add_argument("--tasks", type=str, default="")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--real_quant", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--attn_implementation", type=str, default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--abits", type=int, default=16)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--lwc", action="store_true", default=True)
    p.add_argument("--let", action="store_true", default=False)
    p.add_argument("--symmetric", action="store_true", default=False)
    p.add_argument("--disable_zero_point", action="store_true", default=False)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lwc_lr", type=float, default=1e-2)
    p.add_argument("--let_lr", type=float, default=5e-3)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--aug_loss", action="store_true", default=False)
    p.add_argument("--deactive_amp", action="store_true", default=False)
    return p.parse_args()


def _disable_init():
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def _model_dir_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _add_model_type(model) -> None:
    name = model.config._name_or_path.lower()
    if "llama-2" in name:
        model.model_type = "llama2"
    elif "opt" in name:
        model.model_type = "opt"
    else:
        raise ValueError(f"Unsupported Model: {model.config._name_or_path}")


def _variants(selected: str):
    if selected == "all":
        return ["base", "hadamard", "orthogonal"]
    return [selected]


def _import_omniquant(omniquant_path: str):
    root = Path(omniquant_path).expanduser().resolve()
    if not (root / "main.py").exists() or not (root / "quantize" / "omniquant.py").exists():
        raise FileNotFoundError(
            f"OpenGVLab/OmniQuant repo not found at {root}. "
            "Clone https://github.com/OpenGVLab/OmniQuant and pass --omniquant_path."
        )

    sys.path.insert(0, str(root))
    omniquant_mod = importlib.import_module("quantize.omniquant")
    datautils_mod = importlib.import_module("datautils")
    return omniquant_mod.omniquant, datautils_mod.get_loaders


def _prefer_spinkv_imports(omniquant_path: str) -> None:
    root = str(Path(omniquant_path).expanduser().resolve())
    while root in sys.path:
        sys.path.remove(root)
    sys.path.append(root)

    utils_mod = sys.modules.get("utils")
    utils_file = getattr(utils_mod, "__file__", "") if utils_mod is not None else ""
    if utils_file and str(Path(utils_file).resolve()).startswith(root):
        del sys.modules["utils"]


def _load_model(model_id: str, device: str, attn_implementation: str):
    print(f"Loading {model_id} ...")
    config = AutoConfig.from_pretrained(model_id, attn_implementation=attn_implementation)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    _add_model_type(model)
    if hasattr(model.config, "do_layer_norm_before") and not model.config.do_layer_norm_before:
        raise ValueError("post-LN OPT (e.g. opt-350m) is not supported. Use opt-1.3b or larger.")
    return model, tokenizer


def _make_hook(args, model_dir: str):
    return SimpleNamespace(
        collect=False,
        cq=False,
        bfp=False,
        offline=True,
        qk_rotate=None,
        orth_dir=args.orth_dir,
        model_dir=model_dir,
    )


def _prepare_variant(model, args, model_dir: str, variant: str):
    if variant == "base":
        return

    _prefer_spinkv_imports(args.omniquant_path)
    from rotquant.apply import apply_rotate

    hook = _make_hook(args, model_dir)
    rotate_device = "cpu"
    if variant == "hadamard":
        # Hadamard absorbed-weight model. QK online rotate and FFN online rotate are disabled.
        hook.offline = True
        hook.qk_rotate = None
        apply_rotate(model, rotate_device, hook, rotate="hadamard")
    elif variant == "orthogonal":
        # Orthogonal matrices are absorbed where possible; residual basis changes stay as SpinKV runtime patches.
        apply_rotate(model, rotate_device, hook, rotate="orthogonal")
    else:
        raise ValueError(f"Unsupported variant: {variant}")


def _build_omni_args(args, variant: str):
    output_dir = Path(args.output_dir) / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    save_dir = None
    if args.save_dir is not None:
        save_dir = str(Path(args.save_dir) / variant)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    net = args.model.split("/")[-1]
    model_family = net.split("-")[0]
    deactive_amp = args.deactive_amp
    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        deactive_amp = True

    omni_args = SimpleNamespace(
        model=args.model,
        cache_dir=args.cache_dir,
        output_dir=str(output_dir),
        save_dir=save_dir,
        resume=args.resume,
        real_quant=args.real_quant,
        calib_dataset=args.calib_dataset,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        seed=args.seed,
        tasks=args.tasks,
        eval_ppl=args.eval_ppl,
        num_fewshot=args.num_fewshot,
        wbits=args.wbits,
        abits=args.abits,
        group_size=args.group_size,
        alpha=args.alpha,
        let_lr=args.let_lr,
        lwc_lr=args.lwc_lr,
        wd=args.wd,
        epochs=args.epochs,
        let=args.let,
        lwc=args.lwc,
        aug_loss=args.aug_loss,
        symmetric=args.symmetric,
        disable_zero_point=args.disable_zero_point,
        a_dynamic_method="per_token",
        w_dynamic_method="per_channel",
        limit=args.limit,
        multigpu=False,
        deactive_amp=deactive_amp,
        attn_implementation=args.attn_implementation,
        net=net,
        model_family=model_family,
        act_scales=None,
        act_shifts=None,
    )

    omni_args.weight_quant_params = {
        "n_bits": omni_args.wbits,
        "per_channel_axes": [0],
        "symmetric": omni_args.symmetric,
        "dynamic_method": omni_args.w_dynamic_method,
        "group_size": omni_args.group_size,
        "lwc": omni_args.lwc,
        "disable_zero_point": omni_args.disable_zero_point,
    }
    omni_args.act_quant_params = {
        "n_bits": omni_args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": omni_args.a_dynamic_method,
    }
    omni_args.q_quant_params = dict(omni_args.act_quant_params)
    omni_args.k_quant_params = dict(omni_args.act_quant_params)
    omni_args.v_quant_params = dict(omni_args.act_quant_params)
    omni_args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    return omni_args


def _save_quantized_model(lm, omni_args, omniquant_path: str):
    if omni_args.save_dir is None:
        return

    # Reuse OmniQuant cleanup semantics before save_pretrained.
    root = Path(omniquant_path).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    int_linear = importlib.import_module("quantize.int_linear")
    int_llama_layer = importlib.import_module("models.int_llama_layer")
    int_opt_layer = importlib.import_module("models.int_opt_layer")
    QuantLinear = int_linear.QuantLinear
    QuantLlamaDecoderLayer = int_llama_layer.QuantLlamaDecoderLayer
    QuantOPTDecoderLayer = int_opt_layer.QuantOPTDecoderLayer

    for module in lm.model.modules():
        if isinstance(module, QuantLinear):
            if hasattr(module.weight_quantizer, "lowbound_factor"):
                del module.weight_quantizer.lowbound_factor
            if hasattr(module.weight_quantizer, "upbound_factor"):
                del module.weight_quantizer.upbound_factor
        if isinstance(module, (QuantLlamaDecoderLayer, QuantOPTDecoderLayer)) and omni_args.let:
            for name in [
                "qkv_smooth_scale", "qkv_smooth_shift",
                "out_smooth_scale", "out_smooth_shift",
                "fc1_smooth_scale", "fc1_smooth_shift",
            ]:
                if hasattr(module, name):
                    delattr(module, name)

    lm.model.save_pretrained(omni_args.save_dir)
    lm.tokenizer.save_pretrained(omni_args.save_dir)
    print(f"Saved quantized model: {omni_args.save_dir}")


def _basis_change(x: torch.Tensor, R_from: torch.Tensor, R_to: torch.Tensor) -> torch.Tensor:
    return (x @ R_from.to(device=x.device, dtype=x.dtype).T) @ R_to.to(device=x.device, dtype=x.dtype)


def _patch_omniquant_layers_for_spinkv(omniquant_path: str) -> None:
    root = Path(omniquant_path).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    int_llama_layer = importlib.import_module("models.int_llama_layer")
    int_opt_layer = importlib.import_module("models.int_opt_layer")
    QuantLlamaDecoderLayer = int_llama_layer.QuantLlamaDecoderLayer
    QuantOPTDecoderLayer = int_opt_layer.QuantOPTDecoderLayer

    if not getattr(QuantLlamaDecoderLayer, "_spinkv_patched", False):
        orig_init = QuantLlamaDecoderLayer.__init__

        def llama_init(self, config, ori_layer, args):
            orig_init(self, config, ori_layer, args)
            for name in ("_spinkv_R_attn", "_spinkv_R_mlp", "_spinkv_R_next"):
                if hasattr(ori_layer, name):
                    setattr(self, name, getattr(ori_layer, name))

        def llama_forward(
            self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
        ):
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            if hasattr(self, "_spinkv_R_attn"):
                residual = _basis_change(residual, self._spinkv_R_attn, self._spinkv_R_mlp)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if hasattr(self, "_spinkv_R_mlp"):
                residual = _basis_change(residual, self._spinkv_R_mlp, self._spinkv_R_next)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        QuantLlamaDecoderLayer.__init__ = llama_init
        QuantLlamaDecoderLayer.forward = llama_forward
        QuantLlamaDecoderLayer._spinkv_patched = True

    if not getattr(QuantOPTDecoderLayer, "_spinkv_patched", False):
        orig_init = QuantOPTDecoderLayer.__init__

        def opt_init(self, config, ori_layer, args):
            orig_init(self, config, ori_layer, args)
            for name in ("_spinkv_R_attn", "_spinkv_R_mlp", "_spinkv_R_next"):
                if hasattr(ori_layer, name):
                    setattr(self, name, getattr(ori_layer, name))

        def opt_forward(
            self, hidden_states, attention_mask=None, layer_head_mask=None,
            output_attentions=False, use_cache=False, past_key_value=None,
        ):
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = torch.nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training,
            )
            if hasattr(self, "_spinkv_R_attn"):
                residual = _basis_change(residual, self._spinkv_R_attn, self._spinkv_R_mlp)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.fc2(hidden_states)
            hidden_states = torch.nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training,
            )
            if hasattr(self, "_spinkv_R_mlp"):
                residual = _basis_change(residual, self._spinkv_R_mlp, self._spinkv_R_next)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs

        QuantOPTDecoderLayer.__init__ = opt_init
        QuantOPTDecoderLayer.forward = opt_forward
        QuantOPTDecoderLayer._spinkv_patched = True


def _run_variant(args, variant: str, omniquant_fn, get_loaders):
    print(f"\n=== {variant} | official OmniQuant W4A16gs128 ===")
    if variant == "orthogonal":
        _patch_omniquant_layers_for_spinkv(args.omniquant_path)
    model_dir = _model_dir_name(args.model)
    model, tokenizer = _load_model(args.model, args.device, args.attn_implementation)
    _prepare_variant(model, args, model_dir, variant)

    omni_args = _build_omni_args(args, variant)
    lm = _SpinKVLM(model, tokenizer, args.device, args.seqlen)
    lm.model.config.use_cache = False

    cache_dataloader = (
        f"{omni_args.cache_dir}/dataloader_{omni_args.model_family}_"
        f"{omni_args.calib_dataset}_{omni_args.nsamples}_{args.seqlen}.cache"
    )
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        print(f"load calibration from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            omni_args.calib_dataset,
            nsamples=omni_args.nsamples,
            seed=omni_args.seed,
            model=omni_args.model,
            seqlen=lm.seqlen,
        )
        torch.save(dataloader, cache_dataloader)

    omniquant_fn(lm, omni_args, dataloader, act_scales=None, act_shifts=None, logger=_PrintLogger())
    _save_quantized_model(lm, omni_args, args.omniquant_path)

    del lm, model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.wbits != 4 or args.abits != 16 or args.group_size != 128:
        print(f"Warning: requested W{args.wbits}A{args.abits}gs{args.group_size}, not W4A16gs128.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    set_seed(args.seed)
    _disable_init()

    omniquant_fn, get_loaders = _import_omniquant(args.omniquant_path)
    for variant in _variants(args.variant):
        _run_variant(args, variant, omniquant_fn, get_loaders)


if __name__ == "__main__":
    main()
