import torch
import torch.nn as nn


class RMSN(nn.Module):
    """No-op RMSNorm (weight=1, bias=0): fusion 후 plug-in 용."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ---------------- generic primitives ----------------
def fuse_ln_linear(layernorm: nn.Module, linears: list[nn.Linear]) -> None:
    """LN의 weight/bias를 후속 Linear에 흡수시키고 LN을 identity로 만든다."""
    for lin in linears:
        dtype = lin.weight.dtype
        W_ = lin.weight.data.double()

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if lin.bias is None:
                lin.bias = nn.Parameter(
                    torch.zeros(lin.out_features, dtype=torch.float64, device=lin.weight.device)
                )
            b_ = lin.bias.data.double()
            lin.bias.data = (b_ + W_ @ layernorm.bias.data.double()).to(dtype)

        lin.weight.data = (W_ * layernorm.weight.data.double()).to(dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, 'bias') and layernorm.bias is not None:
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def bake_mean_into_linear(linear: nn.Linear) -> None:
    """OPT의 mean-subtraction을 linear weight/bias에 흡수."""
    dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).to(dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = (b_ - b_.mean()).to(dtype)


# ---------------- model-specific ----------------
def fuse_llama_norms(model) -> None:
    for layer in model.model.layers:
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_ln_linear(
            layer.post_attention_layernorm,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )
    fuse_ln_linear(model.model.norm, [model.lm_head])


def fuse_opt_norms(model) -> None:
    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    for emb in [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]:
        W_ = emb.weight.data.double()
        emb.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(emb.weight.dtype)

    for layer in model.model.decoder.layers:
        fuse_ln_linear(
            layer.self_attn_layer_norm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        bake_mean_into_linear(layer.self_attn.out_proj)
        bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(model.model.decoder.final_layer_norm, [model.lm_head])

    # LayerNorm -> RMSN 교체
    hidden = model.config.hidden_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def _rmsn():
        return RMSN(hidden).to(device=device, dtype=dtype)

    for layer in model.model.decoder.layers:
        layer.self_attn_layer_norm = _rmsn()
        layer.final_layer_norm = _rmsn()
    model.model.decoder.final_layer_norm = _rmsn()


def fuse_norms(model) -> None:
    if model.model_type == 'llama2':
        fuse_llama_norms(model)
    elif model.model_type == 'opt':
        fuse_opt_norms(model)
    else:
        raise ValueError(f"Unsupported model_type: {model.model_type}")