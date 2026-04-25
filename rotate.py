import os
import types
import argparse

import torch
import torch.nn as nn
from collections import defaultdict
from hadamard_utils import random_hadamard_matrix
from utils import eval_ppl_wikitext, collect_kv_wikitext
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

# ===================== Utils ======================
def add_model_type(model):
    name = model.config._name_or_path.lower()
    if 'llama-2' in name:
        model.model_type = 'llama2'
    elif 'opt' in name:
        model.model_type = 'opt'
    else:
        raise ValueError(f"Unsupported Model: {model.config._name_or_path}")

class RMSN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

# =================== Quantization ===================
def vq_quantize(kv, codebook, n_channel=4):
    B, H, T, D = kv.shape
    G = D // n_channel
    K, C = codebook.shape[-2], codebook.shape[-1]

    x = kv.reshape(B, H, T, G, C).permute(1, 3, 0, 2, 4).reshape(H*G, B*T, C)
    cb = codebook.reshape(H*G, K, C)

    # cdist: (H*G, B*T, K)
    dist = torch.cdist(x, cb, compute_mode='use_mm_for_euclid_dist')
    idx = dist.argmin(dim=-1)  # (H*G, B*T)

    # gather: (H*G, B*T, C)
    decoded = cb.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))

    # 복원: (H*G, B*T, C) -> (B, H, T, D)
    decoded = decoded.reshape(H, G, B, T, C).permute(2, 0, 3, 1, 4).reshape(B, H, T, D)
    return decoded.to(kv.dtype)

# ===================== LN fuse  =====================
def fuse_ln_linear(layernorm, linears):
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


def bake_mean_into_linear(linear):
    dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).to(dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = (b_ - b_.mean()).to(dtype)


def fuse_llama_norms(model):
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


def fuse_opt_norms(model):
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

    hidden = model.config.hidden_size
    for layer in model.model.decoder.layers:
        layer.self_attn_layer_norm = RMSN(hidden).to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        layer.final_layer_norm = RMSN(hidden).to(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
    model.model.decoder.final_layer_norm = RMSN(hidden).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )


def fuse_norms(model):
    if model.model_type == 'llama2':
        fuse_llama_norms(model)
    elif model.model_type == 'opt':
        fuse_opt_norms(model)


# ==================== Absorb R ====================
def absorb_R_input(linear, R):
    dtype = linear.weight.dtype
    W = linear.weight.data.double()
    linear.weight.data = (W @ R.double()).to(dtype)


def absorb_R_output(linear, R):
    dtype = linear.weight.dtype
    W = linear.weight.data.double()                    # [out, in]
    linear.weight.data = (R.double().T @ W).to(dtype)  # W <- R^T @ W  (= (W^T @ R)^T)
    if linear.bias is not None:
        b = linear.bias.data.double()                   # [out]
        linear.bias.data = (b @ R.double()).to(dtype)   # b <- b @ R


def absorb_R_into_embedding(model, R):
    if model.model_type == 'llama2':
        embs = [model.model.embed_tokens]
    elif model.model_type == 'opt':
        embs = [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    for emb in embs:
        dtype = emb.weight.dtype
        W = emb.weight.data.double()
        emb.weight.data = (W @ R.double()).to(dtype)


# ==================== Online R ====================
def patch_online_rotate(linear, R):
    R_local = R
    def forward_fn(self, x):
        x = x @ R_local.to(x.dtype)
        return torch.nn.functional.linear(x, self.weight, self.bias)
    linear.forward = types.MethodType(forward_fn, linear)


# ==================== LLaMA QK head-dim Rotate ====================
def patch_llama_attention(attn_module, R_head, layer_idx, hook):
    def rotate_head(query_states, key_states, value_states):
        R_h = R_head.to(query_states.dtype)
        query_states = query_states @ R_h
        key_states = key_states @ R_h
        if hook.collect == True:
            key_states.retain_grad()  
            value_states.retain_grad()  
            hook.v[layer_idx].append(value_states)
            hook.k_ropes[layer_idx].append(key_states)
        if hook.cq == True:
            v_cb = hook.v_cb[layer_idx].to(value_states.device, value_states.dtype)
            value_states = vq_quantize(value_states, v_cb, hook.channel)

            if not hook.pre_rope:
                kr_cb = hook.kr_cb[layer_idx].to(key_states.device, key_states.dtype)
                key_states = vq_quantize(key_states, kr_cb, hook.channel)
        return query_states, key_states, value_states
    def patched_forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False, **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if hook.collect == True:
            R_h = R_head.to(key_states.dtype)
            k_pre_rot = key_states @ R_h
            hook.k[layer_idx].append(k_pre_rot.detach().cpu())
        if hook.cq and hook.pre_rope:
            R_h = R_head.to(key_states.dtype)
            k_pre_rot = key_states @ R_h
            k_cb = hook.k_cb[layer_idx].to(key_states.device, key_states.dtype)
            k_pre_rot_q = vq_quantize(k_pre_rot, k_cb, hook.channel)
            key_states = k_pre_rot_q @ R_h.T

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        query_states, key_states, value_states = rotate_head(query_states, key_states, value_states)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    attn_module.forward = types.MethodType(patched_forward, attn_module)


# ==================== OPT QK head-dim Rotate ====================
def patch_opt_attention(attn_module, R_head, layer_idx, hook):
    def rotate_head(query_states, key_states, value_states):
        R_h = R_head.to(query_states.dtype)
        query_states = query_states @ R_h
        key_states = key_states @ R_h
        if hook.collect == True:
            key_states.retain_grad()  
            value_states.retain_grad()  
            hook.k[layer_idx].append(key_states)
            hook.v[layer_idx].append(value_states)
        return query_states, key_states, value_states
    def patched_forward(
        self, hidden_states, key_value_states=None, past_key_value=None,
        attention_mask=None, layer_head_mask=None, output_attentions=False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            query_states, key_states, value_states = rotate_head(query_states, key_states, value_states)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            query_states, key_states, value_states = rotate_head(query_states, key_states, value_states)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    attn_module.forward = types.MethodType(patched_forward, attn_module)


# ==================== 전체 적용 ====================
def apply_llama_rotate(model, device, hook):
    hidden = model.config.hidden_size
    intermediate = model.config.intermediate_size
    head_dim = hidden // model.config.num_attention_heads

    R_res = random_hadamard_matrix(hidden, device=device)
    R_mlp = random_hadamard_matrix(intermediate, device=device)
    R_head = random_hadamard_matrix(head_dim, device=device)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_res)
        for proj in [mlp.gate_proj, mlp.up_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.o_proj, R_res)
        absorb_R_output(mlp.down_proj, R_res)

        absorb_R_input(mlp.down_proj, R_mlp)
        patch_online_rotate(mlp.down_proj, R_mlp)

        patch_llama_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_mlp, R_head
    torch.cuda.empty_cache()


def apply_opt_rotate(model, device, hook):
    hidden = model.config.hidden_size
    ffn_dim = model.config.ffn_dim
    head_dim = hidden // model.config.num_attention_heads

    if model.lm_head.weight.data_ptr() == model.model.decoder.embed_tokens.weight.data_ptr():
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())

    R_res = random_hadamard_matrix(hidden, device=device)
    R_ffn = random_hadamard_matrix(ffn_dim, device=device)
    R_head = random_hadamard_matrix(head_dim, device=device)

    absorb_R_into_embedding(model, R_res)

    for layer_idx, layer in enumerate(model.model.decoder.layers):
        attn = layer.self_attn
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            absorb_R_input(proj, R_res)
        absorb_R_output(attn.out_proj, R_res)
        absorb_R_input(layer.fc1, R_res)

        absorb_R_output(layer.fc2, R_res)

        absorb_R_input(layer.fc2, R_ffn)
        patch_online_rotate(layer.fc2, R_ffn)

        patch_opt_attention(attn, R_head, layer_idx, hook)

    absorb_R_input(model.lm_head, R_res)

    del R_res, R_ffn, R_head
    torch.cuda.empty_cache()


def apply_rotate(model, device, hook):
    add_model_type(model)
    fuse_norms(model)
    if model.model_type == 'llama2':
        apply_llama_rotate(model, device, hook)
    elif model.model_type == 'opt':
        apply_opt_rotate(model, device, hook)

class Hook:
    collect = False
    cq = False
    k_cb = None
    v_cb = None
    kr_cb = None
    channel = 4
    pre_rope = False
    k = defaultdict(list)
    k_ropes = defaultdict(list)
    v = defaultdict(list)
    k_grad = defaultdict(list)
    k_ropes_grad = defaultdict(list)
    v_grad = defaultdict(list)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HF model id (e.g. meta-llama/Llama-2-7b-hf, facebook/opt-1.3b, facebook/opt-6.7b)")
    p.add_argument("--ppl_seq_len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_rotate", action="store_true", help="baseline PPL (no rotation, no fuse)")
    p.add_argument("--collect", action="store_true", help="Collecting KV")
    p.add_argument("--cq", action="store_true", help="Coupled Quantization KV")
    p.add_argument("--pre_rope", action="store_true", help="Quantization Pre KV")
    p.add_argument("--channel", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    def skip(*args, **kwargs): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model_name = args.model.replace("/", "_")+'.pt'

    if hasattr(model.config, 'do_layer_norm_before') and not model.config.do_layer_norm_before:
        raise ValueError("post-LN OPT (e.g. opt-350m) is not supported. Use opt-1.3b or larger.")

    hook = Hook()
    hook.collect = args.collect
    hook.cq = args.cq
    if (args.collect and args.cq):
        raise ValueError("activate only one")
    if hook.cq:
        codebook_path = os.path.join('codebooks', model_name)
        hook.k_cb, hook.v_cb, hook.kr_cb = torch.load(codebook_path)
        hook.channel = args.channel
        hook.pre_rope = args.pre_rope
    
    if not args.no_rotate:
        print("Rotate Model")
        apply_rotate(model, args.device, hook)
    if args.collect == True:
        import numpy as np
        print("Collect Activation from Wikitext2")
        collect_kv_wikitext(model, tokenizer, hook)
        folder_name = 'activations'
        
        os.makedirs(folder_name, exist_ok=True)
        file_path = os.path.join(folder_name, model_name)

        _k = hook.k
        _v = hook.v
        _k_g = hook.k_grad
        _v_g = hook.v_grad
        t_k = torch.from_numpy(np.array([_k[d] for d in sorted(_k)])).squeeze(2)
        t_v = torch.from_numpy(np.array([_v[d] for d in sorted(_v)])).squeeze(2)
        t_v_g = torch.from_numpy(np.array([_v_g[d] for d in sorted(_v_g)])).squeeze(2)

        if model.model_type == 'llama2':
            _k_ropes = hook.k_ropes
            _k_ropes_grad = hook.k_ropes_grad
            t_k_ropes = torch.from_numpy(np.array([_k_ropes[d] for d in sorted(_k_ropes)])).squeeze(2)
            t_k_ropes_grad = torch.from_numpy(np.array([_k_ropes_grad[d] for d in sorted(_k_ropes_grad)])).squeeze(2)
            torch.save((t_k, t_k_ropes_grad, t_v, t_v_g, t_k_ropes), file_path)
        else:
            t_k_g = torch.from_numpy(np.array([_k_g[d] for d in sorted(_k_g)])).squeeze(2)
            torch.save((t_k, t_k_g, t_v, t_v_g), file_path)

        print(f"Activation Saved At {file_path}")
    else:
        print("\n--- PPL evaluation on WikiText-2 ---")
        ppl = eval_ppl_wikitext(model, tokenizer, seq_len=args.ppl_seq_len, device=args.device)
        print(f"\n[{args.model}] PPL: {ppl:.4f}")

if __name__=='__main__':
    main()