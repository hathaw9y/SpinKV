import types
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from utils import bfp_quantize_activation
from ..quantization import vq_quantize, vq_quantize_mantissa


def patch_llama_attention(attn_module, R_head, layer_idx: int, hook) -> None:
    """
    LLaMA self-attention patch.
    R_head=None이면 head-dim Hadamard 회전을 적용하지 않음 (no_rotate 모드).
    """
    rotate = R_head is not None

    def _maybe_quantize_v(value_states):
        if not hook.cq:
            return value_states
        v_cb = hook.v_cb[layer_idx].to(value_states.device, value_states.dtype)
        if hook.mant:
            return vq_quantize_mantissa(
                value_states, v_cb, hook.channel, hook.mant_bits, hook.mant_block_size,
            )
        return vq_quantize(value_states, v_cb, hook.channel)

    def _maybe_quantize_k_post_rope(key_states):
        if not (hook.cq and not hook.pre_rope):
            return key_states
        kr_cb = hook.kr_cb[layer_idx].to(key_states.device, key_states.dtype)
        if hook.mant:
            return vq_quantize_mantissa(
                key_states, kr_cb, hook.channel, hook.mant_bits, hook.mant_block_size,
            )
        return vq_quantize(key_states, kr_cb, hook.channel)

    def _maybe_collect_post_rope(key_states, value_states, key_raw, value_raw):
        """post-RoPE 단계에서 rotated/raw 모두 수집."""
        if not hook.collect:
            return
        key_states.retain_grad()
        value_states.retain_grad()
        hook.v[layer_idx].append(value_states)
        hook.k_ropes[layer_idx].append(key_states)
        # raw post-RoPE는 detached cpu로 저장 (grad 불필요)
        hook.k_ropes_raw[layer_idx].append(key_raw.detach().cpu())
        hook.v_raw[layer_idx].append(value_raw.detach().cpu())

    def rotate_head(query_states, key_states, value_states,
                    key_raw_post, value_raw_post):
        if rotate:
            R_h = R_head.to(query_states.dtype)
            query_states = query_states @ R_h
            key_states = key_states @ R_h
        _maybe_collect_post_rope(key_states, value_states, key_raw_post, value_raw_post)
        value_states = _maybe_quantize_v(value_states)
        key_states = _maybe_quantize_k_post_rope(key_states)
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

        # ---- pre-RoPE collection / quantization ----
        if hook.collect:
            # rotated pre-RoPE key (rotate=False면 raw와 동일)
            if rotate:
                R_h = R_head.to(key_states.dtype)
                k_pre_rot = key_states @ R_h
            else:
                k_pre_rot = key_states
            hook.k[layer_idx].append(k_pre_rot.detach().cpu())
            # raw pre-RoPE key
            hook.k_raw[layer_idx].append(key_states.detach().cpu())

        if hook.cq and hook.pre_rope:
            if rotate:
                R_h = R_head.to(key_states.dtype)
                k_pre_rot = key_states @ R_h
                k_cb = hook.k_cb[layer_idx].to(key_states.device, key_states.dtype)
                if hook.mant:
                    k_pre_rot_q = vq_quantize_mantissa(
                        k_pre_rot, k_cb, hook.channel,
                        hook.mant_bits, hook.mant_block_size,
                    )
                else:
                    k_pre_rot_q = vq_quantize(k_pre_rot, k_cb, hook.channel)
                key_states = k_pre_rot_q @ R_h.T
            else:
                k_cb = hook.k_cb[layer_idx].to(key_states.device, key_states.dtype)
                if hook.mant:
                    key_states = vq_quantize_mantissa(
                        key_states, k_cb, hook.channel,
                        hook.mant_bits, hook.mant_block_size,
                    )
                else:
                    key_states = vq_quantize(key_states, k_cb, hook.channel)

        # ---- RoPE ----
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # post-RoPE raw 스냅샷 (회전 전, quantize 전)
        key_raw_post = key_states
        value_raw_post = value_states

        query_states, key_states, value_states = rotate_head(
            query_states, key_states, value_states, key_raw_post, value_raw_post
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if hook.bfp:
            query_states = bfp_quantize_activation(
                query_states, hook.bfp_block_size, hook.bfp_bits,
            )
            key_states = bfp_quantize_activation(
                key_states, hook.bfp_block_size, hook.bfp_bits,
            )

        if attention_mask is not None and attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
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
