import types
import torch
import torch.nn as nn

from ..quantization import vq_quantize


def patch_opt_attention(attn_module, R_head, layer_idx: int, hook) -> None:
    """
    OPT self-attention patch.
    R_head=None이면 head-dim Hadamard 회전을 적용하지 않음 (no_rotate 모드).
    """
    rotate = R_head is not None

    def _maybe_collect(key_states, value_states, key_raw, value_raw):
        if not hook.collect:
            return
        key_states.retain_grad()
        value_states.retain_grad()
        hook.k[layer_idx].append(key_states)
        hook.v[layer_idx].append(value_states)
        hook.k_raw[layer_idx].append(key_raw.detach().cpu())
        hook.v_raw[layer_idx].append(value_raw.detach().cpu())

    def _maybe_quantize(key_states, value_states):
        if not hook.cq:
            return key_states, value_states
        k_cb = hook.k_cb[layer_idx].to(key_states.device, key_states.dtype)
        v_cb = hook.v_cb[layer_idx].to(value_states.device, value_states.dtype)
        key_states = vq_quantize(key_states, k_cb, hook.channel)
        value_states = vq_quantize(value_states, v_cb, hook.channel)
        return key_states, value_states

    def rotate_head(query_states, key_states, value_states):
        # rotation 직전 값을 raw 사본으로 보관
        key_raw, value_raw = key_states, value_states
        if rotate:
            R_h = R_head.to(query_states.dtype)
            query_states = query_states @ R_h
            key_states = key_states @ R_h
        _maybe_collect(key_states, value_states, key_raw, value_raw)
        key_states, value_states = _maybe_quantize(key_states, value_states)
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
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, "
                    f"but is {layer_head_mask.size()}"
                )
            attn_weights = (
                layer_head_mask.view(1, -1, 1, 1)
                * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            )
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
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

    attn_module.forward = types.MethodType(patched_forward, attn_module)