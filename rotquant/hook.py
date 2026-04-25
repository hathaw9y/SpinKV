from collections import defaultdict


class Hook:
    """KV cache 수집 / quantization 적용 시 공유되는 상태."""
    collect = False
    cq = False
    k_cb = None
    v_cb = None
    kr_cb = None
    channel = 4
    pre_rope = False
    mant = False
    mant_bits = 8
    mant_block_size = 128

    def __init__(self):
        # rotated activations
        self.k = defaultdict(list)           # pre-RoPE, rotated
        self.k_ropes = defaultdict(list)     # post-RoPE, rotated (LLaMA only)
        self.v = defaultdict(list)
        self.k_grad = defaultdict(list)
        self.k_ropes_grad = defaultdict(list)
        self.v_grad = defaultdict(list)

        # non-rotated (raw) activations
        self.k_raw = defaultdict(list)
        self.k_ropes_raw = defaultdict(list)
        self.v_raw = defaultdict(list)
