import torch
from tqdm import tqdm


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


def restore_fp16_from_mantissa(mantissa: torch.Tensor, real_exp: torch.Tensor,
                               mbits: int = 8) -> torch.Tensor:
    truncate_bits = 11 - mbits + 1
    return mantissa * (2 ** truncate_bits) / 1024.0 * real_exp


def bfp_quantize_activation(x: torch.Tensor, block_size: int = 128,
                            mbits: int = 8) -> torch.Tensor:
    if x.shape[-1] % block_size != 0:
        block_size = x.shape[-1]
    restored, _, _ = convert2fp16(x, block_size=block_size, mbits=mbits)
    return restored.to(x.dtype)


# ==================== PPL ====================
@torch.no_grad()
def eval_ppl_wikitext(model, tokenizer, seq_len=2048, device="cuda"):
    from datasets import load_dataset
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    input_ids = enc.input_ids.to(device)

    n_samples = input_ids.shape[1] // seq_len
    nlls = []
    for i in tqdm(range(n_samples)):
        batch = input_ids[:, i * seq_len:(i + 1) * seq_len]
        out = model(batch, labels=batch)
        nlls.append(out.loss.float() * seq_len)
    return torch.exp(torch.stack(nlls).sum() / (n_samples * seq_len)).item()

def collect_kv_wikitext(model, tokenizer, hook, n_samples=16, seq_len=2048, device="cuda"):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    import random
    random.seed(0)
    trainloader = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    def update_grad(data, grad, sample_idx):
        for key, val in data.items():
            sample_data = val[sample_idx]
            grad[key].append(sample_data.grad.detach().cpu())
            val[sample_idx] = sample_data.detach().cpu()  # 그래프 해제
    
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), desc='Calibration'):
        x = data[0].to(device)
        model.zero_grad()
        outputs = model(input_ids = x, labels=x)
        loss = outputs.loss
        loss.backward()
        update_grad(hook.v, hook.v_grad, i)
        if model.model_type == 'llama2':
            update_grad(hook.k_ropes, hook.k_ropes_grad, i)
        elif model.model_type == 'opt':
            update_grad(hook.k, hook.k_grad, i)


@torch.no_grad()
def collect_act_wikitext(model, tokenizer, n_samples=16, seq_len=2048, device="cuda"):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    import random
    random.seed(0)
    trainloader = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        trainloader.append(trainenc.input_ids[:, i:j])

    for x in tqdm(trainloader, total=len(trainloader), desc='Activation'):
        model(input_ids=x.to(device))
