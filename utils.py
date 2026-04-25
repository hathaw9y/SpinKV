import torch
from tqdm import tqdm
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
