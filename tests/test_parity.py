import sys
sys.path.append('../')
from model_runner import ModelRunner
import torch

model_name = "/data/models/Qwen3-0.6B/"
engine = ModelRunner(model_name)
tokenizer = engine.tokenizer
hf_model = engine.model

prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "Q: What is 2+2?\nA:",
]

MAX_NEW = 32

for p in prompts:
    # 你的实现
    ids_mine = engine.generate(p, max_tokens=MAX_NEW)  # [1, prompt_len + MAX_NEW]
    
    # HF 官方 generate
    inputs = tokenizer.encode(p, return_tensors="pt").to(hf_model.device)
    with torch.no_grad():
        ids_hf = hf_model.generate(
            inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            use_cache=False,
        )
    
    # 只比较新生成的 token（从 prompt_len 之后开始）
    prompt_len = inputs.shape[1]
    mine_new = ids_mine[0, prompt_len:].tolist()
    hf_new = ids_hf[0, prompt_len:].tolist()
    
    # 先比较长度
    if len(mine_new) != len(hf_new):
        print(f"[LENGTH MISMATCH] prompt={p!r}")
        print(f"  mine len={len(mine_new)}, hf len={len(hf_new)}")
        print(f"  mine: {mine_new}")
        print(f"  hf:   {hf_new}")
        continue
    
    # 逐 token 比较
    mismatch_idx = None
    for i, (a, b) in enumerate(zip(mine_new, hf_new)):
        if a != b:
            mismatch_idx = i
            break
    
    if mismatch_idx is None:
        print(f"[PASS] {p!r}  ({len(mine_new)} tokens identical)")
    else:
        print(f"[FAIL] {p!r}  divergence at new-token index {mismatch_idx}")
        print(f"  mine: {mine_new}")
        print(f"  hf:   {hf_new}")