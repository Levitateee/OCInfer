import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelRunner:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        self.model.eval()
    def generate(self, prompt, max_tokens):
        # 每次把完整序列喂给模型，取最后一个位置的 logits，采样下一个 token，拼回输入，再喂一次
        # Placeholder for the actual generation logic
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device) # B,N
        for _ in range(max_tokens):
            logits = self.model(input_ids,use_cache=False).logits # B,N,V
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)  # B,1
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)  # Append the new token to the input sequence
        return input_ids