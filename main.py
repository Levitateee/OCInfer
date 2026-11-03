from model_runner import ModelRunner

if __name__ == "__main__":
    # Example usage
    model_name = "/data/models/Qwen3-0.6B/"  # Replace with your model name
    model = ModelRunner(model_name)
    
    prompts = ["Once upon a time",
              "There was a little girl"]

    max_tokens = 50
    for p in prompts:
        prompt = p
        ids = model.generate(prompt, max_tokens)
        print(model.tokenizer.decode(ids[0], skip_special_tokens=False))