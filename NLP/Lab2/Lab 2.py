import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model


# Generate the exact number of words
def generate_next_words(model, tokenizer, text, num_words=2, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    generated = text
    collected = []

    while len(collected) < num_words:
        out = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        # Decode and take the new part
        new_text = tokenizer.decode(out[0], skip_special_tokens=True)
        frag = new_text[len(generated):].strip()

        # Extract full words
        words = re.findall(r'(?<=\s)[A-Za-z\'-]+', " " + frag)
        for w in words:
            if len(collected) < num_words:
                collected.append(w)

        # Update context
        generated = new_text
        input_ids = tokenizer.encode(generated, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

    return " ".join(collected)


# Top 3 alternatives
def generate_top3(model, tokenizer, text, num_words=2):
    results = []
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    for _ in range(3):
        out = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )
        new_text = tokenizer.decode(out[0], skip_special_tokens=True)
        frag = new_text[len(text):].strip()
        words = re.findall(r'(?<=\s)[A-Za-z\'-]+', " " + frag)
        results.append(" ".join(words[:num_words]))

    return results


if __name__ == "__main__":
    tokenizer, model = load_gpt2()

    user_input = input("Enter a 4-word sequence: ").strip()

    # Primary prediction
    primary = generate_next_words(model, tokenizer, user_input, num_words=2)
    print("\nPrimary prediction:", primary)

    # Top-3 alternatives
    top3 = generate_top3(model, tokenizer, user_input)
    print("Top-3 alternatives:", top3)
