

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# adapted from - https://huggingface.co/blog/how-to-generate

seed_text = 'My favorite thing about a wedding at the Breakers Palm Beach is'

# how would your tokenizer be trained? It has to choose the right vocabulary (best guess)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode(seed_text, return_tensors='pt')

greedy_output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

