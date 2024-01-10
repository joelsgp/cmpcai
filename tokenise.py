from datasets import load_dataset
from transformers import GPT2Tokenizer

dataset = load_dataset("text", data_files={"train": "train_eggyboi.txt"})
tokeniser = GPT2Tokenizer.from_pretrained("gpt2")


def tokenise_function(examples):
    return tokeniser(examples["text"])


encoded = dataset.map(tokenise_function)
encoded.save_to_disk("train_eggyboi/")
