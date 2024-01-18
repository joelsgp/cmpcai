from datasets import load_dataset
from transformers import GPT2Tokenizer

from common import DATA_TEXT_PATH, DATASET_DIR, MODEL_NAME


def main():
    dataset = load_dataset("text", data_files={"train": DATA_TEXT_PATH})
    tokeniser = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    def tokenise_function(examples):
        return tokeniser(examples["text"])

    encoded = dataset.map(tokenise_function)
    encoded.save_to_disk(DATASET_DIR)


if __name__ == "__main__":
    main()
