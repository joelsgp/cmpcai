from datasets import load_dataset
from transformers import GPT2Tokenizer

from common import DATA_TEXT_PATH, DATASET_DIR, MODEL_NAME

# https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
# https://huggingface.co/docs/datasets/v2.16.1/en/package_reference/main_classes#datasets.Dataset.map

def main():
    dataset = load_dataset("text", data_files={"train": DATA_TEXT_PATH})
    tokeniser = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    def tokenise_function(examples):
        return tokeniser(examples["text"], truncation=True)

    encoded = dataset.map(tokenise_function)
    encoded.save_to_disk(DATASET_DIR)


if __name__ == "__main__":
    main()
