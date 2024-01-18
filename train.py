from datasets import load_dataset
from transformers import GPT2Model, Trainer, TrainingArguments

from common import DATASET_DIR, MODEL_NAME

# training / fine-tuning tutorial
# https://huggingface.co/docs/transformers/v4.36.1/en/training

# TrainingArguments
# https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments

# GPT2 docs
# https://huggingface.co/gpt2
# https://huggingface.co/docs/transformers/v4.36.1/en/model_doc/gpt2

# train example scripts
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/README.md

OUTPUT_DIR = "eggyboi_model/"


def main():
    model = GPT2Model.from_pretrained(MODEL_NAME)

    dataset = load_dataset(DATASET_DIR)
    dataset_train = dataset["train"]

    training_args = TrainingArguments(output_dir=OUTPUT_DIR)
    trainer = Trainer(model=model, train_dataset=dataset_train, args=training_args)

    trainer.train()


if __name__ == "__main__":
    main()
