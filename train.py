from datasets import load_dataset
from transformers import GPT2Model, Trainer, TrainingArguments

from common import DATASET_DIR, MODEL_NAME

# https://huggingface.co/docs/transformers/v4.36.1/en/training
# https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/trainer#transformers.TrainingArguments
# https://huggingface.co/gpt2
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/README.md

OUTPUT_DIR = "eggyboi_model/"


def main():
    model = GPT2Model.from_pretrained(MODEL_NAME)

    dataset = load_dataset(DATASET_DIR)
    dataset_train = dataset["train"][0]

    training_args = TrainingArguments(output_dir=OUTPUT_DIR)
    trainer = Trainer(model=model, train_dataset=dataset_train, args=training_args)

    trainer.train()


if __name__ == "__main__":
    main()
