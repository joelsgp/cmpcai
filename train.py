from datasets import load_dataset
from transformers import GPT2Model, Trainer, TrainingArguments

from common import DATASET_DIR, MODEL_NAME

OUTPUT_DIR = "eggyboi_model/"

model = GPT2Model.from_pretrained(MODEL_NAME)
dataset = load_dataset(DATASET_DIR)
training_args = TrainingArguments(output_dir=OUTPUT_DIR)
trainer = Trainer(model=model, train_dataset=dataset, args=training_args)


def main():
    trainer.train()


if __name__ == "__main__":
    main()
