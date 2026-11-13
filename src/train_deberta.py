import os
import json
import argparse
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Union
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    set_seed
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0].keys() else None
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    acc = np.mean(preds == label_ids)
    return {"accuracy": float(acc)}

def train(args):
    set_seed(args.seed)
    
    with open(args.train_json, 'r', encoding='utf-8') as f:
        records = json.load(f)
        
    train_records = [r for r in records if r.get('fold') != args.val_fold]
    val_records = [r for r in records if r.get('fold') == args.val_fold]
    
    print(f"Loaded {len(train_records)} train cases, {len(val_records)} validation cases (Fold {args.val_fold}).")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def preprocess_function(examples):
        # Repeat context 4 times for the 4 choices
        first_sentences = [[context] * 4 for context in examples["context"]]
        second_sentences = [
            [examples["choice_A"][i], examples["choice_B"][i], examples["choice_C"][i], examples["choice_D"][i]]
            for i in range(len(examples["context"]))
        ]
        
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=args.max_length
        )
        
        # Unflatten to [batch_size, 4, seq_len]
        return {
            k: [v[i : i + 4] for i in range(0, len(v), 4)]
            for k, v in tokenized_examples.items()
        }

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)
    
    train_tokenized = train_ds.map(preprocess_function, batched=True)
    val_tokenized = val_ds.map(preprocess_function, batched=True)
    
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
    
    output_dir = os.path.join(args.output_dir, f"fold_{args.val_fold}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        save_total_limit=1,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_res = trainer.evaluate()
    print("Final Evaluation Results on Validation Fold:", eval_res)
    
    # Save best model
    trainer.save_model(os.path.join(output_dir, "best_model"))
    print(f"Model saved to {output_dir}/best_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, default='data/mc_train.json')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small')
    parser.add_argument('--output_dir', type=str, default='checkpoints/deberta_mc')
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
