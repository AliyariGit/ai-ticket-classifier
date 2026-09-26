"""Supervised QLoRA fine-tuning for the ticket classification prompt format."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="backend/training/data")
    parser.add_argument("--output-dir", default="backend/artifacts/llama-ticket-adapter")
    parser.add_argument("--base-model", default=os.getenv("LLAMA_BASE_MODEL", "meta-llama/Llama-3.2-3B"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a supported CUDA GPU; CPU training is not supported by this script.")

    data_files = {
        split: os.path.join(args.data_dir, f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    dataset = load_dataset("json", data_files=data_files)
    if not all(len(dataset[split]) for split in data_files):
        raise ValueError("train, validation, and test JSONL files must all be non-empty")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    target_modules = [
        item.strip()
        for item in os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")
        if item.strip()
    ]
    adapter = LoraConfig(
        r=int(os.getenv("LORA_R", "32")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "64")),
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        max_length=2048,
        dataset_text_field="text",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=adapter,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as config:
        json.dump({
            "base_model": args.base_model,
            "lora_r": int(os.getenv("LORA_R", "32")),
            "lora_alpha": int(os.getenv("LORA_ALPHA", "64")),
            "lora_dropout": float(os.getenv("LORA_DROPOUT", "0.05")),
            "target_modules": target_modules,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        }, config, indent=2)


if __name__ == "__main__":
    main()