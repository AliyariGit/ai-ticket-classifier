"""4-bit base-model inference with a separately loaded PEFT adapter."""

from pathlib import Path
import os


class LlamaTicketClassifier:
    def __init__(self, base_model, adapter_path):
        if os.getenv("LLAMA_4BIT", "true").lower() not in ("1", "true", "yes", "on"):
            raise RuntimeError("The Llama provider requires LLAMA_4BIT=true.")
        if not adapter_path:
            raise RuntimeError("LLAMA_ADAPTER_PATH must point to a trained LoRA adapter when LLM_PROVIDER=llama.")
        if not Path(adapter_path).is_dir():
            raise RuntimeError("LLAMA_ADAPTER_PATH does not exist or is not a directory.")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError(
                "Llama inference dependencies are missing. Install backend/requirements-llama.txt."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "LLM_PROVIDER=llama requires a supported CUDA GPU; configure LLM_PROVIDER=ollama or rules instead."
            )

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization,
            torch_dtype=compute_dtype,
            device_map="auto",
        )
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        self.model.eval()
        self.torch = torch

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=300,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)