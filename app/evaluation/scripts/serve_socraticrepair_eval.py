from __future__ import annotations

import modal

app = modal.App("socraticrepair-eval-model")
model_volume = modal.Volume.from_name(
    "socraticrepair-checkpoints",
    create_if_missing=False,
)
cache_volume = modal.Volume.from_name(
    "socraticrepair-hf-cache",
    create_if_missing=False,
)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0",
    "transformers==4.55.3",
    "peft==0.17.1",
    "accelerate==1.10.1",
)


@app.cls(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=30 * 60,
    scaledown_window=15 * 60,
    max_containers=4,
    volumes={
        "/models": model_volume,
        "/hf-cache": cache_volume,
    },
    env={"HF_HOME": "/hf-cache"},
)
class SocraticRepairModel:
    BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
    ADAPTER_PATH = "/models/qwen25coder7b-socraticrepair/epoch-2-adapter"
    MAX_INPUT_TOKENS = 30_000

    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL,
            cache_dir="/hf-cache",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            cache_dir="/hf-cache",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.model = PeftModel.from_pretrained(base, self.ADAPTER_PATH)
        self.model.cuda()
        self.model.eval()

    @modal.method()
    def generate(self, messages: list[dict], max_new_tokens: int = 1200) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens > self.MAX_INPUT_TOKENS:
            raise RuntimeError(
                f"Evaluation prompt has {input_tokens} tokens; refusing to truncate."
            )

        inputs = inputs.to("cuda")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output[0, input_tokens:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
