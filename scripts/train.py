from __future__ import annotations

import json
import random
from pathlib import Path

import modal

# ============================================================================
# Frozen experiment
# ============================================================================

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
GPU = "H100"

MAX_SEQ_LENGTH = 12_288  # measured max TRAIN = 11,211
EPOCHS = 2
LR = 1e-4
WARMUP_RATIO = 0.03
GRAD_ACCUM = 8  # micro-batch 1 -> effective batch 8

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PLAN_REPEAT = 2
EXPECTED_PLAN = 419
EXPECTED_TUTOR = 8_183
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = ROOT / "data" / "prepared_training" / "planner_examples-train.jsonl"
TUTOR_FILE = (
    ROOT / "data" / "prepared_training" / "tutor_examples_step_assessments.jsonl"
)

# ============================================================================
# Modal setup
# ============================================================================

app = modal.App("socraticrepair-sft")

output_vol = modal.Volume.from_name(
    "socraticrepair-checkpoints", create_if_missing=True
)
cache_vol = modal.Volume.from_name("socraticrepair-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.55.3",
        "peft==0.17.1",
        "accelerate==1.10.1",
    )
    .add_local_file(str(PLAN_FILE), "/data/plan.jsonl")
    .add_local_file(str(TUTOR_FILE), "/data/tutor.jsonl")
)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=5 * 60 * 60,
    volumes={
        "/outputs": output_vol,
        "/hf-cache": cache_vol,
    },
    env={
        "HF_HOME": "/hf-cache",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)
def train():
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # 1. Load exactly the final clean TRAIN data.
    # ------------------------------------------------------------------
    plans = read_jsonl("/data/plan.jsonl")
    tutors = read_jsonl("/data/tutor.jsonl")

    assert len(plans) == EXPECTED_PLAN, len(plans)
    assert len(tutors) == EXPECTED_TUTOR, len(tutors)
    assert all(x["task"] == "PLAN" for x in plans)
    assert all(x["task"] == "TUTOR" for x in tutors)

    rows = plans * PLAN_REPEAT + tutors
    random.Random(SEED).shuffle(rows)

    print(
        f"TRAIN: PLAN={len(plans) * PLAN_REPEAT:,}, "
        f"TUTOR={len(tutors):,}, total={len(rows):,}"
    )

    # ------------------------------------------------------------------
    # 2. Tokenise with the exact Qwen chat template.
    #
    # VITAL: prompt tokens get label -100, so loss is computed ONLY on
    # the assistant completion. Also: NO truncation is allowed.
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir="/hf-cache")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class SFTDataset(Dataset):
        def __init__(self, rows):
            self.items = []
            self.lengths = []

            for row in rows:
                prompt_text = tokenizer.apply_chat_template(
                    row["prompt_messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    row["prompt_messages"] + [row["completion"]],
                    tokenize=False,
                    add_generation_prompt=False,
                )

                if not full_text.startswith(prompt_text):
                    raise RuntimeError(f"Assistant boundary failed: {row['id']}")

                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                full_ids = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]

                if full_ids[: len(prompt_ids)] != prompt_ids:
                    raise RuntimeError(f"Token boundary failed: {row['id']}")

                if len(full_ids) > MAX_SEQ_LENGTH:
                    raise RuntimeError(
                        f"{row['id']} has {len(full_ids)} tokens "
                        f"(limit {MAX_SEQ_LENGTH}). Refusing to truncate."
                    )

                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

                self.items.append(
                    {
                        "input_ids": torch.tensor(full_ids, dtype=torch.long),
                        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )
                self.lengths.append(len(full_ids))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    dataset = SFTDataset(rows)

    print(
        f"Tokenisation PASS. Longest sequence: "
        f"{max(dataset.lengths):,}/{MAX_SEQ_LENGTH:,}"
    )

    # micro-batch = 1, so we need no padding and never truncate.
    def collate(batch):
        assert len(batch) == 1
        return {k: v.unsqueeze(0) for k, v in batch[0].items()}

    # ------------------------------------------------------------------
    # 3. Qwen + LoRA.
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir="/hf-cache",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    cache_vol.commit()

    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Needed with frozen base weights + gradient checkpointing.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Quick OOM test on the actual longest example BEFORE the real run.
    # ------------------------------------------------------------------
    longest_i = max(range(len(dataset)), key=lambda i: dataset.lengths[i])
    test_batch = collate([dataset[longest_i]])
    test_batch = {k: v.cuda() for k, v in test_batch.items()}

    model.cuda()
    model.train()
    loss = model(**test_batch).loss
    loss.backward()
    model.zero_grad(set_to_none=True)

    print(f"Longest-example forward/backward PASS (loss={loss.item():.4f})")
    del test_batch, loss
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Train.
    #
    # Frequent resumable checkpoints protect the overnight Modal run.
    # Separate epoch adapters let us compare epoch 1 vs epoch 2 tomorrow.
    # ------------------------------------------------------------------
    run_dir = Path("/outputs/qwen25coder7b-socraticrepair")
    run_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=4,
        report_to="none",
        remove_unused_columns=False,
        seed=SEED,
    )

    class SaveEpochAdapter(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            output_vol.commit()
            return control

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            epoch = int(round(state.epoch or 0))
            if epoch >= 1 and model is not None:
                path = run_dir / f"epoch-{epoch}-adapter"
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                output_vol.commit()
                print(f"Saved {path}")
            return control

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate,
        callbacks=[SaveEpochAdapter()],
    )

    last_checkpoint = get_last_checkpoint(str(run_dir))
    print(
        f"Resuming from {last_checkpoint}"
        if last_checkpoint
        else "Starting fresh training"
    )

    result = trainer.train(resume_from_checkpoint=last_checkpoint or None)

    # ------------------------------------------------------------------
    # 6. Save final LoRA adapter.
    # ------------------------------------------------------------------
    final_dir = run_dir / "final-adapter"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    (run_dir / "train_metrics.json").write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )
    (run_dir / "DONE.txt").write_text(
        "Training complete.\n",
        encoding="utf-8",
    )
    output_vol.commit()

    print("TRAINING COMPLETE")
    print(f"Final adapter: {final_dir}")


@app.local_entrypoint()
def main():
    train.remote()
