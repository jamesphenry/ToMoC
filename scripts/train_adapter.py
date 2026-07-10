#!/usr/bin/env python3
"""train_adapter — LoRA train smolLM:135m on the tool-call flashcards.

Phase 2 of the roadmap. Trains a LoRA adapter that teaches the habit:
emit `TOOL lookup query="..."` when it can't answer, else just answer.
Logs the run to passdb (walltime_s, gpu_mem_used_mb, loss).

Design (KISS):
- base model from models/smollm-135m-instruct (pulled via huggingface_hub)
- flashcards -> text pairs: prompt = "Question: <q>\nAnswer or call a tool:\n"
  target = the A/B `a` field (TOOL line, or the answer / <model> placeholder)
- Type B cards with a=None (coding/summarization) are skipped as training
  targets (no gold answer to supervise); Type B with a concrete answer are kept.
- LoRA on attention proj layers; P4-friendly (r=8, fp16, grad ckpt).

Usage:
  python scripts/train_adapter.py --epochs 3 --lr 2e-4 --out adapters/v1
  python scripts/train_adapter.py --base models/smollm-135m-instruct --data data/raw/flashcards.jsonl --out adapters/v1 --epochs 3
"""
import argparse
import json
import os
import re
import time

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, TaskType

from passdb import PassDB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_BASE = os.path.join(ROOT, "models", "smollm-135m-instruct")
DEFAULT_DATA = os.path.join(ROOT, "data", "raw", "flashcards.jsonl")


class PadCollator:
    """Pads input_ids/attention_mask/labels to the longest in batch.
    Labels are padded with -100 (ignored in loss). Avoids the
    DataCollatorForLanguageModeling 'labels excessive nesting' error."""
    def __init__(self, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, features):
        import torch
        ids = [f["input_ids"] for f in features]
        masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for seq, m, lab in zip(ids, masks, labels):
            out["input_ids"].append(torch.tensor(seq))
            out["attention_mask"].append(torch.tensor(m))
            out["labels"].append(torch.tensor(lab))
        # pad to max length in batch (cap at max_len)
        L = min(self.max_len, max(len(s) for s in out["input_ids"]))
        def padseq(t, val):
            if len(t) >= L:
                return t[:L]
            return torch.cat([t, torch.full((L - len(t),), val, dtype=t.dtype)])
        out["input_ids"] = torch.stack([padseq(s, pad_id) for s in out["input_ids"]])
        out["attention_mask"] = torch.stack([padseq(s, 0) for s in out["attention_mask"]])
        out["labels"] = torch.stack([padseq(s, -100) for s in out["labels"]])
        return out


class FlashcardDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len
        cards = [json.loads(l) for l in open(path) if l.strip()]
        self.rows = []
        for c in cards:
            q = c["q"].strip()
            a = c.get("a")
            if a is None:
                # Type B with no gold answer (coding/summarization): skip train target
                continue
            prompt = f"Question: {q}\nAnswer or call a tool:\n"
            self.rows.append((prompt, a.strip()))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        prompt, target = self.rows[i]
        text = prompt + target + self.tok.eos_token
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        # causal LM: labels = input_ids (we train to reproduce the whole seq,
        # but mask the prompt so loss focuses on the response)
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        p_enc = self.tok(prompt, truncation=True, max_length=self.max_len,
                         return_tensors="pt")["input_ids"].squeeze(0)
        labels[: len(p_enc)] = -100
        return {"input_ids": ids, "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(ROOT, "adapters", "v1"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto")
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = FlashcardDataset(args.data, tokenizer, max_len=args.max_len)
    print(f"training samples: {len(ds)}")

    t0 = time.time()
    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=PadCollator(tokenizer, max_len=args.max_len),
    )
    trainer.train()
    wall = time.time() - t0

    # metrics from the last log
    loss_final = None
    if trainer.state.log_history:
        loss_final = trainer.state.log_history[-1].get("train_loss")

    # save adapter
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # gpu mem (P4)
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\nadapter saved -> {args.out}")
    print(f"walltime_s={wall:.1f} loss_final={loss_final} gpu_mem_mb={gpu_mem}")

    # persist to passdb
    db = PassDB()
    pid = db.new_pass(
        base_model="smollm:135m", lora_r=args.r, lora_alpha=args.alpha,
        epochs=args.epochs, lr=args.lr, num_cards=len(ds),
        loss_final=loss_final, walltime_s=round(wall, 1),
        gpu_mem_used_mb=round(gpu_mem, 1) if gpu_mem else None,
        status="trained")
    db.log_meta(pid, "adapter_path", args.out)
    db.log_meta(pid, "data", os.path.basename(args.data))
    db.summarize(pid)
    db.close()
    print(f"logged pass id={pid}")


if __name__ == "__main__":
    main()
