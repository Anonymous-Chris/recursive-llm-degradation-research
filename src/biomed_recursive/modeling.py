from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from .common import prompt


def bounded_tokens(tokenizer, question, context, max_length, answer=None):
    # Preserve the existing prompt format, trimming only context first.
    before, after = prompt(question, "").split("Context:\n", 1)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    head, tail = encode(before + "Context:\n"), encode(after)
    available = max_length - len(head) - len(tail)
    if available < (2 if answer is not None else 0):
        raise ValueError("Question and prompt markers exceed the token budget")
    target = []
    if answer is not None:
        target = encode(answer)[:available - 1] + [tokenizer.eos_token_id]
    context_ids = encode(str(context))[:available - len(target)]
    return head + context_ids + tail, target


class SupervisedDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.items = []
        for row in frame.itertuples(index=False):
            prefix_ids, answer_ids = bounded_tokens(
                tokenizer, row.question, row.context, max_length, str(row.long_answer).strip()
            )
            ids = prefix_ids + answer_ids
            labels = [-100] * len(prefix_ids) + answer_ids
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def tokenizer_for(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def base_model(model_name: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
    )


def train_adapter(frame: pd.DataFrame, cfg: dict, output_dir: Path, init_adapter: Path | None = None, *, seed: int = 42):
    tokenizer = tokenizer_for(cfg["model_name"])
    tokenizer.padding_side = "right"
    model = base_model(cfg["model_name"])
    train_cfg = cfg["training"]
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=train_cfg["lora_r"], lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=train_cfg["lora_dropout"], target_modules=train_cfg["target_modules"], bias="none",
        )
        model = get_peft_model(model, lora)
    model.config.use_cache = False
    dataset = SupervisedDataset(frame, tokenizer, train_cfg["max_seq_length"])
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"), num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"], per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"], logging_steps=5,
        save_strategy="no", report_to="none", remove_unused_columns=False,
        seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, padding=True)
    Trainer(model=model, args=args, train_dataset=dataset, data_collator=collator).train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return adapter_dir


@torch.inference_mode()
def generate(frame: pd.DataFrame, cfg: dict, adapter_dir: Path, answer_column="predicted_answer") -> pd.DataFrame:
    tokenizer = tokenizer_for(cfg["model_name"])
    model = PeftModel.from_pretrained(base_model(cfg["model_name"]), adapter_dir)
    model.eval()
    rows, gen = [], cfg["generation"]
    device = next(model.parameters()).device
    for start in range(0, len(frame), gen["batch_size"]):
        batch = frame.iloc[start:start + gen["batch_size"]]
        inputs = [{"input_ids": bounded_tokens(
            tokenizer, r.question, r.context, cfg["training"]["max_seq_length"]
        )[0]} for r in batch.itertuples(index=False)]
        encoded = tokenizer.pad(inputs, return_tensors="pt", padding=True).to(device)
        kwargs = dict(max_new_tokens=gen["max_new_tokens"], do_sample=gen["do_sample"],
                      repetition_penalty=gen["repetition_penalty"], no_repeat_ngram_size=gen["no_repeat_ngram_size"],
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if gen["do_sample"]:
            kwargs.update(temperature=gen["temperature"], top_p=gen["top_p"])
        output = model.generate(**encoded, **kwargs)
        prompt_width = encoded.input_ids.shape[1]
        texts = tokenizer.batch_decode(output[:, prompt_width:], skip_special_tokens=True)
        for (_, row), text in zip(batch.iterrows(), texts):
            record = row.to_dict()
            record[answer_column] = text.strip()
            rows.append(record)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)
