#!/usr/bin/env python3
"""Single-file LoRA dynamics experiment for SST-2, QNLI, and mixed clients.

The program trains six independent LoRA adapters from exactly the same
initialization and records one continuous local trajectory per client:

    sst2_0, sst2_1, qnli_0, qnli_1, mixed_0, mixed_1.

One standard training run supports four analyses without retraining:

1. formation of the final BA singular directions over optimizer time;
2. layer-wise A, delta-A, B, and BA client similarities;
3. A drift and the error of using B A0 as a proxy for B A(t);
4. whether similarity predicts validation loss after pairwise operator merging.

An optional freeze-A arm trains only three representative clients and is the
minimal intervention needed to test whether A drift causes B/BA divergence.
It uses the same file, model, data, checkpoints, and analysis implementation.

The frozen RoBERTa masked-LM head and fixed verbalizers avoid adding a local
classification head.  Only q/v LoRA parameters are trainable.  The base model
is loaded once per program invocation and reused across clients and seeds, so
six clients never require six resident copies of RoBERTa-large.

Recommended workflow:

    # Cheap dependency/model/data smoke test
    python fed_svd_dynamics_experiment.py --smoke-test \
      --output-dir outputs/fed_svd_smoke

    # Target-model memory preflight: one RoBERTa-large optimizer step.
    python fed_svd_dynamics_experiment.py --model-name FacebookAI/roberta-large \
      --epochs 1 --train-per-client 32 --reference-per-client 8 \
      --compatibility-per-client 8 --batch-size 32 --eval-batch-size 8 \
      --standard-clients sst2_0 --skip-compatibility \
      --output-dir outputs/roberta_large_preflight_v5

    # Endpoint-selection pilot: three representative standard clients only.
    # Keep this exploratory seed out of the later confirmatory seed summary.
    python fed_svd_dynamics_experiment.py --seeds 41 --epochs 15 \
      --arms standard --standard-clients sst2_0 qnli_0 mixed_0 \
      --skip-compatibility \
      --evaluation-checkpoint-fractions 0 0.1 0.5 0.667 1 \
      --stability-checkpoint-fractions 0.5 0.667 0.8 0.9 1 \
      --output-dir outputs/fed_svd_endpoint_pilot

    # First six-client exploratory seed (standard arm only by default)
    python fed_svd_dynamics_experiment.py --seeds 42 \
      --output-dir outputs/fed_svd_r8_formal

    # Optional later confirmation after the exploratory protocol is frozen
    python fed_svd_dynamics_experiment.py --seeds 42 43 44 45 46 --resume \
      --output-dir outputs/fed_svd_r8_formal

Dependencies: torch, transformers, datasets, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# Hugging Face reads several environment variables during import.  Explicit
# shell settings still win because setdefault is used.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import datasets
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - exercised on the GPU host
    raise SystemExit(
        "Missing dependencies. Install torch, transformers, datasets, numpy, and matplotlib."
    ) from exc


VERSION = "fed_svd_dynamics_v5"
TRAINING_SCHEMA_VERSION = "fed_svd_training_v5"
TASKS = ("sst2", "qnli")
TARGET_NAMES = ("query", "value")
EPS = 1e-12
MASK_PLACEHOLDER = "<|FED_SVD_MASK_PLACEHOLDER|>"
SST2_PROMPT_PREFIX = "Task: sentiment. Sentence:"
SST2_PROMPT_MIDDLE = ""
SST2_PROMPT_SUFFIX = f" The sentiment is positive: {MASK_PLACEHOLDER}."
QNLI_PROMPT_PREFIX = "Task: question answering. Question:"
QNLI_PROMPT_MIDDLE = " Sentence:"
QNLI_PROMPT_SUFFIX = (
    f" Does the sentence contain the answer to the question? {MASK_PLACEHOLDER}."
)
DUAL_FIELD_TRUNCATION_POLICY = "longest_first_with_per_field_minimum_v1"


def progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision not in {"bf16", "fp16"}:
        raise ValueError(f"Unsupported mixed precision mode: {precision}")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass(frozen=True)
class ClientSpec:
    client_id: str
    task_probs: Dict[str, float]
    category: str


@dataclass
class ClientData:
    train: List[Dict[str, Any]]
    reference: List[Dict[str, Any]]
    compatibility: List[Dict[str, Any]]


def client_specs() -> List[ClientSpec]:
    return [
        ClientSpec("sst2_0", {"sst2": 1.0, "qnli": 0.0}, "sst2"),
        ClientSpec("sst2_1", {"sst2": 1.0, "qnli": 0.0}, "sst2"),
        ClientSpec("qnli_0", {"sst2": 0.0, "qnli": 1.0}, "qnli"),
        ClientSpec("qnli_1", {"sst2": 0.0, "qnli": 1.0}, "qnli"),
        ClientSpec("mixed_0", {"sst2": 0.5, "qnli": 0.5}, "mixed"),
        ClientSpec("mixed_1", {"sst2": 0.5, "qnli": 0.5}, "mixed"),
    ]


def render_record(task: str, row: Mapping[str, Any], source_split: str) -> Dict[str, Any]:
    label = int(row["label"])
    if task == "sst2":
        question = ""
        sentence = str(row["sentence"])
        prefix = SST2_PROMPT_PREFIX
        middle = SST2_PROMPT_MIDDLE
        suffix_template = SST2_PROMPT_SUFFIX
        binary = int(label == 1)
    elif task == "qnli":
        question = str(row["question"])
        sentence = str(row["sentence"])
        prefix = QNLI_PROMPT_PREFIX
        middle = QNLI_PROMPT_MIDDLE
        suffix_template = QNLI_PROMPT_SUFFIX
        binary = int(label == 0)  # GLUE QNLI: 0 means entailment
    else:
        raise ValueError(f"Unsupported task: {task}")
    source_index = row.get("idx")
    if source_index is None:
        raise KeyError(f"Dataset row for {task} has no stable idx")
    return {
        "prompt_prefix": prefix,
        "prompt_middle": middle,
        "prompt_suffix_template": suffix_template,
        "question_text": question,
        "sentence_text": sentence,
        "label": binary,
        "task": task,
        "source_id": f"{task}:{source_split}:{int(source_index)}",
    }


def load_rendered_train_rows(
    task: str, seed: int, dataset_revision: Optional[str]
) -> Dict[int, List[Dict[str, Any]]]:
    load_kwargs: Dict[str, Any] = {"split": "train"}
    if dataset_revision is not None:
        load_kwargs["revision"] = dataset_revision
    rows = [
        render_record(task, row, "train")
        for row in load_dataset("nyu-mll/glue", task, **load_kwargs)
        if int(row["label"]) >= 0
    ]
    by_label: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        by_label[int(row["label"])].append(row)
    for label in (0, 1):
        random.Random(seed + 1009 * TASKS.index(task) + 37 * label).shuffle(by_label[label])
    return by_label


class BalancedTaskAllocator:
    """Draw disjoint, approximately label-balanced rows for every client split."""

    def __init__(self, pools: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]]) -> None:
        self.pools = pools
        self.cursor: Dict[Tuple[str, int], int] = defaultdict(int)

    def draw(self, task: str, amount: int) -> List[Dict[str, Any]]:
        if amount == 0:
            return []
        counts = [amount // 2, amount - amount // 2]
        output: List[Dict[str, Any]] = []
        for label, count in enumerate(counts):
            start = self.cursor[(task, label)]
            end = start + count
            source = self.pools[task][label]
            if end > len(source):
                raise ValueError(
                    f"Not enough {task}/label={label} rows: need {end}, have {len(source)}"
                )
            output.extend(dict(row) for row in source[start:end])
            self.cursor[(task, label)] = end
        return output


def task_counts_for_client(spec: ClientSpec, total: int) -> Dict[str, int]:
    sst2 = int(round(total * spec.task_probs["sst2"]))
    return {"sst2": sst2, "qnli": total - sst2}


def allocate_client_data(
    specs: Sequence[ClientSpec],
    train_per_client: int,
    reference_per_client: int,
    compatibility_per_client: int,
    seed: int,
    dataset_revision: Optional[str],
) -> Tuple[Dict[str, ClientData], Dict[str, Any]]:
    pools = {
        task: load_rendered_train_rows(task, seed, dataset_revision)
        for task in TASKS
    }
    allocator = BalancedTaskAllocator(pools)
    data = {spec.client_id: ClientData([], [], []) for spec in specs}
    requested = {
        "train": train_per_client,
        "reference": reference_per_client,
        "compatibility": compatibility_per_client,
    }
    source_locations: Dict[str, List[str]] = defaultdict(list)
    manifest_clients: Dict[str, Any] = {}
    for split_index, (split_name, total) in enumerate(requested.items()):
        for client_index, spec in enumerate(specs):
            destination: List[Dict[str, Any]] = getattr(data[spec.client_id], split_name)
            task_counts = task_counts_for_client(spec, total)
            for task in TASKS:
                destination.extend(allocator.draw(task, task_counts[task]))
            random.Random(
                seed + 7919 * (client_index + 1) + 104729 * (split_index + 1)
            ).shuffle(destination)
            for row in destination:
                source_locations[str(row["source_id"])].append(f"{spec.client_id}:{split_name}")
    duplicates = {key: value for key, value in source_locations.items() if len(value) > 1}
    if duplicates:
        raise AssertionError(f"Data overlap detected: {next(iter(duplicates.items()))}")
    for spec in specs:
        splits: Dict[str, Any] = {}
        for split_name in requested:
            rows = getattr(data[spec.client_id], split_name)
            by_task = {task: sum(row["task"] == task for row in rows) for task in TASKS}
            by_label = {
                task: {
                    str(label): sum(
                        row["task"] == task and int(row["label"]) == label for row in rows
                    )
                    for label in (0, 1)
                }
                for task in TASKS
            }
            if len(rows) != requested[split_name]:
                raise AssertionError(f"Unexpected {spec.client_id}/{split_name} size")
            splits[split_name] = {"task_counts": by_task, "label_counts": by_label}
        manifest_clients[spec.client_id] = {
            "category": spec.category,
            "task_probs": spec.task_probs,
            "splits": splits,
        }
    assigned_content = []
    for client_id, client in sorted(data.items()):
        for split_name in ("train", "reference", "compatibility"):
            for row in getattr(client, split_name):
                assigned_content.append(
                    {
                        "location": f"{client_id}:{split_name}",
                        "source_id": row["source_id"],
                        "task": row["task"],
                        "label": int(row["label"]),
                        "question_text": row["question_text"],
                        "sentence_text": row["sentence_text"],
                    }
                )
    return data, {
        "clients": manifest_clients,
        "unique_source_examples": len(source_locations),
        "duplicate_source_examples": 0,
        "all_train_reference_compatibility_disjoint": True,
        "assignment_sha256": stable_hash(
            {key: sorted(value) for key, value in sorted(source_locations.items())}
        ),
        "assigned_content_sha256": stable_hash({"rows": assigned_content}),
        "dataset_revision_requested": dataset_revision,
    }


def allocate_two_fields(
    first_length: int,
    second_length: int,
    budget: int,
    minimum_per_nonempty_field: int,
) -> Tuple[int, int]:
    """Apply longest-first truncation while preserving both QNLI fields."""
    if budget < 0:
        raise ValueError("Text token budget cannot be negative")
    first_minimum = min(first_length, minimum_per_nonempty_field)
    second_minimum = min(second_length, minimum_per_nonempty_field)
    if first_minimum + second_minimum > budget:
        raise ValueError(
            "max_length leaves too few tokens to preserve the configured minimum for "
            "both QNLI fields"
        )
    if first_length + second_length <= budget:
        return first_length, second_length

    # Find the largest common retained-length cap that fits. This is equivalent
    # to repeatedly truncating the currently longer field, but is O(log n)
    # instead of O(number of removed tokens). Short questions remain intact;
    # two long fields converge to comparable retained lengths.
    lower, upper = 0, max(first_length, second_length)
    best_cap = 0
    while lower <= upper:
        cap = (lower + upper) // 2
        first_at_cap = min(first_length, max(first_minimum, cap))
        second_at_cap = min(second_length, max(second_minimum, cap))
        if first_at_cap + second_at_cap <= budget:
            best_cap = cap
            lower = cap + 1
        else:
            upper = cap - 1
    first_keep = min(first_length, max(first_minimum, best_cap))
    second_keep = min(second_length, max(second_minimum, best_cap))
    remaining = budget - first_keep - second_keep
    candidates = sorted(
        (
            (first_length - first_keep, 0),
            (second_length - second_keep, 1),
        ),
        reverse=True,
    )
    for available, field_index in candidates:
        addition = min(available, remaining)
        if field_index == 0:
            first_keep += addition
        else:
            second_keep += addition
        remaining -= addition
        if remaining == 0:
            break
    return first_keep, second_keep


def encode_prompt_row(
    row: MutableMapping[str, Any],
    tokenizer: Any,
    max_length: int,
    qnli_min_field_tokens: int,
) -> Dict[str, Any]:
    source_id = str(row["source_id"])
    raw_fields = (
        str(row["question_text"]),
        str(row["sentence_text"]),
    )
    if any(MASK_PLACEHOLDER in value for value in raw_fields):
        raise ValueError(f"Reserved mask placeholder occurs in source text: {source_id}")
    suffix_template = str(row["prompt_suffix_template"])
    if suffix_template.count(MASK_PLACEHOLDER) != 1:
        raise ValueError(f"Prompt suffix must contain exactly one placeholder: {source_id}")
    suffix = suffix_template.replace(MASK_PLACEHOLDER, str(tokenizer.mask_token))

    def ids(text: str) -> List[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    prefix_ids = ids(str(row["prompt_prefix"]))
    middle_ids = ids(str(row["prompt_middle"]))
    suffix_ids = ids(suffix)
    question_ids = ids(" " + raw_fields[0]) if raw_fields[0] else []
    sentence_ids = ids(" " + raw_fields[1]) if raw_fields[1] else []
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    fixed_length = len(prefix_ids) + len(middle_ids) + len(suffix_ids) + special_tokens
    text_budget = max_length - fixed_length
    if text_budget < 1:
        raise ValueError(
            f"max_length={max_length} cannot fit the fixed prompt and mask for {source_id}"
        )
    if str(row["task"]) == "qnli":
        question_keep, sentence_keep = allocate_two_fields(
            len(question_ids),
            len(sentence_ids),
            text_budget,
            qnli_min_field_tokens,
        )
    else:
        question_keep = 0
        sentence_keep = min(len(sentence_ids), text_budget)
    content_ids = (
        prefix_ids
        + question_ids[:question_keep]
        + middle_ids
        + sentence_ids[:sentence_keep]
        + suffix_ids
    )
    input_ids = list(tokenizer.build_inputs_with_special_tokens(content_ids))
    mask_count = sum(int(token_id) == int(tokenizer.mask_token_id) for token_id in input_ids)
    if len(input_ids) > max_length or mask_count != 1:
        raise ValueError(
            f"Invalid encoded prompt for {source_id}: length={len(input_ids)}, "
            f"max_length={max_length}, mask_count={mask_count}"
        )
    row["input_ids"] = input_ids
    row["attention_mask"] = [1] * len(input_ids)
    return {
        "source_id": source_id,
        "task": row["task"],
        "original_tokens": fixed_length + len(question_ids) + len(sentence_ids),
        "encoded_tokens": len(input_ids),
        "question_tokens_before": len(question_ids),
        "question_tokens_after": question_keep,
        "sentence_tokens_before": len(sentence_ids),
        "sentence_tokens_after": sentence_keep,
        "was_truncated": int(
            question_keep < len(question_ids) or sentence_keep < len(sentence_ids)
        ),
    }


def encode_client_data(
    data: MutableMapping[str, ClientData],
    tokenizer: Any,
    max_length: int,
    qnli_min_field_tokens: int,
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    for client_id, client in data.items():
        audit[client_id] = {}
        for split_name in ("train", "reference", "compatibility"):
            rows = getattr(client, split_name)
            row_audits = [
                encode_prompt_row(
                    row, tokenizer, max_length, qnli_min_field_tokens
                )
                for row in rows
            ]
            original_lengths = [int(item["original_tokens"]) for item in row_audits]
            encoded_lengths = [int(item["encoded_tokens"]) for item in row_audits]
            truncated = [item for item in row_audits if int(item["was_truncated"]) == 1]
            audit[client_id][split_name] = {
                "examples": len(rows),
                "mean_original_tokens": float(np.mean(original_lengths)),
                "max_original_tokens": max(original_lengths),
                "max_encoded_tokens": max(encoded_lengths),
                "truncated_examples": len(truncated),
                "truncated_fraction": len(truncated) / max(len(rows), 1),
                "truncation_examples": truncated[:20],
            }
    return audit


def run_tokenizer_self_tests(
    tokenizer: Any, max_length: int, qnli_min_field_tokens: int
) -> Dict[str, Any]:
    """Exercise the paid-run failure modes before loading or training any client."""
    synthetic = render_record(
        "qnli",
        {
            "idx": 987654321,
            "label": 0,
            "question": "{literal braces} " + "question " * (max_length * 2),
            "sentence": "sentence " * (max_length * 3),
        },
        "synthetic",
    )
    audit = encode_prompt_row(
        synthetic, tokenizer, max_length, qnli_min_field_tokens
    )
    if int(audit["was_truncated"]) != 1:
        raise AssertionError("Long-prompt tokenizer self-test did not trigger truncation")
    if sum(
        int(token_id) == int(tokenizer.mask_token_id)
        for token_id in synthetic["input_ids"]
    ) != 1:
        raise AssertionError("Long-prompt tokenizer self-test lost or duplicated the mask")
    if min(
        int(audit["question_tokens_after"]),
        int(audit["sentence_tokens_after"]),
    ) < qnli_min_field_tokens:
        raise AssertionError("QNLI field-preservation minimum was not respected")
    return audit


class EncodedDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


class EncodedCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        padded = self.tokenizer.pad(
            {
                "input_ids": [row["input_ids"] for row in rows],
                "attention_mask": [row["attention_mask"] for row in rows],
            },
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
        padded["tasks"] = [str(row["task"]) for row in rows]
        padded["source_ids"] = [str(row["source_id"]) for row in rows]
        return padded


def balanced_epoch_rows(rows: Sequence[Mapping[str, Any]], seed: int) -> List[Mapping[str, Any]]:
    """Interleave task/label strata so early checkpoints reflect client composition."""
    by_stratum: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row["task"]), int(row["label"]))].append(row)
    for offset, stratum in enumerate(sorted(by_stratum)):
        random.Random(seed + 101 * offset).shuffle(by_stratum[stratum])
    strata = sorted(by_stratum)
    output: List[Mapping[str, Any]] = []
    maximum = max(len(by_stratum[stratum]) for stratum in strata)
    for index in range(maximum):
        for stratum in strata:
            if index < len(by_stratum[stratum]):
                output.append(by_stratum[stratum][index])
    return output


class LoRALinear(nn.Module):
    def __init__(
        self, base: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(
            F.linear(self.dropout(x), self.A), self.B
        )


AdapterRegistry = Dict[str, LoRALinear]
AdapterState = Dict[str, torch.Tensor]


def inject_lora(
    model: nn.Module, rank: int, alpha: float, dropout: float
) -> AdapterRegistry:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    registry: AdapterRegistry = {}
    for layer_index, layer in enumerate(model.roberta.encoder.layer):
        attention = layer.attention.self
        for target_name in TARGET_NAMES:
            base = getattr(attention, target_name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"Expected nn.Linear at layer {layer_index}/{target_name}")
            wrapped = LoRALinear(base, rank, alpha, dropout)
            setattr(attention, target_name, wrapped)
            path = f"roberta.encoder.layer.{layer_index}.attention.self.{target_name}"
            registry[path] = wrapped
    expected = len(model.roberta.encoder.layer) * len(TARGET_NAMES)
    if len(registry) != expected:
        raise AssertionError(f"Expected {expected} LoRA targets, found {len(registry)}")
    return registry


def reset_lora_parameters(registry: AdapterRegistry) -> None:
    """Create a fresh standard LoRA initialization without reloading the base model."""
    with torch.no_grad():
        for module in registry.values():
            nn.init.kaiming_uniform_(module.A, a=math.sqrt(5))
            module.B.zero_()


def state_key(module_name: str, factor: str) -> str:
    return f"{module_name}::{factor}"


def extract_state(registry: AdapterRegistry, dtype: torch.dtype = torch.float32) -> AdapterState:
    state: AdapterState = {}
    for module_name, module in registry.items():
        state[state_key(module_name, "A")] = module.A.detach().to("cpu", dtype=dtype).clone()
        state[state_key(module_name, "B")] = module.B.detach().to("cpu", dtype=dtype).clone()
    return state


def clone_state(state: Mapping[str, torch.Tensor]) -> AdapterState:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def load_state(registry: AdapterRegistry, state: Mapping[str, torch.Tensor]) -> None:
    expected = {
        state_key(module_name, factor)
        for module_name in registry
        for factor in ("A", "B")
    }
    if set(state) != expected:
        missing = expected - set(state)
        extra = set(state) - expected
        raise KeyError(f"Adapter state mismatch: missing={list(missing)[:2]}, extra={list(extra)[:2]}")
    with torch.no_grad():
        for module_name, module in registry.items():
            module.A.copy_(state[state_key(module_name, "A")].to(module.A.device, module.A.dtype))
            module.B.copy_(state[state_key(module_name, "B")].to(module.B.device, module.B.dtype))


def module_pair(state: Mapping[str, torch.Tensor], module_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    return state[state_key(module_name, "A")], state[state_key(module_name, "B")]


def module_coordinates(module_name: str) -> Tuple[int, str]:
    parts = module_name.split(".")
    layer_index = int(parts[3])
    target_name = parts[-1]
    return layer_index, target_name


def verbalizer_ids(tokenizer: Any) -> Tuple[int, int]:
    ids: List[int] = []
    for word in (" no", " yes"):
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Verbalizer {word!r} is not one token: {encoded}")
        ids.append(int(encoded[0]))
    return ids[0], ids[1]


def mask_logits(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    no_token_id: int,
    yes_token_id: int,
) -> torch.Tensor:
    mask = input_ids.eq(mask_token_id)
    if not torch.all(mask.sum(dim=1).eq(1)):
        raise ValueError("Every prompt must contain exactly one mask token")
    encoded = model.roberta(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
    )
    positions = mask.to(torch.int64).argmax(dim=1)
    batch_index = torch.arange(input_ids.size(0), device=input_ids.device)
    hidden = encoded.last_hidden_state[batch_index, positions]
    vocabulary_logits = model.lm_head(hidden)
    return vocabulary_logits[:, [no_token_id, yes_token_id]]


def frobenius_inner_low_rank(
    a_i: torch.Tensor,
    b_i: torch.Tensor,
    a_j: torch.Tensor,
    b_j: torch.Tensor,
) -> torch.Tensor:
    ai = a_i.to(torch.float64)
    bi = b_i.to(torch.float64)
    aj = a_j.to(torch.float64)
    bj = b_j.to(torch.float64)
    return torch.trace((bi.T @ bj) @ (aj @ ai.T))


def low_rank_norm_sq(a: torch.Tensor, b: torch.Tensor) -> float:
    value = frobenius_inner_low_rank(a, b, a, b)
    return max(float(value), 0.0)


def flat_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().to(torch.float64).reshape(-1)
    y = right.detach().to(torch.float64).reshape(-1)
    denominator = float(torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y))
    if denominator <= EPS:
        return float("nan")
    return float(torch.dot(x, y) / denominator)


def low_rank_cosine(
    a_i: torch.Tensor,
    b_i: torch.Tensor,
    a_j: torch.Tensor,
    b_j: torch.Tensor,
) -> float:
    norm_i = low_rank_norm_sq(a_i, b_i)
    norm_j = low_rank_norm_sq(a_j, b_j)
    if norm_i <= EPS or norm_j <= EPS:
        return float("nan")
    inner = float(frobenius_inner_low_rank(a_i, b_i, a_j, b_j))
    return inner / math.sqrt(norm_i * norm_j)


def compact_svd(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact compact SVD of B@A without materializing the dense operator."""
    aa = a.detach().to(device="cpu", dtype=torch.float64)
    bb = b.detach().to(device="cpu", dtype=torch.float64)
    q_b, r_b = torch.linalg.qr(bb, mode="reduced")
    q_a, r_a = torch.linalg.qr(aa.T, mode="reduced")
    u_core, singular, vh_core = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    u = q_b @ u_core
    v = q_a @ vh_core.T
    return u, singular, v


def factors_from_svd(
    u: torch.Tensor,
    singular: torch.Tensor,
    v: torch.Tensor,
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keep = min(rank, int(singular.numel()))
    roots = torch.sqrt(torch.clamp(singular[:keep], min=0.0))
    b = u[:, :keep] * roots.unsqueeze(0)
    a = roots.unsqueeze(1) * v[:, :keep].T
    if keep < rank:
        b = F.pad(b, (0, rank - keep))
        a = F.pad(a, (0, 0, 0, rank - keep))
    return a.to(dtype), b.to(dtype)


def compress_sum_of_pairs(
    pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: Sequence[float],
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if len(pairs) != len(weights) or not pairs:
        raise ValueError("pairs and weights must be non-empty and aligned")
    b_blocks: List[torch.Tensor] = []
    a_blocks: List[torch.Tensor] = []
    for (a, b), weight in zip(pairs, weights):
        if weight < 0:
            raise ValueError("Only non-negative merge weights are supported")
        root = math.sqrt(float(weight))
        b_blocks.append(root * b.detach().to(device="cpu", dtype=torch.float64))
        a_blocks.append(root * a.detach().to(device="cpu", dtype=torch.float64))
    b_cat = torch.cat(b_blocks, dim=1)
    a_cat = torch.cat(a_blocks, dim=0)
    u, singular, v = compact_svd(a_cat, b_cat)
    a_new, b_new = factors_from_svd(u, singular, v, rank)
    total_energy = float(torch.sum(singular.square()))
    kept_energy = float(torch.sum(singular[:rank].square()))
    retained = 1.0 if total_energy <= EPS else kept_energy / total_energy
    return a_new, b_new, retained


def weighted_b_cosine(
    b_i: torch.Tensor, b_j: torch.Tensor, a_zero: torch.Tensor
) -> float:
    bi = b_i.to(torch.float64)
    bj = b_j.to(torch.float64)
    gram = a_zero.to(torch.float64) @ a_zero.to(torch.float64).T
    inner = float(torch.trace((bi.T @ bj) @ gram))
    ni = float(torch.trace((bi.T @ bi) @ gram))
    nj = float(torch.trace((bj.T @ bj) @ gram))
    if ni <= EPS or nj <= EPS:
        return float("nan")
    return inner / math.sqrt(ni * nj)


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return float("nan")
    x_rank = rankdata([pair[0] for pair in pairs])
    y_rank = rankdata([pair[1] for pair in pairs])
    if np.std(x_rank) <= EPS or np.std(y_rank) <= EPS:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def automatic_checkpoints(
    total_steps: int,
    interval_steps: int = 8,
    required_fractions: Sequence[float] = (),
) -> List[int]:
    candidates = {0, 1, 2, 4, 8, total_steps}
    candidates.update(range(interval_steps, total_steps + 1, interval_steps))
    for fraction in (
        0.10,
        0.20,
        0.40,
        0.50,
        0.60,
        0.80,
        1.00,
        *required_fractions,
    ):
        candidates.add(max(1, int(round(total_steps * fraction))))
    return sorted(step for step in candidates if 0 <= step <= total_steps)


def experiment_checkpoint_steps(
    args: argparse.Namespace, total_steps: int
) -> List[int]:
    required = (
        list(args.evaluation_checkpoint_fractions)
        + list(args.compatibility_checkpoint_fractions)
        + list(args.freeze_compatibility_checkpoint_fractions)
        + list(args.stability_checkpoint_fractions)
    )
    return automatic_checkpoints(
        total_steps,
        interval_steps=args.trajectory_checkpoint_interval,
        required_fractions=required,
    )


def closest_steps(checkpoints: Sequence[int], fractions: Sequence[float]) -> List[int]:
    total = max(checkpoints)
    selected = {min(checkpoints, key=lambda value: abs(value - fraction * total)) for fraction in fractions}
    return sorted(selected)


def training_parameters(registry: AdapterRegistry, freeze_a: bool) -> List[nn.Parameter]:
    parameters: List[nn.Parameter] = []
    for module in registry.values():
        module.A.requires_grad_(not freeze_a)
        module.B.requires_grad_(True)
        if not freeze_a:
            parameters.append(module.A)
        parameters.append(module.B)
    return parameters


def make_loader(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    batch_size: int,
    shuffle_seed: Optional[int] = None,
) -> DataLoader:
    ordered: Sequence[Mapping[str, Any]]
    if shuffle_seed is None:
        ordered = rows
    else:
        ordered = balanced_epoch_rows(rows, shuffle_seed)
    return DataLoader(
        EncodedDataset(ordered),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=EncodedCollator(tokenizer),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    precision: str,
    collect_examples: bool = False,
) -> Dict[str, Any]:
    model.eval()
    no_id, yes_id = verbalizer_ids(tokenizer)
    total = 0
    correct = 0
    loss_sum = 0.0
    example_losses: List[float] = []
    example_tasks: List[str] = []
    example_labels: List[int] = []
    example_source_ids: List[str] = []
    per_task: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "correct": 0.0, "loss_sum": 0.0}
    )
    for batch in make_loader(rows, tokenizer, batch_size):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            logits = mask_logits(
                model,
                input_ids,
                attention_mask,
                tokenizer.mask_token_id,
                no_id,
                yes_id,
            )
            losses = F.cross_entropy(logits.float(), labels, reduction="none")
        predictions = logits.argmax(dim=-1)
        labels_cpu = labels.detach().cpu()
        predictions_cpu = predictions.detach().cpu()
        losses_cpu = losses.detach().cpu()
        if collect_examples:
            example_losses.extend(float(value) for value in losses_cpu)
            example_tasks.extend(str(task) for task in batch["tasks"])
            example_labels.extend(int(value) for value in labels_cpu)
            example_source_ids.extend(str(value) for value in batch["source_ids"])
        total += int(labels_cpu.numel())
        correct += int(predictions_cpu.eq(labels_cpu).sum())
        loss_sum += float(losses_cpu.sum())
        for task, label, prediction, loss in zip(
            batch["tasks"], labels_cpu, predictions_cpu, losses_cpu
        ):
            stats = per_task[str(task)]
            stats["n"] += 1.0
            stats["correct"] += float(prediction.eq(label))
            stats["loss_sum"] += float(loss)
    result = {
        "n": total,
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "per_task": {
            task: {
                "n": int(stats["n"]),
                "accuracy": stats["correct"] / max(stats["n"], 1.0),
                "loss": stats["loss_sum"] / max(stats["n"], 1.0),
            }
            for task, stats in sorted(per_task.items())
        },
    }
    if collect_examples:
        result["example_losses"] = example_losses
        result["example_tasks"] = example_tasks
        result["example_labels"] = example_labels
        result["example_source_ids"] = example_source_ids
    return result


def flattened_task_metrics(
    metrics: Mapping[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    columns: Dict[str, Any] = {}
    per_task = metrics.get("per_task", {})
    for task in TASKS:
        values = per_task.get(task)
        columns[f"{prefix}{task}_n"] = 0 if values is None else int(values["n"])
        columns[f"{prefix}{task}_accuracy"] = (
            float("nan") if values is None else float(values["accuracy"])
        )
        columns[f"{prefix}{task}_loss"] = (
            float("nan") if values is None else float(values["loss"])
        )
    return columns


def assert_finite_lora_gradients(
    registry: AdapterRegistry, client_id: str, step: int
) -> None:
    named_gradients: List[Tuple[str, str, torch.Tensor]] = []
    for module_name, module in registry.items():
        for factor, parameter in (("A", module.A), ("B", module.B)):
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                raise FloatingPointError(
                    f"Missing LoRA gradient: client={client_id}, step={step}, "
                    f"module={module_name}, factor={factor}"
                )
            named_gradients.append((module_name, factor, parameter.grad))
    flattened = torch.cat(
        [gradient.detach().reshape(-1) for _, _, gradient in named_gradients]
    )
    if bool(torch.isfinite(flattened).all()):
        return
    for module_name, factor, gradient in named_gradients:
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                f"Non-finite LoRA gradient: client={client_id}, step={step}, "
                f"module={module_name}, factor={factor}"
            )
    raise FloatingPointError(
        f"Non-finite LoRA gradient: client={client_id}, step={step}"
    )


def assert_finite_lora_parameters(
    registry: AdapterRegistry, client_id: str, step: int
) -> None:
    named_parameters = [
        (module_name, factor, parameter)
        for module_name, module in registry.items()
        for factor, parameter in (("A", module.A), ("B", module.B))
    ]
    flattened = torch.cat(
        [parameter.detach().reshape(-1) for _, _, parameter in named_parameters]
    )
    if bool(torch.isfinite(flattened).all()):
        return
    for module_name, factor, parameter in named_parameters:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                f"Non-finite LoRA parameter: client={client_id}, step={step}, "
                f"module={module_name}, factor={factor}"
            )
    raise FloatingPointError(
        f"Non-finite LoRA parameter: client={client_id}, step={step}"
    )


def train_client_trajectory(
    model: nn.Module,
    registry: AdapterRegistry,
    client_id: str,
    initial_state: Mapping[str, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    precision: str,
    seed: int,
    freeze_a: bool,
    checkpoint_steps: Sequence[int],
) -> Dict[str, Any]:
    load_state(registry, initial_state)
    assert_finite_lora_parameters(registry, client_id, 0)
    parameters = training_parameters(registry, freeze_a)
    # LoRA weight decay is deliberately zero.  With B0=0 this preserves the
    # exact first-step statement grad(A)=0 instead of moving A via AdamW decay.
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    use_grad_scaler = device.type == "cuda" and precision == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)
    steps_per_epoch = int(math.ceil(len(rows) / float(batch_size)))
    total_steps = steps_per_epoch * epochs
    checkpoints = sorted({int(value) for value in checkpoint_steps})
    if not checkpoints or checkpoints[0] != 0 or checkpoints[-1] != total_steps:
        raise ValueError("checkpoint_steps must include both 0 and total_steps")
    checkpoint_set = set(checkpoints)
    trajectory: Dict[int, AdapterState] = {0: clone_state(initial_state)}
    trace: List[Dict[str, Any]] = []
    fp16_scaler_backoff_steps = 0
    no_id, yes_id = verbalizer_ids(tokenizer)
    step = 0
    started = time.time()
    for epoch in range(epochs):
        loader = make_loader(
            rows,
            tokenizer,
            batch_size,
            shuffle_seed=seed + 104729 * epoch,
        )
        for batch in loader:
            model.train()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = mask_logits(
                    model,
                    input_ids,
                    attention_mask,
                    tokenizer.mask_token_id,
                    no_id,
                    yes_id,
                )
                loss = F.cross_entropy(logits.float(), labels)
            next_step = step + 1
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError(
                    f"Non-finite loss: client={client_id}, step={next_step}, "
                    f"epoch={epoch}"
                )
            if scaler.is_enabled():
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < scale_before:
                    fp16_scaler_backoff_steps += 1
            else:
                loss.backward()
                assert_finite_lora_gradients(registry, client_id, next_step)
                optimizer.step()
            step += 1
            if step in checkpoint_set:
                assert_finite_lora_parameters(registry, client_id, step)
                trace.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        "sst2_in_batch": sum(task == "sst2" for task in batch["tasks"]),
                        "qnli_in_batch": sum(task == "qnli" for task in batch["tasks"]),
                    }
                )
                trajectory[step] = extract_state(registry)
    if step != total_steps or set(trajectory) != checkpoint_set:
        raise AssertionError(
            f"Trajectory incomplete: step={step}/{total_steps}, "
            f"checkpoints={sorted(trajectory)} expected={checkpoints}"
        )
    initial_a = torch.cat(
        [initial_state[state_key(name, "A")].reshape(-1) for name in sorted(registry)]
    )
    first_a = torch.cat(
        [trajectory[1][state_key(name, "A")].reshape(-1) for name in sorted(registry)]
    )
    first_a_displacement = float(torch.linalg.vector_norm(first_a - initial_a))
    if first_a_displacement != 0.0:
        raise AssertionError(
            "A changed on the first optimizer step despite B0=0 and zero weight decay"
        )
    if freeze_a:
        for state in trajectory.values():
            for name in registry:
                if not torch.equal(
                    state[state_key(name, "A")], initial_state[state_key(name, "A")]
                ):
                    raise AssertionError("freeze-A arm changed A")
        shared_a = {
            key: value for key, value in initial_state.items() if key.endswith("::A")
        }
        trajectory = {
            checkpoint: {
                **shared_a,
                **{
                    key: value
                    for key, value in state.items()
                    if key.endswith("::B")
                },
            }
            for checkpoint, state in trajectory.items()
        }
    return {
        "trajectory": trajectory,
        "trace": trace,
        "checkpoints": checkpoints,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "elapsed_seconds": time.time() - started,
        "first_step_a_displacement": first_a_displacement,
        "fp16_scaler_backoff_steps": fp16_scaler_backoff_steps,
        "freeze_a": freeze_a,
    }


def pair_type(left: ClientSpec, right: ClientSpec) -> str:
    if left.category == right.category:
        return f"same_{left.category}"
    categories = {left.category, right.category}
    if categories == {"sst2", "qnli"}:
        return "sst2_qnli"
    if categories == {"sst2", "mixed"}:
        return "sst2_mixed"
    if categories == {"qnli", "mixed"}:
        return "qnli_mixed"
    return "other"


def all_pairs(specs: Sequence[ClientSpec]) -> List[Tuple[ClientSpec, ClientSpec]]:
    return [
        (specs[i], specs[j])
        for i in range(len(specs))
        for j in range(i + 1, len(specs))
    ]


def similarity_rows(
    seed: int,
    arm: str,
    specs: Sequence[ClientSpec],
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    initial_state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> List[Dict[str, Any]]:
    clients = sorted(trajectories)
    common_steps = sorted(set.intersection(*(set(trajectories[c]) for c in clients)))
    spec_by_id = {spec.client_id: spec for spec in specs}
    rows: List[Dict[str, Any]] = []
    for step in common_steps:
        for module_name in module_names:
            layer, target = module_coordinates(module_name)
            a_zero, _ = module_pair(initial_state, module_name)
            for left_index in range(len(clients)):
                for right_index in range(left_index + 1, len(clients)):
                    left_id = clients[left_index]
                    right_id = clients[right_index]
                    ai, bi = module_pair(trajectories[left_id][step], module_name)
                    aj, bj = module_pair(trajectories[right_id][step], module_name)
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "step": step,
                            "layer": layer,
                            "target": target,
                            "module": module_name,
                            "client_i": left_id,
                            "client_j": right_id,
                            "pair_type": pair_type(spec_by_id[left_id], spec_by_id[right_id]),
                            "cos_a": flat_cosine(ai, aj),
                            "cos_delta_a": flat_cosine(ai - a_zero, aj - a_zero),
                            "cos_b": flat_cosine(bi, bj),
                            "cos_ba": low_rank_cosine(ai, bi, aj, bj),
                            "cos_b_weighted_a0": weighted_b_cosine(bi, bj, a_zero),
                        }
                    )
    return rows


def drift_rows(
    seed: int,
    arm: str,
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    initial_state: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client_id, trajectory in sorted(trajectories.items()):
        for step, state in sorted(trajectory.items()):
            for module_name in module_names:
                layer, target = module_coordinates(module_name)
                a, b = module_pair(state, module_name)
                a_zero, _ = module_pair(initial_state, module_name)
                a0_norm = max(float(torch.linalg.vector_norm(a_zero.to(torch.float64))), EPS)
                a_drift = float(torch.linalg.vector_norm((a - a_zero).to(torch.float64))) / a0_norm
                actual_norm_sq = low_rank_norm_sq(a, b)
                proxy_error_a = a - a_zero
                proxy_error_sq = low_rank_norm_sq(proxy_error_a, b)
                proxy_relative = (
                    float("nan")
                    if actual_norm_sq <= EPS
                    else math.sqrt(max(proxy_error_sq, 0.0) / actual_norm_sq)
                )
                gram = a_zero.to(torch.float64) @ a_zero.to(torch.float64).T
                c = float(torch.trace(gram)) / gram.shape[0]
                isotropy = float(
                    torch.linalg.vector_norm(gram - c * torch.eye(gram.shape[0], dtype=gram.dtype))
                    / max(torch.linalg.vector_norm(c * torch.eye(gram.shape[0], dtype=gram.dtype)), EPS)
                )
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "client": client_id,
                        "step": step,
                        "layer": layer,
                        "target": target,
                        "module": module_name,
                        "a_relative_drift": a_drift,
                        "ba_norm": math.sqrt(max(actual_norm_sq, 0.0)),
                        "ba_vs_ba0_relative_error": proxy_relative,
                        "a0_gram_isotropy_error": isotropy,
                    }
                )
    return rows


def subspace_overlap(
    u_left: torch.Tensor,
    v_left: torch.Tensor,
    u_right: torch.Tensor,
    v_right: torch.Tensor,
    rank: int,
) -> float:
    keep = min(rank, u_left.shape[1], u_right.shape[1], v_left.shape[1], v_right.shape[1])
    if keep == 0:
        return float("nan")
    left_score = float(torch.sum((u_left[:, :keep].T @ u_right[:, :keep]).square())) / keep
    right_score = float(torch.sum((v_left[:, :keep].T @ v_right[:, :keep]).square())) / keep
    return 0.5 * (left_score + right_score)


def degeneracy_group_ids(singular: torch.Tensor, relative_gap: float) -> List[int]:
    groups: List[int] = []
    group = 0
    for index, value in enumerate(singular):
        if index > 0:
            previous = float(singular[index - 1])
            gap = abs(previous - float(value)) / max(previous, EPS)
            if gap > relative_gap:
                group += 1
        groups.append(group)
    return groups


def spectral_rows(
    seed: int,
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    module_names: Sequence[str],
    degenerate_gap: float,
    singular_relative_floor: float,
    stability_fractions: Sequence[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    component_rows: List[Dict[str, Any]] = []
    subspace_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    for client_id, trajectory in sorted(trajectories.items()):
        steps = sorted(trajectory)
        final_step = steps[-1]
        comparison_steps = [
            step
            for step in closest_steps(steps, stability_fractions)
            if step < final_step
        ]
        for module_name in module_names:
            layer, target = module_coordinates(module_name)
            final_a, final_b = module_pair(trajectory[final_step], module_name)
            u_final, singular, v_final = compact_svd(final_a, final_b)
            group_ids = degeneracy_group_ids(singular, degenerate_gap)
            original_group_sizes: Dict[int, int] = defaultdict(int)
            for group_id in group_ids:
                original_group_sizes[int(group_id)] += 1
            leading = max(float(singular[0]), EPS)
            total_final_energy = max(float(torch.sum(singular.square())), EPS)
            rank_total = int(singular.numel())
            for comparison_step in comparison_steps:
                comparison_a, comparison_b = module_pair(
                    trajectory[comparison_step], module_name
                )
                u_comparison, singular_comparison, v_comparison = compact_svd(
                    comparison_a, comparison_b
                )
                comparison_leading = max(float(singular_comparison[0]), EPS)
                comparison_total_energy = max(
                    float(torch.sum(singular_comparison.square())), EPS
                )
                for keep in sorted({1, 2, 4, rank_total}):
                    if keep > rank_total:
                        continue
                    final_min_relative_sigma = float(singular[keep - 1]) / leading
                    comparison_min_relative_sigma = (
                        float(singular_comparison[keep - 1]) / comparison_leading
                    )
                    if keep < rank_total:
                        final_boundary_gap = (
                            float(singular[keep - 1] - singular[keep])
                            / max(float(singular[keep - 1]), EPS)
                        )
                        comparison_boundary_gap = (
                            float(
                                singular_comparison[keep - 1]
                                - singular_comparison[keep]
                            )
                            / max(float(singular_comparison[keep - 1]), EPS)
                        )
                    else:
                        final_boundary_gap = float("nan")
                        comparison_boundary_gap = float("nan")
                    energy_identifiable = (
                        final_min_relative_sigma >= singular_relative_floor
                        and comparison_min_relative_sigma >= singular_relative_floor
                    )
                    boundary_identifiable = (
                        keep == rank_total
                        or (
                            final_boundary_gap >= degenerate_gap
                            and comparison_boundary_gap >= degenerate_gap
                        )
                    )
                    comparison_energy = float(
                        torch.sum(singular_comparison[:keep].square())
                    )
                    final_energy = float(torch.sum(singular[:keep].square()))
                    overlap = subspace_overlap(
                        u_comparison, v_comparison, u_final, v_final, keep
                    )
                    invalid_reasons: List[str] = []
                    if not energy_identifiable:
                        invalid_reasons.append("low_energy")
                    if not boundary_identifiable:
                        invalid_reasons.append("boundary_near_degenerate")
                    if not math.isfinite(overlap):
                        invalid_reasons.append("nonfinite_overlap")
                    stability_valid = not invalid_reasons
                    stability_rows.append(
                        {
                            "seed": seed,
                            "client": client_id,
                            "layer": layer,
                            "target": target,
                            "module": module_name,
                            "comparison_step": comparison_step,
                            "comparison_fraction": comparison_step / final_step,
                            "final_step": final_step,
                            "top_m": keep,
                            "subspace_overlap": overlap,
                            "comparison_energy": comparison_energy,
                            "final_energy": final_energy,
                            "comparison_cumulative_energy_fraction": (
                                comparison_energy / comparison_total_energy
                            ),
                            "final_cumulative_energy_fraction": (
                                final_energy / total_final_energy
                            ),
                            "top_m_energy_relative_change": (
                                abs(comparison_energy - final_energy)
                                / max(final_energy, EPS)
                            ),
                            "final_min_relative_sigma": final_min_relative_sigma,
                            "comparison_min_relative_sigma": comparison_min_relative_sigma,
                            "final_boundary_relative_gap": final_boundary_gap,
                            "comparison_boundary_relative_gap": comparison_boundary_gap,
                            "energy_identifiable": int(energy_identifiable),
                            "boundary_identifiable": int(boundary_identifiable),
                            "stability_valid": int(stability_valid),
                            "stability_invalid_reason": (
                                "" if stability_valid else ";".join(invalid_reasons)
                            ),
                        }
                    )
            for step in steps:
                current_a, current_b = module_pair(trajectory[step], module_name)
                current_norm_sq = low_rank_norm_sq(current_a, current_b)
                left = u_final.T @ current_b.to(torch.float64)
                right = current_a.to(torch.float64) @ v_final
                coordinates = left @ right
                diagonal = torch.diagonal(coordinates)
                for rank_index in range(int(singular.numel())):
                    sigma = float(singular[rank_index])
                    coefficient = float(diagonal[rank_index])
                    valid = sigma / leading >= singular_relative_floor
                    component_rows.append(
                        {
                            "seed": seed,
                            "client": client_id,
                            "step": step,
                            "layer": layer,
                            "target": target,
                            "module": module_name,
                            "rank_index": rank_index + 1,
                            "sigma_final": sigma,
                            "final_energy_fraction": sigma * sigma / total_final_energy,
                            "degeneracy_group": group_ids[rank_index],
                            "original_degeneracy_group_size": original_group_sizes[
                                int(group_ids[rank_index])
                            ],
                            "above_relative_floor": int(valid),
                            "coefficient": coefficient,
                            "signed_progress": (
                                float("nan") if sigma <= EPS else coefficient / sigma
                            ),
                            "absolute_progress": (
                                float("nan") if sigma <= EPS else abs(coefficient) / sigma
                            ),
                            "mode_current_energy_fraction": (
                                float("nan")
                                if current_norm_sq <= EPS
                                else coefficient * coefficient / current_norm_sq
                            ),
                        }
                    )
                for keep in sorted({1, 2, 4, int(singular.numel())}):
                    if keep > int(singular.numel()):
                        continue
                    diagonal_capture = float(torch.sum(diagonal[:keep].square()))
                    full_subspace_capture = float(
                        torch.sum(coordinates[:keep, :keep].square())
                    )
                    final_top_m_energy = float(torch.sum(singular[:keep].square()))
                    subspace_rows.append(
                        {
                            "seed": seed,
                            "client": client_id,
                            "step": step,
                            "layer": layer,
                            "target": target,
                            "module": module_name,
                            "top_m": keep,
                            "diagonal_mode_capture": (
                                float("nan")
                                if current_norm_sq <= EPS
                                else diagonal_capture / current_norm_sq
                            ),
                            "full_subspace_capture": (
                                float("nan")
                                if current_norm_sq <= EPS
                                else full_subspace_capture / current_norm_sq
                            ),
                            "final_top_m_formation_progress": (
                                float("nan")
                                if final_top_m_energy <= EPS
                                else full_subspace_capture / final_top_m_energy
                            ),
                        }
                    )
    return component_rows, subspace_rows, stability_rows


def truncate_state(
    state: Mapping[str, torch.Tensor], module_names: Sequence[str], rank_keep: int, full_rank: int
) -> AdapterState:
    output: AdapterState = {}
    for module_name in module_names:
        a, b = module_pair(state, module_name)
        u, singular, v = compact_svd(a, b)
        truncated = singular.clone()
        truncated[rank_keep:] = 0.0
        a_new, b_new = factors_from_svd(u, truncated, v, full_rank)
        output[state_key(module_name, "A")] = a_new
        output[state_key(module_name, "B")] = b_new
    return output


def merged_state(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
    rank: int,
) -> Tuple[AdapterState, float, float]:
    output: AdapterState = {}
    retained: List[float] = []
    total_energy = 0.0
    kept_energy = 0.0
    for module_name in module_names:
        a_i, b_i = module_pair(left, module_name)
        a_j, b_j = module_pair(right, module_name)
        a_new, b_new, energy_retained = compress_sum_of_pairs(
            [(a_i, b_i), (a_j, b_j)], [0.5, 0.5], rank
        )
        output[state_key(module_name, "A")] = a_new
        output[state_key(module_name, "B")] = b_new
        retained.append(energy_retained)
        module_energy = max(
            0.25 * low_rank_norm_sq(a_i, b_i)
            + 0.25 * low_rank_norm_sq(a_j, b_j)
            + 0.5 * float(frobenius_inner_low_rank(a_i, b_i, a_j, b_j)),
            0.0,
        )
        total_energy += module_energy
        kept_energy += energy_retained * module_energy
    globally_weighted = 1.0 if total_energy <= EPS else kept_energy / total_energy
    return output, float(np.mean(retained)), globally_weighted


def state_operator_norm_and_distance(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    module_names: Sequence[str],
    scale: float,
) -> Tuple[float, float]:
    """Return ||left|| and ||left-right|| over all LoRA target operators."""
    left_sq = 0.0
    distance_sq = 0.0
    for module_name in module_names:
        a_left, b_left = module_pair(left, module_name)
        a_right, b_right = module_pair(right, module_name)
        norm_left = low_rank_norm_sq(a_left, b_left)
        norm_right = low_rank_norm_sq(a_right, b_right)
        cross = float(frobenius_inner_low_rank(a_left, b_left, a_right, b_right))
        left_sq += scale * scale * norm_left
        distance_sq += scale * scale * max(norm_left + norm_right - 2.0 * cross, 0.0)
    return math.sqrt(max(left_sq, 0.0)), math.sqrt(max(distance_sq, 0.0))


def paired_loss_uncertainty(
    local_losses: Sequence[float],
    merged_losses: Sequence[float],
    tasks: Sequence[str],
    labels: Sequence[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Task-by-label stratified paired uncertainty for local-minus-merged loss."""
    local = np.asarray(local_losses, dtype=np.float64)
    merged = np.asarray(merged_losses, dtype=np.float64)
    if (
        local.shape != merged.shape
        or local.ndim != 1
        or local.size == 0
        or len(tasks) != local.size
        or len(labels) != local.size
    ):
        raise ValueError("Paired losses, tasks, and labels must be non-empty and aligned")
    if not bool(np.isfinite(local).all()) or not bool(np.isfinite(merged).all()):
        raise ValueError("Paired compatibility losses must all be finite")
    differences = local - merged
    strata: Dict[Tuple[str, int], np.ndarray] = {}
    for key in sorted(set(zip((str(value) for value in tasks), (int(v) for v in labels)))):
        indices = np.asarray(
            [
                index
                for index, (task, label) in enumerate(zip(tasks, labels))
                if str(task) == key[0] and int(label) == key[1]
            ],
            dtype=np.int64,
        )
        if indices.size:
            strata[key] = indices
    lower = float("nan")
    upper = float("nan")
    standard_error = float("nan")
    if bootstrap_samples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        means = np.empty(bootstrap_samples, dtype=np.float64)
        for index in range(bootstrap_samples):
            weighted_mean = 0.0
            for indices in strata.values():
                sampled = indices[
                    rng.integers(0, indices.size, size=indices.size)
                ]
                weighted_mean += (
                    indices.size / differences.size
                ) * float(np.mean(differences[sampled]))
            means[index] = weighted_mean
        lower, upper = (float(value) for value in np.quantile(means, [0.025, 0.975]))
        if bootstrap_samples > 1:
            standard_error = float(np.std(means, ddof=1))
    return {
        "paired_loss_utility_mean": float(np.mean(differences)),
        "stratified_bootstrap_standard_error": standard_error,
        "paired_loss_utility_bootstrap_ci95_low": lower,
        "paired_loss_utility_bootstrap_ci95_high": upper,
        "bootstrap_strata": len(strata),
    }


def evaluate_trajectory_checkpoints(
    seed: int,
    arm: str,
    model: nn.Module,
    registry: AdapterRegistry,
    tokenizer: Any,
    device: torch.device,
    precision: str,
    eval_batch_size: int,
    data: Mapping[str, ClientData],
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    checkpoint_fractions: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for client_id, trajectory in sorted(trajectories.items()):
        steps = closest_steps(sorted(trajectory), checkpoint_fractions)
        for step in steps:
            load_state(registry, trajectory[step])
            metrics = evaluate(
                model,
                data[client_id].reference,
                tokenizer,
                device,
                eval_batch_size,
                precision,
            )
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "client": client_id,
                    "step": step,
                    "checkpoint_fraction": step / max(trajectory),
                    "split": "reference",
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                    "n": metrics["n"],
                    **flattened_task_metrics(metrics),
                }
            )
    return rows


def evaluate_spectral_truncations(
    seed: int,
    model: nn.Module,
    registry: AdapterRegistry,
    tokenizer: Any,
    device: torch.device,
    precision: str,
    eval_batch_size: int,
    data: Mapping[str, ClientData],
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    module_names: Sequence[str],
    rank: int,
    requested_ranks: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ranks = sorted({min(rank, max(1, int(value))) for value in requested_ranks} | {rank})
    for client_id, trajectory in sorted(trajectories.items()):
        final_step = max(trajectory)
        final_state = trajectory[final_step]
        for keep in ranks:
            state = final_state if keep == rank else truncate_state(
                final_state, module_names, keep, rank
            )
            load_state(registry, state)
            metrics = evaluate(
                model,
                data[client_id].reference,
                tokenizer,
                device,
                eval_batch_size,
                precision,
            )
            rows.append(
                {
                    "seed": seed,
                    "client": client_id,
                    "final_step": final_step,
                    "retained_rank_per_target": keep,
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                    "n": metrics["n"],
                    **flattened_task_metrics(metrics),
                }
            )
    return rows


def evaluate_compatibility(
    seed: int,
    model: nn.Module,
    registry: AdapterRegistry,
    tokenizer: Any,
    device: torch.device,
    precision: str,
    eval_batch_size: int,
    specs: Sequence[ClientSpec],
    data: Mapping[str, ClientData],
    trajectories: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
    module_names: Sequence[str],
    rank: int,
    checkpoint_fractions: Sequence[float],
    bootstrap_samples: int,
    arm: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    common_steps = sorted(
        set.intersection(*(set(trajectory) for trajectory in trajectories.values()))
    )
    selected_steps = closest_steps(common_steps, checkpoint_fractions)
    specs_by_id = {spec.client_id: spec for spec in specs}
    client_ids = sorted(trajectories)
    for step in selected_steps:
        local_metrics: Dict[str, Dict[str, Any]] = {}
        for client_id in client_ids:
            load_state(registry, trajectories[client_id][step])
            local_metrics[client_id] = evaluate(
                model,
                data[client_id].compatibility,
                tokenizer,
                device,
                eval_batch_size,
                precision,
                collect_examples=True,
            )
        for left_index in range(len(client_ids)):
            for right_index in range(left_index + 1, len(client_ids)):
                left_id = client_ids[left_index]
                right_id = client_ids[right_index]
                merged, retained, globally_weighted_retained = merged_state(
                    trajectories[left_id][step],
                    trajectories[right_id][step],
                    module_names,
                    rank,
                )
                load_state(registry, merged)
                for receiver, donor in ((left_id, right_id), (right_id, left_id)):
                    local_norm, displacement = state_operator_norm_and_distance(
                        trajectories[receiver][step],
                        merged,
                        module_names,
                        scale=float(next(iter(registry.values())).scale),
                    )
                    merged_metrics = evaluate(
                        model,
                        data[receiver].compatibility,
                        tokenizer,
                        device,
                        eval_batch_size,
                        precision,
                        collect_examples=True,
                    )
                    local = local_metrics[receiver]
                    for field in (
                        "example_source_ids",
                        "example_tasks",
                        "example_labels",
                    ):
                        if local[field] != merged_metrics[field]:
                            raise AssertionError(
                                f"Compatibility evaluations lost paired {field} order"
                            )
                    bootstrap_key = {
                        "seed": seed,
                        "arm": arm,
                        "step": step,
                        "receiver": receiver,
                        "donor": donor,
                    }
                    bootstrap_seed = int(stable_hash(bootstrap_key)[:16], 16) % (2**32)
                    paired_uncertainty = paired_loss_uncertainty(
                        local["example_losses"],
                        merged_metrics["example_losses"],
                        local["example_tasks"],
                        local["example_labels"],
                        bootstrap_samples,
                        bootstrap_seed,
                    )
                    task_columns: Dict[str, Any] = {}
                    for task in TASKS:
                        local_task = local["per_task"].get(task)
                        merged_task = merged_metrics["per_task"].get(task)
                        for metric_name in ("loss", "accuracy"):
                            local_value = (
                                float("nan")
                                if local_task is None
                                else float(local_task[metric_name])
                            )
                            merged_value = (
                                float("nan")
                                if merged_task is None
                                else float(merged_task[metric_name])
                            )
                            task_columns[f"local_{task}_{metric_name}"] = local_value
                            task_columns[f"merged_{task}_{metric_name}"] = merged_value
                            task_columns[f"{task}_{metric_name}_utility"] = (
                                local_value - merged_value
                                if metric_name == "loss"
                                else merged_value - local_value
                            )
                    rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "step": step,
                            "checkpoint_fraction": step / max(common_steps),
                            "receiver": receiver,
                            "donor": donor,
                            "undirected_pair": "--".join(sorted((receiver, donor))),
                            "pair_type": pair_type(specs_by_id[receiver], specs_by_id[donor]),
                            "local_loss": local["loss"],
                            "merged_loss": merged_metrics["loss"],
                            "loss_utility": local["loss"] - merged_metrics["loss"],
                            "relative_loss_utility": (
                                (local["loss"] - merged_metrics["loss"])
                                / max(abs(float(local["loss"])), EPS)
                            ),
                            "local_accuracy": local["accuracy"],
                            "merged_accuracy": merged_metrics["accuracy"],
                            "accuracy_utility": merged_metrics["accuracy"] - local["accuracy"],
                            "mean_operator_energy_retained": retained,
                            "global_energy_weighted_operator_retained": (
                                globally_weighted_retained
                            ),
                            "local_operator_norm": local_norm,
                            "local_to_merged_operator_distance": displacement,
                            "relative_operator_displacement": (
                                float("nan") if local_norm <= EPS else displacement / local_norm
                            ),
                            "loss_utility_per_operator_displacement": (
                                float("nan")
                                if displacement <= EPS
                                else (local["loss"] - merged_metrics["loss"]) / displacement
                            ),
                            "loss_utility_per_relative_operator_displacement": (
                                float("nan")
                                if local_norm <= EPS or displacement <= EPS
                                else (local["loss"] - merged_metrics["loss"])
                                / (displacement / local_norm)
                            ),
                            "n": merged_metrics["n"],
                            **paired_uncertainty,
                            **task_columns,
                        }
                    )
    return rows


def mean_directed_compatibility(
    compatibility: Sequence[Mapping[str, Any]],
    step: int,
    field: str = "loss_utility",
) -> Dict[Tuple[str, str], float]:
    grouped: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = defaultdict(dict)
    for row in compatibility:
        if int(row["step"]) != step:
            continue
        key = tuple(sorted((str(row["receiver"]), str(row["donor"]))))
        value = float(row[field])
        if math.isfinite(value):
            grouped[key][(str(row["receiver"]), str(row["donor"]))] = value
    return {
        key: 0.5 * (directions[(key[0], key[1])] + directions[(key[1], key[0])])
        for key, directions in grouped.items()
        if (key[0], key[1]) in directions and (key[1], key[0]) in directions
    }


def directional_compatibility_summary_rows(
    seed: int,
    arm: str,
    compatibility: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for step in sorted({int(row["step"]) for row in compatibility}):
        by_pair: Dict[Tuple[str, str], Dict[Tuple[str, str], float]] = defaultdict(dict)
        for row in compatibility:
            if int(row["step"]) != step:
                continue
            pair = tuple(sorted((str(row["receiver"]), str(row["donor"]))))
            value = float(row["loss_utility"])
            if math.isfinite(value):
                by_pair[pair][(str(row["receiver"]), str(row["donor"]))] = value
        for pair, directions in sorted(by_pair.items()):
            forward = directions.get((pair[0], pair[1]), float("nan"))
            reverse = directions.get((pair[1], pair[0]), float("nan"))
            complete = math.isfinite(forward) and math.isfinite(reverse)
            output.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "step": step,
                    "client_i": pair[0],
                    "client_j": pair[1],
                    "both_directions_complete": int(complete),
                    "i_receives_j_loss_utility": forward,
                    "j_receives_i_loss_utility": reverse,
                    "mean_bidirectional_loss_utility": (
                        0.5 * (forward + reverse) if complete else float("nan")
                    ),
                    "worst_direction_loss_utility": (
                        min(forward, reverse) if complete else float("nan")
                    ),
                    "absolute_directional_asymmetry": (
                        abs(forward - reverse) if complete else float("nan")
                    ),
                    "either_direction_negative_transfer": (
                        int(min(forward, reverse) < 0) if complete else ""
                    ),
                }
            )
    return output


def exact_qap_pvalue(
    similarity_by_pair: Mapping[Tuple[str, str], float],
    utility_by_pair: Mapping[Tuple[str, str], float],
    clients: Sequence[str],
) -> Tuple[float, float, int, str]:
    """Exact two-sided node-label QAP test for up to eight clients."""
    clients = sorted(set(clients))
    expected = {
        (clients[left], clients[right])
        for left in range(len(clients))
        for right in range(left + 1, len(clients))
    }
    if set(similarity_by_pair) != expected or set(utility_by_pair) != expected:
        return float("nan"), float("nan"), 0, "incomplete_pair_matrix"
    if not all(
        math.isfinite(float(similarity_by_pair[key]))
        and math.isfinite(float(utility_by_pair[key]))
        for key in expected
    ):
        return float("nan"), float("nan"), 0, "nonfinite_pair_value"
    common = sorted(expected)
    observed = spearman(
        [similarity_by_pair[key] for key in common],
        [utility_by_pair[key] for key in common],
    )
    if len(clients) > 8 or not math.isfinite(observed):
        return observed, float("nan"), 0, "unsupported_or_nonfinite_statistic"
    exceed = 0
    permutations_tested = 0
    for permutation in itertools.permutations(clients):
        mapping = dict(zip(clients, permutation))
        permuted_similarity: List[float] = []
        utilities: List[float] = []
        for key in common:
            mapped = tuple(sorted((mapping[key[0]], mapping[key[1]])))
            permuted_similarity.append(similarity_by_pair[mapped])
            utilities.append(utility_by_pair[key])
        statistic = spearman(permuted_similarity, utilities)
        if math.isfinite(statistic) and abs(statistic) >= abs(observed) - 1e-12:
            exceed += 1
        permutations_tested += 1
    return (
        observed,
        exceed / max(permutations_tested, 1),
        permutations_tested,
        "ok",
    )


def matrix_relation_rows(
    seed: int,
    similarities: Sequence[Mapping[str, Any]],
    compatibility: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Seed-level summaries; client pairs are not treated as independent samples."""
    rows: List[Dict[str, Any]] = []
    clients = sorted(
        {
            str(row[field])
            for row in compatibility
            for field in ("receiver", "donor")
        }
        | {
            str(row[field])
            for row in similarities
            for field in ("client_i", "client_j")
        }
    )
    expected_pairs = {
        (clients[left], clients[right])
        for left in range(len(clients))
        for right in range(left + 1, len(clients))
    }
    compat_steps = sorted({int(row["step"]) for row in compatibility})
    for step in compat_steps:
        expected_modules = {
            str(row["module"])
            for row in similarities
            if int(row["step"]) == step
        }
        by_representation: Dict[
            str, Dict[Tuple[str, str], Dict[str, float]]
        ] = {
            "a": defaultdict(dict),
            "delta_a": defaultdict(dict),
            "b": defaultdict(dict),
            "b_weighted_a0": defaultdict(dict),
            "ba": defaultdict(dict),
        }
        for row in similarities:
            if int(row["step"]) != step:
                continue
            key = tuple(sorted((str(row["client_i"]), str(row["client_j"]))))
            module = str(row["module"])
            for representation, field in (
                ("a", "cos_a"),
                ("delta_a", "cos_delta_a"),
                ("b", "cos_b"),
                ("b_weighted_a0", "cos_b_weighted_a0"),
                ("ba", "cos_ba"),
            ):
                value = float(row[field])
                if math.isfinite(value):
                    by_representation[representation][key][module] = value

        averaged_by_representation: Dict[
            str, Dict[Tuple[str, str], float]
        ] = {}
        coverage_by_representation: Dict[str, Dict[str, Any]] = {}
        for representation, values_by_pair in by_representation.items():
            pair_sets = [set(values_by_pair.get(pair, {})) for pair in expected_pairs]
            common_modules = set.intersection(*pair_sets) if pair_sets else set()
            union_modules = set.union(*pair_sets) if pair_sets else set()
            averaged_by_representation[representation] = {
                pair: float(
                    np.mean([values_by_pair[pair][module] for module in common_modules])
                )
                for pair in expected_pairs
                if common_modules
            }
            coverage_by_representation[representation] = {
                "common_module_count": len(common_modules),
                "available_module_union_count": len(union_modules),
                "expected_module_count": len(expected_modules),
                "module_coverage_fraction": (
                    len(common_modules) / len(expected_modules)
                    if expected_modules
                    else float("nan")
                ),
            }
        for utility_field, metric_name in (
            ("loss_utility", "spearman_similarity_vs_loss_utility"),
            (
                "loss_utility_per_relative_operator_displacement",
                "spearman_similarity_vs_displacement_adjusted_loss_utility",
            ),
        ):
            utilities = mean_directed_compatibility(
                compatibility, step, utility_field
            )
            for representation, averaged in averaged_by_representation.items():
                observed, qap_pvalue, qap_permutations, qap_status = exact_qap_pvalue(
                    averaged, utilities, clients
                )
                missing_similarity = sorted(expected_pairs - set(averaged))
                missing_utility = sorted(expected_pairs - set(utilities))
                rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "representation": representation,
                        "pairs": len(set(averaged) & set(utilities)),
                        "expected_pairs": len(expected_pairs),
                        "missing_similarity_pairs": len(missing_similarity),
                        "missing_utility_pairs": len(missing_utility),
                        "missing_similarity_pair_ids": ";".join(
                            f"{left}|{right}" for left, right in missing_similarity
                        ),
                        "missing_utility_pair_ids": ";".join(
                            f"{left}|{right}" for left, right in missing_utility
                        ),
                        **coverage_by_representation[representation],
                        "metric_value": observed,
                        "metric_name": metric_name,
                        "exact_qap_two_sided_p": qap_pvalue,
                        "qap_permutations": qap_permutations,
                        "qap_status": qap_status,
                        "analysis_role": (
                            "unconditional_task_composition_confounded_exploratory"
                        ),
                    }
                )
        # B-vs-BA agreement is computed on the same 15 pair matrix, averaged
        # over layers only after all layer-wise values have been retained.
        b_values = averaged_by_representation["b"]
        ba_values = averaged_by_representation["ba"]
        complete = set(b_values) == expected_pairs and set(ba_values) == expected_pairs
        common = sorted(expected_pairs) if complete else []
        rows.append(
            {
                "seed": seed,
                "step": step,
                "representation": "b_vs_ba",
                "pairs": len(common),
                "expected_pairs": len(expected_pairs),
                "metric_value": spearman(
                    [b_values[key] for key in common],
                    [ba_values[key] for key in common],
                ),
                "metric_name": "spearman_b_vs_ba",
                "exact_qap_two_sided_p": float("nan"),
                "qap_permutations": 0,
                "qap_status": "not_applicable",
                "analysis_role": "descriptive_proxy_agreement",
            }
        )
    return rows


def formation_time_rows(
    spectral: Sequence[Mapping[str, Any]],
    threshold: float,
    sustain_steps: int,
    formation_type: str,
) -> List[Dict[str, Any]]:
    if formation_type not in {"positive_alignment", "absolute_amplitude"}:
        raise ValueError(f"Unknown formation type: {formation_type}")
    progress_field = (
        "signed_progress"
        if formation_type == "positive_alignment"
        else "absolute_progress"
    )
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in spectral:
        key = (
            row["seed"],
            row["client"],
            row["module"],
            row["rank_index"],
        )
        grouped[key].append(row)
    output: List[Dict[str, Any]] = []
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: int(row["step"]))
        above_floor = int(ordered[0]["above_relative_floor"]) == 1
        tau: Optional[int] = None
        confirmation_step: Optional[int] = None
        interval_left: Optional[int] = None
        confirmable_candidate_steps: List[int] = []
        for index, candidate in enumerate(ordered):
            candidate_step = int(candidate["step"])
            confirm_index = next(
                (
                    later
                    for later in range(index, len(ordered))
                    if int(ordered[later]["step"]) >= candidate_step + sustain_steps
                ),
                None,
            )
            if confirm_index is None:
                continue
            confirmable_candidate_steps.append(candidate_step)
            if not above_floor:
                continue
            observed_window = ordered[index : confirm_index + 1]
            if all(
                math.isfinite(float(row[progress_field]))
                and float(row[progress_field]) >= threshold
                for row in observed_window
            ):
                tau = candidate_step
                confirmation_step = int(ordered[confirm_index]["step"])
                interval_left = int(ordered[index - 1]["step"]) if index > 0 else 0
                break
        first = ordered[0]
        final_step = max(int(row["step"]) for row in ordered)
        last_confirmable = (
            max(confirmable_candidate_steps) if confirmable_candidate_steps else None
        )
        signed_values = [float(row["signed_progress"]) for row in ordered]
        nonzero_signs = [
            1 if value > 0 else -1
            for value in signed_values
            if math.isfinite(value) and abs(value) > EPS
        ]
        sign_changes = sum(
            left != right for left, right in zip(nonzero_signs, nonzero_signs[1:])
        )
        first_positive = next(
            (
                int(row["step"])
                for row in ordered
                if math.isfinite(float(row["signed_progress"]))
                and float(row["signed_progress"]) > 0
            ),
            None,
        )
        output.append(
            {
                "seed": key[0],
                "client": key[1],
                "module": key[2],
                "layer": first["layer"],
                "target": first["target"],
                "rank_index": key[3],
                "sigma_final": first["sigma_final"],
                "degeneracy_group": first["degeneracy_group"],
                "original_degeneracy_group_size": first[
                    "original_degeneracy_group_size"
                ],
                "above_relative_floor": int(above_floor),
                "threshold": threshold,
                "formation_type": formation_type,
                "progress_field": progress_field,
                "sustain_steps": sustain_steps,
                "event_definition": "sampled_sustained_event",
                "formation_step": "" if tau is None else tau,
                "formation_interval_left_step": "" if interval_left is None else interval_left,
                "confirmation_step": "" if confirmation_step is None else confirmation_step,
                "observed_sustain_steps": (
                    "" if tau is None or confirmation_step is None else confirmation_step - tau
                ),
                "formed": int(tau is not None),
                "event_status": (
                    "excluded_below_relative_singular_floor"
                    if not above_floor
                    else (
                        "observed_sampled_sustained_event"
                        if tau is not None
                        else (
                            "not_formed_by_last_confirmable_candidate"
                            if last_confirmable is not None
                            else "insufficient_followup_to_confirm_any_candidate"
                        )
                    )
                ),
                "final_observed_step": final_step,
                "last_confirmable_candidate_step": (
                    "" if last_confirmable is None else last_confirmable
                ),
                "tail_without_sufficient_followup": int(
                    last_confirmable is None or last_confirmable < final_step
                ),
                "ever_negative_signed_progress": int(
                    any(math.isfinite(value) and value < 0 for value in signed_values)
                ),
                "signed_progress_sign_changes": sign_changes,
                "first_positive_checkpoint": (
                    "" if first_positive is None else first_positive
                ),
                "minimum_signed_progress": min(
                    (value for value in signed_values if math.isfinite(value)),
                    default=float("nan"),
                ),
            }
        )
    return output


def module_order_rows(formation: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in formation:
        grouped[
            (
                row["seed"],
                row["client"],
                row["module"],
                row["threshold"],
                row["formation_type"],
            )
        ].append(row)
    output: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        # Individual singular vectors are not identifiable within a nearly
        # degenerate block.  The primary ordering statistic therefore uses
        # singleton blocks only; cumulative subspace capture remains saved
        # separately for the degenerate case.
        usable = [
            row
            for row in rows
            if int(row["above_relative_floor"]) == 1
            and int(row["original_degeneracy_group_size"]) == 1
        ]
        formed = [row for row in usable if int(row["formed"]) == 1]
        ranks = [float(row["rank_index"]) for row in formed]
        times = [float(row["formation_step"]) for row in formed]
        comparable = 0
        concordant = 0
        discordant = 0
        tied = 0
        # rank_index is the final-energy order: the smaller index should form first.
        for high, low in itertools.combinations(
            sorted(usable, key=lambda row: int(row["rank_index"])), 2
        ):
            high_formed = int(high["formed"]) == 1
            low_formed = int(low["formed"]) == 1
            if not high_formed and not low_formed:
                continue
            comparable += 1
            if high_formed and not low_formed:
                concordant += 1
            elif not high_formed and low_formed:
                discordant += 1
            else:
                high_time = int(high["formation_step"])
                low_time = int(low["formation_step"])
                if high_time < low_time:
                    concordant += 1
                elif high_time > low_time:
                    discordant += 1
                else:
                    tied += 1
        output.append(
            {
                "seed": key[0],
                "client": key[1],
                "module": key[2],
                "threshold": key[3],
                "formation_type": key[4],
                "total_directions": len(rows),
                "directions_above_floor": sum(
                    int(row["above_relative_floor"]) for row in rows
                ),
                "directions_excluded_below_floor": sum(
                    int(row["above_relative_floor"]) == 0 for row in rows
                ),
                "nondegenerate_directions_used": len(usable),
                "degenerate_directions_excluded": sum(
                    int(row["above_relative_floor"]) == 1
                    and int(row["original_degeneracy_group_size"]) > 1
                    for row in rows
                ),
                "formed_fraction": (
                    float(np.mean([int(row["formed"]) for row in usable]))
                    if usable
                    else float("nan")
                ),
                "formed_directions": len(formed),
                "meets_three_formed_direction_interpretation_threshold": int(
                    len(formed) >= 3
                ),
                # Conditional on observed sustained formation. Positive rho means
                # lower-energy (larger rank index) directions formed later.
                "conditional_spearman_rank_vs_formation_step": spearman(ranks, times),
                "comparable_direction_pairs": comparable,
                "concordant_direction_pairs": concordant,
                "discordant_direction_pairs": discordant,
                "tied_direction_pairs": tied,
                "uninformative_both_unformed_pairs": (
                    math.comb(len(usable), 2) - comparable
                    if len(usable) >= 2
                    else 0
                ),
                "discrete_sustained_event_pairwise_c_index": (
                    float("nan")
                    if comparable == 0
                    else (concordant + 0.5 * tied) / comparable
                ),
                "c_index_is_finite": int(comparable > 0),
                "conditional_spearman_is_finite": int(
                    math.isfinite(spearman(ranks, times))
                ),
            }
        )
    return output


def spectral_functional_alignment_rows(
    seed: int,
    subspace: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Join already-computed spectral formation and task metrics without new forwards."""
    base_by_client = {
        str(row["client"]): row
        for row in validation
        if int(row["step"]) == 0
    }
    output: List[Dict[str, Any]] = []
    top_values = sorted({int(row["top_m"]) for row in subspace})
    for metric in validation:
        client_id = str(metric["client"])
        step = int(metric["step"])
        if client_id not in base_by_client:
            continue
        base = base_by_client[client_id]
        for top_m in top_values:
            formation_values = [
                float(row["final_top_m_formation_progress"])
                for row in subspace
                if str(row["client"]) == client_id
                and int(row["step"]) == step
                and int(row["top_m"]) == top_m
            ]
            output.append(
                {
                    "seed": seed,
                    "analysis_role": "descriptive_time_confounded",
                    "client": client_id,
                    "step": step,
                    "top_m": top_m,
                    "mean_final_top_m_formation_progress": finite_mean(
                        formation_values
                    ),
                    "reference_loss": float(metric["loss"]),
                    "reference_loss_improvement_from_step0": (
                        float(base["loss"]) - float(metric["loss"])
                    ),
                    "reference_accuracy": float(metric["accuracy"]),
                    "reference_accuracy_improvement_from_step0": (
                        float(metric["accuracy"]) - float(base["accuracy"])
                    ),
                }
            )
    return output


def plotting_backend() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


PAIR_COLORS = {
    "same_sst2": "#1f77b4",
    "same_qnli": "#2ca02c",
    "same_mixed": "#9467bd",
    "sst2_qnli": "#d62728",
    "sst2_mixed": "#ff7f0e",
    "qnli_mixed": "#17becf",
}


def finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def finite_median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else float("nan")


def finite_quantile(values: Sequence[float], quantile: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.quantile(finite, quantile)) if finite else float("nan")


def plot_layerwise_similarity(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    plt = plotting_backend()
    steps = sorted({int(row["step"]) for row in rows})
    if not steps:
        return
    selected_steps = closest_steps(steps, (0.10, 0.50, 1.00))
    representations = (("cos_a", "A"), ("cos_b", "B"), ("cos_ba", "BA"))
    for step in selected_steps:
        figure, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True, sharey=False)
        for row_index, target in enumerate(TARGET_NAMES):
            for column_index, (field, title) in enumerate(representations):
                axis = axes[row_index, column_index]
                subset = [
                    row for row in rows
                    if int(row["step"]) == step and str(row["target"]) == target
                ]
                by_pair: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
                for row in subset:
                    by_pair[(str(row["client_i"]), str(row["client_j"]))].append(row)
                for pair, pair_rows in sorted(by_pair.items()):
                    ordered = sorted(pair_rows, key=lambda item: int(item["layer"]))
                    pair_group = str(ordered[0]["pair_type"])
                    axis.plot(
                        [int(item["layer"]) for item in ordered],
                        [float(item[field]) for item in ordered],
                        color=PAIR_COLORS.get(pair_group, "#777777"),
                        alpha=0.25,
                        linewidth=0.8,
                    )
                by_group_layer: Dict[Tuple[str, int], List[float]] = defaultdict(list)
                for row in subset:
                    by_group_layer[(str(row["pair_type"]), int(row["layer"]))].append(
                        float(row[field])
                    )
                for pair_group in sorted({key[0] for key in by_group_layer}):
                    layers = sorted(key[1] for key in by_group_layer if key[0] == pair_group)
                    axis.plot(
                        layers,
                        [finite_mean(by_group_layer[(pair_group, layer)]) for layer in layers],
                        color=PAIR_COLORS.get(pair_group, "#777777"),
                        linewidth=2.1,
                        label=pair_group,
                    )
                axis.axhline(0.0, color="#999999", linewidth=0.6)
                axis.set_title(f"{target} / {title}")
                axis.set_xlabel("Transformer layer")
                axis.set_ylabel("Cosine similarity")
                axis.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
        figure.suptitle(f"Layer-wise client similarity, seed={seed}, step={step}")
        figure.tight_layout(rect=(0, 0.07, 1, 0.96))
        path = output_dir / f"layerwise_similarity_seed{seed}_step{step}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)


def plot_similarity_heatmaps(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    plt = plotting_backend()
    steps = sorted({int(row["step"]) for row in rows})
    if not steps:
        return
    selected_steps = closest_steps(steps, (0.10, 0.50, 1.00))
    representations = (("cos_a", "A"), ("cos_b", "B"), ("cos_ba", "BA"))
    pairs = sorted(
        {(str(row["client_i"]), str(row["client_j"])) for row in rows}
    )
    layers = sorted({int(row["layer"]) for row in rows})
    for step in selected_steps:
        for target in TARGET_NAMES:
            figure, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
            for axis, (field, title) in zip(axes, representations):
                matrix = np.full((len(pairs), len(layers)), np.nan, dtype=np.float64)
                lookup = {
                    (
                        (str(row["client_i"]), str(row["client_j"])),
                        int(row["layer"]),
                    ): float(row[field])
                    for row in rows
                    if int(row["step"]) == step and str(row["target"]) == target
                }
                for pair_index, pair in enumerate(pairs):
                    for layer_index, layer in enumerate(layers):
                        matrix[pair_index, layer_index] = lookup.get((pair, layer), np.nan)
                image = axis.imshow(
                    matrix, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm"
                )
                axis.set_title(title)
                axis.set_xlabel("Transformer layer")
                axis.set_xticks(range(0, len(layers), max(1, len(layers) // 8)))
                axis.set_xticklabels(layers[:: max(1, len(layers) // 8)])
            axes[0].set_yticks(range(len(pairs)))
            axes[0].set_yticklabels(
                [f"{left} vs {right}" for left, right in pairs], fontsize=7
            )
            figure.colorbar(
                image, ax=axes, fraction=0.02, pad=0.02, label="Cosine similarity"
            )
            figure.suptitle(
                f"All 15 client pairs, seed={seed}, step={step}, target={target}"
            )
            figure.subplots_adjust(
                left=0.18, right=0.94, bottom=0.12, top=0.88, wspace=0.08
            )
            figure.savefig(
                output_dir / f"similarity_heatmap_seed{seed}_step{step}_{target}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)


def b_ba_agreement_rows(
    similarities: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in similarities:
        grouped[(int(row["layer"]), str(row["target"]), int(row["step"]))].append(row)
    output: List[Dict[str, Any]] = []
    for (layer, target, step), rows in sorted(grouped.items()):
        b = [float(row["cos_b"]) for row in rows]
        ba = [float(row["cos_ba"]) for row in rows]
        differences = [abs(x - y) for x, y in zip(b, ba) if math.isfinite(x) and math.isfinite(y)]
        output.append(
            {
                "layer": layer,
                "target": target,
                "step": step,
                "spearman_b_vs_ba": spearman(b, ba),
                "mae_b_vs_ba": finite_mean(differences),
            }
        )
    return output


def causal_control_rows(
    standard_similarity: Sequence[Mapping[str, Any]],
    standard_drift: Sequence[Mapping[str, Any]],
    freeze_similarity: Sequence[Mapping[str, Any]],
    freeze_drift: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Matched standard/freeze-A summaries over the three control clients."""
    freeze_clients = {
        str(row["client_i"]) for row in freeze_similarity
    } | {
        str(row["client_j"]) for row in freeze_similarity
    }
    output: List[Dict[str, Any]] = []
    for arm, similarities, drifts in (
        ("standard", standard_similarity, standard_drift),
        ("freeze_a", freeze_similarity, freeze_drift),
    ):
        matched_similarity = [
            row
            for row in similarities
            if str(row["client_i"]) in freeze_clients
            and str(row["client_j"]) in freeze_clients
        ]
        matched_drift = [
            row for row in drifts if str(row["client"]) in freeze_clients
        ]
        steps = sorted(
            {int(row["step"]) for row in matched_similarity}
            & {int(row["step"]) for row in matched_drift}
        )
        for step in steps:
            similarity_step = [
                row for row in matched_similarity if int(row["step"]) == step
            ]
            drift_step = [row for row in matched_drift if int(row["step"]) == step]
            plain_errors = [
                abs(float(row["cos_b"]) - float(row["cos_ba"]))
                for row in similarity_step
                if math.isfinite(float(row["cos_b"]))
                and math.isfinite(float(row["cos_ba"]))
            ]
            weighted_errors = [
                abs(float(row["cos_b_weighted_a0"]) - float(row["cos_ba"]))
                for row in similarity_step
                if math.isfinite(float(row["cos_b_weighted_a0"]))
                and math.isfinite(float(row["cos_ba"]))
            ]
            output.append(
                {
                    "arm": arm,
                    "step": step,
                    "matched_clients": len(freeze_clients),
                    "mean_a_relative_drift": finite_mean(
                        [float(row["a_relative_drift"]) for row in drift_step]
                    ),
                    "mean_abs_cos_b_minus_cos_ba": finite_mean(plain_errors),
                    "mean_abs_cos_b_weighted_a0_minus_cos_ba": finite_mean(
                        weighted_errors
                    ),
                }
            )
    return output


def plot_b_ba_agreement(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    plt = plotting_backend()
    agreement = b_ba_agreement_rows(rows)
    steps = sorted({int(row["step"]) for row in agreement})
    layers = sorted({int(row["layer"]) for row in agreement})
    if not steps or not layers:
        return
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for row_index, target in enumerate(TARGET_NAMES):
        for column_index, (field, title, vmin, vmax, cmap) in enumerate(
            (
                ("spearman_b_vs_ba", "Spearman(B, BA)", -1.0, 1.0, "coolwarm"),
                ("mae_b_vs_ba", "Mean |B - BA cosine|", 0.0, 1.0, "magma"),
            )
        ):
            matrix = np.full((len(layers), len(steps)), np.nan)
            for row in agreement:
                if row["target"] != target:
                    continue
                matrix[layers.index(int(row["layer"])), steps.index(int(row["step"]))] = float(
                    row[field]
                )
            image = axes[row_index, column_index].imshow(
                matrix, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap
            )
            axes[row_index, column_index].set_title(f"{target} / {title}")
            axes[row_index, column_index].set_xlabel("Checkpoint index")
            axes[row_index, column_index].set_ylabel("Transformer layer")
            axes[row_index, column_index].set_xticks(range(len(steps)))
            axes[row_index, column_index].set_xticklabels(steps, rotation=45, fontsize=7)
            axes[row_index, column_index].set_yticks(range(0, len(layers), max(1, len(layers) // 8)))
            axes[row_index, column_index].set_yticklabels(layers[:: max(1, len(layers) // 8)])
            figure.colorbar(image, ax=axes[row_index, column_index], fraction=0.046, pad=0.04)
    figure.suptitle(f"B versus BA agreement over time, seed={seed}")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / f"b_ba_agreement_seed{seed}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_causal_control(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    if not rows:
        return
    plt = plotting_backend()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"standard": "#1f77b4", "freeze_a": "#d62728"}
    for arm in ("standard", "freeze_a"):
        arm_rows = sorted(
            (row for row in rows if str(row["arm"]) == arm),
            key=lambda row: int(row["step"]),
        )
        if not arm_rows:
            continue
        x = [int(row["step"]) for row in arm_rows]
        axes[0].plot(
            x,
            [float(row["mean_a_relative_drift"]) for row in arm_rows],
            marker="o",
            color=colors[arm],
            label=arm,
        )
        axes[1].plot(
            x,
            [float(row["mean_abs_cos_b_minus_cos_ba"]) for row in arm_rows],
            marker="o",
            color=colors[arm],
            linestyle="-",
            label=f"{arm}: plain B",
        )
        axes[1].plot(
            x,
            [
                float(row["mean_abs_cos_b_weighted_a0_minus_cos_ba"])
                for row in arm_rows
            ],
            marker="s",
            color=colors[arm],
            linestyle="--",
            label=f"{arm}: B weighted by A0",
        )
    axes[0].set_title("A movement")
    axes[0].set_ylabel("Mean relative A drift")
    axes[1].set_title("B proxy error relative to BA")
    axes[1].set_ylabel("Mean absolute cosine difference")
    for axis in axes:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"Matched freeze-A causal control, seed={seed}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(
        output_dir / f"freeze_a_causal_control_seed{seed}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_spectral_progress(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    plt = plotting_backend()
    clients = sorted({str(row["client"]) for row in rows})
    ranks = sorted({int(row["rank_index"]) for row in rows})
    steps = sorted({int(row["step"]) for row in rows})
    for client_id in clients:
        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        client_rows = [
            row for row in rows
            if str(row["client"]) == client_id and int(row["above_relative_floor"]) == 1
        ]
        for rank_index in ranks:
            x: List[int] = []
            signed: List[float] = []
            absolute: List[float] = []
            for step in steps:
                subset = [
                    float(row["signed_progress"])
                    for row in client_rows
                    if int(row["rank_index"]) == rank_index and int(row["step"]) == step
                ]
                subset_abs = [
                    float(row["absolute_progress"])
                    for row in client_rows
                    if int(row["rank_index"]) == rank_index and int(row["step"]) == step
                ]
                if subset:
                    x.append(step)
                    signed.append(finite_mean(subset))
                    absolute.append(finite_mean(subset_abs))
            if x:
                axes[0].plot(x, signed, marker="o", markersize=3, label=f"rank {rank_index}")
                axes[1].plot(x, absolute, marker="o", markersize=3, label=f"rank {rank_index}")
        axes[0].axhline(0.0, color="#777777", linewidth=0.7)
        axes[0].set_title("Signed progress")
        axes[1].set_title("Absolute progress")
        for axis in axes:
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel("Projection / final singular value")
            axis.grid(alpha=0.2)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)), frameon=False)
        figure.suptitle(f"Final SVD direction formation: {client_id}, seed={seed}")
        figure.tight_layout(rect=(0, 0.10, 1, 0.94))
        figure.savefig(
            output_dir / f"spectral_progress_seed{seed}_{client_id}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_spectral_functional_alignment(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, seed: int
) -> None:
    if not rows:
        return
    plt = plotting_backend()
    top_values = sorted({int(row["top_m"]) for row in rows})
    columns = 2
    row_count = int(math.ceil(len(top_values) / columns))
    figure, axes = plt.subplots(
        row_count, columns, figsize=(12, 4.5 * row_count), squeeze=False
    )
    clients = sorted({str(row["client"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clients), 1)))
    for axis, top_m in zip(axes.flat, top_values):
        correlations: List[float] = []
        for color, client_id in zip(colors, clients):
            subset = sorted(
                (
                    row
                    for row in rows
                    if int(row["top_m"]) == top_m
                    and str(row["client"]) == client_id
                ),
                key=lambda row: int(row["step"]),
            )
            x = [
                float(row["mean_final_top_m_formation_progress"])
                for row in subset
            ]
            y = [
                float(row["reference_loss_improvement_from_step0"])
                for row in subset
            ]
            correlation = spearman(x, y)
            if math.isfinite(correlation):
                correlations.append(correlation)
            axis.plot(x, y, marker="o", linewidth=1.0, color=color, label=client_id)
        axis.axhline(0.0, color="#777777", linewidth=0.7)
        axis.set_title(
            f"top-{top_m}; median within-client rho={finite_median(correlations):.3f}"
        )
        axis.set_xlabel("Mean formation progress in final top-m subspace")
        axis.set_ylabel("Reference loss improvement from step 0")
        axis.grid(alpha=0.2)
    for axis in list(axes.flat)[len(top_values) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        f"Descriptive (time-confounded): spectral formation versus task improvement, seed={seed}"
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    figure.savefig(
        output_dir / f"spectral_functional_alignment_seed{seed}.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_compatibility_scatter(
    similarities: Sequence[Mapping[str, Any]],
    compatibility: Sequence[Mapping[str, Any]],
    output_dir: Path,
    seed: int,
) -> None:
    plt = plotting_backend()
    steps = sorted({int(row["step"]) for row in compatibility})
    for step in steps:
        utilities = mean_directed_compatibility(compatibility, step)
        averaged: Dict[str, Dict[Tuple[str, str], List[float]]] = {
            "cos_b": defaultdict(list),
            "cos_ba": defaultdict(list),
        }
        for row in similarities:
            if int(row["step"]) != step:
                continue
            key = tuple(sorted((str(row["client_i"]), str(row["client_j"]))))
            averaged["cos_b"][key].append(float(row["cos_b"]))
            averaged["cos_ba"][key].append(float(row["cos_ba"]))
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for axis, field, title in zip(axes, ("cos_b", "cos_ba"), ("B", "BA")):
            common = sorted(set(utilities) & set(averaged[field]))
            x = [finite_mean(averaged[field][key]) for key in common]
            y = [utilities[key] for key in common]
            axis.scatter(x, y, color="#1f77b4", alpha=0.85)
            for key, xx, yy in zip(common, x, y):
                axis.annotate(f"{key[0]}\n{key[1]}", (xx, yy), fontsize=6, alpha=0.75)
            axis.axhline(0.0, color="#d62728", linewidth=0.8)
            axis.set_xlabel(f"Mean layer-wise {title} cosine")
            axis.set_title(f"rho={spearman(x, y):.3f}")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Mean bidirectional loss utility")
        figure.suptitle(f"Similarity versus pairwise sharing utility, seed={seed}, step={step}")
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(
            output_dir / f"compatibility_scatter_seed{seed}_step{step}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def run_math_self_tests() -> Dict[str, float]:
    torch.manual_seed(7)
    dimensions = (17, 13, 4)
    out_features, in_features, rank = dimensions
    a_i = torch.randn(rank, in_features, dtype=torch.float64)
    b_i = torch.randn(out_features, rank, dtype=torch.float64)
    a_j = torch.randn(rank, in_features, dtype=torch.float64)
    b_j = torch.randn(out_features, rank, dtype=torch.float64)
    dense_i = b_i @ a_i
    dense_j = b_j @ a_j
    dense_cosine = float(
        torch.sum(dense_i * dense_j)
        / (torch.linalg.vector_norm(dense_i) * torch.linalg.vector_norm(dense_j))
    )
    compact_cosine = low_rank_cosine(a_i, b_i, a_j, b_j)
    cosine_error = abs(dense_cosine - compact_cosine)
    if cosine_error > 1e-10:
        raise AssertionError(f"Low-rank cosine error: {cosine_error}")

    u, singular, v = compact_svd(a_i, b_i)
    reconstructed = (u * singular.unsqueeze(0)) @ v.T
    svd_error = float(
        torch.linalg.vector_norm(reconstructed - dense_i)
        / torch.linalg.vector_norm(dense_i)
    )
    if svd_error > 1e-10:
        raise AssertionError(f"Compact SVD error: {svd_error}")

    a_merge, b_merge, retained = compress_sum_of_pairs(
        [(a_i, b_i), (a_j, b_j)], [0.5, 0.5], rank
    )
    target = 0.5 * dense_i + 0.5 * dense_j
    _, target_singular, _ = torch.linalg.svd(target, full_matrices=False)
    best_error_sq = float(torch.sum(target_singular[rank:].square()))
    observed_error_sq = float(
        torch.linalg.vector_norm(target - b_merge.to(torch.float64) @ a_merge.to(torch.float64))
        ** 2
    )
    merge_optimality_error = abs(observed_error_sq - best_error_sq) / max(
        float(torch.linalg.vector_norm(target) ** 2), EPS
    )
    if merge_optimality_error > 2e-6:
        raise AssertionError(f"Pair merge is not truncated-SVD optimal: {merge_optimality_error}")

    a_zero = torch.randn(rank, in_features, dtype=torch.float64)
    q, _ = torch.linalg.qr(a_zero.T, mode="reduced")
    a_orthogonal = q.T
    weighted = weighted_b_cosine(b_i, b_j, a_orthogonal)
    plain = flat_cosine(b_i, b_j)
    weighted_error = abs(weighted - plain)
    if weighted_error > 1e-10:
        raise AssertionError(f"Weighted-B identity failed: {weighted_error}")

    arbitrary_a0 = torch.randn(rank, in_features, dtype=torch.float64)
    weighted_arbitrary = weighted_b_cosine(b_i, b_j, arbitrary_a0)
    exact_fixed_a = low_rank_cosine(arbitrary_a0, b_i, arbitrary_a0, b_j)
    fixed_a_error = abs(weighted_arbitrary - exact_fixed_a)
    if fixed_a_error > 1e-10:
        raise AssertionError(f"Fixed-A weighted-B equivalence failed: {fixed_a_error}")

    scale = 0.7
    state_i = {"test::A": a_i, "test::B": b_i}
    state_j = {"test::A": a_j, "test::B": b_j}
    compact_norm, compact_distance = state_operator_norm_and_distance(
        state_i, state_j, ["test"], scale
    )
    dense_norm = scale * float(torch.linalg.vector_norm(dense_i))
    dense_distance = scale * float(torch.linalg.vector_norm(dense_i - dense_j))
    distance_error = max(
        abs(compact_norm - dense_norm),
        abs(compact_distance - dense_distance),
    )
    if distance_error > 1e-9:
        raise AssertionError(f"Low-rank operator distance failed: {distance_error}")

    checkpoint_test = automatic_checkpoints(160, interval_steps=8)
    if 80 not in checkpoint_test or max(
        right - left for left, right in zip(checkpoint_test, checkpoint_test[1:])
    ) > 8:
        raise AssertionError("Dense checkpoint schedule omitted 50% or exceeded interval")

    synthetic_spectral: List[Dict[str, Any]] = []
    progress_by_rank = {
        1: [0.0, 0.6, 0.7, 1.0],
        2: [0.0, 0.2, 0.6, 1.0],
        3: [0.0, 0.2, 0.3, 1.0],
    }
    for rank_index, progress_values in progress_by_rank.items():
        for step, value in zip((0, 8, 16, 24), progress_values):
            synthetic_spectral.append(
                {
                    "seed": 0,
                    "client": "test",
                    "module": "test",
                    "layer": 0,
                    "target": "query",
                    "rank_index": rank_index,
                    "sigma_final": float(4 - rank_index),
                    "degeneracy_group": rank_index,
                    "original_degeneracy_group_size": 1,
                    "above_relative_floor": 1,
                    "signed_progress": value,
                    "absolute_progress": abs(value),
                    "step": step,
                }
            )
    synthetic_formation = formation_time_rows(
        synthetic_spectral, 0.5, 8, "positive_alignment"
    )
    synthetic_order = module_order_rows(synthetic_formation)
    formation_c_index = float(
        synthetic_order[0]["discrete_sustained_event_pairwise_c_index"]
    )
    if abs(formation_c_index - 1.0) > 1e-12:
        raise AssertionError(
            f"Discrete sustained-event formation ordering failed: {formation_c_index}"
        )
    late_unconfirmed = next(
        row for row in synthetic_formation if int(row["rank_index"]) == 3
    )
    if (
        late_unconfirmed["event_status"]
        != "not_formed_by_last_confirmable_candidate"
        or int(late_unconfirmed["last_confirmable_candidate_step"]) != 16
        or int(late_unconfirmed["final_observed_step"]) != 24
    ):
        raise AssertionError("Discrete event confirmation boundary was recorded incorrectly")

    paired_test = paired_loss_uncertainty(
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        ["sst2", "sst2", "qnli"],
        [0, 1, 0],
        100,
        7,
    )
    if abs(paired_test["paired_loss_utility_mean"] - 0.5) > 1e-12:
        raise AssertionError("Paired compatibility loss calculation failed")

    split_degenerate = [
        {
            **row,
            "degeneracy_group": 0,
            "original_degeneracy_group_size": 2,
            "above_relative_floor": int(int(row["rank_index"]) == 1),
        }
        for row in synthetic_formation
        if int(row["rank_index"]) <= 2
    ]
    split_order = module_order_rows(split_degenerate)
    if int(split_order[0]["nondegenerate_directions_used"]) != 0:
        raise AssertionError("A partially filtered degenerate group was treated as identifiable")

    qap_similarity = {("a", "b"): 0.1, ("a", "c"): 0.2}
    qap_utility = {("a", "b"): 0.1, ("a", "c"): 0.2, ("b", "c"): 0.3}
    _, _, _, incomplete_status = exact_qap_pvalue(
        qap_similarity, qap_utility, ["a", "b", "c"]
    )
    if incomplete_status != "incomplete_pair_matrix":
        raise AssertionError("Incomplete QAP matrices were not rejected")
    _, complete_qap_p, complete_qap_permutations, complete_status = exact_qap_pvalue(
        qap_utility, qap_utility, ["a", "b", "c"]
    )
    if (
        complete_status != "ok"
        or complete_qap_permutations != 6
        or not math.isfinite(complete_qap_p)
    ):
        raise AssertionError("Complete exact QAP self-test failed")

    balanced_keeps = allocate_two_fields(100, 384, 106, 16)
    short_question_keeps = allocate_two_fields(10, 384, 106, 16)
    if balanced_keeps != (53, 53):
        raise AssertionError(
            f"Longest-first balanced truncation failed: {balanced_keeps}"
        )
    if short_question_keeps != (10, 96):
        raise AssertionError(
            f"Longest-first short-question preservation failed: {short_question_keeps}"
        )

    return {
        "low_rank_cosine_absolute_error": cosine_error,
        "compact_svd_relative_error": svd_error,
        "rank_r_merge_optimality_relative_error": merge_optimality_error,
        "orthogonal_a0_weighted_b_cosine_error": weighted_error,
        "arbitrary_fixed_a_weighted_b_cosine_error": fixed_a_error,
        "operator_norm_distance_absolute_error": distance_error,
        "merge_energy_retained": retained,
        "checkpoint_schedule_max_gap": float(
            max(right - left for left, right in zip(checkpoint_test, checkpoint_test[1:]))
        ),
        "synthetic_formation_c_index": formation_c_index,
        "synthetic_last_confirmable_candidate_step": int(
            late_unconfirmed["last_confirmable_candidate_step"]
        ),
        "paired_loss_utility_test_mean": paired_test["paired_loss_utility_mean"],
        "incomplete_qap_status": incomplete_status,
        "complete_qap_permutations": complete_qap_permutations,
        "balanced_longest_first_keeps": list(balanced_keeps),
        "short_question_longest_first_keeps": list(short_question_keeps),
    }


def scientific_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    excluded = {
        "seeds",
        "output_dir",
        "resume",
        "validate_only",
        "self_test_only",
        "smoke_test",
        "no_plots",
        "device",
        "allow_cpu",
    }
    arguments = {
        key: value for key, value in vars(args).items() if key not in excluded
    }
    return {
        "version": VERSION,
        "arguments": arguments,
        "clients": [asdict(spec) for spec in client_specs()],
        "head": "frozen_roberta_masked_lm_head",
        "verbalizers": [" no", " yes"],
        "trainable_targets": "all_transformer_query_and_value_lora_only",
        "similarity": "all_layers_A_deltaA_B_and_exact_low_rank_BA",
        "spectral_formation_thresholds": sorted(
            {0.25, args.formation_threshold, 0.75}
        ),
        "spectral_formation_primary": (
            "positive_alignment; sampled checkpoints must remain above threshold through a "
            f"confirmation checkpoint at least {args.formation_sustain_steps} optimizer steps later"
        ),
        "spectral_formation_secondary": "absolute_amplitude_with_sign_reversal_diagnostics",
        "dropout_protocol": (
            "standard train-mode base-model dropout plus configured LoRA dropout; dynamics are "
            "stochastic and inference is based on repetition across seeds"
        ),
        "pilot_guardrail": (
            "A seed used to choose the training endpoint is exploratory; confirmatory summaries "
            "should use fresh seeds after the endpoint and protocol are fixed."
        ),
        "primary_compatibility_endpoint": (
            "raw paired receiver loss utility at the earliest configured compatibility checkpoint; "
            "B-versus-BA predictive association is compared at seed level"
        ),
        "compatibility": (
            "directed_validation_utility_of_equal_operator_average_then_rank_r_compression; "
            "raw_and_relative-displacement-adjusted"
        ),
    }


def training_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    """Only values that can change trained checkpoints or their sampled trajectory."""
    keys = (
        "model_name",
        "model_revision",
        "dataset_revision",
        "rank",
        "lora_alpha",
        "lora_dropout",
        "learning_rate",
        "epochs",
        "train_per_client",
        "batch_size",
        "max_length",
        "qnli_min_field_tokens",
        "data_seed",
        "precision",
        "deterministic",
        "trajectory_checkpoint_interval",
        "evaluation_checkpoint_fractions",
        "compatibility_checkpoint_fractions",
        "freeze_compatibility_checkpoint_fractions",
        "stability_checkpoint_fractions",
    )
    return {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "arguments": {key: getattr(args, key) for key in keys},
        "clients": [asdict(spec) for spec in client_specs()],
        "prompt_protocol": {
            "sst2": {
                "prefix": SST2_PROMPT_PREFIX,
                "middle": SST2_PROMPT_MIDDLE,
                "suffix_template": SST2_PROMPT_SUFFIX,
            },
            "qnli": {
                "prefix": QNLI_PROMPT_PREFIX,
                "middle": QNLI_PROMPT_MIDDLE,
                "suffix_template": QNLI_PROMPT_SUFFIX,
            },
            "mask_placeholder": MASK_PLACEHOLDER,
            "dual_field_truncation_policy": DUAL_FIELD_TRUNCATION_POLICY,
        },
        "optimizer": "AdamW_weight_decay_0",
        "trainable_targets": "all_transformer_query_and_value_lora_only",
        "head": "frozen_roberta_masked_lm_head",
        "verbalizers": [" no", " yes"],
    }


def resource_estimate(args: argparse.Namespace) -> Dict[str, Any]:
    steps_per_client = int(math.ceil(args.train_per_client / args.batch_size)) * args.epochs
    standard_clients = len(args.standard_clients) if "standard" in args.arms else 0
    freeze_clients = len(args.freeze_a_clients) if "freeze_a" in args.arms else 0
    checkpoints = experiment_checkpoint_steps(args, steps_per_client)
    validation_points = len(
        closest_steps(checkpoints, args.evaluation_checkpoint_fractions)
    )
    compatibility_points = (
        0
        if args.skip_compatibility or (standard_clients == 0 and freeze_clients == 0)
        else len(closest_steps(checkpoints, args.compatibility_checkpoint_fractions))
    )
    freeze_compatibility_points = (
        0
        if (
            args.skip_compatibility
            or args.skip_freeze_compatibility
            or freeze_clients == 0
        )
        else len(
            closest_steps(
                checkpoints, args.freeze_compatibility_checkpoint_fractions
            )
        )
    )
    spectral_ranks = len(
        {min(args.rank, max(1, int(value))) for value in args.spectral_eval_ranks}
        | {args.rank}
    )
    validation_examples = (
        (standard_clients + freeze_clients)
        * validation_points
        * args.reference_per_client
    )
    spectral_examples = (
        0
        if not args.run_functional_spectral_eval or standard_clients == 0
        else standard_clients * spectral_ranks * args.reference_per_client
    )
    standard_compatibility_examples = (
        compatibility_points
        * (standard_clients + 2 * math.comb(standard_clients, 2))
        * args.compatibility_per_client
    )
    freeze_compatibility_examples = (
        0
        if (
            args.skip_compatibility
            or args.skip_freeze_compatibility
            or freeze_clients == 0
        )
        else freeze_compatibility_points
        * (freeze_clients + 2 * math.comb(freeze_clients, 2))
        * args.compatibility_per_client
    )
    return {
        "optimizer_steps_per_client": steps_per_client,
        "standard_client_trajectories_per_seed": standard_clients,
        "freeze_a_client_trajectories_per_seed": freeze_clients,
        "total_optimizer_steps_per_seed": steps_per_client * (standard_clients + freeze_clients),
        "training_examples_seen_per_seed": (
            args.train_per_client * args.epochs * (standard_clients + freeze_clients)
        ),
        "forward_only_examples_per_seed": (
            validation_examples
            + spectral_examples
            + standard_compatibility_examples
            + freeze_compatibility_examples
        ),
        "standard_compatibility_forward_examples_per_seed": standard_compatibility_examples,
        "freeze_compatibility_forward_examples_per_seed": freeze_compatibility_examples,
        "learning_rate_candidates": 1,
        "base_model_copies_resident": 1,
        "base_model_loads_per_invocation_with_pending_work": 1,
        "checkpoint_steps": checkpoints,
        "compatibility_forward_only_checkpoints": compatibility_points,
        "freeze_compatibility_forward_only_checkpoints": freeze_compatibility_points,
        "note": (
            "No federated rounds or LR grid; one base model is reused across clients and seeds. "
            "Completed seeds are skipped under --resume."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Single-file LoRA SVD dynamics and client matching experiment",
    )
    parser.add_argument("--model-name", default="FacebookAI/roberta-large")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face commit/tag for exact model reproducibility",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-per-client", type=int, default=500)
    parser.add_argument("--reference-per-client", type=int, default=100)
    parser.add_argument("--compatibility-per-client", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--qnli-min-field-tokens",
        type=int,
        default=16,
        help="Minimum token budget retained for each non-empty QNLI text field",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--data-seed", type=int, default=2027)
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Optional Hugging Face datasets revision/commit for reproducibility",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a non-smoke formal run on CPU (normally blocked as a cost/time safeguard)",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--arms", nargs="+", choices=("standard", "freeze_a"), default=["standard"]
    )
    parser.add_argument(
        "--freeze-a-clients",
        nargs="+",
        default=["sst2_0", "qnli_0", "mixed_0"],
        help="Representative clients for the causal freeze-A control",
    )
    parser.add_argument(
        "--standard-clients",
        nargs="+",
        default=[spec.client_id for spec in client_specs()],
        help=(
            "Standard-arm clients. Keep all six for the formal experiment; a representative "
            "subset can be used for an endpoint-selection pilot."
        ),
    )
    parser.add_argument(
        "--evaluation-checkpoint-fractions", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0]
    )
    parser.add_argument(
        "--compatibility-checkpoint-fractions", type=float, nargs="+", default=[0.1, 0.5, 1.0]
    )
    parser.add_argument(
        "--freeze-compatibility-checkpoint-fractions",
        type=float,
        nargs="+",
        default=[0.1, 1.0],
        help="Cheaper early/final mechanism check for the three-client freeze-A arm",
    )
    parser.add_argument(
        "--stability-checkpoint-fractions",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--trajectory-checkpoint-interval", type=int, default=8)
    parser.add_argument(
        "--spectral-eval-ranks", type=int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument("--formation-threshold", type=float, default=0.5)
    parser.add_argument(
        "--formation-sustain-steps",
        type=int,
        default=8,
        help=(
            "Minimum optimizer-step separation used to confirm a sustained event on the "
            "sampled trajectory; this does not claim every unobserved intermediate step passed."
        ),
    )
    parser.add_argument("--degenerate-relative-gap", type=float, default=0.05)
    parser.add_argument("--singular-relative-floor", type=float, default=1e-3)
    parser.add_argument("--skip-compatibility", action="store_true")
    parser.add_argument("--skip-freeze-compatibility", action="store_true")
    parser.add_argument("--compatibility-bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--run-functional-spectral-eval",
        action="store_true",
        help="Opt in to the extra final-checkpoint spectral truncation evaluations",
    )
    parser.add_argument("--output-dir", default="outputs/fed_svd_roberta_large_r8")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.model_name = "FacebookAI/roberta-base"
    args.model_revision = None
    args.rank = 4
    args.lora_alpha = 8.0
    args.epochs = 1
    args.train_per_client = 24
    args.reference_per_client = 8
    args.compatibility_per_client = 8
    args.batch_size = 8
    args.eval_batch_size = 8
    args.max_length = 128
    args.seeds = [42]
    args.spectral_eval_ranks = [1, 2, 4]
    args.output_dir = str(Path(args.output_dir))


def validate_args(args: argparse.Namespace) -> None:
    if args.rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("rank and lora-alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("lora-dropout must lie in [0, 1)")
    if args.learning_rate <= 0 or args.epochs <= 0:
        raise ValueError("learning-rate and epochs must be positive")
    if min(args.train_per_client, args.reference_per_client, args.compatibility_per_client) <= 0:
        raise ValueError("All client split sizes must be positive")
    # Mixed clients must split equally over tasks, and each per-task block is
    # label balanced.  Multiples of four guarantee both properties exactly.
    for name in ("train_per_client", "reference_per_client", "compatibility_per_client"):
        if getattr(args, name) % 4 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be divisible by 4")
    if args.batch_size <= 0 or args.eval_batch_size <= 0 or args.max_length <= 0:
        raise ValueError("Batch sizes and max-length must be positive")
    if args.qnli_min_field_tokens <= 0:
        raise ValueError("qnli-min-field-tokens must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if len(set(args.arms)) != len(args.arms) or not args.arms:
        raise ValueError("arms must be non-empty and unique")
    valid_clients = {spec.client_id for spec in client_specs()}
    if not set(args.standard_clients) <= valid_clients:
        raise ValueError(f"Unknown standard clients: {set(args.standard_clients) - valid_clients}")
    if len(set(args.standard_clients)) != len(args.standard_clients) or not args.standard_clients:
        raise ValueError("standard-clients must be non-empty and unique")
    if not set(args.freeze_a_clients) <= valid_clients:
        raise ValueError(f"Unknown freeze-A clients: {set(args.freeze_a_clients) - valid_clients}")
    if len(set(args.freeze_a_clients)) != len(args.freeze_a_clients):
        raise ValueError("freeze-A clients must be unique")
    if {"standard", "freeze_a"} <= set(args.arms) and not set(
        args.freeze_a_clients
    ) <= set(args.standard_clients):
        raise ValueError(
            "freeze-a-clients must be included in standard-clients for the matched control"
        )
    for values, name in (
        (args.evaluation_checkpoint_fractions, "evaluation checkpoint fractions"),
        (args.compatibility_checkpoint_fractions, "compatibility checkpoint fractions"),
        (
            args.freeze_compatibility_checkpoint_fractions,
            "freeze compatibility checkpoint fractions",
        ),
        (args.stability_checkpoint_fractions, "stability checkpoint fractions"),
    ):
        if not values or min(values) < 0 or max(values) > 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not 0 < args.formation_threshold <= 2:
        raise ValueError("formation-threshold must be in (0, 2]")
    if args.formation_sustain_steps <= 0:
        raise ValueError("formation-sustain-steps must be positive")
    if args.trajectory_checkpoint_interval <= 0:
        raise ValueError("trajectory-checkpoint-interval must be positive")
    if args.compatibility_bootstrap_samples < 0:
        raise ValueError("compatibility-bootstrap-samples cannot be negative")
    if not 0 <= args.degenerate_relative_gap < 1:
        raise ValueError("degenerate-relative-gap must be in [0, 1)")
    if not 0 <= args.singular_relative_floor < 1:
        raise ValueError("singular-relative-floor must be in [0, 1)")
    if min(args.spectral_eval_ranks) <= 0:
        raise ValueError("spectral-eval-ranks must be positive")


def actual_precision(args: argparse.Namespace, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        progress("CUDA device lacks bf16 support; falling back to fp32")
        return "fp32"
    return args.precision


def client_checkpoint_path(output: Path, seed: int, arm: str, client_id: str) -> Path:
    return output / "checkpoints" / f"seed_{seed}" / arm / f"{client_id}.pt"


def validate_tensor_state(
    state: Mapping[str, Any],
    expected_state: Mapping[str, torch.Tensor],
    context: str,
) -> None:
    if set(state) != set(expected_state):
        missing = sorted(set(expected_state) - set(state))
        extra = sorted(set(state) - set(expected_state))
        raise RuntimeError(f"{context}: state keys mismatch; missing={missing}, extra={extra}")
    for key, expected in expected_state.items():
        value = state[key]
        if not torch.is_tensor(value):
            raise RuntimeError(f"{context}: {key} is not a tensor")
        if tuple(value.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"{context}: {key} shape={tuple(value.shape)} expected={tuple(expected.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{context}: {key} contains non-finite values")


def validate_resumed_client_payload(
    payload: Mapping[str, Any],
    path: Path,
    training_fingerprint: str,
    seed: int,
    arm: str,
    client_id: str,
    expected_steps: Sequence[int],
    initial_state: Mapping[str, torch.Tensor],
) -> None:
    context = f"resume checkpoint {path}"
    expected_metadata = {
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "training_fingerprint": training_fingerprint,
        "seed": seed,
        "arm": arm,
        "client_id": client_id,
        "total_steps": max(expected_steps),
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"{context}: {key}={payload.get(key)!r}, expected={expected!r}"
            )
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise RuntimeError(f"{context}: missing trajectory mapping")
    actual_steps = sorted(int(step) for step in trajectory)
    if actual_steps != sorted(int(step) for step in expected_steps):
        raise RuntimeError(
            f"{context}: checkpoint steps={actual_steps}, expected={list(expected_steps)}"
        )
    compact_freeze = bool(payload.get("trajectory_uses_shared_initial_a"))
    if compact_freeze != (arm == "freeze_a"):
        raise RuntimeError(f"{context}: unexpected compact freeze-A representation")
    expected_disk_state = {
        key: value
        for key, value in initial_state.items()
        if not compact_freeze or key.endswith("::B")
    }
    for step, state in trajectory.items():
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{context}: step {step} is not a state mapping")
        validate_tensor_state(state, expected_disk_state, f"{context}, step={step}")


def train_or_resume_client(
    args: argparse.Namespace,
    output: Path,
    training_fingerprint: str,
    seed: int,
    arm: str,
    client_id: str,
    model: nn.Module,
    registry: AdapterRegistry,
    initial_state: Mapping[str, torch.Tensor],
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    precision: str,
) -> Dict[str, Any]:
    path = client_checkpoint_path(output, seed, arm, client_id)
    total_steps = int(math.ceil(len(records) / float(args.batch_size))) * args.epochs
    expected_steps = experiment_checkpoint_steps(args, total_steps)
    if args.resume and path.is_file():
        payload = load_torch(path)
        validate_resumed_client_payload(
            payload,
            path,
            training_fingerprint,
            seed,
            arm,
            client_id,
            expected_steps,
            initial_state,
        )
        if payload.get("trajectory_uses_shared_initial_a"):
            expanded: Dict[int, AdapterState] = {}
            initial_a = {
                key: value
                for key, value in initial_state.items()
                if key.endswith("::A")
            }
            for step, state in payload["trajectory"].items():
                expanded[int(step)] = {**initial_a, **state}
            payload["trajectory"] = expanded
        progress(f"resume seed={seed} arm={arm} client={client_id}")
        return payload
    progress(f"train seed={seed} arm={arm} client={client_id}")
    client_seed = (
        seed
        + 15485863 * (1 + sorted(spec.client_id for spec in client_specs()).index(client_id))
    )
    # Make each client trajectory invariant to run order and to which earlier
    # clients were restored from checkpoints.
    seed_everything(client_seed, args.deterministic)
    result = train_client_trajectory(
        model=model,
        registry=registry,
        client_id=client_id,
        initial_state=initial_state,
        rows=records,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        precision=precision,
        seed=client_seed,
        freeze_a=arm == "freeze_a",
        checkpoint_steps=experiment_checkpoint_steps(
            args, total_steps
        ),
    )
    payload = {
        "version": VERSION,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "training_fingerprint": training_fingerprint,
        "seed": seed,
        "arm": arm,
        "client_id": client_id,
        **result,
    }
    payload_for_disk = payload
    if arm == "freeze_a":
        payload_for_disk = {
            **payload,
            "trajectory_uses_shared_initial_a": True,
            "trajectory": {
                step: {
                    key: value for key, value in state.items() if key.endswith("::B")
                }
                for step, state in payload["trajectory"].items()
            },
        }
    validate_resumed_client_payload(
        payload_for_disk,
        path,
        training_fingerprint,
        seed,
        arm,
        client_id,
        expected_steps,
        initial_state,
    )
    atomic_torch(path, payload_for_disk)
    return payload


def summarize_seed(
    seed: int,
    standard_similarity: Sequence[Mapping[str, Any]],
    standard_drift: Sequence[Mapping[str, Any]],
    freeze_drift: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
    order_rows: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    agreement = b_ba_agreement_rows(standard_similarity)
    agreement_by_step: Dict[int, List[float]] = defaultdict(list)
    mae_by_step: Dict[int, List[float]] = defaultdict(list)
    for row in agreement:
        correlation = float(row["spearman_b_vs_ba"])
        mae = float(row["mae_b_vs_ba"])
        if math.isfinite(correlation):
            agreement_by_step[int(row["step"])].append(correlation)
        if math.isfinite(mae):
            mae_by_step[int(row["step"])].append(mae)
    drift_by_step: Dict[int, List[float]] = defaultdict(list)
    for row in standard_drift:
        drift_by_step[int(row["step"])].append(float(row["a_relative_drift"]))
    freeze_drift_by_step: Dict[int, List[float]] = defaultdict(list)
    for row in freeze_drift:
        freeze_drift_by_step[int(row["step"])].append(float(row["a_relative_drift"]))
    conditional_order_values = [
        float(row["conditional_spearman_rank_vs_formation_step"])
        for row in order_rows
        if math.isfinite(float(row["conditional_spearman_rank_vs_formation_step"]))
    ]
    c_index_values = [
        float(row["discrete_sustained_event_pairwise_c_index"])
        for row in order_rows
        if math.isfinite(float(row["discrete_sustained_event_pairwise_c_index"]))
    ]
    valid_stability = [row for row in stability if int(row["stability_valid"]) == 1]
    latest_stability_fraction = max(
        (float(row["comparison_fraction"]) for row in stability), default=float("nan")
    )
    latest_valid_stability = [
        row
        for row in valid_stability
        if abs(float(row["comparison_fraction"]) - latest_stability_fraction) <= 1e-12
    ]
    stability_by_fraction_and_rank: Dict[
        Tuple[float, int], List[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in stability:
        stability_by_fraction_and_rank[
            (float(row["comparison_fraction"]), int(row["top_m"]))
        ].append(row)
    stability_summary: Dict[str, Any] = {}
    for (fraction, top_m), group in sorted(stability_by_fraction_and_rank.items()):
        valid = [row for row in group if int(row["stability_valid"]) == 1]
        invalid_reason_counts: Dict[str, int] = defaultdict(int)
        for row in group:
            for reason in str(row.get("stability_invalid_reason", "")).split(";"):
                if reason:
                    invalid_reason_counts[reason] += 1
        overlaps = [float(row["subspace_overlap"]) for row in valid]
        energy_changes = [
            float(row["top_m_energy_relative_change"]) for row in valid
        ]
        stability_summary[f"fraction={fraction:.6g}|top_m={top_m}"] = {
            "total_comparisons": len(group),
            "valid_comparisons": len(valid),
            "valid_fraction": len(valid) / len(group) if group else float("nan"),
            "subspace_overlap_q10": finite_quantile(overlaps, 0.10),
            "subspace_overlap_q25": finite_quantile(overlaps, 0.25),
            "subspace_overlap_median": finite_median(overlaps),
            "energy_relative_change_q25": finite_quantile(energy_changes, 0.25),
            "energy_relative_change_median": finite_median(energy_changes),
            "energy_relative_change_q75": finite_quantile(energy_changes, 0.75),
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        }
    primary_relation_rows = [
        row
        for row in relations
        if str(row["metric_name"]) == "spearman_similarity_vs_loss_utility"
        and str(row["representation"]) in {"b", "ba"}
    ]
    primary_step = min(
        (int(row["step"]) for row in primary_relation_rows), default=None
    )
    primary_by_representation = {
        str(row["representation"]): float(row["metric_value"])
        for row in primary_relation_rows
        if primary_step is not None and int(row["step"]) == primary_step
    }
    return {
        "seed": seed,
        "b_ba_agreement_by_step": {
            str(step): {
                "median_layer_spearman": finite_median(values),
                "median_layer_mae": finite_median(mae_by_step[step]),
            }
            for step, values in sorted(agreement_by_step.items())
        },
        "standard_a_drift_by_step": {
            str(step): finite_median(values)
            for step, values in sorted(drift_by_step.items())
        },
        "freeze_a_drift_by_step": {
            str(step): finite_median(values)
            for step, values in sorted(freeze_drift_by_step.items())
        },
        "final_reference_stability": {
            f"top_{top_m}": finite_median(
                [
                    float(row["subspace_overlap"])
                    for row in latest_valid_stability
                    if int(row["top_m"]) == top_m
                ]
            )
            for top_m in sorted({int(row["top_m"]) for row in stability})
        },
        "final_reference_stability_by_checkpoint_and_rank": stability_summary,
        "final_reference_stability_valid_fraction": (
            len(valid_stability) / len(stability) if stability else float("nan")
        ),
        "final_reference_stability_primary_comparison_fraction": latest_stability_fraction,
        "spectral_order": {
            "module_count": len(order_rows),
            "module_count_with_finite_c_index": len(c_index_values),
            "module_count_meeting_three_formed_direction_threshold": sum(
                int(row["meets_three_formed_direction_interpretation_threshold"])
                for row in order_rows
            ),
            "median_discrete_sustained_event_pairwise_c_index": finite_median(
                c_index_values
            ),
            "module_count_with_conditional_spearman": len(conditional_order_values),
            "median_conditional_spearman_final_rank_vs_formation_step": finite_median(
                conditional_order_values
            ),
            "median_formed_fraction": finite_median(
                [float(row["formed_fraction"]) for row in order_rows]
            ),
        },
        "primary_early_compatibility": {
            "step": primary_step,
            "b_spearman_vs_raw_loss_utility": primary_by_representation.get(
                "b", float("nan")
            ),
            "ba_spearman_vs_raw_loss_utility": primary_by_representation.get(
                "ba", float("nan")
            ),
            "b_minus_ba_spearman": (
                primary_by_representation["b"] - primary_by_representation["ba"]
                if {"b", "ba"} <= set(primary_by_representation)
                else float("nan")
            ),
        },
        "matrix_relation_summaries": list(relations),
    }


def run_seed(
    args: argparse.Namespace,
    output: Path,
    training_fingerprint: str,
    analysis_fingerprint: str,
    specs: Sequence[ClientSpec],
    data: Mapping[str, ClientData],
    tokenizer: Any,
    seed: int,
    model: nn.Module,
    registry: AdapterRegistry,
    device: torch.device,
    precision: str,
) -> Dict[str, Any]:
    seed_everything(seed, args.deterministic)
    reset_lora_parameters(registry)
    module_names = sorted(registry)
    initial_state = extract_state(registry)
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    initial_path = seed_dir / "initial_state.pt"
    if args.resume and initial_path.is_file():
        saved_initial = load_torch(initial_path)
        if saved_initial.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
            raise RuntimeError(f"Initial-state schema mismatch: {initial_path}")
        if saved_initial.get("training_fingerprint") != training_fingerprint:
            raise RuntimeError(f"Initial-state fingerprint mismatch: {initial_path}")
        if saved_initial.get("seed") != seed:
            raise RuntimeError(f"Initial-state seed mismatch: {initial_path}")
        validate_tensor_state(
            saved_initial.get("state", {}), initial_state, f"initial state {initial_path}"
        )
        initial_state = saved_initial["state"]
        load_state(registry, initial_state)
    else:
        atomic_torch(
            initial_path,
            {
                "training_fingerprint": training_fingerprint,
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "seed": seed,
                "state": initial_state,
            },
        )

    planned_jobs: List[Tuple[str, str]] = []
    if "standard" in args.arms:
        planned_jobs.extend(
            ("standard", spec.client_id)
            for spec in specs
            if spec.client_id in args.standard_clients
        )
    if "freeze_a" in args.arms:
        planned_jobs.extend(("freeze_a", client_id) for client_id in args.freeze_a_clients)
    pending_at_start = sum(
        not (
            args.resume
            and client_checkpoint_path(output, seed, arm, client_id).is_file()
        )
        for arm, client_id in planned_jobs
    )
    completed_jobs = 0
    newly_trained_jobs = 0
    observed_training_seconds = 0.0

    def report_client_completion(
        arm: str,
        client_id: str,
        resumed: bool,
        wall_seconds: float,
    ) -> None:
        nonlocal completed_jobs, newly_trained_jobs, observed_training_seconds
        completed_jobs += 1
        if not resumed:
            newly_trained_jobs += 1
            observed_training_seconds += wall_seconds
        remaining_new_training = max(pending_at_start - newly_trained_jobs, 0)
        eta_seconds = (
            observed_training_seconds / newly_trained_jobs * remaining_new_training
            if newly_trained_jobs
            else float("nan")
        )
        progress(
            f"client complete seed={seed} arm={arm} client={client_id} "
            f"status={'resumed' if resumed else 'trained'} "
            f"client_time={format_duration(wall_seconds)} "
            f"jobs={completed_jobs}/{len(planned_jobs)} "
            f"estimated_remaining_training="
            f"{format_duration(eta_seconds) if math.isfinite(eta_seconds) else 'pending'}"
        )

    payloads: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    if "standard" in args.arms:
        for spec in specs:
            if spec.client_id not in args.standard_clients:
                continue
            resumed = args.resume and client_checkpoint_path(
                output, seed, "standard", spec.client_id
            ).is_file()
            client_started = time.time()
            payload = train_or_resume_client(
                args,
                output,
                training_fingerprint,
                seed,
                "standard",
                spec.client_id,
                model,
                registry,
                initial_state,
                data[spec.client_id].train,
                tokenizer,
                device,
                precision,
            )
            payloads["standard"][spec.client_id] = payload
            report_client_completion(
                "standard",
                spec.client_id,
                resumed,
                time.time() - client_started,
            )
    if "freeze_a" in args.arms:
        for client_id in args.freeze_a_clients:
            resumed = args.resume and client_checkpoint_path(
                output, seed, "freeze_a", client_id
            ).is_file()
            client_started = time.time()
            payload = train_or_resume_client(
                args,
                output,
                training_fingerprint,
                seed,
                "freeze_a",
                client_id,
                model,
                registry,
                initial_state,
                data[client_id].train,
                tokenizer,
                device,
                precision,
            )
            payloads["freeze_a"][client_id] = payload
            report_client_completion(
                "freeze_a",
                client_id,
                resumed,
                time.time() - client_started,
            )

    spec_by_id = {spec.client_id: spec for spec in specs}
    result_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    standard_trajectories: Dict[str, Mapping[int, Mapping[str, torch.Tensor]]] = {}
    freeze_trajectories: Dict[str, Mapping[int, Mapping[str, torch.Tensor]]] = {}
    if payloads.get("standard"):
        standard_trajectories = {
            client_id: payload["trajectory"]
            for client_id, payload in payloads["standard"].items()
        }
        result_rows["similarity"] = similarity_rows(
            seed, "standard", specs, standard_trajectories, initial_state, module_names
        )
        result_rows["drift"] = drift_rows(
            seed, "standard", standard_trajectories, initial_state, module_names
        )
        spectral, subspace, stability = spectral_rows(
            seed,
            standard_trajectories,
            module_names,
            args.degenerate_relative_gap,
            args.singular_relative_floor,
            args.stability_checkpoint_fractions,
        )
        result_rows["spectral"] = spectral
        result_rows["subspace"] = subspace
        result_rows["stability"] = stability
        formation: List[Dict[str, Any]] = []
        for threshold in sorted({0.25, args.formation_threshold, 0.75}):
            for formation_type in ("positive_alignment", "absolute_amplitude"):
                formation.extend(
                    formation_time_rows(
                        spectral,
                        threshold,
                        args.formation_sustain_steps,
                        formation_type,
                    )
                )
        result_rows["formation"] = formation
        result_rows["order"] = module_order_rows(formation)
        result_rows["validation"] = evaluate_trajectory_checkpoints(
            seed,
            "standard",
            model,
            registry,
            tokenizer,
            device,
            precision,
            args.eval_batch_size,
            data,
            standard_trajectories,
            args.evaluation_checkpoint_fractions,
        )
        result_rows["spectral_functional_alignment"] = (
            spectral_functional_alignment_rows(
                seed,
                subspace,
                result_rows["validation"],
            )
        )
        if args.run_functional_spectral_eval:
            result_rows["spectral_function"] = evaluate_spectral_truncations(
                seed,
                model,
                registry,
                tokenizer,
                device,
                precision,
                args.eval_batch_size,
                data,
                standard_trajectories,
                module_names,
                args.rank,
                args.spectral_eval_ranks,
            )
        if not args.skip_compatibility:
            result_rows["compatibility"] = evaluate_compatibility(
                seed,
                model,
                registry,
                tokenizer,
                device,
                precision,
                args.eval_batch_size,
                specs,
                data,
                standard_trajectories,
                module_names,
                args.rank,
                args.compatibility_checkpoint_fractions,
                args.compatibility_bootstrap_samples,
                "standard",
            )
            result_rows["relations"] = matrix_relation_rows(
                seed, result_rows["similarity"], result_rows["compatibility"]
            )
            result_rows["compatibility_pair_summary"] = (
                directional_compatibility_summary_rows(
                    seed, "standard", result_rows["compatibility"]
                )
            )

    if payloads.get("freeze_a"):
        freeze_trajectories = {
            client_id: payload["trajectory"]
            for client_id, payload in payloads["freeze_a"].items()
        }
        freeze_specs = [spec_by_id[client_id] for client_id in sorted(freeze_trajectories)]
        result_rows["freeze_similarity"] = similarity_rows(
            seed,
            "freeze_a",
            freeze_specs,
            freeze_trajectories,
            initial_state,
            module_names,
        )
        result_rows["freeze_drift"] = drift_rows(
            seed, "freeze_a", freeze_trajectories, initial_state, module_names
        )
        result_rows["freeze_validation"] = evaluate_trajectory_checkpoints(
            seed,
            "freeze_a",
            model,
            registry,
            tokenizer,
            device,
            precision,
            args.eval_batch_size,
            data,
            freeze_trajectories,
            args.evaluation_checkpoint_fractions,
        )
        if result_rows.get("similarity"):
            result_rows["causal_control"] = causal_control_rows(
                result_rows["similarity"],
                result_rows["drift"],
                result_rows["freeze_similarity"],
                result_rows["freeze_drift"],
            )
        if not args.skip_compatibility and not args.skip_freeze_compatibility:
            result_rows["freeze_compatibility"] = evaluate_compatibility(
                seed,
                model,
                registry,
                tokenizer,
                device,
                precision,
                args.eval_batch_size,
                freeze_specs,
                data,
                freeze_trajectories,
                module_names,
                args.rank,
                args.freeze_compatibility_checkpoint_fractions,
                args.compatibility_bootstrap_samples,
                "freeze_a",
            )
            result_rows["freeze_relations"] = matrix_relation_rows(
                seed,
                result_rows["freeze_similarity"],
                result_rows["freeze_compatibility"],
            )
            result_rows["freeze_compatibility_pair_summary"] = (
                directional_compatibility_summary_rows(
                    seed, "freeze_a", result_rows["freeze_compatibility"]
                )
            )
            for row in result_rows["freeze_relations"]:
                row["arm"] = "freeze_a"
            control_clients = set(freeze_trajectories)
            if result_rows.get("compatibility"):
                matched_standard_similarity = [
                    row
                    for row in result_rows["similarity"]
                    if str(row["client_i"]) in control_clients
                    and str(row["client_j"]) in control_clients
                ]
                matched_standard_compatibility = [
                    row
                    for row in result_rows["compatibility"]
                    if str(row["receiver"]) in control_clients
                    and str(row["donor"]) in control_clients
                ]
                matched_standard_relations = matrix_relation_rows(
                    seed,
                    matched_standard_similarity,
                    matched_standard_compatibility,
                )
                for row in matched_standard_relations:
                    row["arm"] = "standard_matched"
                result_rows["causal_compatibility_relations"] = (
                    matched_standard_relations + result_rows["freeze_relations"]
                )

    metrics_dir = seed_dir / "metrics"
    for name, rows in result_rows.items():
        write_csv(metrics_dir / f"{name}.csv", rows)
    summary = summarize_seed(
        seed,
        result_rows.get("similarity", []),
        result_rows.get("drift", []),
        result_rows.get("freeze_drift", []),
        result_rows.get("stability", []),
        [
            row
            for row in result_rows.get("order", [])
            if abs(float(row["threshold"]) - args.formation_threshold) <= 1e-12
            and str(row["formation_type"]) == "positive_alignment"
        ],
        result_rows.get("relations", []),
    )
    summary["analysis_fingerprint"] = analysis_fingerprint
    summary["training_fingerprint"] = training_fingerprint
    summary["precision"] = precision
    summary["causal_compatibility_relation_summaries"] = result_rows.get(
        "causal_compatibility_relations", []
    )
    summary["spectral_order_sensitivity"] = {}
    for formation_type in ("positive_alignment", "absolute_amplitude"):
        for threshold in sorted(
            {float(row["threshold"]) for row in result_rows.get("order", [])}
        ):
            subset = [
                row
                for row in result_rows["order"]
                if abs(float(row["threshold"]) - threshold) <= 1e-12
                and str(row["formation_type"]) == formation_type
            ]
            c_indices = [
                float(row["discrete_sustained_event_pairwise_c_index"])
                for row in subset
                if math.isfinite(
                    float(row["discrete_sustained_event_pairwise_c_index"])
                )
            ]
            correlations = [
                float(row["conditional_spearman_rank_vs_formation_step"])
                for row in subset
                if math.isfinite(
                    float(row["conditional_spearman_rank_vs_formation_step"])
                )
            ]
            summary["spectral_order_sensitivity"][
                f"{formation_type}@{threshold}"
            ] = {
                "module_count": len(subset),
                "module_count_with_finite_c_index": len(c_indices),
                "median_discrete_sustained_event_pairwise_c_index": finite_median(
                    c_indices
                ),
                "median_formed_fraction": finite_median(
                    [float(row["formed_fraction"]) for row in subset]
                ),
                "module_count_with_conditional_spearman": len(correlations),
                "median_conditional_spearman": finite_median(correlations),
            }
    summary["spectral_functional_alignment"] = {}
    alignment_rows = result_rows.get("spectral_functional_alignment", [])
    for top_m in sorted({int(row["top_m"]) for row in alignment_rows}):
        client_correlations: List[float] = []
        for client_id in sorted({str(row["client"]) for row in alignment_rows}):
            rows_for_client = sorted(
                (
                    row
                    for row in alignment_rows
                    if int(row["top_m"]) == top_m
                    and str(row["client"]) == client_id
                ),
                key=lambda row: int(row["step"]),
            )
            correlation = spearman(
                [
                    float(row["mean_final_top_m_formation_progress"])
                    for row in rows_for_client
                ],
                [
                    float(row["reference_loss_improvement_from_step0"])
                    for row in rows_for_client
                ],
            )
            if math.isfinite(correlation):
                client_correlations.append(correlation)
        summary["spectral_functional_alignment"][f"top_{top_m}"] = {
            "clients_with_finite_within_trajectory_correlation": len(
                client_correlations
            ),
            "median_within_client_spearman_formation_vs_loss_improvement": (
                finite_median(client_correlations)
            ),
        }
    summary["training"] = {
        arm: {
            client_id: {
                "total_steps": payload["total_steps"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "first_step_a_displacement": payload["first_step_a_displacement"],
                "fp16_scaler_backoff_steps": payload.get(
                    "fp16_scaler_backoff_steps", 0
                ),
            }
            for client_id, payload in by_client.items()
        }
        for arm, by_client in payloads.items()
    }
    quality_warnings: List[str] = []
    fp16_backoffs = sum(
        int(payload.get("fp16_scaler_backoff_steps", 0))
        for by_client in payloads.values()
        for payload in by_client.values()
    )
    if fp16_backoffs:
        quality_warnings.append(
            f"FP16 GradScaler reduced its scale {fp16_backoffs} times; parameters remained "
            "finite, but inspect training traces before interpretation."
        )
    stability_key = f"top_{min(4, args.rank)}"
    stability_value = float(
        summary["final_reference_stability"].get(stability_key, float("nan"))
    )
    if math.isfinite(stability_value) and stability_value < 0.90:
        quality_warnings.append(
            "Final SVD reference is not stable enough (median top-subspace overlap < 0.90); "
            "extend training before interpreting spectral order."
        )
    if not math.isfinite(stability_value):
        quality_warnings.append(
            "No identifiable finite top-subspace stability value was available; "
            "do not interpret the final SVD direction order."
        )
    stability_valid_fraction = float(
        summary.get("final_reference_stability_valid_fraction", float("nan"))
    )
    if math.isfinite(stability_valid_fraction) and stability_valid_fraction < 0.5:
        quality_warnings.append(
            "Fewer than half of requested stability comparisons passed the singular-energy "
            "and boundary-gap identifiability checks."
        )
    latest_stability_fraction = float(
        summary.get(
            "final_reference_stability_primary_comparison_fraction", float("nan")
        )
    )
    late_energy_changes = [
        float(row["top_m_energy_relative_change"])
        for row in result_rows.get("stability", [])
        if int(row["stability_valid"]) == 1
        and int(row["top_m"]) == min(4, args.rank)
        and math.isfinite(latest_stability_fraction)
        and abs(float(row["comparison_fraction"]) - latest_stability_fraction) <= 1e-12
    ]
    median_late_energy_change = finite_median(late_energy_changes)
    if math.isfinite(median_late_energy_change) and median_late_energy_change > 0.10:
        quality_warnings.append(
            "Top-mode energy still changes by over 10% across late checkpoints; run a longer "
            "endpoint-selection pilot before treating the final SVD as fixed."
        )
    expected_modules = len(standard_trajectories) * len(module_names)
    c_index_modules = int(
        summary["spectral_order"]["module_count_with_finite_c_index"]
    )
    three_formed_modules = int(
        summary["spectral_order"][
            "module_count_meeting_three_formed_direction_threshold"
        ]
    )
    if expected_modules and three_formed_modules < 0.8 * expected_modules:
        quality_warnings.append(
            "Fewer than 80% of client/modules had at least three usable nondegenerate modes; "
            "interpret conditional rank-versus-time correlations cautiously."
        )
    if expected_modules and c_index_modules < 0.8 * expected_modules:
        quality_warnings.append(
            "Fewer than 80% of client/modules had a finite discrete sustained-event "
            "pairwise C-index; interpret the ordering statistic cautiously."
        )
    retained_values = [
        float(row["global_energy_weighted_operator_retained"])
        for row in result_rows.get("compatibility", [])
    ]
    median_merge_retention = finite_mean(retained_values)
    if math.isfinite(median_merge_retention) and median_merge_retention < 0.90:
        quality_warnings.append(
            "Rank-r compression retained under 90% of pair-average energy on average; "
            "sharing utility includes a material compression effect."
        )
    freeze_proxy_errors = [
        float(row["mean_abs_cos_b_weighted_a0_minus_cos_ba"])
        for row in result_rows.get("causal_control", [])
        if str(row["arm"]) == "freeze_a"
        and math.isfinite(float(row["mean_abs_cos_b_weighted_a0_minus_cos_ba"]))
    ]
    max_freeze_proxy_error = max(freeze_proxy_errors, default=float("nan"))
    if math.isfinite(max_freeze_proxy_error) and max_freeze_proxy_error > 1e-5:
        quality_warnings.append(
            "Freeze-A weighted-B similarity did not numerically match BA similarity."
        )
    failed_qap_rows = [
        row
        for row in result_rows.get("relations", [])
        if str(row.get("representation")) != "b_vs_ba"
        and str(row.get("qap_status")) != "ok"
    ]
    if failed_qap_rows:
        quality_warnings.append(
            f"{len(failed_qap_rows)} QAP summaries were skipped because the complete "
            "finite 15-pair/common-layer matrix requirement was not met."
        )
    reduced_layer_qap_rows = [
        row
        for row in result_rows.get("relations", [])
        if str(row.get("representation")) != "b_vs_ba"
        and math.isfinite(float(row.get("module_coverage_fraction", float("nan"))))
        and float(row["module_coverage_fraction"]) < 1.0
    ]
    if reduced_layer_qap_rows:
        quality_warnings.append(
            f"{len(reduced_layer_qap_rows)} QAP summaries used a strict common finite-layer "
            "subset; inspect module_coverage_fraction before interpretation."
        )
    summary["quality_checks"] = {
        "reference_stability_value": stability_value,
        "fp16_scaler_backoff_steps": fp16_backoffs,
        "reference_stability_valid_fraction": stability_valid_fraction,
        "median_late_top_mode_energy_relative_change": median_late_energy_change,
        "spectral_modules_expected": expected_modules,
        "spectral_modules_with_finite_c_index": c_index_modules,
        "spectral_modules_with_three_formed_directions": three_formed_modules,
        "mean_global_energy_weighted_pair_merge_retention": median_merge_retention,
        "max_freeze_a_weighted_b_proxy_error": max_freeze_proxy_error,
        "qap_rows_skipped": len(failed_qap_rows),
        "qap_rows_with_reduced_common_layer_coverage": len(
            reduced_layer_qap_rows
        ),
        "warnings": quality_warnings,
    }
    atomic_json(seed_dir / "summary.json", summary)

    if not args.no_plots and result_rows.get("similarity"):
        figures = seed_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plot_layerwise_similarity(result_rows["similarity"], figures, seed)
        plot_similarity_heatmaps(result_rows["similarity"], figures, seed)
        plot_b_ba_agreement(result_rows["similarity"], figures, seed)
        plot_spectral_progress(result_rows["spectral"], figures, seed)
        plot_spectral_functional_alignment(
            result_rows["spectral_functional_alignment"], figures, seed
        )
        if result_rows.get("causal_control"):
            plot_causal_control(result_rows["causal_control"], figures, seed)
        if result_rows.get("compatibility"):
            plot_compatibility_scatter(
                result_rows["similarity"], result_rows["compatibility"], figures, seed
            )

    del payloads, standard_trajectories, freeze_trajectories
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    atomic_json(
        seed_dir / "complete.json",
        {
            "analysis_fingerprint": analysis_fingerprint,
            "training_fingerprint": training_fingerprint,
            "plots_generated": not args.no_plots,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return summary


def aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    order_values = [
        float(
            summary["spectral_order"][
                "median_discrete_sustained_event_pairwise_c_index"
            ]
        )
        for summary in summaries
        if math.isfinite(
            float(
                summary["spectral_order"][
                    "median_discrete_sustained_event_pairwise_c_index"
                ]
            )
        )
    ]
    final_top4 = [
        float(summary["final_reference_stability"].get("top_4", float("nan")))
        for summary in summaries
    ]
    primary_b_minus_ba = [
        float(summary.get("primary_early_compatibility", {}).get(
            "b_minus_ba_spearman", float("nan")
        ))
        for summary in summaries
    ]
    relation_groups: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for summary in summaries:
        for row in summary.get("matrix_relation_summaries", []):
            relation_groups[
                (
                    str(row["metric_name"]),
                    str(row["representation"]),
                    int(row["step"]),
                )
            ].append(row)
    relation_aggregate: List[Dict[str, Any]] = []
    for (metric_name, representation, step), rows in sorted(relation_groups.items()):
        values = [float(row["metric_value"]) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        relation_aggregate.append(
            {
                "metric_name": metric_name,
                "representation": representation,
                "step": step,
                "seeds": len(rows),
                "median_metric_value": finite_median(values),
                "q25_metric_value": (
                    float(np.quantile(finite_values, 0.25))
                    if finite_values
                    else float("nan")
                ),
                "q75_metric_value": (
                    float(np.quantile(finite_values, 0.75))
                    if finite_values
                    else float("nan")
                ),
                "finite_seed_count": len(finite_values),
            }
        )
    return {
        "version": VERSION,
        "seeds_completed": [int(summary["seed"]) for summary in summaries],
        "seed_count": len(summaries),
        "spectral_order_c_index_median_across_seeds": finite_median(order_values),
        "final_top4_reference_stability_mean_across_seeds": finite_mean(final_top4),
        "primary_early_b_minus_ba_spearman_median_across_seeds": finite_median(
            primary_b_minus_ba
        ),
        "matrix_relations_across_seeds": relation_aggregate,
        "quality_warnings_by_seed": {
            str(summary["seed"]): summary.get("quality_checks", {}).get("warnings", [])
            for summary in summaries
            if summary.get("quality_checks", {}).get("warnings")
        },
        "interpretation_guardrails": [
            "A discrete sustained-event pairwise C-index above 0.5 supports earlier sampled sustained formation of larger final modes; it is not a continuous-time survival estimate and does not prove functional importance.",
            "B-vs-BA agreement, task-mixture structure, and actual sharing utility are separate endpoints.",
            "Client-pair QAP is reported only within each seed; the 15 pairs are not independent replicates and p-values are not combined across seeds.",
            "QAP p-values are unadjusted across representations/checkpoints and should be treated as exploratory.",
            "Freeze-A is a mechanism intervention: it supports an A-drift explanation but does not prove A drift is the sole cause of matching changes.",
            "Trajectory/function correlations are descriptive and time-confounded.",
            "Pilot seeds used to choose the endpoint are exploratory unless rerun under a protocol fixed in advance.",
        ],
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        apply_smoke_overrides(args)
    validate_args(args)
    tests = run_math_self_tests()
    if args.self_test_only:
        print(json.dumps({"version": VERSION, "math_self_tests": tests}, indent=2))
        return
    protocol = scientific_protocol(args)
    training_specification = training_protocol(args)
    training_fingerprint = stable_hash(training_specification)
    source_sha256 = current_source_sha256()
    analysis_fingerprint = stable_hash(
        {
            "scientific_protocol": protocol,
            "training_fingerprint": training_fingerprint,
            "source_sha256": source_sha256,
        }
    )
    protocol["training_protocol"] = training_specification
    protocol["training_fingerprint"] = training_fingerprint
    protocol["analysis_fingerprint"] = analysis_fingerprint
    protocol["source_sha256"] = source_sha256
    protocol["resource_estimate"] = resource_estimate(args)
    protocol["math_self_tests"] = tests
    if args.validate_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return

    device = resolve_device(args.device)
    if device.type != "cuda" and not args.smoke_test and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA device selected. Refusing an accidental RoBERTa-large CPU run; "
            "use --allow-cpu only if this is intentional."
        )
    precision = actual_precision(args, device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if not args.resume and not protocol_path.is_file() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a fresh directory."
        )
    if protocol_path.is_file() and not args.resume:
        raise FileExistsError(
            f"Output already contains a protocol: {protocol_path}. "
            "Use --resume or choose a new --output-dir; refusing an accidental paid rerun."
        )
    if args.resume and not protocol_path.is_file() and any(output.iterdir()):
        raise RuntimeError(
            "Cannot safely resume a non-empty output directory without protocol.json."
        )
    if args.resume and protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("training_fingerprint") != training_fingerprint:
            raise RuntimeError(
                "Output directory contains training checkpoints from a different training "
                "protocol. Use a new output directory instead of mixing runs."
            )
        if existing.get("analysis_fingerprint") != analysis_fingerprint:
            progress(
                "analysis protocol/source changed; validated training checkpoints may be "
                "reused, but metrics and summaries will be regenerated"
            )
    specs = client_specs()
    data, manifest = allocate_client_data(
        specs,
        args.train_per_client,
        args.reference_per_client,
        args.compatibility_per_client,
        args.data_seed,
        args.dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, revision=args.model_revision, use_fast=True
    )
    if tokenizer.mask_token_id is None:
        raise RuntimeError("Selected tokenizer has no mask token")
    manifest["tokenizer_self_test"] = run_tokenizer_self_tests(
        tokenizer, args.max_length, args.qnli_min_field_tokens
    )
    manifest["tokenization"] = encode_client_data(
        data, tokenizer, args.max_length, args.qnli_min_field_tokens
    )
    manifest_path = output / "data_manifest.json"
    if args.resume and protocol_path.is_file() and not manifest_path.is_file():
        raise RuntimeError(
            "Cannot safely resume existing results without data_manifest.json."
        )
    if args.resume and manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing_manifest.get("assignment_sha256") != manifest["assignment_sha256"]
            or existing_manifest.get("assigned_content_sha256")
            != manifest["assigned_content_sha256"]
        ):
            raise RuntimeError(
                "Dataset assignment changed since the original run; use a new output directory."
            )
    atomic_json(protocol_path, protocol)
    atomic_json(manifest_path, manifest)
    progress(json.dumps(protocol["resource_estimate"], ensure_ascii=False))

    summaries: List[Dict[str, Any]] = []
    shared_model: Optional[nn.Module] = None
    shared_registry: Optional[AdapterRegistry] = None
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        completion_path = seed_dir / "complete.json"
        summary_path = seed_dir / "summary.json"
        if args.resume and completion_path.is_file() and summary_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            complete_enough = args.no_plots or bool(completion.get("plots_generated"))
            if (
                completion.get("analysis_fingerprint") == analysis_fingerprint
                and completed_summary.get("analysis_fingerprint")
                == analysis_fingerprint
                and complete_enough
            ):
                progress(f"reuse completed seed={seed}")
                summaries.append(completed_summary)
                continue
        if shared_model is None or shared_registry is None:
            progress(f"load shared base model once: {args.model_name}")
            shared_model = AutoModelForMaskedLM.from_pretrained(
                args.model_name, revision=args.model_revision
            )
            runtime_manifest_path = output / "runtime_manifest.json"
            if (
                args.resume
                and (output / "checkpoints").exists()
                and not runtime_manifest_path.is_file()
            ):
                raise RuntimeError(
                    "Cannot safely resume existing checkpoints without runtime_manifest.json."
                )
            runtime_manifest = {
                "model_name": args.model_name,
                "requested_model_revision": args.model_revision,
                "resolved_model_commit": getattr(
                    shared_model.config, "_commit_hash", None
                ),
                "resolved_tokenizer_commit": tokenizer.init_kwargs.get(
                    "_commit_hash"
                ),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "datasets_version": datasets.__version__,
                "numpy_version": np.__version__,
                "actual_precision": precision,
                "device_type": device.type,
                "cuda_device_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
            }
            if args.resume and runtime_manifest_path.is_file():
                existing_runtime = json.loads(
                    runtime_manifest_path.read_text(encoding="utf-8")
                )
                old_commit = existing_runtime.get("resolved_model_commit")
                new_commit = runtime_manifest.get("resolved_model_commit")
                if old_commit and new_commit and old_commit != new_commit:
                    raise RuntimeError(
                        "Resolved base-model commit changed since the original run."
                    )
                old_tokenizer_commit = existing_runtime.get(
                    "resolved_tokenizer_commit"
                )
                new_tokenizer_commit = runtime_manifest.get(
                    "resolved_tokenizer_commit"
                )
                if (
                    old_tokenizer_commit
                    and new_tokenizer_commit
                    and old_tokenizer_commit != new_tokenizer_commit
                ):
                    raise RuntimeError(
                        "Resolved tokenizer commit changed since the original run."
                    )
                if existing_runtime.get("actual_precision") != precision:
                    raise RuntimeError(
                        "Actual numeric precision changed since the original run."
                    )
            atomic_json(runtime_manifest_path, runtime_manifest)
            shared_registry = inject_lora(
                shared_model, args.rank, args.lora_alpha, args.lora_dropout
            )
            shared_model.to(device)
        summaries.append(
            run_seed(
                args,
                output,
                training_fingerprint,
                analysis_fingerprint,
                specs,
                data,
                tokenizer,
                seed,
                shared_model,
                shared_registry,
                device,
                precision,
            )
        )
    del shared_registry, shared_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    aggregate = aggregate_summaries(summaries)
    aggregate["analysis_fingerprint"] = analysis_fingerprint
    aggregate["training_fingerprint"] = training_fingerprint
    atomic_json(output / "summary.json", aggregate)
    progress(f"complete: {output.resolve()}")


if __name__ == "__main__":
    main()
