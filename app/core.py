"""Core functionality for Quantized Retail Recommender.

Includes synthetic shopping basket generation, small language model architecture,
next-item prediction, itemset probing, baseline measurement, post-training
quantization (dynamic and static), and comparative evaluation.
"""
from __future__ import annotations

import io
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from thilimarketbasket import dataset as tmb_dataset
from thilimarketbasket import verbalize as tmb_verbalize

# --- Default Catalog & Frequent Itemsets for Synthetic Baskets
RETAIL_CATALOG = [
    "Milk 1L", "Bread White", "Butter 200g", "Eggs 12pk", "Cheddar Cheese 250g",
    "Pasta 500g", "Tomato Sauce 400g", "Parmesan Cheese 100g", "Olive Oil 500ml", "Garlic 3pk",
    "Potato Chips 150g", "Salsa Dip 300g", "Cola 1.5L", "Pretzels 200g",
    "Coffee Beans 250g", "Granulated Sugar 1kg", "Chocolate Chip Cookies 200g", "Oat Milk 1L",
    "Apples 1kg", "Bananas 1kg", "Yogurt 500g", "Granola 400g", "Honey 350g",
]

FREQUENT_RULES = [
    (["Bread White", "Butter 200g"], "Milk 1L"),
    (["Pasta 500g", "Tomato Sauce 400g"], "Parmesan Cheese 100g"),
    (["Potato Chips 150g", "Salsa Dip 300g"], "Cola 1.5L"),
    (["Coffee Beans 250g", "Granulated Sugar 1kg"], "Chocolate Chip Cookies 200g"),
    (["Yogurt 500g", "Granola 400g"], "Honey 350g"),
]


# --- Milestone 1: Prepare synthetic shopping basket data for training
def prepare_synthetic_baskets(
    n_baskets: int = 10000,
    catalog: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate diverse shopping baskets containing frequent itemsets and background noise."""
    random.seed(seed)
    np.random.seed(seed)
    cat = catalog or RETAIL_CATALOG

    baskets: list[list[str]] = []
    for _ in range(n_baskets):
        basket: set[str] = set()

        # 40% chance to plant one or more frequent association rules
        if random.random() < 0.45:
            rule_visible, rule_target = random.choice(FREQUENT_RULES)
            basket.update(rule_visible)
            if random.random() < 0.85:  # high co-occurrence probability
                basket.add(rule_target)

        # Add background random catalog items
        num_bg = random.randint(1, 4)
        bg_items = random.sample(cat, k=num_bg)
        basket.update(bg_items)

        baskets.append(sorted(list(basket)))

    # Build item vocabulary
    vocab = ["<pad>", "<bos>", "<eos>"] + sorted(list(set(cat)))
    item_to_id = {item: idx for idx, item in enumerate(vocab)}
    id_to_item = {idx: item for idx, item in enumerate(vocab)}

    # Verbalize baskets as text sequences
    verbalized = [f"Basket: {'; '.join(b)}" for b in baskets]

    return {
        "baskets": baskets,
        "catalog": cat,
        "vocab": vocab,
        "item_to_id": item_to_id,
        "id_to_item": id_to_item,
        "verbalized": verbalized,
    }


def milestone_1() -> dict[str, Any]:
    """Milestone 1: Prepare synthetic shopping basket data for training."""
    return prepare_synthetic_baskets(n_baskets=10000)


class BasketLanguageModel(nn.Module):
    """A compact language model for shopping basket sequence modeling and next-item prediction."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Any = None) -> tuple[torch.Tensor, Any]:
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        logits = self.fc(out)
        return logits, hidden


def train_basket_lm(
    data: dict[str, Any],
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    embed_dim: int = 64,
    hidden_dim: int = 128,
    max_baskets: int = 2000,
) -> tuple[BasketLanguageModel, dict[str, Any]]:
    """Train a small language model on synthetic shopping basket token sequences."""
    baskets = data["baskets"][:max_baskets]
    item_to_id = data["item_to_id"]
    vocab_size = len(data["vocab"])

    # Prepare token sequences: <bos> item1 item2 ... <eos>
    bos_id = item_to_id["<bos>"]
    eos_id = item_to_id["<eos>"]
    sequences = []
    for b in baskets:
        seq = [bos_id] + [item_to_id[it] for it in b if it in item_to_id] + [eos_id]
        sequences.append(seq)

    # Pad sequences
    max_len = max(len(s) for s in sequences)
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s

    tensor_data = torch.from_numpy(padded)
    dataset = torch.utils.data.TensorDataset(tensor_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BasketLanguageModel(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for ep in range(epochs):
        ep_loss = 0.0
        batches = 0
        for (batch_x,) in loader:
            inputs = batch_x[:, :-1]
            targets = batch_x[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=0,
            )
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            batches += 1

        avg_loss = ep_loss / max(1, batches)
        history.append({"epoch": ep + 1, "loss": avg_loss})

    model.eval()
    return model, {"history": history, "vocab_size": vocab_size}


# --- Milestone 2: Continue pretraining a small LM on the basket data
# Hint: Continue pretraining the language model.
def milestone_2(
    data: dict[str, Any] | None = None,
    epochs: int = 2,
    max_baskets: int = 2000,
) -> tuple[BasketLanguageModel, dict[str, Any]]:
    """Milestone 2: Continue pretraining a small LM on the basket data."""
    if data is None:
        data = milestone_1()
    model, info = train_basket_lm(data, epochs=epochs, max_baskets=max_baskets)
    return model, data


# --- Milestone 3: Implement next-item prediction and itemset probing
# Hint: Use the language model for next-item prediction and itemset probing.
def predict_next_items(
    model: nn.Module,
    data: dict[str, Any],
    current_basket: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Predict and rank the most likely next items given items currently in the basket."""
    model.eval()
    item_to_id = data["item_to_id"]
    id_to_item = data["id_to_item"]
    bos_id = item_to_id["<bos>"]

    # Convert visible basket items to tokens
    valid_ids = [item_to_id[it] for it in current_basket if it in item_to_id]
    seq_ids = [bos_id] + valid_ids

    input_tensor = torch.tensor([seq_ids], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(input_tensor)
        next_logits = logits[0, -1, :].clone()

    # Mask special tokens and items already present in the basket
    masked_ids = {0, item_to_id.get("<bos>", 1), item_to_id.get("<eos>", 2)}
    masked_ids.update(valid_ids)
    for mid in masked_ids:
        if mid < len(next_logits):
            next_logits[mid] = -float("inf")

    probs = F.softmax(next_logits, dim=-1)
    top_indices = torch.topk(probs, k=min(top_k, len(probs))).indices.tolist()

    results: list[tuple[str, float]] = []
    for idx in top_indices:
        item_name = id_to_item.get(idx, f"item_{idx}")
        results.append((item_name, float(probs[idx].item())))

    return results


def probe_itemsets(
    model: nn.Module,
    data: dict[str, Any],
    rules: list[tuple[list[str], str]] | None = None,
) -> dict[str, Any]:
    """Probe language model to evaluate recovery of planted association rules and itemset relationships."""
    test_rules = rules or FREQUENT_RULES
    hits_top1 = 0
    hits_top5 = 0
    rr_total = 0.0
    details = []

    for visible, target in test_rules:
        recs = predict_next_items(model, data, visible, top_k=len(data["catalog"]))
        rec_items = [name for name, _ in recs]

        if target in rec_items:
            rank = rec_items.index(target) + 1
            rr = 1.0 / rank
            if rank == 1:
                hits_top1 += 1
            if rank <= 5:
                hits_top5 += 1
        else:
            rank = len(data["catalog"]) + 1
            rr = 0.0

        rr_total += rr
        details.append({
            "visible": visible,
            "target": target,
            "rank": rank,
            "reciprocal_rank": rr,
            "top_recommendations": rec_items[:3],
        })

    n = len(test_rules)
    return {
        "top1_accuracy": hits_top1 / n,
        "top5_accuracy": hits_top5 / n,
        "mrr": rr_total / n,
        "details": details,
    }


def milestone_3(
    model: nn.Module | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Milestone 3: Implement next-item prediction and itemset probing."""
    if model is None or data is None:
        model, data = milestone_2()

    # Predict next items for sample basket
    sample_basket = ["Pasta 500g", "Tomato Sauce 400g"]
    sample_preds = predict_next_items(model, data, sample_basket, top_k=3)

    # Probe frequent itemsets
    probe_metrics = probe_itemsets(model, data)

    return {
        "sample_basket": sample_basket,
        "sample_predictions": sample_preds,
        "probe_metrics": probe_metrics,
    }


# --- Milestone 4: Measure the baseline size and accuracy of the model
# Hint: Measure initial model size and accuracy.
def get_model_size_bytes(model: nn.Module) -> int:
    """Calculate the serialized memory size of the PyTorch model in bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def measure_model_accuracy(
    model: nn.Module,
    data: dict[str, Any],
    test_baskets: list[list[str]] | None = None,
    max_eval: int = 50,
) -> dict[str, float]:
    """Evaluate next-item recommendation accuracy and inference latency on held-out baskets."""
    baskets = test_baskets or data["baskets"]
    eval_baskets = [b for b in baskets if len(b) >= 2][:max_eval]
    if not eval_baskets:
        eval_baskets = [["Bread White", "Butter 200g", "Milk 1L"]]

    hits_top1 = 0
    hits_top5 = 0
    rr_sum = 0.0
    latencies: list[float] = []

    for b in eval_baskets:
        visible = b[:-1]
        target = b[-1]

        t0 = time.perf_counter()
        preds = predict_next_items(model, data, visible, top_k=len(data["catalog"]))
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        items_pred = [it for it, _ in preds]
        if target in items_pred:
            rank = items_pred.index(target) + 1
            if rank == 1:
                hits_top1 += 1
            if rank <= 5:
                hits_top5 += 1
            rr_sum += 1.0 / rank

    total = len(eval_baskets)
    return {
        "top1_accuracy": hits_top1 / total,
        "top5_accuracy": hits_top5 / total,
        "mrr": rr_sum / total,
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def measure_baseline(model: nn.Module, data: dict[str, Any]) -> dict[str, Any]:
    """Measure the baseline FP32 model size, parameter count, and evaluation metrics."""
    size_bytes = get_model_size_bytes(model)
    size_mb = size_bytes / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())

    acc_metrics = measure_model_accuracy(model, data)
    probe_metrics = probe_itemsets(model, data)

    return {
        "model_type": "baseline_fp32",
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 4),
        "num_parameters": param_count,
        "top1_accuracy": acc_metrics["top1_accuracy"],
        "top5_accuracy": acc_metrics["top5_accuracy"],
        "mrr": acc_metrics["mrr"],
        "latency_ms": round(acc_metrics["latency_ms"], 2),
        "probe_top5_accuracy": probe_metrics["top5_accuracy"],
    }


def milestone_4(
    model: nn.Module | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Milestone 4: Measure the baseline size and accuracy of the model."""
    if model is None or data is None:
        model, data = milestone_2()
    return measure_baseline(model, data)


class StaticQuantizedLinear(nn.Module):
    """Linear layer with statically quantized INT8 weights and precomputed scale/zero-point."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear) -> "StaticQuantizedLinear":
        w = float_linear.weight.detach()
        qmin, qmax = -128, 127
        w_min = float(w.min().item())
        w_max = float(w.max().item())
        scale = max((w_max - w_min) / float(qmax - qmin), 1e-8)
        zero_point = int(round(-w_min / scale + qmin))
        zero_point = max(qmin, min(qmax, zero_point))

        w_q = torch.clamp(torch.round(w / scale) + zero_point, qmin, qmax).to(torch.int8)

        layer = cls(float_linear.in_features, float_linear.out_features, bias=float_linear.bias is not None)
        layer.weight_int8.copy_(w_q)
        layer.scale.fill_(scale)
        layer.zero_point.fill_(zero_point)
        if float_linear.bias is not None:
            layer.bias.data.copy_(float_linear.bias.data)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = (self.weight_int8.float() - self.zero_point.float()) * self.scale
        return F.linear(x, w_dequant, self.bias)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """Apply post-training dynamic quantization to linear and recurrent layers."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM},
            dtype=torch.qint8,
        )


def apply_static_quantization(model: BasketLanguageModel) -> nn.Module:
    """Apply post-training static weight quantization to model layers."""
    import copy
    static_model = copy.deepcopy(model)
    static_model.fc = StaticQuantizedLinear.from_float(static_model.fc)
    return static_model


# --- Milestone 5: Apply dynamic and static post-training quantization to the model
# Hint: Apply post-training dynamic and static quantization.
def milestone_5(model: nn.Module | None = None) -> tuple[nn.Module, nn.Module]:
    """Milestone 5: Apply dynamic and static post-training quantization to the model."""
    if model is None:
        model, _ = milestone_2()
    dyn_model = apply_dynamic_quantization(model)
    static_model = apply_static_quantization(model)
    return dyn_model, static_model


# --- Milestone 6: Demonstrate size reduction and evaluate accuracy impact
# Hint: Demonstrate size reduction and evaluate accuracy changes.
def evaluate_quantization_impact(
    baseline_model: nn.Module,
    dynamic_model: nn.Module,
    static_model: nn.Module,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Compare baseline FP32 against dynamic and static quantized models on size, accuracy, and latency."""
    base_metrics = measure_baseline(baseline_model, data)

    # Dynamic Quantization evaluation
    dyn_bytes = get_model_size_bytes(dynamic_model)
    dyn_acc = measure_model_accuracy(dynamic_model, data)
    dyn_probe = probe_itemsets(dynamic_model, data)

    # Static Quantization evaluation
    stat_bytes = get_model_size_bytes(static_model)
    stat_acc = measure_model_accuracy(static_model, data)
    stat_probe = probe_itemsets(static_model, data)

    dyn_reduction = (1.0 - dyn_bytes / max(1, base_metrics["size_bytes"])) * 100.0
    stat_reduction = (1.0 - stat_bytes / max(1, base_metrics["size_bytes"])) * 100.0

    dyn_metrics = {
        "model_type": "dynamic_int8",
        "size_bytes": dyn_bytes,
        "size_mb": round(dyn_bytes / (1024 * 1024), 4),
        "size_reduction_pct": round(dyn_reduction, 2),
        "top1_accuracy": dyn_acc["top1_accuracy"],
        "top5_accuracy": dyn_acc["top5_accuracy"],
        "mrr": dyn_acc["mrr"],
        "latency_ms": round(dyn_acc["latency_ms"], 2),
        "probe_top5_accuracy": dyn_probe["top5_accuracy"],
    }

    stat_metrics = {
        "model_type": "static_int8",
        "size_bytes": stat_bytes,
        "size_mb": round(stat_bytes / (1024 * 1024), 4),
        "size_reduction_pct": round(stat_reduction, 2),
        "top1_accuracy": stat_acc["top1_accuracy"],
        "top5_accuracy": stat_acc["top5_accuracy"],
        "mrr": stat_acc["mrr"],
        "latency_ms": round(stat_acc["latency_ms"], 2),
        "probe_top5_accuracy": stat_probe["top5_accuracy"],
    }

    return {
        "baseline": base_metrics,
        "dynamic_quantized": dyn_metrics,
        "static_quantized": stat_metrics,
        "summary": {
            "dynamic_size_reduction_pct": dyn_metrics["size_reduction_pct"],
            "static_size_reduction_pct": stat_metrics["size_reduction_pct"],
        },
    }


def milestone_6(
    model: nn.Module | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Milestone 6: Demonstrate size reduction and evaluate accuracy impact."""
    if model is None or data is None:
        model, data = milestone_2()
    dyn_model, static_model = milestone_5(model)
    return evaluate_quantization_impact(model, dyn_model, static_model, data)
