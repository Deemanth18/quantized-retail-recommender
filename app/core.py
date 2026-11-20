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
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Measure the baseline size and accuracy of the model
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Apply dynamic and static post-training quantization to the model
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Demonstrate size reduction and evaluate accuracy impact
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
