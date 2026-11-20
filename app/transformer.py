"""A compact causal Transformer for next-SKU prediction.

The model is intentionally small and designed for local experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class SKUTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_length: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        length = tokens.shape[1]
        if length > self.max_length:
            tokens = tokens[:, -self.max_length :]
            length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        # True entries are masked, so each position can only attend to its past.
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def predict_next(self, tokens: torch.Tensor, top_k: int = 5):
        self.eval()
        logits = self(tokens)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        return torch.topk(probs, k=min(top_k, self.vocab_size), dim=-1)


def encode_sequences(sequences: Iterable[Iterable[str]]):
    sequences = [list(map(str, seq)) for seq in sequences if len(seq) >= 2]
    vocab = sorted({sku for seq in sequences for sku in seq})
    stoi = {sku: i for i, sku in enumerate(vocab)}
    examples = []
    for seq in sequences:
        ids = [stoi[s] for s in seq]
        for end in range(1, len(ids)):
            examples.append((ids[:end], ids[end]))
    return vocab, examples


def train_transformer(
    sequences: Iterable[Iterable[str]],
    epochs: int = 20,
    learning_rate: float = 1e-3,
    d_model: int = 64,
    max_length: int = 32,
    device: str | None = None,
):
    vocab, examples = encode_sequences(sequences)
    if not examples:
        raise ValueError("At least one sequence with two or more SKUs is required")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SKUTransformer(len(vocab), d_model=d_model, max_length=max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    model.train()
    for _ in range(max(1, int(epochs))):
        total = 0.0
        for prefix, target in examples:
            x = torch.tensor([prefix[-max_length:]], dtype=torch.long, device=device)
            y = torch.tensor([target], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x)[:, -1, :], y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        losses.append(total / len(examples))
    return model, vocab, losses


def save_checkpoint(model: SKUTransformer, vocab: list[str], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "vocab_size": model.vocab_size,
                "max_length": model.max_length,
                "d_model": model.token_embedding.embedding_dim,
            },
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str = "cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = SKUTransformer(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["vocab"]


def predict_skus(model: SKUTransformer, vocab: list[str], history: list[str], top_k: int = 5):
    stoi = {sku: i for i, sku in enumerate(vocab)}
    known = [stoi[sku] for sku in history if sku in stoi]
    if not known:
        return []
    device = next(model.parameters()).device
    x = torch.tensor([known[-model.max_length :]], dtype=torch.long, device=device)
    values, indices = model.predict_next(x, top_k=top_k)
    return [
        {"sku": vocab[int(idx)], "score": float(score)}
        for score, idx in zip(values[0].cpu(), indices[0].cpu())
    ]
