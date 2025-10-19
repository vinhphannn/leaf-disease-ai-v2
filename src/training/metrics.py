"""Metrics helpers."""
from __future__ import annotations

from typing import List, Tuple
import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
	preds = logits.argmax(dim=1)
	return preds.eq(labels).float().mean().item()
