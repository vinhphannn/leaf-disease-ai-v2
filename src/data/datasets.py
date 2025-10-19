"""Dataset utilities for training/validation/test loading.

Placeholders for future extensions: custom splits, caching, sampling, etc.
"""
from __future__ import annotations

from torchvision import datasets
from torchvision import transforms as T


def build_imagefolder(root: str, tfms: T.Compose) -> datasets.ImageFolder:
	return datasets.ImageFolder(root, transform=tfms)
