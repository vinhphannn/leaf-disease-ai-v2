"""Model factory using TIMM."""
from __future__ import annotations

import timm
from torch import nn


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
	return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
