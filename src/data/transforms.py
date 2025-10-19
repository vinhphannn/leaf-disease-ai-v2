"""Image transforms used for train/val/test."""
from __future__ import annotations

from torchvision import transforms as T


def build_train_transforms(img_size: int = 224) -> T.Compose:
	return T.Compose([
		T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
		T.RandomHorizontalFlip(),
		T.ColorJitter(0.2, 0.2, 0.15, 0.05),
		T.RandAugment(num_ops=2, magnitude=9),
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])


def build_eval_transforms(img_size: int = 224) -> T.Compose:
	return T.Compose([
		T.Resize(int(img_size * 1.15)),
		T.CenterCrop(img_size),
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])
