"""Checkpoint helpers aligned with stage-aware naming."""
from __future__ import annotations

import os
import glob
from typing import List, Optional


def stage_tag(stage: int) -> str:
	return f"stage{stage}"


def list_all_checkpoints(root: str = ".") -> List[str]:
	pats = [
		"checkpoint_stage*_best.pth",
		"checkpoint_stage*_latest.pth",
		"checkpoint_stage*_epoch_*.pth",
		"checkpoint_best.pth",
		"checkpoint_latest.pth",
		"checkpoint_epoch_*.pth",
	]
	out: List[str] = []
	for pat in pats:
		out.extend(glob.glob(os.path.join(root, pat)))
	return out


def newest_checkpoint(paths: List[str]) -> Optional[str]:
	return max(paths, key=lambda p: os.path.getmtime(p)) if paths else None
