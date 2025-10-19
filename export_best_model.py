import os
import glob
import argparse
from typing import List, Optional, Tuple

import torch
import timm


def discover_checkpoints(search_dirs: List[str]) -> List[str]:
    patterns = [
        "checkpoint_stage*_best.pth",
        "checkpoint_best.pth",
        "checkpoint_stage*_latest.pth",
        "checkpoint_latest.pth",
        "checkpoint_stage*_epoch_*.pth",
        "checkpoint_epoch_*.pth",
    ]
    found: List[str] = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(d, pat)))
    # dedupe
    seen = set()
    unique = []
    for p in found:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def find_best_checkpoint(search_dirs: List[str]) -> Optional[str]:
    candidates = discover_checkpoints(search_dirs)
    if not candidates:
        return None
    stage_best = [p for p in candidates if os.path.basename(p).startswith("checkpoint_stage") and p.endswith("_best.pth")]
    if stage_best:
        return max(stage_best, key=lambda p: os.path.getmtime(p))
    global_best = [p for p in candidates if os.path.basename(p) == "checkpoint_best.pth"]
    if global_best:
        return max(global_best, key=lambda p: os.path.getmtime(p))
    # fallback newest
    return max(candidates, key=lambda p: os.path.getmtime(p))


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def load_model_from_ckpt(ckpt_path: str, device: torch.device, model_name: str) -> Tuple[torch.nn.Module, List[str]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get('classes')
    if classes is None:
        raise SystemExit("Checkpoint missing 'classes'; cannot determine num_classes.")
    model = build_model(model_name=model_name, num_classes=len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes


def main():
    parser = argparse.ArgumentParser(description="Export clean model file from best checkpoint")
    parser.add_argument("--checkpoint_dirs", nargs='*', default=[".", "./checkpoint", "./checkpoints"], help="Directories to search for checkpoints")
    parser.add_argument("--model_name", default="caformer_s18.sail_in1k", help="TIMM model name used during training")
    parser.add_argument("--out", default="models/model_best.pth", help="Output file path for clean model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = find_best_checkpoint(args.checkpoint_dirs)
    if not ckpt_path:
        raise SystemExit("No checkpoint found in provided directories.")
    print(f"Using checkpoint: {ckpt_path}")

    model, classes = load_model_from_ckpt(ckpt_path, device, args.model_name)

    # Save a clean bundle with just model weights and classes
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': classes,
        'model_name': args.model_name,
    }, args.out)

    print(f"Exported clean model to: {args.out}")


if __name__ == "__main__":
    main()


