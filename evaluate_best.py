import os
import glob
import argparse
import time
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm


def discover_checkpoints(search_dirs: List[str]) -> List[str]:
    """Return a list of checkpoint paths discovered in the given directories."""
    patterns = [
        "checkpoint_stage*_best.pth",
        "checkpoint_stage*_latest.pth",
        "checkpoint_stage*_epoch_*.pth",
        "checkpoint_best.pth",
        "checkpoint_latest.pth",
        "checkpoint_epoch_*.pth",
    ]
    found: List[str] = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(d, pat)))
    # Dedupe while preserving order
    seen = set()
    unique = []
    for p in found:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def find_best_checkpoint(search_dirs: List[str]) -> Optional[str]:
    """Pick the most appropriate checkpoint file to evaluate.

    Priority:
    1) Any stage-specific best (newest by mtime)
    2) Global best
    3) Otherwise the newest among latest/epoch files
    """
    candidates = discover_checkpoints(search_dirs)
    if not candidates:
        return None

    stage_best = [p for p in candidates if os.path.basename(p).startswith("checkpoint_stage") and "_best.pth" in p]
    if stage_best:
        return max(stage_best, key=lambda p: os.path.getmtime(p))

    global_best = [p for p in candidates if os.path.basename(p) == "checkpoint_best.pth"]
    if global_best:
        # If multiple directories contain it, pick the newest
        return max(global_best, key=lambda p: os.path.getmtime(p))

    # Fallback: newest by mtime
    return max(candidates, key=lambda p: os.path.getmtime(p))


def build_transforms(img_size: int = 224):
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return val_tfms


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    amp_enabled = device.type == "cuda"
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return running_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, List[str]]:
    """Recreate the model and load weights from a saved checkpoint.

    The training notebook used TIMM and stored `classes` in the checkpoint. We will
    reconstruct the same architecture with `caformer_s18.sail_in1k` by default.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)

    classes = checkpoint.get('classes')
    if classes is None:
        raise RuntimeError("Checkpoint missing 'classes'. Cannot infer num_classes.")

    num_classes = len(classes)
    model_name = "caformer_s18.sail_in1k"  # must match the training config
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model, classes


def main():
    parser = argparse.ArgumentParser(description="Evaluate best plant disease model checkpoint")
    parser.add_argument("--data_dir", default="./data/valid", help="Path to validation dataset root (ImageFolder)")
    parser.add_argument("--checkpoint_dirs", nargs='*', default=[".", "./checkpoint"], help="Directories to search for checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = find_best_checkpoint(args.checkpoint_dirs)
    if not ckpt_path:
        raise SystemExit("No checkpoint found. Ensure training has produced a checkpoint.")
    print(f"Using checkpoint: {ckpt_path}")

    model, classes = load_model_from_checkpoint(ckpt_path, device)

    # Dataset and loader
    tfms = build_transforms(args.img_size)
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"Validation data directory not found: {args.data_dir}")
    val_dataset = datasets.ImageFolder(args.data_dir, transform=tfms)

    # Validate that class order matches
    if val_dataset.classes != classes:
        # Map dataset classes to checkpoint class order if possible
        cls_to_idx_ckpt = {c: i for c, i in checkpoint_class_to_idx_from_ckpt(ckpt_path).items()}
        try:
            remapped_samples = []
            for path, _ in val_dataset.samples:
                cls_name = os.path.basename(os.path.dirname(path))
                remapped_samples.append((path, cls_to_idx_ckpt[cls_name]))
            val_dataset.samples = remapped_samples
            val_dataset.targets = [s[1] for s in remapped_samples]
            print("Remapped validation labels to checkpoint class order.")
        except Exception as e:
            raise SystemExit(
                "Class order mismatch and could not remap. Ensure validation directory classes match training classes."
            ) from e

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start = time.time()
    val_loss, val_acc, preds, labels = validate(model, val_loader, device)
    dur = time.time() - start
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Eval time: {dur:.1f}s")

    # Extended metrics and reports
    try:
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
    except Exception:
        print("scikit-learn not available; skipping detailed metrics.")
        return

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")

    # Per-class report
    report = classification_report(labels, preds, target_names=val_dataset.classes, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)

    # Save artifacts
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    # Summary
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Samples: {len(labels)}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Macro Precision: {precision:.6f}\n")
        f.write(f"Macro Recall: {recall:.6f}\n")
        f.write(f"Macro F1: {f1:.6f}\n")

    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Save confusion matrix as CSV
    try:
        import numpy as np
        import csv
        cm_path = os.path.join(out_dir, "confusion_matrix.csv")
        with open(cm_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([""] + val_dataset.classes)
            for i, row in enumerate(cm):
                writer.writerow([val_dataset.classes[i]] + row.tolist())
        print(f"Saved metrics to {out_dir}/metrics.txt, classification_report.txt, confusion_matrix.csv")
    except Exception:
        print("Failed to save confusion matrix CSV; numpy/csv issue.")


def checkpoint_class_to_idx_from_ckpt(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_to_idx = ckpt.get('class_to_idx')
    if class_to_idx is None:
        # Fallback from ordered `classes`
        classes = ckpt.get('classes')
        if not classes:
            raise RuntimeError("Checkpoint missing both 'class_to_idx' and 'classes'.")
        return {c: i for i, c in enumerate(classes)}
    return class_to_idx


if __name__ == "__main__":
    main()


