import os
import argparse
import time
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm  # progress bar


def build_eval_transforms(img_size: int = 224):
	return transforms.Compose([
		transforms.Resize(int(img_size * 1.15)),
		transforms.CenterCrop(img_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])


def load_clean_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, List[str]]:
	bundle = torch.load(model_path, map_location=device)
	classes = bundle.get('classes')
	model_name = bundle.get('model_name', 'caformer_s18.sail_in1k')
	if not classes:
		raise SystemExit("Model file missing 'classes'.")
	model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
	model.load_state_dict(bundle['model_state_dict'])
	model.to(device)
	model.eval()
	return model, classes


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device):
	criterion = nn.CrossEntropyLoss()
	model.eval()
	running_loss, correct, total = 0.0, 0, 0
	all_preds, all_labels = [], []
	amp_enabled = device.type == 'cuda'
	for imgs, labels in tqdm(loader, desc="Evaluating", unit="batch"):
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


def main():
	parser = argparse.ArgumentParser(description='Evaluate a saved clean model (.pth)')
	parser.add_argument('--model_path', default='models/model_best.pth')
	parser.add_argument('--data_dir', default='./data/valid')
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--img_size', type=int, default=224)
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Device: {device}')

	if not os.path.isfile(args.model_path):
		raise SystemExit(f"Model file not found: {args.model_path}")
	if not os.path.isdir(args.data_dir):
		raise SystemExit(f"Data directory not found: {args.data_dir}")

	model, classes = load_clean_model(args.model_path, device)
	tfms = build_eval_transforms(args.img_size)
	dataset = datasets.ImageFolder(args.data_dir, transform=tfms)

	# Align class order if needed
	if dataset.classes != classes:
		cls_to_idx = {c: i for i, c in enumerate(classes)}
		try:
			remapped = []
			for p, _ in dataset.samples:
				cname = os.path.basename(os.path.dirname(p))
				remapped.append((p, cls_to_idx[cname]))
			dataset.samples = remapped
			dataset.targets = [s[1] for s in remapped]
			print('Remapped dataset class indices to match model classes.')
		except Exception as e:
			raise SystemExit('Class order mismatch and remap failed. Ensure class names match.') from e

	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	t0 = time.time()
	val_loss, val_acc, preds, labels = validate(model, loader, device)
	dt = time.time() - t0
	print(f'Validation Loss: {val_loss:.4f}')
	print(f'Validation Accuracy: {val_acc:.4f}')
	print(f'Eval time: {dt:.1f}s')

	try:
		from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
		acc = accuracy_score(labels, preds)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
		print(f'Macro Precision: {precision:.4f}')
		print(f'Macro Recall:    {recall:.4f}')
		print(f'Macro F1:        {f1:.4f}')

		report = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)
		cm = confusion_matrix(labels, preds)

		out_dir = os.path.join('outputs')
		os.makedirs(out_dir, exist_ok=True)
		with open(os.path.join(out_dir, 'metrics_from_model.txt'), 'w', encoding='utf-8') as f:
			f.write(f"Model: {args.model_path}\n")
			f.write(f"Data: {args.data_dir}\n")
			f.write(f"Samples: {len(labels)}\n")
			f.write(f"Accuracy: {acc:.6f}\n")
			f.write(f"Macro Precision: {precision:.6f}\n")
			f.write(f"Macro Recall: {recall:.6f}\n")
			f.write(f"Macro F1: {f1:.6f}\n")
		with open(os.path.join(out_dir, 'classification_report_from_model.txt'), 'w', encoding='utf-8') as f:
			f.write(report)

		import csv
		cm_path = os.path.join(out_dir, 'confusion_matrix_from_model.csv')
		with open(cm_path, 'w', newline='', encoding='utf-8') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([''] + classes)
			for i, row in enumerate(cm):
				writer.writerow([classes[i]] + row.tolist())
		print('Saved outputs under outputs/: metrics_from_model.txt, classification_report_from_model.txt, confusion_matrix_from_model.csv')
	except Exception:
		print('scikit-learn not available; skipping detailed metrics saving.')


if __name__ == '__main__':
	main()


