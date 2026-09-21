import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_dataset(data_dir: str, min_classes: int = 2) -> None:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data folder not found: {root}")
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if len(classes) < min_classes:
        raise ValueError(
            f"Expected at least {min_classes} class folders under {root}, but found {classes}. "
            "Download the Kaggle Cats vs Dogs dataset and keep the structure train/cat and train/dog."
        )


def build_transforms(image_size: int = 224, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if augment:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    return transforms.Compose(ops)


def split_dataset(dataset, val_ratio: float = 0.2, limit: int | None = None):
    if limit is not None:
        indices = list(range(len(dataset)))
        rng = np.random.default_rng(42)
        indices = rng.choice(indices, size=min(limit, len(indices)), replace=False).tolist()
        dataset = Subset(dataset, indices)

    total = len(dataset)
    val_size = max(1, int(round(total * val_ratio)))
    train_size = total - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError('The dataset is too small for a proper train/validation split.')

    indices = list(range(total))
    random.Random(42).shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_loaders(data_dir: str, batch_size: int = 32, val_ratio: float = 0.2, augment: bool = True, limit: int | None = None):
    ensure_dataset(data_dir)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=build_transforms(augment=augment))
    train_subset, val_subset = split_dataset(full_dataset, val_ratio=val_ratio, limit=limit)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, full_dataset.classes


def compute_metrics(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred_labels = (preds >= 0.5).astype(int)
    tp = int(((pred_labels == 1) & (labels == 1)).sum())
    tn = int(((pred_labels == 0) & (labels == 0)).sum())
    fp = int(((pred_labels == 1) & (labels == 0)).sum())
    fn = int(((pred_labels == 0) & (labels == 1)).sum())

    acc = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def to_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def save_checkpoint(path, model, optimizer, epoch, best_metric, history, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
        'history': history,
        'config': config,
    }
    torch.save(checkpoint, path)
