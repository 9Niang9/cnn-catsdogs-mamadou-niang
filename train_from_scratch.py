import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam, SGD
from torchvision.models import resnet18

from utils import compute_metrics, get_device, make_loaders, save_checkpoint, set_seed, to_json

ROOT = Path(__file__).resolve().parent


class SmallCNN(nn.Module):
    def __init__(self, dropout: float = 0.35):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.model(x)


def build_optimizer(model, optimizer_name, learning_rate, weight_decay):
    if optimizer_name.lower() == 'adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name.lower() == 'sgd':
        return SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Optimiseur non pris en charge : {optimizer_name}')


def train_once(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    all_probs = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs)
        all_labels.append(labels)
    metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_labels))
    return float(sum(losses) / max(len(losses), 1)), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            losses.append(loss.item())
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)
            all_labels.append(labels)
    metrics = compute_metrics(torch.cat(all_probs), torch.cat(all_labels))
    return float(sum(losses) / max(len(losses), 1)), metrics


def save_history(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Entraîner un CNN from scratch sur le dataset Cats vs Dogs.')
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'cosine'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, classes = make_loaders(args.data_dir, batch_size=args.batch_size, val_ratio=args.val_ratio, augment=True, limit=args.limit)

    model = SmallCNN(dropout=args.dropout).to(device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
    elif args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_once(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
        }
        history.append(entry)
        print(f"Époque {epoch:02d} | perte_train={train_loss:.4f} | perte_val={val_loss:.4f} | acc_val={val_metrics['accuracy']:.4f} | precision_val={val_metrics['precision']:.4f} | rappel_val={val_metrics['recall']:.4f}")
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = {
                'epoch': epoch,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'config': {
                    'optimizer': args.optimizer,
                    'lr': args.lr,
                    'dropout': args.dropout,
                    'scheduler': args.scheduler,
                    'classes': classes,
                },
            }

    if best_state is not None:
        save_checkpoint(checkpoint_dir / 'from_scratch_best.pth', model, optimizer, best_state['epoch'], best_val_acc, history, best_state['config'])
    save_history(history, checkpoint_dir / 'from_scratch_history.json')

    plt.figure(figsize=(12, 6))
    epochs = list(range(1, len(history) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h['train_loss'] for h in history], label='Train loss')
    plt.plot(epochs, [h['val_loss'] for h in history], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Entraînement CNN from scratch')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h['val_accuracy'] for h in history], label='Précision validation')
    plt.plot(epochs, [h['val_precision'] for h in history], label='Précision validation')
    plt.plot(epochs, [h['val_recall'] for h in history], label='Rappel validation')
    plt.xlabel('Époque')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plot_dir = ROOT / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / 'from_scratch_metrics.png', dpi=150)
    plt.close()
    print(f'Modèle enregistré dans : {checkpoint_dir / "from_scratch_best.pth"}')
    print(f'Graphiques enregistrés dans : {plot_dir / "from_scratch_metrics.png"}')


if __name__ == '__main__':
    main()
