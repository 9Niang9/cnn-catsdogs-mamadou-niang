import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam, SGD
from torchvision.models import resnet18, ResNet18_Weights

from utils import compute_metrics, get_device, make_loaders, save_checkpoint, set_seed, to_json


def build_model(freeze_backbone: bool = True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


def build_optimizer(model, optimizer_name, learning_rate, weight_decay):
    params = model.parameters() if optimizer_name.lower() == 'sgd' else model.parameters()
    if optimizer_name.lower() == 'adam':
        return Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name.lower() == 'sgd':
        return SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
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


def main():
    parser = argparse.ArgumentParser(description='Fine-tuner un ResNet18 pré-entraîné sur Cats vs Dogs.')
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--freeze_backbone', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'step', 'cosine'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, classes = make_loaders(args.data_dir, batch_size=args.batch_size, val_ratio=args.val_ratio, augment=True, limit=args.limit)
    model = build_model(freeze_backbone=bool(args.freeze_backbone)).to(device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.scheduler == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    elif args.scheduler == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_acc = -1.0
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
            save_checkpoint(Path(args.checkpoint_dir) / 'transfer_learning_best.pth', model, optimizer, epoch, best_val_acc, history, {
                'optimizer': args.optimizer,
                'lr': args.lr,
                'freeze_backbone': bool(args.freeze_backbone),
                'scheduler': args.scheduler,
                'classes': classes,
            })

    with open(Path(args.checkpoint_dir) / 'transfer_learning_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(12, 6))
    epochs = list(range(1, len(history) + 1))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h['train_loss'] for h in history], label='Train loss')
    plt.plot(epochs, [h['val_loss'] for h in history], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Entraînement par transfert learning')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h['val_accuracy'] for h in history], label='Précision validation')
    plt.plot(epochs, [h['val_precision'] for h in history], label='Précision validation')
    plt.plot(epochs, [h['val_recall'] for h in history], label='Rappel validation')
    plt.xlabel('Époque')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plot_dir = Path('plots')
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / 'transfer_learning_metrics.png', dpi=150)
    plt.close()
    print(f'Modèle enregistré dans : {Path(args.checkpoint_dir) / "transfer_learning_best.pth"}')
    print(f'Graphiques enregistrés dans : {plot_dir / "transfer_learning_metrics.png"}')


if __name__ == '__main__':
    main()
