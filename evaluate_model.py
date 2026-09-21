import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

from utils import build_transforms, compute_metrics, get_device, set_seed


def load_model_from_checkpoint(checkpoint_path, model_type='scratch'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if model_type == 'scratch':
        from train_from_scratch import SmallCNN
        model = SmallCNN(dropout=0.35)
    else:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def evaluate_model(checkpoint_path, data_dir, model_type='scratch'): 
    set_seed(42)
    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path, model_type)
    model.to(device)
    model.eval()
    dataset = ImageFolder(data_dir, transform=build_transforms(224, False))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    metrics = compute_metrics(probs, labels)
    print(metrics)

    # confusion matrix
    preds = (probs >= 0.5).int()
    cm = torch.zeros(2, 2, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t.long(), p.long()] += 1
    print('Confusion matrix:\n', cm.numpy())

    plt.imshow(cm.numpy(), cmap='Blues')
    plt.xticks([0, 1], ['cat', 'dog'])
    plt.yticks([0, 1], ['cat', 'dog'])
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    out_dir = Path('plots')
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved Cats vs Dogs model checkpoint.')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--model_type', type=str, default='scratch', choices=['scratch', 'transfer'])
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.data_dir, args.model_type)


if __name__ == '__main__':
    main()
