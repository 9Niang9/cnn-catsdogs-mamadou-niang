# CNN from scratch vs Transfer Learning – Cats vs Dogs

## Objectif

Ce projet compare deux approches sur le même corpus Cats vs Dogs :

- Expérience A : CNN entraîné from scratch avec 3 blocs convolutionnels, BatchNorm, Dropout et réglage manuel de l’optimiseur.
- Expérience B : transfert learning avec un modèle pré-entraîné ResNet18, avec adaptation des couches finales sur le problème binaire.

L’objectif est de mesurer l’impact du transfert learning sur la vitesse de convergence, la performance finale et la robustesse du modèle.

---

## Environnement

### Installer les dépendances

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# ou sur Windows PowerShell : .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Le dépôt contient aussi un environnement Python déjà présent dans le workspace, qui peut être utilisé si le cours l’a préparé :

```powershell
& 'c:\Users\mamad\Desktop\DIT\Deep_Learnig\DL_PYTORCH\dl_pytorch\Scripts\python.exe' -m pip install -r .\requirements.txt
```

### Vérifier le GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

- Si `cuda` est disponible, les scripts utilisent automatiquement le GPU.
- Sinon, ils se lancent sur CPU.

---

## Organisation des données

Le projet attend un jeu de données structuré comme suit :

```text
data/
├─ train/
│  ├─ cat/
│  └─ dog/
├─ test/
│  ├─ cat/
│  └─ dog/
└─ ...
```

### Télécharger le corpus

Le corpus Cats vs Dogs peut être téléchargé via KaggleHub :

```python
import kagglehub

# Download latest version
path = kagglehub.competition_download('dogs-vs-cats')
print("Path to competition files:", path)
```

Ensuite, il faut extraire le zip dans le dossier `data/` du dépôt.

L’alternative Fourni dans le projet est :

```bash
python download_data.py
```

Important :

- Les données ne doivent pas être poussées vers GitHub.
- Le dépôt contient un `.gitignore` qui ignore `data/`, `checkpoints/`, `runs/`, `plots/` et les fichiers `.pth`.
- KaggleHub nécessite une authentification Kaggle valide pour télécharger le dataset.

---

## Séed et reproductibilité

Le seed est fixé à 42 dans les scripts :

```python
set_seed(42)
```

Cela rend les expériences reproductibles, y compris les séparations train/validation et la génération des lots.

---

## Modèles et bonnes pratiques

### Expérience A — CNN from scratch

Le modèle de base contient :

- 4 blocs convolutionnels
- Batch Normalization après chaque bloc convolutionnel
- ReLU
- MaxPooling
- Dropout sur les couches entièrement connectées
- Classifier binaire final

Justification :

- `BatchNorm` stabilise la distribution de chaque batch et accélère la convergence.
- `Dropout` réduit le surapprentissage, surtout sur un petit corpus.

### Expérience B — Transfer learning

Le script utilise `ResNet18` pré-entraîné sur ImageNet. Les couches finales sont remplacées par un classifieur binaire et, selon le mode choisi, le backbone est gelé ou fine-tuné.

---

## Commandes d’entraînement

### 1) From scratch

```bash
python train_from_scratch.py \
  --data_dir data/train \
  --epochs 12 \
  --batch_size 32 \
  --optimizer adam \
  --lr 1e-3 \
  --dropout 0.35 \
  --scheduler cosine \
  --val_ratio 0.2 \
  --checkpoint_dir checkpoints
```

### 2) Transfer learning

```bash
python train_transfer_learning.py \
  --data_dir data/train \
  --epochs 8 \
  --batch_size 32 \
  --optimizer adam \
  --lr 1e-4 \
  --freeze_backbone 1 \
  --scheduler cosine \
  --val_ratio 0.2 \
  --checkpoint_dir checkpoints
```

### 3) Optimiseurs et learning rate testés

Au minimum, les scripts supportent :

- `adam`
- `sgd`

Le code est prêt pour des essais de learning rate, avec option de scheduler (`none`, `step`, `cosine`).

---

## Évaluation et rechargement du meilleur modèle

### Recharger le meilleur checkpoint

```bash
python evaluate_model.py \
  --checkpoint checkpoints/from_scratch_best.pth \
  --data_dir data/test \
  --model_type scratch
```

```bash
python evaluate_model.py \
  --checkpoint checkpoints/transfer_learning_best.pth \
  --data_dir data/test \
  --model_type transfer
```

Les scripts produisent également :

- les métriques en console,
- la matrice de confusion sous `plots/confusion_matrix.png`,
- les courbes de loss/métriques sous `plots/`.

---

## Métriques suivies

Les scripts calculent, à chaque époque et sur la validation :

- `train_loss`
- `val_loss`
- accuracy
- precision
- recall

Les résultats sont enregistrés dans :

```text
checkpoints/
├─ from_scratch_history.json
├─ from_scratch_best.pth
├─ transfer_learning_history.json
├─ transfer_learning_best.pth
```

---

## Résultats attendus et comparaison

| Modèle | Optimiseur | LR | Epochs | Val Accuracy | Val Precision | Val Recall |
|---|---:|---:|---:|---:|---:|---:|
| CNN from scratch | Adam | 1e-3 | 12 | À remplir | À remplir | À remplir |
| Transfer learning (ResNet18) | Adam | 1e-4 | 8 | À remplir | À remplir | À remplir |

### Analyse attendue

Le transfert learning est généralement attendu pour :

- converger plus vite,
- atteindre un meilleur score sur un petit corpus,
- fournir une meilleure robustesse au départ de l’apprentissage.

Le CNN from scratch reste utile si l’on veut un modèle léger, totalement personnalisé, ou si le jeu de données est suffisamment volumineux. En pratique, sur Cats vs Dogs, le transfert learning a habituellement un avantage net, surtout lorsqu’on dispose d’un petit nombre d’images.

---

## Limites et pistes d’amélioration

- Ajouter plus de données d’augmentation pour améliorer la généralisation.
- Tester d’autres backbones (EfficientNet, MobileNetV3, DenseNet).
- Explorer un scheduler plus agressif ou un fine-tuning complet du backbone.
- Ajouter TensorBoard ou W&B pour la journalisation visuelle.
- Faire un split plus explicite train/validation/test si le corpus est large.

---

## Note sur ce workspace

Le dossier de données de ce workspace ne contenait pas encore le corpus complet Cats vs Dogs au moment de la validation. La structure du projet est donc prête et reproductible, mais les expérimentations complètes doivent être lancées une fois le jeu de données téléchargé et placé dans `data/` selon la hiérarchie ci-dessus.
