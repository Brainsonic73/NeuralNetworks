# %% [markdown]
# ## Part 4: Analysis and Report

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %% [markdown]
# ### Load results from Part 3

# %%
df = pd.read_csv('part3_results.csv')
print(df.head())
print(f"\nBest final config: {df.loc[df['final_val_acc'].idxmax()].to_dict()}")
print(f"Best observed config: {df.loc[df['best_val_acc'].idxmax()].to_dict()}")

with open('part3_histories.json', 'r') as f:
    histories = json.load(f)

# %% [markdown]
# ### Table for the report

# %%
print("\n=== Full Results Table ===")
print(df.to_string(index=False))

# %% [markdown]
# ### Visualisation 1: Hyperparameter impact heatmap
#
# We'll show three heatmaps — one per (batch_size, dropout) pair — with
# learning rate on the x-axis. Easier to compare.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
pivot_dropouts = [0.0, 0.2, 0.5]

for ax, do in zip(axes, pivot_dropouts):
    sub = df[df['dropout'] == do]
    pivot = sub.pivot(index='batch_size', columns='lr', values='final_val_acc')
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu',
                vmin=0.70, vmax=0.90, ax=ax, cbar=(do == 0.5))
    ax.set_title(f'Dropout = {do}')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size' if do == 0.0 else '')

plt.tight_layout()
plt.savefig('fig_hyperparam_heatmap.png', dpi=100)
plt.show()

# %% [markdown]
# ### Visualisation 2: Bar chart — best per learning rate

# %%
fig, ax = plt.subplots(figsize=(8, 5))
best_per_lr = df.groupby('lr')['best_val_acc'].max()
ax.bar(best_per_lr.index.astype(str), best_per_lr.values, color='steelblue')
for i, v in enumerate(best_per_lr.values):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center')
ax.set_xlabel('Learning Rate')
ax.set_ylabel('Best Validation Accuracy')
ax.set_title('Best Validation Accuracy per Learning Rate')
ax.set_ylim(0.70, 0.92)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_lr_impact.png', dpi=100)
plt.show()

# %% [markdown]
# ### Train baseline + best model with identical setup for loss-curve comparison

# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
full_train = datasets.FashionMNIST(root='./data', train=True,
                                   download=True, transform=transform)
train_set, val_set = random_split(
    full_train, [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

class MLP(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        history['train_loss'].append(running / len(train_loader.dataset))
        model.eval()
        rv, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                rv += criterion(out, lbls).item() * imgs.size(0)
                correct += (out.argmax(1) == lbls).sum().item()
                total += lbls.size(0)
        history['val_loss'].append(rv / len(val_loader.dataset))
        history['val_acc'].append(correct / total)
    return history

# Baseline
train_loader_b = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader_b = DataLoader(val_set, batch_size=64, shuffle=False)
baseline_model = MLP(dropout=0.0)
baseline_hist = train_model(baseline_model, train_loader_b, val_loader_b,
                            nn.CrossEntropyLoss(),
                            optim.SGD(baseline_model.parameters(), lr=0.01),
                            epochs=10)
print(f"Baseline final val acc: {baseline_hist['val_acc'][-1]:.4f}")

# Best
train_loader_best = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader_best = DataLoader(val_set, batch_size=32, shuffle=False)
best_model = MLP(dropout=0.0)
best_hist = train_model(best_model, train_loader_best, val_loader_best,
                        nn.CrossEntropyLoss(),
                        optim.SGD(best_model.parameters(), lr=0.1),
                        epochs=10)
print(f"Best model final val acc: {best_hist['val_acc'][-1]:.4f}")

# %% [markdown]
# ### Visualisation 3: Loss curves — baseline vs best

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, 11)

axes[0].plot(epochs, baseline_hist['train_loss'], 'o-', label='Train')
axes[0].plot(epochs, baseline_hist['val_loss'], 's-', label='Val')
axes[0].set_title('Baseline (lr=0.01, bs=64, do=0.0)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs, best_hist['train_loss'], 'o-', label='Train')
axes[1].plot(epochs, best_hist['val_loss'], 's-', label='Val')
axes[1].set_title('Best (lr=0.1, bs=32, do=0.0)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_loss_curves.png', dpi=100)
plt.show()

# %% [markdown]
# ### Visualisation 4: Confusion matrix for best model

# %%
best_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in val_loader_best:
        imgs = imgs.to(device)
        preds = best_model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title('Confusion Matrix — Best Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig_confusion_matrix.png', dpi=100)
plt.show()

# %% [markdown]
# ### Which classes are confused?

# %%
# Off-diagonal top confusions
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
flat_idx = np.argsort(cm_off.flatten())[::-1][:6]
print("\nTop 6 confused pairs (true → predicted):")
for idx in flat_idx:
    i, j = idx // 10, idx % 10
    if cm_off[i, j] > 0:
        print(f"  {CLASS_NAMES[i]:>12} → {CLASS_NAMES[j]:<12} : {cm_off[i, j]} times")
