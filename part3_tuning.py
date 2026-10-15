# %% [markdown]
# ## Part 3: Hyperparameter Optimisation
#
# Grid search over 3 hyperparameters:
# - Learning rate: {0.001, 0.01, 0.1}
# - Batch size: {32, 64, 128}
# - Dropout: {0.0, 0.2, 0.5}
#
# 3 x 3 x 3 = 27 combinations, each trained for 10 epochs.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import json
import time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ### Data setup (same as Parts 1 & 2)

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

# %% [markdown]
# ### Model definition (with configurable dropout)

# %%
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
        x = self.fc3(x)
        return x

# %% [markdown]
# ### Training function (returns history)

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)

    for epoch in range(epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * images.size(0)

        history['train_loss'].append(running_train_loss / len(train_loader.dataset))

        # Validate
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        history['val_loss'].append(running_val_loss / len(val_loader.dataset))
        history['val_acc'].append(correct / total)

    return history

# %% [markdown]
# ### Grid search

# %%
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [32, 64, 128]
dropouts = [0.0, 0.2, 0.5]

results = []
all_histories = {}

total_runs = len(learning_rates) * len(batch_sizes) * len(dropouts)
run_id = 0
start_time = time.time()

for lr in learning_rates:
    for bs in batch_sizes:
        for do in dropouts:
            run_id += 1
            key = f"lr{lr}_bs{bs}_do{do}"
            print(f"\n[{run_id}/{total_runs}] {key}")

            # Fresh dataloaders with this batch size
            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

            torch.manual_seed(42)
            model = MLP(dropout=do)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr)

            t0 = time.time()
            hist = train_model(model, train_loader, val_loader,
                              criterion, optimizer, epochs=10)
            elapsed = time.time() - t0

            final_val_acc = hist['val_acc'][-1]
            best_val_acc = max(hist['val_acc'])

            print(f"   Final val acc: {final_val_acc:.4f}  (best: {best_val_acc:.4f})  "
                  f"Time: {elapsed:.1f}s")

            results.append({
                'lr': lr,
                'batch_size': bs,
                'dropout': do,
                'final_val_acc': final_val_acc,
                'best_val_acc': best_val_acc,
                'train_time_s': elapsed
            })
            all_histories[key] = hist

# %% [markdown]
# ### Save results

# %%
df = pd.DataFrame(results)
df.to_csv('part3_results.csv', index=False)

with open('part3_histories.json', 'w') as f:
    json.dump(all_histories, f)

print(f"\nTotal time: {(time.time() - start_time)/60:.1f} minutes")
print("\n=== Results table ===")
print(df.to_string(index=False))

# %% [markdown]
# ### Best configuration

# %%
best_row = df.loc[df['final_val_acc'].idxmax()]
print(f"\nBest configuration:")
print(f"  lr = {best_row['lr']}")
print(f"  batch_size = {best_row['batch_size']}")
print(f"  dropout = {best_row['dropout']}")
print(f"  Final val acc = {best_row['final_val_acc']:.4f}")
