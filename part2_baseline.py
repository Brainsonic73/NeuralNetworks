# %% [markdown]
# ## Part 2: Baseline MLP
#
# Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10
# Trained for 10 epochs with SGD, lr=0.01, batch_size=64.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ### Reuse data setup

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

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# %% [markdown]
# ### Network Architecture

# %%
class MLP(nn.Module):
    """
    Simple MLP for Fashion-MNIST.
    784 -> 128 -> 64 -> 10, with ReLU activations.
    """
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

# Sanity check
model_test = MLP()
x_dummy = torch.randn(4, 1, 28, 28)
print(f"Output shape: {model_test(x_dummy).shape}")

# %% [markdown]
# ### Training loop

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Trains a model and validates it each epoch.
    Returns history dict with train_loss, val_loss, val_acc per epoch.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    model = model.to(device)

    print("Starting Training...")
    for epoch in range(epochs):
        # Training
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

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation
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

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print("Finished Training")
    return history

# %% [markdown]
# ### Train the baseline

# %%
baseline_model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(baseline_model.parameters(), lr=0.01)

baseline_history = train_model(
    baseline_model, train_loader, val_loader,
    criterion, optimizer, epochs=10
)

baseline_acc = baseline_history['val_acc'][-1]
print(f"\nBaseline final validation accuracy: {baseline_acc:.4f}")

# %% [markdown]
# ### Plot the baseline loss curves

# %%
fig, ax = plt.subplots(figsize=(8, 5))
epochs = range(1, 11)
ax.plot(epochs, baseline_history['train_loss'], marker='o', label='Train Loss')
ax.plot(epochs, baseline_history['val_loss'], marker='s', label='Val Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Baseline Model — Loss Curves')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_loss_curves.png', dpi=100)
plt.show()
print("Saved baseline_loss_curves.png")
