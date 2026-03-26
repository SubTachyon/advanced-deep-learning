import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# settings
data_path = "data"
device = "cuda" # if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 100 #50 #20
lr = 0.001

# print(torch.cuda.is_available())

# load data
spectra = np.load(f"{data_path}/spectra.npy").astype(np.float32)
labels = np.load(f"{data_path}/labels.npy").astype(np.float32)

# keep only t_eff, log_g, fe_h
labels = labels[:, -4:-1]

# normalize spectra like the exercise says
spectra = np.log(np.maximum(spectra, 0.2))

# plot a few spectra
for i in range(5):
    plt.figure(figsize=(10, 4))
    plt.plot(spectra[i])
    plt.title(f"Star {i}")
    plt.xlabel("Wavelength bin")
    plt.ylabel("Log flux")
    plt.tight_layout()
    plt.savefig(f"spectrum_{i}.png")
    plt.close()

# split data: 70% train, 15% val, 15% test
n = len(spectra)
idx = np.random.permutation(n)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train = spectra[train_idx]
X_val = spectra[val_idx]
X_test = spectra[test_idx]

y_train = labels[train_idx]
y_val = labels[val_idx]
y_test = labels[test_idx]

# standardize using training data only
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0) + 1e-8

X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# convert to torch
X_train = torch.tensor(X_train).unsqueeze(1)
X_val = torch.tensor(X_val).unsqueeze(1)
X_test = torch.tensor(X_test).unsqueeze(1)

y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_test)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# 1D CNN
model = nn.Sequential(
    nn.Conv1d(1, 16, kernel_size=7, padding=3),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Conv1d(16, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Conv1d(32, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool1d(2),

    nn.Conv1d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(32), #nn.AdaptiveAvgPool1d(1),

    nn.Flatten(),
    nn.Linear(64 * 32, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []

# training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}  train loss: {train_loss:.4f}  val loss: {val_loss:.4f}")

# test predictions
model.eval()
preds = []
truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
        truth.append(yb.numpy())

preds = np.vstack(preds)
truth = np.vstack(truth)

# undo label normalization - DUH...!
preds = preds * y_std + y_mean
truth = truth * y_std + y_mean

names = ["t_eff", "log_g", "fe_h"]

# print simple metrics
for i, name in enumerate(names):
    mae = np.mean(np.abs(preds[:, i] - truth[:, i]))
    print(f"{name} MAE: {mae:.4f}")

# plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# plot predicted vs true
for i, name in enumerate(names):
    plt.figure(figsize=(5, 5))
    plt.scatter(truth[:, i], preds[:, i], s=10)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_true.png")
    plt.close()

print("Done! Saved plots to the current folder.")
