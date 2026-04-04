import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.ticker import FuncFormatter

# settings
data_path = "data"
device = "cuda" # if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 30
lr = 1e-3
min_log_sigma = -6.0
max_log_sigma = 3.0
seed = 42

# load data
spectra = np.load(f"{data_path}/spectra.npy").astype(np.float32)
labels = np.load(f"{data_path}/labels.npy").astype(np.float32)

# keep only t_eff, log_g, fe_h
labels = labels[:, -4:-1]

# normalize spectra
spectra = np.log(np.maximum(spectra, 0.2))

# # plot a few spectra
# for i in range(5):
#     plt.figure(figsize=(10, 4))
#     plt.plot(spectra[i])
#     plt.title(f"Star {i}")
#     plt.xlabel("Wavelength bin")
#     plt.ylabel("Log flux")
#     plt.tight_layout()
#     plt.savefig(f"spectrum_{i}.png")
#     plt.close()

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

# 1D CNN that predicts both values and uncertainties
# outputs are: [mu_t_eff, mu_log_g, mu_fe_h, log_sigma_t_eff, log_sigma_log_g, log_sigma_fe_h]
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
    nn.AdaptiveAvgPool1d(32),

    nn.Flatten(),
    nn.Linear(64 * 32, 64),
    nn.ReLU(),
    nn.Linear(64, 6)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses = []
best_val_loss = float("inf")
best_state = None

# training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        out = model(xb)
        mu = out[:, :3]
        log_sigma = out[:, 3:]
        log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)

        inv_var_term = 0.5 * ((yb - mu) / torch.exp(log_sigma)) ** 2
        loss = torch.mean(inv_var_term + log_sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            mu = out[:, :3]
            log_sigma = out[:, 3:]
            log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)

            inv_var_term = 0.5 * ((yb - mu) / torch.exp(log_sigma)) ** 2
            val_loss += torch.mean(inv_var_term + log_sigma).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch + 1}/{epochs} | train NLL: {train_loss:.4f} | val NLL: {val_loss:.4f}")

# restore best validation model
if best_state is not None:
    model.load_state_dict(best_state)

# test predictions
model.eval()
preds_mu = []
preds_sigma = []
truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        out = model(xb)
        mu = out[:, :3]
        log_sigma = out[:, 3:]
        log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)

        preds_mu.append(mu.cpu().numpy())
        preds_sigma.append(np.exp(log_sigma.cpu().numpy()))
        truth.append(yb.numpy())

preds_mu = np.vstack(preds_mu)
preds_sigma = np.vstack(preds_sigma)
truth = np.vstack(truth)

# undo label normalization
preds_mu_phys = preds_mu * y_std + y_mean
truth_phys = truth * y_std + y_mean
preds_sigma_phys = preds_sigma * y_std  # sigma rescales linearly

###############################
###### OUTPUT AND PLOTS #######
###############################

names = ["t_eff", "log_g", "fe_h"]

# print("\nTest-set metrics:")
# for i, name in enumerate(names):
#     residuals = preds_mu_phys[:, i] - truth_phys[:, i]
#     sigma_i = preds_sigma_phys[:, i]
#     mae = np.mean(np.abs(residuals))
#     rmse = np.sqrt(np.mean(residuals ** 2))
#     mean_sigma = np.mean(sigma_i)
#     pull = residuals / (sigma_i + 1e-12)
#     pull_mean = np.mean(pull)
#     pull_std = np.std(pull)
#     coverage_1sigma = np.mean(np.abs(residuals) <= sigma_i)
#     coverage_2sigma = np.mean(np.abs(residuals) <= 2.0 * sigma_i)

#     print(f"{name}:")
#     print(f"MAE = {mae:.4f}")
#     print(f"RMSE = {rmse:.4f}")
#     print(f"mean predicted sigma = {mean_sigma:.4f}")
#     print(f"pull mean = {pull_mean:.4f} (ideal 0)")
#     print(f"pull std = {pull_std:.4f} (ideal 1)")
#     print(f"1 sigma coverage = {coverage_1sigma:.4f} (ideal ~0.6827)")
#     print(f"2 sigma coverage = {coverage_2sigma:.4f} (ideal ~0.9545)")

# # overall uncertainty calibration summary
# all_residuals = preds_mu_phys - truth_phys
# all_pulls = all_residuals / (preds_sigma_phys + 1e-12)
# print("\nOverall calibration summary across all targets:")
# print(f"Pull mean = {np.mean(all_pulls):.4f}")
# print(f"Pull std  = {np.std(all_pulls):.4f}")
# print(f"1 sigma coverage = {np.mean(np.abs(all_residuals) <= preds_sigma_phys):.4f}")
# print(f"2 sigma coverage = {np.mean(np.abs(all_residuals) <= 2.0 * preds_sigma_phys):.4f}")

# loss curve
# fucking y axis...
plt.figure(figsize=(8, 4))
loss_floor = min(min(train_losses), min(val_losses))
loss_shift = 1.0 - loss_floor if loss_floor <= 0 else 0.0
plt.plot(np.array(train_losses) + loss_shift, label="train")
plt.plot(np.array(val_losses) + loss_shift, label="val")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
plt.xlabel("Epoch")
plt.ylabel("Gaussian NLL loss")
plt.title("Training and validation NLL")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve_uncertainty.png")
plt.close()

# predicted vs true with representative error bars
rng = np.random.default_rng(seed)

for i, name in enumerate(names):
    plt.figure(figsize=(5, 5))

    n_points = len(truth_phys)
    n_sample = min(100, n_points)
    sample = rng.choice(n_points, size=n_sample, replace=False)

    mask = np.ones(n_points, dtype=bool)
    mask[sample] = False

    # lighter points without error bars
    plt.scatter(
        truth_phys[mask, i],
        preds_mu_phys[mask, i],
        s=10,
        color="#7fb3d5",
    )

    # darker points with error bars
    plt.scatter(
        truth_phys[sample, i],
        preds_mu_phys[sample, i],
        s=14,
        color="#1f4e79",
    )

    plt.errorbar(
        truth_phys[sample, i],
        preds_mu_phys[sample, i],
        yerr=preds_sigma_phys[sample, i],
        fmt="none",
        ecolor="#1f4e79",
        alpha=0.35,
        elinewidth=1,
        capsize=2,
    )

    minv = min(truth_phys[:, i].min(), preds_mu_phys[:, i].min())
    maxv = max(truth_phys[:, i].max(), preds_mu_phys[:, i].max())
    plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=2)

    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"{name}: prediction vs truth")
    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_true_uncertainty.png")
    plt.close()

# pull distributions
for i, name in enumerate(names):
    residuals = preds_mu_phys[:, i] - truth_phys[:, i]
    pull = residuals / (preds_sigma_phys[:, i] + 1e-12)
    plt.figure(figsize=(6, 4))
    plt.hist(pull, bins=40, density=True)
    x = np.linspace(-4, 4, 400)
    std_normal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)
    plt.plot(x, std_normal)
    plt.xlabel(f"Pull for {name}: (pred - true) / sigma")
    plt.ylabel("Density")
    plt.title(f"Pull distribution for {name}\nmean={np.mean(pull):.3f}, std={np.std(pull):.3f}")
    plt.tight_layout()
    plt.savefig(f"{name}_pull_distribution.png")
    plt.close()

print("\nDone! Saved plots to the current folder.")