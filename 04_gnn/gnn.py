
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool

# settings
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "data")
train_file = os.path.join(data_path, "train.pq")
val_file = os.path.join(data_path, "val.pq")
test_file = os.path.join(data_path, "test.pq")
output_path = os.path.join(script_dir, "results")

batch_size = 64
epochs = 30  # after ~20 val loss was still improving slightly
lr = 1e-3
k = 8  # tried 5 first, 8 worked a bit better

device = "cuda" if torch.cuda.is_available() else "cpu"

# labels for plots
coord_keys = ["x", "y"]
coord_display_names = ["x position", "y position"]


# load parquet and turn each event into one graph
class NeutrinoDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_file, feature_mean=None, feature_std=None):
        df = pd.read_parquet(parquet_file)
        self.graphs = []

        # compute normalization stats from the training set only
        if feature_mean is None or feature_std is None:
            feature_sum = torch.zeros(3, dtype=torch.float32)
            feature_sq_sum = torch.zeros(3, dtype=torch.float32)
            total_photons = 0

            for _, row in df.iterrows():
                times = torch.tensor(row["data"][0], dtype=torch.float32)
                xpos = torch.tensor(row["data"][1], dtype=torch.float32)
                ypos = torch.tensor(row["data"][2], dtype=torch.float32)

                features = torch.stack([times, xpos, ypos], dim=1)
                feature_sum += features.sum(dim=0)
                feature_sq_sum += (features ** 2).sum(dim=0)
                total_photons += features.size(0)

            self.feature_mean = feature_sum / total_photons
            self.feature_std = torch.sqrt(feature_sq_sum / total_photons - self.feature_mean ** 2)

            # avoid division by zero
            self.feature_std[self.feature_std == 0] = 1.0
        else:
            self.feature_mean = feature_mean
            self.feature_std = feature_std

        for _, row in df.iterrows():
            times = torch.tensor(row["data"][0], dtype=torch.float32).view(-1, 1)
            xpos = torch.tensor(row["data"][1], dtype=torch.float32).view(-1, 1)
            ypos = torch.tensor(row["data"][2], dtype=torch.float32).view(-1, 1)

            # photon features: time and detector position
            x = torch.cat([times, xpos, ypos], dim=1)
            x = (x - self.feature_mean) / self.feature_std

            # DynamicEdgeConv needs at least k + 1 nodes
            while x.size(0) < k + 1:
                x = torch.cat([x, x[-1:].clone()], dim=0)

            y = torch.tensor([[row["xpos"], row["ypos"]]], dtype=torch.float32)
            self.graphs.append(Data(x=x, y=y))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


print(f"Using device: {device}")
print("Loading data...")

train_dataset = NeutrinoDataset(train_file)
val_dataset = NeutrinoDataset(val_file, train_dataset.feature_mean, train_dataset.feature_std)
test_dataset = NeutrinoDataset(test_file, train_dataset.feature_mean, train_dataset.feature_std)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

os.makedirs(output_path, exist_ok=True)


# GNN model that predicts neutrino interaction positions
class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()

        hidden = 128

        self.conv1 = DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(6, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ),
            k=k,
            aggr="mean",
        )

        self.conv2 = DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ),
            k=k,
            aggr="mean",
        )

        self.conv3 = DynamicEdgeConv(
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            ),
            k=k,
            aggr="mean",
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)

        # keep features from all graph layers
        x = torch.cat([x1, x2, x3], dim=1)
        x = global_mean_pool(x, batch)

        return self.mlp(x)


def make_model():
    model = SimpleGNN()
    return model


###############################
########## TRAINING ###########
###############################
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    train_loss_sum = 0.0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)
        loss = loss_fn(pred, batch.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * batch.num_graphs

    train_loss = train_loss_sum / len(loader.dataset)
    return train_loss


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)
        loss = loss_fn(pred, batch.y)

        loss_sum += loss.item() * batch.num_graphs
        all_preds.append(pred.cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    mae = np.mean(np.abs(preds - targets))
    position_error = np.mean(np.linalg.norm(preds - targets, axis=1))
    mean_loss = loss_sum / len(loader.dataset)

    return mean_loss, mae, position_error, preds, targets


def train_model():
    print("Start training...")

    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_mae, val_pos_error, _, _ = evaluate(model, val_loader, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train MSE: {train_loss:.4f} | "
            f"val MSE: {val_loss:.4f} | "
            f"val MAE: {val_mae:.4f} | "
            f"val mean position error: {val_pos_error:.4f}"
        )

    return model, loss_fn, train_losses, val_losses


###############################
###### OUTPUT AND PLOTS #######
###############################
def save_plots(train_losses, val_losses, test_preds, test_targets):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("gnn loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(test_targets[:, 0], test_targets[:, 1], s=8, alpha=0.5, label="true")
    plt.scatter(test_preds[:, 0], test_preds[:, 1], s=8, alpha=0.5, label="pred")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("True vs predicted neutrino positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "test_predictions.png"), dpi=150)
    plt.close()

    # trying to visualize error/shift
    dx = test_preds[:, 0] - test_targets[:, 0]
    dy = test_preds[:, 1] - test_targets[:, 1]
    r = np.linalg.norm(test_targets, axis=1)

    plt.figure(figsize=(6, 4))
    plt.scatter(r, dx, s=6, alpha=0.3)
    plt.xlabel("true radius")
    plt.ylabel("x residual")
    plt.title("X residual vs radius")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "x_bias_vs_radius.png"), dpi=150)
    plt.close()

    # left/right bias?
    dx = test_preds[:, 0] - test_targets[:, 0]

    plt.figure(figsize=(6, 4))
    plt.scatter(test_targets[:, 0], dx, s=6, alpha=0.3)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("true x")
    plt.ylabel("x residual")
    plt.title("X residual vs true x")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "x_residual_vs_true_x.png"), dpi=150)
    plt.close()

    for i, key in enumerate(coord_keys):
        plt.figure(figsize=(6, 6))
        plt.scatter(test_targets[:, i], test_preds[:, i], s=8, alpha=0.3, zorder=1)

        minv = min(test_targets[:, i].min(), test_preds[:, i].min())
        maxv = max(test_targets[:, i].max(), test_preds[:, i].max())

        plt.plot([minv, maxv], [minv, maxv], linestyle="--", color="red", zorder=10)
        plt.xlabel(f"true {coord_display_names[i]}")
        plt.ylabel(f"predicted {coord_display_names[i]}")
        plt.title(f"Predicted vs true {key}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"pred_vs_true_{key}.png"), dpi=150)
        plt.close()


###############################
############ MAIN #############
###############################
model, loss_fn, train_losses, val_losses = train_model()

print("\nEvaluating test set")

test_loss, test_mae, test_pos_error, test_preds, test_targets = evaluate(model, test_loader, loss_fn)

print("\nTest results")
print(f"test loss = {test_loss:.4f}")
print(f"test MAE = {test_mae:.4f}")
print(f"test mean position error = {test_pos_error:.4f}")

save_plots(train_losses, val_losses, test_preds, test_targets)

print(f"\nDone! Saved plots to {output_path}")
