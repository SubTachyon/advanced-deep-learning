import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# settings
os.makedirs("results", exist_ok=True)

batch_size = 64
epochs = 30
lr = 1e-3
hidden_dim = 128
num_heads = 4 # at least 2 heads for the exercise
num_layers = 2 # at least 2 transformer layers for the exercise
feedforward_dim = 256
dropout = 0.02

# set to smaller numbers while debugging, then back to None
max_train_events = None
max_val_events = None
max_test_events = None

seed = 123

device = "cuda" if torch.cuda.is_available() else "cpu"

# labels for plots, kept from the GNN exercise
coord_keys = ["x", "y"]
coord_display_names = ["x position", "y position"]


def set_seed(value):
    torch.manual_seed(value)
    np.random.seed(value)
    if device == "cuda":
        torch.cuda.manual_seed_all(value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(seed)


# load parquet and turn each event into one variable-length photon sequence
class NeutrinoDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_file, feature_mean=None, feature_std=None, max_events=None):
        df = pd.read_parquet(parquet_file)

        if max_events is not None:
            df = df.iloc[:max_events]

        self.events = []
        self.labels = []

        # compute normalization stats from the training set only
        if feature_mean is None or feature_std is None:
            feature_sum = torch.zeros(3, dtype=torch.float32)
            feature_sq_sum = torch.zeros(3, dtype=torch.float32)
            total_photons = 0

            for _, row in df.iterrows():
                times = torch.tensor(np.asarray(row["data"][0]), dtype=torch.float32)
                xpos = torch.tensor(np.asarray(row["data"][1]), dtype=torch.float32)
                ypos = torch.tensor(np.asarray(row["data"][2]), dtype=torch.float32)

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
            times = torch.tensor(np.asarray(row["data"][0]), dtype=torch.float32).view(-1, 1)
            xpos = torch.tensor(np.asarray(row["data"][1]), dtype=torch.float32).view(-1, 1)
            ypos = torch.tensor(np.asarray(row["data"][2]), dtype=torch.float32).view(-1, 1)

            # photon features: time and detector position
            x = torch.cat([times, xpos, ypos], dim=1)
            x = (x - self.feature_mean) / self.feature_std

            # I sort by time so the input order is at least deterministic.
            # The transformer still gets the real hit time as a normal feature.
            order = torch.argsort(x[:, 0])
            x = x[order]

            y = torch.tensor([row["xpos"], row["ypos"]], dtype=torch.float32)

            self.events.append(x)
            self.labels.append(y)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return {
            "data": self.events[idx],
            "label": self.labels[idx],
        }


def collate_fn_transformer(batch):
    """
    Custom collate function from the transformer template idea.

    The events have different numbers of photons, so first concatenate all hits
    into one tensor and remember the length of each event. The transformer pads
    them back to equal length inside the forward pass.
    """
    data_list = []
    labels = []
    lengths = []

    for b in batch:
        tensordata = b["data"]

        lengths.append(tensordata.shape[0])
        data_list.append(tensordata)
        labels.append(b["label"].unsqueeze(0))

    labels = torch.cat(labels, dim=0)
    data_vec = torch.cat(data_list, dim=0)

    return [data_vec, lengths], labels


print(f"Using device: {device}")
print("Loading data...")

train_dataset = NeutrinoDataset("data/train.pq", max_events=max_train_events)
val_dataset = NeutrinoDataset(
    "data/val.pq",
    train_dataset.feature_mean,
    train_dataset.feature_std,
    max_events=max_val_events,
)
test_dataset = NeutrinoDataset(
    "data/test.pq",
    train_dataset.feature_mean,
    train_dataset.feature_std,
    max_events=max_test_events,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_transformer,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_transformer,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_transformer,
)


# Transformer model that predicts neutrino interaction positions
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        # input embedding: 3 photon features -> hidden transformer dimension
        self.input_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            activation="relu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data):
        src, lengths = data

        # 1) embed the input data into the hidden dimension
        src = self.input_embedding(src)

        # 2) split the data back into a list of tensors, one for each event
        parts = src.split(lengths, dim=0)

        # 3) pad inputs with zeros so all batch items have the same length
        padded = pad_sequence(parts, batch_first=True)
        batch_size_now, max_len, _ = padded.shape

        # 4) build padding mask: True = padding token, False = real photon hit
        mask = torch.zeros(batch_size_now, max_len, dtype=torch.bool, device=padded.device)
        for i, length in enumerate(lengths):
            mask[i, length:] = True

        # 5) run transformer encoder while ignoring padding tokens
        enc_out = self.encoder(padded, src_key_padding_mask=mask)

        # 6) masked mean pooling over photon hits to get one vector per event
        valid_mask = ~mask
        summed = (enc_out * valid_mask.unsqueeze(-1)).sum(dim=1)
        pooled = summed / torch.tensor(lengths, device=enc_out.device, dtype=enc_out.dtype).view(-1, 1)

        # 7) predict x/y position
        return self.mlp(pooled)


def make_model():
    model = SimpleTransformer()
    return model


def move_batch_to_device(batch):
    data, labels = batch
    data_vec, lengths = data
    data = [data_vec.to(device), lengths]
    labels = labels.to(device)
    return data, labels


###############################
########## TRAINING ###########
###############################
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    train_loss_sum = 0.0

    for batch_idx, batch in enumerate(loader):
        data, labels = move_batch_to_device(batch)

        pred = model(data)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_batch_size = labels.shape[0]
        train_loss_sum += loss.item() * cur_batch_size

        print(
            f"\rBatch {batch_idx + 1}/{len(loader)} | loss: {loss.item():.4f}",
            end="",
        )

    print()
    train_loss = train_loss_sum / len(loader.dataset)
    return train_loss


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    loss_sum = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        data, labels = move_batch_to_device(batch)

        pred = model(data)
        loss = loss_fn(pred, labels)

        loss_sum += loss.item() * labels.shape[0]
        all_preds.append(pred.cpu())
        all_targets.append(labels.cpu())

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
            f"Epoch {epoch + 1:02d}/{epochs} finished | "
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
    plt.title("transformer loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(test_targets[:, 0], test_targets[:, 1], s=8, alpha=0.5, label="true")
    plt.scatter(test_preds[:, 0], test_preds[:, 1], s=8, alpha=0.5, label="pred")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("True vs predicted neutrino positions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/test_predictions.png", dpi=150)
    plt.close()

    residuals = test_preds - test_targets
    position_errors = np.linalg.norm(residuals, axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(position_errors, bins=50, alpha=0.8)
    plt.xlabel("position error")
    plt.ylabel("count")
    plt.title("Test position error")
    plt.tight_layout()
    plt.savefig("results/position_error_histogram.png", dpi=150)
    plt.close()

    # trying to visualize error/shift, copied from the GNN exercise idea
    dx = residuals[:, 0]
    dy = residuals[:, 1]
    r = np.linalg.norm(test_targets, axis=1)

    plt.figure(figsize=(6, 4))
    plt.scatter(r, dx, s=6, alpha=0.3)
    plt.xlabel("true radius")
    plt.ylabel("x residual")
    plt.title("X residual vs radius")
    plt.tight_layout()
    plt.savefig("results/x_bias_vs_radius.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(r, dy, s=6, alpha=0.3)
    plt.xlabel("true radius")
    plt.ylabel("y residual")
    plt.title("Y residual vs radius")
    plt.tight_layout()
    plt.savefig("results/y_bias_vs_radius.png", dpi=150)
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
        plt.savefig(f"results/pred_vs_true_{key}.png", dpi=150)
        plt.close()


###############################
############ MAIN #############
###############################
if __name__ == "__main__":
    model, loss_fn, train_losses, val_losses = train_model()

    print("\nEvaluating test set")

    test_loss, test_mae, test_pos_error, test_preds, test_targets = evaluate(model, test_loader, loss_fn)

    print("\nTest results")
    print(f"test loss = {test_loss:.4f}")
    print(f"test MAE = {test_mae:.4f}")
    print(f"test mean position error = {test_pos_error:.4f}")

    save_plots(train_losses, val_losses, test_preds, test_targets)

    print("\nDooooooooooooone the whoooole cooooourssse!")
