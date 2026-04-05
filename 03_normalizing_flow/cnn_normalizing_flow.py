import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import jammy_flows

# settings
data_path = "data"
device = "cuda" # if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 30
lr = 1e-3
seed = 42
n_samples_eval = 300
n_samples_pdf_plot = 5000

# load data
spectra = np.load(f"{data_path}/spectra.npy").astype(np.float32)
labels = np.load(f"{data_path}/labels.npy").astype(np.float32)
output_path = "results"

# keep only t_eff, log_g, fe_h
labels = labels[:, -4:-1]

# labels and units for plots
label_keys = ["t_eff", "log_g", "fe_h"]
label_display_names = ["T_eff", "log g", "[Fe/H]"]
label_units = ["K", "dex", "dex"]
density_units = ["K^-1", "dex^-1", "dex^-1"]

# normalize spectra
spectra = np.log(np.maximum(spectra, 0.2))

# split data: 70% train, 15% val, 15% test
n = len(spectra)
indices = np.random.permutation(n)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

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
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

os.makedirs(output_path, exist_ok=True)

# 1D CNN that predicts flow parameters
def make_encoder(n_flow_params):
    encoder = nn.Sequential(
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
        nn.Linear(64, n_flow_params),
    )
    return encoder


def format_value_with_unit(value, unit, decimals=3):
    return f"{value:.{decimals}f} {unit}"


###############################
########## TRAINING ###########
###############################
def train_model(flow_type):
    print(f"Training {flow_type}")

    options = {}
    options["t"] = {}
    options["g"] = {}

    if flow_type == "diagonal_gaussian":
        options["t"]["cov_type"] = "diagonal"
        flow_defs = "t"

    elif flow_type == "full_gaussian":
        options["t"]["cov_type"] = "full"
        flow_defs = "t"

    elif flow_type == "full_flow":
        options["t"]["cov_type"] = "full"
        flow_defs = "gggt"
        options["g"]["fit_normalization"] = 1
        options["g"]["upper_bound_for_widths"] = 1.0
        options["g"]["lower_bound_for_widths"] = 0.01

    else:
        raise ValueError("Unknown flow_type")

    pdf = jammy_flows.pdf(
        "e3",
        flow_defs,
        options_overwrite=options,
        amortize_everything=True,
        amortization_mlp_use_custom_mode=True,
    )

    n_flow_params = pdf.total_number_amortizable_params
    encoder = make_encoder(n_flow_params)

    encoder = encoder.to(device)
    pdf = pdf.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(pdf.parameters()),
        lr=lr,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_encoder_state = None
    best_pdf_state = None

    for epoch in range(epochs):
        encoder.train()
        pdf.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            flow_params = encoder(xb)

            if flow_type == "full_flow":
                flow_params = flow_params.to(torch.float64)
                yb = yb.to(torch.float64)

            log_pdf_values, _, _ = pdf(yb, amortization_parameters=flow_params)
            loss = -log_pdf_values.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        train_losses.append(train_loss)

        encoder.eval()
        pdf.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                flow_params = encoder(xb)

                if flow_type == "full_flow":
                    flow_params = flow_params.to(torch.float64)
                    yb = yb.to(torch.float64)

                log_pdf_values, _, _ = pdf(yb, amortization_parameters=flow_params)
                loss = -log_pdf_values.mean()

                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_pdf_state = copy.deepcopy(pdf.state_dict())

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train NLL: {train_loss:.4f} | "
            f"val NLL: {val_loss:.4f}"
        )

    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
    if best_pdf_state is not None:
        pdf.load_state_dict(best_pdf_state)

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title(f"{flow_type} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{flow_type}_loss.png"))
    plt.close()

    return encoder, pdf


###############################
###### OUTPUT AND PLOTS #######
###############################
def visualize_full_flow_pdfs(encoder, pdf, n_examples=3, n_samples=5000):
    print("\nSaving example PDFs from the Gaussianization flow...")

    encoder.eval()
    pdf.eval()

    example_indices = np.arange(min(n_examples, len(X_test)))

    with torch.no_grad():
        for plot_number, idx in enumerate(example_indices):
            xb = X_test[idx:idx + 1].to(device)
            y_true = y_test[idx].numpy() * y_std + y_mean

            flow_params = encoder(xb)
            flow_params = flow_params.to(torch.float64)

            repeated_params = flow_params.repeat_interleave(n_samples, dim=0)

            samples, _, _, _ = pdf.sample(
                amortization_parameters=repeated_params,
                allow_gradients=False,
            )

            samples = samples.view(1, n_samples, 3)
            samples = samples.squeeze(0).cpu().numpy()
            samples = samples * y_std + y_mean

            fig, axes = plt.subplots(3, 1, figsize=(8, 9))

            for dim in range(3):
                mu = samples[:, dim].mean()
                sigma = samples[:, dim].std()

                axes[dim].hist(samples[:, dim], bins=50, density=True, alpha=0.6)
                axes[dim].axvline(y_true[dim], color="red", label="true value")
                axes[dim].axvline(mu, color="green", label="predicted mean")
                axes[dim].set_title(f"{label_display_names[dim]} [{label_units[dim]}]")
                axes[dim].set_xlabel(f"{label_display_names[dim]} [{label_units[dim]}]")
                axes[dim].set_ylabel(f"Probability density [{density_units[dim]}]")

                if dim == 0:
                    axes[dim].legend()

                text = (
                    f"mean={format_value_with_unit(mu, label_units[dim])}\n"
                    f"std={format_value_with_unit(sigma, label_units[dim])}\n"
                    f"true={format_value_with_unit(y_true[dim], label_units[dim])}"
                )
                axes[dim].text(
                    0.98,
                    0.95,
                    text,
                    transform=axes[dim].transAxes,
                    ha="right",
                    va="top",
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"full_flow_pdf_example_{plot_number}.png"))
            plt.close()


###############################
############ MAIN #############
###############################

# diagonal gaussian
diagonal_encoder, diagonal_pdf = train_model("diagonal_gaussian")

print("Evaluating diagonal_gaussian")

diagonal_encoder.eval()
diagonal_pdf.eval()

diagonal_all_means = []
diagonal_all_stds = []
diagonal_all_truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        flow_params = diagonal_encoder(xb)

        batch_size_here = flow_params.shape[0]
        repeated_params = flow_params.repeat_interleave(n_samples_eval, dim=0)

        samples, _, _, _ = diagonal_pdf.sample(
            amortization_parameters=repeated_params,
            allow_gradients=False,
        )

        samples = samples.view(batch_size_here, n_samples_eval, 3)
        means = samples.mean(dim=1)
        stds = samples.std(dim=1)

        diagonal_all_means.append(means.cpu().numpy())
        diagonal_all_stds.append(stds.cpu().numpy())
        diagonal_all_truth.append(yb.numpy())

diagonal_pred_mean = np.vstack(diagonal_all_means)
diagonal_pred_std = np.vstack(diagonal_all_stds)
diagonal_truth = np.vstack(diagonal_all_truth)

diagonal_pred_mean_phys = diagonal_pred_mean * y_std + y_mean
diagonal_truth_phys = diagonal_truth * y_std + y_mean
diagonal_pred_std_phys = diagonal_pred_std * y_std

rng = np.random.default_rng(seed)

for i, key in enumerate(label_keys):
    plt.figure(figsize=(5, 5))

    n_points = len(diagonal_truth_phys)
    n_sample = min(100, n_points)
    sample = rng.choice(n_points, size=n_sample, replace=False)

    mask = np.ones(n_points, dtype=bool)
    mask[sample] = False

    plt.scatter(
        diagonal_truth_phys[mask, i],
        diagonal_pred_mean_phys[mask, i],
        s=10,
        color="#7fb3d5",
    )

    plt.scatter(
        diagonal_truth_phys[sample, i],
        diagonal_pred_mean_phys[sample, i],
        s=14,
        color="#1f4e79",
    )

    plt.errorbar(
        diagonal_truth_phys[sample, i],
        diagonal_pred_mean_phys[sample, i],
        yerr=diagonal_pred_std_phys[sample, i],
        fmt="none",
        ecolor="#1f4e79",
        alpha=0.35,
        elinewidth=1,
        capsize=2,
    )

    minv = min(diagonal_truth_phys[:, i].min(), diagonal_pred_mean_phys[:, i].min())
    maxv = max(diagonal_truth_phys[:, i].max(), diagonal_pred_mean_phys[:, i].max())
    plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=2)

    plt.xlabel(f"True {label_display_names[i]} [{label_units[i]}]")
    plt.ylabel(f"Predicted {label_display_names[i]} [{label_units[i]}]")
    plt.title(f"diagonal_gaussian: {label_display_names[i]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"diagonal_gaussian_{key}_pred_vs_true.png"))
    plt.close()

    residuals = diagonal_pred_mean_phys[:, i] - diagonal_truth_phys[:, i]
    sigma = diagonal_pred_std_phys[:, i] + 1e-12
    pull = residuals / sigma

    plt.figure(figsize=(6, 4))
    plt.hist(pull, bins=40, density=True, alpha=0.7)

    x = np.linspace(-4, 4, 400)
    normal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
    plt.plot(x, normal)

    plt.xlabel(f"Pull for {label_display_names[i]} [(prediction - true) / sigma]")
    plt.ylabel("Probability density")
    plt.title(f"diagonal_gaussian: {label_display_names[i]} pull")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"diagonal_gaussian_{key}_pull.png"))
    plt.close()


# full gaussian
full_gaussian_encoder, full_gaussian_pdf = train_model("full_gaussian")

print("Evaluating full_gaussian")

full_gaussian_encoder.eval()
full_gaussian_pdf.eval()

full_gaussian_all_means = []
full_gaussian_all_stds = []
full_gaussian_all_truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        flow_params = full_gaussian_encoder(xb)

        batch_size_here = flow_params.shape[0]
        repeated_params = flow_params.repeat_interleave(n_samples_eval, dim=0)

        samples, _, _, _ = full_gaussian_pdf.sample(
            amortization_parameters=repeated_params,
            allow_gradients=False,
        )

        samples = samples.view(batch_size_here, n_samples_eval, 3)
        means = samples.mean(dim=1)
        stds = samples.std(dim=1)

        full_gaussian_all_means.append(means.cpu().numpy())
        full_gaussian_all_stds.append(stds.cpu().numpy())
        full_gaussian_all_truth.append(yb.numpy())

full_gaussian_pred_mean = np.vstack(full_gaussian_all_means)
full_gaussian_pred_std = np.vstack(full_gaussian_all_stds)
full_gaussian_truth = np.vstack(full_gaussian_all_truth)

full_gaussian_pred_mean_phys = full_gaussian_pred_mean * y_std + y_mean
full_gaussian_truth_phys = full_gaussian_truth * y_std + y_mean
full_gaussian_pred_std_phys = full_gaussian_pred_std * y_std

rng = np.random.default_rng(seed)

for i, key in enumerate(label_keys):
    plt.figure(figsize=(5, 5))

    n_points = len(full_gaussian_truth_phys)
    n_sample = min(100, n_points)
    sample = rng.choice(n_points, size=n_sample, replace=False)

    mask = np.ones(n_points, dtype=bool)
    mask[sample] = False

    plt.scatter(
        full_gaussian_truth_phys[mask, i],
        full_gaussian_pred_mean_phys[mask, i],
        s=10,
        color="#7fb3d5",
    )

    plt.scatter(
        full_gaussian_truth_phys[sample, i],
        full_gaussian_pred_mean_phys[sample, i],
        s=14,
        color="#1f4e79",
    )

    plt.errorbar(
        full_gaussian_truth_phys[sample, i],
        full_gaussian_pred_mean_phys[sample, i],
        yerr=full_gaussian_pred_std_phys[sample, i],
        fmt="none",
        ecolor="#1f4e79",
        alpha=0.35,
        elinewidth=1,
        capsize=2,
    )

    minv = min(full_gaussian_truth_phys[:, i].min(), full_gaussian_pred_mean_phys[:, i].min())
    maxv = max(full_gaussian_truth_phys[:, i].max(), full_gaussian_pred_mean_phys[:, i].max())
    plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=2)

    plt.xlabel(f"True {label_display_names[i]} [{label_units[i]}]")
    plt.ylabel(f"Predicted {label_display_names[i]} [{label_units[i]}]")
    plt.title(f"full_gaussian: {label_display_names[i]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"full_gaussian_{key}_pred_vs_true.png"))
    plt.close()

    residuals = full_gaussian_pred_mean_phys[:, i] - full_gaussian_truth_phys[:, i]
    sigma = full_gaussian_pred_std_phys[:, i] + 1e-12
    pull = residuals / sigma

    plt.figure(figsize=(6, 4))
    plt.hist(pull, bins=40, density=True, alpha=0.7)

    x = np.linspace(-4, 4, 400)
    normal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
    plt.plot(x, normal)

    plt.xlabel(f"Pull for {label_display_names[i]} [(prediction - true) / sigma]")
    plt.ylabel("Probability density")
    plt.title(f"full_gaussian: {label_display_names[i]} pull")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"full_gaussian_{key}_pull.png"))
    plt.close()


# full flow
full_flow_encoder, full_flow_pdf = train_model("full_flow")

print("Evaluating full_flow")

full_flow_encoder.eval()
full_flow_pdf.eval()

full_flow_all_means = []
full_flow_all_stds = []
full_flow_all_truth = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        flow_params = full_flow_encoder(xb)
        flow_params = flow_params.to(torch.float64)

        batch_size_here = flow_params.shape[0]
        repeated_params = flow_params.repeat_interleave(n_samples_eval, dim=0)

        samples, _, _, _ = full_flow_pdf.sample(
            amortization_parameters=repeated_params,
            allow_gradients=False,
        )

        samples = samples.view(batch_size_here, n_samples_eval, 3)
        means = samples.mean(dim=1)
        stds = samples.std(dim=1)

        full_flow_all_means.append(means.cpu().numpy())
        full_flow_all_stds.append(stds.cpu().numpy())
        full_flow_all_truth.append(yb.numpy())

full_flow_pred_mean = np.vstack(full_flow_all_means)
full_flow_pred_std = np.vstack(full_flow_all_stds)
full_flow_truth = np.vstack(full_flow_all_truth)

full_flow_pred_mean_phys = full_flow_pred_mean * y_std + y_mean
full_flow_truth_phys = full_flow_truth * y_std + y_mean
full_flow_pred_std_phys = full_flow_pred_std * y_std

rng = np.random.default_rng(seed)

for i, key in enumerate(label_keys):
    plt.figure(figsize=(5, 5))

    n_points = len(full_flow_truth_phys)
    n_sample = min(100, n_points)
    sample = rng.choice(n_points, size=n_sample, replace=False)

    mask = np.ones(n_points, dtype=bool)
    mask[sample] = False

    plt.scatter(
        full_flow_truth_phys[mask, i],
        full_flow_pred_mean_phys[mask, i],
        s=10,
        color="#7fb3d5",
    )

    plt.scatter(
        full_flow_truth_phys[sample, i],
        full_flow_pred_mean_phys[sample, i],
        s=14,
        color="#1f4e79",
    )

    plt.errorbar(
        full_flow_truth_phys[sample, i],
        full_flow_pred_mean_phys[sample, i],
        yerr=full_flow_pred_std_phys[sample, i],
        fmt="none",
        ecolor="#1f4e79",
        alpha=0.35,
        elinewidth=1,
        capsize=2,
    )

    minv = min(full_flow_truth_phys[:, i].min(), full_flow_pred_mean_phys[:, i].min())
    maxv = max(full_flow_truth_phys[:, i].max(), full_flow_pred_mean_phys[:, i].max())
    plt.plot([minv, maxv], [minv, maxv], color="red", linewidth=2)

    plt.xlabel(f"True {label_display_names[i]} [{label_units[i]}]")
    plt.ylabel(f"Predicted {label_display_names[i]} [{label_units[i]}]")
    plt.title(f"full_flow: {label_display_names[i]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"full_flow_{key}_pred_vs_true.png"))
    plt.close()

    residuals = full_flow_pred_mean_phys[:, i] - full_flow_truth_phys[:, i]
    sigma = full_flow_pred_std_phys[:, i] + 1e-12
    pull = residuals / sigma

    plt.figure(figsize=(6, 4))
    plt.hist(pull, bins=40, density=True, alpha=0.7)

    x = np.linspace(-4, 4, 400)
    normal = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)
    plt.plot(x, normal)

    plt.xlabel(f"Pull for {label_display_names[i]} [(prediction - true) / sigma]")
    plt.ylabel("Probability density")
    plt.title(f"full_flow: {label_display_names[i]} pull")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"full_flow_{key}_pull.png"))
    plt.close()

visualize_full_flow_pdfs(
    full_flow_encoder,
    full_flow_pdf,
    n_examples=3,
    n_samples=n_samples_pdf_plot,
)

print(f"\nDone! Saved plots to {output_path}")
