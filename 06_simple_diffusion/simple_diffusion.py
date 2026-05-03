import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# settings
script_dir = os.path.dirname(__file__)
output_path = os.path.join(script_dir, "results")
os.makedirs(output_path, exist_ok=True)

batch_size = 64
epochs = 50
lr = 0.8e-4
hidden_dim = 64
time_steps = 250
beta = 0.02
sample_count = 1000
sample_every = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

alpha = 1.0 - beta
beta = torch.tensor(beta, device=device)
alpha = torch.tensor(alpha, device=device)
alpha_bar = alpha ** torch.arange(time_steps + 1, device=device)

# data
data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([1.0, 2.0])),
    torch.distributions.Normal(torch.tensor([-4.0, 4.0]), torch.tensor([1.0, 1.0])),
)

dataset = data_distribution.sample(torch.Size([10000])).to(device)
dataset_validation = data_distribution.sample(torch.Size([1000])).to(device)


# model
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, t):
        t = t.float() / time_steps
        x = torch.stack([x, t], dim=1)
        return self.net(x).squeeze(1)


def make_noisy_sample(x0, t):
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bar[t]
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return xt, noise


def sample_reverse(model, count, save_history=False):
    model.eval()
    x = torch.randn(count, device=device)
    history = []

    if save_history:
        history.append(x.detach().cpu())

    with torch.no_grad():
        for step in range(time_steps, 0, -1):
            t = torch.full((count,), step, device=device, dtype=torch.long)
            predicted_noise = model(x, t)

            mean = (1.0 / torch.sqrt(alpha)) * (
                x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar[step])) * predicted_noise
            )

            if step > 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = mean + torch.sqrt(beta) * z

            if save_history and (step % 25 == 0 or step == 1):
                history.append(x.detach().cpu())

    if save_history:
        return x, history
    return x


###############################
########## TRAINING ###########
###############################
def train_model():
    print(f"Training on {device}...")

    model = DiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    validation_losses = []

    progress = tqdm(range(epochs))
    for epoch in progress:
        model.train()
        indices = torch.randperm(dataset.shape[0], device=device)
        shuffled_dataset = dataset[indices]

        loss_sum = 0.0
        seen = 0

        for i in range(0, shuffled_dataset.shape[0] - batch_size, batch_size):
            x0 = shuffled_dataset[i:i + batch_size]
            t = torch.randint(1, time_steps + 1, (batch_size,), device=device)

            xt, noise = make_noisy_sample(x0, t)
            predicted_noise = model(xt, t)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * batch_size
            seen += batch_size

        train_loss = loss_sum / seen
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            x0 = dataset_validation
            t = torch.randint(1, time_steps + 1, (x0.shape[0],), device=device)
            xt, noise = make_noisy_sample(x0, t)
            predicted_noise = model(xt, t)
            validation_loss = criterion(predicted_noise, noise).item()
            validation_losses.append(validation_loss)

        progress.set_description(
            f"epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | val {validation_loss:.4f}"
        )

        if (epoch + 1) % sample_every == 0:
            samples = sample_reverse(model, sample_count).detach().cpu().numpy()
            save_distribution_plot(samples, f"samples_epoch_{epoch + 1:04d}.png")

    return model, train_losses, validation_losses


###############################
###### OUTPUT AND PLOTS #######
###############################
def save_distribution_plot(samples, file_name):
    bins = np.linspace(-10, 10, 50)
    data = dataset.detach().cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, density=True, histtype="step", linewidth=2, label="true distribution")
    plt.hist(samples, bins=bins, density=True, alpha=0.6, label="sampled distribution")
    plt.xlabel("sample value")
    plt.ylabel("density")
    plt.title("Diffusion model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), dpi=150)
    plt.close()


def save_plots(model, train_losses, validation_losses):
    samples, history = sample_reverse(model, sample_count, save_history=True)
    samples = samples.detach().cpu().numpy()

    save_distribution_plot(samples, "final_distribution.png")

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("mse loss")
    plt.title("noise prediction loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_curve.png"), dpi=150)
    plt.close()

    bins = np.linspace(-10, 10, 50)
    plt.figure(figsize=(10, 6))
    for i in [0, 2, 4, 6, 8, len(history) - 1]:
        plt.hist(history[i].numpy(), bins=bins, density=True, histtype="step", linewidth=1.5, label=f"stage {i}")
    plt.xlabel("sample value")
    plt.ylabel("density")
    plt.title("distribution during reverse diffusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "diffusion_stages.png"), dpi=150)
    plt.close()


###############################
############ MAIN #############
###############################
model, train_losses, validation_losses = train_model()
save_plots(model, train_losses, validation_losses)

print(f"Done! Saved results to {output_path}")
