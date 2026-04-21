import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# settings
script_dir = os.path.dirname(__file__)
dataset_path = os.path.join(script_dir, "dataset")
output_path = os.path.join(script_dir, "results")
os.makedirs(output_path, exist_ok=True)

batch_size = 64
epochs = 30 #50 is overtraining
lr = 2e-4
latent_dim = 64
image_dim = 28 * 28
hidden_dim = 256
sample_every = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

print("Loading MNIST...")

dataset = datasets.MNIST(root=dataset_path, train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# models
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, image_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


def make_models():
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    return generator, discriminator


def save_sample_images(generator, fixed_noise, step_name):
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
        grid = make_grid(fake, nrow=4, normalize=True)
        save_image(grid, os.path.join(output_path, f"samples_{step_name}.png"))
    generator.train()


###############################
########## TRAINING ###########
###############################
def train_model():
    print("Start training...")

    generator, discriminator = make_models()
    opt_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(16, latent_dim, device=device)
    save_sample_images(generator, fixed_noise, "before_training")

    generator_losses = []
    discriminator_losses = []

    for epoch in range(epochs):
        generator_loss_sum = 0.0
        discriminator_loss_sum = 0.0

        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, image_dim).to(device)
            cur_batch_size = real.shape[0]

            real_labels = torch.ones(cur_batch_size, 1, device=device)
            fake_labels = torch.zeros(cur_batch_size, 1, device=device)

            # train discriminator
            noise = torch.randn(cur_batch_size, latent_dim, device=device)
            fake = generator(noise)

            disc_real = discriminator(real)
            loss_disc_real = criterion(disc_real, real_labels)

            disc_fake = discriminator(fake.detach())
            loss_disc_fake = criterion(disc_fake, fake_labels)

            loss_discriminator = 0.5 * (loss_disc_real + loss_disc_fake)

            opt_discriminator.zero_grad()
            loss_discriminator.backward()
            opt_discriminator.step()

            # train generator
            output = discriminator(fake)
            loss_generator = criterion(output, real_labels)

            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            generator_loss_sum += loss_generator.item() * cur_batch_size
            discriminator_loss_sum += loss_discriminator.item() * cur_batch_size

            print(
                f"\rEpoch [{epoch + 1:02d}/{epochs}] Batch {batch_idx + 1}/{len(loader)} | "
                f"loss discriminator: {loss_discriminator.item():.4f} | "
                f"loss generator: {loss_generator.item():.4f}",
                end="",
            )

        epoch_generator_loss = generator_loss_sum / len(loader.dataset)
        epoch_discriminator_loss = discriminator_loss_sum / len(loader.dataset)

        generator_losses.append(epoch_generator_loss)
        discriminator_losses.append(epoch_discriminator_loss)

        print()
        print(
            f"Epoch {epoch + 1:02d}/{epochs} finished | "
            f"mean discriminator loss: {epoch_discriminator_loss:.4f} | "
            f"mean generator loss: {epoch_generator_loss:.4f}"
        )

        if (epoch + 1) % sample_every == 0 or epoch == 0 or epoch == epochs - 1:
            save_sample_images(generator, fixed_noise, f"epoch_{epoch + 1:02d}")

    return generator, discriminator, generator_losses, discriminator_losses, fixed_noise


###############################
###### OUTPUT AND PLOTS #######
###############################
def save_plots(generator_losses, discriminator_losses, generator, fixed_noise):
    plt.figure(figsize=(8, 4))
    plt.plot(generator_losses, label="generator")
    plt.plot(discriminator_losses, label="discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_curve.png"), dpi=150)
    plt.close()

    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_noise).reshape(-1, 1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i, 0].cpu().numpy(), cmap="gray")
        ax.axis("off")

    plt.suptitle("Generated MNIST digits after training")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "final_generated_digits.png"), dpi=150)
    plt.close()


###############################
############ MAIN #############
###############################
generator, discriminator, generator_losses, discriminator_losses, fixed_noise = train_model()

save_plots(generator_losses, discriminator_losses, generator, fixed_noise)

print(f"\nDone! Saved results to {output_path}")