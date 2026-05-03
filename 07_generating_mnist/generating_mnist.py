import os
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
except ImportError as error:
    raise ImportError(
        "Install the diffusion library with: pip install denoising_diffusion_pytorch"
    ) from error

# settings
script_dir = os.path.dirname(__file__)
dataset_path = os.path.join(script_dir, "dataset")
output_path = os.path.join(script_dir, "results")
os.makedirs(output_path, exist_ok=True)

batch_size = 128
epochs = 50
lr = 4e-4
image_size = 28
time_steps = 1000
sampling_timesteps = 250
sample_every = 5
sample_count = 16

dim = 32
dim_mults = (1, 2, 5)
cond_drop_prob = 0.1
cond_scale = 2.0

max_batches = None
max_validation_batches = None
seed = 123

# old stuff I tried
# batch_size = 64
# epochs = 30
# lr = 1e-4
# sampling_timesteps = 500
# cond_scale = 3.0
# cond_scale = 5.0  # made things worse honestly
# dim = 64
# dim_mults = (1, 2, 4)
# image_size = 32  # with padding, but I went back to 28

target_labels = torch.tensor(
    [
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 0, 1,
        2, 3, 4, 5,
    ],
    dtype=torch.long,
)

# maybe use a different fixed ordering?
# target_labels = torch.tensor(
#     [
#         0, 0, 1, 1,
#         2, 2, 3, 3,
#         4, 4, 5, 5,
#         6, 6, 7, 7,
#     ],
#     dtype=torch.long,
# )

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(value):
    torch.manual_seed(value)
    if device == "cuda":
        torch.cuda.manual_seed_all(value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(seed)

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),  # no, for this exercise [0,1] is the point
    # transforms.Pad(2),  # used when I briefly switched to 32x32
])

print("Loading MNIST...")

dataset = datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=transform,
    download=True,
)

validation_dataset = datasets.MNIST(
    root=dataset_path,
    train=False,
    transform=transform,
    download=True,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    # num_workers=2,
    # pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=2,
    # pin_memory=True,
)


# model
def make_model():
    model = Unet(
        dim=dim,
        num_classes=10,
        cond_drop_prob=cond_drop_prob,
        dim_mults=dim_mults,
        channels=1,
        # flash_attn=False,  # not using this path here
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=time_steps,
        sampling_timesteps=sampling_timesteps,
        # objective="pred_noise",
        # beta_schedule="cosine",
    )

    return diffusion.to(device)


def save_image_grid(images, labels, file_name):
    images = images.detach().cpu().clamp(0, 1)
    labels = labels.detach().cpu()

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(int(labels[i])), fontsize=10)
        ax.axis("off")

        # tried putting prediction/confidence here at one point,
        # but that was overcomplicating it
        # ax.set_xlabel(str(int(labels[i])), fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name), dpi=150)
    plt.close()


def save_sample_images(diffusion, step_name):
    diffusion.eval()
    labels = target_labels[:sample_count].to(device)

    with torch.no_grad():
        # I want the same grid every time for comparison, otherwise it's annoying
        with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
            set_seed(seed)

            # unconditional version:
            # samples = diffusion.sample(batch_size=sample_count)

            samples = diffusion.sample(
                classes=labels,
                cond_scale=cond_scale,
            )

            # tried this before but it didn't really solve much
            # samples = diffusion.sample(
            #     classes=labels,
            #     cond_scale=1.5,
            # )

        save_image_grid(samples, labels, f"samples_{step_name}.png")

    diffusion.train()


###############################
########## TRAINING ###########
###############################
def train_model():
    print(f"Training on {device}...")

    diffusion = make_model()

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses = []
    validation_losses = []

    save_sample_images(diffusion, "before_training")

    for epoch in range(epochs):
        diffusion.train()
        loss_sum = 0.0
        seen = 0

        for batch_idx, (real, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            real = real.to(device)
            labels = labels.to(device)

            loss = diffusion(real, classes=labels)

            optimizer.zero_grad()
            loss.backward()

            # maybe clip grads? didn't seem necessary
            # torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)

            optimizer.step()

            cur_batch_size = real.shape[0]
            loss_sum += loss.item() * cur_batch_size
            seen += cur_batch_size

            print(
                f"\rEpoch [{epoch + 1:02d}/{epochs}] Batch {batch_idx + 1}/{len(loader)} | "
                f"loss: {loss.item():.4f}",
                end="",
            )

        train_loss = loss_sum / seen
        train_losses.append(train_loss)

        diffusion.eval()
        validation_loss_sum = 0.0
        validation_seen = 0

        with torch.no_grad():
            for batch_idx, (real, labels) in enumerate(validation_loader):
                if max_validation_batches is not None and batch_idx >= max_validation_batches:
                    break

                real = real.to(device)
                labels = labels.to(device)

                loss = diffusion(real, classes=labels)

                cur_batch_size = real.shape[0]
                validation_loss_sum += loss.item() * cur_batch_size
                validation_seen += cur_batch_size

        validation_loss = validation_loss_sum / validation_seen
        validation_losses.append(validation_loss)

        print()
        print(
            f"Epoch {epoch + 1:02d}/{epochs} finished | "
            f"train loss: {train_loss:.4f} | validation loss: {validation_loss:.4f}"
        )

        # scheduler.step()

        # save every few epochs so I can actually see if it's learning anything
        if (epoch + 1) % sample_every == 0 or epoch == epochs - 1:
            save_sample_images(diffusion, f"epoch_{epoch + 1:02d}")

        # if epoch > 10 and validation_losses[-1] > validation_losses[-2]:
        #     print("validation got worse, maybe stop?")
        #     break

    return diffusion, train_losses, validation_losses


###############################
###### OUTPUT AND PLOTS #######
###############################
def save_plots(train_losses, validation_losses, diffusion):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DDPM training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_curve.png"), dpi=150)
    plt.close()

    # I already save checkpoint samples, so this felt redundant
    # save_sample_images(diffusion, "final")

    # also thought about saving a smoothed curve, but no
    # plt.figure(figsize=(8, 4))
    # plt.plot(train_losses, label="training")
    # plt.plot(validation_losses, label="validation")
    # plt.yscale("log")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_path, "loss_curve_log.png"), dpi=150)
    # plt.close()


###############################
############ MAIN #############
###############################
if __name__ == "__main__":
    diffusion, train_losses, validation_losses = train_model()
    save_plots(train_losses, validation_losses, diffusion)

    # save checkpoint?
    # torch.save(diffusion.state_dict(), os.path.join(output_path, "mnist_diffusion.pt"))

    print(f"\nDone! Saved results to {output_path}")