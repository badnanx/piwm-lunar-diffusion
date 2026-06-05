import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.dataset import LunarFrameDataset
from lunar_diffusion.autoencoder import ConvVAE, vae_loss


def save_reconstruction_grid(images, recon, save_path, num_images=8):
    images = images[:num_images].detach().cpu()
    recon = recon[:num_images].detach().cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))

    for i in range(num_images):
        axes[0, i].imshow(images[i].permute(1, 2, 0))
        axes[0, i].axis("off")
        axes[0, i].set_title("real")

        axes[1, i].imshow(recon[i].permute(1, 2, 0))
        axes[1, i].axis("off")
        axes[1, i].set_title("recon")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def run_epoch(model, loader, optimizer, device, kl_weight, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for images, states in tqdm(loader, leave=False):
        images = images.to(device)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(
                recon, images, mu, logvar, kl_weight=kl_weight
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "recon_loss": total_recon / n,
        "kl_loss": total_kl / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/autoencoder")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_dataset = LunarFrameDataset(
        args.train_dir,
        state_key="states",
        max_files=args.max_train_files,
    )
    test_dataset = LunarFrameDataset(
        args.test_dir,
        state_key="states",
        max_files=args.max_test_files,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.kl_weight, train=True
        )
        test_metrics = run_epoch(
            model, test_loader, optimizer, device, args.kl_weight, train=False
        )

        print("train:", train_metrics)
        print("test: ", test_metrics)

        # Save a reconstruction grid from one test batch.
        model.eval()
        images, _ = next(iter(test_loader))
        images = images.to(device)
        with torch.no_grad():
            recon, _, _ = model(images)

        grid_path = os.path.join(args.output_dir, f"recon_epoch_{epoch:03d}.png")
        save_reconstruction_grid(images, recon, grid_path)
        print("saved:", grid_path)

        # Save latest checkpoint.
        latest_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "test_loss": test_metrics["loss"],
            },
            latest_path,
        )

        # Save best checkpoint.
        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "test_loss": test_metrics["loss"],
                },
                best_path,
            )
            print("saved best:", best_path)


if __name__ == "__main__":
    main()
