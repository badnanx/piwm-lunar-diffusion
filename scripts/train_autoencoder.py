import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.dataset import LunarFrameDataset
from lunar_diffusion.autoencoder import ConvVAE, vae_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    kl_weight,
    state_weight,
    state_indices,
    kl_on_physical,
    train=True,
):
    model.train() if train else model.eval()
    desc = "train" if train else "test"

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_state = 0.0

    for images, states in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        states = states.to(device)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss, state_loss = vae_loss(
                recon,
                images,
                mu,
                logvar,
                states=states,
                state_indices=state_indices,
                kl_weight=kl_weight,
                state_weight=state_weight,
                kl_on_physical=kl_on_physical,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_state += state_loss.item() * batch_size

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "recon_loss": total_recon / n,
        "kl_loss": total_kl / n,
        "state_loss": total_state / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/autoencoder")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--state_weight", type=float, default=0.1)
    parser.add_argument(
        "--state_indices",
        type=int,
        nargs="+",
        default=[0, 1, 4],
        help="State indices to supervise. Default: 0 1 4 = x y theta.",
    )
    parser.add_argument(
        "--kl_on_physical",
        action="store_true",
        help="If set, apply KL to physical latent dims too. Default: KL only on residual dims.",
    )
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.state_weight > 0 and len(args.state_indices) > args.latent_dim:
        raise ValueError("Number of state indices cannot exceed latent_dim.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("state_weight:", args.state_weight)
    print("state_indices:", args.state_indices)
    print("kl_weight:", args.kl_weight)
    print("kl_on_physical:", args.kl_on_physical)
    print("patience:", args.patience)
    print("min_delta:", args.min_delta)
    print("seed:", args.seed)

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

    viz_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.kl_weight,
            args.state_weight,
            args.state_indices,
            args.kl_on_physical,
            train=True,
        )
        test_metrics = run_epoch(
            model,
            test_loader,
            optimizer,
            device,
            args.kl_weight,
            args.state_weight,
            args.state_indices,
            args.kl_on_physical,
            train=False,
        )

        print("train:", train_metrics)
        print("test: ", test_metrics)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(row)

        model.eval()
        images, _ = next(iter(viz_loader))
        images = images.to(device)
        with torch.no_grad():
            recon, _, _ = model(images)

        grid_path = os.path.join(args.output_dir, f"recon_epoch_{epoch:03d}.png")
        save_reconstruction_grid(images, recon, grid_path)
        print("saved:", grid_path)

        latest_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "test_loss": test_metrics["loss"],
                "history": history,
            },
            latest_path,
        )

        improved = test_metrics["loss"] < (best_test_loss - args.min_delta)

        if improved:
            best_test_loss = test_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "test_loss": test_metrics["loss"],
                    "history": history,
                },
                best_path,
            )
            print("saved best:", best_path)
        else:
            epochs_without_improvement += 1
            print(
                f"no improvement for {epochs_without_improvement}/"
                f"{args.patience} epochs"
            )

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time

    summary = {
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss,
        "total_epochs_run": history[-1]["epoch"] if history else 0,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "args": vars(args),
        "final_train": history[-1]["train"] if history else None,
        "final_test": history[-1]["test"] if history else None,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining summary")
    print("----------------")
    print(f"best_epoch: {summary['best_epoch']}")
    print(f"best_test_loss: {summary['best_test_loss']}")
    print(f"total_epochs_run: {summary['total_epochs_run']}")
    print(f"elapsed_minutes: {summary['elapsed_minutes']:.2f}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
