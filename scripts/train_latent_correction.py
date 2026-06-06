import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from lunar_diffusion.correction import LatentCorrectionMLP
from lunar_diffusion.latent_dataset import LatentTransitionDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_baseline_rmse = 0.0
    total_corrected_rmse = 0.0
    total_correction_rmse = 0.0
    n_seen = 0

    desc = "train" if train else "test"

    for batch in tqdm(loader, desc=desc, leave=False):
        z_pred_next = batch["z_pred_next"].to(device)
        z_next = batch["z_next"].to(device)
        correction = batch["correction"].to(device)

        with torch.set_grad_enabled(train):
            pred_correction = model(z_pred_next)
            z_corrected = z_pred_next + pred_correction

            # Train directly on correction residual.
            loss = F.mse_loss(pred_correction, correction, reduction="mean")

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = z_pred_next.size(0)
        n_seen += batch_size

        baseline_rmse = rmse(z_pred_next, z_next)
        corrected_rmse = rmse(z_corrected, z_next)
        correction_rmse = rmse(pred_correction, correction)

        total_loss += loss.item() * batch_size
        total_baseline_rmse += baseline_rmse.item() * batch_size
        total_corrected_rmse += corrected_rmse.item() * batch_size
        total_correction_rmse += correction_rmse.item() * batch_size

    return {
        "loss": total_loss / n_seen,
        "baseline_rmse": total_baseline_rmse / n_seen,
        "corrected_rmse": total_corrected_rmse / n_seen,
        "correction_rmse": total_correction_rmse / n_seen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_npz", required=True)
    parser.add_argument("--output_dir", default="outputs/latent_correction")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = LatentTransitionDataset(args.latent_npz)

    n_total = len(dataset)
    n_val = int(round(n_total * args.val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=generator,
    )

    print("n_train:", n_train)
    print("n_val:", n_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LatentCorrectionMLP(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_rmse = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        print("train:", train_metrics)
        print("val:  ", val_metrics)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        latest_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "history": history,
            },
            latest_path,
        )

        improved = val_metrics["corrected_rmse"] < (best_val_rmse - args.min_delta)

        if improved:
            best_val_rmse = val_metrics["corrected_rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "history": history,
                },
                best_path,
            )
            print("saved best:", best_path)
        else:
            epochs_without_improvement += 1
            print(f"no improvement for {epochs_without_improvement}/{args.patience} epochs")

        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time

    summary = {
        "best_epoch": best_epoch,
        "best_val_corrected_rmse": best_val_rmse,
        "total_epochs_run": history[-1]["epoch"] if history else 0,
        "elapsed_minutes": elapsed / 60.0,
        "final_train": history[-1]["train"] if history else None,
        "final_val": history[-1]["val"] if history else None,
        "args": vars(args),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining summary")
    print("----------------")
    print(f"best_epoch: {best_epoch}")
    print(f"best_val_corrected_rmse: {best_val_rmse}")
    print(f"elapsed_minutes: {elapsed / 60.0:.2f}")
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
