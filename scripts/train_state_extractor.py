import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.dataset import LunarFrameDataset
from lunar_diffusion.visible_dataset import LunarVisibleFrameDataset
from lunar_diffusion.state_extractor import StateExtractorCNN


STATE_NAMES = {
    0: "x",
    1: "y",
    2: "vx",
    3: "vy",
    4: "theta",
    5: "omega",
    6: "left_leg",
    7: "right_leg",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse_per_dim(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))


def run_epoch(model, loader, optimizer, device, state_indices, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_pred = []
    all_target = []
    n_seen = 0

    desc = "train" if train else "test"

    for images, states in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        states = states.to(device)
        target = states[:, state_indices]

        with torch.set_grad_enabled(train):
            pred = model(images)
            loss = F.mse_loss(pred, target, reduction="mean")

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_seen += batch_size

        all_pred.append(pred.detach().cpu())
        all_target.append(target.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    rmse = rmse_per_dim(all_pred, all_target)

    return {
        "loss": total_loss / n_seen,
        "rmse": rmse.tolist(),
        "rmse_mean": float(rmse.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/state_extractor")
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 4])
    parser.add_argument("--visible_only", action="store_true", help="Train/evaluate only on fully visible lander frames.")
    parser.add_argument("--visible_margin", type=int, default=0, help="Pixel margin for fully visible filtering.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("state_key:", args.state_key)
    print("visible_only:", args.visible_only)
    print("visible_margin:", args.visible_margin)
    print("state_indices:", args.state_indices)
    print("state_names:", [STATE_NAMES.get(i, f"state_{i}") for i in args.state_indices])

    dataset_cls = LunarVisibleFrameDataset if args.visible_only else LunarFrameDataset

    if args.visible_only:
        train_dataset = dataset_cls(
            args.train_dir,
            state_key=args.state_key,
            max_files=args.max_train_files,
            margin=args.visible_margin,
        )
        test_dataset = dataset_cls(
            args.test_dir,
            state_key=args.state_key,
            max_files=args.max_test_files,
            margin=args.visible_margin,
        )
    else:
        train_dataset = dataset_cls(
            args.train_dir,
            state_key=args.state_key,
            max_files=args.max_train_files,
        )
        test_dataset = dataset_cls(
            args.test_dir,
            state_key=args.state_key,
            max_files=args.max_test_files,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = StateExtractorCNN(output_dim=len(args.state_indices)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.state_indices, train=True
        )
        test_metrics = run_epoch(
            model, test_loader, optimizer, device, args.state_indices, train=False
        )

        print("train:", train_metrics)
        print("test: ", test_metrics)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
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

        improved = test_metrics["loss"] < (best_loss - args.min_delta)

        if improved:
            best_loss = test_metrics["loss"]
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
        "best_loss": best_loss,
        "total_epochs_run": history[-1]["epoch"] if history else 0,
        "elapsed_minutes": elapsed / 60.0,
        "state_key": args.state_key,
        "visible_only": args.visible_only,
        "visible_margin": args.visible_margin,
        "state_indices": args.state_indices,
        "state_names": [STATE_NAMES.get(i, f"state_{i}") for i in args.state_indices],
        "final_train": history[-1]["train"] if history else None,
        "final_test": history[-1]["test"] if history else None,
        "args": vars(args),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining summary")
    print("----------------")
    print(f"best_epoch: {best_epoch}")
    print(f"best_loss: {best_loss}")
    print(f"elapsed_minutes: {elapsed / 60.0:.2f}")
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
