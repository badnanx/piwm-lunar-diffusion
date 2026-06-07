import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.autoencoder import ConvVAE, kl_divergence
from lunar_diffusion.crop_loss import state_guided_crop_mse
from lunar_diffusion.dynamics import LatentDynamicsMLP
from lunar_diffusion.pair_dataset import LunarPairDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_prediction_grid(img_t, img_next, recon_t, recon_next, pred_next, save_path, num_images=6, title=None):
    img_t = img_t[:num_images].detach().cpu()
    img_next = img_next[:num_images].detach().cpu()
    recon_t = recon_t[:num_images].detach().cpu()
    recon_next = recon_next[:num_images].detach().cpu()
    pred_next = pred_next[:num_images].detach().cpu()

    rows = [
        ("real t", img_t),
        ("recon t", recon_t),
        ("real t+1", img_next),
        ("recon t+1", recon_next),
        ("pred t+1", pred_next),
    ]

    # Extra first column is reserved for large row labels.
    fig, axes = plt.subplots(
        len(rows),
        num_images + 1,
        figsize=(2 * (num_images + 1), 2 * len(rows)),
        gridspec_kw={"width_ratios": [0.9] + [1.0] * num_images},
    )

    for r, (label, imgs) in enumerate(rows):
        label_ax = axes[r, 0]
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        for c in range(num_images):
            ax = axes[r, c + 1]
            ax.imshow(imgs[c].permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title is not None else None)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def select_state(states, state_indices):
    return states[:, state_indices]


def compute_losses(
    model,
    dynamics,
    img_t,
    img_next,
    state_t,
    state_next,
    action_t,
    state_indices,
    kl_weight,
    p1_weight,
    p2_weight,
    dynamics_weight,
    pred_recon_weight,
    crop_weight,
    pred_crop_weight,
    crop_size,
):
    k = len(state_indices)

    # Encode current and next images.
    mu_t, logvar_t = model.encode(img_t)
    z_t = model.reparameterize(mu_t, logvar_t)

    mu_next, logvar_next = model.encode(img_next)
    z_next = model.reparameterize(mu_next, logvar_next)

    # Decode true/current latents.
    recon_t = model.decode(z_t)
    recon_next = model.decode(z_next)

    # Dynamics predicts next latent from current latent + action.
    # Use mu_t for dynamics because it is deterministic/stable.
    pred_mu_next = dynamics(mu_t, action_t)
    pred_img_next = model.decode(pred_mu_next)

    # Reconstruction losses.
    recon_t_loss = F.mse_loss(recon_t, img_t, reduction="mean")
    recon_next_loss = F.mse_loss(recon_next, img_next, reduction="mean")
    recon_loss = 0.5 * (recon_t_loss + recon_next_loss)

    # Predicted-next image loss.
    pred_recon_loss = F.mse_loss(pred_img_next, img_next, reduction="mean")

    crop_loss = 0.5 * (
        state_guided_crop_mse(
            pred_images=recon_t,
            target_images=img_t,
            states=state_t,
            crop_size=crop_size,
        )
        + state_guided_crop_mse(
            pred_images=recon_next,
            target_images=img_next,
            states=state_next,
            crop_size=crop_size,
        )
    )

    pred_crop_loss = state_guided_crop_mse(
        pred_images=pred_img_next,
        target_images=img_next,
        states=state_next,
        crop_size=crop_size,
    )

    # PIWM P1: physical latent dims match physical state.
    target_t = select_state(state_t, state_indices)
    target_next = select_state(state_next, state_indices)

    p1_t_loss = F.mse_loss(mu_t[:, :k], target_t, reduction="mean")
    p1_next_loss = F.mse_loss(mu_next[:, :k], target_next, reduction="mean")
    p1_loss = 0.5 * (p1_t_loss + p1_next_loss)

    # PIWM P2: latent physical change matches physical state change.
    delta_state = target_next - target_t
    p2_target = mu_t[:, :k] + delta_state
    p2_loss = F.mse_loss(mu_next[:, :k], p2_target, reduction="mean")

    # Dynamics latent prediction loss.
    # Detach target so dynamics learns to chase encoder latents,
    # while encoder is trained by recon/P1/P2.
    dynamics_latent_loss = F.mse_loss(pred_mu_next, mu_next.detach(), reduction="mean")
    dynamics_state_loss = F.mse_loss(pred_mu_next[:, :k], target_next, reduction="mean")
    dynamics_loss = dynamics_latent_loss + dynamics_state_loss

    # KL only on residual latent dims, matching PIWM Principle 1 idea.
    if k < mu_t.size(1):
        kl_t = kl_divergence(mu_t[:, k:], logvar_t[:, k:])
        kl_next = kl_divergence(mu_next[:, k:], logvar_next[:, k:])
        kl_loss = 0.5 * (kl_t + kl_next)
    else:
        kl_loss = torch.zeros((), device=img_t.device)

    total_loss = (
        recon_loss
        + kl_weight * kl_loss
        + p1_weight * p1_loss
        + p2_weight * p2_loss
        + dynamics_weight * dynamics_loss
        + pred_recon_weight * pred_recon_loss
        + crop_weight * crop_loss
        + pred_crop_weight * pred_crop_loss
    )

    metrics = {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "pred_recon_loss": pred_recon_loss,
        "kl_loss": kl_loss,
        "p1_loss": p1_loss,
        "p2_loss": p2_loss,
        "dynamics_loss": dynamics_loss,
        "dynamics_latent_loss": dynamics_latent_loss,
        "dynamics_state_loss": dynamics_state_loss,
        "crop_loss": crop_loss,
        "pred_crop_loss": pred_crop_loss,
    }

    outputs = {
        "recon_t": recon_t,
        "recon_next": recon_next,
        "pred_img_next": pred_img_next,
        "mu_t": mu_t,
        "mu_next": mu_next,
        "pred_mu_next": pred_mu_next,
    }

    return metrics, outputs


def run_epoch(
    model,
    dynamics,
    loader,
    optimizer,
    device,
    state_indices,
    kl_weight,
    p1_weight,
    p2_weight,
    dynamics_weight,
    pred_recon_weight,
    crop_weight,
    pred_crop_weight,
    crop_size,
    train=True,
):
    model.train() if train else model.eval()
    dynamics.train() if train else dynamics.eval()

    desc = "train" if train else "test"

    totals = {}

    n_seen = 0

    for img_t, img_next, state_t, state_next, action_t in tqdm(loader, desc=desc, leave=False):
        img_t = img_t.to(device)
        img_next = img_next.to(device)
        state_t = state_t.to(device)
        state_next = state_next.to(device)
        action_t = action_t.to(device)

        with torch.set_grad_enabled(train):
            metrics, _ = compute_losses(
                model=model,
                dynamics=dynamics,
                img_t=img_t,
                img_next=img_next,
                state_t=state_t,
                state_next=state_next,
                action_t=action_t,
                state_indices=state_indices,
                kl_weight=kl_weight,
                p1_weight=p1_weight,
                p2_weight=p2_weight,
                dynamics_weight=dynamics_weight,
                pred_recon_weight=pred_recon_weight,
                crop_weight=crop_weight,
                pred_crop_weight=pred_crop_weight,
                crop_size=crop_size,
            )

            loss = metrics["loss"]

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = img_t.size(0)
        n_seen += batch_size

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value.item() * batch_size

    return {key: value / n_seen for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/piwm_pair_crop")
    parser.add_argument("--state_key", default="states", help="State key to use: states, noisy_states_2, noisy_states_5, noisy_states_10")

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_actions", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--p1_weight", type=float, default=0.1)
    parser.add_argument("--p2_weight", type=float, default=0.1)
    parser.add_argument("--dynamics_weight", type=float, default=0.1)
    parser.add_argument("--pred_recon_weight", type=float, default=0.1)
    parser.add_argument("--crop_weight", type=float, default=1.0)
    parser.add_argument("--pred_crop_weight", type=float, default=0.5)
    parser.add_argument("--crop_size", type=int, default=32)

    parser.add_argument(
        "--state_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Default: 0 1 2 3 4 5 = x y vx vy theta omega.",
    )

    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument("--max_train_files", type=int, default=None)
    parser.add_argument("--max_test_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("state_key:", args.state_key)
    print("state_indices:", args.state_indices)
    print("kl_weight:", args.kl_weight)
    print("p1_weight:", args.p1_weight)
    print("p2_weight:", args.p2_weight)
    print("dynamics_weight:", args.dynamics_weight)
    print("pred_recon_weight:", args.pred_recon_weight)
    print("crop_weight:", args.crop_weight)
    print("pred_crop_weight:", args.pred_crop_weight)
    print("crop_size:", args.crop_size)
    print("patience:", args.patience)
    print("seed:", args.seed)

    train_dataset = LunarPairDataset(
        args.train_dir,
        state_key=args.state_key,
        action_key="acts",
        max_files=args.max_train_files,
    )
    test_dataset = LunarPairDataset(
        args.test_dir,
        state_key=args.state_key,
        action_key="acts",
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
    dynamics = LatentDynamicsMLP(
        latent_dim=args.latent_dim,
        num_actions=args.num_actions,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(dynamics.parameters()),
        lr=args.lr,
    )

    best_test_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            dynamics=dynamics,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            state_indices=args.state_indices,
            kl_weight=args.kl_weight,
            p1_weight=args.p1_weight,
            p2_weight=args.p2_weight,
            dynamics_weight=args.dynamics_weight,
            pred_recon_weight=args.pred_recon_weight,
            crop_weight=args.crop_weight,
            pred_crop_weight=args.pred_crop_weight,
            crop_size=args.crop_size,
            train=True,
        )

        test_metrics = run_epoch(
            model=model,
            dynamics=dynamics,
            loader=test_loader,
            optimizer=optimizer,
            device=device,
            state_indices=args.state_indices,
            kl_weight=args.kl_weight,
            p1_weight=args.p1_weight,
            p2_weight=args.p2_weight,
            dynamics_weight=args.dynamics_weight,
            pred_recon_weight=args.pred_recon_weight,
            crop_weight=args.crop_weight,
            pred_crop_weight=args.pred_crop_weight,
            crop_size=args.crop_size,
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
        dynamics.eval()

        img_t, img_next, state_t, state_next, action_t = next(iter(viz_loader))
        img_t = img_t.to(device)
        img_next = img_next.to(device)
        state_t = state_t.to(device)
        state_next = state_next.to(device)
        action_t = action_t.to(device)

        with torch.no_grad():
            _, outputs = compute_losses(
                model=model,
                dynamics=dynamics,
                img_t=img_t,
                img_next=img_next,
                state_t=state_t,
                state_next=state_next,
                action_t=action_t,
                state_indices=args.state_indices,
                kl_weight=args.kl_weight,
                p1_weight=args.p1_weight,
                p2_weight=args.p2_weight,
                dynamics_weight=args.dynamics_weight,
                pred_recon_weight=args.pred_recon_weight,
                crop_weight=args.crop_weight,
                pred_crop_weight=args.pred_crop_weight,
                crop_size=args.crop_size,
            )

        grid_path = os.path.join(args.output_dir, f"pred_grid_epoch_{epoch:03d}.png")
        save_prediction_grid(
            img_t=img_t,
            img_next=img_next,
            recon_t=outputs["recon_t"],
            recon_next=outputs["recon_next"],
            pred_next=outputs["pred_img_next"],
            save_path=grid_path,
            title=f"PIWM P4-lite crop predictions — epoch {epoch}",
        )
        print("saved:", grid_path)

        latest_path = os.path.join(args.output_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dynamics_state_dict": dynamics.state_dict(),
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
                    "dynamics_state_dict": dynamics.state_dict(),
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
