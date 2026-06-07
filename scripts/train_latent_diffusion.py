import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot_actions(actions, num_actions=4):
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions.astype(np.int64)] = 1.0
    return out


class LatentCorrectionDataset(Dataset):
    """
    Dataset for latent diffusion correction.

    From exported PIWM transitions:
      z_t
      z_next
      z_pred_next
      correction = z_next - z_pred_next

    We train diffusion on correction vectors.
    """

    def __init__(
        self,
        npz_path,
        state_dim=6,
        num_actions=4,
        correction_mean=None,
        correction_std=None,
        cond_mean=None,
        cond_std=None,
    ):
        data = np.load(npz_path)

        self.z_t = data["z_t"].astype(np.float32)
        self.z_next = data["z_next"].astype(np.float32)
        self.z_pred_next = data["z_pred_next"].astype(np.float32)
        self.correction = data["correction"].astype(np.float32)
        self.state_next = data["state_next"].astype(np.float32)[:, :state_dim]
        self.action_t = data["action_t"].astype(np.int64)

        action_oh = one_hot_actions(self.action_t, num_actions=num_actions)

        # Conditioning vector:
        #   predicted next latent
        #   current latent
        #   next physical state dims
        #   action one-hot
        self.cond = np.concatenate(
            [
                self.z_pred_next,
                self.z_t,
                self.state_next,
                action_oh,
            ],
            axis=1,
        ).astype(np.float32)

        if correction_mean is None:
            correction_mean = self.correction.mean(axis=0, keepdims=True)
        if correction_std is None:
            correction_std = self.correction.std(axis=0, keepdims=True)

        if cond_mean is None:
            cond_mean = self.cond.mean(axis=0, keepdims=True)
        if cond_std is None:
            cond_std = self.cond.std(axis=0, keepdims=True)

        self.correction_mean = correction_mean.astype(np.float32)
        self.correction_std = np.maximum(correction_std.astype(np.float32), 1e-6)

        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = np.maximum(cond_std.astype(np.float32), 1e-6)

        self.correction_norm = (
            (self.correction - self.correction_mean) / self.correction_std
        ).astype(np.float32)

        self.cond_norm = (
            (self.cond - self.cond_mean) / self.cond_std
        ).astype(np.float32)

    def __len__(self):
        return len(self.correction)

    def __getitem__(self, idx):
        return {
            "correction_norm": torch.from_numpy(self.correction_norm[idx]),
            "cond_norm": torch.from_numpy(self.cond_norm[idx]),
            "z_next": torch.from_numpy(self.z_next[idx]),
            "z_pred_next": torch.from_numpy(self.z_pred_next[idx]),
            "correction": torch.from_numpy(self.correction[idx]),
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        timesteps: (B,) integer tensor
        returns: (B, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -np.log(10000)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )

        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb


class LatentDiffusionMLP(nn.Module):
    """
    Tiny DDPM denoiser for latent correction vectors.

    Input:
      noisy correction x_t
      timestep embedding
      conditioning vector

    Output:
      predicted noise epsilon
    """

    def __init__(
        self,
        latent_dim,
        cond_dim,
        time_dim=64,
        hidden_dim=512,
        num_layers=4,
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        in_dim = latent_dim + cond_dim + time_dim

        layers = []
        dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, timesteps, cond):
        t_emb = self.time_embed(timesteps)
        x = torch.cat([x_t, t_emb, cond], dim=1)
        return self.net(x)


class DiffusionSchedule:
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps

        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


def diffusion_training_loss(model, schedule, x0, cond):
    """
    Standard DDPM noise prediction objective.

    x0:
      normalized correction vector

    cond:
      normalized conditioning vector
    """
    batch_size = x0.size(0)
    device = x0.device

    t = torch.randint(
        low=0,
        high=schedule.num_steps,
        size=(batch_size,),
        device=device,
    )

    noise = torch.randn_like(x0)

    alpha_bar_t = schedule.alpha_bars[t].view(batch_size, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    pred_noise = model(x_t=x_t, timesteps=t, cond=cond)

    return nn.functional.mse_loss(pred_noise, noise, reduction="mean")


@torch.no_grad()
def sample_correction_norm(model, schedule, cond, latent_dim):
    """
    Sample normalized correction vectors from the learned reverse process.
    """
    device = cond.device
    batch_size = cond.size(0)

    x = torch.randn(batch_size, latent_dim, device=device)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        beta_t = schedule.betas[step]
        alpha_t = schedule.alphas[step]
        alpha_bar_t = schedule.alpha_bars[step]

        pred_noise = model(x_t=x, timesteps=t, cond=cond)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )

        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

    return x


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


@torch.no_grad()
def evaluate_latent_rmse(
    model,
    schedule,
    loader,
    device,
    latent_dim,
    correction_mean,
    correction_std,
    max_eval_batches=None,
):
    model.eval()

    baseline_rmses = []
    corrected_rmses = []
    correction_rmses = []

    correction_mean_t = torch.from_numpy(correction_mean).to(device)
    correction_std_t = torch.from_numpy(correction_std).to(device)

    for batch_idx, batch in enumerate(loader):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        cond = batch["cond_norm"].to(device)
        z_pred_next = batch["z_pred_next"].to(device)
        z_next = batch["z_next"].to(device)
        true_correction = batch["correction"].to(device)

        sampled_norm = sample_correction_norm(
            model=model,
            schedule=schedule,
            cond=cond,
            latent_dim=latent_dim,
        )

        sampled_correction = sampled_norm * correction_std_t + correction_mean_t
        z_corrected = z_pred_next + sampled_correction

        baseline_rmses.append(rmse(z_pred_next, z_next))
        corrected_rmses.append(rmse(z_corrected, z_next))
        correction_rmses.append(rmse(sampled_correction, true_correction))

    return {
        "baseline_rmse": float(np.mean(baseline_rmses)),
        "corrected_rmse": float(np.mean(corrected_rmses)),
        "correction_rmse": float(np.mean(correction_rmses)),
    }


def run_train_epoch(model, schedule, loader, optimizer, device):
    model.train()

    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        x0 = batch["correction_norm"].to(device)
        cond = batch["cond_norm"].to(device)

        loss = diffusion_training_loss(
            model=model,
            schedule=schedule,
            x0=x0,
            cond=cond,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


@torch.no_grad()
def run_loss_epoch(model, schedule, loader, device):
    model.eval()

    losses = []
    for batch in tqdm(loader, desc="val-loss", leave=False):
        x0 = batch["correction_norm"].to(device)
        cond = batch["cond_norm"].to(device)

        loss = diffusion_training_loss(
            model=model,
            schedule=schedule,
            x0=x0,
            cond=cond,
        )

        losses.append(loss.item())

    return {"loss": float(np.mean(losses))}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--num_actions", type=int, default=4)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_eval_batches", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("train_npz:", args.train_npz)
    print("output_dir:", args.output_dir)
    print("latent_dim:", args.latent_dim)
    print("state_dim:", args.state_dim)
    print("diffusion_steps:", args.diffusion_steps)
    print("hidden_dim:", args.hidden_dim)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("seed:", args.seed)

    full_dataset = LatentCorrectionDataset(
        npz_path=args.train_npz,
        state_dim=args.state_dim,
        num_actions=args.num_actions,
    )

    cond_dim = full_dataset.cond_norm.shape[1]

    n_total = len(full_dataset)
    n_val = max(1, int(args.val_fraction * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=generator,
    )

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

    model = LatentDiffusionMLP(
        latent_dim=args.latent_dim,
        cond_dim=cond_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    schedule = DiffusionSchedule(
        num_steps=args.diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_metric = float("inf")
    best_epoch = None
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print()
        print(f"Epoch {epoch}/{args.epochs}")

        train_metrics = run_train_epoch(
            model=model,
            schedule=schedule,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        val_loss_metrics = run_loss_epoch(
            model=model,
            schedule=schedule,
            loader=val_loader,
            device=device,
        )

        val_rmse_metrics = evaluate_latent_rmse(
            model=model,
            schedule=schedule,
            loader=val_loader,
            device=device,
            latent_dim=args.latent_dim,
            correction_mean=full_dataset.correction_mean,
            correction_std=full_dataset.correction_std,
            max_eval_batches=args.max_eval_batches,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val_loss": val_loss_metrics,
            "val_rmse": val_rmse_metrics,
        }
        history.append(row)

        print("train:", train_metrics)
        print("val_loss:", val_loss_metrics)
        print("val_rmse:", val_rmse_metrics)

        metric = val_rmse_metrics["corrected_rmse"]

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            no_improve = 0

            best_path = os.path.join(args.output_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "correction_mean": full_dataset.correction_mean,
                    "correction_std": full_dataset.correction_std,
                    "cond_mean": full_dataset.cond_mean,
                    "cond_std": full_dataset.cond_std,
                    "cond_dim": cond_dim,
                },
                best_path,
            )
            print("saved best:", best_path)
        else:
            no_improve += 1
            print(f"no improvement for {no_improve}/{args.patience} epochs")

        history_path = os.path.join(args.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= args.patience:
            print("early stopping")
            break

    elapsed = time.time() - start_time

    summary = {
        "best_epoch": best_epoch,
        "best_val_corrected_rmse": best_metric,
        "total_epochs_run": len(history),
        "elapsed_minutes": elapsed / 60.0,
        "args": vars(args),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Latent diffusion training summary")
    print("---------------------------------")
    print("best_epoch:", best_epoch)
    print("best_val_corrected_rmse:", best_metric)
    print("total_epochs_run:", len(history))
    print("elapsed_minutes:", f"{elapsed / 60.0:.2f}")
    print("saved summary:", summary_path)


if __name__ == "__main__":
    main()
