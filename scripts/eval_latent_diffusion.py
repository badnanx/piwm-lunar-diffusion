import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing from scripts/ when this file is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_latent_diffusion import (
    LatentCorrectionDataset,
    LatentDiffusionMLP,
    DiffusionSchedule,
    sample_correction_norm,
)


def rmse_np(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--state_dim", type=int, default=6)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt["args"]

    correction_mean = ckpt["correction_mean"]
    correction_std = ckpt["correction_std"]
    cond_mean = ckpt["cond_mean"]
    cond_std = ckpt["cond_std"]

    dataset = LatentCorrectionDataset(
        npz_path=args.latent_npz,
        state_dim=args.state_dim,
        num_actions=args.num_actions,
        correction_mean=correction_mean,
        correction_std=correction_std,
        cond_mean=cond_mean,
        cond_std=cond_std,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LatentDiffusionMLP(
        latent_dim=args.latent_dim,
        cond_dim=ckpt["cond_dim"],
        time_dim=ckpt_args["time_dim"],
        hidden_dim=ckpt_args["hidden_dim"],
        num_layers=ckpt_args["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    schedule = DiffusionSchedule(
        num_steps=ckpt_args["diffusion_steps"],
        beta_start=ckpt_args["beta_start"],
        beta_end=ckpt_args["beta_end"],
        device=device,
    )

    correction_mean_t = torch.from_numpy(correction_mean).to(device)
    correction_std_t = torch.from_numpy(correction_std).to(device)

    z_next_all = []
    z_pred_all = []
    z_diff_all = []
    corr_true_all = []
    corr_sample_all = []

    with torch.no_grad():
        for batch in loader:
            cond = batch["cond_norm"].to(device)
            z_next = batch["z_next"].to(device)
            z_pred_next = batch["z_pred_next"].to(device)
            true_correction = batch["correction"].to(device)

            # Average multiple stochastic diffusion samples if requested.
            sampled_corrections = []
            for _ in range(args.num_samples):
                sampled_norm = sample_correction_norm(
                    model=model,
                    schedule=schedule,
                    cond=cond,
                    latent_dim=args.latent_dim,
                )
                sampled_correction = sampled_norm * correction_std_t + correction_mean_t
                sampled_corrections.append(sampled_correction)

            sampled_correction = torch.stack(sampled_corrections, dim=0).mean(dim=0)
            z_corrected = z_pred_next + sampled_correction

            z_next_all.append(z_next.cpu().numpy())
            z_pred_all.append(z_pred_next.cpu().numpy())
            z_diff_all.append(z_corrected.cpu().numpy())
            corr_true_all.append(true_correction.cpu().numpy())
            corr_sample_all.append(sampled_correction.cpu().numpy())

    z_next_all = np.concatenate(z_next_all, axis=0)
    z_pred_all = np.concatenate(z_pred_all, axis=0)
    z_diff_all = np.concatenate(z_diff_all, axis=0)
    corr_true_all = np.concatenate(corr_true_all, axis=0)
    corr_sample_all = np.concatenate(corr_sample_all, axis=0)

    baseline_rmse = rmse_np(z_pred_all, z_next_all)
    corrected_rmse = rmse_np(z_diff_all, z_next_all)
    correction_rmse = rmse_np(corr_sample_all, corr_true_all)

    metrics = {
        "num_transitions": int(len(z_next_all)),
        "num_samples": int(args.num_samples),
        "baseline_rmse": baseline_rmse,
        "diffusion_corrected_rmse": corrected_rmse,
        "diffusion_correction_rmse": correction_rmse,
        "improvement_abs": baseline_rmse - corrected_rmse,
        "improvement_ratio": baseline_rmse / corrected_rmse if corrected_rmse > 0 else None,
        "checkpoint": args.checkpoint,
        "latent_npz": args.latent_npz,
    }

    with open(args.output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Latent diffusion evaluation")
    print("--------------------------")
    print("num_transitions:", metrics["num_transitions"])
    print("num_samples:", metrics["num_samples"])
    print("baseline_rmse:", f"{baseline_rmse:.6f}")
    print("diffusion_corrected_rmse:", f"{corrected_rmse:.6f}")
    print("diffusion_correction_rmse:", f"{correction_rmse:.6f}")
    print("improvement_abs:", f"{metrics['improvement_abs']:.6f}")
    print("improvement_ratio:", f"{metrics['improvement_ratio']:.3f}x")
    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
