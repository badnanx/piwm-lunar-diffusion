import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.correction import LatentCorrectionMLP
from lunar_diffusion.latent_dataset import LatentTransitionDataset


def rmse(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2))


def rmse_per_dim(x, y):
    return torch.sqrt(torch.mean((x - y) ** 2, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    latent_dim = ckpt_args.get("latent_dim", args.latent_dim)
    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)

    dataset = LatentTransitionDataset(args.latent_npz)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LatentCorrectionMLP(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_z_pred_next = []
    all_z_next = []
    all_z_corrected = []
    all_pred_correction = []
    all_true_correction = []

    with torch.no_grad():
        for batch in tqdm(loader):
            z_pred_next = batch["z_pred_next"].to(device)
            z_next = batch["z_next"].to(device)
            true_correction = batch["correction"].to(device)

            pred_correction = model(z_pred_next)
            z_corrected = z_pred_next + pred_correction

            all_z_pred_next.append(z_pred_next.cpu())
            all_z_next.append(z_next.cpu())
            all_z_corrected.append(z_corrected.cpu())
            all_pred_correction.append(pred_correction.cpu())
            all_true_correction.append(true_correction.cpu())

    z_pred_next = torch.cat(all_z_pred_next, dim=0)
    z_next = torch.cat(all_z_next, dim=0)
    z_corrected = torch.cat(all_z_corrected, dim=0)
    pred_correction = torch.cat(all_pred_correction, dim=0)
    true_correction = torch.cat(all_true_correction, dim=0)

    baseline_rmse = rmse(z_pred_next, z_next)
    corrected_rmse = rmse(z_corrected, z_next)
    correction_rmse = rmse(pred_correction, true_correction)

    baseline_rmse_per_dim = rmse_per_dim(z_pred_next, z_next)
    corrected_rmse_per_dim = rmse_per_dim(z_corrected, z_next)

    improvement_ratio = baseline_rmse / corrected_rmse
    improvement_abs = baseline_rmse - corrected_rmse

    results = {
        "latent_npz": args.latent_npz,
        "checkpoint": args.checkpoint,
        "num_transitions": int(z_next.shape[0]),
        "latent_dim": int(z_next.shape[1]),
        "baseline_rmse": float(baseline_rmse),
        "corrected_rmse": float(corrected_rmse),
        "correction_rmse": float(correction_rmse),
        "improvement_abs": float(improvement_abs),
        "improvement_ratio": float(improvement_ratio),
        "baseline_rmse_per_dim": baseline_rmse_per_dim.tolist(),
        "corrected_rmse_per_dim": corrected_rmse_per_dim.tolist(),
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nLatent correction evaluation")
    print("----------------------------")
    print("num_transitions:", results["num_transitions"])
    print("baseline_rmse:", f"{results['baseline_rmse']:.6f}")
    print("corrected_rmse:", f"{results['corrected_rmse']:.6f}")
    print("correction_rmse:", f"{results['correction_rmse']:.6f}")
    print("improvement_abs:", f"{results['improvement_abs']:.6f}")
    print("improvement_ratio:", f"{results['improvement_ratio']:.3f}x")
    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
