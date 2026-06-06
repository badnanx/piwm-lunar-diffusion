import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.autoencoder import ConvVAE
from lunar_diffusion.dynamics import LatentDynamicsMLP
from lunar_diffusion.pair_dataset import LunarPairDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)

    parser.add_argument("--state_key", default="states")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)
    num_actions = ckpt_args.get("num_actions", args.num_actions)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    dynamics = LatentDynamicsMLP(
        latent_dim=args.latent_dim,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    dynamics.load_state_dict(checkpoint["dynamics_state_dict"])

    model.eval()
    dynamics.eval()

    dataset = LunarPairDataset(
        data_dir=args.data_dir,
        state_key=args.state_key,
        action_key="acts",
        max_files=args.max_files,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_z_t = []
    all_z_next = []
    all_z_pred_next = []
    all_state_t = []
    all_state_next = []
    all_action_t = []

    with torch.no_grad():
        for img_t, img_next, state_t, state_next, action_t in tqdm(loader):
            img_t = img_t.to(device)
            img_next = img_next.to(device)
            action_t_device = action_t.to(device)

            # Use mu, not sampled z, for deterministic latent export.
            mu_t, _ = model.encode(img_t)
            mu_next, _ = model.encode(img_next)

            pred_mu_next = dynamics(mu_t, action_t_device)

            all_z_t.append(mu_t.cpu().numpy().astype(np.float32))
            all_z_next.append(mu_next.cpu().numpy().astype(np.float32))
            all_z_pred_next.append(pred_mu_next.cpu().numpy().astype(np.float32))

            all_state_t.append(state_t.numpy().astype(np.float32))
            all_state_next.append(state_next.numpy().astype(np.float32))
            all_action_t.append(action_t.numpy().astype(np.int64))

    z_t = np.concatenate(all_z_t, axis=0)
    z_next = np.concatenate(all_z_next, axis=0)
    z_pred_next = np.concatenate(all_z_pred_next, axis=0)
    state_t = np.concatenate(all_state_t, axis=0)
    state_next = np.concatenate(all_state_next, axis=0)
    action_t = np.concatenate(all_action_t, axis=0)

    correction = z_next - z_pred_next

    np.savez_compressed(
        args.output_path,
        z_t=z_t,
        z_next=z_next,
        z_pred_next=z_pred_next,
        correction=correction,
        state_t=state_t,
        state_next=state_next,
        action_t=action_t,
    )

    summary = {
        "output_path": args.output_path,
        "num_pairs": int(z_t.shape[0]),
        "latent_dim": int(z_t.shape[1]),
        "state_key": args.state_key,
        "checkpoint": args.checkpoint,
        "arrays": {
            "z_t": list(z_t.shape),
            "z_next": list(z_next.shape),
            "z_pred_next": list(z_pred_next.shape),
            "correction": list(correction.shape),
            "state_t": list(state_t.shape),
            "state_next": list(state_next.shape),
            "action_t": list(action_t.shape),
        },
        "correction_mean_abs": float(np.mean(np.abs(correction))),
        "correction_rmse": float(np.sqrt(np.mean(correction ** 2))),
    }

    summary_path = args.output_path.replace(".npz", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nExport summary")
    print("--------------")
    print("num_pairs:", summary["num_pairs"])
    print("latent_dim:", summary["latent_dim"])
    print("correction_mean_abs:", summary["correction_mean_abs"])
    print("correction_rmse:", summary["correction_rmse"])
    print("saved:", args.output_path)
    print("saved summary:", summary_path)


if __name__ == "__main__":
    main()
