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


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--state_key", default="states", help="State key to evaluate against")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--state_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
    )
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
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_mu_t = []
    all_mu_next = []
    all_pred_mu_next = []
    all_state_t = []
    all_state_next = []

    state_indices = args.state_indices
    k = len(state_indices)

    with torch.no_grad():
        for img_t, img_next, state_t, state_next, action_t in tqdm(loader):
            img_t = img_t.to(device)
            img_next = img_next.to(device)
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            action_t = action_t.to(device)

            mu_t, _ = model.encode(img_t)
            mu_next, _ = model.encode(img_next)
            pred_mu_next = dynamics(mu_t, action_t)

            all_mu_t.append(mu_t[:, :k].cpu())
            all_mu_next.append(mu_next[:, :k].cpu())
            all_pred_mu_next.append(pred_mu_next[:, :k].cpu())
            all_state_t.append(state_t[:, state_indices].cpu())
            all_state_next.append(state_next[:, state_indices].cpu())

    mu_t = torch.cat(all_mu_t, dim=0)
    mu_next = torch.cat(all_mu_next, dim=0)
    pred_mu_next = torch.cat(all_pred_mu_next, dim=0)
    state_t = torch.cat(all_state_t, dim=0)
    state_next = torch.cat(all_state_next, dim=0)

    delta_state = state_next - state_t
    p2_target = mu_t + delta_state

    p1_t_rmse = rmse(mu_t, state_t)
    p1_next_rmse = rmse(mu_next, state_next)
    dynamics_state_rmse = rmse(pred_mu_next, state_next)
    dynamics_latent_rmse = rmse(pred_mu_next, mu_next)
    p2_rmse = rmse(mu_next, p2_target)

    results = {
        "num_pairs": int(mu_t.shape[0]),
        "state_key": args.state_key,
        "state_indices": state_indices,
        "state_names": [STATE_NAMES.get(i, f"state_{i}") for i in state_indices],
        "p1_t_rmse": p1_t_rmse.tolist(),
        "p1_next_rmse": p1_next_rmse.tolist(),
        "dynamics_state_rmse": dynamics_state_rmse.tolist(),
        "dynamics_latent_rmse": dynamics_latent_rmse.tolist(),
        "p2_rmse": p2_rmse.tolist(),
        "means": {
            "p1_t_rmse": float(p1_t_rmse.mean()),
            "p1_next_rmse": float(p1_next_rmse.mean()),
            "dynamics_state_rmse": float(dynamics_state_rmse.mean()),
            "dynamics_latent_rmse": float(dynamics_latent_rmse.mean()),
            "p2_rmse": float(p2_rmse.mean()),
        },
    }

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation summary")
    print("------------------")
    print("num_pairs:", results["num_pairs"])
    print("state_names:", results["state_names"])
    for key, value in results["means"].items():
        print(f"{key}: {value:.6f}")
    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
