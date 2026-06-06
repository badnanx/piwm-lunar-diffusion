import argparse
import os
import sys
from pathlib import Path

# Let this script import helper functions from other scripts.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from lunar_diffusion.autoencoder import ConvVAE
from lunar_diffusion.dynamics import LatentDynamicsMLP
from lunar_diffusion.pair_dataset import LunarPairDataset
from scripts.train_piwm_pair_baseline import compute_losses, save_prediction_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)
    num_actions = ckpt_args.get("num_actions", args.num_actions)

    dataset = LunarPairDataset(
        args.data_dir,
        state_key=args.state_key,
        action_key="acts",
    )

    generator = torch.Generator().manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

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

    img_t, img_next, state_t, state_next, action_t = next(iter(loader))

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
            kl_weight=0.0001,
            p1_weight=1.0,
            p2_weight=0.5,
            dynamics_weight=1.0,
            pred_recon_weight=0.1,
        )

    checkpoint_epoch = checkpoint.get("epoch", "unknown")

    save_prediction_grid(
        img_t=img_t,
        img_next=img_next,
        recon_t=outputs["recon_t"],
        recon_next=outputs["recon_next"],
        pred_next=outputs["pred_img_next"],
        save_path=args.output_path,
        num_images=args.num_images,
        title=f"PIWM pair predictions — checkpoint epoch {checkpoint_epoch}",
    )

    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
