import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torch.utils.data import DataLoader

# Allow imports from repo root and src.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lunar_diffusion.autoencoder import ConvVAE
from lunar_diffusion.crop_loss import state_xy_to_pixel
from lunar_diffusion.dynamics import LatentDynamicsMLP
from lunar_diffusion.pair_dataset import LunarPairDataset
from scripts.train_piwm_pair_crop import compute_losses


def draw_crop_box(ax, state, img_h, img_w, crop_size):
    state_batch = state.detach().cpu().view(1, -1)
    px, py = state_xy_to_pixel(state_batch, img_h, img_w)

    cx = float(px[0])
    cy = float(py[0])
    half = crop_size / 2.0

    rect = Rectangle(
        (cx - half, cy - half),
        crop_size,
        crop_size,
        fill=False,
        edgecolor="cyan",
        linewidth=2.5,
    )
    ax.add_patch(rect)


def save_boxed_grid(
    img_t,
    img_next,
    recon_t,
    recon_next,
    pred_next,
    state_t,
    state_next,
    save_path,
    crop_size=32,
    num_images=6,
    title=None,
):
    img_t = img_t[:num_images].detach().cpu()
    img_next = img_next[:num_images].detach().cpu()
    recon_t = recon_t[:num_images].detach().cpu()
    recon_next = recon_next[:num_images].detach().cpu()
    pred_next = pred_next[:num_images].detach().cpu()
    state_t = state_t[:num_images].detach().cpu()
    state_next = state_next[:num_images].detach().cpu()

    rows = [
        ("real t", img_t, state_t),
        ("recon t", recon_t, state_t),
        ("real t+1", img_next, state_next),
        ("recon t+1", recon_next, state_next),
        ("pred t+1", pred_next, state_next),
    ]

    fig, axes = plt.subplots(
        len(rows),
        num_images + 1,
        figsize=(2 * (num_images + 1), 2 * len(rows)),
        gridspec_kw={"width_ratios": [0.9] + [1.0] * num_images},
    )

    for r, (label, imgs, states) in enumerate(rows):
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
            img = imgs[c].permute(1, 2, 0).clamp(0, 1)
            ax.imshow(img)
            ax.axis("off")

            img_h = imgs.shape[2]
            img_w = imgs.shape[3]
            draw_crop_box(
                ax=ax,
                state=states[c],
                img_h=img_h,
                img_w=img_w,
                crop_size=crop_size,
            )

    if title is not None:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title is not None else None)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


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
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state_indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    checkpoint_epoch = checkpoint.get("epoch", "unknown")

    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)
    num_actions = ckpt_args.get("num_actions", args.num_actions)

    dataset = LunarPairDataset(
        data_dir=args.data_dir,
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
            crop_weight=1.0,
            pred_crop_weight=0.5,
            crop_size=args.crop_size,
        )

    save_boxed_grid(
        img_t=img_t,
        img_next=img_next,
        recon_t=outputs["recon_t"],
        recon_next=outputs["recon_next"],
        pred_next=outputs["pred_img_next"],
        state_t=state_t,
        state_next=state_next,
        save_path=args.output_path,
        crop_size=args.crop_size,
        num_images=args.num_images,
        title=f"PIWM P4-lite crop predictions — best checkpoint epoch {checkpoint_epoch}",
    )

    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
