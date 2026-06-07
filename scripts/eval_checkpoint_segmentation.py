import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing helper functions from scripts/ when run as a file.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunar_diffusion.autoencoder import ConvVAE
from lunar_diffusion.crop_loss import state_xy_to_pixel
from lunar_diffusion.dynamics import LatentDynamicsMLP
from lunar_diffusion.pair_dataset import LunarPairDataset
from scripts.analyze_lander_segmentation import lander_mask_rgb, mask_center


KINDS = [
    "real_t",
    "recon_t",
    "real_next",
    "recon_next",
    "pred_next",
]


def tensor_to_image_np(x):
    """
    x: torch tensor [3,H,W], assumed in [0,1]
    returns [H,W,3] float numpy in [0,1]
    """
    return x.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def state_center_pixels(state, img_h, img_w):
    state_t = torch.tensor(state[None, :], dtype=torch.float32)
    px, py = state_xy_to_pixel(state_t, img_h=img_h, img_w=img_w)
    return float(px[0]), float(py[0])


def analyze_image(img, state, purple_threshold):
    img_h, img_w = img.shape[:2]

    true_x, true_y = state_center_pixels(state, img_h, img_w)
    visible_expected = (
        true_x >= 0
        and true_x < img_w
        and true_y >= 0
        and true_y < img_h
    )

    mask = lander_mask_rgb(img, purple_threshold=purple_threshold)
    detected_center = mask_center(mask)

    row = {
        "visible_expected": bool(visible_expected),
        "detected": detected_center is not None,
        "true_x": true_x,
        "true_y": true_y,
        "mask_pixels": int(mask.sum()),
        "pixel_error": None,
        "det_x": None,
        "det_y": None,
    }

    if detected_center is not None:
        det_x, det_y = detected_center
        row["det_x"] = det_x
        row["det_y"] = det_y
        row["pixel_error"] = float(
            np.sqrt((det_x - true_x) ** 2 + (det_y - true_y) ** 2)
        )

    return row


def summarize_rows(rows):
    n = len(rows)

    visible_rows = [r for r in rows if r["visible_expected"]]
    offscreen_rows = [r for r in rows if not r["visible_expected"]]
    detected_rows = [r for r in rows if r["detected"]]
    visible_detected_rows = [r for r in visible_rows if r["detected"]]

    errors_visible = [
        r["pixel_error"]
        for r in visible_detected_rows
        if r["pixel_error"] is not None
    ]

    mask_pixels = [r["mask_pixels"] for r in rows]

    return {
        "num_frames": n,
        "num_visible_expected": len(visible_rows),
        "num_offscreen_expected": len(offscreen_rows),
        "num_detected": len(detected_rows),
        "num_visible_detected": len(visible_detected_rows),
        "detection_rate_all": len(detected_rows) / max(n, 1),
        "detection_rate_visible_expected": len(visible_detected_rows) / max(len(visible_rows), 1),
        "mean_pixel_error_visible_detected": float(np.mean(errors_visible)) if errors_visible else None,
        "median_pixel_error_visible_detected": float(np.median(errors_visible)) if errors_visible else None,
        "p90_pixel_error_visible_detected": float(np.percentile(errors_visible, 90)) if errors_visible else None,
        "mean_mask_pixels": float(np.mean(mask_pixels)) if mask_pixels else None,
        "median_mask_pixels": float(np.median(mask_pixels)) if mask_pixels else None,
    }


def save_overlay_grid(examples, output_path, purple_threshold):
    """
    examples is list of dicts with:
      kind -> image/state/analysis
    """
    num_images = len(examples)
    rows = KINDS

    fig, axes = plt.subplots(
        len(rows),
        num_images + 1,
        figsize=(2 * (num_images + 1), 2 * len(rows)),
        gridspec_kw={"width_ratios": [0.9] + [1.0] * num_images},
    )

    for r, kind in enumerate(rows):
        label_ax = axes[r, 0]
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.5,
            kind,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

        for c, ex in enumerate(examples):
            ax = axes[r, c + 1]
            ax.axis("off")

            img = ex[kind]["image"]
            row = ex[kind]["analysis"]

            ax.imshow(img)

            mask = lander_mask_rgb(img, purple_threshold=purple_threshold)
            overlay = np.zeros_like(img)
            overlay[:, :, 1] = mask.astype(np.float32)
            overlay[:, :, 2] = mask.astype(np.float32)
            ax.imshow(overlay, alpha=0.45)

            ax.scatter([row["true_x"]], [row["true_y"]], marker="x", s=50)

            if row["detected"]:
                ax.scatter([row["det_x"]], [row["det_y"]], marker="o", s=30)
                if row["visible_expected"]:
                    title = f"err={row['pixel_error']:.1f}"
                else:
                    title = "offscreen+det"
            else:
                title = "not detected" if row["visible_expected"] else "offscreen"

            ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_actions", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--purple_threshold", type=float, default=0.08)
    parser.add_argument("--num_overlay_images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {})

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

    all_rows_by_kind = {kind: [] for kind in KINDS}
    overlay_examples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="batches")):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            img_t, img_next, state_t, state_next, action_t = batch

            img_t = img_t.to(device)
            img_next = img_next.to(device)
            state_t = state_t.to(device)
            state_next = state_next.to(device)
            action_t = action_t.to(device)

            # Deterministic eval: decode mu, not sampled z.
            mu_t, _ = model.encode(img_t)
            mu_next, _ = model.encode(img_next)

            recon_t = model.decode(mu_t)
            recon_next = model.decode(mu_next)

            pred_mu_next = dynamics(mu_t, action_t)
            pred_next = model.decode(pred_mu_next)

            batch_size = img_t.size(0)

            for i in range(batch_size):
                images = {
                    "real_t": tensor_to_image_np(img_t[i]),
                    "recon_t": tensor_to_image_np(recon_t[i]),
                    "real_next": tensor_to_image_np(img_next[i]),
                    "recon_next": tensor_to_image_np(recon_next[i]),
                    "pred_next": tensor_to_image_np(pred_next[i]),
                }

                states = {
                    "real_t": state_t[i].detach().cpu().numpy(),
                    "recon_t": state_t[i].detach().cpu().numpy(),
                    "real_next": state_next[i].detach().cpu().numpy(),
                    "recon_next": state_next[i].detach().cpu().numpy(),
                    "pred_next": state_next[i].detach().cpu().numpy(),
                }

                example = {}

                for kind in KINDS:
                    analysis = analyze_image(
                        img=images[kind],
                        state=states[kind],
                        purple_threshold=args.purple_threshold,
                    )
                    analysis["kind"] = kind
                    all_rows_by_kind[kind].append(analysis)

                    example[kind] = {
                        "image": images[kind],
                        "analysis": analysis,
                    }

                if len(overlay_examples) < args.num_overlay_images:
                    overlay_examples.append(example)

    summary = {
        kind: summarize_rows(rows)
        for kind, rows in all_rows_by_kind.items()
    }

    summary_path = os.path.join(args.output_dir, "segmentation_summary.json")
    rows_path = os.path.join(args.output_dir, "segmentation_rows.json")
    overlay_path = os.path.join(args.output_dir, "segmentation_overlay.png")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(rows_path, "w") as f:
        json.dump(all_rows_by_kind, f, indent=2)

    if overlay_examples:
        save_overlay_grid(
            examples=overlay_examples,
            output_path=overlay_path,
            purple_threshold=args.purple_threshold,
        )
        print("saved:", overlay_path)

    print()
    print("Checkpoint segmentation summary")
    print("-------------------------------")
    print("checkpoint:", args.checkpoint)

    for kind in KINDS:
        s = summary[kind]
        print()
        print(kind)
        print("  num_visible_expected:", s["num_visible_expected"])
        print("  detection_rate_visible_expected:", f"{s['detection_rate_visible_expected']:.3f}")
        print("  mean_pixel_error_visible_detected:", s["mean_pixel_error_visible_detected"])
        print("  median_pixel_error_visible_detected:", s["median_pixel_error_visible_detected"])
        print("  p90_pixel_error_visible_detected:", s["p90_pixel_error_visible_detected"])
        print("  mean_mask_pixels:", s["mean_mask_pixels"])

    print()
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
