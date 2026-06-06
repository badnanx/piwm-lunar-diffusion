import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from lunar_diffusion.autoencoder import ConvVAE


# Renderer constants, copied from the Slack renderer.
SCALE = 30.0
VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE
HALF_W = W / 2.0
HALF_H = H / 2.0

LEG_DOWN = 18
HELIPAD_Y = H / 4.0
Y_OFFSET = HELIPAD_Y + LEG_DOWN / SCALE

LANDER_POLY = (
    np.array(
        [
            (-14, +17),
            (-17, 0),
            (-17, -10),
            (+17, -10),
            (+17, 0),
            (+14, +17),
        ],
        dtype=np.float32,
    )
    / SCALE
)

LEG_AWAY = 20 / SCALE
LEG_LEN = 18 / SCALE


def state_to_world(state):
    x = state[0] * HALF_W + HALF_W
    y = state[1] * HALF_H + Y_OFFSET
    angle = state[4]
    return x, y, angle


def world_to_pixel(x, y, img_w, img_h):
    px = x / W * img_w
    py = img_h - (y / H * img_h)
    return px, py


def rotate_points(points, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ rotation.T


def lander_pixel_bounds(state, img_h=100, img_w=150):
    x, y, angle = state_to_world(state)

    points_local = [LANDER_POLY]

    for sign in (-1.0, 1.0):
        hip_local = np.array([[sign * LEG_AWAY, -0.1]], dtype=np.float32)
        foot_local = np.array([[sign * (LEG_AWAY + 0.15), -LEG_LEN]], dtype=np.float32)
        points_local.extend([hip_local, foot_local])

    all_world_points = []
    for local in points_local:
        rotated = rotate_points(local, angle)
        world = rotated + np.array([x, y], dtype=np.float32)
        all_world_points.append(world)

    all_world_points = np.concatenate(all_world_points, axis=0)

    pixel_points = np.array(
        [world_to_pixel(px, py, img_w, img_h) for px, py in all_world_points],
        dtype=np.float32,
    )

    left = float(pixel_points[:, 0].min())
    right = float(pixel_points[:, 0].max())
    top = float(pixel_points[:, 1].min())
    bottom = float(pixel_points[:, 1].max())

    return left, right, top, bottom


def is_fully_visible(state, img_h=100, img_w=150, margin=0):
    left, right, top, bottom = lander_pixel_bounds(state, img_h=img_h, img_w=img_w)
    return (
        left >= margin
        and right < img_w - margin
        and top >= margin
        and bottom < img_h - margin
    )


def collect_visible_examples(data_dir, num_images):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    examples = []

    for path in files:
        with np.load(path) as data:
            imgs = data["imgs"]
            states = data["states"]

            for frame_idx in range(len(imgs)):
                state = states[frame_idx]
                if is_fully_visible(state):
                    examples.append(
                        {
                            "img": imgs[frame_idx],
                            "state": state,
                            "file": os.path.basename(path),
                            "frame_idx": frame_idx,
                        }
                    )

                if len(examples) >= num_images:
                    return examples

    return examples


def save_grid(examples, recon, save_path):
    n = len(examples)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i, ex in enumerate(examples):
        real_img = ex["img"]
        recon_img = recon[i].detach().cpu().permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0.0, 1.0)

        axes[0, i].imshow(real_img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"real\n{ex['file']}:{ex['frame_idx']}")

        axes[1, i].imshow(recon_img)
        axes[1, i].axis("off")
        axes[1, i].set_title("recon")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_path", default="outputs/visible_recon_grid.png")
    parser.add_argument("--num_images", type=int, default=12)
    parser.add_argument("--latent_dim", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = collect_visible_examples(args.data_dir, args.num_images)
    print(f"collected {len(examples)} visible examples")

    if len(examples) == 0:
        raise RuntimeError("No visible examples found.")

    imgs = np.stack([ex["img"] for ex in examples], axis=0)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        recon, _, _ = model(imgs)

    save_grid(examples, recon, args.output_path)
    print("saved:", args.output_path)


if __name__ == "__main__":
    main()
