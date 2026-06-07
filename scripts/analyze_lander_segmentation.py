import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lunar_diffusion.crop_loss import state_xy_to_pixel


def load_npz_files(data_dir, max_files=None):
    files = sorted(Path(data_dir).glob("*.npz"))
    if max_files is not None:
        files = files[:max_files]
    return files


def normalize_images(obs):
    """
    Accepts obs as uint8 [0,255] or float [0,1].
    Returns float32 [N,H,W,3] in [0,1].
    """
    obs = obs.astype(np.float32)
    if obs.max() > 1.5:
        obs = obs / 255.0
    return obs


def get_obs_and_states(npz_path, state_key):
    data = np.load(npz_path)

    if "obs" in data:
        image_key = "obs"
    elif "imgs" in data:
        image_key = "imgs"
    else:
        raise KeyError(f"{npz_path} missing image key. Expected obs or imgs; keys={list(data.keys())}")

    if state_key not in data:
        raise KeyError(f"{npz_path} missing {state_key}; keys={list(data.keys())}")

    obs = normalize_images(data[image_key])
    states = data[state_key].astype(np.float32)

    # Some datasets may store obs as [T, H, W, C].
    # Some may store [T, something, H, W, C]. Use first camera if needed.
    if obs.ndim == 5:
        obs = obs[:, 0]

    if obs.ndim != 4:
        raise ValueError(f"Expected obs shape [T,H,W,C], got {obs.shape}")

    return obs, states


def lander_mask_rgb(img, purple_threshold=0.08):
    """
    Simple color-based lander detector.

    Lunar Lander body is purple-ish. We detect pixels where red/blue are
    meaningfully stronger than green.

    img: [H,W,3] float in [0,1]
    returns mask: [H,W] bool
    """
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Purple/magenta-ish: red and blue high relative to green.
    purple_score = 0.5 * (r + b) - g

    # Avoid selecting dark sky noise.
    brightness = (r + g + b) / 3.0

    mask = (
        (purple_score > purple_threshold)
        & (r > g + 0.03)
        & (b > g + 0.03)
        & (brightness > 0.08)
    )

    return mask


def mask_center(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def state_center_pixels(state, img_h, img_w):
    state_t = torch.tensor(state[None, :], dtype=torch.float32)
    px, py = state_xy_to_pixel(state_t, img_h=img_h, img_w=img_w)
    return float(px[0]), float(py[0])


def analyze_file(npz_path, state_key, purple_threshold):
    obs, states = get_obs_and_states(npz_path, state_key)

    rows = []
    for t in range(len(obs)):
        img = obs[t]
        state = states[t]

        img_h, img_w = img.shape[:2]

        mask = lander_mask_rgb(img, purple_threshold=purple_threshold)
        detected = mask_center(mask)
        true_px, true_py = state_center_pixels(state, img_h, img_w)

        if detected is None:
            rows.append(
                {
                    "detected": False,
                    "pixel_error": None,
                    "det_x": None,
                    "det_y": None,
                    "true_x": true_px,
                    "true_y": true_py,
                    "mask_pixels": int(mask.sum()),
                }
            )
        else:
            det_x, det_y = detected
            pixel_error = float(np.sqrt((det_x - true_px) ** 2 + (det_y - true_py) ** 2))
            rows.append(
                {
                    "detected": True,
                    "pixel_error": pixel_error,
                    "det_x": det_x,
                    "det_y": det_y,
                    "true_x": true_px,
                    "true_y": true_py,
                    "mask_pixels": int(mask.sum()),
                }
            )

    return rows, obs, states


def save_overlay_grid(obs, states, rows, output_path, purple_threshold, max_images=12):
    n = min(max_images, len(obs))
    cols = min(6, n)
    rows_n = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows_n, cols, figsize=(2.5 * cols, 2.5 * rows_n))
    axes = np.array(axes).reshape(rows_n, cols)

    for i in range(rows_n * cols):
        ax = axes.flat[i]
        ax.axis("off")

        if i >= n:
            continue

        img = obs[i]
        state = states[i]
        result = rows[i]

        mask = lander_mask_rgb(img, purple_threshold=purple_threshold)
        ax.imshow(img)

        # Mask overlay in cyan.
        overlay = np.zeros_like(img)
        overlay[:, :, 1] = mask.astype(np.float32)
        overlay[:, :, 2] = mask.astype(np.float32)
        ax.imshow(overlay, alpha=0.45)

        true_x = result["true_x"]
        true_y = result["true_y"]
        ax.scatter([true_x], [true_y], marker="x", s=60)

        if result["detected"]:
            ax.scatter([result["det_x"]], [result["det_y"]], marker="o", s=35)
            title = f"err={result['pixel_error']:.1f}px"
        else:
            title = "not detected"

        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize(all_rows):
    n = len(all_rows)
    detected_rows = [r for r in all_rows if r["detected"]]
    detection_rate = len(detected_rows) / max(n, 1)

    errors = [r["pixel_error"] for r in detected_rows if r["pixel_error"] is not None]
    mask_pixels = [r["mask_pixels"] for r in all_rows]

    summary = {
        "num_frames": n,
        "num_detected": len(detected_rows),
        "detection_rate": detection_rate,
        "mean_pixel_error_detected": float(np.mean(errors)) if errors else None,
        "median_pixel_error_detected": float(np.median(errors)) if errors else None,
        "p90_pixel_error_detected": float(np.percentile(errors, 90)) if errors else None,
        "mean_mask_pixels": float(np.mean(mask_pixels)) if mask_pixels else None,
        "median_mask_pixels": float(np.median(mask_pixels)) if mask_pixels else None,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--state_key", default="states")
    parser.add_argument("--max_files", type=int, default=10)
    parser.add_argument("--purple_threshold", type=float, default=0.08)
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = load_npz_files(args.data_dir, max_files=args.max_files)

    all_rows = []
    overlay_candidates = []

    for file_idx, npz_path in enumerate(tqdm(files, desc="files")):
        rows, obs, states = analyze_file(
            npz_path=npz_path,
            state_key=args.state_key,
            purple_threshold=args.purple_threshold,
        )

        for r in rows:
            r["file"] = str(npz_path)
        all_rows.extend(rows)

        if args.save_overlay:
            for local_idx, row in enumerate(rows):
                overlay_candidates.append(
                    {
                        "img": obs[local_idx],
                        "state": states[local_idx],
                        "row": row,
                    }
                )

    if args.save_overlay and overlay_candidates:
        rng = np.random.default_rng(args.seed)
        n = min(12, len(overlay_candidates))
        selected_indices = rng.choice(len(overlay_candidates), size=n, replace=False)
        selected = [overlay_candidates[i] for i in selected_indices]

        overlay_obs = np.stack([item["img"] for item in selected], axis=0)
        overlay_states = np.stack([item["state"] for item in selected], axis=0)
        overlay_rows = [item["row"] for item in selected]

        overlay_path = os.path.join(args.output_dir, "segmentation_overlay.png")
        save_overlay_grid(
            obs=overlay_obs,
            states=overlay_states,
            rows=overlay_rows,
            output_path=overlay_path,
            purple_threshold=args.purple_threshold,
            max_images=12,
        )
        print("saved:", overlay_path)

    summary = summarize(all_rows)

    summary_path = os.path.join(args.output_dir, "summary.json")
    rows_path = os.path.join(args.output_dir, "rows.json")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(rows_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    print()
    print("Lander segmentation summary")
    print("---------------------------")
    print("num_frames:", summary["num_frames"])
    print("num_detected:", summary["num_detected"])
    print("detection_rate:", f"{summary['detection_rate']:.3f}")
    print("mean_pixel_error_detected:", summary["mean_pixel_error_detected"])
    print("median_pixel_error_detected:", summary["median_pixel_error_detected"])
    print("p90_pixel_error_detected:", summary["p90_pixel_error_detected"])
    print("mean_mask_pixels:", summary["mean_mask_pixels"])
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
