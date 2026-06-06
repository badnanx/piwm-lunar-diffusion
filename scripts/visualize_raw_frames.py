import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/raw_frames")
    parser.add_argument("--num_files", type=int, default=3)
    parser.add_argument("--frames_per_file", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[: args.num_files]

    for file_path in files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        with np.load(file_path) as data:
            imgs = data["imgs"]
            states = data["states"]

            frame_indices = np.linspace(
                0, len(imgs) - 1, args.frames_per_file, dtype=int
            )

            for frame_idx in frame_indices:
                img = imgs[frame_idx]
                state = states[frame_idx]

                save_path = os.path.join(
                    args.output_dir, f"{file_name}_frame_{frame_idx:04d}.png"
                )

                plt.figure(figsize=(9, 6))
                plt.imshow(img)
                plt.axis("off")
                plt.title(
                    f"{file_name}, frame {frame_idx}\\n"
                    f"x={state[0]:.3f}, y={state[1]:.3f}, "
                    f"vx={state[2]:.3f}, vy={state[3]:.3f}, "
                    f"theta={state[4]:.3f}"
                )
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()

                print("saved:", save_path)


if __name__ == "__main__":
    main()
