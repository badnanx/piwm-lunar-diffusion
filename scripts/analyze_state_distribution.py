import argparse
import glob
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))

    all_states = []
    for path in files:
        with np.load(path) as data:
            all_states.append(data["states"])

    states = np.concatenate(all_states, axis=0)
    y = states[:, 1]

    print("num_frames:", len(y))
    print("y min:", y.min())
    print("y max:", y.max())
    print("y mean:", y.mean())
    print("y std:", y.std())

    for threshold in [1.4, 1.25, 1.0, 0.5, 0.2, 0.0]:
        count = np.sum(y > threshold)
        pct = 100 * count / len(y)
        print(f"y > {threshold:4.2f}: {count:6d} frames ({pct:5.2f}%)")

    print()
    print("percentiles:")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"{p:3d}%: {np.percentile(y, p):.4f}")


if __name__ == "__main__":
    main()
