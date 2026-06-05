import argparse
import glob
import os
import numpy as np

def describe(name, arr):
    msg = f"  {name}: shape={arr.shape}, dtype={arr.dtype}"
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        msg += f", min={arr.min():.4f}, max={arr.max():.4f}"
    print(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    print("data_dir:", args.data_dir)
    print("num_files:", len(files))

    total_frames = 0
    lengths = []

    for path in files[:args.max_files]:
        print("\nFILE:", os.path.basename(path))
        with np.load(path) as data:
            print("keys:", data.files)
            for k in data.files:
                describe(k, data[k])
            if "imgs" in data:
                total_frames += data["imgs"].shape[0]
                lengths.append(data["imgs"].shape[0])

    full_total = 0
    for path in files:
        with np.load(path) as data:
            if "imgs" in data:
                full_total += data["imgs"].shape[0]

    print("\nfirst_lengths:", lengths)
    print("total_frames:", full_total)

if __name__ == "__main__":
    main()
