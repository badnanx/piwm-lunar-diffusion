import argparse
import json
import os

import matplotlib.pyplot as plt


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)


def get_metric(history, split, metric):
    epochs = []
    values = []

    for row in history:
        if split in row and metric in row[split]:
            epochs.append(row["epoch"])
            values.append(row[split][metric])

    return epochs, values


def plot_metric(history, metric, output_dir):
    train_epochs, train_values = get_metric(history, "train", metric)
    test_epochs, test_values = get_metric(history, "test", metric)

    if len(train_values) == 0 and len(test_values) == 0:
        print(f"skipping {metric}: not found")
        return

    plt.figure(figsize=(8, 5))

    if len(train_values) > 0:
        plt.plot(train_epochs, train_values, marker="o", label=f"train {metric}")

    if len(test_values) > 0:
        plt.plot(test_epochs, test_values, marker="o", label=f"test {metric}")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("saved:", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    history = load_history(args.history)

    metrics = [
        "loss",
        "recon_loss",
        "pred_recon_loss",
        "kl_loss",
        "p1_loss",
        "p2_loss",
        "dynamics_loss",
        "dynamics_latent_loss",
        "dynamics_state_loss",
        "state_loss",
    ]

    for metric in metrics:
        plot_metric(history, metric, args.output_dir)


if __name__ == "__main__":
    main()
