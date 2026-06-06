import argparse
import json


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


METRIC_KEYS = [
    "p1_t_rmse",
    "p1_next_rmse",
    "dynamics_state_rmse",
    "dynamics_latent_rmse",
    "p2_rmse",
]


def fmt(x):
    if x is None:
        return "-"
    return f"{x:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    with open(args.metrics, "r") as f:
        metrics = json.load(f)

    state_indices = metrics.get("state_indices")
    state_names = metrics.get("state_names")

    if state_names is None and state_indices is not None:
        state_names = [STATE_NAMES.get(i, f"state_{i}") for i in state_indices]

    print()
    print("PIWM evaluation summary")
    print("-----------------------")
    print(f"file: {args.metrics}")
    print(f"num_pairs: {metrics.get('num_pairs', '-')}")
    print(f"state_key: {metrics.get('state_key', '-')}")
    print(f"state_indices: {state_indices}")
    print(f"state_names: {state_names}")

    print()
    print("Mean metrics")
    print("------------")

    means = metrics.get("means", {})
    for key in METRIC_KEYS:
        if key in means:
            print(f"{key}: {means[key]:.6f}")
        elif key in metrics and isinstance(metrics[key], (int, float)):
            print(f"{key}: {metrics[key]:.6f}")

    if not state_names:
        print()
        print("No state_names found; cannot print per-dimension table.")
        return

    print()
    print("Per-dimension RMSE")
    print("------------------")

    header = ["dim", "name"] + [key for key in METRIC_KEYS if key in metrics and isinstance(metrics[key], list)]
    widths = [max(len(h), 4) for h in header]

    rows = []
    for i, name in enumerate(state_names):
        row = [str(i), name]
        for key in header[2:]:
            values = metrics.get(key)
            value = values[i] if values is not None and i < len(values) else None
            row.append(fmt(value))
        rows.append(row)
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  ".join("-" * widths[i] for i in range(len(header))))

    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    print()
    print("Interpretation hints")
    print("--------------------")
    print("p1_t_rmse: encoder physical dims vs state_t")
    print("p1_next_rmse: encoder physical dims vs state_next")
    print("dynamics_state_rmse: dynamics predicted physical dims vs state_next")
    print("dynamics_latent_rmse: dynamics predicted latent vs encoder next latent")
    print("p2_rmse: physical latent delta consistency")


if __name__ == "__main__":
    main()
