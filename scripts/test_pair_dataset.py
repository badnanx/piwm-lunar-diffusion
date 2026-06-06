import argparse

from torch.utils.data import DataLoader

from lunar_diffusion.pair_dataset import LunarPairDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    dataset = LunarPairDataset(
        data_dir=args.data_dir,
        state_key="states",
        action_key="acts",
        max_files=args.max_files,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    img_t, img_next, state_t, state_next, action_t = next(iter(loader))

    print("Pair batch loaded successfully.")
    print("img_t shape:", img_t.shape)
    print("img_next shape:", img_next.shape)
    print("state_t shape:", state_t.shape)
    print("state_next shape:", state_next.shape)
    print("action_t shape:", action_t.shape)
    print("action_t dtype:", action_t.dtype)
    print("first state_t:", state_t[0])
    print("first state_next:", state_next[0])
    print("first action_t:", action_t[0])


if __name__ == "__main__":
    main()
