import argparse

from torch.utils.data import DataLoader

from lunar_diffusion.dataset import LunarFrameDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    dataset = LunarFrameDataset(
        data_dir=args.data_dir,
        state_key="states",
        max_files=args.max_files,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    images, states = next(iter(loader))

    print("Batch loaded successfully.")
    print("images shape:", images.shape)
    print("images dtype:", images.dtype)
    print("images min/max:", images.min().item(), images.max().item())
    print("states shape:", states.shape)
    print("states dtype:", states.dtype)
    print("first state:", states[0])


if __name__ == "__main__":
    main()
