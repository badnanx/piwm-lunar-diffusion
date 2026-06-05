import argparse

import torch
from torch.utils.data import DataLoader

from lunar_diffusion.dataset import LunarFrameDataset
from lunar_diffusion.autoencoder import ConvVAE, vae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--max_files", type=int, default=5)
    args = parser.parse_args()

    dataset = LunarFrameDataset(args.data_dir, max_files=args.max_files)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    images, states = next(iter(loader))

    model = ConvVAE(latent_dim=args.latent_dim)

    recon, mu, logvar = model(images)
    loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)

    print("Autoencoder forward pass successful.")
    print("input images shape:", images.shape)
    print("recon images shape:", recon.shape)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)
    print("loss:", loss.item())
    print("recon_loss:", recon_loss.item())
    print("kl_loss:", kl_loss.item())


if __name__ == "__main__":
    main()
