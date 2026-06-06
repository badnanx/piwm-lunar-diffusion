import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Small convolutional VAE for Lunar Lander images.

    Input image shape:
        (B, 3, 100, 150)

    Latent shape:
        (B, latent_dim)

    Output reconstruction shape:
        (B, 3, 100, 150)

    PIWM-style idea:
        Some dimensions of mu can be supervised to match selected
        physical state variables, e.g. x, y, theta.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: image -> compressed feature map
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),     # 100x150 -> 50x75
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 50x75 -> 25x37
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 25x37 -> 12x18
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 12x18 -> 6x9
            nn.ReLU(),
        )

        self.feature_dim = 256 * 6 * 9

        # VAE latent distribution parameters
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # Decoder: latent -> reconstructed image
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6x9 -> 12x18
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 12x18 -> 24x36
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 24x36 -> 48x72
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 48x72 -> 96x144
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.reshape(z.size(0), 256, 6, 9)
        recon = self.decoder_conv(h)

        # Decoder naturally gives 96x144, so resize back to 100x150.
        recon = F.interpolate(recon, size=(100, 150), mode="bilinear", align_corners=False)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon,
    x,
    mu,
    logvar,
    states=None,
    state_indices=None,
    kl_weight: float = 1e-4,
    state_weight: float = 0.0,
):
    """
    VAE loss.

    recon_loss:
        Makes reconstructed image look like input image.

    kl_loss:
        Keeps the latent distribution close to standard normal.

    state_loss:
        PIWM-style supervision. Forces the first k dimensions of mu
        to match selected physical state variables.

        Example:
            state_indices = [0, 1, 4]

        Then:
            mu[:, 0] -> x
            mu[:, 1] -> y
            mu[:, 2] -> theta
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    if states is not None and state_weight > 0.0:
        if state_indices is None:
            raise ValueError("state_indices must be provided when state_weight > 0.")

        if len(state_indices) > mu.size(1):
            raise ValueError(
                f"Need at least {len(state_indices)} latent dimensions, "
                f"but latent_dim is {mu.size(1)}."
            )

        target_state = states[:, state_indices]
        predicted_state = mu[:, : len(state_indices)]
        state_loss = F.mse_loss(predicted_state, target_state, reduction="mean")
    else:
        state_loss = torch.zeros((), device=x.device)

    total_loss = recon_loss + kl_weight * kl_loss + state_weight * state_loss

    return total_loss, recon_loss, kl_loss, state_loss
