import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamicsMLP(nn.Module):
    """
    Simple latent dynamics model.

    Input:
        latent_t: (B, latent_dim)
        action_t: (B,), integer action labels

    Output:
        predicted latent_{t+1}: (B, latent_dim)

    We use one-hot actions because Lunar Lander actions are discrete.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        input_dim = latent_dim + num_actions

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        action_one_hot = F.one_hot(action_t, num_classes=self.num_actions).float()
        x = torch.cat([latent_t, action_one_hot], dim=1)
        return self.net(x)
