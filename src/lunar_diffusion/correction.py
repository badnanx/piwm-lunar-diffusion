import torch
import torch.nn as nn


class LatentCorrectionMLP(nn.Module):
    """
    Simple MLP correction model.

    Input:
        z_pred_next

    Output:
        predicted correction

    Then:
        z_corrected_next = z_pred_next + predicted_correction
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_pred_next: torch.Tensor) -> torch.Tensor:
        return self.net(z_pred_next)
