import torch
import torch.nn as nn


class StateExtractorCNN(nn.Module):
    """
    CNN that predicts selected physical state variables from an image.

    This is an early P4 / constraint-checker component:
        image -> physical state estimate
    """

    def __init__(self, output_dim: int = 3):
        super().__init__()

        self.conv = nn.Sequential(
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

        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.reshape(h.size(0), -1)
        return self.head(h)
