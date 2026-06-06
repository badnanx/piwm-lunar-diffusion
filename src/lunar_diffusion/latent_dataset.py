import numpy as np
import torch
from torch.utils.data import Dataset


class LatentTransitionDataset(Dataset):
    """
    Dataset for latent transition correction.

    Expected .npz keys:
        z_t
        z_next
        z_pred_next
        correction
        state_t
        state_next
        action_t

    Main correction task:
        input:  z_pred_next
        target: z_next

    Equivalent residual task:
        input:  z_pred_next
        target: correction = z_next - z_pred_next
    """

    def __init__(self, npz_path: str):
        self.npz_path = npz_path

        with np.load(npz_path) as data:
            self.z_t = data["z_t"].astype(np.float32)
            self.z_next = data["z_next"].astype(np.float32)
            self.z_pred_next = data["z_pred_next"].astype(np.float32)
            self.correction = data["correction"].astype(np.float32)
            self.state_t = data["state_t"].astype(np.float32)
            self.state_next = data["state_next"].astype(np.float32)
            self.action_t = data["action_t"].astype(np.int64)

        n = self.z_t.shape[0]

        for name, arr in [
            ("z_next", self.z_next),
            ("z_pred_next", self.z_pred_next),
            ("correction", self.correction),
            ("state_t", self.state_t),
            ("state_next", self.state_next),
            ("action_t", self.action_t),
        ]:
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} has length {arr.shape[0]}, expected {n}"
                )

        print(
            f"Loaded LatentTransitionDataset: {npz_path}, "
            f"{n} transitions, latent_dim={self.z_t.shape[1]}"
        )

    def __len__(self):
        return self.z_t.shape[0]

    def __getitem__(self, idx):
        return {
            "z_t": torch.from_numpy(self.z_t[idx]),
            "z_next": torch.from_numpy(self.z_next[idx]),
            "z_pred_next": torch.from_numpy(self.z_pred_next[idx]),
            "correction": torch.from_numpy(self.correction[idx]),
            "state_t": torch.from_numpy(self.state_t[idx]),
            "state_next": torch.from_numpy(self.state_next[idx]),
            "action_t": torch.tensor(self.action_t[idx], dtype=torch.long),
        }
