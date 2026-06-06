import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lunar_diffusion.visibility import is_fully_visible


class LunarVisibleFrameDataset(Dataset):
    """
    Single-frame dataset that only returns frames where the lander is fully visible.

    Each item:
        image, state

    This is useful for testing whether image -> state extraction improves
    when the lander is not clipped/off-screen.
    """

    def __init__(
        self,
        data_dir: str,
        state_key: str = "states",
        max_files: Optional[int] = None,
        margin: int = 0,
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.state_key = state_key
        self.margin = margin

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in data_dir: {self.data_dir}")

        self.index = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "imgs" not in data:
                    raise KeyError(f"{path} is missing key 'imgs'")
                if self.state_key not in data:
                    raise KeyError(f"{path} is missing key '{self.state_key}'")

                states = data[self.state_key]

                for frame_idx in range(len(states)):
                    if is_fully_visible(states[frame_idx], margin=margin):
                        self.index.append((file_idx, frame_idx))

        if len(self.index) == 0:
            raise ValueError(f"No fully visible frames found in: {self.data_dir}")

        print(
            f"Loaded LunarVisibleFrameDataset: {len(self.files)} files, "
            f"{len(self.index)} visible frames, state_key={state_key}, margin={margin}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, frame_idx = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            img = data["imgs"][frame_idx]
            state = data[self.state_key][frame_idx]

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        state = torch.from_numpy(state.astype(np.float32))

        return img, state
