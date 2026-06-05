import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LunarFrameDataset(Dataset):
    """
    Loads Lunar Lander .npz trajectory files and exposes individual frames.

    Each .npz file is expected to contain:
      - imgs:   (T, H, W, 3), uint8
      - states: (T, 8), float32

    Optional state_key can be:
      - states
      - noisy_states_2
      - noisy_states_5
      - noisy_states_10
    """

    def __init__(
        self,
        data_dir: str,
        state_key: str = "states",
        max_files: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.state_key = state_key

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in data_dir: {data_dir}")

        self.index = []
        self.lengths = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "imgs" not in data:
                    raise KeyError(f"{path} is missing key 'imgs'")
                if state_key not in data:
                    raise KeyError(f"{path} is missing key '{state_key}'")

                n = data["imgs"].shape[0]
                self.lengths.append(n)

                for frame_idx in range(n):
                    self.index.append((file_idx, frame_idx))

        print(
            f"Loaded LunarFrameDataset: {len(self.files)} files, "
            f"{len(self.index)} frames, state_key={state_key}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, frame_idx = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            img = data["imgs"][frame_idx]
            state = data[self.state_key][frame_idx]

        # Image: uint8 HWC [0, 255] -> float CHW [0, 1]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        # State: numpy float32 -> torch float32
        state = torch.from_numpy(state.astype(np.float32))

        return img, state
