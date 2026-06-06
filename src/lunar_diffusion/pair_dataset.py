import glob
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LunarPairDataset(Dataset):
    """
    Loads Lunar Lander .npz trajectory files and exposes consecutive frame pairs.

    Each item is:
        img_t, img_next, state_t, state_next, action_t

    Expected .npz keys:
        imgs:   (T, H, W, 3), uint8
        acts:   (T,), int action labels, usually 0..3
        states: (T, 8), float32

    Pairing:
        frame t      -> frame t+1
        state t      -> state t+1
        action_t     -> action taken at t

    This is the dataset we need for PIWM Principle 2 and dynamics.
    """

    def __init__(
        self,
        data_dir: str,
        state_key: str = "states",
        action_key: str = "acts",
        max_files: Optional[int] = None,
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.state_key = state_key
        self.action_key = action_key

        self.files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
        if max_files is not None:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in data_dir: {self.data_dir}")

        self.index = []
        self.lengths = []

        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "imgs" not in data:
                    raise KeyError(f"{path} is missing key 'imgs'")
                if self.state_key not in data:
                    raise KeyError(f"{path} is missing key '{self.state_key}'")
                if self.action_key not in data:
                    raise KeyError(f"{path} is missing key '{self.action_key}'")

                num_frames = data["imgs"].shape[0]
                num_actions = data[self.action_key].shape[0]

                # We need t and t+1. Usually acts has same length as imgs in your data.
                # Safe usable pairs are t = 0 ... min(T-1, A)-1.
                num_pairs = min(num_frames - 1, num_actions)

                if num_pairs <= 0:
                    continue

                self.lengths.append(num_pairs)

                for t in range(num_pairs):
                    self.index.append((file_idx, t))

        if len(self.index) == 0:
            raise ValueError(f"No valid consecutive pairs found in: {self.data_dir}")

        print(
            f"Loaded LunarPairDataset: {len(self.files)} files, "
            f"{len(self.index)} pairs, state_key={state_key}, action_key={action_key}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, t = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            img_t = data["imgs"][t]
            img_next = data["imgs"][t + 1]
            state_t = data[self.state_key][t]
            state_next = data[self.state_key][t + 1]
            action_t = data[self.action_key][t]

        # Images: uint8 HWC [0, 255] -> float CHW [0, 1]
        img_t = img_t.astype(np.float32) / 255.0
        img_next = img_next.astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_t).permute(2, 0, 1)
        img_next = torch.from_numpy(img_next).permute(2, 0, 1)

        # States: numpy float32 -> torch float32
        state_t = torch.from_numpy(state_t.astype(np.float32))
        state_next = torch.from_numpy(state_next.astype(np.float32))

        # Action: scalar int -> torch long
        action_t = torch.tensor(action_t, dtype=torch.long)

        return img_t, img_next, state_t, state_next, action_t
