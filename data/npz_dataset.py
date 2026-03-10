import os
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any


class NPZ_Dataset(Dataset):
    """Dataset for SEN12MS-CR stored in .npz format.

    Each .npz file contains:
        s1:    SAR images (2 channels)
        s2:    Cloudy optical images (13 channels)
        label: Cloud-free target images (13 channels)
        masks: Cloud masks (4 channels)
        paths: File paths
    """

    def __init__(self, root: str, split: str = 'all', data_range=1.0, crop_size=None):
        super().__init__()
        self.root = root
        self.split = split
        self.data_range = data_range
        self.crop_size = crop_size
        self.load_file()

    def load_file(self):
        if 'train' in self.split or 'test' in self.split:
            print(f'Processing paths for {self.split} split')
        npz_file_path = os.path.join(self.root, f"{self.split}.npz")
        npz_file = np.load(npz_file_path, allow_pickle=True)
        self.sar_list = npz_file['s1']
        self.cloudy_list = npz_file['s2']
        self.target_list = npz_file['label']
        self.masks_list = npz_file['masks']
        self.paths_list = npz_file['paths']
        length_list = [len(l) for l in [self.sar_list, self.cloudy_list,
                                        self.target_list, self.masks_list, self.paths_list]]
        assert max(length_list) == min(length_list), "数据集长度不一致！"
        if 'train' in self.split or 'test' in self.split:
            print(f'{len(self)} samples loaded, with SAR: {self.sar_list[0].shape}, '
                  f'Cloudy: {self.cloudy_list[0].shape}, Target: {self.target_list[0].shape}, '
                  f'masks: {self.masks_list[0].shape}')

    def __len__(self) -> int:
        return len(self.sar_list)

    def __getitem__(self, idx, return_path=False) -> Dict[str, Any]:
        sample = {
            "SAR": self.sar_list[idx].astype(np.float32) * self.data_range,
            "cloudy": self.cloudy_list[idx].astype(np.float32) * self.data_range,
            "target": self.target_list[idx].astype(np.float32) * self.data_range,
            "masks": self.masks_list[idx].astype(np.float32),
        }

        if self.crop_size is not None:
            h, w = sample["SAR"].shape[-2:]
            ch, cw = self.crop_size, self.crop_size
            if h > ch:
                top = np.random.randint(0, h - ch)
            else:
                top = 0
                ch = h
            if w > cw:
                left = np.random.randint(0, w - cw)
            else:
                left = 0
                cw = w
            for key in sample:
                sample[key] = sample[key][..., top:top + ch, left:left + cw].copy()

        if return_path:
            sample["paths"] = self.paths_list[idx]

        return sample
