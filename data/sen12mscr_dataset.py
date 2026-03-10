"""SEN12MS-CR dataset loader (raw .tif files).

Provides the original SEN12MS-CR dataset class that reads SAR, cloudy, and
cloud-free optical .tif patches directly from disk.

Reference: https://patricktum.github.io/cloud_removal/sen12mscr/
Directory structure expected:
    root/
    ├── ROIs1158_spring_s1/
    │   └── s1_1/
    │       ├── ROIs1158_spring_s1_1_p1.tif
    │       └── ...
    ├── ROIs1158_spring_s2/
    │   └── s2_1/ ...
    ├── ROIs1158_spring_s2_cloudy/
    │   └── s2_cloudy_1/ ...
    ...
"""

import os
import warnings
import numpy as np
from natsort import natsorted
from tqdm import tqdm

import rasterio
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def read_tif(path):
    return rasterio.open(path)


def read_img(tif):
    return tif.read().astype(np.float32)


def rescale(img, old_min, old_max):
    return (img - old_min) / (old_max - old_min)


def process_MS(img, method='default'):
    """Pre-process multi-spectral image."""
    if method == 'default':
        img = np.clip(img, 0, 10000)
        img = rescale(img, 0, 10000)
    elif method == 'resnet':
        img = np.clip(img, 0, 10000)
        img /= 2000
    return img


def process_SAR(img, method='default'):
    """Pre-process SAR image."""
    if method == 'default':
        img = np.clip(img, -25, 0)
        img = rescale(img, -25, 0)
    elif method == 'resnet':
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([
            (2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0])
             / (dB_max[0] - dB_min[0]))[None, ...],
            (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1])
             / (dB_max[1] - dB_min[1]))[None, ...],
        ], axis=0)
    return img


# ---------------------------------------------------------------------------
# SEN12MSCR Dataset
# ---------------------------------------------------------------------------

class SEN12MSCR(Dataset):
    """SEN12MS-CR raw dataset: single-temporal cloud removal triplets.

    Each sample contains an (S1, S2_cloudy, S2_clear) triplet of co-registered
    SAR, cloudy optical, and cloud-free optical patches.
    Returns raw dict: {'input': {'S1': ..., 'S2': ...}, 'target': {'S2': ...}}.
    """

    # Official splits (S1 seed directories)
    SPLITS = {
        'train': [
            'ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107',
            'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40',
            'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64',
            'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35',
            'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141',
            'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26',
            'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71',
            'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136',
            'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91',
            'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19',
            'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109',
            'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128',
            'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27',
            'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105',
            'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40',
            'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145',
            'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26',
            'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100',
            'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58',
            'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 'ROIs1158_spring_s1/s1_126',
            'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39',
            'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 'ROIs1158_spring_s1/s1_75',
            'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142',
            'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 'ROIs1158_spring_s1/s1_97',
            'ROIs1158_spring_s1/s1_147',
            'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25',
            'ROIs1868_summer_s1/s1_124', 'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135',
            'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42',
            'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139',
            'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55',
            'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_100',
            'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143',
            'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89',
            'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72',
            'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146',
            'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 'ROIs1868_summer_s1/s1_102',
            'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132',
            'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 'ROIs1868_summer_s1/s1_4',
            'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86',
            'ROIs1868_summer_s1/s1_47',
            'ROIs2017_winter_s1/s1_68', 'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62',
            'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64',
            'ROIs2017_winter_s1/s1_21', 'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112',
            'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59',
            'ROIs2017_winter_s1/s1_49', 'ROIs2017_winter_s1/s1_104', 'ROIs2017_winter_s1/s1_81',
            'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 'ROIs2017_winter_s1/s1_94',
            'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47',
        ],
        'val': [
            'ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65',
            'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 'ROIs1868_summer_s1/s1_80',
            'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17',
            'ROIs2017_winter_s1/s1_84',
        ],
        'test': [
            'ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140',
            'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 'ROIs1868_summer_s1/s1_119',
            'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108',
            'ROIs2017_winter_s1/s1_63',
        ],
    }

    def __init__(self, root, split='all', season='all',
                 rescale_method='default'):
        self.root_dir = root
        self.split = split
        self.season = season
        self.method = rescale_method

        assert split in ['all', 'train', 'val', 'test']
        assert season in ['all', 'spring', 'summer', 'fall', 'winter']

        splits = {k: list(v) for k, v in self.SPLITS.items()}
        splits['all'] = splits['train'] + splits['val'] + splits['test']

        if season != 'all':
            for k in splits:
                splits[k] = [r for r in splits[k] if season in r]

        self.split_rois = splits[split]
        self.paths = self._get_paths()

        if not self.paths:
            warnings.warn(
                f"No data found for split='{split}', season='{season}' "
                f"in root='{root}'. Check directory structure.")

    def _get_paths(self):
        print(f'\nProcessing paths for {self.split} split (season={self.season})')
        paths = []
        seeds_S1 = natsorted([
            d for d in os.listdir(self.root_dir)
            if '_s1' in d and not d.endswith('.tar')
        ])
        for seed in tqdm(seeds_S1, desc='Indexing ROIs'):
            rois = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois:
                roi_dir = os.path.join(self.root_dir, seed, roi)
                patches_S1 = natsorted([
                    os.path.join(roi_dir, f) for f in os.listdir(roi_dir)
                ])
                patches_S2 = [
                    p.replace('/s1', '/s2').replace('_s1', '_s2')
                    for p in patches_S1
                ]
                patches_S2_cloudy = [
                    p.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy')
                    for p in patches_S1
                ]
                for i in range(len(patches_S1)):
                    if not all(os.path.isfile(p) for p in
                               [patches_S1[i], patches_S2[i], patches_S2_cloudy[i]]):
                        continue
                    if not any(r in patches_S1[i] for r in self.split_rois):
                        continue
                    paths.append({
                        'S1': patches_S1[i],
                        'S2': patches_S2[i],
                        'S2_cloudy': patches_S2_cloudy[i],
                    })
        print(f'{len(paths)} samples found')
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        s1 = process_SAR(read_img(read_tif(p['S1'])), self.method)
        s2 = process_MS(read_img(read_tif(p['S2'])), self.method)
        s2_cloudy = process_MS(read_img(read_tif(p['S2_cloudy'])), self.method)

        np.nan_to_num(s1, copy=False)
        np.nan_to_num(s2, copy=False)
        np.nan_to_num(s2_cloudy, copy=False)

        return {
            'input': {
                'S1': s1,
                'S2': s2_cloudy,
                'S1 path': p['S1'],
                'S2 path': p['S2_cloudy'],
            },
            'target': {
                'S2': s2,
                'S2 path': p['S2'],
            },
        }


# ---------------------------------------------------------------------------
# Wrapper: adapt to ECRformer training format
# ---------------------------------------------------------------------------

class SEN12MSCR_Dataset(Dataset):
    """SEN12MS-CR dataset wrapper for ECRformer training.

    Wraps SEN12MSCR to produce the dict format expected by
    CloudRemovalModel: {SAR, cloudy, target}.

    Matches ``find_dataset_using_name('sen12mscr')``.

    Args:
        root:           Path to the SEN12MS-CR dataset root.
        split:          One of 'all', 'train', 'val', 'test'.
        data_range:     Scaling factor applied after loading (default 1.0).
        crop_size:      If set, random-crop patches to this size.
        season:         Season filter: 'all', 'spring', 'summer', 'fall', 'winter'.
        rescale_method: Pre-processing method: 'default' or 'resnet'.
    """

    def __init__(self, root, split='train', data_range=1.0, crop_size=None,
                 season='all', rescale_method='default'):
        self.dataset = SEN12MSCR(
            root, split=split, season=season, rescale_method=rescale_method)
        self.data_range = data_range
        self.crop_size = crop_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw = self.dataset[idx]
        s1 = raw['input']['S1'].astype(np.float32) * self.data_range
        cloudy = raw['input']['S2'].astype(np.float32) * self.data_range
        target = raw['target']['S2'].astype(np.float32) * self.data_range

        if self.crop_size is not None:
            h, w = s1.shape[-2:]
            cs = self.crop_size
            top = np.random.randint(0, max(h - cs, 0) + 1)
            left = np.random.randint(0, max(w - cs, 0) + 1)
            s1 = s1[..., top:top+cs, left:left+cs].copy()
            cloudy = cloudy[..., top:top+cs, left:left+cs].copy()
            target = target[..., top:top+cs, left:left+cs].copy()

        return {
            'SAR': s1,
            'cloudy': cloudy,
            'target': target,
        }
