#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LuojiaSET Dataset Implementation
基于SEN12MSCR数据集读取方法，为LuojiaSET数据集实现的PyTorch Dataset类

LuojiaSET数据集组织结构：
- 按云覆盖率分层：0-20%, 20%-40%, 40%-60%, 60%-80%, 80%-100%
- 五种数据模态：s1/ (SAR), s2/ (清晰光学), s2_cloudy/ (有云光学), 
  cloud_detection_results/ (云检测结果), land_cover_maps/ (土地覆盖图)
- 文件命名：ROIs_XX_s[1|2|2_cloudy]_pYYYY.tif
"""

import os
import glob
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window


class LuojiaSET_Dataset(Dataset):
    """
    LuojiaSET数据集的PyTorch Dataset实现

    参考SEN12MSCR的实现方法，适配LuojiaSET的云覆盖率分层组织结构
    """

    # 数据集常量
    CLOUD_COVERAGE_DIRS = ['0-20%', '20%-40%',
                           '40%-60%', '60%-80%', '80%-100%']
    DATA_MODALITIES = ['s1', 's2', 's2_cloudy',
                       'cloud_detection_results', 'land_cover_maps']

    # 光谱波段配置 (参考SEN12MSCR)
    S2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05',
                'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    S1_BANDS = ['VV', 'VH']

    def __init__(self,
                 root_dir: str,
                 cloud_coverage_ranges: Optional[List[str]] = None,
                 modalities: Optional[List[str]] = None,
                 patch_size: int = 256,
                 use_random_crop: bool = False,
                 normalize: bool = True,
                 transform=None):
        """
        初始化LuojiaSET数据集

        Args:
            root_dir: 数据集根目录路径 (包含云覆盖率目录)
            cloud_coverage_ranges: 要使用的云覆盖率范围列表，None表示使用全部
            modalities: 要加载的数据模态列表，None表示使用全部
            patch_size: 图像patch尺寸
            use_random_crop: 是否使用随机裁剪
            normalize: 是否进行归一化
            transform: 额外的数据变换
        """
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.use_random_crop = use_random_crop
        self.normalize = normalize
        self.transform = transform

        # 设置要使用的云覆盖率范围和模态
        self.cloud_coverage_ranges = cloud_coverage_ranges or self.CLOUD_COVERAGE_DIRS
        self.modalities = modalities or self.DATA_MODALITIES

        # 验证输入参数
        self._validate_inputs()

        # 扫描并构建数据索引
        self.data_index = self._build_data_index()

        print(f"LuojiasetDataset initialized:")
        print(f"  Root directory: {self.root_dir}")
        print(f"  Cloud coverage ranges: {self.cloud_coverage_ranges}")
        print(f"  Modalities: {self.modalities}")
        print(f"  Total samples: {len(self.data_index)}")

    def _validate_inputs(self):
        """验证输入参数的有效性"""
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")

        for coverage in self.cloud_coverage_ranges:
            if coverage not in self.CLOUD_COVERAGE_DIRS:
                raise ValueError(f"Invalid cloud coverage range: {coverage}")

        for modality in self.modalities:
            if modality not in self.DATA_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")

    def _build_data_index(self) -> List[Dict]:
        """
        构建数据索引，扫描所有可用的数据文件

        Returns:
            包含每个样本文件路径信息的字典列表
        """
        data_index = []

        for coverage_range in self.cloud_coverage_ranges:
            coverage_dir = self.root_dir / coverage_range
            if not coverage_dir.exists():
                print(f"Warning: Coverage directory not found: {coverage_dir}")
                continue

            # 以s2模态为基准，扫描所有可用的patch
            s2_dir = coverage_dir / 's2'
            if not s2_dir.exists():
                print(f"Warning: s2 directory not found in {coverage_dir}")
                continue

            s2_files = sorted(glob.glob(str(s2_dir / "ROIs_*_s2_p*.tif")))

            for s2_file in s2_files:
                s2_path = Path(s2_file)
                # 从文件名提取ROI和patch信息
                filename_parts = s2_path.stem.split('_')
                if len(filename_parts) >= 4:
                    roi_id = filename_parts[1]  # 例如 "03"
                    patch_id = filename_parts[3]  # 例如 "p1104"

                    # 构建该patch的所有模态文件路径
                    sample_info = {
                        'coverage_range': coverage_range,
                        'roi_id': roi_id,
                        'patch_id': patch_id,
                        'files': {}
                    }

                    # 检查各个模态的文件是否存在
                    all_files_exist = True
                    for modality in self.modalities:
                        if modality == 's1':
                            file_pattern = f"ROIs_{roi_id}_s1_{patch_id}.tif"
                        elif modality == 's2':
                            file_pattern = f"ROIs_{roi_id}_s2_{patch_id}.tif"
                        elif modality == 's2_cloudy':
                            file_pattern = f"ROIs_{roi_id}_s2_cloudy_{patch_id}.tif"
                        elif modality == 'cloud_detection_results':
                            file_pattern = f"ROIs_{roi_id}_s2_cloudy_{patch_id}.tif"
                        elif modality == 'land_cover_maps':
                            file_pattern = f"ROIs_{roi_id}_s2_{patch_id}.tif"
                        else:
                            continue

                        file_path = coverage_dir / modality / file_pattern
                        if file_path.exists():
                            sample_info['files'][modality] = file_path
                        else:
                            all_files_exist = False
                            break

                    # 只有当所有需要的模态文件都存在时才添加到索引中
                    if all_files_exist:
                        data_index.append(sample_info)

        return data_index

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的数据样本

        Args:
            idx: 数据索引

        Returns:
            包含各模态数据的字典
        """
        sample_info = self.data_index[idx]
        sample = {}

        # 加载各个模态的数据
        for modality in self.modalities:
            if modality in sample_info['files']:
                file_path = sample_info['files'][modality]

                if modality in ['s1', 's2', 's2_cloudy']:
                    # 多光谱/SAR数据
                    data = self._read_tif(file_path)
                    if self.normalize:
                        if modality == 's1':
                            data = self._process_SAR(data)
                        else:
                            data = self._process_MS(data)
                else:
                    # 单波段数据 (云检测结果、土地覆盖图等)
                    data = self._read_tif(file_path)

                # sample[modality] = data
                if modality == 's1':
                    sample['SAR'] = data.astype(np.float32)
                elif modality == 's2':
                    sample['target'] = data.astype(np.float32)
                elif modality == 's2_cloudy':
                    sample['cloudy'] = data.astype(np.float32)
                elif modality == 'cloud_detection_results':
                    sample['masks'] = data.astype(np.float32)
                elif modality == 'land_cover_maps':
                    sample['land_cover'] = data.astype(np.float32)

        # 添加元数据
        sample['metadata'] = {
            'coverage_range': sample_info['coverage_range'],
            'roi_id': sample_info['roi_id'],
            'patch_id': sample_info['patch_id']
        }

        # 应用额外的变换
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _read_tif(self, file_path: Path) -> np.ndarray:
        """
        读取TIF文件 (参考SEN12MSCR实现)

        Args:
            file_path: TIF文件路径

        Returns:
            图像数据数组，形状为 (C, H, W)
        """
        try:
            with rasterio.open(file_path, 'r') as src:
                if self.use_random_crop and (src.width > self.patch_size or src.height > self.patch_size):
                    # 随机裁剪
                    max_row = max(0, src.height - self.patch_size)
                    max_col = max(0, src.width - self.patch_size)
                    row_off = random.randint(0, max_row)
                    col_off = random.randint(0, max_col)

                    window = Window(col_off, row_off,
                                    min(self.patch_size, src.width - col_off),
                                    min(self.patch_size, src.height - row_off))
                    data = src.read(window=window)
                else:
                    # 读取全部数据
                    data = src.read()

                # 处理无效值
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                return data.astype(np.float32)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # 返回默认形状的零数组
            if 's1' in str(file_path):
                return np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)
            else:
                return np.zeros((12, self.patch_size, self.patch_size), dtype=np.float32)

    def _process_MS(self, data: np.ndarray) -> np.ndarray:
        """
        多光谱数据预处理 (参考SEN12MSCR实现)

        Args:
            data: 原始多光谱数据

        Returns:
            处理后的数据
        """
        # 归一化到 [0, 1] 范围
        # Sentinel-2 L1C产品的典型值范围是0-10000
        data = np.clip(data / 10000.0, 0, 1)
        return data

    def _process_SAR(self, data: np.ndarray) -> np.ndarray:
        """
        SAR数据预处理 (参考SEN12MSCR实现)

        Args:
            data: 原始SAR数据

        Returns:
            处理后的数据
        """
        # SAR数据通常需要对数变换和归一化
        # 避免对零值取对数
        # data = np.where(data > 0, data, 1e-6)
        # data = 10 * np.log10(data)

        # 归一化到合理范围
        data = np.clip(data, -25, 0)  # 假设范围 [-30, 20] dB
        data = (data + 25) / 25.0  # 归一化到 [0, 1]
        return data

    def get_sample_info(self, idx: int) -> Dict:
        """
        获取指定索引样本的元数据信息

        Args:
            idx: 数据索引

        Returns:
            样本的元数据信息
        """
        return self.data_index[idx]

    def filter_by_coverage(self, coverage_ranges: List[str]) -> 'LuojiaSET_Dataset':
        """
        根据云覆盖率范围过滤数据集

        Args:
            coverage_ranges: 要保留的云覆盖率范围列表

        Returns:
            过滤后的新数据集实例
        """
        filtered_dataset = LuojiaSET_Dataset(
            root_dir=str(self.root_dir),
            cloud_coverage_ranges=coverage_ranges,
            modalities=self.modalities,
            patch_size=self.patch_size,
            use_random_crop=self.use_random_crop,
            normalize=self.normalize,
            transform=self.transform
        )
        return filtered_dataset

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息

        Returns:
            包含数据集统计信息的字典
        """
        stats = {
            'total_samples': len(self.data_index),
            'coverage_distribution': {},
            'roi_distribution': {},
            'modalities': self.modalities
        }

        # 统计云覆盖率分布
        for sample in self.data_index:
            coverage = sample['coverage_range']
            stats['coverage_distribution'][coverage] = stats['coverage_distribution'].get(
                coverage, 0) + 1

            roi = sample['roi_id']
            stats['roi_distribution'][roi] = stats['roi_distribution'].get(
                roi, 0) + 1

        return stats


class LuojiaSET_Subset(Dataset):
    """
    LuojiaSET数据集的子集类，用于数据集划分
    """

    def __init__(self, dataset: 'LuojiaSET_Dataset', indices: List[int]):
        """
        初始化数据集子集

        Args:
            dataset: 原始数据集
            indices: 子集包含的索引列表
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # 将子集索引映射到原始数据集索引
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

    def get_statistics(self) -> Dict:
        """获取子集的统计信息"""
        stats = {
            'total_samples': len(self.indices),
            'coverage_distribution': {},
            'roi_distribution': {},
            'modalities': self.dataset.modalities
        }

        # 统计云覆盖率和ROI分布
        for idx in self.indices:
            sample_info = self.dataset.data_index[idx]
            coverage = sample_info['coverage_range']
            roi = sample_info['roi_id']

            stats['coverage_distribution'][coverage] = stats['coverage_distribution'].get(
                coverage, 0) + 1
            stats['roi_distribution'][roi] = stats['roi_distribution'].get(
                roi, 0) + 1

        return stats


def create_luojiaset_datasets(root_dir: str,
                              train_ratio: float = 0.7,
                              val_ratio: float = 0.15,
                              test_ratio: float = 0.15,
                              random_seed: int = 42,
                              **kwargs) -> Tuple[LuojiaSET_Dataset, LuojiaSET_Dataset, LuojiaSET_Dataset]:
    """
    创建训练、验证和测试数据集（按比例随机划分，确保云覆盖率分布均匀）

    Args:
        root_dir: 数据集根目录
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.15
        test_ratio: 测试集比例，默认0.15
        random_seed: 随机种子，默认42
        **kwargs: 传递给Dataset的其他参数

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    import random

    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"比例之和必须等于1.0，当前为: {train_ratio + val_ratio + test_ratio}")

    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 创建完整数据集获取所有样本索引
    full_dataset = LuojiaSET_Dataset(root_dir, **kwargs)
    total_samples = len(full_dataset)

    if total_samples == 0:
        raise ValueError("数据集为空，请检查数据路径")

    print(f"总样本数: {total_samples}")
    print(f"划分比例 - 训练集: {train_ratio}, 验证集: {val_ratio}, 测试集: {test_ratio}")

    # 按云覆盖率分层采样，确保每个子集的云覆盖率分布均匀
    coverage_samples = {}
    for i, sample_info in enumerate(full_dataset.data_index):
        coverage = sample_info['coverage_range']
        if coverage not in coverage_samples:
            coverage_samples[coverage] = []
        coverage_samples[coverage].append(i)

    print("原始云覆盖率分布:")
    for coverage, samples in coverage_samples.items():
        print(f"  {coverage}: {len(samples)} 样本")

    # 对每个云覆盖率范围内的样本进行随机划分
    train_indices = []
    val_indices = []
    test_indices = []

    for coverage, samples in coverage_samples.items():
        # 随机打乱该云覆盖率范围内的样本
        random.shuffle(samples)

        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val  # 剩余的分配给测试集

        train_indices.extend(samples[:n_train])
        val_indices.extend(samples[n_train:n_train + n_val])
        test_indices.extend(samples[n_train + n_val:])

        print(f"  {coverage} 划分: 训练={n_train}, 验证={n_val}, 测试={n_test}")

    print(f"\n最终划分结果:")
    print(f"  训练集: {len(train_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本")
    print(f"  测试集: {len(test_indices)} 样本")

    # 创建子数据集
    train_dataset = LuojiaSET_Subset(full_dataset, train_indices)
    val_dataset = LuojiaSET_Subset(full_dataset, val_indices)
    test_dataset = LuojiaSET_Subset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 使用示例
    root_dir = "/mnt/ramdisk/LuojiaSET-OSFCR"

    # 创建完整数据集
    dataset = LuojiaSET_Dataset(
        root_dir=root_dir,
        modalities=['s1', 's2', 's2_cloudy'],  # 只加载主要模态
        patch_size=256,
        normalize=True
    )

    print(f"Dataset size: {len(dataset)}")
    print("Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample 0:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

    # 创建训练/验证/测试集
    train_ds, val_ds, test_ds = create_luojiaset_datasets(
        root_dir=root_dir,
        modalities=['s2', 's2_cloudy'],  # 只用光学数据
        normalize=True
    )

    print(f"\nTrain set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Test set: {len(test_ds)} samples")
