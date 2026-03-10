"""Checkpoint utilities for automatic resume."""

import os
import glob


def find_latest_checkpoint(save_dir, log_name):
    """Find the latest checkpoint file for automatic resume.

    Args:
        save_dir: Base directory for experiments.
        log_name: Experiment name.

    Returns:
        tuple: (checkpoint_path, version_number), or (None, None) if not found.
    """
    exp_dir = os.path.join(save_dir, log_name)

    if not os.path.exists(exp_dir):
        print(f"实验目录不存在: {exp_dir}")
        return None, None

    version_dirs = glob.glob(os.path.join(exp_dir, "version_*"))
    if not version_dirs:
        print(f"未找到版本目录: {exp_dir}")
        return None, None

    def extract_version_number(path):
        try:
            return int(os.path.basename(path).split('_')[-1])
        except (ValueError, IndexError):
            return -1

    version_dirs.sort(key=extract_version_number)
    latest_version_dir = version_dirs[-1]
    version_num = extract_version_number(latest_version_dir)

    checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"checkpoints目录不存在: {checkpoint_dir}")
        return None, None

    # Prefer last.ckpt
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"找到最新checkpoint: {last_ckpt} (版本: {version_num})")
        return last_ckpt, version_num

    # Fall back to latest epoch checkpoint
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch=*.ckpt"))
    if not ckpt_files:
        print(f"未找到任何checkpoint文件: {checkpoint_dir}")
        return None, None

    def extract_epoch(filename):
        basename = os.path.basename(filename)
        try:
            return int(basename.split('epoch=')[1].split('-')[0])
        except (ValueError, IndexError):
            return -1

    ckpt_files.sort(key=extract_epoch)
    latest_ckpt = ckpt_files[-1]
    print(f"找到最新checkpoint: {latest_ckpt} (版本: {version_num})")

    return latest_ckpt, version_num
