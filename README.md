<div align="center">

  <h2><b> ECRformer: An Efficient Cloud Removal Transformer with Semantic-Decoupled Learning for Multimodal Satellite Imagery </b></h2>

  **[ISPRS Journal of Photogrammetry and Remote Sensing](https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing)**

  Zaiyan Zhang, Jie Li\*, Yuanqi Liang, Jining Yan, Yi Xiao, Xin Su, Qiangqiang Yuan\*

  *Wuhan University & China University of Geosciences & Zhengzhou University*

</div>

<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zzaiyan/ECRformer)

</div>

## Introduction

Remote sensing imagery is essential for global environmental monitoring, but frequent cloud cover severely limits the utility of optical images. Fusing cloud-prone optical images with cloud-penetrating Synthetic Aperture Radar (SAR) data offers a path to all-weather Earth observation. However, this task faces a dual challenge: the escalating computational cost of state-of-the-art methods and the inherent ill-posedness of the reconstruction under information loss, which complicates the learning process.

To tackle this, we propose **ECRformer** (Efficient Cloud Removal Transformer). ECRformer pairs an efficient architecture with a principled learning paradigm to address both challenges through:

1. A suite of efficient attention mechanisms, including **Cross-Covariance Attention (XCA)** for computationally-aware multimodal feature fusion and **Multi-Dilation Window Attention (MDWA)** for capturing multi-scale spatial context with linear complexity;
2. The **Semantic-Decoupled Feature Learning (SDFL)** paradigm, a novel training strategy that decomposes the ill-posed reconstruction task into two well-defined sub-problems: structure recovery and texture rendering. By applying asymmetric supervision (structural loss on the encoder, texture loss on the decoder), SDFL provides a more principled learning process.

These improvements enhance reconstruction quality, training stability, and reliability, culminating in new **state-of-the-art (SOTA)** performance on both the SEN12MS-CR and LuojiaSET-OSFCR large-scale optical-SAR cloud removal datasets. Notably, ECRformer surpasses previous SOTA methods by **1.23/0.90 dB in PSNR**, while requiring only **28.9% of the parameters** and **24.5% of the FLOPs**, providing a powerful, efficient, and reliable solution for multimodal cloud removal.

## Main Results

### Quantitative Comparison on SEN12MS-CR and LuojiaSET-OSFCR

| Method | Venue | MAE ↓ | SAM ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MAE ↓ | SAM ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-------|-------|-------|--------|--------|---------|-------|-------|--------|--------|---------|
| | | **SEN12MS-CR** | | | | | **LuojiaSET-OSFCR** | | | | |
| SAR-Opt-cGAN | IGARSS'18 | 0.0431 | 15.494 | 25.59 | 0.764 | 0.476 | 0.0457 | 15.953 | 25.31 | 0.752 | 0.498 |
| DSen2-CR | ISPRS'20 | 0.0313 | 9.472 | 27.76 | 0.874 | 0.354 | 0.0317 | 9.511 | 27.68 | 0.873 | 0.359 |
| GLF-CR | ISPRS'22 | 0.0280 | 8.981 | 28.64 | 0.885 | 0.321 | 0.0284 | 9.039 | 28.57 | 0.884 | 0.327 |
| UnCRtainTS | CVPRW'23 | 0.0272 | 8.324 | 28.90 | 0.880 | 0.287 | 0.0299 | 8.495 | 28.05 | 0.878 | 0.294 |
| DiffCR | TGRS'24 | 0.0191 | 5.821 | 31.77 | 0.902 | 0.244 | 0.0194 | 5.886 | 31.71 | 0.900 | 0.263 |
| HPN-CR | TGRS'25 | 0.0242 | 7.637 | 30.23 | 0.898 | 0.275 | 0.0246 | 7.692 | 30.17 | 0.897 | 0.299 |
| EMRDM | CVPR'25 | 0.0179 | 5.267 | 32.14 | 0.924 | **0.181** | 0.0182 | 5.338 | 32.15 | 0.921 | 0.201 |
| **ECRformer-Light** | *Ours* | 0.0178 | 5.026 | 32.75 | 0.920 | 0.224 | 0.0182 | 5.185 | 32.41 | 0.918 | 0.235 |
| **ECRformer** | *Ours* | **0.0164** | **4.693** | **33.37** | **0.932** | 0.188 | **0.0167** | **4.751** | **33.05** | **0.929** | **0.196** |

### Performance vs. Efficiency

| Method | PSNR / SSIM | Params | FLOPs | Training Time (GPU·h) |
|--------|-------------|--------|-------|-----------------------|
| DSen2-CR | 27.76 / 0.874 | 18.95M | 1241.18G | 212.9 |
| GLF-CR | 28.64 / 0.885 | 14.83M | 249.71G | 142.4 |
| UnCRtainTS | 28.90 / 0.880 | 0.52M | 28.56G | 89.5 |
| DiffCR | 31.77 / 0.902 | 22.91M | 45.86G | 396.0 |
| HPN-CR | 30.23 / 0.898 | 3.69M | 19.61G | 130.4 |
| EMRDM | 32.14 / 0.924 | 39.13M | 417.85G | 231.7 |
| **ECRformer-Light** | 32.75 / 0.920 | **3.70M** | **35.78G** | **78.4** |
| **ECRformer** | **33.37 / 0.932** | 11.29M | 102.47G | 142.1 |

## Project Structure

```
ECRformer/
├── train.py                 # Training entry point (CloudRemovalModel)
├── models/
│   ├── ecrformer_model.py   # ECRformerModel (main architecture)
│   ├── module.py            # Attention & block implementations
│   └── module_util.py       # LayerNorm variants, PreNorm, helpers
├── config/
│   ├── base_config.py       # Base configuration
│   ├── ecrformer_config.py  # ECRformer config
│   ├── ecrformer_light_config.py       # ECRformer-Light config
│   └── ecrformer_sen12mscr_config.py   # ECRformer on raw SEN12MS-CR .tif
├── data/
│   ├── npz_dataset.py       # Pre-processed NPZ dataset
│   └── sen12mscr_dataset.py # SEN12MS-CR raw .tif dataset + wrapper
└── util/
    ├── util.py              # Metrics, initialization, param counting
    ├── augment.py           # Train/test augmentation
    ├── checkpoint.py        # Auto-resume from checkpoint
    ├── EMA.py               # Exponential Moving Average callback
    └── pytorch_ssim/        # SSIM module
```

## Model Variants

| Variant | Embed Dim | Depth per Stage | Bottleneck | Refinement | Params | FLOPs |
|---------|-----------|-----------------|------------|------------|--------|-------|
| ECRformer | 48 | [2, 3, 2] | 2 | 4 | 11.29M | 102.47G |
| ECRformer-Light | 32 | [2, 2, 1] | 1 | 2 | 3.70M | 35.78G |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0

```bash
pip install -r requirements.txt
```

Additional dependencies for raw SEN12MS-CR .tif loading:
```bash
pip install rasterio natsort
```

## Dataset Preparation

### Option A: Pre-processed NPZ format (recommended)

Place your `.npz` files under the dataset root. Each file should contain arrays `s1`, `s2`, `label`.

Edit `config/ecrformer_config.py`:
```python
self.dataset.root = "/path/to/npz_datasets"
self.dataset.split = ["ALL_train", "ALL_test"]
```

### Option B: Raw SEN12MS-CR .tif

Download the SEN12MS-CR dataset from: https://patricktum.github.io/cloud_removal/sen12mscr/

The expected directory structure:
```
SEN12MSCR/
├── ROIs1158_spring_s1/
│   └── s1_1/
│       └── ROIs1158_spring_s1_1_p1.tif
├── ROIs1158_spring_s2/
│   └── s2_1/ ...
├── ROIs1158_spring_s2_cloudy/
│   └── s2_cloudy_1/ ...
...
```

Edit `config/ecrformer_sen12mscr_config.py`:
```python
self.dataset.root = "/path/to/SEN12MSCR"
```

## Training

```bash
# Train ECRformer (default)
python train.py --config ecrformer --gpu 0

# Train ECRformer-Light
python train.py --config ecrformer_light --gpu 0

# Train ECRformer with raw SEN12MS-CR
python train.py --config ecrformer_sen12mscr --gpu 0

# Train with custom experiment name
python train.py --config ecrformer --name my_experiment --gpu 0

# Disable auto-resume
python train.py --config ecrformer --no-resume --gpu 0
```

Training logs are saved to `./experiments/` and can be viewed with TensorBoard:
```bash
tensorboard --logdir experiments/
```

## Contact

If you have any questions or suggestions, feel free to contact us.

📧 Email: zzaiyan@whu.edu.cn

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026ecrformer,
    title     = {ECRformer: An Efficient Cloud Removal Transformer with Semantic-Decoupled Learning for Multimodal Satellite Imagery},
    author    = {Zhang, Zaiyan and Li, Jie and Liang, Yuanqi and Yan, Jining and Xiao, Yi and Su, Xin and Yuan, Qiangqiang},
    journal   = {ISPRS Journal of Photogrammetry and Remote Sensing},
    year      = {2026},
    publisher = {Elsevier}
}
```

## Acknowledgements

- [Restormer](https://github.com/swz30/Restormer) — Transposed Attention and GatedFFN design.
- [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/) — Training, validation and testing.
- [LuojiaSET-OSFCR](https://github.com/RSIIPAC/LuojiaSET-OSFCR) — Cross-domain testing. 
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) — Useful toolkit.
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) — Elegant training framework.
