<div align="center">

  <h2><b> ECRformer: An Efficient Cloud Removal Transformer with Semantic-Decoupled Learning for Multi-Modal Satellite Imagery (Under Review) </b></h2>

</div>


<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/zzaiyan/ECRformer)

</div>


## Introduction

Remote sensing imagery is essential for global environmental monitoring, but frequent cloud cover severely limits the utility of optical images. Fusing cloud-prone optical images with cloud-penetrating Synthetic Aperture Radar (SAR) data offers a path to all-weather Earth observation. However, this task faces a dual challenge: the escalating computational cost of state-of-the-art methods and the inherent ill-posedness of the reconstruction under information loss, which complicates the learning process. To tackle this, we propose ECRformer (Efficient Cloud Removal Transformer). ECRformer pairs an efficient architecture with a principled learning paradigm to address both challenges through: 1) a suite of efficient attention mechanisms, including Cross-Covariance Attention (XCA) for computationally-aware multi-modal feature fusion and Multi-Dilation Window Attention (MDWA) for capturing multi-scale spatial context with linear complexity; and 2) the Semantic-Decoupled Feature Learning (SDFL) paradigm, a novel training strategy that decomposes the ill-posed reconstruction task into two well-defined sub-problems: structure recovery and texture rendering. By applying asymmetric supervision (structural loss on the encoder, texture loss on the decoder), SDFL provides a more principled learning process. These improvements enhance reconstruction quality, training stability, and reliability, culminating in new state-of-the-art (SOTA) performance on both the SEN12MS-CR and LuojiaSET-OSFCR large-scale optical-SAR cloud removal datasets. Notably, ECRformer surpasses previous SOTA methods by 1.23/0.90 dB in PSNR, while requiring only 28.9\% of the parameters and 24.5\% of the FLOPs, providing a powerful, efficient, and reliable solution for multi-modal cloud removal.

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

Place your `.npz` files under the dataset root. Each file should contain arrays `s1`, `s2`, `label`, `paths`.

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

## Model Variants

| Variant | features_start | num_blocks | num_refine | Params |
|---------|---------------|------------|------------|--------|
| ECRformer | 48 | [2, 3, 2, 2] | 4 | ~11.3M |
| ECRformer-Light | 32 | [2, 2, 1, 1] | 2 | ~3.7M |

## Contact

If you have any questions or suggestions, feel free to contact me.

📧Email: zzaiyan@whu.edu.cn

## Citation

If you find this work useful, please cite:

*TODO*

```bibtex
@article{ecr_former,
    title     = {ECRformer: An Efficient Cloud Removal Transformer with Semantic-Decoupled Learning for Multi-Modal Satellite Imagery},
    author    = {TODO},
    journal   = {TODO},
    year      = {2026},
}
```

## Acknowledgements

- [Restormer](https://github.com/swz30/Restormer) — Transposed Attention and GatedFFN design.
- [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/) — Dataset.
- [Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning) — Elegant training framework.

## License

*TODO*
