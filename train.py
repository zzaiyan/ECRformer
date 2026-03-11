import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, Namespace

from util.pytorch_ssim import SSIM
from util.util import count_parameters, initialize_weights, compute_metric
from util.augment import TestAugment, TrainAugment
from util.checkpoint import find_latest_checkpoint

from models import find_model_using_name
from data import find_dataset_using_name
from config import find_config_using_name


class CloudRemovalModel(pl.LightningModule):
    """PyTorch Lightning module for cloud removal with ECRformer.

    Supports multi-scale projection loss (down_proj + up_proj).
    """

    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.input = config.net.input
        self.output = config.net.output

        net_class = find_model_using_name(config.net.name)
        self.net = net_class(**config.net.cfg)

        self.SSIM = SSIM()
        self.downsample = nn.AvgPool2d(2, 2)
        def loss_SSIM(x1, x2): return 1 - self.SSIM(x1, x2)
        self.loss_fn = [nn.L1Loss(), loss_SSIM]

        self.loss_weight = config.train.loss_weight
        self.proj_weight = config.train.proj_weight

        self.lr = config.train.lr
        self.train_augment = TrainAugment(crop_size=config.dataset.crop_size)
        self.save_hyperparameters()

        initialize_weights(self.net)

        down_weight, up_weight = self.proj_weight
        if down_weight == 0.:
            self.net.down_proj.requires_grad_(False)
        if up_weight == 0.:
            self.net.up_proj.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    @torch.no_grad()
    def fuse_input(self, batch):
        """Fuse input channels and prepare target tensor."""
        if self.training:
            batch = self.train_augment.augment(batch)
        target = torch.cat([batch[key] for key in self.output], dim=1)
        merged = torch.cat([batch[key] for key in self.input], dim=1)
        return batch, merged, target

    def training_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        down_projs, up_projs = projs

        # Multi-scale projection loss
        down_loss_sum, up_loss_sum = 0, 0
        mid_target = target
        for down_proj, up_proj in zip(down_projs, up_projs[::-1]):
            down_loss = [fn(down_proj, mid_target) for fn in self.loss_fn]
            down_loss = sum(l * w for l, w in zip(down_loss, [0.1, 0.9]))
            down_loss_sum = down_loss_sum + down_loss

            up_loss = [fn(up_proj, mid_target) for fn in self.loss_fn]
            up_loss = sum(l * w for l, w in zip(up_loss, [0.9, 0.1]))
            up_loss_sum = up_loss_sum + up_loss

            mid_target = self.downsample(mid_target)

        down_loss = down_loss_sum / (len(down_projs) + 1e-3)
        up_loss = up_loss_sum / (len(up_projs) + 1e-3)

        # Main loss
        loss_list = [fn(pred, target) for fn in self.loss_fn]
        loss = sum(l * w for l, w in zip(loss_list, self.loss_weight))

        # Combined loss
        down_weight, up_weight = self.proj_weight
        loss = loss * (1 - down_weight - up_weight) + \
            down_loss * down_weight + up_loss * up_weight

        self.log('train_MAE', loss_list[0], prog_bar=True)
        self.log('train_SSIM', 1 - loss_list[1], prog_bar=True)
        self.log('down_proj', down_loss, prog_bar=True)
        self.log('up_proj', up_loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        metrics = compute_metric(pred, target, size_average=True)
        self.log('valid_RMSE', metrics['RMSE'], prog_bar=False, on_epoch=True)
        self.log('valid_MAE', metrics['MAE'], prog_bar=True, on_epoch=True)
        self.log('valid_PSNR', metrics['PSNR'], prog_bar=False, on_epoch=True)
        self.log('valid_SAM', metrics['SAM'], prog_bar=False, on_epoch=True)
        self.log('valid_SSIM', metrics['SSIM'], prog_bar=True, on_epoch=True)
        valid_loss = metrics['MAE'] * 0.9 + (1 - metrics['SSIM']) * 0.1
        self.log('valid_loss', valid_loss, prog_bar=False, on_epoch=True)
        if len(self.trainer.optimizers) > 0:
            self.log('learning_rate',
                     self.trainer.optimizers[0].param_groups[0]['lr'],
                     prog_bar=False, on_epoch=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred_result = self.forward(merged)
        return batch, pred_result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr * 5., weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[120, 150, 170, 180, 190, 200], gamma=0.5)
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def main(config):
    torch.set_float32_matmul_precision("highest")
    pl.seed_everything(config.seed)

    # 创建数据集
    print("\n加载数据集……")
    dataset_class = find_dataset_using_name(config.dataset.name)
    root = config.dataset.root
    split = config.dataset.split
    data_range = config.dataset.data_range
    crop_size = config.dataset.crop_size

    if isinstance(split, list):
        assert len(split) == 2
        train_split, valid_split = split
        train_dataset = dataset_class(
            root, split=train_split, data_range=data_range, crop_size=crop_size)
        valid_dataset = dataset_class(
            root, split=valid_split, data_range=data_range)
    else:
        dataset = dataset_class(root, split=split, data_range=data_range)
        train_size = int(config.dataset.train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size])

    num_workers = config.train.num_workers
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.train_bs, drop_last=True,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.train.valid_bs,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True)

    # 创建模型
    print("\n创建模型……")
    model = CloudRemovalModel(config)
    print(f"模型类: {model.net.__class__.__name__}")
    count_parameters(model)

    # 回调
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_SSIM', verbose=False, mode='max',
        auto_insert_metric_name=True, save_last=True, save_top_k=3)
    early_stop_callback = EarlyStopping(
        monitor='valid_SSIM', patience=config.train.early_stop,
        verbose=False, mode='max')

    # 日志
    log_name = config_name
    if config.name:
        log_name = f"{log_name}_{config.name}"

    save_dir = config.train.save_dir

    # 自动断点续训
    ckpt_path = config.train.ckpt_path
    resume_version = None

    if ckpt_path is None and not getattr(config, 'no_resume', False):
        auto_ckpt_path, version_num = find_latest_checkpoint(save_dir, log_name)
        if auto_ckpt_path is not None:
            ckpt_path = auto_ckpt_path
            resume_version = version_num
            print(f"自动断点续训: {ckpt_path}")
        else:
            print("未找到可用checkpoint，从头开始训练")
    elif getattr(config, 'no_resume', False):
        print("已禁用断点续训")
        ckpt_path = None

    if resume_version is not None:
        tb_logger = TensorBoardLogger(
            save_dir=save_dir, name=log_name, version=resume_version)
    else:
        tb_logger = TensorBoardLogger(save_dir=save_dir, name=log_name)

    # 训练器
    print("\n创建训练器……")
    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        min_epochs=35,
        gradient_clip_val=.5,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=[tb_logger],
        devices=config.train.gpu,
        **config.optim.__dict__,
    )

    # 开始训练
    print("\n开始训练……")
    print(f"实验目录: {os.path.join(save_dir, log_name)}")
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader, ckpt_path=ckpt_path)
    print("\n训练完成!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='ecrformer', help="配置文件名")
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true',
                        help="禁用自动断点续训")
    args = parser.parse_args()

    config_name = args.config
    print(f"配置: {config_name}")
    config_class = find_config_using_name(config_name)
    config = config_class()
    config.name = args.name
    config.train.gpu = [args.gpu]
    config.no_resume = args.no_resume

    main(config)
