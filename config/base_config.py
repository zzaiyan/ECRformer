from argparse import Namespace
from typing import Any


class BaseConfig(Namespace):
    NUM_CHANS = {
        'SAR': 2,
        'cloudy': 13,
        'target': 13,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.seed = 42

        # 数据集
        self.dataset = Namespace(
            name='npz',
            root=r"/path/to/datasets",
            split=["ALL_train", "ALL_test"],
            train_ratio=0.8,    # 当split只包含一个元素时，才使用train_ratio划分训练/验证集
            data_range=1.0,
            crop_size=128,
        )

        # 训练
        self.train = Namespace(
            max_epoch=1000,
            early_stop=100,
            lr=1e-4,
            loss_weight=[0.9, 0.1],
            proj_weight=[0., 0.],
            train_bs=16,
            valid_bs=32,
            num_workers=8,
            ckpt_path=None,
            save_dir='./experiments',
        )

        # 训练优化（传递给 pl.Trainer）
        self.optim = Namespace(
            accelerator='auto',
            precision=32,
        )

        # 网络
        self.net = Namespace(
            name='ecrformer',
            input=['SAR', 'cloudy'],
            output=['target'],
            cfg=dict(),
        )

        self.net.cfg['in_chans'] = [self.NUM_CHANS[key] for key in self.net.input]
        self.net.cfg['out_chans'] = sum(self.NUM_CHANS[key] for key in self.net.output)
