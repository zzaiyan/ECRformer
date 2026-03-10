from typing import Any
from .base_config import BaseConfig


class ECRformerConfig(BaseConfig):
    """Configuration for ECRformer (default variant)."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.net.name = 'ecrformer'

        self.net.cfg = dict(
            features_start=48,
            num_blocks=[2, 3, 2, 2],
            block_type=['ecrformer', 'ecrformer'],
            cbam='1ca2+1sa2',
            bottle_neck='tsa',
            num_refine=4,
            pos_encoding=None,
            drop_path_rate=0.,
        )

        self.net.output = ['target']

        self.train.train_bs = 8
        self.train.valid_bs = 4
        self.optim.accumulate_grad_batches = 2
        self.train.proj_weight = [0.05, 0.05]

        self.net.cfg['in_chans'] = [self.NUM_CHANS[key]
                                    for key in self.net.input]
        self.net.cfg['out_chans'] = sum(
            [self.NUM_CHANS[key] for key in self.net.output])
