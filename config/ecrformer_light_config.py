from typing import Any
from .ecrformer_config import ECRformerConfig


class ECRformerLightConfig(ECRformerConfig):
    """Configuration for ECRformer-Light (lightweight variant)."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.net.cfg['features_start'] = 32
        self.net.cfg['num_blocks'] = [2, 2, 1, 1]
        self.net.cfg['num_refine'] = 2

        self.net.cfg['in_chans'] = [self.NUM_CHANS[key]
                                    for key in self.net.input]
        self.net.cfg['out_chans'] = sum(
            [self.NUM_CHANS[key] for key in self.net.output])
