from typing import Any
from .ecrformer_config import ECRformerConfig


class ECRformerSEN12MSCRConfig(ECRformerConfig):
    """Configuration for ECRformer on original SEN12MS-CR .tif dataset.

    Inherits all network parameters from ECRformerConfig;
    only overrides dataset settings.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.dataset.name = 'sen12mscr'
        self.dataset.root = r"/path/to/SEN12MSCR"
        self.dataset.split = ['train', 'val']
