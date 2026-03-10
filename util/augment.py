import torch


class TestAugment:
    """Test-time augmentation: identity, flips, and rotations."""

    def __init__(self, num_augment=6):
        self.operator = [
            lambda x: x,
            lambda x: torch.flip(x, [-1]),
            lambda x: torch.flip(x, [-2]),
            lambda x: torch.rot90(x, 1, [-1, -2]),
            lambda x: torch.rot90(x, 2, [-1, -2]),
            lambda x: torch.rot90(x, 3, [-1, -2]),
        ][:num_augment]
        self.inv_operators = [
            lambda x: x,
            lambda x: torch.flip(x, [-1]),
            lambda x: torch.flip(x, [-2]),
            lambda x: torch.rot90(x, -1, [-1, -2]),
            lambda x: torch.rot90(x, -2, [-1, -2]),
            lambda x: torch.rot90(x, -3, [-1, -2]),
        ][:num_augment]

    def apply(self, image):
        return [f(image) for f in self.operator]

    def inverse(self, preds):
        return [f(pred) for pred, f in zip(preds, self.inv_operators)]


class TrainAugment:
    """Training augmentation: random flip/rotation + optional random crop."""

    def __init__(self, num_augment=6, crop_size=None):
        self.operator = [
            lambda x: x,
            lambda x: torch.flip(x, [-1]),
            lambda x: torch.flip(x, [-2]),
            lambda x: torch.rot90(x, 1, [-1, -2]),
            lambda x: torch.rot90(x, 2, [-1, -2]),
            lambda x: torch.rot90(x, 3, [-1, -2]),
        ][:num_augment]
        self.crop_size = crop_size

    def augment(self, sample):
        if self.crop_size is not None:
            sample = self.crop(sample)
        op_idx = torch.randint(len(self.operator), (1,)).item()
        operator = self.operator[op_idx]
        sample = {k: operator(v) if isinstance(v, torch.Tensor) else v
                  for k, v in sample.items()}
        return sample

    def crop(self, sample):
        for v in sample.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                h, w = v.shape[-2:]
                break
        else:
            return sample

        top = torch.randint(h - self.crop_size + 1, (1,)).item()
        left = torch.randint(w - self.crop_size + 1, (1,)).item()
        sample = {k: v[..., top:top + self.crop_size, left:left + self.crop_size]
                  if isinstance(v, torch.Tensor) else v
                  for k, v in sample.items()}
        return sample
