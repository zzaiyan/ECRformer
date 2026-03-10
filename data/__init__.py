import importlib
import torch.utils.data


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    The class called <DatasetName>Dataset() will be instantiated.
    It has to be a subclass of torch.utils.data.Dataset, and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.replace('_', '').lower() == target_dataset_name.lower() \
           and issubclass(cls, torch.utils.data.Dataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of Dataset with class name that matches %s in lowercase."
            % (dataset_filename, target_dataset_name))

    return dataset
