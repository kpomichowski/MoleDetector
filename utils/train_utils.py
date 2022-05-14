import torch
import os
import re

from dataset.data_set import LesionsDataset
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms


mean = [0.7636298672977316, 0.5460409399886097, 0.5704622818258704]
std = [0.14052498579388373, 0.15315615440423488, 0.17051305095192615]

composed_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.65),
        transforms.RandomVerticalFlip(p=0.65),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True),
    ]
)


def get_device(gpu: bool) -> torch.device:
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    return device


def get_sampler(train_dataset, oversample: bool=True) -> None or WeightedRandomSampler:
    if not oversample: return None
    target = torch.tensor(
        [sample.get("target") for sample in train_dataset], dtype=torch.int
    )
    class_sample_count = torch.tensor(
        [(target == t).sum() for t in torch.unique(target, sorted=True)]
    )
    weight = 1.0 / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in target])
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def get_data_loaders(datasets: dict, batch_size: int =64, over_sample: bool =True) -> dict:
    loaders = {}
    modes = ['train', 'eval', 'test']

    for mode in modes:
        dataset = datasets.get(mode)
        if mode != 'train':
            over_sample = False
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            sampler=get_sampler(dataset, over_sample)
        )

        loaders[mode] = loader

    return loaders


def get_datasets(path_to_csv: str, path_to_image_folder: str) -> dict:
    datasets = {}
    if not os.path.exists(path_to_csv): raise RuntimeError(f'Path to .csv files does not exist.')

    def find_csv_filenames(path, suffix='.csv'):
        file_names = os.listdir(path)
        return ( file_name for file_name in file_names if file_name.endswith(suffix) )

    pattern = r'\w+(train|test|val).csv'
    prog = re.compile(pattern, re.IGNORECASE)
    for csv_file in find_csv_filenames(path=path_to_csv):
        match = prog.match(csv_file)
        if match:
            csv_file_name = match.group(0)
            try:
                mode = csv_file_name.split('.')[0].split('_')[-1]
                csv_path = path_to_csv + '/' + csv_file_name
                dataset = LesionsDataset(
                    csv_filepath=csv_path,
                    root_dir=path_to_image_folder,
                    transform=composed_transforms if mode.lower() == 'train' else transforms.ToTensor()
                )
                datasets[mode.lower()] = dataset
            except IndexError as e:
                print(e)

    return datasets
