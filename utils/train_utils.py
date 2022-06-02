import torch
import os
import re
import random

from dataset.data_set import LesionsDataset
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms


def get_transforms(input_size, mode="train"):
    mean = [0.7636298672977316, 0.5460409399886097, 0.5704622818258704]
    std = [0.14052498579388373, 0.15315615440423488, 0.17051305095192615]
    if mode == "train":
        composed_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256),
                transforms.RandomHorizontalFlip(p=0.65),
                transforms.RandomVerticalFlip(p=0.65),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.ColorJitter(),
                transforms.RandomPerspective(p=0.5),
                transforms.CenterCrop(size=input_size),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True),
            ]
        )
    else:
        composed_transforms = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True),
            ]
        )
    return composed_transforms


def get_device(gpu: bool) -> torch.device:
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    return device


def get_sampler(train_dataset, oversample: bool) -> None or WeightedRandomSampler:

    if not oversample:
        return None

    target = torch.tensor(
        [torch.argmax(sample.get("target")) for sample in train_dataset],
        dtype=torch.int,
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


def get_data_loaders(
    datasets: dict, batch_size: int = 64, over_sample: bool = True
) -> dict:
    loaders = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        dataset = datasets.get(mode)
        if mode != "train":
            over_sample = False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
            sampler=get_sampler(dataset, over_sample),
        )

        loaders[mode] = loader

    return loaders


def get_datasets(
    path_to_csv: str, path_to_image_folder: str, unique: bool, input_size: int = 224
) -> dict:
    datasets = {}
    if not os.path.exists(path_to_csv):
        raise RuntimeError(f"Path to .csv files does not exist.")

    def find_csv_filenames(path, suffix=".csv"):
        file_names = os.listdir(path)
        return (file_name for file_name in file_names if file_name.endswith(suffix))

    pattern = r"\w+(train|test|val).csv" if unique else r"\w+(train|test|val)_org.csv"

    prog = re.compile(pattern, re.IGNORECASE)
    for csv_file in find_csv_filenames(path=path_to_csv):
        match = prog.match(csv_file)
        if match:
            csv_file_name = match.group(0)
            try:
                index = -2 if not unique else -1
                mode = csv_file_name.split(".")[0].split("_")[index]
                csv_path = path_to_csv + "/" + csv_file_name
                dataset = LesionsDataset(
                    csv_filepath=csv_path,
                    root_dir=path_to_image_folder,
                    transform=get_transforms(input_size=input_size, mode=mode),
                )
                datasets[mode.lower()] = dataset
            except IndexError as e:
                print(e)

    return datasets


def count_model_parameters(model):
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    model_total_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    return model_trainable_params, model_total_params


def count_classes(num_classes: int, dataset: LesionsDataset):
    labels = torch.zeros(num_classes, dtype=torch.long)
    for sample in dataset:
        labels += sample.get("target")
    return labels


def unfreeze_layers(model, layers: tuple or list) -> None:
    layer_index = 0
    for layer in model.children():
        layer_index += 1
        if layer_index in layers:
            blocks = random.choices(layer, k=len(layers))
            for block in blocks:
                for parameter in block.parameters():
                    parameter.requires_grad = True
