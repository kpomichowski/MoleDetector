import torch
import os
import re
import time
import matplotlib.pyplot as plt
import numpy as np

from dataset.data_set import LesionsDataset
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms
from torchvision import models


def get_transforms(input_size, mode="train"):
    mean = [0.7636298672977316, 0.5460409399886097, 0.5704622818258704]
    std = [0.14052498579388373, 0.15315615440423488, 0.17051305095192615]
    if mode == "train":
        composed_transforms = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(p=0.65),
                transforms.RandomVerticalFlip(p=0.65),
                transforms.RandomRotation(degrees=(0, 90)),
                transforms.RandomCrop(input_size),
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


def get_sampler(
    train_dataset, oversample: bool = True
) -> None or WeightedRandomSampler:
    if not oversample:
        return None

    target = torch.tensor(
        [torch.argmax(sample.get("target")) for sample in train_dataset], dtype=torch.int
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
    path_to_csv: str, path_to_image_folder: str, input_size: int = 224
) -> dict:
    datasets = {}
    if not os.path.exists(path_to_csv):
        raise RuntimeError(f"Path to .csv files does not exist.")

    def find_csv_filenames(path, suffix=".csv"):
        file_names = os.listdir(path)
        return (file_name for file_name in file_names if file_name.endswith(suffix))

    pattern = r"\w+(train|test|val).csv"
    prog = re.compile(pattern, re.IGNORECASE)
    for csv_file in find_csv_filenames(path=path_to_csv):
        match = prog.match(csv_file)
        if match:
            csv_file_name = match.group(0)
            try:
                mode = csv_file_name.split(".")[0].split("_")[-1]
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


def plot_save_loss_acc(
    model_name: str, epoch: int, data: tuple, path_to_save_plot: str = None
) -> None:

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(121)
    ax1 = fig.add_subplot(122)

    axis_labels = ["loss", "acc"]
    titles = ["Training/Validation loss.", "Training/Validation accuracy."]

    train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist = data

    # Train loss history plot
    ax.plot(
        np.arange(1, len(train_loss_hist) + 1),
        train_loss_hist,
        c="b",
        label="Training loss",
    )
    # Val loss history plot
    ax.plot(
        np.arange(1, len(val_loss_hist) + 1),
        val_loss_hist,
        c="g",
        label="Validation loss",
    )

    # Train loss accuracy plot
    ax1.plot(
        np.arange(1, len(train_acc_hist) + 1),
        train_acc_hist,
        c="b",
        label="Training accuracy",
    )

    # Val loss accuracy plot
    ax1.plot(
        np.arange(1, len(val_acc_hist) + 1),
        val_acc_hist,
        c="g",
        label="Validation accuracy",
    )

    for ax in fig.axes:
        axis_index = fig.axes.index(ax)
        axis_label = axis_labels[axis_index]
        ax.set_xlabel("epochs")
        ax.set_ylabel(axis_label)
        ax.set_title(titles[axis_index])
        ax.legend()

    plt.show()
    if path_to_save_plot and os.path.exists(path_to_save_plot):
        file_name = f"{int(time.time())}_{model_name}_epoch_{epoch}_plot.png"
        fig.savefig(path_to_save_plot + file_name)
    else:
        raise RuntimeError(f'Folder "./plots" does not exist in project structure.')


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model: str = "vgg19",
    num_classes: int = 7,
    feature_extraction: bool = True,
    pretrained: bool = True,
    progress: bool = True,
):

    model_ft = None
    input_size = 0

    if model == "vgg19":
        model_ft = models.vgg19(pretrained=pretrained, progress=progress)
        model_ft.name = "vgg19"
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extraction)
        model_ft.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=1024, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1024, out_features=128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )
        input_size = 224
    elif model == "inceptionv3":
        ...
    elif model == "resnet50":
        ...
    elif model == "xception":
        ...
    elif model == "mobilenet":
        ...
    else:
        raise RuntimeError(f"Inaproperiate model name.")

    return model_ft, input_size