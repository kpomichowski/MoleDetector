import torch
import os
import re
import random
import time

from utils.sampler import StratifiedBatchSampler
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
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.ColorJitter(brightness=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.CenterCrop(size=input_size),
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
    datasets: dict, stratify: bool, batch_size: int = 64, over_sample: bool = True,
) -> dict:
    loaders = {}
    modes = ["train", "val", "test"]

    for mode in modes:
        dataset = datasets.get(mode)
        if mode != "train":
            over_sample = False

        if not stratify:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                drop_last=True,
                sampler=get_sampler(dataset, over_sample),
            )
        elif stratify and mode in ["val", "train"]:
            loader = DataLoader(
                dataset=dataset,
                batch_sampler=StratifiedBatchSampler(
                    dataset.targets, batch_size=batch_size
                ),
                num_workers=0,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=0,
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


def save_on_checkpoint(model, epoch_number: int) -> None:

    model_name = model.name
    model_state_dict = model.state_dict()
    optimizer_state_dict = model.optimizer.state_dict()
    _epoch = epoch_number
    _lr = model.lr

    try:
        import google.colab
        USING_COLAB = True
    except:
        USING_COLAB = False

    filename = f"{model_name}_{_epoch}_{int(time.time())}_checkpoint.pth"
    path = (
        f"./model_weights/" + filename
        if not USING_COLAB
        else f"/content/drive/My Drive/HAM10000_checkpoints/" + filename
    )

    if os.path.exists(path):
        torch.save(
            {
                "model": model,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "epoch": _epoch,
                "lr": _lr,
            },
            path,
        )

    print(f"[INFO] Succesfully saved checkpoint to the {path} at epoch {epoch_number}.")
