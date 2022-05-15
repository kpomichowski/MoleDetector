import argparse

from base import base_trainer
from utils.train_utils import (
    get_device,
    get_data_loaders,
    get_datasets,
    initialize_model,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training models for skin cancer detection."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="./data/HAM10000_original.csv",
        help="Source path to .csv data.",
    )

    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Oversampling imbalanced class in data by WeightedRandomSampler.",
    )
    parser.add_argument(
        "--no-oversample",
        dest="oversample",
        action="store_false",
        help="Classes won't be oversampled.",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./data/HAM10000",
        help="Source path to folder that contains skin cancer images.",
    )

    parser.add_argument("--model", type=str, help="Model to train and evaluate data.")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The size of batches in dataloader."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="The default learning rate value."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="The number of epochs to train model."
    )
    parser.add_argument(
        "--gpu", action="store_true", help="CUDA cores support for training model."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="CPU only for training model.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="The optimizer name to update the weights of model.\nAvailable optimizers: `adam`.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        help="Scheduler name to change the value of learning rate.\nAvailable schedulers: `plateau`.",
    )

    args = parser.parse_args()

    """
        Argument --csv seeks path that contains three .csv files with proper names.
        For each csv that is split into train, validation and test set, this function 
        `get_datasets` seeks those files with respect to their file name.
        For example, function get_datsets will return datasets, if path to csv contains the following files:
            dataset_train.csv, dataset_val.csv, dataset_test.csv.
    """

    device = get_device(gpu=args.gpu)

    model, input_size = initialize_model(
        model=args.model,
        num_classes=7,
        feature_extraction=True,
        progress=True,
        pretrained=True,
    )

    datasets = get_datasets(
        path_to_csv=args.csv,
        path_to_image_folder=args.image_folder,
        input_size=input_size,
    )

    data_loaders = get_data_loaders(
        datasets=datasets, over_sample=args.oversample, batch_size=args.batch_size,
    )

    trainer = base_trainer.BaseTrainer(
        model=model,
        scheduler=args.scheduler,
        optimizer=args.optimizer,
        lr=args.lr,
        device=device,
    )

    model, metrics = trainer.train(data_loaders=data_loaders, num_epochs=args.epochs)
