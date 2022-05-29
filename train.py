import argparse

from trainer.trainer import Trainer
from utils.train_utils import (
    get_device,
    get_data_loaders,
    get_datasets,
    initialize_model,
    plot_metrics,
    save_model,
    plot_confusion_matrix,
)


if __name__ == "__main__":

    MODEL_WTS_DST_PATH = "./model_weights/"

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

    parser.add_argument(
        "--model", type=str, default="vgg19", help="Model to train and evaluate data."
    )
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

    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for ReduceLROnPlateau."
    )

    parser.add_argument(
        "--weighted_loss", action="store_true", help="Use weights for CrossEntropyLoss."
    )

    parser.add_argument(
        "--no-weighted_loss",
        action="store_false",
        dest="weighted_loss",
        help="Loss will be applied with no additional weights.",
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
        show_progress=True,
        pretrained=True,
    )

    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters of the model: {model_total_params}")

    datasets = get_datasets(
        path_to_csv=args.csv,
        path_to_image_folder=args.image_folder,
        input_size=input_size,
    )

    data_loaders = get_data_loaders(
        datasets=datasets, over_sample=args.oversample, batch_size=args.batch_size,
    )

    trainer = Trainer(
        model=model,
        scheduler=args.scheduler,
        optimizer=args.optimizer,
        lr=args.lr,
        patience=args.patience,
        weighted_loss=args.weighted_loss,
        device=device,
        validate=True,
    )

    model = trainer.train(data_loaders=data_loaders, num_epochs=args.epochs)

    save_model(model=model, path=MODEL_WTS_DST_PATH, epochs=args.epochs)

    # Evaluation of the model
    recall, precision, accuracy, F1_score, cm = trainer.eval(
        data_loader=data_loaders.get("test")
    )

    plot_confusion_matrix(
        confusion_matrix=cm, model_name=model.name, path_to_save_plot="./plots/"
    )

    plot_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=F1_score,
        model_name=model.name,
        path_to_save_plot="./plots/",
    )
