import argparse

from trainer.trainer import Trainer
from utils.train_utils import (
    get_device,
    get_data_loaders,
    get_datasets,
    count_classes,
    count_model_parameters,
)

from models.pretrained_models_selection import initialize_model, save_model

from utils.plots import plot_metrics, plot_confusion_matrix


if __name__ == "__main__":

    MODEL_WTS_DST_PATH = "./model_weights/"

    parser = argparse.ArgumentParser(
        description="Training models for skin cancer detection."
    )

    parser.add_argument(
        "--csv", type=str, default="./data/", help="Source path to .csv data."
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
        "--checkpoints",
        type=int,
        nargs="?",
        help="Creates checkpoints while training the model. Weights of the model will be saved in Google Drive folder, if training is not perfomer at Google Colab, the weights will be saved in `model_weights` folder.",
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="./data/HAM10000",
        help="Source path to folder that contains skin cancer images.",
    )

    parser.add_argument(
        "--model", type=str, default="vgg19", help="Model to train and evaluate data."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="The size of batches in dataloader."
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
        help="The optimizer name to update the weights of model.\nAvailable optimizers: `adam`, `sgd`.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        help="Scheduler name to change the value of learning rate.\nAvailable schedulers: `plateau`, `cosine`.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau. Default value is 5.",
    )

    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="Patience for ReduceLROnPlateau. Default value is 0.5.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropyloss",
        help="Default: CrossEntropyLoss. Possible loss functions: `crossentropyloss`, `focalloss`.",
    )

    parser.add_argument(
        "--unfreeze-weights",
        action="store_true",
        help="Partially unfrozen layers for the model.",
    )

    if hasattr(parser.parse_known_args()[0], "unfreeze_weights"):
        if parser.parse_known_args()[0].unfreeze_weights:
            parser.add_argument(
                "--layers",
                type=int,
                nargs="+",
                help="Specify which in which layers, weights should be unfrozen. Example: `--layers 2 3`. In 2nd and 3rd layers weights will be unfrozen.",
            )

    if hasattr(parser.parse_known_args()[0], "optimizer"):
        if parser.parse_known_args()[0].optimizer.lower() == "sgd":
            parser.add_argument(
                "--momentum",
                type=float,
                help="Momentum factor using Stochastic Gradient Descent optimizer.",
            )

    if hasattr(parser.parse_known_args()[0], "loss"):

        if parser.parse_known_args()[0].loss.lower() == "crossentropyloss":
            parser.add_argument(
                "--weighted-loss",
                action="store_true",
                help="Use weights for CrossEntropyLoss.",
            )
        else:
            parser.add_argument(
                "--gamma",
                default=2,
                type=int,
                help="Gamma value for localfoss function.",
            )
            parser.add_argument(
                "--alpha",
                action="store_true",
                help="Alpha parameter (weights) will be applied to focal loss function.",
            )

    if hasattr(parser.parse_known_args()[0], "oversample"):
        if not parser.parse_known_args()[0].oversample:
            parser.add_argument(
                "--stratify",
                action="store_true",
                help="Startifies the number of samples to achieve equally distributed samples within batch.",
            )

    parser.add_argument(
        "--unique",
        action="store_true",
        help="Load datasets with the unique lesion IDs.",
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

    model_trainable_params, model_total_params = count_model_parameters(model=model)
    print(
        f"[INFO] Total trainable parameters of the model {model.name}: {model_trainable_params}"
    )
    print(
        f"[INFO] Total non trainable parameters of the model {model.name}: {model_total_params}"
    )

    datasets = get_datasets(
        path_to_csv=args.csv,
        path_to_image_folder=args.image_folder,
        input_size=input_size,
        unique=args.unique,
    )

    data_loaders = get_data_loaders(
        datasets=datasets,
        over_sample=args.oversample,
        batch_size=args.batch_size,
        stratify=args.stratify if hasattr(args, "stratify") else None,
    )

    if hasattr(args, "weighted_loss") or hasattr(args, "alpha"):
        weighted_loss = getattr(args, "weighted_loss", None)
        alpha = getattr(args, "alpha", None)
        if weighted_loss or alpha:
            class_count = count_classes(num_classes=7, dataset=datasets.get("train"))
        else:
            class_count = None

    trainer = Trainer(
        model=model,
        scheduler=args.scheduler,
        optimizer=args.optimizer,
        lr=args.lr,
        loss=args.loss,
        patience=args.patience,
        unfreeze_weights=args.unfreeze_weights,
        layers=args.layers if hasattr(args, "layers") else None,
        gamma=args.gamma if hasattr(args, "gamma") else None,
        checkpoints=args.checkpoints if hasattr(args, "checkpoints") else None,
        factor=args.factor if hasattr(args, "factor") else None,
        momentum=args.factor if hasattr(args, "momentum") else None,
        class_count=class_count,
        device=device,
        validate=True,
    )

    model = trainer.train(data_loaders=data_loaders, num_epochs=args.epochs)

    save_model(model=model, path=MODEL_WTS_DST_PATH, epochs=args.epochs)

    # Evaluation of the model
    model_metrics = trainer.eval(data_loader=data_loaders.get("test"))

    plot_confusion_matrix(
        confusion_matrix=model_metrics.get("cm"),
        model_name=model.name,
        path_to_save_plot="./plots/",
    )

    plot_metrics(
        metrics=model_metrics,
        model_name=model.name,
        path_to_save_plot="./plots/",
        metric_type="avg",
    )

    plot_metrics(
        metrics=model_metrics,
        model_name=model.name,
        path_to_save_plot="./plots/",
        metric_type="per_class",
    )
