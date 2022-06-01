import torch
import time
import os
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model: str = "vgg19",
    num_classes: int = 7,
    feature_extraction: bool = True,
    pretrained: bool = True,
    show_progress: bool = True,
):

    model_ft = None
    input_size = 0

    model = model.lower()

    if model == "vgg19":
        model_ft = models.vgg19(pretrained=pretrained, progress=show_progress)
        model_ft.name = "vgg19"
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extraction)
        num_features = model_ft.classifier[0].in_features
        model_ft.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features, out_features=128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )
        input_size = 224
    elif model == "densenet121":
        model_ft = models.densenet121(pretrained=True, progress=True)
        model_ft.name = "densenet121"
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extraction)
        num_features = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features, out_features=128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )
        input_size = 224
    elif model == "resnet50":
        model_ft = models.resnet50(pretrained=pretrained, progress=show_progress)
        model_ft.name = "resnet50"
        set_parameter_requires_grad(model_ft, feature_extracting=feature_extraction)
        num_features = model_ft.fc.in_features
        model_ft.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=num_features, out_features=128, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=128, out_features=num_classes, bias=True),
        )
        input_size = 224
    else:
        raise RuntimeError(f"Inaproperiate model name.")

    return model_ft, input_size


def save_model(model, path: str, epochs: int):
    if os.path.exists(path=path):
        print(f"[INFO] - Saving the model...")
        model_fname = f"{model.name}_{epochs}_{int(time.time())}"
        PATH = path + model_fname + ".pth"
        torch.save(model.state_dict(), PATH)
    else:
        raise RuntimeError(f"Given path does not exist.")
