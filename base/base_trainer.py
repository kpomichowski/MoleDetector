import inspect
import time
import torch
import copy
import abc
import os

from losses import losses
from datetime import datetime
from utils import plots
from torch import optim
from tqdm import tqdm
from utils import train_utils


def get_kwargs_params(obj, **kwargs):
    object_params = list(inspect.signature(obj).parameters)
    return {k: kwargs.pop(k) for k in dict(kwargs) if k in object_params}


class BaseTrainer(metaclass=abc.ABCMeta):

    __schedulers = {"plateau": optim.lr_scheduler.ReduceLROnPlateau}

    __optimizers = {"adam": optim.Adam, "sgd": optim.SGD}

    __losses = {
        "crossentropyloss": torch.nn.CrossEntropyLoss,
        "focalloss": losses.focal_loss,
    }

    def __init__(
        self,
        model,
        device: torch.device,
        optimizer: str,
        loss: str,
        unfreeze_weights: bool,
        layers: list,
        class_count: torch.tensor,
        gamma: int,
        lr: float = 1e-3,
        scheduler: str = None,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.criterion = self.__init_loss(
            loss=loss, class_count=class_count, gamma=gamma
        )
        self.optimizer = self.__init_optimizer(
            optimizer_name=optimizer, lr=lr, **kwargs
        )
        self.scheduler = self.__init_scheduler(scheduler_name=scheduler, **kwargs)
        self.unfreeze_weights = unfreeze_weights
        if self.unfreeze_weights:
            self.layers = layers

        self.checkpoints = kwargs.get("checkpoints")

    def __init_loss(
        self, loss: str, class_count: torch.tensor or None, gamma: int = None
    ):
        if class_count is not None:
            num_samples = class_count
            normed_weights = [1 - (x / torch.sum(num_samples)) for x in num_samples]
            normed_weights = torch.FloatTensor(normed_weights).to(self.device)

        loss_ = self.__losses.get(loss)
        if loss is None:
            raise RuntimeError(
                f"Specified loss does not exist. Available losses are: `focalloss`, `crossentropyloss`."
            )
        if class_count is not None and loss == "crossentropyloss":
            criterion = loss_(weight=normed_weights)
        elif class_count is not None and loss == "focalloss":
            # criterion = loss_(
            #     gamma=gamma, alpha=normed_weights, reduction="mean", device=self.device
            # )
            criterion = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                model="FocalLoss",
                alpha=normed_weights,
                gamma=gamma,
                reduction="mean",
                force_reload=False,
                verbose=True,
            )
        elif class_count is None and loss == "focalloss":
            # criterion = loss_(gamma=gamma, reduction="mean", device=self.device)
            criterion = torch.hub.load(
                "adeelh/pytorch-multi-class-focal-loss",
                alpha=None,
                gamma=gamma,
                reduction="mean",
                force_reload=False,
                verbose=True,
            )
        else:
            criterion = loss_()

        return criterion

    def __init_scheduler(self, scheduler_name: str, **kwargs):
        if scheduler_name:
            scheduler = self.__schedulers.get(scheduler_name.lower())
            scheduler_params = get_kwargs_params(scheduler, **kwargs)
            if not scheduler:
                raise RuntimeError(f"Inaproperiate name of a scheduler.")

            if scheduler_name == "plateau":
                scheduler = scheduler(
                    self.optimizer,
                    verbose=True,
                    factor=scheduler_params.pop("factor", 0.2),
                    patience=scheduler_params.pop("patience", 5),
                    **scheduler_params,
                )
            elif scheduler_name == "cosine":
                scheduler = scheduler(
                    self.optimizer, T_0=10, T_mult=1, verbose=True, **scheduler_params
                )
            else:
                raise NotImplementedError
            scheduler.name = scheduler_name
            return scheduler

    def __init_optimizer(self, optimizer_name, lr, **kwargs):
        optimizer = self.__optimizers.get(optimizer_name.lower())
        if not optimizer:
            raise RuntimeError(f"Inaproperiate name of an optimizer.")
        optimizer_params = get_kwargs_params(obj=optimizer, **kwargs)
        if optimizer_name == "adam":
            optimizer = optimizer(
                filter(lambda param: param.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=1e-6,
                **optimizer_params,
            )
        elif optimizer_name == "sgd":
            optimizer = optimizer(
                filter(lambda param: param.requires_grad, self.model.parameters()),
                lr=lr,
                momentum=optimizer_params.pop("momentum", 0.9),
                weight_decay=1e-6,
                **optimizer_params,
            )
        else:
            raise NotImplementedError

        return optimizer

    def train(self, data_loaders: dict, num_epochs: int):
        self.model.to(self.device)
        return self.__train_loop(data_loaders=data_loaders, num_epochs=num_epochs)

    @abc.abstractmethod
    def _train_one_epoch(self, data_loaders: dict, epoch: int = 1):
        raise NotImplementedError

    @abc.abstractmethod
    def _validate_one_epoch(self, data_loaders: dict, epoch: int = 1):
        raise NotImplementedError

    def __train_loop(self, data_loaders: dict, num_epochs: int = 25):
        time_start = time.time()

        train_loss_history, train_acc_history = [], []
        val_loss_history, val_acc_history = [], []

        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_validation_acc = 0.0
        is_unfreezed = False
        unfrozen = []

        train_loader = data_loaders.get("train")
        validation_loader = data_loaders.get("val")

        for epoch in tqdm(range(1, num_epochs + 1)):

            # logger here to add info about number of epoch
            if epoch != 1:
                print("\n\n", "*" * 90)
            else:
                print("\n", "*" * 90)

            print(
                f'\n\n[{datetime.now().isoformat(" ", "seconds")}]\n\n\t [INFO] Current epoch: {epoch} of {num_epochs}\n'
            )

            training_acc, training_loss = self._train_one_epoch(
                data_loader=train_loader, epoch=epoch
            )

            if epoch % 20 == 0 and not is_unfreezed:
                if self.unfreeze_weights and self.layers:
                    self.layers = tuple(self.layers)
                    print(
                        f"\n[INFO] - epoch {epoch} - unfreezing weights in the following layers: {self.layers}."
                    )
                    time.sleep(3)
                    train_utils.unfreeze_layers(model=self.model, layers=self.layers)
                    is_unfreezed = True
                    print(
                        f"[INFO] Total trainable params after unfreeze layers: {train_utils.count_model_parameters(self.model)[0]}."
                    )

            # store the history of train accuracy and loss
            train_acc_history.append(training_acc)
            train_loss_history.append(training_loss)

            validation_acc, validation_loss = self._validate_one_epoch(
                data_loader=validation_loader, epoch=epoch
            )

            # store the history of validation accuracy and loss
            val_acc_history.append(validation_acc)
            val_loss_history.append(validation_loss)

            # Plot for every five epochs training and validation loss/accuracy.
            if epoch % 5 == 0:
                self.__plot_loss_and_acc(
                    data=(
                        train_loss_history,
                        val_loss_history,
                        train_acc_history,
                        val_acc_history,
                    ),
                    epoch=epoch,
                )

            if self.checkpoints:
                checkpoint = self.checkpoints
                if epoch % checkpoint == 0:
                    self.model.optimizer = self.optimizer
                    if self.scheduler.name == "cosine":
                        self.model.lr = self.scheduler.get_last_lr()
                    else:
                        self.model.lr = self.optimizer.param_groups[0]["lr"]
                    train_utils.save_on_checkpoint(model=self.model, epoch_number=epoch)

            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - time_start
        print(
            f'[{datetime.now().isoformat(" ", "seconds")}]\n\t [INFO] Training complete: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        )

        self.model.load_state_dict(best_model_weights)
        return self.model

    def eval(self, data_loader: dict):
        self.model.to(self.device)
        return self._eval(data_loader=data_loader)

    @abc.abstractmethod
    def _eval(self, data_loader: dict):
        raise NotImplementedError

    def __plot_loss_and_acc(self, data, epoch):
        model_name = self.model.name
        plots.plot_save_loss_acc(
            model_name=model_name, data=data, path_to_save_plot=f"./plots/", epoch=epoch
        )

    def _compute_acc(self, predicts: torch.Tensor, target_gt: torch.Tensor):
        batch_len = predicts.size(0)
        corrects = torch.sum(predicts == target_gt).sum().item()
        return batch_len, corrects

    def _compute_loss(
        self, model_output: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        return self.criterion(model_output, targets)
