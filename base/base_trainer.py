import inspect
import time
import torch
import copy
import abc

from datetime import datetime
from utils import train_utils
from torch import optim
from tqdm import tqdm


class BaseTrainer(metaclass=abc.ABCMeta):

    __schedulers = {
        "plateau": optim.lr_scheduler.ReduceLROnPlateau,
    }

    __optimizers = {
        "adam": optim.Adam,
    }

    def __init__(
        self,
        model,
        device: torch.device,
        optimizer: str,
        lr: float = 1e-3,
        scheduler: str = None,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self.__init_optimizer(
            optimizer_name=optimizer, lr=lr, **kwargs
        )
        self.scheduler = self.__init_scheduler(scheduler_name=scheduler, **kwargs)

    def train(self, data_loaders: dict, num_epochs: int):
        data_loaders__c = copy.deepcopy(data_loaders)
        if "test" in data_loaders.keys():
            data_loaders__c.pop("test")
        self.model.to(self.device)
        return self.__train_loop(data_loaders=data_loaders, num_epochs=num_epochs)

    def _compute_loss(
        self, model_output: torch.tensor, targets: torch.tensor
    ) -> torch.tensor:
        return self.criterion(model_output, targets)

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

        train_loader = data_loaders.get("train")
        validation_loader = data_loaders.get("val")

        for epoch in tqdm(range(1, num_epochs + 1)):
            # logger here to add info about number of epoch
            if epoch != 1:
                print("\n", "*" * 70)
            print(
                f'\n\n[{datetime.now().isoformat(" ", "seconds")}]\n\n\t [INFO] Current epoch: {epoch} of {num_epochs}\n'
            )
            print("\n", "*" * 70)

            training_acc, training_loss = self._train_one_epoch(
                data_loader=train_loader, epoch=epoch
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

            if validation_acc > best_validation_acc:
                best_validation_acc = validation_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - time_start
        print(
            f'[{datetime.now().isoformat(" ", "seconds")}]\n\t [INFO] Training complete: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        )

        self.model.load_state_dict(best_model_weights)
        return self.model

    def eval(self, data_loader: dict, mode: str):
        self.model.to(self.device)
        self._eval(data_loaders=data_loader, mode="test")

    @abc.abstractmethod
    def _eval(self, data_loaders: dict, mode: str):
        raise NotImplementedError

    def __plot_loss_and_acc(self, data, epoch):
        model_name = self.model.name
        train_utils.plot_save_loss_acc(
            model_name=model_name,
            data=data,
            path_to_save_plot=f"./plots/",
            epoch=epoch,
        )

    def __init_scheduler(self, scheduler_name: str, **kwargs):
        if scheduler_name:
            scheduler = self.__schedulers.get(scheduler_name.lower())
            scheduler_params = self.__get_kwargs_params(scheduler, **kwargs)
            if not scheduler:
                raise RuntimeError(f"Inaproperiate name of a scheduler.")
            return scheduler(
                self.optimizer,
                verbose=True,
                factor=scheduler_params.pop("factor", 0.5),
                patience=scheduler_params.pop("patience", 10),
                **scheduler_params,
            )

    def __init_optimizer(self, optimizer_name, lr, **kwargs):
        optimizer = self.__optimizers.get(optimizer_name.lower())
        if not optimizer:
            raise RuntimeError(f"Inaproperiate name of an optimizer.")
        optimizer_params = self.__get_kwargs_params(obj=optimizer, **kwargs)
        return optimizer(self.model.parameters(), lr=lr, **optimizer_params)

    @staticmethod
    def __get_kwargs_params(obj, **kwargs):
        object_params = list(inspect.signature(obj).parameters)
        return {k: kwargs.pop(k) for k in dict(kwargs) if k in object_params}

    def _compute_acc(self, predicts: torch.Tensor, target_gt: torch.Tensor):
        batch_len = predicts.size(0)
        corrects = torch.sum(predicts == target_gt).sum().item()
        return batch_len, corrects
