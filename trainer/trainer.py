import torch
import numpy as np

from datetime import datetime
from base import base_trainer


class Trainer(base_trainer.BaseTrainer):
    def __init__(
        self, model, device, optimizer, lr=1e-3, scheduler=None, validate=True, **kwargs
    ):
        super().__init__(
            model=model,
            device=device,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            **kwargs,
        )
        self.do_validation = validate

    def _train_one_epoch(self, data_loaders: dict, epoch: int = 1) -> tuple:
        self.model.train()

        running_loss, correct_total = 0, 0
        for batch_index, samples in enumerate(data_loaders.get("train")):

            inputs, targets = samples.get("input"), samples.get("target")
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Decoding one-hot encoded target
            targets_ = torch.argmax(targets, dim=1)

            self.optimizer.zero_grad()

            if self.model.name == "inceptionv3":
                logits, aux_outputs = self.model(inputs)
                loss = self.criterion(model_output=logits, targets=targets_)
                loss_ = self.criterion(model_output=aux_outputs, targets=targets_)
                loss = loss * 0.4 * loss_
            else:
                logits = self.model(inputs)
                loss = self._compute_loss(model_output=logits, targets=targets_)

            _, predictions = torch.max(logits, dim=1)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            batch_length, correct_predicts = self._compute_acc(
                predicts=predictions, target_gt=targets_
            )
            
            correct_total += correct_predicts

        training_loss, training_acc = self.__compute_metrics(
            correct_total=correct_total,
            running_loss=running_loss,
            total_batches=len(data_loaders.get("train")),
            total_items=batch_length * len(data_loaders.get('train')),
        )

        if self.do_validation:
            validation_data_loader = data_loaders.get("val")
            val_acc, val_loss = self.__validate_one_epoch(
                data_loader=validation_data_loader, epoch=epoch
            )
        else:
            return training_acc, training_loss

        print(
            f"[{datetime.now()}]\n\t [INFO] training epoch loss: {training_loss} | training epoch acc.: {training_acc} |"
        )
        print(
            f"[{datetime.now()}]\n\t [INFO] validation epoch loss: {val_loss} | validation epoch acc.: {val_acc} |"
        )

        return training_acc, training_loss, val_acc, val_loss

    def __validate_one_epoch(self, data_loader, epoch: int) -> tuple:
        self.model.eval()

        total_items, running_loss, correct_total = 0, 0, 0
        with torch.no_grad():

            for batch_index, samples in enumerate(data_loader):

                inputs, targets = samples.get("input"), samples.get("target")
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs)
                targets_ = torch.argmax(targets, dim=1)
                _, predictions = torch.max(logits, dim=1)

                loss = self._compute_loss(model_output=logits, targets=targets_)

                running_loss += loss.item()
                batch_length, correct_predicts = self._compute_acc(
                    predicts=predictions, target_gt=targets_
                )

                correct_total += correct_predicts
                total_items += batch_length

            val_acc, val_loss = self.__compute_metrics(
                correct_total=correct_total,
                running_loss=running_loss,
                total_batches=len(data_loader),
                total_items=batch_length * len(data_loader),
            )

        if self.scheduler:
            self.scheduler.step(loss)

        return val_acc, val_loss

    def _eval(self, data_loaders, mode):
        pass

    def __compute_metrics(
        self,
        correct_total: int,
        running_loss: float,
        total_items: int,
        total_batches: int,
    ) -> tuple:
        acc = 100 * correct_total / total_items
        r_loss = running_loss / total_batches
        return acc, r_loss
