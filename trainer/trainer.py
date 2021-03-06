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

    def _train_one_epoch(self, data_loader: dict, epoch: int = 1) -> tuple:

        self.model.train()

        running_loss, correct_total = 0, 0
        iters = len(data_loader)

        for batch_index, samples in enumerate(data_loader):

            inputs, targets = samples.get("input"), samples.get("target")
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Decoding one-hot encoded target
            targets_ = torch.argmax(targets, dim=1)

            logits = self.model(inputs)
            loss = self._compute_loss(model_output=logits, targets=targets_)

            _, predictions = torch.max(logits, dim=1)

            running_loss += loss.item() * inputs.size(0)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler.name == "cosine":
                self.scheduler.step(epoch + batch_index / iters)

            batch_length, correct_predicts = self._compute_acc(
                predicts=predictions, target_gt=targets_
            )

            correct_total += correct_predicts

        training_acc, training_loss = self.__compute_metrics(
            correct_total=correct_total,
            running_loss=running_loss,
            total_items=batch_length * len(data_loader),
        )

        print(
            f'[{datetime.now().isoformat(" ", "seconds")}]\n\t [INFO] Epoch: {epoch} | training epoch loss: {training_loss} | training epoch acc.: {training_acc} |'
        )

        return training_acc, training_loss

    def _validate_one_epoch(self, data_loader, epoch: int) -> tuple:

        self.model.eval()

        running_loss, correct_total = 0, 0

        with torch.no_grad():

            for _, samples in enumerate(data_loader):

                inputs, targets = samples.get("input"), samples.get("target")
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                logits = self.model(inputs)
                targets_ = torch.argmax(targets, dim=1)
                _, predictions = torch.max(logits, dim=1)

                loss = self._compute_loss(model_output=logits, targets=targets_)
                running_loss += loss.item() * inputs.size(0)
                batch_length, correct_predicts = self._compute_acc(
                    predicts=predictions, target_gt=targets_
                )

                correct_total += correct_predicts

            val_acc, val_loss = self.__compute_metrics(
                correct_total=correct_total,
                running_loss=running_loss,
                total_items=batch_length * len(data_loader),
            )

        print(
            f"\n\t [INFO] Epoch: {epoch} | validation epoch loss: {val_loss} | validation epoch acc.: {val_acc} |"
        )

        if self.scheduler.name == "plateau":
            self.scheduler.step(val_loss)

        return val_acc, val_loss

    def _eval(self, data_loader, num_classes=7):

        self.model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        correct_total = 0
        with torch.no_grad():

            for samples in data_loader:
                inputs, targets = samples.get("input"), samples.get("target")
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                targets_ = torch.argmax(targets, dim=1)
                _, predictions = torch.max(outputs, dim=1)

                for target, prediction in zip(targets_.view(-1), predictions.view(-1)):
                    confusion_matrix[target.long(), prediction.long()] += 1

                _, correct_predicts = self._compute_acc(
                    predicts=predictions, target_gt=targets_
                )

                correct_total += correct_predicts

        TP = np.diag(confusion_matrix)
        FP = np.sum(confusion_matrix, axis=0) - TP
        FN = np.sum(confusion_matrix, axis=1) - TP
        recall = TP / (TP + FN)
        F1_score = TP / (TP + 0.5 * (FP + FN))
        precision = np.divide(
            TP,
            TP + FP,
            out=np.zeros_like(TP, dtype=np.float16),
            where=((TP != 0) & (FP != 0)),
        )
        accuracy = np.divide(
            np.diag(confusion_matrix),
            np.sum(confusion_matrix, axis=1),
            where=np.sum(confusion_matrix, axis=1) != 0,
        )

        print(f"Confusion matrix:\n", confusion_matrix)
        print(f"Per class acc.: {accuracy}")
        print(f"Per class precision: {precision}")
        print(f"Per class recall: {recall}")
        print(f"Per class F1 score: {F1_score}")
        print(f"\nAvg precision on test data set:", np.mean(precision))
        print(f"\nAvg recall on test data set:", np.mean(recall))
        print(f"\nAvg F1 score: {np.mean(F1_score)}")

        metrics = {
            "avg": [
                np.mean(recall),
                np.mean(precision),
                np.mean(accuracy),
                np.mean(F1_score),
            ],
            "per_class": [accuracy, precision, recall, F1_score],
            "cm": confusion_matrix,
        }

        return metrics

    def __compute_metrics(
        self, correct_total: int, running_loss: float, total_items: int
    ) -> tuple:
        acc = 100 * (correct_total / total_items)
        r_loss = running_loss / total_items
        accuracy, loss = np.round(acc, 4), np.round(r_loss, 4)
        return accuracy, loss
