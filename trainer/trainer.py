from matplotlib.pyplot import axis
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

        for batch_index, samples in enumerate(data_loader):

            inputs, targets = samples.get("input"), samples.get("target")
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Decoding one-hot encoded target
            targets_ = torch.argmax(targets, dim=1)

            self.optimizer.zero_grad()

            logits = self.model(inputs)
            print(logits.size())

            loss = self._compute_loss(model_output=logits, targets=targets_)

            _, predictions = torch.max(logits, dim=1)

            running_loss += loss.item() * inputs.size(0)
            loss.backward()
            self.optimizer.step()

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

            for batch_index, samples in enumerate(data_loader):

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

        if self.scheduler:
            self.scheduler.step(loss)

        return val_acc, val_loss

    def _eval(self, data_loader, num_classes=7):

        self.model.eval()
        # TODO: create confusion metric, recall, precision, accuracy, auc roc.
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

                batch_length, correct_predicts = self._compute_acc(
                    predicts=predictions, target_gt=targets_
                )

                correct_total += correct_predicts

        TP = np.diag(confusion_matrix)
        FP = np.sum(confusion_matrix, axis=0) - TP
        FN = np.sum(confusion_matrix, axis=1) - TP
        TN = []
        for i in range(num_classes):
            temp = np.delete(confusion_matrix, i, axis=0)
            temp = np.delete(temp, i, axis=1)
            TN.append(sum(sum(temp)))

        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1_score = TP / (TP + 0.5 * (FP + FN))
        accuracy = (TP + TN) / (TP + FP + FN + TN)

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
        self, correct_total: int, running_loss: float, total_items: int,
    ) -> tuple:
        acc = 100 * correct_total / total_items
        r_loss = running_loss / total_items
        accuracy, loss = np.round(acc, 3), np.round(r_loss, 3)
        return accuracy, loss
