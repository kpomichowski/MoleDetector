import inspect
import time
import torch.nn
import copy

from torch import optim
from tqdm import tqdm


class BaseTrainer:

    __schedulers = {
        'plateau': optim.lr_scheduler.ReduceLROnPlateau,
    }

    __optimizers = {
        'adam': optim.Adam,
    }

    def __init__(self, model, device, optimizer, num_epochs=25, lr=1e-3, scheduler=None, **kwargs):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.optimizer = self.__init_optimizer(optimizer_name=optimizer, lr=lr, **kwargs)
        self.scheduler = self.__init_scheduler(scheduler_name=scheduler, **kwargs)

    def train(self, data_loaders):
        return self.__train_loop(data_loaders=data_loaders)

    def __train_loop(self, data_loaders):
        time_start = time.time()
        train_acc_hist, val_acc_hist = [], []
        train_loss_hist, val_loss_hist = [], []
        best_acc = 0.
        best_model_wts = copy.deepcopy(self.model.state_dict())
        self.model.to(self.device)
        for epoch in tqdm(range(self.num_epochs)):
            # logger here to add info about number of epoch
            print('\nEpoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0
                correct_predicts = 0

                for batch in data_loaders.get(phase):

                    inputs, targets = batch.get('input'), batch.get('target')

                    inputs = inputs.to(self.device)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(self.device)

                    # reset the parameter of gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        _, preds = torch.max(outputs, dim=1)

                        if phase == 'train':
                            loss.backward()
                            # update the parameters of model
                            self.optimizer.step()

                    # running statistics
                    running_loss += loss.item() * inputs.size(0)
                    correct_predicts += torch.sum(preds == targets.data).float()

                if phase == 'train' and self.scheduler:
                    self.scheduler.step(loss)

                # epoch stats
                epoch_loss = running_loss / len(data_loaders.get(phase))
                epoch_acc = correct_predicts / len(data_loaders.get(phase))

                # logger here for loss info and acc about training data
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                if phase == 'val':
                    val_loss_hist.append(epoch_loss)
                    val_acc_hist.append(epoch_acc)
                else:
                    train_loss_hist.append(epoch_loss)
                    train_acc_hist.append(epoch_acc)

            time_end = time.time() - time_start
            # logger for time elapsed info and best acc

        self.model.load_state_dict(best_model_wts)
        return self.model, ((train_loss_hist, train_acc_hist), (val_loss_hist, val_acc_hist))

    def eval(self):
        raise NotImplementedError

    def __eval(self):
        raise NotImplementedError

    def __init_scheduler(self, scheduler_name, **kwargs):
        if scheduler_name:
            scheduler = self.__schedulers.get(scheduler_name.lower())
            scheduler_params = self.__get_kwargs_params(scheduler, **kwargs)
            if not scheduler: raise RuntimeError(f'Inaproperiate name of a scheduler.')
            return scheduler(
                self.optimizer,
                verbose=True,
                factor=scheduler_params.get('factor', 0.01),
                patience=scheduler_params.get('patience', 5),
                **scheduler_params,
            )

    def __init_optimizer(self, optimizer_name, lr, **kwargs):
        optimizer = self.__optimizers.get(optimizer_name.lower())
        if not optimizer: raise RuntimeError(f'Inaproperiate name of an optimizer.')
        optimizer_params = self.__get_kwargs_params(obj=optimizer, **kwargs)
        return optimizer(self.model.parameters(), lr=lr, **optimizer_params)

    @staticmethod
    def __get_kwargs_params(obj, **kwargs):
        object_params = list(inspect.signature(obj).parameters)
        return {k: kwargs.pop(k) for k in dict(kwargs) if k in object_params}