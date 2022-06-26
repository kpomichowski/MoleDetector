# MoleDetector - HAM2018 Multiclass skin lesion classification


### Multiclass classifcation using deep learning CNN's with PyTorch library.

![Different skin lesion types](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/SkinLesionsTypes.png)

### Data explanation:

 - `bkl` - considered as `benign keratosis-like lesions` - **(benign)** case,
 - `bcc` - `basal cell carcinoma` - **melanoma** case,
 - `akiec` - `Bowen's disease` - easy treatable cancer (**melanoma** case),
 - `df` - dermatofibroma - (**benign** case),
 - `mel`  - **melanoma** case,
 - `nv` - melanocytic nevi (**benign** case),
 - `vasc` - vascular lesions (**benign** case).


PyTorch software for multiclassification of the skin cancer.
File `train.py` provides all functionalities for training.

The HAM10000 dataset is strong imbalanced, so some methods were applied (see `Training info` section):

![Data class distribution](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/data_distribution.png)

# `train.py` usage:

Type command `python train.py --help` to get the following available instructions:

```

usage: train.py [-h] [--csv CSV] [--oversample] [--no-oversample] [--checkpoints [CHECKPOINTS]] [--image-folder IMAGE_FOLDER] [--model MODEL] [--batch-size BATCH_SIZE] [--lr LR] [--epochs EPOCHS] [--gpu] [--no-gpu] [--optimizer OPTIMIZER] [--scheduler SCHEDULER] [--patience PATIENCE] [--factor FACTOR] [--loss LOSS]
                [--unfreeze-weights]

Training models for skin cancer detection.

optional arguments:
  -h, --help            show this help message and exit
  --csv CSV             Source path to .csv data.
  --oversample          Oversampling imbalanced class in data by WeightedRandomSampler.
  --no-oversample       Classes won't be oversampled.
  --checkpoints [CHECKPOINTS]
                        Creates checkpoints while training the model. Weights of the model will be saved in Google Drive folder.
  --image-folder IMAGE_FOLDER
                        Source path to folder that contains skin cancer images.
  --model MODEL         Model to train and evaluate data.
  --batch-size BATCH_SIZE
                        The size of batches in dataloader.
  --lr LR               The default learning rate value.
  --epochs EPOCHS       The number of epochs to train model.
  --gpu                 CUDA cores support for training model.
  --no-gpu              CPU only for training model.
  --optimizer OPTIMIZER
                        The optimizer name to update the weights of model. Available optimizers: `adam`, `sgd`.
  --scheduler SCHEDULER
                        Scheduler name to change the value of learning rate. Available schedulers: `plateau`, `cosine`.
  --patience PATIENCE   Patience for ReduceLROnPlateau. Default value is 5.
  --factor FACTOR       Patience for ReduceLROnPlateau. Default value is 0.5.
  --loss LOSS           Default: CrossEntropyLoss. Possible loss functions: `crossentropyloss`, `focalloss`.
  --unfreeze-weights    Partially unfrozen layers for the model.
```

# Example of usage the script `train.py`

```
python train.py --csv data/ --image-folder data/HAM10000/ --model efficientnet --batch-size 32 --lr 0.00001 --epochs 60 --optimizer adam --loss focalloss --alpha --gamma 2 --scheduler plateau --patience 5 --factor 0.2 --gpu --unique --no-oversample
```

# Available models to train (with transfer learning):
 * `ResNet50` (`--model resnet50`),
 * `DenseNet121` (`--model densenet121`),
 * `EfficientNetB0` (`--model efficientnet`),
 * `VGG19` (`--model vgg19`).

# Data folder destination

The `data` folder should be located in the root folder of the `MoleDetector` project for convenience, but it's not necessary.
Notice that the name matters: the folder that contains the all training/test data should be name `data`.

The `data` folder should contain:
 * train/val/test csv files of images; csv files should be named `*_train.csv`, `*_val.csv`, `*_test.csv`, where ("\*\") is filename wildcard (any filename).
 * folder `HAM10000`, which contains all sample images `.jpg`.

While running the script you should point the path to the csv files and folder with skin lesion images.
If you put the csv files and `HAM10000` image folder in `data` at the root path, there is no need to provide these locations.

# Training information of the model `EfficientNetB0`.

### Training info:
 * Model was trained without oversampling the images (option --oversample duplicates the augemented samples along the batch with PyTorch `WeightedRandomSampler`),
 * For each band in `training`, `validation` data, mean, standard deviation of each channel was calculated,
 * On training data, there were applied transformations such as: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `ColorJitter`,
 * Loss function `FocalLoss` [link](https://arxiv.org/abs/1708.02002v2) was implemented, because of the imbalanced dataset.
 * Loss function was weighted for each class due to data imbalance (parameter `--alpha`) with parameter `--gamma 2`.
 * Different models were tested: `VGG19`, `ResNet50`, `EfficientNetB0` (the best option),
 * Optimizer: `Adam`, with `weight_decay`: 1e-4
 * Hyperparameters: `batch_size`: 32, `learning_rate`: 0.00001, `epochs`: 60.

### Training information:

| Train | Validation |
|:--------:|:----------:|
| ~88%     |~90%        |


# Training plots:

Training/validation curves: 

![Training/Validation curves](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/1656279234_EfficientNetB0_epoch_60_plot.png)

# Test plots:

Metrics (Precision, Recall, Accuracy, F1 score) for each class:

![Metrics for each class](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/1656279252_metrics_EfficientNetB0_test_per_class.png)

Test confusion matrix:

![Confusion matrix](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/1656279252_cm_EfficientNetB0_test.png)


Average of metrics: Precision, Recall, Accuracy, F1 score:

![Avg. of metrics Accuracy, Recall, F1 Score, Precision](https://github.com/kpomichowski/MoleDetector/blob/trainer/TrainingImages/1656279252_metrics_EfficientNetB0_test_avg.png)


