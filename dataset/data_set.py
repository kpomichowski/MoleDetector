import os
import torch
import pandas as pd

from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class LesionsDataset(Dataset):
    """
    Lesions custom dataset:
    Args:
        csv_filepath (str): Path to the csv that contains the lesions samples.
        root_dir (str): Directory with all images of mole lesions.
        transform (callable, optional): Optional transform to be applied on sample.
    """

    def __init__(self, csv_filepath: str, root_dir: str, transform=None, transform_target=None):
        self.lesion_dataset = pd.read_csv(csv_filepath, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, item_index):

        if torch.is_tensor(item_index):
            item_index = item_index.tolist()

        image_name = self.lesion_dataset.iloc[item_index, 1] + ".jpg"
        lesion_type = self.lesion_dataset.iloc[item_index, -2]
        image_source_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_source_path)

        if self.transform:
            image = self.transform(image)

        target = one_hot(torch.tensor(int(lesion_type)), num_classes=7)

        sample = {"input": image, "target": target}
        return sample

    def __len__(self):
        return len(self.lesion_dataset)
