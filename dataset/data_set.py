import os
import io
import torch
import pandas as pd

from skimage import io
from torch.utils.data import Dataset


class LesionsDataset(Dataset):
    """
    Lesions custom dataset:
    Args:
        csv_filepath (str): Path to the csv that contains the lesions samples.
        root_dir (str): Directory with all images of mole lesions.
        transform (callable, optional): Optional transform to be applied on sample.
    """
    def __init__(self, csv_filepath: str, root_dir: str, transform = None):
        self.lesion_dataset = pd.read_csv(csv_filepath, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, item_index):

        if torch.is_tensor(item_index):
            item_index = item_index.tolist()

        image_name = self.lesion_dataset.iloc[item_index, 1] + '.jpg'
        lesion_type = self.lesion_dataset.iloc[item_index, -1]
        dx_type = self.lesion_dataset.iloc[item_index, 2]

        image_source_path = os.path.join(self.root_dir, image_name)
        image = io.imread(image_source_path)

        sample = {'image': image, 'lesion_type': lesion_type, 'dx': dx_type}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.lesion_dataset)
