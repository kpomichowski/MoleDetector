import cv2
import os
import numpy as np

from tqdm import tqdm


def get_mean_std_from_channels(images_path: str, channel_num: int) -> tuple:
    """
    :param images_path: str -> path to folders with images
    :param channel_num: int -> number of channels for each image (channel_num=3 in case of RGB images)
    :return: tuple -> returns the tuple of lists: mean, std foreach following channel: r, g, b
    """

    channel_sum = np.zeros(channel_num)
    channel_sum_squared = np.zeros(channel_num)
    pixel_num = 0
    H, W = 224, 224

    for image in tqdm(os.listdir(images_path)):
        image = cv2.imread(os.path.join(images_path, image))
        image = cv2.resize(image, (H, W), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        pixel_num += image.size / channel_num
        channel_sum += np.sum(image, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(image), axis=(0, 1))

    rgb_mean = list(channel_sum / pixel_num)
    rgb_std = list(np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean)))

    print(f"mean: {rgb_mean}\nstd: {rgb_std}")
    return rgb_mean, rgb_std


if __name__ == "__main__":
    std, mean = get_mean_std_from_channels(
        images_path="../data/HAM10000", channel_num=3
    )
