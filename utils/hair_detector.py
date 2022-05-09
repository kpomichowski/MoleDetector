import cv2 as cv
import numpy as np
import os


test_path = r'D:\personalProject\data\HAM10000_images_part_1'

KERNEL = cv.getStructuringElement(1, (32, 32))


def image_to_BGR(image_path: str) -> np.array:
    return cv.imread(image_path, cv.IMREAD_COLOR)


def show_image(image: str or np.array) -> None:
    if not isinstance(image, np.ndarray):
        cv.imshow('image', image_to_BGR(image_path=image))
    else:
        cv.imshow('image', image)
    cv.waitKey(0)


def hair_removal(image):
    grey_scale_im = cv.cvtColor(image_to_BGR(image), cv.COLOR_RGB2GRAY)
    blackhat = cv.morphologyEx(grey_scale_im, cv.MORPH_BLACKHAT, kernel=KERNEL)
    _, thresh = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    #dst = cv.inpaint(image_to_BGR(image), thresh, 1, cv.INPAINT_TELEA)
    show_image(thresh)


for s, image in enumerate(os.listdir(test_path)):
    hair_removal(test_path + r"\\" + image)
    if s == 100: break


