from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import parser
import argparse
import pathlib
import numpy as np
import random
from PIL import ImageOps


def to_binary(data):
    for d in data:
        image_array = d["image"]
        image_array = image_array.astype('uint8')
        # print(image_)
        is_tfl = d["is_tfl"]
        # im=Image.open(image_)
        # image_array = np.array()
        # image_array.convert('rgb')
        with open('data.bin', 'ab') as file:
            np.save(file, image_array)
        with open('labels.bin', 'ab') as f2:
            f2.write(is_tfl)


def crop(path_image: str, x, y):
    im = Image.open(path_image)
    im2 = ImageOps.expand(im, border=40, fill='black')
    im3 = np.asarray(im2, dtype="uint8")
    left = y
    top = x
    right = y + 81
    bottom = x + 81
    cropped_image = im3[left:right, top:bottom]
    return cropped_image


def find_tl(_image: np.ndarray, all_tl):
    rand = random.randint(0, len(all_tl))
    tl_x = all_tl[0][rand]
    tl_y = all_tl[1][rand]
    return tl_x, tl_y


def find_not_tf(_image: np.ndarray, all_tl):
    mask = np.in1d(np.arange(np.shape(_image)[1]), all_tl[1])
    no_tl = np.where(~mask)[0]
    rand = random.randint(0, len(no_tl) - 1)
    not_tl_x = random.randint(0, np.shape(_image)[0])
    not_tl_y = no_tl[rand]
    return not_tl_x, not_tl_y


def lable_data(path_image, grey_image: np.ndarray):
    all_tl = np.where(grey_image == 19)
    # print(all_tl)
    if len(all_tl[0]) == 0:
        return []
    tl_x, tl_y = find_tl(grey_image, all_tl)
    not_tl_x, not_tl_y = find_not_tf(grey_image, all_tl)
    tl = crop(path_image, tl_x, tl_y)
    ntl = crop(path_image, not_tl_x, not_tl_y)
    return [{"image": tl, "is_tfl": b"00000001"}, {"image": ntl, "is_tfl": b"00000000"}]


def load_images():
    files = glob.glob(os.path.join("gtFine", "train", "*", "*labelids.png"))
    res = []
    for image_path in files:
        # image_array = np.array(Image.open(image_path))
        # image_array = plt.imread(image_path)
        image_array = np.array(Image.open(image_path), dtype='uint8')
        image_name = "\\".join(image_path.split("\\")[-2:])
        color_name = image_name.replace("gtFine_labelIds.png", "leftImg8bit.png")
        color_path = os.path.join("leftImg8bit_trainvaltest", "leftImg8bit", "train", color_name)
        dict_res = lable_data(color_path, image_array)
        # print(dict_res)
        to_binary(dict_res)


def all_to_binary(all):
    for im in all:
        to_binary(im)
        # print(im)
        pass


def load_data_and_labels():
    images = load_images()
    # all_to_binary(images)


load_data_and_labels()
# yourself()
# a=plt.imread("aachen_000031_000019_gtFine_labelIds.png")
# lable_data("aachen_000031_000019_leftImg8bit.png",a)

# im = Image.open("miss.PNG")
# to_binary({"image": im, "is_tfl": b"00000001"})
# crop("miss.PNG", 50, 60)
