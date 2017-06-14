import cv2
import numpy as np
import matplotlib.pyplot as plt
from carnd_vehicle_detection import ROOT_DIR
from random import random
import os
from glob import glob
from matplotlib import image as mpimg

MY_VEHICLE_IMGS_EXPR = os.path.join(ROOT_DIR, 'images', 'vehicles', "my_additional_images", "*png")
MY_AUGMENTED_VEHICLE_IMGS_DIR = os.path.join(ROOT_DIR, 'images', 'vehicles', "my_additional_images_augmented")
MY_NONVEHICLE_IMGS_EXPR = os.path.join(ROOT_DIR, 'images', 'nonvehicles', "my_additional_images", "*png")
MY_AUGMENTED_NONVEHICLE_IMGS_DIR = os.path.join(ROOT_DIR, 'images', 'nonvehicles', "my_additional_images_augmented")

VEHICLE_FILENAMES = glob(MY_VEHICLE_IMGS_EXPR)
NONVEHICLE_FILENAMES = glob(MY_NONVEHICLE_IMGS_EXPR)


def change_filenames(filenames=VEHICLE_FILENAMES):
    for filename in filenames:
        splitted = os.path.split(filename)
        last_part = splitted[-1]
        prefix = splitted[:-1]
        last_part = change_spaces_to_underscores(last_part)
        os.rename(filename, os.path.join(os.path.join(*prefix), last_part))
        # print(os.path.join(os.path.join(*prefix), last_part))

def change_spaces_to_underscores(string):
    return string.replace(" ", "_")


def read_images(filenames=VEHICLE_FILENAMES):
    return [(mpimg.imread(filename)[:, :, :3] * 255).astype(np.uint8) for filename in filenames]



def resize_and_save_images(images, filenames):
    for image, fname in zip(images, filenames):
        image = cv2.resize(image, (64, 64))
        print("about to save the image with the old name, size is now {}".format(image.shape))
        mpimg.imsave(fname, image)


def index_filename(filename, index):
    splitted = os.path.split(filename)
    base_path = os.path.join(*splitted[:-1])
    last_part = splitted[-1]
    dot_splitted = last_part.split(".")
    base_name = ".".join(dot_splitted[:-1]) + "_{}.".format(index) + dot_splitted[-1]
    new_filename = os.path.join(base_path, base_name)
    return new_filename.replace("my_additional_images", "my_additional_augmented_images")

def save_rotated_scaled_versions(filenames, images, rounds):
    for round in range(rounds):

        rotation_angles = [18 * (random() - 0.5) for _ in range(len(images))]
        scale_coeffs = [1.1 + 0.2 * random() for _ in range(len(images))]

        for ind, img_and_filename in enumerate(zip(images, filenames)):
            img, filename = img_and_filename
            print("round {}, ind {}, filename {}".format(round, ind, filename))
            matr = cv2.getRotationMatrix2D((32, 32), rotation_angles[ind], scale_coeffs[ind])
            new_image = cv2.warpAffine(img, matr, (64, 64))
            new_image_filename = index_filename(filename, "round_{}_image_{}".format(round, ind))
            # print(new_image_filename)
            mpimg.imsave(new_image_filename, new_image[:, :, :3])


if __name__ == "__main__":
    fnames = VEHICLE_FILENAMES
    images = read_images(fnames)
    resize_and_save_images(images, fnames)
    images = read_images(fnames)
    save_rotated_scaled_versions(fnames, images, 130)

