"""Adopted from https://github.com/garythung/trashnet/blob/master/data/resize.py"""

import os

import numpy as np
import imageio.v2 as imageio

from PIL import Image

w = h = 256  # Dimensions of the resized image


def resize(image: np.ndarray, dim1: int, dim2: int) -> Image:
    """Resizes an image to the given dimensions."""
    return Image.fromarray(image).resize((dim1, dim2))


def loop_files(directory: str, file_path: str) -> None:
    """Loops through all the files in a given directory and resizes them."""
    try:
        os.makedirs(file_path)
    except OSError:
        if not os.path.isdir(file_path):
            raise

    for subdir, _, files in os.walk(directory):
        for file in files:
            if len(file) <= 4 or file[-4:] != '.jpg':
                continue

            pic = imageio.imread(os.path.join(subdir, file))
            dim1 = len(pic)
            dim2 = len(pic[0])
            if dim1 > dim2:
                pic = np.rot90(pic)

            resized_img = resize(pic, w, h)
            imageio.imsave(os.path.join(file_path, file), resized_img)


def main() -> None:
    """Main function to resize the images."""
    prepath = os.path.join(os.getcwd(), 'dataset-original')
    glass_dir = os.path.join(prepath, 'glass')
    paper_dir = os.path.join(prepath, 'paper')
    cardboard_dir = os.path.join(prepath, 'cardboard')
    plastic_dir = os.path.join(prepath, 'plastic')
    metal_dir = os.path.join(prepath, 'metal')
    trash_dir = os.path.join(prepath, 'trash')

    dest_path = os.path.join(os.getcwd(), 'dataset-resized')

    try:
        os.makedirs(dest_path)
    except OSError:
        if not os.path.isdir(dest_path):
            raise

    loop_files(glass_dir, os.path.join(dest_path, 'glass'))
    loop_files(paper_dir, os.path.join(dest_path, 'paper'))
    loop_files(cardboard_dir, os.path.join(dest_path, 'cardboard'))
    loop_files(plastic_dir, os.path.join(dest_path, 'plastic'))
    loop_files(metal_dir, os.path.join(dest_path, 'metal'))
    loop_files(trash_dir, os.path.join(dest_path, 'trash'))


if __name__ == '__main__':
    main()
