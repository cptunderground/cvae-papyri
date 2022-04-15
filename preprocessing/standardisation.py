import gzip
import math
import random
from time import sleep

import cv2
from PIL import Image
import os
import shutil
from array import *
from random import shuffle
import preprocessing.padding
import logging
from util.base_logger import logger

def png_to_ipx3():
    Names = [['./data/training-data-standardised/alpha', 'train'], ['./data/test-data-standardised/alpha', 't10k']]

    for name in Names:

        data_image = array('B')
        data_label = array('B')

        FileList = []
        # for dirname in os.listdir(name[0])[1:]:  # [1:] Excludes .DS_Store from Mac OS
        path = f"{name[0]}"  # os.path.join(name[0], dirname)
        # print(path)
        for filename in os.listdir(path):
            logger.debug((name[0], filename))
            if filename.endswith(".png"):
                FileList.append(f"{name[0]}/{filename}")  # os.path.join(name[0], dirname, filename))

        shuffle(FileList)  # Usefull for further segmenting the validation set

        for filename in FileList:

            label = 0  # int(filename.split('/')[2])

            Im = Image.open(filename)

            pixel = Im.load()

            width, height = Im.size

            for x in range(0, width):
                for y in range(0, height):
                    data_image.append(pixel[y, x])

            data_label.append(label)  # labels start (one unsigned byte each)

        hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

        # header for label array

        header = array('B')
        header.extend([0, 0, 8, 1, 0, 0])
        header.append(int('0x' + hexval[2:][:2], 16))
        header.append(int('0x' + hexval[2:][2:], 16))

        data_label = header + data_label

        # additional header for images array

        if max([width, height]) <= 256:
            header.extend([0, 0, 0, width, 0, 0, 0, height])
        else:
            raise ValueError('Image exceeds maximum size: 256x256 pixels');

        header[3] = 3  # Changing MSB for image data (0x00000803)

        data_image = header + data_image

        output_file = open(f'data/MNIST/raw/{name[1]}-images-idx3-ubyte.gz', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_image)
        # output_file.close()

        output_file = open(f'data/MNIST/raw/{name[1]}-labels-idx1-ubyte.gz', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_label)

        output_file = open(f'data/MNIST/raw/{name[1]}-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()
        # output_file.close()

        output_file = open(f'data/MNIST/raw/{name[1]}-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()


def crop_img(img):
    logger.debug(img.shape)  # Print image shape
    height, width = img.shape

    if (height > width):
        # crop height
        diff = height - width
        cropped_img = img[0:height - diff, 0:width]

    if (width > height):
        # crop width
        diff = width - height
        cropped_img = img[0:height, 0:width - diff]

    if (width == height):
        cropped_img = img

    logger.debug(cropped_img.shape)
    return cropped_img


def scale_img(img, px):
    dim = (px, px)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


def otsu(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_img[1]


def standardise(dimension=28, mode="gray-scale"):
    print(f"cv2.version={cv2.__version__}")

    path_raw = "./data/raw"

    path_train = "./data/training-data"
    path_test = "./data/test-data"
    path_raw_cleaned = "./data/raw-cleaned"

    path_train_std = "./data/training-data-standardised"
    path_test_std = "./data/test-data-standardised"
    path_raw_cleaned_std = "./data/raw-cleaned-standardised"


    paths = []
    paths.append((path_train, path_train_std))
    paths.append((path_test, path_test_std))
    paths.append((path_raw_cleaned, path_raw_cleaned_std))

    max_resolution = 0
    min_resolution = 1000
    max_resolution_name = None
    min_resolution_name = None

    for tup in paths:
        src_directory = tup[0]
        dst_directory = tup[1]

        logger.debug(src_directory, dst_directory)

        for dir in os.listdir(src_directory):
            logger.debug(dir)

            for file in os.listdir(f"{src_directory}/{dir}"):
                logger.debug(file)
                filename = os.fsdecode(file)

                if filename.endswith(".png"):
                    # read img as bw

                    logger.debug((src_directory, filename))

                    img = cv2.imread(f'{src_directory}/{dir}/{filename}', 0)

                    img_height = img.shape[0]
                    img_width = img.shape[1]

                    img_max_res = max(img_height, img_width)
                    img_min_res = min(img_height, img_width)

                    if (img_max_res > max_resolution):
                        max_resolution = img_max_res
                        max_resolution_name = filename

                    if (img_min_res < min_resolution):
                        min_resolution = img_min_res
                        min_resolution_name = filename

                    if (mode == "otsu"):
                        img = otsu(img)
                    #img = preprocessing.padding.pad(img,200)


                    img = crop_img(img)
                    img = scale_img(img, dimension)

                    # save image to folder
                    final_label = f'{dst_directory}/{dir}/{filename[:-4]}-std.png'
                    logger.debug(final_label)
                    cv2.imwrite(final_label, img)
                    continue
                else:
                    continue
    sleep(2)
    logger.info(f"Global maximum resolution={max_resolution} - file={max_resolution_name}")
    logger.info(f"Global minimum resolution={min_resolution} - file={min_resolution_name}")
    sleep(10)
    return mode

def setup():
    path_raw = "./data/raw"

    path_raw_cleaned = "./data/raw-cleaned"
    path_train = "./data/training-data"
    path_test = "./data/test-data"
    path_raw_cleaned_std = "./data/raw-cleaned-standardised"
    path_train_std = "./data/training-data-standardised"
    path_test_std = "./data/test-data-standardised"

    path_out = "./out"
    path_out_tsne = f'./out/tsne'
    path_out_tsne_covae = f'./out/tsne/CovAE'
    path_out_tsne_covae_otsu = f'./out/tsne/CovAE/otsu/'
    path_out_tsne_covae_gs = f'./out/tsne/CovAE/gray-scale/'

    paths = []
    paths.append(path_raw_cleaned)
    paths.append(path_raw_cleaned_std)
    paths.append(path_train)
    paths.append(path_train_std)
    paths.append(path_test)
    paths.append(path_test_std)

    paths_tup = []
    paths_tup.append((path_train, path_train_std))
    paths_tup.append((path_test, path_test_std))
    paths_tup.append((path_raw_cleaned, path_raw_cleaned_std))

    out_paths = []
    out_paths.append(path_out)
    out_paths.append(path_out_tsne)
    out_paths.append(path_out_tsne_covae)
    out_paths.append(path_out_tsne_covae_otsu)
    out_paths.append(path_out_tsne_covae_gs)

    for path in paths and out_paths:
        if (os.path.isdir(path)):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            logger.warning(f"{path} already exists")

def generate_training_sets():
    path_raw = "./data/raw"

    path_raw_cleaned = "./data/raw-cleaned"
    path_train = "./data/training-data"
    path_test = "./data/test-data"
    path_raw_cleaned_std = "./data/raw-cleaned-standardised"
    path_train_std = "./data/training-data-standardised"
    path_test_std = "./data/test-data-standardised"

    path_out = "./out"
    path_out_tsne = f'./out/tsne'
    path_out_tsne_covae = f'./out/tsne/CovAE'
    path_out_tsne_covae_otsu = f'./out/tsne/CovAE/otsu/'
    path_out_tsne_covae_gs = f'./out/tsne/CovAE/gray-scale/'


    paths = []
    paths.append(path_raw_cleaned)
    paths.append(path_raw_cleaned_std)
    paths.append(path_train)
    paths.append(path_train_std)
    paths.append(path_test)
    paths.append(path_test_std)

    paths_tup = []
    paths_tup.append((path_train, path_train_std))
    paths_tup.append((path_test, path_test_std))
    paths_tup.append((path_raw_cleaned, path_raw_cleaned_std))

    out_paths = []
    out_paths.append(path_out)
    out_paths.append(path_out_tsne)
    out_paths.append(path_out_tsne_covae)
    out_paths.append(path_out_tsne_covae_otsu)
    out_paths.append(path_out_tsne_covae_gs)

    for path in (paths + out_paths):
        if (os.path.isdir(path)):
            logger.warning(f"{path} already exists")
        else:
            os.mkdir(path)

    """
    if (os.path.isdir(path_train)):
        shutil.rmtree(path_train)
    if (os.path.isdir(path_test)):
        shutil.rmtree(path_test)
    if (os.path.isdir(path_train_std)):
        shutil.rmtree(path_train_std)
    if (os.path.isdir(path_test_std)):
        shutil.rmtree(path_test_std)
    if (os.path.isdir(path_raw_cleaned)):
        shutil.rmtree(path_raw_cleaned)

    os.mkdir(path_train)
    os.mkdir(path_test)
    os.mkdir(path_train_std)
    os.mkdir(path_test_std)
    os.mkdir(path_raw_cleaned)
    """
    with os.scandir(path_raw) as entries:

        total_files = 0
        total_train_files = 0
        total_test_files = 0
        for entry in entries:
            logger.debug(entry)

            for path in paths:
                if (os.path.isdir(f"{path}/{entry.name}")):
                    shutil.rmtree(f"{path}/{entry.name}")
                os.mkdir(f"{path}/{entry.name}")

            """
            if (os.path.isdir(f"{path_train}/{entry.name}")):
                shutil.rmtree(f"{path_train}/{entry.name}")
            os.mkdir(f"{path_train}/{entry.name}")

            if (os.path.isdir(f"{path_test}/{entry.name}")):
                shutil.rmtree(f"{path_test}/{entry.name}")
            os.mkdir(f"{path_test}/{entry.name}")

            if (os.path.isdir(f"{path_train_std}/{entry.name}")):
                shutil.rmtree(f"{path_train_std}/{entry.name}")
            os.mkdir(f"{path_train_std}/{entry.name}")

            if (os.path.isdir(f"{path_test_std}/{entry.name}")):
                shutil.rmtree(f"{path_test_std}/{entry.name}")
            os.mkdir(f"{path_test_std}/{entry.name}")

            if (os.path.isdir(f"{path_raw_cleaned}/{entry.name}")):
                shutil.rmtree(f"{path_raw_cleaned}/{entry.name}")
            os.mkdir(f"{path_raw_cleaned}/{entry.name}")
            
            if (os.path.isdir(f"{path_raw_cleaned_std}/{entry.name}")):
                shutil.rmtree(f"{path_raw_cleaned_std}/{entry.name}")
            os.mkdir(f"{path_raw_cleaned_std}/{entry.name}")
            """


            with os.scandir(f"{path_raw}/{entry.name}") as classes:
                num_files = len(os.listdir(f"{path_raw}/{entry.name}"))
                num_test = math.floor(num_files / 10)
                num_train = num_files - num_test

                logger.info(f"total files found in {entry.name}:{num_files}")
                logger.info(f"total training files in {entry.name}:{num_train}")
                logger.info(f"total testing files in {entry.name}:{num_test}")

                total_files += num_files
                total_train_files += num_train
                total_test_files += num_test

                files = set(os.listdir(f"{path_raw}/{entry.name}"))
                train_files = set(random.sample(files, num_train))
                test_files = files - train_files





                for file in files:
                    file_name = file
                    file_name = f'{entry.name}{file_name[1:-4].replace(".", "_")}.png'
                    shutil.copy(f'{path_raw}/{entry.name}/{file}', f'{path_raw_cleaned}/{entry.name}/{file_name}')


                for file in train_files:
                    file_name = file
                    file_name = f'{entry.name}{file_name[1:-4].replace(".", "_")}.png'
                    shutil.copy(f'{path_raw}/{entry.name}/{file}', f'{path_train}/{entry.name}/{file_name}')

                for file in test_files:
                    file_name = file
                    file_name = f'{entry.name}{file_name[1:-4].replace(".", "_")}.png'
                    shutil.copy(f'{path_raw}/{entry.name}/{file}', f'{path_test}/{entry.name}/{file_name}')

        logger.info(f"total files found:{total_files}")
        logger.info(f"total training files:{total_train_files}")
        logger.info(f"total testing files:{total_test_files}")