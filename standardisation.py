import gzip

import cv2
from PIL import Image
import os
import shutil
from array import *
from random import shuffle

def png_to_ipx3():
    Names = [['./data/training-data-standardised', 't10k'], ['./data/test-data-standardised', 'test']]

    for name in Names:

        data_image = array('B')
        data_label = array('B')

        FileList = []
        #for dirname in os.listdir(name[0])[1:]:  # [1:] Excludes .DS_Store from Mac OS
        path = f"{name[0]}" #os.path.join(name[0], dirname)
            #print(path)
        for filename in os.listdir(path):
            print((name[0], filename))
            if filename.endswith(".png"):
                FileList.append(f"{name[0]}/{filename}") #os.path.join(name[0], dirname, filename))

        shuffle(FileList)  # Usefull for further segmenting the validation set

        for filename in FileList:

            label = 0 #int(filename.split('/')[2])

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

        output_file = open(name[1] + '-images-idx3-ubyte.gz', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_image)
        #output_file.close()

        output_file = open(name[1] + '-labels-idx1-ubyte.gz', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_label)

        output_file = open(name[1] + '-images-idx3-ubyte', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_image)
        # output_file.close()

        output_file = open(name[1] + '-labels-idx1-ubyte', 'wb')
        with gzip.open(output_file, "wb") as f:
            f.write(data_label)

    # gzip resulting files




def crop_img(img):
    print(img.shape)  # Print image shape
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

    print(cropped_img.shape)
    return cropped_img


def scale_img(img, px):
    dim = (px, px)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized_img


def otsu(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return otsu_img[1]




def standardise():
    print(f"cv2.version={cv2.__version__}")

    path_train = "./data/training-data"
    path_test = "./data/test-data"
    path_train_std = "./data/training-data-standardised"
    path_test_std = "./data/test-data-standardised"

    paths = []
    paths.append((path_train, path_train_std))
    paths.append((path_test, path_test_std))

    if (os.path.isdir(path_train_std)):
        shutil.rmtree(path_train_std)
    if (os.path.isdir(path_test_std)):
        shutil.rmtree(path_test_std)

    os.mkdir(path_train_std);
    os.mkdir(path_test_std);

    for tup in paths:
        src_directory = tup[0]
        dst_directory = tup[1]

        for file in os.listdir(src_directory):
            filename = os.fsdecode(file)
            if filename.endswith(".png"):
                # read img as bw
                print((src_directory, filename))
                img = cv2.imread(f'{src_directory}/{filename}', 0)

                img = otsu(img)
                img = crop_img(img)
                img = scale_img(img, 28)


                # save image to folder
                final_label = f'{dst_directory}/{filename[:-4]}-std.png'
                print(final_label)
                cv2.imwrite(final_label, img)
                continue
            else:
                continue
