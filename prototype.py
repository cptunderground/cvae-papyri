import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util.base_logger import logger

import os
import cv2

if __name__ == '__main__':
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

    data = np.zeros((200,200))
    logger.info(data.shape)

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



                    continue
                else:
                    continue

    logger.info(f"Global maximum resolution={max_resolution} - file={max_resolution_name}")
    logger.info(f"Global minimum resolution={min_resolution} - file={min_resolution_name}")

    data = np.zeros((max_resolution,max_resolution))



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

                    data[img_width-1][img_height-1] += 1

                    continue
                else:
                    continue

    logger.info(data)

    nx, ny = max_resolution, max_resolution
    x = range(nx)
    y = range(ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)

    plt.show()