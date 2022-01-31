import cv2
import os
import shutil


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
