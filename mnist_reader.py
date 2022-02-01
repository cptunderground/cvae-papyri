import gzip
import random

from mnist import MNIST

def mnist_read():
    with gzip.open('data/raw/compressed/train-images-idx3-ubyte.gz', 'r') as fin:
        for line in fin:
            print('got line', line)

    with gzip.open('data/raw/compressed/t10k-images-idx3-ubyte.gz', 'r') as fin:
        for line in fin:
            print('got line', line)


    mndata = MNIST('data/raw/uncompressed')

    print("TRAINING IMAGES")
    images, labels = mndata.load_training()
    for index in range (0, len(images)):
        print(mndata.display(images[index]))
        print(labels[index])

    print("TEST IMAGES")
    images, labels = mndata.load_testing()
    for index in range (0, len(images)):
        print(mndata.display(images[index]))
        print(labels[index])