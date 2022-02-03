# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import autoencoder
import mnist_reader
import standardisation



if __name__ == '__main__':
    print('starting main')

    dimension = 28

    standardisation.standardise(dimension)
    standardisation.png_to_ipx3()
    mnist_reader.mnist_read()

    autoencoder.run_ae()