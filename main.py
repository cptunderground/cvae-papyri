# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import autoencoder
import mnist_reader
import standardisation



if __name__ == '__main__':


    dimension = 28

    standardisation.generate_training_sets()
    standardisation.standardise(dimension)

    #unused
    #standardisation.png_to_ipx3()
    #mnist_reader.mnist_read()

    #autoencoder.run_ae()
    autoencoder.run_cae()