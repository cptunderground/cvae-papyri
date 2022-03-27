# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import dim_reduction.custom_tsne
import dim_reduction.tsne
from autoencoders import autoencoder
from preprocessing import standardisation

if __name__ == '__main__':


    dimension = 28

    #standardisation.generate_training_sets()
    #standardisation.standardise(dimension)

    #unused
    #standardisation.png_to_ipx3()
    #mnist_reader.mnist_read()

    #autoencoder.run_ae()
    autoencoder.run_cae()

    #dim_reduction.custom_tsne.custom_tsne()
    #dim_reduction.tsne.tsne()