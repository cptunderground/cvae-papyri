# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import dim_reduction.custom_tsne
import dim_reduction.tsne
from autoencoders import autoencoder
from preprocessing import standardisation

if __name__ == '__main__':
    dimension = 28
    epochs = 100



    standardisation.generate_training_sets()

    dim_reduction.tsne.tsne(mode="raw-cleaned", folder='./data/raw-cleaned')



    mode = standardisation.standardise(dimension=dimension, mode="gray-scale")
    dim_reduction.tsne.tsne(mode=mode, folder='./data/raw-cleaned-standardised')
    X, y = autoencoder.run_cae(epochs=epochs, mode=mode)

    mode = standardisation.standardise(dimension=dimension, mode="otsu")
    dim_reduction.tsne.tsne(mode=mode, folder='./data/raw-cleaned-standardised')
    X, y = autoencoder.run_cae(epochs=epochs, mode=mode)

    # unused
    # standardisation.png_to_ipx3()
    # mnist_reader.mnist_read()

    # autoencoder.run_ae()
    # X, y = autoencoder.run_cae()

    # dim_reduction.custom_tsne.custom_tsne(X, y)
