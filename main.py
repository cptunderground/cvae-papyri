# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import dim_reduction.custom_tsne
import dim_reduction.tsne
from autoencoders import autoencoder
from preprocessing import standardisation
import argparse
import logging

if __name__ == '__main__':

    modes = ['test', 'cluster', 'full']
    processing_modes = ['gray-scale', 'otsu']
    test_config = {
        'logging_lvl': logging.DEBUG,
        'epochs': 5,
        'dimension': 28,
        'processing_mode': 'gray-scale',
        'tqdm': True
    }
    cluster_config = {
        'logging_lvl': logging.INFO,
        'epochs': 30,
        'dimension': 28,
        'processing_mode': 'gray-scale',
        'tqdm': False
    }
    full_config = {
        'logging_lvl': logging.INFO,
        'epochs': 30,
        'dimension': 28,
        'processing_mode': 'gray-scale',
        'tqdm': True
    }

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="papyri-cvae arguments")
    parser.add_argument('mode', metavar='MODE', type=str, nargs=1,
                        help=f'select the mode for running the program. Availabel modes: {modes}')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--processing_mode', type=str)

    args = parser.parse_args()
    logging.info(args.epochs)
    mode = args.mode[0]

    if not (mode in modes):
        logging.info(f'Your given mode \'{mode}\' was not found in the available modes of {modes}.')
        logging.info(f'Starting program in default mode')
        logging.basicConfig(level=logging.INFO)
        dimension = 28
        epochs = 30
        processing_mode = 'gray-scale'
        tqdm_mode = True

    else:
        selected_config = eval(f'{mode}_config')
        logging.basicConfig(level=selected_config['logging_lvl'])
        dimension = selected_config['dimension']
        epochs = selected_config['epochs']
        processing_mode = selected_config['processing_mode']
        tqdm_mode = selected_config['tqdm']

    if args.epochs is not None:
        epochs = args.epochs
    if args.processing_mode is not None:
        processing_mode = args.processing_mode

    logging.info('STARTING PROGRAM')
    logging.info('Selected parameters:')
    logging.info(f'epochs={epochs}')
    logging.info(f'dimension={dimension}')
    logging.info(f'processing_mode={processing_mode}')
    logging.info(f'tqdm_mode={tqdm_mode}')

    standardisation.generate_training_sets()

    # dim_reduction.tsne.tsne(mode="raw-cleaned", folder='./data/raw-cleaned')

    standardisation.standardise(dimension=dimension, mode=processing_mode)
    dim_reduction.tsne.tsne(mode=processing_mode, folder='./data/raw-cleaned-standardised')
    X, y = autoencoder.run_cae(epochs=epochs, mode=processing_mode)

    # unused
    # standardisation.png_to_ipx3()
    # mnist_reader.mnist_read()

    # autoencoder.run_ae()
    # X, y = autoencoder.run_cae()

    # dim_reduction.custom_tsne.custom_tsne(X, y)
