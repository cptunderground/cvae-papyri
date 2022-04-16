import os
from datetime import datetime

import dim_reduction.custom_tsne
import dim_reduction.tsne
import util.utils
import util.report
import util.base_logger
from autoencoders import autoencoder
from preprocessing import standardisation
from util.base_logger import logger
import argparse
import logging


def get_root():
    return globals().get("run_root")


def set_root(root):
    globals().update(run_root=root)


if __name__ == '__main__':

    modes = ['test', 'cluster', 'full', 'init']
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



    parser = argparse.ArgumentParser(description="papyri-cvae arguments")
    parser.add_argument('mode', metavar='MODE', type=str, nargs=1,
                        help=f'select the mode for running the program. Available modes: {modes}')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--processing_mode', type=str)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--name', type=str)

    args = parser.parse_args()
    mode = args.mode[0]
    name = args.name if (args.name is not None) else 'unnamed'
    name = f'{name}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'
    print(name)

    run_path = f'out/{name}'

    run_root = run_path
    util.utils.set_root(run_root)

    util.utils.create_folder(run_path)

    util.base_logger.set_FileHandler(run_root, name)

    util.report.set_mdPath(run_root)
    util.report.create_report(f"report-{name}", title=f"Report for {name}")
    util.report.write_to_report(f"This is your Report for the run {name}. It summarizes all calculations made.")

    logger.info(f'Program starting in {mode}-mode')
    if mode == 'init':
        logger.info('Program initialises folders')
        standardisation.generate_training_sets()
        logger.warning('Program has initialised itself - exiting program...')
        exit(0)

    if not (mode in modes):
        logger.info(f'Your given mode \'{mode}\' was not found in the available modes of {modes}.')
        logger.info(f'Starting program in default mode')
        logger.setLevel(level=logging.INFO)
        dimension = 28
        epochs = 30
        processing_mode = 'gray-scale'
        tqdm_mode = True

    else:
        selected_config = eval(f'{mode}_config')
        logger.setLevel(level=selected_config['logging_lvl'])
        dimension = selected_config['dimension']
        epochs = selected_config['epochs']
        processing_mode = selected_config['processing_mode']
        tqdm_mode = selected_config['tqdm']

    if args.epochs is not None:
        epochs = args.epochs
    if args.processing_mode is not None:
        processing_mode = args.processing_mode

    logger.info('STARTING PROGRAM')
    logger.info('Selected parameters:')
    logger.info(f'epochs={epochs}')
    logger.info(f'dimension={dimension}')
    logger.info(f'processing_mode={processing_mode}')
    logger.info(f'tqdm_mode={tqdm_mode}')

    if args.generate:
        standardisation.generate_training_sets()

    # dim_reduction.tsne.tsne(mode="raw-cleaned", folder='./data/raw-cleaned')

    standardisation.standardise(dimension=dimension, mode=processing_mode)

    dim_reduction.tsne.tsne(mode=processing_mode, folder='./data/raw-cleaned-standardised')
    X, y = autoencoder.run_cae(epochs=epochs, mode=processing_mode, tqdm_mode=tqdm_mode)

    if args.report:
        print("Evaluating results and summarizing them in report")

    util.report.save_report()
    # unused
    # standardisation.png_to_ipx3()
    # mnist_reader.mnist_read()

    # autoencoder.run_ae()
    # X, y = autoencoder.run_cae()

    # dim_reduction.custom_tsne.custom_tsne(X, y)
