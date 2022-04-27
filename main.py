import argparse
import logging
from datetime import datetime

from util.run import Run
import dim_reduction.tsne
import util.base_logger
import util.report
import util.utils
from autoencoders import autoencoder
from preprocessing import standardisation
from util.base_logger import logger


def get_root():
    return globals().get("run_root")


def set_root(root):
    globals().update(run_root=root)


if __name__ == '__main__':

    run = Run.fromfile("./_config/standard_config.json")
    modes = ['test', 'cluster', 'full', 'init']
    processing_modes = ['gray-scale', 'otsu']




    parser = argparse.ArgumentParser(description="papyri-cvae arguments")

    parser.add_argument('--config', type=str)
    parser.add_argument('--generate', action="store_true")



    args = parser.parse_args()
    print(args.config)

    if args.config == None:
        config_file = "./_config/standard_config.json"
    else:
        config_file = args.config
    run = Run().fromfile(config_file)

    run.name = f'{run.name}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'


    run_path = f'out/{run.name}'

    run.root= run_path


    util.utils.create_folder(run.root)

    util.base_logger.set_FileHandler(run.root, run.name)

    util.report.set_mdPath(run.root)
    util.report.create_report(f"report-{run.name}", title=f"Report for {run.name}")
    util.report.write_to_report(f"This is your Report for the run {run.name}. It summarizes all calculations made.")

    logger.info(f'Program starting in {run.mode}-mode')
    if run.mode == 'init':
        logger.info('Program initialises folders')
        standardisation.generate_training_sets()
        logger.warning('Program has initialised itself - exiting program...')
        exit(0)



    logger.info('STARTING PROGRAM')
    logger.info('Selected parameters:')
    logger.info(run)


    util.report.header1("Config")
    util.report.write_to_report(config_file)
    run.write_to_md()




    if args.generate:
        standardisation.generate_training_sets()

    #dim_reduction.tsne.tsne(mode="raw-cleaned", folder='./data/raw-cleaned')

    standardisation.standardise(run)

    #dim_reduction.tsne.tsne(mode=run.mode, folder='./data/raw-cleaned-standardised')
    X, y = autoencoder.run_cae(run)

    print("Evaluating results and summarizing them in report")
    util.report.save_report()                               
    # unused
    # standardisation.png_to_ipx3()     
    # mnist_reader.mnist_read()

    # autoencoder.run_ae()
    # X, y = autoencoder.run_cae()

    # dim_reduction.custom_tsne.custom_tsne(X, y)
