import argparse
from datetime import datetime

import util.base_logger
import util.report
import util.utils
from autoencoders import autoencoder
from preprocessing import standardisation
from util.base_logger import logger
from util.config import Config

if __name__ == '__main__':

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
    config = Config().fromfile(config_file)

    config.name_time = f'{config.name}-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'

    run_path = f'out/{config.name_time}'

    config.root = run_path

    util.utils.create_folder(config.root)

    util.base_logger.set_FileHandler(config.root, config.name)

    util.report.set_mdPath(config.root)
    util.report.create_report(f"report-{config.name}", title=f"Report for {config.name}")
    util.report.write_to_report(f"This is your Report for the run {config.name}. It summarizes all calculations made.")

    logger.info(f'Program starting in {config.mode}-mode')
    if config.mode == 'init':
        logger.info('Program initialises folders')
        standardisation.generate_training_sets()
        logger.warning('Program has initialised itself - exiting program...')
        exit(0)

    logger.info('STARTING PROGRAM')
    logger.info('Selected parameters:')
    logger.info(config)

    util.report.header1("Config")
    util.report.write_to_report(config_file)
    config.write_to_md()

    if args.generate:
        standardisation.generate_training_sets()
        standardisation.standardise(config)

    if config.train:
        autoencoder.train(config)
    else:
        autoencoder.evaluate(config)
    print("Evaluating results and summarizing them in report")
    util.report.save_report()
