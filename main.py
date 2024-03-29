import argparse
from datetime import datetime

import torch

from util import decorators
import util.base_logger
import util.report
import util.utils
from autoencoders import evaluate_resnetAE, evaluate_frag_CVAE, evaluate_convAE
from autoencoders import evaluate_char_CVAE
from autoencoders import train_resnetAE
from autoencoders import train_covAE
from autoencoders.char_CVAE import char_CVAE
from autoencoders.char_CVAE import train_char_cvae

from autoencoders.frag_CVAE import frag_CVAE
from autoencoders.frag_CVAE import train_frag_cvae

from preprocessing import standardisation
from util.base_logger import logger
from util.config import Config
from util.result import Result


@decorators.timed
def run():
    modes = ['test', 'cluster', 'full', 'init']
    processing_modes = ['gray-scale', 'otsu']

    parser = argparse.ArgumentParser(description="papyri-cvae arguments")

    parser.add_argument('--config', type=str)
    parser.add_argument('--generate', action="store_true")

    args = parser.parse_args()
    print(args.config)

    if args.config is None:
        config_file = "./_config/standard_config.json"
    else:
        config_file = args.config

    config = Config().fromfile(config_file)

    if config.name_time is None:
        config.name_time = f'{config.name}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        run_path = f'out/{config.name_time}'
        config.root = run_path

        util.utils.create_folder(config.root)

        config.saveJSON()

        util.base_logger.set_FileHandler(config.root, config.name)

        util.report.set_mdPath(config.root)
        util.report.create_report(f"report-{config.name}", title=f"Report for {config.name}")
        util.report.write_to_report(
            f"This is your Report for the run {config.name}. It summarizes all calculations made.")

        logger.info('STARTING PROGRAM')
        logger.info('Selected parameters:')
        logger.info(config)

        util.report.header1("Config")
        util.report.write_to_report(config_file)
        config.write_to_md()

    # if args.generate:
    # standardisation.generate_training_sets()
    # standardisation.standardise(config)

    if config.train:
        #result, config = train_covAE.train(config)
        result, config = train_resnetAE.train(config)
        result, config = train_char_cvae(config, result)
        result, config = train_frag_cvae(config, result)

        config.train = False
        out_path = result.saveJSON()
        config.result_path = out_path
        config.saveJSON()

    if config.evaluate:
        result = Result.fromfile(config.result_path)

        #evaluate_convAE.evaluate(config, result)
        evaluate_resnetAE.evaluate(config, result)
        evaluate_char_CVAE.evaluate(config, result)
        evaluate_frag_CVAE.evaluate(config, result)

    print("Evaluating results and summarizing them in report")
    # util.report.save_report()


if __name__ == '__main__':
    run()
