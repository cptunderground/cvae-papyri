import json
import logging

import torch


from util.base_logger import logger

modes = ['default', 'testing', 'cluster', 'full', 'init']
processing_modes = ['gray-scale', 'otsu']


class Run:
    def __init__(self, name: str = "unnamed", letters: list = None,
                 train: bool = False, model: str = f'./models/models-autoencodergray-scale.pth',
                 mode: str = "default", batch_size = 128, epochs: int = 30, dimensions: int = 28,
                 tqdm: bool = False, processing: str = "gray-scale"):
        checkArgs(mode, processing)

        self.name = name
        self.name_time = None

        self.train = train
        self.model = model
        #model_cls = autoencoders.autoencoder.Network()
        #self.model = autoencoders.autoencoder.ConvAutoEncoder(model_cls.load_state_dict(torch.load(model)))
        self.letters = letters
        self.logging = logging.ERROR
        self.mode = mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.dimensions = dimensions
        self.tqdm = tqdm
        self.processing = processing
        self.root = None

    @classmethod
    def fromfile(self, path_to_file):
        with open(path_to_file, encoding="UTF-8") as file:
            data = json.load(file)
            run = Run(name=data["name"], train=data["train"], letters=data["letters"],
                      mode=data["mode"], epochs=data["epochs"], dimensions=data["dimensions"],
                      tqdm=data["tqdm"], batch_size=data["batch_size"],
                      processing=data["processing"])
            return run

    def __repr__(self):
        return str(self.__dict__)

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def saveJSON(self) -> None:
        with open(f'{self.name}.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)

    def setRoot(self, root_path: str):
        self.root = root_path

    def getRoot(self):
        return self.root

    def write_to_md(self):
        pass


def checkArgs(mode, proc):
    ex = False
    if not mode in modes:
        logger.error(f"Selected mode \"{mode}\" is not supported. Please select one of the following modes: {modes}")
        ex = True
    if not proc in processing_modes:
        logger.error(
            f"Selected processing mode \"{proc}\" is not supported. Please select one of the following modes: {processing_modes}")
        ex = True
    if ex:
        exit(1)

if __name__ == '__main__':
    standard_conf = Run(name="standard_config", letters=['alpha'], train=True, mode="default", epochs=30, dimensions=28,
                        tqdm=False, processing="gray-scale")

    standard_conf.saveJSON()
    testi = Run.fromfile("./standard_config.json")
    print(testi)
