import json
import logging
from base_logger import logger

modes = ['default', 'testing', 'cluster', 'full', 'init']
processing_modes = ['gray-scale', 'otsu']


class Run:
    def __init__(self, name: str = "unnamed", mode: str = "default", epochs: int = 30, dimensions: int = 28,
                 tqdm: bool = False, processing: str = "gray-scale"):
        checkArgs(mode, processing)

        self.name = name
        self.logging = logging.ERROR
        self.mode = mode
        self.epochs = epochs
        self.dimensions = dimensions
        self.tqdm = tqdm
        self.processing = processing
        self.root = None

    def __repr__(self):
        return str(self.__dict__)

    def toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def saveJSON(self) -> None:
        with open('run.json', 'w') as outfile:
            json.dump(self.__dict__, outfile)

    def setRoot(self, root_path: str):
        self.root = root_path

    def getRoot(self):
        return self.root


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


def configFromJSON(path) -> Run:
    with open('run.json') as json_file:
        data = json.load(json_file)

    run = Run(name=data["name"], mode=data["mode"], epochs=data["epochs"], dimensions=data["dimensions"], tqdm=data["tqdm"],
              processing=data["processing"])
    return run


if __name__ == '__main__':
    run1 = Run(mode="testing", epochs=3, dimensions=3, tqdm=True, processing="gray-scale")
    print(run1)
