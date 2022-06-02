import json


class Config:
    """
    Represents the JSON config for the auto-encoder as an instance during runtime.
    """

    def __init__(self, name: str = "unnamed", name_time: str = None, train: bool = False, evaluate: bool = True,
                 model_class: str = None, model_path: str = None, letters_to_eval: list = None, logging_val: int = 40,
                 batch_size: int = 128, epochs: int = 30, tqdm: bool = False, root=None, result_path=None):
        self.name = name
        self.name_time = name_time

        self.train = train
        self.evaluate = evaluate
        self.model_class = model_class
        self.model_path = model_path
        self.letters_to_eval = letters_to_eval
        self.logging_val = logging_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.tqdm = tqdm
        self.root = root

        self.result_path = result_path

    @classmethod
    def fromfile(cls, path_to_file):
        with open(path_to_file, encoding="UTF-8") as file:
            data = json.load(file)
            config = Config(name=data["name"],
                            name_time=data["name_time"],
                            train=data["train"],
                            evaluate=data["evaluate"],
                            model_class=data["model_class"],
                            model_path=data["model_path"],
                            letters_to_eval=data["letters_to_eval"],
                            logging_val=data["logging_val"],
                            batch_size=data["batch_size"],
                            epochs=data["epochs"],
                            tqdm=data["tqdm"],
                            root=data["root"],
                            result_path=data["result_path"])
            return config

    def __repr__(self):
        return str(self._toJSON())

    def _toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def saveJSON(self) -> None:
        if self.root is not None:
            with open(f'./{self.root}/{self.name}.json', 'w') as outfile:
                json.dump(self.__dict__, outfile)
        else:
            with open(f'../_config/{self.name}.json', 'w') as outfile:
                json.dump(self.__dict__, outfile)

    def write_to_md(self):
        pass


if __name__ == '__main__':
    # __debug = Config().fromfile("../_config/__debug.json")

    standard_conf = Config(name="standard",
                           train=True,
                           evaluate=True,
                           model_class="resnet_AE",
                           model_path=None,
                           letters_to_eval=["alpha", "beta"],
                           logging_val=40,
                           batch_size=128,
                           epochs=30,
                           tqdm=True)

    cluster_3_gpu = Config(name="cluster_3_gpu",
                           train=True,
                           evaluate=True,
                           model_class="resnet_AE",
                           model_path=None,
                           letters_to_eval=["alpha", "beta"],
                           logging_val=40,
                           batch_size=512,
                           epochs=3,
                           tqdm=False)

    cluster_30_gpu = Config(name="cluster_30_gpu",
                            train=True,
                            evaluate=True,
                            model_class="resnet_AE",
                            model_path=None,
                            letters_to_eval=["alpha", "beta"],
                            logging_val=40,
                            batch_size=512,
                            epochs=30,
                            tqdm=False)

    cluster_100_gpu = Config(name="cluster_100_gpu",
                             train=True,
                             evaluate=True,
                             model_class="resnet_AE",
                             model_path=None,
                             letters_to_eval=["alpha", "beta"],
                             logging_val=40,
                             batch_size=512,
                             epochs=100,
                             tqdm=False)

    cluster_1000_gpu = Config(name="cluster_1000_gpu",
                              train=True,
                              evaluate=True,
                              model_class="resnet_AE",
                              model_path=None,
                              letters_to_eval=["alpha", "beta"],
                              logging_val=40,
                              batch_size=512,
                              epochs=1000,
                              tqdm=False)

    testing = Config(name="testing",
                     train=True,
                     evaluate=True,
                     model_class="resnet_AE",
                     model_path=None,
                     letters_to_eval=["alpha", "beta"],
                     logging_val=40,
                     batch_size=128,
                     epochs=2,
                     tqdm=True)

    standard_conf.saveJSON()
    cluster_3_gpu.saveJSON()
    cluster_30_gpu.saveJSON()
    cluster_100_gpu.saveJSON()
    cluster_1000_gpu.saveJSON()
    testing.saveJSON()
