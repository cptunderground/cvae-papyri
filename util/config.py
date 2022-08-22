import json

all_labels_char = ["alpha", "beta", "chi", "delta", "epsilon", "eta", "gamma", "iota", "kappa", "lambda", "mu", "nu",
                   "omega", "omicron", "phi", "pi", "psi", "rho", "sigma", "tau", "theta", "xi", "ypsilon", "zeta"]
all_labels_frags = [
    '61210', '61226', '61228', '60402', '59170', '60221', '60475', '60998', '60291', '60941', '60810', '60468',
    '61140', '60251', '60246', '60891', '60670', '60398', '60589', '60343', '60809', '61026', '60326', '60663',
    '60220', '60812', '60242', '60400', '60842', '60324', '61236', '60304', '60462', '60934', '61239', '60808',
    '61106', '60276', '61212', '61244', '60476', '60633', '60238', '61240', '60910', '61245', '61124', '60333',
    '61138', '60901', '60306', '60214', '61213', '61165', '61246', '60215', '60492', '60258', '60940', '60732',
    '60216', '60364', '60479', '60847', '60583', '61122', '60283', '60740', '60255', '65858', '60471', '60701',
    '61117', '60359', '61073', '60367', '60219', '60337', '60312', '60771', '61112', '60867', '60421', '60764',
    '60217', '60248', '60411', '60253', '60290', '60659', '60481', '61141', '66764', '60267', '60369', '60965']


class Config:
    """
    Represents the JSON config for the auto-encoder as an instance during runtime.
    """

    def __init__(self, name: str = "unnamed", name_time: str = None, train: bool = False, evaluate: bool = True,
                 model_class: str = None, model_path: str = None, cvae_char: str = None, cvae_char_path: str = None,
                 cvae_frag: str = None, cvae_frag_path: str = None, chars_to_eval: list = None,
                 chars_to_train: list = None, frags_to_eval: list = None, frags_to_train: list = None,
                 logging_val: int = 40, batch_size: int = 128, epochs: int = 30, epochs_cvae: int = 1000,
                 tqdm: bool = False, root=None,
                 result_path=None):

        self.name = name
        self.name_time = name_time

        self.train = train
        self.evaluate = evaluate

        self.model_class = model_class
        self.model_path = model_path

        self.cvae_char = cvae_char
        self.cvae_char_path = cvae_char_path

        self.cvae_frag = cvae_frag
        self.cvae_frag_path = cvae_frag_path

        self.logging_val = logging_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_cvae = epochs_cvae
        self.tqdm = tqdm
        self.root = root

        self.result_path = result_path

        self.chars_to_eval = chars_to_eval
        self.chars_to_train = chars_to_train

        self.frags_to_eval = frags_to_eval
        self.frags_to_train = frags_to_train

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

                            cvae_char=data["cvae_char"],
                            cvae_char_path=data["cvae_char_path"],

                            cvae_frag=data["cvae_frag"],
                            cvae_frag_path=data["cvae_frag_path"],

                            chars_to_eval=data["chars_to_eval"],
                            chars_to_train=data["chars_to_train"],
                            frags_to_eval=data["frags_to_eval"],
                            frags_to_train=data["frags_to_train"],

                            logging_val=data["logging_val"],
                            batch_size=data["batch_size"],
                            epochs=data["epochs"],
                            epochs_cvae=data["epochs_cvae"],
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

                           cvae_char="char_CVAE",
                           cvae_char_path=None,

                           cvae_frag="frag_CVAE",
                           cvae_frag_path=None,

                           chars_to_eval=all_labels_char,
                           chars_to_train=all_labels_char,
                           frags_to_eval=all_labels_frags,
                           frags_to_train=all_labels_frags,

                           logging_val=40,
                           batch_size=128,
                           epochs=30,
                           epochs_cvae=100,
                           tqdm=True)

    cluster_3_gpu = Config(name="cluster_3_gpu",
                           train=True,
                           evaluate=True,
                           model_class="resnet_AE",
                           model_path=None,
                           cvae_char="char_CVAE",
                           cvae_char_path=None,

                           cvae_frag="frag_CVAE",
                           cvae_frag_path=None,

                           chars_to_eval=all_labels_char,
                           chars_to_train=all_labels_char,
                           frags_to_eval=all_labels_frags,
                           frags_to_train=all_labels_frags,

                           logging_val=40,
                           batch_size=512,
                           epochs=3,
                           epochs_cvae=100,
                           tqdm=False)

    cluster_30_gpu = Config(name="cluster_30_gpu",
                            train=True,
                            evaluate=True,
                            model_class="resnet_AE",
                            model_path=None,
                            cvae_char="char_CVAE",
                            cvae_char_path=None,

                            cvae_frag="frag_CVAE",
                            cvae_frag_path=None,

                            chars_to_eval=all_labels_char,
                            chars_to_train=all_labels_char,
                            frags_to_eval=all_labels_frags,
                            frags_to_train=all_labels_frags,

                            logging_val=40,
                            batch_size=512,
                            epochs=30,
                            epochs_cvae=100,
                            tqdm=False)

    cluster_100_gpu = Config(name="cluster_100_gpu",
                             train=True,
                             evaluate=True,
                             model_class="resnet_AE",
                             model_path=None,
                             cvae_char="char_CVAE",
                             cvae_char_path=None,

                             cvae_frag="frag_CVAE",
                             cvae_frag_path=None,

                             chars_to_eval=all_labels_char,
                             chars_to_train=all_labels_char,
                             frags_to_eval=all_labels_frags,
                             frags_to_train=all_labels_frags,

                             logging_val=40,
                             batch_size=512,
                             epochs=100,
                             epochs_cvae=100,
                             tqdm=False)

    cluster_1000_gpu = Config(name="cluster_1000_gpu",
                              train=True,
                              evaluate=True,
                              model_class="resnet_AE",
                              model_path=None,

                              cvae_char="char_CVAE",
                              cvae_char_path=None,

                              cvae_frag="frag_CVAE",
                              cvae_frag_path=None,

                              logging_val=40,
                              batch_size=512,
                              epochs=1000,
                              epochs_cvae=100,
                              tqdm=False,

                              chars_to_eval=all_labels_char,
                              chars_to_train=all_labels_char,
                              frags_to_eval=all_labels_frags,
                              frags_to_train=all_labels_frags

                              )

    cluster_10000_gpu = Config(name="cluster_10000_gpu",
                               train=True,
                               evaluate=True,
                               model_class="resnet_AE",
                               model_path=None,

                               cvae_char="char_CVAE",
                               cvae_char_path=None,

                               cvae_frag="frag_CVAE",
                               cvae_frag_path=None,

                               logging_val=40,
                               batch_size=512,
                               epochs=1000,
                               epochs_cvae=20000,
                               tqdm=False,

                               chars_to_eval=all_labels_char,
                               chars_to_train=all_labels_char,
                               frags_to_eval=all_labels_frags,
                               frags_to_train=all_labels_frags

                               )

    testing = Config(name="testing",
                     train=True,
                     evaluate=True,
                     model_class="resnet_AE",
                     model_path=None,
                     cvae_char="char_CVAE",
                     cvae_char_path=None,

                     cvae_frag="frag_CVAE",
                     cvae_frag_path=None,

                     chars_to_eval=all_labels_char,
                     chars_to_train=all_labels_char,
                     frags_to_eval=all_labels_frags,
                     frags_to_train=all_labels_frags,

                     logging_val=40,
                     batch_size=128,
                     epochs=1,
                     epochs_cvae=2,
                     tqdm=True)

    standard_conf.saveJSON()
    cluster_3_gpu.saveJSON()
    cluster_30_gpu.saveJSON()
    cluster_100_gpu.saveJSON()
    cluster_1000_gpu.saveJSON()
    cluster_10000_gpu.saveJSON()
    testing.saveJSON()
