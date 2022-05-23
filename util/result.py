import json


class Result:
    def __init__(self, root, name):
        self.root = root
        self.name = f"results-{name}"

        self.model = None
        self.epochs = None
        self.batch_size = None

        self.optimizer = None
        self.optimizer_args = None
        self.loss = None
        self.loss_args = None
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

    def __repr__(self):
        return str(self._toJSON())

    @classmethod
    def fromfile(self, path_to_file):
        with open(path_to_file, encoding="UTF-8") as file:
            data = json.load(file)
            run = Result(name=data["name"],
                         train=data["train"],
                         evaluate=data["evaluate"],
                         model_class=data["model_class"],
                         model_path=data["model_path"],
                         letters_to_eval=data["letters_to_eval"],
                         logging_val=data["logging_val"],
                         batch_size=data["batch_size"],
                         epochs=data["epochs"],
                         tqdm=data["tqdm"])
            return run

    def _toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def saveJSON(self) -> None:
        if self.root != None:
            with open(f'./{self.root}/{self.name}.json', 'w') as outfile:
                json.dump(self._toJSON(), outfile)
        else:
            pass

