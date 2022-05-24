import json


class Result:
    def __init__(self, root, name):
        self.root = root
        self.name = f"result-{name}"

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
            result = Result(name=data["name"],
                            root=data["root"])

            result.model = data["model"]
            result.epochs = data["epochs"]
            result.batch_size = data["batch_size"]

            result.optimizer = data["optimizer"]
            result.optimizer_args = data["optimizer_args"]
            result.loss = data["loss"]
            result.loss_args = data["loss_args"]

            result.train_loss = data["train_loss"]
            result.valid_loss = data["valid_loss"]
            result.test_loss = data["test_loss"]

            return result

    def _toJSON(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)

    def saveJSON(self) -> None:
        if self.root != None:
            out_path = f'./{self.root}/{self.name}.json'
            with open(out_path, 'w') as outfile:
                json.dump(self.__dict__, outfile)
            return out_path
        else:
            pass
