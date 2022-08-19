import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import util
from util import c_transforms
from util import decorators
from util.base_logger import logger
from util.config import Config
from util.enc_dataset import EncodedDataset
from util.result import Result

input_size = 48
labels_length = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_labels_char = ["alpha", "beta", "chi", "delta", "epsilon", "eta", "gamma", "iota", "kappa", "lambda", "mu", "nu",
                   "omega", "omicron", "phi", "pi", "psi", "rho", "sigma", "tau", "theta", "xi", "ypsilon", "zeta"]


class char_CVAE(nn.Module):
    def __init__(self, input_size, hidden_size=48):  # 30, 48, 96 for hidden_size for characters, 100 for frags
        super(char_CVAE, self).__init__()
        input_size_with_label = input_size + labels_length
        hidden_size += labels_length

        self.fc1 = nn.Linear(input_size_with_label,
                             input_size_with_label)  # output <= to 48 + 24 (input size + labels) for chars, larger for frags
        # if needed one more fc layer
        self.fc21 = nn.Linear(input_size_with_label, hidden_size)
        self.fc22 = nn.Linear(input_size_with_label, hidden_size)

        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, input_size_with_label)
        self.fc4 = nn.Linear(input_size_with_label, input_size)

    def encode(self, x, labels):
        x = x.view(-1, 1 * 48)
        x = torch.cat((x, labels), 1)
        x = self.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def decode(self, z, labels):
        torch.cat((z, labels), 1)
        z = self.relu(self.fc3(z))
        z = torch.sigmoid(self.fc4(z))
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, labels):
        # targets = one_hot(targets,labels_length-1).float().to(DEVICE)
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z, labels)
        return x, mu, logvar


def one_hot(x, max_x):
    return torch.eye(max_x + 1)[x]


def vae_loss_fn(x, recon_x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def plot_gallery(images, h, w, n_row=3, n_col=6):
    plt.figure(figsize=(2 * n_col, 2 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.axis("off")
        plt.imshow(images[i].reshape(h, w), cmap=matplotlib.cm.binary)
    plt.show()


def plot_loss(history):
    loss, val_loss = zip(*history)
    plt.figure(figsize=(15, 9))
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


def evaluate(losses, autoencoder, dataloader, flatten=True):
    model = lambda x, y: autoencoder(x, y)[0]
    loss_sum = []
    inp, out = [], []
    loss_fn = nn.MSELoss()

    label_encoder_char = preprocessing.LabelEncoder()
    label_encoder_char.fit(all_labels_char)

    for inputs, labels, frags in dataloader:
        inputs = inputs.to(DEVICE)
        labels = label_encoder_char.transform([labels])
        labels = one_hot(labels, 23).to(DEVICE)

        if flatten:
            inputs = inputs.view(inputs.size(0), 48)

        outputs = model(inputs, labels)
        loss = loss_fn(inputs, outputs)
        loss_sum.append(loss)
        inp = inputs
        out = outputs

    """
    with torch.set_grad_enabled(False):
        plot_gallery([inp[0].detach().cpu(), out[0].detach().cpu()], 64, 64, 1, 2)
    """
    losses.append((sum(loss_sum) / len(loss_sum)).item())


@decorators.timed
def _train_char_cvae(net, dataloader, test_dataloader, flatten=True, epochs=20):
    print("training")
    validation_losses = []
    optim = torch.optim.Adam(net.parameters())

    label_encoder_char = preprocessing.LabelEncoder()
    label_encoder_char.fit(all_labels_char)

    log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        for i in range(epochs):
            for batch, labels, frags in dataloader:

                labels = label_encoder_char.transform([labels])

                batch = batch.to(DEVICE)
                labels = one_hot(labels, 23).to(DEVICE)

                if flatten:
                    batch = batch.view(batch.size(0), 48)

                optim.zero_grad()
                x, mu, logvar = net(batch, labels)
                loss = vae_loss_fn(batch, x[:, :48], mu, logvar)
                loss.backward()
                optim.step()
            evaluate(validation_losses, net, test_dataloader, flatten=True)
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=i + 1, v_loss=validation_losses[i]))
    plt.show()
    return validation_losses


def get_latent_data(net, dataset, count=1000, is_cvae=False):
    latent_vectors = []
    latent_labels = []
    img_inputs = []
    rounds = count / 100
    i = 0

    label_encoder_char = preprocessing.LabelEncoder()
    label_encoder_char.fit(all_labels_char)

    with torch.set_grad_enabled(False):
        dataset_loader = DataLoader(dataset, batch_size=100, shuffle=True)
        for inputs, labels, frags in dataset_loader:
            inputs = inputs.to(DEVICE)

            labels = label_encoder_char.transform(labels)
            labels_one_hot = one_hot(labels, 23).to(DEVICE)
            if is_cvae:
                outputs, mu, logvar = net(inputs, labels_one_hot)
            else:
                outputs = net(inputs, labels_one_hot)
            outputs = outputs.cpu()
            if i == 0:
                latent_vectors = outputs
                latent_labels = labels
                img_inputs = inputs
            else:
                latent_vectors = torch.cat((latent_vectors, outputs), 0)
                latent_labels = torch.cat((latent_labels, labels), 0)
                img_inputs = torch.cat((img_inputs, inputs), 0)
            if i > rounds:
                break
            i += 1
    return img_inputs, latent_vectors, latent_labels


def plot_tsne(net, mode, count, dataset, is_cvae=False):
    img_inputs, latent_vectors, latent_labels = get_latent_data(net=net, count=count, is_cvae=is_cvae, dataset=dataset)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title('t-SNE')
    coords = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors)
    if mode == 'imgs':
        for image, (x, y) in zip(img_inputs.cpu(), coords):
            im = OffsetImage(image.reshape(28, 28), zoom=1, cmap='gray')
            ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
            ax.add_artist(ab)
        ax.update_datalim(coords)
        ax.autoscale()
    elif mode == 'dots':
        classes = latent_labels
        plt.scatter(coords[:, 0], coords[:, 1], c=classes)
        plt.colorbar()
        for i in range(10):
            class_center = np.mean(coords[classes == i], axis=0)
            text = TextArea('{}'.format(i))
            ab = AnnotationBbox(text, class_center, xycoords='data', frameon=True)
            ax.add_artist(ab)
    plt.show()


@decorators.timed
def split_data(_dataset, random_state=42):
    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=random_state)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=random_state)
    return _trainset, _testset, _validset


def train_char_cvae(config: Config, result: Result):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = transforms.Compose([
        c_transforms.CustomPad(padding=[0, 0, 0, 0], fill=(255, 255, 255, 1)),
        transforms.Resize([64, 64]),
        transforms.Grayscale()
    ])

    t_prime = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # _dataset = datasets.ImageFolder('./data/raw', transform=transforms.Compose([t, t_prime]))

    print("loading data")

    _dataset = EncodedDataset(config.model_path, './data/raw-cleaned-custom',
                              transform=transforms.Compose([t, t_prime]))

    print("data loaded")

    random_state = 42

    print("splitting")

    _trainset, _testset, _validset = split_data(_dataset)

    batch = config.batch_size

    print("splitted")
    print("creating loaders")

    train_loader = torch.utils.data.DataLoader(_trainset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(_validset, batch_size=batch)

    print("loaders created")
    print("Setting up CVAE")
    cvae = char_CVAE(48).to(DEVICE)
    losses_train = []
    losses_valid = []
    losses_test = []
    optimal_model = None
    current_valid_loss = 10000

    num_epochs = config.epochs

    print("training")
    validation_losses = []

    optim = torch.optim.Adam(cvae.parameters())

    label_encoder_char = preprocessing.LabelEncoder()
    label_encoder_char.fit(all_labels_char)

    for epoch in range(num_epochs):
        cum_train_loss = 0
        cum_valid_loss = 0
        cum_test_loss = 0

        ###################
        # train the models #
        ###################

        if config.tqdm:
            loop_train = tqdm(train_loader, total=len(train_loader))
        else:
            loop_train = train_loader
        log_template = "\nEpoch {ep:03d} val_loss {v_loss:0.4f}"

        for batch, labels, frags in loop_train:
            labels = label_encoder_char.transform(labels)

            batch = batch.to(DEVICE)
            labels = one_hot(labels, 23).to(DEVICE)

            batch = batch.view(batch.size(0), 48)

            optim.zero_grad()
            x, mu, logvar = cvae(batch, labels)
            loss = vae_loss_fn(batch, x[:, :48], mu, logvar)
            loss.backward()
            optim.step()

            cum_train_loss += loss.item()

            if config.tqdm:
                loop_train.set_description(f'char_CVAE - Training Epoch  [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_train_loss)
            else:
                logger.info(f'char_CVAE - Training Epoch  [{epoch + 1:2d}/{num_epochs}]')

        tqdm._instances.clear()

        ######################
        # validate the model #
        ######################

        if config.tqdm:
            loop_valid = tqdm(valid_loader, total=len(valid_loader))
        else:
            loop_valid = valid_loader

        for batch, labels, frags in loop_valid:
            labels = label_encoder_char.transform(labels)

            batch = batch.to(DEVICE)
            labels = one_hot(labels, 23).to(DEVICE)

            batch = batch.view(batch.size(0), 48)

            cvae.eval()
            optim.zero_grad()
            with torch.no_grad():
                x, mu, logvar = cvae(batch, labels)
            loss_valid = vae_loss_fn(batch, x[:, :48], mu, logvar)

            cum_valid_loss += loss_valid.item()
            if config.tqdm:
                loop_train.set_description(f'char_CVAE - Validation Epoch  [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_train_loss)
            else:
                logger.info(f'char_CVAE - Validation Epoch  [{epoch + 1:2d}/{num_epochs}]')

        if current_valid_loss > cum_valid_loss:
            optimal_model = (cvae, epoch)
            current_valid_loss = cum_valid_loss

        tqdm._instances.clear()

        ##################
        # test the model #
        ##################

        if config.tqdm:
            loop_test = tqdm(test_loader, total=len(test_loader))
        else:
            loop_test = test_loader

        for batch, labels, frags in loop_test:
            labels = label_encoder_char.transform(labels)

            batch = batch.to(DEVICE)
            labels = one_hot(labels, 23).to(DEVICE)

            batch = batch.view(batch.size(0), 48)

            cvae.eval()
            optim.zero_grad()
            with torch.no_grad():
                x, mu, logvar = cvae(batch, labels)
            loss_test = vae_loss_fn(batch, x[:, :48], mu, logvar)

            if config.tqdm:
                loop_train.set_description(f'char_CVAE - Test Epoch  [{epoch + 1:2d}/{num_epochs}]')
            else:
                logger.info(f"char_CVAE - Test Epoch  [{epoch + 1:2d}/{num_epochs}]")

            cum_test_loss += loss_test.item()

        losses_train.append(cum_train_loss)
        losses_valid.append(cum_valid_loss)
        losses_test.append(cum_test_loss)
        print(f'Epoch={epoch} done.')

    torch.save(optimal_model[0], f'./{config.root}/char_CVAE-optimal-{optimal_model[1]}-{config.name_time}.pth')
    config.cvae_char_path = f'./{config.root}/char_CVAE-optimal-{optimal_model[1]}-{config.name_time}.pth'
    result.char_cvae = f'./{config.root}/char_CVAE-optimal-{optimal_model[1]}-{config.name_time}.pth'

    result.char_cvae_train_loss = losses_train
    result.char_cvae_valid_loss = losses_valid
    result.char_cvae_test_loss = losses_test

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_train)
    util.utils.create_folder(f"./{config.root}/net_eval/char_CVAE")
    plt.title("Train Loss - char_CVAE")
    plt.savefig(f"./{config.root}/net_eval/char_CVAE/char_CVAE_loss_train.png")
    # util.report.image_to_report("net_eval/loss_train.png", "Network Training Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_valid)
    util.utils.create_folder(f"./{config.root}/net_eval/char_CVAE")
    plt.title("Validation Loss - char_CVAE")
    plt.savefig(f"./{config.root}/net_eval/char_CVAE/char_CVAE_loss_valid.png")
    # util.report.image_to_report("net_eval/loss_valid.png", "Network Validation Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_test)
    util.utils.create_folder(f"./{config.root}/net_eval/char_CVAE")
    plt.title("Test Loss - char_CVAE")
    plt.savefig(f"./{config.root}/net_eval/char_CVAE/char_CVAE_loss_test.png")
    # util.report.image_to_report("net_eval/loss_test.png", "Network Test Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_train, label="train_loss")
    plt.plot(losses_valid, label="valid_loss")
    plt.plot(losses_test, label="test_loss")
    util.utils.create_folder(f"./{config.root}/net_eval/char_CVAE")
    plt.title("All Losses - Log Scale")
    plt.legend()
    # plt.yscale("log")
    plt.savefig(f"./{config.root}/net_eval/char_CVAE/char_CVAE_loss_all.png")
    # util.report.image_to_report("net_eval/loss_all.png", "Network All Loss")
    plt.close()

    return result, config
