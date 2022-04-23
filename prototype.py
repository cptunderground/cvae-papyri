import math

import cv2
import umap.umap_ as umap


from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import os

import numpy as np
import scipy
import torch

import torch.nn as nn

import pyro
import pyro.distributions as dist

import torch;
from torchvision import datasets, transforms
from tqdm import tqdm

torch.manual_seed(0)

import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt;plt.rcParams['figure.dpi'] = 200

from torch.utils.data import Dataset, DataLoader


class Decoder_pyro(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.fc21(hidden)
        return loc_img


class Encoder_pyro(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 500 hidden units
    def __init__(self, z_dim=50, hidden_dim=500, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder_pyro(z_dim, hidden_dim)
        self.decoder = Decoder_pyro(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = torch.sigmoid(self.decoder(z))

            pyro.sample("obs", dist.ContinuousBernoulli(probs=loc_img).to_event(1), obs=x.reshape(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = torch.sigmoid(self.decoder(z))
        out = loc_img / (2 * loc_img - 1) + 1. / (2 * torch.atanh(1 - 2 * loc_img))
        out[out == float('inf')] = 0.5
        return out.to(device)
        # return loc_img


def train(svi, train_loader, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    loop = tqdm(train_loader, total=len(train_loader))
    for d in loop:
        # if on GPU put mini-batch into CUDA memory
        x = d['re_image']
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


class ImgFolderWithLabel(datasets.ImageFolder):
    def __getitem__(self, item):
        img, target = super(ImgFolderWithLabel, self).__getitem__(item)
        return img, target


def loss_plot_pyro(loss):
    fig = plt.figure(figsize=(5, 5))
    plt.plot(np.arange(len(loss)), loss)
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.savefig("loss.png")
    plt.close()



class Papyri_dataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        re_image = image / 255
        target = int(self.targets[item])
        return {'re_image': torch.tensor(re_image.reshape(28, 28)[np.newaxis, :, :], dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)
                }


def plot_latent_var_pyro(epoch, autoencoder, data, nei, num_batches=100):
    print("in umap plot")
    stack = []
    stacky = []
    autoencoder = autoencoder.eval()
    with torch.no_grad():
        for i, d in enumerate(data):

            x = d['re_image']
            y = d['target'].to('cpu').detach().numpy().tolist()
            z, sigma = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy().tolist()
            stack.extend(z)
            stacky.extend(y)
            if i > num_batches:

                umaper = umap.UMAP(n_components=2, n_neighbors=nei)
                x_umap = umaper.fit_transform(stack)
                plt.scatter(x_umap[:, 0], x_umap[:, 1], s=2, c=stacky, cmap='tab10')
                plt.colorbar()
                plt.xlabel('UMAP 1')
                plt.ylabel('UMAP 2')


                break

    plt.savefig(f"./umap/umap{epoch}.png")
    plt.close()



class Mnist_dataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        re_image = image / 255
        target = int(self.targets[item])
        return {'re_image': torch.tensor(re_image.reshape(28, 28)[np.newaxis, :, :], dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.long)
                }


if __name__ == '__main__':
    # assert pyro.__version__.startswith('1.7.0')
    print(f"start programm")

    dataset = datasets.ImageFolder(
        './data/raw-cleaned-standardised',
        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )

    dataloader = DataLoader(dataset)
    images = dataset.imgs
    targets = dataset.targets
    samples = len(images)
    test_samples = math.floor(samples / 10)
    train_samples = samples - test_samples


    print(f"test_samples={test_samples}")
    print(f"trains_samples={train_samples}")

    X = []

    for i in images:
        img = cv2.imread(str(i[0]).replace("\\\\", "/"), 0)
        if img is not None:
            X.append(img)

    X = np.array(X)
    X = np.reshape(X,(X.shape[0], X.shape[1]**2))
    y = np.array(targets)



    #_X, _y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    #_train_samples = 60000
    #_test_samples =10000
    #print(_X,_y)

    print(X,y)

    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=test_samples, test_size=test_samples)

    pyro.distributions.enable_validation(False)
    pyro.set_rng_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_CUDA = True if torch.cuda.is_available() else False

    LEARNING_RATE = 3.0e-4

    NUM_EPOCHS = 30

    i_dataset = Mnist_dataset(images=X_train, targets=y_train)

    i_data_loader = torch.utils.data.DataLoader(i_dataset, num_workers=8)

    pyro.clear_param_store()

    # setup the VAE
    vae = VAE(use_cuda=USE_CUDA)

    # setup the optimizer
    adam_args = {"lr": LEARNING_RATE}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(NUM_EPOCHS):
        total_epoch_loss_train = train(svi, i_data_loader, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        plot_latent_var_pyro(epoch, vae, i_data_loader, 100)
        print("epoch end")

    torch.save(vae.state_dict(), f'./models/models-vae.pth')
    loss_plot_pyro(train_elbo)
    plot_latent_var_pyro(epoch, vae, i_data_loader, 100)
