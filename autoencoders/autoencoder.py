import logging
import random
import time

import numpy as np
import torch
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist

from sklearn.manifold._t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction import image
from skimage import io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot
from matplotlib.pyplot import imshow
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from mnist import MNIST
from torchvision import datasets, transforms
from torchviz import make_dot

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 32, out_features=256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=256, out_features=5)

    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        t = t.reshape(t.size(0), -1)
        t = self.fc(t)
        t = self.out(t)

        return t


class ConvAutoEncoder(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=4,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self, t):
        enc = self.encoder(t)
        dec = self.decoder(enc)

        return enc, dec


def get_label(label):
    switcher = {
        "tensor(0)":"alpha",
        "tensor(1)":"beta",
        "tensor(2)":"delta",
        "tensor(3)":"epsilon",
        "tensor(4)":"gamma",
    }

    return switcher.get(label)

def run_cae(epochs=30, mode="not_selected", tqdm_mode=True):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = "CovAE"
    train_set = datasets.ImageFolder(
        # './data/training-data-standardised',
        './data/raw-cleaned-standardised',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )
    test_set = datasets.ImageFolder(
        './data/raw-cleaned-standardised',
        # './data/test-data-standardised',
        # './data/test-data-manual',
        # './data/test-data-manual-otsu',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2877)
    logging.debug(test_loader.batch_size)

    # take 5 random letters from testset

    pretrained_model = Network()
    logging.info(pretrained_model)
    # pretrained_model.load_state_dict(torch.load('models/pretrained/model-run(lr=0.001, batch_size=256).ckpt', map_location=device))

    model = ConvAutoEncoder(pretrained_model)
    logging.info(model)

    b = torch.randn(16, 1, 5, 5)

    input_names = ['Image']
    output_names = ['Label']
    torch.onnx.export(model, b, 'AE.onnx', input_names=input_names, output_names=output_names)


    summary(model, (1, 28, 28), 2592)
    # to check if our weight transfer was successful or not
    # list(list(pretrained_model.layer2.children())[0].parameters()) == list(
    #    list(model.encoder.children())[4].parameters())
    #
    for layer_num, child in enumerate(model.encoder.children()):
        if layer_num < 8:
            for param in child.parameters():
                param.requires_grad = False

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([  # parameters which need optimization
        {'params': model.encoder[8:].parameters()},
        {'params': model.decoder.parameters()}
    ], lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1 / 3, patience=3, verbose=True)

    num_epochs = epochs
    for epoch in range(num_epochs):
        train_loss = 0
        ###################
        # train the models #
        ###################
        if tqdm_mode: loop = tqdm(train_loader, total=len(train_loader))
        else: loop = train_loader

        for batch in loop:
            images = batch[0].to(device)
            _, outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            if tqdm_mode:
                loop.set_description(f'Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop.set_postfix(loss=train_loss)
        logging.info(f'Epoch={epoch} done.')

        scheduler.step(train_loss)

        torch.save(model.state_dict(), f'./models/models-autoencoder{mode}.pth')

        images, labels = next(iter(test_loader))
        # images, labels = next(iter(train_loader))
        images = images.to(device)

        # get sample outputs
        encoded_imgs, decoded_imgs = model(images)
        # prep images for display
        images = images.cpu().numpy()

        # use detach when it's an output that requires_grad
        encoded_imgs = encoded_imgs.detach().cpu().numpy()
        decoded_imgs = decoded_imgs.detach().cpu().numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, decoded_imgs], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)


        fig.savefig('images/original_decoded.png', bbox_inches='tight')
        plt.close()

        encoded_img = encoded_imgs[0]  # get the 7th image from the batch (7th image in the plot above)

        fig = plt.figure(figsize=(4, 4))
        for fm in range(encoded_img.shape[0]):
            ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
            ax.set_title(f'feature map: {fm}')
            ax.imshow(encoded_img[fm], cmap='gray')


        fig.savefig('images/encoded_img_alpha')
        plt.close()

        encoded_img = encoded_imgs[3]  # get 1st image from the batch (here '7')

        fig = plt.figure(figsize=(4, 4))
        for fm in range(encoded_img.shape[0]):
            ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
            ax.set_title(f'feature map: {fm}')
            ax.imshow(encoded_img[fm], cmap='gray')


        fig.savefig('images/encoded_img_epsilon')
        plt.close()

        # X, y = load_digits(return_X_y=True)

        data = []
        folder = './data/training-data-standardised'

        # print(encoded_imgs)
        # print(labels)
        # print(len(encoded_imgs))
        # print(len(labels))

        for i in range(len(encoded_imgs)):
            data.append([encoded_imgs[i], labels[i]])

        # print(data)

        features, images = zip(*data)
        y = images
        X = np.array(features)

        print(X.shape)
        X.reshape(-1)
        print(X.shape)
        X.ravel()
        print(X.shape)
        X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        print(X.shape)

        y_list = list(y)

        for item in range(len(y_list)):

            y_list[item] = get_label(str(y_list[item]))

        y = tuple(y_list)

        print(y)
        y_set = set(y)
        y_len = len(y_set)
        print(y_len)
        palette = sns.color_palette("bright", y_len)
        MACHINE_EPSILON = np.finfo(np.double).eps
        n_components = 2
        perplexity = 30

        # X_embedded = fit(X,y, MACHINE_EPSILON, n_components, perplexity)

        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
        # plt.show()

        tsne = TSNE()
        X_embedded = tsne.fit_transform(X)

        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

        plt.title(f"tsne_{name}_epoch_{epoch}_mode_{mode}")
        plt.savefig(f'./out/tsne/{name}/{mode}/tsne_{name}_epoch_{epoch}_mode_{mode}.png')
        plt.show(block=True)

    summary(model, (1, 28, 28))

    return features, images


"""

IMAGE_SIZE = 784
IMAGE_WIDTH = IMAGE_HEIGHT = 28


class AutoEncoder(nn.Module):

    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out


def run_ae():
    # Hyperparameters
    code_size = 20
    num_epochs = 5
    batch_size = 128
    lr = 0.002
    optimizer_cls = optim.Adam

    # Load data

    train_data = datasets.ImageFolder(
        './data/training-data-standardised',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )
    test_data = test_set = datasets.ImageFolder(
        './data/test-data-standardised',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4,
                                               drop_last=True)

    # Instantiate models
    autoencoder = AutoEncoder(code_size)
    loss_fn = nn.BCELoss()
    optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)


    # Training loop
    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)

        for i, (images, _) in enumerate(train_loader):  # Ignore image labels
            out, code = autoencoder(Variable(images))

            optimizer.zero_grad()
            loss = loss_fn(out, images)
            loss.backward()
            optimizer.step()

        print("Loss = %.3f" % loss.data)

    # Try reconstructing on test data
    test_image = random.choice(test_data)

    test_reconst, _ = autoencoder(test_image)

    torchvision.utils.save_image(test_image.data, 'orig.png')
    torchvision.utils.save_image(test_reconst.data, 'reconst.png')
"""
