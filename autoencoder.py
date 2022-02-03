import random

import torch
#from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

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
            nn.Linear(in_features=7*7*32, out_features=256),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=256, out_features=10)

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
            *list(pretrained_model.layer1.children()),
            *list(pretrained_model.layer2.children()),
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


def run_ae():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_set = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )
    test_set = torchvision.datasets.MNIST(
        root='./data/',
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=1)

    pretrained_model = Network()
    #pretrained_model.load_state_dict(torch.load('models/pretrained/model-run(lr=0.001, batch_size=256).ckpt', map_location=device))
    model = ConvAutoEncoder(pretrained_model)
    model

    # to check if our weight transfer was successful or not
    list(list(pretrained_model.layer2.children())[0].parameters()) == list(list(model.encoder.children())[4].parameters())
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

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = 0
        ###################
        # train the models #
        ###################
        loop = tqdm(train_loader, total=len(train_loader))
        for batch in loop:
            images = batch[0].to(device)
            _, outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            loop.set_description(f'Epoch [{epoch + 1:2d}/{num_epochs}]')
            loop.set_postfix(loss=train_loss)

        scheduler.step(train_loss)

        torch.save(model.state_dict(), 'models/models-autoencoder.pth')

        images, labels = next(iter(test_loader))
        images = images.to(device)

        # get sample outputs
        encoded_imgs, decoded_imgs = model(images)
        # prep images for display
        images = images.cpu().numpy()

        # use detach when it's an output that requires_grad
        encoded_imgs = encoded_imgs.detach().cpu().numpy()
        decoded_imgs = decoded_imgs.detach().cpu().numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, decoded_imgs], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()
        fig.savefig('images/original_decoded.png', bbox_inches='tight')
        plt.close()

        encoded_img = encoded_imgs[6]  # get the 7th image from the batch (7th image in the plot above)

        fig = plt.figure(figsize=(4, 4))
        for fm in range(encoded_img.shape[0]):
            ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
            ax.set_title(f'feature map: {fm}')
            ax.imshow(encoded_img[fm], cmap='gray')

        plt.show()
        fig.savefig('images/encoded_img_4')
        plt.close()

        encoded_img = encoded_imgs[0]  # get 1st image from the batch (here '7')

        fig = plt.figure(figsize=(4, 4))
        for fm in range(encoded_img.shape[0]):
            ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
            ax.set_title(f'feature map: {fm}')
            ax.imshow(encoded_img[fm], cmap='gray')

        plt.show()
        fig.savefig('images/encoded_img_7')
        plt.close()
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
    train_data = datasets.MNIST('./data', train=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4,
                                               drop_last=True)

    # Instantiate models
    autoencoder = AutoEncoder(code_size)
    loss_fn = nn.BCELoss()
    optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

    test_image = random.choice(test_data)
    test_image = Variable(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]))
    print(test_image)
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
    test_image = Variable(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]))
    test_reconst, _ = autoencoder(test_image)

    torchvision.utils.save_image(test_image.data, 'orig.png')
    torchvision.utils.save_image(test_reconst.data, 'reconst.png')
"""