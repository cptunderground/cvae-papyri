import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
from umap.umap_ import UMAP

import util._transforms as _transforms
import util.report
import util.utils
from util.base_logger import logger
from util.run import Run


def get_label(label):
    switcher = {
        "tensor(0)": "alpha",
        "tensor(1)": "beta",
        "tensor(2)": "chi",
        "tensor(3)": "delta",
        "tensor(4)": "epsilon",
        "tensor(5)": "eta",
        "tensor(6)": "gamma",
        "tensor(7)": "iota",
        "tensor(8)": "kappa",
        "tensor(9)": "lambda",
        "tensor(10)": "mu",
        "tensor(11)": "nu",
        "tensor(12)": "omega",
        "tensor(13)": "omicron",
        "tensor(14)": "phi",
        "tensor(15)": "pi",
        "tensor(16)": "psi",
        "tensor(17)": "rho",
        "tensor(18)": "sigma",
        "tensor(19)": "tau",
        "tensor(20)": "theta",
        "tensor(21)": "xi",
        "tensor(22)": "ypsilon",
        "tensor(23)": "zeta",

    }

    return switcher.get(label)


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


def train(run: Run):
    util.report.header1("Auto-Encoder")

    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = "CovAE"
    t = transforms.Compose([
        _transforms._Pad(padding=[0, 0, 0, 0], fill=(255, 255, 255)),
        transforms.Resize([28, 28]),
        transforms.Grayscale()]
    )

    _dataset = datasets.ImageFolder(
        # './data/raw-cleaned-standardised',
        './data/raw',
        # './data/test-data-manual',
        # './data/test-data-manual-otsu',

        transform=transforms.Compose([t, transforms.ToTensor()])
    )

    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=42)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=42)

    logger.info(f"len(_trainset)={len(_trainset)}")
    logger.info(f"len(_validset)={len(_validset)}")
    logger.info(f"len(_testset)={len(_testset)}")

    """
    test_idx = [i for i in range(len(test_set)) if
                test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters]]
    # build the appropriate subset
    subset_test = Subset(test_set, test_idx)

    train_idx = [i for i in range(len(train_set)) if
                 train_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters]]
    # build the appropriate subset
    subset_train = Subset(train_set, train_idx)
    """

    train_loader = torch.utils.data.DataLoader(_trainset)
    test_loader = torch.utils.data.DataLoader(_testset)
    valid_loader = torch.utils.data.DataLoader(_validset)
    logger.debug(f"testloader batchsize={test_loader.batch_size}")

    # take 5 random letters from testset

    pretrained_model = Network()
    #logger.info(pretrained_model)
    # util.report.write_to_report(pretrained_model)

    # pretrained_model.load_state_dict(torch.load('models/pretrained/model-run(lr=0.001, batch_size=256).ckpt', map_location=device))

    model = ConvAutoEncoder(pretrained_model)
    #logger.info(model)

    b = torch.randn(16, 1, 5, 5)

    input_names = ['Image']
    output_names = ['Label']

    # TODO fix
    # util.report.write_to_report(summary(model, (1, 28, 28), 2592))
    # to check if our weight transfer was successful or not
    # list(list(pretrained_model.layer2.children())[0].parameters()) == list(
    #    list(model.encoder.children())[4].parameters())
    #
    for layer_num, child in enumerate(model.encoder.children()):
        if layer_num < 8:
            for param in child.parameters():
                param.requires_grad = False

    model.to(device)

    ###TODO understand losses
    criterion = nn.MSELoss()
    ###TODO understand optimizer

    """Old optimizer
    optimizer = optim.Adam([  # parameters which need optimization
        {'params': model.encoder[8:].parameters()},
        {'params': model.decoder.parameters()}
    ], lr=0.01)"""

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    ###TODO
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1 / 3, patience=3, verbose=True)

    losses_train = []
    losses_valid = []
    losses_test = []
    optimal_model = None
    current_valid_loss = 10000

    num_epochs = run.epochs
    for epoch in range(num_epochs):
        cum_train_loss = 0
        cum_valid_loss = 0
        cum_test_loss = 0
        ###################
        # train the models #
        ###################
        if run.tqdm:
            loop_train = tqdm(train_loader, total=len(train_loader))
        else:
            loop_train = train_loader

        for batch in loop_train:
            images = batch[0].to(device)
            _, outputs = model.train()(images)
            loss_train = criterion(outputs, images)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            cum_train_loss += loss_train.item() * images.size(0)
            if run.tqdm:
                loop_train.set_description(f'Training Epoch  [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_train_loss)

        if run.tqdm:
            loop_valid = tqdm(valid_loader, total=len(valid_loader))
        else:
            loop_valid = valid_loader

        for batch in loop_valid:
            images = batch[0].to(device)
            _, outputs = model.eval()(images)
            loss_valid = criterion(outputs, images)

            cum_valid_loss += loss_valid.item() * images.size(0)
            if run.tqdm:
                loop_train.set_description(f'Validation Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_valid_loss)

        if current_valid_loss > cum_valid_loss:
            optimal_model = (model, epoch)
            current_valid_loss = cum_valid_loss

        if run.tqdm:
            loop_test = tqdm(test_loader, total=len(test_loader))
        else:
            loop_test = test_loader

        for batch in loop_test:
            images = batch[0].to(device)
            _, outputs = model.eval()(images)
            loss_test = criterion(outputs, images)

            cum_test_loss += loss_test.item() * images.size(0)
            if run.tqdm:
                loop_train.set_description(f'Test Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_test_loss)



        losses_train.append(cum_train_loss)
        losses_valid.append(cum_valid_loss)
        losses_test.append(cum_test_loss)
        logger.info(f'Epoch={epoch} done.')

        #scheduler.step(cum_train_loss)

        images, labels = next(iter(test_loader))
        # images, labels = next(iter(train_loader))
        images = images.to(device)

        # get sample outputs
        encoded_imgs, decoded_imgs = model.eval()(images)
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

        fig.savefig(f'./{run.root}/original_decoded.png', bbox_inches='tight')

        plt.close()

        encoded_img = encoded_imgs[0]  # get the 7th image from the batch (7th image in the plot above)

        fig = plt.figure(figsize=(4, 4))
        for fm in range(encoded_img.shape[0]):
            ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
            ax.set_title(f'feature map: {fm}')
            ax.imshow(encoded_img[fm], cmap='gray')

        fig.savefig(f'./{run.root}/encoded_img_alpha')
        plt.close()

    torch.save(optimal_model[0].state_dict(), f'./models/optimal-model-{optimal_model[1]}-ae-{run.name_time}.pth')
    torch.save(optimal_model[0], f'./{run.root}/optimal-model-{optimal_model[1]}-ae-{run.name_time}.pth')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-num_epochs:])
    util.utils.create_folder(f"./{run.root}/net_eval")
    plt.title("Train Loss")
    plt.savefig(f"./{run.root}/net_eval/loss_train.png")
    util.report.image_to_report("net_eval/loss_train.png", "Network Training Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_valid[-num_epochs:])
    util.utils.create_folder(f"./{run.root}/net_eval")
    plt.title("Validation Loss")
    plt.savefig(f"./{run.root}/net_eval/loss_valid.png")
    util.report.image_to_report("net_eval/loss_valid.png", "Network Validation Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_test[-num_epochs:])
    util.utils.create_folder(f"./{run.root}/net_eval")
    plt.title("Test Loss")
    plt.savefig(f"./{run.root}/net_eval/loss_test.png")
    util.report.image_to_report("net_eval/loss_test.png", "Network Test Loss")
    plt.close()


def evaluate(run):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    name = "CovAE"

    test_set = datasets.ImageFolder(
        './data/raw-cleaned-standardised',
        # './data/__test-data-standardised',
        # './data/test-data-manual',
        # './data/test-data-manual-otsu',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=11082)
    idx = [i for i in range(len(test_set)) if
           test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters]]
    # build the appropriate subset
    subset = Subset(test_set, idx)

    train_loader = torch.utils.data.DataLoader(subset)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=11082)
    logger.debug(test_loader.batch_size)
    model = Network()
    model = ConvAutoEncoder(model)
    model.load_state_dict(torch.load(run.model))
    model.eval()

    # util.report.write_to_report(pretrained_model)

    # pretrained_model.load_state_dict(torch.load('models/pretrained/model-run(lr=0.001, batch_size=256).ckpt', map_location=device))

    logger.info(model)

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

    fig.savefig(f'./{run.root}/original_decoded.png', bbox_inches='tight')
    plt.close()

    encoded_img = encoded_imgs[0]  # get the 7th image from the batch (7th image in the plot above)

    fig = plt.figure(figsize=(4, 4))
    for fm in range(encoded_img.shape[0]):
        ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
        ax.set_title(f'feature map: {fm}')
        ax.imshow(encoded_img[fm], cmap='gray')

    fig.savefig(f'./{run.root}/encoded_img_alpha')
    plt.close()

    encoded_img = encoded_imgs[3]  # get 1st image from the batch (here '7')

    fig = plt.figure(figsize=(4, 4))
    for fm in range(encoded_img.shape[0]):
        ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
        ax.set_title(f'feature map: {fm}')
        ax.imshow(encoded_img[fm], cmap='gray')

    fig.savefig(f'./{run.root}/encoded_img_epsilon')
    plt.close()

    # X, y = load_digits(return_X_y=True)

    data = []
    folder = './data/__training-data-standardised'

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

    X.reshape(-1)

    X.ravel()

    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))

    y_list = list(y)
    y_list_old = y_list
    y_old = y
    for item in range(len(y_list)):
        y_list[item] = get_label(str(y_list[item]))

    y = tuple(y_list)

    y_set = set(y)
    y_len = len(y_set)

    palette = sns.color_palette("bright", y_len)
    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30

    # X_embedded = fit(X,y, MACHINE_EPSILON, n_components, perplexity)

    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # plt.show()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)

    umap = UMAP()

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"tsne_{name}_final_eval_mode_{run.processing}")
    util.utils.create_folder(f"./{run.root}/{name}/{run.processing}")
    plt.savefig(f'./{run.root}/{name}/{run.processing}/tsne_{name}_final_eval_mode_{run.processing}.png')
    util.report.image_to_report(f"{name}/{run.processing}/tsne_{name}_final_eval_mode_{run.processing}.png",
                                f"TSNE final_eval")
    plt.close()

    umap = UMAP()
    X_embedded = umap.fit_transform(X)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"umap_{name}_final_eval_mode_{run.processing}")
    util.utils.create_folder(f"./{run.root}/{name}/{run.processing}")
    plt.savefig(f'./{run.root}/{name}/{run.processing}/umap_{name}_final_eval_mode_{run.processing}.png')
    util.report.image_to_report(f"{name}/{run.processing}/umap_{name}_final_eval_mode_{run.processing}.png",
                                f"UMAP final_eval")
    plt.close()


