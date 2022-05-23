import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

import autoencoders.resnet_ae
import util._transforms as _transforms
import util.report
import util.utils
import util.decorators
from util.base_logger import logger
from util.config import Config
from util.result import Result


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


@util.decorators.timed
def train(config: Config):
    dim = 224

    dim = math.floor(dim / 4) * 4
    logger.info(f"Adjusted dim to %4=0 {dim}")
    util.report.header1("Auto-Encoder")

    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t = transforms.Compose([
        _transforms._Pad(padding=[0, 0, 0, 0], fill=(255, 255, 255)),
        transforms.Resize([64, 64]),
        transforms.Grayscale()
    ])

    t_prime = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _dataset = datasets.ImageFolder(
        './data/raw',

        transform=transforms.Compose([t, t_prime])
    )

    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=42)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=42)

    logger.info(f"len(_trainset)={len(_trainset)}")
    logger.info(f"len(_validset)={len(_validset)}")
    logger.info(f"len(_testset)={len(_testset)}")

    batch = config.batch_size

    train_loader = torch.utils.data.DataLoader(_trainset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(_validset, batch_size=batch)
    logger.info(f"testloader batchsize={test_loader.batch_size}")

    ae_resnet18 = autoencoders.resnet_ae.resnet_AE(z_dim=24, nc=1)

    #logger.info(ae_resnet18)

    ae_resnet18.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(ae_resnet18.parameters(), lr=0.0001)

    logger.info(f"optimizer:{optimizer.__module__}")
    logger.info(f"optimizer defaults:{optimizer.defaults}")
    logger.info(f"loss:{criterion}")
    logger.info(f"loss defaults:{criterion.parameters()}")


    result = Result(root=config.root, name=config.name)

    result.model = ae_resnet18.__module__
    result.epochs = config.epochs
    result.batch_size = config.batch_size

    result.optimizer = optimizer.__module__
    result.optimizer_args = optimizer.defaults
    result.loss = str(criterion)
    #result.loss = criterion.__dict__

    logger.info(f"result obj : {result}")


    ###TODO
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1 / 3, patience=3, verbose=True)

    losses_train = []
    losses_valid = []
    losses_test = []
    optimal_model = None
    current_valid_loss = 10000

    num_epochs = config.epochs
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

        for batch in loop_train:

            images = batch[0].to(device)

            ae_resnet18.train()
            _enc, _dec = ae_resnet18(images)

            loss_train = criterion(_dec, images)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            cum_train_loss += loss_train.item() / len(train_loader.dataset)

            if config.tqdm:
                loop_train.set_description(f'Training Epoch  [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_train_loss)

        tqdm._instances.clear()

        ######################
        # validate the model #
        ######################
        if config.tqdm:
            loop_valid = tqdm(valid_loader, total=len(valid_loader))
        else:
            loop_valid = valid_loader

        for batch in loop_valid:
            images = batch[0].to(device)

            ae_resnet18.eval()
            with torch.no_grad():
                _enc, _dec = ae_resnet18(images)
            loss_valid = criterion(_dec, images)

            cum_valid_loss += loss_valid.item() / len(valid_loader.dataset)
            if config.tqdm:
                loop_train.set_description(f'Validation Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_valid_loss)

        if current_valid_loss > cum_valid_loss:
            optimal_model = (ae_resnet18, epoch)
            current_valid_loss = cum_valid_loss

        tqdm._instances.clear()

        ##################
        # test the model #
        ##################
        if config.tqdm:
            loop_test = tqdm(test_loader, total=len(test_loader))
        else:
            loop_test = test_loader

        for batch in loop_test:
            images = batch[0].to(device)

            ae_resnet18.eval()
            with torch.no_grad():
                _enc, _dec = ae_resnet18(images)

            loss_test = criterion(_dec, images)

            cum_test_loss += loss_test.item() / len(test_loader.dataset)
            if config.tqdm:
                loop_train.set_description(f'Test Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_test_loss)

        losses_train.append(cum_train_loss)
        losses_valid.append(cum_valid_loss)
        losses_test.append(cum_test_loss)

        logger.info(f'Epoch={epoch} done.')

        # scheduler.step(cum_train_loss)

        images, labels = next(iter(test_loader))
        # images, labels = next(iter(train_loader))
        images = images.to(device)

        # get sample outputs
        encoded_imgs = ae_resnet18.encoder.eval()(images)
        decoded_imgs = ae_resnet18.decoder.eval()(encoded_imgs)
        # prep images for display
        images = images.cpu().numpy()

        # use detach when it's an output that requires_grad
        encoded_imgs = encoded_imgs.detach().cpu().numpy()
        decoded_imgs = decoded_imgs.detach().cpu().numpy()

        decoded_imgs = np.reshape(decoded_imgs, (
            decoded_imgs.shape[0], decoded_imgs.shape[2], decoded_imgs.shape[3], decoded_imgs.shape[1]))
        images = np.reshape(images, (images.shape[0], images.shape[2], images.shape[3], images.shape[1]))

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(12, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, decoded_imgs], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.savefig(f'./{config.root}/original_decoded.png', bbox_inches='tight')

        plt.close()

    torch.save(optimal_model[0].state_dict(), f'./models/optimal-model-{optimal_model[1]}-ae-{config.name_time}.pth')
    torch.save(optimal_model[0], f'./{config.root}/optimal-model-{optimal_model[1]}-ae-{config.name_time}.pth')

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-num_epochs:])
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Train Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_train.png")
    util.report.image_to_report("net_eval/loss_train.png", "Network Training Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_valid[-num_epochs:])
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Validation Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_valid.png")
    util.report.image_to_report("net_eval/loss_valid.png", "Network Validation Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_test[-num_epochs:])
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Test Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_test.png")
    util.report.image_to_report("net_eval/loss_test.png", "Network Test Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train[-num_epochs:], label = "train_loss")
    plt.plot(losses_valid[-num_epochs:], label = "valid_loss")
    plt.plot(losses_test[-num_epochs:], label = "test_loss")
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("All Losses - Log Scale")
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"./{config.root}/net_eval/loss_all.png")
    util.report.image_to_report("net_eval/loss_all.png", "Network All Loss")
    plt.close()



    result.train_loss = losses_train
    result.valid_loss = losses_valid
    result.test_loss = losses_test
    logger.info(result)
    result.saveJSON()
