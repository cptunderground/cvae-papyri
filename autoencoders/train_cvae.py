import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset
from torchsummary import summary
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

import autoencoders.resnet_ae
import util.c_transforms as c_transforms
import util.decorators
from util.base_logger import logger
from util.config import Config
from util.enc_dataset import EncodedDataset
from util.result import Result
from util.c_dataset import PapyriDataset


@util.decorators.timed
def train_cvae(config: Config):
    # util.report.header1("Auto-Encoder")

    logger.info(f"torch.cuda.is_available()={torch.cuda.is_available()}")

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

    _dataset = EncodedDataset(config.model_path, './data/raw-cleaned-custom',
                              transform=transforms.Compose([t, t_prime]))

    random_state = 42

    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=random_state)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=random_state)

    logger.info(f"len(_trainset)={len(_trainset)}")
    logger.info(f"len(_validset)={len(_validset)}")
    logger.info(f"len(_testset)={len(_testset)}")

    batch = config.batch_size

    train_loader = torch.utils.data.DataLoader(_trainset, batch_size=batch)
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(_validset, batch_size=batch)
    logger.info(f"testloader batchsize={test_loader.batch_size}")

    cvae_char = autoencoders.cvae.CVAE()
    cvae_frag = autoencoders.cvae.CVAE()

    cvae_char.to(device)
    cvae_frag.to(device)

    criterion = nn.MSELoss()

    optimizer_char = optim.Adam(cvae_char.parameters(), lr=0.0001)
    optimizer_frag = optim.Adam(cvae_frag.parameters(), lr=0.0001)

    scheduler_char = optim.lr_scheduler.ReduceLROnPlateau(optimizer_char, factor=1 / 3, patience=25, verbose=True)
    scheduler_frag = optim.lr_scheduler.ReduceLROnPlateau(optimizer_frag, factor=1 / 3, patience=25, verbose=True)

    ##############################################################

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

            cum_train_loss += loss_train.item()

            if config.tqdm:
                loop_train.set_description(f'Training Epoch  [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_train_loss)

        scheduler.step(cum_train_loss)
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

            cum_valid_loss += loss_valid.item()
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

            cum_test_loss += loss_test.item()
            if config.tqdm:
                loop_train.set_description(f'Test Epoch [{epoch + 1:2d}/{num_epochs}]')
                loop_train.set_postfix(loss=cum_test_loss)

        losses_train.append(cum_train_loss)
        losses_valid.append(cum_valid_loss)
        losses_test.append(cum_test_loss)
        logger.info(f'Epoch={epoch} done.')

    torch.save(optimal_model[0], f'./{config.root}/optimal-model-{optimal_model[1]}-ae-{config.name_time}.pth')
    config.model_path = f'./{config.root}/optimal-model-{optimal_model[1]}-ae-{config.name_time}.pth'
    result.model = f'./{config.root}/optimal-model-{optimal_model[1]}-ae-{config.name_time}.pth'

    # scheduler.step(cum_train_loss)

    result.train_loss = losses_train
    result.valid_loss = losses_valid
    result.test_loss = losses_test

    logger.info(result)

    return result, config
