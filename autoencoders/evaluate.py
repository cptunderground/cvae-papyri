import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from torch.utils.data import dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from umap.umap_ import UMAP

import util.report
import util.utils
from autoencoders.convautoencoder import ConvAutoEncoder as CAE
from util.base_logger import logger


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
           test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in run.letters_to_eval]]
    # build the appropriate subset
    subset = Subset(test_set, idx)

    train_loader = torch.utils.data.DataLoader(subset)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=11082)
    logger.debug(test_loader.batch_size)

    model = CAE()
    model.load_state_dict(torch.load(run.model_path))
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
