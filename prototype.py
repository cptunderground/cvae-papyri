import numpy as np
import sklearn
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import transforms

import umap.umap_
import umap.plot
import sklearn.cluster as cluster

from autoencoders.autoencoder import Network, get_label, ConvAutoEncoder
from util.base_logger import logger

import util


def draw_umap(data,labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.umap_.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()

    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels, s=100)
    plt.title(title, fontsize=18)
    plt.show()
    plt.savefig
    plt.close()

def evaluate(letters: list, root: str, eval_name: str, letter_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    util.utils.create_folder(root)
    test_set = datasets.ImageFolder(
        './data/raw-cleaned-standardised',
        # './data/__test-data-standardised',
        # './data/test-data-manual',
        # './data/test-data-manual-otsu',

        transform=transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=11082)
    idx = [i for i in range(len(test_set)) if
           test_set.imgs[i][1] in [test_set.class_to_idx[letter] for letter in letters]]
    # build the appropriate subset
    subset = Subset(test_set, idx)

    train_loader = torch.utils.data.DataLoader(subset)
    test_loader = torch.utils.data.DataLoader(subset, batch_size=len(test_set))
    logger.debug(test_loader.batch_size)
    model = Network()
    model = ConvAutoEncoder(model)
    model.load_state_dict(torch.load("./_models/models-autoencoder-cluster_50.pth"))
    model.eval()
    model.to(device)

    # pretrained_model.load_state_dict(torch.load('models/pretrained/model-run(lr=0.001, batch_size=256).ckpt', map_location=device))

    logger.info(model)

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
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(12, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, decoded_imgs], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.savefig(f'./{root}/original_decoded.png', bbox_inches='tight')
    plt.close()

    encoded_img = encoded_imgs[0]  # get the 7th image from the batch (7th image in the plot above)

    fig = plt.figure(figsize=(4, 4))
    for fm in range(encoded_img.shape[0]):
        ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
        ax.set_title(f'feature map: {fm}')
        ax.imshow(encoded_img[fm], cmap='gray')

    fig.savefig(f'./{root}/encoded_img_alpha')
    plt.close()

    encoded_img = encoded_imgs[3]  # get 1st image from the batch (here '7')

    fig = plt.figure(figsize=(4, 4))
    for fm in range(encoded_img.shape[0]):
        ax = fig.add_subplot(2, 2, fm + 1, xticks=[], yticks=[])
        ax.set_title(f'feature map: {fm}')
        ax.imshow(encoded_img[fm], cmap='gray')

    fig.savefig(f'./{root}/encoded_img_epsilon')
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

    X_embedded = TSNE().fit_transform(X)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"tsne_{eval_name}_final_eval_mode_{letter_name}")

    plt.savefig(f'./{root}/tsne_{eval_name}_final_eval_mode_{letter_name}.png')

    plt.close()
    print(encoded_imgs.shape)
    print(labels)
    print(labels.shape)

    encoded_imgs = encoded_imgs.reshape((encoded_imgs.shape[0], 196))

    X_embedded = umap.umap_.UMAP().fit_transform(encoded_imgs)

    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # plt.title(f"umap_{eval_name}_final_eval_{letter_name}")
    # plt.savefig(f'./{root}/umap_{eval_name}_final_eval_mode_{letter_name}.png')
    # plt.close()

    mapper = umap.umap_.UMAP(n_components=2).fit(encoded_imgs)

    umap.plot.points(mapper, labels=labels, theme="fire")

    plt.title(f"umap_{eval_name}_final_eval_{letter_name}")
    plt.savefig(f'./{root}/umap_{eval_name}_scatter_{letter_name}.png')
    plt.show()
    plt.close()

    standard_embedding = umap.umap_.UMAP(random_state=42).fit_transform(encoded_imgs)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels, s=0.1, cmap='Spectral');
    plt.title(f"umap_{eval_name}_ncluster_{letter_name}")
    plt.savefig(f'./{root}/umap_{eval_name}_ncluster_{letter_name}.png')
    plt.show()
    plt.close()

    kmeans_labels = cluster.KMeans(n_clusters=len(letters)).fit_predict(encoded_imgs)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral');
    plt.title(f"umap_{eval_name}_kmeans_{letter_name}")
    plt.savefig(f'./{root}/umap_{eval_name}_kmeans_{letter_name}.png')
    plt.show()
    plt.close()



    fit = umap.umap_.UMAP(n_components=3)
    u = fit.fit_transform(encoded_imgs);
    fig = plt.figure()


    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels, s=10)
    plt.title(f'n_components = 3 - {letter_name}')
    plt.savefig(f'./{root}/umap_{eval_name}_ncomp3_{letter_name}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    eval_name = "eval50"
    root = "./_out/eval/50"

    letters = ["alpha",
    "delta"

    ]
    evaluate(letters, root, eval_name, "all")
