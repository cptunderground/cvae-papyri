import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import dataset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

import util.report
import util.utils
from util import _transforms


def evaluate(config, result):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ae_resnet18 = torch.load(config.model_path)
    ae_resnet18.eval()

    criterion = eval(f"nn.{result.loss}")

    losses_train = result.train_loss
    losses_valid = result.valid_loss
    losses_test = result.test_loss

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

    test_loader = torch.utils.data.DataLoader(_testset, batch_size=1)

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

    ### Calculate worst, best and some random loss on testset
    test_losses = []
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=1)

    if config.tqdm:
        test_loop = tqdm(test_loader, total=len(test_loader))
    else:
        test_loop = test_loader

    for image, label in test_loop:
        image = image.to(device)
        label = label.to(device)

        ae_resnet18.eval()
        with torch.no_grad():
            _enc, _dec = ae_resnet18(image)

        loss = criterion(_dec, image)
        test_losses.append((loss.cpu().numpy(), image.cpu().numpy(), _dec.cpu().numpy(), label.cpu().numpy()))

    test_losses.sort(key=lambda s: s[0])

    """
    print("test losses", test_losses)
    print("test losses first 5", test_losses[:5])
    print("test losses last 5", test_losses[-5:])
    """

    rows = 6
    cols = 10
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)

    for i in range(0, cols):
        axes[0, i].imshow(np.squeeze(test_losses[i][1]), cmap="gray")
        # axes[0, i].set_title(test_losses[i][0])
        axes[1, i].imshow(np.squeeze(test_losses[i][2]), cmap="gray")
        # axes[1, i].set_title(test_losses[i][0])

        axes[2, i].imshow(np.squeeze(test_losses[(len(test_losses) // 2) + i][1]), cmap="gray")
        # axes[2, i].set_title(test_losses[(len(test_losses)//2)+i][0])
        axes[3, i].imshow(np.squeeze(test_losses[(len(test_losses) // 2) + i][2]), cmap="gray")
        # axes[3, i].set_title(test_losses[(len(test_losses)//2)+i][0])

        axes[4, i].imshow(np.squeeze(test_losses[-1 - i][1]), cmap="gray")
        # axes[4, i].set_title(test_losses[-1-i][0])
        axes[5, i].imshow(np.squeeze(test_losses[-1 - i][2]), cmap="gray")
        # axes[5, i].set_title(test_losses[-1-i][0])

    for i in range(0, cols):
        for j in range(0, rows):
            axes[j, i].get_xaxis().set_visible(False)
            axes[j, i].get_yaxis().set_visible(False)


    print(losses_train)
    print(losses_valid)
    print(losses_test)

    fig.suptitle("10/10/10 - best/avg/worst - decoding")
    fig.savefig(f"./{config.root}/10-10-10-decod.png")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Train Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_train.png")
    #util.report.image_to_report("net_eval/loss_train.png", "Network Training Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_valid)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Validation Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_valid.png")
    #util.report.image_to_report("net_eval/loss_valid.png", "Network Validation Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_test)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Test Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_test.png")
    #util.report.image_to_report("net_eval/loss_test.png", "Network Test Loss")
    plt.close()

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses_train, label="train_loss")
    plt.plot(losses_valid, label="valid_loss")
    plt.plot(losses_test, label="test_loss")
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("All Losses - Log Scale")
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"./{config.root}/net_eval/loss_all.png")
    #util.report.image_to_report("net_eval/loss_all.png", "Network All Loss")
    plt.close()


    """
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

    plt.title(f"tsne_{name}_final_eval_mode_{config.processing}")
    util.utils.create_folder(f"./{config.root}/{name}/{config.processing}")
    plt.savefig(f'./{config.root}/{name}/{config.processing}/tsne_{name}_final_eval_mode_{config.processing}.png')
    util.report.image_to_report(f"{name}/{config.processing}/tsne_{name}_final_eval_mode_{config.processing}.png",
                                f"TSNE final_eval")
    plt.close()

    umap = UMAP()
    X_embedded = umap.fit_transform(X)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"umap_{name}_final_eval_mode_{config.processing}")
    util.utils.create_folder(f"./{config.root}/{name}/{config.processing}")
    plt.savefig(f'./{config.root}/{name}/{config.processing}/umap_{name}_final_eval_mode_{config.processing}.png')
    util.report.image_to_report(f"{name}/{config.processing}/umap_{name}_final_eval_mode_{config.processing}.png",
                                f"UMAP final_eval")
    plt.close()
    
    """

    pass