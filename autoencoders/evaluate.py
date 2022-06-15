import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn import cluster, preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import dataset, Subset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm

import seaborn as sns
from umap import plot
from umap.umap_ import UMAP

import util.report
import util.utils
from util import c_transforms
from util.c_dataset import PapyriDataset


def draw_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = UMAP(
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
    plt.close()


def get_num_label_from_name(label):
    switcher = {
        "alpha": 0,
        "beta": 1,
        "chi": 2,
        "delta": 3,
        "epsilon": 4,
        "eta": 5,
        "gamma": 6,
        "iota": 7,
        "kappa": 8,
        "lambda": 9,
        "mu": 10,
        "nu": 11,
        "omega": 12,
        "omicron": 13,
        "phi": 14,
        "pi": 15,
        "psi": 16,
        "rho": 17,
        "sigma": 18,
        "tau": 19,
        "theta": 20,
        "xi": 21,
        "ypsilon": 22,
        "zeta": 23,

    }
    return switcher.get(label, label)


def get_label_from_tensor(label):
    switcher = {
        "tensor([0], device='cuda:0')": "alpha",
        "tensor([1], device='cuda:0')": "beta",
        "tensor([2], device='cuda:0')": "chi",
        "tensor([3], device='cuda:0')": "delta",
        "tensor([4], device='cuda:0')": "epsilon",
        "tensor([5], device='cuda:0')": "eta",
        "tensor([6], device='cuda:0')": "gamma",
        "tensor([7], device='cuda:0')": "iota",
        "tensor([8], device='cuda:0')": "kappa",
        "tensor([9], device='cuda:0')": "lambda",
        "tensor([10], device='cuda:0')": "mu",
        "tensor([11], device='cuda:0')": "nu",
        "tensor([12], device='cuda:0')": "omega",
        "tensor([13], device='cuda:0')": "omicron",
        "tensor([14], device='cuda:0')": "phi",
        "tensor([15], device='cuda:0')": "pi",
        "tensor([16], device='cuda:0')": "psi",
        "tensor([17], device='cuda:0')": "rho",
        "tensor([18], device='cuda:0')": "sigma",
        "tensor([19], device='cuda:0')": "tau",
        "tensor([20], device='cuda:0')": "theta",
        "tensor([21], device='cuda:0')": "xi",
        "tensor([22], device='cuda:0')": "ypsilon",
        "tensor([23], device='cuda:0')": "zeta",

    }

    return switcher.get(label)


def evaluate(config, result):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ae_resnet18 = torch.load(config.model_path)
    ae_resnet18.eval()

    criterion = eval(f"nn.{result.loss}")

    losses_train = result.train_loss
    losses_valid = result.valid_loss
    losses_test = result.test_loss

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

    _dataset = PapyriDataset('./data/raw-cleaned-custom', transform=transforms.Compose([t, t_prime]))

    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=result.random_state)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=result.random_state)

    #######################################################################################
    # Calculate worst, best and some random loss on testset
    #######################################################################################

    test_losses = []
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=1)

    if config.tqdm:
        test_loop = tqdm(test_loader, total=len(test_loader))
    else:
        test_loop = test_loader

    for image, label_char, label_frag in test_loop:
        image = image.to(device)

        ae_resnet18.eval()
        with torch.no_grad():
            _enc, _dec = ae_resnet18(image)

        loss = criterion(_dec, image)
        test_losses.append((loss.cpu().numpy(), image.cpu().numpy(), _dec.cpu().numpy(), label_char[0],
                            label_frag[0]))

    test_losses.sort(key=lambda s: s[0])

    #############################################################################################
    # Plot 10/10/10
    #############################################################################################

    rows = 6
    cols = 6
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

    fig.suptitle(f"{cols}/{cols}/{cols} - best/median/worst - decoding")
    fig.tight_layout()
    fig.savefig(f"./{config.root}/{cols}-{cols}-{cols}-decod.png", dpi=300)
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_train)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Train Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_train.png")
    # util.report.image_to_report("net_eval/loss_train.png", "Network Training Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_valid)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Validation Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_valid.png")
    # util.report.image_to_report("net_eval/loss_valid.png", "Network Validation Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_test)
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("Test Loss")
    plt.savefig(f"./{config.root}/net_eval/loss_test.png")
    # util.report.image_to_report("net_eval/loss_test.png", "Network Test Loss")
    plt.close()

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses_train, label="train_loss")
    plt.plot(losses_valid, label="valid_loss")
    plt.plot(losses_test, label="test_loss")
    util.utils.create_folder(f"./{config.root}/net_eval")
    plt.title("All Losses - Log Scale")
    plt.legend()
    plt.yscale("log")
    plt.savefig(f"./{config.root}/net_eval/loss_all.png")
    # util.report.image_to_report("net_eval/loss_all.png", "Network All Loss")
    plt.close()

    ################################################################################
    # Individual letter clustering
    ################################################################################

    print([letter for letter in config.letters_to_eval])
    print(_testset[0][1])
    idx = [i for i in range(len(_testset)) if
           _testset[i][1] in [letter for letter in config.letters_to_eval]]
    # build the appropriate subset
    print(idx)
    test_subset = Subset(_testset, idx)
    print(test_subset)

    ind_eval_loader = torch.utils.data.DataLoader(test_subset, batch_size=1)

    if config.tqdm:
        ind_loop = tqdm(ind_eval_loader, total=len(ind_eval_loader))
    else:
        ind_loop = ind_eval_loader

    encoded_images = []
    decoded_images = []
    labels_char_list = []
    labels_frag_list = []

    for image, label_char, label_frag in ind_loop:
        image = image.to(device)

        ae_resnet18.eval()
        with torch.no_grad():
            _enc, _dec = ae_resnet18(image)

        encoded_images.append(_enc.cpu().numpy())
        decoded_images.append(_dec.cpu().numpy())
        labels_char_list.append(label_char[0])  # cpu().numpy())
        labels_frag_list.append(label_frag[0])  # cpu().numpy())

    print("enc_images", encoded_images)
    print(labels_char_list)
    print(labels_frag_list)
    print(len(encoded_images))
    print(len(labels_char_list))

    y = labels_char_list
    f = labels_frag_list
    X = np.array(encoded_images)
    Z = np.array(decoded_images)

    print(X.shape)
    print(Z.shape)

    X.reshape(-1)

    X.ravel()

    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))

    y_list = list(y)
    y_list_num = list(y)

    f_list = list(f)
    f_list_num = list(f)

    for item in range(len(y_list)):
        y_list[item] = (str(y_list[item]))
        y_list_num[item] = (y_list[item])

    for item in range(len(f_list)):
        f_list[item] = (str(f_list[item]))
        f_list_num[item] = (f_list[item])

    y = tuple(y_list)
    f = tuple(f_list)
    labels_char_list = np.array(y_list_num)
    frags = np.array(f_list_num)

    print(frags)
    print(labels_char_list)
    print(frags.shape)
    print(labels_char_list.shape)

    y_set = set(y)
    y_len = len(y_set)

    f_set = set(f)
    f_len = len(f_set)

    print(f"y_set={y_set}")
    print(f"f_set={f_set}")
    print(f"y_len={y_len}")
    print(f"f_len={f_len}")

    palette_char = sns.color_palette("bright", y_len)
    palette_frag = sns.color_palette("bright", f_len)
    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(labels_char_list)
    labels_char_list_enumerated = label_encoder.transform(labels_char_list)

    # X_embedded = fit(X,y, MACHINE_EPSILON, n_components, perplexity)

    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette_char=palette_char)

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette_char)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"tsne_final_eval")
    util.utils.create_folder(f"./{config.root}/net_eval/")
    plt.savefig(f'./{config.root}/net_eval/tsne_final_eval_mode.png')
    # util.report.image_to_report(f"{name}/{config.processing}/tsne_{name}_final_eval_mode_{config.processing}.png",f"TSNE final_eval")
    plt.close()

    umap = UMAP()
    X_embedded = umap.fit_transform(X)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette_char)
    # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

    plt.title(f"umap_final_eval_mode")
    plt.savefig(f'./{config.root}/net_eval/umap_final_eval_mode.png')
    # util.report.image_to_report(f"{name}/{config.processing}/umap_{name}_final_eval_mode_{config.processing}.png",f"UMAP final_eval")
    plt.close()

    ####################################################################################################
    # 3D UMAP
    ####################################################################################################

    mapper = UMAP(n_components=2).fit(X)

    plot.points(mapper, labels=labels_char_list, theme="fire")

    plt.title(f"umap_final_eval_{config.letters_to_eval}_char")
    plt.savefig(f'./{config.root}/net_eval/umap_scatter_{config.letters_to_eval}_char.png')
    plt.close()

    plot.points(mapper, labels=frags, theme="fire")
    plt.legend().remove()

    plt.title(f"umap_final_eval_{config.letters_to_eval}_frag")
    plt.savefig(f'./{config.root}/net_eval/umap_scatter_{config.letters_to_eval}_frag.png')
    plt.close()

    standard_embedding = UMAP(random_state=42).fit_transform(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels_char_list_enumerated, s=5, cmap='Spectral')
    plt.title(f"umap_ncluster_{config.letters_to_eval}")
    #plt.legend(labels_char_list)
    plt.savefig(f'./{config.root}/net_eval/umap_ncluster_{config.letters_to_eval}.png')
    plt.close()

    kmeans_labels = cluster.KMeans(n_clusters=len(config.letters_to_eval)).fit_predict(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=5, cmap='Spectral')
    plt.title(f"umap_kmeans_{config.letters_to_eval}")
    #plt.legend(labels_char_list)
    plt.savefig(f'./{config.root}/net_eval/umap_kmeans_{config.letters_to_eval}.png')
    plt.close()

    kmeans_labels_frags = cluster.KMeans(n_clusters=f_len).fit_predict(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels_frags, s=5, cmap='Spectral')
    plt.title(f"umap_kmeans_frag")
    # plt.legend(labels_char_list)
    plt.savefig(f'./{config.root}/net_eval/umap_kmeans_frag.png')
    plt.close()

    fit = UMAP(n_components=3)
    u = fit.fit_transform(X)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels_char_list_enumerated, s=10)
    plt.title(f'n_components = 3 - {config.letters_to_eval}')
    plt.savefig(f'./{config.root}/net_eval/umap_ncomp3_{config.letters_to_eval}.png')
    plt.close()
