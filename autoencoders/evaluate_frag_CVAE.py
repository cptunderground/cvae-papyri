import math

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms
from sklearn import cluster, preprocessing
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import dataset, Subset
from torchvision.transforms import transforms
from tqdm import tqdm
from umap import plot
from umap.umap_ import UMAP

from torch import nn

import util.report
import util.utils
from autoencoders.char_CVAE import char_CVAE, one_hot
from autoencoders.frag_CVAE import frag_CVAE
from util import c_transforms
from util.base_logger import logger
from util.c_dataset import PapyriDataset

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from util.config import Config
from util.result import Result


def draw_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
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


def evaluate(config: Config, result: Result):
    util.utils.create_folder(f"./{config.root}/net_eval")
    util.utils.create_folder(f"./{config.root}/net_eval/frag_CVAE")
    util.utils.create_folder(f"./{config.root}/net_eval/frag_CVAE_eval")

    nn

    ae_resnet18 = torch.load(config.model_path)
    ae_resnet18.eval()

    frag_cvae = frag_CVAE(48)
    frag_cvae = torch.load(config.cvae_frag_path)
    frag_cvae.eval()

    all_labels_frags = [
        '61210', '61226', '61228', '60402', '59170', '60221', '60475', '60998', '60291', '60941', '60810', '60468',
        '61140', '60251', '60246', '60891', '60670', '60398', '60589', '60343', '60809', '61026', '60326', '60663',
        '60220', '60812', '60242', '60400', '60842', '60324', '61236', '60304', '60462', '60934', '61239', '60808',
        '61106', '60276', '61212', '61244', '60476', '60633', '60238', '61240', '60910', '61245', '61124', '60333',
        '61138', '60901', '60306', '60214', '61213', '61165', '61246', '60215', '60492', '60258', '60940', '60732',
        '60216', '60364', '60479', '60847', '60583', '61122', '60283', '60740', '60255', '65858', '60471', '60701',
        '61117', '60359', '61073', '60367', '60219', '60337', '60312', '60771', '61112', '60867', '60421', '60764',
        '60217', '60248', '60411', '60253', '60290', '60659', '60481', '61141', '66764', '60267', '60369', '60965']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = eval(f"nn.{result.loss}")

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

    _trainset, _testset = train_test_split(_dataset, test_size=0.2, random_state=42)
    _trainset, _validset = train_test_split(_trainset, test_size=0.25, random_state=42)

    print([letter for letter in config.chars_to_eval])
    print(_testset[0][1])
    idx = [i for i in range(len(_testset)) if
           _testset[i][1] in [letter for letter in config.chars_to_eval]]
    # build the appropriate subset
    print(idx)
    _testset = Subset(_testset, idx)
    print(_testset)

    label_encoder_frag = preprocessing.LabelEncoder()
    label_encoder_frag.fit(all_labels_frags)

    #######################################################################################
    # Calculate worst, best and some random loss on testset
    #######################################################################################

    test_losses = []
    org_enc_dec = []
    test_loader = torch.utils.data.DataLoader(_testset, batch_size=1)

    test_loop = tqdm(test_loader, total=len(test_loader))

    for image, label_char, label_frag in test_loop:
        image = image.to(device)

        label = label_encoder_frag.transform(label_frag)

        ae_resnet18.eval()
        with torch.no_grad():
            _enc, _dec = ae_resnet18(image)

            label = one_hot(label, 95).to(device)

            _enc = _enc.view(_enc.size(0), 48)

            mu, logvar = frag_cvae.encode(_enc, label)
            _enc_cvae = frag_cvae.reparameterize(mu, logvar)

            _dec_cvae = frag_cvae.decode(_enc_cvae, label)

            org_enc_dec.append(
                (image.cpu().numpy(), _enc_cvae.cpu().numpy(), _dec.cpu().numpy(), label_char, label_frag))

        loss = criterion(_dec, image)
        test_losses.append((loss.cpu().numpy(), image.cpu().numpy(), _dec.cpu().numpy(), label_char[0],
                            label_frag[0]))

    #############################################################################################
    # Euclidean Distance Pseudo Random Samples
    #############################################################################################
    samples = [0, 2, 6, 10, 11, 13, 18, 20, 22, 52, 60, 81, 101, 116, 138, 164, 192, 219, 236]
    # samples = [0]
    r = math.floor(len(_testset) / 10)
    for _num in samples:
        num = _num * 10
        base = org_enc_dec[num]
        distances = []

        for o_e_d in org_enc_dec:
            b = base[1]
            e = o_e_d[1]
            distance = np.linalg.norm(b - e)

            distances.append((distance, o_e_d))

        distances.sort(key=lambda s: s[0])

        rows = 2
        cols = 5
        fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
        axes[0, 0].imshow(np.squeeze(distances[0][1][0]), cmap="gray")
        axes[0, 0].set_title(f"{distances[0][1][3][0]}\n {distances[0][1][4][0]}")

        axes[0, 1].imshow(np.squeeze(distances[1][1][0]), cmap="gray")
        axes[0, 1].set_title(f"{distances[1][1][3][0]}\n{distances[1][1][4][0]}")

        axes[0, 2].imshow(np.squeeze(distances[2][1][0]), cmap="gray")
        axes[0, 2].set_title(f"{distances[2][1][3][0]}\n{distances[2][1][4][0]}")

        axes[0, 3].imshow(np.squeeze(distances[3][1][0]), cmap="gray")
        axes[0, 3].set_title(f"{distances[3][1][3][0]}\n{distances[3][1][4][0]}")

        axes[0, 4].imshow(np.squeeze(distances[4][1][0]), cmap="gray")
        axes[0, 4].set_title(f"{distances[4][1][3][0]}\n{distances[4][1][4][0]}")

        axes[1, 0].imshow(np.squeeze(distances[0][1][2]), cmap="gray")
        axes[1, 1].imshow(np.squeeze(distances[1][1][2]), cmap="gray")
        axes[1, 2].imshow(np.squeeze(distances[2][1][2]), cmap="gray")
        axes[1, 3].imshow(np.squeeze(distances[3][1][2]), cmap="gray")
        axes[1, 4].imshow(np.squeeze(distances[4][1][2]), cmap="gray")

        fig.suptitle(f"fragCVAE - Euclidean Distance - Test Sample Index {num}", fontsize="x-large")
        plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/frag_CVAE-euclid-sample-{_num}.png")
        plt.show()
        plt.close()

        print(distances[0][0])
        print(distances[1][0])
        print(distances[2][0])
        print(distances[3][0])

        plt.hist([distances[x][0] for x in range(len(distances))], bins=50)

        plt.title(f"fragCVAE - Euclid - Test Sample Index {num} - Histogram")
        plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/frag_CVAE-euclid-sample-{_num}-hist.png")
        plt.show()
        plt.close()

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
    fig.savefig(f"./{config.root}/net_eval/char_CVAE_eval/{cols}-{cols}-{cols}-decod.png", dpi=300)
    plt.close()

    ################################################################################
    # Individual letter clustering
    ################################################################################

    ind_eval_loader = torch.utils.data.DataLoader(_testset, batch_size=1)

    if config.tqdm:
        ind_loop = tqdm(ind_eval_loader, total=len(ind_eval_loader))
    else:
        ind_loop = ind_eval_loader

    org_enc_dec = []
    decoded_images = []
    labels_char_list = []
    labels_frag_list = []

    for image, label_char, label_frag in ind_loop:
        image = image.to(device)
        label = label_encoder_frag.transform(label_frag)
        ae_resnet18.eval()
        with torch.no_grad():
            _enc, _dec = ae_resnet18(image)

            label = one_hot(label, 95).to(device)

            _enc = _enc.view(_enc.size(0), 48)

            mu, logvar = frag_cvae.encode(_enc, label)
            _enc_cvae = frag_cvae.reparameterize(mu, logvar)

            _dec_cvae = frag_cvae.decode(_enc_cvae, label)

        # TODO change to _enc_cvae
        org_enc_dec.append(_enc_cvae.cpu().numpy())
        decoded_images.append(_dec.cpu().numpy())
        labels_char_list.append(label_char[0])  # cpu().numpy())
        labels_frag_list.append(label_frag[0])  # cpu().numpy())

    print("enc_images", org_enc_dec)
    print(labels_char_list)
    print(labels_frag_list)
    print(len(org_enc_dec))
    print(len(labels_char_list))

    y = labels_char_list
    f = labels_frag_list
    X = np.array(org_enc_dec)
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

    print(f_set)
    print(f_len)

    print(f"y_set={y_set}")
    print(f"f_set={f_set}")
    print(f"y_len={y_len}")
    print(f"f_len={f_len}")

    frag_dict = {f: 0 for f in f_set}

    for f in labels_frag_list:
        frag_dict[f] += 1

    print(frag_dict)

    plt.figure(figsize=(18, 3))
    plt.bar(frag_dict.keys(), frag_dict.values(), width=0.5, align="edge", color='g')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    frag_count = [item for item in frag_dict.items()]
    print(frag_count)
    print(f"len(frag_count)={len(frag_count)}")
    frag_count_val = 0
    for key, value in frag_count:
        frag_count_val += value

    print(f"frag_count_val={frag_count_val}")

    frag_count_max = [(key, value) for key, value in frag_dict.items() if value >= 50]
    frag_count_max_count = 0

    for key, value in frag_count_max:
        frag_count_max_count += value

    frag_count.sort(key=lambda x: x[1], reverse=True)

    keys, values = zip(*frag_count)

    plt.figure(figsize=(18, 3))
    plt.bar(keys, values, width=0.5, align="edge", color='g')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    r1 = np.mean(values)
    print("Mean: ", r1)

    r2 = np.std(values)
    print("std: ", r2)

    r3 = np.var(values)
    print("variance: ", r3)

    print(frag_count_max)
    print(f"len(frag_count_max)={len(frag_count_max)}")
    print(f"frag_count_max_count={frag_count_max_count}")
    print(frag_count)

    palette_char = sns.color_palette("bright", y_len)
    palette_frag = sns.color_palette("bright", f_len)
    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30

    label_encoder_char = preprocessing.LabelEncoder()
    label_encoder_char.fit(labels_char_list)
    labels_char_list_enumerated = label_encoder_char.transform(labels_char_list)

    label_encoder_frag = preprocessing.LabelEncoder()
    label_encoder_frag.fit(labels_frag_list)
    labels_frag_list_enumerated = label_encoder_frag.transform(labels_frag_list)

    for n_neighbor in [2, 5, 10, 15, 20, 100, 200]:
        umap = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbor)
        X_embedded = umap.fit_transform(X)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full',
                        palette=sns.color_palette("hls", y_len), s=10)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels_char_list_enumerated, s=2, cmap='tab20')
        # sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full')

        plt.title(f"fragCVAE - UMAP - n_neighbors={n_neighbor}")
        plt.tight_layout()
        plt.savefig(f'./{config.root}/net_eval/frag_CVAE_eval/fragCVAE_umap_n{n_neighbor}.png', bbox_inches='tight')
        plt.show()
        plt.close()

    ####################################################################################################
    # 3D UMAP
    ####################################################################################################

    mapper = UMAP(n_components=2).fit(X)

    plot.points(mapper, labels=labels_char_list, theme="fire")

    plt.title(f"umap_final_eval_{config.chars_to_eval}_char")
    plt.savefig(f'./{config.root}/net_eval/frag_CVAE_eval/umap_scatter_{config.chars_to_eval}_char.png')
    plt.close()

    plot.points(mapper, labels=frags, theme="fire")
    plt.legend().remove()

    plt.title(f"umap_final_eval_{config.chars_to_eval}_frag")
    plt.savefig(f'./{config.root}/net_eval/frag_CVAE_eval/umap_scatter_{config.chars_to_eval}_frag.png')
    plt.close()

    standard_embedding = UMAP(random_state=42).fit_transform(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels_char_list_enumerated, s=5, cmap='Spectral')
    plt.title(f"umap_ncluster_{config.chars_to_eval}")
    # plt.legend(labels_char_list)
    plt.savefig(f'./{config.root}/net_eval/frag_CVAE_eval/umap_ncluster_{config.chars_to_eval}.png')
    plt.close()

    ####################################################################################################
    # K MEANS - Letter Labels
    ####################################################################################################
    print(y_len)

    """
    for k in range(3, y_len):
        kmeans_labels = cluster.KMeans(n_clusters=k).fit_predict(X)
        # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=5, cmap='tab20')
        # plt.title(f"umap_kmeans_{config.letters_to_eval}")
        # plt.legend(labels_char_list)
        # plt.savefig(f'./{config.root}/net_eval/umap_kmeans_{config.letters_to_eval}.png')
        # plt.close()

        ars = adjusted_rand_score(labels_char_list_enumerated, kmeans_labels)
        amis = adjusted_mutual_info_score(labels_char_list_enumerated, kmeans_labels)

        logger.info(f"k={k} - Adjusted rand index of kmeans clustered character labels")
        logger.info(str((ars, amis)))

    ####################################################################################################
    # K MEANS - Fragment Labels
    ####################################################################################################
    for k in range(3, f_len):
        kmeans_labels_frags = cluster.KMeans(n_clusters=k).fit_predict(X)
        # plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels_frags, s=5, cmap='gist_ncar')
        # plt.title(f"umap_kmeans_frag")
        # plt.legend(labels_char_list)
        # plt.savefig(f'./{config.root}/net_eval/umap_kmeans_frag.png')
        # plt.close()

        ars = adjusted_rand_score(labels_char_list_enumerated, kmeans_labels_frags)
        amis = adjusted_mutual_info_score(labels_char_list_enumerated, kmeans_labels_frags)

        logger.info(f"k={k} - Adjusted rand index of kmeans clustered character labels")
        logger.info(str((ars, amis)))
    ####################################################################################################
    
    """
    fit = UMAP(n_components=3)
    u = fit.fit_transform(X)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=labels_char_list_enumerated, s=10)
    plt.title(f'n_components = 3 - {config.chars_to_eval}')
    plt.savefig(f'./{config.root}/net_eval/frag_CVAE_eval/umap_ncomp3_{config.chars_to_eval}.png')
    plt.close()

    ####################################################################################################
    # UMAP Enhanced Clustering - Character Labels
    ####################################################################################################

    logger.info("####################################################################################################")
    logger.info("# UMAP Enhanced Clustering - Character Labels")
    logger.info("####################################################################################################")

    standard_embedding = UMAP(random_state=42).fit_transform(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=labels_char_list_enumerated, s=5,
                cmap='gist_ncar')
    plt.title(f"umap_char_labels_gist")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/umap_char_labels_gist.png")
    plt.show()

    logger.info(f"distinct char labels: {len(y_set)}")
    kmeans_labels = cluster.KMeans(n_clusters=len(y_set)).fit_predict(X)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=5, cmap='tab20')
    plt.title(f"kmeans_char_labels_tab20")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/kmeans_char_labels_tab20.png")
    plt.show()

    ars = adjusted_rand_score(labels_char_list_enumerated, kmeans_labels)
    amis = adjusted_mutual_info_score(labels_char_list_enumerated, kmeans_labels)

    logger.info(f"Adjusted rand index of kmeans clustered character labels")
    logger.info(str((ars, amis)))

    clusterable_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(X)

    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels_char_list_enumerated, s=5,
                cmap='gist_ncar')
    plt.title(f"clusterable_embeddings_umap_char_labels")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/clusterable_embeddings_umap_char_labels.png")
    plt.show()

    hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=40).fit_predict(clusterable_embedding)
    print("hdbscan labels")
    print(hdbscan_labels)
    clustered = (hdbscan_labels >= 0)
    plt.scatter(standard_embedding[~clustered, 0], standard_embedding[~clustered, 1], color=(0.5, 0.5, 0.5), s=5,
                alpha=0.5)
    plt.title(f"hdbscan_char_labels_1")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/hdbscan_char_labels_1.png")
    plt.show()

    plt.scatter(standard_embedding[clustered, 0], standard_embedding[clustered, 1], c=hdbscan_labels[clustered], s=5,
                cmap='gist_ncar')
    plt.title(f"hdbscan_char_labels_2")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/hdbscan_char_labels_2.png")
    plt.show()

    ars = adjusted_rand_score(labels_char_list_enumerated, hdbscan_labels)
    amis = adjusted_mutual_info_score(labels_char_list_enumerated, hdbscan_labels)
    logger.info(f"adjusted rand index hdbscan:")
    logger.info(str((ars, amis)))

    ars = adjusted_rand_score(labels_char_list_enumerated[clustered], hdbscan_labels[clustered])
    amis = adjusted_mutual_info_score(labels_char_list_enumerated[clustered], hdbscan_labels[clustered])
    logger.info(f"adjusted rands index hdbscan clustered:")
    logger.info(str((ars, amis)))

    clustered_sum = np.sum(clustered) / labels_char_list_enumerated.shape[0]
    logger.info(f"clustered sum:")
    logger.info(clustered_sum)

    ####################################################################################################
    # UMAP Enhanced Clustering - Fragment Labels
    ####################################################################################################

    logger.info("####################################################################################################")
    logger.info("# UMAP Enhanced Clustering - Fragment Labels")
    logger.info("####################################################################################################")

    standard_embedding = UMAP(random_state=42).fit_transform(X)

    active_keys = keys[:5]
    passive_keys = keys[5:]

    for n_neighbor in [2, 5, 10, 15, 20, 100, 200]:
        standard_embedding = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbor).fit_transform(X)
        active_embeddings = [x for i, x in enumerate(standard_embedding) if labels_frag_list[i] in active_keys]
        active_embeddings = np.asarray(active_embeddings)
        active_labels = [x for i, x in enumerate(labels_frag_list) if labels_frag_list[i] in active_keys]
        active_labels = label_encoder_frag.transform(active_labels)

        passive_embeddings = [x for i, x in enumerate(standard_embedding) if labels_frag_list[i] in passive_keys]
        passive_embeddings = np.asarray(passive_embeddings)
        passive_labels = [x for i, x in enumerate(labels_frag_list) if labels_frag_list[i] in passive_keys]
        passive_labels = label_encoder_frag.transform(passive_labels)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.scatter(active_embeddings[:, 0], active_embeddings[:, 1], c=active_labels, s=20, cmap='gist_ncar')
        ax.scatter(passive_embeddings[:, 0], passive_embeddings[:, 1], c="gray", s=2)
        # ax.legend(l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(f"fragCVAE - UMAP - Fragment Labels - n_neighbors={n_neighbor}")
        plt.tight_layout()
        plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/umap_frag_labels_active_passive_n{n_neighbor}.png",
                    bbox_inches='tight')
        plt.show()
        plt.close()

    active_embeddings = [x for i, x in enumerate(standard_embedding) if labels_frag_list[i] in active_keys]
    active_embeddings = np.asarray(active_embeddings)
    active_labels = [x for i, x in enumerate(labels_frag_list) if labels_frag_list[i] in active_keys]
    active_labels = label_encoder_frag.transform(active_labels)

    passive_embeddings = [x for i, x in enumerate(standard_embedding) if labels_frag_list[i] in passive_keys]
    passive_embeddings = np.asarray(passive_embeddings)
    passive_labels = [x for i, x in enumerate(labels_frag_list) if labels_frag_list[i] in passive_keys]
    passive_labels = label_encoder_frag.transform(passive_labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(active_embeddings[:, 0], active_embeddings[:, 1], c=active_labels, s=20, cmap='gist_ncar')
    ax.scatter(passive_embeddings[:, 0], passive_embeddings[:, 1], c="gray", s=5)
    plt.title(f"umap_frag_labels_active_passive")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/umap_frag_labels_active_passive.png")
    plt.show()
    plt.close()

    print(f"distinct frag labels: {len(f_set)}")
    kmeans_labels = cluster.KMeans(n_clusters=len(f_set)).fit_predict(X)

    active_keys = keys[:5]
    active_keys = label_encoder_frag.transform(active_keys)
    passive_keys = keys[5:]
    passive_keys = label_encoder_frag.transform(passive_keys)

    active_embeddings = [x for i, x in enumerate(standard_embedding) if kmeans_labels[i] in active_keys]
    active_embeddings = np.asarray(active_embeddings)
    active_labels = [x for i, x in enumerate(labels_frag_list) if kmeans_labels[i] in active_keys]
    active_labels = label_encoder_frag.transform(active_labels)

    passive_embeddings = [x for i, x in enumerate(standard_embedding) if kmeans_labels[i] in passive_keys]
    passive_embeddings = np.asarray(passive_embeddings)
    passive_labels = [x for i, x in enumerate(labels_frag_list) if kmeans_labels[i] in passive_keys]
    passive_labels = label_encoder_frag.transform(passive_labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(active_embeddings[:, 0], active_embeddings[:, 1], c=active_labels, s=20, cmap='gist_ncar')
    ax.scatter(passive_embeddings[:, 0], passive_embeddings[:, 1], c="gray", s=5)
    plt.title(f"umap_kmeans_clustered_frag_labels_active_passive")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/umap_kmeans_clustered_frag_labels_active_passive.png")
    plt.show()

    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=5, cmap='gist_ncar')
    plt.title("scatter-kmeans-gist")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/scatter-kmeans-gist.png")
    plt.show()

    ars = adjusted_rand_score(labels_frag_list_enumerated, kmeans_labels)
    amis = adjusted_mutual_info_score(labels_frag_list_enumerated, kmeans_labels)
    logger.info(f"Adjusted rand index of kmeans clustered fragment labels")
    logger.info(str((ars, amis)))

    clusterable_embedding = UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=2,
        random_state=42,
    ).fit_transform(X)

    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], c=labels_frag_list_enumerated, s=5,
                cmap='gist_ncar')
    plt.title(f"umap_clusterable_embeddings_frag_labels")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/umap_clusterable_embeddings_frag_labels.png")
    plt.show()

    hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=20).fit_predict(clusterable_embedding)
    clustered = (hdbscan_labels >= 0)

    plt.scatter(standard_embedding[~clustered, 0], standard_embedding[~clustered, 1], color=(0.5, 0.5, 0.5), s=5,
                alpha=0.5)
    plt.title(f"hdbscan_frag_labels_1")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/hdbscan_frag_labels_1.png")
    plt.show()

    plt.scatter(standard_embedding[clustered, 0], standard_embedding[clustered, 1], c=hdbscan_labels[clustered], s=5,
                cmap='gist_ncar')
    plt.title(f"hdbscan_frag_labels_2")
    plt.savefig(f"./{config.root}/net_eval/frag_CVAE_eval/hdbscan_frag_labels_2.png")
    plt.show()

    ars = adjusted_rand_score(labels_frag_list_enumerated, hdbscan_labels)
    amis = adjusted_mutual_info_score(labels_frag_list_enumerated, hdbscan_labels)
    logger.info(f"adjusted rand index hdbscan fragments:")
    logger.info(str((ars, amis)))

    ars = adjusted_rand_score(labels_frag_list_enumerated[clustered], hdbscan_labels[clustered])
    amis = adjusted_mutual_info_score(labels_frag_list_enumerated[clustered], hdbscan_labels[clustered])
    logger.info(f"adjusted rand index hdbscan clustered fragments:")
    logger.info(str((ars, amis)))

    clustered_sum = np.sum(clustered) / labels_frag_list_enumerated.shape[0]
    logger.info(f"clustered sum:")
    logger.info(clustered_sum)
