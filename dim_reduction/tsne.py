import logging
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
from torch.utils.data import datasets
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


def fit(X,y, eps, n_components, perplexity):
    n_samples = X.shape[0]

    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(P, degrees_of_freedom, n_samples, n_components=n_components, X_embedded=X_embedded,y=y)


def _tsne(P, degrees_of_freedom, n_samples, X_embedded, n_components,y):
    params = X_embedded.ravel()

    obj_func = _kl_divergence

    params = _gradient_descent(obj_func, params,y, [P, degrees_of_freedom, n_samples, n_components])

    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    MACHINE_EPSILON = np.finfo(np.double).eps
    X_embedded = params.reshape(n_samples, n_components)

    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _gradient_descent(obj_func, p0,y, args, it=0, n_iter=100,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    P, degrees_of_freedom, n_samples, n_components = args


    plt.ion()
    for i in range(it, n_iter):

        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        print("[t-SNE] Iteration %d: error = %.7f,"
              " gradient norm = %.7f"
              % (i + 1, error, grad_norm))

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break

        if grad_norm <= min_grad_norm:
            break

        X_embedded = p.reshape(n_samples, n_components)
        x_data=X_embedded[:, 0]
        y_data=X_embedded[:, 1]

        palette = sns.color_palette("bright", 5)
        sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette, x="x", y="y")
        plt.show()
        plt.pause(0.0001)
        plt.clf()


    return p


def tsne(mode, folder):
    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", 5)

    #X, y = load_digits(return_X_y=True)

    data = []
    #folder = './data/raw-cleaned-standardised'

    for f in os.listdir(folder):
        logging.debug(f)
        for filename in os.listdir(f"{folder}/{f}"):
            logging.debug(filename)
            image = cv2.imread(os.path.join(folder, f, filename))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (28, 28))
                image = image.flatten()
                data.append([image, f])
                # data.append([image, f"{folder}/{f}/{filename}"])
    logging.debug(data)
    features, images = zip(*data)
    y = images
    X = np.array(features)
    logging.debug(features)
    logging.debug(images)
    logging.debug(X.shape)

    """
    features = np.array(features)
    pca = PCA(n_components=50)
    pca.fit(features)
    pca_features = pca.transform(features)

    num_images_to_plot = len(images)

    if len(images) > num_images_to_plot:
        sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
        images = [images[i] for i in sort_order]
        pca_features = [pca_features[i] for i in sort_order]

    X = np.array(pca_features)
    #tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)


    print(X)
    print(y)
    """

    MACHINE_EPSILON = np.finfo(np.double).eps
    n_components = 2
    perplexity = 30

    #X_embedded = fit(X,y, MACHINE_EPSILON, n_components, perplexity)

    #sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    #plt.show()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)

    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)

    plt.savefig(f'./out/tsne_{mode}.png')
    plt.show()



