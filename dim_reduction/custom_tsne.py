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

def custom_tsne():

    data = []
    folder = './data/test-data-standardised'

    for f in os.listdir(folder):
        print(f)
        for filename in os.listdir(f"{folder}/{f}"):
            print(filename)
            image = cv2.imread(os.path.join(folder,f,filename))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (28,28))
                image = image.flatten()
                data.append([image, f])
                #data.append([image, f"{folder}/{f}/{filename}"])

    features, images = zip(*data)
    print(features)
    print(images)

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
    tsne = TSNE(n_components=2, learning_rate=150, perplexity=30, angle=0.2, verbose=2).fit_transform(X)

    tx, ty = tsne[:, 0], tsne[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

    palette = sns.color_palette("bright", 5)
    sns.scatterplot(tsne[:, 0], tsne[:, 1], hue= images, palette=palette)
    #sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    plt.show()


