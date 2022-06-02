import numpy as np
import umap.plot
import umap.umap_
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

import util.decorators
from util.c_transforms import CustomPad


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()


def draw_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
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


if __name__ == '__main__':

    @util.decorators.timed
    def loopi():
        for i in range(100000):
            pass

    loopi()




    orig_img = Image.open("./data/raw-cleaned/alpha/alpha_60583_[-0_5-0_5]_bt1_Iliad_14_228_25.png")
    plt.imshow(orig_img)
    plt.show()

    rgb = orig_img.convert("RGB")
    plt.imshow(rgb)
    plt.show()

    t = transforms.Compose([
        CustomPad(padding=[0, 0, 0, 0], fill=(255, 255, 255)),
        transforms.Resize([224, 224]),
        transforms.Grayscale()]
    )
    padded_imgs = t(rgb)
    plt.imshow(padded_imgs, cmap="gray", vmin=0, vmax=255)
    plt.show()

    """
    eval_name = "eval50"
    root = "./_out/eval/50"

    letters = ["alpha",
    "delta"

    ]
    evaluate(letters, root, eval_name, "all")"""
