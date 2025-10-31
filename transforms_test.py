# sphinx_gallery_thumbnail_path = "../../gallery/assets/transforms_thumbnail.png"

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


orig_img = Image.open(Path('C:/Users/AncilunKiang/Desktop/DesktopFile/矿物数据集/36/新/811分割/train') / 'Albite/1427_1.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(3407)


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


if __name__ == '__main__':
    data_transform_train = T.Compose([T.RandomResizedCrop(224),  # 随机裁剪
                                      T.RandAugment(),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])

    augmenter = T.RandAugment()
    imgs = [
        [data_transform_train(orig_img).permute(1, 2, 0) for _ in range(4)]
        for i in range(4)
    ]
    plot(imgs)
