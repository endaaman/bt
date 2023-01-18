import itertools

import cv2
import torch
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType

from endaaman.torch import pil_to_tensor


def overlay_heatmap(mask: torch.Tensor, img: torch.Tensor, alpha=1.0, threshold=0.2, cmap=cv2.COLORMAP_JET):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, H, W) and each element has value in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
    """
    mask = mask.squeeze().cpu()
    img = img.squeeze().cpu()
    mask_mask = (mask > threshold) * alpha
    heatmap = (255 * mask).type(torch.uint8).numpy()
    heatmap = cv2.applyColorMap(heatmap, cmap)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    # BGR -> RGB
    heatmap = heatmap[[2, 1, 0], :, :]

    masked = img*(-mask_mask+1) + heatmap*mask_mask
    # masked = masked.clamp(max=1.0)
    return heatmap, masked


def concat_grid_images_float(images, n_cols=1):
    poss = []
    y = 0
    x = 0
    max_y = 0
    max_x = 0
    for idx, image in enumerate(images):
        poss.append((x, y))
        x += image.width
        max_x = max(max_x, x)
        max_y = max(max_y, y+image.height)
        if (idx + 1) % n_cols == 0:
            x = 0
            y = max_y

    bg = Image.new('RGB', (max_x, max_y))
    for i, pos in zip(images, poss):
        bg.paste(i, (pos[0], pos[1], pos[0]+i.width, pos[1]+i.height))
    return bg



def select_side(W, w, idx=None):
    count = int(np.ceil(W / w))
    if count <= 1:
        return 0
    overwrap = (w * count - W) / (count - 1)
    if idx is None:
        idx = np.random.randint(0, count)
    return int(idx * w - overwrap * idx)

def grid_split_with_overwrap(img, size, flattern=False):
    hor_count = int(np.ceil(img.width / size))
    ver_count = int(np.ceil(img.height / size))
    iii = []
    for v_idx in range(ver_count):
        ii = []
        for h_idx in range(hor_count):
            x = select_side(img.width, size, h_idx)
            y = select_side(img.height, size, v_idx)
            ii.append(img.crop((x, y, x+size, y+size)))
        iii.append(ii)

    if flattern:
        iii = list(itertools.chain.from_iterable(iii))
    return iii

def n_split(x, n):
    return [(x + i) // n for i in range(n)]

def grid_split_by_size(img, size, flattern=False):
    hor_sizes = n_split(img.width, max(img.width//size, 1))
    ver_sizes = n_split(img.height, max(img.height//size, 1))
    iii = []
    y = 0

    for ver_size in ver_sizes:
        ii = []
        x = 0
        for hor_size in hor_sizes:
            ii.append(img.crop((x, y, x+hor_size, y+ver_size)))
            x += hor_size
        y += ver_size
        iii.append(ii)

    if flattern:
        iii = list(itertools.chain.from_iterable(iii))
    return iii


def grid_split(img, size, overwrap=True, flattern=False):
    if overwrap:
        return grid_split_with_overwrap(img, size, flattern)
    return grid_split_by_size(img, size, flattern)


def test_grid():
    img = Image.open('/home/ken/Dropbox/Pictures/osu.png')
    albu = A.Compose([
        GridRandomCrop(width=800, height=800)
    ])

    for i in range(20):
        a = albu(image=np.array(img))['image']
        Image.fromarray(a).save(f'tmp/grid/{i}.png')


def test_grid2():
    from torchvision.utils import make_grid
    from endaaman.torch import pil_to_tensor, tensor_to_pil
    i = Image.open('/home/ken/Dropbox/Pictures/piece.jpg')
    ii = grid_split(i, 500, overwrap=False, flattern=True)
    tt = [pil_to_tensor(i) for i in ii]
    tensor_to_pil(make_grid(tt, nrow=2, padding=0)).save('grid.png')


if __name__ == '__main__':
    i = Image.open('/home/ken/Dropbox/Pictures/piece.jpg')
    ii = grid_split(i, 2000, overwrap=False, flattern=True)
    # ii = [
    #     Image.open('/home/ken/Dropbox/Pictures/piece.jpg'),
    #     Image.open('/home/ken/Dropbox/Pictures/osu.png'),
    #     Image.open('/home/ken/Dropbox/Pictures/enda_chan.png'),
    #     Image.open('/home/ken/Dropbox/Pictures/dl.png'),
    # ]

    # print(ii)
    # concat_grid_images(ii, n_cols=2).save('g.png')
