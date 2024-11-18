import itertools
from functools import lru_cache

import cv2
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.Image import Image as ImageType

from endaaman.ml import pil_to_tensor
from endaaman import grid_split


COLORS = {
    'L': '#1f77b4',
    'M': '#ff7f0e',
    'G': '#2ca02c',
    'A': 'red',
    'O': 'blue',
    'B': '#AC64AD',
}

FG_COLORS = {
    'L': 'white',
    'M': 'white',
    'G': 'white',
    'A': 'white',
    'O': 'white',
    'B': 'white',
}

def pad16(image: ImageType, with_dimension:bool=False) -> ImageType:
    width, height = image.size
    # 16の倍数となる最小サイズを計算
    new_width = ((width + 15) // 16) * 16
    new_height = ((height + 15) // 16) * 16
    # 新しい黒い背景画像を作成
    if image.mode in ('RGBA', 'LA'):
        # アルファチャンネルがある場合は透明な黒背景
        background = Image.new(image.mode, (new_width, new_height), (0, 0, 0, 0))
    else:
        # アルファチャンネルがない場合は黒背景
        background = Image.new(image.mode, (new_width, new_height), 0)
    # 画像を中央に配置するための座標を計算
    x = (new_width - width) // 2
    y = (new_height - height) // 2
    background.paste(image, (x, y))
    if with_dimension:
        return background, (x, y, new_width, new_height)
    return background

@lru_cache
def get_font():
    return ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=36)

def draw_frame(size: [int, int], text, bg_color, fg_color='#fff'):
    font = get_font()

    image = Image.new('RGBA', (size[0], size[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        xy=((0, 0), (size[0]-1, size[1]-1)),
        outline=bg_color,
        width=1,
    )
    # text = item['pred'] + ' '+ text
    bb = draw.textbbox(xy=(0, 0), text=text, font=font, spacing=8)
    draw.rectangle(
        xy=(0, 0, bb[2]+4, bb[3]+4),
        fill=bg_color,
    )
    draw.text(
        xy=(0, 0),
        text=text,
        font=font,
        fill=fg_color,
    )
    return image


def show_fold_diag(df):
    for fold, df0 in df.groupby('fold'):
        counts = {}
        counts['all'] = len(df0)
        for diag, df1 in df0.groupby('diag'):
            counts[diag] = len(df1)
        print(fold, ' '.join(f'{k}:{v}' for k, v in counts.items()))

def calc_white_area(image, min=230, max=255):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, min, max, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 各白色領域の面積を計算
    areas = [cv2.contourArea(cnt) for cnt in contours]

    # # 面積の閾値を設定（広いと狭いを区別する閾値）
    # large_area_threshold = 1000  # 例: 広いと判断する面積の閾値
    # small_area_threshold = 500   # 例: 狭いと判断する面積の閾値
    # # 広い領域と狭い領域を区別
    # large_areas = [cnt for cnt, area in zip(contours, areas) if area >= large_area_threshold]
    # small_areas = [cnt for cnt, area in zip(contours, areas) if area < small_area_threshold]
    if len(areas) > 0:
        ratio = np.max(areas)/gray.shape[0]/gray.shape[1]
    else:
        ratio = 0
    return ratio


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
    assert n_cols > 0
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

def grid_compose_image(image_grid: list[list[Image.Image]]) -> Image.Image:
    cell_width = image_grid[0][0].width
    cell_height = image_grid[0][0].height
    rows = len(image_grid)
    cols = len(image_grid[0])

    grid_width = cell_width * cols
    grid_height = cell_height * rows
    grid_image = Image.new('RGBA', (grid_width, grid_height), (0, 0, 0, 0))

    for i in range(rows):
        for j in range(cols):
            x = j * cell_width
            y = i * cell_height
            grid_image.paste(
                image_grid[i][j],
                (x, y, x + cell_width, y + cell_height)
            )
    return grid_image


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
