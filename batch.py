from glob import glob

import torch
from PIL import Image, ImageOps, ImageFile
import numpy as np
from tqdm import tqdm

from endaaman import Commander

class CMD(Commander):
    def run_mean_std(self):
        pp = glob('data/*/*.jpg')
        mm = []
        ss = []
        scale = 0
        for p in tqdm(pp):
            i = np.array(Image.open(p)).reshape(-1, 3)
            size = i.shape[0]
            mm.append(np.mean(i, axis=0) * size / 255)
            ss.append(np.std(i, axis=0) * size / 255)
            scale += size

        mean = (np.sum(mm, axis=0) / scale).tolist()
        std = (np.sum(ss, axis=0) / scale).tolist()
        print('mean', mean)
        print('std', std)


if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
