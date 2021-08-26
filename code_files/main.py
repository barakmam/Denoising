import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

path = '/datasets/Sherlock/FF++_deomposedToImages_partial/aaqaifqrwn'
path = '/datasets/coco2017/train2017'
file_name = os.listdir(path)

im = np.array(Image.open(path + '/' + file_name[40]))
plt.imshow(im)
plt.show()