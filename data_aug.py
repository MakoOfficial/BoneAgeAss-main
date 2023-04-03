import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib

d2l.set_figsize()
img = d2l.Image.open('../data/archive/small-training-dataset/10245.png')
d2l.plt.imshow(img)
