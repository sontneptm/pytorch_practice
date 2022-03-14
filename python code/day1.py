import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms

class BaseTransform():
    def __init__(self, resize, mean, std) -> None:
        self.base_transform=transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.base_transform(img)


image_path = 'exam_source/1_image_classification/data/goldenretriever-3724972_640.jpg'
img = Image.open(image_path)

plt.imshow(img)
plt.show()

# use_pretrained = True
# net = models.vgg16(pretrained=use_pretrained)
# net.eval()

# print(net)