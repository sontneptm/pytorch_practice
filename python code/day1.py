from statistics import mean
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

class ILSVRCPredictor():
    def __init__(self, class_index) -> None:
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxid)][1]

        return predicted_label_name

image_path = 'exam_source/1_image_classification/data/goldenretriever-3724972_640.jpg'
img = Image.open(image_path)

plt.imshow(img)
plt.show()

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = BaseTransform(resize, mean, std)
img_transformed = transform(img)

img_transformed = img_transformed.numpy().transpose((1,2,0))
img_transformed = np.clip(img_transformed,0,1)

plt.imshow(img_transformed)
plt.show()

# use_pretrained = True
# net = models.vgg16(pretrained=use_pretrained)
# net.eval()

# print(net)
