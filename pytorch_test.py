import tvm
# import onnx
import numpy as np
import tvm.relay as relay
from PIL import Image
import time

mean = [123, 117, 104]
std = [58.395, 57.12, 57.375]


def transform_image(image):
    image = image - np.array(mean)
    image /= np.array(std)
    image = np.array(image).transpose(2, 0, 1)
    image = image[np.newaxis, :].astype('float32')
    return image


img = Image.open('../datasets/images/plane.jpg').resize((224, 224))  # 这里我们将图像resize为特定大小
x = transform_image(img)
