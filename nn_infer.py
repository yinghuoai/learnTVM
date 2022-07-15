import numpy as np
import mxnet as mx
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
import onnx

# 获得预先训练好的模型

model = mx.gluon.model_zoo.vision.resnet18_v2(pretrained=True)
print(len(model.features))

#print(model)

# 加载的模型在Imagenet 1K数据集上进行训练，
# 该数据集包含1000个类中约100万个自然对象图像。
# 模型分为两部分，主体部分模型。特征包含13个块，
# 输出层是一个密集层，有1000个输出。

# 下面的代码块加载Imagenet数据集中每个类的文本标签。


# 预处理数据

model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

################################################################################
# Downloading, Preprocessing, and Loading the Test Image
# ------------------------------------------------------
#
# Each model is particular when it comes to expected tensor shapes, formats and
# data types. For this reason, most models require some pre and
# post-processing, to ensure the input is valid and to interpret the output.
# TVMC has adopted NumPy's ``.npz`` format for both input and output data.
#
# As input for this tutorial, we will use the image of a cat, but you can feel
# free to substitute this image for any of your choosing.
#
# .. image:: https://s3.amazonaws.com/model-server/inputs/kitten.jpg
#    :height: 224px
#    :width: 224px
#    :align: center
#
# Download the image data, then convert it to a numpy array to use as an input to the model.

# 开始下载图片
print("开始下载图片")

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 开始对图片进行处理
print("调整图片尺寸格式")
# Resize it to 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Our input image is in HWC layout while ONNX expects CHW input, so convert the array
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

#开始用Relay编译模型
print("开始用Relay编译模型")
###############################################################################
# Compile the Model With Relay
# ----------------------------
#
# The next step is to compile the ResNet model. We begin by importing the model
# to relay using the `from_onnx` importer. We then build the model, with
# standard optimizations, into a TVM library.  Finally, we create a TVM graph
# runtime module from the library.





######################################################################
# .. admonition:: Defining the Correct Target
#
#   Specifying the correct target can have a huge impact on the performance of
#   the compiled module, as it can take advantage of hardware features
#   available on the target. For more information, please refer to
#   :ref:`Auto-tuning a convolutional network for x86 CPU <tune_relay_x86>`.
#   We recommend identifying which CPU you are running, along with optional
#   features, and set the target appropriately. For example, for some
#   processors ``target = "llvm -mcpu=skylake"``, or ``target = "llvm
#   -mcpu=skylake-avx512"`` for processors with the AVX-512 vector instruction
#   set.
#

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

relay_mod, relay_params = relay.frontend.from_onnx(onnx_model, shape_dict)


target = 'llvm'
with tvm.transform.PassContext(opt_level=3):
    graph, mod , params = relay.build(relay_mod, target=target, params=relay_params)

# 编译后的模块由三部分组成：
# graph是一个描述神经网络的json字符串，
# mod是一个库，包含用于运行推理的所有编译运算符，
# params是一个字典，将参数名映射到权重

print(type(graph), type(mod), type(params))

# 现在我们可以创建一个运行时来运行模型推理，
# 即神经网络的前向传递。
# 创建运行时需要json中的神经网络定义（即图形）
# 和包含编译运算符机器代码（即mod）的库，
# 以及可以从目标构建的设备上下文。
# 这里的设备是由llvm指定的CPU。
# 接下来，我们使用set_输入加载参数，
# 并通过输入数据来运行工作负载。
# 由于这个网络只有一个输出层，
# 我们可以通过get_output（0）得到它，
# 一个（1000）形状矩阵。最终输出是一个1000长度的NumPy向量。



ctx = tvm.context(target)
rt = tvm.contrib.graph_runtime.create(graph, mod, ctx)
rt.set_input(**params)
rt.run(data=tvm.nd.array(img_data))
scores = rt.get_output(0).asnumpy()[0]
scores.shape

