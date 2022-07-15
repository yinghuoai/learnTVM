import time
import tvm
from tvm import relay

import numpy as np

from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision


# 导入TVM和Pytorch并加载ResNet18模型
######################################################################
# -------------------------------
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()


# 加载一张测试图片，并执行一些后处理

from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
# 新增Batch维度
img = np.expand_dims(img, 0)

# Relay导入TorchScript模型并编译到LLVM后端

######################################################################
# Import the graph to Relay
# 将图结构导入Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
# 将 PyTorch 图转换为中继图。输入名称可以是任意的。
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

print("===========打印mod和params")
print(mod)
print("===================================")
print(params)
print("===========开始relay.build")
######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
# 使用给定的输入规范将图形编译为 llvm 目标。
target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)


# 在目标硬件上进行推理并输出分类结果
# 通过计时函数来计算推理耗时

######################################################################
# Execute the portable graph on TVM
# 在TVM上运行
# ---------------------------------
# Now we can try deploying the compiled model on target.
# 现在可以尝试在目标上部署编译后的模型。
from tvm.contrib import graph_runtime



tvm_t0 = time.time()
for i in range(10000):
    dtype = "float32"
    m = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
tvm_t1 = time.time()

# 接下来我们在1000类的字典里面查询一下Top1概率对应的类别并输出，
# 同时也用Pytorch跑一下原始模型看看两者的结果是否一致和推理耗时情况。

#####################################################################
# Look up synset name
# 查找同义词集名称
# -------------------
# Look up prediction top 1 index in 1000 class synset.
# 在 1000 个类同义词集中查找预测前 1 个索引。
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.asnumpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
torch_t0 = time.time()
for i in range(10000):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]
torch_t1 = time.time()

tvm_time = tvm_t1 - tvm_t0
torch_time = torch_t1 - torch_t0

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
print('Relay time: ', tvm_time / 10000.0, 'seconds')
print('Torch time: ', torch_time / 10000.0, 'seconds')

