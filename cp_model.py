import onnx
from tvm.contrib.download import download_testdata
from PIL import  Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor

# 下载和加载onnx模型

model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet18-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet18-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# 下载、预处理和加载测试图像

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

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



##### 使用Relay编译模型

target = "llvm"

# The input name may vary across model types. You can use a tool
# like Netron to check input names
input_name = "data"
shape_dict = {input_name: img_data.shape}

mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)


# 这里是实际的编译过程，所以需要进里面修改
with tvm.transform.PassContext(opt_level=3):
    print("开始实际relay编译过程")
    lib = relay.build(mod, target=target, params=params)

###################

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))


# 在 TVM 运行时执行

dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()



# 收集基本性能数据

import timeit

timing_number = 10
timing_repeat = 10
unoptimized = (
    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
    * 1000
    / timing_number
)
unoptimized = {
    "mean": np.mean(unoptimized),
    "median": np.median(unoptimized),
    "std": np.std(unoptimized),
}

# 输出：在多个批次中多次重复运行计算，
# 然后收集一些关于均值、中值和标准差的基础统计数据。
print(unoptimized)


# 后处理输出
# 将 ResNet-50 v2 的输出呈现为更易于阅读的形式。

from scipy.special import softmax

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))


# # 开始调整模型

# import tvm.auto_scheduler as auto_scheduler
# from tvm.autotvm.tuner import XGBTuner
# from tvm import autotvm

# number = 10
# repeat = 1
# min_repeat_ms = 0
# timeout = 10

# # create a TVM runner

# runner = autotvm.LocalRunner(
#     number=number,
#     repeat=repeat,
#     timeout=timeout,
#     min_repeat_ms=min_repeat_ms,
#     enable_cpu_cache_flush=True,
# )

# tuning_option = {
#     "tuner": "xgb",
#     "trials": 10,
#     "early_stopping": 100,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(build_func="default"), runner=runner
#     ),
#     "tuning_records": "resnet-50-v2-autotuning.json",
# }

# # begin by extracting the tasks from the onnx model
# tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# # Tune the extracted tasks sequentially.
# for i, task in enumerate(tasks):
#     prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
#     tuner_obj = XGBTuner(task, loss_type="rank")
#     tuner_obj.tune(
#         n_trial=min(tuning_option["trials"], len(task.config_space)),
#         early_stopping=tuning_option["early_stopping"],
#         measure_option=tuning_option["measure_option"],
#         callbacks=[
#             autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
#             autotvm.callback.log_to_file(tuning_option["tuning_records"]),
#         ],
#     )

# # 使用调整数据编译优化模型
# with autotvm.apply_history_best(tuning_option["tuning_records"]):
#     with tvm.transform.PassContext(opt_level=3, config={}):
#         lib = relay.build(mod, target=target, params=params)

# dev = tvm.device(str(target), 0)
# module = graph_executor.GraphModule(lib["default"](dev))

# # 验证优化模型是否运行并产生相同的结果：

# dtype = "float32"
# module.set_input(input_name, img_data)
# module.run()
# output_shape = (1, 1000)
# tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

# scores = softmax(tvm_output)
# scores = np.squeeze(scores)
# ranks = np.argsort(scores)[::-1]
# for rank in ranks[0:5]:
#     print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

# # 比较调整和未调整的模型

# import timeit

# timing_number = 10
# timing_repeat = 10
# optimized = (
#     np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
#     * 1000
#     / timing_number
# )
# optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


# print("optimized: %s" % (optimized))
# print("unoptimized: %s" % (unoptimized))
