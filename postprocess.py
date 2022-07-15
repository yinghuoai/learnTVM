#!python ./postprocess.py
import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# 打开输出文件，也就是predictions.npz并读取输出张量，获取排名前5的预测输出，并显示出各自的名称和可能性大小
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

            