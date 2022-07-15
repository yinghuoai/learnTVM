import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 绘制比较四组tConv 的分组直方图

labels = ['1x1x5x5(little conv2d)', '1x1x8x8(conv2d)', '1x1x8x8-4(split 4 little conv2d)', '1x1x8x8-4(4 thread)']
little_conv_means = [0.032381]
conv_means = [0.024514]
split4conv_means = [0.036042]
Thread_conv_means = [0.002723]

x = np.arange(1)  # the label locations
width = 0.50  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, little_conv_means, width, label='Little-Conv')
rects2 = ax.bar(x + width/2, conv_means, width, label='Conv')
rects3 = ax.bar(x + width/2+width, split4conv_means, width, label='Split4Conv')
rects4 = ax.bar(x + width/2+width+width, Thread_conv_means, width, label='Thread_Conv')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time(s)')
# ax.set_xlabel('Type')
ax.set_title('Time from compute the same size of Conv ')
ax.set_xticks(x + width)
ax.set_xticklabels(['base:1x1x8x8 Conv2d'])



ax.legend()