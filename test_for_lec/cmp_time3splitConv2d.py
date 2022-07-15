import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 绘制比较三组splitConv 的分组直方图

labels = ['1x1x8x8-4', '1x1x128x128-4', '1x1x258x258-16']
conv_means = [0.024514, 0.019140, 0.019753]
Thread_conv_means = [0.002723, 0.002147, 0.021263]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, conv_means, width, label='Conv')
rects2 = ax.bar(x + width/2, Thread_conv_means, width, label='Thread_Conv')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time(s)')
ax.set_xlabel('Size')
ax.set_title('Time from diferent size of Conv and Thread_Conv')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()