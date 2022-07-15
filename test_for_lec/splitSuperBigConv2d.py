
# 观察将大的Conv2d分割并且用多线程方式调用TensorIR的方式是否有优化效果

import multiprocessing
from pydoc import tempfilepager
from graphviz import view
import numpy as np
from torch import row_stack
import tvm
from tvm.ir.module import IRModule
from tvm.runtime.module import BenchmarkResult, Module
from tvm.runtime.ndarray import cpu
from tvm.script import tir as T
import time


# Exercise 2: 2D Convolution 2D卷积
# 二维卷积，这是图像处理中常见的操作。
# 其中，A 是输入张量，
# W 是权重张量，
# b 是批量指标，
# k 是输出通道，
# i 和 j 是图像高度和宽度的指标，
# di 和 dj 是权重指标，
# q 是输入通道，
# strides 是过滤窗口的步幅。
# 在这个练习中，我们选择了一个小而简单的情况，大步 = 1，填充 = 0。

N, CI, H, W, CO, K = 1, 1, 258, 258, 2, 3
OUT_H, OUT_W = H-K+1, W-K+1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)





# 这里没有运行，但是在python 的shell中运行得出了结果
# 输入： data：1*1*8*8，四维矩阵，（默认左上到右下） 0-63
#       weight: 2*1*3*3, 四维矩阵， 两个3*3， 分别是0-8和9-17，
#       这个其实就是可以看作是两个卷积核，对应最后得到的两个输出结果
# 输出： 1*2*6*6， 两个6*6，
#       1. 474-2094， 同行相邻间隔为36，同列相邻间隔为288
#       2. 1203-6468， 同行相邻间隔为117，同列相邻间隔为936


# 尝试TensorIR实现，这是一个完整的conv的实现：1*1*258*258
@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer[ (1, 1, 258, 258), "int64"],
            B: T.Buffer[ (2, 1, 3, 3), "int64"],
            C: T.Buffer[ (1, 2, 256, 256), "int64"]):
        T.func_attr({"global_symbol":"conv", "tir.noalias":True})
        for b, k, i, j in T.grid(1, 2, 256, 256):
            with T.block("C"):
                vb = T.axis.spatial(1, b)
                vk = T.axis.spatial(2, k)
                vi = T.axis.spatial(256, i)
                vj = T.axis.spatial(256, j)
                for di, dj in T.grid(3,3):
                    
                    with T.init():
                        C[vb, vk, vi, vj] = T.int64(0)
                    C[vb, vk, vi, vj] += A[vb, 0, vi+di, vj+dj]*B[vk, 0,di, dj ]
                # C[vb, vk, vi, vj] += T.int64(1)

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
conv_begin = time.time()
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
conv_end = time.time()
# print("data_tvm:\n", data_tvm)
# print("weight_tvm:\n", weight_tvm)
print("conv_tvm:",np.shape(conv_tvm))
print(conv_tvm)
print("Time cost of MyConv %f s" % (conv_end-conv_begin))


import torch

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
# 验证
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

f_timer_conv = rt_lib.time_evaluator("conv", tvm.cpu())
# print("Time cost of MyConv %g sec" % f_timer_conv(data_tvm, weight_tvm, conv_tvm).mean)
# print("Time cost of MyConv %f ms" % (f_timer_conv(data_tvm, weight_tvm, conv_tvm).mean*1e3))



# 用多线程，16个1*1*65*65

from threading import Thread
from time import sleep, ctime
import time

# 创建 Thread 的子类
class MyThread(Thread):
    def __init__(self, threadID, name,  func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)   # 不要忘记调用Thread的初始化方法
        self.func = func
        self.args = args
        self.threadID = threadID
        self.name = name 
        

    def run(self):
        print("Starting " + self.name)
        self.func(*self.args)
        print_time(self.name)
        print("Exiting " + self.name)

exitFlag = 0
def print_time(threadName):
    
    if exitFlag:
        (Thread).exit()
    
    print("%s: %s" % (threadName, time.ctime(time.time())))
    

N, CI, H, W, CO, K = 1, 1, 258, 258, 2, 3
OUT_H, OUT_W = H-K+1, W-K+1
# r_d=0
# c_d=0

@tvm.script.ir_module
class MyThreadSplit4Conv():
    @T.prim_func
    def conv(A: T.Buffer[ (1, 1, 258, 258), "int64"],
            B: T.Buffer[ (2, 1, 3, 3), "int64"],
            C: T.Buffer[ (1, 2, 256, 256), "int64"],
            r_d: T.Buffer[(1,2), "int32"]):
        T.func_attr({"global_symbol":"conv", "tir.noalias":True})
        # tmp 
        Y = T.alloc_buffer(( 1, 1, 66, 66), dtype="int64")
        # r_d = T.var("int32")
        # c_d = T.var("int32")
        # r_d = T.alloc_buffer((1,2), dtype="int32")
        
        # r_d[0,0] = T.int32(0)
        # r_d[0,1] = T.int32(0)
        
        
        # for num in T.grid(4):
        for tt, ttt, row, col in T.grid(1, 1, 66, 66):
            with T.block("Y"):
                dd = T.axis.spatial(1, tt)
                ddd = T.axis.spatial(1, ttt)
                rr = T.axis.spatial(66, row)
                cc = T.axis.spatial(66, col)
                with T.init():
                    Y[dd, ddd, rr, cc] = T.int64(0)
                Y[dd, ddd, rr, cc] = A[dd, ddd,rr+r_d[0,0],cc+r_d[0,1]]
                    #  Y[dd, ddd, rr, cc] = A[dd, ddd,rr+T.int64(3)*(num/2),cc+T.int64(3)*(num%2)]

                    # compute Y
        
        
        # store little convd Y to C
        
        for b, k, i, j in T.grid(1, 2, 64, 64):
            with T.block("C"):
                vb = T.axis.spatial(1, b)
                vk = T.axis.spatial(2, k)
                vi = T.axis.spatial(64, i)
                vj = T.axis.spatial(64, j)
                for di, dj in T.grid(3,3):    
                    with T.init():
                        C[vb, vk, vi+r_d[0,0], vj+r_d[0,1]] = T.int64(0)
                    C[vb, vk, vi+r_d[0,0], vj+r_d[0,1]] += Y[vb, 0, vi+di, vj+dj]*B[vk, 0,di, dj ]

        
        

r_d_data = np.empty(shape=(1,2), dtype="int32")
print("r_d_data:\n", r_d_data)



threadsplit4conv_lib = tvm.build(MyThreadSplit4Conv, target="llvm")

threadsplit4conv_tvm = tvm.nd.array(np.zeros((N, CO, OUT_H, OUT_W), dtype=np.int64))
print("threadsplit4conv_tvm before:",np.shape(threadsplit4conv_tvm))
print(threadsplit4conv_tvm) 

r_d_data[0,0] = 0
r_d_data[0,1] = 0

r_d_tvm = tvm.nd.array(r_d_data)
t1 = MyThread(threadID=1, name="Thread-1",  func=threadsplit4conv_lib["conv"], args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 0
r_d_data[0,1] = 64

r_d_tvm = tvm.nd.array(r_d_data)
t2 = MyThread(threadID=2, name="Thread-2", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 0
r_d_data[0,1] = 128

r_d_tvm = tvm.nd.array(r_d_data)
t3 = MyThread(threadID=3, name="Thread-3",  func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 0
r_d_data[0,1] = 192
r_d_tvm = tvm.nd.array(r_d_data)
t4 = MyThread(threadID=4, name="Thread-4", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )

### 第二层


r_d_data[0,0] = 64
r_d_data[0,1] = 0

r_d_tvm = tvm.nd.array(r_d_data)
t5 = MyThread(threadID=5, name="Thread-5",  func=threadsplit4conv_lib["conv"], args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 64
r_d_data[0,1] = 64

r_d_tvm = tvm.nd.array(r_d_data)
t6 = MyThread(threadID=6, name="Thread-6", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 64
r_d_data[0,1] = 128

r_d_tvm = tvm.nd.array(r_d_data)
t7 = MyThread(threadID=7, name="Thread-7",  func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 64
r_d_data[0,1] = 192
r_d_tvm = tvm.nd.array(r_d_data)
t8 = MyThread(threadID=8, name="Thread-8", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )


### 第三层


r_d_data[0,0] = 128
r_d_data[0,1] = 0

r_d_tvm = tvm.nd.array(r_d_data)
t9 = MyThread(threadID=9, name="Thread-9",  func=threadsplit4conv_lib["conv"], args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 128
r_d_data[0,1] = 64

r_d_tvm = tvm.nd.array(r_d_data)
t10 = MyThread(threadID=10, name="Thread-10", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 128
r_d_data[0,1] = 128

r_d_tvm = tvm.nd.array(r_d_data)
t11 = MyThread(threadID=11, name="Thread-11",  func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 128
r_d_data[0,1] = 192
r_d_tvm = tvm.nd.array(r_d_data)
t12 = MyThread(threadID=12, name="Thread-12", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )

### 第四层


r_d_data[0,0] = 192
r_d_data[0,1] = 0

r_d_tvm = tvm.nd.array(r_d_data)
t13 = MyThread(threadID=13, name="Thread-13",  func=threadsplit4conv_lib["conv"], args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 192
r_d_data[0,1] = 64

r_d_tvm = tvm.nd.array(r_d_data)
t14 = MyThread(threadID=14, name="Thread-14", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 192
r_d_data[0,1] = 128

r_d_tvm = tvm.nd.array(r_d_data)
t15 = MyThread(threadID=15, name="Thread-15",  func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )
r_d_data[0,0] = 192
r_d_data[0,1] = 192
r_d_tvm = tvm.nd.array(r_d_data)
t16 = MyThread(threadID=16, name="Thread-16", func=threadsplit4conv_lib["conv"],args=(data_tvm, weight_tvm, threadsplit4conv_tvm, r_d_tvm) )



begin_time = time.time()
# 启动线程运行
t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()
t9.start()
t10.start()
t11.start()
t12.start()
t13.start()
t14.start()
t15.start()
t16.start()
# 等待所有线程执行完毕
t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()
t9.join()
t10.join()
t11.join()
t12.join()
t13.join()
t14.join()
t15.join()
t16.join()

end_time = time.time()
print("time cost of myThreadSplit4Conv2d is %f s" %((end_time-begin_time)))

print("threadsplit4conv_tvm:",np.shape(threadsplit4conv_tvm))
print(threadsplit4conv_tvm) 
np.testing.assert_allclose(threadsplit4conv_tvm.numpy(), conv_torch, rtol=1e-5)


# 改成在prim_func中分成4个1*1*66*66

N, CI, H, W, CO, K = 1, 1, 258, 258, 2, 3
OUT_H, OUT_W = H-K+1, W-K+1
# r_d=0
# c_d=0

@tvm.script.ir_module
class MySplit4Conv:
    @T.prim_func
    def conv(A: T.Buffer[ (1, 1, 258, 258), "int64"],
            B: T.Buffer[ (2, 1, 3, 3), "int64"],
            C: T.Buffer[ (1, 2, 256, 256), "int64"]):
        T.func_attr({"global_symbol":"conv", "tir.noalias":True})
        # tmp 
        Y = T.alloc_buffer(( 1, 1, 130, 130), dtype="int64")
        # r_d = T.var("int32")
        # c_d = T.var("int32")
        r_d = T.alloc_buffer((1,2), dtype="int32")
        
        r_d[0,0] = T.int32(0)
        r_d[0,1] = T.int32(0)
        
        
        for num in T.grid(4):
            for tt, ttt, row, col in T.grid(1, 1, 130, 130):
                with T.block("Y"):
                    dd = T.axis.spatial(1, tt)
                    ddd = T.axis.spatial(1, ttt)
                    rr = T.axis.spatial(130, row)
                    cc = T.axis.spatial(130, col)
                    with T.init():
                        Y[dd, ddd, rr, cc] = T.int64(0)
                    Y[dd, ddd, rr, cc] = A[dd, ddd,rr+r_d[0,0],cc+r_d[0,1]]
                        #  Y[dd, ddd, rr, cc] = A[dd, ddd,rr+T.int64(3)*(num/2),cc+T.int64(3)*(num%2)]

                        # compute Y
            
            
            # store little convd Y to C
            
            for b, k, i, j in T.grid(1, 2, 128, 128):
                with T.block("C"):
                    vb = T.axis.spatial(1, b)
                    vk = T.axis.spatial(2, k)
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j)
                    for di, dj in T.grid(3,3):    
                        with T.init():
                            C[vb, vk, vi+r_d[0,0], vj+r_d[0,1]] = T.int64(0)
                        C[vb, vk, vi+r_d[0,0], vj+r_d[0,1]] += Y[vb, 0, vi+di, vj+dj]*B[vk, 0,di, dj ]

            
            # reset r_d, c_d
            if (r_d[0,0]== 0 and r_d[0,1]==0):
                r_d[0,1] = 128
            elif (r_d[0,0]==0 and r_d[0,1]==128):
                r_d[0,0] = 128
                r_d[0,1] = 0
            elif (r_d[0,0]== 128 and r_d[0,1]==0):
                r_d[0,1]= 128
                
     


split4conv_lib = tvm.build(MySplit4Conv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
split4conv_tvm = tvm.nd.array(np.zeros((N, CO, OUT_H, OUT_W), dtype=np.int64))
split4conv_begin = time.time()

split4conv_lib["conv"](data_tvm, weight_tvm, split4conv_tvm)
split4conv_end = time.time()     
print("split4conv_tvm :",np.shape(split4conv_tvm))
print(split4conv_tvm) 
np.testing.assert_allclose(split4conv_tvm.numpy(), conv_torch, rtol=1e-5)

print("Time cost of MySplit4Conv %f s" % (split4conv_end-split4conv_begin))

print(Module.time_evaluator(split4conv_lib, func_name="conv", dev=cpu(0), repeat=100))

