from pydoc import tempfilepager
from graphviz import view
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# Example: Element-wise Add 添加元素

# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
print("a:\n",a)
print("b:\n",b)
c_np = a + b
print("c_np:\n",c_np)

# low_level numpy version 这一步是将高级计算抽象ndarray转换为低级python实现
def lnumpy_add(a: np.ndarray, b:np.ndarray, c:np.ndarray):
    for i in range(4):
        for j in range(4):
            c[i, j] = a[i, j] + b[i, j]

c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
print("c_lnumpy:\n",c_lnumpy)

# 接下来是进一步将低级numpy实现转换为TensorIR
# 然后结果进行比较
# TensorIR version
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer[ (4, 4), "int64"],
            B: T.Buffer[ (4, 4), "int64"],
            C: T.Buffer[ (4, 4), "int64"]):
        T.func_attr({"global_symbol":"add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty( (4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)

# Exercise 1: Broadcast Add 广播加
# 编写一个TensorIR函数， 在广播的时候添加两个数组

# init data
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
# numpy version
print("a:\n",a)
print("b:\n",b)
c_np = a + b
print("c_np:\n",c_np)

@tvm.script.ir_module
class MyAdd2:
    @T.prim_func
    def add(A: T.Buffer[ (4, 4), "int64"],
            B: T.Buffer[ (4,), "int64"],
            C: T.Buffer[ (4, 4), "int64"]):
        T.func_attr({"global_symbol":"add", "tir.noalias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vj]

rt_lib = tvm.build(MyAdd2, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)


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

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H-K+1, W-K+1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

# torch version
# torch version
# import torch   

# data_torch = torch.Tensor(data)
# weight_torch = torch.Tensor(weight)
# conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
# conv_torch = conv_torch.numpy().astype(np.int64)
# print("conv_torch:\n", conv_torch)



# 这里没有运行，但是在python 的shell中运行得出了结果
# 输入： data：1*1*8*8，（默认左上到右下） 0-63
#      weight: 2*1*3*3, 两个3*3， 分别是0-8和9-17
# 输出： 1*2*6*6， 两个6*6，
#       1. 474-2094， 同行相邻间隔为36，同列相邻间隔为288
#       2. 1203-6468， 同行相邻间隔为117，同列相邻间隔为936

# 尝试TensorIR实现
@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(A: T.Buffer[ (1, 1, 8, 8), "int64"],
            B: T.Buffer[ (2, 1, 3, 3), "int64"],
            C: T.Buffer[ (1, 2, 6, 6), "int64"]):
        T.func_attr({"global_symbol":"conv", "tir.noalias":True})
        for b, k, i, j in T.grid(1, 2, 6, 6):
            with T.block("C"):
                vb = T.axis.spatial(1, b)
                vk = T.axis.spatial(2, k)
                vi = T.axis.spatial(6, i)
                vj = T.axis.spatial(6, j)
                for di, dj in T.grid(3,3):
                    
                    with T.init():
                        C[vb, vk, vi, vj] = T.int64(0)
                    C[vb, vk, vi, vj] += A[vb, 0, vi+di, vj+dj]*B[vk, 0,di, dj ]
                # C[vb, vk, vi, vj] += T.int64(1)

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
print("conv_tvm:\n",conv_tvm)
# np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

f_timer_conv = rt_lib.time_evaluator("conv", tvm.cpu())
print("Time cost of MyConv %g sec" % f_timer_conv(data_tvm, weight_tvm, conv_tvm).mean)

# exercise 2: How to transform TensorIR 如何转换TensorIR

# Parallel, Vectorize and Unroll 并行，矢量化和展开

@tvm.script.ir_module
class MyAdd3:
    @T.prim_func
    def add(A: T.Buffer[ (4, 4), "int64"],
            B: T.Buffer[ (4, 4), "int64"],
            C: T.Buffer[ (4, 4), "int64"]):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]
sch = tvm.tir.Schedule(MyAdd3)
print("before:", sch.mod.script())
block = sch.get_block("C", func_name = "add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors = [2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
print("after:", sch.mod.script())

# exercise 3: transform a batch matmul program 转换一批矩乘程序
# @tvm.script.ir_module
# class TargetModule:
#     @T.prim_func
#     def bmm_relu(A: T.Buffer[(16, 128, 128), "float32"],
#                  B: T.Buffer[(16, 128, 128), "float32"],
#                  C: T.Buffer[(16, 128, 128), "float32"]):
#         T.func_attr({"global_symbol": "bmm_relu", "tir_noalias":True})
#         Y = T.alloc_buffer([16, 128, 128], dtype="float32")
#         for i0 in T.parallel(16):
#             for i1, i2_0 in T.grid(128, 16):
#                 for ax0_init in T.vectorized(8):
#                     with T.block("Y_init"):
#                         n, i = T.axis.remap("SS", [i0, i1])
#                         j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
        


# import tvm.contrib.debugger
# 看看能不能将relay，lower成实际的低级代码
import tvm.relay as relay


# relay example

def example():
    x = relay.var("x", relay.TensorType((1, 3, 3, 1), "float32"))
    net = relay.nn.conv2d(x, relay.var("weight"),    
                        channels=2,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        data_layout='NHWC')
    net = relay.nn.bias_add(net, relay.var("bias"))
    net = relay.nn.relu(net)
    net = relay.add(net, relay.var("add_w", shape=[1, 3, 3, 2], dtype="float32"))
    net = relay.multiply(net, relay.var("mul_w", shape=[1, 3, 3, 2], dtype="float32"))
    net = relay.nn.softmax(net)
    return relay.Function(relay.analysis.free_vars(net), net)
f = example()
# f = relay.Function(relay.analysis.free_vars(D), D)
mod = tvm.IRModule.from_expr(f)

seq = tvm.transform.Sequential([
    relay.transform.SimplifyInference(),
    relay.transform.FoldConstant(),
    relay.transform.EliminateCommonSubexpr(),
    relay.transform.FoldScaleAxis(),
    relay.transform.FuseOps()])
with relay.build_config(opt_level=3):
    mod = seq(mod)
    
print(dir(mod))
print(mod.get_global_vars())
print(mod.source_map)
print("================开始打印mod================")
print(mod)
print("================结束打印mod================")

lib =tvm.build(mod, target="llvm")
print(lib.get_source())


