# TensorIR: Tensor Program Abstraction Case Study Action.ipynb

import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)

# a @ b is equal to np.matmul(a, b) 矩乘
c_mm_relu = np.maximum(a_np @ b_np, 0)

print(a_np)
print(b_np)
print(c_mm_relu)

# 接下来尝试如果自己实现矩乘+relu，想实现方式

def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k==0:
                    Y[i, j]=0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)
    
c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
print(c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)

# 用tvm.script 用TensorIR 来实现矩乘+relu

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol":  "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                # vi = T.axis.spatial(128, i)
                # vj = T.axis.spatial(128, j)
                # vk = T.axis.reduce(128, k)
                # 这里还有个语法糖可以使用
                # 等价于
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

# 等价于以下实现
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias":True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):                      
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


# transformation 变化循环结构顺序等也就是具体实现中的具体循环结构

def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C:np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0*4 + j1
                    if k==0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol = 1e-5)



# 导入Ipython，使得tvm.script打印出来的是语法高亮的
# import IPython

# def code2html(code):
#     """helper function to use pygments to turn the code string into highlighted html"""
#     import pygments
#     from pygments.lexers import Python3Lexer
#     from pygments.formatters import HtmlFormatter
#     formatter = HtmlFormatter()
#     html = pygments.highlight(code, Python3Lexer(), formatter=formatter)
#     return "<style>%s<style>%s\n" % (formatter.get_style_defs(".highlight"), html)

# IPython.display.HTML(code2html(MyModule.script()))

# ##############################################


print(type(MyModule))
print(type(MyModule["mm_relu"]))
print(type(MyModuleWithAxisRemapSugar))
print(type(MyModuleWithAxisRemapSugar["mm_relu"]))

# 一个IRModule中可以包含多个张量函数
@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A: T.Buffer[(128, 128), "float32"],
           B: T.Buffer[(128, 128), "float32"],
           Y: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol" : "mm", "tir.noalias":True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
    
    @T.prim_func
    def relu(A: T.Buffer[(128, 128), "float32"],
             B: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol" : "relu", "tir.noalias":True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))

print(MyModuleWithTwoFunctions.script())
print(type(MyModuleWithTwoFunctions))
print(type(MyModuleWithTwoFunctions["mm"]))
print(type(MyModuleWithTwoFunctions["relu"]))



# 上面改变循环结构和顺序是通过手动实现的，那么如何实现自动做这些事儿呢？
print(MyModule.script())

# 继续自动变换，就需要用schedule
# 1.指定sch对应的是哪个mod
sch = tvm.tir.Schedule(MyModule)
# 2. 获取指定的sch中的指定函数的block
block_Y = sch.get_block("Y", func_name = "mm_relu")
# 3. 然后可以拿到对应的mod指定的sch中的特定函数的block的外层循环
i, j, k = sch.get_loops(block_Y)
# 4.将j分裂为j0，j1， 用split函数
j0, j1 =sch.split(j, factors =[None, 4])

print(sch.mod.script())

# 5. 轻微改动顺序
sch.reorder(j0, k, j1)
print(sch.mod.script())

# Getting to Another Variant

block_C = sch.get_block("C", "mm_relu")
# 使用一个名为 reverse_computer_at 的原语将块 C 移动到 Y 的内部循环中
sch.reverse_compute_at(block_C, j0)
print(sch.mod.script())

# 到目前为止，我们已经将简化初始化和更新步骤放在一个块主体中。
# 这种组合形式为循环转换带来了方便
# (因为初始化和更新的外部循环 i、 j 通常需要彼此保持同步)。

# 循环转换之后，我们可以将 Y 元素的初始化与简化更新分离开来。
# 我们可以通过decompose_reduction原语来做到这一点。
# (注意: 在以后的编译中，tvm 也会隐式地执行这个操作，因此这一步主要是显式地执行，并查看最终效果)。

sch.decompose_reduction(block_Y, k)
print(sch.mod.script())

# 构建和运行
rt_lib = tvm.build(MyModule, target="llvm")

# 保存输入和输出
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
print(type(c_nd))

# 从runtime Module中获得可以运行的函数
func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# 构建转换之后的程序
rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

# 比较两者的时间差异
# Time _ evaluator 是一个帮助器基准测试函数，
# 可用于比较不同生成函数的运行性能。

f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)
 
# exercise 练习
def transform(mod, jfactor):
    sch = tvm.tir.Schedule(mod)
    block_Y = sch.get_block("Y", func_name = "mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors = [None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

mod_tranformed = transform(MyModule, jfactor=8)

rt_lib_transformed = tvm.build(mod_tranformed, "llvm")
f_timer_transformed = rt_lib_transformed.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed mod_transformed %g sec " %f_timer_transformed(a_nd, b_nd, c_nd).mean)


# 与TensorIR 创建和交互的方法 
# 1.用TVM.script 创建TensorIR 程序
# 2.用TE 张量表达式 创建TensorIR 程序

from tvm import te

A = te.placeholder( (128, 128), "float32", name = "A")
B = te.placeholder( (128, 128), "float32", name = "B")
k = te.reduce_axis( (0, 128), "k")
Y = te.compute( (128, 128), lambda i, j: te.sum( A[i, k] * B[k, j], axis=k), name = "Y")
C = te.compute( (128, 128), lambda i, j: te.max(Y[i, j], 0), name = "C")

# 上面的过程就是已经对计算的输入、输出和中间计算的步骤都进行了描述
# 接下来就是希望创建一个具有两个输入AB一个输出C的函数

te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol" : "mm_relu"})
MyModuleFromTe = tvm.IRModule({"mm_relu" : te_func})
print(MyModuleFromTe.script())
