# 课程来自 https://tvm.d2l.ai/chapter_getting_started/vector_add.html 向量加法
# dive into dl compiler
import numpy as np
import  tvm
from tvm import te # te stands for tensor expression

# save to the d2ltvm package
np.random.seed(0)
n = 100
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b

# Save to the d2ltvm package.
def get_abc(shape, constructor=None):
    """Return random a, b and empty c with the same shape.
    """
    np.random.seed(0)
    a = np.random.normal(size=shape).astype(np.float32)
    b = np.random.normal(size=shape).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

def vector_add(n):
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')

    return A, B, C

A, B, C = vector_add(n)
print(A.dtype,A.shape)
print(B.dtype,B.shape)
print(C.dtype,C.shape)
print(type(A),type(C))
print(type(A.op))
print(type(B.op))
print(type(C.op))

# 创建一个默认schedule，可以设置循环顺序和多线程访问等
s = te.create_schedule(C.op)

# A schedule consists of several stages. Each stage corresponds to an operation to describe how it is scheduled. We can access a particular stage by either s[C] or s[C.op].

print(type(s), type(s[C]), type(s[A]))

#稍后我们将看到如何更改执行计划，
# 以更好地利用硬件资源来提高其效率。
# 这里，让我们通过打印类似C的伪代码来查看默认执行计划。

print(tvm.lower(s, [A, B, C], simple_mode=True)
)

# produce c {
#   for (i, 0, 100) {
#     c[i] = (a[i] + b[i])
#   }
# }

# lower方法接受schedule和输入输出张量。simple_mode=True将以简单紧凑的方式打印程序。
# 请注意，该程序根据输出形状添加了适当的for循环。总的来说，它与vector_add非常相似。

# 一旦定义了计算和调度，我们就可以用tvm将它们编译成一个可执行模块
# tvm.build

mod = tvm.build(s, [A, B, C])
print(type(mod))

# 为ABC赋值并运行
# The tensor data must be tvm.ndarray.NDArray object.
# 最简单的方法是先创建NumPy ndarray对象，然后通过TVM将它们转换为TVM ndarray。nd。大堆我们可以通过asnumpy方法将它们转换回NumPy。
x = np.ones(2)
y = tvm.nd.array(x)
print(type(y),type(y.numpy()))

# 现在，让我们构造数据并将其作为TVM Ndarray返回。

a, b, c = get_abc(100, tvm.nd.array)

mod(a,b,c)
print(np.testing.assert_array_equal(a.asnumpy()+b.asnumpy(), c.asnumpy()))

# TVM有参数约束（形状和数据类型校验，可以try-catch捕获异常

# 编译后的模块可以保存到磁盘中，

# A compiled a module can be saved into disk,
mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)
# and then loaded back later.
loaded_mod = tvm.runtime.load_module(mod_fname)

a, b, c = get_abc(100, tvm.nd.array)
loaded_mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())


# Implementing an operator using TVM has three steps:
#
#     Declare the computation by specifying input and output shapes and how each output element is computed.
#
#     Create a schedule to (hopefully) fully utilize the machine resources.
#
#     Compile to the hardware target.
#
# In addition, we can save the compiled module into disk so we can load it back later.

