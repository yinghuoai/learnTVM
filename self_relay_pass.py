# 演示如何使用pass infra来执行某种优化，
# 并且为relay程序创建优化通道, 同样的方法可以用于tir

import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

# 创建一个简单的relay
# 本文件中将针对这个relay进行各种优化

def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape=shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)


# 优化这个relay

# 比如手动应用优化pass
f = example()
mod = tvm.IRModule.from_expr(f)
fold_const = relay.transform.FoldConstant()
mod = fold_const(mod)
print(mod)

# 同样类似的，比如还可以应用消除公共子表达式的pass
mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)

# 还有算符融合pass，需要设置优化级别
mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
print(mod)

# 还可以使用Sequential打包多个pass

f = example()
mod = tvm.IRModule.from_expr(f)
# Glob the interested passes.
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)
mod1 = seq(mod)
print(mod1)

# 使用python装饰器实现一个pass
# 通过pass infra编写自定义优化pipeline
# 用户可以简单定义一个装饰类来执行函数级别优化
# 之后，当调用这个pass，将访问给定mod中的每个函数
# 并且替换掉函数中的每个常量

# 用c的倍数替换所有常量
@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    # 简单测试函数：将一个参数替换为另一个参数
    def __init__(self, multiplier):
        self.multiplier = multiplier

    # 这个函数可以定义一个pass
    def transform_function(self, func, mod, ctx):
        obj = self
        
        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)
        return ReplaceConstant().visit(func)

f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print(mod3)


# 调试pass


@tvm.instrument.pass_instrument
class PrintIR:
    # 打印pass的name,IR,只在pass执行前
    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        mod = seq(mod)
print(mod)

