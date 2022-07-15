import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

def example():
    shape = (1,64,54,54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64,64,3,3))
    x = relay.var("x", relay.TensorType((1,64,56,56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c,c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y,c)
    z1 = relay.add(y,c)
    z2 = relay.add(z,z1)
    return relay.Function([x,weight], z2)


# 编写一个给一个conv op注册一个输出数据排布更改的Pass
# 这个Pass将卷积层的NCHW数据排布变化成NCHW16c的数据排布

@relay.op.register_alter_op_layout("nn.conv2d", level=101)
def alter_conv2d(attrs, inputs, tinfos, out_types):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs["data_layout"] = "NCHW16c"
    return relay.nn.conv2d(data, weight, **new_attrs)


#通过表达式创建IRModule
f=example()
mod = tvm.IRModule.from_expr(f)
print(mod)

fold_const = relay.transform.FoldConstant()
mod = fold_const(mod)
print(mod)

mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)
print(mod)


# 使用 Sequential 应用一系列Pass
# Now let's execute some passes through :py:class:`tvm.transform.Sequential`

mod = tvm.IRModule.from_expr(f)

seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2)
    ]
)

mod1 = seq(mod)
print(mod1)

with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)
print(mod2)

seq1 = tvm.transform.Sequential([relay.transform.AlterOpLayout()])
with tvm.transform.PassContext(opt_level=3):
    with tvm.target.Target("llvm"):
        mod3 = seq1(mod)
print(mod3)

# 使用Python装饰器实现一个Pass
# 例如，用户可以简单的定义一个装饰器类来实现函数级别的优化

@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""
    def __init__(self, multiplier):
        self.multiplier = multiplier

    # 这个函数可以定义一个Pass
    def transform_function(self,func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)

f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod5 = custom_pass(mod)
print("===========mod5==========")
print(mod5)

# 调试一个Pass

f = example()
mod = tvm.IRModule.from_expr(f)
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(),
        relay.transform.AlterOpLayout(),
    ]
)

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)


with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        # Perform the optimizations.
        mod = seq(mod)
print(mod)

print("done")
