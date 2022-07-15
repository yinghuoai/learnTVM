#这里主要是来演示一些开发人员如何使用Pass Infra来进行某种优化，并为Relay程序创建优化管道。这里的方法同样适用于tir。首先导入一些必要的包。

import numpy as np
import tvm
from tvm import  te
import tvm.relay as relay

#接下来，展示了一个简单的Relay程序，该程序将用于执行各种实例Pass的例子。同样，用户也可以编写一个tir原始函数并应用Pass。

def example():
    shape = (1,64,54,54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64,64,3,3))
    x = relay.var("x", relay.TensorType((1,64,56,56),"float32"))
    conv = relay.nn.conv2d(x,weight)
    y = relay.add(c,c)
    y = relay.multiply(y,relay.const(2, "float32"))
    y = relay.add(conv,y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    # 通过relay.Function()前面的是输入，后面的是输出，上面的代码是过程
    return relay.Function([x, weight], z2)

# 然后这里给一个conv op注册一个输出数据排布更改的Pass，
# 这个Pass将卷积层的NCHW数据排布变化成NCHW16c的数据排布

@relay.op.register_alter_op_layout("nn.conv2d", level=101)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs["data_layout"] = "NCHW16c"
    return  relay.nn.conv2d(data, weight, **new_attrs)

# 在应用Pass之前看一下Relay程序是什么样子：

# func = example()
# print(func)

# 现在要优化程序。Relay具有许多优化功能。
# 我们将选择其中的一些应用到这个示例程序中。
# 手动应用优化Passes，这里使用一个常量折叠的Pass。

# 首先创建一个能包含一个或多个Relay的Relay Module

f = example()
# 通过函数直接创建一个Relay Module
mod = tvm.IRModule.from_expr(f)

# 现在我们可以在这个Module中应用常量折叠
#fold_const这里是一个回调函数，它不接受任何参数。

fold_const = relay.transform.FoldConstant()

# 然后，我们可以在给定模块上调用 pass。
# 请注意，常量折叠传递在函数级别工作。function-level
# 也就是说，模块中的每个函数都将应用pass优化。
# 用户无需手动迭代各个函数即可应用此pass。

mod = fold_const(mod)

# 可以从更新的程序中看到常量已经被折叠了

print(mod)

# 接下来可以以类似的方式应用更多优化
# 例如，可以消除z和z1使用的公共表达式
# 即使用EliminateCommonSubexpr Pass

mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)


# 还有一些优化，比如fuse，带有一些配置参数
# 例如，opt_level 0 将不允许运算融合在一起。
# 用户可以通过fuse_opt_level 来启用它

mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

print(mod)


# 使用Sequential应用一系列Pass
# 像上面这样应用pass实际上很麻烦，可能需要用户更好地理解他们之间的依赖关系
# 例如，目前fusion在let bindings上效果不佳。
# 因此，如果在融合之前应用 relay.transform.ToANormalForm() ，我们将无法融合可融合的运算符，
# 因为此Pass为每个表达式生成 let bindings以规范 Relay 程序。

# 因此，Relay 提供了 tvm.transform.Sequential，
# 通过指定每个Pass所需的passes并将它们打包为一个整体来执行，
# 从而减轻开发人员明确处理这些问题的负担

# Pass Infra中的 tvm.transform.Sequential 用于优化Pass。


# 通过py:class:`tvm.transform.Sequential`来执行一些passes

f = example()
mod = tvm.IRModule.from_expr(f)

# 将想要用到的passes打包

seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)

mod1 = seq(mod)
print(mod1)

# 转换后的Relay程序中，可以看到仍然有两个相同的加法操作。
# 这说明根本没有执行消除公共子表达式这个Pass
# 原因是优化级别小于或者等于2的pass才会在Sequential下默认执行
# 但是，Pass Infra提供了一个配置接口，供用户自定义他们想要执行的优化级别

with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)
print(mod2)

# 更改优化级别之后
# 可以看到消除公共子表达式的这个Pass执行了

# 除此之外，还可以使用disabled_pass配置来禁用某些Pass
# 例如：禁用消除公共子表达式的Pass

with tvm.transform.PassContext(opt_level=3, disabled_pass=["EliminateCommonSubexpr"]):
    mod3 = seq(mod)
print(mod3)

# 可以看到设置生效了，再次显示两个同样的加法操作

# 至此，应用的Pass是与目标设备无关的。
# Pass Infra 还提供了一些硬件感知Pass
# 比如 layout alteration pass

with tvm.transform.PassContext(opt_level=3):
    mod4 = seq(mod)
print(mod4)

seq1 = tvm.transform.Sequential([relay.transform.AlterOpLayout()])

with tvm.transform.PassContext(opt_level=3):
    with tvm.target.Target("llvm"):
        mod5 = seq1(mod)
print(mod5)

# 可以看到更改之后的效果

# 接下来，使用Python装饰器通过pass infra编排定制的优化pass。
# 此功能极大的简化了实现passes的难度。
# 例如，用户可以简单的定义一个装饰器类来实现函数级别的优化
# 如下面，transform_function包装了一个类，将所有的常量乘以c
# 然后，当调用这个自定义Pass之后
# 给定Module中的每一个函数都会被访问，并且函数中的每个常量都会被替换


@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """将一个参数替换为另一个参数的简单测试函数"""

    def __init__(self, multiplier):
        self.multiplier = multiplier


    # 这个函数可以定义一个Pass
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)


f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert  custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print(mod3)

# 调试一个Pass
# TVM为用户提供即插即用的调试pass，
# 通过特殊的pass（PrintIR）完成特定pass后打印IR
# 转储整个模块的IR

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

# 通过在 ``FoldConstant`` 之后插入 ``PrintIR`` 通道，
# 当 ``FoldConstant`` 完成时，通道基础将转储模块 IR。
# 用户可以在任何想要调试的 pass 之后插入这个 pass 来查看优化效果。
# 构建配置对象还公开了一种更灵活的调试机制。
# 可以传递一个跟踪函数，
# 该函数可用于在每次传递之前和之后执行任意代码。
# 跟踪函数将接收一个 :py::class:`tvm.IRModule`、一个 :py:class:`tvm.transform.PassInfo` 对象和一个布尔值，
# 一个布尔值指示您是在传递之前还是之后执行。
# 下面是一个例子。

@tvm.instrument.pass_instrument
class PrintIR:
    """在Pass执行之前打印pass的名称，也就是IR"""

    def run_before_pass(self, mod, info):
        print("Running pass:{}", info)
        print(mod)

with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        # 执行优化
        mod = seq(mod)
print(mod)
print("done")

# TVM Relay 表示的计算图是一个DAG，
# 但在算符融合的Pass中构建的其实是后支配树，
# 这里不是通过拓扑序来处理，而是通过DFS序来进行处理


# 这里看看能不能直接可视化Relay IR


import relay_viz
from mxnet.gluon.model_zoo import vision
import vta

env = vta.get_env()
#device = "vta"
device = "arm_cpu"
target = env.target if device == "vta" else env.target_vta_cpu

# Dictionary lookup for when to start/end bit packing
pack_dict = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
}


# 要编译的 Gluon 模型的名称
# ``start_pack`` 和 ``stop_pack``
# 标签指示从哪里开始和结束图形打包中继传递：
# 换句话说，从哪里开始和完成卸载到 VTA。

# model = "resnet18_v1"
# assert model in pack_dict
#
#
# dtype_dict = {"data": "float32"}
# shape_dict = {"data": (env.BATCH, 3, 224, 224)}
#
#
#
#
# # Get off the shelf gluon model, and convert to relay
# gluon_model = vision.get_model(model, pretrained=True)
#
#
#
# # Start front end compilation
# mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)


vizer = relay_viz.RelayVisualizer(mod)
vizer.render("output.html")


# TVM的算符融合
# 操作符融合的想法是来源于单个Kernel函数会节省将中间结果写回全局内存的时间消耗。




