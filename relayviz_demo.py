from pytest import param
from tvm.contrib import relay_viz
from tvm.relay.testing import resnet
from unicodedata import name
import tvm
import tvm.relay as relay
import numpy as np
import time


#mod, param = resnet.get_workload(num_layers=18)

# simple mod

# dshape = (1, 16, 64, 64)
# x = relay.var("x", shape=dshape)
# x = relay.add(x, relay.const(1, "float32"))
# y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# y1 = relay.add(relay.const(1, "float32"), y)
# y = relay.add(y, y1)
# z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=16)
# z = relay.add(z2, z3)
# func = relay.Function(relay.analysis.free_vars(z), z)

# 尝试再设计一个比如说看看连续两个con2d+bn+relu的结构

# # 构造BN
# def batch_norm(data,
#                gamma=None,
#                beta=None,
#                moving_mean=None,
#                moving_var=None,
#                **kwargs):
#     name = kwargs.get('name')
#     kwargs.pop('name')
#     if not gamma:
#         gamma = relay.var(name + '_gamma')
#     if not beta:
#         beta = relay.var(name + '_beta')
#     if not moving_mean:
#         moving_mean = relay.var(name + '_moving_mean')
#     if not moving_var:
#         moving_var = relay.var(name + '_moving_var')
#     return relay.nn.batch_norm(data,
#                                gamma=gamma,
#                                beta=beta,
#                                moving_mean=moving_mean,
#                                moving_var=moving_var,
#                                **kwargs)[0]


# # 构造卷积
# def conv2d(data, weight=None, **kwargs):
#     name = kwargs.get('name')
#     kwargs.pop('name')
#     if not weight:
#         weight = relay.var(name + '_weight')
#     return relay.nn.conv2d(data, weight, ** kwargs)


# # 构造卷积+BN+ReLU的simpleNet
# def simplenet(data, name, channels, kernel_size=(3,3), strides=(1,1),
#               padding=(1,1), epsilon =1e-5):
#     conv = conv2d(
#         data=data,
#         channels=channels,
#         kernel_size=kernel_size,
#         strides=strides,
#         padding=padding,
#         data_layout='NCHW',
#         name=name+'_conv')
#     bn = batch_norm(data=conv, epsilon=epsilon, name=name+'_bn')
#     act = relay.nn.relu(data=bn)
#     return act

# data_shape = (1, 3, 224, 224)
# kernel_shape = (32, 3, 3, 3)
# dtype = "float32"
# data = relay.var("data", shape=data_shape, dtype=dtype)
# act = simplenet(data, "graph", 32, strides=(2,2))
# func = relay.Function(relay.analysis.free_vars(act), act)

# 构造2/3/4个连续卷积算子的网络并看优化

# dshape = (1, 1, 64, 64)
# x = relay.var("x", dtype="float32", shape=(1, 2, 2, 2))
# y1 = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=1)
# y2 = relay.nn.conv2d(y1, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=1)
# y3 = relay.nn.conv2d(y2, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=1)

# 构造a+b
input1 = relay.const(1, dtype="float32")
input2 = relay.const(-1, dtype="float32")
input3 = relay.const(1, dtype="float32")
input4 = relay.const(-1, dtype="float32")
x = relay.var("x",dtype="float32")
x = relay.add(input1, input2)
y = relay.var("y",dtype="float32")
y = relay.add(x, input3)
z = relay.var("z",dtype="float32")
z = relay.add(y, input4)

testinput = relay.const(0, dtype="float32")


func = relay.Function(relay.analysis.free_vars(z), z)
# print(y1)


mod = tvm.IRModule.from_expr(func)
# print("unoptimize func "+str(mod["main"]))



# 设置target
target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host)

# 在目标硬件上进行推理并输出分类结果
# 通过计时函数来计算推理耗时

######################################################################
# Execute the portable graph on TVM
# 在TVM上运行
# ---------------------------------
# Now we can try deploying the compiled model on target.
# 现在可以尝试在目标上部署编译后的模型。
from tvm.contrib import graph_runtime

input_name = "input0"

tvm_t0 = time.time()
for i in range(10000):
    dtype = "float32"
    m = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))
    # Set inputs
    # m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    # Execute
    m.run()
    # Get outputs
    tvm_output = m.get_output(0)
tvm_t1 = time.time()

tvm_time = tvm_t1 - tvm_t0
print('Relay time: ', tvm_time / 10000.0, 'seconds')
print('output: ', tvm_output)
print("==========================")
print(mod["main"].body)
print("==========================")


# graphviz attributes
graph_attr = {"color": "red"}
node_attr = {"color": "blue"}
edge_attr = {"color": "black"}

# VizNode is passed to the callback.
# We want to color NCHW conv2d nodes. Also give Var a different shape.
def get_node_attr(node):
    if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
        return {
            "fillcolor": "green",
            "style": "filled",
            "shape": "box",
        }
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}

# Create plotter and pass it to viz. Then render the graph.
dot_plotter = relay_viz.DotPlotter(
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    get_node_attr=get_node_attr)

viz = relay_viz.RelayVisualizer(
    mod,
    #relay_param=param,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("3conv2d")



# optimize simplenet mod

# mod, param= relay.optimize(mod, target="llvm")


# # 设置target
# target = "llvm"
# target_host = "llvm"
# ctx = tvm.cpu(0)
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, target_host=target_host)

# # 在目标硬件上进行推理并输出分类结果
# # 通过计时函数来计算推理耗时

# ######################################################################
# # Execute the portable graph on TVM
# # 在TVM上运行
# # ---------------------------------
# # Now we can try deploying the compiled model on target.
# # 现在可以尝试在目标上部署编译后的模型。
# from tvm.contrib import graph_runtime

# tvm_t2 = time.time()
# for i in range(10000):
#     dtype = "float32"
#     m = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))
#     # Set inputs
#     # m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
#     # Execute
#     m.run()
#     # Get outputs
#     tvm_output = m.get_output(0)
# tvm_t3 = time.time()

# tvm_time = tvm_t3 - tvm_t2
# print('Relay time: ', tvm_time / 10000.0, 'seconds')

# viz = relay_viz.RelayVisualizer(
#     mod,
#     relay_param=param,
#     plotter=dot_plotter,
#     parser=relay_viz.DotVizParser())
# viz.render("3conv2d-optimized")