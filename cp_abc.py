import onnx
from tvm.contrib.download import download_testdata
from PIL import  Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import time
import torch
from tvm.contrib import relay_viz



# 第一步，用relay构建初始的mod
dshape = (1, 1, 3, 3)
# tmp = relay.const(2, "int32")
A = relay.var("input0", dtype="float32", shape=(1, 1, 3, 3))
B = relay.nn.conv2d(A, relay.ones(shape=(1,1,1,1), dtype="float32"), kernel_size=(1, 1), padding=(0, 0), channels=1)
C = relay.multiply(B, relay.const(2, "float32"))
Path1 = relay.power(B, relay.const(-1, "float32"))
Path2 = relay.power(C, relay.const(-1, "float32"))
D = relay.multiply(Path1, Path2)
print(D)
func = relay.Function(relay.analysis.free_vars(D), D)
mod = tvm.IRModule.from_expr(func)

# 第二步，用relay构建用了交换律和结合律的mod2
dshape = (1, 1, 3, 3)
# tmp = relay.const(2, "int32")
A2 = relay.var("input1", dtype="float32", shape=(1, 1, 3, 3))
B2 = relay.nn.conv2d(A2, relay.ones(shape=(1,1,1,1), dtype="float32"), kernel_size=(1, 1), padding=(0, 0), channels=1)
C2 = relay.power(B2, relay.const(-2, "float32"))
# Path = relay.power(C2, relay.const(1, "float32"))
D2 = relay.divide(C2, relay.const(2, "float32"))
#print(D2)  
print(D2)
func2 = relay.Function(relay.analysis.free_vars(D2), D2)
mod2 = tvm.IRModule.from_expr(func2)


# 这里可以插入运行之前的两个mod，还没有经过build的relay可视化的操作

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
viz.render("mod_before")

viz2 = relay_viz.RelayVisualizer(
    mod2,
    #relay_param=param,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz2.render("mod2_before")





# 设置同一输入，然后分别运行多次，比较推理时间
# 设置随机输入张量数据
# shape_list = [("input0", dshape)]
# random_input = relay.random.np.random.random_sample(shape_list[0][1])
random_input = np.ones(shape=dshape, dtype="float32")*2014+23
# random_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))

# 设置target
input_name = "input0"
target = "llvm"
target_host = "llvm"
ctx = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target)

# 在目标硬件上进行推理并输出分类结果
# 通过计时函数来计算推理耗时

######################################################################
# Execute the portable graph on TVM
# 在TVM上运行
# ---------------------------------
# Now we can try deploying the compiled model on target.
# 现在可以尝试在目标上部署编译后的模型。
from tvm.contrib import graph_runtime


with tvm.transform.PassContext(opt_level=2):
    lib = relay.build(mod, target=target)
# 测试推理时间
tvm_t0 = time.time()
for i in range(10000):
    dtype = "float32"
    m = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input("input0", random_input)
    # Execute
    m.run()
    # Get outputs
    mod_output = m.get_output(0)
tvm_t1 = time.time()

tvm_time = tvm_t1 - tvm_t0



with tvm.transform.PassContext(opt_level=2):
    lib2 = relay.build(mod2, target=target)
# 测试推理时间
tvm_t3 = time.time()
for i in range(10000):
    dtype = "float32"
    m = graph_runtime.graph_executor.GraphModule(lib2["default"](ctx))
    # Set inputs
    m.set_input("input1", random_input)
    # Execute
    m.run()
    # Get outputs
    mod2_output = m.get_output(0)
tvm_t4 = time.time()

mod_time = tvm_t1 - tvm_t0
mod2_time = tvm_t4 - tvm_t3

print('mod Relay time: ', mod_time / 10000.0, 'seconds')
print('mod output: ', mod_output)
print("==========================")
print('mod2 Relay time: ', mod2_time / 10000.0, 'seconds')
print('mod2 output: ', mod2_output)
print("==========================")


# 最后可以在运行之后，添加对两个mod的relay可视化的操作

