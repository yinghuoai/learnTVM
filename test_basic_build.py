from pytest import param
from tvm.contrib import relay_viz
from tvm.relay.testing import resnet
from unicodedata import name
import tvm
import tvm.relay as relay
import numpy as np
import time


tgt = "llvm"
dev = tvm.cpu()
# func
a = relay.var("a", dtype="float32", shape=(16, 8))
b = relay.var("b", dtype="float32", shape=(8, 8))
c = relay.var("c", dtype="float32", shape=(16, 8))
x = relay.nn.dense(a, b)
print(x)
y = relay.nn.relu(x)
z = y + c
func = relay.Function([a, b, c], z)
A = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), device=dev)
B = tvm.nd.array(np.random.uniform(-1, 1, (8, 8)).astype("float32"), device=dev)
C = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), device=dev)
params = {"b": B, "c": C}
# build
targets = {tvm.tir.IntImm("int32", dev.device_type): tgt}
mod = tvm.IRModule.from_expr(func)
func_in_mod = mod["main"]
assert mod["main"] == func_in_mod, "cannot compare function to itself"

lib = relay.build(mod, targets, "llvm", params=params)
assert mod["main"] == func_in_mod, "relay.build changed module in-place"

# test
rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
rt.set_input("a", A)
rt.run()
out = rt.get_output(0)

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
viz.render("basic")

