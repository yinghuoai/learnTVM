from this import d
import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import rand, run_opt_pass
import torch
import torchvision.models as models

from tvm.contrib import relay_viz
import time


K_ELEMWISE = 0
K_BROADCAST = 1

# 测试匹配const

def test_match_const():
    conv2d = is_op("nn.conv2d")(wildcard(), is_constant())
    pattern = is_op("nn.bias_add")(conv2d, wildcard())

    x = relay.var("x", shape=(1, 3, 224, 224))
    w = relay.var("w", shape=(3, 3, 3, 3))
    b = relay.var("b", shape=(3,))
    conv2d = relay.op.nn.conv2d(x, w)
    out = relay.op.nn.bias_add(conv2d, b)
    func = relay.Function([x, w, b], out)
    mod = tvm.IRModule.from_expr(func)

    assert not pattern.match(mod["main"].body)
    mod["main"] = bind_params_by_name(mod["main"], {"w": tvm.nd.array(np.ones(shape=(3, 3, 3, 3)))})
    assert pattern.match(mod["main"].body)

# 匹配
def test_match_diamond():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    path1 = is_op("nn.relu")(is_conv2d)
    path2 = is_op("nn.leaky_relu")(is_conv2d)
    diamond = is_op("add")(path1, path2)

    # Expr
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)


def test_match_dominator():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Deeper Branch
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    relu = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Single Branch
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Check
    assert diamond.match(out)

    # Fuzzy path/nested Diamond
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()) | is_op(
        "add"
    )(wildcard(), wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    tanh = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = tanh + leaky_relu

    assert diamond.match(out)

class ConvCallback(DFPatternCallback):
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        
        self.y1 = is_op("nn.conv2d")(wildcard(), wildcard())
        self.y2 = is_op("nn.conv2d")(self.y1, wildcard())
        self.y3 = is_op("nn.conv2d")(self.y2, wildcard())
        self.pattern = self.y1

    def callback(self, pre: Expr, post: Expr, node_map: tvm.ir.container.Map) -> Expr:
        y1 = node_map[self.y1][0]
        y2 = node_map[self.y2][0]
        y3 = node_map[self.y3][0]
        # print("y1:",y1)
        
        return relay.const(1, dtype="int32")

if __name__=='__main__':
    #prepare model and input
    model = models.resnet18(pretrained=True)
    
    dshape = (1, 1, 3, 3)
    input0 = relay.var("input0", dtype="float32", shape=(1, 1, 3, 3))
    y1 = relay.nn.conv2d(input0, relay.var("w1"), kernel_size=(1, 1), padding=(0, 0), channels=1)
    y2 = relay.nn.conv2d(y1, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=1)
    y3 = relay.nn.conv2d(y2, relay.var("w3"), kernel_size=(1, 1), padding=(0, 0), channels=1)

    func = relay.Function(relay.analysis.free_vars(y3), y3)
    # print(y1)
    print("输出未重写结果：", func)
    
    


    mod = tvm.IRModule.from_expr(func)
    
    # 绑定参数
    mod["main"] = bind_params_by_name(mod["main"], {"w1": tvm.nd.array(np.ones(shape=(1, 1, 1, 1), dtype="float32"))})
    mod["main"] = bind_params_by_name(mod["main"], {"w2": tvm.nd.array(np.ones(shape=(1, 1, 1, 1), dtype="float32"))})
    mod["main"] = bind_params_by_name(mod["main"], {"w3": tvm.nd.array(np.ones(shape=(1, 1, 1, 1), dtype="float32"))})

    # 模式匹配，接下来开始尝试将这个mod中的部分子图替换掉
    con1 = is_op("nn.conv2d")(wildcard(), wildcard())
    con2 = is_op("nn.conv2d")(con1, wildcard())
    con3 = is_op("nn.conv2d")(con2, wildcard())
    
    print("是否匹配到3个连续卷积：", con3.match(mod["main"].body))
    
    # 尝试模式重写
    from tvm.relay.dataflow_pattern import rewrite
    
    
    re_out = rewrite(ConvCallback(), func)
    
    print("输出重写结果：", re_out)
    
    
    # 将rewirte的结果写入mod
    mod = tvm.IRModule.from_expr(re_out)

    
    ########################################
        
    # 设置随机输入张量数据
    shape_list = [("input0", dshape)]
    random_input = relay.random.np.random.random_sample(shape_list[0][1])
    print("random input: " , random_input)

    # fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
    # print(fake_input)
    # graph = torch.jit.trace(model,fake_input)
    # #main function
    # mod, params = relay.frontend.from_pytorch(graph, shape_list)
    # 用可视化打印出未优化之前的resnet18 relay格式

    input_name = "input0"



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
    viz.render("re_3conv")

  
    #optimize the mod
    #step 1 create target
    #step 1 create PassContext
    #   with tvm.transform.PassContext(opt_level=3):
    #     #step 3 optimize
    #     mod,params=auto_optimize(mod,target,params)
    #   print("optimize func "+str(mod["main"]))


    # 设置target
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



    # 测试推理时间
    tvm_t0 = time.time()
    for i in range(10000):
        dtype = "float32"
        m = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))
        # Set inputs
        m.set_input(input_name, random_input)
        # Execute
        m.run()
        # Get outputs
        tvm_output = m.get_output(0)
    tvm_t1 = time.time()

    tvm_time = tvm_t1 - tvm_t0

    
    print('Relay time: ', tvm_time / 10000.0, 'seconds')
    print('output: ', tvm_output)
    print("==========================")
    # print(mod["main"].body)
    print("==========================")
