import tvm
from tvm import relay
from tvm.contrib import relay_viz
from tvm.relay.dataflow_pattern import *



def test_fuse_simple():
    """Simple testcase:假设我们要完成一个y = exp(x+1.0)的计算图"""
    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        return relay.Function([x], z)
    
    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        f1 = relay.Function([x], z)
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        return relay.Function([x], y)
    
    z = before()
    #print(z)
    mod = tvm.IRModule.from_expr(z)

    seq = tvm.transform.Sequential(
        [
            relay.transform.FuseOps(fuse_opt_level=2),
        ]
    )
    
    mod = seq(mod)
    # mod = relay.transform.InferType()(mod)
    #print(mod)
    
    # 测试数据流匹配
    # 测试Node常量是否匹配,包括expr，var，costant，wildcard(任意)，Call，Funtion
    
    # ep = is_expr(relay.var("x", shape=(10, 10)))
    # print(isinstance(ep, ExprPattern))
    # assert isinstance(ep.expr, relay.Var)
    print(mod["main"].body)
    
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
    viz.render("after")
    
test_fuse_simple()