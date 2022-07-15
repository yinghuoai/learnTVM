import tvm
from tvm import te
import numpy as np

from tvm.ir import transform


# 首先编写一个简单的向量相加，
# 并且使用默认的schedule构建它。
# 然后，使用自定义lower pass直接操作IR

n = tvm.tir.const(128, "int32")
a = te.placeholder((n,), name="a")
b = te.placeholder((n,), name="b")
c = te.compute((n,), lambda i: a[i] + b[i], name="c")

sch = te.create_schedule(c.op)
ir = tvm.lower(sch, [a, b, c])
print(ir)

# 编写pass

# 必须使用数组来存储IR访问的结果
loops = []

def find_width8(op):
    """Find all the 'tir.For' nodes whose extent can be divided by 8."""
    if isinstance(op, tvm.tir.For):
        if isinstance(op.extent, tvm.tir.IntImm):
            if op.extent.value % 8 ==0:
                loops.append(op)


# IR 转换
# transform接口和visitor接口不同，
# transform同时支持前序和后序回调
# visitor只支持后序回调，
# 如果需要保留原始IR节点，返回None；
# 如果要更改，使用功能TVM IR maker构建并返回这个值

def vectorize8(op):
    """Split can vectorize the loops found in `find_width8`."""
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = te.var(name + ".outer"), te.var(name + ".inner")
        body = tvm.tir.stmt_functor.substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.tir.For(li, 0, 8, tvm.tir.ForKind.VECTORIZED, body)
        body = tvm.tir.For(lo, 0, extent // 8, tvm.tir.ForKind.SERIAL, body)
        return body
    return None


@tvm.tir.transform.prim_func_pass(opt_level=0)
def vectorize(f, mod, ctx):
    global loops

    tvm.tir.stmt_functor.post_order_visit(f.body, find_width8)

    if not loops:
        return sf

    # The last list arugment indicates what kinds of nodes will be transformed.
    # Thus, in this case only `For` nodes will call `vectorize8`
    return f.with_body(tvm.tir.stmt_functor.ir_transform(f.body, None, vectorize8, ["tir.For"]))

with tvm.transform.PassContext(config={"tir.add_lower_pass": [(1, vectorize)]}):
    print(tvm.lower(sch, [a, b, c]))




