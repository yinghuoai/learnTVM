# 可以模仿第5讲mlc 自动程序优化
# 将高并发运行多个小的splitConv2d
# 改为自动高并发尝试运行并得出最小时间，和最小优化尝试
# 简单的实现就是for循环

# 不过具体有两个问题
# 一个就是自动：通过求出out张量的H，W等，然后用求因子的函数求出各个因子，
#              然后这些因子就可以作为split的小conv2d的H， W

# 还有一个就是如何将自动和@TVM.script编写的prim_fun()里面的参数联系起来？
# 这个就有一个问题，以为输入里面如果不是具体数字会报错。