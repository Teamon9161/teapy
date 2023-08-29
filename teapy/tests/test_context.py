# import os

# import teapy as tp
# from teapy import context as ct
# from teapy.testing import assert_allclose

# # 设置环境变量
# os.environ["RUST_BACKTRACE"] = "1"

# c = ct("b").ts_sum(2)  # .alias('c')
# # d = ct('a').mean().alias('d')

# dd = tp.DataDict(a=[1.0, 2, 3, 4], b=[4, 3, 1])

# # dd = dd.with_columns([ct('a').mean().alias('d')])

# c.eval(context=dd)

# d.eval(context=dd)

# assert_allclose(c.eval(context=dd).view, [4, 7, 4])
# assert dd["c"].step == 0
# dd.eval()
# assert dd["d"].view == 5
