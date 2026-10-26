# author: jkisme
import torch

# 求导
def test1():
    #1准备数据
    x = torch.tensor([3.0], requires_grad=True)

    #2 前向传播
    y = 0 * x + 5

    #3 反向传播,求梯度
    y.backward()

    #4 拿到结果梯度
    print(f"导数结果:{x.grad}")

def test2():
    # 1.【准备数据 & 标记叶子节点】：4.0随便写，直线的导数至于K有关
    x = torch.tensor([4.0], requires_grad=True)  # 在x=4处求导

    # 2.【前向传播 & 构建计算图】：直线函数 的 导数就是K，是一个常量
    y = 2 * x + 1

    # 3.【反向传播 & 计算梯度】
    y.backward()

    # 4.【查看梯度 & 提取结果】
    print(f"导数为: {x.grad}")  # 输出: tensor([2.]) ，如果k=3.0，导数就是3.

def test3():
    # 1.【准备数据 & 标记叶子节点】
    x = torch.tensor([2.0], requires_grad=True)

    # 2.【前向传播 & 构建计算图】
    y = x ** 2

    # 3.【反向传播 & 计算梯度】
    y.backward()

    # 4.【查看梯度 & 提取结果】 ** 等价于 ^
    # 输出: tensor([4.]),数学求导公式知道x ** 2的导数=2*x^(2-1) = 2x
    print(f"导数为: {x.grad}")
    # 修改x的值为3 - 导数为6，修改x的值为4 - 导数为8...

def train01():
    # 1.准备数据 和 参数
    # 1.1.准备目标数据
    #输入 1，答案应该是 5
	#输入 3，答案应该是 10    期望的结果是：y = 2.5x + 2.5
    x = torch.tensor([1.0, 3.0]) # 输入值
    y = torch.tensor([5.0, 10.0]) # 目标值
    # 1.2.初始化参数（从0开始猜）
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # 2.训练1000轮 = 1000次参数更新
    for epoch in range(2000):
        # 2.1.正向传播
        y_prep = w * x + b
        # 2.2.计算损失值
        loss = ((y_prep - y) ** 2).mean() # 或MSE()
        # 2.3.反向传播：自动微分
        loss.backward()
        # 2.4.更新参数
        with torch.no_grad():
            w -= 0.1 * w.grad # 学习步长/学习率写死0.01，可以改
            b -= 0.1 * b.grad
            w.grad.zero_() #梯度清零，不然会累加
            b.grad.zero_()
        # 打印进度
        if epoch % 100 == 0:
            print(f"轮数{epoch}: loss={loss.item():.4f}, "
                  f"w={w.item():.4f}, b={b.item():.4f}") # :.4f表示把这个数字显示成浮点数，并且保留4位小数

    print(f"\n最终: y = {w.item():.4f}x + {b.item():.4f}")

def train02():
    # 1.准备数据 和 参数
    # 1.1.准备目标数据
    x = torch.linspace(-2.0, 2.0, 20) # 输入值，从 -2 到 2，均匀取 20 个点
    y = x ** 2 # 目标值，目标值就是 x 的平方，也就是说，真实规律是 y = x²
    # 1.2.初始化参数（从0开始猜）
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)

    # 2.训练1000轮 = 1000次参数更新
    for epoch in range(2000):
        # 2.1.正向传播
        y_prep = a * (x ** 2) + b * x + c
        # 2.2.计算损失值
        loss = ((y_prep - y) ** 2).mean() # 或MSE()
        # 2.3.反向传播：自动微分
        loss.backward()
        # 2.4.更新参数
        with torch.no_grad():
            a -= 0.01 * a.grad # 学习步长/学习率写死0.01，可以改
            b -= 0.01 * b.grad
            c -= 0.01 * c.grad
            a.grad.zero_() #梯度清零，不然会累加
            b.grad.zero_()
            c.grad.zero_()
        # 打印进度
        if epoch % 200 == 0:
            print(f"轮数{epoch}: loss={loss.item():.4f}, "
                  f"a={a.item():.4f}, b={b.item():.4f}, c={c.item():.4f}") # :.4f表示把这个数字显示成浮点数，并且保留4位小数

    print(f"\n最终: y = {a.item():.4f}x**2 + {b.item():.4f}x + {c.item():.4f}")

train02()
