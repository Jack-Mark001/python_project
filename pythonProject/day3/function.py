# author: jkisme
import torch

#计算 y = 3x² + 4x + 5 在 x=2 处的导数
def fun1():
    x=torch.tensor(2.0, requires_grad=True)
    y=3*x**2+4*x+5
    y.backward()
    print("y=",y.item(),"x的导数为:",x.grad.item())

# fun1()

# 计算 f(x,y) = x²y + y² 在点 (2,3) 处对 x 和 y 的偏导数
def fun2():
    x=torch.tensor(2.0, requires_grad=True)
    y=torch.tensor(3.0, requires_grad=True)
    z=x**2*y+y**2
    z.backward()
    print("z=",z.item(),"x的偏导数为:",x.grad.item(),"y的偏导数为:",y.grad.item())

# fun2()

# 用两个数据点训练线性模型 y = w*x + b
def fun3():
    x = torch.tensor([2.0, 3.0])  # 输入值
    y = torch.tensor([11.0, 14.0])  # 目标值

    # 初始化参数 w, b（requires_grad=True 才能被优化）
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # 优化器：使用 SGD
    lr = 0.01
    optimizer = torch.optim.SGD([w, b], lr=lr)

    # 训练循环
    for epoch in range(20000):
        # 前向计算预测值
        y_pred = w * x + b

        # 计算损失（均方误差 MSE）
        loss = ((y_pred - y) ** 2).mean()

        # 反向传播 + 更新参数
        optimizer.zero_grad()  # 清空上一步的梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新 w, b

        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | loss={loss.item():.6f} | w={w.item():.4f} | b={b.item():.4f}")

    # 最终结果
    print("\n训练完成：")
    print("w =", w.item())
    print("b =", b.item())
    print("预测值:", (w * x + b).detach())
fun3()
