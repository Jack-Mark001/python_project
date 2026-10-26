# author: jkisme

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd; import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def house_price_prediction(area):
    # 设置图表风格，让图表更好看
    sns.set_style("whitegrid")
    # -------------------------- 设置中文字体 start --------------------------
    plt.rcParams['font.sans-serif'] = [
        # Windows 优先
        'SimHei', 'Microsoft YaHei',
        # macOS 优先
        'PingFang SC', 'Heiti TC',
        # Linux 优先
        'WenQuanYi Micro Hei', 'DejaVu Sans'
    ]
    # 修复负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 准备数据
    # 假设我们有房屋面积和对应的价格数据
    # 房屋面积（平方米）
    house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120]).reshape(-1, 1)
    # 房屋价格（万元）
    house_prices = np.array([150, 180, 210, 240, 270, 300, 330, 360])

    # 2. 创建并训练模型
    # 创建线性回归模型
    model = LinearRegression()
    # 用数据训练模型（学习面积和价格之间的关系）
    model.fit(house_sizes, house_prices)

    # 3. 使用模型进行预测
    # 预测 85 平方米的房屋价格
    predicted_price = model.predict([[area]])
    print(f"{area} 平方米的房屋预测价格：{predicted_price[0]:.2f} 万元")

    # 4. 可视化结果
    plt.scatter(house_sizes, house_prices, color='blue', label='实际数据')
    plt.plot(house_sizes, model.predict(house_sizes), color='red', label='预测线')
    plt.scatter([area], predicted_price, color='green', s=100, label='预测点')
    plt.xlabel('房屋面积（平方米）')
    plt.ylabel('房屋价格（万元）')
    plt.title('RUNOOB 机器学习测试 -- 房屋面积与价格关系')
    plt.legend()
    plt.grid(True)
    plt.show()


def house2(area):
    # 设置图表风格
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                       'PingFang SC', 'Heiti TC',
                                       'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ============ 1. 构造数据 ============
    # 特征：面积(m²)、房龄(年)、地区
    np.random.seed(42)
    n = 120

    areas = np.random.randint(40, 160, n)  # 面积 40~160 m²
    ages = np.random.randint(0, 31, n)  # 房龄 0~30 年

    # 地区：市中心 / 郊区 / 远郊，不同地区单价不同
    regions = np.random.choice(['市中心', '郊区', '远郊'], n, p=[0.35, 0.4, 0.25])
    region_base = {'市中心': 120, '郊区': 70, '远郊': 40}  # 该地区基础价
    region_unit = {'市中心': 5.0, '郊区': 3.5, '远郊': 2.2}  # 每平米单价系数

    # 价格公式（万元）：
    #   基础价 + 面积*单价系数 - 房龄*1.5 + 噪声
    prices = np.array([
        region_base[r] + region_unit[r] * a - 1.5 * age + np.random.normal(0, 12)
        for r, a, age in zip(regions, areas, ages)
    ])
    prices = np.clip(prices, 20, None)  # 价格不能为负

    # 构建 DataFrame
    df = pd.DataFrame({
        '面积': areas,
        '房龄': ages,
        '地区': regions,
        '价格': prices
    })
    print("数据预览：")
    print(df.head(), "\n")
    print("数据统计：")
    print(df.describe(), "\n")

    # ============ 2. 独热编码 + 训练模型 ============
    # 把「地区」转成 0/1 列（drop_first 避免共线性，也可不 drop）
    df_encoded = pd.get_dummies(df, columns=['地区'], drop_first=True, dtype=int)
    print("编码后的列名：", list(df_encoded.columns), "\n")

    # 特征列（去掉价格）
    feature_cols = [c for c in df_encoded.columns if c != '价格']
    X = df_encoded[feature_cols].values
    y = df_encoded['价格'].values

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # ============ 3. 打印模型参数 ============
    print("模型系数：")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:>10s} : {coef:8.2f}")
    print(f"  截距       : {model.intercept_:8.2f}\n")

    # ============ 4. 评估 ============
    y_pred = model.predict(X_test)
    print(f"MSE : {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R²  : {r2_score(y_test, y_pred):.4f}\n")

    # ============ 5. 预测新房子 ============
    new_houses = pd.DataFrame({
        '面积': [90, 120, 60],
        '房龄': [5, 10, 0],
        '地区': ['市中心', '郊区', '远郊']
    })
    # 注意：编码时要用和训练时相同的列顺序
    new_encoded = pd.get_dummies(new_houses, columns=['地区'], dtype=int)
    new_encoded = new_encoded.reindex(columns=feature_cols, fill_value=0)

    preds = model.predict(new_encoded.values)
    print("新房子预测：")
    for i, row in new_houses.iterrows():
        print(f"  {row['地区']} {row['面积']}m² 房龄{row['房龄']}年 → {preds[i]:.2f} 万元")

    # ============ 6. 可视化 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：真实值 vs 预测值
    axes[0].scatter(y_test, y_pred, color='#3498db', alpha=0.7, s=55)
    lim = [min(y_test.min(), y_pred.min()) - 10,
           max(y_test.max(), y_pred.max()) + 10]
    axes[0].plot(lim, lim, 'r--', linewidth=2, label='完美预测线 y=x')
    axes[0].set_xlabel('真实价格（万元）')
    axes[0].set_ylabel('预测价格（万元）')
    axes[0].set_title(f'真实值 vs 预测值（R²={r2_score(y_test, y_pred):.3f}）')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # 图2：按地区着色的 面积-价格 散点（展示地区差异）
    colors = {'市中心': '#e74c3c', '郊区': '#2ecc71', '远郊': '#3498db'}
    for r, c in colors.items():
        sub = df[df['地区'] == r]
        axes[1].scatter(sub['面积'], sub['价格'], color=c, alpha=0.65,
                        label=r, s=45)
    axes[1].set_xlabel('面积（平方米）')
    axes[1].set_ylabel('价格（万元）')
    axes[1].set_title('不同地区的面积-价格关系')
    axes[1].legend(title='地区')
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

house2(100)

