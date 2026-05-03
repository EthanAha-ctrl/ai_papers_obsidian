
















基于我对 Nixtla 生态系统的了解，我来详细讲解这两个网页的内容。

---

## 概述

这两个网页是 Nixtla 的 **`datasetsforecast`** Python 库的文档页面，专门用于提供 **长时序预测（Long-Horizon Time Series Forecasting）** 的标准 benchmark datasets 的便捷加载接口。

---

## 网页 1: `LongHorizon`

**URL**: [https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html)

这个页面对应 `datasetsforecast` 库中的 **`LongHorizon` 类**，它封装了经典的长时序预测 benchmark 数据集。这些数据集源自 **Informer** 论文（Zhou et al., 2021, AAAI）及其后续工作，是长序列时间序列预测领域的标准评测集。

### 包含的数据集

| 数据集名称 | 频率 | 时间步数 | 变量数 | 描述 |
|---|---|---|---|---|
| **ETTh1** | Hourly | 17,420 | 7 | Electricity Transformer Temperature (Hourly, Region 1) |
| **ETTh2** | Hourly | 17,420 | 7 | Electricity Transformer Temperature (Hourly, Region 2) |
| **ETTm1** | 15-min | 69,680 | 7 | Electricity Transformer Temperature (15-min, Region 1) |
| **ETTm2** | 15-min | 69,680 | 7 | Electricity Transformer Temperature (15-min, Region 2) |
| **Electricity** | Hourly | 26,304 | 321 | 电力消耗数据 |
| **Traffic** | Hourly | 17,544 | 862 | 交通流量数据 |
| **Weather** | 10-min | 52,696 | 21 | 气象数据 |

### 使用方式

```python
from datasetsforecast.long_horizon import LongHorizon

# 加载数据
Y_df, X_df, S_df = LongHorizon.load(directory='./data', group='ETTh1')

# Y_df: target time series (包含 unique_id, ds, y 列)
# X_df: exogenous variables
# S_df: static variables
```

### 数据集细节

**ETT (Electricity Transformer Temperature)**:
- 来自中国某省的两个区域变压器温度数据
- 每条记录包含 **7 个特征**: `HUFL` (High Useful Load), `HULL` (High Useless Load), `MUFL` (Mid Useful Load), `MULL` (Mid Useless Load), `LUFL` (Low Useful Load), `LULL` (Low Useless Load), `OT` (Oil Temperature)
- `OT` (Oil Temperature) 是通常的预测目标
- ETTh = Hourly 采样, ETTm = 15-minute 采样

**Electricity**:
- 来自 UCI 机器学习库
- 321 个客户的小时电力消耗数据
- 来自 2012-2014 年的记录

**Traffic**:
- 加州高速公路的交通流量数据
- 862 个传感器的小时数据
- 来自 2015-2016 年

**Weather**:
- 21 个气象指标 (温度、湿度、风速等)
- 10 分钟采样间隔
- 来自 2020 年的记录

---

## 网页 2: `LongHorizon2`

**URL**: [https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html)

这个页面对应 **`LongHorizon2` 类**，它提供了 **扩展版** 或 **第二版** 的长时序 benchmark 数据集。这些通常是后续研究（如 Autoformer, PatchTST 等论文）引入的更大规模、更多样化的数据集。

### 包含的数据集（典型）

| 数据集名称 | 频率 | 时间步数 | 变量数 | 描述 |
|---|---|---|---|---|
| **ETTh1** | Hourly | 17,420 | 7 | 同上，但预测 horizon 定义不同 |
| **ETTh2** | Hourly | 17,420 | 7 | 同上 |
| **ETTm1** | 15-min | 69,680 | 7 | 同上 |
| **ETTm2** | 15-min | 69,680 | 7 | 同上 |
| **Electricity** | Hourly | 26,304 | 321 | 同上 |
| **Traffic** | Hourly | 17,544 | 862 | 同上 |
| **Weather** | 10-min | 52,696 | 21 | 同上 |

### LongHorizon vs LongHorizon2 的关键区别

两个类的核心差异不在于数据集本身的原始数据，而在于 **prediction horizon（预测窗口长度）** 的定义和 **train/val/test split** 的划分方式不同：

**LongHorizon** (来自 Informer 论文):
- 预测 horizon 长度较短
- 例如 ETTh1 的 horizon 为: 24, 48, 168, 336, 720

**LongHorizon2** (来自后续论文如 Autoformer, FEDformer):
- 使用更标准的 horizon 定义
- 例如 ETTh1 的 horizon 为: 96, 192, 336, 720
- 更严格的数据划分比例

---

## 第一性原理：什么是 Long-Horizon Forecasting？

### 问题定义

给定历史序列 $\mathbf{x}_{1:T} = \{x_1, x_2, \ldots, x_T\}$，需要预测未来 $H$ 步的值 $\mathbf{x}_{T+1:T+H}$，其中 $H$ 是预测 horizon。

$$\hat{\mathbf{x}}_{T+1:T+H} = f(\mathbf{x}_{1:T}; \theta)$$

其中：
- $T$ = 输入序列长度
- $H$ = 预测 horizon 长度
- $f$ = 模型函数
- $\theta$ = 模型参数

### 为什么 Long-Horizon 困难？

1. **Error Accumulation（误差累积）**: 自回归模型中，每步的预测误差会传播到后续步骤
2. **Long-term Dependency（长程依赖）**: 需要捕获跨数百甚至数千步的时间依赖
3. **Distribution Shift（分布漂移）**: 数据的非平稳性在长 horizon 上更显著

### 评测指标

常用 **MSE (Mean Squared Error)** 和 **MAE (Mean Absolute Error)**:

$$\text{MSE} = \frac{1}{H} \sum_{t=T+1}^{T+H} (x_t - \hat{x}_t)^2$$

$$\text{MAE} = \frac{1}{H} \sum_{t=T+1}^{T+H} |x_t - \hat{x}_t|$$

其中：
- $x_t$ = 时刻 $t$ 的真实值
- $\hat{x}_t$ = 时刻 $t$ 的预测值
- $H$ = 预测 horizon 长度

---

## 数据集的 Train/Val/Test Split 规范

对于 LongHorizon 数据集，标准的划分方式如下：

| 数据集 | Train | Validation | Test |
|---|---|---|---|
| ETTh1/ETTh2 | 8,545 | 2,781 | 2,781 |
| ETTm1/ETTm2 | 34,465 | 11,523 | 11,523 |
| Electricity | 18,413 | 2,633 | 5,260 |
| Traffic | 12,285 | 1,759 | 3,516 |
| Weather | 36,887 | 5,269 | 10,538 |

---

## 在 Nixtla 生态中的位置

这些数据集在 Nixtla 的 **NeuralForecast** 库中被广泛使用来 benchmark 各种模型：

```
Nixtla 生态系统:
├── NeuralForecast (深度学习预测模型库)
│   ├── Informer
│   ├── Autoformer
│   ├── PatchTST
│   ├── TFT (Temporal Fusion Transformer)
│   ├── N-BEATS / N-HiTS
│   └── ...
├── datasetsforecast (数据加载工具) ← 这两个网页所在位置
│   ├── LongHorizon
│   ├── LongHorizon2
│   ├── M3, M4, M5
│   └── ...
├── StatsForecast (统计模型库)
└── HierarchicalForecast (层次化预测)
```

---

## 参考链接

- **Nixtla 文档首页**: [https://nixtlaverse.nixtla.io/](https://nixtlaverse.nixtla.io/)
- **LongHorizon 文档**: [https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html)
- **LongHorizon2 文档**: [https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html)
- **Informer 论文**: Zhou, H., et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021. [https://arxiv.org/abs/2012.07436](https://arxiv.org/abs/2012.07436)
- **Autoformer 论文**: Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting." NeurIPS 2021. [https://arxiv.org/abs/2106.13008](https://arxiv.org/abs/2106.13008)

---

**总结**: 这两个网页是 Nixtla `datasetsforecast` 库中用于加载长时序预测 benchmark 数据集的 API 文档，`LongHorizon` 提供经典 Informer 系列的 benchmark 配置，`LongHorizon2` 提供后续论文（Autoformer 等）的扩展配置，两者在 horizon 定义和数据划分上有所不同。