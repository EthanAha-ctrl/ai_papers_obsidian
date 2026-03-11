## 一、RoPE基础回顾

### 1.1 RoPE核心公式

RoPE (Rotary Position Embedding) 通过旋转矩阵将位置信息编码到Transformer的Attention中：

**频率计算公式：**
```
θ_i = 1 / (base^(2i/d_model))  其中 base = 10000
```

**旋转矩阵应用：**
```
f_q(x, m) = R_m W_q x
f_k(x, n) = R_n W_k x
```

其中旋转矩阵：
```
R_m = [[cos(mθ_0), -sin(mθ_0)], [sin(mθ_0), cos(mθ_0)] ⊗ ...]
```

### 1.2 RoPE的局限性

**问题表现：**
- **位置外推失效**：当序列长度超过训练长度时，attention score会剧烈波动
- **高频成分衰减**：远距离的位置信息丢失
- **训练长度约束**：通常限制在2k-4k tokens

参考链接：https://arxiv.org/abs/2104.09864

---

## 二、Dynamic RoPE (NTK-aware Scaling)

### 2.1 核心思想

Dynamic RoPE通过**动态调整base值**来实现外推，关键创新在于：

**NTK (Neural Tangent Kernel) 感知缩放：**
```python
def dynamic_ntk_alpha(current_seq_len, max_seq_len, base=10000):
    # 动态计算alpha值
    alpha = (current_seq_len / max_seq_len) - (max_seq_len / current_seq_len)
    # 调整base
    new_base = base * alpha ** (d_model / (d_model - 2))
    return new_base
```

### 2.2 技术细节

**公式推导：**

原始RoPE频率：
```
θ_i = 1 / (10000^(2i/d))
```

NTK-aware缩放后：
```
θ_i' = 1 / ((10000 * α)^(2i/d))
```

**实现代码框架：**

```Python
import torch

import torch.nn as nn

import matplotlib.pyplot as plt

  

class DynamicRoPE(nn.Module):

    def __init__(self, dim, max_seq_len=4096, base=10000.0):

        super().__init__()

        self.dim = dim

        self.base = base

        self.max_seq_len = max_seq_len

  

    def get_freqs(self, seq_len, device):

        # 动态计算base / Dynamic base calculation

        # 如果序列长度超过max_seq_len，则根据比例调整base

        if seq_len > self.max_seq_len:

            # 这里的缩放策略可以根据具体论文调整，这里沿用原代码的思路

            current_alpha = (seq_len / self.max_seq_len) ** 0.5

            effective_base = self.base * current_alpha

        else:

            effective_base = self.base

        # 生成频率 / Generate frequencies

        # theta_i = 1 / (base ^ (2i/d))

        inv_freq = 1.0 / (effective_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))

        return inv_freq

  

    def forward(self, x, seq_len=None):

        # x shape: [batch, seq_len, dim]

        if seq_len is None:

            seq_len = x.shape[1]

        device = x.device

        inv_freq = self.get_freqs(seq_len, device)

        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)

        freqs = torch.outer(t, inv_freq) # [seq_len, dim/2]

        # 为了方便计算，通常将cos/sin扩展到与x相同的形状

        # 这里我们生成 [seq_len, dim] 的 cos 和 sin

        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()

  

def rotate_half(x):

    """Rotates half the hidden dims of the input."""

    x1 = x[..., :x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)

  

def apply_rotary_pos_emb(x, cos, sin):

    """

    Apply rotary position embeddings.

    x: [batch, seq_len, dim]

    cos, sin: [seq_len, dim] (broadcastable to x)

    """

    # 简单的广播机制，假设 cos/sin 已经 match 了 seq_len

    return (x * cos) + (rotate_half(x) * sin)

  

def visualize_pe(dim=128, max_seq_len=50, test_seq_len=100):

    print(f"Initializing DynamicRoPE with dim={dim}, max_seq_len={max_seq_len}")

    model = DynamicRoPE(dim, max_seq_len)

    # 模拟输入

    print(f"Generating embeddings for test_seq_len={test_seq_len}...")

    x = torch.randn(1, test_seq_len, dim)

    cos, sin = model(x)

    # 可视化

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)

    plt.imshow(cos.cpu().numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    plt.title(f'Cosine (Seq Len {test_seq_len}, Max {max_seq_len})')

    plt.xlabel('Dimension')

    plt.ylabel('Position')

    plt.colorbar()

  

    plt.subplot(1, 2, 2)

    plt.imshow(sin.cpu().numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    plt.title(f'Sine (Seq Len {test_seq_len}, Max {max_seq_len})')

    plt.xlabel('Dimension')

    plt.ylabel('Position')

    plt.colorbar()

    plt.tight_layout()

    plt.show()

    print("Visualization complete.")

  

if __name__ == "__main__":

    # 运行示例

    visualize_pe(dim=64, max_seq_len=50, test_seq_len=120)
```

### 2.3 NTK理论解释

**关键洞察：**
- **低频维度**负责处理**长程依赖**
- **高频维度**负责处理**局部模式**

通过扩大base值：
- **低频成分的周期变长**，使模型能处理更长距离
- **保持相对位置的稳定性**

**Scaling Law公式：**
```
最佳base = 原始base × (目标长度/原始长度)^(d_model/(d_model-2))
```

参考链接：https://arxiv.org/abs/2310.05209

---

## 三、YaRN (Yet another RoPE extension)

### 3.1 YaRN创新点

YaRN结合了**多项式插值**和**动态缩放**，核心包括三个组件：

1. **Beta Scaling**：温度参数缩放
2. **Interpolation**：位置插值
3. **NTK-aware Scaling**：保留NTK特性

### 3.2 YaRN完整公式

**1. 位置插值：**
```
θ_i' = θ_i × β^(2i/d_model)
```

**2. Beta参数计算：**
```python
def compute_beta(s, L, L_train, d_model):
    # s: 目标长度
    # L: 原始长度
    # L_train: 训练长度
    ratio = s / L
    beta = 1 + (ratio - 1) * (d_model / (d_model - 2))
    return beta
```

**3. 完整YaRN变换：**
```
位置变换: m' = m × L_train / L
频率变换: θ_i'' = θ_i × β^(2i/d_model)
```

### 3.3 YaRN架构流程图

```
输入序列 → [位置缩放] → [RoPE编码] → [Attention计算]
            ↓
       m' = m × α
            ↓
       θ_i' = θ_i × β^(2i/d)
            ↓
       旋转矩阵 R_{m',θ'}
            ↓
       Attention(q ⊙ R, k ⊙ R)
```

### 3.4 YaRN实现代码


```Python
import torch

import torch.nn as nn

import matplotlib.pyplot as plt

import math

  

class YaRNRoPE(nn.Module):

    def __init__(self, d_model, max_seq_len=4096, base=10000.0, original_max_seq_len=4096):

        super().__init__()

        self.d_model = d_model

        self.base = base

        self.max_seq_len = max_seq_len

        self.original_max_seq_len = original_max_seq_len

  

    def get_freqs(self, seq_len, device):

        # 计算扩展因子 / Calculate scale factor

        scale = 1.0

        if seq_len > self.original_max_seq_len:

            scale = seq_len / self.original_max_seq_len

        # 如果没有扩展，直接返回标准频率

        if scale <= 1.0:

            two_i = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)

            return self.base ** (-two_i / self.d_model)

  

        # YaRN 参数 / YaRN parameters

        beta_fast = 32

        beta_slow = 1

        # 生成基础频率 / Generate base frequencies

        two_i = torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)

        theta = self.base ** (-two_i / self.d_model)

        # 计算波长 lambda = 2 * pi / freq

        # 计算 r = L / lambda = L * freq / (2 * pi)

        # 这里 L 是原始最大长度

        r = self.original_max_seq_len * theta / (2 * math.pi)

        # 计算 ramp 函数 / Calculate ramp function

        # r < beta_slow (低频/长波长) -> 全插值 (freq / scale) -> 需要拉伸以覆盖更长上下文

        # r > beta_fast (高频/短波长) -> 无插值 (freq) -> 保持局部性

        # alpha = (r - beta_slow) / (beta_fast - beta_slow)

        # alpha 限制在 [0, 1]

        # alpha = 0 -> r <= beta_slow -> 全插值

        # alpha = 1 -> r >= beta_fast -> 无插值

        alpha = (r - beta_slow) / (beta_fast - beta_slow)

        alpha = torch.clamp(alpha, 0.0, 1.0)

        # 混合频率 / Mix frequencies

        # theta_interp = theta / scale

        # theta_no_interp = theta

        # final_theta = alpha * theta_no_interp + (1 - alpha) * theta_interp

        theta_interp = theta / scale

        final_theta = alpha * theta + (1 - alpha) * theta_interp

        return final_theta

  

    def forward(self, x, seq_len=None):

        if seq_len is None:

            seq_len = x.shape[1]

        device = x.device

        theta = self.get_freqs(seq_len, device)

        t = torch.arange(seq_len, device=device, dtype=theta.dtype)

        # 外积计算 theta_pos = t * theta

        freqs = torch.outer(t, theta)

        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos(), emb.sin()

  

def visualize_pe(d_model=128, original_max_len=50, test_seq_len=100):

    print(f"Initializing YaRNRoPE with d_model={d_model}, original_max_len={original_max_len}")

    # max_seq_len 这里设为 test_seq_len 只是为了初始化，实际上 YaRN 逻辑依赖 original_max_seq_len

    model = YaRNRoPE(d_model, max_seq_len=test_seq_len, original_max_seq_len=original_max_len)

    # 模拟输入

    print(f"Generating embeddings for test_seq_len={test_seq_len} (Scale={test_seq_len/original_max_len})...")

    x = torch.randn(1, test_seq_len, d_model)

    cos, sin = model(x)

    # 可视化

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)

    plt.imshow(cos.cpu().numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    plt.title(f'YaRN Cosine (Seq {test_seq_len}, Orig Max {original_max_len})')

    plt.xlabel('Dimension')

    plt.ylabel('Position')

    plt.colorbar()

  

    plt.subplot(1, 2, 2)

    plt.imshow(sin.cpu().numpy(), aspect='auto', cmap='RdBu', vmin=-1, vmax=1)

    plt.title(f'YaRN Sine (Seq {test_seq_len}, Orig Max {original_max_len})')

    plt.xlabel('Dimension')

    plt.ylabel('Position')

    plt.colorbar()

    plt.tight_layout()

    plt.show()

    print("Visualization complete.")

  

if __name__ == "__main__":

    # 运行示例：扩展 2.4 倍

    visualize_pe(d_model=64, original_max_len=50, test_seq_len=120)
```
让我来理解一下这段代码: 最终的theta是两个theta的alpha blending: 一个是原本正常的theta, 一个叫theta_intp. theta_intp 来来自于简单的对原本theta的linear scaling. 然后看alpha, alpha来自于 1 到 32 之间的一个缩放比例。 你怎么看我的理解. respond in Chinese.

简单总结一下你的观点：

1. **混合机制 (Alpha Blending)**：没得说，就是拿 
    
    ```
    alpha
    ```
    
     做线性插值。
    - ```
        final_theta = alpha * 原本theta + (1-alpha) * 插值theta
        ```
        
2. **来源 (Scaling)**：
    
    ```
    theta_intp
    ```
    
     确实就是简单的 Linear Scaling (
    
    ```
    theta / scale
    ```
    
    )，这和最原始的线性插值 RoPE 是一样的。
3. **Alpha 的阈值 (1 到 32)**：
    
    ```
    alpha
    ```
    
     确实是由指标 
    
    ```
    r
    ```
    
     是否落在 
    
    ```
    1
    ```
    
     (
    
    ```
    beta_slow
    ```
    
    ) 到 
    
    ```
    32
    ```
    
     (
    
    ```
    beta_fast
    ```
    
    ) 这个区间决定的。

为了让你理解得更透彻，这里稍微补充一点点物理含义，这能帮你理解**为什么要这么做**：

这里的 

**`r`**

 其实代表了 **“在原始上下文长度 (比如 4k) 内，这个频率的波能转多少圈”**。

- **高频部分 (
    
    ```
    r > 32
    ```
    
    )**：波转得很快（转了32圈以上）。这些维度负责捕捉**局部关注力**（比如 adj、noun 搭配）。对于局部关系，不管是 4k 窗口还是 100k 窗口，它们的相对位置关系是不变的，所以**不需要缩放**，保持原样最好 (
    
    ```
    alpha=1
    ```
    
     -> 用 
    
    ```
    theta
    ```
    
    )。保留原样能防止网络变“傻”，避免在这个微观尺度上混淆。
- **低频部分 (
    
    ```
    r < 1
    ```
    
    )**：波转得很慢（转不完一圈）。这些维度负责捕捉**长程依赖**（全局语义）。当窗口变长时，我们需要把这个波“拉长”才能覆盖新的长度，否则模型就“看不见”更远的地方了，所以需要**完全缩放** (
    
    ```
    alpha=0
    ```
    
     -> 用 
    
    ```
    theta_intp
    ```
    
    )。
- **中间部分 (
    
    ```
    1 ~ 32
    ```
    
    )**：做一个平滑过渡。

所以 YaRN 的精髓就在于：**不要一刀切**（全缩放会损失局部精度，全不缩放会丢失长程视野），而是根据频率的高低，**混合**使用两种策略.

如果我们看**最终频率** $\theta_{final}$ 和**原始频率** $\theta$ 的关系：

在中间那个过渡区间（Ramp 区间）：

1. $\alpha$ 是 $\theta$ 的线性函数：$\alpha \approx c_1 \cdot \theta + c_2$
2. 插值公式是：$\theta_{final} = \theta \cdot [\alpha + \frac{1-\alpha}{scale}]$

把 1 代入 2，你会发现 $\theta_{final}$ 确实变成了 $\theta$ 的**二次多项式（Quadratic Polynomial）**： $$ \theta_{final} \propto \theta^2 + \dots $$

所以你说得一点没错，YaRN 本质上就是设计了一个**分段函数**：两头是线性的（一头斜率是 $1/s$，一头是 $1$），中间用一个多项式曲线把它**平滑地“对”在一起**。

这么做唯一的目的就是**消除割裂感**。如果直接从“拉伸”跳到“不拉伸”，模型在那个频率点会懵掉（出现 Artifacts）；用曲线平滑过渡一下，为了让模型觉得自然一点。
### 3.5 YaRN训练效率对比

| 方法 | 训练Token数 | 训练步数 | 推理速度 |
|------|------------|---------|---------|
| PI (Positional Interpolation) | 50B | 4000 | 基准 |
| Dynamic RoPE | 5B | 1000 | 1.2x |
| **YaRN** | **500M** | **400** | **1.5x** |

参考链接：https://arxiv.org/abs/2309.00071

---

## 四、超长上下文处理机制对比

### 4.1 方法对比表

| 方法                              | 核心机制      | 最大扩展倍数   | 训练需求   | 优点     | 缺点    |
| ------------------------------- | --------- | -------- | ------ | ------ | ----- |
| **原始RoPE**                      | 固定base    | 1x       | 无      | 简单     | 长度受限  |
| **Linear Scaling**              | 线性缩小位置    | ~2x      | 少      | 简单     | 精度下降快 |
| **PI (Position Interpolation)** | 位置插值      | ~8x      | 中等     | 稳定     | 扩展有限  |
| **Dynamic RoPE**                | 动态base调整  | ~16x     | 较少     | 外推能力强  | 调参复杂  |
| **YaRN**                        | 多项式插值+NTK | **~32x** | **最少** | **最优** | 实现略复杂 |

### 4.2 外推性能曲线

```
性能 vs 扩展长度
100% |           PI
     |          /   \
 80% |         /     \      Dynamic RoPE
     |        /       \     /\
 60% |       /         \   /  \
     |      /           \ /    \
 40% |     /             X      \_____ YaRN
     |    /             / \
 20% |   /_____________/   \___________
     |  /
  0% +------------------------------------
       2K   4K   8K   16K   32K   64K   128K
```

### 4.3 注意力权重分析

**原始RoPE在长上下文问题：**
```
当m, n超过训练长度时：
Attention(q, k) = q·k^T / √d

问题：
- 相对位置信息失效
- 高频分量衰减导致注意力"塌陷"
```

**YaRN的解决方案：**
```
1. 位置映射：m → m' = m × α (α < 1)
2. 频率缩放：θ → θ' = θ × β (β > 1)
3. 保持相对距离的线性关系：
   |m' - n'| ∝ |m - n|
```

---

## 五、实验数据与性能评估

### 5.1 Passkey Retrieval任务

| 模型 | 方法 | Context Length | Accuracy |
|------|------|---------------|----------|
| LLaMA-7B | 原始RoPE | 2K | 100% |
| LLaMA-7B | 原始RoPE | 4K | 0% |
| LLaMA-7B | PI | 8K | 95% |
| LLaMA-7B | Dynamic RoPE | 16K | 98% |
| LLaMA-7B | **YaRN** | **32K** | **99%** |
| LLaMA-7B | **YaRN** | **128K** | **97%** |

### 5.2 Language Modeling Perplexity

| Context Length | PI | Dynamic RoPE | YaRN | Full Retrain |
|---------------|----|-------------|------|--------------|
| 2K (baseline) | 3.45 | 3.45 | 3.45 | 3.42 |
| 4K | 3.52 | 3.48 | **3.46** | 3.44 |
| 8K | 3.78 | 3.55 | **3.49** | 3.47 |
| 16K | 4.12 | 3.68 | **3.54** | 3.51 |
| 32K | 5.67 | 3.92 | **3.62** | 3.58 |

### 5.3 训练成本对比

```
相对训练成本（以PI为基准）：

YaRN:    ████ 10%
Dynamic: ████████████ 40%
PI:      ██████████████████████████████ 100%
Full Retrain: ████████████████████████████████████████████████████████ 250%
```

---

## 六、实际应用建议

### 6.1 方法选择指南

```python
def select_rope_method(target_length, base_length, training_budget):
    if target_length <= 2 * base_length:
        return "Positional Interpolation"  # 简单场景
    elif target_length <= 8 * base_length:
        return "Dynamic RoPE"  # 平衡性能
    else:
        return "YaRN"  # 超长上下文
```

### 6.2 实际调参示例

**对于LLaMA-2-7B扩展到32K上下文：**

```python
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 应用YaRN配置
config = model.config
config.rope_theta = 10000  # 原始base
config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,  # 扩展因子 32K / 4K
    "beta": 1.5,     # YaRN beta参数
    "original_max_position_embeddings": 4096
}

# 微调（仅需少量步骤）
# 训练约400步，使用约500M tokens即可
```

参考链接：https://github.com/jquesnelle/yarn

### 6.3 推理时动态调整

```python
class AdaptiveRoPEInference:
    def __init__(self, base_length=4096):
        self.base_length = base_length
        self.current_length = base_length
        
    def adjust_for_context(self, input_length):
        if input_length <= self.base_length:
            return {"scaling": "none", "beta": 1.0}
        elif input_length <= 2 * self.base_length:
            return {"scaling": "linear", "beta": 1.2}
        else:
            # 使用YaRN动态调整
            beta = 1 + (input_length / self.base_length - 1) * (d / (d - 2))
            return {"scaling": "yarn", "beta": beta}
```

---

## 七、前沿研究方向

### 7.1 混合频率策略

```python
# 不同维度使用不同缩放策略
def hybrid_scaling(theta, dim, position):
    # 低频维度：使用较大的扩展倍数
    low_freq_idx = dim < dim // 4
    theta[low_freq_idx] *= beta_large
    
    # 中频维度：使用中等扩展
    mid_freq_idx = (dim // 4 <= dim) & (dim < 3 * dim // 4)
    theta[mid_freq_idx] *= beta_medium
    
    # 高频维度：保持原样或轻微扩展
    theta[~mid_freq_idx & ~low_freq_idx] *= beta_small
```

### 7.2 自适应RoPE

研究在推理时根据输入特征动态调整RoPE参数：

```python
class AdaptiveRoPE:
    def forward(self, x):
        # 根据输入内容动态计算最佳beta
        content_aware_beta = self.predict_beta(x)
        return self.apply_rope(x, beta=content_aware_beta)
```

### 7.3 多尺度的RoPE

结合多个不同base的RoPE，捕获不同尺度的位置信息：

```python
class MultiScaleRoPE:
    def __init__(self, bases=[1000, 10000, 100000]):
        self.bases = bases
        
    def forward(self, x):
        # 计算多个尺度的RoPE
        embeddings = [self.compute_rope(x, base) for base in self.bases]
        # 加权融合
        return self.fuse_embeddings(embeddings)
```

---

## 八、总结

### 核心要点总结：

1. **Dynamic RoPE**：通过动态调整base值，实现有效的外推，特别适合需要平滑扩展的场景

2. **YaRN**：当前最先进的方法，结合了多项式插值和NTK-aware scaling，在训练效率和性能上达到最优平衡

3. **选择策略**：
   - < 2x扩展：PI（简单有效）
   - 2x-8x扩展：Dynamic RoPE（性能好）
   - > 8x扩展：YaRN（最佳选择）

4. **实践建议**：
   - 优先考虑YaRN，其训练成本仅为传统方法的10%
   - 微调时使用渐进式增加序列长度的策略
   - 推理时可动态调整beta参数以适应不同长度

参考资源：
- 原始RoPE论文：https://arxiv.org/abs/2104.09864
- YaRN论文：https://arxiv.org/abs/2309.00071
- Scaling Laws论文：https://arxiv.org/abs/2310.05209
- YaRN实现：https://github.com/jquesnelle/yarn
- PI论文：https://arxiv.org/abs/2306.15595

## 各模型RoPE类型对比

### 1. **Qwen 3 (通义千问)**
- **RoPE类型**: 标准RoPE (不使用RoPE Scaling)
- **rope_theta**: 1,000,000 (非常大的基频)
- **特点**:
  - 通过超大的base_theta实现长上下文能力
  - 不使用特殊的长上下文RoPE scaling技术
  - 滑动窗口(sliding window)辅助长文本处理
- **上下文长度**: 32K (支持通过滑动窗口扩展到131K)

### 2. **DeepSeek (DeepSeek-V2/V3, R1)**
- **RoPE类型**: **YARN (Yet another RoPE extension)** 
- **rope_theta**: 10,000
- **YARN Scaling配置**:
  ```json
  "rope_scaling": {
    "type": "yarn",
    "factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 4096
  }
  ```
- **特点**:
  - 使用YARN扩展技术支持超长上下文(163K)
  - 专门的部分头维度用于RoPE(qk_rope_head_dim: 64)
  - 分离的nope头维度用于其他信息(qk_nope_head_dim: 128)

### 3. **GLM-4 (智谱AI)**
- **RoPE类型**: **原始RoPE (Original RoPE)**
- **配置参数**:
  ```json
  "original_rope": true
  ```
- **特点**:
  - 使用标准RoPE实现，无特殊扩展
  - 通过其他机制(如SwiGLM)优化长文本处理
  - 上下文长度: 8K-32K

### 4. **MiniMax**
- **RoPE类型**: *配置信息未公开*
- **状态**:
  - HuggingFace上有MiniMax-M2.1等模型发布
  - 技术细节未完全开源
  - 可能使用私有或改进的RoPE方案
  - 具体实现方式不详

## RoPE类型对比总结

| 模型 | RoPE类型 | theta值 | 长上下文技术 | 最大长度 |
|------|----------|---------|--------------|----------|
| **Qwen 3** | 标准RoPE | 1,000,000 | 滑动窗口 | 32K-131K |
| **DeepSeek** | **YARN** | 10,000 | YARN scaling | 163K |
| **GLM-4** | 原始RoPE | - | SwiGLM | 8K-32K |
| **MiniMax** | 未公开 | - | 未公开 | 未公开 |

## 关键技术说明

### YARN RoPE (DeepSeek使用)
YARN是DeepSeek团队采用的高级RoPE扩展方法，特点包括：
- **多尺度缩放**: 结合快速和慢速两种beta值
- **原始保留**: 保持原始最大长度的参考点
- **非线性插值**: 改进的频率位置编码插值方法

### 大RoPE Theta (Qwen使用)
Qwen采用了极其简单的方案：
- **超大基频**: 1,000,000使远距离token的频率变化更缓慢
- **无需额外scaling**: 直接通过theta实现长距离依赖
- **计算效率高**: 无需复杂的插值计算

### MiniMax的局限性
由于MiniMax的技术细节未完全公开，无法确定其使用的具体RoPE实现方式。这可能是因为：
- 商业机密考虑
- 采用私有改进方案
- 或使用标准RoPE但未公开细节