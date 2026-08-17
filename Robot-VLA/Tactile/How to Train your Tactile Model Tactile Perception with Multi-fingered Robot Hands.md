---
source_pdf: How to Train your Tactile Model Tactile Perception with Multi-fingered
  Robot Hands.pdf
paper_sha256: 4fa038e170e0db1a379e367679ea7281280e1048d2901658796b685b3449a015
processed_at: '2026-08-05T00:08:02-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 在干嘛

## 一句话

五指机器人手上 5 个触觉传感器长得差不多但出图不一样，CNN 每换一个就得重新采 3000 张图重训一遍，他们用 ViT 替代 CNN，做到训完 4 个就能直接用第 5 个，不用重训。

---

## 问题是什么

想象你给机器人做了 5 根手指头，每根指尖装一个 TacTip 指尖传感器。TacTip 长啥样呢？就是一个 3D 打印的小橡胶头，里面有一圈白色小圆点（叫 marker），顶上有个 camera 拍这些 marker。手指一碰东西，橡胶皮变形，marker 跟着动，camera 拍下 marker 位移的图，算法从图里推断"我压下去多深、斜了多少度、受力多大"。

听起来简单，但麻烦在于：5 个传感器都是 3D 打印的，打印这东西不精确。每个的 lens 角度微微歪一点、LED 灯位置偏一点、gel 厚度不均一点、marker 大小略有差异。结果就是——同一个 pose 压下去，5 个 sensor 出来的图长得都不太一样（paper 里 Fig. 2 给了对照，亮度、glare、marker 形态全不一样）。

之前主流做法是用 CNN 学"图 → pose"。CNN 在单 sensor 上学得特别好，MAE 能做到 0.1mm、0.85° 这种精度。但你一旦换个新 sensor，CNN 立刻崩掉——pose 误差从 0.85° 飙到 5.77°，差不多 7 倍退化。原因也好理解：CNN 学的是"这个位置的这块纹理 = 这个 pose"，本质是局部模板匹配。新 sensor 的纹理位置、亮度全偏了，模板全废。

工程上这意味着啥？你换一个指尖就得重新采 3000 个数据点（UR5 机械臂一个个 pose 压过去，一个传感器采一下午），然后重训模型。五指手 5 个指尖还好，要是未来做 20 指手、或者传感器磨损了要换，这就完全 scalable 不起来。

---

## 他们的招：用 ViT

ViT（Vision Transformer）跟 CNN 最大的区别是它不靠卷积扫局部，而是把图切成一堆 patch，扔进 transformer 让任意 patch 之间互相 attend。它学到的是"patch 和 patch 之间什么关系"，而不是"这个局部位置长啥样"。

放到 TacTip 这个场景，这个区别恰好特别关键。

TacTip 的图里，真正有信息的信号是：**所有 marker 作为一个群体怎么一起动**。

- 压深了：marker 整体往里收
- 往左斜了：左边 marker 压得更深、右边翘起来
- 往右斜了：反过来

也就是说，单个 marker 在哪个像素位置不重要，**marker 之间的相对位移关系**才重要。CNN 会把"marker A 在像素 (120, 80)"也当成特征学进去，sensor 一变就废；ViT 的 self-attention 天然就是建模"任意 marker 对任意 marker 的关系"，对绝对位置不敏感，sensor 换了 marker 摆位偏一点，相对关系还是稳定的。

他们用的具体配置：
- 拿 Google 的 `vit-base-patch16-224`（ImageNet 预训练好的 ViT-base）当 backbone
- 224×224 输入，切成 196 个 16×16 patch
- 12 层 transformer，12 个 attention head，embedding 维度 768
- 后面接 4 层 FC 做回归头，输出 6 个数：z（深度）、Rx、Ry（倾角）、Fx、Fy、Fz（力）
- 微调时用 LoRA，不更新原 ViT 权重，只训练低秩矩阵 $W_A W_B$ 和回归头，参数量小、不容易把预训练表征打坏
- 先冻住 transformer 只训回归头几个 epoch，再逐步解冻，防止 catastrophic forgetting

输出 6 维 pose+force，loss 是统一的 MSE。15000 张图（5 sensor × 3000），80/20 split。

---

## 实验怎么设计的

他们设了三档难度，把"泛化"这个含糊的概念拆成可测量的东西：

| 实验 | 训练 | 测试 | 模拟场景 |
|---|---|---|---|
| Tr1-Te1 | 1 个 sensor | 同一个 | 单指 baseline |
| Tr5-Te1 | 5 个全上 | 训练里见过的某一个 | 多指已部署 |
| Tr4-TeU | 4 个 | 第 5 个，没见过 | 换新 sensor |

Tr4-TeU 用 5-fold 交叉验证——轮流留一个当 unseen，5 个都轮一遍取平均。这是评估泛化的标准做法。

---

## 结果怎么样

直接看 Table IV 关键数字：

| | z (mm) | Rx (°) | Ry (°) | Fx (N) | Fy (N) | Fz (N) |
|---|---|---|---|---|---|---|
| **Tr1-Te1 CNN** | 0.101 | 0.855 | 0.850 | 0.126 | 0.124 | 0.383 |
| **Tr1-Te1 TacViT** | 0.065 | 1.032 | 0.895 | 0.115 | 0.116 | 0.285 |
| **Tr4-TeU CNN** | 0.212 | 5.768 | 4.698 | 0.392 | 0.426 | 0.822 |
| **Tr4-TeU TacViT** | 0.080 | 2.714 | 1.833 | 0.185 | 0.165 | 0.434 |

读法：

1. **单 sensor 上 CNN 略赢**：Tr1-Te1 上 CNN 的 Rx/Ry 是 0.85°，TacViT 是 1.03°，CNN 小胜。意料之中——ViT 没啥 inductive bias，小数据上 prior 弱，单 domain 上拼不过 CNN 的局部特化能力。

2. **5 个一起训，CNN 还行**：Tr5-Te1 上 CNN 没崩，甚至 z 还更准了。说明 5 个 sensor 的差异还在 CNN 的拟合能力范围内。

3. **unseen sensor 上 CNN 彻底崩，TacViT 稳**：这就是论文的高光时刻。CNN 的 Rx 从 0.85° 飙到 5.77°，差不多 7 倍退化；TacViT 从 1.03° 升到 2.71°，只退化 2.6 倍。z 方向更夸张——TacViT 在 **unseen sensor** 上的误差 0.08mm，比 CNN 在 **seen sensor** 上的 0.10mm 还小。完全不是一个量级的鲁棒性。

4. **z 比 Rx/Ry 更好迁移**：为啥？z 是"所有 marker 整体收一下"这种全局协同信号，跨 sensor 几乎完全不变。Rx/Ry 是"左右两边 marker 差多少"这种局部对比信号，更容易被 sensor 的 lens distortion、glare 干扰。所以 z 几乎无损迁移，Rx/Ry 退化明显但仍远好于 CNN。

---

## 为什么 ViT 在这个奇怪的场景能 work

直觉上挺反直觉的——TacTip 图里全是小圆点，跟 ImageNet 的猫狗汽车完全不像，ImageNet 预训练的 ViT 凭啥有用？

拆开看：
- TacTip 图里的 marker 本质是**高对比度的小 blob**
- ViT 在 ImageNet 上学的底层视觉 primitive（边缘检测、blob 中心定位、对比度梯度）对 blob 位移检测直接有用
- 全局 self-attention 让模型把"marker 群体的位移场"当成一个整体来理解，而不是逐像素模板匹配
- LoRA 微调只动低秩子空间，保住了预训练的通用视觉 prior，不让 ViT 在小数据上把底层表征打坏

所以他们其实没用啥 tactile 领域的特化设计，纯粹是"通用视觉 backbone + 小心微调 + 合适的架构归纳偏置"三件事凑齐了。

---

## 工程上意味着啥

之前的 workflow：
1. 3D 打印一个新 TacTip
2. 装到 UR5 上，采 3000 个 pose 数据（半天到一天）
3. 训 CNN（几小时）
4. 部署

之后如果有 TacViT：
1. 3D 打印一个新 TacTip
2. **直接用**，不用采数据不用训

对多指手特别值——5 个指尖本来就要 5 套数据 5 套模型，现在一套模型通吃，再加一个指尖也直接能用。对 sensor wear-and-tear 场景也值——传感器磨损了换一个，模型不动。

---

## 这篇 paper 的弱点

人话讲也得说清楚弱点，不然就是软文了：

1. **CNN baseline 太弱**。他们比的是 4 conv + 2 FC 的玩具 CNN，没拿 ResNet18 pretrained、ConvNeXt-Tiny 这种 modern CNN 来比。ConvNeXt 论文已经证明纯 CNN 加上 ViT 风格的训练 recipe 也能很强。所以"CNN 不行 ViT 行"这个结论下得有点早——可能只是"弱 CNN 不行，pretrained ViT 行"。

2. **ViT 用了 ImageNet 预训练，CNN 没有**。公平性打折。真正干净的对照是 CNN 也用 ImageNet pretrain 再 fine-tune。

3. **Loss 设计有问题**。6 维输出共用一个 MSE，force 量级（N，0-10）天然比 pose（mm，°）大好几倍，梯度被 force 主导。这可能就是 TacViT 在 Tr5-Te1 上 Rx/Ry 比 CNN 差的原因——loss 没加权，pose 信号被稀释。

4. **没做消融**。LoRA vs 全量微调？patch size 16 适不适合 marker 大小？从零训 vs ImageNet pretrain？这些都没测，只是拍脑袋选了 ViT-base。

5. **只测了 TacTip**。GelSight、DIGIT、GelSlim 这些主流 VBTS 没碰。GelSight 是连续弹性体表面，不是离散 marker array，ViT 是不是还占优不知道。

6. **统计不严谨**。5 个 sensor 这么小样本，只给 mean 和 strip plot，没 confidence interval 没 paired t-test，差异是不是显著存疑。

---

## 我看完之后的 intuition

这篇 paper 给我最大的 takeaway 是个挺通用的事：**当你的信号本质是"一组元素的相对关系"而不是"特定位置的局部纹理"时，CNN 的 locality prior 是个负债，ViT 的 global attention 是个资产。**

TacTip 恰好是这个类型的 sensor——marker 是离散的、unlabeled 的、位置不严格固定的，你要的是"这群 marker 怎么一起动"，不是"这个像素这块花纹长啥样"。所以 ViT 完美匹配。

换到 GelSight 那种连续表面形变的 sensor，是不是 ViT 还比 CNN 强？不一定，因为信号分布不一样了。这种"什么归纳偏置匹配什么 sensor"的问题，比单纯说"ViT 比 CNN 强"更有意思，也是这篇 paper 没碰但值得做的下一步。

再往远了想——如果有人攒一个 1000 万张的 tactile 大数据集，把所有 TacTip/GelSight/DIGIT 的图都聚起来从零训一个 ViT，做成 tactile 版的 foundation model，那才是这个方向真正 solve 问题的样子。这篇 paper 只是在这条路上往前迈了一小步，证明了"用 transformer 跨 sensor 泛化"这个方向可行，离终点还远。

---

参考链接：
- TacViT paper: https://arxiv.org/abs/2410.02779
- ViT 原文: https://arxiv.org/abs/2010.11929
- LoRA: https://arxiv.org/abs/2106.09685
- TacTip 综述: https://ieeexplore.ieee.org/document/9506776
- ConvNeXt（CNN 反击 ViT）: https://arxiv.org/abs/2201.03545
- vit-base-patch16-224 模型: https://huggingface.co/google/vit-base-patch16-224

---

# TacViT：用 Vision Transformer 替代 CNN 解决多指手触觉传感器的跨传感器泛化问题

这篇 paper 的核心 insight 一句话就能说清楚：在 vision-based tactile sensors (VBTS) 上做接触 pose/force 回归时，CNN 强 locality prior 反而成了跨传感器部署的负担，而 ViT 的全局 self-attention 学到的是「marker 群体位移模式」这种 domain-invariant 信号，因此可以在完全未见过的新传感器上零重训泛化。下面我按动机 → 方法 → 数据 → intuition → 批判这条线展开。

---

## I. 问题动机：为什么 CNN 在这个场景下"恰好"失败

**多指手的部署瓶颈。** 一个五指手有 5 个 TacTip-style 指尖传感器，3D 打印工艺会引入 lens 曲率、illumination LED 摆位、gel 厚度、marker 分布、wear pattern 的微小差异。Fig. 2 直观展示了同一设计的 5 个传感器出图差异非常明显（亮度、glare、marker 大小都不同）。

**CNN 失败的本质。** CNN 的 inductive bias 是 *locality + translation invariance + hierarchical receptive field*。它学到的特征本质上是「这个局部纹理块 → 这个 pose 值」的模板匹配。当新传感器的 glare 模式、marker 形态偏移后，这些 local template 全部失效，性能从 ~0.85° 飙到 ~5.77°（Table IV 的 Tr4-TeU）。

**ViT 的优势反直觉。** ViT 几乎没有 spatial inductive bias，靠 ImageNet 大规模预训练学通用视觉 prior，再靠 self-attention 让任意 patch 都能 attend 到任意 patch。这意味着它不会被「特定传感器的局部模板」绑死，而能学到 *marker 之间相对位移的全局统计规律* —— 这个规律跨传感器是稳定的。

参考文献：
- ViT 原论文：https://arxiv.org/abs/2010.11929
- TacTip 综述：https://ieeexplore.ieee.org/document/9506776
- 之前 CNN-based TacTip 工作：https://arxiv.org/abs/2003.01860

---

## II. 方法架构详解

### A. 整体 pipeline（Fig. 3）

```
TacTip image (224×224)
      ↓  切成 196 个 16×16 patch
Linear patch embedding  E = W_p · X_p + b_p   (dim=768)
      ↓  + positional embedding
Transformer Encoder × 12 层
   每层: Multi-Head Self-Attention (12 heads) → MLP (768→3072→768)
      ↓  取 [CLS] token 输出
Regression Head (4 层 FC + ReLU)
      ↓
ŷ = (z, R_x, R_y, F_x, F_y, F_z)   6 个回归输出
```

### B. 关键公式与变量含义

**1. Patch embedding**

$$E = W_p X_p + b_p$$

- $X_p \in \mathbb{R}^{P \times P \times 3}$：单个 patch，$P=16$，所以 $X_p$ 是 $16 \times 16 \times 3$ 的 RGB 小图块
- $W_p \in \mathbb{R}^{768 \times (P^2 \cdot 3)}$：learnable linear projection，把 $16\cdot16\cdot3 = 768$ 维像素向量投影到 embedding 维度 $d=768$（这里数字撞上了是巧合，ViT 默认就是 768）
- $b_p \in \mathbb{R}^{768}$：bias
- 224/16 = 14，14×14 = 196 个 patch（Table I 中 "Number of Patches = 196" 对得上）

**2. Self-attention（单层）**

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- $Q = X W_Q, K = X W_K, V = X W_V$，$d_k = 768/12 = 64$
- $\sqrt{d_k}$ 是 scaled factor 防止 dot product 过大导致 softmax 饱和
- 12 heads 各自做 64 维 attention，concat 后再投影

**3. LoRA fine-tuning**

$$W_{\text{new}} = W_{\text{frozen}} + \Delta W, \quad \Delta W = W_A W_B$$

- $W_A \in \mathbb{R}^{d \times r}, W_B \in \mathbb{R}^{r \times d}$，$r \ll d$（论文里没给具体 r，估计是 4 或 8，参考原 LoRA 论文默认）
- $W_{\text{frozen}}$ 是 ImageNet 预训练权重，冻结不动
- 只有 $W_A, W_B$ 和 regression head 是可训练的，参数量从 ViT-base 的 ~86M 降到 ~1M 量级
- 直觉：把"任务相关的微调方向"约束在低秩子空间，避免在小数据（15000 张）上把预训练表征打崩

参考：LoRA 原文 https://arxiv.org/abs/2106.09685

**4. Loss**

$$L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

- $N$：batch size（论文 batch=128？Table I 没写清，CNN 是 16）
- $\hat{y}_i$：6 维预测向量
- $y_i$：6 维 ground truth，由 UR5 末端执行器位姿 + 力传感器读数获得

注意输出是 6-DoF 的混合：3 个 pose（$z, R_x, R_y$，注意 $x, y$ 是 shear 不预测）+ 3 个 force（$F_x, F_y, F_z$）。**所有维度共用一个 MSE**，没有加权，这意味着 force 量级（N，0~10）会主导 loss 相对 pose（mm，0~4；°，0~20）。这是一个潜在弱点，下面会展开。

### C. 数据采集 protocol

UR5 机械臂 + 平面接触目标，每个传感器采 3000 个样本，5 个传感器共 15000。位姿范围：

- $x, y \in [-2, 2]$ mm （shear 切向位移，作为扰动而非标签）
- $z \in [0, 4]$ mm （法向压入深度，是回归 label）
- $R_x, R_y \in [-20°, 20°]$ （倾斜角，是回归 label）
- $F_x, F_y \in [-3, 3]$ N, $F_z \in [0, 10]$ N （力 label，由位姿派生）

关键设计：**故意 shear** —— 把传感器压上表面后再切向位移一段 $(x,y)$，让模型学会忽略这种切向扰动，提高后续 tactile servoing 的鲁棒性。这是一个很强的 task prior。

参考数据采集原方案：https://journals.sagepub.com/doi/10.1177/02783649241240206

### D. 训练细节

- 硬件：单卡 RTX 3070 + CUDA 11.8 + PyTorch 2.6
- **渐进式 unfreezing**：先只训 regression head，几个 epoch 后再解冻部分 transformer layer
- 目的：防止 catastrophic forgetting 把 ImageNet 表征打坏
- 80/20 train/val split，全数据集 15000 → 12000 训练 + 3000 验证

---

## III. 实验设计：三档难度梯度

这是论文最精巧的部分，把"泛化"这个含糊概念拆成了三个可测量的等级：

| Experiment | 训练数据 | 测试数据 | 难度 | 模拟场景 |
|---|---|---|---|---|
| **Tr1-Te1** | 1 sensor | 同一 sensor | 易 | 单指 deployment baseline |
| **Tr5-Te1** | 5 sensors | 训练中见过的某 1 sensor | 中 | 多指已部署，在线推理 |
| **Tr4-TeU** | 4 sensors | 未训练过的第 5 sensor | 难 | 新增 / 更换传感器 |

Tr4-TeU 用 5-fold 交叉验证（轮流留一个 sensor 当 unseen），这是评估泛化的金标准设计。

---

## IV. 结果数据深度解读

### A. Table IV 总览（mean MAE）

| 设置 | 模型 | z (mm) | Rx (°) | Ry (°) | Fx (N) | Fy (N) | Fz (N) |
|---|---|---|---|---|---|---|---|
| Tr1-Te1 | CNN | **0.101** | **0.855** | **0.850** | **0.126** | **0.124** | **0.383** |
|  | TacViT | 0.065 | 1.032 | 0.895 | 0.115 | 0.116 | 0.285 |
| Tr5-Te1 | CNN | **0.082** | 0.964 | 1.112 | **0.133** | 0.130 | 0.347 |
|  | TacViT | 0.063 | 1.651 | 1.386 | 0.121 | 0.107 | 0.334 |
| Tr4-TeU | CNN | 0.212 | 5.768 | 4.698 | 0.392 | 0.426 | 0.822 |
|  | TacViT | **0.080** | **2.714** | **1.833** | **0.185** | **0.165** | **0.434** |

### B. 三个关键观察

**1. Tr1-Te1（in-domain）CNN 小赢。** CNN 在 single-sensor 上 MAE 略低于 TacViT，尤其 Rx/Ry（0.85° vs 1.03°）。这符合 ViT 在小数据上"prior 弱"的预期。论文引用的 acceptable threshold 是 $z < 0.1$mm, $R < 2.5°$, $F < 1$N，所以两者都过关。

**2. Tr5-Te1（multi-sensor seen）CNN 反而最好。** 注意 CNN 的 z MAE 从 0.101 降到 0.082，Rx 从 0.855 升到 0.964，但 Rx/Ry 整体还是低于 TacViT 的 1.65°/1.39°。这暗示 **多传感器训练对 CNN 反而是轻微的 distribution 扩展，但还没超出 CNN 的拟合能力**。TacViT 这里反而比 Tr1-Te1 更差，可能是 5 个传感器的数据让它更倾向学通用特征，丢失了对单 sensor 的特化精度。

**3. Tr4-TeU（unseen sensor）ViT 完胜。** 这是论文核心 claim 的支撑：
- CNN Rx MAE 从 0.85° → 5.77°，**约 6.8× 退化**
- TacViT Rx MAE 从 1.03° → 2.71°，**约 2.6× 退化**
- CNN z MAE 从 0.10 → 0.21mm，**约 2.1× 退化**
- TacViT z MAE 从 0.065 → 0.080mm，**仅 1.23× 退化**，几乎无损

z 方向 TacViT 的鲁棒性极其漂亮 —— 在 unseen sensor 上比 CNN 在 seen sensor 上还准 2.5 倍。

### C. 单 sensor 维度看（Table III Tr4-TeU 行）

注意 sensor #1 和 #5 是 CNN 的灾难点（Rx 8.93° 和 8.59°），但 TacViT 在 #1 上 Rx 也有 6.92° 的相对较差值。这说明**不是所有 unseen sensor 都同样难**，#1 可能存在制造上更显著的偏移（Fig. 2 也佐证 #1 偏暗）。但 TacViT 整体 spread 小（Fig. 4 strip plot 显示 TacViT 点更聚集），mean 和 variance 都占优。

---

## V. Build Your Intuition：为什么 ViT 在这个任务上对

### 1. TacTip 图像的本质信号

TacTip 出图是一组 marker（白色小圆点钉在黑色橡胶皮肤内侧），接触皮肤变形时 marker 会发生 *径向位移*。给定一个 contact pose (z, Rx, Ry)，对应的 marker 位移场是：

- 法向压入 z：所有 marker 整体向中心聚拢（或向外扩张，取决于接触几何）
- 倾斜 Rx：marker 沿某轴产生梯度位移（一侧被压深，另一侧抬起）
- 倾斜 Ry：类似但方向正交

**关键观察**：这些信号是 *全局空间相关性* —— 一个 marker 的位移只有和周围 marker 的位移对比才有意义。CNN 通过 stacked convolution 也能学到这种关系，但它会把"marker A 应该出现在像素 (120, 80)"这种 *绝对位置* 也学进去，于是新传感器 marker 摆位微变就崩。ViT 的 self-attention 天生建模 *任意 marker 对任意 marker 的相对关系*，且 patch + positional embedding 的设计让"绝对位置"信号比 CNN 弱得多。

### 2. 为什么 z 泛化得最好，Rx/Ry 略差

z 是一个**全局协同信号**：所有 marker 同步响应，attention 只要能聚合"整体亮度模式 / marker 整体收缩程度"就够了，跨传感器几乎完全可迁移。

Rx/Ry 是**局部差异信号**：需要识别"哪一边的 marker 位移更大"，依赖更精细的局部几何对比，更容易被传感器特定的 lens distortion / glare pattern 干扰。这解释了 Table IV 中 TacViT 在 Tr4-TeU 上 z 退化 1.23× 而 Rx/Ry 退化 2.6× 的差异。

### 3. LoRA 在这里的双重作用

不只是参数效率。LoRA 的 $\Delta W = W_A W_B$ 是一个 *低秩扰动*，物理上等价于"在预训练表征空间里沿少量方向做 task-specific 旋转"。Tactile 图像和 ImageNet 自然图像分布差异巨大，全量微调会把底层 attention pattern 全部重写，丢失 generalizable structure；LoRA 强制保留预训练的"通用视觉 grouping"能力，只让任务相关的 head 和少部分方向适应 tactile 域。这与论文中"渐进式 unfreezing"策略目的一致。

参考：LoRA 表达能力分析 https://arxiv.org/abs/2310.17513

### 4. 为什么 ImageNet 预训练对 tactile 图像竟然有效

这看似反直觉 —— tactile 图像是 marker array，跟猫狗汽车完全不像。但 ViT 学到的 *底层视觉 primitive*（边缘、blob 检测、对称性、对比度梯度）对 marker displacement field 同样有用：marker 本质是高对比度 blob，位移检测就是 blob center localization 任务。所以 ImageNet 预训练 + 小数据 fine-tune 在这里 work 得很好。这也呼应论文 discussion 中"如果有个 large-scale tactile pre-training dataset 会更好"的判断 —— 类似 LLM 时代的 domain-adapted pretraining。

---

## VI. 批判性思考 / 改进方向

### A. 方法学上的弱点

1. **Loss 加权缺失**：6 维输出共用 MSE，force 量级（N，0~10）天然比 pose（mm，°）数值大 ~5-10 倍，会主导梯度。应该按维度标准化或加权。这可能是 TacViT 在 Tr5-Te1 上 Rx/Ry 比 CNN 差的部分原因 —— loss 被低优先级的 force 项稀释了。

2. **没报告 LoRA rank r**：Table I 给了架构但漏了 LoRA 配置，无法复现。

3. **没做消融**：
   - LoRA vs full fine-tuning
   - ImageNet 预训练 vs 从零训练
   - Regression head 深度（4 层 FC 怎么定的）
   - Patch size（16 在 224×224 上对应 14×14 patch，对 marker 尺寸是否合适？marker 直径可能 5-10 像素，patch 16 会把多个 marker 切进一个 patch）

4. **CNN baseline 不够强**：Table II 的 CNN 是 4 conv + 2 FC 的玩具结构，没有跟 ResNet18 / ConvNeXt-Tiny 等 modern CNN 对比。ConvNeXt 论文 (https://arxiv.org/abs/2201.03545) 显示纯 CNN 加上 ViT 风格的训练 recipe 也能达到很强性能，这里的 CNN 失败可能是 baseline 选择问题，不是 CNN 架构本身限制。

5. **只测 TacTip**：GelSight、DIGIT、GelSlim 等其他 VBTS 没验证。GelSight 图像是连续弹性体表面形变（不是离散 marker），ViT 是否还占优未知。

6. **统计报告不完整**：只给 mean 和 strip plot，没有 confidence interval、paired t-test。在 5 个 sensor 的小样本上，差异是否统计显著存疑。

### B. Conceptual gap

- **"Global attention = better generalization" 是 sufficient 还是 necessary？** 论文 claim ViT 比 CNN 强是 *因为* 全局 attention，但没做实验隔离 self-attention 的贡献。可能 *预训练规模* 和 *架构* 是耦合的 —— ViT 用了 ImageNet 预训练，CNN baseline 没用，公平性存疑。一个更干净的对照是：CNN 也用 ImageNet 预训练（比如 ResNet18 pretrained）再 fine-tune。

- **Marker tracking 替代方案**：传统方法（光流、marker correspondence）天然跨传感器，因为它们显式建模 marker 而非像素。深度学习方案是否真的胜过 classical marker-tracking + 几何拟合？这论文没比。

### C. 延伸方向

- **Tactile Foundation Model**：把 1000 个 TacTip / GelSight / DIGIT 传感器数据聚成 10M 级别 dataset，从零训练一个 ViT，做成 tactile 版 CLIP。这能彻底解决跨传感器问题。
- **Cross-modal pre-training**：用 visuotactile paired data 训练，让 tactile encoder 和 visual encoder 对齐 embedding。
- **Sensor ID conditioning**：给 ViT 加一个 sensor embedding（类似 positional），train 时见过，inference 时新 sensor 用 zero-shot 或 few-shot adaptation。这是另一种泛化路径。

参考跨模态方向：
- AnyRotate sim-to-real touch：https://arxiv.org/abs/2405.07391
- PACT perception-action causal transformer：https://arxiv.org/abs/2303.01548

---

## VII. 一句话总结 intuition

TacViT 用 ViT 的 global self-attention 把 TacTip 图像中的 marker array 当成一组 unordered tokens 来建模，学到的不是"这个像素 pattern → 这个 pose"的 sensor-specific 模板，而是"marker 群体位移关系 → 这个 pose"的 sensor-agnostic 几何规律。再加上 LoRA 的小心 fine-tuning 保护了 ImageNet 预训练的通用视觉 prior，使得在 unseen sensor 上仍能从位移场推断出接触参数。代价是在 in-domain single-sensor 上比 CNN 略差 0.2°，换得跨传感器 deployment 时不重新采数据 3000 个的工程红利。

参考论文与开源资源：
- TacViT 论文（ICRA workshop / arXiv）https://arxiv.org/abs/2410.02779
- ViT 原文 https://arxiv.org/abs/2010.11929
- LoRA https://arxiv.org/abs/2106.09685
- vit-base-patch16-224 HuggingFace 模型卡 https://huggingface.co/google/vit-base-patch16-224
- TacTip 项目主页 https://www.brl.ac.uk/researchthemes/tactile-robotics.aspx
- Lloyd & Lepora tactile servoing https://journals.sagepub.com/doi/10.1177/02783649241240206
- ConvNeXt（CNN 对比 baseline 缺失参考）https://arxiv.org/abs/2201.03545
- MonoViT（ViT 用在深度估计的相邻工作）https://arxiv.org/abs/2208.00050
