---
source_pdf: lola.pdf
paper_sha256: e14acedb737b60bc12b4a19a923609e456b3400539a9eb0c26964ee83cf4c52b
processed_at: '2026-08-05T15:48:03-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LoLA 人话版

好，Andrej，我换个画风，用白话讲讲这篇 paper 到底在干啥，那些公式背后真正想说的人话是什么。

---

## 这篇 paper 到底想解决啥

一句话：**让 robot 能干那种要折腾好几分钟的任务**，比如做披萨——拿锅、放烤箱、关烤箱门、转旋钮、出炉、切、刷酱、撒料。整个流程下来 2-5 分钟，中间几百个 action step，任何一步歪一点，后面全崩。

现有 VLA model 像 RT-2 [https://arxiv.org/abs/2307.15818]、OpenVLA [https://arxiv.org/abs/2406.09246] 甚至 π0 [https://arxiv.org/abs/2410.24164] 基本都是 **看一张图、出一帧 action** 的模式。这就好比你让一个人蒙着眼开车，每秒睁开看一眼路就闭上，开 5 分钟——你大概会撞墙。

LoLA 想做的事很直白：**让 robot 能"记得住事"**，并且**知道自己的身体在哪**，从而在长程任务里保持一致。

---

## 三个根本困难，人话版

### 困难 1：Error 会 compound

假设单步 success probability 是 $p$，episode length $T$，总 success 大约 $p^T$。这是乘法关系。

打个比方：你让 robot 关烤箱门，单步成功率 80%。如果你要它做 10 步，总 success $0.8^{10} \approx 10.7\%$。Paper Table 6 里 Franka 的 multi-step 任务，π0 在 T4→T5→T6 上只有 12.4%，完全符合这个 decay 直觉。

**要做长程，必须每步更准 + 有 history 来纠偏**。

### 困难 2：VLM 的 embedding 和 action 根本不在一个空间

这是 paper 最深的洞察。VLM（比如 Qwen2.5-VL [https://arxiv.org/abs/2502.13923]）在 web 数据上训出来，它的 embedding 知道"这是红色杯子"、"杯子在桌子左边"——这些是**视觉语义空间**里的点。

而 robot 的 action $a_t$ 是什么？是 joint velocity，是 end-effector 的 delta pose，定义在**物理坐标系**里。Robot state $s_t$（joint angles $q_t$ + gripper pose $p_t$）也在这同一个物理空间里，action 就是 state 的增量：

$$s_{t+1} = s_t \oplus a_t \quad (\text{Eq.3})$$

这里 $s_t$ 是当前 state 向量，$a_t$ 是 action 向量，$\oplus$ 是物理 space 的加法（关节空间直接相加，或 SE(3) pose 复合）。

**直觉**：你给 robot 看 VLM 说"红色杯子在左边"，VLM 给出 embedding $e \in \mathbb{R}^{4096}$。然后你让 action head 去生成 $a_t \in \mathbb{R}^{7}$（7 DoF）。这个 $e$ 和 $a_t$ 之间隔着十万八千里。传统方法就是把 $e$ 和 $s_t$ 拼起来丢给下一层，让 attention 自己学会去对齐——这件事很难，因为 attention 没有任何先验知道这两个 embedding 的"对应关系"。

### 困难 3：长 history 太贵

25 帧 history × 4 个视角 × 224×224 图像，光过 vision encoder 就要烧一大堆 token。算 attention 又是 $O(n^2)$。

---

## 三个 trick 的直觉解释

### Trick 1：不对称分辨率

当前帧 $V_t$ 给 224×224 高清，history 全部降到 112×112（Eq.1）：

$$\mathcal{V}_{hme} = \{V_{t-n}^{\downarrow}, \ldots, V_{t-2}^{\downarrow}, V_{t-1}^{\downarrow}, V_t\}$$

上标 $\downarrow$ 就是"下采样"的意思，$V_{t-n}^{\downarrow}$ 是 $t-n$ 时刻的下采样帧，$n$ 是历史窗口长度（论文里 25）。

**人话**：你开车时眼睛要盯前方看清路标，余光扫后视镜知道"后面有车在动"就够了。Action 对当前帧是**position-sensitive**（gripper 对准要 sub-centimeter），对 history 是**velocity/phase-sensitive**（知道"我在刷酱的中段"就行，细节不重要）。所以 history 给低分辨率，token 数减半，$O(n^2)$ attention 成本降一半多。

### Trick 2：SALR——这篇 paper 的灵魂

这部分我重点讲，因为它真的有点意思。

**传统做法**（π0, OpenVLA 等都是）：把 state $s_t$（一串数字，比如 7 个关节角 + gripper width）投影成 embedding，作为**一个 token** concat 到 VLM 输出后面，让下游 attention 自己去混合。

**LoLA 的做法**：让 state 在 VLM 的**每一层、每个 KV channel** 上都参与调制。具体地：

Step 1：state $s_t$ 投影成 embedding，送进一个**并行的 State Transformer**（也是 28 层，跟 VLM 等深）。State Transformer 每层做 self-attention，输出一个 query $Q_r \in \mathbb{R}^{N_s \times H}$。

变量解释：
- $N_s$：State Transformer 的 attention head 数，论文里是 8
- $H$：hidden dim，论文里是 1024（比 VLM 的 3584 小，省钱）

Step 2：在 VLM 的第 $i$ 层，State Transformer 第 $i$ 层的 $Q_r$ 和 VLM 第 $i$ 层的 KV 做外积：

$$K^*[i, j, :] = Q_r[i, :] \odot K_i[j, :] \quad (\text{Eq.4})$$
$$V^*[i, j, :] = Q_r[i, :] \odot V_i[j, :] \quad (\text{Eq.5})$$

- $i$：state head 的索引（1 到 $N_s = 8$）
- $j$：VLM 的 KV head 索引（1 到 $N_v$）
- $\odot$：element-wise 乘法
- $K_i[j, :] \in \mathbb{R}^H$：VLM 第 $i$ 层第 $j$ 个 head 的 Key 向量
- $Q_r[i, :] \in \mathbb{R}^H$：state 第 $i$ 个 head 的 Query

**人话**：这就相当于**给 VLM 每一个 KV channel 学一个 state-dependent scalar gate**。如果某个 visual feature（比如"背景墙是白色的"）在当前 robot state 下对 action 没用，$Q_r$ 对应那个 channel 的维度会训练成接近 0，直接把那个 channel 抑制掉。

这和 FiLM conditioning [https://arxiv.org/abs/1709.07871] 的精神类似，但只有 multiplicative term，没 additive bias。也和 Hypernetworks [https://arxiv.org/abs/1609.09106] 有家族相似——用一个网络（state → $Q_r$）去"调制"另一个网络（VLM）的中间表示。

**为什么这个比 concat 强？** Concat 是 additive 融合，state 和 VL feature 在不同 sub-manifold 独立存在，attention 后面才混合，混合得很慢、很间接。Multiplicative 让 state **直接 gate** 每个 channel，是 dense per-channel modulation，相当于在 representation 层面就做完物理 grounding。

Step 3：再乘一个 learnable mask $M_k, M_v$：

$$K' = K^* \odot M_k, \quad V' = V^* \odot M_v \quad (\text{Eq.6})$$

$M_k, M_v \in \mathbb{R}^{(N_s \cdot N_v) \times H}$ 是模型参数，不依赖输入。作用是**全局抑制 action-irrelevant 噪声**（比如 background distractor）。这部分像一个 learnable sparsity prior。

Step 4：压缩到 $K^a, V^a \in \mathbb{R}^{N_a \times H}$，喂给 action expert 做 cross-attention。

### Trick 3：CFM Action Expert

Action expert 是 28 层 Transformer decoder，用 Conditional Flow Matching [https://arxiv.org/abs/2210.02747] 生成 action chunk。

输入：
1. SALR 出来的 $\{K^a, V^a\}$（cross-attention 的 KV）
2. Noisy action trajectory（self-attention 主输入）
3. Noise timestep $z$（sinusoidal embedding）

训练时预测 noise / vector field，推理时从 $\mathcal{N}(0, I)$ 出发多步 denoise 得到 action chunk $\{a_t, a_{t+1}, \ldots, a_{t+s}\}$，$s$ 是预测步数。

**为什么用 flow matching 不用 AR token？** 这是个分裂：
- AR token（RT-2, OpenVLA）把 action 离散化到 256 bins，丢精度，且 chunk 内 step 之间独立假设不合理
- Flow matching 在连续空间建模 multi-modal distribution，一次性生成整个 chunk，平滑性天然好

π0 走的就是这条路，LoLA 直接继承。

---

## 整体信号流（人话版）

```
看到的东西（25 帧历史 + 4 视角当前帧）
      ↓
语言指令（"把锅放烤箱里"）
      ↓
VLM（Qwen2.5-VL 7B）每层吐出 KV
      ↓                                  ← state 也并行进 State Transformer
                                       ↓
SALR: state 在每一层"摸一遍" VLM 的 KV
      ↓
压缩成 action-conditioned KV
      ↓
Flow Matching Action Expert 生成 10+ 步 action
      ↓
Robot 执行，下一帧再循环
```

---

## 实验数字的关键读法

### Table 1（Google Robot, SIMPLER）

| Method | Avg (Visual Matching) |
|---|---|
| RT-2-X | 46.3% |
| π0 | 52.7% |
| **LoLA** | **61.5%** |

特别在 "Move Near"（需要 spatial reasoning）上 LoLA 71.7% vs π0 35.0%，SALR 的 state-grounding 帮助大。

### Table 2（WidowX, SIMPLER）

| Method | Avg |
|---|---|
| π0 | 41.7% |
| CogACT | 51.3% |
| **LoLA** | **71.9%** |

WidowX 4 个任务里 3 个是 long-horizon（put X on Y / stack A on B）。LoLA 在 "Put Spoon on Towel" 上 95.8% vs π0 62.5%。

### Table 3（LIBERO）

| Method | LIBERO-Long | Avg |
|---|---|---|
| OpenVLA | 53.7% | 76.5% |
| π0 | 85.4% | 92.2% |
| **LoLA** | **88.2%** | **96.2%** |

LIBERO-Long 的 88.2% 是核心信号——这个 suite 是 "A and B" 复合任务，没 history 几乎不可能 maintain coherence。Diffusion Policy 在 LIBERO-Long 上只有 50.5%，对比鲜明。

### Table 6（Franka 多步真机）

| Method | T1→T2→T3 | T4→T5→T6 | T7→T8→T9 |
|---|---|---|---|
| π0 | 17.8% | 12.4% | 16.6% |
| LoLA | **5.9%** | **33.1%** | **28.9%** |

注意 T1→T2→T3 这组 LoLA 反而输给 π0。我推测原因：T1 是 episode 起始，没有有效 history，SALR 的优势发挥不出来；且 T1 单步 success 15.4% 太低，cascade 下来"完成至少两步"这个 metric 直接被 T1 bottleneck 卡死。T4→T5→T6 的 33.1% 才是 SALR + history 真正发挥的场景。

### Table 5（Ablation）

| FrozenVL | MF | SALR | Avg |
|---|---|---|---|
| | | | 30.3% |
| | √ | | 41.7% |
| | | √ | 44.8% |
| | √ | √ | 57.3% |
| √ | √ | √ | **1.1%** |

**最后一行 1.1% 是震撼教育**：VLM 必须可训练，frozen VLM 即使配 SALR + history 也废。这说明 SALR 不是"零成本利用 frozen VLM"的方法，它依赖 VLM 的 representation 被 co-adapted 到 action-relevant 子空间。这点和 OpenVLA 的 fine-tune 经验一致。

### Table 7（State ablation）

| | Goal | Object | Spatial | Long | Avg |
|---|---|---|---|---|---|
| w/o state | 84.2% | 90.0% | 93.0% | 71.5% | 84.7% |
| w/ state | 91.4% | 96.0% | 95.5% | 82.0% | 91.2% |

LIBERO-Long 上 +10.5%，证实 state-grounding 在长程任务价值最大。

### Table 9（Bi-manual Aloha BusyBox）

| Method | Avg |
|---|---|
| Diffusion Policy | 8.3% |
| π0 | 30.0% |
| **LoLA** | **46.7%** |

Bi-manual 18 维 state 比 7-DoF 信息量大，SALR 的外积调制价值更显著。

---

## 工程量级

- **模型大小**：~10B 参数
  - VLM backbone：Qwen2.5-VL-7B（28 层，hidden=3584，28 heads）
  - State Transformer：28 层，hidden=1024，8 heads
  - Action Expert：28 层 CFM decoder，hidden=1280，10 heads

- **训练**：32 × A100 40GB，batch 1280（这很大），AdamW lr=2.5e-5，cosine decay，5000 warmup，14 天

- **数据**：1.1M trajectories / 62M timestamps，OXE [https://arxiv.org/abs/2310.08864] + AgiBot [https://arxiv.org/abs/2503.06669] 混合

- **并行策略**：Intra-node FSDP（ZeRO-3）+ Inter-node DDP，激活 checkpoint 全开。理由是跨节点 FSDP 通信太贵，hybrid 平衡 memory 和 comm

- **真机数据采集**：Franka 7-DoF + 1-DoF gripper，Xbox controller teleop，20Hz 记录
  - 左摇杆：XY 平动
  - 右摇杆：XY 旋转（roll/pitch）
  - 左右 trigger：Z 平动
  - 左右 bumper：Z 旋转（yaw）
  - A 键：gripper 开合

---

## 我看到的几个有意思的地方

### 1. T1 单步 LoLA 反不如 π0

T1（pick up pan）LoLA 15.4% vs π0 46.2%。这是**没 history 时 SALR 可能反而拖后腿**的信号。State Transformer 增加的 capacity 在 short-horizon 起始步上可能 overfit 到某种 prior，或者 frozen-VLM-style 的初始阶段没训好。Paper 没解释，这是个 weakness。

### 2. T1→T2→T3 chain 的诡异

5.9% vs π0 17.8%。这个 metric 是"完成至少两步"，T1 success 15.4% 太低，$P(\text{at least 2}) \approx P(T1) \cdot P(T2|T1) + P(T2) \cdot P(T3|T2)$ 类似这种。T1 把上限卡死了。这是 metric 设计的 caveat，paper 没说明。

### 3. Learnable mask 的独立 ablation 缺失

$M_k, M_v$ 的贡献没单独消融，混在 SALR 整体里。不知道 mask 占多少 gain。猜测它主要起 sparsity prior 作用，避免外积空间 $(N_s \cdot N_v \cdot H)$ 太大导致 overfit。

### 4. State Transformer 28 层是否 overkill

State Transformer 只处理**一个 state vector**（不是 sequence），28 层 self-attention 在单 token 上其实是 28 层 MLP。每层做一次 self-attention 等于做一次 identity + MLP refinement。我怀疑 4-8 层就够了，paper 没做深度 ablation。

### 5. Outer product 的 memory 成本

$K^* \in \mathbb{R}^{N_s \times N_v \times H} = 8 \times N_v \times 1024$ 每层都要算，28 层累积下来 inference latency 应该不小。Paper 没报告 single-step inference time，这是 deployment 关心的硬指标。

### 6. 25 帧是怎么选的

Paper 说 "25 historical frames make a balance"，但没给 10/25/50 的对比。直觉上 25 帧 @ 20Hz = 1.25 秒 history，对于 pizza 任务（每个 sub-task 10-30 秒）来说只覆盖一个 sub-task 的中段。更长 history 可能能 capture "我在整个 recipe 的哪一步"这种 phase 信息。

---

## 几个 Karpathy 式联想

1. **SALR ≈ Hypernetwork for KV cache**。用 state 生成一个 modulation pattern 去改写 VLM 的 KV，本质上是"用 state 当 keys 去检索 / filter VLM 的知识"。这和 [VALL-E](https://arxiv.org/abs/2301.02111) 的 acoustic prompt 调制 TTS 模型的 spirit 相近。

2. **Asymmetric resolution 对应 information bottleneck**。当前帧要 sub-centimeter precision（gripper 对准），history 只要 phase。这是 position vs velocity 信息论差异的直接体现。

3. **VLM 必须 fine-tune 的发现很重要**。Table 5 的 1.1% 证明 frozen-VLM-pilot 的路子在 action generation 上走不通。原因可能是 web vision feature 的"语义轴"和 action 的"物理轴"重叠太少，需要 task-specific adaptation 把它们拽到一起。

4. **Long-horizon 最终需要 closed-loop recovery**。Paper conclusion 承认 perturbation-rich 任务还是 limitation。下一步应该是把 SALR 升级成 recurrent，让 state representation 携带 task progress，而不只是当前 proprioception。

5. **Bi-manual 的 gain 更大暗示什么**。Bi-manual 18 维 state vs 单臂 7 维，SALR 的 gain 在 bi-manual 上从 +9% 涨到 +16.7%。维度越高，multiplicative gating 的信息 gain 越大。这暗示 LoLA 的方法对 **humanoid（30-50+ DoF）** 可能价值更大——值得 follow up。

---

## 一句话总结

LoLA 把 robot proprioception 从"被动拼接的 token"升级为"主动调制 VLM 每层 KV 的 per-channel gate"，配合不对称 history 分辨率和 CFM action expert，在 long-horizon 上拿到显著 gain。剩下两个硬骨头——closed-loop recovery 和 short-horizon 上的 negative transfer——是显而易见的 next step。

---

## Reference 链接

方法核心：
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- TraceVLA: https://arxiv.org/abs/2412.10345
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Octo: https://arxiv.org/abs/2405.12213
- RDT-1B: https://arxiv.org/abs/2410.07864

VLM 骨干：
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- PaliGemma: https://arxiv.org/abs/2407.07726

方法背景：
- Flow Matching: https://arxiv.org/abs/2210.02747
- FiLM: https://arxiv.org/abs/1709.07871
- Hypernetworks: https://arxiv.org/abs/1609.09106
- FSDP/ZeRO: https://arxiv.org/abs/1911.06423

数据集：
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- AgiBot World: https://arxiv.org/abs/2503.06669
- LIBERO: https://arxiv.org/abs/2306.03310
- SIMPLER: https://arxiv.org/abs/2405.05941
- BridgeData V2: https://arxiv.org/abs/2308.12952

---

# LoLA: Long Horizon Latent Action Learning 深度解析

你好 Andrej！这篇 paper 我读了几遍，让我从架构哲学、数学机制、实验逻辑三个层面给你展开讲讲，重点关注那些**只有自己实现过 VLA 才会关心的细节**。

---

## 1. 一句话定位

LoLA 是一个 ~10B 参数的 VLA model，骨干用 Qwen2.5-VL-7B [https://arxiv.org/abs/2502.13923]，专门针对 **long-horizon manipulation** 做了三件事：(1) 不对称分辨率处理 history vs current；(2) 用一个并行的 State Transformer 通过 outer-product fusion 把 robot proprioception 真正"注入"到 VLM 的 KV cache 里；(3) 用 CFM action expert 解码多步 action。整体气质很接近 π0 [https://arxiv.org/abs/2410.24164] 的"VLM + flow matching action expert"骨架，但在 conditioning 机制上做了关键升级。

---

## 2. 为什么 long-horizon 是真正的硬骨头

paper 在 Section 1 列了三个挑战，我重组成两个本质问题：

**Compounding error 问题**：设 episode length 为 T，每步 success probability p，naive BC 整体 success ~ p^T。在 pizza-making 这种 ~2.3 分钟任务里 T 可能上百，所以即便单步 90%，长程 success 也会崩到很低——Table 6 里 π0 在 T1→T2→T3 上只有 17.8%，完全符合这个 decay。LoLA 把它拉到 5.9%（这一组偏低，原因后面讲），T4→T5→T6 拉到 33.1%（π0 是 12.4%，~2.67× improvement）。

**Modality gap 问题**：这是更深的洞察。VLM 输出的 embedding 是 web-scale 视觉语义空间里的点，它知道"杯子是红的"，但它不知道"end-effector 当前 z=0.32m, gripper width=0.05m"。而 action $a_t$ 是定义在 state 上的 delta：$s_{t+1} = s_t \oplus a_t$（Eq.3，$\oplus$ 是 SE(3) pose composition 或 joint space addition）。这意味着 **action 和 state 共享同一个 physical manifold**，而 VL embedding 完全不在这个 manifold 上。传统方法把 state 简单 concat 到 VL embedding 后面做 late fusion，本质上 state 只是一个 parallel token，attention 层得从零学会做 grounding——这件事很 inefficient，而且容易让 background features 误导 action。

这个 modality gap 的诊断是这篇 paper 的核心 motivation，也直接催生了 SALR。

---

## 3. 视觉编码：Asymmetric Resolution 的设计直觉

### 3.1 Current Observation Encoding (COE)
高保真输入：primary view $V_t$ + multi-view $\mathcal{V}_m = \{V_{sec}, \ldots, V_{wrist}\}$。wrist view 给 fine-grained egocentric 信息，secondary view 处理 occlusion。这些走 224×224，送进 vision encoder 得到 $F_{coe}$。

### 3.2 Historical Motion Encoding (HME)
25 帧 history，全部下采样到 112×112：

$$\mathcal{V}_{hme} = \{V_{t-n}^{\downarrow}, \ldots, V_{t-3}^{\downarrow}, V_{t-2}^{\downarrow}, V_{t-1}^{\downarrow}, V_t\} \tag{1}$$

这里的 intuition 是 **action 对当前帧是 position-sensitive 的，但对 history 是 velocity/phase-sensitive 的**。打个 Karpathy 风格的比方：你在 fast driving 时眼睛盯前方要看高清路标，但余光看后视镜只要知道"后面有车在动"就够了，不需要看清楚车牌。这样 25 帧 history 的 token 数大约减半，长序列 attention 的 $O(n^2)$ 开销显著下降。

但注意：当前帧 $V_t$ 同时出现在 COE 和 HME 里。这是一个稍微 tricky 的设计——可能为了让 history 的最后一帧和 current 的高分辨率帧有一个 implicit alignment point。

### 3.3 组装
$F_{in} = \{F_{coe}, F_{hme}, F_l\}$（Eq.2），送进 L-layer VLM。每一层 i 产出 $K_i, V_i \in \mathbb{R}^{N_v \times H}$。这些 KV cache 不会直接被 action head 用，要先进 SALR。

---

## 4. SALR：这篇 paper 的真正灵魂

### 4.1 核心公式拆解

State $s_t = (q_t, p_t)$，包含 joint angles $q_t$ 和 end-effector pose $p_t$。先经过 state projection layer 升到维度 H，得到 initial state embedding，送进 State Transformer。

State Transformer 也是 L=28 层（与 VLM 等深，这是关键！），每层内部先 self-attention refine state，然后投影出 state 的 query：

$$Q_r \in \mathbb{R}^{N_s \times H}$$

其中 $N_s$ 是 State Transformer 的 attention head 数（论文里是 8），$H=1024$ 是 State Transformer 的 hidden dim（比 VLM 的 3584 小很多，是 efficiency 考量）。

然后第 i 层做 **outer-product fusion**：

$$K^*[i, j, :] = Q_r[i, :] \odot K_i[j, :] \tag{4}$$
$$V^*[i, j, :] = Q_r[i, :] \odot V_i[j, :] \tag{5}$$

这里 $\odot$ 是 element-wise product，$i \in [1, N_s]$, $j \in [1, N_v]$。结果 shape 是 $(N_s, N_v, H)$——所以叫"outer product"，因为它在 head 维度上做笛卡尔积，每对 组合都有一个独立的 fused representation。

**为什么用乘法而不是加法？** 这是 paper 最核心的设计选择。我的理解：
- Additive fusion（concat 或 sum）让 state 和 VL feature 在不同 sub-manifold 里独立存在，attention 后面再混合；
- Multiplicative fusion 让 **state 直接 gate/modulate 每个 VL feature channel**。如果某个 visual feature 在当前 physical state 下 irrelevant，$Q_r$ 的对应维度会被训成接近 0，直接把那个 channel 抑制掉。这等价于 state 给 VL embedding 学一个 conditional mask，但 mask 是 dense per-channel 的。

这有点像 FiLM conditioning [https://arxiv.org/abs/1709.07871] 的精神，但 FiLM 用 affine transform $\gamma \odot x + \beta$，这里只有 multiplicative term，没有 additive bias——可能是因为 KV 本身已经被 VLM pretrain 充分校准，不需要再加 bias。

### 4.2 Learnable Mask

接下来：
$$K' = K^* \odot M_k, \quad V' = V^* \odot M_v \tag{6}$$

$M_k, M_v \in \mathbb{R}^{(N_s \cdot N_v) \times H}$。这是一个**不依赖输入的 learnable mask**，是模型的参数。它的作用是**全局抑制 action-irrelevant 噪声**，比如 background distractor。注意这里和 state-dependent 的 $Q_r$ 不同——$M_k/M_v$ 是 unconditional 的，相当于一个 learnable sparsity prior。

我猜测这个 mask 训完之后大概率是稀疏的（很多接近 0 的 channel），相当于在 $(N_s \cdot N_v \cdot H) \approx 8 \cdot N_v \cdot 1024$ 这个超大空间里挑出真正 action-relevant 的 subset。这和 "sparsely-activated mixture of experts" [https://arxiv.org/abs/2009.06742] 在哲学上有点像。

### 4.3 Latent Space Compression

最后把 $K', V'$ 压到 $K^a, V^a \in \mathbb{R}^{N_a \times H}$，$N_a$ 是 action expert 的 head 数。论文没明确说怎么压，但从 shape 推测是一个 linear projection 或 pooling。这个 $N_a$ 维度的 KV 就是 action expert 的 cross-attention 的 K/V 输入。

### 4.4 整体信号流

```
[State s_t] ──→ State Projection ──→ State Transformer (28 layers)
                                          │
                                          │ Q_r per layer
                                          ↓
[Multi-view current] ──┐                 │
[25-frame history] ──→ VLM (28 layers) ──┤ outer product ⊙
[Language]            ──┘                ↓
                                  K*, V* (N_s × N_v × H)
                                          │
                                          │ × M_k, M_v (learnable)
                                          ↓
                                  K', V'
                                          │ compress
                                          ↓
                                  K^a, V^a ──→ Action Expert (CFM, 28 layers)
                                                       │ cross-attn
                                                       ↓
                                              {a_t, a_{t+1}, ..., a_{t+s}}
```

---

## 5. Action Expert: Conditional Flow Matching

这部分 paper 写得比较 skim，但实际上是 π0 直接的延续。CFM [https://arxiv.org/abs/2210.02747] 是 flow-based generative model 的一种，相比 diffusion 用 ODE 而不是 SDE 做去噪，路径更直，采样步数可以更少。

Action expert 是 M=28 层 Transformer Decoder，hidden=1280，10 heads。条件有三：
1. SALR 出来的 $\{K^a, V^a\}$（cross-attention 的 K/V）
2. Noisy action trajectory（self-attention 主输入）
3. Noise timestep $z$（sinusoidal embedding）

训练目标：predict noise/vector field。推理时多步 denoise 从 $\mathcal{N}(0, I)$ 出发生成 $s$ 步 action chunk。

**为什么不用 autoregressive action token？** 这是 π0/LoLA 派系 vs RT-2/OpenVLA 派系 [https://arxiv.org/abs/2307.15818, https://arxiv.org/abs/2406.09246] 的根本分歧：
- AR token 把 continuous action 离散化到 256 bins，丢精度；多步 action 之间有强 temporal correlation，AR 假设独立很 inefficient
- Flow matching 直接在连续空间建模 multi-modal distribution，可以一次性生成整个 chunk，平滑性更好

---

## 6. 实验数据的关键 reading

### 6.1 SIMPLER Google Robot (Table 1)
- Visual Matching 平均：LoLA 61.5% vs π0 52.7% vs RT-2-X 46.3%
- 关键 win 在 "Move Near"（71.7% vs 35.0%）——这是一个需要 spatial reasoning 的任务，SALR 的 grounding 帮助大
- "Open Top Drawer and Place Apple" 是组合任务，LoLA 26.9% vs π0 16.0%

### 6.2 SIMPLER WidowX (Table 2)
平均 71.9% vs π0 41.7% vs CogACT 51.3%。WidowX 的 4 个任务里 3 个是 long-horizon (Put X on Y, Stack A on B)。LoLA 在 "Put Spoon on Towel" 拿到 95.8%，π0 是 62.5%。

### 6.3 LIBERO (Table 3)
- 平均 96.2% vs π0 92.2%
- LIBERO-Long 88.2% vs π0 85.4%
- LIBERO-Object 99.6%（接近 ceiling）

LIBERO-Long 的 88.2% 很 impressive，因为 LIBERO-Long 任务有 ~10 步 stages，diffusion policy 只有 50.5%，OpenVLA 53.7%。

### 6.4 真实 Franka (Table 4 + Table 6)
这里 paper 自己也承认 absolute 数字偏低：π0 单步平均 36.8%，LoLA 46.1%。T1 (pick up pan) LoLA 只有 15.4% 反而比 π0 的 46.2% 差——这点很有意思，paper 没有解释。我猜测是因为 T1 是 episode 起始步，没有有效 history 可用，SALR 的优势发挥不出来，反而 State Transformer 的额外 capacity 在 short-horizon 上 overfit 或欠拟合到某种 prior 上。这是一个**值得 follow up 的 failure mode**。

但 multi-step (Table 6)：
- T1→T2→T3: LoLA 5.9% vs π0 17.8% （这里 LoLA 反而输！）
- T4→T5→T6: LoLA 33.1% vs π0 12.4%
- T7→T8→T9: LoLA 28.9% vs π0 16.6%

T1→T2→T3 输 π0 我没看懂——可能是 T1 success rate 太低（15.4%）导致整个 chain 早就崩了，"完成至少两步"的 metric 在这种 case 下其实变成了"完成 T1 和 T2"，所以 LoLA 的乘积 0.154 × P(T2|T1) 反而更低。这是 metric 设计的 caveat。

### 6.5 Bi-Manual Aloha BusyBox (Table 9)
46.7% vs π0 30.0% vs DP 8.3%。Bi-manual 的高自由度让 SALR 的 state-grounding 价值更明显——18 维 state（每 arm 7 joint + gripper）比 7-DoF 单臂的 information content 更大，外积调制的效果更显著。

### 6.6 Ablation (Table 5)
最 striking 的 ablation：
- 只有 MF（多帧）：41.7%
- 只有 SALR：44.8%
- 两者都加：57.3%
- **FrozenVL + MF + SALR：1.1%** ← 这个数字非常震撼

最后这一行说明：VLM 必须 fine-tune，光靠 frozen VLM 的 embedding 即使有 SALR 也完全不行。这和 OpenVLA 的发现一致——action generation 需要 representation adaptation。这也间接说明 SALR 不是"零成本"利用 frozen VLM 的方法，它依赖 VLM 的 representation 被 co-adapted。

### 6.7 State ablation (Table 7)
w/o state 84.7% → w/ state 91.2%，在 LIBERO-Long 上从 71.5% → 82.0%（+10.5%）。这证实了 state-grounding 在 long-horizon 上价值最大。

---

## 7. 与其他 VLA 的定位对比

| 模型 | VLM backbone | Action head | State fusion 方式 | Long-horizon 支持 |
|---|---|---|---|---|
| RT-2 [https://arxiv.org/abs/2307.15818] | PaLI-X / PaLM-E | AR token (256 bins) | token concat | × |
| OpenVLA [https://arxiv.org/abs/2406.09246] | Prismatic 7B | AR token | token concat | × |
| Octo [https://arxiv.org/abs/2405.12213] | ViT (93M) | Diffusion head 3M | 跨 attention | partial |
| RDT-1B [https://arxiv.org/abs/2410.07864] | ViT | Diffusion 1B | 跨 attention | partial (bi-manual) |
| CogACT [https://arxiv.org/abs/2411.19650] | VLM | Diffusion | cognition token conditioning | partial |
| π0 [https://arxiv.org/abs/2410.24164] | PaliGemma 3B | Flow matching | late concat | partial (chunk=10) |
| SpatialVLA [https://arxiv.org/abs/2501.15830] | VLM | AR/diffusion | spatial encoding | partial |
| **LoLA** | Qwen2.5-VL 7B | Flow matching | **outer-product SALR** | √ (25 frames) |

LoLA 的 unique selling point 是 **multiplicative state grounding + long history**。其他 VLA 要么没有 history (RT-2, OpenVLA, π0)，要么 state 只是 token (Octo, RDT)，要么用 cognition token 做间接触发 (CogACT)。

---

## 8. 工程细节（Section 6 of supplementary）

### 8.1 数据
1.1M trajectories, 62M timestamps，混合 OXE [https://arxiv.org/abs/2310.08864] + AgiBot [https://arxiv.org/abs/2503.06669]。Fractal 14.3%, Kuka 14.4%, Bridge 13.3%, AgiBot 7.1% 占大头。AgiBot 是中文团队自己采的 dataset，对 cooking/kitchen 任务覆盖好。

### 8.2 训练
- 32 × A100 40GB
- Batch size 1280 (!) 这个很大
- AdamW, lr 2.5e-5, cosine decay, 5000 warmup
- 14 天训练
- FSDP intra-node + DDP inter-node（hybrid sharding，因为纯 FSDP 跨节点通信太贵）
- Activation checkpointing 全开

### 8.3 Real-world 数据采集
20Hz teleop（Xbox controller），6-DoF mapping：左摇杆 XY 平动 + 右摇杆 XY 旋转 + triggers Z 平动 + bumpers Z 旋转 + A 键 gripper。这个 mapping 设计得挺合理，能 cover 6 DoF continuous control。

### 8.4 Franka benchmark 构成
- 22 atomic sub-tasks 组成 7 sequential groups (G1-G7)
- 6 个 end-to-end episodes (E1-E6)，平均 3.6 min，最长 E1 "Full Baking Cycle" 5.2 min
- 全部 cooking 主题，pizza making 是 narrative thread

---

## 9. 我的几个 takeaways 和疑问

### Intuition builders:

1. **Outer product = state-conditioned channel gating**。把它理解为给 VLM 的每个 KV channel 学一个 state-dependent scalar weight，比 concat 强在"state 在每层都重新参与调制"。

2. **Asymmetric resolution 对应 position vs velocity 信息论**。当前帧要 sub-centimeter precision（gripper 对齐），history 只需 phase 信息。

3. **VLM 必须可训练**。Table 5 的 FrozenVL=1.1% 证明 representation gap 太大，SALR 也救不了 frozen VLM。

4. **LIBERO-Long 的 88.2% 是真正的信号**。这个 suite 设计成 "A and B" 复合任务，没有 history 上下文几乎不可能维持 coherence。

### 疑问 / 可能的 weakness:

1. **T1 在 Franka 单步任务上 LoLA 反而不如 π0**（15.4% vs 46.2%）。这可能暗示 SALR 在 short-horizon 上有 negative transfer，或者 State Transformer 的初始化没收敛好。paper 没解释。

2. **T1→T2→T3 chain 上 LoLA 5.9% vs π0 17.8%**，与 T4→T5→T6 的 33.1% vs 12.4% 形成鲜明对比。同样三个连续任务，为什么差距这么大？是 T1 success 太低 cascade 还是别的？metric 设计上"完成至少两步"很容易被 first step bottleneck。

3. **Learnable mask $M_k, M_v$ 的 ablation 缺失**。paper 没单独消融 mask 的贡献，只消融了 SALR 整体。不知道 mask 占了多少 gain。

4. **Outer product 在 $N_s \cdot N_v \cdot H$ 上的 memory 成本**。$8 \cdot N_v \cdot 1024$ 每层都要算，28 层累积下来不小的开销。paper 没报告 single-step inference latency。

5. **State Transformer 28 层是不是 overkill？** 它只处理一个 state vector（不是 sequence），28 层 self-attention 在单 token 上其实是 28 层 MLP。感觉可以 ablate 不同深度。

6. **Action chunk size $s$ 没明确报告**。从实验描述推测可能在 10-50 步之间，这个数字对 long-horizon consistency 很关键。

### Karpathy 视角的几个联想:

- SALR 的 outer-product fusion 让我想到 **Hypernetworks** [https://arxiv.org/abs/1609.09106] 的精神——用一个网络生成另一个网络的参数。这里 state 生成 KV cache 的"调制参数"。如果把 $Q_r$ 看成 hypernetwork 输出，那 $K^* = Q_r \odot K$ 就是 weights that depend on state。这和 "conditional computation" 路线很近。

- LoRA / hypernetwork / cross-attention 三者其实是 spectrum：cross-attention 是 sparse 调制，outer-product 是 dense 调制，hypernetwork 是 full weights。LoLA 选了 dense per-channel modulation，这是计算和表达力的 sweet spot。

- 这个 paper 的 SALR 本质上是在做 **"VLM 作为 perception，state 作为 adapter"** 的架构。VLM 提供通用 visual reasoning，state 把它"翻译"到 robot 自己的 physical frame。这种 separation of concerns 很优雅。

- 长程任务最终还是要解决 **closed-loop error correction**。paper 在 conclusion 里承认"complex novel perturbation-rich long-horizon tasks"仍是 limitation。下一步应该是把 SALR 升级成 recurrent 的，让 state representation 自己携带 task progress 信息，而不只是当前 proprioception。

---

## 10. Reference 链接汇总

主方法参考：
- LoLA 本身（这篇 paper）
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- TraceVLA: https://arxiv.org/abs/2412.10345
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Octo: https://arxiv.org/abs/2405.12213
- RDT-1B: https://arxiv.org/abs/2410.07864

VLM 骨干：
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- PaliGemma: https://arxiv.org/abs/2407.07726
- Prismatic VLMs: https://arxiv.org/abs/2405.12981

数据集：
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- AgiBot World: https://arxiv.org/abs/2503.06669
- LIBERO: https://arxiv.org/abs/2306.03310
- SIMPLER: https://arxiv.org/abs/2405.05941
- BridgeData V2: https://arxiv.org/abs/2308.12952

方法背景：
- Flow Matching: https://arxiv.org/abs/2210.02747
- FiLM conditioning: https://arxiv.org/abs/1709.07871
- Hypernetworks: https://arxiv.org/abs/1609.09106
- Mixture of Experts (sparse): https://arxiv.org/abs/2009.06742
- FSDP/ZeRO: https://arxiv.org/abs/1911.06423

---

总结一句：LoLA 的贡献集中在 SALR 这个 multiplicative state-grounding mechanism 上，它把 robot proprioception 从"被动拼接的 token"升级为"主动调制 VLM KV 的 per-channel gate"。配合 asymmetric temporal sampling，在 long-horizon 上拿到了显著 gain。剩下的硬骨头是 closed-loop recovery 和 short-horizon 上可能的 negative transfer——这两个是未来 VLA 进化的 obvious next step。
