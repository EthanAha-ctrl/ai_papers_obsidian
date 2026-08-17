---
source_pdf: VTLA.pdf
paper_sha256: a3f2197448da871a2d5903939ac2975e6b1419ffdcc00b28b74a2e05d4815dbc
processed_at: '2026-08-13T03:30:56-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VTLA

## 一、这篇 paper 在干嘛

一句话：**教机器人插钥匙**。

你拿钥匙开锁的时候，眼睛看个大概位置，然后钥匙碰到锁孔那一刻，手指头会"感觉"到歪没歪，微调一下就插进去了。VTLA 就是想让机器人也这么干——**用眼睛看个大概，用触觉做微调，用语言理解任务**。

之前的工作要么只用眼睛（VLA），要么只用触觉（TLA），就像你闭着眼睛开锁或者不看锁孔纯靠手感，都不太行。VTLA 把两路信号都接上，再加一层 preference learning 来"校准"输出，效果就上来了。

参考 TLA: https://arxiv.org/abs/2503.08548

---

## 二、任务有多难

**Peg-in-hole**，就是往孔里插销钉。

难点在哪？**Clearance 只有 0.6mm**。什么概念？一根头发丝大概 0.07mm，0.6mm 也就是 8 根头发丝的余量。机器人 gripper 抓销钉本身就有 ~0.5mm 的位置抖动，GelStereo 触觉传感器 noise 大概 0.1mm。**signal 和 noise 几乎在一个量级**。

而且销钉形状有 5 种：square, triangle, hexagon, pentagon, round。每种形状的对称性不一样，round 最简单（转哪个角度都能进），square 次之（只有 4 个角度对得上），triangle 最难（只有 3 个）。

机器人初始位置随机偏移 ±2.5mm + ±5°，最多试 15 次。这就好比把你眼睛蒙上、手绑在棍子上，让你开锁，还得 15 次以内搞定。

---

## 三、数据是怎么来的

**全在 simulation 里造**，NVIDIA Isaac Gym + 自己写的 visuotactile simulator。

28000 个样本，每个样本包含：
- 左手指 4 帧触觉图（拼成 2×2 grid）
- 右手指 4 帧触觉图（拼成 2×2 grid）
- 1 张手腕相机拍的视觉图
- 1 个 action label `[x, y, rz]`

**为什么用 simulation？** 因为真实机器人采数据太慢了。28000 次 insertion，每次 15 步，机器人得跑几十万次，几个月。simulation 一晚上搞定。

**但 simulation 和 real 有 gap**，尤其是触觉。硅胶的 Young's modulus 会老化，friction 受湿度影响，sim 里很难精确建模。作者的解法是 **domain randomization**：把这些物理参数都随机化，让模型学到的是"在任意物理参数下都能 work"的策略，而不是死记硬背某一套参数。

具体随机化范围：
- Young's modulus: U(1.0e5, 5.0e5) Pa — 硅胶软硬变化 5 倍
- Friction coefficient: U(0.2, 0.7) — 摩擦系数变化 3.5 倍
- 光照强度: U(0.2, 0.6) — 亮暗变化 3 倍
- 图像 scale/rotation/translation/shear 都随机

这就像你训练一个人开锁，让他用各种材质的钥匙、各种天气、各种光线条件下都练一遍，他到真实场景就更容易泛化。

---

## 四、核心 Trick 1: VGTE — 把视觉放最后

### 问题

LLM 有个毛病叫 **recency bias**：注意力偏向序列末尾的 token。离末尾越近的 token，被"看见"得越清楚。

这本来是缺陷，但作者反过来了：**既然模型对末尾 token 更敏感，那我就把最重要的信息放末尾**。

### 哪个模态最重要？

在 insertion 任务里，**视觉在前期最重要**（approach 阶段，要看到 hole 在哪），触觉在后期最重要（contact 后才能感知 misalignment）。但 action prediction 是最后一步，所以"离 action 最近的 token"会被重点关注。

作者的排列：**tactile 在前，vision 在后**。这样 vision token 离 action token 最近，被赋予更高权重，模型更依赖视觉做粗定位。

### 触觉怎么处理？

触觉是 4 帧时序图像。Qwen2-VL 用 3D convolution 处理 video，但 3D conv 在 4 帧 short clip 上表现不好——kernel 设计受限，而且预训练 weight 是在 semantic video（动作识别、视频问答）上学的，跟触觉的 pressure distribution 分布完全不一样。

作者的解法：**把 4 帧拼成 2×2 grid 的单张图**，然后用 ViT 编码。ViT 的 self-attention 在 2×2 grid 上能自然捕捉 4 帧之间的 spatial-temporal 关系。这样 **temporal fusion 发生在 tokenization 之前**，绕过了 LLM 内部对 temporal reasoning 的弱点。

参考 Qwen2-VL: https://arxiv.org/abs/2409.12191

### 直觉

就像你读一段话，最后那句话印象最深。作者把视觉信息放在"最后一句"，让模型"印象最深"。

---

## 五、核心 Trick 2: DPO — 把分类当回归做

### 问题

LLM 本质是 **classification**——预测下一个 token 是 vocabulary 里哪个词，用 cross-entropy loss。

但 robotic control 是 **regression**——输出连续的 `[x, y, rz]` 数值。

问题出在哪？action `[-0.9, 0.4, 0.013]` 被 tokenize 成离散字符 `[-`, `0`, `.`, `9`, `,`, ` `, `0`, `.`, `4`, ...，每个 token 独立算 cross-entropy。

但数值是有拓扑结构的：
- `[0.9, 0.4, 0.013]` 和 `[-0.9, 0.4, 0.013]` 物理上方向完全相反
- `[0.9, 0.4, 0.013]` 和 `[0.95, 0.4, 0.013]` 物理上几乎一样

NTP loss 看不到这种结构，只看到 token 是否匹配。就像你让模型猜"答案是 0.9"，模型猜 0.95，NTP 会说"完全错"，但物理上 0.95 已经很接近了。

### 解法

作者用 **DPO (Direct Preference Optimization)** 来模拟 regression-like supervision。

具体步骤：
1. 用 SFT 后的模型在训练样本上生成多个候选 action（不同 sampling 配置）
2. 计算每个候选和 ground truth 的 L1 距离
3. 距离近的标记为 `chosen`，远的标记为 `rejected`
4. 用 DPO loss 优化模型

DPO loss 公式：

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_{\theta}(y_{\mathrm{chosen}} \mid x)}{\pi_{\mathrm{ref}}(y_{\mathrm{chosen}} \mid x)} - \log \frac{\pi_{\theta}(y_{\mathrm{reject}} \mid x)}{\pi_{\mathrm{ref}}(y_{\mathrm{reject}} \mid x)}\right)$$

变量解释：
- $x$: input（视觉 + 触觉 + 文本 tokens）
- $y_{\mathrm{chosen}}$: 离 ground truth 更近的 action
- $y_{\mathrm{reject}}$: 离 ground truth 更远的 action
- $\pi_{\theta}$: 正在训练的模型
- $\pi_{\mathrm{ref}}$: 冻住的 reference 模型（防止训练跑偏）
- $\beta$: 控制偏好信号尖锐度，β 大信号尖锐，β 小信号平滑
- $\sigma$: sigmoid 函数

### 为什么这等价于 regression？

DPO 在隐式地告诉模型："离 ground truth 越近的 action，reward 越高"。这等价于在 action space 上施加一个 **单调下降的 implicit reward function** $r(y) \propto -\|y - y_{\mathrm{gt}}\|_1$。

这就是 regression 的本质——让输出分布向 ground truth 收敛。但 DPO 是相对比较，不是绝对回归，所以不会 collapse 到单点，保留了泛化能力。

同时 KL anchor（$\pi_{\mathrm{ref}}$）防止 overfitting 到 sampled ground truth，这正是 paper 说的 "alleviating overfitting to ground-truth actions"。

参考 DPO: https://arxiv.org/abs/2305.18290

### 直觉

就像教小孩投篮，你不一定要他每次都投进（绝对回归），而是说"这次比上次准"（相对比较）。慢慢地他的动作就收敛到正确姿势了，但不会因为某一次没投进就推翻所有学习。

---

## 六、实验结果讲了什么

### 6.1 数据集上（Table 1）

| Method | ID GCR | OOD GCR |
|--------|--------|---------|
| DP (Diffusion Policy) | 7.8% | 8.5% |
| TLA (只有触觉) | 15.3% | 14.4% |
| VLA (只有视觉) | 46.1% | 29.5% |
| **VTLA** | **47.3%** | **31.2%** |

**几个直觉**：

1. **DP 几乎挂了**（7.8%）。Diffusion policy 在 contact-rich 任务上需要海量数据和高维 action modeling，28000 samples 不够。这印证了 paper 核心论点：contact-rich 任务需要 LLM-level reasoning。

2. **TLA 远逊于 VLA**（15.3% vs 46.1%）。只有触觉没有视觉，机器人不知道 hole 在哪，瞎撞。

3. **VTLA 比 VLA 略好**（47.3% vs 46.1%）。触觉是 marginal improvement，ID 提升小，OOD 提升大。说明触觉主要帮助 generalization——视觉在 OOD 形状下可能 ambiguous，触觉提供 complementary 信息。

### 6.2 不同 Clearance（Table 2）

| Method | 2.0mm | 1.6mm | 1.0mm | 0.6mm |
|--------|-------|-------|-------|-------|
| DP | 42% | 32% | 28% | 22% |
| VLA | 100% | 98% | 90% | 80% |
| TLA | 94% | 90% | 80% | 80% |
| **VTLA** | **100%** | **98%** | **96%** | **90%** |

**直觉**：clearance 越小越难。0.6mm 是极限设定，VTLA 还能 90%。DP 在小 clearance 下直接崩溃，因为 diffusion 学不到精细的 action distribution。

注意一个反直觉的点：0.6mm 下 VTLA 的 step 数（5.91）比 VLA（5.55）还多。但 success rate 更高（90% vs 80%）。说明 VTLA 更"保守"，走小步，宁可多走几步也要保证成功。这是 **success rate vs efficiency** 的 trade-off。

### 6.3 DPO Ablation（Table 4）

| Method | ID GCR | OOD GCR |
|--------|--------|---------|
| w/o DPO | 47.5% | 27.0% |
| w/ DPO-1k | 47.5% | 31.4% |
| w/ DPO-2k | 47.3% | 31.2% |

**直觉**：DPO 主要提升 OOD generalization（+16%），ID 几乎不变。这很符合 DPO 本质——SFT 阶段模型 overfit 到 training distribution，DPO 通过相对比较"软化"了这种 overfitting。

**Data scaling 边际效应为 0**（1k → 2k 几乎无提升）。作者推测：preference data 只来自两种 sampling config，多样性不足。这与 RLHF 经验一致——preference data 的 diversity 比 quantity 更重要。

### 6.4 Real-world Sim2Real（Table 5, 6, 7）

**最 striking 的发现**：

1. **VTLA real-world OOD shapes 100% success**（Table 6）。甚至比 ID 还好。Sim2Real gap 在 VTLA 上很小。

2. **TLA 在 real world 上崩溃**：sim 中 80%+，real 中只有 30-40%（Table 7）。这是 tactile sim2real 的经典难题——硅胶老化、温度漂移、表面污染很难在 sim 中建模。VTLA 通过 vision 弥补了 tactile 的 sim2real gap。

3. **VTLA 比 VLA 更高效**：pentagon 上 1.85 step vs 2.3 step。触觉帮助 VTLA 更快找到正确方向，因为 contact 后的触觉信号直接指示了 misalignment 方向。

### 6.5 Poor Lighting 实验（Appendix C）

这是隐藏的 gem。光线暗的情况下，VLA 模型看不到 hole 位置，完全失败。VTLA 靠触觉还能完成。

这就是 **multimodal fusion 的核心价值**：单一模态脆弱，融合模态鲁棒。就像自动驾驶里 camera + LiDAR + radar 的 redundancy 设计——一种 sensor 挂了，另一种还能顶上。

---

## 七、硬件配置

- **Robot arm**: UR3 6-DoF
- **Gripper**: Robotiq 2F-85
- **Wrist camera**: Intel RealSense D405
- **Tactile sensors**: 2× GelStereo 2.0，装在 gripper 指尖
- **Tactile frame rate**: 20 FPS

GelStereo 2.0 是 visuotactile sensor——硅胶层变形通过 stereo vision 重建，输出 tactile image。每个指尖一个，所以 left + right 两路触觉。

参考 GelStereo 2.0: https://ieeexplore.ieee.org/document/10309321

---

## 八、几个值得琢磨的点

### 8.1 Token 顺序也是 inductive bias

VTLA 把 recency bias 从 bug 变成 feature。这启发一个更 general 的设计原则：**在 LLM-based policy 中，token 顺序本身是一种 inductive bias**。

可以联想到 prompt engineering 里"最后一句话最重要"，in-context learning 里 recency effect，CoT reasoning 里关键 step 放最后。

参考 recency bias: https://arxiv.org/abs/2310.01427

### 8.2 DPO 作为 Regression Proxy 的推广

VTLA 用 DPO 处理 continuous action 的思路，可以推广到其他 LLM-based regression 任务：
- Time series forecasting with LLMs
- Code generation with continuous parameters
- Scientific computing 数值解

但要注意：preference data 依赖 ground truth，如果 ground truth 噪声大，preference signal 会被污染。

### 8.3 Tactile Sim2Real 仍是 open problem

TLA 在 real world 掉到 30-40%，揭示 tactile sim2real 很难。可能方向：
- Tactile domain adaptation（类似 visual 的 CyCADA）
- Real-to-sim tactile matching（用 real 数据校准 sim 物理参数）
- Tactile foundation models（如 AnyTouch https://arxiv.org/abs/2502.12191）

### 8.4 Action Tokenization 的局限

把 `[-0.9, 0.4, 0.013]` tokenize 成字符 token，损失了数值结构。可能改进：
- Bin-based action tokenization（RT-2 思路）
- Continuous action head（在 LLM 外接 MLP/diffusion head，π0 思路 https://arxiv.org/abs/2410.24164）
- 数值 token 的 metric-aware embedding

### 8.5 几何对称性的理论解释

Round peg 上所有方法都 90%+，可以用群论解释：
- Round 的对称群是 $SO(2)$（连续旋转对称）
- Square 是 $\mathbb{Z}_4$（4 阶循环群）
- Pentagon 是 $\mathbb{Z}_5$
- Hexagon 是 $\mathbb{Z}_6$

对称阶数越高，effective configuration space 越小，任务越容易。如果模型能显式利用 symmetry（如 equivariant networks https://arxiv.org/abs/2406.16639），可能进一步提升。

### 8.6 Multi-resolution Sensing

0.6mm clearance 接近 sensor noise 极限。VTLA 能 work 说明：
- Vision 提供 ~mm 级粗定位
- Tactile 提供 ~0.1mm 级精修
- 两者协同才能在 0.6mm 下成功

这是 **coarse-to-fine optimization** 的经典思想。

### 8.7 数据效率惊人

28000 samples + 7B 模型，达到 90%+ real-world success。对比：
- RT-2: web-scale 数据
- GR-2: web-scale video pre-training
- π0: 大规模 robot trajectories

VTLA 的小数据成功归因于：任务聚焦 + VLM pre-training strong prior + domain randomization + DPO 二次提炼。

---

## 九、一句话总结

**VTLA = VLM + 触觉 + DPO 校准**，用 LLM 的 recency bias 放大视觉权重，用 DPO 把分类问题伪装成回归问题，在 0.6mm clearance 的 peg-in-hole 任务上达到 90%+ 成功率，Sim2Real 表现优秀。

核心 insight：**LLM 的一些"缺陷"（recency bias, classification loss）可以被反向利用成为设计杠杆**。这不是 patch LLM，而是理解 LLM 的特性后顺势而为。

---

参考资源：
- VTLA 项目主页: 见 paper 中链接
- TLA paper: https://arxiv.org/abs/2503.08548
- DPO paper: https://arxiv.org/abs/2305.18290
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- π0: https://arxiv.org/abs/2410.24164
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- GelStereo 2.0: https://ieeexplore.ieee.org/document/10309321
- Touch100k: https://arxiv.org/abs/2406.03813
- Recency bias: https://arxiv.org/abs/2310.01427
- Equivariant Diffusion: https://arxiv.org/abs/2406.16639
- AnyTouch: https://arxiv.org/abs/2502.12191

---

# VTLA Paper 深度解析

## 一、Paper 核心定位

VTLA (Vision-Tactile-Language-Action) 是一篇针对 **contact-rich manipulation** 的工作，具体任务锁定在 **peg-in-hole insertion**。从研究脉络看，这是 TLA (Tactile-Language-Action, arXiv:2503.08548) 的直接延续，并且补上了视觉模态这块拼图。

核心 motivation 可以拆成三层：
- VLA models (RT-2, OpenVLA, GR-1/2, RDT-1B, π0) 在 pick-and-place 这类 visually-dominant 任务上很强，但是 contact-rich 场景下 vision 信息不足；
- TLA 只有 tactile，缺乏 global perception，存在 performance ceiling；
- LLM 的 NTP loss 本质是 classification，与 robotic control 的 regression 性质存在 mismatch。

VTLA 同时 tackle 这三个问题，分别用 **VGTE tokens**、**vision-tactile 融合**、**DPO preference learning** 来对应解决。

参考链接：
- TLA paper: https://arxiv.org/abs/2503.08548
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- DPO: https://arxiv.org/abs/2305.18290

---

## 二、任务设定与数据构造

### 2.1 Peg-in-hole 任务形式化

任务流程：
1. Gripper 抓住 peg，定位到 hole 上方
2. 引入 **3-DOF misalignment**: x-axis, y-axis 平移 + z-axis 旋转
3. Gripper 下降尝试 insertion
4. 若 collision 发生 → retract 重试
5. 若无 collision 且达到 insertion depth → success
6. 最多 15 次尝试，否则 fail

数据集统计：
- **28,000 assembly samples**
- **5 种 peg-hole shapes** (square, triangle, hexagon, pentagon, round)
- **Clearance 范围**: 0.6 mm ~ 2.0 mm
- 每个 sample 含：left/right tactile image sequences (2×2 grid, 4 帧) + 1 张 wrist visual image + action label

Action label 是三维向量 `[x, y, rz]`，例如 `[-0.9, 0.4, 0.013]`：
- `x = -0.9`: x 方向 correction（负方向）
- `y = 0.4`: y 方向 correction
- `rz = 0.013`: z 轴旋转 correction

### 2.2 Instruction Format

数据被组织成 chat-like instruction 格式：

```
<|im_start|>user
<|vision_start|> TactileLeftHand.png <|vision_end|>
<|vision_start|> TactileRightHand.png <|vision_end|>
<|vision_start|> WristImage.png <|vision_end|>
Given the tactile images from robot left and right fingertips and a wrist camera image...
Predict action: <|im_end|>
<|im_start|>assistant
[-0.9, 0.4, 0.013] <|im_end|>
```

注意一个细节：**tactile 在前，vision 在后**。这看似随机，其实是 VGTE 设计的核心，下文详述。

### 2.3 Domain Randomization

为 zero-shot Sim2Real，作者在 simulation (NVIDIA Isaac Gym + 自建 visuotactile simulator) 中大量随机化：

| 类别 | 参数 | 分布 |
|------|------|------|
| Physical | Young's modulus | U(1.0e5, 5.0e5) Pa |
| Physical | Poisson ratio | U(0.3, 0.48) |
| Physical | Friction coefficient | U(0.2, 0.7) |
| Task | Peg offset x in gripper | U(-1.0, 1.0) mm |
| Task | Peg offset z in gripper | U(-1.0, 1.0) mm |
| Task | Contact depth | U(0.6, 0.9) mm |
| Vision | Light intensity | U(0.2, 0.6) |
| Vision | Scale | U(0.9, 1.1) |
| Vision | Translation | U(-10, 10) pixels |
| Vision | Rotation | U(-3, 3) deg |
| Vision | Shearing | U(-3, 3) deg |

Physical 参数随机化背后的 intuition：GelStereo 类 visuotactile sensor 的硅胶层会随时间老化，Young's modulus 和 Poisson ratio 会漂移，real world 中很难精确标定。Friction 受材料、湿度等多因素影响，sim 中建模困难。这种 "modeling the uncertainty" 的思路与 **Bayesian domain randomization** 思想接近。

---

## 三、VGTE: Vision-Guided Temporally Enhanced Tokens（核心创新 1）

### 3.1 设计 intuition

VGTE 解决两个具体问题：

**问题 A: VLM 的 Recency Bias**

Transformer-based LLM 存在 **recency bias**（Peysakhovich & Lerer, 2023, https://arxiv.org/abs/2310.01427）——注意力倾向于关注序列末尾的 token。这本来是 long-context 下的缺陷，但 VTLA 反过来利用它：**把 vision input 放在 tactile 之后、action prediction 之前**，让 vision 离 action token 最近，从而被赋予更高权重。

为什么 vision 应该被强调？参考 Lee et al. 2020 (https://arxiv.org/abs/2007.12988) 的发现：在 vision-tactile 任务中，**visual observation 在任务早期阶段（approach phase）起主导作用**。Insertion 任务中，wrist camera 看到 hole 大致位置，这是 global cue；tactile 只在 contact 后才有信号，是 local cue。所以 vision 需要在 token 序列中"被记住"。

**问题 B: VLM 的 Temporal Reasoning 弱**

Qwen2-VL 系列 (https://arxiv.org/abs/2409.12191) 用 **3D convolution** 处理 video，适合 high-level semantic video understanding（如动作识别、视频问答）。但 tactile image sequences 有两个特殊性：
- **短时序**：只 4 帧
- **Low-level**：每帧是 contact pressure distribution，不是 semantic content

3D conv 在 4 帧 short clip 上 kernel 设计受限，且预训练 weight 是在 semantic video 上学的，与 tactile 物理量分布不匹配。

### 3.2 VGTE 架构

VTLA 的解法：
1. 把 4 帧 tactile images 拼成 **2×2 grid**（单张图）
2. 通过 **pre-trained ViT** 编码成 temporally-aware tokens
3. Vision image 单独通过 vision encoder
4. 所有 tokens + text tokens 一起喂入 LLM

这样 **temporal fusion 发生在 tokenization 之前**，绕过了 LLM 内部对 temporal 依赖处理的弱点。ViT 的 self-attention 在 2×2 grid 上能自然捕捉 4 帧之间的 spatial-temporal 关系。

### 3.3 Token 流水线

```
TactileLeft 4 frames ──┐
                       ├──> 2x2 grid ──> ViT ──> tactile tokens
TactileRight 4 frames ─┘                                │
                                                        ├──> LLM ──> action tokens
Wrist Image ──────────────> Vision Encoder ──> vision tokens
                                                        │
Text instruction ─────────> Tokenizer ──> text tokens ──┘
```

冻结部分：vision encoder + modality adapter；可训练部分：LLM only。这种 partial freeze 与 OpenVLA (https://arxiv.org/abs/2406.09246) 和 Prismatic VLMs (https://arxiv.org/abs/2402.08846) 的实践一致——保留 pre-trained visual grounding，避免 catastrophic forgetting。

---

## 四、Stage 1: Supervised Fine-Tuning with NTP Loss

### 4.1 公式

$$\mathcal{L}_{\mathrm{NTP}} = -\sum_{n=1}^{N} \log P_{\theta}(x_n \mid x_{<n})$$

变量解释：
- $N$: input token 序列总长度（包括 vision/tactile/text/action tokens）
- $x_n$: position $n$ 处的 ground-truth token
- $x_{<n}$: position $n$ 之前的所有 token（context）
- $\theta$: VTLA 可训练参数（LLM 部分）
- $P_{\theta}(\cdot)$: 模型在 softmax over vocabulary 上的概率

### 4.2 NTP 的本质问题

这里有一个关键的 insight：action `[-0.9, 0.4, 0.013]` 被 tokenize 成离散 tokens（比如 `[-`, `0`, `.`, `9`, `,`, ` `, `0`, `.`, `4`, ...）。每个 token 独立 cross-entropy loss。

但是 robotic control 是 regression 问题，存在**距离度量**：
- `[0.9, 0.4, 0.013]` 离 `[-0.9, 0.4, 0.013]` 远（x 方向完全相反）
- `[0.95, 0.4, 0.013]` 离 `[-0.9, 0.4, 0.013]` 也远，但物理上离 `[0.9, 0.4, 0.013]` 很近
- NTP 看不到这种拓扑结构，只看到 token-level 是否匹配

这就像把 regression 硬塞进 classification 框架。RT-2 也面临同样问题，但 RT-2 任务相对宽松（pick-and-place 容错大），insertion 任务对 action 精度极高（0.6 mm clearance），所以问题暴露明显。

### 4.3 训练配置

- Base model: **Qwen2-VL 7B** (https://arxiv.org/abs/2409.12191)
- Learning rate: $5 \times 10^{-4}$
- Batch size: 64
- Epochs: 10
- Framework: LlamaFactory (https://arxiv.org/abs/2403.13372)

---

## 五、Stage 2: DPO Preference Learning（核心创新 2）

### 5.1 Reformulation: Multi-label Regression via Preference

VTLA 把 action prediction 重新表述为 multi-label 问题，用 DPO 模拟 regression-like supervision。

**Preference data 构造**：
1. 用 SFT 后的 VTLA 在 training samples 上以不同 generation 配置（temperature / top-p / sampling）生成多个候选 actions
2. 计算每个候选与 ground-truth 的距离（L1）
3. 距离近的标记为 `chosen`，远的标记为 `rejected`
4. 总共 2,400 条 preference pairs（从两个 sampling 配置对比生成）

### 5.2 DPO Loss 公式

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_{\theta}(y_{\mathrm{chosen}} \mid x)}{\pi_{\mathrm{ref}}(y_{\mathrm{chosen}} \mid x)} - \log \frac{\pi_{\theta}(y_{\mathrm{reject}} \mid x)}{\pi_{\mathrm{ref}}(y_{\mathrm{reject}} \mid x)}\right)$$

变量含义：
- $x$: input tokens（vision + tactile + text）
- $y_{\mathrm{chosen}}$: preferred response（更接近 ground-truth 的 action）
- $y_{\mathrm{reject}}$: rejected response（更远的 action）
- $\pi_{\theta}$: trainable VTLA（从 SFT 模型初始化）
- $\pi_{\mathrm{ref}}$: frozen reference VTLA（也从 SFT 模型初始化，作为 KL anchor）
- $\beta > 0$: preference signal sharpness 控制系数。$\beta$ 大 → 偏好信号尖锐；$\beta$ 小 → 偏好信号平滑
- $\sigma(\cdot)$: sigmoid 函数
- $\log \frac{\pi_{\theta}}{\pi_{\mathrm{ref}}}$: log probability ratio，衡量 policy 相对 reference 的偏移

### 5.3 为什么 DPO 等价于 "Regression-like Supervision"

这是整篇 paper 最 clever 的部分。让我深入推演：

**原始 RLHF**：先训 reward model $r(x, y)$，再用 PPO 优化 $\pi_{\theta}$ 最大化 $r$ 同时 KL-约束到 $\pi_{\mathrm{ref}}$。

**DPO 闭式解**：DPO 推导出 reward 可以直接表达为 $r(x, y) = \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \text{const}$，从而绕过显式 reward model。

**VTLA 的应用**：chosen/reject 是按 **L1 距离 ground truth** 排序的，所以 DPO 在隐式地告诉模型 "更近的 action 应该有更高 reward"。这等价于在 action space 上施加一个 **单调下降的 implicit reward function** $r(y) \propto -\|y - y_{\mathrm{gt}}\|_1$。

这正好是 regression 的本质——让 $\pi_{\theta}$ 的输出分布向 ground truth 收敛，同时不 collapse 到单点（因为 DPO 是相对比较，不是绝对回归）。同时，KL anchor 防止 overfitting 到 sampled ground truth，这正是 paper 中提到的 "alleviating overfitting to ground-truth actions"。

可以联想到：
- **Reward-weighted regression** (Peters & Schaal, 2007): 把 RL 转成 weighted regression
- **Decision Transformer** (Chen et al., 2021): 用 conditional sequence modeling 处理 RL
- **CPC (Contrastive Predictive Coding)**: 用对比学习建模时序依赖

参考：DPO 原始 paper https://arxiv.org/abs/2305.18290

### 5.4 DPO 配置

- Learning rate: $5 \times 10^{-6}$（比 SFT 小 100 倍，防止破坏 SFT 学到的知识）
- Batch size: 32
- Epochs: 3

---

## 六、实验结果深度分析

### 6.1 Dataset 评估（Table 1）

| Method | ID GCR(%) | L1 x | L1 y | L1 rz | OOD GCR(%) | L1 x | L1 y | L1 rz |
|--------|-----------|------|------|-------|------------|------|------|-------|
| DP | 7.8 | 0.826 | 0.819 | 1.421 | 8.5 | 0.821 | 0.843 | 1.407 |
| VLA | 46.1 | 0.210 | 0.247 | 0.886 | 29.5 | 0.353 | 0.351 | 1.221 |
| TLA | 15.3 | 0.531 | 0.677 | 1.427 | 14.4 | 0.509 | 0.675 | 1.462 |
| **VTLA** | **47.3** | **0.181** | **0.224** | 0.904 | **31.2** | **0.305** | **0.324** | **1.136** |

关键观察：

1. **DP (Diffusion Policy) 几乎失败**：GCR 只有 7.8%。Diffusion policy 在 contact-rich 任务上需要大量数据和高维 action modeling，28000 samples 不够。这印证了 paper 的核心论点：contact-rich 任务需要 LLM-level reasoning。
   
2. **TLA 远逊于 VLA**（15.3% vs 46.1%）：单 tactile 无法提供 global 信息，机器人不知道 hole 在哪。这佐证了 vision 在 approach phase 的主导地位。

3. **VTLA 比 VLA 略好**（47.3% vs 46.1% ID, 31.2% vs 29.5% OOD）：tactile 提供的 marginal improvement。注意 ID 提升小（1.2%），OOD 提升大（1.7%），说明 tactile 主要帮助 generalization，因为 visual cue 在 OOD 形状下可能 ambiguous，tactile 提供 complementary contact state 信息。

4. **L1 rz 普遍较大**（0.9+）：旋转 prediction 比平移难，因为旋转的 visual cue 不明显，且 tactile 对旋转的敏感度低于平移。

### 6.2 不同 Clearance 的 Square Peg（Table 2）

| Method | 2.0mm Suc/Step | 1.6mm | 1.0mm | 0.6mm |
|--------|---------------|-------|-------|-------|
| DP | 42 / 2.47 | 32 / 2.63 | 28 / 4.85 | 22 / 3.54 |
| VLA | 100 / 2.28 | 98 / 3.24 | 90 / 3.28 | 80 / 5.55 |
| TLA | 94 / 3.27 | 90 / 3.60 | 80 / 4.97 | 80 / 5.48 |
| **VTLA** | **100 / 2.12** | **98 / 2.87** | **96 / 4.64** | **90 / 5.91** |

Intuition:
- **Clearance 越小，step 越多**：物理上 misalignment tolerance 越小，需要更多次 correction。
- **0.6mm clearance 下 VTLA step 反而比 VLA 多**（5.91 vs 5.55）：VTLA 更 "保守"，倾向于小步 correction，所以 success rate 高（90% vs 80%）。这是 success rate vs efficiency 的 trade-off。
- **DP 在小 clearance 下崩溃**：clearance 2.0mm 时 42%，0.6mm 时 22%。Diffusion 在数据稀疏的 contact-rich region 学不到精准 action distribution。

### 6.3 不同 Shape 的 0.6mm Clearance（Table 3）

| Method | Square | Triangle | Hexagon | Pentagon | Round |
|--------|--------|----------|---------|----------|-------|
| DP | 22/3.54 | 30/3.87 | 28/3.00 | 26/5.61 | 10/3.80 |
| VLA | 80/5.55 | 82/5.02 | 84/3.83 | 82/4.41 | 94/4.81 |
| TLA | 80/5.48 | 74/4.27 | 80/5.25 | 80/4.60 | 92/3.54 |
| **VTLA** | **90/5.91** | **88/4.53** | **90/4.68** | **92/3.97** | 92/4.74 |

关键 insight：**Round peg 上所有方法都 90%+**。这是因为 round shape 的 **geometric isotropy**——任意旋转角度都对齐，所以 task 本质上是 2-DOF (x, y) 而不是 3-DOF。这印证了：**任务的几何对称性决定了难度**，与 SVD 矩阵条件数的概念类似——对称性高 = effective DoF 低 = 容易。

Pentagon 是 5-fold 对称，Hexagon 是 6-fold，Square 是 4-fold。对称阶数越高，难度应该越低。但实验中 square 反而比 pentagon 难（VTLA: 90 vs 92）。我推测原因是：square 的 visual feature 更明显（边角清晰），pentagon 边角多但每条边短，visual cue 反而模糊，所以差异不大。

### 6.4 DPO Ablation（Table 4）

| Method | ID GCR | OOD GCR | OOD L1 x | OOD L1 y | OOD L1 rz |
|--------|--------|---------|----------|----------|-----------|
| VTLA w/o DPO | 47.5 | 27.0 | 0.349 | 0.367 | 1.223 |
| VTLA w/ DPO-1k | 47.5 | 31.4 | 0.305 | 0.324 | 1.137 |
| VTLA w/ DPO-2k | 47.3 | 31.2 | 0.305 | 0.324 | 1.136 |

Intuition:
- **DPO 主要提升 OOD generalization**（27.0% → 31.4%，+16%），ID 几乎不变。这非常符合 DPO 的本质：SFT 阶段模型 overfit 到 training distribution 的 ground truth，DPO 通过相对比较"软化"了这种 overfitting，让 model 学到 action space 的拓扑结构，而非死记硬背 specific actions。
- **Data scaling 边际效应为 0**（1k → 2k 几乎无提升）。作者推测：当前 preference data 只来自两种 sampling config，多样性不足。这与 RLHF 中的观察一致——preference data 的 diversity 比 quantity 更重要。
- **L1 误差 OOD 上降幅约 10%**：所有维度都改善，证明 DPO 不是只优化某一维。

可以联想到：
- **LLM 中 DPO 的 overfitting 现象**：https://arxiv.org/abs/2305.18290 也提到 DPO 容易 overfit preference data
- **DPO 的 length bias 问题**：在 VTLA 中不适用，因为 action 长度固定
- **IPO (Identity Preference Optimization)**：DPO 的改进版，更稳健
- **KTO (Kahneman-Tversky Optimization)**：不需要成对 preference，只需 binary signal

### 6.5 Real-world Sim2Real（Table 5, 6, 7）

Table 5 - 不同 clearance（real world）:
| | 1.6mm Suc/Step | 1.0mm | 0.6mm |
|-|---------------|-------|-------|
| VTLA | 100/1.60 | 100/1.95 | 95/4.31 |

Table 6 - 不同 shape 0.6mm（real world）:
| Shape | Suc/Step |
|-------|----------|
| Square (ID) | 95/4.31 |
| Triangle (ID) | 95/3.94 |
| Hexagon (ID) | 95/3.52 |
| Pentagon (OOD) | 100/1.85 |
| Round (OOD) | 100/5.2 |

Table 7 - 与 baseline 对比（real world 0.6mm）:
| Method | Triangle Suc/Step | Pentagon Suc/Step |
|--------|-------------------|-------------------|
| VLA | 90/4.06 | 100/2.3 |
| TLA | 30/2.00 | 40/1.88 |
| **VTLA** | **95/3.94** | **100/1.85** |

**最 striking 的发现**：

1. **VTLA real-world OOD shapes 100% success**：甚至比 ID 还好。这说明 Sim2Real gap 在 VTLA 上很小，且 OOD shape 反而可能因为几何特性更容易（如 pentagon 是 5-fold 对称）。

2. **TLA 在 real world 上崩溃**：sim 中 80%+，real 中只有 30-40%。这是 tactile sim2real 的经典难题——tactile sensor 的物理特性（硅胶老化、温度漂移、表面污染）很难在 sim 中精确建模。VTLA 通过 vision 弥补了 tactile 的 sim2real gap，vision 的 sim2real 相对成熟（domain randomization 有效）。

3. **VTLA vs VLA 效率**：pentagon 上 1.85 step vs 2.3 step。tactile 帮助 VTLA 更快找到正确方向，因为 contact 后的 tactile 信号直接指示了 misalignment 方向。

### 6.6 Poor Lighting 实验（Appendix C）

这是 paper 中一个隐藏的 gem。在 dim lighting 下，vision image 质量严重退化，VLA 模型无法识别 hole 位置（step 3-14 都在错误方向尝试），而 VTLA 通过 tactile 仍能完成。

这是 **multimodal fusion 的核心价值证明**：单一模态脆弱，融合模态鲁棒。可以联想到：
- **Sensor fusion 的 Bayesian 视角**：当一种 sensor 的 likelihood 退化时，另一种 sensor 的 prior 仍然能维持 posterior 估计
- **Robust statistics**：breakdown point 的概念——多模态提高系统的 breakdown point
- **Self-driving 中的 sensor redundancy**：camera + LiDAR + radar 的 redundancy 设计哲学

---

## 七、Hardware Setup

- **Robot arm**: UR3 6-DoF
- **Gripper**: Robotiq 2F-85
- **Wrist camera**: Intel RealSense D405
- **Tactile sensors**: 2× GelStereo 2.0 (https://arxiv.org/abs/2307.06148) 安装在 gripper 指尖
- **Tactile frame rate**: 20 FPS (ROS)
- **Misalignment 范围**: x ∈ [-2.5, 2.5] mm, y ∈ [-2.5, 2.5] mm, rz ∈ [-5°, 5°]

RealSense D405 是 wrist-mounted，提供 RGB-D 但 paper 中只用 RGB。GelStereo 2.0 是 visuotactile sensor，通过 stereo vision 重建硅胶层变形，输出 tactile image。每个 fingertip 一个，所以 left + right 两路 tactile。

---

## 八、关键 Insights 与延伸联想

### 8.1 Recency Bias 的反向利用

VTLA 把 recency bias 从 bug 变成 feature。这启发一个更 general 的设计原则：**在 LLM-based policy 中，token 顺序本身是一种 inductive bias**。可以联想到：
- **Prompt engineering 中的 "last sentence matters most"**
- **In-context learning 中的 recency effect** (https://arxiv.org/abs/2304.04893)
- **Chain-of-thought 的 reasoning step 排序**：关键 step 放最后

### 8.2 DPO 作为 Regression Proxy 的 Generalization

VTLA 用 DPO 处理 continuous action 的思路，可以推广到其他 LLM-based regression 任务：
- **Time series forecasting with LLMs**: 用 preference 学习数值序列的相对接近度
- **Code generation with continuous parameters**: hyperparameter tuning
- **Scientific computing**: 数值解的 preference optimization

但要注意 DPO 的局限：preference data 的生成依赖 ground truth，如果 ground truth 噪声大，preference signal 会被污染。

### 8.3 Tactile Sim2Real 的开放问题

TLA 在 real world 上掉到 30-40%，揭示 tactile sim2real 仍是 open problem。可能方向：
- **Tactile domain adaptation**: 类似 visual 的 CyCADA / Domain-Adversarial NN
- **Real-to-sim tactile matching**: 用 real tactile 数据校准 sim 的 silicone 物理参数
- **Tactile foundation models**: 像 AnyTouch (https://arxiv.org/abs/2502.12191) 试图统一不同 sensor 的 tactile representation

### 8.4 Action Tokenization 的局限

把 `[-0.9, 0.4, 0.013]` tokenize 成字符 token (`[-`, `0`, `.`, `9`, ...) 损失了数值结构。可能改进：
- **Bin-based action tokenization** (RT-2): 把每维离散化成 256 bins，每个 bin 一个 token
- **Continuous action head**: 在 LLM 之外接一个 MLP/diffusion head 专门输出 continuous action（π0 的思路 https://arxiv.org/abs/2410.24164）
- **Token embedding for numbers**: 学习数值 token 的 metric-aware embedding

### 8.5 与其他 VLA 工作的定位对比

| Model | Modalities | Action Head | 关键特点 |
|-------|-----------|-------------|---------|
| RT-2 | V+L | Token (bin) | VLA 范式开创 |
| OpenVLA | V+L | Token (bin) | 开源 7B |
| GR-1/2 | V+L | Token | video pre-training |
| RDT-1B | V+L | Diffusion | bimanual, 1B |
| π0 | V+L | Flow matching | 通用 robot control |
| TLA | T+L | Token | tactile-only |
| **VTLA** | **V+T+L** | **Token + DPO** | **contact-rich** |

VTLA 是少数同时融合 vision + tactile 的 LLM-based policy。可以对比：
- **Jones et al. 2025** (https://arxiv.org/abs/2501.04693): 把 heterogeneous sensors 接入 generalist policy
- **3D-VLA** (https://arxiv.org/abs/2403.09631): 3D scene理解

### 8.6 关于几何对称性的理论解释

Round peg 上所有方法都 90%+ 的现象，可以用 **SO(3) 群论** 解释：
- Round peg 的对称群是 $SO(2)$（连续旋转对称）
- Square peg 的对称群是 $\mathbb{Z}_4$（4 阶循环群）
- Pentagon 是 $\mathbb{Z}_5$
- Hexagon 是 $\mathbb{Z}_6$

对称阶数越高，对称群越大，task 的 effective configuration space 越小。这是为什么 round 最容易。如果模型能显式利用这种 symmetry（如 equivariant networks），可能进一步提升效率。可以联想到：
- **Equivariant diffusion policy** (https://arxiv.org/abs/2406.16639)
- **SE(3) equivariant manipulation**

### 8.7 关于 0.6mm Clearance 的物理意义

0.6mm clearance 是 paper 中最难设定。考虑 GelStereo sensor 的 noise level（通常 ~0.1mm），peg-hole 接触时 gripper 的位置 uncertainty（~0.5mm），这接近 **signal-to-noise ratio 的极限**。VTLA 在此设定下 90% success 说明：
- Vision 提供 ~mm 级粗定位
- Tactile 提供 ~0.1mm 级精修
- 两者协同才能在 0.6mm clearance 下成功

这是 **multi-resolution sensing** 的经典思想，类似于 coarse-to-fine optimization。

### 8.8 关于训练数据效率

28000 samples + 2 个 stage（SFT + DPO），用 7B 模型，达到 90%+ real-world success。这个数据效率令人印象深刻，对比：
- RT-2: web-scale + robotics 数据
- GR-2: web-scale video pre-training
- π0: 大规模 robot trajectories

VTLA 的小数据成功归因于：
1. 任务聚焦（只做 insertion）
2. VLM pre-training 提供 strong prior
3. Domain randomization 增强泛化
4. DPO 二次提炼

可以联想到 **data-efficient imitation learning** 的研究方向。

### 8.9 Limitations 的延伸思考

Paper 提到两个 limitations：
1. Tactile 用 vision encoder 编码，丢失 tactile 特有特征
2. Visual-tactile 深度融合在 LLM 中未充分探索

我的额外思考：
- **Tactile-language alignment 缺失**：类似 CLIP 的 image-text alignment，tactile-text alignment 数据集稀缺（Touch100k 是初期工作 https://arxiv.org/abs/2406.03813）
- **无 tactile-language pre-training**：VTLA 直接用 visual VLM，tactile 是 "hijack" vision encoder。如果先做 tactile-language pre-training 再 finetune，可能更好
- **Action chunking 缺失**：VTLA 每次预测 1 个 action（next-step），而 diffusion policy / π0 预测 action chunk（未来 N 步），前者在 contact-rich 任务中可能更稳健（reactive），但效率低
- **No closed-loop control**：模型预测一次 action，gripper 执行，再观测，再预测。这是 model-based MPC 的简化版，但 latency 高（LLM 推理慢）。可以联想到 **Reactive Diffusion Policy** (https://arxiv.org/abs/2503.02881) 的 fast-slow 思路

---

## 九、可能的改进方向

基于以上分析，我想到几个 potential 改进：

1. **Tactile-specific encoder**: 用 tactile 数据预训练一个 encoder（如 mask autoencoder on tactile videos），替代直接用 vision ViT
2. **Continuous action head**: 在 LLM 后接 diffusion / flow head，绕过 tokenization
3. **Tactile-language pre-training stage 0**: 在 SFT 之前，先用 Touch100k 等 tactile-text 数据做 alignment
4. **Action chunk prediction**: 预测未来 K 步 actions，提高 efficiency
5. **Equivariant design**: 在 action space 上施加 SE(2) equivariance
6. **Online DPO with human feedback**: 让人类标注 preference，而非用 L1 距离自动生成
7. **Multi-modal alignment loss**: 在 SFT 阶段加入 vision-tactile contrastive loss，强化跨模态对齐
8. **Tactile sim2real via GAN**: 用 GAN 把 sim tactile 转成 real tactile 风格

---

## 十、Summary

VTLA 的核心 contribution 是把 **LLM-based VLA 范式扩展到 contact-rich manipulation**，关键 tricks 是：

1. **VGTE tokens**：利用 VLM 的 recency bias，把 vision 放在 token 序列末尾，并用 ViT 在 tokenization 前融合 tactile 时序信息
2. **DPO as regression proxy**：把 continuous regression 转成 preference 比较，弥补 NTP loss 的 classification 本质
3. **Vision-tactile 互补**：vision 提供 global perception，tactile 提供 local contact state，两者在 poor lighting / small clearance 等极端情况下互补

实验结果证明：
- ID GCR 47.3%（vs VLA 46.1%, TLA 15.3%, DP 7.8%）
- OOD GCR 31.2%（+16% with DPO）
- Real-world 0.6mm clearance 95% success
- Real-world OOD shapes 100% success

未来方向：tactile-language pre-training、continuous action head、equivariant design、online preference learning。

整体看，这是一篇 **engineering-heavy 但 insight 清晰**的工作，把 LLM 的几个已知特性（recency bias, NTP classification limitation）转化为可利用的设计杠杆。对 contact-rich manipulation 这个具体问题，VTLA 提供了一个 solid baseline，但 tactile sim2real 和 action representation 仍是 open challenges。

---

参考资源：
- VTLA 项目主页: 见 paper 中链接
- TLA paper: https://arxiv.org/abs/2503.08548
- DPO paper: https://arxiv.org/abs/2305.18290
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- π0: https://arxiv.org/abs/2410.24164
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- GelStereo 2.0: https://ieeexplore.ieee.org/document/10309321
- Touch100k: https://arxiv.org/abs/2406.03813
- LlamaFactory: https://arxiv.org/abs/2403.13372
- Recency bias: https://arxiv.org/abs/2310.01427
