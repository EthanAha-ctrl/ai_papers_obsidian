---
source_pdf: PointVLA.pdf
paper_sha256: a95e4b96250e1cbc45a7c5bea3c94460ec4fc55c80a9cebdb156a2b5d892b521
processed_at: '2026-08-06T05:09:28-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 PointVLA

好，Andrej，我把之前那套技术拆解放一边，咱们像在 NeurIPS 走廊上聊天那样，把这 paper 从头到尾"翻译"一遍。

---

## 一句话版本

现在大家手里都有一堆 super 强的 VLA model（DexVLA、OpenVLA、π0），它们都是用海量 2D image 训出来的。你想让它们"看见" 3D 世界，但拿 3D 数据 retrain 一遍太贵，而且会 destroy 掉原来 2D 学到的东西。PointVLA 的做法：**VLM 完全 freeze，action expert 也基本 freeze，只在 action expert 内部几个"不重要"的 transformer block 上挂一个 super 轻量的 3D adapter**，像给一台已经训练好的 robot brain 装一个 3D 眼镜，brain 本身一点都不动。

---

## 这篇 paper 想解决的真正痛点

我先说背景。VLA model 的训练范式现在很清楚了：

- 你拿一个 VLM（Qwen2-VL 2B、PaliGemma 3B 这种）当 backbone；
- 在它后面接一个 action expert（diffusion transformer 或 LLM-style token predictor）；
- 用大规模 robot dataset（Open X-Embodiment、DROID、AgiBot-World）做 imitation learning。

OpenVLA 用了 4k hours data，π0 用 10k hours，DexVLA Stage 1 用 100 hours cross-embodiment。这些 2D data 规模很大，pretrain 出来的 representation 很值钱。

但问题来了：**robot 真的要跟物理世界交互，2D image 不够**。

举个例子，你给 robot 看一张 laundry detergent 的照片（屏幕显示的），从 robot 的 top camera 俯视，RGB 上几乎跟真 detergent 一模一样。2D VLA model（OpenVLA、DexVLA）直接被骗，会无限循环去抓那个不存在的瓶子。这就是 paper 里说的 **object hallucination**：robot 在物理 grounding 上 hallucinate 了。

再举个例子，训练时桌面 3mm foam，测试时换成 52mm foam（高出来 49mm）。2D VLA 全部 fail，因为它们对 z 轴 absolute depth 是盲的，只会去训练时见过的那个高度抓，把高出来的面包往下按，然后抓空。PointVLA 因为有 point cloud，直接读出 absolute height，gripper 自动调整，成功抓起来。

这两个例子直接说明：**3D 信息对 robot manipulation 是刚需**。但问题是——

### 为什么不能直接 fine-tune 一个 3D VLA？

最 naive 的做法是把 point cloud tokenize 成 3D token，塞进 LLM input（像 [LLaVA-3D](https://arxiv.org/abs/2409.18125) 那样）。听起来合理，但有两个坑：

**坑 1：catastrophic forgetting**。你的 2D data 是几万 hours，你的 3D data 是几十 hours。你拿小数据 fine-tune，原来的 2D-text aligned embedding 会被 pull 走，模型语言理解退化、视觉 grounding 退化。

**坑 2：domain gap**。2D pixel 和 3D geometric structure 之间本来就有大 gap，小数据 fine-tune 时模型容易 overfit 到 point cloud 的 surface artifacts（比如某个传感器的 noise pattern），generalization 反而变差。

paper 里有个很直接的证据：在 RoboTwin benchmark 上，把 RGB 加进 DP3（一个纯 3D diffusion policy），结果**很多任务性能反而下降**。比如 Bottle Adjust 50 demos 时，纯 DP3 是 70.7%，加了 RGB 变成 27.7%。这说明粗暴的 multi-modal fusion 反而 hurt performance。

所以 PointVLA 的核心 paradigm：**3D 不是 primary input modality，而是 complementary conditioning signal**。3D 不参与 VLM 的 reasoning，只参与 action expert 的 refinement。

---

## 整体架构，我用流水账讲一遍

Robot 有 3 个相机（2 wrist + 1 top），其中一个 top camera 是 RealSense L515（solid-state LiDAR，depth 精度比 D435i 高很多），专门采集 point cloud。

数据流分两路：

**2D 路径（保留原 VLA）**：
- 3 个 RGB view + language instruction 喂给 Qwen2-VL 2B（DexVLA 用的 VLM）；
- VLM 输出 state embedding，传给 action expert；
- Action expert 是 ScaleDP，1B 参数的 diffusion transformer，32 个 block，输出 14 维 action × chunk size 50。

**3D 路径（新加的）**：
- L515 给出 point cloud $P \in \mathbb{R}^{N \times 3}$；
- 一个轻量的 hierarchical CNN encoder 把 $P$ 编码成 $z_{3D} \in \mathbb{R}^{C_{3D} \times N'}$；
- 这个 $z_{3D}$ 通过一个 bottleneck 对齐到 action embedding 维度；
- 然后在 action expert 的 **selected blocks**（比如 block 11-15）上，用一个小 MLP adapter 把 $z_{3D}$ 加到 hidden state 上。

公式上：

$$
h_i' = h_i + \text{MLP}_i^{\text{adapter}}(z_{3D})
$$

其中 $h_i$ 是 block $i$ 原本输出的 hidden state，$i \in \{11, 12, 13, 14, 15\}$。$\text{MLP}_i^{\text{adapter}}$ 是新加的可训练参数，$z_{3D}$ 是 point cloud embedding。VLM 完全 frozen，action expert 的其他 27 个 block 也 frozen，只这 5 个 adapter 训练。

---

## 最有意思的部分：Skip Block Analysis

这是 paper 里我觉得最 clever 的实验。问题很简单：action expert 有 32 个 block，3D 信号该注入哪几个？

naive 答案是"全部都注入"，但这有两个问题：
1. 计算太贵；
2. 每个 block 都改，原有的 2D pretrain 表征肯定会被破坏。

作者的思路：**找那些"本来就没那么重要"的 block，在它们身上动手**。

怎么找？做个 skip experiment：在 DexVLA 的 shirt folding 任务上，每次 inference 时 skip 一个 block（让 input 直接 bypass 这个 block），看任务还能不能完成。

结果很 striking：

- **Block 0-10（前 11 个）**：skipping 任何一个，gripper 都无法闭合，任务直接 fail。这些 block 是 critical 的；
- **Block 11-31**：skipping 单个 block，performance 几乎不掉。

这说明 diffusion transformer 的前 1/3 block 学的是 fundamental action primitive（"planner"），后 2/3 是 refinement（"refiner"），存在大量冗余。

进一步，从 block 11 开始连续 skip：
- Skip 1-4 个：还能跑；
- Skip 5 个以上：开始 fail。

所以 5 是个 sweet spot。作者就选 block 11-15 这 5 个 block 来 inject 3D 信号。

### 为什么这个结果直觉上 make sense？

Diffusion transformer 的 forward 是 iterative denoising。早期 block 处理 noisy action $\hat{a}_T$，提取 coarse trajectory plan；晚期 block refine 细节。3D 信息（深度、3D 形状）本质上属于 fine-grained spatial refinement，不是 coarse plan。所以把它注入 refiner block，正好补上 2D 模型对 z 轴、对 3D shape 的盲区，而不会破坏 coarse plan。

这跟 LLM 里发现的"早期 layer 编码 syntax、晚期 layer 编码 semantics"是 reverse 过来的——diffusion action expert 里是"早期 plan、晚期 refine"。这个观察我觉得对未来 VLA 的 layer-wise analysis、distillation、pruning 都有启发。

---

## 3D Encoder 的设计选择

这里有个反直觉的发现：用预训练的 3D encoder（PointNet++、Point-MAE、Point-BERT）反而 hurt performance。因为这些 encoder 的 inductive bias 来自 classification / segmentation，跟 manipulation 需要的几何信息不 match。

所以作者用一个超级简单的 hierarchical CNN：
- 几层 1D convolution over points（point-wise MLP）；
- 中间夹 max pooling 降点密度；
- 最后把每层 conv 的输出 concat 起来，形成 multi-scale 3D 表征。

低层抓 edge、curvature，高层抓 scene layout。整个 encoder 参数量很小，跟 iDP3 的设计类似。作者也承认这个 encoder 不强，留作 future work。我觉得这是 paper 的一个 limitation——如果换更强的 3D encoder（比如 Point Transformer、Point-VicC、3D foundation model），性能应该还能往上推。

---

## 实验里最"哇"的几个 finding

### 1. Few-shot multi-tasking

4 个 task，每个 20 demos，共 80 demos。这种数据量对 Diffusion Policy 来说直接 fail，因为 20 demos 不足以让 action space 在 multi-task 间 disentangle。ScaleDP-1B 即使把参数量加到 1B，也没救。DexVLA 因为有 100 hours cross-embodiment pretrain，能学，但 performance 比 PointVLA 略低。PointVLA 最好，说明 3D 信息提升了 sample efficiency。

### 2. Long-horizon packing

这个 task 难度拉满：bimanual UR5e 从 **moving conveyor belt** 抓 2 袋 laundry detergent 装箱，还要封箱。任务分 5 个 sub-step。

看 Avg. Len.：
- Octo: 0.27
- OpenVLA: 0.36
- Diffusion Policy: 0.36
- ScaleDP-1B: 0.72
- DexVLA: 1.72
- **PointVLA: 2.36**

特别看 Step 5（封箱）：只有 PointVLA 能到 2/11，DexVLA 是 0/11。封箱这个动作对 z 轴深度极度敏感，2D VLA 在最后这步就崩了。

### 3. Real-vs-Photo Discrimination

这是最 demo-friendly 的实验。把真 detergent 换成屏幕显示的照片。
- 2D VLA：被骗，进入 infinite grasping loop（一直试图抓空气）；
- PointVLA：point cloud 显示那个空间是空的（depth 是平的），识别出"这是 photo"，拒绝抓取。

这个实验第一次明确把 "robot hallucination" 提出来。在 LLM 里 hallucination 是说错话，在 VLA 里 hallucination 是对一个根本不存在的物理 object 做动作，直接安全风险。3D 作为 "reality anchor" 切断了 RGB 的 hallucination 路径。

### 4. Height Adaptability

训练 foam 3mm，测试 foam 52mm。差 49mm 的 z 轴 offset。2D 方法全 fail（去训练高度抓，把面包按下去）。PointVLA 直接成功，因为 point cloud 给出 absolute z，gripper 自动调整。

这个实验说明一个 deep point：2D 模型学到的 grasp pose 是 camera frame 里的 2D location，对 absolute metric depth 是盲的。3D 直接解出 metric depth。

### 5. RoboTwin Simulation

8+ 个 task，20 和 50 demos 两种设置。PointVLA 在所有 task 上都 best。特别值得注意：

- **DP3 (Point Cloud + RGB) 经常比 DP3 (纯 Point Cloud) 还差**。这非常 counterintuitive——加了 RGB 信息反而 hurt。这正好印证了 paper 的 thesis：粗暴 multi-modal fusion 会有 negative transfer，需要 surgical integration；
- **Diffusion Policy 在 20 demos 基本全 fail**，2D 方法 sample efficiency 差；
- **PointVLA 在 sample efficiency 上完胜**。

---

## 给你的几个直觉（Karpathy-style mental model）

### 直觉 1：PointVLA 是 "3D prosthetic limb"

DexVLA 是一个 2D vision 的 brain，已经学会抓东西、叠衣服。PointVLA 不改 brain（VLM frozen），不改 spinal cord（action expert 大部分 block frozen），只在运动皮层末梢（block 11-15）装一个 3D prosthetic limb。这个 prosthetic 提供 depth + 3D shape 的 supplementary signal，让原本的 motor plan 在执行时被 3D 校准。

### 直觉 2：为什么 additive injection 比 token concat 好

如果把 point cloud tokenize 成 K 个 3D token 塞进 LLM input：
- LLM 的 self-attention 会被新 token dilute，2D visual tokens 的 attention weight 重新分配；
- 这是 catastrophic forgetting 的主因。

Additive injection 在 action expert 内部做，绕开 LLM 的 attention，VLM 完全 untouched。3D 信息只在 action prediction 阶段起作用，相当于在 "perception → action" 的 mapping 上加 3D prior，而不在 perception 上加 3D observation。这是个关键的 paradigm choice。

### 直觉 3：为什么是 block 11-15，不是 block 0-4

3D 信息属于 fine-grained spatial refinement（gripper 应该到哪个 z 高度），属于 coarse plan（抓哪个物体）应该交给 2D VLM 来做。coarse plan 已经在 2D pretrain 里学得很好了，3D 只需要补 fine-grained 部分。所以注入 refiner block（晚期 block）是 "spatially informative but plan-agnostic" 的最佳位置。

### 直觉 4：与 LoRA 的精神同源

LoRA：低秩矩阵注入 frozen LLM，学新任务；
PointVLA：3D adapter 注入 frozen action expert，学新 modality。

两者都是 freeze backbone + 加 lightweight module，目标都是"不要破坏 pretrain 的好东西"。PointVLA 可以看成 "modality PEFT"。

---

## 我看到的 Limitations

1. **3D encoder 太弱**：hierarchical CNN 是 2020 年水平。如果换 Point Transformer、3D foundation model，性能应该还能涨；
2. **Skip block analysis 只在 shirt folding 上做**，被默认 generalize 到所有 task，没验证 cross-task consistency；
3. **只有 5 个 block 上加 injection**，对 3D-heavy task（透明物体、镜面物体、occlusion 严重场景）可能 capacity 不够；
4. **Single L515 camera 的 point cloud**，occlusion 时信号 degraded。Multi-view point cloud fusion 没做；
5. **Real-vs-photo 只是 binary 判断**，没给 ROC、没量化，只有 qualitative evidence。

---

## 延伸联想（让 mind 跑一会）

### 联想 1：把这套 paradigm 套到 π0 上

[π0](https://arxiv.org/abs/2410.24164) 的 action expert 也是 flow matching based diffusion transformer，而且更大。理论上 π0 的 action expert 冗余 block 更多，injection 空间更丰富。PointVLA 的 skip block analysis 方法可以直接迁移过去，做 π0-3D injection 版本。

### 联想 2：multi-modality injection 的 roadmap

按 PointVLA 的 paradigm，force / tactile / audio / thermal 都可以用同样的方式 inject。Robot foundation model 会变成 "2D-centric core + 多个 modality adapter" 的 modular 架构。每个 modality 有自己的 lightweight adapter，挂在不同 block 上。这是未来 robot foundation model 的一个可能 roadmap。

### 联想 3：Robot hallucination 作为新研究方向

Real-vs-photo 实验第一次明确把 robot hallucination 提出来。这跟 LLM hallucination 类比，但发生在物理 grounding 层。未来可以：
- 用 force/tactile 做 reality check；
- 用 multi-view consistency 做 hallucination detection；
- 用 active perception（move camera to verify）做 hallucination correction；
- 用 3D scene graph 做 grounding verification。

### 联想 4：Skip block analysis → layer-wise interpretability for VLA

这个分析揭示 diffusion transformer 内部有 planner / refiner 分化。这对 RLHF、distillation、pruning 都有启发：
- Pretraining 阶段 prune 后期 block 的 capacity，省计算；
- Fine-tune 阶段只在 refiner block 上 tune，保留 planner；
- Distillation 阶段从 planner block 蒸出 small model。

### 联想 5：Metric depth foundation model 替代 raw point cloud

[Depth Anything V2](https://arxiv.org/abs/2406.09414) 这种 metric depth foundation model 可以直接从 RGB 输出 metric depth。未来 route 是：用 Depth Anything 输出的 metric depth map 作为 condition，不用 LiDAR。这会让 PointVLA 在没有 L515 的部署场景也能用。

### 联想 6：与 EgoExo4D / 4D 数据的关系

[EgoExo4D](https://egoexo4d-data.org/) 提供 paired ego/exo video + 3D。可以用来 pretrain 3D encoder，再做 PointVLA-style injection。这是个数据 scaling 的方向。

### 联想 7：3D data scaling law

Paper 没探索 3D data 数量对 PointVLA 的影响。如果 3D data 增加 10x、100x，5 个 block 的 injection capacity 够不够？这是个开放问题，类似 Chinchilla 对 LLM 的 scaling law 分析。

### 联想 8：与 Lift3D 的对比

[Lift3D (Jia et al., 2024)](https://arxiv.org/abs/2411.18623) 把 2D pretrained model 升级到 3D，但需要 full fine-tune。PointVLA 用 adapter 逃避 full fine-tune，效率高但 capacity trade-off 不明。两条路线未来可能 converge：先 Lift3D-style lift，再 PointVLA-style inject。

### 联想 9：Diffusion Policy 3D 扩展路线的收敛

DP → DP3 → iDP3 这条线是 "pure 3D imitation learning"。PointVLA 是 "2D VLA + 3D injection"。两条线未来可能 converge：用 iDP3 的 3D encoder 作为 PointVLA 的 3D encoder，可能进一步提性能。

### 联想 10：Robot foundation model 的 modular future

整个 robotics community 在朝 "foundation model + adapter" 走。PointVLA 是这个方向的 3D 版本。未来 vision：一个 universal robot foundation model，根据 deployment scenario 灵活挂载 modality adapter（3D、force、tactile、audio），所有 adapter 都是 lightweight、surgical、不破坏 core。这是 modular robot foundation model 的雏形。

---

## Reference

- [PointVLA project page](https://pointvla.github.io)
- [DexVLA paper](https://arxiv.org/abs/2502.05855)
- [OpenVLA paper](https://arxiv.org/abs/2406.09246)
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [DP3 paper](https://arxiv.org/abs/2403.03954)
- [iDP3 paper](https://arxiv.org/abs/2410.10803)
- [RoboTwin](https://arxiv.org/abs/2409.02920)
- [Qwen2-VL](https://arxiv.org/abs/2409.12191)
- [LLaVA-3D](https://arxiv.org/abs/2409.18125)
- [3DVLA](https://arxiv.org/abs/2402.09469)
- [SpatialVLA](https://arxiv.org/abs/2501.15830)
- [Lift3D](https://arxiv.org/abs/2411.18623)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [3D Diffuser Actor](https://arxiv.org/abs/2402.10885)
- [Prismatic VLMs](https://arxiv.org/abs/2402.07865)
- [SUGAR: 3D pretraining for robotics](https://arxiv.org/abs/2312.06747)
- [AgiBot-World](https://agibot-world.com/)
- [Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [FFN-SkipLLM (layer skipping for LLM)](https://arxiv.org/abs/2404.03865)
- [Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [EgoExo4D dataset](https://egoexo4d-data.org/)
- [Flux distillation](https://github.com/black-forest-labs/flux)
- [Discrete Policy](https://arxiv.org/abs/2501.17701)

---

简单总结一句给 Andrej：**PointVLA = "不要 retrain foundation model，给它挂一个 3D 小耳机"**。这个 paradigm 简单、cost-effective、且实验上有几个 very compelling 的 demo（real-vs-photo、height adaptability）。它对未来 robot foundation model 的 modular 设计我觉得有启发意义。如果你之后要做 VLA + 3D 的实验，这个 paper 的 skip block analysis 方法可以直接拿来用。

---

# PointVLA 深度解析：把 3D 世界注入预训练 VLA

好的 Andrej，我们来逐层剥开这篇 paper。这篇工作的核心 insight 我先一句话框定：**如何在不破坏预训练 2D VLA 表示的前提下，把 3D point cloud 当作"侧条件信号"嫁接进 action expert**。整个方法其实是一种"外科手术式的 modality injection"，让我从 motivation 到 architecture 到 experiment 一一拆开。

---

## 1. Motivation：为什么不能直接 fine-tune 3D 进 VLA？

### 1.1 数据规模的非对称性

VLA 模型（OpenVLA、π0、DexVLA）预训练在 thousands of hours 的 2D robot data 上，比如 OpenVLA 用了 4k hours open-source data，π0 用 10k hours proprietary data，DexVLA Stage 1 跑 100 hours cross-embodiment data。相对地，3D robot dataset（point cloud + trajectory）的规模要小 1-2 个数量级。

如果直接把 point cloud tokenize 后塞进 LLM（像 LLaVA-3D 那样），会出现两个灾难性问题：

1. **Catastrophic forgetting**：2D visual-text aligned embedding space 会被小规模 3D 数据 pull 走，原有的语言理解、视觉 grounding 全部退化；
2. **Domain gap**：2D pixel 与 3D geometric structure 之间存在巨大 domain gap，小数据 fine-tune 时模型会过拟合到 point cloud 的 surface artifacts，反而 generalization 变差。

### 1.2 PointVLA 的 paradigm shift

作者把 3D point cloud 当作 **complementary conditioning signal**，而非 primary input modality。这意味着：

- VLM backbone（Qwen2-VL 2B）完全 freeze，2D 视觉 token 路径不动；
- Action expert（ScaleDP 1B 的 diffusion transformer）也大部分 freeze；
- 只在 action expert 内部 selected blocks 上加 lightweight 3D adapter，做 additive injection。

这种设计本质上是 **Parameter-Efficient Fine-tuning (PEFT) 思想向 multi-modal conditioning 的延伸**，类似于 LoRA / Prefix tuning，但目标不是让模型学新任务，而是让模型 access 一个新 modality。

---

## 2. Architecture 拆解

### 2.1 整体数据流

参考 [PointVLA project page](https://pointvla.github.io)，整体框架可以用一个公式描述：

$$
a_t = f_\theta(x_{2D}, l, h_{t-1}) + \Delta a_\phi(x_{3D}, h_{t-1})
$$

其中：
- $a_t \in \mathbb{R}^{14}$ 是 bimanual robot 的 action chunk（chunk size = 50，所以实际是 $a_t \in \mathbb{R}^{14 \times 50}$）；
- $f_\theta$ 是 frozen action expert（ScaleDP，1B 参数，32 个 diffusion transformer blocks）；
- $x_{2D}$ 是 multi-view RGB（3 个相机：2 wrist + 1 top）；
- $l$ 是 language instruction；
- $h_{t-1}$ 是 action expert 内部 hidden states；
- $\Delta a_\phi$ 是 3D injector 的贡献，参数 $\phi$ 远小于 $\theta$。

### 2.2 Point Cloud Encoder

作者参考 [iDP3 (Ze et al., 2024)](https://arxiv.org/abs/2410.10803) 和 [DP3 (Ze et al., 2024)](https://arxiv.org/abs/2403.03954)，发现预训练的 3D encoder（比如 PointNet++、Point-MAE）反而 hurt performance，因为它们的 inductive bias 来自分类/分割任务，与 manipulation 的几何需求 misaligned。

所以他们用了一个**简化的 hierarchical convolutional encoder**：

$$
z_{3D} = \text{Concat}\left[\text{Conv}_1(P), \text{Pool}(\text{Conv}_2(P)), \text{Pool}^2(\text{Conv}_3(P)), \ldots\right]
$$

其中：
- $P \in \mathbb{R}^{N \times 3}$ 是原始 point cloud（RealSense L515 采集），$N$ 是点数；
- $\text{Conv}_i$ 是 1D convolution over points（point-wise MLP）；
- $\text{Pool}$ 是 max pooling，逐步降密度；
- 最终 $z_{3D} \in \mathbb{R}^{C_{3D} \times N'}$，$N' \ll N$，$C_{3D}$ 是 channel 维度。

低层 conv 抓 low-level geometric features（边缘、曲率），高层 conv 抓 high-level scene layout（物体大致位置、桌面高度）。Multi-level concat 是关键的 multi-scale 设计。

### 2.3 Action Embedding Bottleneck

Action expert 输出的 action embedding 维度可能很大（chunk size 50 × action dim 14 → 700 维序列），直接与 $z_{3D}$ 交互计算量高。所以加了一个 bottleneck：

$$
\hat{z}_{action} = W_{down} \, z_{action}, \quad W_{down} \in \mathbb{R}^{C_{3D} \times C_{action}}
$$

把 action embedding 压缩到与 $z_{3D}$ 对齐的 channel 维度 $C_{3D}$，方便后续 additive fusion。这个 $W_{down}$ 是 trainable 的。

### 2.4 Injection Block

对于被选中的 block $i \in \mathcal{S}$（$\mathcal{S}$ 是 selected block indices，比如 {11, 12, 13, 14, 15}），injection 操作为：

$$
h_i' = h_i + \text{MLP}_i^{\text{adapter}}(z_{3D})
$$

其中：
- $h_i$ 是 block $i$ 原本输出的 hidden state；
- $\text{MLP}_i^{\text{adapter}}$ 是一个小型 adapter MLP（参数量远小于 block $i$ 本身）；
- 加法是 element-wise broadcast 到 sequence 维度。

**为什么用 additive 而非 concat / cross-attention？** 我的猜测：additive 不改变 hidden state 的 shape，对预训练 transformer 的 attention pattern 干扰最小。Cross-attention 会引入新的 attention matrix，破坏原模型的 attention 分布；concat 会改变 positional encoding 的范围。Additive 类似于 residual connection，是最"温和"的 modality fusion。

---

## 3. Skip Block Analysis：怎么选哪些 block 注入？

这是这篇 paper 最有意思的实验，本质上是 **layer redundancy analysis**，类似 [FFN-SkipLLM (Jaiswal et al., 2024)](https://arxiv.org/abs/2404.03865)、[Sliced Recursive Transformer (Shen et al., 2022)](https://arxiv.org/abs/2205.00513) 和 flux distillation [Flux](https://github.com/black-forest-labs/flux) 中的 layer skipping 思路。

### 3.1 实验设置

在 DexVLA 的"shirt folding"任务上做分析。DexVLA 的 action expert 是 1B 参数、32 个 diffusion transformer blocks。Metric 是 average score：把长任务切成多步，按完成 step 数打分。

### 3.2 Single-block skipping

每次 skip 1 个 block，跑一遍推理：

- **Block 0–10（前 11 个）**：skipping 任何一个都导致 gripper 无法闭合，任务失败。这些 block 编码了 fundamental action primitives，是 critical 的；
- **Block 11–31**：skipping 单个 block 几乎不影响 performance。

这暗示 diffusion transformer 的后 21 个 block 存在大量冗余，类似于 LLM 中的 "late-layer redundancy" 现象。

### 3.3 Multi-block consecutive skipping

从 block 11 开始，连续 skip 1, 2, 3, 4, 5 个 block：

- Skip 5 个连续 block（11–15）后模型才开始 fail；
- 所以策略：**把 3D injection 加在 block 11–15（或类似的 5 个 block 区间）**，因为这 5 个 block 本身的输出对结果影响小，被 3D 信号"overwrite"也不会破坏 2D pretraining。

### 3.4 我的 interpretation

这个实验揭示了一个很 deep 的现象：**Diffusion transformer 的早期 block 学习 low-level action token prediction，后期 block 学习 refinement**。前 11 个 block 是 "planner"，后 21 个 block 是 "refiner"。Refiner 是冗余的，所以可以被 3D 信号 hijack 而不破坏 planner。

这跟 [Prismatic VLMs](https://arxiv.org/abs/2402.07865) 中发现"LLM 后期 layer 负责 high-level semantics、早期 layer 负责底层 token"的观察有点像，但 reversed：在 diffusion action expert 里，**早期是 high-level plan，后期是 low-level refinement**。

---

## 4. 实验详解

### 4.1 Real-world setups

两种 bimanual embodiment：

| Setup | Arms | DOF | Cameras | Freq |
|---|---|---|---|---|
| Bimanual UR5e | 2× UR5e + Robotiq gripper | 14 | 2× wrist D435i + 1× top L515 | 15Hz |
| Bimanual AgileX | 2× 6-DOF AgileX | 14 | 2× wrist + 1× base | 30Hz |

Point cloud 用 RealSense L515（注意：L515 是 solid-state LiDAR，比 D435 的 stereo depth 精度更高，更适合 manipulation）。Chunk size = 50。

### 4.2 Few-shot multi-tasking

4 个任务，每个 20 demos，共 80 demos：

1. **ChargePhone**：抓 smartphone 放到无线充电板；
2. **WipePlate**：双手协同，一只手拿 sponge，另一只手拿 plate，sponge 擦 plate；
3. **PlaceBread**：抓面包放盘子，下面垫 foam（为后面 height adaptability 实验铺垫）；
4. **TransportFruit**：抓朝向随机的 banana 放中央 box。

结果（Fig 6 中）：PointVLA > DexVLA > ScaleDP-1B > Diffusion Policy。Diffusion Policy 直接 fail，因为 20 demos 太少，action representation space 在 multi-task 间 entangle（这与 [Discrete Policy (Wu et al., 2025)](https://arxiv.org/abs/2501.17701) 的 observation 一致：multi-task 需要 disentangled action space）。

### 4.3 Long-horizon packing task

Bimanual UR5e 从 moving conveyor belt 抓 2 袋 laundry detergent 装箱。任务分 5 个 sequential sub-step：

| Method | Step1 | Step2 | Step3 | Step4 | Step5 | Avg.Len |
|---|---|---|---|---|---|---|
| Octo | 1/11 | 1/11 | 0 | 0 | 0 | 0.27 |
| OpenVLA | 2/11 | 1/11 | 0 | 0 | 0 | 0.36 |
| Diffusion Policy | 2/11 | 1/11 | 0 | 0 | 0 | 0.36 |
| ScaleDP-1B | 4/11 | 2/11 | 0 | 0 | 0 | 0.72 |
| DexVLA | 2/11 | 5/11 | 1/11 | 1/11 | 0 | 1.72 |
| **PointVLA** | 3/11 | 1/11 | 1/11 | 2/11 | 2/11 | **2.36** |

PointVLA 比 DexVLA 高 0.64 平均长度。关键看 Step 5（最后封箱）：只有 PointVLA 能稳定到 2/11，DexVLA 是 0/11。这说明 3D 信息在 long-horizon 末端 step 的精细操作（比如闭合纸箱 flaps）特别重要，因为这些动作对 z 轴深度敏感。

### 4.4 Real-vs-Photo Discrimination

这是 paper 最炫的实验。把真 laundry detergent 换成屏幕上显示的 laundry detergent 照片。从 egocentric top camera 看，照片与真物在 RGB 上几乎 indistinguishable；但从 exocentric 视角（侧面相机）和 3D point cloud 看，明显是平面的。

- **OpenVLA / DexVLA**：被 RGB 欺骗，反复试图抓"空气中的照片"，进入 infinite grasping loop；
- **PointVLA**：point cloud 显示对应空间是空的（depth 是平的），识别出 "no real object"，拒绝抓取。

这本质上是一个 **object hallucination 问题**，类似 LLM 中的 hallucination，但发生在 VLA 的物理 grounding 层。Point cloud 作为 "reality anchor" 直接切断 RGB-only 模型的 hallucination 路径。

### 4.5 Height Adaptability

训练时 foam 厚 3mm，测试时换 52mm foam（差 49mm）。

- 2D 方法（OpenVLA / DP / ScaleDP-1B / DexVLA）：在训练高度尝试 grasp，把高出来的面包"按下去"再抓，失败；
- **PointVLA**：point cloud 直接给出面包的 absolute height，gripper 自动调整到正确 z 高度。

这个实验的 intuition：2D 模型学到的 grasp pose 是 camera frame 内的 2D location，对 z 轴 absolute position 是盲的；3D point cloud 提供 metric depth，直接解出 $z_{object}$。

### 4.6 RoboTwin Simulation

[RoboTwin (Mu et al., 2024)](https://arxiv.org/abs/2409.02920) 是 14-DOF bimanual benchmark。在 20 和 50 demos 两种设置下测了 8+ 任务：

| Task | DP3 (PC) | DP3 (PC+RGB) | DP | PointVLA |
|---|---|---|---|---|
| Block Hammer Beat (20) | 47.7 | 44.7 | 0.0 | **61.2** |
| Blocks Stack Easy (50) | 17.0 | 17.0 | 0.0 | **24.3** |
| Bottle Adjust (50) | 70.7 | 27.7 | 24.7 | **74.5** |
| Container Place (50) | 74.0 | 58.3 | 16.3 | **81.3** |
| Mug Hanging Easy (50) | 14.0 | 2.0 | 0.0 | **19.1** |

观察：

1. **DP3 (PC+RGB) 经常比 DP3 (PC) 还差**，比如 Bottle Adjust 50 demos：27.7 vs 70.7。这验证了 paper 的核心 thesis：直接把 RGB 塞进 3D 模型会 hurt performance，因为 domain gap 和数据稀缺导致 overfit；
2. **PointVLA 在所有任务上都 best**，包括最难的双瓶抓取、mug hanging 等；
3. **DP 在 20 demos 下基本全 fail**，纯 2D 模型数据效率极低。

---

## 5. Intuition Building：把所有 piece 串起来

让我给你一个 mental model，Andrej：

### 5.1 类比：PointVLA 是 "3D prosthetic limb" 给 2D VLA

DexVLA 是一个有 2D vision 的 "身体"，已经学会抓东西、叠衣服。PointVLA 不改造它的 brain（VLM），也不改造它的 spinal cord（frozen action expert 大部分 block），只在它的"运动皮层末梢"（block 11-15）装上一个 3D prosthetic limb。这个 prosthetic limb 提供 depth + 3D shape 的 supplementary signal，让原本的 motor plan 在执行时被"3D 校准"。

### 5.2 为什么 additive injection 比 token concat 好？

如果 point cloud tokenize 成 K 个 3D token 塞进 LLM：
- LLM 的 self-attention 会被新 token dilute，2D visual tokens 的 attention weight 重新分配；
- 这是 catastrophic forgetting 的主因。

Additive injection 在 action expert 内部做，绕开 LLM 的 attention，保持 VLM 完全 untouched。3D 信息只在 action prediction 阶段起作用，相当于在 " perception → action" 的 mapping 上加 3D prior，而非在 perception 上加 3D observation。

### 5.3 为什么 skip block 11-15 而非 0-4？

Diffusion transformer 的 forward 过程是 iterative denoising：
- 早期 block 处理 noisy action $\hat{a}_T$，提取 coarse trajectory；
- 晚期 block refine 细节。

3D 信息最适合在 "refinement" 阶段注入，因为 3D 主要提供 fine-grained spatial 信息（gripper 应该精确到哪个 z 高度），不是 coarse plan（应该抓哪个物体）。所以注入晚期的冗余 block 是 "spatially informative but plan-agnostic" 的最佳位置。

### 5.4 与其他 modality injection 工作的关系

- **[LLaVA-3D (Zhu et al., 2024)](https://arxiv.org/abs/2409.18125)**：把 3D token 塞进 LLM input，属于 "early fusion"；
- **[3D Diffuser Actor (Ke et al., 2024)](https://arxiv.org/abs/2402.10885)**：在 policy 内做 3D conditioning，属于 "mid fusion"；
- **PointVLA**：在 action expert 的 redundant block 做 late-stage additive injection，属于 "late fusion + surgical"。

三者代表 modality fusion 的光谱，PointVLA 选 late fusion 因为它要 preserve 2D pretraining。

### 5.5 与 PEFT 的关系

PointVLA 可被视为 "modality PEFT"：
- LoRA：低秩矩阵注入 frozen LLM，学新任务；
- PointVLA：3D adapter 注入 frozen action expert，学新 modality。

两者都是 freeze backbone + 加 lightweight module，但目标不同：LoRA 学 task adaptation，PointVLA 学 modality expansion。

### 5.6 Limitations 我看出来的（paper 没明说）

1. **Point cloud encoder 是手工设计的 hierarchical CNN**，没用 Point Transformer / Point-VicC / PointGPT 这些更强的 3D encoder。作者承认这是 future work；
2. **Skip block analysis 是 task-specific**（在 shirt folding 上做的），但被默认 transfer 到所有任务，没有验证 cross-task consistency；
3. **3D injector 只在 5 个 block 上加**，可能对某些 3D-heavy 任务（如透明物体抓取）insufficient；
4. **Point cloud 来自 single L515 camera**，occlusion 严重时 3D signal 会 degraded。Multi-view point cloud fusion 没做；
5. **Real-vs-photo 是 binary 判断**，没量化 ROC，只有 qualitative evidence。

---

## 6. 我的延伸思考 & 相关联想

### 6.1 与 Ego3D / 4D foundation models 的关系

最近 [SUGAR (Chen et al., 2024)](https://arxiv.org/abs/2312.06747) 等 3D pretraining for robotics 工作正在兴起。PointVLA 的 paradigm（3D as auxiliary signal）可以延伸到：未来如果有大规模 3D robot dataset，可以把 3D encoder 也 pretrain，然后再 inject。这会形成 "2D VLA + 3D pretrained encoder + adapter" 的三段式架构。

### 6.2 与 π0 / π0-FAST 的关系

[π0 (Black et al., 2024)](https://arxiv.org/abs/2410.24164) 的 action expert 也是 flow matching / diffusion based。PointVLA 的 skip block analysis 方法可以直接迁移到 π0 上，做 3D injection。理论上 π0 的 action expert 更大，冗余 block 更多，injection 空间更丰富。

### 6.3 与 SpatialVLA / 3DVLA 的对比

- [3DVLA (Huang et al., 2024)](https://arxiv.org/abs/2402.09469)：从 3D scene 生成 + VQA + action 的 unified model，但只在 sim 上验证；
- [SpatialVLA (Qu et al., 2025)](https://arxiv.org/abs/2501.15830)：在 VLA 内学 spatial representation，仍以 2D 为主，加 3D-aware representation；
- **PointVLA**：保留 2D 主干，3D 作为 condition。

三者走的是不同路线：3DVLA 是 "3D-native from scratch"，SpatialVLA 是 "2D + spatial encoding"，PointVLA 是 "2D + 3D adapter"。

### 6.4 与 Lift3D 的对比

[Lift3D (Jia et al., 2024)](https://arxiv.org/abs/2411.18623) 把 2D pretrained model 升级到 3D，但需要 full fine-tune。PointVLA 用 adapter 逃避 full fine-tune，效率更高但 capacity trade-off 不明。

### 6.5 与 Diffusion Policy 的 3D 扩展路线

- DP → [DP3](https://arxiv.org/abs/2403.03954) → [iDP3](https://arxiv.org/abs/2410.10803) 这条线是 "pure 3D imitation learning"，没有 VLM backbone；
- PointVLA 是 "2D VLA + 3D injection"，两条路线未来可能 converge：用 DP3/iDP3 的 3D encoder 作为 PointVLA 的 3D encoder，可能进一步提性能。

### 6.6 Skip Block Analysis 的延伸：Layer Importance for Action vs Perception

这个实验揭示 diffusion transformer 内部有 "planner block" 和 "refiner block" 的分化。这对 RLHF、pretraining 优化都有启发：

- Pretraining 阶段可以 prune 后期 block 的 capacity，省计算；
- Fine-tune 阶段只在 refiner block 上 tune，保留 planner；
- Distillation 阶段可以从 planner block 蒸出 small model。

### 6.7 Robotic Hallucination 作为一个新方向

Real-vs-photo 实验第一次正式提出 "robot hallucination" 问题。这在 LLM/VLM 已有大量研究（object hallucination、grounding hallucination），但 robot hallucination 的定义还模糊。PointVLA 提供了一个物理 grounding 的方式来 mitigate。未来可以：
- 用 force/tactile sensor 做 "reality check"；
- 用 multi-view consistency 做 hallucination detection；
- 用 active perception（move camera to verify）做 hallucination correction。

### 6.8 Height adaptability 与 metric depth

PointVLA 解决了 2D VLA 的 "absolute depth blindness"。这与 [Depth Anything V2](https://arxiv.org/abs/2406.09414) 等 metric depth foundation model 的兴起呼应。未来 route 可以是：把 metric depth foundation model 的输出（而非 raw point cloud）作为 condition，更轻量。

### 6.9 与 EgoExo4D / 4D 数据的关系

[EgoExo4D](https://egoexo4d-data.org/) 等数据集提供 paired ego/exo video + 3D。可以用来 pretrain 3D encoder，再做 PointVLA-style injection。

### 6.10 Scaling laws of 3D data

Paper 没探索 3D data scaling 对 PointVLA 的影响。如果 3D data 增加 10x、100x，PointVLA 的 injection capacity（5 个 block）是否 sufficient？这是个开放问题，类似 Chinchilla 对 LLM 的 scaling law 分析。

---

## 7. 总结：PointVLA 的贡献与启示

**核心贡献**：
1. 提出 3D as auxiliary conditioning 而非 primary input 的 paradigm；
2. 用 skip block analysis 找出 safe injection points，最小化对 2D pretraining 的破坏；
3. 在 real robot 上验证了 3 个 2D VLA 无法解决的 setting：real-vs-photo、height adaptability、long-horizon packing；
4. Few-shot 多任务下比 SOTA 2D 方法（OpenVLA、DexVLA、Diffusion Policy）显著更强。

**Intuition**：PointVLA 是一个 surgical modality expansion framework，类似 LoRA 之于 LLM finetuning。它通过 (a) freeze backbone, (b) lightweight adapter on redundant blocks, (c) additive late-stage fusion 这三招，把 3D 世界嫁接到 2D VLA 上而不破坏它。

**启示给 robotics community**：未来我们要 "augment" foundation model 而非 "retrain" 它。3D、force、audio、tactile 这些 modality 都可以用 PointVLA-style paradigm 逐个 inject。Robot foundation model 会变成一个 2D-centric core + 多个 modality adapter 的 modular 架构。

**启示给 VLM community**：LLM 的 skip block analysis 思路可以延伸到 VLA / VLM。Understanding which layers encode what kind of knowledge (perception vs planning vs refinement) 对 model interpretability 和 efficient finetuning 都重要。

---

## Reference Links

- PointVLA project: https://pointvla.github.io
- DexVLA: https://arxiv.org/abs/2502.05855
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- DP3: https://arxiv.org/abs/2403.03954
- iDP3: https://arxiv.org/abs/2410.10803
- RoboTwin: https://arxiv.org/abs/2409.02920
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- LLaVA-3D: https://arxiv.org/abs/2409.18125
- 3DVLA: https://arxiv.org/abs/2402.09469
- SpatialVLA: https://arxiv.org/abs/2501.15830
- Lift3D: https://arxiv.org/abs/2411.18623
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- 3D Diffuser Actor: https://arxiv.org/abs/2402.10885
- Prismatic VLMs: https://arxiv.org/abs/2402.07865
- SUGAR: https://arxiv.org/abs/2312.06747
- AgiBot-World: https://agibot-world.com/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- ScaleDP (DexVLA action expert): https://arxiv.org/abs/2502.05855
- FFN-SkipLLM: https://arxiv.org/abs/2404.03865
- Depth Anything V2: https://arxiv.org/abs/2406.09414
- EgoExo4D: https://egoexo4d-data.org/
- Flux.1: https://github.com/black-forest-labs/flux
- Discrete Policy: https://arxiv.org/abs/2501.17701

如果你想 deep dive 某一块（比如 skip block analysis 的统计学解释、3D encoder 的具体 conv 结构、long-horizon evaluation metric 的设计），告诉我，我可以再展开。
