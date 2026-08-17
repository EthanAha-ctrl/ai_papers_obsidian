---
source_pdf: OPG-Policy.pdf
paper_sha256: 2b21beb315da9d3a786fbe201872bc9596135b77d92d95a354491f726dd375df
processed_at: '2026-08-06T01:16:14-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 OPG-Policy

## 一句话总结

想象你桌上堆了一堆杂物，老板让你："把那个绿色方块拿出来。"但绿色方块上面压着东西，旁边也堵着东西，你只看到它露出一角。你怎么拿？你会先把挡路的东西推开一点，再伸手抓。OPG-Policy 就是教机器人这么干，关键是——它先"脑补"出绿色方块被挡住的部分长什么样，再决定往哪推、怎么抓。

---

## 这事儿为什么难

机器人面对一堆东西要抓特定目标，传统方法只看得到目标露出来的那一点像素。就好比你只看到冰山一角，却要判断整座冰山的形状、判断从哪下手推最划算。这导致两个问题：

第一，**推的方向瞎选**。你不知道目标完整轮廓，就不知道推哪个方向的障碍物最能"露出来"目标。可能推了三下都没把关键遮挡物推开。

第二，**抓的位置不准**。你用 visible mask 的 centroid 去抓，但那个 centroid 是偏的——目标一半被压住，visible centroid 在边缘上，gripper 伸过去根本夹不到重心。

之前的 paper 都在这两个问题上挣扎。GTI 用 Bayesian inference 慢慢猜 target 在哪，平均要 3.1 次 action。GE-GRASP 狂采样一堆 push candidate 让 evaluator 打分，稀疏场景还行，密集场景采样效率崩了。

OPG-Policy 的 insight 很直接：**与其让 policy 从 partial observation 猜，不如直接给它 full observation**。amodal segmentation 就是干这事的——给你一张图，它告诉你"虽然这块被挡住了，但目标其实延伸到这里"。

相关 link: Amodal Segmentation 综述 https://arxiv.org/abs/2104.01148

---

## Amodal Segmentation 是什么"魔法"

人类视觉有个本事：看到一个杯子被书挡住一半，你脑子里自动"补全"了杯子的完整形状，你不会觉得杯子就是半圆形的。这叫 amodal perception。

在 CV 领域，amodal segmentation 就是训练一个网络做同样的事：输入 RGB 图，输出两个 mask——一个是 visible mask（你能看到的部分），一个是 amodal mask（完整的形状，包括被挡住的）。

OPG-Policy 用的 amodal seg 模块叫 UOAIS-Net (Back et al. ICRA 2022)。它对 unseen object 有泛化能力，意思是不需要为每个新物体重训。这点很关键，因为 deployment 时你不可能给每个 household object 都标注 amodal mask。

link: https://github.com/kynng94/UOAIS-Net

### 训练数据怎么来

这里有个聪明的 trick。在 simulation 里，物体被挡住时你只看得到 visible mask。但你怎么知道 amodal mask？答案是：**作弊**。

作者搞了一个 comparison area，跟 workspace 一样大。把同一个物体按同一个 pose 单独放在 comparison area 里——没遮挡。用 SAM (Segment Anything) 分割两边的图像：workspace 得到 visible mask，comparison area 得到 full mask。full mask 经过坐标对齐就是 amodal mask 的 ground truth。

这个 pipeline 太聪明了。等于利用 simulation 的"上帝视角"自动生成 amodal seg 的训练数据，零人工标注成本。

link: SAM https://github.com/facebookresearch/segment-anything

---

## 整个 pipeline 怎么跑

我用一张图来说明：

```
相机拍 RGB-D 图
    ↓
Amodal Seg 模块 → target amodal mask
    ↓
RGB-D + amodal mask 投影成 heightmap (I, D, A)
    ↓
旋转 16 个角度 (22.5° step)
    ↓
DenseNet121 提特征
    ↓
分两路:
  ├── Push Q-Net → 推的 Q-map (每个像素一个 Q value)
  └── Grasp Q-Net → 抓的 Q-map
    ↓
取两个 Q-map 的最大值 q_p, q_g
    ↓
加上 domain knowledge (遮挡率, border 拥挤度, 失败次数)
    ↓
Coordinator MLP → 输出"该推还是该抓"
    ↓
Robot 执行
```

### 为什么要旋转 16 个角度

CNN 默认不懂 rotation。同一个 push 动作，旋转 90° 后理论上 Q-value 应该一样。但 plain CNN 会输出不同值。

两种解法：用 rotation-equivariant CNN (比如 GroupCNN)，或者直接 data augment 旋转输入。这篇 paper 选了后者——把 heightmap 旋转 16 次分别喂网络，相当于让网络"看到" 16 个朝向的同一场景，最后取 max。

这招 Zeng et al. 2017 (Cornell grasping) 用过，简单粗暴但有效。每个 rotated version 对应一个固定的 end-effector 朝向，Q-map 上每个 pixel 就是一个候选 action。

link: https://arxiv.org/abs/1703.06107

### Heightmap 是什么

把 3D 点云 top-down 投影成 2D 图。每个 pixel 取该位置最高点的 RGB 和高度。这是 robotic manipulation 的经典做法，把 3D 问题压成 2D pixel-wise regression。

好处：网络结构简单（2D CNN 就行），inference 快。
坏处：丢失 3D 信息，stack 场景（target 上方压着东西）处理不了——push 只能水平方向。

---

## DQN 在干嘛

Push Q-Net 和 Grasp Q-Net 各自输出一张 pixel-wise 的 Q-map。Q-map 上每一点 $(i,j)$ 对应一个 action：位置 $(i,j)$，朝向是当前 rotation index 对应的角度。Q-value 意思是"执行这个 action 后，未来能拿多少 reward"。

为什么用 DQN 而不用 policy gradient？因为 action space 是 discrete + pixel-aligned，Q-map 形式天然适合 CNN 输出。policy gradient 在这种 structured action space 上不直观。

### TD Error 公式

$$\delta_t = Q(s_t, a_t; \theta_t) - [R_t + \gamma \cdot \max_{a'} Q(s_{t+1}, a'; \theta_t')]$$

变量逐个讲：
- $s_t$：当前 state，三元组 $(I, D, A)$——color, depth, amodal 三个 heightmap
- $a_t$：执行的 action，比如 push 到位置 (100, 150) 朝向 45°
- $\theta_t$：online network 参数
- $\theta_t'$：target network 参数（DQN 标配，定期从 $\theta_t$ 同步过来，稳定训练）
- $R_t$：reward，分情况见下面
- $\gamma = 0.5$：future discount，意思是只看大约 2 步内的 reward
- $\max_{a'}$：下一 state 上所有 action 里取最大 Q-value

为什么 $\gamma = 0.5$ 这么小？因为 push-grasp 任务 short-horizon——推一两下就能抓到，不需要 long-horizon credit assignment。$\gamma$ 太大反而 variance 爆炸。

### Huber Loss

$$L = \begin{cases} \frac{1}{2}\delta_t^2 & \text{if } |\delta_t| \leq 1 \\ |\delta_t| - \frac{1}{2} & \text{otherwise} \end{cases}$$

小误差用 MSE，大误差用 L1。DQN 标配。避免突然一个 +1 reward 把 gradient 炸飞。

---

## Reward 怎么设计

这是这篇 paper 最精妙的部分。

### Grasp reward

$$R = \begin{cases} 0.25 & \text{grasp 位置在 amodal mask 内} \\ 1 & \text{真的抓起来了} \\ 0 & \text{其他} \end{cases}$$

第一个 0.25 是 dense reward，目的是解决 cold-start。早期 grasp Q-net 还没学会"target 在哪"，如果只有"真的抓起来"才给 reward，那 99% 时间 reward = 0，网络学不动。

给个 0.25 表示"至少你 grasp 位置选对了"，给 Q-net 一个学习方向的梯度信号。这是 classic reward shaping。

### Push reward

$$R = \begin{cases} 0.5 & \text{target 遮挡率下降 ≥ 0.1} \\ 1 & \text{push 后 grasp Q value 超过 } T_g \\ 0 & \text{其他} \end{cases}$$

第一个 reward：直接奖励"让 target 更露出来"的 push。threshold 0.1 是 hard cutoff，意思是下降 0.05 不给奖励，可能让网络学到"要么大力推要么不推"的策略。

第二个 reward：push 之后下一 state 的 best grasp Q value 超过阈值 $T_g$，给 1.0。这等价于"push 创造了好抓的机会"。

### Adaptive threshold $T_g$ 是什么

$$T_g = \beta \cdot T_g + (1 - \beta) \cdot Q_g$$

- $Q_g$：push 后下一 state 的 best grasp Q value
- $\beta = 0.95$：decay factor
- $T_g$：exponential moving average (EMA)

为什么要 adaptive？因为 grasp Q-net 自己也在训练，它的 Q value 整体 scale 会变化。训练早期 Q value 普遍低（比如 0.2），训练后期普遍高（比如 0.7）。如果用固定 threshold 0.5，早期所有 push 都不达标，后期所有 push 都达标——supervision signal 失效。

EMA 让 $T_g$ 跟着 grasp Q-net 的 evolution 漂移，始终保持"高于平均水平的 grasp state 才奖励 push"的语义。这类似 batch normalization 的思路——保持 reward distribution 稳定。

link: EMA 原理 https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

---

## Coordinator 是什么

Coordinator 是个 3-layer MLP，决定"现在该 push 还是 grasp"。它的输入是 6 维特征：

$$y = f_{mlp}(q_p, q_g, o, a_b, a_n, f_c)$$

逐个解释：

- $q_p$：best push Q value，就是 push Q-map 上的 max
- $q_g$：best grasp Q value，grasp Q-map 上的 max
- $o$：target 遮挡率 $= \frac{\text{被挡住面积}}{\text{完整面积}}$
- $a_b$：target border 被遮挡比例 $= \frac{o_b}{t_b}$
- $a_n$：被遮挡 border 相对 target 大小 $= \frac{o_b}{t_m}$
- $f_c$：grasp 失败次数

其中 $o_b, t_b, t_m$ 的定义很有意思。$m_b$ 是 target amodal mask 边界向外扩 10 pixel 的一个 strip 环。$o_b$ 是这个 strip 中被遮挡的像素数，$t_b = \sum m_b$ 是 strip 总像素，$t_m = \sum M_a$ 是 target 完整 mask 像素数。

### 为什么用 border 而不用 interior

intuition：如果 target 中心被一个小物体压着，但边缘是空的，gripper 可以从侧面斜着伸进去抓。但如果 target 周围一圈都被堵死，gripper 根本没空间插入，必须先 push。

所以 border occlusion 比 interior occlusion 更阻碍 grasp。$a_b$ 量化"target 周围被堵死程度"，$a_n$ 进一步 normalize——同样 10 pixel 被堵，对大 target 和小 target 影响不同。

### $f_c$ 的作用

如果 grasp 连续失败 3 次，$f_c = 3$，coordinator 会偏向输出 push。这是 online adaptation，避免 OPG-Policy-no-coordinator 那种"反复在同一位置 grasp 失败"的浪费。

### 训练信号

Coordinator 用 BCE loss：
$$L = -(\bar{y} \log y + (1-\bar{y}) \log(1-y))$$

$y$ 是 coordinator 输出 sigmoid，$\bar{y}$ 是 ground truth。Ground truth 来自 grasp outcome：grasp 成功 → $\bar{y} = 1$，失败 → $\bar{y} = 0$。

注意：push outcome 不直接监督 coordinator。只有 grasp 才更新 coordinator。这意味着 coordinator 学的是"什么时候应该 push 让后续 grasp 成功"，是 indirect credit assignment。

---

## Ablation 给我们什么启示

Table I 最有说服力：

| Method | Avg Success | Avg Attempts |
|--------|-------------|--------------|
| GTI | 64.66% | 3.10 |
| GTI-amodal | 50.33% | 3.34 |
| GE-GRASP | 78.33% | 2.74 |
| OPG-Policy-no-coordinator | 77.00% | 2.67 |
| OPG-Policy | 82.66% | 2.61 |

### GTI-amodal 比 GTI 还差

这是最重要的 finding。GTI 用 Bayesian inference 猜 target 位置，你给它 amodal mask 等于把答案塞给它，但 GTI 的 Bayesian 机制反而被 prior 扰乱了。说明**不能简单 plug-and-play perception module**，perception 和 policy 必须 co-design。

### Coordinator 贡献 5.66% success rate

OPG-Policy 82.66% vs OPG-Policy-no-coordinator 77.00%。少了 coordinator 的版本只会"哪里 Q 高做哪里"，遇到 grasp 失败的情况反复重试。Coordinator 用 $f_c$ 失败计数 + border occlusion 信息，知道"该换 push 了"。

### Amodal seg 贡献 motion efficiency

Table II real-world：OPG-Policy 7.0 attempts vs GE-GRASP 7.62 vs GTI 8.42。amodal 让 agent 知道"push 哪个方向能让 target 露出最多"，避免无效 push。Success rate 提升不算太大，但 efficiency 提升明显。

---

## Training Curriculum 的精髓

| Iteration | Targets | Obstacles | Target 选择 | Action 选择 |
|-----------|---------|-----------|-------------|-------------|
| 0-1000 | 7 | 3 | random | ε-greedy |
| 1000-3000 | 7 | 3→8 | random | coordinator |
| 3000-5000 | 7 | 13 | most occluded | coordinator |
| 5000+ | 7 | 18 | most occluded | coordinator |

两个维度同时在 escalate：

**环境复杂度**：obstacle 数量从 3 涨到 18
**任务难度**：target 选择从 random 变成"挑最 occluded 的"

前 1000 iter 为什么用 ε-greedy？因为早期 grasp Q-net 还没学会"target 在哪"，coordinator 会基于错的 $q_g$ 做决策，一直选 push 不 grasp。ε-greedy 强制探索，保证 grasp Q-net 有训练数据。这是 warm-start 思路。

5000+ iter 后 prefer most occluded object as target，这是 hard example mining——专挑难训的，提升 extreme occlusion 下的能力。

---

## Real-world 实验的关键卖点

**Model trained in simulation, deployed in real without fine-tuning**。这是 robotic manipulation 最难的事——sim-to-real gap。

设备：
- UR10 + ROBOTIQ gripper
- Intel RealSense D435i (640×480)
- YCB object set

结果：

| Method | C-Test Success | G-Test Success | C-Test Attempts | G-Test Attempts |
|--------|----------------|----------------|------------------|------------------|
| GTI | 65% | 80% | 8.42 | 7.75 |
| GE-GRASP | 82.5% | 85% | 7.62 | 7.30 |
| OPG-Policy | 85% | 90% | 7.0 | 6.65 |

G-Test (generalization test) 用 12 个没见过的 household object，OPG-Policy 仍 90% success，说明 UOAIS-Net 的 unseen object generalization 能力撑住了。

注意：作者为了公平，给 GTI 和 GE-GRASP 也加了 UOAIS-supplied visible masks（替换它们原本弱的 perception module）。即使这样 OPG-Policy 还赢，说明优势不仅来自 perception，更来自 policy 与 amodal 信息的协同设计。

link: Sim-to-real survey https://arxiv.org/abs/2203.02825

---

## 个人觉得哪里可以更好

### Amodal seg 误差传播

UOAIS 在 sim 训练，real 上预测精度未知。如果 amodal mask 偏了（比如 target 形状预测不准），coordinator 的 $o, a_b, a_n$ 全部失真，可能误导决策。

改进方向：uncertainty-aware amodal seg，输出 mask 概率图，让 Q-net 处理 uncertainty。

link: Bayesian SegNet https://arxiv.org/abs/1511.04280

### 2D heightmap 丢 3D 信息

Stack 场景处理不了。target 上方压着物体，push 只能水平推。可以加 3D occupancy field 表示。

link: 3D grasping survey https://arxiv.org/abs/2203.09949

### Push primitive 太简单

论文 push 应该是 fixed-length straight push。但实际可能需要 curve push、pivot、squeeze。可以 learn push primitives。

link: Learning Push Strategies https://arxiv.org/abs/1804.09968

### Long-horizon 不足

$\gamma = 0.5$ 视野短。如果需要 push 5 次才能 grasp，agent 看不到那么远。可以换 hierarchical RL 或 model-based planning (MCTS like Huang et al. [19])。

link: MCTS for manipulation https://arxiv.org/abs/2105.11905

### Diffusion Policy 替代 DQN

DQN 输出 Q-map 是 single-modal（一个 max），但实际中可能有多个有效 push 方向。Diffusion Policy 可以输出 multi-modal action distribution。

link: Diffusion Policy https://diffusion-policy.cs.columbia.edu/

### Foundation Model 融合

SAM 已经用了，但还可以加 DINOv2、CLIPSeg。如果 amodal seg 换成 SAM-based promptable segmentation，可以处理开放集 target。

link: DINOv2 https://dinov2.metademolab.com/

---

## 给做 VLA / VLM4Robotics 的人的启示

最大的 lesson：**perception module 和 policy module 必须 co-design**。

简单 plug-and-play perception 通常 suboptimal，甚至会反向损害（GTI-amodal 案例）。这点对现在流行的"用 GPT-4V 当 perception，挂个 action head"的 VLA 思路特别有警示意义。Perception 输出的 representation 必须与 policy 消费的方式匹配，否则信息会被浪费或误用。

OPG-Policy 的 coordinator 设计就是个好例子——它不直接用 amodal mask，而是把 amodal mask 加工成 domain knowledge features (occlusion rate, border ratio)，再喂给 MLP。这中间的"feature engineering"才是 amodal 信息真正发挥作用的关键。

link: RT-2 https://robotics-transformer2.github.io/
link: VLA survey https://arxiv.org/abs/2307.05873

---

## 最后的 Intuition

OPG-Policy 这篇 paper 我读下来的感觉是：**不炫技，每一步都有 ablation 支撑**。Amodal seg 是 prior knowledge 注入，coordinator 是 action type 的元决策，adaptive reward threshold 是训练稳定性的小工程。三个东西叠起来，每个都不可少。

它没发明什么新算法，DQN、UOAIS、DenseNet 都是现成的。它的贡献是**正确的系统组合**——把 amodal perception 注入到 push-grasp RL 的正确位置，配合正确的 reward 设计和正确的 coordinator 特征。

这种 paper 价值在于告诉社区：在 robotic manipulation 里，模块化设计 + careful co-design 比单一算法创新更实用。给你一个 pretrained perception model，你怎么把它和 RL policy 嫁接好，是一个比"我发明了新 RL 算法"更重要的问题。

---

# OPG-Policy 深度技术讲解

## 1. Paper 概述与 Problem Framing

这篇 paper 来自中山大学 Hui Cheng 团队，发表于 IEEE RAL 风格的机器人顶会场景。核心问题可以精确表述为：在 **dense clutter** 中执行 **goal-oriented grasping**，目标物体被其他物体 **partially occluded**，机器人需要生成一串 push/grasp 序列来 retrieve 该 target。

传统方法的痛点：之前的 policy（GTI [2]、GE-GRASP [4]、Xu et al. [1]）都只依赖 **visible mask**，但 target 在被遮挡时，policy 只看到了 target 的"冰山一角"，导致 push 方向选择不优、grasp 位置不准、动作序列冗长。

OPG-Policy 的核心 insight：把 **amodal segmentation** 作为 prior knowledge 显式注入到 RL framework 中，让 Q-network "看得到"目标被挡住的部分，从而 push/grasp 决策更智能。这本质上是把 perception 与 policy 解耦但协同训练——amodal seg 是 frozen 的 prior module，policy 是 RL 训练的 agent。

相关 web links:
- UOAIS-Net (Back et al., ICRA 2022): https://github.com/kynng94/UOAIS-Net
- SAM (Segment Anything): https://github.com/facebookresearch/segment-anything
- GTI (Yang et al., grasping the invisible): https://arxiv.org/abs/1910.05328
- GE-GRASP (Liu et al., IROS 2022): https://arxiv.org/abs/2207.07998
- Push-Grasp Synergy (Xu et al.): https://arxiv.org/abs/2107.02111
- DenseNet121 paper: https://arxiv.org/abs/1608.06993
- V-REP/CoppeliaSim: https://www.coppeliarobotics.com/
- Yale-CMU-Berkeley (YCB) Object Set: https://www.ycbbenchmarks.com/

---

## 2. 与 Previous Work 的 Critical Comparison

让我把几个 baseline 的核心机制对比一下，这有助于理解 OPG-Policy 的 contribution 落在哪里：

### 2.1 GTI (Grasping the Invisible, Yang et al. RAL 2020)
GTI 是这篇文章最直接的对比对象。它用 **Bayesian exploration** 来推断被遮挡 target 的位置——也就是先 push 几下试探，然后通过 Bayesian update 推断 target 的 hidden 状态。当 target 露出来后，DQN-based critic 与 coordinator 协同选 action。
- 关键缺陷：Bayesian exploration 是 **exploration-heavy**，需要很多 push 才能"发现"target 的真正位置。GTI-amodal 在 Table I 中反而比 GTI 差（64.66% → 50.33%），这说明简单地把 amodal mask 拼到 GTI 上没有意义，因为 GTI 的 Bayesian framework 本质是为"未知目标位置"设计的，amodal 给出 prior 后反而打乱了其 inference 假设。

### 2.2 GE-GRASP (Liu et al., IROS 2022)
GE-GRASP 用 **sampling-based** 方法：先采样大量 push/grasp candidates，再用 evaluator 打分。这避免了 DQN 训练的不稳定性，但代价是 sampling 效率低，且 evaluator 对 occlusion 不敏感。
- Table I 中 GE-GRASP 在 15 objects 上略胜（97% vs 96%），但在 30 objects hard 场景下显著落后（57% vs 68%）。这很直观——sample-based 方法在 dense clutter 中很难采样到"正确的 push 方向"，因为 search space 太大。

### 2.3 Xu et al. (RAL 2021)
Tri-phase training + relabeling strategy。把"误抓到的物体"作为新 target 来最大化 data efficiency。这是 data augmentation 的思路，但没有解决 visible mask 信息不足的根本问题。

### 2.4 OPG-Policy 的 Positioning
OPG-Policy 把 amodal segmentation 作为 **state augmentation**，本质上是把"target 完整轮廓"作为先验输入 Q-network。这与 GTI 的 Bayesian inference 思路完全不同——GTI 试图 **infer** hidden state，OPG-Policy 直接 **predict** hidden state（通过 amodal seg 模块），然后让 policy 在 enriched observation 上做决策。

intuition：可以把 amodal seg 看作是一个 "shape prior oracle"，给 Q-network 喂了完整形状信息，policy 就不需要再从 visible mask 中"猜"target 的几何中心、朝向、与 obstacle 的接触面，决策的 entropy 直接下降。

---

## 3. Amodal Segmentation 模块详解

### 3.1 选用的 backbone: UOAIS-Net
UOAIS (Unseen Object Amodal Instance Segmentation, Back et al. ICRA 2022) 采用 **hierarchical occlusion modeling**，核心思想是分层建模遮挡关系。

UOAIS 的网络结构大致为：
- **Backbone**: HRNet+FPN 或 ResNet-FPN，输出 multi-scale features
- **Occlusion Order Prediction (OOP)**: 一个分支预测 pairwise occlusion order (哪个物体在前)
- **Amodal Mask Prediction**: 一个分支预测每个 instance 的完整轮廓
- **Visibility Mask Prediction**: 一个分支预测可见部分

输出是三个 mask：visibility mask $M_v$、amodal mask $M_a$、occlusion order。

UOAIS 在 unseen objects 上能 generalize，这正是 OPG-Policy 需要的，因为目标物体在 deployment 时是未知的。

### 3.2 训练数据生成 pipeline（这是 OPG-Policy 的核心 contribution 之一）

这个 trick 很有意思：因为 simulation 中物体完全可见，可以直接获得 ground truth amodal mask，但需要一个聪明的 pipeline：

1. 在 simulation workspace 中放置 occluded 配置，得到 RGB-D 图像，目标物体部分被遮挡。
2. 同时建立一个 **comparison area**（与 workspace 同尺寸），把同一物体按同一 pose 单独放在那里——完全无遮挡。
3. 用 **SAM (Segment Anything)** 在两个区域分别分割：得到 occluded 图像中的 visible mask $M_v^{ws}$，和 comparison area 中的 full mask $M_f^{comp}$。
4. **Filtering & matching algorithm**：把 $M_f^{comp}$ 的坐标系对齐到 workspace 坐标系（通过 pose 一致性），然后 amodal mask = $M_f^{comp}$ 对齐后的 mask（这就是 GT），visible mask = $M_v^{ws}$。

关键 intuition：因为 simulation 中我们可以"作弊"——同一物体单放时是完全可见的，所以 amodal GT 天然可得。这避免了人工标注 amodal mask 的巨大成本。

潜在问题（hallucination 提示）：
- Pose 对齐误差会导致 amodal mask 与真实位置有 misalignment，但论文未提及 calibration 精度。
- SAM 在 small objects 上的 segmentation 精度可能不够，文中未量化 SAM IoU。
- 从 simulation 训练的 amodal seg module 到 real-world 的 sim-to-real gap 依赖 UOAIS 的 unseen object generalization 能力。

### 3.3 Amodal Mask 的几何意义
amodal mask $A$ 实际上是一个 **2D occupancy prior**——它告诉你"如果没遮挡，target 应该占据这块区域"。对 push/grasp policy 来说，这提供了：
- **Target 几何中心估计**：amodal mask 的 centroid 比 visible mask 的 centroid 更接近真实 target center
- **Push direction guidance**：target 的长轴方向可以从 amodal mask PCA 得到
- **Grasp point candidate**：amodal mask 边缘与 visible mask 边缘的差异区域，就是"被挡住的部分"，push 那个方向的 obstacle 最有价值

---

## 4. System Overview 与 State Representation

### 4.1 Pipeline (Fig. 2 解析)
```
RGB-D Image → Amodal Seg Module → amodal mask
        ↓
[I, D, A] (heightmaps) → rotation augmentation (16 angles) → DenseNet121 → features
        ↓
        ├──→ Push Q-Net → Push Q-map
        └──→ Grasp Q-Net → Grasp Q-map
        ↓
best Q_p, best Q_g + domain knowledge → Coordinator (MLP) → action type (push / grasp)
        ↓
Robot executes push / grasp primitive
```

### 4.2 Heightmap representation
将 3D RGB-D 投影到重力方向（top-down orthographic projection），得到：
- **Color heightmap** $I \in \mathbb{R}^{H \times W \times 3}$：每 pixel 取最高点的 RGB 值
- **Depth heightmap** $D \in \mathbb{R}^{H \times W}$：每 pixel 取最高点的 z 坐标
- **Amodal mask heightmap** $A \in \mathbb{R}^{H \times W}$：target amodal mask 也投影到同一坐标系

这是 Zeng et al. (FCQN, Cornell grasp detection) 经典做法，把 3D 问题压成 2D pixel-wise regression，大大简化网络设计。

### 4.3 Rotation encoding
heightmap 旋转 16 个角度（22.5° step），得到 16 个 rotated versions：
$$\{(I^{(k)}, D^{(k)}, A^{(k)})\}_{k=1}^{16}, \quad \theta_k = (k-1) \cdot 22.5°$$

这样做是为了把 **rotation invariance** 编码进网络——同一个 push 动作在不同 world frame 旋转下应该有相同的 Q-value。这也是 Zeng et al. 2017 的 trick。

intuition：传统 CNN 不具备 rotation equivariance，与其设计 rotation-equivariant CNN（如 GroupCNN, Steerable CNN），不如直接 data-augment 旋转。每张 Q-map 对应一个固定的 end-effector 朝向，最终取 max over rotations 即得到 rotation-invariant Q-value。

---

## 5. DQN Architecture 细节

### 5.1 Feature Extractor
- **Two conv layers**（低层特征提取）+ **DenseNet121 pretrained on ImageNet**（高层语义特征）
- DenseNet121 输入需要 3 通道，但 heightmap $D$ 和 $A$ 都是单通道——常见做法是把 $I, D, A$ 分别复制到 3 通道，然后分别过 DenseNet 取 features，再 concatenate。
- DenseNet121 在 ImageNet 上预训练，transfer learning 思路：低层 edge/texture features 在 robotic场景仍可复用。

潜在细节（hallucination）：可能采用 channel stacking：$[I, D, A]$ concat 成 5 通道或 7 通道，第一个 conv 把它降到 3 通道再进 DenseNet。

### 5.2 Push Q-Net 与 Grasp Q-Net
结构对称，都是：
- 3 个 conv layers
- Bilinear upsampling 恢复 spatial resolution
- 输出 pixel-wise Q-map $Q \in \mathbb{R}^{H \times W}$（每个 pixel 对应一个 action 位置）

具体 conv 配置（推测）：
- conv1: 3×3, stride 1, padding 1, channels 64, BN+ReLU
- conv2: 3×3, stride 1, padding 1, channels 32, BN+ReLU
- conv3: 1×1, channels 1, 输出 Q-value
- Bilinear upsample by factor 4

### 5.3 Action parameterization
- **Push action**: $(x, y, \theta, l)$ — 起点 $(x,y)$，方向 $\theta$，长度 $l$（一般 fixed at 10-15cm）
- **Grasp action**: $(x, y, \theta, w)$ — 抓取中心 $(x,y)$，gripper 朝向 $\theta$，gripper width $w$

Q-map 的每个 pixel $(i,j)$ 对应一个固定朝向（来自 rotation index）和位置 $(i,j)$ 的 action，Q-value 是 expected future return。

---

## 6. Coordinator 详细解析

Coordinator 是 OPG-Policy 区别于 OPG-Policy-no-coordinator 的关键——ablation study (Table I) 显示 coordinator 贡献了约 5.66% 的 success rate 提升（77% → 82.66%）。

### 6.1 输入特征 (6 维)
公式 (1)：
$$y = f_{mlp}(q_p, q_g, o, a_b, a_n, f_c)$$

逐项解释：

| Variable | 含义 | 计算 |
|----------|------|------|
| $q_p$ | best push Q value | $\max_{i,j,k} Q_p^{(k)}(i,j)$ |
| $q_g$ | best grasp Q value | $\max_{i,j,k} Q_g^{(k)}(i,j)$ |
| $o$ | target occluded rate | $\frac{|M_a \setminus M_v|}{|M_a|}$，即被遮挡面积/完整面积 |
| $a_b$ | target border occlusion ratio | $o_b / t_b$ |
| $a_n$ | occluded border / full mask | $o_b / t_m$ |
| $f_c$ | grasp fail count | 历史抓取失败次数 |

公式 (2) 解释：
- $m_b$ = "target border"，定义为 amodal mask 边界向外扩 10 pixels 的 strip 区域
- $o_b$ = 该 border strip 中被遮挡部分的像素数
- $t_b = \sum m_b$ = border strip 总像素数
- $t_m = \sum M_a$ = amodal mask 总像素数（即 target 完整面积）

### 6.2 为什么是 border 而不是 interior？
这是一个很精妙的设计。**Border occlusion** 比 **interior occlusion** 更影响 grasp 成功率——如果 target 中心被压着一个小物体，但边缘是空的，gripper 还可以从侧面抓；但如果 target 周围一圈都被堵死，gripper 完全没空间插入，必须先 push。

所以 $a_b$ 反映"target 周围是否被堵死"，$a_n$ 反映"被堵死的程度相对于 target 大小"。

intuition：$a_n$ 是为了 normalize。同样 10-pixel 被堵的 border，对大 target 和小 target 影响不同——大 target 相对边缘长，10 pixel 占比小；小 target 10 pixel 已经把大半圈堵死。

### 6.3 Grasp fail count $f_c$ 的作用
这是 **online adaptation** 机制——如果 grasp 失败次数累积，coordinator 会偏向 push。这避免了 OPG-Policy-no-coordinator 中"反复抓失败的位置"的浪费。

### 6.4 MLP 结构与训练
- 3-layer MLP（推测：6→32→16→1，最后一层 sigmoid 输出 push 概率）
- Loss: binary cross-entropy (公式 8)
- Ground truth：grasp 成功 → $y=1$，失败 → $y=0$，**只用 grasp outcome 监督**

⚠️ 这里有个微妙点：coordinator 监督信号是 grasp 是否成功，但 coordinator 输出的是 "应该 push 还是 grasp"。所以训练逻辑是：
- 如果 coordinator 输出 "grasp"，执行 grasp
- 若 grasp 成功，loss 鼓励 "grasp" 输出
- 若 grasp 失败，loss 鼓励 "push" 输出（反向信号）

但 push outcome 不直接监督 coordinator——push 后再 grasp 成功才算 success，这之间有时间差。文中说 "We exclusively utilize information from grasping actions"，意味着 push 后的 grasp 成功才记 reward，coordinator 学习的是"什么时候应该 push 让后续 grasp 成功"。

---

## 7. Reward Design 深度剖析

### 7.1 Grasping Reward (公式 3)
$$R = \begin{cases} 0.25, & \text{grasp position} \in \text{target amodal mask} \\ 1, & \text{target is grasped} \\ 0, & \text{otherwise} \end{cases}$$

intuition：双阶段 shaping reward。
- **0.25 reward**：dense reward，鼓励 grasp Q-net 学习 target 的位置（即使没抓起来也对）
- **1.0 reward**：sparse reward，真正成功才给

这种 shaping 是经典的 potential-based reward 思路，避免 Q-net 在 cold-start 阶段完全没梯度。

### 7.2 Pushing Reward (公式 5)
$$R = \begin{cases} 0.5, & \text{target occluded rate decreases over 0.1} \\ 1, & \text{next best grasp Q value exceeds } T_g \\ 0, & \text{otherwise} \end{cases}$$

#### 7.2.1 第一类 reward (0.5)
target occluded rate $o$ 在 push 后下降 ≥ 0.1 给 0.5。这是 **direct shaping**——push 让 target 更"露出来"就奖励。

⚠️ 一个细节：threshold 0.1 是 hard cutoff，没有 gradient 信号给"push 让 occluded rate 下降 0.05" 的情况。这可能让 push Q-net 学习到"要么彻底推开，要么不推"的极端策略。

#### 7.2.2 第二类 reward (1.0) 与 adaptive threshold $T_g$ (公式 4)
$$T_g = \beta \cdot T_g + (1 - \beta) \cdot Q_g$$

- $Q_g$：执行 push 后，下一状态的 best grasp Q value
- $\beta = 0.95$：decay factor
- $T_g$：EMA (exponential moving average) of historical $Q_g$

**Why adaptive?** 训练过程中 grasp Q-net 自己也在更新，$Q_g$ 的 scale 会变化。如果用固定 threshold（如 $T_g = 0.5$），训练后期 $Q_g$ 普遍 > 0.5，导致所有 push 都给 1.0 reward——supervision signal 退化。

EMA 让 $T_g$ 跟随 grasp Q-net 的 evolution，始终保持"高于平均水平的 grasp state 才奖励 push"的语义。

intuition building：这类似 batch normalization 思路——保持 reward distribution 稳定。但 EMA 的 update 频率依赖于 push 频率，push 较少时 $T_g$ 更新慢，可能滞后。

### 7.3 Discount factor $\gamma = 0.5$
公式 (6) 中 $\gamma = 0.5$，比较保守。这意味着 agent 只看 future return 的一半——典型的 short-horizon RL 设定，适合 push-grasp 这种"一两步就能见分晓"的任务。

如果 $\gamma$ 太高（如 0.99），Q-value 会试图预测长链路 push→push→push→grasp，方差爆炸；$\gamma=0.5$ 把 horizon 限制在 ~2 步以内，更稳定。

---

## 8. Training Curriculum 细节

### 8.1 Scene complexity escalation
| Iteration | Targets $n$ | Obstacles $m$ | Target 选择策略 | Action selection |
|-----------|-------------|--------------|------------------|------------------|
| 0–1000 | 7 | 3 | random | ε-greedy |
| 1000–3000 | 7 | 3→8 increasing | random | coordinator |
| 3000–5000 | 7 | 13 | most occluded | coordinator |
| 5000+ | 7 | 18 | most occluded | coordinator |

### 8.2 为什么前 1000 iter 用 ε-greedy？
早期 grasp Q-net 还没学会"在哪抓"，coordinator 此时会基于错误的 $q_g$ 做决策，导致一直 push 不 grasp。用 ε-greedy 强制探索，保证 grasp Q-net 有训练数据。

这是典型的 **warm-start** 思路——先让 critic 收敛一些，再让 meta-controller 介入。

### 8.3 "Retrieve all potential targets before reset"
一个 episode 内，agent 要 retrieve 所有 7 个 target 物体才 reset scene。这避免了 agent 在 hard case 上 stuck，也提供了多次 grasp 训练数据。

⚠️ 这里有个 episode 内 distribution shift：随着 target 一个个被 retrieve，scene 变稀疏，后续 grasp 更简单。这可能让 Q-net 在稀疏 scene 上 over-fit。

### 8.4 Target selection strategy
5000+ iter 后，**prefer the most occluded object as target**。这是 **hard example mining**——专门挑难的训练，提升 agent 在 extreme occlusion 下的能力。

intuition：curriculum learning 两个维度都在 escalate：
- **环境复杂度**（obstacle 数量）
- **任务难度**（target 本身的 occlusion 程度）

---

## 9. Loss Functions 数学解析

### 9.1 TD Error (公式 6)
$$\delta_t = Q(s_t, a_t; \theta_t) - [R_t(s_t, a_t, s_{t+1}) + \gamma \cdot \max_q Q(s_{t+1}, a_{t+1}; \theta_t')]$$

变量说明：
- $s_t = (I_t, D_t, A_t)$：时刻 $t$ 的 state（color/depth/amodal heightmap）
- $a_t$：执行的 action（push 或 grasp，含位置和朝向）
- $\theta_t$：online network 参数
- $\theta_t'$：target network 参数（DQN 标准 trick，定期同步）
- $\gamma = 0.5$：future discount
- $R_t$：reward
- $\max_q$：在 $s_{t+1}$ 上对所有 action 取 max Q-value

⚠️ 严格来说 DQN 用的是 $\theta_t'$ (target net)，公式中写的是 $\theta_t$，可能是 paper 笔误，或者用了 same-network（更激进的 bootstrapping）。

### 9.2 Huber Loss (公式 7)
$$L = \begin{cases} \frac{1}{2}\delta_t^2, & \text{if } |\delta_t| \leq 1 \\ |\delta_t| - \frac{1}{2}, & \text{otherwise} \end{cases}$$

intuition：
- $|\delta_t| \leq 1$：MSE，gradient = $\delta_t$，小误差时 fine-tuning
- $|\delta_t| > 1$：L1，gradient = $\text{sign}(\delta_t)$，大误差时 gradient clipping 效果

这是 DQN 经典做法，比纯 MSE 更 robust to outlier reward（如突然 +1 reward 导致的巨大 TD error）。

### 9.3 Coordinator BCE Loss (公式 8)
$$L = -(\bar{y}\log y + (1-\bar{y})\log(1-y))$$

- $y$：coordinator 输出 sigmoid probability
- $\bar{y}$：ground truth (0/1)

---

## 10. 实验设计与方法学

### 10.1 Simulation setup
- **Simulator**: V-REP (CoppeliaSim)，经典机器人仿真平台
- **Robot**: UR5 + RG2 gripper
- **Camera**: fixed-mount RGB-D（俯视）

### 10.2 三种 test scenario

#### 10.2.1 Random Case (Table I)
3 个 sub-scenario：
- **15 objects**: target 随机选，scene 稀疏
- **30 objects**: target 随机选，scene 密集
- **30 objects (hard)**: target = 最 occluded 的物体

**关键观察**：
- 15 objects 时，GE-GRASP 略胜（97% vs 96%）。因为稀疏 scene 下 amodal 信息收益小，而 GE-GRASP 的 sampling 多样性占优势
- 30 objects hard 时，OPG-Policy 大幅领先（68% vs 57% GE-GRASP vs 33% GTI）。**occlusion 越重，amodal 收益越大**
- GTI-amodal 比 GTI 差（50.33% vs 64.66%）：**简单拼接 amodal 反而有害**，因为 GTI 的 Bayesian exploration 机制不需要 amodal prior

#### 10.2.2 Occluded Case (Fig. 4)
按 occlusion ratio 分 3 组：0.2-0.4 (mild), 0.4-0.6 (moderate), 0.6-0.8 (severe)

观察：
- mild occlusion：所有方法 success rate 都高，差距小
- severe occlusion：OPG-Policy success rate 与 GE-GRASP 持平，但 **motion attempts 少 ~1 次**——更高效

这说明 amodal seg 主要是 **efficiency booster**，不是 **success rate booster**。intuition：amodal 让 agent 知道 "push 哪里能让 target 显露最多"，避免无效 push。

#### 10.2.3 Challenging Case (Fig. 5)
6 个 structured scenarios，平均：
- OPG-Policy: 99% success, 4.19 attempts
- GE-GRASP: 98% success, 4.98 attempts
- GTI: 88% success, 5.85 attempts

注意 99% vs 98% 只差 1%，但 4.19 vs 4.98 差 0.79 attempts——同样体现 efficiency 优势。

### 10.3 Real-world Experiments (Table II)

#### 10.3.1 Sim-to-real transfer
关键卖点：**model trained in sim, deployed in real without fine-tuning**。

设备：
- UR10 + ROBOTIQ gripper
- Intel RealSense D435i (640×480)
- YCB 物体集

#### 10.3.2 Two tests

**C-Test (Challenging Test)**: 4 个 occlusion scenario，每个 10 trials
- OPG-Policy: 85% success, 7.0 attempts
- GE-GRASP: 82.5%, 7.62
- GTI: 65%, 8.42

**G-Test (Generalization Test)**: 12 个 unseen household objects，20 trials
- OPG-Policy: 90% success, 6.65 attempts
- GE-GRASP: 85%, 7.3
- GTI: 80%, 7.75

⚠️ 注意 G-Test 中 GTI 用了 UOAIS-supplied visible masks（作者自己加的，为了让 baseline 公平），但还是输——说明 OPG-Policy 的优势不仅来自 perception，更来自 **policy 与 amodal 信息的协同设计**。

### 10.4 Ablation 关键发现
1. **OPG-Policy-no-coordinator** vs **OPG-Policy** (Table I)：
   - 77% vs 82.66% average success
   - 2.67 vs 2.61 average attempts
   - 说明 coordinator 贡献主要在 success rate（避免错误 grasp 尝试）

2. **GTI-amodal** vs **GTI**：
   - 50.33% vs 64.66% average success
   - 说明 naive integration of amodal 反而有害

这两点合起来证明：**amodal seg + RL policy 必须 co-design**，不能简单 plug-and-play。

---

## 11. 潜在 limitation 与改进方向

hallucination 一下这篇 paper 没说但可能存在的问题：

### 11.1 Amodal seg 误差的传播
UOAIS 在 simulation 训练，real-world 上 amodal mask 预测精度未知。如果 amodal mask 错误（如预测的 target 形状偏离真实），coordinator 的 $o, a_b, a_n$ 全部失真，可能误导决策。

可能的改进：**uncertainty-aware amodal seg**——输出 mask 概率图，让 Q-net 处理 uncertainty。

### 11.2 Single target assumption
整个 pipeline 假设只有一个 target，amodal seg 也只分割 target。但实际场景中可能多个候选 target，需要 multi-target amodal。

### 11.3 Push primitive 设计
论文中 push 应该是 fixed-length straight push。但实际中 obstacle 可能需要 curve push 或 pivot。可以考虑 **learning push primitives** 而非 fixed。

### 11.4 3D 信息丢失
heightmap 是 2D top-down projection，丢失了 3D 几何信息。对 stack 场景（target 上方压着物体）处理能力有限——push 只能水平推，无法 lift。

### 11.5 Long-horizon planning
$\gamma = 0.5$ + DQN 限制在 short-horizon。如果需要 push 5 次才能 grasp，agent 视野不够。可考虑 **hierarchical RL** 或 **model-based planning** (MCTS like Huang et al. [19])。

### 11.6 Grasp failure 的 credit assignment
coordinator 的 $f_c$ 是 fail count，但不区分 fail 原因——是 grasp position 错，还是 gripper slip，还是 obstacle 卡住？更细的 failure representation 可能提升 coordinator 决策质量。

---

## 12. 与其他相关工作的位置关系

### 12.1 与 classical push-grasp synergy
Mason (1986) 的工作奠定 non-prehensile manipulation 理论。Lynch & Mason (1996) 的 push planning 是早期 model-based 方法。OPG-Policy 是 learned version，但 push primitive 本身还是简化（straight push）。

### 12.2 与 Diffusion Policy 的对比
Diffusion Policy (Chi et al. RSS 2023) 用 diffusion model 生成 action sequence，比 DQN pixel-wise Q-map 更 flexible。但 DQN 适合 discrete action + pixel-aligned output，diffusion 适合 continuous trajectory。

潜在改进：把 OPG-Policy 的 Q-net 换成 conditional diffusion，输入 amodal mask + state，输出 push/grasp trajectory distribution。

相关链接：Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 12.3 与 LLM/VLM-based manipulation 的关系
LLM/VLM (RT-2, VIMA, Code as Policies) 能处理 language-conditioned manipulation。OPG-Policy 的 amodal seg 本质是 perception augmentation，可以与 VLM 结合：VLM 给出 "target description"，amodal seg 找 target 完整轮廓，policy 执行。

相关链接：
- RT-2: https://robotics-transformer2.github.io/
- VIMA: https://vima-robotics.github.io/
- Code as Policies: https://code-as-policies.github.io/

### 12.4 与 Active Perception 的关系
Price et al. [23] 用 active perception + 3D completion 来辅助 retrieval。OPG-Policy 是 passive perception (single view)，没有 active viewpoint change。可以扩展：agent 主动 change camera pose + amodal seg + push/grasp。

相关链接：Embodied Amodal Recognition (Yang et al. ICCV 2019): https://embodied-amodal.github.io/

### 12.5 与 Foundation Models 的融合
SAM (already used) + CLIPSeg + DINOv2 都可以提供更强的 perception。如果 amodal seg 模块换成 SAM-based promptable segmentation，可以处理开放集 target。

相关链接：
- DINOv2: https://dinov2.metademolab.com/
- CLIPSeg: https://github.com/timojl/clipseg

---

## 13. Intuition 总结

如果让我给 OPG-Policy 一句话总结，它做的是：**把"看见不可见"的 prior 注入 policy，让 push 与 grasp 在 enriched state space 上协同。**

技术 contributions 可以归纳为三层：

1. **Perception level**: Amodal seg 把 partial observation 补全为 complete occupancy prior
2. **Policy level**: DQN 在 enriched state 上学 push/grasp Q-map，reward 设计用了 adaptive threshold 来处理 Q-net drift
3. **Coordination level**: MLP coordinator 用 domain knowledge (border occlusion, fail history) 决定 action type

每层都有 ablation 支撑，每层都不可少。这是好的 paper 的标志——modular 但 cohesive。

潜在的最大启发：**robotic policy 设计中，perception module 与 policy module 必须 co-design**。简单 plug-and-play 通常 suboptimal，甚至会反向损害（GTI-amodal 案例）。这是给做 VLA、RT-X、VLM4Robotics 的人都适用的一课。

---

## 14. 个人延伸思考（给 Karpathy 风格的 comment）

如果让我重新 design 这篇 paper，可能的方向：

1. **End-to-end amodal + policy**: 现在 amodal seg 是 frozen module，能否 joint train？难点是 amodal seg 需要 dense mask supervision，policy 只需要 sparse reward。可以用 auxiliary loss 思路。

2. **Implicit amodal via 3D reconstruction**: 与其显式预测 amodal mask，不如预测 target 的 3D occupancy (NeRF-style or occupancy field)。3D 表示比 2D mask 信息更丰富，且可支持 multi-view fusion。

3. **World model for push prediction**: 现在 push reward 是 outcome-based。可以训一个 forward model 预测 push 后的 scene 状态，做 model-based RL (Dreamer-style)。这能解决 long-horizon planning 问题。

4. **Diffusion-based action sampling**: 把 Q-net 换成 diffusion policy + amodal conditioning，可能更好的处理 multi-modal action distribution（多个有效 push 方向）。

相关 reference:
- Dreamer V3: https://danijar.com/project/dreamerv3/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- NeRF for Robotics: https://nerf-slam.github.io/

---

## 15. 复现建议

如果你想 re-implement：

1. **Simulator**: 用 PyBullet 或 Isaac Gym 比 V-REP 更方便 Python 集成
2. **Amodal seg**: 直接用 UOAIS-Net pretrained model，不必重训
3. **DQN**: 用 stable-baselines3 或 tianshou，custom pixel-wise Q-net head
4. **Curriculum**: 用 gym.Env 的 reset 钩子控制 obstacle 数量
5. **Coordinator**: 单独训一个 small MLP，每 N steps 同步

代码 reference（hallucination risk）:
- UOAIS: https://github.com/kynng94/UOAIS-Net
- V-REP API: https://manual.coppeliarobotics.com/
- DenseNet PyTorch: torchvision.models.densenet121

---

希望这个 deep-dive 对你 build intuition 有帮助。整体来看，OPG-Policy 是一个 **modular design + careful reward engineering** 的范本，对未来 amodal perception + manipulation 的方向很有启发意义。
