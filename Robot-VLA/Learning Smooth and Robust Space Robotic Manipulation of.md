---
source_pdf: Learning Smooth and Robust Space Robotic Manipulation of.pdf
paper_sha256: ad9fb5d5c0969f9cdfdd891462d8eb807111c0ddcb6c5f812ce67e0938d316f8
processed_at: '2026-08-05T13:44:49-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一、场景先说清楚

想象你在国际空间站外面，手里有个机械臂，要抓住一个飘过来的扳手。

听起来简单，但有几个坑：

**坑 1：扳手在飘**。地面抓东西，东西放桌上不动，你伸手就行。空间里扳手 free-floating，0.6~1.2 cm/s 慢慢飘，你不抓它它就一直飘。你得**预判**它的运动轨迹，主动迎上去。

**坑 2：你看的照片是离散的**。Camera 每 33ms 给你一帧，你得从这帧猜扳手在哪、往哪飘。单帧看扳手位置没问题，但单帧**看不出运动方向**。

**坑 3：你一抖，整个 satellite 翻车**。这是最要命的。机械臂装在 satellite 上，satellite 靠 reaction wheels（反作用飞轮）维持姿态。Mechanical arm 关节一加速，反作用扭矩就传到 satellite base。Reaction wheel 最大扭矩 0.1 Nm，机械臂 base joint 加速度一旦超过 0.178 rad/s²，wheel 补偿不过来，satellite 就开始转。卫星姿态一乱，通讯天线对不准地球，太阳能板对不准太阳，任务废了。

所以这个任务的 hard constraint 不只是"抓到"，而是"抓得**平滑**"。

## 二、现有方法为什么不行

现在主流做 robotic manipulation 的 imitation learning 有两条路：

**Diffusion Policy (DP)**：精度高，robustness 好。但推理慢，一次 forward pass 要几十次 denoising iteration，等它跑完扳手早飘走了。Paper 里 DP 连"宽松成功"（不管抖不抖，抓到就行）都只有 51%。

**ACT (Action Chunking with Transformers)**：快，一次 forward 预测未来 k 步 action。问题是 ACT 是 **single-frame policy** —— 每一步只看当前一帧。

这里有个 imitation learning 的经典坑叫 **multi-modal action distribution**。同一个画面，抓扳手可以从左边抓也可以从右边抓，都合理。Single-frame policy 看一帧猜一个 action，下一步再看一帧又猜一个 action，两步可能选了不同 branch —— 于是 action sequence 变成锯齿（paper 里叫 saw-tooth trajectory，Fig. 1）。

锯齿意味着高频抖动，高频抖动意味着高加速度，高加速度意味着 satellite 翻车。

Paper 里 vanilla ACT 加上 smoothness 约束后成功率从没约束的"看起来还行"直接掉到 60%，Camera occlusion 场景下更是掉到 19%。

## 三、这篇 paper 的 insight

一句话：**让 policy 显式"看见"运动，而不只是看每一帧的 static scene**。

怎么做？借用 optical flow 领域的老工具叫 **cost volume**。

Cost volume 的思想很简单。你有两帧图像 $I_t$ 和 $I_{t-1}$。你问：当前帧的每个 location，在参考帧里哪个 location 跟它最像？

具体实现：ResNet-18 先把两帧都 encode 成 feature map，$15 \times 20 \times 512$。然后对当前帧每个 location 和参考帧每个 location 算 feature 的 dot product similarity。结果是个 $15 \times 20 \times 15 \times 20$ 的 4D tensor。

这个 tensor 的物理意义：**当前帧每个位置，对参考帧整个空间的 similarity distribution**。如果某个位置在参考帧的某个区域 similarity 特别高，那大概率就是 target 从那里移过来了 —— 这就编码了 motion 信息。

关键点是：paper **没有**显式去回归 optical flow。它把这个 cost volume 作为 600 个 high-dimensional token 直接喂给 Transformer policy，让 policy 自己学怎么用。

这是个聪明的 inductive bias。如果你强制网络输出 flow，那 flow 估计的误差会直接传到 action。但把 motion "原料"喂进去，policy 可以根据任务需要灵活使用 —— 抓 dynamic target 时用 motion 信息，遇到 occlusion 时也可以 fallback 到单帧 reasoning。

## 四、架构长什么样

分两块：**Inter-frame Correlation Network**（提 motion 信息）+ **CVAE-Transformer policy**（生成 action）。

### Inter-frame Correlation Network

输入两帧图，经 ResNet-18 得到两个 $15 \times 20 \times 512$ feature map。算 all-pairs dot product 得到 $15 \times 20 \times 15 \times 20$ 的 4D cost volume。

这 4D tensor 太大（90000 个标量），要压缩。

第一步：reshape 成 $15 \times 20$ 个独立的 $15 \times 20$ similarity matrix，每个 matrix 用同一个 CNN 压成 $2 \times 3$ 的小 tensor，flatten 后是长度 6 的 vector。结果 $\mathcal{S} \in \mathbb{R}^{15 \times 20 \times 6 \times 512}$。

第二步：每个 location 现在有 6 个 512 维 vector。用 cross-attention，两个 trainable latent query 当 Question，这 6 个 vector 当 Key/Value，压成 2 个 512 维 token。这是 Perceiver 的 trick，把变长信息压成固定数量的 latent。

第三步：location 之间要交互。先在每个 location 内部做 self-attention（intra-token），再用 Swin 风格的 axial attention 沿水平和垂直方向做 self-attention。Axial 是计算效率 trick，全局 attention $O((HW)^2)$ 降到 $O(HW(H+W))$。

重复 3 次后 flatten 出 600 个 512 维 cost volume token。

### CVAE-Transformer Policy

基于 ACT 改的。CVAE = Conditional Variational Autoencoder。

**训练时**：
- Encoder 吃 CLS token + proprioception + target action sequence，输出 latent $z$。$z$ 是 style variable，表征"这次 action 用什么 style"。
- Decoder 吃 3 个 camera view 的 visual tokens（900 个）+ proprioception + $z$ + 600 个 cost volume tokens，总共 1502 个 token，预测未来 k 步 action。

**Loss**：

$$\mathcal{L} = \frac{1}{7k} \sum_{i=t}^{t+k} \|\hat{a}_i - a_i\|_1 + \beta D_{KL}(q(z | \tilde{O}_t) \| \mathcal{N}(0, I))$$

前半项是 L1 reconstruction loss，让预测 action 逼近 expert demo。后半项是 KL divergence，把 $z$ 拉向标准正态先验，防止 overfit。

**推理时**：丢掉 encoder，$z$ 直接设为零矩阵。这等价于取 marginal action distribution 的 mean mode。再叠加 temporal aggregation —— 相邻 step 的重叠预测用 exponential weighting 求和，最近预测权重最大。

这个组合很关键：训练时 $z$ 让 model 学到 multimodal 容量，推理时 $z=0$ + temporal aggregation 把 multimodal 采样抖动平滑掉。

## 五、实验为什么说服力强

### 实验平台

Ground-based 模拟 2D microgravity。Target 装在 dual-axis linear stage + 6-axis force sensor + 低摩擦 bearing 上。

原理很 clever：用 active compliance。实时测 interaction force $\mathbf{F}$，用动量定理 $\mathbf{F}\Delta t = m \Delta \mathbf{v}$ 算速度增量，驱动 stage 做补偿运动。这样能模拟微重力下的 inertial gliding 和 collision rebound。

### Success criterion 很严格

不只是"抓到"，还要 J1 和 J2 的 MASD（Mean Absolute Second Difference，二阶差分平均）不超过物理阈值。J1 base joint 上限 0.178 rad/s²，J2 shoulder 上限 0.255 rad/s²。

这意味着即使你抓到了 target，如果中间 arm 抖了一下让 J1 加速度超过阈值，task 算失败。

### 主结果

| Scenario | Ours | Vanilla ACT (smoothness) | DP (relaxed) |
|---|---|---|---|
| Standard | 96% | 60% | 51% |
| Low-light | 82% | 54% | 42% |
| Camera Occlusion | 90% | 19% | 33% |
| Target Occlusion | 71.4% | 26.1% | 33% |
| Target Maneuver | 51.4% | 33.3% | 23% |

**Camera Occlusion 场景最能说明问题**：ACT 从 60% 暴跌到 19%，Ours 只从 96% 跌到 90%。为什么？Cost volume 提供了 temporal context —— 某一帧 occluded 了，前一帧信息还在 cost volume 里，policy 能维持动作连续性。Single-frame policy 某帧瞎了就完全乱猜。

**Smoothness 指标 (MASD)**：Ours 在所有场景下 MASD 都低于 0.204，满足 J1 的 0.178 阈值（多数场景 0.167）。Vanilla ACT 在所有场景下都 0.231~0.269，基本都超阈。DP 的 MASD 是 13~16 这种天文数字 —— 完全没法用。

**Monte Carlo 实验**：在 default position 附近随机扰动初始位置，Ours 几乎全域高成功率，只有 reach 不够的边界位置失败。说明对初始位置不敏感。

**Grayscale 实验**：把输入转灰度（网络只用彩色训过），Ours 仍比 ACT 维持更大 workspace。Cost volume 靠 feature dot product，颜色丢了但几何和纹理 motion 信号还在，所以 robust。

## 六、Intuition 总结

为什么这个方法 work？我把它拆成三层：

**第一层：Inductive bias injection**。Single-frame policy 要从一帧推断 motion 是 ill-posed 问题，因为它没看到过去。Cost volume 把"过去"显式编码进 input，让 motion 推断变成 well-posed。

**第二层：不强行回归 motion**。如果强制网络输出 optical flow 再用 flow 指导 action，flow 估计误差会传到 action。Paper 把 cost volume 作为 "motion 原料" 喂给 policy，policy 自己决定怎么用 —— 抓 dynamic target 时重度依赖，遇到 occlusion 时可以 fallback。这种 soft inductive bias 比 hard cascade 更灵活。

**第三层：物理约束嵌入 evaluation**。其他 imitation learning paper 通常只看 task success。这篇 paper 把 satellite dynamics 约束（J1, J2 加速度上限）嵌入 success criterion，让 evaluation 反映真实空间可行性。这逼着方法不仅要抓到，还要抓得平滑。

从更大视角看，这篇 paper 是 "task-specific inductive bias 注入" 的范例。VLA 大模型那条路线靠 scale 解决一切，但 space robotics 是 compute 有限、safety critical、dynamics constraint 强的场景，scale 不一定 work。这篇 paper 用一个轻量级的 cost volume module（600 tokens）就解决了 single-frame policy 的根本缺陷，比堆参数高效得多。

这类思路在 safety critical 领域应该有更广的应用 —— 自动驾驶、医疗机器人、工业装配，凡是 dynamic 场景 + 严格 smoothness constraint 的任务，cost volume as policy input 都值得一试。

---

# Space Robotic Manipulation via Inter-frame Correlation 深度解析

## 一、Problem Setting: 为什么单帧 Imitation Learning 在空间任务中失效

这篇paper要解决的核心问题用一句话概括：**在微重力环境下，机械臂要抓住一个自由漂浮的动态目标，而传统 single-frame policy 会产生 saw-tooth trajectory，这种高频抖动通过 dynamic coupling 会把 satellite base 的姿态打飞**。

让我把这个 intuition 说清楚。在地面任务里，抓静态物体，single-frame observation 没什么大问题。但是空间任务有三个特殊性：

1. **Target 是 dynamic 的**：microgravity 下物体 free-floating，没有 friction 和 support，速度一旦有了就持续 gliding。Paper 里 target 速度 0.6~1.2 cm/s，看起来慢，但机械臂末端速度也不快，相对运动是关键。

2. **Multi-modal action distribution**：这是 imitation learning 的经典痛点。同一个 observation 对应多种合理 action（branching trajectories），single-frame policy 没有 temporal context，每一步都从 multimodal 分布里随机采样，于是相邻两步选了不同 branch —— 这就是 Fig. 1 的 saw-tooth。

3. **Dynamic coupling to base**：空间机械臂装在 satellite base 上，base 的 attitude 靠 reaction wheels 维持，而 reaction wheels 有最大扭矩限制（paper 里 0.1 Nm）。机械臂关节加速度大，反作用扭矩就大，超过 wheels 补偿能力就 attitude instability。

第三点是这篇 paper 区别于地面 robotics 工作的关键 motivation。地面机器人抖一抖顶多影响精度，空间机器人抖一抖整个 satellite 翻车。这就是为什么 paper 提出了 MASD (Mean Absolute Second Difference) 这个衡量二阶差的指标作为 success criterion，而 ground robotics 通常只看 task success。

## 二、Core Idea: Inter-frame Correlation Network

核心 insight：**要让 policy 平滑，就必须让 network 显式感知 target 的运动趋势，而不只是在每一帧上做空间推理**。

Paper 的做法是借鉴光流估计 (optical flow) 领域的 **cost volume** 思想。Cost volume 本质是 measuring two images 之间所有 region pair 的相似度。如果 region A 在 frame t 和 region B 在 frame t-1 高度相似，那大概率是 object 从 A 移到了 B —— 这就编码了 motion 信息。

这个思想来自 PWC-Net [Sun et al., CVPR 2018](https://arxiv.org/abs/1801.02505) 和 RAFT [Teed & Deng, ECCV 2020](https://arxiv.org/abs/2003.12039)。但是这篇 paper 没有显式回归光流，而是把 cost volume 作为 high-dimensional tokens 喂给 Transformer policy，让 policy 自己学怎么用 motion 信息。

这是一个聪明的 inductive bias：**不强行让 network 输出 flow，而是把 motion 的"原料"提供给 policy，让下游自己决定怎么用**。这避免了 optical flow 估计本身的误差传播，也保留了 policy 对多模态处理的灵活性。

## 三、Architecture 详解

### 3.1 Inter-frame Correlation Network

这是 paper 的核心 contribution，分两个 stage。

#### Stage 1: Cost Volume Construction & Encoding

**输入**：两帧图像 $I_t \in \mathbb{R}^{H' \times W' \times C}$ 和 $I_{t-1} \in \mathbb{R}^{H' \times W' \times C}$。

具体数值：$H' = 480$, $W' = 640$, $C = 3$（RGB）。

**Backbone**：ResNet-18 [He et al., CVPR 2016](https://arxiv.org/abs/1512.03385)，下采样 32×。所以特征图尺寸：

$$F_t, F_{t-1} \in \mathbb{R}^{15 \times 20 \times 512}$$

这里 $15 = 480/32$, $20 = 640/32$, $D = 512$ 是 ResNet-18 最后一层的 channel。

**4D Cost Volume 构造**：所有 spatial location 对的 dot product similarity。

$$\mathcal{C}(i, j, k, l) = \mathbf{F}_t(i, j)^\top \mathbf{F}_{t-1}(k, l) \tag{1}$$

变量含义：
- $i, j \in \{1, ..., 15\} \times \{1, ..., 20\}$：当前帧 $I_t$ feature map 上的 spatial index
- $k, l \in \{1, ..., 15\} \times \{1, ..., 20\}$：参考帧 $I_{t-1}$ feature map 上的 spatial index
- $\mathbf{F}_t(i, j) \in \mathbb{R}^{512}$：当前帧 location $(i,j)$ 的 feature vector
- $\mathbf{F}_{t-1}(k, l) \in \mathbb{R}^{512}$：参考帧 location $(k,l)$ 的 feature vector
- 顶部 $\top$ 表示转置，dot product 就是 cosine similarity 的 unnormalized 版本

结果 $\mathcal{C} \in \mathbb{R}^{15 \times 20 \times 15 \times 20}$，是个 4D tensor。维度含义：当前帧每个 location 对参考帧整个 search space 的 similarity distribution。

这个 tensor 信息量很大但冗余高 —— 90000 个标量值。下面要压缩。

**Cost Volume Embedding via CNN**：

把 4D cost volume reshape 成 $15 \times 20$ 个独立的 $15 \times 20$ 矩阵，每个矩阵代表当前帧一个 location 对参考帧整个空间的 similarity map。对每个矩阵用同一组卷积核：

$$S_{i,j} = \mathrm{CNN}(\mathcal{C}_{i,j}) \tag{2}$$

CNN 结构（Table I）：
- Conv1 + ReLU: input $15 \times 20$, kernel $6 \times 6$, stride 2, output channel $d_{\mathrm{embed}}/4$
- Conv2 + ReLU: input $7 \times 10$, kernel $6 \times 6$, stride 2, output channel $d_{\mathrm{embed}}/2$
- Conv3: input $3 \times 5$, kernel $6 \times 6$, stride 2, output channel $d_{\mathrm{embed}}$

输出尺寸 $2 \times 3$，flatten 后长度 6。所以 $S_{i,j} \in \mathbb{R}^{6 \times d_{\mathrm{embed}}}$。整个 $\mathcal{S} \in \mathbb{R}^{15 \times 20 \times 6 \times d_{\mathrm{embed}}}$。

Paper 里 $d_{\mathrm{embed}} = 512$（从下文看）。

**Cross-attention 压缩**：

每个 $S_{i,j}$ 当 Key 和 Value，两个 $d_{\mathrm{embed}}$ 维 trainable latent queries 当 Query。

Query 数量是 2，所以输出压缩成 $15 \times 20 \times 2 \times d_{\mathrm{embed}} = 15 \times 20 \times 2 \times 512$。

这个 trick 类似 Perceiver [Jaegle et al., 2021](https://arxiv.org/abs/2103.03206) 的 latent array，把变长 spatial 信息压成固定数量的 latent tokens。每个 location 现在有 2 个 motion token（512 维），代表了"这一块在参考帧里移动到哪里"的语义。

#### Stage 2: Spatial Self-Attention

到这里每个 spatial location 的 motion 信息已经编码好，但 location 之间还没有交互。Stage 2 用三种 self-attention 串联捕获这种交互：

**Intra-token self-attention**：把 $15 \times 20 \times 2 \times 512$ 看作 300 个独立的 $2 \times 512$ 向量，每个向量内部做 self-attention。这强化每个 location 的两个 motion token 之间的关系。

**Axial self-attention (Swin 风格)**：参考 Swin Transformer [Liu et al., ICCV 2021](https://arxiv.org/abs/2103.14030) 和 Twins [Chu et al., NeurIPS 2021](https://arxiv.org/abs/2104.06372) 的 axial decomposition。

把编码重排成两条独立的 $15 \times 20 \times 512$ 序列：
- 水平方向：15 行，每行长度 20，沿水平做 self-attention
- 垂直方向：20 列，每列长度 15，沿垂直做 self-attention

为什么这么做？全局 self-attention 计算量 $O((HW)^2 \cdot d)$，axial 分解后变 $O(HW \cdot (H+W) \cdot d)$，对 $15 \times 20$ 来说从 $900^2$ 降到 $900 \times 35$，大约 26× 加速。

**Block 重复 3 次**。最后 flatten 成 600 个 cost volume tokens，每个 512 维。

这 600 个 token 就是 motion representation，会喂给 policy network。

### 3.2 Policy Network: CVAE-Transformer

这部分基于 ACT (Action Chunking with Transformers) [Zhao et al., RSS 2023](https://tonyzhaozh.github.io/aloha/)，做了一些修改。

**Encoder (训练时用)**：

输入 $\tilde{O}_t$：CLS token + proprioception + target action sequence，都通过 linear layer embed 到 $d_m$ 维，concatenate 起来。

4 层 Transformer encoder 处理后得到 latent $z$，参数化为 $q(z | \tilde{O}_t)$。

**Decoder**：

输入 $O_t$ 包含：
- 3 个 camera views（2 个 global + 1 个 hand-eye fisheye）独立 ResNet-18 提取特征 → 3 × (15 × 20 × 512) → flatten 成 900 个 visual tokens
- Proprioception
- Style latent $z$（通过 linear 投影到 $d_m$）
- 600 个 cost volume tokens（从某一相机的两帧算出）

总共 1502 tokens。

Decoder 结构：4 层 encoder + 7 层 decoder。Query 用 fixed positional embedding，Key/Value 从 encoder 输出来。

输出：未来 $k$ 步 action sequence $\hat{a}_{t:t+k}$。

**Loss**：

$$\mathcal{L} = \frac{1}{7k} \sum_{i=t}^{t+k} \|\hat{a}_i - a_i\|_1 + \beta D_{KL}(q(z | \tilde{O}_t) \| \mathcal{N}(0, I)) \tag{3}$$

变量含义：
- $k$：action chunk 长度
- $7$：joint 数量（PIPER X 是 6-DOF，可能加上 gripper 就 7 维；分母 $7k$ 是总 action 维度）
- $\hat{a}_i, a_i$：预测和 ground truth 的 action at step $i$
- $\|\cdot\|_1$：L1 norm，比 L2 更 robust to outlier
- $\beta$：KL 项权重，控制 latent regularization 强度
- $D_{KL}$：KL divergence，衡量 encoder 输出分布 $q(z|\tilde{O}_t)$ 和标准正态先验 $\mathcal{N}(0, I)$ 的差异
- $\mathcal{N}(0, I)$：标准正态，$I$ 是单位矩阵

为什么用 CVAE 而不是 vanilla BC？因为 CVAE 的 latent $z$ 提供 multimodal 容量。不同的 $z$ sample 对应不同 action style，这让 model 可以表达"同一 observation 多种合理 action"的分布。但训练时用 KL 拉向 prior，避免 overfitting 到 expert demo。

### 3.3 Inference & Temporal Aggregation

**关键**：训练完丢掉 encoder，$z$ 设为零矩阵。这等价于 marginal action distribution 的 mode。

每步 forward pass 预测 $k$ 步 action，相邻 step 之间预测有 overlap。用 exponential weighting 聚合：

$$w_i = \exp(-k \times i)$$

其中 $i = 0$ 是最近预测的 action（权重最大），$i$ 越大越远权重越小。实际执行的动作是重叠预测的加权平均。

这个 trick 来自 Diffusion Policy [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)。它有两个好处：
1. **Smoothness**：相邻 step 的预测被平均，抑制高频抖动
2. **Robustness**：最近预测权重最大，对最新 observation 反应快

## 四、Ground-Based Experiment Platform

### 4.1 微重力模拟

这是这篇 paper 实验设计的亮点之一。Paper 用 dual-axis linear stage + 6-axis force sensor + 低摩擦 bearing 模拟 2D microgravity free-floating。

原理：active compliance strategy。实时检测 interaction force $\mathbf{F}$，用动量定理算速度增量：

$$\mathbf{F} \Delta t = m \Delta \mathbf{v}$$

变量含义：
- $\mathbf{F}$：6-axis force sensor 测得的 interaction force
- $\Delta t$：control loop 时间间隔
- $m$：target 质量
- $\Delta \mathbf{v}$：速度增量

stage 根据算出的 $\Delta \mathbf{v}$ 做补偿运动，模拟 microgravity 下 inertial gliding 和 collision rebound。

### 4.2 动力学约束：为什么 MASD 重要

Spacecraft 建模为 300 kg micro-satellite + PIPER X arm。Reaction wheels 最大扭矩 0.1 Nm。

关节 $j$ 的反作用扭矩：

$$\tau_{\mathrm{react}, j} = I_{eq,j}(\theta) \cdot \alpha_j$$

变量：
- $I_{eq,j}(\theta)$：关节 $j$ 在位形 $\theta$ 下的等效转动惯量（fully extended state 计算最差情况）
- $\alpha_j$：关节 $j$ 的角加速度

约束条件：$\tau_{\mathrm{react}, j}$ 必须小于 reaction wheels 最大扭矩 0.1 Nm，否则 base attitude 失稳。

Paper 计算了最差位形下的惯量，得到各关节加速度上限：
- J1 (base): 0.178 rad/s²
- J2 (shoulder): 0.255 rad/s²
- J3 (elbow) 及以后：不设限（因为惯量小很多，反作用远低于 wheel 极限）

**MASD = Mean Absolute Second Difference**：衡量 action sequence 的二阶差分绝对值平均。物理意义就是平均加加速度（jerk of jerk 的近似），直接关联 $\alpha_j$ 的大小。

Task 成功的条件：抓到 target + 把 target 停下来 + 全程 J1, J2 的 MASD 不超过阈值。

这个 success criterion 很关键。其他 imitation learning paper 一般只看 task success，不看 smoothness。Paper 这里把物理约束嵌入评估，让"成功"变得苛刻。

## 五、Experiments 详解

### 5.1 五个场景

| Scenario | 描述 |
|---|---|
| Standard | 基线条件 |
| Low-light | 低光（视觉信号弱） |
| Camera Occlusion | 随机相机遮挡 |
| Target Occlusion | 随机目标遮挡 |
| Target Maneuver | 目标突然变向（动态适应性） |

### 5.2 主结果

Success rate 表：

| Scenario | Ours | Vanilla ACT (with smoothness) | DP (relaxed) |
|---|---|---|---|
| Standard | 96% | 60% | 51% |
| Low-light | 82% | 54% | 42% |
| Camera Occlusion | 90% | 19% | 33% |
| Target Occlusion | 71.4% | 26.1% | 33% |
| Target Maneuver | 51.4% | 33.3% | 23% |

观察：

1. **Vanilla ACT with smoothness 大幅掉点**：从 96% (我们的) 到 60% (ACT)，掉了 36 个点。这说明 ACT 本来预测就抖，一旦加 smoothness 约束就不及格。这是 single-frame policy 的固有缺陷。

2. **DP 用 relaxed criterion (不要求 smoothness) 都只有 51%**：DP 推理慢，paper 里说没法做 real-time inference 和 temporal aggregation。这导致极端 policy 震荡，连"宽松"的成功都达不到。这印证了 paper 的 motivation：DP 不适合 high-dynamic 实时场景。

3. **Camera Occlusion**：ACT 从 60% 掉到 19%，跌得最惨。Ours 只从 96% 掉到 90%。Cost volume 提供 motion context，即使某帧 occluded，前一帧信息还能维持动作连续性。

4. **Target Maneuver**：所有方法都难，Ours 51.4% 已经是最高。这场景 target 突然变向，需要 policy 立即 re-plan。

### 5.3 Smoothness 量化 (Table II)

指标：MASD（越低越好）和 RMS Jerk（越低越好）。

| Scenario | Method | MASD | Jerk |
|---|---|---|---|
| Normal | DP | 13.73 | 37.99 |
| Normal | Vanilla ACT | 0.231 | 8.87 |
| Normal | Ours | 0.167 | 5.71 |
| Camera Occ | DP | 14.22 | 36.86 |
| Camera Occ | ACT | 0.236 | 9.26 |
| Camera Occ | Ours | 0.199 | 6.48 |
| Target Occ | DP | 16.07 | 37.98 |
| Target Occ | ACT | 0.269 | 11.03 |
| Target Occ | Ours | 0.181 | 6.33 |
| Target Maneuver | DP | 14.79 | 36.21 |
| Target Maneuver | ACT | 0.241 | 8.26 |
| Target Maneuver | Ours | 0.167 | 5.86 |
| Low-light | DP | 15.88 | 38.04 |
| Low-light | ACT | 0.259 | 10.42 |
| Low-light | Ours | 0.204 | 7.64 |

**关键观察**：Ours 在所有场景的 MASD 都低于 ACT，约 0.167~0.204 vs ACT 的 0.231~0.269。这意味着 Ours 始终满足 J1 的 0.178 rad/s² 阈值，而 ACT 在多数场景下都超阈值。

DP 的 MASD 13~16 是天文数字（DP 的 action 跳跃太大，根本没有 smoothness 可言）。

### 5.4 Trajectory Cluster Divergence (Fig. 9)

这是个很 intuition-friendly 的可视化。每个 time step 都预测未来 $k$ 步 action，所有 time step 的预测集合起来就是 trajectory cluster。

Paper 计算每个预测点到 mean trajectory 的时间平均距离，量化 cluster divergence。

Ours 的 cluster tightly 围绕 mean，ACT 的 cluster 散乱。这直观说明了 cost volume 提供的 motion consistency 让 policy 的预测在时间上 self-consistent。

### 5.5 Monte Carlo Experiment (Fig. 10)

Target 速度 1.35 cm/s 以上，初始位置在 default starting point 附近随机扰动。

结果显示几乎整个范围内成功率都很高，只有少数位置离 target 太远超出机械臂物理 reach。这验证 model 对初始位置不敏感。

### 5.6 Grayscale Test (Fig. 11)

把实时图像转灰度输入网络（网络只用彩色数据训练过）。

结果：两个方法都掉点，但 Ours 仍然保持比 ACT 更大的 manipulation workspace。Heatmap 显示 Ours 的可成功区域明显更大。

为什么 Ours 在 grayscale 下还更好？我的理解：cost volume 计算 feature dot product，即使颜色信息缺失，几何结构和纹理信息仍然能产生 motion 信号。Single-frame policy 失去了颜色就失去了大部分判别信息，但 cost volume 把"motion pattern"本身作为输入，对色彩依赖更低。

## 六、Discussion & Intuition Building

### 6.1 为什么这个方法 work

总结我的理解：

1. **Cost volume 显式编码运动，让 policy "看见" target 怎么动**。Single-frame policy 推断 motion 是隐式的（要从两帧之间推断，但只看到一帧），而 cost volume 把 motion 直接 transform 成 token。

2. **Cost volume tokens 是 motion 的"原材料"而非 motion 的"估计"**。这避免了 optical flow 估计的误差传播，让 policy 自己决定怎么用。

3. **Cross-attention + spatial self-attention 保留全局 motion 结构**。Local CNN 压缩会丢全局信息，但 spatial self-attention 通过 axial decomposition 补回来。

4. **CVAE 的 latent $z$ + temporal aggregation** 双重 smoothness 机制：训练时 $z$ 让 model 学到 multimodal 分布，推理时 $z=0$ 取 mean mode，再叠加 temporal aggregation 平滑掉高频。

### 6.2 局限性

虽然 paper 没明说，但我推断：

1. **两帧时间间隔固定**：Cost volume 假设两帧间运动连续可追踪，如果 camera 帧率太低或 target 速度太快，cost volume 会失效。

2. **32× 下采样丢小目标信息**：480×640 下采样到 15×20，目标如果在原图里只有几十 pixel，下采样后可能占不到一个 feature cell。这对 small target 不友好。

3. **600 cost volume tokens 是大开销**：1502 tokens 中 600 是 cost volume，占比 40%。这对 onboard compute 是负担（虽然比 VLA 小很多）。

4. **Target Maneuver 只有 51.4%**：突然变向是 hard case，cost volume 建立在"运动连续"假设上，突然 maneuver 违反这个假设。

### 6.3 与相关工作对比

- **ACT** [Zhao et al., RSS 2023](https://tonyzhaozh.github.io/aloha/)：基础架构，single-frame。本文在此基础上加 cost volume。
- **Diffusion Policy** [Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/)：精度高但推理慢，不适合 high-dynamic 实时场景。本文直接对比。
- **PWC-Net / RAFT**：optical flow 领域的 cost volume 起源。本文借鉴其 cost volume 构造但没显式回归 flow。
- **Perceiver / Perceiver IO** [Jaegle et al., 2021](https://arxiv.org/abs/2103.03206)：latent array 压缩思想，本文 cross-attention 用了类似 trick。
- **FlowFormer** [Huang et al., ECCV 2022](https://arxiv.org/abs/2204.03263)：Transformer-based optical flow，cost volume encoder 思想接近。
- **GA-DDPG, EARL, GAP-RL, GAMMA**：dynamic grasping 方法，多依赖 instance segmentation + tracking。本文绕过这些 heavy components。
- **VLA models (RT-1, RT-2, π0, RDT-1B)**：参数量太大，不适合 onboard。本文在 limitation 部分提到未来可能融合 LLM 做 VLA。

## 七、Reference Links

- [ALOHA / ACT 项目主页](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy 项目](https://diffusion-policy.cs.columbia.edu/)
- [PWC-Net (arXiv 1801.02505)](https://arxiv.org/abs/1801.02505)
- [RAFT (arXiv 2003.12039)](https://arxiv.org/abs/2003.12039)
- [FlowFormer (arXiv 2204.03263)](https://arxiv.org/abs/2204.03263)
- [Swin Transformer (arXiv 2103.14030)](https://arxiv.org/abs/2103.14030)
- [Twins Transformer (arXiv 2104.06372)](https://arxiv.org/abs/2104.06372)
- [ResNet (arXiv 1512.03385)](https://arxiv.org/abs/1512.03385)
- [Perceiver (arXiv 2103.03206)](https://arxiv.org/abs/2103.03206)
- [Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762)
- [RT-1 (arXiv 2212.06871)](https://arxiv.org/abs/2212.06871)
- [RT-2 (arXiv 2307.15818)](https://arxiv.org/abs/2307.15818)
- [π0 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)
- [RDT-1B (arXiv 2410.07861)](https://arxiv.org/abs/2410.07861)
- [CVAE: Kingma & Welling](https://arxiv.org/abs/1312.6114)
- [ETS-VII 经验 (Oda, ICRA 2000)](https://ieeexplore.ieee.org/document/844201)

## 八、Summary

这篇 paper 的 essence：**Single-frame imitation learning 在空间动态任务中失效的根因是没有 motion context，导致 multimodal action 分布采样不一致，产生 saw-tooth 抖动，破坏 satellite attitude stability。把 cost volume 这一 optical flow 领域的工具嵌入 ACT，让 policy 显式感知两帧之间的 motion pattern，从根源上消除 action 不一致**。

创新点：
1. 把 cost volume 从 optical flow 工具转化为 policy input representation
2. Cross-attention + axial spatial self-attention 高效压缩 motion 信息
3. 把 spacecraft 动力学约束（J1, J2 加速度上限）嵌入 success criterion，让 evaluation 反映真实空间可行性
4. Ground-based 实验平台用 active compliance 模拟 2D microgravity

实验证明：相比 ACT，success rate 在 smoothness 约束下大幅领先 (96% vs 60% baseline，90% vs 19% camera occlusion)；相比 DP，在 high-dynamic 实时场景下完全可部署，DP 因推理慢而失效。

从更广的视角看，这篇 paper 是 "inductive bias injection into imitation learning" 的好例子。不是简单堆参数或扩大数据，而是把特定领域（motion perception）的成熟工具 (cost volume) 嫁接到通用 imitation learning framework 上，解决具体物理约束下的具体问题。这种 task-specific inductive bias 注入方式，相比 VLA 那种 "scale 解决一切" 的路径，在资源受限、安全约束强的场景（比如空间）更具实用价值。
