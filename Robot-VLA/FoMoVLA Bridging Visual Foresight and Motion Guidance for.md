---
source_pdf: FoMoVLA Bridging Visual Foresight and Motion Guidance for.pdf
paper_sha256: 95a08f74c3799de0319bdd4f449a9ebb9201371ce74e4005d6be92fce75e29d7
processed_at: '2026-08-18T22:34:25-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FoMoVLA

好嘞 Andrej，我换个路子，假装在 Tesla 旁边跟你喝咖啡聊这篇 paper。

---

## 一句话版

**现在的 VLA 模型就像一个只看脚下的人走路——它知道现在看见啥、听见啥指令，然后盲猜下一步该动哪儿。FoMoVLA 给它装了个"未来眼"和"路径规划器"，告诉它"终点长啥样"和"路上东西怎么动"，然后这两个东西还得对得上号。最骚的是，这俩辅助模块只在训练时用，部署时全丢，但模型已经被"开过光"了。**

---

## 先说问题：现在 VLA 为啥蠢

你想想现在的 OpenVLA、$\pi_0$、GR00T 这些——它们本质就一个公式：

$$\pi: (o_t, \text{instruction}) \rightarrow \text{action chunk}$$

输入当前画面 + 指令，输出一段 action。就这么简单。

这有啥问题？

**它活在"现在"。** 就像你开车只盯着车头前面三米，从来不看远方。短任务（grab the mug）还行，一旦任务长（put both the soup and the box in the basket），模型就懵了——它没有一个"目标长啥样"的概念，也没有"中间东西怎么动"的概念，全靠 reactive mapping 硬记。

那之前有人尝试解决吗？有，两条路：

**路线一：predict future frames**（WorldVLA、DreamVLA 这类）。让模型 predict 未来的画面。听起来很对，但实际有两个坑：
- 大部分像素是**静态的**——桌面纹理、背景墙、灯光都不动，模型花一堆 capacity 学这些没用的东西
- OOD 一来就崩——Table 2 里 WorldVLA 在 LIBERO-Plus 上只有 25.0%，惨不忍睹

**路线二：predict motion**（FlowVLA、JOPAT 这类）。让模型 predict 哪些点往哪儿动。问题是它只看"动"，不看"终点长啥样"——你问它"这罐可乐最后去哪儿"，它说不清楚。

FoMoVLA 的 insight 简单到离谱：**这两个东西本来就是 complementary 的啊！** future 告诉你 "where to go"，motion 告诉你 "how to get there"，**你把它们一起学、还让它们互相 condition，不就齐活儿了？**

---

## 再说怎么做：三个组件

### 组件一：Point Tracking——"路上东西怎么动"

这个特别直觉。你把当前画面切成 8×8 = 64 个 patch，每个 patch 中心对应一个 query point。然后让模型 predict：这些点在未来 T=8 帧里怎么移动？

监督从哪儿来？用 **CoTracker-v3**（Oxford 的 frozen teacher），它跟踪每个点未来 T 帧的 2D displacement，作为 ground truth。

为啥用 64 个 sparse 点，不用 dense pixel？Table 4 ablation 说明：dense（16×16=256 点）在 Long suite 上反而**更差**（94.8 vs 97.6）。因为 dense supervision 会把模型注意力稀释到每个 pixel 上，反而看不清"task-relevant 的运动"。Sparse 64 点是一种**隐式 attention prior**，逼模型聚焦在关键位置。

Loss 也很干净：

$$\mathcal{L}_{\text{disp}} = \frac{1}{NT}\sum v^\star \|\hat{\mathbf{d}} - \mathbf{d}^\star\|^2$$

- $v^\star$：visibility mask，被遮挡的点不算 gradient
- $\hat{\mathbf{d}}$：模型预测的 displacement
- $\mathbf{d}^\star$：CoTracker 给的 GT

外加一个 smoothness 项惩罚二阶差分（discrete acceleration），逼轨迹 piecewise linear——**物理上就是鼓励匀速运动，符合机器人该有的样子**。

### 组件二：Future Feature Prediction——"终点长啥样"

这个也直觉。append 16 个 learnable `<Foresight>` token，让它们的 hidden state encode "未来 T 帧之后的画面长啥样"。

但**不 predict raw pixel**，predict 的是 VLM latent space 的 feature。为啥？因为：
- pixel prediction 算力贵
- pixel 大部分是 static 的，浪费 capacity
- latent feature 抽象，更 task-relevant

监督用 **EMA teacher**：维护一个 VLM encoder 的 exponential moving average copy，momentum $\mu = 0.999$。teacher 接收 $o_{t+T}$，输出 target feature $\mathbf{z}^\star$。student 用 cosine similarity loss 去 match。

为啥要 EMA，不直接用 student 自己当 target？因为 bootstrap 不稳定——target 跟着 student 一起跑容易 collapse。EMA 慢更新保证 target 稳定。**这个 trick 你在 BYOL、DINO、JEPA 里见过无数次了**。

Loss：
$$\mathcal{L}_{\text{foresight}} = 1 - \frac{1}{M}\sum_i \cos(\hat{\mathbf{z}}_i, \mathbf{z}^\star_i)$$

K=16 个 token 比目标 M=64 patches 少很多，这是个**有意的 bottleneck**——逼模型压缩 future state，学 global scene change 而不是 patch-level texture。Table 4 ablation：bottleneck 比直接 64 token 在 Long 上**高 2.6pp**。压缩出泛化，老套路了。

### 组件三：FCCA——让它俩对上号（这才是 paper 的灵魂）

如果你把上面两个 Loss 简单加起来一起优化会怎样？

Table 1 告诉你：+1.8pp。有点用，但没爆炸。

为啥没爆炸？因为**这两个 objective 学的是 representation space 里完全不同的东西**：
- Foresight 学的是"未来外观"
- Tracking 学的是"空间运动"

它们在共享的 VLM 参数上 gradient 互相打架。Table 4 里有个 ablation 把它俩塞进同一组 token：Long suite 直接从 97.6 掉到 85.8——**灾难性 gradient interference**。

FCCA 的设计超简单：

$$\tilde{\mathbf{H}}_{\text{vis}} = \mathbf{H}_{\text{vis}} + \text{MHA}(\text{LN}(\mathbf{H}_{\text{vis}}), \text{LN}(\mathbf{H}_{\text{fut}}), \text{LN}(\mathbf{H}_{\text{fut}}))$$

翻译成人话：**让每个 image token 去 cross-attend 一下 foresight tokens，把"未来长啥样"的信息注入到"要预测怎么动"的 token 里**。

- Query = image features（N=64 个 spatial token）
- Key/Value = foresight features（K=16 个 future token）

这就让 motion prediction 知道"我要往哪儿去"——如果知道终点的可乐在篮子里，那中间的运动轨迹就该是"可乐往篮子方向动"，而不是乱跑。

**最骚的 trick 是 zero-init**：MHA 的 output projection 初始化成 0，训练开始时 $\text{MHA}(\cdot) = 0$，整个 FCCA 就是 identity mapping，**完全不扰动 pretrained VLM feature**。随着训练慢慢学起来。

这个 pattern 你在 ControlNet、LoRA adapter 里都见过——**warm-start residual，避免一上来就把预训练好的 representation 砸了**。

效果如何？Table 1：从无 FCCA 的 98.3 涨到有 FCCA 的 98.8（Long suite 上 95.8 → 97.6，**+1.8pp**）。

但更 striking 的是 Appendix B.2 的 point tracking 质量：

| Metric | 无 FCCA | 有 FCCA | 提升 |
|--------|---------|---------|------|
| ATE-moving | 3.8px | 2.3px | **−40%** |
| Survival@10px | 78.0% | **95.3%** | +17.3pp |

Survival@10px 是说"最终帧误差 < 10px 的点占比"——从 78% 跳到 95%，意思是**绝大多数点都被 FCCA 拉到 tight error bound 里了**。这是全局一致性的提升，不只是均值变好。

---

## 推理时：啥都不留

这是 paper 最实用主义的地方——**所有 auxiliary branch 只在训练时存在，部署时全丢**：
- Point tracking head 丢
- Visibility head 丢  
- MAE decoder 丢
- FCCA 丢
- EMA teacher 丢

只留 **K=16 个 `<Foresight>` token** 在输入序列里，让 action head 能 cross-attend 到 future-informed representation。

代价是多少？Table 8：median latency +9.4ms，GPU memory +0.1GB。

换来了 LIBERO Long suite +5.6pp。这个 trade-off 简直是**白嫖**。

为啥能这么干？因为辅助 supervision 的"knowledge"已经被吸收进 VLM 主干的 weights 里了——就像你上学时做很多辅导题，毕业之后不带着辅导题上班，但题里学到的东西已经在脑子里了。

---

## 实验数据说话

### LIBERO（标准 benchmark）

| 方法 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| X-VLA | 98.2 | 98.6 | 97.8 | 97.6 | 98.1 |
| LangForce | 99.2 | 99.6 | 99.4 | 95.2 | 98.4 |
| **FoMoVLA** | 98.4 | 99.6 | 99.4 | **97.6** | **98.8** |

LIBERO SOTA。最大 gain 在 Long suite（+5.6 over base）——长任务最需要 temporal reasoning，foresight + tracking 最能发挥作用。

### LIBERO-Plus OOD（zero-shot）

| 方法 | 总分 |
|------|------|
| StarVLA base | 74.1 |
| WorldVLA | 25.0（崩了）|
| **FoMoVLA** | **80.5** |

+6.4pp over base。注意 FoMoVLA **没有任何额外 pretrain**，达到了需要 VLA-JEPA pretrain 的 Abot-M0 同等水平（80.5%）。说明 spatio-temporal supervision 本身就提供了大量 inductive bias。

最大 gain 在 language perturbation（+5.5）——foresight 强制 model 把 language-conditioned future state encode 出来，逼它真理解指令而不是表面 matching。

camera/robot state 上 gain 小是预期内：训练时 fixed view + fixed pose，纯 2D motion signal 难 generalize 到 novel view。作者也老实承认了。

### RoboCasa GR-1 Tabletop

| 方法 | Avg |
|------|-----|
| StarVLA-GR00T base | 47.8 |
| GR00T-N1.6 | 47.6 |
| **FoMoVLA** | **56.9** |

+9.1pp over backbone。注意这里是 **egocentric view**（机器人头部相机，view 会随身体动而 shift），FoMoVLA 依然 robust——说明它不依赖 fixed camera。

---

## Intuition：为啥这玩意儿 work

我把这篇 paper 的精髓压成三句话：

1. **Future state 和 motion path 是两个互补的东西，缺一不可**——只有 future 不知道怎么去，只有 motion 不知道去哪儿
2. **Auxiliary supervision 只在训练时用，但学到的东西已经融进 backbone**——部署零成本
3. **Multi-task 不能各练各的，必须让一个 condition 另一个**——FCCA 就是把"未来"作为 prior 注入到"运动"的预测里

这跟你讲 World Models、JEPA、model-based RL 的思想一脉相承——**用 learned world model 提供 inductive bias，但不直接拿它做 planning，而是让它"熏陶" representation**。FoMoVLA 就是这个思想的 surgical 应用版：不做 multi-step rollout（贵），只 predict 一步 goal state（便宜）；不做 dense pixel prediction（贵且 OOD 崩），做 latent feature prediction（便宜且 robust）；不做 dense optical flow（贵），做 sparse 64 点 tracking（便宜且聚焦）。

---

## 一个有意思的细节：Attention Mask

FoMoVLA 把输入序列重排成：

```
[Instruction] → [Current Image] → [Foresight] → [Action]
```

为啥 instruction 放前面？因为 VLM 是 causal attention——如果 image 在 text 前面，image token 的 hidden state 就 attend 不到后面的 text，导致 image feature 不是 goal-aware 的。**你没法从不知道指令的 image feature 里 decode "这个点该往哪儿动"**。

Foresight tokens 之间用 **bidirectional attention**——让 K=16 个 token 自由交换信息，形成 coherent future representation。

Action tokens 看前面所有——能 leverage future-informed representation。

这个不对称 mask 设计很 elegant，每个 modality 各司其职：
- Instruction 单向 condition 下游，不被 visual 反向污染
- Image 被 condition 于 instruction
- Foresight 自由聚合全局 context
- Action 看到所有

---

## 跟你熟悉的工作的联系

- **World Models / Dreamer** (https://worldmodels.github.io)：思想一样，但 Dreamer 在 latent space multi-step rollout，FoMoVLA 只 predict 一步 goal state，更轻量
- **JEPA** (https://arxiv.org/abs/2301.08243)：latent space prediction + EMA teacher，FoMoVLA 完全是 JEPA 套路
- **CoTracker** (https://arxiv.org/abs/2307.07635)：Oxford Vedaldi 组的，FoMoVLA 用它当 frozen teacher
- **$\pi_0$ Flow Matching** (https://arxiv.org/abs/2410.24164)：action head 用这个，但 FoMoVLA 证明这套 supervision 跟 action decoder 解耦
- **ControlNet** (https://arxiv.org/abs/2302.08453)：zero-init + residual 的 trick 同款
- **BYOL** (https://arxiv.org/abs/2006.07733)：EMA teacher + cosine loss，FoMoVLA 的 future prediction 完全是 BYOL 套路

---

## 一些脑洞

作者承认的 limitation：**只有 2D tracking，没有 3D geometry**。camera perturbation 上 gain 小就是这个原因。未来方向显然是 3D point tracking 或结合 depth。

我自己想到几个方向：
- **Multi-step future**：predict $o_{t+T/2}, o_{t+T}, o_{t+3T/2}$ 多个 keyframe，让 model 学 intermediate subgoal
- **Generative foresight**：现在 foresight 是 deterministic，能不能做成 diffusion in latent，学 multi-modal future distribution，对 stochastic 环境 robust
- **Active point selection**：让 model 自己 learn 哪些点 task-relevant，类似 attention-based point selection
- **3D-aware tracking**：2D pixel displacement 换 3D scene coordinate，natural extension

---

## 总结

FoMoVLA 这篇 paper 的精髓就一句话：**"未来"和"路径"要一起学，还要让路径 condition 在未来上**。

技术上：
- Future feature prediction 用 EMA teacher + MAE decoder（JEPA-style）
- Sparse point tracking 用 frozen CoTracker-v3 当 teacher
- FCCA 用 zero-init cross-attention 把 future 注入 motion
- 全部 training-only，inference 时只多 16 个 token，+9.4ms latency

效果：
- LIBERO SOTA 98.8%
- RoboCasa GR-1 56.9%（+9.1 over backbone）
- LIBERO-Plus OOD 80.5%（+6.4 over StarVLA）

这就是用 model-based RL 的思想"开光" VLA representation，但 surgical 到不破坏 pretrained model、不加部署成本的程度。我觉得这就是你喜欢的那个 flavor——**简洁、surgical、有效**。

References:
- Project: https://liauto-research.github.io/FoMoVLA
- CoTracker: https://arxiv.org/abs/2307.07635
- $\pi_0$: https://arxiv.org/abs/2410.24164
- JEPA: https://arxiv.org/abs/2301.08243
- LIBERO: https://arxiv.org/abs/2306.11377
- RoboCasa: https://arxiv.org/abs/2406.02524
- World Models: https://worldmodels.github.io
- BYOL: https://arxiv.org/abs/2006.07733
- ControlNet: https://arxiv.org/abs/2302.08453
- FlowVLA: https://arxiv.org/abs/2508.18269
- DreamVLA: https://arxiv.org/abs/2507.04447
- WorldVLA: https://arxiv.org/abs/2506.21539

---

# FoMoVLA：用 Future Foresight + Motion Tracking 一起把 VLA 从 "反应式" 升级成 "前瞻式"

Hi Andrej，这篇 paper 我觉得挺对你胃口的——它本质上是把 robotics policy learning 那种 "predict future + track motion" 的思想塞进 VLA 的 latent space，然后通过一个 zero-init 的 cross-attention 把两个 auxiliary objective 真正耦合起来，而不是各练各的。下面我从大图景讲到每个公式、每个 ablation，尽量帮你 build intuition。

项目主页：https://liauto-research.github.io/FoMoVLA

---

## 1. 大图景：VLA 为什么需要 foresight + motion

传统 VLA 的 policy 定义很简单：

$$\pi: (o_t, l) \mapsto \mathbf{a}_{t:t+T} \in \mathbb{R}^{T \times D}$$

其中 $o_t \in \mathbb{R}^{H \times W \times 3}$ 是当前观测，$l$ 是 language instruction，$T$ 是 action chunk length，$D$ 是 action dim（end-effector pose / joint position 等）。这是一个**纯 reactive mapping**——模型只看当下，不看未来。

作者的核心论点（我非常 buy）：actionable foresight 应该 encode 两件事：
- **where to go**：future state（goal state）
- **how to get there**：continuous motion path

之前的工作要么做 dense pixel-level future prediction（representational redundancy，太多 static content），要么做 keyframe-level future representation（只刻画 target state，丢失了 dynamic interaction）。FoMoVLA 的 insight 是：**future feature prediction 和 sparse 2D point tracking 是天然 complementary 的**——前者给 goal，后者给 path。

---

## 2. 整体架构（Fig. 2）

输入 sequence 重新排成：

```
[Instruction tokens] → [Current Image tokens (M=64)] → [Foresight tokens (K=16)] → [Action tokens]
```

为什么 text 在 image 前面？因为标准 autoregressive VLM 用的是 causal attention——如果 image tokens 在 text 前面，image token 的 hidden state 就 attend 不到后面的 text，导致 visual feature 不是 goal-aware 的。把 text 放前面，image tokens 就能 attend 到完整的 instruction，这是后续从 image hidden state decode point motion 的**必要前提**。

三个 training-only 分支：

| 分支 | 作用 | 监督来源 | 输入位置 |
|------|------|---------|----------|
| Point Tracking | per-frame 2D motion | frozen CoTracker-v3 | N=64 image token positions |
| Future Feature Prediction | goal state | EMA teacher (momentum 0.999) | K=16 `<Foresight>` token positions |
| FCCA | 把 future condition 注入 motion | — | module inserted between VLM and tracking head |

**关键卖点**：推理时所有 auxiliary branches 全部丢弃，只有 K=16 个 `<Foresight>` token 留下来 cross-attend 给 action head。Inference overhead 只有 +9.4ms latency 和 +0.1GB memory（Table 8），训练多了 60.7M 参数（+1.3%），部署参数和 vanilla 完全一样（4599.3M）。

---

## 3. Flow Matching Action Head（Preliminaries）

Action head 用的是 conditional flow matching（$\pi_0$ style），路径定义为：

$$\mathbf{x}_\tau = (1-\tau)\mathbf{x}_0 + \tau \mathbf{x}_1, \quad \tau \sim \mathcal{U}[0,1]$$

- $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：standard Gaussian noise
- $\mathbf{x}_1 = \mathbf{a}_{t:t+T}$：ground-truth action chunk
- $\tau$：interpolation time，从 0（noise）到 1（data）

训练目标是让 DiT velocity field $v_\theta(\mathbf{x}_\tau, \tau, \mathbf{c})$ 匹配 constant conditional velocity $\mathbf{x}_1 - \mathbf{x}_0$：

$$\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau, \mathbf{x}_0, \mathbf{x}_1}\left[\big\|v_\theta(\mathbf{x}_\tau, \tau, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\big\|^2\right]$$

这里 $\mathbf{c}$ 是 VLM 输出的 context representation，在 FoMoVLA 里包含了 future-informed hidden states（因为 action tokens 在 sequence 末尾，会 attend 到 foresight tokens）。

推理时 integrate $\mathrm{d}\mathbf{x}/\mathrm{d}\tau = v_\theta(\mathbf{x}_\tau, \tau, \mathbf{c})$ from $\tau=0$ to $\tau=1$，恢复 action chunk。

这个设计的好处：action head 是**模型无关**的——Table 5 显示 FoMoVLA 可以 plug-in 到 StarVLA-$\pi$、StarVLA-OFT、StarVLA-GR00T 三个不同的 action decoder 上，都有一致 +1.4~2.2% 的 gain。说明 FoMoVLA 改善的是 VLM 的 visual representation，不是 action head。

---

## 4. Point Tracking 分支：让 image tokens 编码 motion

### 4.1 Sparse point selection

ViT patch stride 16，输入 224×224 → 14×14 feature map，再被 spatial merger 压到 8×8 = 64 image tokens。每个 image token 对应原图的一个 patch cell，把 patch center 映射回原图坐标，得到 N=64 个 query points。

注意是**4× sparser**——他们 ablate 过 16×16（256 points），结果 Table 4 显示在 long-horizon 上反而更差（94.8 vs 97.6）。原因：dense supervision 会逼着模型 over-fit 到每个点的局部 motion，反而稀释了 task-relevant 的 motion signal。Sparse 是一种 implicit attention prior。

### 4.2 监督信号

用 frozen **CoTracker-v3** (Karaev et al. 2024, https://arxiv.org/abs/2307.07635) 作为 teacher，给定 $o_t$ 和 N 个 grid points，追踪整个 action chunk 的 T 帧，得到：

- $\mathbf{d}^\star \in \mathbb{R}^{N \times T \times 2}$：每个点每帧的 2D displacement（按 image size 归一化）
- $\mathbf{v}^\star \in \{0,1\}^{N \times T}$：visibility label（点是否被遮挡）

注意这个 $\mathbf{d}^\star$ 是 **displacement**，不是 absolute position——所以即使 view shift，只要相对 motion 一致就能 match。

### 4.3 Alignment heads

从 VLM 在 N=64 个 image token 位置提取 hidden state $\mathbf{h} \in \mathbb{R}^{N \times d}$（d 是 VLM hidden dim），两个轻量 MLP：

$$\hat{\mathbf{d}} = f_{\text{track}}(\mathbf{h}) \in \mathbb{R}^{N \times T \times 2}$$
$$\hat{\mathbf{v}} = f_{\text{vis}}(\mathbf{h}) \in \mathbb{R}^{N \times T}$$

$f_{\text{track}}$ 是两层 MLP：LayerNorm → Linear → GELU → Linear。$f_{\text{vis}}$ 结构类似但 hidden dim 更小。

### 4.4 Motion loss

$$\mathcal{L}_{\text{track}} = \mathcal{L}_{\text{disp}} + \mathcal{L}_{\text{vis}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

三个分量：

**(1) Displacement regression**（masked MSE）：
$$\mathcal{L}_{\text{disp}} = \frac{1}{NT}\sum_{n,t} v^\star_{n,t} \|\hat{\mathbf{d}}_{n,t} - \mathbf{d}^\star_{n,t}\|^2$$

- $v^\star_{n,t}$ 是 ground-truth visibility mask——被遮挡的点不贡献 gradient，避免 noisy supervision
- 这是**核心 motion signal**

**(2) Visibility classification**：
$$\mathcal{L}_{\text{vis}} = \text{BCE}(\hat{\mathbf{v}}, \mathbf{v}^\star)$$

**(3) Smoothness regularization**（二阶时序差分）：
$$\mathcal{L}_{\text{smooth}} = \frac{1}{N(T-2)}\sum_{n,t} v^\star_{n,t} \|\Delta^2 \hat{\mathbf{d}}_{n,t}\|^2$$

其中 $\Delta^2 \hat{\mathbf{d}}_{n,t} = \hat{\mathbf{d}}_{n,t+1} - 2\hat{\mathbf{d}}_{n,t} + \hat{\mathbf{d}}_{n,t-1}$。

**Intuition**：这个二阶差分就是 discrete acceleration，惩罚它等于鼓励轨迹是**piecewise linear**的——物理上对应"匀速运动"，避免训练里出现莫名其妙的 jerk。这是 robots 应该有的 motion prior。$\lambda_{\text{smooth}} = 0.1$。

---

## 5. Future Feature Prediction：compact goal state

### 5.1 为什么不用 raw pixels？

Dense future frame prediction（像 WorldVLA、DreamVLA 那样）的问题：
1. **Computational overhead 大**
2. **大部分 pixel 是 static**—— tabletop 任务的背景、桌面纹理都不变，但 dense pixel prediction 会逼模型花大量 capacity 在这些 irrelevant static content 上

FoMoVLA 选了 latent feature prediction：在 VLM 的 feature space 里 predict $o_{t+T}$ 的 visual feature。Loss 简单干净，是 cosine similarity reconstruction。

### 5.2 K-token bottleneck

Append K=16 个 learnable `<Foresight>` tokens，让它们的 hidden state encode $o_{t+T}$ 的 visual features $\mathbf{z}^\star \in \mathbb{R}^{M \times d}$（M=64 个 visual patches）。

注意 K=16 << M=64 这个 bottleneck 是**有意的**：强迫 model 把 future state 压缩到一个紧凑的 manifold，逼它 capture global scene-level changes（"哪些东西变了"）而不是 patch-level texture memorization（"每个 pixel 长什么样"）。

Table 4 的 ablation 对比了 "MAE (16→64 tokens)" vs "Direct-64 (64 tokens)"：在 Long suite 上 MAE 是 97.6，Direct 是 95.0——bottleneck 越紧，long-horizon reasoning 越好。这是一个典型的 "compression → generalization" 现象。

### 5.3 EMA Teacher

为了给 future prediction 提供稳定的 regression target，维护一个 VLM vision encoder 的 EMA shadow copy，momentum $\mu = 0.999$：

$$\theta_{\text{teacher}}^{(t+1)} = \mu \cdot \theta_{\text{teacher}}^{(t)} + (1-\mu) \cdot \theta_{\text{student}}^{(t)}$$

- $\theta_{\text{teacher}}$：EMA teacher 参数（不接收 gradient）
- $\theta_{\text{student}}$：当前 VLM encoder 参数
- $\mu = 0.999$：very slow update, target stable

EMA teacher 接收 $o_{t+T}$，输出 target feature $\mathbf{z}^\star$。

**Why EMA**：如果直接用 student 自己当 target，会有 **bootstrap 不稳定**（target 跟着 student 跑，容易 collapse）。EMA 慢更新保证 target 是一个 quasi-fixed 的 anchor。这跟 BYOL、DINO 一脉相承的思想——你训练过 ViT/JEPA 应该一眼就 get。

### 5.4 MAE Decoder

K=16 个 foresight hidden state + (M−K)=48 个 learnable mask tokens，过一个 2-layer ViT decoder with sinusoidal positional embeddings，输出重构 $\hat{\mathbf{z}} \in \mathbb{R}^{M \times d}$。

Loss：

$$\mathcal{L}_{\text{foresight}} = 1 - \frac{1}{M}\sum_{i=1}^{M} \frac{\hat{\mathbf{z}}_i \cdot \mathbf{z}^\star_i}{\|\hat{\mathbf{z}}_i\| \|\mathbf{z}^\star_i\|}$$

是 1 − cosine similarity，是标量。MAE decoder 只在 training 时存在，inference 时整个丢掉——只保留 K=16 个 `<Foresight>` token 的 hidden state 给 action head 用。

---

## 6. FCCA：Future-Conditioned Cross-Attention（核心 contribution）

这是这篇 paper 真正的"灵魂"——其他两个分支单独看都不算 novel（latent future prediction + sparse tracking 都是已知 trick），但**让两者真正耦合**才是关键。

### 6.1 为什么不能各练各的？

如果只把 $\mathcal{L}_{\text{foresight}}$ 和 $\mathcal{L}_{\text{track}}$ 当成两个 independent auxiliary loss 并行加，会出现什么问题？

Table 1 的 ablation 给了答案：
- Base: 96.5%
- +Future Prediction: 97.5%（+1.0）
- +Tracking: 97.8%（+1.3）
- +Future + Tracking（无 FCCA）: 98.3%（+1.8）
- +Future + Tracking + FCCA: **98.8%（+2.3）**

独立 multi-task training 是有 gain 的（98.3 > 97.5 和 97.8），但 synergy 有限——两个 objective 在 representation space 里学的是**不一致的东西**：foresight 学的是 "future appearance"，tracking 学的是 "spatial motion"，它们的 gradient 在 shared parameters 上**互相打架**。

Table 4 的 "Shared goal tokens" ablation 更明显：如果让 tracking head 也用 K=16 个 foresight tokens 作为 input，Long suite 掉到 85.8%（从 97.6% 掉 12%！）。原因就是 gradient interference。

### 6.2 FCCA 的设计

在 VLM 视觉 token extraction 之后、tracking projector 之前，插入一个 lightweight cross-attention：

$$\tilde{\mathbf{H}}_{\text{vis}} = \mathbf{H}_{\text{vis}} + \text{MHA}(\text{LN}(\mathbf{H}_{\text{vis}}), \text{LN}(\mathbf{H}_{\text{fut}}), \text{LN}(\mathbf{H}_{\text{fut}}))$$

- $\mathbf{H}_{\text{vis}} \in \mathbb{R}^{N \times d}$：grid 位置的 image features（Q）
- $\mathbf{H}_{\text{fut}} \in \mathbb{R}^{K \times d}$：foresight 位置的 hidden states（K 和 V）
- MHA：multi-head attention，8 heads
- LN：layer normalization

也就是：**Query = image features, Key/Value = foresight features**。让每个 spatial image token 去 "看一眼" future state，把 goal-aware 的信息注入到 motion predictor 的 input 里。

### 6.3 Zero-initialization（关键 trick）

MHA 的 output projection **零初始化**——这意味着训练开始时 $\text{MHA}(\cdot) = 0$，所以 $\tilde{\mathbf{H}}_{\text{vis}} = \mathbf{H}_{\text{vis}}$，完全等于 identity mapping，**不扰动 pretrained VLM features**。

随着训练进行，FCCA 渐渐学起来把 future 信息注入 spatial tokens。

**Intuition**：这是一个标准的 "warm-start residual" pattern——你不想一上来就用 random-initialized module 破坏 VLM 已经学好的 representation。这种 zero-init + residual 在 ControlNet、adapter tuning 里都有出现。

### 6.4 FCCA 带来的 tracking 质量提升

Appendix B.2 量化了 FCCA 对 tracking quality 的影响（10,000 个 LIBERO-10 样本，用 frozen CoTracker 当 GT）：

| Metric | Vanilla | +FCCA | Δ |
|--------|---------|-------|----|
| ATE-all | 1.2 px | 0.9 px | −25.0% |
| ATE-moving | 3.8 px | 2.3 px | **−39.5%** |
| Median TE | 3.7 px | 2.1 px | −43.2% |
| Survival@10px | 78.0% | **95.3%** | +17.3 pp |

- ATE = Average Trajectory Error
- "moving" subset：displacement > 3px 的点（task-relevant）
- Survival@10px：最终帧 error < 10px 的比例

这个 +17.3pp 的 Survival@10px 跳升说明 FCCA 不只是 mean error 降了，而是把"大部分点都收敛到 tight error bound"——这是**全局一致性**的体现。Fig. 3 的 qualitative 也支持这个：no-FCCA 时 point trajectory 方向错乱，加 FCCA 后轨迹方向和 GT 对齐。

---

## 7. Attention Mask 设计（Appendix A.2）

Input 序列：Instruction → Current Image → Foresight → Action

四个 modality 之间的 attention 规则：

| From \ To | Instruction | Current Image | Foresight | Action |
|-----------|--------------|---------------|-----------|--------|
| Instruction | causal | ✗ | ✗ | ✗ |
| Current Image | ✓ (causal to preceding text) | causal (within image) | ✗ | ✗ |
| Foresight | ✓ | ✓ | **bidirectional** | ✗ |
| Action | ✓ | ✓ | ✓ | — |

关键点：
- **Instruction 单向 condition 下游**——不被 visual observation 反向影响，保持 language prior 纯净
- **Foresight tokens 之间 bidirectional**——K 个 token 自由交换信息，形成 coherent future representation
- **Action tokens 看到所有前面**——能 leverage future-informed representation

这个不对称 mask 设计很 elegant，让每个 modality 各司其职。

---

## 8. 主实验结果

### 8.1 LIBERO（Table 1）

LIBERO 4 个 suite（Spatial/Object/Goal/Long），每 suite 10 tasks，每 task 20 rollouts，T=8：

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| Base (StarVLA-GR00T) | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |
| +Future Pred | 99.0 | 99.4 | 97.2 | 94.4 | 97.5 |
| +Tracking | 98.6 | 99.2 | 99.0 | 94.4 | 97.8 |
| +Future +Track | 98.8 | 99.4 | 99.2 | 95.8 | 98.3 |
| **Full (+FCCA)** | **98.4** | **99.6** | **99.4** | **97.6** | **98.8** |

对比 SOTA VLA：$\pi_0$ 94.1, X-VLA 98.1, LangForce 98.4, Cosmos Policy 98.5, Spatial Forcing 98.5——FoMoVLA 98.8 是 LIBERO 上的新 SOTA。

**最大 gain 在 Long suite**：base 92.0 → full 97.6（+5.6）。Long-horizon 任务最需要 temporal reasoning，foresight + tracking 最能发挥作用。

### 8.2 LIBERO-Plus OOD（Table 2）

7 个 perturbation 维度（camera, robot state, language, lighting, background, noise, layout），10,030 个 instance，**zero-shot 评估**：

| Method | Camera | Robot | Lang | Light | BG | Noise | Layout | Total |
|--------|--------|-------|------|-------|----|-------|--------|-------|
| StarVLA | 52.5 | 49.8 | 88.5 | 95.7 | 95.7 | 73.0 | 76.9 | 74.1 |
| **FoMoVLA** | 64.0 | 62.2 | **94.0** | 94.1 | **96.2** | 82.2 | 79.6 | **80.5** |

+6.4 pp overall。最大 gain 在 language perturbation（+5.5）和 background（+0.5）、layout（+2.7）。

为什么 language perturbation 上 gain 最大？我猜想是：foresight branch 强制 model 把 language-conditioned future state 显式 encode，逼着 model 真正理解 "做什么" 而不是 surface-level matching——所以 perturb language 时 robustness 更好。

为什么 camera/robot state gain 小？因为训练时是 fixed view + fixed initial pose，纯 2D motion signal 难以 generalize 到 novel view。作者也明确承认了这个 limitation。

对比 Abot-M0（80.5%）—— FoMoVLA 在**没有 pretrain** 的情况下达到 Abot-M0（带 VLA-JEPA pretrain）的水平，这说明 spatio-temporal supervision 本身就 provide 了大量 inductive bias。

### 8.3 RoboCasa GR-1 Tabletop（Table 3）

24 个 tabletop pick-place 任务，1000 demos/task，50 rollouts/task，T=16。Egocentric view（机器人头部相机）。

| Method | Avg |
|--------|-----|
| StarVLA-GR00T | 47.8 |
| GR00T-N1.6 | 47.6 |
| StarVLA-OFT | 48.8 |
| **FoMoVLA** | **56.9** |

+9.1 pp over backbone，比 StarVLA-OFT 高 8.1 pp。egocentric view 引入了 continuous viewpoint change，这个 setting 下 FoMoVLA 依然 robust——验证了 model 不依赖 fixed camera。

Table 7 是 per-task ablation：从 vanilla 47.8 → +Future 54.4 → +Tracking 55.6 → +Both 56.6 → +FCCA 56.9。每一步都有 marginal gain，FCCA 提供 final nudge。

---

## 9. Ablation 深读

### 9.1 Point grid density

| Grid | Long | Avg |
|------|------|-----|
| 8×8 (64 pts) | 97.6 | 98.8 |
| 16×16 (256 pts) | 94.8 | 98.1 |

Dense 反而更差——尤其在 Long suite 上掉 2.8 pp。我的理解：dense supervision 等价于让 model 学每个 pixel 的 optical flow，这 dilutes task-relevant motion。Sparse 8×8 grid 本身是一种 "structured attention prior"，model 只需要 attend 到 64 个 task-relevant 的 spatial location。

### 9.2 Foresight bottleneck

| Design | Long | Avg |
|--------|------|-----|
| MAE (16→64) | 97.6 | 98.8 |
| Direct-64 (64 tokens) | 95.0 | 97.8 |

MAE bottleneck（K=16 → M=64）比直接用 64 tokens 学 future 更好。compression → generalization 的经典案例。

### 9.3 Coupling 设计（关键 ablation）

| Design | Long | Avg |
|--------|------|-----|
| Shared goal tokens (track uses K=16 foresight) | 85.8 | 94.8 |
| Separate query tokens (each branch own K=16) | 95.8 | 98.2 |
| Shared image tokens (both use N=64 image) | 95.6 | 97.4 |
| **Image + Goal CrossAttn (FoMoVLA)** | **97.6** | **98.8** |

"Shared goal tokens" 灾难性——Long 掉到 85.8。因为两个 objective 在 K=16 个 shared embedding 上 gradient 直接打架：foresight 学 future appearance，tracking 学 spatial motion，两者 representation 完全不同。

最佳设计是 FoMoVLA：tracking head 用 image tokens 作为 input（保持 spatial 位置信息），通过 FCCA 从 foresight tokens cross-attend 过来 goal context——既保留了 spatial grounding，又注入了 future conditioning，没有 gradient interference。

### 9.4 跨 action head 的 scalability（Table 5）

| Backbone | Base | +FoMoVLA | Δ |
|----------|------|----------|----|
| StarVLA-$\pi$ | 95.7 | 97.9 | +2.2 |
| StarVLA-OFT | 96.6 | 98.0 | +1.4 |
| StarVLA-GR00T | 96.5 | 98.8 | +2.3 |

FoMoVLA 跨 action head 都有 +1.4~2.3 的 gain，最大 gain 在 Long suite（StarVLA-$\pi$ +8.2 in Long!）。这证明 FoMoVLA 改的是 **VLM representation**，不依赖 action decoder。

---

## 10. 训练细节超参

- **Backbone**：StarVLA-GR00T (https://arxiv.org/abs/2604.05014)
- **Input**：224×224, ViT patch 16 → 14×14 → spatial merger → 8×8 = 64 image tokens
- **K**：16 foresight tokens
- **EMA momentum** $\mu = 0.999$
- **Loss weights**：$\lambda_1 = 0.1$（foresight）, $\lambda_2 = 0.3$（tracking total）, $\lambda_{\text{smooth}} = 0.1$
- **Optimizer**：AdamW ($\beta_1=0.9$, $\beta_2=0.95$), per-module LR
  - VLM: $1 \times 10^{-5}$
  - Auxiliary heads: $1 \times 10^{-4}$
- **Hardware**：8×H20 GPUs, DeepSpeed ZeRO-2
- **LIBERO**：30K steps, batch size 12
- **RoboCasa**：100K steps, batch size 8

**Note**：VLM LR 比 auxiliary head 小 10×——因为 VLM 是 pretrained 的，要 protected；auxiliary head 是 from scratch 的，需要 fast learning。这种 per-module LR 设计很关键。

---

## 11. Inference Cost（Table 8）

| Metric | Vanilla | FoMoVLA |
|--------|---------|---------|
| Median Latency | 94.3 ms | 103.7 ms |
| Mean Latency | 97.5 ms | 103.8 ms |
| GPU Memory | 9.3 GB | 9.4 GB |

只多 +9.4 ms latency（约 +10%）和 +0.1 GB memory。原因是 inference 时所有 auxiliary module 全丢，只多 K=16 个 token 的 forward cost。这个 trade-off 非常划算——LIBERO Long 上 +5.6 pp 换来 +9.4 ms。

---

## 12. 给你的 Intuition 总结

我觉得这篇 paper 之所以 work，本质上是把 **"model-based RL 的 predictive model"** 思想翻译到了 VLA 的 latent space，但做得非常 surgical：

1. **不 predict dense future**：用 latent feature 代替 raw pixel，避开 representational redundancy
2. **不 predict dense motion**：用 sparse 8×8 grid 代替 dense optical flow，避开 supervision dilution
3. **不各练各的**：FCCA 让 motion prediction condition 在 future state 上，从 independent regularizer 变成 **goal-conditioned motion planning 的 latent 实现**
4. **不增加 inference cost**：所有 auxiliary 都 training-only，部署零负担

我觉得最 elegant 的点是 FCCA 的 zero-init + residual——这是把 "future-conditioned motion" 这个概念上 abstract 的东西，落到了一个具体可训练的、不破坏 pretrained feature 的 module 上。

### 12.1 和 Karpathy 你自己工作的联系

你应该会想到几个相关 line：

1. **World Models (PlaNet, World Models, Dreamer)** (https://worldmodels.github.io)：model-based RL 用 learned world model 做 planning。FoMoVLA 的 future feature prediction 就是 latent world model 的 lightweight 版本——但**不 rollout 多步**，只 predict 一步 goal state。这跟 Dreamer 的 multi-step roll-out 不同，更接近 JEPA-style。

2. **JEPA** (LeCun, https://arxiv.org/abs/2301.08243)：predict in latent space，不用 pixel reconstruction。FoMoVLA 的 future feature prediction 完全是 JEPA 思想——EMA teacher 也是 BYOL/JEPA 的标准 trick。

3. **CoTracker** (https://arxiv.org/abs/2307.07635)：Andrej 你应该熟悉这个 work，Oxford 的 Vedaldi group。他们做的是 dense / sparse point tracking，FoMoVLA 把它当成 frozen teacher 提供 supervision signal，类似 distillation。

4. **Flow Matching for Action** ($\pi_0$, https://arxiv.org/abs/2410.24164)：这个 action head 你应该熟。FoMoVLA 的 action head 就是 $\pi_0$-style，但他们证明这套 spatio-temporal supervision 跟 action decoder 解耦，可以 plug-in 任何 action head。

5. **Latent Surgical / Surgical Foundation Models**：你之前在 lidar / surgical robot 的 talk 里提过类似 "把未来信息塞回 current representation" 的想法——FCCA 就是这个思想的具体 instantiation。

### 12.2 Limitation 和未来方向

作者自己承认的 limitation：只有 **2D point tracking**，没有 **3D geometry**。当 view shift 大时（camera perturbation 上 gain 小）2D motion signal 不够 informative。未来方向是 3D motion prediction——可能结合 Gaussian Splatting 或者 NeRF 的 scene representation。

我自己的脑洞：
1. **Multi-step future**：现在只 predict $o_{t+T}$，能不能 predict $o_{t+T/2}, o_{t+T}, o_{t+3T/2}$ 多个 keyframe？这会让 model 学到 "intermediate subgoal"，对 long-horizon 更好。
2. **Generative foresight**：现在的 foresight 是 deterministic reconstruction，能不能做成 generative（diffusion in latent space）让 model 学到 multi-modal future distribution？这对 stochastic environment 更 robust。
3. **Active point selection**：现在是 fixed 8×8 grid，能不能让 model 自己 learn 哪些点 task-relevant？类似 attention-based point selection。
4. **3D-aware tracking**：把 2D pixel displacement 换成 3D scene coordinate + depth，natural extension。

### 12.3 你可能想知道的 "为什么不..."

- **Why not just future frame prediction (WorldVLA)？** Table 2 显示 WorldVLA 在 LIBERO-Plus 上只有 25.0%（vs FoMoVLA 80.5%）。Dense pixel prediction 在 OOD 上崩盘——model 学了太多 static spurious correlation。
- **Why not just optical flow (FlowVLA)？** Table 1 显示 FlowVLA avg 88.1（LIBERO 上），比 FoMoVLA 低 10.7pp。Optical flow 只 capture short-term motion，没有 long-horizon goal state representation。
- **Why not just point tracking (JOPAT)？** JOPAT 97.8 avg——已经不错，但 Long suite 96.4 < FoMoVLA 97.6。纯 tracking 缺 future state 的 holistic encoding。
- **Why not combine them naively without FCCA？** Table 1 ablation：+Future+Track (无FCCA) 98.3 vs +FCCA 98.8。差 0.5pp 听起来不多，但在 Long suite 上是 95.8 → 97.6（差 1.8pp），且 tracking quality 的 Survival@10px 从 78% → 95.3%。FCCA 真的让两个 objective 协同。

---

## References（关键 link）

- Project page: https://liauto-research.github.io/FoMoVLA
- CoTracker: https://arxiv.org/abs/2307.07635
- $\pi_0$ (Flow Matching VLA): https://arxiv.org/abs/2410.24164
- LIBERO benchmark: https://arxiv.org/abs/2306.11377
- RoboCasa: https://arxiv.org/abs/2406.02524
- StarVLA backbone: https://arxiv.org/abs/2604.05014
- FlowVLA (related point tracking VLA): https://arxiv.org/abs/2508.18269
- DreamVLA (related future prediction VLA): https://arxiv.org/abs/2507.04447
- WorldVLA (related world model VLA): https://arxiv.org/abs/2506.21539
- VLA-JEPA (related latent world model): https://arxiv.org/abs/2602.10098
- JEPA (LeCun): https://arxiv.org/abs/2301.08243
- Diffusion Policy (related action head): https://arxiv.org/abs/2303.04137
- LIBERO-Plus (OOD benchmark): https://arxiv.org/abs/2510.13626

---

## TL;DR

FoMoVLA = VLA + 2 个 training-only auxiliary supervision + 1 个 cross-attention module 把它们耦合：

1. **Future feature prediction**：K=16 foresight tokens + EMA teacher + MAE decoder + cosine loss → goal state representation
2. **Sparse 2D point tracking**：N=64 grid points + frozen CoTracker-v3 teacher + masked MSE + smoothness regularization → motion path representation
3. **FCCA**：zero-init cross-attention（Q=image, KV=foresight）让 motion prediction condition 在 future state 上

Inference 时只保留 K=16 foresight tokens（+9.4ms / +0.1GB），其他全丢。

LIBERO SOTA 98.8%, RoboCasa 56.9%（+9.1 over backbone），LIBERO-Plus OOD 80.5%（+6.4 over StarVLA）。

核心 insight：**future 和 motion 是 complementary 的，但只有在 FCCA 让它们 condition 在彼此上时才真正协同**——独立 multi-task training 不够，必须 explicit coupling。

我觉得这是一个很 clean 的 case study：怎么把 model-based RL 的 predictive model 思想 surgical 地 transplant 到 VLA 上，不破坏 pretrained representation，不加 inference cost，但显著提升 spatio-temporal reasoning。
