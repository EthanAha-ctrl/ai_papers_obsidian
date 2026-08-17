---
source_pdf: Chameleon Episodic Memory for Long-Horizon.pdf
paper_sha256: 3fecae9c37e44bdc7fe4f556410b526d833fd29e4e9cb5aace8fdd13ee6c0c35
processed_at: '2026-08-03T15:29:36-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Chameleon 用人话讲

## 一句话版本

机器人做 long-horizon 任务时经常遇到"三个杯子长得一模一样，但球藏在哪个杯子取决于之前的 swap history"这种问题。现有方法要么把 memory 压成文字（丢了细节），要么存一堆图片用 similarity 检索（相似≠有用）。Chameleon 说我直接抄人脑的 EC-HC-PFC episodic memory 电路，把感知、记忆、决策分成三个模块，让 memory 写入时保留 geometry 细节，读取时用 future prediction 来 drive，policy 只看一个 compact state。

---

## 问题到底是什么

想象你在玩 shell game：三个一模一样的杯子，ball 藏在其中一个，然后 shuffle。shuffle 完了你看到三个杯子，视觉上完全 identical，但是你要抓对那个。

这就是 **perceptual aliasing**：observation $o_t$ 看起来一样，但是最优动作 $a_t^*$ 取决于 history。observation level 上 non-Markovian。

这个场景在 robotics 里到处都是：
- 你让 robot "clean the plate that just had raw meat"——三个 plate 视觉上一样干净，哪个是 contaminated 的取决于之前 user 放了哪个
- 你让 robot 依次加三种 seasoning——每次回到 prep pose，spoon 视觉上一样，加了哪些取决于之前 action
- drawer 里取东西、container 整理、多步 cooking……都是 aliasing

现有方法的两种死法：

**死法一：language-centric memory**。把 experience 总结成 "ball is under a cup"。语义没错，但是哪个 cup？信息丢了。RAG-style 的东西在 robotics 上经常这样挂掉。Ref: [RAG original paper](https://arxiv.org/abs/2005.11401)

**死法二：visual buffer + similarity retrieval**。存一堆历史 frame，用 embedding similarity 找相关的。问题是相似场景可能对应不同 history——三个杯子的 observation 几乎一模一样，similarity retrieval 会把 decision-irrelevant 的 episode 也捞回来，干扰决策。Ref: [Embodied-RAG](https://arxiv.org/abs/2409.18313)

所以 thesis 是两条：
1. Memory **写入**时要保留 disambiguating 的 fine-grained cues（不能 semantic compress）
2. Memory **读取**要由 decision utility 驱动（不能 similarity 驱动）

---

## 人脑怎么解决这个的

作者从神经科学里扒了三个功能原则：

**Entorhinal Cortex (EC)**：把多模态 sensory 整合成 unified representation。相当于一个 multimodal fusion 接口。

**Dentate Gyrus (DG) → pattern separation**：DG 的 granule cells 极其 sparse，把相似的 EC 输入 decorrelate 成 distinct 的 population codes。功能上就是"相似的 experience 存成不同的 trace，减少 interference"。

**CA3 → pattern completion**：CA3 有 recurrent collateral 形成 auto-associative network，partial cue 能 trigger 整个 stored pattern 的恢复。功能上就是"从部分线索恢复完整 episode"。

**CA1**：把 HC recall 整合成 output representation 传给 downstream。

**PFC (Prefrontal Cortex)**：goal-directed control——根据当前 goal bias 哪些 memory 该被 recall，并且做 prospective simulation（"想象未来会怎样"）来指导决策。

Ref: [Bakker 2008 pattern separation](https://www.science.org/doi/10.1126/science.1152882), [Allen & Fortin 2013](https://www.pnas.org/doi/10.1073/pnas.1301199110), [ scrub jays episodic memory](https://link.springer.com/article/10.3758/s13420-024-00617-x)

Chameleon 把这套搬过来：perception 做 EC-style integration，memory 做 DG-style separation + CA3-style completion，HoloHead 做 PFC-style prospective imagination。

---

## 架构：Perception → Memory → Policy

整体就三步：

```
cameras + proprioception → [Perception] → x_t (patch tokens)
                                              ↓
                              [Memory] ← h_{t-1} → h_t (decision state)
                                              ↓
                              [Policy] → future EE trajectory
```

关键设计：**policy 只看 $h_t$**。这是 single-state conditioning。如果 robot 成功完成 long-horizon task，那 disambiguation 所需的信息**必然**在 $h_t$ 里。这把 memory 的责任 explicit 化——不能甩锅给 policy 自己去历史里捞东西。

---

## Perception：geometry-grounded 双流

### 为什么要两路

Ventral stream（"what"）：frozen DINOv2 提 patch tokens，给 appearance evidence。这个大家都会做。

Dorsal stream（"where/how"）：这是 paper 的一个亮点。用 **end-effector (EE) pose 作为 action-centric geometric anchor**，把 3D 几何信息注入 2D patch。

具体做法：每个 patch $i$ 在 view $v$ 里算一个 7 维 descriptor：

$$g_{t,i}^v = [u_{t,i}^v, r_{t,i}^v, \rho_{t,i}^v, \cos\theta_{t,i}^v]$$

- $u_{t,i}^v$：patch center 的 2D image coordinate
- $r_{t,i}^v$：从 camera center 穿过 patch 的 3D unit ray
- $\rho_{t,i}^v = \|u_{t,i}^v - u_t^{\text{EE},v}\|_2$：patch 到 EE 投影的 image-plane 距离（衡量 action relevance——离 EE 近的 patch 更重要）
- $\cos\theta_{t,i}^v$：patch ray 和 camera-to-EE 方向的夹角余弦

这个 descriptor 编码了"这个 patch 在 image 哪里、3D 哪个方向、离 action center 多近、朝 action center 的视角如何"。

### Cross-view attention 怎么做

Front camera 和 hand camera 互相 enhance，但是 attention logits 加了两个 bias：

$$\text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B_{ab}^{\text{epi}} + b_t^a \mathbf{1}^\top + \mathbf{1}(b_t^b)^\top\right)$$

**Epipolar bias** $B_{ab}^{\text{epi}}$：用 fundamental matrix $F_{ab}$ 算 point-to-epipolar-line distance 的负数。两个 patch 如果不符合 epipolar geometry（不可能对应同一个 3D 点），attention 就被压低。这是几何先验。

**Unary geometric bias** $b_t^a, b_t^b$：从 EE-anchored descriptor 学出来的 per-patch bias，broadcast 成 pairwise 形式。让 attention 倾向于 action-relevant patches。

这两个 bias 加起来，cross-view 通信只在"几何上可能对应 + action 上相关"的 patches 之间发生。Figure 5 的可视化很 clean：有 dorsal stream 时 attention 集中在正确 target 上，没有时 attention 在 distractors 上 diffuse。

最后 FiLM modulation 把 geometry code 注入 feature：

$$\tilde{V}_t^v(i) = \gamma(C_{t,i}^v) \odot \bar{V}_t^v(i) + \beta(C_{t,i}^v)$$

输出 $x_t = \text{Concat}(\tilde{V}_t^f, \tilde{V}_t^h) \in \mathbb{R}^{512 \times 768}$（两个 view 各 256 tokens）。

**直觉**：ventral 给"看到什么"，dorsal 给"和 action 的几何关系"，两者通过 geometry-constrained attention 融合。这避免了"把两个视角里不相关的 region 混在一起"的错误，对应生物学的 ventral/dorsal 双流分工。Ref: [Goodale & Milner](https://www.sciencedirect.com/science/article/pii/0028393292900279)

---

## Memory：最复杂的部分

### 多模态融合

把 visual tokens + proprioception token + phase token 各加 modality/view embedding，concat 后 layer norm，得到 $z_t \in \mathbb{R}^{M \times d}$。

### Dynamic context modulation

每一层用上一层的 working state 给所有 token 加 bias：

$$\tilde{z}_t^{(\ell)} = z_t + \mathbf{1}(W_c^{(\ell)} h_t^{(\ell-1)})^\top$$

意思：同一个 evidence set 在不同层会被不同的 working context 重新解读。Layer 1 可能关注 immediate sensory contingencies，Layer 2 可能关注 long-horizon abstract recall。

### Spatial-Temporal Anchors（pattern separation 的实现）

这是最关键的设计。把 visual evidence 显式 factorize 到 spatial × temporal 两个维度。

**Spatial anchors**：用 routing weights $\pi_{t,a,i}^{(\ell)}$ 把 $N=512$ 个 visual tokens 分成 $A=8$ 个 component：

$$u_{t,a}^{(\ell)} = \sum_{i=1}^N \pi_{t,a,i}^{(\ell)} \bar{x}_{t,i}$$

每个 spatial anchor 是 visual evidence 的一个 localized summary。然后把 proprioception token 和 phase token attach 上去，形成 $\tilde{z}_t^{(\ell,a)} \in \mathbb{R}^{3 \times d}$。

**Temporal anchors**：每个 spatial anchor 用 $B=4$ 个 learned temporal query 读出，得到 slot feature：

$$f_{t,a,b}^{(\ell)} = \text{Attn}(q_{a,b}^{(\ell)}, \tilde{z}_t^{(\ell,a)})$$

最终得到 $8 \times 4 = 32$ 个 spatiotemporal slots。Spatial index 选 evidence 的哪个 component，temporal index 选用哪个时间尺度处理。

**为什么这能实现 pattern separation**：相似的 observation 如果在 spatial 或 temporal 上有 subtle 差异，会被路由到不同的 slot combinations。不同 slot 维护不同的 latent trajectory，自然 decorrelate 了。

### Episodic + Working Memory 分离

每个 slot $(a,b)$ 维护一个 episodic state $m_{t,a,b}^{(\ell)}$，用 Mamba-style selective SSM 更新：

$$m_{t,a,b}^{(\ell)} = \text{SSM}_{\text{epi}}^{(\ell)}(m_{t-1,a,b}^{(\ell)}; \theta_{t,a,b}^{(\ell)})$$

关键 trick：**每个 temporal index $b$ 赋予不同的 base step size $\Delta t_b^{(0)}$**。Paper 里用 $\{0.001, 0.005, 0.02, \text{flexible}\}$，对应 short → long half-life。同时 step size 还会被 slot content 调制。

这是 explicit multi-timescale prior——有的 slot 记短期细节，有的 slot 记长期 context。比单纯 learn 一个 RNN 的 forgetting gate 更 interpretable。

为了 efficiency，作者用 fused scan kernel 一次更新所有 32 个 episodic states。

Episodic readout 聚合：

$$r_t^{(\ell)} = \frac{1}{AB} \sum_{a,b} W_{a,b}^{(\ell)} r_{t,a,b}^{(\ell)}$$

然后注入 Working Memory：

$$h_t^{(\ell)} = \text{SSM}_{\text{work}}^{(\ell)}\left(h_{t-1}^{(\ell)}; \text{LN}(h_{t-1}^{(\ell)} + \text{Proj}_r^{(\ell)}(r_t^{(\ell)}))\right)$$

**Asymmetric design 是关键**：
- Episodic Memory: slot-indexed, persistent, latent（存 rich traces）
- Working Memory: compact, exposed, 是层间唯一通信通道

storage 和 exposure 分离。Standard RNN 里一个 state 同时承担存储 + 计算 + 输出三个角色，容易 interference。这里 rich traces 在 episodic bank 里保留，只有压缩 working summary 传到上层。

### Hierarchical Fusion

L=2 层的 working readout 用 task query $q_{\text{task}}$ 融合：

$$h_t = \sum_\ell \alpha_{t,\ell} v_{t,\ell}, \quad \alpha_{t,\ell} = \text{softmax}(\langle q_{\text{task}}, k_{t,\ell}\rangle)$$

不是简单平均，是 task-conditioned 自适应选择。

### HoloHead：PFC-style prospective imagination

这是让 $h_t$ 变成 **predictive** representation 的关键 regularizer。

条件在 $h_t$ 上，预测两组 waypoints：
- **Anchor waypoints** ($N_a=8$): 接下来 8 帧的 EE 位置（dense, short-horizon）
- **Compass waypoints** ($N_c=8$): 当前 phase 剩余部分用 log-spacing 采样（sparse-to-dense, long-horizon）

Compass 的 sampling 公式：

$$\Delta_j^{(c)} = \lfloor (N_a+1)^{1-\alpha_j} R_t^{\alpha_j} \rfloor, \quad \alpha_j = \frac{j-1}{N_c-1}$$

$R_t$ 是当前 phase 剩余帧数。近端 dense，远端 sparse，最后一个固定在 phase endpoint。

Loss 是 2D 和 3D 各自的 L1：

$$\mathcal{L}_{\text{holo}} = \sum_{\xi \in \{2D, 3D\}} \left(\|\hat{W}_{t,A}^\xi - W_{t,A}^\xi\|_1 + \|\hat{W}_{t,C}^\xi - W_{t,C}^\xi\|_1\right)$$

**为什么这个重要**：没有 HoloHead，$h_t$ 容易 collapse 到 instantaneous appearance（只编码当前看到的东西）。HoloHead 逼迫 $h_t$ 必须能 generate 未来的轨迹——这意味着 $h_t$ 必须编码 task-relevant latent state（球在哪个杯子、已经加了哪些 seasoning），因为只有这些信息才能 predict 未来。

这本质上是用 self-supervised forward prediction 来 shape latent state，和 [Dreamer](https://arxiv.org/abs/1912.01603) 的 world model 思路相通，但用在 supervised imitation learning 里。

---

## Policy：Conditional Rectified Flow

$$x_\tau = (1-\tau)x_0 + \tau x_1, \quad u_\tau = x_1 - x_0$$
$$\hat{u}_\tau = v_\theta(x_\tau, \tau, c_t), \quad c_t = W_{\text{ctx}} h_t$$
$$\mathcal{L}_{\text{flow}} = \mathbb{E}[\|\hat{u}_\tau - u_\tau\|^2]$$

- $x_1$: ground-truth 未来 H=8 步 EE trajectory
- $x_0 \sim \mathcal{N}(0,I)$: 噪声起点
- $\tau \sim \mathcal{U}(0,1)$: flow time
- $c_t$: memory readout 投影

用 rectified flow 而不是 diffusion，因为 path 更直，ODE 求解更稳定，对 long-horizon 更友好。Inference 50 步 ODE。

Velocity network 是 6-layer Transformer，用 AdaLN 让 $\tau$ 调制每个 block（DiT-style）。Self-attention 在 trajectory tokens 间，cross-attention 从 trajectory tokens 到 single conditioning token $c_t$。

Ref: [Rectified Flow](https://arxiv.org/abs/2209.03003), [Flow Matching](https://arxiv.org/abs/2210.02747), [DiT](https://arxiv.org/abs/2212.09748)

---

## Camo-Dataset：三个 perceptually aliased 任务

| Task | Memory Type | 关键挑战 | Chance Level |
|------|-------------|----------|--------------|
| Clean a specified plate | Episodic (event-object binding) | 三个 plate 视觉一样，哪个 contaminated | DSR=1/3 |
| Play shell game | Spatial (occluded tracking) | 三个 cup 一样，ball 在哪 | DSR=1/3 |
| Add various seasonings | Sequential (sub-goal tracking) | 三个 spoon 一样，已加哪些 | DSR=1/27 |

每个 task 120 demos，test 36 trials。Chance level 故意设计成不同——sequential task 是 1/27，因为 9 phases 顺序敏感。

**为什么这三个 task 互补**：
- Episodic：测 event-object binding（HC 功能）
- Spatial：测 occluded object tracking（EC-HC cognitive map）
- Sequential：测 sub-goal tracking（PFC working memory）

---

## 实验结果讲人话

### 主结果

Chameleon 在三个 task 上 DSR (Cohen's κ)：
- Episodic: **100% (100%)**——完美克服 aliasing
- Spatial: **73.5% (60.3%)**——大幅超过 baseline
- Sequential: **72.2% (71.2%)**——baseline 全部 0%

Baseline 的惨状：
- Diffusion Policy / Flow Matching / ACT 在 sequential task 上 DSR=0%, κ=-3.8（比随机猜还差）
- 它们全部 collapse 到单一 dominant behavior，不管 underlying memory state 是什么都预测同一个动作

**MSR（manipulation success rate）** Chameleon 在 episodic 上 86.1%，比 Diffusion Policy 的 91.7% 略低。这很合理——Chameleon 的创新在 memory decision，low-level execution 没有显著超越 baseline。

### Cohen's κ 为什么重要

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

$p_o$ 是实际 DSR，$p_e$ 是 chance level DSR。

κ=1 完美，κ=0 和随机一样，κ<0 比随机还差。Chameleon 在 episodic 上 κ=100 意味着 $p_o=1, p_e=1/3$，所以 $\kappa = (1-1/3)/(1-1/3)=1$。

Baseline κ 经常是 0 或负数，说明它们根本没学到 memory-dependent decision。

### Ablation 怎么解读

**w/o Memory**：所有 task 退化到 chance level。Sanity check 通过——benchmark 确实 non-Markovian。

**Memory Bank（similarity retrieval）**：MSR=0%，完全失败。因为 visually similar 但 decision-irrelevant 的 episode 被错误 retrieve，policy 混乱。这是对 RAG-style 方法的直接打击。

**Vanilla Mamba**：把 structured memory 换成 monolithic Mamba。Sequential task 直接归零。**单纯的 recurrence 不够**——必须要有 spatial-temporal slot structure 来 reduce interference。Mamba 的 selectivity 在这里不够，因为它没有 explicit multi-timescale prior + slot-indexed engram。

**w/o Dorsal Stream**：Episodic 几乎没变（96.8 vs 100），但 spatial 大幅下降（51.5 vs 73.5）。Dorsal stream 主要在 spatial reasoning 上发挥作用。Figure 5 显示去掉 dorsal 后 attention 在 distractors 上 diffuse。

**w/o HoloHead**：所有 task 下降，sequential 归零。Latent imagination 对 $h_t$ 稳定性至关重要——没有它 $h_t$ collapse 到 instantaneous appearance。

### Mechanistic Validation

UMAP 可视化 $h_t$：episodic/spatial task 上不同 latent state 分到不同 manifold cluster（pattern separation 成功）。Sequential task 在 shared manifold 上沿 stage 演化。

HoloHead rollout：从随机 timestep 预测未来 trajectory，保持 goal-consistent（pattern completion 成功）。

---

## 我的几点直觉

### 1. Single-state conditioning 是强约束

Policy 只看 $h_t$，逼 memory 承担全部 disambiguation 责任。这种 architectural constraint 比 auxiliary loss 更有 teeth。可以推广到其他 modular design——给中间模块一个 single-interface constraint，逼它承担责任。

### 2. Multi-timescale prior 很优雅

$\Delta t_b^{(0)} = \{0.001, 0.005, 0.02, \text{flexible}\}$ 对应 explicit half-life schedule。这比让 RNN 自己 learn forgetting gate 更 interpretable，也更 data-efficient。类似 signal processing 里的 multi-resolution analysis（wavelet 思路），但在 SSM 框架里。

### 3. HoloHead 是 self-supervised world model

本质上是用 forward prediction shape latent state，和 [Dreamer](https://arxiv.org/abs/1912.01603), [V-JEPA](https://arxiv.org/abs/2301.08243), [PlaTe](https://arxiv.org/abs/2306.06494) 一类工作思想相通。但是用在 supervised imitation learning 里，巧妙地把 RL 里的 world model 思路搬到 IL。

### 4. EE-anchored geometry 是好 inductive bias

用 EE pose 作为 geometric anchor，把 action relevance 注入 perception。这比单纯 epipolar constraint 多了 action-aware prior。在 robotics 里 EE 是 action 的直接执行者，用它做 anchor 很自然。

### 5. 局限

- 只在 single UR5e 上做，三个 controlled task。Real-world aliasing 可能更复杂（光照变化、视角扰动）
- $\psi_t$ phase indicator 在 open-ended setting 里怎么定义？
- Compass waypoint 需要 phase endpoint $t_{\text{end}}$，inference 时怎么知道？
- Memory Bank ablation 的 similarity retrieval 可能太 naive，更 sophisticated 的 [RAPTOR](https://arxiv.org/abs/2310.06825) 或 [graph retrieval](https://arxiv.org/abs/2404.17724) 可能有不同表现
- 没和 VLA foundation model 结合，zero-shot 能力弱。未来方向应该是 episodic memory stack + foundation model = structural encoding + high-level priors

### 6. 和你之前工作的联系

Karpathy 你在 [nanoGPT](https://github.com/karpathy/nanogpt) 里强调 simplicity。Chameleon 不算 simple——spatial-temporal slots + episodic/working 分离 + HoloHead regularization 不少 moving parts。但每个 component 都有清晰 functional motivation。

如果你要 build 类似 intuition，我会关注：
- **Single-state sufficiency constraint**：是否可以推广到其他 modular architectures（比如 VLA 里的 memory-augmented policy）
- **Multi-timescale prior 在 SSM 里**：是否可以替代 attention-based memory
- **Predictive regularizer 作为 memory shaper**：forward prediction shape latent state 的思路可以推广到其他 long-horizon RL

---

## 相关参考

- [Chameleon GitHub](https://github.com/gxyes/MARS_Chameleon)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/)
- [HippoRAG](https://arxiv.org/abs/2405.14831)
- [Embodied-RAG](https://arxiv.org/abs/2409.18313)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [V-JEPA](https://arxiv.org/abs/2301.08243)
- [DiT](https://arxiv.org/abs/2212.09748)
- [π0](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Bakker 2008 pattern separation](https://www.science.org/doi/10.1126/science.1152882)
- [Allen & Fortin 2013](https://www.pnas.org/doi/10.1073/pnas.1301199110)
- [Goodale & Milner ventral/dorsal](https://www.sciencedirect.com/science/article/pii/0028393292900279)
- [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
- [RAPTOR hierarchical retrieval](https://arxiv.org/abs/2310.06825)
- [nanoGPT](https://github.com/karpathy/nanogpt)

---

# Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation 深度讲解

## 1. 核心动机：Perceptual Aliasing 这个问题到底是什么

这篇 paper 抓住了一个非常本质的 robotics 问题：**perceptual aliasing**。在 shell game（猜杯子游戏）里，三个视觉上完全一样的杯子在 decision time 看起来一样，但是 ball 藏在哪个杯子里取决于 swap history。这意味着 observation level 上决策是 **non-Markovian** 的——同一个 $o_t$ 可能对应不同的最优动作 $a_t^*$，正确动作依赖于历史交互。

Karpathy 你肯定熟悉 POMDP 的概念，但这里作者强调的 **不是** 经典的 belief state 那一套（虽然思想上接近）。他们想要的是一种 **bio-inspired** 的 episodic memory 结构，从人类的 EC-HC-PFC（Entorhinal Cortex - Hippocampus - Prefrontal Cortex）回路里拿设计灵感。

为什么现有方法不够用？作者批判两类：

**Language-centric memory (RAG-style)**：把 experience 压缩成 text-like traces，比如 "the ball is under a cup"。这种 semantic compression 在 shell game 里就是灾难——语义正确但不足以 disambiguate 哪个杯子。Ref: [Lewis et al. 2020 RAG](https://arxiv.org/abs/2005.11401), [Karpukhin et al. 2020 DPR](https://arxiv.org/abs/2004.04906)

**Visual history buffer + similarity retrieval**：保留视觉历史但用 similarity-based retrieval，问题是相似场景可能对应不同 history，retrieval 会引入干扰。Ref: [Embodied-RAG](https://arxiv.org/abs/2409.18313), [HippoRAG](https://arxiv.org/abs/2405.14831)

所以核心 thesis 是：**memory writing** 必须保留 disambiguating 的 fine-grained perceptual cues，**memory retrieval** 必须由 decision utility 驱动，**而不是** perceptual similarity。

---

## 2. 整体架构：Perception → Memory → Policy 三段式

整体 pipeline 的三段式（Eq. 1）：

$$x_t = \Phi_{\text{perc}}(I_t^f, I_t^h, s_t, \psi_t)$$
$$h_t = \Phi_{\text{mem}}(x_t, h_{t-1})$$
$$\hat{x}_{t:t+H} = \Phi_{\text{pol}}(h_t)$$

变量说明：
- $I_t^f$: front camera 的 RGB image（固定视角，提供 global context）
- $I_t^h$: hand-mounted side camera 的 RGB image（提供 contact-scale 细节，应对 occlusion）
- $s_t$: robot proprioceptive state（end-effector pose + gripper state）
- $\psi_t$: optional task-phase indicator（一个 0/1 信号，标记 observe phase vs act phase，**但不**揭示 hidden task state）
- $x_t \in \mathbb{R}^{N \times d}$: perception 输出的 fused patch tokens
- $h_t$: 唯一的 decision state，是 memory 模块和 policy 之间唯一的接口
- $\hat{x}_{t:t+H}$: 预测的未来 H 步 end-effector pose trajectory

设计哲学：**single-state conditioning**——policy 只看到 $h_t$，如果 agent 在 long-horizon control 上成功，那么 disambiguation 所需的信息**必然**在 $h_t$ 里被表示。这是一种 sufficiency constraint，把 memory 的作用 explicit 化。

---

## 3. Perception 模块：Geometry-Grounded 双流融合

### 3.1 Ventral Stream：appearance evidence

用 frozen DINOv2 提取每个 view 的 patch tokens：

$$V_t^v = \text{DINO}(I_t^v) \in \mathbb{R}^{N_v \times d}$$

其中 $v \in \{f, h\}$，$N_v = 256$，$d = 768$。这里 DINO 提供强大的 pretrained visual prior，提供 "what" 的细粒度证据。Ref: [DINOv2](https://arxiv.org/abs/2304.07193)

### 3.2 Dorsal Stream：EE-anchored geometry codes

这是这篇文章一个关键创新点。作者用 end-effector (EE) pose 作为 action-centric geometric anchor，把 3D 信息投影到 2D image plane，给每个 patch 算几何描述符。

首先把 EE 3D 位置 $p_t \in \mathbb{R}^3$ 投影到每个 view：

$$u_t^{\text{EE},v} = \Pi(K^v, T_t^v, p_t) \in \mathbb{R}^2$$

变量：
- $K^v$: 相机 intrinsics
- $T_t^v \in SE(3)$: world-to-camera extrinsics（注意 hand camera 的 $T_t^h$ 随 EE 移动变化）
- $\Pi$: 标准 pinhole projection

对每个 patch $i$ 在 view $v$ 里，定义：

$$r_{t,i}^v = \frac{(K^v)^{-1} \tilde{u}_{t,i}^v}{\|(K^v)^{-1} \tilde{u}_{t,i}^v\|_2} \in \mathbb{R}^3$$

$$g_{t,i}^v = [u_{t,i}^v, r_{t,i}^v, \rho_{t,i}^v, \cos\theta_{t,i}^v] \in \mathbb{R}^7$$

变量解释：
- $u_{t,i}^v \in \mathbb{R}^2$: patch center 的 normalized image coordinate
- $\tilde{u}_{t,i}^v$: 它的 homogeneous form $\in \mathbb{P}^2$
- $r_{t,i}^v$: 从相机中心穿过 patch 的单位 ray 方向（3D）
- $\rho_{t,i}^v = \|u_{t,i}^v - u_t^{\text{EE},v}\|_2$: patch center 到 EE 投影的 image-plane 距离（衡量 action relevance）
- $\cos\theta_{t,i}^v = r_{t,i}^{v\top} d_t^v$: patch ray 和 camera-to-EE 方向的 cosine similarity，其中 $d_t^v$ 是 camera frame 下从相机中心到 EE 的单位方向向量

这个 7 维 descriptor $g_{t,i}^v$ 编码了 "这个 patch 在 image 哪里 + 在 3D 哪个方向 + 离 EE 多近 + 朝 EE 的视角如何"。一个 MLP 把它转成 unary patch bias $b_{t,i}^v$ 和 conditioning code $C_{t,i}^v$。

### 3.3 Geometry-Biased Bidirectional Cross-View Enhancement

这个是 cross-view attention 的关键设计（Eq. 3）：

$$\text{Attn}_{a \to b}(Q,K,U) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B_{ab}^{\text{epi}} + b_t^a \mathbf{1}^\top + \mathbf{1}(b_t^b)^\top\right) U$$

这里有三个 bias 项：

**Epipolar feasibility bias** $B_{ab}^{\text{epi}}$：基于 fundamental matrix $F_{ab}$ 计算的 negative point-to-epipolar-line distance：

$$d_{ij} = \frac{(\tilde{u}_{t,j}^{b\top} F_{ab} \tilde{u}_{t,i}^a)^2}{\|(F_{ab}\tilde{u}_{t,i}^a)_{1:2}\|_2^2 + \epsilon}$$
$$B_{ab}^{\text{epi}}(i,j) = -\frac{d_{ij}}{\tau}$$

这个项惩罚那些不符合 epipolar geometry 的 patch pairs，确保 cross-view attention 只在几何上可能对应的 patches 之间发生。$\tau$ 是 temperature。

**Unary geometric biases** $b_t^a \mathbf{1}^\top + \mathbf{1}(b_t^b)^\top$：来自 dorsal stream 的 per-patch bias 被 broadcast 成 pairwise 形式，让 attention 倾向于 action-relevant patches（靠近 EE 投影的 patches）。这个设计避免了显式的 pairwise matcher，但仍然让 action relevance 影响 cross-view 通信。

最后做 FiLM-style modulation（Eq. 4）：

$$\tilde{V}_t^v(i) = \gamma(C_{t,i}^v) \odot \bar{V}_t^v(i) + \beta(C_{t,i}^v)$$

$$x_t = \text{Concat}(\tilde{V}_t^f, \tilde{V}_t^h) \in \mathbb{R}^{N \times d}$$

其中 $\gamma(\cdot), \beta(\cdot)$ 是 learned affine functions。

**直觉**：ventral 给 appearance，dorsal 给 geometry，cross-view attention 被 epipolar + EE-anchor 双重约束，避免 incompatible evidence 混合。这对应大脑 ventral "what" stream + dorsal "where/how" stream 的分工。Ref: [Goodale & Milner ventral/dorsal streams](https://www.sciencedirect.com/science/article/pii/0028393292900279)

---

## 4. Memory 模块：Hierarchical Differentiable Memory Stack

这是 paper 最核心也最复杂的部分。

### 4.1 Multimodal Fusion (Eq. 5-6)

把 visual tokens、proprioception token、phase token 各自加上 learned modality embedding 和 view embedding：

$$\bar{x}_t^v = \tilde{V}_t^v + A_{\text{vis}}^{\text{mod}} + A_v^{\text{view}}, \quad v \in \{f,h\}$$
$$\bar{p}_t = p_t + A_{\text{prop}}^{\text{mod}}$$
$$\bar{\varphi}_t = \varphi_t + A_{\text{phase}}^{\text{mod}}$$

$$z_t = \text{LN}(\text{Concat}[\bar{x}_t^f, \bar{x}_t^h, \bar{p}_t, \bar{\varphi}_t]) \in \mathbb{R}^{M \times d}$$

$M = N + 1$（无 phase）或 $M = N + 2$（有 phase）。

### 4.2 Dynamic Context Modulation (Eq. 7)

每一层用上一层的 working state 给所有 token 加 bias：

$$\tilde{z}_t^{(\ell)} = z_t + \mathbf{1}(W_c^{(\ell)} h_t^{(\ell-1)})^\top \in \mathbb{R}^{M \times d}$$

变量：
- $h_t^{(\ell-1)} \in \mathbb{R}^{d_w}$: layer $\ell-1$ 的 working state
- $W_c^{(\ell)} \in \mathbb{R}^{d \times d_w}$: 投影矩阵
- $\mathbf{1} \in \mathbb{R}^M$: all-ones vector

**直觉**：同一个 evidence set $z_t$ 在不同层会被不同的 working context 重新 interpret，content-adaptive。这避免了 explicit token rewriting。

### 4.3 Spatial and Temporal Anchors (Eq. 8-9) —— Pattern Separation

这是实现 DG-style pattern separation 的核心机制。

**Spatial anchors**：用 routing weights 把 visual evidence 分解成 $A$ 个 component：

$$u_{t,a}^{(\ell)} = \sum_{i=1}^N \pi_{t,a,i}^{(\ell)} \bar{x}_{t,i} \in \mathbb{R}^d$$

其中 routing weights $\pi_{t,a,i}^{(\ell)} = \text{softmax}(\text{Router}^{(\ell)}(\bar{x}_{t,i}))$ 满足 $\sum_a \pi_{t,a,i}^{(\ell)} = 1$。每个 spatial anchor $a$ 形成一个 visual summary，然后和 proprioception + phase token concat：

$$\tilde{z}_t^{(\ell,a)} = \text{Concat}[u_{t,a}^{(\ell)}, \bar{p}_t, \bar{\varphi}_t] \in \mathbb{R}^{3 \times d}$$

**Temporal anchors**：每个 spatial anchor 进一步被 $B$ 个 temporal query 读出，对应不同的时间尺度：

$$f_{t,a,b}^{(\ell)} = \text{Attn}(q_{a,b}^{(\ell)}, \tilde{z}_t^{(\ell,a)}) \in \mathbb{R}^d$$

这里 $q_{a,b}^{(\ell)}$ 是 learned temporal queries，$a = 1,...,A$，$b = 1,...,B$。

最终得到 $A \times B$ 的 spatiotemporal slot matrix $\{f_{t,a,b}^{(\ell)}\}$。Paper 里 $A=8$，$B=4$，所以每层 32 个 slots。

**直觉**：这相当于把 evidence 显式 factorize 到 spatial×temporal 两个维度。spatial index 决定关注 evidence 的哪个 component，temporal index 决定用哪个时间尺度处理。这是 pattern separation 的实现——不同 spatial-temporal component 被分配到不同 latent trajectories。

### 4.4 Spatial-Temporal Memory: Episodic + Working States (Eq. 10-13)

每个 slot $(a,b)$ 维护一个 episodic state $m_{t,a,b}^{(\ell)}$，用 Mamba-style selective SSM 更新：

$$\theta_{t,a,b}^{(\ell)} = \text{Proj}_{a,b}^{(\ell)}(f_{t,a,b}^{(\ell)})$$
$$m_{t,a,b}^{(\ell)} = \text{SSM}_{\text{epi}}^{(\ell)}(m_{t-1,a,b}^{(\ell)}; \theta_{t,a,b}^{(\ell)})$$

关键设计：每个 temporal index $b$ 被赋予不同的 base step size $\Delta t_b^{(0)}$，对应 explicit half-life schedule。Paper 里用的是 $\{0.001, 0.005, 0.02, \text{flexible}\}$，对应 short to long retention timescales。同时 step size 还会被 slot content进一步调制。Ref: [Mamba](https://arxiv.org/abs/2312.00752)

为了 efficiency，作者实现了一个 fused slot-wise scan kernel，所有 $A \times B$ episodic states 在 single pass 里更新。

Episodic readout 聚合成 recall vector：

$$r_t^{(\ell)} = \frac{1}{AB} \sum_{a=1}^A \sum_{b=1}^B W_{a,b}^{(\ell)} r_{t,a,b}^{(\ell)} \in \mathbb{R}^{d_w}$$

其中 $r_{t,a,b}^{(\ell)} = \text{Read}_{\text{epi}}^{(\ell)}(m_{t,a,b}^{(\ell)})$。

然后注入 Working Memory（Eq. 13）：

$$h_t^{(\ell)} = \text{SSM}_{\text{work}}^{(\ell)}\left(h_{t-1}^{(\ell)}; \text{LN}(h_{t-1}^{(\ell)} + \text{Proj}_r^{(\ell)}(r_t^{(\ell)}))\right)$$

这里 $h_t^{(\ell)} \in \mathbb{R}^{d_w}$ 是 layer $\ell$ 的唯一输出。

**Asymmetric design**：
- Episodic Memory: slot-indexed, persistent, latent（存储 rich traces）
- Working Memory: compact, exposed, 是 layer 间唯一通信通道

这样 storage 和 exposure 分离：rich traces 在 episodic bank 里保留，但只有压缩的 working summary 传到上层。**避免** standard RNN 里一个 state 同时承担存储、计算、输出三个角色导致的 interference。

### 4.5 Hierarchical Memory Fusion (Eq. 14)

L 层 working readout 用 task query $q_{\text{task}}$ 融合：

$$\alpha_{t,\ell} = \text{softmax}_\ell(\langle q_{\text{task}}, k_{t,\ell}\rangle)$$
$$h_t = \sum_{\ell=1}^L \alpha_{t,\ell} v_{t,\ell}$$

其中 $k_{t,\ell} = U_\ell y_t^{(\ell)}$，$v_{t,\ell} = \text{Enc}_\ell(y_t^{(\ell)})$，$y_t^{(\ell)} = \text{LN}(W_o^{(\ell)} h_t^{(\ell)})$。

**直觉**：这不是简单平均。不同层 capture 不同 abstraction level（shallow layers 强调 short-horizon sensory contingencies，deeper layers 强调 long-horizon abstract recall），task query 自适应选择最相关的 working summaries。这是两阶段 selection：先 slot-level factorization，再 layer-wise hierarchical fusion。

### 4.6 HoloHead: Latent Imagination Objective (Eq. 15-16)

这是让 $h_t$ 变成 **predictive** representation 的关键 regularizer。

$$\begin{bmatrix}\hat{w}_{t:t+N_a}^{2D} \\ \hat{w}_{t:t+N_c}^{2D} \\ \hat{w}_{t:t+N_a}^{3D} \\ \hat{w}_{t:t+N_c}^{3D}\end{bmatrix} = \text{HoloHead}(h_t)$$

预测两组 waypoints：
- **Anchor waypoints** ($N_a = 8$): 接下来 $N_a$ 个连续帧的 EE 位置（short-horizon，dense）
- **Compass waypoints** ($N_c = 8$): 当前 phase 剩余部分用 logarithmic spacing 采样（long-horizon，sparse-to-dense）

Compass waypoints 的 sampling 公式（Eq. 41）：

$$\alpha_j = \frac{j-1}{N_c - 1}$$
$$\Delta_j^{(c)} = \lfloor (N_a + 1)^{1-\alpha_j} R_t^{\alpha_j} \rfloor, \quad j = 1, ..., N_c$$

其中 $R_t = t_{\text{end}} - t$ 是当前 phase 剩余帧数。这个 log-spacing 让 Compass waypoints 在近端 dense、远端 sparse，且最后一个 waypoint 固定在 phase endpoint。

Loss（Eq. 16）：

$$\mathcal{L}_{\text{holo}} = \sum_{\xi \in \{2D, 3D\}} \left(\|\hat{w}_{t:t+N_a}^\xi - w_{t:t+N_a}^\xi\|_1 + \|\hat{w}_{t:t+N_c}^\xi - w_{t:t+N_c}^\xi\|_1\right)$$

2D 和 3D 各自用 L1 loss，权重都是 0.5。

**直觉**：这相当于 PFC 的 prospective simulation——把 episodic fragments 重新组合成对未来轨迹的预测。Anchor 段迫使 $h_t$ 编码 immediate actionable geometry，Compass 段迫使 $h_t$ 编码 phase-level intent。这个 regularizer 防止 $h_t$ collapse 到 instantaneous appearance，让它变成 decision-sufficient 的 predictive state。这呼应了 [forward prediction in world models](https://arxiv.org/abs/1803.10122) 的思路。

---

## 5. Policy: Conditional Rectified Flow Matching (Eq. 17)

$$x_\tau = (1-\tau)x_0 + \tau x_1, \quad u_\tau = x_1 - x_0$$
$$\hat{u}_\tau = v_\theta(x_\tau, \tau, c_t), \quad \mathcal{L}_{\text{flow}} = \mathbb{E}[\|\hat{u}_\tau - u_\tau\|^2]$$

变量：
- $x_1 \in \mathbb{R}^{H \times d_{\text{raw}}}$: ground-truth future trajectory
- $x_0 \sim \mathcal{N}(0, I)$: 初始噪声
- $\tau \sim \mathcal{U}(0,1)$: flow time
- $c_t = W_{\text{ctx}} h_t$: memory readout 投影到 policy hidden dim
- $v_\theta$: velocity network（6-layer Transformer，hidden 384，8 heads）

Inference 时解 ODE $\dot{x} = v_\theta(x, \tau, c_t)$，50 步 fixed-step solver。

这里用 **rectified flow** 而不是 diffusion，因为 rectified flow 的路径更直，ODE 求解更稳定，对 long-horizon control 更友好。Ref: [Rectified Flow](https://arxiv.org/abs/2209.03003), [Flow Matching](https://arxiv.org/abs/2210.02747), [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)

Policy 内部用 AdaLN（adaptive layer norm）让 flow time $\tau$ 调制每个 block，类似 [DiT](https://arxiv.org/abs/2212.09748) 的做法。Self-attention 在 trajectory tokens 之间，cross-attention 从 trajectory tokens 到 single conditioning token $c_t$。

---

## 6. Camo-Dataset：三个 Perceptually Aliased 任务

作者构造了三个 task category，对应三种 memory 类型：

| Category | Task | Memory Type | Biological Substrate | Chance Level |
|----------|------|-------------|---------------------|--------------|
| Episodic | Clean a specified plate | Event-object binding | HC-centered | DSR=1/3, CSR=1/3 |
| Spatial | Play shell game | Occluded object tracking | EC-HC cognitive map | DSR=1/3, CSR=1/3 |
| Sequential | Add various seasonings | Sub-goal tracking | PFC-centered working memory | DSR=1/27, CSR=1/9 |

**Clean a specified plate**：用户放一个 contaminated plate（视觉上干净），robot 要清洁指定的那个。Decision stage 三个 plate 视觉上相似。

**Play shell game**：用户藏 cube 在一个 cup 下，随机 swap cups，robot 要追踪正确的 cup。Decision stage 三个 cup 完全相同。这个 task 对应 spatial tracking。

**Add various seasonings**：依次加三种 seasoning（green-red-yellow 顺序），每次操作后回到 prep pose。Decision stage 三个 spoon 相似，必须记住已经加了哪些。这个 task 是 9 phases 的长序列，chance level 是 $1/27$。

每个 task 收集 120 demonstrations，test 时做 36 trials。Frame rate 30 FPS，training 时 temporal stride 4 downsample。

---

## 7. 实验结果分析

### 7.1 主实验（Table 2）

| Method | Episodic DSR(κ) | Episodic SR | Spatial DSR(κ) | Spatial SR | Sequential DSR(κ) | Sequential SR |
|--------|-----------------|-------------|----------------|------------|---------------------|----------------|
| Diffusion Policy | 33.3 (0.0) | 30.6 | 34.3 (1.4) | 33.3 | 0.0 (-3.8) | 0.0 |
| Flow Matching | 30.0 (-5.0) | 25.0 | 25.7 (-11.4) | 25.0 | 0.0 (-3.8) | 0.0 |
| ACT | 28.0 (-8.0) | 19.4 | 35.5 (3.2) | 30.6 | NA (NA) | 0.0 |
| **Chameleon** | **100.0 (100.0)** | **86.1** | **73.5 (60.3)** | **69.4** | **72.2 (71.2)** | **36.1** |

观察：
1. **Episodic task** Chameleon DSR 100%，Cohen's κ 也是 100%，意味着完全克服了 chance level。这说明在 event-object binding 上 episodic memory 设计非常有效。
2. **Sequential task** 是最难的（chance level 1/27）。Diffusion Policy/Flow Matching/ACT 全部 0% DSR，κ 是负的（-3.8），意味着它们甚至比随机猜还差——collapse 到单一 dominant behavior。Chameleon 72.2% DSR, 71.2% κ 说明 structured memory 在长序列上能保持 task progression。
3. **MSR (Manipulation Success Rate)** Chameleon 在 episodic task 上 86.1%，比 Diffusion Policy 的 91.7% 略低。这说明 Chameleon 主要提升 decision，manipulation 执行能力没有显著超越 baseline——这是合理的，因为创新点在 memory 而不是 low-level control。

### 7.2 Cohen's κ 的意义

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

其中 $p_o$ 是 observed DSR，$p_e$ 是 chance-level DSR。

κ = 1 表示完全一致，κ = 0 表示和 chance 一样，κ < 0 表示比 chance 还差。Chameleon 在 episodic 上 κ = 100.0，意味着 $p_o = 1$ 且 $p_e = 1/3$，所以 $\kappa = (1 - 1/3)/(1 - 1/3) = 1$。在 sequential 上 $p_e = 1/27$，$\kappa = (0.722 - 0.037)/(1 - 0.037) \approx 0.712$，对得上。

### 7.3 Ablation Study（Table 2 下半部分）

| Variant | Episodic DSR(κ) | Episodic CSR | Spatial DSR(κ) | Spatial CSR | Sequential DSR(κ) | Sequential CSR |
|---------|-----------------|--------------|----------------|-------------|---------------------|----------------|
| w/o Memory | 37.0 (5.6) | 36.7 | 34.5 (1.7) | 35.0 | 0.0 (-3.8) | 6.7 |
| Memory Bank* | NA | 33.3 | NA | 31.7 | NA | 11.7 |
| Vanilla Mamba* | 28.0 (-8.0) | 31.7 | 30.3 (-4.5) | 28.3 | 0.0 (-3.8) | 8.3 |
| w/o Dorsal Stream | 96.8 (95.2) | 100.0 | 51.5 (27.3) | 65.0 | 66.7 (65.4) | 71.7 |
| w/o HoloHead | 60.7 (41.1) | 61.7 | 46.7 (20.0) | 53.3 | 0.0 (-3.8) | 13.3 |

关键发现：

1. **w/o Memory**：所有 task 退化到接近 chance level，证实 benchmark 是 non-Markovian at observation level。这是 paper 的 sanity check。

2. **Memory Bank（similarity-based retrieval）**：DSR 都是 NA（因为 manipulation 全部失败 MSR=0%），CSR 接近 chance level（33.3%, 31.7%, 11.7%）。这说明 similarity-based retrieval 在 aliasing 下不稳定——visually similar 但 decision-irrelevant 的 episode 被错误检索，导致 policy 混乱。这是对 RAG-style 方法的直接打击。

3. **Vanilla Mamba**：把整个 structured memory 换成 monolithic Mamba backbone。结果 sequential task 完全失败（0% DSR），episodic/spatial 也接近 chance。这说明**单纯的 recurrence 不够**——必须要有 anchor-structured, multi-timescale 的 episodic state 来 reduce interference。Ref: [Mamba](https://arxiv.org/abs/2312.00752)

4. **w/o Dorsal Stream**：episodic 几乎没变（96.8 vs 100.0），但 spatial task 大幅下降（51.5 vs 73.5 DSR）。这证实了 dorsal stream 主要在 spatial reasoning 任务上发挥作用。Figure 5 的可视化显示，去掉 dorsal stream 后 cross-view attention 在 distractors 上 diffuse，无法 focus 到正确 target。这个 ablation 非常 clean，对应了生物学上 dorsal "where" stream 的功能。

5. **w/o HoloHead**：所有 task 都下降，sequential task 直接归零。这说明 latent imagination objective 对 $h_t$ 的稳定性至关重要——没有它 $h_t$ 会 collapse 到 instantaneous appearance，丢失 predictive content。

### 7.4 Mechanistic Validation

**Pattern separation**（Figure 4）：UMAP projection 显示 $h_t$ 在 episodic/spatial task 上从重叠 cluster 分离到 distinct manifolds，对应不同 latent states。Sequential task 则在 shared manifold 上沿着不同 stage 演化。这说明 $h_t$ 确实实现了 event-biased separation。

**Pattern completion**（Figure 6）：从随机 timesteps rollout HoloHead predictions，trajectory 保持 goal-consistent。这证实 $h_t$ 能从 partial cues 恢复 task-relevant latent state。

---

## 8. 与生物学的对应（Appendix C, Table 7）

| Biological Component | Biological Role | Chameleon Module | Functional Correspondence |
|---------------------|-----------------|------------------|---------------------------|
| Ventral stream | Object identity (what) | Frozen DINO patch tokens | Local visual evidence |
| Dorsal stream | Spatial relations (where) | EE-anchored geometry + cross-view attention | Spatial structure, cross-view correspondence |
| EC | Multimodal integration | Multimodal fusion | Unified evidence set |
| DG | Pattern separation | Spatial anchors + spatiotemporal slots | Decorrelate overlapping perceptual evidence |
| HC circuit | Episodic encoding/retrieval | Hierarchical memory stack | Structured episodic memory across slots and time |
| CA3 | Pattern completion | Slot-wise episodic SSM | Distributed episodic states supporting context recovery |
| CA1 | Integrates recall into output | Aggregated recall vector + working readout | Compact representation for downstream use |
| HC-PFC interaction | Prospective cognition | Working state $h_t$ + hierarchical fusion | Aggregates recalled content for decision-making |
| PFC imagination | Simulate future states | HoloHead latent imagination | Predicts multi-horizon future EE trajectories |

这个对应很有趣，但作者也诚实地承认：模型**没有**显式实现生物学上的 pattern separation mapping，slot decomposition 只是在功能上类似。Ref: [Bakker et al. 2008 pattern separation](https://www.science.org/doi/10.1126/science.1152882), [Allen & Fortin 2013 evolution of episodic memory](https://www.pnas.org/doi/10.1073/pnas.1301199110)

---

## 9. Implementation Details 关键数字

- **Trainable parameters**: Chameleon ≈ 49.89M（vs Diffusion Policy 328.83M, Flow Matching 79.90M, ACT 78.13M）
- **Training**: 20K steps, single 4090 GPU, AdamW lr=1e-4, batch size 4, bfloat16, cosine annealing, EMA decay 0.999
- **Inference**: 82 ms/control step, 50 rectified-flow ODE steps
- **Sequence training**: chunk length 512, temporal stride 4, policy loss on last 16 frames
- **Memory slots**: $A=8$ spatial × $B=4$ temporal = 32 slots/layer, $L=2$ layers
- **Episodic SSM**: $(d_{\text{state}}, d_{\text{conv}}, \text{expand}) = (128, 4, 1)$
- **Working SSM**: $(32, 4, 2)$
- **Base temporal priors** $\Delta t_b^{(0)}$: $\{0.001, 0.005, 0.02, \text{flexible}\}$

参数量比 baseline 都小很多——这说明 structured inductive bias 比单纯堆参数更有效。

---

## 10. 我的批判性思考

### 10.1 优势

1. **Single-state conditioning 是强约束**：把 memory 的作用 explicit 化，policy 只看 $h_t$，逼迫 memory 模块承担 disambiguation 的责任。这种 architectural constraint 比 soft auxiliary loss 更有 teeth。

2. **Multi-timescale 设计有 physics grounding**：$\Delta t_b^{(0)}$ 对应 explicit half-life，这比单纯 learn 一个 RNN 更 interpretable。这也呼应了 [Hierarchical RNN](https://arxiv.org/abs/1609.01704) 和 [Dilated RNN](https://arxiv.org/abs/1710.02244) 的思路，但更 explicit。

3. **HoloHead 是 self-supervised predictive objective**：类似于 [Dreamer](https://arxiv.org/abs/1912.01603) 的 latent imagination，但用在 supervised imitation learning 里。这个 regularizer 让 $h_t$ 不只是 reconstructive 而是 predictive，对 long-horizon 很重要。

4. **Cross-view geometry integration 优雅**：EE-anchored 的设计把 action relevance 注入 perception，让 cross-view attention 不会在 distractors 上浪费。这比单纯的 epipolar constraint 多了 action-aware 的 prior。

### 10.2 局限和疑问

1. **Biological analogy 的深度有限**：作者承认 DG-style pattern separation 在模型里只是 functional analogy。真正的 DG 用极其 sparse 的 granule cells 实现 decorrelation，Chameleon 的 spatial-temporal slots 更像是一种 structured memory addressing，没有显式的 sparse coding。但作为 engineering 这种 analogy 已经足够 motivate 架构设计。

2. **Generalization 范围**：实验只在 single UR5e platform 上做，三个 task 都是 controlled lab setup。作者在 limitations 里诚实地提到 cross-embodiment transfer 和 event segmentation 是 future work。但 real-world 的 perceptual aliasing 可能比这些 task 复杂得多——比如光照变化、视角微小扰动等。

3. **Task phase signal $\psi_t$ 的依赖**：虽然 $\psi_t$ 不揭示 hidden state，但仍然是一个 phase indicator。在更 open-ended 的 setting 里 phase boundary 怎么定义？这是一个 unaddressed 的 assumption。

4. **Episodic vs Parametric memory 的结合**：作者说 Chameleon **没有** incorporate VLA 的 semantic abstraction，所以 zero-shot generalization 弱。未来的方向应该是把 episodic memory stack 和 foundation model 结合——结构化 encoding + 高层 priors。这让我想到 [HippoRAG](https://arxiv.org/abs/2405.14831) 在 LLM 上的工作，可能是 robot 版本的延伸方向。

5. **Memory Bank ablation 的极端结果**：Memory Bank variant MSR=0% 完全失败，这个结果有点 suspicious。可能 similarity-based retrieval 的 implementation 太 naive，更 sophisticated 的 retrieval（比如 [RAPTOR](https://arxiv.org/abs/2310.06825) 的 hierarchical retrieval 或 [graph-based retrieval](https://arxiv.org/abs/2404.17724)）可能有不同表现。不过作者的核心论点——similarity ≠ utility——是合理的。

6. **HoloHead 的 Compass waypoint 依赖 phase endpoint**：$R_t = t_{\text{end}} - t$ 需要知道 phase 什么时候结束。在 training 时可以 ground truth 知道，但 deployment 时如何知道？这暗示 inference 时 phase segmentation 仍然需要外部提供。

### 10.3 与近期工作的联系

- 这篇文章和 [V-JEPA](https://arxiv.org/abs/2301.08243), [DreamerV3](https://arxiv.org/abs/2301.04104) 这一类 predictive world model 有思想上的联系——都是用 forward prediction 来 shape latent state。
- 和 [Mamba](https://arxiv.org/abs/2312.00752), [Vision Mamba](https://arxiv.org/abs/2401.16645) 的关系：作者用 selective SSM 作为 episodic state 的 backbone，但加了 multi-timescale prior + spatial-temporal slot structure。这是 SSM 在 robotics memory 上的 application。
- 和 [π0](https://arxiv.org/abs/2410.24164), [OpenVLA](https://arxiv.org/abs/2406.09246) 这类 VLA 的关系：作者明确说 Chameleon **没有** VLA 的 zero-shot 能力，但 episodic memory stack 可能是 VLA 的一个补充——VLA 提供 parametric prior，Chameleon-style memory 提供 non-parametric episode-level disambiguation。

### 10.4 对你（Karpathy）的可能兴趣点

你之前在 [build-nanogpt](https://github.com/karpathy/build-nanogpt) 和 [nanoGPT](https://github.com/karpathy/nanogpt) 里强调 architecture simplicity。Chameleon 的设计其实**不算** simple——spatial-temporal slots + episodic/working separation + HoloHead regularization 有不少 moving parts。但每一个 component 都有清晰的 functional motivation。

如果你想 build intuition，我建议关注：
1. **Single-state sufficiency constraint**：这是一种 strong inductive bias，强迫 memory 模块承担全部 disambiguation 责任，是否可以推广到其他 modular architectures？
2. **Multi-timescale prior $\Delta t_b^{(0)}$**：这对应 signal processing 里的 multi-resolution analysis（类似 wavelet），在 SSM 里有明确物理意义（half-life）。是否可以替代更复杂的 attention-based memory？
3. **Predictive regularizer 作为 memory shaper**：HoloHead 不是 policy 的一部分，但 shape 了 memory。这种 auxiliary task shaping latent state 的思路可以推广到其他长 horizon RL 问题。

---

## 11. 总结

Chameleon 的核心贡献是把 perceptual aliasing 问题 formalize 成 memory-intensive manipulation 的 central bottleneck，并从 EC-HC-PFC episodic memory 电路里提取三个 functional principle：
1. **Disambiguating encoding at write time**（geometry-grounded perception）
2. **Pattern separation via structured slot decomposition**（spatial-temporal anchors）
3. **Goal-directed recall via predictive imagination**（HoloHead）

实验结果在三个 controlled aliasing task 上 convincingly 超过 baseline，ablation 干净地证实每个 component 的必要性。Limitation 主要在 generalization scope 和对 VLA foundation model 的整合。

Code: https://github.com/gxyes/MARS_Chameleon

相关参考链接：
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [ACT: Action Chunking Transformer](https://tonyzhaozh.github.io/aloha/)
- [HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs](https://arxiv.org/abs/2405.14831)
- [Embodied-RAG](https://arxiv.org/abs/2409.18313)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [V-JEPA](https://arxiv.org/abs/2301.08243)
- [Bakker et al. 2008 Pattern Separation in CA3 and DG](https://www.science.org/doi/10.1126/science.1152882)
- [Allen & Fortin 2013 Evolution of Episodic Memory](https://www.pnas.org/doi/10.1073/pnas.1301199110)
- [Goodale & Milner Ventral/Dorsal Streams](https://www.sciencedirect.com/science/article/pii/0028393292900279)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
