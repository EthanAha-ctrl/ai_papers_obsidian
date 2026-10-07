---
source_pdf: GENERATIVE VIEW STITCHING.pdf
paper_sha256: 2eade7f7bae6a7d0a6a5d3c3730fd23ac319b4f0f74e2c1ccb2d0feed0c41bf2
processed_at: '2026-08-04T14:19:31-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GVS 用人话讲

---

## 一句话版本

你有一个只能看 8 帧的 video model，你想生成 120 帧的 video，而且 camera 路线是提前定好的。老办法（autoregressive）是一步步往后推，但它走到第 50 帧可能撞墙了，因为生成时不知道后面要往哪走。GVS 的办法是：**把所有 chunk 同时 denoise，让每一 chunk 都能看到前面和后面的邻居，大家一起商量着来，谁也不撞墙**。

---

## 背景设定

现在的 video diffusion model（Runway Gen-4, Veo-3, HunyuanVideo, Wan, Luma Dream Machine 这些）context window 通常只有 5-10 秒。训练更长 context 的 model 又贵又难。所以大家都在想办法：怎么用 short-context model 生成长视频。

主流思路是 **autoregressive rollout**——生成 8 帧，把这 8 帧当 history，再生成下 8 帧，循环往复。配合 history guidance 或 RAG retrieval，能跑几百帧还算稳定。MIT 这个组之前就做过这个方向（[DFoT / History-Guided Video Diffusion](https://arxiv.org/abs/2502.07272)）。

---

## AR 的致命问题

AR 的 logic 是 **causal**：当前帧只能看 past，看不到 future。

这在"用户实时操控 camera"的场景下还好——用户会本能地避免撞墙，所以 camera trajectory 是动态生成的，不会和场景冲突。

但如果 camera trajectory 是 **预先定义好的**（比如电影一镜到底、自动驾驶 simulation），AR 就完蛋了。举个例子：

- 你定义了一条 camera trajectory：往前走 50 米然后左转
- AR 生成第 1-40 帧时，不知道后面要左转，可能就生成了一堵墙挡在左转方向
- 到第 41 帧要左转了，model 被迫"穿墙"，生成 OOD frame
- 之后 distribution shift 累积，generation 直接 collapse

这就是 Figure 1 展示的 failure mode。AR 不是技术不行，是 **architecture 层面看不到 future**，无解。

---

## 那 stitching 是什么思路

换个思路：**不要一步步推，一次性把所有 chunk 都 denoise**。

具体怎么做：
1. 把 120 帧的 target video 切成 30 个 chunk，每个 chunk 4 帧
2. 每个 chunk 在 denoise 时，把它的前后邻居也拉进来，组成一个 8-frame context window：`[2 帧 past][4 帧 target][2 帧 future]`
3. 这个 context window 送进 model，model 输出 denoised target chunk
4. 用 denoised target chunk 更新全局的 video estimate，邻居的 denoised 结果**丢掉**（只是临时借来当 conditioning 的）
5. 所有 chunk **同时** denoise，step by step 从纯 noise 走向 clean

这样每个 chunk 都能"看到"前后邻居，信息是双向流动的，不会出现"走到死胡同"的问题。

Figure 2 的图很直观。

---

## 为什么不用专门训练一个 stitching model

之前有个工作 [CompDiffuser (Luo et al., 2025)](https://arxiv.org/abs/2503.07559) 做过 trajectory stitching for robot planning，思路类似。但它需要训练一个 special model：用 special encoder + AdaLN 把 neighbor chunks 作为独立 conditioning input 注入。

对 video 来说，训练一个 custom model 成本不可接受（video model 训一次几百万美金起步）。

GVS 的关键发现：**Diffusion Forcing (DF) 训练出来的 model 天然就支持 stitching**，不需要任何额外训练。

为什么？因为 DF 训练时就是每个 token 独立加 noise，model 学习的是 $\epsilon_\theta(\mathbf{x}_0^{k_0}, \mathbf{x}_1^{k_1}, \ldots, \mathbf{x}_n^{k_n})$ 这种 mixed noise level 的输入。所以你给它一个 `[$\mathbf{x}_{t-1}^k$, $\mathbf{x}_t^k$, $\mathbf{x}_{t+1}^k$]` 的 sequence，它完全知道怎么 joint denoise。

这是论文最漂亮的 insight 之一：**好的 training framework 应该提供超出其原始目的的 affordances**。DF 当初设计是为了 flexible conditioning 和 history guidance，结果发现 stitching 也免费送了。

参考 [Diffusion Forcing (Chen et al., 2024)](https://boyuan.space/diffusion-forcing/)。

---

## 但 vanilla stitching 有个坑

你按上面说的做了，发现 temporal consistency 很差（Figure 3 第一行）。

原因：你想要的是 conditional distribution $p(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_{t+1})$ 的 score，但你实际算的是 joint distribution $p(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1})$ 的 score。

什么意思呢？在 AR sampling 里，past frames 的 noise level 很低（接近 clean），target frames 的 noise level 很高。model 看到清晰的 past + 模糊的 target，就知道"target 应该和 past 一致"。conditioning signal 很强。

但在 stitching 里，target 和 neighbors **一样 noisy**（同步 denoising）。model 看到三个同样模糊的 chunks，搞不清楚谁该跟谁走。conditioning signal 太弱了。

---

## StochSync 的解法和它的局限

之前有个工作 [StochSync (Yeo et al., 2025)](https://arxiv.org/abs/2410.18795) 提出一个 trick：用 **maximum stochasticity**。

DDIM 的 denoising step 里有三项：predicted clean signal、directional noise、fresh random noise。如果把 stochasticity 拉满 $\sigma^k = \sqrt{1-\alpha^{k-1}}$，directional noise 项就消失了，每步变成：

$$\mathbf{x}^{k-1} = \sqrt{\alpha^{k-1}}\hat{\mathbf{x}}^0 + \sqrt{1-\alpha^{k-1}}\,\epsilon$$

**直觉**：每步只保留 model 预测的 clean sample $\hat{\mathbf{x}}^0$，然后重新加 fresh noise。这样 error 不会累积——每步都"重新洗牌"。consistency 确实变好了。

但代价是 **oversmoothing**：model 的精细预测每步都被 reset 掉，high-frequency detail 慢慢磨没了。生成出来的 video 像涂了一层凡士林，模糊糊的。

[Karras et al., 2022](https://arxiv.org/abs/2206.00364) 和 StochSync 自己都观察到了这个现象。

---

## Omni Guidance：GVS 的核心创新

StochSync 是用"磨平细节"来换取 consistency，治标不治本。GVS 的思路是：**直接强化 conditioning signal，让 model 真正知道"target 要跟邻居一致"**。

但这里有个技术难点：标准 [Classifier-Free Guidance (CFG)](https://arxiv.org/abs/2207.12598) 假设 conditioning signal 是固定的（比如 text embedding），独立于 model weights。但在 GVS 里，conditioning signal 是 model 自己 co-evolving 的 noisy neighbor estimate，**它本身依赖 model weights**。标准 CFG 的理论假设不成立。

GVS 借鉴了 [Inner Guidance / VideoJAM (Chefer et al., 2025)](https://arxiv.org/abs/2410.11758) 的思路：不按标准 CFG 那样在 score 层面做线性组合，而是**直接修改 sampling distribution**。

最终推导出的 guided score function（Eq.8）：

$$\tilde{\epsilon}_\theta = (1+\gamma)\,\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1}) - \gamma\,\epsilon_\theta(\emptyset, \mathbf{x}_t^k, \emptyset | \emptyset, \emptyset, \emptyset)$$

**人话翻译**：
- 第一项 $\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1})$：正常 forward pass，neighbors 和 camera 都给 model 看
- 第二项 $\epsilon_\theta(\emptyset, \mathbf{x}_t^k, \emptyset | \emptyset, \emptyset, \emptyset)$：把 neighbors 替换成 pure noise、noise level 拉到最大，camera 也 drop 掉。相当于"没有邻居、没有 camera"的 baseline forward pass
- 两者做差再加权，就是在强化"neighbors 和 camera 对 target 的影响"

第二项的实现依赖 DF backbone 的能力：可以把 neighbor tokens 替换成 pure Gaussian noise，并设置它们的 noise level 为 $K-1$（最大）。model 看到 `[pure noise @ max level][target @ current level][pure noise @ max level]`，就相当于"没有 neighbor conditioning"。

$\gamma$ 实践中设为 1。

**效果**：Table 2 的 ablation 非常清楚。没有 Omni Guidance 时，加大 stochasticity 能改善 consistency 但 quality 掉得厉害。有 Omni Guidance 时，consistency 在各个 stochasticity level 都更好，而且可以用 partial stochasticity（$\eta = 0.9$ 而非 1.0）保留 detail。两者配合是最好的。

**intuition**：Omni Guidance 是 **explicit** 的 consistency 机制（直接强化 conditioning），stochasticity 是 **implicit** 的 consistency 机制（通过 reset 消除 error 累积）。两者正交，可以组合使用，fine-tune consistency vs. quality 的 tradeoff。

---

## Loop Closing：另一个坑

理论上，stitching 的 receptive field 应该随 denoising step 增长——chunk 0 通过 chunk 1 间接影响 chunk 2，多步累积后应该能覆盖全局。类似 [CNN 的 effective receptive field](https://arxiv.org/abs/1701.04128) 随 depth 增长。

**但实际上不是**。Figure 4 做了个实验：120-frame panorama，不做 explicit loop closing，结果 video 末尾根本无法 visually 回到起点。信息传播衰减太快。

Table 3 的数据更直接：

| Loop Closing | Omni Guidance | LRC (Panorama 1-loop) |
|---|---|---|
| ✗ | ✗ | 0.950（很差） |
| ✗ | ✓ | 0.962（Omni Guidance 也救不了） |
| ✓ | ✗ | 0.201 |
| ✓ | ✓ | 0.141（最好） |

**结论**：effective receptive field 不是 global，必须 explicit 注入 long-range constraints。

---

## Cyclic Conditioning 怎么做

思路：每个 target chunk 在 denoising 过程中，**交替使用两套 context windows**：

1. **Temporal windows**：时间上相邻的 chunks（standard stitching windows）
2. **Spatial windows**：时间上 distant 但空间上 close 的 chunks

比如 Panorama 1-loop：chunk 0（loop 起点）和 chunk T-1（loop 终点）在 3D 空间中是同一个位置。spatial window 会同时包含这两端，强制它们 visually 一致。

每个 denoising step $k$，target chunk $t$ 轮流使用不同的 windows。Algorithm 1 里这行就是 cycling：

```
w_t^k ← W[t][k mod |W[t]|]
```

Figure 8 展示了所有 trajectory 的 spatial windows 设计。Panorama 2-loop 还有额外的 spatial windows 连接两个 loop 的对应帧。

**Impossible Staircase 应用**（Section 4.3）特别酷：[Oscar Reutersvärd's Impossible Staircase](https://en.wikipedia.org/wiki/Penrose_stairs) 是那种"一直往上走却回到原点"的视觉错觉。GVS 能生成一个 120-frame 的 video，camera 一直在往上爬楼梯，但最后视觉上回到了起点。靠的就是 cyclic conditioning 把两端缝起来，而且 Appendix A.2 还把两端 disconnected 的 camera segments 替换成了 continuous straight line 来鼓励视觉连续性。

---

## 实验结果一眼看

Table 1 主结果，挑几个关键数字：

**Stairs benchmark**（最能体现 collision avoidance）：
- AR：CA = 0.075（7.5% frames 撞墙），F2FC = 0.166
- StochSync：CA = 0（但靠 shape-shifting scene 作弊），F2FC = 0.204
- GVS：CA = 0，F2FC = 0.139，IQ = 0.635（quality 最高）

**Panorama 2-loop**（最能体现 loop closing）：
- AR：LRC = 0.171，但 Figure 6 显示是"last-minute loop closure"，视觉上不连续
- StochSync：LRC = 0.279，loop closing 效果一般
- GVS：LRC = 0.116，真正的 visual loop closure

**Scaling**：Figure 9 展示 GVS 生成 1080-frame（18 层楼梯）video，全程 collision-free 且 stable。AR 在这个长度几乎必然 collapse。

---

## Backbone 和实现细节

Backbone 是 [DFoT from Song et al. (2025)](https://arxiv.org/abs/2502.07272)：
- U-ViT architecture（[Hoogeboom et al., 2023](https://arxiv.org/abs/2301.11093)）
- Context window = 8 frames，input size $8 \times 256 \times 256$
- 训练在 [RealEstate10K](https://arxiv.org/abs/1805.09517) 上
- Camera conditioning：relative poses → ray encodings，通过 AdaLN 注入

GVS hyperparameters：
- Target chunk size = 4 帧
- Overlap = 2 帧（左右各 2）
- Stochasticity $\eta = 0.9$
- Omni Guidance $\gamma = 1$
- Denoising steps：50 steps linear schedule

---

## Limitations

### External image conditioning 失败

给 GVS 一张 context frame + camera trajectory，GVS 无法把 context frame 的 scene 信息传播到整个 video（Figure 11）。

**Root cause**：stitching 的 symmetry（所有 chunks 平等 co-evolve）和 external conditioning 的 asymmetry（context frame 固定）冲突。neighboring chunks 互相影响，但 external context frame 不受 target video 影响，信息流不对称。

论文留给 future work：通过 modulating per-frame noise levels 来控制信息传播方向。

### Wide-baseline loop closing 失败

180° orbit 连接 forward 和 backward segments（Figure 12-13），GVS 无法 loop close。

**Root cause**：backbone 在 RealEstate10K 上训练，该数据集只有 small viewpoint shifts。wide-baseline cameras 是 OOD，backbone 直接就 track 不了 camera motion。这不是 GVS 方法的问题，是 backbone 的问题。解决方向：在 [DL3DV](https://arxiv.org/abs/2312.16256) 或 [ScanNet++](https://arxiv.org/abs/2308.11417) 等 wide-baseline multi-view dataset 上训练 backbone。

### Structurally similar segments

Stairs trajectory 里，upward staircase 起点和 downward staircase 终点在结构上 identical（只差 rigid transformation）。backbone context window 短 + 用 relative poses（丧失全局位置信息），GVS 分不清这两个 segment，有时会在 downward staircase 底部错误生成 ascending steps。

---

## 几个深层 intuition

### Stitching vs. AR 是 inference-time 的 causal structure 选择

AR 是 causal/forward：past → present。Stitching 是 bidirectional/simultaneous：past + future → present。有 future constraints 就选 stitching，没有就选 AR（AR 的 strong conditioning 更好）。

### DF 的独立 noise level 是隐藏的 affordance

DF 训练时每个 token 独立 noising，本意是为了 flexible conditioning。结果发现 stitching 也免费送了。这说明好的 training framework 应该设计得"over-complete"一些，留出 unintended affordances。

### Theoretical vs. Effective Receptive Field

理论上 stitching 的 receptive field 应该 global，但实际信息衰减严重。diffusion 的信息传播不是无损的——每步有 noise injection 和 model capacity 限制，信号会衰减。所以 explicit long-range constraints 是必要的工程手段，类似 transformer global attention 相对于 CNN local receptive field 的角色。

### Loop closing 是 3D consistency 的 proxy

GVS 的 loop closing 本质上在 enforcing **spatial consistency**——不同时间但相同空间位置的 frames 应该 depict 相同 scene。类似 NeRF 的 multi-view consistency，但在生成框架里实现，不需要 explicit 3D representation。

---

## 一句话总结

GVS = DF backbone（免费提供 stitching 能力）+ Omni Guidance（强化 bidirectional conditioning）+ Cyclic Conditioning（explicit long-range constraints）。三个 component 组合，实现了 training-free 的 camera-guided long video generation，避免 collision，enable loop closure，还能生成 Impossible Staircase。

---

## 参考链接汇总

- [GVS 项目主页](https://andrewsonga.github.io/gvs)
- [Diffusion Forcing](https://boyuan.space/diffusion-forcing/)
- [History-Guided Video Diffusion (DFoT)](https://arxiv.org/abs/2502.07272)
- [StochSync](https://arxiv.org/abs/2410.18795)
- [CompDiffuser](https://arxiv.org/abs/2503.07559)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598)
- [VideoJAM (Inner Guidance)](https://arxiv.org/abs/2410.11758)
- [RealEstate10K](https://arxiv.org/abs/1805.09517)
- [MEt3R](https://arxiv.org/abs/2504.20774)
- [VBench](https://arxiv.org/abs/2311.13582)
- [Video Depth Anything](https://arxiv.org/abs/2501.12275)
- [WorldExplorer](https://arxiv.org/abs/2409.12169)
- [DL3DV](https://arxiv.org/abs/2312.16256)
- [ScanNet++](https://arxiv.org/abs/2308.11417)
- [Penrose Stairs](https://en.wikipedia.org/wiki/Penrose_stairs)
- [MultiDiffusion](https://arxiv.org/abs/2302.08113)
- [Effective Receptive Field](https://arxiv.org/abs/1701.04128)
- [Simple Diffusion (U-ViT)](https://arxiv.org/abs/2301.11093)
- [Karras et al. Design Space](https://arxiv.org/abs/2206.00364)

---

# Generative View Stitching (GVS) 深度解析

这篇 paper 来自 MIT CSAIL 的 Vincent Sitzmann 组，一作 Chonghyuk Song。核心问题是：如何用 short-context video diffusion model 生成 faithful to predefined camera trajectory 的 long video，同时避免 collision 和 enable loop closure。项目主页：https://andrewsonga.github.io/gvs

---

## 1. 问题动机：为什么 AR sampling 不够用

现有的 video diffusion models（Runway Gen-4, Veo-3, HunyuanVideo, Wan, Luma Dream Machine 等）context window 通常只有 5-10 秒。要生成更长的视频，主流方法是 **autoregressive (AR) rollout**：用 short-horizon model 反复推演，配合 history guidance 或 RAG-style retrieval 来维持 long-range consistency。

AR 的根本缺陷在于 **causal direction**：当前生成只能 condition on past，无法看到 future。这在 camera-guided generation 中是致命的。设想你有一条预定义的 camera trajectory，AR 在第 $t$ 帧可能生成一堵墙，但第 $t+10$ 帧的 camera pose 要求"穿过"这堵墙。由于 AR 无法 plan ahead，它会被迫生成 OOD frames，之后 distribution shift 累积，generation 迅速 collapse（Figure 1 直观展示了这个 failure mode）。

这个问题本质上是一个 **planning under future constraints** 问题，与 robot planning 中的 goal-conditioned generation 非常类似。

---

## 2. Diffusion Background：关键公式拆解

### 2.1 Forward Process

$$\mathbf{x}^k = \sqrt{\alpha^k}\,\mathbf{x}^0 + \sqrt{1-\alpha^k}\,\epsilon$$

变量含义：
- $\mathbf{x}^0 \in \mathbb{R}^{H \times W \times C}$：clean data sample（比如一帧或一组 video frames）
- $\mathbf{x}^k$：noise level $k$ 下的 noised version
- $k \in \{0, 1, \ldots, K-1\}$：discrete noise levels，$k$ 越大 noise 越多
- $\alpha^k \in [0,1]$：noise schedule 的累积参数，$\alpha^0 \approx 1$（几乎无 noise），$\alpha^{K-1} \approx 0$（纯 noise）
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：标准 Gaussian noise

### 2.2 Reverse Process (DDIM-style)

论文的 Eq.(1)：

$$\mathbf{x}^{k-1} = \sqrt{\alpha^{k-1}}\left(\frac{\mathbf{x}^k - \sqrt{1-\alpha^k}\,\epsilon_\theta(\mathbf{x}^k, k)}{\sqrt{\alpha^k}}\right) + \sqrt{1-\alpha^{k-1}-(\sigma^k)^2}\cdot\epsilon_\theta(\mathbf{x}^k, k) + \sigma^k\,\epsilon$$

注意：论文 Eq.(1) 第三项写的是 $\sigma^k \epsilon_\theta(\mathbf{x}^k, k)$，但根据标准 DDIM（[Song et al., 2021a, DDIM](https://arxiv.org/abs/2010.02502)）和论文后面的描述（"$\sigma^k$ controls the level of stochasticity i.e., the amount of random noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ injected"），第三项应该是 $\sigma^k \epsilon$ 即 fresh random noise，这是论文的笔误。

拆解三项：
1. **Predicted clean signal** $\sqrt{\alpha^{k-1}}\hat{\mathbf{x}}^0$，其中 $\hat{\mathbf{x}}^0 = \frac{\mathbf{x}^k - \sqrt{1-\alpha^k}\epsilon_\theta}{\sqrt{\alpha^k}}$
2. **Directional component** $\sqrt{1-\alpha^{k-1}-(\sigma^k)^2}\cdot\epsilon_\theta$：deterministic 的 noise 方向
3. **Stochastic component** $\sigma^k \epsilon$：fresh randomness

关键参数 $\sigma^k$ 的两个极端：
- $\sigma^k = 0$：deterministic DDIM，每步完全由 model 预测决定
- $\sigma^k = \sqrt{1-\alpha^{k-1}}$：**maximum stochasticity**，此时第二项系数 $\sqrt{1-\alpha^{k-1}-(\sigma^k)^2} = 0$，即完全丢弃 directional component，只保留 predicted $\hat{\mathbf{x}}^0$ + fresh noise。这是 [StochSync (Yeo et al., 2025)](https://arxiv.org/abs/2410.18795) 的核心 trick，起 error correction 作用但会导致 oversmoothing。

---

## 3. Diffusion Forcing (DF)：为什么它是 stitching 的天然基础

[Diffusion Forcing (Chen et al., 2024)](https://boyuan.space/diffusion-forcing/) 是一个 sequence diffusion training framework，核心特点：**每个 token 有独立的 noise level**。

训练时，sequence $[\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_n]$ 中每个 $\mathbf{x}_i$ 被独立地 noised 到 random level $k_i$，model 学习 $\epsilon_\theta(\mathbf{x}_0^{k_0}, \mathbf{x}_1^{k_1}, \ldots, \mathbf{x}_n^{k_n}, k_0, \ldots, k_n)$。

这个设计带来的 sampling affordance 极其重要：你可以**选择性地 mask context 的某些 token 为高 noise**，而 target token 为低 noise，从而实现 variable-length conditioning。这正是 history guidance（[Song et al., 2025, History-Guided Video Diffusion](https://arxiv.org/abs/2502.07272)）的基础——把 past frames 放在 context 里，noise level 设为低（接近 clean），target frames noise level 设为高，model 就会 condition on past 来生成 present。

DF 已经被广泛采用：[Decart Oasis](https://oasis-model.github.io/), [Sand.ai Magi-1](https://arxiv.org/abs/2505.13211), [SkyReels-V2](https://arxiv.org/abs/2504.13074), [Yin et al. BAR](https://arxiv.org/abs/2504.01197) 等。

---

## 4. GVS 的核心设计

### 4.1 Compositional Distribution

GVS 将 camera-guided video 的 distribution 写成 compositional 形式（Eq.3）：

$$p_\theta(\mathbf{x} | \mathbf{p}) \propto \prod_{t=0}^{T-1} p_t(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_{t+1}, \mathbf{p}_{t-1}, \mathbf{p}_t, \mathbf{p}_{t+1})$$

变量含义：
- $\mathbf{x} = [\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_{T-1}]$：target video，分成 $T$ 个 non-overlapping chunks
- $\mathbf{p} = [\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_{T-1}]$：predefined camera trajectory（per-chunk camera poses）
- $\mathbf{p}_{-1} \triangleq \mathbf{p}_0, \mathbf{p}_T \triangleq \mathbf{p}_{T-1}$：boundary padding
- $\mathbf{x}_{-1}, \mathbf{x}_T$：pure-noise padding frames

每个 chunk $\mathbf{x}_t$ 的 conditional distribution 依赖于**双向** neighbors $\mathbf{x}_{t-1}, \mathbf{x}_{t+1}$ 和对应的 cameras。这与 CompDiffuser ([Luo et al., 2025](https://arxiv.org/abs/2503.07559)) 的形式相同，但实现方式截然不同。

### 4.2 为什么不需要 custom model

CompDiffuser 要训练一个 special encoder + AdaLN 来注入 neighboring chunks 作为独立 conditioning input。这对 video 来说成本不可接受。

GVS 的关键 insight：**DF model 本来就是设计来 joint denoise 一个 sequence 的**。所以直接把 target chunk 和它的 neighbors 拼成一个 input sequence：

$$[\mathbf{x}_{t-1}^k, \mathbf{x}_t^k, \mathbf{x}_{t+1}^k]$$

一起送进 model 做 joint denoising。然后：
- **保留** target chunk $\mathbf{x}_t^{k-1}$ 的 denoised 结果，用来 update stitched sequence $\mathbf{z}$
- **丢弃** denoised conditioning chunks $\mathbf{x}_{t-1}^{k-1}, \mathbf{x}_{t+1}^{k-1}$（它们只是临时用作 conditioning，不写入最终结果）

实践细节（Appendix A.1）：
- Backbone context window = 8 frames
- $\mathcal{T}^{\text{target chunk}} = 4$（每个 target chunk 4 帧）
- $\mathcal{T}^{\text{overlap}} = 2$（左右各 2 帧来自 neighboring chunks）
- 所以一个 8-frame context window = [2 past overlap][4 target][2 future overlap]

Figure 2 清晰展示了这个结构。这个设计完全 training-free，只要 backbone 是 DF-trained 就行。

### 4.3 Vanilla GVS 的问题：weak conditioning

Vanilla GVS 虽然 simple，但 temporal consistency 很差（Figure 3 top row）。原因是一个 distribution mismatch：

我们**想要**的是 conditional distribution $p(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_{t+1})$ 的 score，但 vanilla GVS 实际计算的是 **joint distribution** $p(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1})$ 的 score。

在 AR sampling 中，past context 通常 noise level 远低于 target（past 接近 clean，target 接近 noise），所以 conditioning signal 很强。但 vanilla GVS 中，target chunk $\mathbf{x}_t^k$ 和 neighbors $\mathbf{x}_{t-1}^k, \mathbf{x}_{t+1}^k$ **一样 noisy**（同步 denoising），conditioning signal 很弱——model 看到的是三个同样模糊的 chunks，很难从中提取出"target 应该与 neighbors 一致"的信号。

---

## 5. Omni Guidance：核心技术创新

### 5.1 为什么标准 CFG 不适用

标准 [Classifier-Free Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598) 假设 conditioning signal $\mathbf{c}$ 独立于 model weights $\theta$：

$$\tilde{\epsilon}_\theta = (1+w)\epsilon_\theta(\mathbf{x}^k | \mathbf{c}) - w\,\epsilon_\theta(\mathbf{x}^k | \emptyset)$$

这里 $\mathbf{c}$ 是固定的（比如 text embedding）。但在 GVS 中，guidance signal 是 model 自己 co-evolving 的 noisy estimate of neighbors $\mathbf{x}_{t-1}^k, \mathbf{x}_{t+1}^k$，**这些 estimate 本身依赖 $\theta$**。所以标准 CFG 的独立性假设被破坏。

### 5.2 Inner Guidance 思路

论文借鉴 [Inner Guidance / VideoJAM (Chefer et al., 2025)](https://arxiv.org/abs/2410.11758) 的思路：直接修改 sampling distribution，而不是在 score function 层面做标准 CFG。

目标 distribution（Eq.4-6）：

$$\tilde{p}_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1}) \propto p_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1}) \cdot \left[\frac{p_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1})}{p_\theta(\mathbf{x}_{t-1:t+1}^k)}\right]^{\gamma_1} \cdot \left[\frac{p_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1})}{p_\theta(\mathbf{x}_t^k | \mathbf{p}_{t-1:t+1})}\right]^{\gamma_2}$$

分解三个 factor 的 intuition：

1. **Base term** $p_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1})$：原始 conditional distribution，给定 camera trajectory 生成 video chunks
2. **Camera adherence factor** $\left[\frac{p(\mathbf{x}|\mathbf{p})}{p(\mathbf{x})}\right]^{\gamma_1}$：likelihood ratio，增强对 camera trajectory 的忠实度。$\gamma_1$ 大 → 更严格遵循 camera pose
3. **Neighbor consistency factor** $\left[\frac{p(\mathbf{x}_{t-1:t+1}|\mathbf{p})}{p(\mathbf{x}_t|\mathbf{p})}\right]^{\gamma_2}$：这个 ratio 的 intuition 是 $p(\mathbf{x}_{t-1:t+1}|\mathbf{p}) / p(\mathbf{x}_t|\mathbf{p}) \propto p(\mathbf{x}_{t-1}, \mathbf{x}_{t+1} | \mathbf{x}_t, \mathbf{p})$，即给定 target chunk 和 camera，neighbors 的 conditional likelihood。$\gamma_2$ 大 → target 更努力与 neighbors 一致

### 5.3 Score function 修改

对应 Eq.(7)：

$$\tilde{\epsilon}_\theta = (1+\gamma_1+\gamma_2)\,\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1}) - \gamma_1\,\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \emptyset, \emptyset, \emptyset) - \gamma_2\,\epsilon_\theta(\emptyset, \mathbf{x}_t^k, \emptyset | \mathbf{p}_{t-1:t+1})$$

三个 score terms：
- $\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1})$：**full conditioning**——所有 chunks + camera 都给定
- $\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \emptyset, \emptyset, \emptyset)$：**fully unconditional**——camera 和 neighbors 都 drop，用于 camera guidance（standard CFG 的 null condition）
- $\epsilon_\theta(\emptyset, \mathbf{x}_t^k, \emptyset | \mathbf{p}_{t-1:t+1})$：**target-only with camera**——neighbors 被替换为 pure noise 且 noise level 设为 maximum，只有 target chunk + camera。这一项是 neighbor consistency 的"negative" term

第三项的实现关键：利用 DF backbone 的能力，将 neighboring chunks 的 tokens 替换为 pure Gaussian noise $\emptyset$，并把它们的 noise level 设为 $k = K-1$（maximum）。Model 看到 [pure noise @ max noise level][target @ current noise level][pure noise @ max noise level]，就相当于"没有 neighbor conditioning"。

### 5.4 实践简化

论文在实践中 merge $\gamma_1, \gamma_2$ 为单一 $\gamma$（Eq.8）：

$$\tilde{\epsilon}_\theta = (1+\gamma)\,\epsilon_\theta(\mathbf{x}_{t-1:t+1}^k | \mathbf{p}_{t-1:t+1}) - \gamma\,\epsilon_\theta(\emptyset, \mathbf{x}_t^k, \emptyset | \emptyset, \emptyset, \emptyset)$$

这里第二项同时 drop camera 和 neighbors。实践中 $\gamma = 1$。

### 5.5 与 Fractional History Guidance 的关系

这是 [History-Guided Video Diffusion (Song et al., 2025)](https://arxiv.org/abs/2502.07272) 的 generalization。关键区别：

| | History Guidance (AR) | Omni Guidance (GVS) |
|---|---|---|
| Conditioning direction | Past → Present | Past + Future → Present |
| Neighbor noise levels | **Fixed**（past frames noise level 恒定低） | **Co-evolving**（neighbors 和 target 同步 denoising，noise level 随步变化） |
| 应用场景 | AR rollout | Parallel stitching |

这个 co-evolving 特性是 stitching 的本质——所有 chunks 同时从 noise 走向 clean，互相影响。

---

## 6. Stochasticity 与 Omni Guidance 的协同

### 6.1 Maximum stochasticity 的局限

StochSync 用 $\sigma^k = \sqrt{1-\alpha^{k-1}}$（即 $\eta = 1$），这会让 denoising step 变成：

$$\mathbf{x}^{k-1} = \sqrt{\alpha^{k-1}}\hat{\mathbf{x}}^0 + \sqrt{1-\alpha^{k-1}}\,\epsilon$$

完全丢弃 directional component，只保留 predicted clean + fresh noise。这确实增强 consistency（因为每步都"重新洗牌"，error 不会累积），但代价是 oversmoothing——model 的精细预测被反复 reset，high-frequency detail 丢失（[Karras et al., 2022](https://arxiv.org/abs/2206.00364) 有类似观察）。

### 6.2 Omni Guidance 的作用

Table 2 的 ablation 非常清晰：

**Without Omni Guidance (Table 2a)**：
- $\eta = 0 \to 1.0$：F2FC 从 0.153 → 0.061（Straight line），consistency 大幅改善
- 但 IQ 从 0.537 → 0.422，IS 从 2.17 → 1.54，quality 明显下降

**With Omni Guidance (Table 2b)**：
- 同样 $\eta$ 范围，F2FC 更低（更好），且 IQ/AQ/IS 显著更高
- 关键：$\eta = 0.9$ 时，IQ = 0.615（远高于 without OG 的任何 $\eta$），同时 F2FC = 0.080（excellent consistency）

Intuition：Omni Guidance 直接强化 conditioning signal，不需要靠 oversmoothing 来"作弊"实现 consistency。这样就解放了 stochasticity 参数，可以用 partial stochasticity $\eta \in (0, 1)$ 而非 maximum，保留 high-frequency detail。

### 6.3 一个有趣的 corner case

Table 2b 中 Straight line 在 $\eta = 1.0$ 时 F2FC = 0.071，反而比 $\eta = 0.9$ 的 0.080 更好——但论文指出这是因为 $\eta = 1.0$ 的 oversmoothing 太严重，以至于"consistency 任务变简单了"（场景都被磨平了，当然一致）。这提示我们：**metric 不能脱离 visual quality 单独看**。

---

## 7. Cyclic Conditioning：Loop Closing 机制

### 7.1 为什么需要 explicit loop closing

理论上，GVS 的 receptive field 应该随 denoising step 增长——类似 [CNN effective receptive field (Luo et al., 2017)](https://arxiv.org/abs/1701.04128) 随 depth 增长。因为每步 denoising，chunk $t$ 通过 neighbors $t-1, t+1$ 间接获得 $t-2, t+2$ 的信息，多步累积后应该能覆盖全局。

但 Figure 4 的实验表明：实际信息传播远不如理论。120-frame panorama 不做 explicit loop closing 时，video 末尾无法 visually "return to the same place"。

Table 3 的数据更直接：

| Loop Closing | Omni Guidance | LRC (Panorama 1-loop) |
|---|---|---|
| ✗ | ✗ | 0.950（极差） |
| ✗ | ✓ | 0.962（Omni Guidance 救不了） |
| ✓ | ✗ | 0.201（大幅改善） |
| ✓ | ✓ | 0.141（最佳） |

这说明 **effective receptive field 不是 global**，信息衰减很快，必须 explicit 注入 long-range constraints。

### 7.2 Cyclic Conditioning 设计

Figure 5 和 Figure 8 展示了完整设计。核心思路：每个 target chunk $t$ 在 denoising 过程中，交替使用两套 context windows：

1. **Temporal windows**：包含 timesteps $\{t\mathcal{T}^{\text{chunk}} - \mathcal{T}^{\text{overlap}}, \ldots, t(\mathcal{T}^{\text{chunk}}+1) + \mathcal{T}^{\text{overlap}} - 1\}$，即时间上相邻的 chunks。这是 standard GVS 的 windows。

2. **Spatial windows**：包含时间上 distant 但空间上 close 的 chunks。比如 Panorama 1-loop 中，chunk 0（loop 起点）和 chunk T-1（loop 终点）在 3D 空间中是同一位置，所以 spatial window 会同时包含这两端，强制它们 visually 一致。

Algorithm 1 中的关键行：
```
w_t^k ← W[t][k mod |W[t]|]  // Cycle through context windows
```
每个 denoising step $k$，target chunk $t$ 轮流使用 $\mathcal{W}[t]$ 中的不同 windows。如果只有 temporal windows，$|\mathcal{W}[t]| = 1$；如果还有 spatial windows，$|\mathcal{W}[t]| = 2$，就交替使用。

Figure 8 展示了所有 trajectory 的 spatial windows：
- Panorama 1-loop：一组 spatial windows 连接 loop 两端
- Panorama 2-loop：额外的 spatial windows 连接两个 loop 的对应帧（"Spatial windows 1-12"）
- Circle/Spiral：类似设计
- Staircase circuit：spatial windows 连接 circuit 上的 spatially co-located 点

### 7.3 Impossible Staircase 应用

Section 4.3 展示了一个酷炫应用：[Oscar Reutersvärd's Impossible Staircase (Penrose & Penrose, 1958)](https://en.wikipedia.org/wiki/Penrose_stairs)。trajectory 两端高度不同，但通过 cyclic conditioning + 特殊的 camera segment 设计（Appendix A.2：将两端 disconnected camera segments 替换为 continuous straight line），生成视觉上连续的 loop——你"一直往上走"却回到了原点。

---

## 8. 实验数据深度解读

### 8.1 Backbone

[DFoT from Song et al. (2025)](https://arxiv.org/abs/2502.07272)：
- Architecture：U-ViT（[Hoogeboom et al., 2023, Simple Diffusion](https://arxiv.org/abs/2301.11093)）
- Input：$8 \times 256 \times 256$ video
- Context window：8 frames
- Training data：[RealEstate10K (Zhou et al., 2018)](https://arxiv.org/abs/1805.09517)
- Conditioning：per-frame noise levels + camera poses via AdaLN
- Camera encoding：relative poses w.r.t. first frame → high-dimensional ray encodings

### 8.2 Baselines

1. **History-Guided AR Sampling**：deterministic DDIM ($\sigma^k = 0$)，50 steps，history guidance scale = 4，4 history frames，stabilization = 0.02。对于 loop-closing trajectories，augmented with FOV-based retrieval memory（[Zhou et al., 2025, Stable Virtual Camera](https://arxiv.org/abs/2503.18513); [Xiao et al., 2025, WorldMem](https://arxiv.org/abs/2411.09901)）。

2. **StochSync**：maximum stochasticity DDIM，25 steps from $K=900$ to $K_{\text{stop}}=270$，multi-step clean sample computation（initial 50 steps, linearly decrease），two alternating non-overlapping window sets offset by 4 frames。guidance scale 降到 4（默认 7.5 太 unrealistic）。

### 8.3 Metrics

- **F2FC (Frame-to-Frame Consistency)**：[MEt3R (Asim et al., 2025)](https://arxiv.org/abs/2504.20774) cosine similarity，averaged over consecutive frame pairs。越低越好（论文用 ↓，可能是 1-cosine 或 distance）
- **LRC (Long-Range Consistency)**：MEt3R cosine，averaged over temporally distant but spatially close pairs（基于 FOV overlap 判断 spatial closeness）
- **CA (Collision Avoidance)**：用 [Schneider et al., 2025, WorldExplorer](https://arxiv.org/abs/2409.12169) 的 collision detection，基于 [Video Depth Anything (Chen et al., 2025b)](https://arxiv.org/abs/2501.12275) 的 metric depth，depth < threshold 则 collision
- **IQ, AQ**：[VBench (Huang et al., 2024)](https://arxiv.org/abs/2311.13582) 的 imaging quality 和 aesthetic quality
- **IS**：[Inception Score (Salimans et al., 2016)](https://arxiv.org/abs/1606.03498)，仅用于 ablation

### 8.4 主结果 Table 1 解读

以 Stairs（最能体现 collision avoidance 的 benchmark）为例：

| Method | F2FC | LRC | IQ | AQ | CA |
|---|---|---|---|---|---|
| AR | 0.166 | N/A | 0.513 | 0.345 | **0.075** |
| StochSync | 0.204 | N/A | 0.571 | 0.417 | 0 |
| GVS | **0.139** | N/A | **0.635** | 0.400 | 0 |

- AR 的 CA = 0.075：7.5% 的 frames 撞墙，这是 AR 无法 plan ahead 的直接后果
- StochSync CA = 0 但 F2FC = 0.204：它通过 shape-shifting scene 来"避免"collision，本质是作弊——场景变形让 camera 不会撞到东西，但 temporal consistency 崩了
- GVS：CA = 0 且 F2FC = 0.139 且 IQ = 0.635（最高）：真正避免了 collision 同时保持 consistency 和 quality

Panorama 2-loop 更能体现 loop closing：

| Method | F2FC | LRC |
|---|---|---|
| AR | 0.169 | 0.171 |
| StochSync | 0.259 | 0.279 |
| GVS | **0.155** | **0.116** |

AR 的 LRC = 0.171 看起来还行，但 Figure 6 显示 AR 是"last-minute loop closure"——快到终点时强行 stitch visually inconsistent scenes，数字上 close 但视觉上 discontinuous。GVS 的 LRC = 0.116 是真正的 visual loop closure。

### 8.5 Scaling

Figure 9 展示 GVS 生成 1080-frame（18-story staircase）video，全程 collision-free 且 stable。这是 AR 难以实现的——AR 在如此长的 horizon 上几乎必然 encounter collision 并 collapse。

---

## 9. Limitations 深度分析

### 9.1 External Image Conditioning 失败（Appendix C.1）

Figure 11 展示了一个关键 failure mode：给 GVS 一张 context frame + camera trajectory，GVS 无法将 context frame 的 scene 信息传播到整个 video。

Root cause：在 stitching 中，neighboring chunks 互相影响（bidirectional），但 external context frame 是**固定的**——它不受 target video 影响。这打破了 stitching 的 symmetry。具体表现：
- $k=999$（几乎纯 noise）：frame $\tau=2$ 已经 commit 到 context frame 的 scene
- $k=759$：frame $\tau=12$ 及以后 commit 到**不同的 scene**（因为没有直接信息传播路径）
- 最终：context frame 只传播到 frame $\tau=4$，$\tau=6, \tau=8$ 出现 awkward transition

Potential solution（论文留给 future work）：通过 modulating per-frame noise levels 来控制信息传播。这其实回到了 DF 的核心 affordance——独立 noise levels。如果能让 context frame 保持低 noise（强 signal），target frames 保持高 noise（弱 signal），信息就会从 context 流向 target。但这与 stitching 的"同步 denoising"哲学冲突，需要更精细的设计。

### 9.2 Wide-Baseline Loop Closing 失败（Appendix C.2）

Figure 12-13：Forward-Orbit-Backward trajectory（180° orbit 连接 forward 和 backward segments）。GVS 无法 loop-close。

Root cause：backbone 在 RealEstate10K 上训练，该数据集只有 small viewpoint shifts 的 trajectory。当 spatial context window 包含 wide-baseline cameras（180° 对视），这是 OOD，backbone 无法正确 track camera motion。

Figure 13 的对照实验：直接用 full-sequence diffusion（非 stitching）也失败，证明问题在 backbone 而非 GVS 方法本身。

Potential solution：在 [DL3DV (Ling et al., 2024)](https://arxiv.org/abs/2312.16256) 或 [ScanNet++ (Yeshwanth et al., 2023)](https://arxiv.org/abs/2308.11417) 等 wide-baseline multi-view dataset 上训练 backbone。

### 9.3 Structurally Similar Segments（Appendix C.3）

Figure 1 的 Stairs trajectory：upward staircase 的起点和 downward staircase 的终点在结构上 identical（只差 rigid transformation）。由于 backbone context window 短 + 用 relative poses（丧失全局位置信息），GVS 无法区分这两个 segment，有时会在 downward staircase 底部错误生成 ascending steps。

---

## 10. 核心 intuition 总结

让我尝试提炼几个深层 insight：

**Insight 1: Stitching vs. Autoregression 是 inference-time 的 causal structure 选择**

AR 是 causal/forward：past → present，无法看到 future。Stitching 是 bidirectional/simultaneous：past + future → present。对于有 future constraints 的任务（predefined trajectory, goal-conditioned planning），stitching 是天然选择。但 stitching 代价是失去了 AR 的"strong conditioning"（past 接近 clean），需要 Omni Guidance 来弥补。

**Insight 2: DF 的独立 noise level 是 stitching 的隐藏 affordance**

DF 训练时每个 token 独立 noising，本意是为了 flexible conditioning。但这个设计恰好提供了 stitching 所需的一切：joint denoising of mixed-noise-level tokens，selective masking via noise level manipulation。这是一个很好的例子：**好的 training framework 应该提供超出其原始目的的 affordances**。

**Insight 3: Guidance scale 与 stochasticity 是正交的 consistency 控制手段**

Stochasticity 通过"重新洗牌"消除 error 累积，是 **implicit** 的 consistency 机制，代价是 oversmoothing。Omni Guidance 通过直接强化 conditioning signal，是 **explicit** 的 consistency 机制，不牺牲 detail。两者组合可以 fine-tune consistency vs. quality 的 tradeoff。

**Insight 4: Theoretical vs. Effective Receptive Field**

理论上 stitching 的 receptive field 应该 global（每步增长），但实际信息衰减严重。这提醒我们：**diffusion 的信息传播不是无损的**——每步 denoising 都有 noise injection 和 model capacity 限制，信号会衰减。Explicit long-range constraints（cyclic conditioning）是必要的工程手段，类似于 [non-local means](https://en.wikipedia.org/wiki/Non-local_means) 或 [transformer 的 global attention](https://arxiv.org/abs/1706.03762) 相对于 CNN local receptive field 的角色。

**Insight 5: Loop closing 是 3D consistency 的 proxy**

GVS 的 loop closing 本质上是在 enforcing **spatial consistency**——不同时间但相同空间位置的 frames 应该 depict 相同 scene。这类似于 NeRF 的 multi-view consistency，但在生成框架中实现。Cyclic conditioning 可以看作在 sampling 时注入 3D structure prior，即使 model 本身没有显式 3D representation。

---

## 11. 相关工作的 broader context

### 11.1 Diffusion Stitching 谱系

- [MultiDiffusion (Bar-Tal et al., 2023)](https://arxiv.org/abs/2302.08113)：image generation 的多 region stitching
- [SyncTweedies (Kim et al., 2024)](https://arxiv.org/abs/2403.15470)：general synchronized diffusion framework
- [StochSync (Yeo et al., 2025)](https://arxiv.org/abs/2410.18795)：arbitrary space（panorama, mesh texture）的 stitching
- [CompDiffuser (Luo et al., 2025)](https://arxiv.org/abs/2503.07559)：goal-conditioned planning 的 stitching，需要 custom model
- [Stitch-OPE (Goli et al., 2025)](https://arxiv.org/abs/2410.18780)：off-policy evaluation 的 trajectory stitching

GVS 是第一个 **video generation** 的 stitching method，且是 training-free 的。

### 11.2 Long Video Generation 谱系

- **AR-based**：[DFoT (Song et al., 2025)](https://arxiv.org/abs/2502.07272), [SkyReels-V2 (Chen et al., 2025a)](https://arxiv.org/abs/2504.13074), [Magi-1 (Sand.ai, 2025)](https://arxiv.org/abs/2505.13211)
- **Retrieval-based**：[WorldMem (Xiao et al., 2025)](https://arxiv.org/abs/2411.09901), [Context as Memory (Yu et al., 2025)](https://arxiv.org/abs/2501.10200)
- **Compression-based**：[TTT (Zhang et al., 2025)](https://arxiv.org/abs/2505.23884), [One-Minute Video (Dalal et al., 2025)](https://arxiv.org/abs/2410.07791)
- **Stitching-based**：GVS（本文）

GVS 的独特价值：**non-causal**，适合有 future constraints 的任务。

### 11.3 Guidance 技术谱系

- [Classifier Guidance (Dhariwal & Nichol, 2021)](https://arxiv.org/abs/2105.05233)：需要 classifier
- [Classifier-Free Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)：不需要 classifier，但要求 conditioning 独立于 weights
- [History Guidance (Song et al., 2025)](https://arxiv.org/abs/2502.07272)：DF-specific，variable-length history conditioning
- [Inner Guidance (Chefer et al., 2025)](https://arxiv.org/abs/2410.11758)：conditioning signal 来自 model 自身
- **Omni Guidance (本文)**：bidirectional inner guidance，conditioning 来自 co-evolving past + future

---

## 12. 个人思考与开放问题

1. **Omni Guidance 的 $\gamma_1, \gamma_2$ 合并是否损失信息？** 论文实践中 merge 为单一 $\gamma$，但 $\gamma_1$（camera adherence）和 $\gamma_2$（neighbor consistency）语义不同。在某些 trajectory 上可能需要不同权重——比如 Straight line 更需要 camera adherence，Panorama 更需要 neighbor consistency。Adaptive $\gamma$ 可能是改进方向。

2. **Cyclic conditioning 的 spatial windows 是手工设计的。** Figure 8 的 spatial windows 针对每种 trajectory 手动定义。能否自动化？比如基于 camera pose 的 FOV overlap 自动发现 spatial neighbors，类似 retrieval-based methods 的思路，但嵌入到 stitching framework 中。

3. **External image conditioning 的根本矛盾。** Stitching 的 symmetry（所有 chunks 平等 co-evolve）与 external conditioning 的 asymmetry（context frame 固定）冲突。可能需要"软"外部条件——context frame 也参与 denoising 但被 strong prior 锚定，类似 [instruct-pix2pix](https://arxiv.org/abs/2211.09800) 的 image conditioning 但在 DF noise level 层面控制。

4. **与 3D-aware generation 的关系。** GVS 隐式实现了 3D consistency（loop closing），但 backbone 本身没有 3D representation。如果结合 [pixelSplat](https://arxiv.org/abs/2312.02111) 或 [MVSplat](https://arxiv.org/abs/2403.09633) 的 explicit 3D prior，可能解决 wide-baseline 失败问题。

5. **Compute 效率。** Vanilla GVS 并行 denoise 所有 windows，VRAM 需求高（H200）。Scalable version 逐 window denoising 但更慢。这与 [speculative decoding](https://arxiv.org/abs/2302.01318) 类似的 tradeoff——并行 vs. 串行的 memory-latency tradeoff。未来可能需要 chunk-level parallelism + step-level pipelining 的混合策略。

---

## 参考 links

- **GVS 项目主页**：https://andrewsonga.github.io/gvs
- **Diffusion Forcing**：https://boyuan.space/diffusion-forcing/
- **History-Guided Video Diffusion (DFoT)**：https://arxiv.org/abs/2502.07272
- **StochSync**：https://arxiv.org/abs/2410.18795
- **CompDiffuser**：https://arxiv.org/abs/2503.07559
- **DDIM**：https://arxiv.org/abs/2010.02502
- **Classifier-Free Guidance**：https://arxiv.org/abs/2207.12598
- **VideoJAM (Inner Guidance)**：https://arxiv.org/abs/2410.11758
- **RealEstate10K**：https://arxiv.org/abs/1805.09517
- **MEt3R**：https://arxiv.org/abs/2504.20774
- **VBench**：https://arxiv.org/abs/2311.13582
- **Video Depth Anything**：https://arxiv.org/abs/2501.12275
- **WorldExplorer**：https://arxiv.org/abs/2409.12169
- **DL3DV**：https://arxiv.org/abs/2312.16256
- **ScanNet++**：https://arxiv.org/abs/2308.11417
- **Penrose Stairs (Wikipedia)**：https://en.wikipedia.org/wiki/Penrose_stairs
- **MultiDiffusion**：https://arxiv.org/abs/2302.08113
- **WorldMem**：https://arxiv.org/abs/2411.09901
- **Stable Virtual Camera**：https://arxiv.org/abs/2503.18513
- **Magi-1 (Sand.ai)**：https://arxiv.org/abs/2505.13211
- **SkyReels-V2**：https://arxiv.org/abs/2504.13074
- **Oasis (Decart)**：https://oasis-model.github.io/
- **Effective Receptive Field (Luo et al.)**：https://arxiv.org/abs/1701.04128
- **Simple Diffusion (U-ViT)**：https://arxiv.org/abs/2301.11093
- **Elucidating Design Space (Karras)**：https://arxiv.org/abs/2206.00364
