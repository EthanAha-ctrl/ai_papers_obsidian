---
source_pdf: Finetuning Vision-Language-Action Models Requires.pdf
paper_sha256: 40d8534c1b869d61fd59e64bc9867a6981400747582e772e3d5160a0b0733c22
processed_at: '2026-08-18T12:54:40-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CLP 用人话讲一遍

## 一句话总结

这篇 paper 在说一件听起来有点反直觉的事：**modern VLA models (π0, GR00T-N1.5) 的 transformer layers 里有一大堆是"在睡大觉"的，你可以在 fine-tuning 之前直接删掉 30–50%，训练快 40%，推理快 30%，性能居然不掉，小数据下反而还涨**。Project page 在 <https://clpvla.github.io/>。

## 背景为什么有意思

Robot learning 这两年的 trajectory 很清楚：

- RT-2 / OpenVLA 那一代是 autoregressive 出 discrete action token ([RT-2, CoRL 2023](https://arxiv.org/abs/2307.15818); [OpenVLA](https://arxiv.org/abs/2406.09246))
- π0 / GR00T-N1.5 这一代转向 continuous action generation via flow matching / diffusion ([π0](https://arxiv.org/abs/2410.24164); [GR00T-N1.5](https://arxiv.org/abs/2503.14734))，trajectory 更平滑、physics prior 更强
- 代价是 model 越来越大：π0 3.5B、GR00T-N1.5 2.7B、SmolVLA 都 450M

大 model 带来两个具体的 pain：
1. **Downstream fine-tuning 贵到离谱**: LIBERO 上 fine-tune 一次 20 hours on 4×A100。这对 academic lab 是真问题，你想试个 idea 等一周才能看结果
2. **Real-time inference 慢**: π0 inference ~211ms，对应 ~5Hz control frequency，对动态 manipulation 不够

已有的 acceleration 方法 ([EfficientVLA](https://arxiv.org/abs/2411.10950), [VLA-Cache](https://arxiv.org/abs/2510.00517), [SpecPrune-VLA](https://arxiv.org/abs/2506.10514), [MoLe-VLA](https://arxiv.org/abs/2411.10950)) 要么只加速 inference 不动 training，要么引入 dynamic routing module + distillation pipeline，architectural 复杂度爆炸。CLP 想说的是：**你其实根本不需要这些花活，static 删 layer 就够了**。

## 核心发现：deep VLA 大量 layer 在"睡大觉"

这是 paper 里 Figure 2 最 striking 的图。作者用 CKA 这个 metric 量了 π0 和 GR00T-N1.5 各 module (VLM backbone, action head, DiT blocks) 的相邻 layer 表示相似度，结果发现：

**heatmap 上有大片 contiguous 的 dark red 区域 (CKA ≈ 1)**，意思是一大串相邻 layer 输出的 representation 几乎一模一样。

用比喻讲：transformer 就像一条流水线，每一站（layer）应该做点不一样的事——有的站做 attention、有的站做 MLP transformation、有的站融 multimodal info。但 CKA 告诉我们，π0 和 GR00T-N1.5 这条流水线上，**有一大堆相邻工位几乎不做任何 transform，原料进来到出去一模一样**。这些 layer 就是 redundant relay，把它们删了应该不伤 performance。

这个 finding 跟 LLM 圈最近一两年的发现完全一脉相承：
- [ShortGPT](https://arxiv.org/abs/2403.03753): "Layers in LLMs are more redundant than you expect"
- [Gromov et al.](https://arxiv.org/abs/2403.17887): "The unreasonable ineffectiveness of the deeper layers"
- [Nguyen et al., ICLR 2021](https://arxiv.org/abs/2010.15341): "Do wide and deep networks learn the same things?"

Lipline 一直知道 deep transformer 有 layer redundancy。CLP 把这件事在 VLA + continuous action generation 上系统化验证了一遍，并把它变成 actionable pipeline。

## CKA 是啥——用大白话

CKA = Centered Kernel Alignment，paper 公式是：

$$\mathrm{CKA}(H_i, H_j) = \frac{\|H_j^\top H_i\|_F^2}{\|H_i^\top H_i\|_F \cdot \|H_j^\top H_j\|_F} \tag{3}$$

变量解释：
- $H_i, H_j \in \mathbb{R}^{n \times d}$: 两个 layer 的 hidden states，$n$ 个 token，每个 token 是 $d$ 维向量
- $H H^\top \in \mathbb{R}^{n \times n}$: Gram matrix，entry $(p, q)$ 是 token $p$ 和 token $q$ 的内积相似度，描述"这一层 token 之间的相对几何"
- $\|\cdot\|_F$: Frobenius norm，所有元素平方和的开方
- 输出 $\in [0, 1]$

直觉: CKA 衡量的是 **"layer $i$ 的 token-to-token 相对几何关系"跟"layer $j$ 的 token-to-token 相对几何关系"统计上有多依赖**。如果 CKA = 1，意味着两层的 token 相对位置关系完全等价，layer $j$ 对 layer $i$ 没做任何 "informational transform"——即使数值上可能差很多（因为正交变换 / scaling 不影响 CKA）。

为什么不用 MSE 或 cosine？作者做了 ablation（Figure 3-d）：MSE / cosine / random / keep-first 全都不稳定，只有 CKA 能保留 post-pruning fine-tuning 时的 global topology，让 manifold 能 restore 回来。原因很简单：MSE 和 cosine 是 point-wise / vector-wise，对正交变换和 scaling 敏感；CKA 是 statistical dependence，对 coordinate frame 不敏感——它关心的不是"数值长得像不像"，而是"token 之间的相对结构有没有变"。

## CLP 做法的直觉

paper Algorithm 1 写得很清楚，但 intuition 几句话能讲明白：

1. **跑一遍 calibration set**：从 training data 里抽小批 examples 做单次 forward pass，记录每个 layer 的 hidden states
2. **算相邻 layer 的 CKA**: $s_\ell = \mathrm{CKA}(H_{\ell-1}, H_\ell)$，$s_\ell \to 1$ 说明这层 redundant
3. **Block grouping**: 不要直接 Top-K 单挑，因为 calibration set 偶然 noise 可能让某层 $s_\ell$ 偏高。设 threshold $\tau$，把连续满足 $s_\ell \geq \tau$ 的 layer 聚成 block
4. **每个 block 留首层当 anchor**: block 内部第一层通常是 "transition into plateau"，保留它作为 entry point，其余全进 candidate pool
5. **Top-K 删除**: 从 candidate pool 按 $s_\ell$ 排序选 $k$ 个最 redundant 的删掉
6. **Static remove + native finetune**: 因为 transformer block 都共享 hidden dim，删 layer = 直接把 predecessor output 接到 successor input，**不用加任何 auxiliary module、distillation loss、routing parameter**，直接 native fine-tune

最关键的设计决策是 **static + pre-fine-tuning**。MoLe-VLA 和 DeeR-VLA 都搞 dynamic layer skipping，但 dynamic 需要 routing module 在 runtime 决定走哪条路径，这就引入了额外的 trainable parameter、auxiliary loss、inference overhead，还跟 downstream learning algorithm (chain-of-thought, future prediction) 耦合不好。CLP 的 bet 是：**redundancy 不是 input-dependent 的，而是 structural 的，所以你不需要 dynamic 判断**——offline 算一次 CKA，永久删掉，就完了。

## 为什么 prune 反而能涨点？——这其实是核心 insight

这跟很多 human 的 intuition 相反：少 layer = 少 capacity = 性能更差，对吧？但 Table 6 的数据非常 striking：

10% LIBERO data:
- π0 baseline: 77.7%
- π0-MoLe (带 dynamic routing): 79.7%
- **π0-CLP (静态删除 12 层): 84.6%**

**少 layer 比多 layer 涨了 6.9%，比带 routing module 的 MoLe 还好**。

作者的解释是 **implicit regularization**：
- Deep model 在 web-scale pre-training 时为了 broad generalization 保留了 excess capacity
- Fine-tune 到 specific manipulation task 时，这些 excess capacity 变成"过剩"
- 数据少时，过剩 capacity 容易 overfit 到 demonstrations 里的 spurious correlations (sensor noise, controller artifacts, demonstrator habit)
- 删掉 redundant layer = 强制 remaining layer 做 meaningful work = regularization
- Fine-tuning 期间 remaining layers reorganize feature pathways，restore 原 manifold 的 expressive geometry

这跟 LLM/Vision 圈的 lottery ticket hypothesis、network pruning for regularization 完全一脉相承。但 paper 的 Figure 3-f PCA visualization 给了个很 elegant 的 evidence：

- **Base model**: hidden states 在 broad manifold 上 spread
- **Pruned before fine-tuning**: representations collapse 成 narrow subspace（破坏了 latent geometry）
- **Pruned after fine-tuning**: manifold restored，distribution 接近 original

"manifold restoration" 这个现象 build 我的 intuition 是：transformer 的深度既是 functional hierarchy（layer 1 学底层 feature，layer N 学高层 semantic），也是 **capacity reservoir**。删掉一截 reservoir，剩下的 layer 在 fine-tuning 期间会"重新分工"，把丢失的 expressive capacity 在自己的有限 depth 里重建出来。这个 rebuilding 过程本身就是个更强的 learning signal，迫使 layer 学更 generalizable 而不是 memorize。

这个 insight 比单纯"删 layer 省 compute"重要得多。它暗示 robot learning 圈过去几年迷恋 "scale is all you need" 可能 misplaced——很多 capacity 是 dormant 的，**真正的瓶颈是有效 capacity，不是总 capacity**。

## 几个 striking 的实验数据

**Training time (Table 1)**:
- π0: 15.5h → 11.2h (-27.8%)
- GR00T-N1.5: 10.7h → 7.4h (-30.8%)
- SmolVLA: 24.75h → 8.83h (-64.3%！)

**Inference latency (RTX 4070)**:
- π0: 211ms → 152ms (-27.9%, ~6.5Hz)
- GR00T-N1.5: 121ms → 85ms (-29.8%, ~11.7Hz)

**LIBERO baseline comparison (Table 2)**:
- π0 baseline 94.6% → π0-CLP 93.9% (-0.7%, 1.39× speedup)
- GR00T-N1.5 baseline 93.9% → GR00T-N1.5-CLP 93.0% (-0.9%, 1.42× speedup)
- EfficientVLA 在 Long task掉到 72.1% (baseline 90.0%) — token pruning 在 long-horizon 上不稳定，CLP Long 86.4% 稳得多

**Real-world 10 tasks across 4 embodiments (Table 4)**:
- Groceries ToBasket (UR10, 2800 demos): 90 → 89 (-1)
- Serve Napkin (UR5, 100 demos): 45 → 65 (+20)
- Screwdriver ToBasket (UR5, 100 demos): 15 → 30 (+15)
- Banana ToPot (ALOHA single, 150 demos): 65 → 75 (+10)
- Fold Shorts (ALOHA bimanual, 202 demos): 90 → 95 (+5)

Pattern 极其清晰：**数据越少，pruning 提升越大**。数据丰富的 task 略降（-1 ~ -5），数据稀缺的 task 大幅提升（+10 ~ +20）。这是 regularization hypothesis 的强证据。

**SimplerEnv (Table 3)**: GR00T-N1.5 baseline 16.57% → CLP 20.0%，training time 22.9h → 15.7h。SimplerEnv 是 hard benchmark，baseline 才 16% 多，CLP 提升 20% 相对幅度。

**具体剪哪些 layer (Table 5)**:
- π0 (18 layers VLM + action): 删 1, 2, 4, 6, 8, 9 — early + middle layers 偏多
- GR00T-N1.5 DiT action head (16 layers): 删 1, 2, 4, 5, 6, 7, 10, 11 — 一半
- SmolVLA (16 layers): 删 1, 2, 5, 6, 14, 15

Pattern 跟 LLM 圈 finding 一致：early layers 和 middle layers 更 redundant，最后几层 critical。

## VLA architecture 的 formalization — 公式 build intuition

paper Section 3 把 modern continuous-control VLA 抽象成两段式，这给了个统一的 framework 来分析：

**VLM backbone**:
$$H_0^{\mathrm{vlm}} = \mathrm{Embed}(\mathbf{x}^{\mathrm{lang}}, \mathbf{x}^{\mathrm{img}}), \quad H_\ell^{\mathrm{vlm}} = F_\ell^{\mathrm{vlm}}(H_{\ell-1}^{\mathrm{vlm}}) \tag{1}$$

- $\mathbf{x}^{\mathrm{lang}}$: language instruction
- $\mathbf{x}^{\mathrm{img}} \in \mathbb{R}^{H \times W \times 3}$: RGB observation
- $H_\ell^{\mathrm{vlm}} \in \mathbb{R}^{n \times d}$: 第 $\ell$ 个 VLM layer 输出，$n$ token 数、$d$ hidden dim
- $F_\ell^{\mathrm{vlm}}$: 第 $\ell$ 个 transformer block
- $Z = H_{N_v}^{\mathrm{vlm}}$: 最终 context representation

**Action head (flow matching based)**:
$$H_0^{\mathrm{act}} = \mathrm{Embed}_{\mathrm{act}}(\mathbf{a}_t, t), \quad H_m^{\mathrm{act}} = F_m^{\mathrm{act}}(H_{m-1}^{\mathrm{act}}; \Phi_m(Z)) \tag{2}$$

- $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$: target action chunk，$T_a$ action horizon，$d_a$ 单步 action 维度
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise，跟 $\mathbf{a}$ 同形
- $t \in [0, 1]$: flow matching timestep
- $\mathbf{a}_t = (1-t)\epsilon + t\mathbf{a}$: 从 noise 到 target action 的线性插值路径
- $\Phi_m(Z)$: VLM context $Z$ 注入第 $m$ action layer 的 cross-conditioning（cross-attention 或 token prefixing）

Flow matching loss:
$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{a}, \epsilon} \left[ \left\| f_{\mathrm{act}}(Z, \mathbf{a}_t, t) - (\mathbf{a} - \epsilon) \right\|_2^2 \right]$$

target $(\mathbf{a} - \epsilon)$ 是 $\mathbf{a}_t$ 对 $t$ 的导数——沿 linear path 的 constant velocity，flow matching 让 model 学这个 ground-truth velocity field。

**为什么这个 formalization 重要**: 它把 VLM 和 action module 都写成 deep transformer stacks 的形式，所以 intermediate hidden states 在结构上 compatible，可以用同一个 CKA metric 同时分析。这就是为什么 CLP 能同时 prune VLM backbone 和 action head。

## 我的联想——这个工作在更大图景里

**1. Foundation model efficiency 的下一波 from LLM to VLA**

LLM 圈 layer pruning 已经成熟（[ShortGPT](https://arxiv.org/abs/2403.03753), [SliceGPT](https://arxiv.org/abs/2401.15024), [LayerSkip](https://arxiv.org/abs/2404.16710)）。VLA 紧跟是自然的，因为 VLA backbone 就是 LLM/VLM。但 VLA 多了 action head 和 flow matching 这一块，paper 把这个也 cover 了。

**2. Static vs Dynamic 的古老 debate**

MoLe-VLA / DeeR-VLA / AC²-VLA 都是 dynamic layer skipping，但 paper Table 6 显示 static pruning 反而更好。这跟 LLM 圈 early-exit 工作（[LayerSkip](https://arxiv.org/abs/2404.16710)）的发现一致——dynamic 通常需要复杂 routing module 训练，但收益不大。Static 是简单且强力的 baseline，community 经常忽视。

**3. Pretraining vs Fine-tuning 的 capacity 分配**

Paper 的 finding 暗示一个更深的 hypothesis：**deep model 的 depth = (necessary functional hierarchy) + (capacity reservoir for adaptation)**。Pre-training 阶段为了 broad generalization 保留 reservoir，fine-tune 到 specific task 时 reservoir 大部分 dormant。这跟 [Gromov et al.](https://arxiv.org/abs/2403.17887) 的 "unreasonable ineffectiveness of deeper layers" 完全呼应。

**4. Implicit regularization via structural modification**

Pruning 作为 regularization 不是新概念（[Han et al. 2015](https://arxiv.org/abs/1506.02626)），但在 robot learning 圈还是第一次系统化验证。这暗示：**robot data 的 noise / spurious correlation 问题可能比想象严重**，需要比 LoRA / full fine-tuning 更强的 inductive bias。

**5. Robot learning 的 compute-centric turn**

最近一系列工作 ([FOCA](https://arxiv.org/abs/2506.01844), [HAMLET](https://arxiv.org/abs/2502.05476), CLP) 都在重新思考 compute allocation: 不是简单"加 data 加参数"，而是"把 compute 用在 information-rich 的地方"。CLP 是这个方向最 simple 且 effective 的 instance。

## Limitations 我自己加点 critical view

Paper section 7 自己承认的：
1. **Global pruning criterion**: 没区分 action token vs state token 的 redundancy pattern。我猜 action token 可能 redundancy 分布更不均匀，因为 action space 比 vision/language space 稠密
2. **只验证 post-pretraining fine-tuning**: pretraining stage 没试。如果 pruned model 不能 multi-task 继续训练，deployment 灵活性下降
3. **Cross-embodiment generalization 没验证**: 一个 embodiment 上算的 CKA profile 是否 transfer 到另一个？

我自己加几个 concern：
4. **Calibration set sensitivity 没分析**: $\mathcal{D}_{\mathrm{cal}}$ 大小、分布对结果影响没量化
5. **Threshold $\tau$ 的具体值和 tuning 方法 没披露**: 这影响 reproducibility
6. **Table 5 typo**: GR00T-N1.5 VLM 行写 "pruned 5" 但 indices 列了 7 个，需要查 appendix 或 code 确认
7. **跟 LoRA 的 interaction 没分析**: 实际部署时 LoRA + CLP 应该可以叠加，但 interaction 没说
8. **Long-horizon task 上 stability**: LIBERO Long 上 CLP 86.4% vs baseline 90.0%，掉了 3.6%。虽然比 EfficientVLA (72.1%) 好得多，但 long-horizon planning 是 VLA 最 critical 的能力，pruning 可能伤这部分

## 总结——一句话 build intuition

CLP 这篇 paper 的核心 insight 可以压缩成：

**Deep VLA 是 over-parameterized for fine-tuning。Pre-training 需要 deep capacity 学 broad features，downstream task 只用 subset，剩余 layer 是 dormant capacity。CKA 量化 dormant 程度，Top-K 删掉，剩下的 layer 在 fine-tuning 时会 "manifold restore"——既省 compute 又起 regularization 作用，小数据下反而涨点。**

更深层的信号是给 robot learning community 的：**别再盲目 scale-up 了，先看看你的 model 有多少 dead weight**。Robot data 的 compute 不是无限的，academic lab 跑不起 10B VLA，但跑得起 pruned 3B → 2B 的版本，还可能性能更好。这是 robot learning 从"scale-obsessed"走向"compute-aware"的重要 signal。

---

**Reference links**:
- CLP project page: <https://clpvla.github.io/>
- CKA original paper: <https://arxiv.org/abs/1905.00414>
- π0: <https://arxiv.org/abs/2410.24164>
- GR00T-N1.5: <https://arxiv.org/abs/2503.14734>
- SmolVLA: <https://arxiv.org/abs/2506.01844>
- ShortGPT (LLM layer pruning): <https://arxiv.org/abs/2403.03753>
- Unreasonable Ineffectiveness of Deeper Layers: <https://arxiv.org/abs/2403.17887>
- Flow Matching: <https://arxiv.org/abs/2210.02747>
- LIBERO: <https://arxiv.org/abs/2311.11540>
- RoboCasa: <https://robocasa.ai.github.io/>
- SimplerEnv: <https://simpler-env.github.io/>
- OpenVLA: <https://arxiv.org/abs/2406.09246>
- MoLe-VLA: AAAI 2026
- DeeR-VLA: NeurIPS 2024
- EfficientVLA: NeurIPS 2026
- VLA-Cache: NeurIPS 2026
- SpecPrune-VLA: ICML 2026
- FOCA (VinRobotics ICML 2026): future-conditioned VLA adaptation

---

# CLP: CKA-Guided Layer Pruning for VLA Models — 深度技术解析

## 1. 论文核心 thesis

这篇 paper 来自 VinUniversity / VinRobotics / Stanford / DFKI 等多机构合作（project leads 是 Duy Nguyen 和 Ngo Anh Vien），project page: <https://clpvla.github.io/>。核心 claim 非常直接：**modern continuous-control VLA foundations (π0, GR00T-N1.5) 存在严重的 layer-wise representational redundancy，可以用一个 training-free 的 CKA-based pipeline 在 fine-tuning 之前永久砍掉 30–50% 的 transformer layers，training 加速 40–50%，inference 加速 ~30%，performance 甚至还能涨**。这跟 LLM 圈 ShortGPT ([arxiv 2403.03753](https://arxiv.org/abs/2403.03753)) 和 Gromov et al. 的 "unreasonable ineffectiveness of deeper layers" ([arxiv 2403.17887](https://arxiv.org/abs/2403.17887)) 一脉相承，但首次系统化地搬到 VLA + continuous action generation 上。

## 2. Motivation: 为什么现有 VLA acceleration 方法不够

作者把现有方法分成三类，并指出各自的硬伤：

**A. Training-free inference acceleration** (VLA-Cache [16], EfficientVLA [8], SpecPrune-VLA [9], ADP [33])
- 只剪 token、cache KV、speculative decode
- **完全不降低 downstream fine-tuning 的 cost**，而 LIBERO 上 fine-tune 一次要 20h on 4×A100（[FOCA, ICML 2026](https://arxiv.org/abs/2506.01844)），这才是 robot learning 真正的 bottleneck

**B. Lightweight from-scratch** (RoboMamba [10], FLOWER-VLA [11], SmolVLA [12], NORA [13])
- 用 Mamba / Florence-2 / Qwen-2.5-VL 重新设计小模型
- **失去了 large pretrained backbone 的 broad capability**，generalization 差

**C. Training-adaptive** (DeeR-VLA [14], MoLe-VLA [15], AC²-VLA [34])
- 用 Mixture-of-Layers / early-exit / action-prior router 动态跳层
- **引入 auxiliary routing modules + distillation pipeline + 额外 training objectives**，architectural 复杂度很高，跟 downstream learning algorithm（chain-of-thought reasoning, future knowledge prediction 等）耦合差

CLP 的定位是 fourth bucket: **pre-fine-tuning structural compression，static，no auxiliary modules，no distillation，just remove layers and finetune natively**。

## 3. VLA Architecture formalization — 公式逐项解析

作者把 modern continuous-control VLA (π0 [arxiv 2410.24164](https://arxiv.org/abs/2410.24164), GR00T-N1.5 [arxiv 2503.14734](https://arxiv.org/abs/2503.14734), SmolVLA [arxiv 2506.01844](https://arxiv.org/abs/2506.01844)) 抽象成 decoupled 两段式架构。这点很关键：因为 VLM backbone 和 action head 都是 transformer stacks，所以 hidden states 在结构上 compatible，可以用同一个 CKA metric 同时分析。

### 3.1 VLM Backbone (context extractor)

$$H_0^{\mathrm{vlm}} = \mathrm{Embed}(\mathbf{x}^{\mathrm{lang}}, \mathbf{x}^{\mathrm{img}}), \quad H_\ell^{\mathrm{vlm}} = F_\ell^{\mathrm{vlm}}(H_{\ell-1}^{\mathrm{vlm}}) \quad \forall \ell \in \{1, \ldots, N_v\} \tag{1}$$

变量含义：
- $\mathbf{x}^{\mathrm{lang}}$: language instruction (e.g. "pick up the red cup and place it in the basket")
- $\mathbf{x}^{\mathrm{img}} \in \mathbb{R}^{H \times W \times 3}$: 当前帧 RGB observation
- $H_\ell^{\mathrm{vlm}} \in \mathbb{R}^{n \times d}$: 第 $\ell$ 个 VLM transformer layer 之后的 hidden token representation；$n$ 是 token 数（vision patch tokens + language tokens + 任何 special tokens），$d$ 是 hidden dimension
- $F_\ell^{\mathrm{vlm}}$: 第 $\ell$ 个 transformer block（multi-head attention + MLP + residual + LayerNorm）
- $N_v$: VLM backbone 总层数
- $Z = H_{N_v}^{\mathrm{vlm}}$: 最终 context representation，喂给 action head

### 3.2 Action-Generation Head (flow-matching based)

Action head 采用 flow matching objective（[Lipman et al., ICLR 2023](https://arxiv.org/abs/2210.02747)），目标是学一个 velocity field $f_{\mathrm{act}}$ 把 Gaussian noise 传输到 target action chunk。

$$H_0^{\mathrm{act}} = \mathrm{Embed}_{\mathrm{act}}(\mathbf{a}_t, t), \quad H_m^{\mathrm{act}} = F_m^{\mathrm{act}}(H_{m-1}^{\mathrm{act}}; \Phi_m(Z)) \quad \forall m \in \{1, \ldots, N_a\} \tag{2}$$

变量含义：
- $\mathbf{a} \in \mathbb{R}^{T_a \times d_a}$: target action chunk，$T_a$ 是 action horizon（一次预测多少步），$d_a$ 是单步 action 维度（e.g. 7-DoF end-effector pose + gripper）
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: standard Gaussian noise，跟 $\mathbf{a}$ 同形状
- $t \in [0,1]$: flow matching 时间步，$t=0$ 是 pure noise，$t=1$ 是 target action
- $\mathbf{a}_t = (1-t)\epsilon + t\mathbf{a}$: linear interpolation path，沿这条路径从 noise 走到 target
- $H_m^{\mathrm{act}}$: 第 $m$ 个 action transformer layer 之后的 hidden action-token representation
- $F_m^{\mathrm{act}}$: 第 $m$ 个 action transformer block
- $N_a$: action head 总层数
- $\Phi_m(Z)$: cross-conditioning signal，从 VLM context $Z$ 派生出来注入第 $m$ 个 action layer；具体实现可以是 decoder cross-attention（query 来自 action tokens，key/value 来自 $Z$）或者 token prefixing（把 $Z$ 直接拼到 action tokens 前面）

最终输出: $\hat{\mathbf{u}}_t = f_{\mathrm{act}}(Z, \mathbf{a}_t, t) = H_{N_a}^{\mathrm{act}}$，即预测的 velocity field。

**Flow Matching Loss**:

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t, \mathbf{a}, \epsilon} \left[ \left\| f_{\mathrm{act}}(Z, \mathbf{a}_t, t) - (\mathbf{a} - \epsilon) \right\|_2^2 \right]$$

解释一下 target: 在 linear interpolation path $\mathbf{a}_t = (1-t)\epsilon + t\mathbf{a}$ 上，对 $t$ 求导得到 $\frac{d\mathbf{a}_t}{dt} = \mathbf{a} - \epsilon$。所以 $(\mathbf{a} - \epsilon)$ 就是这条直线的 constant velocity，也就是 flow matching 想让 model 学的 ground-truth velocity field。Model 预测 $\hat{\mathbf{u}}_t$，loss 是 MSE。

**关键点 for CLP**: 这个 formulation 把 VLM 和 action module 都写成 deep transformer stacks 的形式，让 intermediate hidden states 在结构上 compatible，可以用同一个 CKA metric 同时做 layer-wise similarity 分析。

## 4. CKA — Centered Kernel Alignment 数学细节

$$\mathrm{CKA}(H_i, H_j) = \frac{\mathrm{HSIC}(K_i, K_j)}{\sqrt{\mathrm{HSIC}(K_i, K_i) \cdot \mathrm{HSIC}(K_j, K_j)}} = \frac{\|H_j^\top H_i\|_F^2}{\|H_i^\top H_i\|_F \cdot \|H_j^\top H_j\|_F} \tag{3}$$

变量含义：
- $H_i, H_j \in \mathbb{R}^{n \times d}$: 两个 layers 的 hidden states，跨 $n$ 个 token，每个 token 是 $d$ 维向量
- $K_i = H_i H_i^\top \in \mathbb{R}^{n \times n}$: layer $i$ 的 Gram matrix，entry $(p,q)$ 是 token $p$ 和 token $q$ 在 layer $i$ 的内积相似度
- HSIC: Hilbert-Schmidt Independence Criterion (centered 版本)，衡量两个 Gram matrices 的统计依赖
- $\|\cdot\|_F$: Frobenius norm，$\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$
- Output $\in [0, 1]$

第二个等号怎么来的: 对于 linear kernel 和 centered data，HSIC 有 closed form $\mathrm{HSIC}(K_i, K_j) = \frac{1}{(n-1)^2} \mathrm{tr}(\tilde{K}_i H \tilde{K}_j H)$，其中 $\tilde{K}$ 是 centered Gram matrix。对于 unnormalized CKA，在合适的 normalization 下化简成 Frobenius norm 比例。

**CKA 比 cosine / MSE 好在哪？** ([Kornblith et al., ICML 2019](https://arxiv.org/abs/1905.00414); [Nguyen et al., ICLR 2021](https://arxiv.org/abs/2010.15341))

- **Invariant to orthogonal transformations**: 对 $H_i$ 做 $H_i \to H_i R$（$R$ 正交），CKA 不变。这意味着 token space 的旋转、basis 变换不影响 similarity，符合"我们关心的是 statistical structure 而非 coordinate frame"的直觉
- **Invariant to isotropic scaling**: $H_i \to \alpha H_i$ 不改变 CKA。MSE 和 cosine 都没这个性质
- **捕捉 pairwise statistical dependence**: 不是 point-wise 距离，是整个 Gram matrix 之间的统计依赖

直觉: CKA = 1 意味着 layer $j$ 的 pairwise token similarity matrix 完全 predict layer $i$ 的，反之亦然。两个 layers 在"token 之间的相对几何关系"上完全等价，只是可能在 absolute coordinate 上有 orthogonal transform / scaling。这种情况下，layer $j$ 相对 layer $i$ 几乎不引入新的 representational transformation — 它是 redundant relay。

## 5. CLP Algorithm — 完整 pipeline

### 5.1 Calibration 阶段

从 training episodes 采样一个 compact calibration set $\mathcal{D}_{\mathrm{cal}}$（不需要太多，只要能 capture task distribution）。Forward pass 整个 $\mathcal{D}_{\mathrm{cal}}$ 通过 pretrained policy，对每个 layer $\ell \in \mathcal{T}_{\mathcal{M}} = \{1, \ldots, L_{\mathcal{M}}\}$ 提取 hidden states，跨 examples concatenate 得到 unified activation matrix $\bar{H}_\ell^{\mathcal{M}}$。$\mathcal{M}$ 是 prunable module（VLM backbone 或 action head）。

### 5.2 Sequential redundancy scoring

$$s_\ell^{\mathcal{M}} = \mathrm{CKA}(\bar{H}_{\ell-1}^{\mathcal{M}}, \bar{H}_\ell^{\mathcal{M}}), \quad \ell = 2, \ldots, L_{\mathcal{M}} \tag{4}$$

变量：
- $s_\ell^{\mathcal{M}} \in [0,1]$: layer $\ell$ 相对前一层的 representational similarity score
- $s_\ell \to 1$: layer $\ell$ 几乎不改变 representation，是 pruning candidate
- $s_\ell \to 0$: layer $\ell$ 做了 major transformation，必须保留

### 5.3 Block grouping (避免局部噪声)

直接 Top-K 选最高 $s_\ell$ 会有问题：calibration set 的 sampling noise 可能让某个 isolated layer $s_\ell$ 偶然偏高，单独剪掉会破坏 contiguity。作者引入一个 similarity threshold $\tau$，把 consecutive layers 聚成 contiguous block：

$$s_\ell^{\mathcal{M}} \geq \tau, \quad \ell = 2, \ldots, L_{\mathcal{M}}$$

满足这个条件的 consecutive layers 归入同一个 block $B$，得到 block 集合 $B_{\mathcal{M}} = \{B_1, \ldots, B_Q\}$。

### 5.4 Anchor retention + candidate pool

对每个 block $B$，保留 initial layer $r(B)$ 作为 "functional anchor"，其余加入 candidate pruning set:

$$\mathcal{P}_{\mathcal{M}} = \bigcup_{B \in B_{\mathcal{M}}} (B \setminus \{r(B)\}) \tag{5}$$

**为什么保留 block 的第一层？** 直觉: 每个 plateau block 代表一段"representational stagnation"，进入 plateau 的第一个 layer 通常是 "transition into plateau"，它可能承担了 setup / routing information 给后续 redundant layers 用的角色。如果把它也剪了，incoming information 可能无处可去，破坏更严重。保留 anchor 等于保留 plateau 的 "entry point"。

### 5.5 Top-K selection

从 candidate pool 中选 Top-K most redundant（按 $s_\ell$ 排序）:

$$\mathcal{R}_{\mathcal{M}} = \mathrm{TopK}_{\ell \in \mathcal{P}_{\mathcal{M}}}(s_\ell^{\mathcal{M}}, k_{\mathcal{M}}) \tag{6}$$

变量：
- $k_{\mathcal{M}}$: target pruning budget for module $\mathcal{M}$
- $\mathcal{R}_{\mathcal{M}}$: final removal set，$|\mathcal{R}_{\mathcal{M}}| = k_{\mathcal{M}}$

### 5.6 Static removal

$$\pi_\theta^{\mathrm{pruned}} = \mathrm{RemoveLayers}(\pi_\theta, \mathcal{R}_{\mathcal{M}})$$

因为 transformer blocks 都共享 identical hidden dimension $d$，layer removal 就是简单把 predecessor 的 output 直接接到 successor 的 input（reshape 一下 residual connection 的 indexing 就行）。**No auxiliary routing parameters, no distillation losses, no architectural modifications**。

整个 pipeline 在 Algorithm 1 (Appendix) 里。

## 6. Representational Plateaus — 实证发现

Figure 2 是 paper 最 striking 的 visualization。作者对 π0 和 GR00T-N1.5 的三个 sub-module（VLM backbone, action head, DiT blocks）分别画了 pairwise CKA heatmap。

关键观察:
- Heatmap 上有大片 contiguous 的 dark red 区域（CKA ≈ 1）
- 这些 "plateaus" 跨越多个连续 layers，说明这些 layers 几乎做 identical operation
- Major feature transformation 只发生在少数几个 transition layers 上

**为什么会有这种 plateau？** 我的 interpretation:

1. **Pre-trained backbone 的 over-parameterization**: π0 的 VLM backbone 来自 PaliGemma / Gemma，这些 model 在 web-scale pre-training 时为了 generalization 和 downstream adaptation 保留了 excess capacity。当 fine-tune 到 specific manipulation task 时，这些 layers 实际很少 contribute
2. **Depth-as-capacity-reserve**: transformer 的深度不全是 "必要的功能层级"，一部分是 "reserve for adaptation"。Pre-training 把这些 reserve 留着，fine-tune 时激活其中一部分，其他保持 dormant
3. **Action head 的 diffusion/DiT 同样冗余**: 即使是 task-specific 的 action head，也继承了 DiT (Diffusion Transformer) 架构，而这些 DiT 内部也存在跟 image generation 一样的 layer redundancy

## 7. 实验结果 — 详细数据解读

### 7.1 Efficiency comparison (Table 1)

| Model | Model Size↓ | Trainable Params↓ | Training Time↓ (60000 steps) | FLOPs↓ | Inference↓ (RTX 4070) |
|---|---|---|---|---|---|
| π0 (3.5B) | 22.9% → 2.7B | 25.8% → 2.3B | 27.8% (15.5h → 11.2h) | 28.5% (3073 → 2196.5) | 27.9% (211ms → 152ms) |
| GR00T-N1.5 (2.7B) | 25.9% → 2B | 30.1% → 0.75B | 30.8% (10.7h → 7.4h) | 49.3% (1010 → 512.4) | 29.8% (121ms → 85ms) |
| SmolVLA (450M) | 21.3% → 354M | 37% → 63M | 64.3% (24.75h → 8.83h) | 10.41% (598.4 → 536.1) | 31.84% (201ms → 137ms) |

观察:
- **Training time reduction 27–64%**，直接打到 fine-tuning bottleneck
- **GR00T-N1.5 的 FLOPs reduction 最激进 (49.3%)**，因为它的 DiT action head 16 层剪掉 8 层，而 DiT 是计算重头
- **SmolVLA training time reduction 最大 (64.3%)**，因为小模型 pruning 比例相对影响更显著，而且 SmolVLA 在 RTX 4070 上 inference 时间基数小
- **Inference latency 全部 < 160ms**，对应 > 6Hz control frequency，对 real-time manipulation 友好

### 7.2 LIBERO benchmark (Table 2)

LIBERO 四个 task suite: Spatial, Object, Goal, Long ([arxiv 2311.11540](https://arxiv.org/abs/2311.11540))

| Method | Spatial | Object | Goal | Long | Avg | Speedup |
|---|---|---|---|---|---|---|
| OpenVLA-OFT [6] | 97.6 | 96.5 | 97.9 | 94.5 | 96.6 | 1.00× |
| FastV [42] | 94.6 | 95.8 | 94.0 | 88.8 | 93.3 | 1.44× |
| EfficientVLA [8] | 96.5 | 91.1 | 96.0 | 72.1 | 88.9 | 1.52× |
| ADP [33] | 97.6 | 98.4 | 97.4 | 84.2 | 94.4 | 1.35× |
| **π0** | 94.6 | 98.2 | 95.4 | 90.0 | 94.6 | 1.00× |
| π0-SpecPrune-VLA | 96.6 | 98.0 | 95.2 | 84.2 | 93.5 | 1.31× |
| **π0-CLP** | 95.0 | 99.2 | 95.0 | 86.4 | **93.9** | **1.39×** |
| GR00T-N1.5 | 90.8 | 98.4 | 95.4 | 91.0 | 93.9 | 1.00× |
| **GR00T-N1.5-CLP** | 89.4 | 98.8 | 95.8 | 88.6 | **93.0** | **1.42×** |
| SmolVLA | 71.8 | 92.2 | 87.4 | 57.2 | 77.15 | 1.00× |
| **SmolVLA-CLP** | 75.6 | 93.0 | 81.6 | 56.2 | **76.75** | **1.47×** |

观察:
- **CLP 几乎 match baseline**（最多掉 0.9%），同时 1.39–1.47× speedup
- 对比 EfficientVLA: 它 Long task 掉到 72.1%（baseline 90.0%），而 CLP Long 86.4%，**stability 远好**
- 对比 ADP: ADP speedup 1.35× vs CLP 1.39×，但 ADP 只做 inference 不降低 training cost
- **SmolVLA 上 Spatial 反而提升**（71.8 → 75.6），暗示小模型 + 小数据下 pruning 的 regularization 效应

### 7.3 Few-shot regime — Regularization effect (Table 6)

10% LIBERO data，跟 MoLe-VLA ([arxiv MoLe-VLA, AAAI 2026](https://arxiv.org/abs/2411.10950)) 对比:

| Model | Long | Goal | Object | Spatial | Avg | Training Hours |
|---|---|---|---|---|---|---|
| π0 baseline | 58.8 | 87.8 | 82.6 | 81.6 | 77.7 | 15.5 |
| π0-MoLe | 60.2 | 88.2 | 86.0 | 84.4 | 79.7 | 15.6 |
| **π0-CLP** | **66.2** | **90.6** | **89.0** | **92.6** | **84.6** | **11.2** |

这个结果是 paper 最 striking 的 finding: **pruning 不仅不掉点，反而提升 6.9% success rate，同时 training time 还减少 28%**。MoLe-VLA 引入了 dynamic layer skipping module + auxiliary training objective，却输给完全 training-free 的 CLP。

**为什么 pruning 会提升 performance?** 作者 hypothesis 是 implicit regularization:
- Redundant layers 提供 excess capacity
- Limited data 下，excess capacity 容易 overfit 到 task-specific noise（demonstrations 里的 spurious correlations, sensor noise, controller artifacts）
- 移除 redundant layers 后，remaining layers forced to 学更 generalizable representations
- Fine-tuning 期间，remaining layers reorganize，恢复 expressive latent structure

这跟 lottery ticket hypothesis / network pruning for image classification 是一脉相承的 idea，但放到 VLA 上还是第一次系统地验证。

### 7.4 Manifold restoration — PCA visualization (Figure 3-f)

作者用 PCA 把 high-dimensional hidden states 投到 2D 可视化:
- **Base model**: latent representations 分布在 broad manifold（state/future tokens 和 action tokens 都有 diverse spread）
- **Pruned model before fine-tuning**: representations collapsed 成 narrow subspace — pruning破坏了 latent geometry
- **Pruned model after fine-tuning**: manifold restored，distribution close to original

这个 "manifold restoration" 现象很优雅: pruning 是破坏性的，但 fine-tuning 让 remaining layers 重新组织 feature pathways，恢复原 manifold 的 expressive capacity。这解释了为什么 compressed policy 能 maintain baseline-level manipulation performance。

### 7.5 Real-world deployment — 10 tasks across 4 embodiments (Table 4)

GR00T-N1.5 上验证，覆盖 UR10, UR5, ALOHA Single, ALOHA Bimanual:

| Task | Embodiment | Baseline | CLP | Δ |
|---|---|---|---|---|
| Groceries ToBasket | UR10 | 90 | 89 | -1 |
| Open Kettle | UR10 | 100 | 95 | -5 |
| Close Kettle | UR10 | 100 | 100 | 0 |
| Serve Napkin | UR5 | 45 | 65 | **+20** |
| Screwdriver ToBasket | UR5 | 15 | 30 | **+15** |
| Banana ToPot | ALOHA Single | 65 | 75 | **+10** |
| Cube ToDrawer | ALOHA Single | 75 | 60 | -15 |
| Block Stacking | ALOHA Single | 80 | 75 | -5 |
| Fold Shorts | ALOHA Bimanual | 90 | 95 | **+5** |
| Fly Towel | ALOHA Bimanual | 75 | 70 | -5 |
| **Avg** | | **73.5** | **75.9** | **+2.4** |

观察:
- **整体 avg 提升 2.4%**，不是掉点，是涨点
- **数据 scarce 的任务提升最大**: Serve Napkin (100 demos) +20%, Screwdriver (100 demos) +15%, Banana ToPot (150 demos) +10%
- **数据丰富的任务略降**: Groceries (2800 demos) -1%, Open Kettle (300 demos) -5%
- 这进一步验证了 regularization hypothesis: 数据越少，pruning 的 regularization effect 越显著

Real-world training time 数据 (Table 8): Groceries 11.8h → 8h (1.47×), FoldShorts 6.5h → 4.4h (1.47×), FlyTowel 3.2h → 2.1h (1.52×) — wall-clock training speedup 1.4–1.94×。

### 7.6 SimplerEnv (Table 3)

GR00T-N1.5 在 Bridge dataset 上 fine-tune，然后 evaluate SimplerEnv WidowX:

| Model | Training Time | Avg Success (7 tasks) |
|---|---|---|
| GR00T-N1.5 | 22.9h | 16.57% |
| GR00T-N1.5-CLP | 15.7h | **20.0%** |

Training time -31%，success rate +3.4%。SimplerEnv 是出了名的 hard benchmark（baseline 才 16.57%），CLP 提升 20% 相当可观。

### 7.7 Ablation: 为什么 CKA 而非 MSE / Cosine? (Figure 3-d)

对比 4 种 block selection strategies on GR00T-N1.5 across LIBERO:
- **CKA**: 最稳定，across all benchmarks 接近 baseline
- **MSE**: 不稳定，特别在 Long 和 Spatial 任务掉得厉害
- **Cosine**: 类似 MSE
- **Random**: 严重 degradation
- **Keep-First** (保留前 k 层，剪后面): 严重 degradation

**直觉解释**: MSE 和 Cosine 是 point-wise / vector-wise metrics，对 orthogonal transformations 和 scaling 敏感。两个 layers 可能 statistically dependent 但 point-wise 距离大（因为 orthogonal transform）。CKA 捕捉 statistical dependence，更能反映 "redundancy in representational structure"。Random 和 Keep-First 的失败说明 "选哪些层剪" 不是 trivial 的，需要 principled criterion。

Figure 3-f 的 PCA 也显示: 替代方法（MSE, Cosine, Random）的 hidden states 都 collapse 成 isolated subspaces，而 CKA 保留了 global topology，这对 post-pruning fine-tuning 让 manifold restore 很关键。

## 8. Pruning configurations (Table 5)

具体剪了哪些 layer index:

| Model | Module | Original Layers | Pruned Count | Pruned Indices |
|---|---|---|---|---|
| π0 | VLM + Action expert | 18 | 12 | 1, 2, 4, 6, 8, 9 |
| GR00T-N1.5 | VLM | 12 | 5 | 3, 4, 5, 6, 7, 8, 9 (注: paper 表格这里似乎有 typo，indices 7 个但 pruned count 写 5) |
| GR00T-N1.5 | VL-self-attention | 4 | 3 | 2 |
| GR00T-N1.5 | DiT Action head | 16 | 8 | 1, 2, 4, 5, 6, 7, 10, 11 |
| SmolVLA | VLM + Action expert | 16 | 10 | 1, 2, 5, 6, 14, 15 |

观察:
- **Early layers 倾向被剪**: π0 剪 1,2,4,6,8,9；GR00T-N1.5 剪 1,2,4,5,6,7,10,11；SmolVLA 剪 1,2,5,6,14,15 — 前几层 redundancy 高
- 这跟 LLM pruning 的 finding 一致（[ShortGPT](https://arxiv.org/abs/2403.03753) 发现 middle layers 最 redundant）

## 9. Limitations & Future Work (Section 7)

作者诚实承认:
1. **Global pruning criterion**: 没考虑 modality-specific token dynamics（action tokens vs state tokens 可能 redundancy pattern 不同）
2. **只验证 post-pretraining fine-tuning**: pretraining stage 没试 — 但作者 hint CLP 可以作为 "layer-selection prior during pretraining-stage adaptation"，即当你想从 VLM 演化到 VLA 时，CLP 能告诉你哪些 VLM layers 值得保留/adapt

## 10. 我的 critical analysis — 这个工作的真正贡献

### 10.1 Conceptual contribution
**"VLA models 需要 fewer layers than we think"** 这个 thesis 对 robot learning community 是个 paradigm shift。Robot learning 一直被 "scale is all you need" 主导，社区倾向于 ever-growing models (π0 3.5B → 后续可能 10B+)。CLP 指出 deep VLA 的深度有大量 "dead weight"，pre-training 阶段为了 broad generalization 留的 excess capacity，downstream adaptation 阶段几乎不激活。

### 10.2 Methodological contribution
- **CKA as actionable diagnostic** (而非仅 analysis tool): 之前 CKA 主要用于 paper 里画 figure "看 model 学了什么"，CLP 把它变成 actionable model modification pipeline
- **Static > Dynamic**: MoLe-VLA / DeeR-VLA 引入 routing module + auxiliary objectives 才能 dynamic skip layers。CLP 证明: 你不需要 dynamic — 直接 static 删掉就行，performance 还更好。这是个重要的 simplification signal
- **Pre-finetuning compression 而非 during-finetuning**: 跟 LoRA 等 PEFT 方法 orthogonal，可以叠加

### 10.3 Empirical contribution
- **10 real-world tasks across 4 embodiments**: 比 EfficientVLA (1-2 tasks)、VLA-Cache (1-2 tasks) 的 evaluation 全面得多
- **Few-shot regime 提升**: 不仅是 efficiency win，还是 performance win，这个 counterintuitive 结果会推动社区重新思考 VLA capacity

### 10.4 潜在 concerns
1. **Calibration set sensitivity**: paper 没详细分析 $\mathcal{D}_{\mathrm{cal}}$ selection 对结果的影响，如果 calibration set 偏向某类 trajectory，CKA profile 可能 biased
2. **Threshold $\tau$ tuning**: paper 没给 $\tau$ 的具体值和 sensitivity analysis，这影响 reproducibility
3. **Pre-training stage 没验证**: 如果 pruning 后的 model 不能继续 pre-train 或 multi-task train，那 deployment 灵活性下降
4. **Cross-embodiment generalization**: 一个 embodiment 上算的 CKA profile 是否 transfer 到另一个 embodiment？没明确验证
5. **Table 5 inconsistency**: GR00T-N1.5 VLM 行写 "pruned 5" 但 indices 有 7 个，影响 reproducibility
6. **Action token dynamics**: 作者自己承认 action tokens 和 state tokens 的 redundancy pattern 可能不同，但用了 global criterion

### 10.5 Broader implications

这个 work 让我联想到几个 trend:

1. **Foundation model efficiency 的下一波**: LLM 圈已经有 ShortGPT, SliceGPT, LayerSkip 等 layer pruning 工作。VLA 紧跟是自然的，因为 VLA backbone 就是 LLM/VLM
2. **Robot learning 的 compute-centric turn**: 从 "more data" 转向 "smarter compute allocation"。FOCA ([arxiv 2506.01844](https://arxiv.org/abs/2506.01844)) 用 future knowledge prediction，CLP 用 structural pruning，都是同一方向
3. **CKA / representational geometry 的实用化**: 之前 CKA 是 "做 analysis 画 figure"，现在变成 actionable。可能推动更多 "diagnostic-as-method" 工作
4. **VLA over-parameterization hypothesis 的实证**: 跟 "deep double descent" / "lottery ticket" 在 vision/NLP 的发现呼应，robot 终于也加入这个 narrative

### 10.6 Open questions for community
- 这种 redundancy 是 VLA 特有，还是所有 multi-modal foundation models 都有？
- 如果不同 embodiments / tasks 有不同 redundant layers，是否要做 "per-embodiment pruning"？
- Pruned model 能否继续 multi-task pre-training，或者只能 single-task fine-tune？
- 跟 LoRA / prefix tuning 怎么组合最佳？LoRA + CLP 应该可以叠加，但 interaction 没分析
- CKA threshold $\tau$ 能否自动 tune？是否有 closed-form optimal？
- pruning 之后剩余 layers 的 attention pattern 怎么变化？是否 shifted 到不同的 "head specialization"？

## 11. 总结 — Build your intuition

CLP 的核心 insight 可以压缩成 5 句话:

1. **Deep VLA 是 over-parameterized for fine-tuning**: pre-training 需要 deep capacity 学 broad features，但 downstream task 只用 subset
2. **CKA 是 information bottleneck detector**: 当 $s_\ell \to 1$，layer $\ell$ 相对 $\ell-1$ 几乎不做 transformation，是 redundant relay
3. **Pruning = structural regularization**: 移除 redundant layers 强制 remaining layers 做 meaningful work，避免 overfitting task-specific noise，这解释了 few-shot regime 的 +6.9% 提升
4. **Fine-tuning = manifold restoration**: 剩余 layers 在 fine-tuning 中 reorganize feature pathways，恢复 original latent geometry — paper Figure 3-f 的 PCA 显示这个 process
5. **Static > Dynamic**: 相比 MoLe-VLA 等 dynamic layer skipping，static pruning 简单、no runtime overhead、no auxiliary modules、performance 更好

这个 work 跟 LLM 圈的 layer pruning 工作遥相呼应，但首次系统化带到 VLA + continuous action generation 上，并且用 extensive real-world validation (10 tasks, 4 embodiments) 证明不只是 paper win，是真的 deployable。在 robot learning 越来越 "scale-obsessed" 的当下，是个非常重要的 "less is more" 信号。

---

**Reference links**:
- Project page: <https://clpvla.github.io/>
- CKA original: <https://arxiv.org/abs/1905.00414>
- π0: <https://arxiv.org/abs/2410.24164>
- GR00T-N1.5: <https://arxiv.org/abs/2503.14734>
- SmolVLA: <https://arxiv.org/abs/2506.01844>
- ShortGPT (LLM layer pruning): <https://arxiv.org/abs/2403.03753>
- Unreasonable Ineffectiveness of Deeper Layers: <https://arxiv.org/abs/2403.17887>
- Flow Matching: <https://arxiv.org/abs/2210.02747>
- LIBERO benchmark: <https://arxiv.org/abs/2311.11540>
- RoboCasa: <https://robocasa.ai.github.io/>
- SimplerEnv: <https://simpler-env.github.io/>
- OpenVLA: <https://arxiv.org/abs/2406.09246>
- MoLe-VLA: AAAI 2026
- DeeR-VLA: NeurIPS 2024
- FOCA (VinRobotics ICML 2026): future-conditioned VLA adaptation
