---
source_pdf: Causal-JEPA Learning World Models through Object-Level Latent Interventions.pdf
paper_sha256: 559be7ac89d76f128a9f2ae7cd97561aebb7c91bd8d55bc4aea107b420fc1c15
processed_at: '2026-08-03T15:20:02-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Andrej，我们把学术论文的包装剥掉，直接从第一性原理和工程直觉来聊 Causal-JEPA 这篇 paper。

核心问题其实非常简单：**怎么让神经网络真正理解物理世界里的“互动”，并且用最省算力的方式去做 world modeling 和 planning？**

### 1. 痛点在哪里？

现有的 world model 基本分两派，各有各的毛病：

第一派是 patch-based（像 DINO-WM）。它把图片切成 16x16 的小块，然后用 Transformer 去预测未来的 patch。效果不错，但是计算量极大。想象一下，你只是为了预测一个球在桌子上滚，你要在几百个 patch token 之间跑 self-attention，这是巨大的浪费。

第二派是 object-centric（像 SlotFormer）。它先用 Slot Attention 把图片压成几个 slot，每个 slot 代表一个 object（比如红球、蓝球、桌子）。预测空间一下子从几百个 patch 降到了几个 slot，极其高效。**但是**，模型很容易在这里走捷径。它往往只盯着 slot 自己的历史轨迹去做 linear extrapolation（“这个球刚才往左走，肯定继续往左走”），完全无视旁边正有一个方块飞过来要砸它。也就是说，它没有学到 **interaction**。

以前的同行怎么解决呢？要么在架构里强行拆分 self-dynamics 和 interaction dynamics（像 OCVP-Seq），要么用 sparse attention 去逼模型聚焦（像 SPARTAN）。这些都是在架构上打补丁。

C-JEPA 的直觉极其漂亮：**不要改架构，改 objective。通过 training objective 里的 masking，把 interaction reasoning 变成数学上的“必须项”。**

### 2. 核心机制的“人话”解释

想象你在看一场台球比赛。如果我要你预测红球下一秒去哪，你会怎么做？
如果红球现在匀速直线运动，你可以猜它继续走。但如果黑球正撞向它呢？你**必须**看黑球的位置和速度，才能算出红球被撞后的反弹轨迹。

C-JEPA 的训练就是基于这个常识。它在训练时，故意把红球在过去这几帧的轨迹**全部涂黑**（mask 掉），只留一个最初的位置告诉你“这是红球”。然后它问模型：“红球现在在哪？未来去哪？”

模型如果还想最小化 loss，它**绝对不可能**靠红球自己历史轨迹去猜，因为那些历史信息已经被抹掉了。它唯一的出路，就是去看黑球的轨迹，去推理：“黑球在 t=2 的时候轨迹发生突变了，说明它撞到了什么东西，那个东西肯定是被我 mask 掉的红球，所以红球现在应该在碰撞点附近，并且获得了黑球传递过来的动量。”

这就是 paper 里说的 **Latent Intervention**。Masking 在这里起到了因果干预的作用，强迫模型把 attention 放在 object 之间的相互作用上。

### 3. 技术细节拆解与公式解析

我们来看看这套机制在数学和架构上是怎么实现的。

#### 3.1 Object-Level Mask Token 的构造
这是我认为全篇最精妙的一个工程细节。

对于时间步 $\tau$ 被选中的 object $i$，它的 mask token 这样构造：

$$ \tilde{z}_\tau^i = \phi(z_{t_0}^i) + e_\tau \quad (\text{Eq. 3}) $$

*   $z_{t_0}^i$：这是 **identity anchor**。在历史窗口的最初始时刻 $t_0$，模型保留了这个 object 的真实 latent state。
*   $\phi$：一个 learnable linear projection，把初始 state 投影一下。
*   $e_\tau$：learnable temporal positional encoding，告诉模型“现在是第 $\tau$ 个时间步”。

**为什么这么设计？** 
因为 Slot Attention 是 permutation-equivariant 的，slot 的顺序是随机的。如果你像传统的 MAE 那样填一个 shared `[MASK]` token，模型根本不知道它要预测的是哪个 object 的轨迹。保留 $t_0$ 的信息作为 anchor，等于给模型发了一张身份证：“你在预测 1 号球”。这个细节完美解决了 set-based prediction 的身份识别问题。参考链接：Slot Attention (https://arxiv.org/abs/2006.15055)

#### 3.2 Learning Objective 的拆解
模型的 loss 是在 latent space 里的 L2 distance：

$$ \mathcal{L}_{\text{mask}} = \underbrace{\mathbb{E}[\|\hat{z}_\tau^i - z_\tau^i\|_2^2 \mid i \in \mathcal{M}, \tau \leq t]}_{\mathcal{L}_{\text{history}}} + \underbrace{\mathbb{E}[\|\hat{Z}_\tau - Z_\tau\|_2^2 \mid \tau > t]}_{\mathcal{L}_{\text{future}}} \quad (\text{Eq. 6}) $$

*   $\mathcal{L}_{\text{history}}$：填补被 mask 掉的历史信息。$i \in \mathcal{M}$ 表示被 mask 的 object，$\tau \leq t$ 表示历史窗口。这一项直接切断了“靠自身历史走捷径”的可能。
*   $\mathcal{L}_{\text{future}}$：预测未来 $\tau > t$ 的轨迹。这是标准的 world modeling 目标。
*   $\hat{z}, z$：分别是预测的 latent state 和 frozen encoder 提取出的 target latent state。

两者结合起来，模型为了在 future prediction 上做好，就会把在 history completion 里学到的 interaction pattern 直接复用过来。

#### 3.3 架构选择
Predictor $f$ 用的是 ViT-style 的 **Bidirectional Transformer**（类似 BERT），并且用了 frozen target encoder 的 JEPA 范式。

这点很关键。SlotFormer 用的是 autoregressive Transformer（类似 GPT），只能看前面的 token。但物理世界的交互是跨越时间的：A 撞 B，B 撞 C，你要预测 C，可能需要看 A 的信息。Bidirectional attention 让模型在填补 mask 时，能够全局统筹所有的 contextual objects。参考链接：I-JEPA (https://arxiv.org/abs/2301.08243), V-JEPA (https://arxiv.org/abs/2404.08471)

### 4. 实验数据的直觉印证

我们来看两个核心实验表格。

#### 4.1 CLEVRER Visual Reasoning (Table 1)

这是一个视频问答数据集，包含 Counterfactual reasoning（反事实推理，比如“如果红球没被撞，它会去哪”）。

| Model | Mask 数量 | Avg per que. (%) | Counterfactual per que. (%) |
| :--- | :--- | :--- | :--- |
| OC-JEPA (无 history mask) | 0 | 82.79 | 47.68 |
| C-JEPA | 1 | 83.95 (+1.16) | 49.67 (+1.99) |
| C-JEPA | 3 | 87.61 (+4.82) | 63.60 (+15.92) |
| C-JEPA | 4 | **89.40 (+6.61)** | **68.81 (+21.13)** |

**直觉解读**：OC-JEPA 是 C-JEPA 的完美对照组，架构完全一样，只是训练时不 mask 历史。数据表明，只要加上 object masking，整体准确率稳步上升。最可怕的是 Counterfactual 推理，直接暴涨 21%。这完美印证了我们的直觉：训练时的 Latent Intervention 让模型学会了真正的因果链条，所以在面对“如果...会怎样”的假设性问题时，它能够反推出来。

#### 4.2 Push-T Manipulation (Table 3)

这是一个机器人推 T 型木块的任务，用 MPC 规划。

| Model | Token 数量 | Success Rate (%) |
| :--- | :--- | :--- |
| DINO-WM (patch-based) | 196 × 384 = 75,264 | 91.33 (参考基准) |
| OC-DINO-WM (仅换 slot，无 mask) | 6 × 128 = 768 | 60.67 (-30.66) |
| OC-JEPA (加 JEPA loss，无 history mask) | 6 × 128 = 768 | 76.00 (+15.33) |
| C-JEPA (完整版) | 6 × 128 = 768 | **88.67 (+28.00)** |

**直觉解读**：这个表格简直是这篇 paper 灵魂的缩影。
1. DINO-WM 效果最好，但用的 feature 数量是天文数字。
2. OC-DINO-WM 证明了光把 patch 换成 object slot 是会崩盘的。因为 slot 太少了，一旦没有 interaction 约束，那点微弱的 latent 信息根本不足以表达复杂的动力学。
3. C-JEPA 用了完全一样的 768 个 feature（**仅仅占 patch 方法的 1.02%**），却把成功率拉回到了 88.67%，几乎追平 DINO-WM。
4. 实际跑规划的时间：DINO-WM 花了 5763 秒，C-JEPA 只花了 673 秒，**快了 8.6 倍**。因为规划全在极低维的 latent space 里做 Cross-Entropy Method (CEM) 搜索，维度低搜索就快。

### 5. 理论层面的 Intuition (Theorem 1)

作者用一套理论证明了为什么 Masking 能 force 出 interaction。核心概念叫 **Influence Neighborhood** $\mathcal{N}_t(i)$。

公式表示为：
$$ p(z_t^i \mid Z_T^{(-i)}) = p(z_t^i \mid \mathcal{N}_t(i)) \quad (\text{Eq. 8}) $$

*   $z_t^i$：object $i$ 在时刻 $t$ 的状态。
*   $Z_T^{(-i)}$：除了 object $i$ 自身历史之外的所有 context 信息（其他物体、action 等）。
*   $\mathcal{N}_t(i)$：能够预测 object $i$ 状态的**最小充分集**。

**直觉**：在 Markov blanket 和 Pearl 的因果图里，你要找 causal parents。但这在真实高维世界里极难做到，充满了 unobserved confounders。作者务实得多，他们定义的 $\mathcal{N}_t(i)$ 只是一个 **predictively sufficient set**。只要在 masking 干预下，某些 variables 能帮你把 $z_t^i$ 预测出来，那它们就是你的 influence neighborhood。

Theorem 1 证明了：在 L2 loss 下，Bayes optimal predictor **必须**利用 $\mathcal{N}_t(i)$ 里的信息，否则 loss 不可能降到最低。这从数学上堵死了模型走捷径的可能。这其实和 Invariant Risk Minimization (IRM, https://arxiv.org/abs/1907.02893) 的思想非常像，只不过 C-JEPA 是在单一数据集内通过 mask 制造了多种“环境”干预。

### 6. 一些发散的工程直觉与联想

和 Andrej 你聊聊我看完这篇 paper 后的一些更野的直觉：

1.  **Train-Test Mismatch 的妥协**：训练时用 BERT-style 的 bidirectional attention 做 mask completion，推理时却用 GPT-style 的 forward prediction。这在 NLP 里是大忌，但在 world model 里居然 work。这背后暗示了一个深刻的物理直觉：物理定律是时间对称的（至少在微观层面），你在双向补全时学到的 interaction graph，在正向推演时同样适用。这说明模型抓到了 invariant mechanics，仅仅是表面条件概率的转换。参考：V-JEPA2 (https://arxiv.org/abs/2506.09985)
2.  **Object Encoder 的天花板**：Table 1 里，如果用 SAVi 代替 VideoSAUR，mask 4 个 object 时性能反而崩了。因为 SAVi 的 object alignment 不够 robust，你 mask 掉太多，模型连身份证都认不全了。C-JEPA 的上限被死死按在了 frozen encoder 的质量上。未来的破局点必须是 end-to-end joint training，但这又容易 representation collapse，非常难搞。参考：SAVi (https://openreview.net/forum?id=E7iUobBQqtV)
3.  **和 LLM 的同构性**：你做 nanoGPT 时，LLM 通过 next-word prediction 学到了语法和推理。C-JEPA 做的是 next-state prediction in latent space，加上 masked-state completion。LLM 靠预测下一个词学到了 world knowledge，C-JEPA 靠预测被 mask 掉的 object 状态学到了 physical interaction。两者底层哲学完全一致：**通过极具挑战性的自监督预测任务，强迫网络重构出底层的生成规则。**
4.  **Hungarian Matching 的痛**：在 Push-T 规划时，由于 slot 顺序不固定，比较 predicted state 和 goal state 还得做一次 Hungarian matching 算法去 align。这是一个很 ugly 的工程 hack。理论上一个好的 object-centric world model 应该在 time axis 上自然保持 slot 一致性，但这在目前的开源 model 里几乎没人能完美做到。

总结一下，C-JEPA 把 JEPA 家族从 patch level 提升到了 object level，并且用一个极度简单的 masking trick 植入了 causal inductive bias。它没有引入任何新的复杂架构，只是改变了自监督的 pretext task，就实现了算力降维打击。这非常符合好的 AI research 的审美：用极其简单的数学机制解决根本的 representation 问题。

---

# Causal-JEPA: 深度技术解读

Andrej, 这是一篇让我很兴奋的 paper，因为它把 LeCun 的 JEPA 路线图 (I-JEPA → V-JEPA → V-JEPA2) 和 object-centric representation 这两个脉络终于缝起来了，而且缝合的方式非常巧妙——**通过 masking 来 induce causal inductive bias**，而不是靠 architectural factorization 或 sparse regularization。下面我从 motivation、公式、架构、实验、理论五个层面给你拆解，最后给你 build intuition。

---

## 1. Motivation: 为什么单靠 object-centric 不够

先回顾一下 LeCun 的 JEPA 哲学 (LeCun, 2022; https://openreview.net/pdf?id=4cYrrCbsyq):
- 抛弃 pixel reconstruction (MAE 那条路)，直接在 latent space 做预测
- 用 mask + predict 学到 predictive relationship，representation 自然 align 到下游 prediction/control
- I-JEPA (Assran et al., 2023, https://arxiv.org/abs/2301.08243) 在 image 上做 patch masking
- V-JEPA (Bardes et al., 2024, https://arxiv.org/abs/2404.08471) 扩展到 spatiotemporal tube
- V-JEPA2 (Assran et al., 2025, https://arxiv.org/abs/2506.09985) 整合 understanding + planning

Object-centric representations (Slot Attention, Locatello et al., 2020, https://arxiv.org/abs/2006.15055) 给出 N 个 slots $\{s_t^1, \dots, s_t^N\}$，每个 slot 对应一个 entity，permutation-equivariant。这看起来很自然地契合 world model——毕竟物理世界是 object 之间相互作用。

**但是！** 现有 object-centric world models 有几个 known failure modes:
- SlotFormer (Wu et al., 2023, https://openreview.net/forum?id=TFbwV6I0VLg) 没有 explicit interaction constraint，容易 fallback 到 self-dynamics
- C-SWM (Kipf et al., 2020, https://openreview.net/forum?id=H1gax6VtDB) 需要预先固定 graph
- SPARTAN (Lei et al., 2025, https://openreview.net/forum?id=uS5ch7GjZ4) 用 sparse attention 正则化
- OCVP-Seq (Villar-Corrales et al., 2023, https://arxiv.org/abs/2309.16591) 通过 architectural factorization 分开 self-dynamics 和 interaction

这些方法都是在**架构上**加 constraint。C-JEPA 的 key insight 是：**通过 objective 本身来 enforce interaction reasoning**，让 interaction 成为 functionally necessary 而非 architectural enforced。

---

## 2. 核心方法: Object-Level Masking as Latent Intervention

### 2.1 Problem Setup

给定视频序列 $\{X_t\}_{t}$，用 frozen object-centric encoder $g$ 把每一帧映射成 slots:

$$S_t = g(X_t) = \{s_t^1, \ldots, s_t^N\}, \quad s_t^i \in \mathbb{R}^d \quad (\text{Eq. 1})$$

- $X_t \in \mathbb{R}^{H \times W \times C}$: pixel observation
- $g$: VideoSAUR (Zadaianchuk et al., 2023, https://arxiv.org/abs/2307.07395) 或 SAVi (Kipf et al., 2022, https://openreview.net/forum?id=E7iUobBQqtV)
- $N$: fixed number of slots (CLEVRER 用 7，Push-T 用 4，含 1 个 background slot)
- $d$: slot dimensionality (实验里固定为 128)
- permutation-equivariant w.r.t. slot ordering

History window 长度 $T_h$，prediction horizon 长度 $T_p$。定义:
- $T := \{t-T_h+1, \ldots, t\}$ (history index set)
- $\mathcal{T} := \{t-T_h+1, \ldots, t+T_p\}$ (full history-future interval)

对每个时间步 $\tau \in T$，把 object set 分成 masked 和 context 两部分:

$$S_\tau^m = \{s_\tau^i \mid i \in \mathcal{M}_\tau\}, \quad S_\tau^c = \{s_\tau^j \mid j \notin \mathcal{M}_\tau\} \quad (\text{Eq. 2})$$

- $\mathcal{M}_\tau \subset \{1, \dots, N\}$: time-varying masked index set
- masked slots 用 mask token $\tilde{S}_\tau^m$ 替换
- context slots 保留

Auxiliary variables $U_t = \{a_t, p_t\}$:
- $a_t$: action
- $p_t$: proprioceptive signal
- 当成额外的 entity tokens $Z_t = \{S_t, U_t\}$

### 2.2 The Mask Token Construction (这是最精妙的地方)

这是 C-JEPA 的核心创新。对于 object $i$ 在时刻 $\tau$ 被 mask:

$$\tilde{z}_\tau^i = \phi(z_{t_0}^i) + e_\tau \quad (\text{Eq. 3})$$

变量解析:
- $\phi$: learnable linear projection, 把 anchor 投影到 mask space
- $z_{t_0}^i$: **identity anchor**——只在最早时间步 $t_0$ 保留该 object 的真实 latent
- $e_\tau$: learnable embedding + temporal positional encoding

**为什么需要 identity anchor？** 因为 Slot Attention 的 permutation-equivariance：如果不给 anchor，predictor 没法知道"我是要预测哪个 object"。这是和 patch-based MAE 的根本区别——patch 有 spatial position，slot 没有 object identity。所以必须 leak 一点点信息——最早的 latent 作为"我是谁"的 anchor，但中间所有时间步的 state 都被 mask 掉。

这个设计其实非常像 BERT 的 [MASK] token + segment embedding 的混合，但更巧妙——anchor 携带 identity，mask embedding 携带 temporal position。

### 2.3 Masking 策略的几何理解

看 Figure 2 的描述，masking 跨越整个 history window $T$，**除了** $t_0$ 这个 identity anchor 时刻。也就是说，object $i$ 在 $t_0$ 之后的每一帧都被 mask，predictor 必须从其他 objects 的 trajectory 推断 object $i$ 在中间时刻的状态。

直觉：如果你看到球 A 撞了球 B，但你 mask 掉了球 B 的整条轨迹（除了初始位置），predictor 必须通过 A 的运动 + 接触时刻推断 B 的反弹轨迹。这就是 latent intervention——你没有改变 transition mechanism (球的物理不变)，但你切断了观察 B 的能力。

### 2.4 Learning Objective

$$\hat{Z}_\mathcal{T} = f(\bar{Z}_\mathcal{T}) \quad (\text{Eq. 4})$$

- $f$: ViT-style masked transformer with **bidirectional attention** (像 BERT 不是 GPT)
- $\bar{Z}_\mathcal{T}$: masked input sequence over $\mathcal{T}$
- 注意：bidirectional attention 是关键设计选择，让 predictor 能 joint infer masked history + predict future

Loss:

$$\mathcal{L}_{\text{mask}} = \mathbb{E}\left[\sum_{\tau \in \mathcal{T}} \sum_{i=1}^N \mathbf{1}[\bar{z}_\tau^i \neq z_\tau^i] \|\hat{z}_\tau^i - z_\tau^i\|_2^2\right] \quad (\text{Eq. 5})$$

- $\mathbf{1}[\bar{z}_\tau^i \neq z_\tau^i]$: indicator, 只对被 mask 的 tokens 计算 loss
- 注意是对**所有** $\tau \in \mathcal{T}$ 求和，包括 history 和 future

分解:

$$\mathcal{L}_{\text{mask}} = \underbrace{\mathbb{E}[\|\hat{z}_\tau^i - z_\tau^i\|_2^2 \mid i \in \mathcal{M}, \tau \leq t]}_{\mathcal{L}_{\text{history}}} + \underbrace{\mathbb{E}[\|\hat{Z}_\tau - Z_\tau\|_2^2 \mid \tau > t]}_{\mathcal{L}_{\text{future}}} \quad (\text{Eq. 6})$$

- $\mathcal{L}_{\text{history}}$: masked completion on history, 防止 trivial self-dynamics shortcut
- $\mathcal{L}_{\text{future}}$: forward world modeling, 标准的 next-state prediction

**两个 loss 的组合是关键**。如果只有 future prediction，模型可以学 $s_{t+1}^i \approx s_t^i + \text{small drift}$，根本不学 interaction。但加上 history masking，你必须从其他 objects 推断 $s_\tau^i$，迫使 attention 必须捕捉 A→B 的 interaction 才能完成 completion。

### 2.5 Inference

推理时**不 mask history**，只 mask future tokens (future 本来就是要预测的)。这样 C-JEPA 就是标准的 forward world model，可以直接 rollout 做 planning。

训练时 bidirectional + mask，推理时 unidirectional forward——这是一个 train-test mismatch，但作者论证 (Remark 3) 这是合理的：训练时学到的 interaction structure $\mathcal{N}_t(i)$ 是 direction-agnostic 的，但在 Assumption 2 (shared transition mechanism) 下，这些 interaction constraints 在 forward dynamics 下依然有效。

---

## 3. 架构图解析

### 3.1 Training Pipeline (Figure 1)

```
[History Frames X_{t-2}, X_{t-1}, X_t] ──→ [Frozen Encoder g] ──→ [Slots S_{t-2}, S_{t-1}, S_t]
                                                                        │
                                                                        ▼
                                                              [Object-Level Masking]
                                                              (random |M| objects masked
                                                               across history except t_0)
                                                                        │
                                                                        ▼
                                                              [Masked Input \bar{Z}_\mathcal{T}]
                                                              + Auxiliaries U_t (action, proprio)
                                                                        │
                                                                        ▼
                                                              [Predictor f (ViT, bidirectional)]
                                                                        │
                                                                        ▼
                                                              [\hat{Z}_\mathcal{T}: predicted masked + future]
                                                                        │
                                                                        ▼
                                                              [L2 loss vs target slots]
                                                              (target from frozen encoder)
```

### 3.2 Frozen Encoder 细节

主实验用 VideoSAUR:
- Backbone: frozen DINOv2 ViT-S/14 (Oquab et al., 2024, https://openreview.net/forum?id=a68SUt6zFt)
- 196 patches/frame, dim 384
- Project to 128-dim, Slot Attention 2 iterations
- 训练 100k steps, Adam, lr $10^{-4}$ linearly scaled by batch size, exponential decay with 2k warmup
- Objective: feature reconstruction on DINOv2 + temporal similarity loss

SAVi baseline:
- 64×64 image, CNN backbone + Slot Attention
- N=7 slots, dim 128
- stochastic SAVi with Gaussian prior variance 0.01
- 8 epochs, Adam, cosine schedule with 2.5% warmup

### 3.3 Predictor 架构

- 6 Transformer layers
- 16 attention heads, head dim 64
- MLP hidden 2048
- 基于 stable-pretraining (Balestriero et al., 2025, https://arxiv.org/abs/2511.19484) 和 stable-worldmodel (Maes et al., 2026, https://arxiv.org/abs/2602.08968) 框架
- Slot dim 全程 128

Training:
- Push-T: history=3, frame skip=5, predict 1 future step
- CLEVRER: history=6, frame skip=2, predict 10 future steps
- 30 epochs, Adam, batch 256, lr $5 \times 10^{-4}$
- Masking: Push-T mask 0-2 objects, CLEVRER mask 0-4 objects

### 3.4 为什么用 ViT 而不是 autoregressive Transformer?

作者在 Appendix E.1 论证: object dynamics 不是 first-order Markov，interaction 跨多个 time step。Autoregressive (像 SlotFormer) 强加 sequential dependency，可能 bias 到 local self-dynamics。Masked prediction 让 model joint attend 整个 history window，更适合 interaction reasoning。

这点其实很关键——SlotFormer 用 causal mask + autoregressive rollout，C-JEPA 用 bidirectional + joint completion。BERT vs GPT 的差别在这里以 world model 的形式重现。

---

## 4. 实验结果详解

### 4.1 CLEVRER Visual Reasoning

CLEVRER (Yi et al., 2020, https://openreview.net/forum?id=HkxYzANYDB): synthetic video with multi-object interactions, 4 类问题:
- Descriptive (per question)
- Predictive (per option + per question)
- Explanatory (per option + per question)
- Counterfactual (per option + per question) ← **重点考察**

Evaluation pipeline:
1. Train world model
2. Rollout 128-frame input video → 160 frames (generate imagined trajectory)
3. Train ALOE (Ding et al., 2021, https://openreview.net/forum?id=lHmhW2zmVN) on these trajectories for VQA

ALOE: Transformer, 12 layers, 8 heads, FFN 512, shared embedding dim 16, MLP classifier hidden 128. 400 epochs, Adam lr $10^{-3}$ cosine decay.

**Table 1: Masking 数量的 ablation (VideoSAUR encoder)**

| Model | |M| | Avg per que. (%) | CF per opt. (%) | CF per que. (%) |
|-------|---|---|---|---|
| OC-JEPA (V) | 0 | 82.79 | 79.53 | 47.68 |
| C-JEPA (V) | 1 | 83.95 (+1.16) | 80.34 (+0.81) | 49.67 (+1.99) |
| C-JEPA (V) | 2 | 84.56 (+1.77) | 80.61 (+1.08) | 50.25 (+2.57) |
| C-JEPA (V) | 3 | 87.61 (+4.82) | 86.49 (+6.96) | 63.60 (+15.92) |
| C-JEPA (V) | 4 | **89.40 (+6.61)** | **88.67 (+9.14)** | **68.81 (+21.13)** |

观察:
- OC-JEPA 是 C-JEPA 的 history-unmasked 版本，**完全相同架构**，只去掉 object masking——这是关键的 controlled ablation
- Masking 越多，counterfactual 准确率提升越大——从 47.68% 到 68.81%，**absolute +21.13%**
- 但 SAVi encoder 的 ablation 显示 mask 4 个会下降 (因为 SAVi 不如 VideoSAUR robust)
- 这说明 masking 的最优 budget 依赖于 encoder 质量

**Table 2: vs 其他 object-centric baselines**

| Model | Avg (%) | CF opt (%) | CF que. (%) |
|-------|---|---|---|
| SlotFormer | 79.44 | 79.28 | 47.29 |
| SlotFormer (-recon) | 44.94 (-34.50) | 55.62 (-23.66) | 11.10 (-36.19) |
| OCVP-Seq | 83.11 | 83.21 | 56.06 |
| OCVP-Seq (-recon) | 80.09 (-3.02) | 77.46 (-5.75) | 43.00 (-13.06) |
| OC-JEPA | 77.28 | 76.69 | 41.10 |
| C-JEPA | **83.88** | **85.16** | **60.19** |

关键 takeaways:
- SlotFormer 严重依赖 reconstruction loss——去掉掉 34%
- OCVP-Seq 靠 architectural factorization 部分缓解
- C-JEPA **完全无 reconstruction**，counterfactual +13% over OC-JEPA

### 4.2 Push-T Manipulation

Push-T (Bekris et al., 2025, https://arxiv.org/abs/2303.04137): 机器人推 T 形物体到目标位姿，contact-rich。

Planning via MPC:
$$a_{t:t+H-1}^* = \arg\min_{a_{t:t+H-1}} \|\hat{S}_{t+H} - S_g\|_2^2 \quad (\text{Eq. 7})$$

- $H=5$ planning horizon, $B=5$ action block size, total action seq length 25
- CEM: 300 samples, 30 elites, 30 iterations
- Goal: 25 steps after initial frame
- Success: position error < 20 (workspace 0-450), orientation error < π/9
- Hungarian matching 用于 slot alignment

**Table 3: Push-T 结果**

| Token×d | Model | Success (%) |
|---------|---|---|
| 196×384 | DINO-WM | 91.33 (ref.) |
| 196×384 | DINO-WM-Reg. | 88.00 (-3.33) |
| 6×128 | OC-DINO-WM | 60.67 (-30.66) |
| 6×128 | OC-JEPA (+JEPA) | 76.00 (+15.33) |
| 6×128 | C-JEPA (+Mask) | **88.67 (+28.00)** |

Token 数对比:
- DINO-WM: 196 patches × 384 dim = 75,264 features
- C-JEPA: 6 slots × 128 dim = 768 features
- **C-JEPA 只用 1.02% 的 features**

Efficiency:
- DINO-WM: 5,763 seconds 平均 (50 trajectories, 3 seeds, L40s)
- C-JEPA: 673 seconds
- **8.6× faster**

这个 progress 看起来很 clean:
- DINO-WM (patch) → OC-DINO-WM (slot, autoregressive): 性能暴跌，说明 object-centric alone 不行
- OC-DINO-WM → OC-JEPA (slot, JEPA-style): 恢复一部分，说明 joint latent prediction 比 autoregressive 好
- OC-JEPA → C-JEPA (slot, JEPA + masking): 几乎追平 patch-based DINO-WM

### 4.3 Masking Strategy Ablation (Appendix J)

三种 masking:
- **Object-level**: mask entire slots
- **Token-level**: random individual tokens
- **Tube-level**: contiguous spatiotemporal tubes

Table A4 在 CLEVRER 上比较。在 matched budget 下:
- Tube 56%: 89.46% avg, 69.81% CF
- Object 4/7: 89.40% avg, 68.81% CF
- Token 56%: 89.32% avg, 68.88% CF

Table A5 在 Push-T 上:
- Object 1/4: 88.67%
- Object 2/4: 82.67%
- Token 25%: 84.67%
- Token 50%: 84.00%
- Tube 25%: 55.33%
- Tube 50%: 5.33% (崩溃)

Object-level masking 在 control 下更稳定——tube-level 高 budget 直接崩。直觉：tube 可能 mask 掉所有 object 在某段时间，导致 information 完全缺失；object-level 始终保留其他 objects 的完整 trajectory，所以总有 contextual info 可用。

---

## 5. 理论分析

这是 paper 的 Section 6，我详细给你拆。

### 5.1 Assumptions

四个假设:
1. **Temporally Directed Predictive Dependencies**: future state 由 past observations + auxiliary 决定，没有 instantaneous causal cycles
2. **Shared Transition Mechanism**: conditional distribution 跨 trajectory 不变 (stationarity)
3. **Object-Aligned Latent Representation**: slot ↔ coherent object, sufficient abstraction
4. **Finite-History Sufficiency**: $T_h$ 足够预测 future (但允许 higher-order dynamics, 比如 velocity 需要 multiple frames)

**注意作者强调不假设**:
- causal sufficiency (允许 unobserved confounders)
- first-order Markov (允许 higher-order)
- global sparsity (Pandaram et al., 2025, https://arxiv.org/abs/2511.08086 表明这太限制)

### 5.2 Definition 1: Influence Neighborhood

$$\mathcal{N}_t(i) \subseteq Z_T^{(-i)}: \quad p(z_t^i \mid Z_T^{(-i)}) = p(z_t^i \mid \mathcal{N}_t(i)) \quad (\text{Eq. 8})$$

- $Z_T^{(-i)}$: 所有 entity tokens 除了 object $i$ 的 history (保留 identity anchor)
- $\mathcal{N}_t(i)$: minimal sufficient subset
- minimality: 没有严格子集满足这个条件

这是 Markov blanket 的弱化版——不要求 full causal graph，只要求 conditional sufficiency under masking。

### 5.3 Theorem 1

$$\hat{z}_t^{i*} = \mathbb{E}[z_t^i \mid Z_T^{(-i)}] = \mathbb{E}[z_t^i \mid \mathcal{N}_t(i)] \quad (\text{Eq. 10})$$

**Proof sketch** (Appendix L.1):
- 用 standard conditional risk decomposition: $\mathbb{E}[\|Y - h(X)\|_2^2 \mid X] = \mathbb{E}[\|Y - m(X)\|_2^2 \mid X] + \|m(X) - h(X)\|_2^2$
- Bayes-optimal predictor 是 conditional mean $m(X) = \mathbb{E}[Y \mid X]$
- 用 Definition 1 的 conditional equivalence，把 $Z_T^{(-i)}$ 替换成 $\mathcal{N}_t(i)$

这是一个 fairly straightforward 的结果，但概念上重要：它说明**任何** optimal predictor 必须用 $\mathcal{N}_t(i)$ 里的信息，否则 risk 严格更高。

### 5.4 Corollary 1: Discovery of Intervention-Stable Influence Neighborhoods

优化 $\mathcal{L}_{\text{mask}}$ 鼓励 attention 集中在 $\mathcal{N}_t(i)$。这和 Invariant Causal Prediction (ICP, Peters et al., 2016, https://doi.org/10.1111/rssb.12167) 和 IRM (Arjovsky et al., 2020, https://arxiv.org/abs/1907.02893) 类比——但 C-JEPA 不需要 multiple environments，而是在 single dataset 内通过 masking 模拟 interventions。

### 5.5 我对理论的几点 intuition

1. **Influence neighborhood ≠ causal parents**。作者明确说 (Appendix B.2): $\mathcal{N}_t(i)$ 可能包含 causally downstream 的 variables, correlated through latent confounders, or otherwise informative。这是务实的——真实系统有 confounders，full causal discovery 不可行 (Seitzer et al., 2021, https://arxiv.org/abs/2011.07793)。

2. **Direction-agnostic abstraction (Remark 3)**。训练用 bidirectional attention，所以 $\mathcal{N}_t(i)$ 没有方向性。但 forward dynamics 用的是同一个 transition mechanism (Assumption 2)，所以学到的 interaction structure 在 forward prediction 下依然有效。这其实是一个 strong claim，paper 没有完全严格证明，依赖于 mechanism invariance。

3. **和 counterfactual reasoning 的连接**。Object-level masking 切断观察 = counterfactual-like query: "如果 B 的轨迹 unknown (但物理不变)，A 的运动如何 imply B 的状态？" 这种 query 在训练时反复出现，predictor 学到的 representation 自然 align 到 counterfactual 推理所需的信息结构。这解释了为什么 counterfactual VQA 提升最大 (+21%) 而不是 descriptive (+3%)。

---

## 6. Build Intuition: 几个关键 design choice 的 why

### 6.1 为什么 object-level masking 比 patch-level 好?

Patch-level MAE 学的是 local texture correlation。Object-level masking 学的是 entity-level interaction。在 Push-T 上，C-JEPA 用 1% 的 features 追平 patch-based DINO-WM——这说明 object 是物理正确的 abstraction level。Patch 是 convenience，object 是 ontology。

### 6.2 为什么需要 identity anchor?

没有 anchor，permutation-equivariance 让 predictor 不知道在预测哪个 object。Anchor 携带 identity（"我是球 B"），mask embedding 携带 temporal position ("我在 t=3 时刻")。两者组合 = "球 B 在 t=3 的状态是什么" 的 query。这是 set-based prediction 的必然要求。

### 6.3 为什么 bidirectional attention?

Interaction 是 non-local in time: A 在 $t=2$ 撞 B，B 在 $t=5$ 撞 C。Autoregressive 必须按顺序生成，可能丢失 long-range interaction。Bidirectional 让 predictor joint infer 所有 masked tokens，attention 可以 attend 任意 token，捕捉 interaction graph 的任意结构。

### 6.4 为什么训练时 mask，推理时不 mask?

训练时 masking = data augmentation via latent intervention。模型被迫学 interaction structure。推理时不需要 intervention，只需 forward prediction。学到的 $\mathcal{N}_t(i)$ 在 forward dynamics 下依然有效——这是 Assumption 2 (shared transition) 的功劳。

### 6.5 和 LeCun 的宏大 vision 的关系

LeCun 2022 paper (https://openreview.net/pdf?id=4cYrrCbsyq) 提出的 JEPA 是 hierarchical: observer → world model → actor + critic。C-JEPA 处于中间层——world model 的 latent predictor。它学到的 object-centric interaction structure 可以为上层 actor-critic 提供 predictive substrate。V-JEPA2 已经展示了 planning 能力，C-JEPA 给出了 object-centric 的更 efficient 版本。

### 6.6 和你 (Karpathy) 的 nanoGPT / 教学角度的联系

你会注意到这本质上是 "BERT for object dynamics"——masked completion 训练，forward 推理。和 GPT 的 autoregressive 不同，BERT-style 训练但 GPT-style 推理 (forward only)。这种 train-test mismatch 在 NLP 里很少见 (除了 ELMO 之类的)，但在 world model 里很自然——因为物理 transition mechanism 是固定的，bidirectional training 学到的是 invariant interaction structure。

如果让我用你的 teaching style 解释: 想象一个学生复习物理题。Masked completion = "给你球的初始位置 + A 的完整轨迹 + 物理规律，预测 B 在中间时刻的位置"。Forward prediction = "给所有当前状态，预测未来"。前者强迫学生学牛顿第三定律 (interaction)，后者只需要运动学。C-JEPA 的训练就是让学生反复做前者，考试做后者——但学到的 interaction law 在两种任务下都适用。

### 6.7 Limitations 我看到

1. **Encoder-dependent ceiling**: Table 1 显示 VideoSAUR 在 mask=4 还在涨 (89.4%), SAVi 在 mask=4 已经崩 (73.28%)。这说明 object alignment 是 bottleneck。未来需要 jointly refine encoder。
2. **Hungarian matching 在 inference**: Push-T 用 Hungarian matching align predicted vs goal slots——这是工程 hack，理论上应该 emerge slot consistency。
3. **Theory 是 sufficient not necessary**: Theorem 1 说 optimal predictor must use $\mathcal{N}_t(i)$，但没说 attention pattern 一定 align。Corollary 1 是 informal argument。
4. **没在 real-world video 上测**: CLEVRER 和 Push-T 都是 synthetic。Object-centric encoder 在 real video 上质量未知。
5. **Bidirectional training → forward inference 的 transfer 是 hand-wavy**: Remark 3 的论证依赖 Assumption 2, 但 Assumption 2 是 strong stationarity，real world 不一定满足。

### 6.8 一些可能的延伸联想

- **Multi-agent**: object = agent, masking = "agent i 的 internal state unknown"。这和 theory of mind 有 connection。
- **Counterfactual video generation**: C-JEPA 的 latent intervention 可以用来 generate "what if object B wasn't there" 的 counterfactual video。
- **Causal discovery**: influence neighborhood 是 soft causal graph。可以用来做 causal discovery on top of learned world model。
- **Compositionality**: object-centric + masking = compositional generalization。如果训练时见过 (A, B) 和 (B, C) interaction，能 generalize 到 (A, C) 吗？
- **Connection to your work on "Software 2.0"**: C-JEPA 的 object-level masking 是一种 "inductive bias programming"——通过 objective design 而非代码实现 causal structure。

---

## 7. 总结

C-JEPA 的核心贡献:
1. **Object-level masking 作为 latent intervention**: 把 masking 从 patch 提升到 object，induce counterfactual-like queries
2. **JEPA × Object-centric 的首次整合**: 让 LeCun 的 JEPA 路线和 object-centric representation 终于相遇
3. **Decoder-free + 1% features**: 实现了 efficient world modeling，8.6× faster MPC
4. **Causal inductive bias 通过 objective**: 不需要 architectural factorization，让 interaction 成为 functionally necessary

代码: https://github.com/galilai-group/cjepa

我的整体感觉: 这篇 paper 设计很 clean，ablation 严谨 (OC-JEPA vs C-JEPA 是 perfect controlled experiment)，理论虽然不复杂但 framing 正确。最大的 open question 是 real-world generalization 和 encoder 的 jointly refinement。如果未来能看到 C-JEPA 在 real robot manipulation 或 autonomous driving 上 work，那就 really exciting 了。

如果你想深挖某一块 (比如 VideoSAUR 的具体 attention 机制, CEM 的实现细节, 或 influence neighborhood 和 Pearl's do-calculus 的精确关系), 告诉我，我可以再展开。
