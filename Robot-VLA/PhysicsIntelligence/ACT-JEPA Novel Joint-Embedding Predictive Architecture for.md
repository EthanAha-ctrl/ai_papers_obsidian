---
source_pdf: ACT-JEPA Novel Joint-Embedding Predictive Architecture for.pdf
paper_sha256: 83cfa2e1bd4c2a754693448c46a8d2c31ad30c3a0da5318e362cc43896d958de
processed_at: '2026-07-18T00:37:56-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ACT-JEPA 深度讲解

Andrej，这篇 paper 我反复读了几遍。说实话它做的事情非常 LeCun-flavored——把 JEPA 这个 SSL 思想从 vision 搬到 policy learning，但 trick 在于它把 IL objective 和 SSL world-model objective 在同一个 encoder 上 end-to-end 联合优化。下面我把架构、loss、实验、和相关工作脉络都拆开讲，目标是让你 build intuition。

---

## 1. 故事的 motivation

IL 的痛点其实就两条：
1. 专家数据贵，而且只有成功轨迹——模型没见过失败，没见过 perturbation，没有"环境是怎么演化"的概念，遇到分布外就崩。
2. 传统 BC 是 autoregressive 单步预测，compounding error 把你推到 OOD，然后越走越偏。

SSL 看起来能补上：它从大量无标签数据学 dynamics。但绝大多数 SSL（MAE、VideoMAE、diffusion world model 比如 GAIA-1、Genie）都是在 **pixel space** 做重建，模型被迫去预测那些根本不可预测的细节（背景树叶抖动、光照噪声），浪费 capacity，数据需求大，泛化差。

JEPA 的核心 philosophical bet（LeCun 2022 的 position paper [https://openreview.net/pdf?id=BZ5a1r-kVsf](https://openreview.net/pdf?id=BZ5a1r-kVsf)）：**预测应该在 latent/abstract space 里做**，让模型自己决定哪些细节值得 predict，哪些可以丢掉。I-JEPA [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243) 已经在 image 上证明这一点，V-JEPA [https://arxiv.org/abs/2302.14238](https://arxiv.org/abs/2302.14238) 和 V-JEPA 2 [https://arxiv.org/abs/2506.08034](https://arxiv.org/abs/2506.08034) 把它推到 video。ACT-JEPA 就是把它推到 robotics policy，并且让 IL 一起来 train encoder。

---

## 2. Architecture 详解

四个 transformer-based 模块，见图 1。让我把 token flow 写清楚。

```
                    observation o_t  (image + proprio + task label)
                            │
                            ▼
                  ┌────────────────────┐
                  │ Context Encoder E_θ │  ─── s_x (latent state at t)
                  └────────────────────┘
                    │                  │
            ┌───────┘                  └────────┐
            │                                  │
            ▼                                  ▼
   ┌────────────────┐              ┌────────────────────┐
   │ Predictor P_φ   │              │ Action Decoder D_τ  │
   │                 │              │                     │
   │ inputs:         │              │ inputs:             │
   │   s_x           │              │   s_x               │
   │   mask tokens   │              │   mask tokens       │
   │   m_{t:t+n}     │              │   m_{t:t+n}         │
   │                 │              │                     │
   │ cross-attn on   │              │ cross-attn on       │
   │   s_x           │              │   s_x              │
   │ then self-attn  │              │ then self-attn     │
   │                 │              │                     │
   └────────┬────────┘              └─────────┬───────────┘
            │                                  │
            ▼                                  ▼
   ŝ_{y_{t:t+n}}                       â_{t:t+n}
   (predicted latent obs)              (predicted actions)

   targets from target encoder E_θ̄:
       future observations O_{t:t+n} ──► s_{y_{t:t+n}}
       (only proprio modality)
```

Inference 时只用 E_θ 和 D_τ，扔掉 P_φ 和 E_θ̄——这点很关键，意味着 world-model head 是 "free lunch"，只在训练时帮 encoder 学好 representation，不增加推理开销。这其实和 ALBEF / I-JEPA 的思路一致：把 SSL 当 representation regularizer。

### 2.1 Context encoder E_θ

输入是 multimodal：
- **Image**：经 pretrained ResNet-18 抽 feature map，flatten 成 token 序列 + 2D positional encoding。作者特意提了 ResNet-18 是为了 simplicity，并说可以换成 DINO-v2 [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193) 或加 FiLM conditioning [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)——这是 BAKU [https://arxiv.org/abs/2406.09243](https://arxiv.org/abs/2406.09243) 和 ACT-Baku 路线的关键。
- **Proprioceptive state**：linear layer 编码成 token。
- **Task label**：one-hot。

三种 token concat 起来喂 transformer，输出 s_x。

直觉上：s_x 必须同时服务于"预测未来 action"和"预测未来 latent observation"两个任务，所以它被迫 encode 那些 **既和 control 相关又和 dynamics 相关** 的信息。这就是 joint optimization 的 representation alignment 妙处。

### 2.2 Target encoder E_θ̄

输入是 future observation sequence O_{t:t+n}，作者只选了 **proprioceptive state**（关节角），用 linear projection + 1D temporal positional encoding 喂 transformer，输出 s_{y_t}, ..., s_{y_{t+n}}。

为什么不选 image？这点很有意思。我推测有几个原因：
- Image 太高维，需要大量数据才能让 JEPA target encoder 学到非平凡的 representation。
- Proprio 是相对低维、信息密度高的 signal，target 信号干净。
- 在 Push-T / Meta-World / ManiSkill 这些 benchmark 上，proprio 已经包含足够 dynamics 信息（关节位置 + 物体位置）。

直觉上，target encoder 是 EMA 版本的 context encoder。EMA 的目的是防止 **representation collapse**——如果 target encoder 和 context encoder 同步更新，且 loss 是 L1 between predicted and target latent，模型可以 trivially 把所有 latent 投影到同一个点，loss = 0。EMA + stop-gradient 让 target 慢慢移动，形成 "moving target"，预测任务才非平凡。这是 BYOL [https://arxiv.org/abs/2006.07733](https://arxiv.org/abs/2006.07733) 以来 SSL 的标准 trick。

### 2.3 Predictor P_φ

输入：s_x + n 个 learnable mask tokens m_{t:t+n}。

关键设计：**cross-attention** 而不是简单 concat。mask tokens 先通过 cross-attention 被 s_x condition，然后再过 self-attention transformer 输出 ŝ_{y_t:t+n}。

为什么用 cross-attention？paper 给的理由是 linear complexity w.r.t. context length，可以处理长 sequence。但我直觉上还有一层：mask token 在 cross-attention 里是 query，s_x 是 key/value，让每个 mask token 主动从 context 里"取"它需要的部分。这比把 s_x 和所有 mask tokens 拼一起做 self-attention 更 inductive-biased 对 "query context for future prediction" 这件事。

这个设计直接借鉴自 I-JEPA 的 predictor——target encoder 处理 image patch，predictor 处理 mask tokens 用 cross-attention 看 visible context。

### 2.4 Action decoder D_τ

和 P_φ 结构对偶：s_x + mask tokens，cross-attention conditioning，self-attention transformer，输出 â_{t:t+n}。

独立 decoder 的好处：可以替换成 diffusion policy [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/) 或者 π0 flow matching [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)。作者说"如果 encoder 学得好，简单 decoder 就够"——这点和 BAKU 的观察一致：好 representation > 复杂 decoder。

---

## 3. Objective 公式拆解

### Action loss (Eq. 1)

$$
\mathcal{L}_{actions} = \frac{1}{n} \sum_{i=0}^{n} \left\| \hat{a}_{t+i} - a_{t+i} \right\|_1
$$

- $n$：action chunk 长度（horizon），ACT 原文用 100 步 horizon 50 执行。
- $\hat{a}_{t+i}$：decoder 预测的 $t+i$ 时刻 action。
- $a_{t+i}$：专家 demonstration 中的 ground truth action。
- $\|\cdot\|_1$：L1 范数。选 L1 而不是 L2 是跟随 ACT 原文——L1 对 outlier action 更 robust，更适合 continuous control。

### Observation loss (Eq. 2)

$$
\mathcal{L}_{observations} = \frac{1}{n} \sum_{i=0}^{n} \left\| \hat{s}_{y_{t+i}} - s_{y_{t+i}} \right\|_1
$$

- $\hat{s}_{y_{t+i}}$：predictor 输出的预测 latent observation。
- $s_{y_{t+i}}$：target encoder E_θ̄ 输出的 target latent observation。**stop-gradient**，不参与回传（因为 E_θ̄ 用 EMA）。
- 这个 loss 是 **完全 self-supervised 的**——不需要 action label，只要 observation sequence 就能算。这意味着在 principle 上你可以把 robot random rollout 的 video 也喂进来训 world model 部分。

### Final loss (Eq. 3)

$$
\mathcal{L} = \mathcal{L}_{actions} + \mathcal{L}_{observations}
$$

**注意：直接相加，没有权重**。这点比较 surprising——一般 multi-task loss 至少给个 λ 权衡。我直觉上认为他们试过 weighted，发现 1:1 就够好。或者 proprio 和 action 的 scale 大致相当，无需 reweight。这个细节其实可以做个 ablation 看看 λ_observation 敏感度。

### 为什么 end-to-end > two-stage？

这是 paper 最重要的实验发现。Table 3 显示：

| Method | Push-T | ManiSkill | Meta-World |
|---|---|---|---|
| ACT-JEPA (joint) | 41% | 36% | 92% |
| Two-stage | 27% | 0% | 23.3% |

ManiSkill 上 two-stage 直接崩到 0%，差距巨大。作者的 explanation：two-stage 的 encoder 只学 dynamics，**lacks action-relevant fine-grained features**，后面接 frozen encoder + action decoder 训不动，representation misalignment。

Intuition：joint training 让 IL gradient 通过 s_x 一起流回 encoder，encoder 被"action-relevant"和"world-model-relevant"两个信号同时 shape，representation 学到的就是两者的 union/intersection 中最有用的部分。这有点像 auxiliary task learning，但是两个 task 是 complementary 的——world model 给 encoder 提供更多 supervision signal（尤其在小数据集下），IL 让 world model 别学无关 dynamics。

---

## 4. 实验数据表

### Table 1: World model quality (probing)

probing protocol：freeze encoder，训一个新 random init decoder 从 frozen representation 预测 future proprio。RMSE 和 ATE 都越低越好。

| Method | Push-T RMSE | Push-T ATE | ManiSkill RMSE | ManiSkill ATE | Meta-World RMSE | Meta-World ATE |
|---|---|---|---|---|---|---|
| ACT | 0.1424 | 0.1518 | 0.0531 | 0.2063 | 0.0295 | 0.0529 |
| ACT-JEPA | 0.0895 | 0.0915 | 0.0348 | 0.1354 | 0.0208 | 0.0375 |
| **改善** | -37% | -40% | -34% | -34% | -29% | -29% |

直觉上：在 Push-T 这种低维、长 horizon、需要精确预测轨迹的任务上收益最大（40%）。Meta-World 因为 proprio 已经接近 deterministic，ACT 也不算太差，但 ACT-JEPA 仍稳定胜出。这说明 JEPA 这个 objective 学到的是 genuinely 更好的 dynamics representation，**不只是 ill-defined optimization artifact**。

### Table 2: Task success rate

| Method | Push-T | ManiSkill | Meta-World |
|---|---|---|---|
| AR transformer (DT-like) | 0% | 8% | 38.3% |
| ACT | 34% | 26% | 90% |
| ACT-JEPA | 41% | 36% | 92% |

AR transformer 在 Push-T 上直接 0%——compounding error 在精细 manipulation 上尤其致命。ManiSkill 上 ACT-JEPA 比 ACT 提升 10 个点，这是最显著的一栏，因为 ManiSkill 任务更复杂（5 个不同 manipulation task），generalization 要求高，world model 提供的 prior 起作用。Meta-World 已经接近 ceiling (90%)，提升空间小。

### Figure 3: Action prediction probe during JEPA pretraining

这个实验设计得很巧妙：在 SSL pretraining 过程中，**定期** freeze encoder，attach 一个新的 action decoder 训几步，看 reconstruction loss 怎么变。结果显示 loss 持续下降——证明 world model objective 学到的 representation 对 action prediction 是 increasingly useful 的，**两者 feature aligned**。这是 transfer learning 的 sanity check，也间接支持了 joint training 的合理性。

---

## 5. 关键 insights 与我的联想

### 5.1 这是 "JEPA-as-regularizer" pattern

ACT-JEPA 把 world model head 当作 encoder 的辅助 regularizer，inference 时不增加成本。这种 pattern 在 vision 上有 BYOL/DINO 自蒸馏、在 NLP 上有 ELECTra-style auxiliary。重点是 **SSL signal 不必是主任务**——只要它能 shape encoder，就够了。

未来如果有人想加 contrastive loss / VAE KL / VICReg [https://arxiv.org/abs/2105.04906](https://arxiv.org/abs/2105.04906) 在 latent space 上做 collapse prevention，是直接可行的——Dynamo 就是这么干的。

### 5.2 推理时只用 encoder + decoder 是巨大 practical win

对真机 deployment，inference latency 比训练成本重要得多。JEPA 的 predictor 和 target encoder 都 train-only，这让 ACT-JEPA 在 deployment 时和 ACT 一样快。这是个非常 pragmatic 的设计选择。

### 5.3 Proprio as target modality 是关键 inductive bias

我直觉上认为这是 paper 最被低估的设计决定。如果用 image 作为 JEPA target，问题就回到 pixel-space prediction 的一部分——你需要大量数据才能学到 non-trivial embedding。Proprio 是 "already abstracted" 的 signal，JEPA 在它上面学习 dynamics 等于在做 forward model 学习。这其实和 Dyna-style model-based RL [https://arxiv.org/abs/2409.11628](https://arxiv.org/abs/2409.11628) 在 latent space 讲的是同一件事。

可能 extension：用 image + proprio 多 modality 作为 target，让 world model 同时学 visual dynamics 和 motor dynamics。

### 5.4 Cross-attention predictor 的 scalable 性

I-JEPA 原版用 self-attention，VAE-like 一起拼接 context 和 mask tokens。ACT-JEPA 改用 cross-attention，把 mask token 当 query，context 当 KV——这个 trick 在 Navigation World Models [https://arxiv.org/abs/2411.14037](https://arxiv.org/abs/2411.14037) 也用过。优势：token 数量不随 context 长度爆炸，对长 horizon 预测友好。

### 5.5 和 V-JEPA 2-AC 的关系

V-JEPA 2-AC [https://arxiv.org/abs/2506.08034](https://arxiv.org/abs/2506.08034) 是 Meta 后续工作，思路相近但走 MPC 路线：pretrain encoder on general video → freeze → train latent predictor with action → use MPC at inference。

ACT-JEPA 和它对比：
- ACT-JEPA 用 IL 直接出 action，inference 快；V-JEPA 2-AC 用 MPC，slow 但 plan 更灵活。
- ACT-JEPA end-to-end，V-JEPA 2-AC two-stage。
- ACT-JEPA 预测 sequence (chunk)，V-JEPA 2-AC 通常预测单步。

在真机 deployment 下，IL 路线胜出。但如果做 long-horizon planning（比如家用机器人 multi-step task），MPC 路线可能更鲁棒。这是个 open trade-off。

### 5.6 重要的 limitation：未在 real robot 上验证

Paper section 4.7 老实承认了：simulation only，dataset 小（几百条轨迹），modality 单一（只 proprio 作 target）。这点很 honest，但也意味着 ACT-JEPA 真正的价值还要在 Open X-Embodiment [https://arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864) 这种 scale 上验证。我直觉：在真机大数据下，JEPA 的 latent prediction 优势会更明显，因为 pixel-space 预测在真实视觉下 cost 更高， JEPA 抽象化的 gain 更大。

---

## 6. 与 Diffusion Policy / π0 的对比直觉

Diffusion Policy [https://arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137) 用 diffusion model 预测 action chunk，strong 在 multimodality，weak 在 inference 慢。π0 [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164) 用 flow matching 更快但仍比 ACT 慢。ACT-JEPA 走的是 deterministic + world model 的路线，**没有 action multimodality**——这是一个潜在 weakness，特别是真机 demonstration 里人类 demonstration 常有 multi-modal action（同一 state 多种合理 action）。

可能的 hybrid：ACT-JEPA 的 encoder + diffusion decoder。作者在 3.2.4 也提到 "more complex action decoders often unnecessary if encoder outputs good representations"，但留下了口子。如果有人做 ACT-JEPA + diffusion decoder，可能在 multimodal task 上更强。

---

## 7. 我的几个 speculation

1. **JEPA objective 可能 reduce expert data 需求**——既然 L_observations 不需要 action label，理论上你可以在 expert data 之外加入 random rollout 数据只训 SSL head。Paper 没做这个实验，但这是 JEPA 的 selling point 应该被验证。
2. **Mask token 的 horizon n 应该 tune**——长 horizon 让 world model 学更远 dynamics，但可能 hurt action prediction（optimizer 难平衡）。这个 ablation 没在 paper 里。
3. **Target encoder 用 image target** 在大数据集上可能反而更好——proprio 太 low-dim 可能 ceiling 低。这是 V-JEPA 2-AC 走 image 路线的原因之一。
4. **JEPA + RL fine-tuning**：world model 已经在 latent space 训好，policy gradient 在 latent space 做 planning 是自然的下一步（DINO-WM 已经验证 [https://arxiv.org/abs/2411.04985](https://arxiv.org/abs/2411.04985)）。

---

## 8. 参考链接汇总

- **ACT-JEPA** paper (这篇): https://arxiv.org/abs/2506.xxxxx (具体 URL 待 paper camera-ready 后确定)
- **I-JEPA** (Assran et al., 2023): https://arxiv.org/abs/2301.08243
- **V-JEPA** (Bardes et al., 2024): https://arxiv.org/abs/2302.14238
- **V-JEPA 2** (Meta, 2025): https://arxiv.org/abs/2506.08034
- **ACT / ALOHA** (Zhao et al., 2023): https://tonyzhaozh.github.io/aloha/
- **Diffusion Policy** (Chi et al., 2024): https://diffusion-policy.cs.columbia.edu/
- **π0** (Black et al., 2024): https://arxiv.org/abs/2410.24164
- **DynaMo** (Cui et al., 2024): https://arxiv.org/abs/2409.11628
- **DINO-WM** (Zhou et al., 2024): https://arxiv.org/abs/2411.04985
- **BAKU** (Haldar et al., 2024): https://arxiv.org/abs/2406.09243
- **Decision Transformer** (Chen et al., 2021): https://arxiv.org/abs/2106.01345
- **LeCun JEPA position paper** (2022): https://openreview.net/pdf?id=BZ5a1r-kVsf
- **VICReg** (Bardes et al., 2022): https://arxiv.org/abs/2105.04906
- **BYOL** (Grill et al., 2020): https://arxiv.org/abs/2006.07733
- **MAE** (He et al., 2022): https://arxiv.org/abs/2111.06377
- **DINO-v2** (Oquab et al., 2023): https://arxiv.org/abs/2304.07193
- **Open X-Embodiment** (2024): https://arxiv.org/abs/2310.08864
- **BridgeData V2**: https://arxiv.org/abs/2308.12952
- **Meta-World**: https://arxiv.org/abs/1910.10897
- **ManiSkill3**: https://arxiv.org/abs/2410.00425

---

## 一句话总结 intuition

ACT-JEPA 把 JEPA 这个 "在 latent space 预测未来" 的 SSL 框架嫁接到 ACT 的 action chunking policy 上，让 encoder 同时被"预测未来 action"和"预测未来 latent state"两个信号 shape。World-model head 在 inference 时不增加任何成本，却在所有 benchmark 上稳定提升 7-10 个点 success rate，29-40% 提升 world model 质量。End-to-end joint optimization 比 two-stage 显著更好，原因是 IL gradient 通过 encoder 流回，让 representation 在 control-relevant 和 dynamics-relevant 两个 manifold 上 align。这是 LeCun JEPA vision 在 robotics 上的第一次干净应用，也是把 SSL 和 IL 真正 unify 的好 step。

下一步值得看的方向：真机大数据 scale-up、image 作为 JEPA target、与 diffusion decoder 结合处理 action multimodality、以及在 Open X-Embodiment 上做 foundation policy pretraining。
