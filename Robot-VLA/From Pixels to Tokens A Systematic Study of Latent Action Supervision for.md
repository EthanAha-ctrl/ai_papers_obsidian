---
source_pdf: From Pixels to Tokens A Systematic Study of Latent Action Supervision
  for.pdf
paper_sha256: 73fb825a1a3ba89258e94fa77a9fb3c6b66e7e1933538506dea4ba3d4165a65a
processed_at: '2026-08-04T10:49:51-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 一句话说清楚这篇在干嘛

你手头有一堆机器人数据，有的来自单臂 JAKA，有的来自双臂 RoboTwin，有的甚至是人手操作视频 (Ego4D)。这些数据的 action space 完全不一样——7维、14维、26维混在一起，VLA model 训起来会互相打架 (negative transfer)。

大家想了个办法：先学一个 latent action model，把 raw actions 或 visual transitions 压缩成一串 discrete tokens，然后用这些 tokens 当 supervision signal 去训 VLA。问题是，community 里各家做法五花八门，有人用 image transitions 学 latent，有人直接 tokenize actions，有人让 VLM predict latent tokens，有人只做 feature alignment。谁也说不清哪种好、为什么好、什么时候该用哪种。

这篇 paper 做的事情很简单：**搭一个统一的 baseline，把所有这些乱七八糟的做法归到同一个 framework 下，然后 head-to-head 比。**

## 核心直觉：两个 Perspective

作者把所有方法归成两大类，划分标准是**信息流的 mapping direction**。

### Perspective 1: Image-based Latent Actions — "看图说话"

从 vision 到 action 的 forward direction。你给 VLM 两帧图 $o_t$ 和 $o_{t+\delta}$，让它学会预测"这两帧之间发生了什么动作"。这个预测出来的东西就是 image-based latent action。

直觉是这样：想象你教小孩开车，你给他看一段行车记录仪，让他描述"刚才方向盘转了多少、油门踩了多深"。小孩不需要真的摸方向盘，光看画面就能推断出动作的 abstract pattern。这就是 image-based latent action 在做的事情——它捕捉的是 **visual state transition**，抽象掉了具体的 motor commands。

对应到公式 (Eq 1)：
$$c_t^{img} = E_\theta([o_t; o_{t+\delta}])$$

- $o_t$: 起始帧 observation
- $o_{t+\delta}$: $\delta=32$ 步之后的结束帧
- $E_\theta$: encoder，基于 frozen DINOv2 + Transformer
- $c_t^{img} \in \mathbb{R}^d$: continuous embedding，$d=128$

然后做 vector quantization：
$$z_t^{img} = VQ(c_t^{img}), \quad z_t^{img} \in \{1, \ldots, K_{img}\}^P$$

- $K_{img}=16$: codebook 只有 16 个 entry，非常 compact
- $P=4$: 每个 timestep 对应 4 个 discrete tokens

### Perspective 2: Action-based Latent Actions — "把动作变成语言"

从 action 到 token 的 reverse direction。你直接拿 raw action chunk $\mathsf{a}_t$，训一个 VQ-VAE 把它 tokenize 成 discrete tokens。

直觉：不同机器人的 action dimensionality 不同，但如果你把所有 actions 都映射到一个 shared codebook，VLM 看到的就都是统一的 token sequence。就像不管你讲英语还是中文，翻译成世界语之后大家都能读。

对应公式 (Eq 2)：
$$c_t^{act} = E_\phi(\mathsf{a}_t), \quad c_t^{act} \in \mathbb{R}^{H \times d}$$

- $\mathsf{a}_t \in \mathbb{R}^{H \times m}$: action chunk，$H$ 是 chunk size，$m$ 是 action dimension
- $E_\phi$: 作者自研的 encoder
- $z_t^{act} = VQ(c_t^{act}), \quad z_t^{act} \in \{1, \ldots, K_{act}\}^H$
- $K_{act}=256$: action codebook 比 image 的大得多

**两种 perspective 的本质区别**：image-based latent action 编码的是"场景应该怎么变"（high-level plan），action-based latent action 编码的是"电机该怎么动"（low-level motor primitive）。前者更 abstract 更 scene-invariant，后者更 concrete 更 motor-specific。

## Action-based Latent Action Model 的架构细节

这部分是 paper 的一个 technical contribution，设计得挺巧。作者没有直接用一个简单 MLP 做 encoder，而是结合了 frequency domain 和 time domain。

**为什么这么设计？** Action chunk 里有两类信息：
- **Low-frequency trend**: 整体轨迹的大方向，比如机械臂从左移到右
- **High-frequency detail**: 瞬间的微小调整，比如 grasping 那一刻的 fine motor control

FFT 擅长捕捉前者，1D temporal convolution 擅长捕捉后者。两者 concatenate 起来，encoder 就能同时看到 coarse 和 fine 的 action dynamics。

具体流程 (Appendix A.2)：
1. 对 $\mathsf{a}_t$ 做 FFT，取 real 和 imaginary parts concatenate
2. Linear projection 到 $d=128$
3. 3层 1D temporal conv，kernel size=3，dilation rates $\{1, 2, 4\}$（指数增长 receptive field）
4. 2层 Transformer encoder，4 heads，feedforward dim=256
5. VQ quantization 到 codebook ($K_{act}=256$)

Decoder 很轻量：per-timestep MLP (128→256→$m$)。

**Latent Consistency Loss** (Eq 18) 值得展开讲：
$$\mathcal{L}_{mask} = \sum_{h \in \mathcal{M}} \|E_\phi(\tilde{\mathsf{a}}_t)_h - E_\phi(\mathsf{a}_t^*)_h\|_2^2$$

- $\mathcal{M}$: masked timestep indices，mask ratio=0.15
- $\tilde{\mathsf{a}}_t$: 从 masked latent sequence decode 再 re-encode 的 action
- 这个 loss 强制 latent representation 在 partial information 下保持稳定

直觉：这跟 MAE ([Masked Autoencoder](https://arxiv.org/abs/2111.06377)) 的思想一样。如果你真的学到了 action 的本质结构，那即使缺了几个 timestep 的信息，你也能 infer 出来。这个 regularization 让 tokenization 更 stable，downstream VLA 训练时不会因为 latent representation 抖动而 degrade。

## 四种 Integration Strategy 的人话版

作者在 unified baseline 上 instantiate 了四种策略，核心区别在于**latent action 怎么 inject 到 VLM 里**。

### Strategy 1: LA-Align — "潜移默化"

最轻量的做法。不增加任何 placeholder，只在 VLM 的某个中间层（layer 17 out of 29）加一个 auxiliary loss，让那个层的 hidden state 跟 ground-truth latent embedding 做 cosine similarity alignment。

公式 (Eq 5)：
$$\mathcal{L}_{latent} = -\mathbb{E}_t[S(\phi_{align}(\nu_t^{(k)}), c_t^{img})]$$

- $\nu_t^{(k)}$: layer $k$ 的 placeholder hidden state
- $\phi_{align}$: linear projection 从 $d_{hidden}$ 到 latent space ($\mathbb{R}^{128}$)
- $S(\cdot, \cdot)$: cosine similarity
- $c_t^{img}$: ground-truth continuous latent embedding

**直觉**：你跟 VLM 说"我不要求你输出特定 tokens，但你的 internal representation 要在语义上跟 latent action 对齐"。这是一种软约束，VLM 自己决定怎么 internalize 这个信号。

**缺点**：约束太弱，VLM 可能学到 shortcut——比如 hidden state 表面上 aligned 但实际上没有真正理解 planning。

### Strategy 2: LA-Direct — "直接告诉我你的 plan"

用 $P \times H$ 个 placeholder tokens 直接 predict discrete latent action tokens。VLM 的输出空间被显式约束成 codebook 上的 probability distribution。

公式 (Eq 6, 7)：
$$\pi_{vlm}(z_{t:t+H-1}^{img} | o_t, \ell) = Softmax(\phi_{explicit}(\nu_t^{(N_{layer})}))$$
$$\mathcal{L}_{latent} = -\sum_{h=0}^{H-1} \log \pi_{vlm}(z_{t+h}^{img*} | o_t, \ell)$$

- $\phi_{explicit}$: linear projection 到 codebook logits ($K_{img}=16$)
- $\nu_t^{(N_{layer})}$: final-layer placeholder states
- $z_{t+h}^{img*}$: ground-truth latent action token at position $h$

然后这些 latent placeholder 的 representation 直接 feed 给 action head：
$$\hat{a}_{t:t+H-1} = f_{head}([h_t^{img}; h_t^{latent}; s_t])$$

**直觉**：VLM 同时干两件事——predict discrete latent tokens (显式 planning) 和 provide action representation。这两个任务 share 同一个 backbone 但通过不同 head 输出。latent tokens 的 prediction 像一个 auxiliary task，regularize backbone 学到更 structured 的 representation。

### Strategy 3: LA-Cond — "先想后做"

这是 LA-Direct 的升级版。Placeholders 分成两段：
- **Latent segment**: $P \times H$ 个 tokens，predict $z_t^{img}$（high-level plan）
- **Action segment**: $H$ 个 tokens，form action representation（low-level execution）

关键设计：通过 causal attention mask，action segment 可以 attend to latent segment，但 latent segment 不能 attend to action segment。这就实现了"先 plan 后 execute"的 hierarchy。

Action head 的 input (Eq 9)：
$$\hat{a}_{t:t+H-1} = f_{head}([h_t^{img}; h_t^{latent}; h_t^{act}; s_t])$$

比 LA-Direct 多了 $h_t^{act}$——专门的 action representation，conditioned on predicted latent plan。

**直觉**：这是 "think before you act" 的 explicit implementation。VLM 先 generate high-level plan (latent tokens)，然后基于这个 plan generate low-level action representation。相当于强制 VLM 走一个 plan → action 的 pipeline。

**LA-Cond vs LA-Direct 的 tradeoff**：
- LA-Direct: latent 和 action 共享 placeholder，implicit conditioning。更简单，但 planning 和 execution 的 boundary 模糊
- LA-Cond: 显式 hierarchical，plan → action 的 causal dependency 明确。但参数更多，training 更难

### Strategy 4: LA-Tok — "把 action 变成 VLM 的母语"

最直接的做法。用 action-based latent action model 把 continuous action chunk tokenize 成 discrete tokens，然后让 VLM 直接 predict 这些 tokens。

公式 (Eq 10)：
$$\pi_{vlm}(z_t^{act} | o_t, \ell) = Softmax(\phi_{tok}(\nu_t^{(N_{layer})}))$$

- $\phi_{tok}$: linear projection 到 action codebook logits ($K_{act}=256$)
- $z_t^{act} \in \{1, \ldots, 256\}^H$: discrete action tokens

**与 LA-Direct 的核心区别**：
- LA-Direct: latent tokens 来自 image transitions，是 visual plan
- LA-Tok: latent tokens 来自 actions 本身，是 motor primitive

LA-Tok 的 intuition：VLM 的 backbone 是在 discrete token space 上 pretrain 的（语言 tokens、image patch tokens）。如果你把 actions 也变成 tokens，VLM 就能用它最擅长的方式去 model actions，不需要额外的 continuous regression head 去 adapt。

## 实验结果的人话总结

### Finding 1: Image-based 擅长 long-horizon，Action-based 擅长 motor control

LIBERO-Long (Table 1)：
- Image-based strategies (LA-Align, LA-Direct, LA-Cond): +8.4% 到 +10.8%
- Action-based (LA-Tok): +6.8%

RoboTwin 2.0 (Table 2)：
- LA-Tok: +17.5%（碾压）
- Image-based: +10% 到 +13.3%

**为什么？** Long-horizon 任务的核心挑战是 multi-stage planning——你需要先想好"先做 A 再做 B 最后做 C"。Image-based latent actions 编码的是 visual state transitions，天然适合做这种 high-level planning supervision。而 RoboTwin 的双臂 coordinated manipulation 核心挑战是 motor precision——你需要精确控制两个手臂的协调时序。Action-based latent actions 把 continuous actions tokenize 成 motor primitives，这种 structured supervision 对学复杂 motor behavior 更有效。

### Finding 2: 直接 predict discrete tokens 最 effective

比较四种 strategy 的 overall performance：
- LA-Align (implicit alignment): 最弱
- LA-Direct (explicit direct): LIBERO 上最强
- LA-Tok (explicit tokenization): RoboTwin 上最强
- LA-Cond (explicit conditional): 居中但最 stable

**为什么 explicit > implicit？** Implicit alignment (cosine similarity) 是一个 bounded loss，数值小，gradient signal 弱。Explicit token prediction (cross-entropy) 是一个 strong supervision——VLM 被强制要求输出 specific tokens，这种 constraint 更 effective 地 shapes backbone representation。

### Finding 3: Discrete tokens > Continuous representations

Table 4 的 ablation：
- LA-Direct (discrete): 97.1% vs LA-Direct (continuous): 94.4% (+2.7%)
- LA-Tok (discrete): 95.5% vs LA-Tok (continuous): 93.3% (+2.2%)

**三个原因**：
1. **Gradient stability**: Cross-entropy 比 MSE 的 gradient 更 stable，不会因为 outlier 产生 huge gradient
2. **Information bottleneck**: Discretization 强制 VLM learn compressed representation，避免 overfitting to noise
3. **Compatibility with pretraining**: VLM backbone 是在 discrete token space 上 pretrain 的，discrete supervision 更 compatible

但注意：continuous variants 仍然 outperform baseline (94.4 vs 93.1)，说明 latent action supervision 本身就有价值，discrete 只是进一步 amplify。

### Finding 4: OOD Generalization — Image-based 碾压

Real-world pick-and-place (Table 3)：
- Baseline OOD average: 17（严重退化）
- LA-Cond OOD: 70（best）
- LA-Direct OOD: 67
- LA-Tok OOD: 50

**为什么 image-based 在 OOD 上强？** Image-based latent actions 编码的是 visual state transitions，这比 raw actions 更 abstract、更 scene-invariant。当 scene configuration 变化时（distractor objects、target layout 改变），visual plan 的 semantics 保持稳定——"把碗放到盘子上"这个 visual transition 不管在什么 scene 下都是一样的。而 motor details 可能需要 adapt——具体的 joint angles 会因为 object position 变化而不同。

### Finding 5: Multi-task Joint Training — LA-Cond 消除 negative transfer

这是最 intriguing 的结果 (Table 11, Fig 5)。

在 10 个 RoboTwin 任务上 joint training：
- **Baseline**: Single-task avg 35.4% → Multi-task avg 53.6%，但 move can pot 从 44% 跌到 13% (-31%)，move playingcard away 从 60% 跌到 49% (-11%)
- **LA-Cond**: Single-task avg 42.7% → Multi-task avg 74.5%，**没有任何 task 退化**

**为什么 latent action supervision 能消除 negative transfer？**

当不同任务 mix 时，action spaces 可能 conflict——有些任务主要用 gripper，有些主要用 wrist。Continuous action regression 会让 VLM 在 conflicting gradients 之间 compromise，导致某些 task 性能退化。

Latent action tokens 提供了一种 task-agnostic 的中间表示。不同任务的 actions 可能不同，但它们的 latent abstractions 可以 share common structure。LA-Cond 的 conditional decoding 进一步加强这个效果：VLM 先 predict latent plan（task-specific but abstract），再基于 plan generate action（concrete but conditioned）。这种 hierarchical decomposition让 multi-task learning 更 stable。

### Finding 6: Data Efficiency

LIBERO-Long, 50% data: LA-Tok 94.0% vs Baseline 80.0% (+14.0%)

Latent actions 作为 compressed supervision signal，显著降低 sample complexity。比起直接 predict high-dimensional continuous actions，predict low-dimensional discrete tokens 是更 easy 的学习任务，VLM 能更快 converge。

### Finding 7: Placeholder Length Ablation — 证明 gains 不是来自更多 tokens

Table 9：
- Baseline: 93.1%
- Baseline (PH-Direct, 同样长度的 placeholder 但无 latent supervision): 91.9% (-1.2%)
- LA-Direct: 97.1% (+4.0%)

这个 ablation 很关键。有人可能质疑：LA-Direct 之所以好，是因为它用了 $P \times H = 32$ 个 placeholder（Baseline 只有 $H=8$ 个），更多 tokens 自然更好。但 Baseline (PH-Direct) 用了同样长度的 placeholder 却 actually degrade。更长 sequence 带来更多 parameters to learn，但没有 inductive bias 的 guidance 反而 hurt。

## Action Head 和 Training 的细节

### Unified Baseline

基于 Qwen3-VL-2B，用 LoRA ($r=64, \alpha=16$) fine-tune 所有 linear layers。VLM 原始 weights frozen，只更新 LoRA parameters 和 action head。

**Input formulation** (Appendix B.1)：
- Visual: primary camera + wrist camera，都 resize 到 224×224
- Language: instruction + suffix "Please predict the next H robot actions: [ACTION]...[ACTION]"
- Proprioceptive state $s_t$: projected to $d_{hidden}$ via MLP

**Placeholder aggregation** (Eq 3)：
$$h_t^{act} = Agg(\{\nu_t^{(k)}\}_{k \in \mathcal{K}_{layers}})$$

- $\mathcal{K}_{layers} = \{\lceil N_{layer}/2 \rceil, \ldots, N_{layer}\}$: 取后半部分层（layer 15-28 for Qwen3-VL-2B 的 29 层）
- $Agg(\cdot)$: 沿 layer dimension stack

**为什么取后半部分层？** Table 5 的 ablation：
- Early layers (align at layer 5): -1.5%（insufficient semantic abstraction）
- Final layer (align at layer 28): 95.0%（specialized for low-level control，too narrow）
- Mid-to-late (layer 17): 97.0%（sweet spot）

### Training Configuration

| Hyperparameter | LIBERO | RoboTwin | JAKA |
|----------------|--------|----------|------|
| Base VLM | Qwen3-VL-2B | Qwen3-VL-2B | Qwen3-VL-2B |
| Training Steps | 80,000 | 30,000 | 30,000 |
| Batch Size | 128 | 128 | 128 |
| Learning Rate | 1e-4 | 1e-4 | 1e-4 |
| Action Chunk Size $H$ | 8 | 25 | 20 |

所有方法用完全相同的 backbone、dataset、optimizer settings，只差 latent action supervision 的方式。这保证了 comparison 的 fairness。

## 这篇 Paper 的深层 Insight

### 1. Latent Action 本质上是 Curriculum

直接 predict continuous actions 是个 hard task——high-dimensional、multi-modal、noisy。Latent action supervision 把这个 task decompose 成两步：
1. 先 predict discrete tokens（easy task，low-dimensional，structured）
2. 再从 tokens decode 出 continuous actions（deterministic mapping）

这相当于一个 implicit curriculum learning。VLM 先学会 high-level structure，再 refine 到 low-level details。这就是为什么 latent action supervision 能提升 data efficiency——它降低了 effective sample complexity。

### 2. Discrete Tokens 是 VLM 的"母语"

VLM 的 backbone 是在 discrete token space 上 pretrain 的——language tokens、image patch tokens、special tokens。当你用 discrete latent action tokens 做 supervision，你在用 VLM 最熟悉的 signal format 去 train 它。这比 continuous regression 更 compatible，gradient signal 更 clean。

Table 4 的 ablation 证实了这点：即使用 continuous latent representations（已经比 raw actions 更 structured），效果还是比 discrete tokens 差 2-3 个百分点。

### 3. Planning 和 Execution 的 Decoupling

LA-Direct > LA-Cond 在大多数任务上，说明在 VLM 内部 decouple planning 和 execution 比 joint modeling 更 effective。

但 LA-Cond 在 multi-task joint training 上更 stable，说明 explicit conditioning 在 heterogeneous task distributions 下有优势——它强制 VLM 走 plan → action 的 pipeline，不同任务的 plan 和 action 可以分别 adapt，不会互相 interfere。

### 4. Formulation-Task Correspondence

这篇 paper 最重要的 empirical finding：**没有 universal best 的 latent action formulation，只有 task-specific 的 best choice。**

- Long-horizon reasoning → image-based latent actions
- Motor control precision → action-based latent actions
- OOD generalization → image-based latent actions
- Multi-task stability → explicit conditional decoding

这给 community 一个 clear guidance：根据你的 downstream task characteristics 选合适的 latent action formulation，而不是盲目追求 SOTA latent action model。

## 局限性和 Open Questions

### Paper 自己承认的
1. 四种 strategies 可能不 cover 所有 possible formulations
2. 没探索 stronger latent representations 是否会 amplify/narrow gaps
3. Real-world 只在单臂 JAKA 上验证

### 我觉得的额外 open questions

**Hybrid formulation**: 能否 combine image-based 和 action-based latent actions？比如先用 image-based 做 high-level planning，再用 action-based 做 low-level execution。Paper 没有 try 这个。

**Codebook size selection**: $K_{img}=16$ vs $K_{act}=256$ 差异巨大。这个 size 怎么 auto-select？太小会 lose information，太大会 collapse 到 continuous case。

**Cross-embodiment generalization**: 在更 diverse embodiments（人形、四足、dexterous hand）上 pattern 是否 hold？Paper 只测了单臂和双臂。

**Inference latency**: Paper 说 single forward pass，但 placeholder 长度从 $H=8$ 增加到 $P \times H = 32$ 会带来 computational overhead。实际部署时 latency 如何？

**Latent action model quality 的影响**: Paper 用了 fixed latent action models。如果用更强的 latent action model（比如更大的 codebook、更好的 architecture），四种 strategy 的 relative ranking 会变吗？

## 对 VLA 领域的 Practical Guidance

### 什么时候用什么 Strategy

| 场景 | 推荐 Strategy | 理由 |
|------|--------------|------|
| Long-horizon manipulation (LIBERO-Long) | LA-Direct | Image-based visual plan supervision 对 multi-stage planning 特别有效 |
| Dual-arm coordination (RoboTwin) | LA-Tok | Action-based tokens 对 motor precision 和 coordination 更有效 |
| OOD scene generalization | LA-Direct 或 LA-Cond | Image-based latent actions 更 scene-invariant |
| Multi-task joint training | LA-Cond | Conditional decoding 消除 negative transfer |
| Low-data regime | LA-Tok | Latent tokens 作为 compressed supervision 降低 sample complexity |
| General purpose (uncertain) | LA-Direct | 最 consistent 的 overall performance |

### Implementation Tips

1. **永远用 discrete tokens** 做 supervision，不要用 continuous representations
2. **取 VLM 后半部分层** 的 placeholder representations，不要用 early 或 final layer
3. **Codebook size 要 task-specific**：image-based 可以小 (16)，action-based 要大 (256)
4. **λ 的设置要 normalize loss scale**：cosine similarity loss 用 $\lambda=1.0$，cross-entropy loss 用 $\lambda=0.1$
5. **Latent action model 要 freeze**，不要和 VLA joint training，否则 latent representation 会 drift

## 相关工作链接

- [OpenVLA](https://arxiv.org/abs/2406.09246) - Kim et al., 2024
- [π0](https://arxiv.org/abs/2410.24164) - Black et al., 2024
- [π0-FAST](https://arxiv.org/abs/2501.09747) - Pertsch et al., 2025，action tokenization via DCT
- [π0.5](https://arxiv.org/abs/2504.16054) - Physical Intelligence, 2025，open-world generalization
- [LAPA](https://arxiv.org/abs/2410.11758) - Ye et al., 2024，latent action pretraining from videos
- [UniVLA](https://arxiv.org/abs/2505.06111) - Bu et al., 2025，task-centric latent actions
- [GR00T N1](https://arxiv.org/abs/2503.14734) - Bjorck et al., 2025，humanoid foundation model
- [Moto-GPT](https://arxiv.org/abs/2503.13425) - Chen et al., 2025，latent motion tokens
- [villa-x](https://arxiv.org/abs/2507.23682) - Chen et al., 2025，enhanced latent action modeling
- [Genie](https://arxiv.org/abs/2402.15391) - Bruce et al., 2024，generative interactive environments
- [VQ-VAE](https://arxiv.org/abs/1711.00937) - Van Den Oord et al., 2017，neural discrete representation learning
- [MAE](https://arxiv.org/abs/2111.06377) - He et al., 2022，masked autoencoders
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2022，low-rank adaptation
- [LIBERO](https://arxiv.org/abs/2306.03310) - Liu et al., 2023，benchmark for lifelong robot learning
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) - Chen et al., 2025，bimanual manipulation benchmark
- [Spatial Forcing](https://arxiv.org/abs/2510.12276) - Li et al., 2025，implicit spatial representation alignment
- [VLA-Adapter](https://arxiv.org/abs/2509.09372) - Wang et al., 2025，tiny-scale VLA paradigm
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) - Kim et al., 2025，optimized fine-tuning recipe
- [RDT-1B](https://arxiv.org/abs/2410.07864) - Liu et al., 2024，diffusion foundation model for bimanual
- [ACT](https://arxiv.org/abs/2304.13705) - Zhao et al., 2023，action chunking transformer
- [DP3](https://arxiv.org/abs/2403.03954) - Ze et al., 2024，3D diffusion policy
- [Ego4D](https://arxiv.org/abs/2110.07058) - Grauman et al., 2022，egocentric video dataset
- [UMI](https://arxiv.org/abs/2402.10329) - Chi et al., 2024，universal manipulation interface
- [DINOv2](https://arxiv.org/abs/2304.07193) - Oquab et al., 2023，self-supervised vision features
- [Latent Action with Distractors](https://arxiv.org/abs/2502.00379) - Nikulin et al., 2025，supervision requirements

---

# From Pixels to Tokens: Latent Action Supervision for VLA Models 深度解析

## 1. 核心问题与动机

这篇paper针对VLA (Vision-Language-Action) models训练中的一个核心痛点：**heterogeneous action spaces**。当你在不同robotic platforms（比如单臂JAKA、双臂RoboTwin、人形机器人）和human videos (Ego4D) 上joint training时，action semantics不一致会导致negative transfer。

Latent actions提供了一种intermediate representation来bridge这个gap，但community里的integration方式非常fragmented：
- LAPA ([paper](https://arxiv.org/abs/2410.11758)) 用discrete token pretraining
- UniVLA ([paper](https://arxiv.org/abs/2505.06111)) 用task-centric latent actions
- Moto-GPT ([paper](https://arxiv.org/abs/2503.13425)) joint generation
- villa-x ([paper](https://arxiv.org/abs/2507.23682)) 联合latent+action

这些方法的formulation和architecture entangle在一起，无法fair comparison。这篇paper的核心贡献是**在unified baseline下隔离latent action formulation和integration strategy的影响**。

## 2. 两个Perspective的intuition

### Perspective 1: Regularizing the Trajectory (Forward Mapping)

从VL space → Action space的forward direction。Image-based latent actions扮演high-level visual plan的角色，告诉VLM"接下来场景应该怎么变化"。

**Intuition**: 想象你教一个人开车，你可以告诉他"把车开到那个停车场"（high-level plan），也可以告诉他"方向盘转30度，油门踩到50%"（low-level action）。Image-based latent actions相当于前者——它捕捉的是visual state transitions，抽象掉了motor details。

### Perspective 2: Unifying the Target Space (Reverse Mapping)

从Action space → VL token space的reverse direction。Action-based latent actions把continuous actions映射成discrete tokens，让VLM可以直接predict。

**Intuition**: 不同机器人的action dimensionality不同（7-DOF vs 14-DOF双臂），但如果你把所有actions都tokenize到一个shared codebook，VLM看到的都是统一的token sequence，heterogeneous problem就消失了。

## 3. Latent Action Models细节

### 3.1 Image-based Latent Action Model (基于UniVLA)

公式(1)的三步：
$$c_t^{img} = E_\theta([o_t; o_{t+\delta}]), \quad c_t^{img} \in \mathbb{R}^d$$

- $E_\theta$: encoder（基于frozen DINOv2 + Transformer）
- $o_t$: start observation at timestep $t$
- $o_{t+\delta}$: end observation，$\delta=32$ 是temporal horizon
- $[o_t; o_{t+\delta}]$: concatenation
- $d$: embedding dimension（paper里是128）

$$z_t^{img} = VQ(c_t^{img}), \quad z_t^{img} \in \{1, \ldots, K_{img}\}^P$$

- $VQ(\cdot)$: vector quantization operator
- $K_{img}=16$: codebook size
- $P=4$: 每个timestep对应4个discrete tokens

$$\hat{o}_{t+\delta} = D_\theta([o_t; e(z_t^{img})])$$

- $D_\theta$: decoder重建future observation
- $e(\cdot)$: codebook embedding lookup

**关键设计决策**：作者在UniVLA基础上加了5% action supervision（Appendix A.1的Eq 14）：
$$\mathcal{L}_{act}^{img} = \|\hat{a}_t - a_t^*\|_2^2$$

这个auxiliary head只在latent action learning阶段用，downstream policy learning时丢弃。Table 6的ablation显示：没有action supervision时，所有策略都drop约0.2-0.3%，证明少量action supervision显著提升latent quality。这呼应了Nikulin et al. ([paper](https://arxiv.org/abs/2502.00379))的发现——pure unsupervised latent action learning在有distractors时会degenerate。

### 3.2 Action-based Latent Action Model (作者自研)

这是paper的一个technical contribution。架构很巧妙——结合frequency domain和time domain：

**Encoder结构**：
1. **FFT augmentation**: 对action chunk $\mathsf{a}_t \in \mathbb{R}^{H \times m}$做Fast Fourier Transform，concatenate real和imaginary parts。这捕捉low-frequency trends（比如整体轨迹方向）
2. **1D Temporal Convolutions**: 3层，kernel size=3，dilation rates $\{1, 2, 4\}$。这捕捉fast variations across neighboring timesteps
3. **Transformer Encoder**: 2层，4 heads，feedforward dim=256。整合整个chunk的信息

**为什么FFT + Conv的组合？** FFT擅长全局趋势（比如机械臂从左到右的整体运动），convolution擅长局部细节（比如grasping瞬间的微小调整）。这种coarse-to-fine的inductive bias让codebook学到更structured的representation。

**Latent Consistency Loss** (Eq 18)：
$$\mathcal{L}_{mask} = \sum_{h \in \mathcal{M}} \|E_\phi(\tilde{\mathsf{a}}_t)_h - E_\phi(\mathsf{a}_t^*)_h\|_2^2$$

- $\mathcal{M}$: masked timestep indices（mask ratio 0.15）
- $\tilde{\mathsf{a}}_t$: 从masked latent sequence decode再re-encode的action
- 这个loss强制latent space在partial information下保持稳定

**Intuition**: 这类似于MAE (Masked Autoencoder)的思想——如果latent representation真的捕捉了action的本质，那么即使missing一些timesteps，剩下的latents也应该能infer出完整的action structure。这stabilize了tokenization。

## 4. Unified VLA Baseline架构

基于Qwen3-VL-2B，用LoRA ($r=64, \alpha=16$) fine-tune。关键设计：

**Parallel Decoding**: 单次forward pass预测整个action chunk，而不是autoregressive。

**Placeholder机制** (Eq 3)：
$$h_t^{act}/h_t^{latent} = Agg(\{\nu_t^{(k)}\}_{k \in \mathcal{K}_{layers}})$$

- $\nu_t^{(k)} \in \mathbb{R}^{H \times d_{hidden}}$: layer $k$的placeholder representation
- $\mathcal{K}_{layers} = \{\lceil N_{layer}/2 \rceil, \ldots, N_{layer}\}$: 取后半部分层（Qwen3-VL-2B有29层，所以从layer 15开始）
- $Agg(\cdot)$: 沿layer dimension stack

**为什么取后半部分层？** Table 5的ablation显示：early layers (-1.5%) insufficient semantic abstraction，final layer也suboptimal因为specialized for low-level control。Mid-to-late layers是sweet spot——既有semantic abstraction又还没完全collapse到output space。

**Action Head** (Eq 3第二行)：
$$\hat{a}_{t:t+H-1} = f_{head}([h_t^{img}; h_t^{act}; s_t])$$

- $h_t^{img}$: aggregated image token representations
- $h_t^{act}$: aggregated placeholder representations
- $s_t$: robot proprioceptive state（projected to $d_{hidden}$ via MLP）
- $f_{head}$: 受VLA-Adapter ([paper](https://arxiv.org/abs/2509.09372))启发的custom head

## 5. 四种Integration Strategies深度解析

### Strategy 1: LA-Align (Implicit Representation Alignment)

**Architecture**: Fig 3(b)，不增加placeholder，只在layer $k=17$加alignment constraint。

**Loss** (Eq 5)：
$$\mathcal{L}_{latent} = -\mathbb{E}_t[S(\phi_{align}(\nu_t^{(k)}), c_t^{img})]$$

- $\phi_{align}(\cdot)$: linear projection from $d_{hidden}$ to latent embedding space ($\mathbb{R}^{128}$)
- $S(\cdot, \cdot)$: cosine similarity
- $c_t^{img}$: ground-truth continuous latent embedding (pre-quantization)

**Intuition**: 这是"软约束"——不强制VLM输出特定tokens，而是让它的internal representation在语义上aligned with latent action embedding。类似Spatial Forcing ([paper](https://arxiv.org/abs/2510.12276))对3D spatial signals的做法。

**缺点**: 约束太弱，VLM可能学到shortcut而不真正internalize planning。

### Strategy 2: LA-Direct (Explicit Direct Decoding)

**Architecture**: Fig 3(c)，用 $P \times H$ 个latent placeholders直接predict discrete tokens。

**Loss** (Eq 6, 7)：
$$\pi_{vlm}(z_{t:t+H-1}^{img} | o_t, \ell) = Softmax(\phi_{explicit}(\nu_t^{(N_{layer})}))$$
$$\mathcal{L}_{latent} = -\sum_{h=0}^{H-1} \log \pi_{vlm}(z_{t+h}^{img*} | o_t, \ell)$$

- $\phi_{explicit}$: linear projection head to codebook logits (size $K_{img}=16$)
- $\nu_t^{(N_{layer})}$: final-layer placeholder states
- $z_{t+h}^{img*}$: ground-truth latent action token at position $h$

**Action prediction** (Eq 8)：
$$\hat{a}_{t:t+H-1} = f_{head}([h_t^{img}; h_t^{latent}; s_t])$$

这里 $h_t^{latent}$ 是latent placeholder的aggregated representation，直接feed给action head。

**关键insight**: VLM backbone同时承担两个任务——predict discrete latent tokens（显式planning）和提供action representation。这两个任务share同一个backbone但通过不同head输出。

### Strategy 3: LA-Cond (Explicit Conditional Decoding)

**Architecture**: Fig 3(d)，placeholders分成两段：
- Latent segment: $P \times H$ 个tokens，predict $z_t^{img}$
- Action segment: $H$ 个tokens，form action representation

**关键设计**: 通过causal attention mask让action segment conditioned on latent segment。即action placeholders attend to latent placeholders，但latent placeholders不attend to action placeholders。

**Action prediction** (Eq 9)：
$$\hat{a}_{t:t+H-1} = f_{head}([h_t^{img}; h_t^{latent}; h_t^{act}; s_t])$$

比LA-Direct多了 $h_t^{act}$——专门的action representation，conditioned on predicted latent plan。

**Intuition**: 这是"think before you act"的explicit implementation。VLM先generate high-level plan (latent tokens)，然后基于这个plan generate low-level action representation。

**与LA-Direct的区别**: LA-Direct是latent和action共享placeholder，implicit conditioning；LA-Cond是显式hierarchical，强制plan → action的causal dependency。

### Strategy 4: LA-Tok (Action-to-Token Mapping)

**Architecture**: Fig 3(e)，用 $H$ 个placeholders直接predict action-based latent tokens。

**Loss** (Eq 10)：
$$\pi_{vlm}(z_t^{act} | o_t, \ell) = Softmax(\phi_{tok}(\nu_t^{(N_{layer})}))$$

- $\phi_{tok}$: linear projection to action codebook logits ($K_{act}=256$)
- $z_t^{act} \in \{1, \ldots, 256\}^H$: discrete action tokens

**Action prediction**: 
$$\hat{\mathsf{a}}_t = f_{head}([h_t^{img}; h_t^{latent}; s_t])$$

Action head从latent representation deterministically decode出continuous action chunk。

**与LA-Direct的核心区别**: 
- LA-Direct: latent tokens来自image transitions (visual plan)
- LA-Tok: latent tokens来自actions themselves (motor primitives)

## 6. 实验结果深度分析

### 6.1 LIBERO Benchmark (Table 1)

| Method | Spatial | Object | Goal | Long | AVG |
|--------|---------|--------|------|------|-----|
| Baseline | 96.6 | 97.2 | 92.8 | 85.8 | 93.1 |
| LA-Align | 97.4 (+0.8) | 98.6 (+1.4) | 97.2 (+4.4) | 94.8 (+9.0) | 97.0 (+3.9) |
| LA-Direct | 97.2 (+0.6) | 99.4 (+2.2) | 95.8 (+3.0) | 96.6 (+10.8) | 97.1 (+4.0) |
| LA-Cond | 97.0 (+0.4) | 98.6 (+1.4) | 95.8 (+3.0) | 94.2 (+8.4) | 96.6 (+3.5) |
| LA-Tok | 97.0 (+0.4) | 100.0 (+2.8) | 92.2 (-0.6) | 92.6 (+6.8) | 95.5 (+2.4) |

**关键观察**:
1. **LIBERO-Long上image-based策略提升巨大** (+8.4%到+10.8%)，而LA-Tok只有+6.8%。Long-horizon任务需要multi-stage planning，image-based latent actions提供的visual plan supervision特别有效。
2. **LA-Direct > LA-Cond**: 在long-horizon上96.6% vs 94.2%。Decoupling latent plan prediction from action representation learning比joint modeling更有效。
3. **LA-Tok在Goal上actually degrade** (-0.6%)。可能因为Goal任务更依赖visual reasoning而非motor precision。

### 6.2 RoboTwin 2.0 (Table 2)

| Method | Move Card | Place Container | Move Can | Pick Dual | AVG |
|--------|-----------|-----------------|----------|-----------|-----|
| Baseline | 73 | 86 | 46 | 37 | 60.5 |
| LA-Align | 78 (+5) | 88 (+2) | 55 (+9) | 61 (+24) | 70.5 (+10.0) |
| LA-Direct | 76 (+3) | 96 (+10) | 64 (+18) | 51 (+14) | 71.8 (+11.3) |
| LA-Cond | 76 (+3) | 89 (+3) | 52 (+6) | 78 (+41) | 73.8 (+13.3) |
| LA-Tok | 89 (+16) | 89 (+3) | 70 (+24) | 64 (+27) | 78.0 (+17.5) |

**关键观察**:
1. **LA-Tok大幅领先** (+17.5%)。RoboTwin是双臂coordinated manipulation，motorically complex。Action-based latent actions提供的structured motor supervision特别有效。
2. **LA-Cond在Pick Dual Bottles上+41%**。这个任务需要dual-arm coordination，joint modeling latent+action让VLM能先plan协调动作再execute。
3. **Pattern**: Image-based策略在visual reasoning重的任务强，action-based策略在motor control重的任务强。

### 6.3 Real-World Pick-and-Place (Table 3)

| Method | Mango ID | Mango OOD | Sponge ID | Sponge OOD | Bottle ID | Bottle OOD | Total |
|--------|----------|-----------|-----------|------------|-----------|------------|-------|
| Baseline | 100 | 30 | 60 | 20 | 70 | 0 | 47 |
| LA-Align | 90 | 80 | 90 | 60 | 60 | 30 | 68 (+21) |
| LA-Direct | 100 | 70 | 90 | 80 | 60 | 50 | 75 (+28) |
| LA-Cond | 90 | 90 | 90 | 80 | 50 | 40 | 73 (+26) |
| LA-Tok | 100 | 60 | 90 | 70 | 60 | 20 | 67 (+20) |

**OOD Generalization的关键发现**:
- Baseline OOD average: 17 (严重退化)
- LA-Cond OOD average: 70 (best)
- LA-Direct OOD average: 67

Image-based策略在OOD上明显更强。**Intuition**: Image-based latent actions编码的是visual state transitions，这比raw actions更抽象、更scene-invariant。当scene configuration变化时，visual plan的semantics保持稳定，而motor details可能需要adapt。

### 6.4 Discrete vs Continuous Supervision (Table 4)

| Method | Spatial | Object | Goal | Long | AVG |
|--------|---------|--------|------|------|-----|
| LA-Direct (C) | 95.4 | 97.0 | 90.0 | 95.2 | 94.4 |
| LA-Direct | 97.2 | 98.6 | 95.8 | 96.6 | 97.1 (+2.7) |
| LA-Tok (C) | 95.6 | 98.4 | 87.6 | 91.6 | 93.3 |
| LA-Tok | 97.0 | 100.0 | 92.2 | 92.6 | 95.5 (+2.2) |

**为什么discrete > continuous?** 
1. **Gradient stability**: Cross-entropy loss on discrete tokens比MSE on continuous embeddings更stable
2. **Information bottleneck**: Discretization强制VLM learn compressed representation，避免overfitting to noise
3. **Alignment with LLM pretraining**: VLM backbone pretrained on discrete tokens，discrete supervision更compatible

但注意：continuous variants仍然outperform baseline (94.4 vs 93.1)，说明latent action supervision本身就有价值，discrete只是进一步amplify。

### 6.5 Multi-task Joint Training (Table 11, Fig 5)

这是最intriguing的结果。在10个RoboTwin任务上joint training：

- **Baseline**: Single-task avg 35.4% → Multi-task avg 53.6%，但move can pot从44%跌到13% (-31%)，move playingcard away从60%跌到49% (-11%)。Classic negative transfer。
- **LA-Cond**: Single-task avg 42.7% → Multi-task avg 74.5%，**没有任何task退化**。所有task都improve或保持。

**为什么latent action supervision消除negative transfer?**

当不同任务mix时，action spaces可能conflict（比如有些任务主要用gripper，有些主要用wrist）。Continuous action regression会让VLM在conflicting gradients之间compromise。而latent action tokens提供了一种**task-agnostic的中间表示**——不同任务的actions可能不同，但它们的latent abstractions可以share common structure。

LA-Cond的conditional decoding进一步加强这个效果：VLM先predict latent plan（task-specific but abstract），再基于plan generate action（concrete but conditioned）。这种hierarchical decomposition让multi-task learning更stable。

### 6.6 Data Efficiency (Appendix E.1, Fig 8)

- LIBERO-Long, 50% data: LA-Tok 94.0% vs Baseline 80.0% (+14.0%)
- LIBERO-Goal, 33% data: LA-Tok 86.5% vs Baseline 78.6% (+7.9%)

Latent actions作为compressed supervision signal，显著降低sample complexity。**Intuition**: 比起直接predict high-dimensional continuous actions，predict low-dimensional discrete tokens是更easy的学习任务，VLM能更快converge。

### 6.7 Placeholder Length Ablation (Fig 6, Table 9-10)

| Method | Spatial | Object | Goal | Long | AVG |
|--------|---------|--------|------|------|-----|
| Baseline | 96.6 | 97.2 | 92.8 | 85.8 | 93.1 |
| Baseline (PH-Direct) | 94.0 | 94.8 | 93.0 | 85.8 | 91.9 (-1.2) |
| LA-Direct | 97.2 | 98.6 | 95.8 | 96.6 | 97.1 (+4.0) |

这个ablation很关键——它证明gains不是来自increased sequence capacity。Baseline (PH-Direct)有和LA-Direct一样长的placeholder但没有任何latent supervision，结果actually degrade (-1.2%)。更长sequence带来更多parameters to learn，但没有inductive bias的guidance反而hurt。

## 7. Action Head架构细节

基于VLA-Adapter设计。Input包括三部分concatenated：
1. $h_t^{img}$: image token representations aggregated from VLM后半部分层
2. $h_t^{latent}$ (或 $h_t^{act}$): placeholder representations
3. $s_t$: proprioceptive state projected via MLP

Action head output: $\hat{\mathsf{a}}_t \in \mathbb{R}^{H \times m}$

Training loss (Eq 4):
$$\mathcal{L}_{action} = \mathbb{E}_t\left[\sum_{\tau=0}^{H-1} \|\hat{a}_{t+\tau} - a_{t+\tau}^*\|_2^2\right]$$

标准L2 regression on action chunk。

## 8. Training Configuration (Table 7, 8)

**Placeholder配置很关键**:

| Strategy | Latent PH | Action PH | Total | λ |
|----------|-----------|-----------|-------|---|
| Baseline | 0 | H | H | 0 |
| LA-Align | 0 | H | H | 1.0 |
| LA-Direct | P×H | 0 | P×H | 0.1 |
| LA-Cond | P×H | H | P×H+H | 0.1 |
| LA-Tok | H | 0 | H | 0.1 |

**为什么LA-Align的λ=1.0而其他是0.1?** Cosine similarity loss bounded in [-1,1]，numerical magnitude小；cross-entropy loss on tokens数值大。用不同λ normalize contribution。

**Hyperparameters**:
- LIBERO: 80k steps, batch 128, lr 1e-4, H=8
- RoboTwin: 30k steps, batch 128, lr 1e-4, H=25
- JAKA: 30k steps, batch 128, lr 1e-4, H=20
- LR scheduler: step decay, factor 0.5 every 10k steps

## 9. 局限性与未来方向

Paper自己承认的limitations:
1. 四种strategies可能不cover所有possible formulations
2. 没探索stronger latent representations是否会amplify/narrow gaps
3. Real-world只在单臂JAKA上验证

**我认为的额外open questions**:
- Image-based和action-based能否combine？Paper没有try hybrid
- Latent action model的codebook size如何auto-select？$K_{img}=16$ vs $K_{act}=256$差异巨大
- 在更diverse embodiments（人形、四足）上pattern是否hold？
- Inference latency如何？Paper说single forward pass，但placeholder长度增加会带来computational overhead

## 10. 对VLA领域的implications

这篇paper给community的practical guidance：

1. **如果你的任务偏long-horizon reasoning**: 用LA-Direct (image-based, explicit direct decoding)
2. **如果你的任务偏motor control**: 用LA-Tok (action-based tokenization)
3. **如果你要multi-task joint training**: 用LA-Cond (conditional decoding减少negative transfer)
4. **永远用discrete tokens**而不是continuous representations做supervision

**更深层的insight**: Latent actions本质上是一种**curriculum**——它让VLM先learn easy task (predict abstract tokens)再learn hard task (predict continuous actions)。这种decomposition比end-to-end regression更sample-efficient且generalizable。

**与π0-FAST ([paper](https://arxiv.org/abs/2501.09747))的关系**: π0-FAST用DCT compression做action tokenization，是action-based latent action的特例。但这篇paper显示，action-based不是universal best——在long-horizon和OOD generalization上image-based更强。

**与OpenVLA-OFT ([paper](https://arxiv.org/abs/2502.19645))的关系**: OpenVLA-OFT是strong baseline (97.1% on LIBERO)，但它用7B model。这篇paper的LA-Direct用2B达到97.1%，显示latent action supervision能让小model达到大model的性能。

## References

- [VQ-VAE](https://arxiv.org/abs/1711.00937) - Van Den Oord et al., 2017
- [OpenVLA](https://arxiv.org/abs/2406.09246) - Kim et al., 2024
- [π0](https://arxiv.org/abs/2410.24164) - Black et al., 2024
- [LAPA](https://arxiv.org/abs/2410.11758) - Ye et al., 2024
- [UniVLA](https://arxiv.org/abs/2505.06111) - Bu et al., 2025
- [GR00T N1](https://arxiv.org/abs/2503.14734) - Bjorck et al., 2025
- [π0-FAST](https://arxiv.org/abs/2501.09747) - Pertsch et al., 2025
- [LIBERO](https://arxiv.org/abs/2306.03310) - Liu et al., 2023
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) - Chen et al., 2025
- [DINOv2](https://arxiv.org/abs/2304.07193) - Oquab et al., 2023
- [LoRA](https://arxiv.org/abs/2106.09685) - Hu et al., 2022
- [Genie](https://arxiv.org/abs/2402.15391) - Bruce et al., 2024
- [Moto-GPT](https://arxiv.org/abs/2503.13425) - Chen et al., 2025
- [villa-x](https://arxiv.org/abs/2507.23682) - Chen et al., 2025
- [Spatial Forcing](https://arxiv.org/abs/2510.12276) - Li et al., 2025
- [VLA-Adapter](https://arxiv.org/abs/2509.09372) - Wang et al., 2025
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) - Kim et al., 2025
- [Latent Action Learning with Distractors](https://arxiv.org/abs/2502.00379) - Nikulin et al., 2025
- [Qwen-VL](https://arxiv.org/abs/2308.12966) - Bai et al., 2023
- [ACT](https://arxiv.org/abs/2304.13705) - Zhao et al., 2023
- [DP3](https://arxiv.org/abs/2403.03954) - Ze et al., 2024
- [RDT-1B](https://arxiv.org/abs/2410.07864) - Liu et al., 2024
- [Ego4D](https://arxiv.org/abs/2110.07058) - Grauman et al., 2022
- [UMI](https://arxiv.org/abs/2402.10329) - Chi et al., 2024
