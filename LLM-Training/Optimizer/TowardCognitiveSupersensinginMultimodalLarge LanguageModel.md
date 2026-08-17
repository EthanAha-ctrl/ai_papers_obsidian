---
source_pdf: TowardCognitiveSupersensinginMultimodalLarge LanguageModel.pdf
paper_sha256: b0ce5f882734693f3a6dce9cd1b4a2c042ada6df6d0275475dc62debde6c9358
processed_at: '2026-08-12T16:56:13-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

## 一句话说清楚这帮人想干嘛

现在的 multimodal LLM 有个毛病：**它能看，但不能"想"**。

你给它一张图，它能描述里面有什么——"一只猫坐在桌子上"。但你让它做 Raven's Progressive Matrices 那种抽象推理题，或者让它想象"把这个方块转 90 度会变成什么样"，它就歇菜了。

原因很简单：大家都在用**文字**做 Chain-of-Thought reasoning。但很多视觉操作——mental rotation、spatial transformation、pattern induction——本质上**不是文字能表达的东西**。你硬要用 token 去描述"旋转过程中的中间状态"，信息就丢了。

这帮人的核心想法：给 MLLM 装一个**"mind's eye"**——一个在 latent space 里做视觉推理的模块，跟文字 reasoning 并行工作。

GitHub: https://github.com/PediaMedAI/Cognition-MLLM
HuggingFace: https://huggingface.co/datasets/PediaMedAI/CogSense-Bench

---

## 为什么 Text-only CoT 不行——一个直觉例子

假设给你一道 Raven's 题目：左边一组图案按某个规则变化，让你选右边该填什么。

人类怎么解题？你脑子里会**想象**中间状态——"如果这个形状往左移一格，颜色变深，那下一步应该是..."。这个"想象"过程是**连续的、visual 的、非语言的**。

现在 MLLM 怎么做？它把所有中间步骤翻译成文字 token："The shape in position 1 is a triangle, it moves left by one unit, the color changes from light to dark..." 然后基于这些 token 推理。

问题来了：**spatial relations 被压成 linear text token，就塌缩了**。几何关系、连续变换、holistic structure——这些信息用离散文字表达本身就是 information loss。这就像用 ASCII art 画蒙娜丽莎，信息丢了 90%。

这就是为什么 GPT-5.2 在他们新提出的 CogSense-Bench 上只有 40.3% accuracy，而人类是 88.4%。

参考 Schulze Buschof et al. (2025) Nature Machine Intelligence: https://www.nature.com/articles/s42256-024-00955-2

---

## CogSense-Bench：测五个认知维度

他们造了一个 benchmark，不测"认不认识"，测"能不能想"。五个维度，每个都有 cognitive science 理论 backing：

| 维度 | 通俗解释 | 对应任务 | 人类水平 |
|------|----------|---------|---------|
| **Fluid Intelligence** | 没见过的问题能不能推理出来 | Raven's matrices、PGM | 82.7 |
| **Crystallized Intelligence** | 用学过的知识做归纳 | Bongard problems | 91.3 |
| **Visuospatial Cognition** | 3D 空间结构理解 | Bongard-LOGO | 88.5 |
| **Mental Simulation** | 脑内模拟物理过程 | KiVA、STARE、ARC-AGI | 97.9 |
| **Visual Routines** | 视觉搜索 + 注意力控制 | CVR | 78.7 |

训练集 105K，测试集 1000 题。数据来源都是经典 cognitive science benchmark——RAVEN、PGM、Bongard、ARC-AGI 这些。

Table A1 数据统计：
- Fluid Intelligence: MaRs-VQA 1.4K + PGM 10K + RAVEN 18K
- Crystallized Intelligence: Bongard-RWR+ 16K + Bongard-HOI 23K  
- Visuospatial: Bongard-LOGO 12K
- Mental Simulation: KiVA 1.4K + STARE 4K + ARC-AGI 1.6K + ARC-AGI-2 8K
- Visual Routines: CVR 10K

关键 insight：这些任务都需要 **(i) multi-step explicit reasoning** + **(ii) 维护和操作 answer-oriented 的 internal visual states**。Text-only CoT 两样都做不到。

参考 Cattell (1963) Fluid/Crystallized intelligence: https://doi.org/10.1037/h0043040
参考 Battaglia et al. (2013) Intuitive physics: https://www.pnas.org/doi/10.1073/pnas.1318523110

---

## Architecture：LVIP Head 到底是个什么东西

先看整体数据流：

```
输入: 图像 V + 文字问题 Q
     ↓
V → Visual Encoder → visual features → Projection → visual tokens (h_V)
Q → Tokenize → text tokens (h_Q)
     ↓
[h_V, h_Q] → LLM Backbone → hidden states
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
              Text Decoder              LVIP Head (g_ψ)
              生成 (Z, y)              预测 ĥ_y (latent imagery)
```

**LVIP Head 的具体设计**：

输入 $V$ 里包含 question image + candidate option images。Backbone 输出所有 visual tokens 的 hidden states $\mathbf{H}_V \in \mathbb{R}^{N \times d}$。从中 extract 出 option images 对应的 subset $\mathbf{H}_{\mathrm{opt}} \in \mathbb{R}^{M \times d}$，做 average pooling 得到 $\bar{\mathbf{h}}_{\mathrm{opt}}$，过 two-layer MLP：

$$
\hat{h}_y = g_\psi(\bar{\mathbf{h}}_{\mathrm{opt}})
$$

**Supervision target**：ground-truth answer image $V_y$ 过 frozen visual encoder 得到的 embedding：
$$
h_y = \mathrm{Enc}_{\mathrm{vis}}(V_y)
$$

用 MSE loss 拉近 $\hat{h}_y$ 和 $h_y$。

**直觉**：这个 head 强制 backbone 在 option tokens 位置上 encode 出足够 recover ground-truth answer 的 visual information。相当于告诉 backbone："你在做 reasoning 的时候，内部 representation 要能"看见"正确答案长什么样"。

这个思路跟 cognitive science 里的 **constructive matching** 对应——人类做这类题目时确实会先在脑子里构造一个候选答案的 image，再去 options 里找最接近的。

参考 Ganis & Schendan (2011) Visual imagery: https://onlinelibrary.wiley.com/doi/10.1002/wcs.103

---

## 三阶段 Training——一步步看

### Stage I: Reasoning Chain Generation

问题：high-quality cognitive CoT data 极其稀缺。

解法：用强大的 teacher MLLM $\mathcal{M}_T$ 生成 reasoning chain：

$$
(Z, \hat{y}) \sim \mathcal{M}_T(\cdot | \mathcal{V}, \mathcal{Q}, \mathcal{P}_{\mathrm{gen}})
$$

- $Z$: reasoning rationale (中间推理步骤)
- $\hat{y}$: predicted answer
- $\mathcal{P}_{\mathrm{gen}}$: task-specific generation prompt

Filter: 只保留 $\hat{y}_i = y_i$（答案对）且 no hallucination 的 samples。最终得到 $\mathcal{D}_{\mathrm{chain}} = \{(\mathcal{V}_i, \mathcal{Q}_i, Z_i, y_i)\}_{i=1}^N$。

### Stage II: SFT with LVIP

Loss function 公式 (2):

$$
\mathcal{L}_{\mathrm{SFT}} = -\underbrace{\sum_{t=1}^{|x|} \log q_\theta(x_t | X, x_{<t})}_{\text{标准 autoregressive CE}} + \underbrace{\beta \cdot \mathrm{MSE}(\hat{h}_y, h_y)}_{\text{LVIP auxiliary loss}}
$$

变量解释：
- $x = (Z, y)$: target sequence，rationale + answer 拼起来
- $|x|$: sequence 长度
- $x_t$: 第 $t$ 个 token
- $x_{<t}$: 前 $t-1$ 个 tokens
- $q_\theta$: model distribution，参数 $\theta$
- $X = (V, Q)$: multimodal input
- $\hat{h}_y$: LVIP 预测的 latent imagery
- $h_y$: ground-truth answer 的 visual embedding
- $\beta$: 平衡系数，控制 LVIP loss 权重

**直觉**：两个 loss 联合优化。CE loss 保证 text reasoning 正确，MSE loss 强制 latent representation 对齐到 ground-truth answer 的 visual encoding。两者耦合，semantic reasoning 跟 visual world modeling 必须同步对齐。

### Stage III: RL with Latent Rationales

这步用 **GFlowNet** 做多样性 seeking RL，而不是 PPO/GRPO。为什么？因为 reasoning path 有多个 valid 解，GFlowNet 能 sample proportional to reward，保持 diversity，不会 collapse 到单一 mode。

Target posterior:
$$
p^*(Z | X, y) \propto \exp(R(Z; X, y))
$$

Reward 设计是核心，分两部分：
$$
R(Z; X, y) = \alpha R_{\mathrm{ans}}(Z; X, y) + \gamma R_{\mathrm{lvip}}(Z; X, y)
$$

**Answer evidence reward** 公式 (5):
$$
R_{\mathrm{ans}}(Z; X, y) = \log q_{\theta_0}(y | X, Z)
$$
- $q_{\theta_0}$: **frozen scorer**——SFT model 的固定副本，避免 moving-target 问题
- 这就是"给定 input 和 rationale，frozen model 给出正确答案的 log probability"

**LVIP reward** 公式 (6):
$$
R_{\mathrm{lvip}}(Z; X, y) = -\|g_\psi(\bar{\mathbf{h}}_{\mathrm{opt}}(X, Z; \theta)) - h_y\|_2^2
$$
- $g_\psi$: LVIP head，RL 阶段 frozen
- $\bar{\mathbf{h}}_{\mathrm{opt}}(X, Z; \theta)$: backbone 接收 $[X; Z]$ 后 option tokens 的 average-pooled hidden state
- $h_y$: ground-truth answer image embedding
- 负号：距离越小 reward 越大

**关键 insight**：LVIP reward 提供 **representation-level dense supervision**。Text answer reward 是 sparse binary signal（对/错），LVIP reward 是 continuous distance，能告诉 model "你的 reasoning 让内部 representation 离正确答案更近了"。这对 long rationale trajectory 的 credit assignment 很重要。

**Token-Wise Marginal Reward**：GFlowNet 需要每个 prefix 的 reward signal。但每个 token 都跑 frozen scorer 太贵。作者用 **sparse anchor + linear interpolation**：

Anchor positions: $t \in \{0, \lambda, 2\lambda, \ldots\}$，间隔 $\lambda$。

对 anchor 之间的位置 $i \in \{0, 1, \ldots, t^+ - t\}$，线性插值公式 (7):
$$
\widetilde{R}(\tau_{t+i}) = \widetilde{R}(\tau_t) + \frac{i}{t^+ - t}\Big(\widetilde{R}(\tau_{t^+}) - \widetilde{R}(\tau_t)\Big)
$$

- $\tau_t = (z_{1:t}, \top)$: prefix state，$\top$ 表示 termination 可选
- $t^+ = \min(t + \lambda, n)$: next anchor
- $i$: 当前 segment 内 offset

Anchor 处的 reward 公式 (8):
$$
R(\tau_t; X, y) = \alpha \log q_{\theta_0}(y | X, z_{1:t}) + \gamma R_{\mathrm{lvip}}(\tau_t; X, y)
$$

然后用 **SubTB loss** (Madan et al., 2023) 训练 $q_\theta(Z | X)$。

参考 SubTB GFlowNet: https://proceedings.mlr.press/v202/madan23a.html

### Reference-Guided GFlowNet Fine-tuning

为降 variance，用 reference rationale $Z_{\mathrm{ref}}$ 锚定 exploration。对每个 $X$，sample $m$ 个 candidates $\{Z_i\}_{i=1}^m$，acceptance criterion 公式 (9):

$$
\mathbb{I}(Z_i) = \mathbf{1}[R(Z_i; X, y) \geq R(Z_{\mathrm{ref}}; X, y) + \log \delta_s]
$$

- $Z_{\mathrm{ref}}$: reference（可能是 SFT 阶段生成的 rationale）
- $\delta_s \in (0, 1]$: step-dependent slack，$s$ 是 training step
- $\log \delta_s \leq 0$：允许 sample 比 reference 稍差一点点

只对 accepted trajectories 优化：
$$
\mathcal{L}(\theta) = \sum_{i=1}^m \mathbb{I}(Z_i) \cdot \mathcal{L}_{\mathrm{subTB}}(Z_i; \theta)
$$

### Inference：Bayesian Aggregation over Latent Rationales

Inference 时把 $Z$ 当 latent variable，sample $N$ 条 rationales，每条 decode 出 answer $y_i$，用 length-normalized evidence score 选最优：

$$
S_i = \frac{1}{|Z_i| + |y_i|} \log q_{\theta_0}(y_i | X, Z_i)
$$

Output: $\hat{y} = y_{i^*}$ where $i^* = \arg\max_i S_i$。

这是 **MAP-style selection over sampled latent rationales**——类似 Self-Consistency CoT (Wang et al., 2022)，但在 latent space 操作，用 length normalization 防止偏好短 rationale。

参考 Self-Consistency: https://arxiv.org/abs/2203.11171

---

## 实验结果——数字说话

### Main Results (Table 1)

| Model | Fluid Intel. | Cryst. Intel. | Visuosp. Cog. | Mental Simu. | Visual Rout. | Avg |
|-------|-------------|---------------|---------------|--------------|--------------|-----|
| **Human** | 82.7 | 91.3 | 88.5 | 97.9 | 78.7 | **88.4** |
| Gemini 2.5 Flash | 23.2 | 40.2 | 31.0 | 40.2 | 45.3 | 36.3 |
| GPT-o3 | 4.7 | 51.4 | 20.4 | 38.7 | 43.0 | 32.3 |
| GPT-5.2 | 29.4 | 35.9 | 57.5 | 60.0 | 37.6 | 40.3 |
| Claude Sonnet 4 | 22.5 | 31.3 | 26.6 | 58.0 | 34.4 | 32.6 |
| Grok 4 Fast | 13.0 | 45.4 | 41.6 | 21.3 | 37.6 | 31.7 |
| Llama-4-Scout-17B | 20.3 | 29.9 | 35.4 | 48.7 | 41.9 | 31.8 |
| Qwen3-VL-30B | 30.8 | 34.0 | 37.2 | 56.0 | 40.9 | 37.4 |
| **CogSense-8B** | **63.8** | **91.0** | **69.0** | **68.0** | **50.5** | **73.8** |

几个值得注意的点：

1. **8B 模型比 GPT-5.2 高 33.5 个点**——这不是微小提升，是巨大 gap
2. **Crystallized Intelligence 上 CogSense-8B (91.0) 几乎追平 human (91.3)**——Bongard problems 这类 format 结构化的任务，model 学到 pattern 后能逼近人类
3. **GPT-o3 Fluid Intelligence 只有 4.7**——非常可疑。可能是 GPT-o3 的 long reasoning mode 在 abstract pattern matrices 上 over-think，反而答错。这说明更长的 text reasoning 不等于更好的 visual reasoning
4. **所有 frontier models 都远低于 human**——验证了 cognitive gap 假设

### General Ability (Table 2)——有没有忘掉基础知识

| Model | HallusionBench | AI2D | GQA | ScienceQA | RealWorldQA | ChartQA | BLINK | MMStar |
|-------|----------------|------|-----|-----------|-------------|---------|-------|--------|
| Qwen3-VL-8B (base) | 61.1 | 85.4 | 71.4 | 92.6 | 71.5 | 88.6 | 64.7 | 70.9 |
| CogSense-8B | 60.5 | 85.1 | 71.8 | 92.6 | 71.9 | 84.7 | 65.3 | 66.8 |

**几乎没掉点**。ChartQA 降 3.9，MMStar 降 4.1，但 GQA、ScienceQA、RealWorldQA、BLINK 反而略升。说明 cognitive training 学到的是 **generalizable visual reasoning pattern**，不是 task-specific overfitting。

### Ablation Study (Table 3)——拆解每个组件的贡献

| Variant | Fluid | Cryst. | Visuosp. | Mental Simu. | Visual Rout. | Avg |
|---------|-------|--------|----------|--------------|--------------|-----|
| Qwen3-VL-8B (base) | 31.2 | 34.8 | 31.0 | 45.3 | 40.9 | 35.5 |
| SFT w/o LVIP | 51.1 | 76.6 | 63.7 | 59.3 | 41.9 | 62.3 |
| SFT w/ LVIP | 55.4 | 88.6 | 61.1 | 61.3 | 44.1 | **68.0** |
| SFT w/o LVIP + GRPO | 55.8 | 79.9 | 63.7 | 63.3 | 43.0 | 65.5 |
| SFT w/ LVIP + GRPO | 59.1 | 89.9 | 64.6 | 65.3 | 46.2 | 70.8 |
| **CogSense-8B (full)** | **63.8** | **91.0** | **69.0** | **68.0** | **50.5** | **73.8** |

逐层拆解：

1. **Base → SFT w/o LVIP**: 35.5 → 62.3 (+26.8)
   - 光是加 reasoning chain 数据就几乎翻倍。说明 cognitive CoT data 本身极其 effective

2. **SFT w/o LVIP → SFT w/ LVIP**: 62.3 → 68.0 (+5.7)
   - LVIP 的 latent imagery grounding 贡献 +5.7。Crystallized Intelligence 提升最大（76.6 → 88.6），说明 visual concept learning 特别受益于 latent grounding

3. **SFT w/o LVIP + GRPO vs SFT w/o LVIP**: 62.3 → 65.5 (+3.2)
   - Standard GRPO 在 text-only rationale 上也有用，但有限

4. **SFT w/ LVIP + GRPO vs SFT w/ LVIP**: 68.0 → 70.8 (+2.8)
   - GRPO 在 LVIP 基础上还能加 buff

5. **CogSense-8B (GFlowNet) vs SFT w/ LVIP + GRPO**: 70.8 → 73.8 (+3.0)
   - **这 +3.0 就是 GFlowNet vs GRPO 的纯 gap**。证明 diversity-seeking latent rationale sampling 优于 deterministic policy optimization

这个 ablation 非常干净地 isolate 了三个 contributions: data (+26.8)、LVIP (+5.7)、GFlowNet RL (+3.0)。

### Out-of-Domain (Table 4)——能不能迁移

| Model | Chemistry | Math |
|-------|-----------|------|
| Qwen3-VL-8B (base) | 39.2 | 26.0 |
| CogSense-8B | 45.4 (+6.2) | 34.8 (+8.8) |

EMMA benchmark 上的提升确认 visual cognition patterns 能迁移到 math/chemistry VQA。Math 提升 +8.8 尤其显著。

参考 EMMA benchmark: https://arxiv.org/abs/2501.05444

---

## 几个值得深挖的 Intuition

### 为什么 LVIP 这么简单的东西会 work

LVIP head 就是 two-layer MLP，架构上简单到不能再简单。但它 work 的关键不在架构复杂度，而在 **supervision signal**：

1. **Ground-truth answer image embedding 作为 target**——这是个 extremely informative signal。Backbone 要让 option tokens 的 hidden state 能 recover 出 ground-truth answer 的 visual encoding，就必须在 reasoning 过程中真正"理解"视觉规则

2. **Auxiliary loss 强制 representation alignment**——standard CE loss 只管 text output，不管 backbone 内部 representation 长什么样。LVIP loss 逼着 backbone 在 option 位置上 organize 出 answer-oriented 的 visual representation

3. **跟 cognitive science constructive matching 对应**——人类做 multiple-choice visual reasoning 时确实会先在脑子里构造候选答案 image，再 match。LVIP 是这个过程的 computational analogue

### 为什么 GFlowNet 比 GRPO 好

GFlowNet 和 GRPO 都是 RL 方法，但哲学不同：

- **GRPO**: 找一个 optimal policy，maximize expected reward。会 collapse 到 single high-reward trajectory
- **GFlowNet**: 学一个 distribution over trajectories，sample proportional to reward。保持 diversity

在 visual reasoning 场景，**multiple valid rationales 存在**。比如解一道 Raven's 题目，可以从颜色角度推理，可以从形状角度推理，可以从位置角度推理——最终都能到正确答案。GRPO 会偏好某一条 path，GFlowNet 保持多条 path 的 distribution。

Ablation 显示 GFlowNet 比 GRPO 高 3.0 个点，这个 gap 就是 diversity 带来的好处。

参考 GFlowNet foundations (Bengio et al., 2023): https://arxiv.org/abs/2111.06477

### LVIP Reward 的 Dense Supervision 价值

Text answer reward 是 sparse binary signal——rationale 对或错。LVIP reward 是 continuous distance——"你的 reasoning 让内部 representation 离正确答案的 visual encoding 有多近"。

对 long rationale trajectory，sparse reward 的 credit assignment 极难——你不知道是第几步的 reasoning 出了问题。LVIP reward 提供了 **每个 prefix 的 dense signal**，因为 backbone conditioned on 不同 prefix $z_{1:t}$ 会产生不同的 hidden state，对应不同的 LVIP distance。

公式 (8) 把这两个 reward 结合：
$$
R(\tau_t; X, y) = \alpha \underbrace{\log q_{\theta_0}(y | X, z_{1:t})}_{\text{discrete answer evidence}} + \gamma \underbrace{R_{\mathrm{lvip}}(\tau_t; X, y)}_{\text{continuous representation grounding}}
$$

这跟 AlphaGo 里 value network + policy network 的思路类似——dense value signal 辅助 sparse reward。

---

## 跟其他工作的关系

### vs. Visual Sketchpad (Hu et al., 2024)

Visual Sketchpad 让 model 生成 **actual visual sketches** 作为 CoT intermediate。这是 explicit visual reasoning，interpretable 但 expensive。

CogSense 是 **latent visual reasoning**——不生成可解释的 visual intermediate，直接在 latent embedding space 操作。Trade-off：latent 更 efficient，但 loses interpretability。

参考 Visual Sketchpad: https://arxiv.org/abs/2406.09285

### vs. Mirage (Yang et al., 2025c)

Mirage 用 latent visual tokens 做 multimodal reasoning，思路相近。区别在于 CogSense 有 explicit supervision target（ground-truth answer image embedding），Mirage 多用 self-supervised signal。

参考 Mirage: https://arxiv.org/abs/2506.17218

### vs. Latent Sketchpad (Zhang et al., 2025)

Latent Sketchpad 让 model 在 latent space 画 "visual thoughts"，跟 CogSense 哲学一致。但 CogSense 加了 GFlowNet RL 优化 rationale sampling，Latent Sketchpad 侧重 SFT。

参考 Latent Sketchpad: https://arxiv.org/abs/2510.24514

### vs. World Models / Dreamer

LVIP 预测 answer image 的 latent embedding，本质是 learning a **conditional world model**——给定 question + reasoning，预测 answer state 的 visual representation。

跟 DreamerV3 (Hafner et al., 2023) 在 RL 场景下学习 dynamics model 类似，但 CogSense 在 VQA context 下，prediction target 是 answer 而不是 next state。

参考 DreamerV3: https://arxiv.org/abs/2301.04104

---

## Limitations 和 Open Questions

1. **LVIP 需要 ground-truth answer image**——只能 train 在 multiple-choice with image options 的 tasks。Open-ended generation 不直接适用。能不能 extend 到 free-form visual generation 是个 open question

2. **Two-layer MLP 可能太简单**——limit latent imagery 的 expressiveness。更深网络或 attention-based head 可能更好

3. **Frozen scorer $q_{\theta_0}$** 在 RL 阶段是 fixed，可能 stale。Periodic update 或 EMA 可能更好

4. **Inference 时 sample N rationales** 增加 compute cost，paper 没讨论 latency tradeoff

5. **Visuospatial Cognition 上 CogSense-8B (69.0) 还远低于 human (88.5)**——3D 空间理解仍是 bottleneck。LVIP 的 latent imagery 可能不够 expressive 来 capture 3D structure

6. **ARC-AGI 单独数字没 report**——如果 LVIP-style 方法能在 ARC-AGI 上突破，会是里程碑。参考 ARC-AGI-2: https://arxiv.org/abs/2505.11831

---

## Final Intuition

这篇 paper 的核心 thesis 用一句话说：**visual reasoning 需要非语言的 substrate**。

Text-only CoT 试图用 language 模拟 System 2 thinking，但 visual cognition 是 non-verbal System 2。你没法用文字完整描述 mental rotation 的中间状态——那是 geometric、continuous、non-symbolic 的。

LVIP head 的角色就是给 MLLM 装了个 **non-verbal System 2 module**——在 latent embedding space 维持和操作 visual representations，跟 text reasoning 并行工作。GFlowNet RL 进一步优化这个 non-verbal reasoning 的 sampling，保持 diversity。

结果：8B 模型在 cognitive reasoning 上碾压 30B frontier models。说明 specialized training 比 scale 更 efficient，前提是 training paradigm 对了。

这跟认知科学里 **visuospatial sketchpad** (Baddeley's working memory model) 完美对应——人类 working memory 有 verbal component (phonological loop) 和 visual component (visuospatial sketchpad)，MLLMs 之前只有 verbal，现在终于有了 visual。

参考 Baddeley's working memory model: https://en.wikipedia.org/wiki/Baddeley%27s_model_of_working_memory

这方向刚起步，期待后续工作把 LVIP extend 到 open-ended generation、video reasoning、robotics planning。

---

# Paper 深度解读: Toward Cognitive Supersensing in Multimodal Large Language Model

## 1. 一句话直觉

这篇 paper 直击 MLLMs 的一个根本性缺陷：**models 能 describe what is present，却无法 mentally operate on visual information**。作者认为问题出在 reasoning substrate——大家都在 text token space 做 CoT，但 visuospatial operations（mental rotation、dynamic simulation、pattern induction）本质上更适合用 continuous latent representations 表达。解决方案是给 MLLM 装一个 "mind's eye"——Latent Visual Imagery Prediction (LVIP) head，让 reasoning chain 一部分从离散 token 空间迁到 latent visual space，再用 GFlowNet-based RL 优化 rationale sampling。

GitHub: https://github.com/PediaMedAI/Cognition-MLLM
HuggingFace: https://huggingface.co/datasets/PediaMedAI/CogSense-Bench

---

## 2. 问题诊断：为什么 text-only CoT 是 brittle interface

作者的核心论点非常 cogent：很多 visual reasoning subroutines 本质上是 **geometric transformations、continuous states、structured visual relations**，把它们强行压缩成 linear text tokens 会引入 representational bottleneck。比如 mental rotation——你在脑子里转一个 3D shape，这个过程是连续的、几何的，tokenizing 它就是 information loss。

这和 cognitive science 里的 **visuospatial sketchpad** (Baddeley's model of working memory) 概念完全对应——人类有 "mind's eye" 维持和 transform 内部 visual representations。现在的 MLLMs 完全没有这个 substrate。

参考 Ganis & Schendan (2011): https://onlinelibrary.wiley.com/doi/10.1002/wcs.103
参考 Schulze Buschof et al. (2025) Nature MI: https://www.nature.com/articles/s42256-024-00955-2

---

## 3. CogSense-Bench：五个 cognitive dimensions

Benchmark 设计 grounded in cognitive science theory，五个维度：

| Dimension | Theory Grounding | 任务类型 |
|-----------|------------------|----------|
| **Fluid Intelligence (Gf)** | Cattell 1963, Structure Mapping Theory (Gentner 1983) | Raven's matrices, PGM - 抽象 rule induction |
| **Crystallized Intelligence (Gc)** | Cattell 1963, Prototype Theory (Rosch 1973) | Bongard problems - 从 variance 抽 semantic concept |
| **Visuospatial Cognition** | Gestalt laws (Wertheimer 1923), Recognition-by-Components (Biederman 1987) | Bongard-LOGO - 把离散 elements 重组为 holistic structure |
| **Mental Simulation** | Intuitive physics (Battaglia et al. 2013), Hypothetico-Deductive Reasoning | KiVA, STARE, ARC-AGI - 模拟 hidden dynamics |
| **Visual Routines** | Ullman 1984, Focused Attention | CVR - visual search + inhibitory control |

Total 1000 test questions。Table A1 显示训练集 105.4K，分布见 Fig 2。

Key insight: 这些任务都需要 (i) explicit multi-step reasoning composing elementary operations, (ii) maintaining 和 manipulating answer-oriented internal visual states during inference。这正是 text-only CoT 做不到的。

参考 Cattell (1963): https://doi.org/10.1037/h0043040
参考 Battaglia et al. (2013): https://www.pnas.org/doi/10.1073/pnas.1318523110
参考 Ullman (1984) Visual Routines: https://www.sciencedirect.com/science/article/pii/0010027784900234

---

## 4. Architecture 深度解析

### 4.1 整体数据流

输入 $X = (V, Q)$，其中 $V$ 是 visual input（包含 question image 和 candidate option images），$Q$ 是 textual prompt。

```
V → Enc_vis(·) → V_γ = {v_i}_{i=1}^T → P(·) → h_V (projected visual tokens)
Q → tokenize → h_Q (textual tokens)
[h_V, h_Q] → Enc_txt(·) → hidden states
                                      ├── text decoder → (Z, y) autoregressive
                                      └── LVIP head g_ψ(·) → ĥ_y (predicted latent imagery)
```

关键设计点：**LVIP head 挂在 backbone 对应 option images 的 visual tokens 上**，从 $\mathbf{H}_V \in \mathbb{R}^{N \times d}$ 里 extract 出 option subset $\mathbf{H}_{\mathrm{opt}} \in \mathbb{R}^{M \times d}$ ($M \leq N$)，average pool 得到 $\bar{\mathbf{h}}_{\mathrm{opt}}$，过 two-layer MLP $g_\psi$ 得到预测 $\hat{h}_y = g_\psi(\bar{\mathbf{h}}_{\mathrm{opt}})$。

Supervision target: $h_y = \mathrm{Enc}_{\mathrm{vis}}(V_y)$，ground-truth option image 通过 frozen visual encoder 得到的 embedding。

这个设计让我联想到 **Mirage** (Yang et al., 2025c) 和 **Latent Sketchpad** (Zhang et al., 2025) 的 latent visual reasoning 思路，但这篇的 novelty 在于 LVIP head 是 explicit auxiliary prediction target，跟 text decoder 联合训练。

参考 Mirage: https://arxiv.org/abs/2506.17218
参考 Latent Sketchpad: https://arxiv.org/abs/2510.24514

### 4.2 三阶段 Training Pipeline

**Stage I: Reasoning Chain Generation**
用 powerful teacher MLLM $\mathcal{M}_T$ 生成 reasoning rationales：
$$
(Z, \hat{y}) \sim \mathcal{M}_T(\cdot | \mathcal{V}, \mathcal{Q}, \mathcal{P}_{gen})
$$
Filter 条件：$\hat{y}_i = y_i$ 且 no hallucination。这步很关键，因为 high-quality cognitive CoT data 极其稀缺。

**Stage II: SFT with LVIP**
公式 (2):
$$
\mathcal{L}_{\mathrm{SFT}} = -\sum_{t=1}^{|x|} \log q_\theta(x_t | X, x_{<t}) + \beta \cdot \mathrm{MSE}(\hat{h}_y, h_y)
$$

变量解释：
- $x = (Z, y)$: target sequence = reasoning rationale concatenated with answer
- $|x|$: sequence 长度
- $x_t$: 第 $t$ 个 token
- $x_{<t}$: 前 $t-1$ 个 tokens (autoregressive conditioning)
- $q_\theta$: parameterized by $\theta$ 的 model distribution
- $X = (V, Q)$: multimodal input
- $\hat{h}_y = g_\psi(\bar{\mathbf{h}}_{\mathrm{opt}})$: LVIP 预测的 latent imagery
- $h_y = \mathrm{Enc}_{\mathrm{vis}}(V_y)$: frozen visual encoder 输出的 ground-truth answer embedding
- $\beta$: 平衡系数，control LVIP loss 权重

直觉：standard autoregressive CE loss + auxiliary MSE regression in latent space。LVIP 强制 backbone hidden states 在 option tokens 位置上 encode 足够 recover ground-truth answer 的 visual information。这相当于给 backbone 一个 "answer-oriented grounding" 信号，semantic reasoning 必须跟 visual world modeling 对齐。

**Stage III: RL with Latent Rationales**

这步最有意思。作者用 **Generative Flow Network (GFlowNet)** 做多样性 seeking RL，而不是 standard PPO/GRPO。原因：rationale space 是 combinatorial，可能有 multiple valid reasoning paths 到达同一答案，GFlowNet 能 sample proportional to reward 而不是 collapse 到 single mode。

Target posterior:
$$
p^*(Z | X, y) \propto \exp(R(Z; X, y))
$$

Reward 分两部分：
$$
R(Z; X, y) = \alpha R_{\mathrm{ans}}(Z; X, y) + \gamma R_{\mathrm{lvip}}(Z; X, y)
$$
- $\alpha, \gamma$: 平衡系数
- $R_{\mathrm{ans}}$: answer evidence (discrete)
- $R_{\mathrm{lvip}}$: LVIP grounding (continuous representation)

Answer evidence:
$$
R_{\mathrm{ans}}(Z; X, y) = \log q_{\theta_0}(y | X, Z)
$$
- $q_{\theta_0}$: **frozen scorer** (固定 SFT model 副本，避免 moving-target)

LVIP reward:
$$
R_{\mathrm{lvip}}(Z; X, y) = -\|g_\psi(\bar{\mathbf{h}}_{\mathrm{opt}}(X, Z; \theta)) - h_y\|_2^2
$$
- $g_\psi$: LVIP head (frozen during RL)
- $\bar{\mathbf{h}}_{\mathrm{opt}}(X, Z; \theta)$: backbone conditioned on $[X; Z]$ 后 option tokens 的 average-pooled hidden state
- $h_y$: ground-truth answer image embedding

直觉：LVIP reward 提供一个 **representation-level 的 dense supervision signal**，弥补 discrete answer reward 的 sparsity。当 $Z$ 是好的 rationale，backbone hidden states 在 option tokens 上应该自然 encode 出更接近 ground-truth 的 visual representation。

### 4.3 Token-Wise Marginal Reward Estimation

GFlowNet 需要 prefix-level training signals，但每 token 评估 scorer 太贵。作者用 **sparse anchor + linear interpolation**：

Anchor indices: $t \in \{0, \lambda, 2\lambda, \ldots\}$，$t^+ = \min(t + \lambda, n)$

For $i \in \{0, 1, \ldots, t^+ - t\}$:
$$
\widetilde{R}(\tau_{t+i}) = \widetilde{R}(\tau_t) + \frac{i}{t^+ - t}(\widetilde{R}(\tau_{t^+}) - \widetilde{R}(\tau_t))
$$

变量：
- $\tau_t = (z_{1:t}, \top)$: prefix state at step $t$，$\top$ 表示 termination action 可用
- $t^+$: next anchor
- $i$: intra-segment offset

Anchor 处用公式 (8):
$$
R(\tau_t; X, y) = \alpha \log q_{\theta_0}(y | X, z_{1:t}) + \gamma R_{\mathrm{lvip}}(\tau_t; X, y)
$$

然后用 **SubTB loss** (Madan et al., 2023) 训练 $q_\theta(Z | X)$。

参考 SubTB GFlowNet: https://proceedings.mlr.press/v202/madan23a.html

### 4.4 Reference-Guided GFlowNet Fine-tuning

为降低 low-quality samples 的 variance，用 reference rationale $Z_{\mathrm{ref}}$ 锚定 exploration。对每个 $X$，sample $m$ candidates $\{Z_i\}_{i=1}^m \sim q_\theta(\cdot | X)$，acceptance criterion:

$$
\mathbb{I}(Z_i) = \mathbf{1}[R(Z_i; X, y) \geq R(Z_{\mathrm{ref}}; X, y) + \log \delta_s]
$$

- $Z_{\mathrm{ref}}$: reference (likely SFT-generated rationale)
- $\delta_s \in (0, 1]$: step-dependent slack，$\log \delta_s \leq 0$
- $s$: training step index

Only optimize on accepted trajectories:
$$
\mathcal{L}(\theta) = \sum_{i=1}^m \mathbb{I}(Z_i) \cdot \mathcal{L}_{\mathrm{subTB}}(Z_i; \theta)
$$

### 4.5 Bayesian Inference over Latent Rationales

Inference 时把 $Z$ 当 latent variable，aggregate evidence：

$$
S_i = \frac{1}{|Z_i| + |y_i|} \log q_{\theta_0}(y_i | X, Z_i)
$$

Sample $N$ rationales，output $\hat{y} = y_{i^*}$ where $i^* = \arg\max_i S_i$。

这是 **MAP-style selection over sampled latent rationales**，length-normalized 防止偏好短 rationale。这种 ensemble 思路让我联想到 Self-Consistency CoT (Wang et al., 2022)，但在 latent rationale space 操作。

参考 Self-Consistency: https://arxiv.org/abs/2203.11171

---

## 5. 实验结果深度分析

### 5.1 Main Results (Table 1)

| Model | Fluid | Cryst. | Visuosp. | Mental Simu. | Visual Rout. | Avg |
|-------|-------|--------|----------|--------------|--------------|-----|
| **Human** | 82.7 | 91.3 | 88.5 | 97.9 | 78.7 | **88.4** |
| Gemini 2.5 Flash | 23.2 | 40.2 | 31.0 | 40.2 | 45.3 | 36.3 |
| GPT-o3 | 4.7 | 51.4 | 20.4 | 38.7 | 43.0 | 32.3 |
| GPT-5.2 | 29.4 | 35.9 | 57.5 | 60.0 | 37.6 | 40.3 |
| Claude Sonnet 4 | 22.5 | 31.3 | 26.6 | 58.0 | 34.4 | 32.6 |
| Grok 4 Fast | 13.0 | 45.4 | 41.6 | 21.3 | 37.6 | 31.7 |
| Llama-4-Scout-17B | 20.3 | 29.9 | 35.4 | 48.7 | 41.9 | 31.8 |
| Gemma-3-27B | 18.5 | 29.4 | 39.8 | 55.3 | 43.0 | 32.7 |
| Qwen3-VL-30B | 30.8 | 34.0 | 37.2 | 56.0 | 40.9 | 37.4 |
| **CogSense-8B (Ours)** | **63.8** | **91.0** | **69.0** | **68.0** | **50.5** | **73.8** |

惊人观察：
1. **8B 模型碾压 GPT-5.2 (40.3%) +33.5**，这是巨大的 gap
2. **Crystallized Intelligence 上 CogSense-8B (91.0) 几乎追平 human (91.3)**——这个数据点耐人寻味，可能是 Bongard problems 这类任务 format 比较结构化，model 学到 pattern 后能完美泛化
3. **Fluid Intelligence 上 GPT-o3 只有 4.7**——这非常可疑，可能 GPT-o3 的 reasoning mode 在 abstract pattern matrices 上反而 over-think 导致 fail
4. **所有 frontier models 都远低于 human**——验证了 cognitive gap 假设

### 5.2 General Ability (Table 2)

| Model | HallusionBench | AI2D | GQA | ScienceQA | RealWorldQA | ChartQA | BLINK | MMStar |
|-------|----------------|------|-----|-----------|-------------|---------|-------|--------|
| Qwen3-VL-8B (base) | 61.1 | 85.4 | 71.4 | 92.6 | 71.5 | 88.6 | 64.7 | 70.9 |
| CogSense-8B | 60.5 | 85.1 | 71.8 | 92.6 | 71.9 | 84.7 | 65.3 | 66.8 |

关键发现：**cognitive training 几乎不损害 general ability**。ChartQA 略降 (-3.9)，MMStar 降 (-4.1)，但 GQA、ScienceQA、RealWorldQA、BLINK 反而略升。这说明 LVIP + RL 学到的是 generalizable visual reasoning pattern，不是 task-specific overfitting。

### 5.3 Ablation Study (Table 3)

| Variant | Fluid | Cryst. | Visuosp. | Mental Simu. | Visual Rout. | Avg |
|---------|-------|--------|----------|--------------|--------------|-----|
| Qwen3-VL-8B (base) | 31.2 | 34.8 | 31.0 | 45.3 | 40.9 | 35.5 |
| SFT w/o LVIP | 51.1 | 76.6 | 63.7 | 59.3 | 41.9 | 62.3 |
| SFT w/ LVIP | 55.4 | 88.6 | 61.1 | 61.3 | 44.1 | **68.0** |
| SFT w/o LVIP + GRPO | 55.8 | 79.9 | 63.7 | 63.3 | 43.0 | 65.5 |
| SFT w/ LVIP + GRPO | 59.1 | 89.9 | 64.6 | 65.3 | 46.2 | 70.8 |
| **CogSense-8B (full)** | **63.8** | **91.0** | **69.0** | **68.0** | **50.5** | **73.8** |

深度解读：

1. **SFT 几乎翻倍 base** (35.5 → 62.3)：reasoning chain 数据本身就 effective
2. **LVIP 单独贡献 +5.7** (62.3 → 68.0)：latent imagery grounding 确实提供额外信号
3. **GRPO on SFT w/o LVIP**: 62.3 → 65.5 (+3.2)
4. **GRPO on SFT w/ LVIP**: 68.0 → 70.8 (+2.8)
5. **CogSense-8B (GFlowNet) vs SFT w/ LVIP + GRPO: 70.8 → 73.8 (+3.0)**——这 +3.0 是 GFlowNet vs GRPO 的 gap，证明 **diversity-seeking latent rationale sampling 优于 deterministic policy optimization**

这个 ablation 干净地 isolate 了三个 contributions：data、LVIP、GFlowNet RL。

### 5.4 Out-of-Domain Evaluation (Table 4)

| Model | Chemistry | Math |
|-------|-----------|------|
| Qwen3-VL-8B (base) | 39.2 | 26.0 |
| CogSense-8B | 45.4 (+6.2) | 34.8 (+8.8) |

EMMA benchmark 上的提升确认 visual cognition patterns 的 generalization。

参考 EMMA: https://arxiv.org/abs/2501.05444

---

## 6. 联想与 Intuition Building

### 6.1 跟 System 2 thinking 的关联

Kahneman 的 System 1 / System 2 framework 里，visual imagery 属于 System 2 的 deliberate reasoning。Text-only CoT 试图用 language 模拟 System 2，但 visual cognition 是 **non-verbal System 2**。CogSense 的 LVIP head 本质是给 MLLM 装了个 "non-verbal System 2 module"。

参考 Lake et al. (2017) Building machines that learn and think like people: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/building-machines-that-learn-and-think-like-people/0E3D2F6F8F7C091C6C8F1F0A4E7D6F4B

### 6.2 跟 World Models 的关联

LVIP 预测 answer image 的 latent embedding，本质是在学习一个 **conditional world model**——给定 question + reasoning，预测 answer state 的 visual representation。这跟 World Models (Ha & Schmidhuber, 2018)、Dreamer (Hafner et al., 2023) 思路相通，但在 VQA context 下。

参考 DreamerV3: https://arxiv.org/abs/2301.04104

### 6.3 跟 Latent Reasoning trend 的关联

最近 latent reasoning 方向很火：
- **Latent Visual Reasoning (LVR)** (Li et al., 2025a): https://arxiv.org/abs/2509.24251
- **Implicit reasoning tokens** (Li et al., 2025c): https://arxiv.org/abs/2512.21218
- **Latent CoT for visual reasoning** (Sun et al., 2025): https://arxiv.org/abs/2510.23925
- **Cambrian-S** (Yang et al., 2025b): https://arxiv.org/abs/2511.04670

这篇 paper 的独特之处：**显式 supervision target (ground-truth answer image embedding) + GFlowNet diversity-seeking RL**。其他工作多用 self-supervised 或 next-token prediction 在 latent space。

### 6.4 GFlowNet 选择的意义

为什么不用 PPO/GRPO？我推测：
1. **Multi-modal reasoning problem 可能有 multiple valid rationales**——GFlowNet sample proportional to reward，能 maintain diversity
2. **Long rationale trajectory 的 credit assignment 难**——SubTB loss with densified prefix rewards 解决这个问题
3. **Avoid mode collapse**——standard RL 容易 collapse 到 single high-reward trajectory，GFlowNet 保持 distribution

参考 GFlowNet foundations: https://arxiv.org/abs/2111.06477

### 6.5 Limitations 我看到的

1. **LVIP 需要 ground-truth answer image**——意味着只能 train 在 multiple-choice with image options 的 tasks。Open-ended generation 上不直接适用
2. **Two-layer MLP for LVIP head** 太简单——可能限制 latent imagery 的 expressiveness
3. **Frozen scorer $q_{\theta_0}$** 在 RL 阶段是 fixed，可能 stale
4. **Inference 时 sample N rationales** 增加 compute cost，没有讨论 latency tradeoff
5. **Bongard-LOGO 上 CogSense-8B 还远低于 human (69.0 vs 88.5)**——visuospatial cognition 仍是 bottleneck

### 6.6 跟 ARC-AGI 的关联

ARC-AGI (Chollet, 2019) 一直被认为是 visual reasoning 的 holy grail。CogSense 在 ARC-AGI / ARC-AGI-2 上做训练和评测，但没 report 单独 ARC-AGI 数字。这可能是 future work 的方向——如果 LVIP-style 方法能在 ARC-AGI 上突破，会是里程碑。

参考 ARC-AGI-2: https://arxiv.org/abs/2505.11831
François Chollet On the Measure of Intelligence: https://arxiv.org/abs/1911.01547

### 6.7 跟 Visual Sketchpad 的对比

Hu et al. (2024) Visual Sketchpad 让 model 生成 actual visual sketches 作为 CoT。这是 "explicit visual reasoning"，而 CogSense 是 "latent visual reasoning"——不生成可解释的 visual intermediate，直接在 latent embedding space 操作。

权衡：latent 更 efficient，但 loses interpretability；explicit 更 interpretable 但 expensive。

参考 Visual Sketchpad: https://arxiv.org/abs/2406.09285

### 6.8 关于 Mental Imagery 的 cognitive science grounding

Paper 提到 vividness of visual imagery questionnaire (VVIQ)——人有不同 vividness 的 mental imagery。MLLMs 现在通过 LVIP 获得 "low-vividness imagery"（latent embedding level），未来可能向 "high-vividness imagery"（pixel-level generation）演进。

参考 Tabi et al. (2022): https://www.sciencedirect.com/science/article/pii/S0010945221004018

---

## 7. 关键 Takeaways

1. **Text is brittle interface for visuospatial reasoning**——核心 thesis，well-argued
2. **Latent visual imagery as reasoning substrate**——通过 LVIP head 实现，简单但 effective
3. **GFlowNet > GRPO for rationale sampling**——diversity-seeking 优于 deterministic optimization，+3.0 gap
4. **8B 模型 + cognitive training > 30B frontier models**——specialized training 比 scale 更 efficient in this domain
5. **OOD generalization confirmed**——visual cognition patterns 可迁移到 math/chemistry VQA

---

## 8. 进一步阅读建议

- Cao et al. (2025) MaRs-VQA: https://arxiv.org/abs/2406.10424
- Yang et al. (2025a) Thinking in Space: https://arxiv.org/abs/2412.14171
- Li et al. (2025b) Imagine while reasoning in space: https://arxiv.org/abs/2501.07542
- Yiu et al. (2025) KiVA: https://openreview.net/forum?id=HioVg0xM3G
- Nie et al. (2020) Bongard-LOGO: https://arxiv.org/abs/2010.13531

---

最后一点 meta 思考：这篇 paper 在某种意义上是 "认知科学 + deep learning" 的 bridge work。LVIP head 设计简单到只有 two-layer MLP，但 ground 在 cognitive science theory（visuospatial sketchpad、mental imagery）让 design choice 有理论支撑。我觉得这种 theory-grounded method design 是值得借鉴的 pattern——不是 chasing SOTA with architectural complexity，而是 minimal intervention grounded in cognitive principles。

期待看到 LVIP 思路 extend 到 open-ended generation、video reasoning、robotics planning。视觉 cognitive supersensing 这条路线刚起步。
