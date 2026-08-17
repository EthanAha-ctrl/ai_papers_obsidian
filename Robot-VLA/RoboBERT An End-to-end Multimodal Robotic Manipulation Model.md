---
source_pdf: RoboBERT An End-to-end Multimodal Robotic Manipulation Model.pdf
paper_sha256: f06933fa5a042680e28d92ad3479bab2dad1a79d1e1d82388adf312a5499a8e8
processed_at: '2026-08-12T00:17:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboBERT 人话版

## 一句话概括

这帮人发现：**别一上来就让 robot 又学看、又学听、又学动，会学崩。先让它闭嘴学会动手，再教它听懂各种话，效果反而更好。** 208M 参数，两张 3090，14 小时，干翻一堆 1B 参数、用 2.7TB 数据预训练的 SOTA。

---

## 为什么这事难

VLA (Vision-Language-Action) 现在的主流玩法是：拿一个巨大的 VLM (比如 PaLM-E、PaLI)，往里灌海量 robot trajectory 数据 fine-tune。RT-2 用 55B 参数，OpenVLA 用 7B，π0 更夸张。问题是：

1. **Robot action data 极度稀缺** — internet 上 text + image 无限，但 "看到这个画面，机械臂该往哪挪" 这种标注数据，撑死了几万条
2. **机器人平台异构** — UR5 和 Franka 的 joint space 完全不同，数据没法混用
3. **算力门槛吓人** — 64 张 A100 训一周，普通 lab 玩不起

所以核心问题：**给你有限的数据和算力，怎么榨干每一滴信息？**

---

## 核心直觉：Curriculum 解耦

这让我想到教小孩打棒球。你不会一上来跟三岁小孩说 "当对方投出 slider 时，你应该调整 bat angle 到 45 度并延迟 swing timing"。你会先让他学会挥棒这个动作本身，等动作稳了，再教他识别球种。

RoboBERT 就是这个思路，但用在 VLA 上。

### Stage 1：闭嘴，先学动手

- Language input 固定成模板化的 "standard language"（比如永远说 "pick up the red block"，不说 "grab the crimson cube"）
- Vision encoder (CLIP ViT) 冻住，只解冻最后一层
- 只训练 language connector + fusion + diffusion policy

为什么？因为同时学三件事太难了：
- Vision 要学 "看到什么"
- Language 要学 "听懂什么"
- Policy 要学 "怎么动"

三个 gradient signal 互相干扰，模型在 loss landscape 里像无头苍蝇。固定 language 等于把问题从三维降到二维，模型只需学 "observation → action" 这个核心 mapping，language 只是个 task label。

而且冻 ViT 有两个好处：
1. CLIP 在 internet 上已经学好 visual concept，不用重新学
2. 冻住大部分层 = backward 只到最后一层 = GPU 省一大半 memory

**Stage 1 每 epoch 40 分钟，10 个 epoch，7 小时搞定。**

### Stage 2：现在教它听懂人话

Stage 1 跑完，模型已经会 "看到 red block 就抓起来" 了。现在把 ViT 全解冻，language 从单一模板换成各种自然语言 paraphrase：
- "grab the red cube"
- "go get that ruby thing"
- "please snatch the crimson block"

这时候 loss 已经很低了（Stage 1 训练好的 policy 基本对了），Stage 2 的 gradient 很小，只是在 language manifold 上做微调，把不同说法 align 到已经学好的 policy 上。不会把 Stage 1 的 weights 推崩。

**Stage 2 每 epoch 90 分钟，5 个 epoch，7.5 小时搞定。**

---

## 消融数据是最有说服力的

看 Table 3a，这个对比真的 dramatic：

| 方法 | ABC→D Avg. Length |
|---|---|
| 直接用 natural language 训 | **1.25** |
| Two-stage 训 | **3.79** |

1.25 是什么概念？CALVIN 是 5 个连续 task，随机猜大概 1.0 左右。所以直接用 NL 训基本等于瞎猜。而 two-stage 是 3.79，**差了 3 倍**。

这说明什么？**不是 "natural language 让模型学不会"，是 "同时学 NL 理解 + policy 会让两者都学不好"。** 解耦之后，NL 反而学得更快，因为 policy 已经稳定了，只需要做 alignment。

---

## Data Augmentation 的故事

这部分让我特别想 build intuition，因为它揭示了 robot learning 和一般 CV 的根本差异。

### Color Jitter 最有效 (+0.65)

CALVIN 的 ABC 和 D 是不同环境，颜色分布不同。如果模型死记 "红色 block 要抓"，换到 D 环境发现 block 是蓝色的就懵了。Color jitter 强制模型学 **shape-based** representation — "管你红蓝，是个 cube 就抓"。

这本质上是 **color domain randomization**，让 model 对颜色 invariant，只关注 task-relevant 的 geometry。

### Affine Transformation 最有害 (-0.25)

这个结果最反直觉，但想想 robot control 的本质就明白了。

一般 image classification 里，translation invariance 是好事 — cat 在左上角还是右下角都是 cat。但 robot control 里，**object 在 image 中的位置直接决定 action**。block 在左边，gripper 就要往左移；block 在右边，gripper 往右移。

你做 affine translation 把 block 往右挪了 15%，但 ground truth action 还是基于真实位置（block 在左边）。模型看到 "block 在右边" 但被告知 "应该往左抓"，这 supervised signal 直接矛盾，模型当然学崩。

更深层：robot 的 eye-in-hand camera 是动的，pixel coordinates 到 world frame 的对应关系是 dynamic 的，affine augmentation 打破了这个对应。

**这件事的 takeaway：robot learning 里 augmentation design 必须 task-aware，不能照搬 CV 的套路。**

### Mixup 想法挺有意思

传统 mixup 是 image classification 里 $x_{mix} = \lambda x_0 + (1-\lambda) x_1$。RoboBERT 把它扩展到 robot：

- 两个 demo 的 RGB frame 做线性插值
- 对应的 language token 也插值
- 对应的 action vector 也插值

$$
x_{mix} = \lambda x_0 + (1-\lambda) x_1, \quad y_{mix} = \lambda y_0 + (1-\lambda) y_1
$$

其中 $\lambda \sim \text{Beta}(0.4, 0.4)$，这个分布是 U 型的，大部分时候 $\lambda$ 接近 0 或 1，也就是说大部分 mixup sample 还是接近原 sample 之一，避免过度混乱。

效果 +0.23，不算大但不亏。更有意思的是它学到了 **trajectory space 上的线性 interpolation** — "从抓杯子到开抽屉之间有连续过渡"，这在 action space 上是某种 implicit smoothing。

### 最佳组合

去掉有害的 Affine，剩下全用：+0.79。不是简单加和，说明 augmentation 之间有 interaction，但去掉有害的之后 synergy 出来了。

---

## 架构为什么这么搭

### 为什么 BERT 不用 LLaMA？

1. BERT 110M，轻量
2. BERT 是 encoder-only，适合 "判断两个句子是不是一个意思" — 这正好是 task-conditioned policy 需要的
3. LLM decoder 推理慢，需要 autoregressive，robot control 要实时性
4. Robot 不需要生成话，只需要听懂话，encoder 够用

### Language Connector 用 Perceiver Resampler

BERT 输出是变长 token sequence，可能 10 个 token 也可能 30 个。下游 fusion 需要固定维度。Perceiver Resamer 用一组可学习 latent query 对 BERT 输出做 cross-attention，压缩成固定数量的 token。这是 Flamingo 的经典操作。

### Fusion 用 Language 当 Query

Cross-attention 里 language 是 Q，vision 是 K/V。这背后的 intuition 很重要：

> 给定 instruction，从 observation 中检索 task-relevant 信息。

如果反过来 vision 当 query，就变成 "visual question answering"，语义就不对了。Robot policy 的本质是 instruction-conditioned，language 是 query 才对。

### Diffusion Policy 为什么必要

直接回归 $\hat{A} = f(L, V)$ 的问题：**multimodal action distribution**。

同一个 "抓杯子" 任务，从左边抓和从右边抓都是对的。MSE regression 会让模型输出这两个的平均 — 一个无效的中间轨迹，机械臂插进桌子。

Diffusion policy 学的是整个 distribution 的 score function，sample 时能产生 valid mode 之一。这是为什么所有 SOTA VLA 基本都用 diffusion 或 flow matching。

Training 目标是预测 noise：

$$
\Delta\theta = \alpha \frac{\partial \left( M_{policy-\theta}(A^0 + \epsilon^k, k \mid M_{fusion-\theta}(\cdot)) - \epsilon^k \right)^2}{\partial(L, V)}
$$

- $A^0$: clean action
- $\epsilon^k$: 第 k 步加的高斯噪声
- $k$: diffusion timestep
- 模型学的是 "给我被污染的 action 和 timestep k，预测加了什么 noise"
- 推理时从纯噪声开始，iteratively denoise 到 clean action

---

## 实验结果解读

### ABCD→D (Table 1)

RoboBERT 4.52，干翻所有 baseline。最有意思的对比：

- **RoboFlamingo 1000M, 4.08** — 用了 LLaMA scale 的 backbone，还是输了
- **GR-1 130M, 4.21** — 用了 2.7TB 感官数据预训练 + proprioception，还是输了
- **MoDE 436M, 4.39** — 用了 mixture of experts + 额外数据，接近但输了

RoboBERT 208M，没 pretrain，没用 proprioception，还是赢了。**这证明架构 design + training paradigm 比 brute-force scaling 重要。**

### ABC→D (Table 2) — 泛化测试

ABC→D 是更难的 setting，D 环境训练时完全没见过。

RoboBERT 从 4.52 跌到 3.79，掉 16%。
RoboFlamingo 从 4.08 跌到 2.47，掉 40%。

RoboBERT 衰减更慢，说明 **color jitter 做的 domain randomization 起作用了**，模型学到 environment-agnostic feature。

### Real Robot (Table 4)

6-DOF RM65B，每个 task 只 25-30 条 demo。对比 RT-1 和 MT-ACT，RoboBERT 全胜。

Sequential task 上优势更明显（86% vs 72% vs 61%），说明 two-stage 学到的 policy 更稳定，long-horizon 不容易 drift。

---

## 让我联想到的事

### 1. 这跟 LLM 的 pretrain → SFT → RLHF 是同构的

LLM 先在 internet 上学 stable language representation，再 SFT align 到 instruction，再 RLHF align 到人类偏好。RoboBERT 是先学 stable policy，再 align 到 natural language。**Curriculum learning 在不同 modality 上都 work，这是个 universal principle。**

### 2. Affine 失败让我想到 sim-to-real 的核心矛盾

Sim-to-real 一直有个矛盾：domain randomization 要 randomize 什么？
- Randomize lighting, color — 好，这是 task-irrelevant
- Randomize object position — 看情况，如果 task 是 "抓起那个东西"，position 是 task-relevant
- Randomize physics (friction, mass) — 好，model 应该对 physics invariant

RoboBERT 的 affine 失败本质上是 **randomize 了 task-relevant 维度**。正确做法是 randomize task-irrelevant 维度，保持 task-relevant 维度不变。如果要做 gripper-aware augmentation（动 object 不动 gripper），应该 segment object region 只对 object 做 affine，gripper 保持原位。

### 3. 对比 π0 的 flow matching

π0 用 flow matching 而不是 diffusion，在 continuous action space 上更高效。RoboBERT 用 CNN-based diffusion，简单但够用。如果 RoboBERT 把 diffusion 换成 flow matching，可能训练更快、sample 质量更高。这是个 obvious next step。

参考：https://arxiv.org/abs/2410.24164

### 4. Action tokenization vs Diffusion 的路线之争

OpenVLA 把 action tokenize 成离散 token，用 LLM autoregressive 生成。好处是可以用 vLLM 等推理加速生态。坏处是连续 action 被量化，精度受限。

RoboBERT 走 diffusion 路线，连续 action space，更自然但推理慢。两条路线会长期共存，最终可能融合 — flow matching + tokenization，或 diffusion + KV cache。

参考：https://openvla.github.io/

### 5. Two-stage 能不能再激进点

Stage 1 用 "standard language" 还是有点 cheating — 毕竟还是用了 language label。更激进的方案：

- Stage 1 完全 unsupervised motor learning，比如 world model pretrain (DreamerV3 风格)
- Stage 2 才引入 language conditioning

这样连 standard language 都不需要，真正 zero-language motor pretrain。参考：https://arxiv.org/abs/2301.04104

### 6. 3D Diffuser Actor 输给 RoboBERT 的启示

3D Diffuser Actor 用 point cloud + camera parameter，有 3D 信息，ABC→D 只 3.27。RoboBERT 用 2D RGB，3.79。

直觉上 3D 应该更好，为什么输了？我猜：
- CALVIN 的 3D 信息质量有限，multi-view stereo 不够准
- 2D RGB + 强 augmentation 反而更 robust
- 3D pipeline 复杂，训练效率低，14 小时 vs 3D Diffuser Actor 可能要几天

**这暗示 2D + good inductive bias + good augmentation 可能比 3D 更 cost-effective。**

### 7. 一个让我兴奋的 hallucination

如果 RoboBERT 的 two-stage 范式 + data augmentation 套到 cross-embodiment 上：
- Stage 1: 在 Platform A 上训好 policy
- Stage 2: 用 Platform B 的少量数据 fine-tune，用类似 LoRA 的 adapter

这样可能实现真正高效的 cross-embodiment transfer，解决 robot data 稀缺的根本问题。现在的 Open X-Embodiment 是暴力收集多平台数据，但没利用 curriculum 解耦。

参考：https://robotics-transformer-x.github.io/

---

## 最终 Takeaway

RoboBERT 的核心贡献其实只有一个 insight：**VLA 训练不要 all-at-once，要 curriculum。**

这个 insight 跨越了具体架构选择 — 你可以把 BERT 换成 LLaMA，把 CNN diffusion 换成 transformer diffusion，把 CLIP 换成 DINOv2，只要保持 "先 stable policy，后 language alignment" 的 curriculum，都能 work。

而 data augmentation 的发现（affine 有害、color jitter 极有效）更是给整个 robot learning 社区提了个醒：**CV 的 augmentation 套路不能照搬，必须 task-aware。**

208M 参数、14 小时训练、两张 3090、SOTA。这是 resource-constrained robotics lab 的 dream paper。

代码：https://anonymouskonto.github.io (匿名 review 版，正式发布后应该在作者主页)

---

## 我的批评

1. **Stage 1 还是需要 standard language label** — 不够 unsupervised，真正的 zero-label motor learning 没探索
2. **没用 proprioception** — GR-1 用了 proprioception 显然有效，RoboBERT 没用，可能是 precision task 上的 bottleneck
3. **Mixup language 可能 damage 语义** — 两个不同 task 的 language token 插值后是无意义 sentence，虽然 Beta(0.4,0.4) 偏向极端，但还是有点 hacky
4. **没测 long-horizon beyond 5 task** — CALVIN 是 5 个 task chain，真实场景可能 50 个，diffusion policy 的 chunk-based inference 在超长 horizon 上还没验证
5. **Affine 失败没深挖** — 只解释了 spatial ambiguity，没尝试 gripper-aware augmentation 这种 obvious 修复

---

## 一句话总结

**别让机器人又学听又学动，先学动再学听，给它点好 curriculum，小模型也能干翻大模型。**

这个 insight 简单到有点 embarrassing — 但这就是好 paper 的特点，让你看完想拍大腿说 "我早该想到的"。

---

# RoboBERT 技术深度讲解

## 1. 核心定位与 Motivation

RoboBERT 是一个 end-to-end language-conditioned VLA (Vision-Language-Action) model, 它的核心 contribution 是 **two-stage training paradigm** + **systematic data augmentation**, 在不依赖大规模 robotics 数据预训练的前提下, 用 208M 参数量在 CALVIN benchmark 上达到了 SOTA (ABCD→D: 4.52, ABC→D: 3.79)。

作者要 build 的核心 intuition 是: 当前主流 VLA 模型 (RT-2, OpenVLA, π0, GR-1) 都依赖 internet-scale pretrain + 大量 robot data fine-tuning, 但 robot action modality 数据本身稀缺、平台异构、计算开销巨大。所以与其暴力 scaling, 不如把 training curriculum 与 modality alignment 解耦, 让有限 data 被更高效地利用。

参考:
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- Foundation Models in Robotics 综述: https://arxiv.org/abs/2312.07843

---

## 2. Problem Formulation 数学拆解

论文给出的目标函数:

$$
\theta = \arg\min_{\theta} \left( M_{\theta}(\mathbf{L}, \mathbf{V}) - \mathbf{A} \right) = \arg\min_{\theta} \left( \hat{\mathbf{A}} - \mathbf{A} \right)
$$

变量解释:
- $\theta$: 模型所有可训练参数
- $\mathbf{L} \in \mathbb{R}^l$: language instruction 向量, $l$ 是 token 序列长度
- $\mathbf{V} \in \mathbb{R}^{t \times c \times h \times w}$: observation tensor
  - $t$: time steps (论文用 1-2 frame)
  - $c$: channels (RGB = 3)
  - $h, w$: image height/width
- $\mathbf{A} \in \mathbb{R}^a$: ground truth action, $a$ = action dimension (CALVIN 是 7-DOF: 6 DOF pose + 1 gripper)
- $\hat{\mathbf{A}}$: model 预测 action

模型被分解为三个子模块:

$$
M_{\theta}(\mathbf{L}, \mathbf{V}) = M_{policy-\theta}\left( M_{fusion-\theta}\left( M_{ext-L\theta}(\mathbf{L}), M_{ext-V\theta}(\mathbf{V}) \right) \right)
$$

- $M_{ext-L\theta}$: language encoder + language connector
- $M_{ext-V\theta}$: vision encoder (CLIP ViT)
- $M_{fusion-\theta}$: cross-modal fusion (transformer decoder without causal mask)
- $M_{policy-\theta}$: diffusion action head

这种 decomposition 让人想到经典的 "encoder-fusion-decoder" 范式, 关键 design choice 在每一段。

---

## 3. 架构细节 (对应 Figure 1)

### 3.1 Language Branch: BERT (110M) + Language Connector

为什么用 BERT 而不是 LLaMA 系列? 
1. **参数量考量**: BERT-base 只有 110M, 与 CLIP ViT 87M 相当, 总参数控制良好
2. **任务匹配**: BERT 是 encoder-only, 适合 sentence understanding / paraphrase inference, 这正好对应 "区分不同 language instruction" 的需求
3. **效率**: 比 decoder-only LLM 推理快, 不需要 autoregressive generation

Language Connector 用的是 **Perceiver Resampler** (出自 Flamingo, https://arxiv.org/abs/2204.14198), 核心作用是把 BERT 输出的变长 token sequence 压缩成固定数量 latent tokens。Perceiver Resamer 通过一组可学习的 latent queries 对 BERT 输出做 cross-attention, 输出 $N_q$ 个 tokens (通常 $N_q \ll l$), 这样下游 fusion 计算量可控。

参考:
- BERT: https://arxiv.org/abs/1810.04805
- Flamingo Perceiver Resampler: https://arxiv.org/abs/2204.14198

### 3.2 Vision Branch: CLIP ViT (87M)

ViT 通过 image-text contrastive 预训练, 提供了 vision-language aligned representation, 这是关键初始化。论文特别强调 **Stage 1 freeze ViT 除了最后一层**, 这背后 intuition:
- CLIP 在 internet-scale 数据上学到的视觉概念 已经足够 general
- 完全 unfreeze 容易 catastrophic forgetting
- 只 unfreeze 最后一层让 ViT 适配 robotic scene 的特殊分布 (gripper camera 视角、indoor scene)
- Frozen backbone 让训练更稳定, 计算 gradient 只需 backward 到最后一层, 显著省 GPU memory

参考:
- ViT: https://arxiv.org/abs/2010.11929
- CLIP: https://arxiv.org/abs/2103.00020

### 3.3 Modality Fusion: Transformer Decoder (without causal mask)

这是借鉴 OpenFlamingo 的设计 (https://arxiv.org/abs/2308.01390)。具体 attention pattern:
- **Query**: language tokens
- **Key/Value**: vision tokens (gripper camera + static camera)
- **Self-attention**: 在 language tokens 之间
- **Cross-attention**: language query vision

为什么 language 当 query? 因为模型本质是 "instruction-conditioned policy": 给定 instruction, 从 observation 中检索 task-relevant 信息。如果反过来 vision 当 query, 就变成 "visual question answering", 与 control 任务语义不匹配。

最后用 **max-pooling** 压缩 latent semantic tokens 成单一 vector, 喂给 diffusion head。这里 max-pooling 比 mean-pooling 更激进, 但在 task-conditioned 设定下, max 倾向于保留 task-relevant 的 dominant signal, 对噪声更鲁棒。

### 3.4 Action Head: CNN-based Diffusion Policy

CNN-based diffusion policy 来自 Chi et al. (https://diffusion-policy.cs.columbia.edu/), 与 transformer-based diffusion policy 相比, CNN 版本对 action sequence 的 local pattern (smoothness, temporal coherence) 更敏感, 计算也更轻。

Diffusion policy training 目标:

$$
\Delta\theta = \alpha \frac{\partial \left( M_{policy-\theta}(\mathbf{A}^0 + \epsilon^k, k \mid M_{fusion-\theta}(\cdot)) - \epsilon^k \right)^2}{\partial(\mathbf{L}, \mathbf{V})}
$$

变量解释:
- $\mathbf{A}^0$: unpolluted (clean) expert action
- $\epsilon^k \sim \mathcal{N}(0, I)$: 第 $k$ 步添加的 Gaussian noise
- $k \in \{1, 2, ..., K\}$: diffusion timestep, $K$ 是 total diffusion steps
- $M_{policy-\theta}$: 学一个 denoising network, 预测当前 step 加的噪声 $\epsilon^k$
- Conditioning: $M_{fusion-\theta}(\cdot)$ 提供 task context, 告诉 denoiser "你在 denoise 哪个任务的 action"

**关键 intuition**: Diffusion policy 相对直接回归 $\hat{\mathbf{A}} = f(\mathbf{L}, \mathbf{V})$ 的好处是能建模 **multimodal action distribution**。同一个 task 可能有多个 valid trajectory (例如抓杯子可以从左边也可以从右边), MSE regression 会让模型输出所有可能轨迹的平均, 在多模态分布上这就是无效 action。Diffusion 通过 score matching 学到整个 distribution, sampling 时能产生 valid modes。

Inference 时从 $\mathbf{A}^K \sim \mathcal{N}(0, I)$ 出发, iteratively denoise 到 $\mathbf{A}^0$, 取最后预测的 chunk 作为当前 action, 然后等待新 observation 重复。

参考:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DDPM: https://arxiv.org/abs/2006.11239

---

## 4. Two-Stage Training Paradigm (核心 contribution)

这是论文最值得 build intuition 的部分。

### 4.1 Stage 1: Stable Policy Learning

```
Trainable: {M_ext-Lθ, M_fusion-θ, M_policy-θ}
Frozen:    M_ext-Vθ (except last ViT layer)
Language:  {standard language}  # 单一固定表述
```

CALVIN dataset 中, 每个 demo 同时有 "standard language" (模板化, 如 "pick up the red block") 和 "natural language" (多样化, 如 "grab the crimson cube", "go ahead and snatch that ruby thing")。Stage 1 只用 standard language, 把 instruction 当作 "task ID"。

为什么这样做? 同时学 (a) visual perception, (b) language understanding, (c) action policy 三件事太难, gradient signal 互相干扰。固定 language 让模型先学 "observation → action" 这个 core mapping, language 只起 task conditioning 作用, 不需要 resolve linguistic ambiguity。Frozen ViT 进一步减低优化维度, 让训练更快收敛 (Stage 1 每 epoch 仅 40 分钟)。

### 4.2 Stage 2: Language Alignment

```
Trainable: {M_ext-Lθ, M_ext-Vθ, M_fusion-θ, M_policy-θ}  # all
Language:  {standard language, natural language}
```

Stage 1 后, model 已经有稳定的 policy $\pi(\cdot \mid \text{standard instruction}, \mathbf{V})$。Stage 2 引入 natural language variants, 目标是让 $\pi(\cdot \mid \text{natural instruction}, \mathbf{V}) \approx \pi(\cdot \mid \text{standard instruction}, \mathbf{V})$。

为什么不会破坏 stage 1 学到的 policy?
1. **Loss 已经很小**: Stage 1 训练后, 标准指令的 BC loss 很低, Stage 2 fine-tune 的 gradient magnitude 自然小
2. **Curriculum effect**: 模型已经在正确的 loss landscape 中, Stage 2 只是 expand language manifold, 不会把 weights 推到完全不同的 basin
3. **All unfreeze**: 此时 ViT 也可以微调, 让 vision encoder 适配更广泛的 visual context (虽然 stage 1 已经够用)

### 4.3 Ablation 验证 (Table 3a)

| Training Method | Dataset | Avg. Length |
|---|---|---|
| NL. directly | ABCD→D | 3.39 ± 0.05 |
| **Two-stage** | **ABCD→D** | **4.52 ± 0.03** |
| NL. directly | ABC→D | 1.25 ± 0.04 |
| **Two-stage** | **ABC→D** | **3.79 ± 0.03** |

直接用 natural language 训练, ABC→D 直接掉到 1.25 (几乎随机), 这是非常 dramatic 的差距。**说明 multi-modal alignment 同时学与 policy 学习是高度 entangled 的**, 解耦后效果提升巨大。

这让我想到 curriculum learning 的研究, 以及 LLM 中 "先学会 reasoning pattern, 再 align to instruction" 的思路。Two-stage 范式本质是 **decoupling perceptual-motor learning from linguistic generalization**。

参考:
- CALVIN benchmark: https://calvinrobot.github.io/
- Instruction tuning: https://arxiv.org/abs/2103.10360

---

## 5. Data Augmentation 详解

这部分对应 Figure 3 与 Table 3b。作者用 ABC→D 做 ablation。

### 5.1 Salt-and-Pepper Noise

随机把 pixel 置为 0 或 255, 模拟 impulse noise。Augmentation strength: SNR = 0.95 (5% pixel 被污染)。Intuition: 强制模型学 holistic feature, 避免 over-reliance 单个 pixel, 类似 dropout 在 pixel level 的作用。

ABC→D Avg. Length: 3.00 → 3.22 (+0.22)

### 5.2 Affine Transformation (有害!)

Translation amplitude = 15%。结果: 3.00 → 2.75 (**-0.25**)

为什么有害? 作者的解释 (非常关键的 intuition): **spatial ambiguity**。Robot manipulation 中, gripper camera 是移动的 (eye-in-hand), pixel coordinates 与 robot world frame 的对应关系是动态的。Affine translation 破坏这种对应, 模型无法判断 "是 cube 真的移到了右边, 还是只是 augmentation 让它看起来在右边"。

这与一般 vision task 不同 - 一般 image classification, translation invariance 是好事; 但在 robot control 中, spatial location 本身是 action-relevant signal。Gripper 应该往哪移动取决于 object 在 image 中位置, 不能 invariance。

### 5.3 Color Jitter (最有效!)

HSV jitter (hue, saturation, value 各 ±0.4)。结果: 3.00 → 3.65 (**+0.65**)

为什么如此有效? 作者指出: "High-frequency features—edges, contours, shapes, and textures—carry more task-relevant information than low-frequency attributes like color or illumination." 这跟 YOLO 训练时用 HSV jitter 思路一致 (https://arxiv.org/abs/1506.02640)。

更深层原因: CALVIN 训练 set 与 test set 的颜色分布存在 shift (不同 layout, 不同 object color), robot 在真实场景中也会遇到不同光照、阴影。Color jitter 本质上做 **domain randomization on color**, 让 model 学 shape-based representation, 这种 representation 在 cross-domain transfer 时更 robust。

ABC→D 是 cross-environment generalization (ABC 训练, D 测试), D 环境颜色与 ABC 不同, 所以 color jitter 收益最大, 这印证了上述 hypothesis。

### 5.4 Robotic Mixup

原版 Mixup (https://arxiv.org/abs/1710.09412):

$$
x_{mix} = \lambda x_0 + (1-\lambda) x_1, \quad y_{mix} = \lambda y_0 + (1-\lambda) y_1
$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$, $\alpha = 0.4$ (相对尖锐的分布, 偏向 0 或 1)。

变量:
- $x_0, x_1$: 两个不同 demo 的 input (RGB frames + language tokens)
- $y_0, y_1$: 对应的 action vectors
- $x_{mix}, y_{mix}$: 合成的新 sample

**特别之处**: 同时对 **RGB frames、language tokens、action vectors** 做 mixup, 而传统 mixup 只对 input 和 label 做。这意味着模型学到在 trajectory space 上的 linear interpolation, 类似于 "在两个 task 之间连续 transition"。

ABC→D 结果: +0.23

但这里有个 concern: mixup language tokens 是否产生无意义 sentence? 例如 "pick up the red" + "open the drawer" 混合后语义混乱。不过作为 regularization, 它确实让 model 学到 robust representation。Beta(0.4, 0.4) 偏向极端, 实际上大部分 sample 接近原 sample 之一, 这避免了过度混乱。

### 5.5 Combining All w/o Aff (最佳组合)

3.00 → 3.79 (**+0.79**)

把所有 augmentation 组合, 但去掉有害的 Affine, 总收益大于单一 augmentation 之和? 不, 论文指出 "improvement is undoubtedly higher than single ones but not simply additions of each increment"。

这是 augmentation 之间的 **interaction effect** - 它们作用在不同的 inductive bias 上 (color, noise, mixup), 组合有 synergy, 但也可能 interference (例如 color jitter + mixup 同时破坏 color 信息)。

---

## 6. 主实验结果深度分析

### 6.1 ABCD→D (Table 1)

| Model | Pretrain | Params | Avg. Length |
|---|---|---|---|
| HULC | N | 100M | 3.06 |
| RoboFlamingo | Y | 1000M | 4.08 |
| DeeR | Y | 1000M | 4.13 |
| GR-1 | Y | 130M | 4.21 |
| MoDE (w/ per.) | Y | 436M | 4.39 |
| **RoboBERT** | **N** | **208M** | **4.52** |

关键观察:
1. RoboFlamingo 与 DeeR 都是 ~1B 参数, RoboBERT 208M 远小但更好 - **架构 + training paradigm 比单纯参数量更重要**
2. GR-1 用了 2.7TB sensory data pretrain, 还加了 proprioception modality, 仍然输给只用 language-labeled demo 的 RoboBERT
3. MoDE 用 mixture-of-experts (436M), 有 extra data, 只比 RoboBERT 略低

### 6.2 ABC→D (Table 2) - Generalization 测试

ABC→D 是更难的 setting: D environment 在训练中完全没出现, 测 zero-shot generalization。

| Model | Avg. Length |
|---|---|
| RoboFlamingo | 2.47 |
| DeeR | 2.82 |
| GR-1 | 3.06 |
| 3D Diffuser Actor | 3.27 |
| MoDE (w/o per.) | 3.39 |
| **RoboBERT** | **3.79** |

RoboBERT 在 cross-environment 上优势更明显 (4.52 → 3.79 衰减 16.6%; 而 RoboFlamingo 4.08 → 2.47 衰减 39.5%)。这说明 **data augmentation (尤其 color jitter) 起到了关键的 domain randomization 作用**, 让 model 学到 environment-agnostic feature。

参考:
- GR-1: https://arxiv.org/abs/2312.13139
- MoDE: https://arxiv.org/abs/2412.12953
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- RoboFlamingo: https://roboflamingo.github.io/
- DeeR-VLA: https://arxiv.org/abs/2411.02359

---

## 7. Real Robot Experiments (Table 4)

6-DOF RM65B arm, 三类 individual task + 四类 sequential task, 对比 RT-1 与 MT-ACT。

| Model | Trans. D. | Trans. C. | Open D. | Close D. | Stack Cube | Trans. P. | Open Door |
|---|---|---|---|---|---|---|---|
| MT-ACT | 72% | 68% | 73% | 80% | 72% | 73% | 78% |
| RT-1 | 61% | 56% | 64% | 72% | 65% | 60% | 72% |
| **RoboBERT** | **86%** | **87%** | **80%** | **92%** | **90%** | **80%** | **82%** |

每个 task 仅 25-30 trajectory (非常少), GPT 生成 language paraphrase 增加语言多样性。RoboBERT 在所有任务上都胜出, sequential task 优势更明显 (说明 long-horizon 任务上 two-stage training 学到的 policy 更稳定)。

参考:
- RT-1: https://arxiv.org/abs/2212.06817
- MT-ACT (RoboAgent): https://arxiv.org/abs/2309.01918

---

## 8. Compute Efficiency

- 硬件: 2 × RTX 3090 24GB
- Stage 1: ABCD→D, 10 epochs × 40 min = ~6.7 hours
- Stage 2: ABCD→D, 5 epochs × 90 min = ~7.5 hours
- Total: ~14 hours on 2 GPUs

对比 OpenVLA / RT-2 这种需要 64+ A100 训练数天的方案, RoboBERT 极度轻量, 这正是论文的卖点 - **不需要 foundation model scale 的 resource, 也能达到 SOTA**。

---

## 9. Build Intuition: 关键 Takeaways

1. **Decoupling modalities during training**: 不要让 model 同时学 perception + language + policy, 用 curriculum (two-stage) 解耦。这跟人类学技能的过程类似 - 先学会 motor pattern, 再听懂多种语言指令。

2. **Frozen pretrained vision encoder 是 friend 不是 enemy**: CLIP 已经学好 visual concept, freezing 它省内存、防 forgetting, 让 model 把 capacity 用在 policy learning 上。

3. **Augmentation 要 task-aware**: 一般 vision 的 augmentation (affine, color jitter) 在 robot control 中效果不同 - affine 破坏 spatial grounding, color jitter 反而促进 cross-environment generalization。**不能照搬 CV 的 augmentation 套路**。

4. **Diffusion policy + lightweight backbone 可以打败 billion-parameter VLA**: multimodal action distribution 建模比单纯堆参数更重要。

5. **Cross-attention with language as query** 是 instruction-conditioned policy 的正确 inductive bias, 与 OpenFlamingo 一致。

6. **Max-pooling fusion**: 简单但有效, 把 variable-length token 压成 fixed vector, 适合 diffusion head 输入。这种 simplicity 优于复杂的 attention pooling 在 data-scarce regime。

---

## 10. 个人延伸思考 (Hallucination zone)

这篇工作让我联想到几个更广的话题:

### 10.1 与 π0 / π0.5 的对比
π0 (https://arxiv.org/abs/2410.24164) 用 flow matching 而非 diffusion, backbone 是 PaLI/PaLM-e scale, 训练于跨 robot 平台数据。π0.5 (https://arxiv.org/abs/2504.16054) 进一步做 open-world generalization。RoboBERT 与它们的区别: 不依赖 cross-embodiment data, 单一 platform 上做 SOTA, 用 two-stage curriculum 替代 data scale。

### 10.2 Action Tokenization vs Diffusion
OpenVLA 把 action tokenize 成离散 token, 用 LLM autoregressive 生成; RoboBERT 用 diffusion 在 continuous space 生成。两者各有优劣: tokenization 适合 LLM 生态 (vLLM 等加速), diffusion 更自然建模 multimodal continuous action。

### 10.3 3D Diffuser Actor 与 RoboBERT
3D Diffuser Actor (https://3d-diffuser-actor.github.io/) 用 3D scene representation (point cloud + camera parameters), 在 ABC→D 上 3.27, 不如 RoboBERT 的 3.79。这有点反直觉 - 3D 信息应该更 geometry-aware。可能原因: CALVIN 的 3D 信息有限, 2D RGB + 强 augmentation 反而更 robust; 同时 3D 处理 pipeline 复杂, 训练效率低。

### 10.4 Two-stage 与 LLM 的 Pretrain → SFT → RLHF
RoboBERT 的 two-stage 跟 LLM 的 pretrain → SRT → RLHF 在结构上有相似性: 先学 stable representation, 再 align 到人类意图。这暗示 VLA 也应该走类似 curriculum, 而 stage 1 不必是 "standard language", 可以是无语言的 pure motor learning, stage 2 再 introduce language conditioning。

### 10.5 Self-supervised Stage 1?
更进一步, stage 1 可以用 self-supervised world model learning (类似 DreamerV3, https://arxiv.org/abs/2301.04104), 学 forward dynamics; stage 2 再 align language。这可能比 BC 更 sample-efficient。

### 10.6 Affine Transformation 失败的更深层原因
让我再深挖一下: 为什么 affine translation 在 robot control 中有害? 可能不只是 spatial ambiguity, 还有:
- **Depth ambiguity**: 2D translation 在 image plane 上模拟 camera shift, 但 3D 中 object 可能离 gripper 距离没变, 这种 cue 不一致让 model 困惑
- **Gripper pose break**: gripper 在 image 中的位置是 action 的直接 conditioning signal, translating image 让 gripper 看起来在别的位置, 但 ground-truth action 是基于真实位置, supervised signal 矛盾

如果是后者, 那 augmentation 应该在 **object region only** (segment 出 object, 只对 object 做 affine), 不动 gripper。这是后续可改进点。

---

## 11. 局限与可改进方向

1. **Stage 1 仍需 labeled data**: 标准 instruction 仍是人工标注, 完全 unsupervised motor learning 没探索
2. **Only 2 cameras, no proprioception**: GR-1 等用 proprioception 收益明显, RoboBERT 没用, 可能限制了 precision task
3. **CNN-based diffusion 限制 long-horizon**: CNN 适合短 chunk, 长 horizon 可能需要 transformer-based diffusion (MoDE 走这条路)
4. **没有 RL fine-tuning**: 只有 BC, 没用 RLHF / DPO 类技术 refine policy
5. **Affine augmentation 失败提示 augmentation design 需要 task-aware**: 可以做 **gripper-aware augmentation** (动 object 不动 gripper)
6. **Mixup language 可能 damage language understanding**: 可以尝试只在 vision/action 做 mixup, 保留 language 原意

---

## 12. 总结

RoboBERT 的核心 insight 是 **efficient VLA 不需要 billion-parameter + internet-scale data**, 通过:
- (a) two-stage curriculum 解耦 policy learning 与 language alignment
- (b) task-aware data augmentation (尤其 color jitter) 实现 cross-domain generalization
- (c) lightweight 但 well-designed architecture (BERT + CLIP + cross-attention + CNN diffusion)

就能在 CALVIN 上达到 4.52 / 3.79 的 SOTA Avg. Length。这个工作对资源受限的 robotics lab 是非常好的 baseline, 也是 VLA curriculum learning 的重要 reference。

代码与 project page: https://anonymouskonto.github.io (匿名 review 版, 正式版应发布在作者主页)

完整 reference list:
- RoboBERT (匿名): https://anonymouskonto.github.io
- CALVIN: https://calvinrobot.github.io/ , https://arxiv.org/abs/2112.03227
- BERT: https://arxiv.org/abs/1810.04805
- ViT: https://arxiv.org/abs/2010
