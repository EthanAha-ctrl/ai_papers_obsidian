---
source_pdf: PriorVLA Prior-Preserving Adaptation for.pdf
paper_sha256: a57535cbf2dd500a773ab9f532f17d079a4e2040fcf9684656b52455844f2a09
processed_at: '2026-08-06T06:17:55-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 PriorVLA

## 一句话版本

大家 fine-tune VLA 时习惯把整个预训练模型当个起点，一路 gradient 踩到底，结果预训练里好不容易学到的"常识"被冲没了。PriorVLA 的做法是：把预训练模型的一份拷贝冻起来当"参考书"，另一份去学新任务，新任务那份随时翻参考书，但参考书本身不许改。

## 一个最好用的比喻

想象一个老师傅（pretrained VLA）带一个徒弟（adaptation expert）。

- 老师傅学过一万种活儿，手艺记在肌肉里
- 现在来了个新活儿，徒弟只有 10 张图纸（few-shot demonstrations）
- 传统做法（full fine-tune）：把老师傅抓过来，让他按这 10 张图纸反复练，练到完全 fit 这 10 张为止。结果他以前会的别的活儿全忘了，换个光线、换个桌面他就不会干了
- PriorVLA 的做法：老师傅站旁边不动，徒弟照图纸练，但每练一步都问老师傅"师傅你看我这步该怎么走"、"师傅这个场景你怎么理解"。老师傅只读不写，徒弟边问边学

这就解释了为什么 PriorVLA 在 OOD 和 few-shot 下特别强 —— 老师傅的知识还在，徒弟只是学会了怎么用老师傅的知识去做新活儿。

## 核心矛盾：fine-tune 到底在破坏什么

这里有个微妙但关键的观察，论文 Section 1 说得很好：

> pretraining 的价值不在 weights 本身，而在 weights 上的 forward-pass computation

啥意思？weights 是一堆数字，但这堆数字 encode 了一种"看见图像 → 理解场景 → 生成动作"的计算流程。这个流程在百万级 robot data 上被验证过能处理各种 lighting、各种 clutter、各种 object、各种 embodiment。

一旦你 full fine-tune，你在用 50 个 demo 重新塑形这整个计算流程。50 个 demo 覆盖不了多少 lighting 变化、多少 object pose 变化，结果流程就被 fit 成"只会这 50 个 demo 那种窄场景"。

更糟的是，VLA 里有两类 prior 会被一起毁掉：
- **Scene prior**（VLM 那部分）：看见杯子知道是杯子、看见 slot 知道是插孔的视觉 grounding
- **Motor prior**（action expert 那部分）：怎么平顺接近、怎么协调 gripper、怎么避免突变的运动学常识

这两类 prior 在 10 个 demo 里基本都恢复不回来。论文做了个非常直观的 VQA probe（Figure 11）：full fine-tune 之后问模型"图里有几个物体"，模型输出的全是乱码；PriorVLA 还能正常回答。这就是 prior forgetting 最直接的证据。

## Dual Action Experts：把"老师傅"和"徒弟"物理分开

论文最干净的设计就是把 action expert 一分为二：

- **Prior Expert (PE)**：原 action expert 冻起来，纯 read-only
- **Adaptation Expert (Ada)**：从相同 weights 初始化，trainable

两个 expert 在每个 denoising step 都看同一个 noisy action chunk，但只有 Ada 的输出用来更新 trajectory，PE 的输出直接扔掉。PE 唯一的作用是它的中间层 representation 被拿去当 motor prior。

这里有个细节值得停一下想：为什么不是 teacher-student distillation？distillation 里 teacher 的输出会进 loss，会反过来影响 student。PriorVLA 里 PE 的 prediction 既不进 loss 也不进 trajectory，PE 只是个"活着的、能 forward 的、内部 representation 可被查询的"prior source。这跟 MoE 也不一样，MoE 是加权混合，PriorVLA 是"PE 只读 + Ada 写入"，功能完全解耦。

## Expert Queries：三组"问问题的嘴"

光冻个 PE 没用，得有 learnable 的 interface 去"读取"它。论文设计了三组 learnable token：

- **Scene Queries (SQ)**：塞到 VLM 输入里，从 frozen VLM 的层间 representation 里抓 task-relevant 的 scene prior。可以理解成一组"可学习的视觉特征探针"
- **Motor Queries (MQ)**：塞到 PE 输入里，单向读 PE 的 noisy action representation。MQ 不能看 VLM，怕被 scene features 淹没
- **Action Queries (AQ)**：塞到 Ada 里，把 SQ 抓来的 scene prior 和 MQ 抓来的 motor prior 整合起来，指导 Ada 的 denoising

这个设计有几个聪明的地方：

1. **PE 的 noisy action tokens 看不见 SQ/MQ**：保证 PE 的 forward pass 跟预训练时一模一样，prior 不被污染
2. **Ada 不直接看 PE 的 raw action KV cache**：作者发现直接看会让训练不稳定，Ada 会"抄近路"复制 frozen state，而不是学会利用 prior
3. **MQ 不看 VLM**：VLM token 数量远多于 PE action token，放开 attention 会让 MQ 被 scene features 主导，失去 motor prior 的纯粹性

这种"接口分离 + 单向读取"的设计是整篇论文最 elegant 的部分。它本质上是在说：frozen prior 不能被 train，但可以被 query。query 本身是 learnable 的，所以下游任务能学会"怎么用 prior"。

## 实验里最戳人的几个数字

我挑几个最能说明问题的：

**RoboTwin 2.0 Hard (OOD)**：PriorVLA 53%，π0.5 是 42%，Diffusion Policy 是 0%。DP 的 0% 是最强证据 —— 没有 pretraining 的模型在 OOD 下基本废掉，pretraining prior 是 OOD 的命根子。

**Few-shot real-world (10 demos)**：PriorVLA ID 48% / OOD 32%，π0.5 是 24% / 10%。10 个 demo 下 π0.5 基本报废，PriorVLA 接近 standard data (100-300 demos) 的水平。这是 prior-preserving 价值最直接的体现。

**Large data 下 Easy 持平 (-1)**：这是个非常诚实的结果。当 downstream data 足够多，ID 性能不再需要 preserved prior，因为 data 自己能 cover ID 分布。但 Hard 仍然 +6，说明 OOD 即使 data 多也还需要 prior。

**Sign-test**：real-world few-shot 16/16 全胜，p=1.5e-5。这不是一两个 outlier task 拉起来的，是 systematic 的提升。

## Ablation 里最关键的对比

**Random PE ≈ w/o PE**：用随机初始化的 frozen branch 替换 PE，结果跟没有 PE 一样。这证明收益来自 pretrained motor prior 本身，不是"多加一个 frozen branch"这种 regularization 效应。

**Trainable PE < Frozen PE**：把 PE 也 fine-tune（参数多 50%），结果反而更差。说明 stable prior source 比 extra capacity 重要。这是反直觉的 —— 一般人觉得多 train 一些参数应该更好，但 prior preservation 的关键是 stability。

**LoRA baseline 灾难性**：LoRA 也只动 25% 参数，但 Hard 只有 17%，PriorVLA 是 49%。这是对"PriorVLA 只是 parameter-efficient fine-tuning"这种误读的有力反驳。LoRA 在原 model 上推 prior，PriorVLA 在 forward pass 空间隔离 prior，本质上完全不同。

## 我读完后的直觉

这篇 paper 真正的贡献是把 VLA adaptation 的 framing 从"怎么高效 fine-tune"升级成"怎么保留并使用 prior"。

之前整个 field 的思路是：pretrained model 是个好起点，我们想办法更高效、更稳定地把它 fit 到下游。PriorVLA 说：pretrained model 不是起点，是参考书。你不能改它，但你要学会查它。

这个 reframing 的价值在于它解释了一个一直在发生但没人正面处理的现象 —— VLA fine-tune 后 VQA 能力消失、OOD 崩溃、few-shot 失效。这些都不是"fine-tune 没调好"的副作用，是"fine-tune 本身就在破坏 prior"的必然结果。PriorVLA 给了一个 architectural 层面的根治方案。

如果说有什么遗憾，是缺少对 PE 内部 representation 的细粒度可视化 —— 比如 scene prior 和 motor prior 在不同 denoising step 怎么 emerge、怎么 interact。作者在 limitations 里承认了。如果有 CKA 或 attention rollout 这种分析，intuition 会更扎实。但整体上，这是 VLA adaptation 方向少见的有 conceptual clarity 又有工程落地的工作。

---

# PriorVLA: Prior-Preserving Adaptation for Vision-Language-Action Models 详解

## 0. TLDR

PriorVLA 在 π0.5 这种 VLA backbone 上做了一个非常巧妙的改造: **把 pretraining 不当成 initialization,而当成 forward-pass 里的 prior source**。具体做法是分两条路:

- **Dual Action Experts (DAE)**: 把 pretrained action expert 复制一份,一份 frozen 作为 Prior Expert (PE, 只读 prior source),一份 trainable 作为 Adaptation Expert (Ada, 负责下游 action generation)。
- **Expert Queries (EQ)**: 引入三组 learnable token (Scene Queries / Motor Queries / Action Queries),通过精心设计的 attention mask,从 frozen VLM 和 frozen PE 的 forward pass 里抽取 scene prior 和 motor prior,再 integrate 到 Ada 里。

结果: 只 train 25% 参数(full fine-tuning 的 1/4),在 RoboTwin 2.0-Hard 上比 π0.5 高 11 个点; LIBERO 上达到 99.1% (SOTA); 真机 8 个任务,standard data 下 81% ID / 57% OOD, few-shot (10 demos) 下 48% ID / 32% OOD,分别超 π0.5 24/22 个点。

Project page: https://priorvla.github.io/

---

## 1. 核心问题与动机 (Motivation)

### 1.1 当前 VLA adaptation 的痛点

预训练 VLA 模型 (π0, π0.5, OpenVLA, RT-2) 在大规模 robot data 上学到了 **broad priors**: 包括 VLM 编码的 task-relevant visual structure, 以及 action expert 编码的 action-generation regularities。这些 priors 是大规模 pretraining 的真正价值所在。

但是当下游 fine-tune 时,几乎所有人用 full fine-tuning,这等价于把 pretraining 只当 initialization。结果:

- 大量参数更新到少量 task-specific data 上,broad priors 被推向 narrow training-distribution patterns
- ID 性能可能提升,但 OOD 性能退化 (prior forgetting)
- few-shot 情况下尤其严重: 10 个 demo 覆盖不了多少 variation, fine-tune 后 policy 被 few-shot data 中的 incidental correlation 主导

### 1.2 核心观察: pretraining 的价值在 forward pass 里

> These priors are not explicit rules or final action outputs, but emerge in the pretrained model's forward pass.

Pretraining 不是简单的 weight initialization,而是 weight 上记录了一种 **forward-pass computation**,这种 computation 在各种 task / embodiment 上都被验证过。一旦 full fine-tune,这个 computation 就被破坏了。

PriorVLA 的立论是: **adaptation 应该学会"使用" pretrained priors,而不只是把 priors 推向下游分布**。

### 1.3 为什么 prior forgetting 在 VLA 里特别致命

VLA 包含两种 prior:

| Prior type | 来源 | 提供什么 | few-shot 数据能否恢复 |
|---|---|---|---|
| Scene prior | VLM (vision encoder + language backbone) | task-relevant visual structure, object grounding, affordance | 不能,few-shot 视觉覆盖太窄 |
| Motor prior | Action expert (flow-matching denoising dynamics) | action-generation regularities, smooth trajectory, gripper coordination | 不能,few-shot 轨迹覆盖太窄 |

Prior forgetting 让 OOD 和 few-shot 设置都被打垮。PriorVLA 通过 architecture 强制保留这两类 prior。

---

## 2. 方法详解

### 2.1 整体架构 (Figure 2 / Figure 3 解析)

PriorVLA 在 π0.5-style backbone (SigLIP vision encoder + Gemma-2B language + Gemma-300M action expert) 基础上插入:

1. 把原来的 action expert 一分为二 → **Dual Action Experts**
2. 在 VLM 输入端加 Scene Queries (SQ)
3. 在 frozen PE 输入端加 Motor Queries (MQ)
4. 在 trainable Ada 输入端加 Action Queries (AQ)

整个 token 序列分成几个 block,block 之间用单向 attention (later block 可以看 earlier block,反过来不行),block 内部 bidirectional self-attention。Figure 3 的 attention mask 表格 (橙色=允许, 空白=禁止):

| Token group | OBS (VLM 原始) | SQ | PE noisy action (N1) | MQ | AQ+AE noisy action (N2) |
|---|---|---|---|---|---|
| OBS | ✓ | ✓ | ✓ | ✗ | ✓ |
| SQ | ✓ | ✓ | ✓ | ✗ | ✓ |
| N1 (PE) | ✓ | ✗ | ✓ | ✗ | ✗ |
| MQ | ✗ | ✗ | ✓ | ✓ | ✗ |
| AQ+N2 (AE) | ✓ | ✓ | ✗ | ✓ | ✓ |

注意几个关键设计:

- **N1 不看 SQ/MQ**: 保留 PE 的 frozen denoising path 不被 query 污染,这是 prior preservation 的关键
- **MQ 不看 OBS/SQ**: appendix A.4 解释, VLM prefix token 数量远多于 PE action token,如果 MQ 可以看 VLM,会被 scene features dominate,失去 motor prior 的角色
- **N2 不直接看 N1**: motor 信息只通过 MQ 路由,作者发现直接访问 PE 的 raw action KV cache 会让训练不稳定 (appendix A.4)
- **AQ 与 N2 同 block bidirectional**: AQ 不解码 action,专职做 multi-source prior 的组织者

### 2.2 Dual Action Experts

把 pretrained AE (action expert) 复制成两份:

- **Prior Expert (PE)**: frozen, read-only,内部 representation 暴露 motor prior
- **Adaptation Expert (Ada)**: trainable, 从相同 weight 初始化,驱动真正的 denoising trajectory

在 denoising step τ,两个 expert 接收同一个 noisy action chunk; Ada 输出用于更新 trajectory, PE 输出直接丢弃。Updated chunk 进入下一步,两个 expert 再次同步。

这个设计非常关键: PE 不是 teacher (它的 prediction 不进 loss,不进 trajectory),它是一个 **read-only forward path**,内部 representation 通过 MQ 暴露给 Ada。两份 expert 共享 denoising trajectory 但功能解耦 —— PE 负责 "preserve", Ada 负责 "specialize"。

### 2.3 Expert Queries

三组 learnable token 的角色分工:

**Scene Queries (SQ)**: 插到 VLM 输入 token 序列里 (跟 multi-view image tokens, prompt tokens, proprioceptive-state tokens 一起),参与 VLM self-attention,从 VLM representations 里 capture task-relevant scene prior。SQ 的 layer-wise KV cache 给 Ada 提供紧凑的 scene-prior interface。

**Motor Queries (MQ)**: 插到 frozen PE 输入序列里,只从 PE 的 noisy action tokens 单向读取 motor prior。MQ 本身 self-attend。这个 one-way 设计让 frozen denoising 过程不被打扰,同时通过 MQ 的 KV cache 暴露 motor prior。

**Action Queries (AQ)**: 插到 trainable Ada 的 noisy action tokens 旁边,在 Ada 内部 integrate scene prior (来自 SQ) 和 motor prior (来自 MQ),指导 Ada 的 denoising。AQ 不被解码为 action,专职做 prior 整合。

---

## 3. 公式详细解析

### 3.1 Policy 基本形式 (公式 1)

$$\hat{\mathbf{a}}_{t:t+H-1} = \pi_\theta(\tilde{\mathbf{a}}_{t:t+H-1}, o_t, l, s_t)$$

变量解释:
- $o_t$: 当前 visual observation (多视角 RGB)
- $l$: language instruction
- $s_t$: proprioceptive state (机器人本体感知)
- $\tilde{\mathbf{a}}_{t:t+H-1}$: noisy action chunk,下标 $t:t+H-1$ 表示从时刻 $t$ 到 $t+H-1$ 的 chunk,$H$ 是 action horizon (RoboTwin/real-world 用 H=50, LIBERO 用 H=10)
- $\hat{\mathbf{a}}_{t:t+H-1}$: denoising prediction (flow matching 下的一步 velocity / noise 预测)
- $\pi_\theta$: 参数为 $\theta$ 的 VLA policy

### 3.2 Dual Action Experts 的同步更新 (公式 2)

$$\tilde{\mathbf{a}}_{PE}^{\tau} = \tilde{\mathbf{a}}_{Ada}^{\tau} = \tilde{\mathbf{a}}^{\tau}, \qquad \tilde{\mathbf{a}}^{\tau+1} = \mathrm{FM}(\tilde{\mathbf{a}}^{\tau}, f_{Ada}^{\tau})$$

变量解释:
- $\tau$: 当前 denoising step
- $\tilde{\mathbf{a}}_{PE}^{\tau}$, $\tilde{\mathbf{a}}_{Ada}^{\tau}$: 两个 expert 在 step $\tau$ 看到的 noisy chunk (强制相同)
- $f_{Ada}^{\tau}$: Adaptation Expert 在 step $\tau$ 的 denoising 输出
- $\mathrm{FM}(\cdot)$: flow-matching update function,典型形式 $\tilde{\mathbf{a}}^{\tau+1} = \tilde{\mathbf{a}}^{\tau} + \Delta\tau \cdot v_\theta$,其中 $v_\theta$ 是预测的 velocity field

关键: PE 的输出完全不出现在公式里,被丢弃。Updated chunk $\tilde{\mathbf{a}}^{\tau+1}$ 同时作为下一步两个 expert 的输入。这个共享 trajectory 的设计让 MQ 在每一步都能读到 PE "认为这一步应该往哪走" 的 internal representation。

### 3.3 Scene Queries 的 attention (公式 3)

$$\mathbf{h}_{sq}^{l+1} = \mathrm{Attn}(\mathbf{Q}_{sq}^l, \mathbf{K}_{obs}^l \| \mathbf{K}_{sq}^l, \mathbf{V}_{obs}^l \| \mathbf{V}_{sq}^l)$$

变量解释:
- $\mathbf{h}_{sq}^{l+1}$: Scene Query token 在第 $l+1$ 层的 hidden state
- $\mathbf{Q}_{sq}^l = \mathbf{h}_{sq}^l \mathbf{W}_Q^l$: SQ 的 query 投影,$\mathbf{W}_Q^l$ 是 query projection matrix
- $\mathbf{K}_{obs}^l, \mathbf{V}_{obs}^l$: 原始 VLM input tokens (OBS, 包括 image / prompt / proprioceptive tokens) 的 K, V 投影
- $\mathbf{K}_{sq}^l, \mathbf{V}_{sq}^l$: SQ 自己的 K, V 投影 (SQ 之间互相 attend)
- $\|$: concatenation

含义: SQ 一边跟 OBS 一起 bidirectional self-attention,一边把自己的 layer-wise KV cache 暴露给 Ada。可以理解成一组 "可学习的 hooks",从 frozen VLM 的层层 representation 里捞出 task-relevant 的部分。

### 3.4 Prior Expert 的 frozen denoising path (公式 4)

$$\mathbf{h}_{\tilde{a}^{pe}}^{l+1} = \mathrm{Attn}(\mathbf{Q}_{\tilde{a}^{pe}}^l, \mathbf{K}_{obs}^l \| \mathbf{K}_{\tilde{a}^{pe}}^l, \mathbf{V}_{obs}^l \| \mathbf{V}_{\tilde{a}^{pe}}^l)$$

变量解释:
- $\mathbf{h}_{\tilde{a}^{pe}}^{l+1}$: PE 的 noisy action tokens 在第 $l+1$ 层的 hidden state
- $\mathbf{Q}_{\tilde{a}^{pe}}^l$: PE noisy action 的 query 投影
- $\mathbf{K}_{obs}^l, \mathbf{V}_{obs}^l$: VLM 的 K, V (PE 仍可以看 VLM)
- $\mathbf{K}_{\tilde{a}^{pe}}^l, \mathbf{V}_{\tilde{a}^{pe}}^l$: PE noisy action 自己的 K, V (block 内 bidirectional)

关键点: **PE 的 noisy action tokens 不 attend SQ 也不 attend MQ**。这是为了让 PE 的 forward pass 跟 pretraining 时一模一样,从而保留 motor prior 不被 query 污染。它只是 frozen weights 的 forward,但每次 forward 的输入 (noisy chunk) 是被 Ada 控制的。

### 3.5 Motor Queries 的单向 read (公式 5)

$$\mathbf{h}_{mq}^{l+1} = \mathrm{Attn}(\mathbf{Q}_{mq}^l, \mathbf{K}_{mq}^l \| \mathbf{K}_{\tilde{a}^{pe}}^l, \mathbf{V}_{mq}^l \| \mathbf{V}_{\tilde{a}^{pe}}^l)$$

变量解释:
- $\mathbf{h}_{mq}^{l+1}$: Motor Query token 在第 $l+1$ 层的 hidden state
- $\mathbf{Q}_{mq}^l$: MQ 的 query 投影
- $\mathbf{K}_{mq}^l, \mathbf{V}_{mq}^l$: MQ 自己的 K, V (MQ 之间 self-attend)
- $\mathbf{K}_{\tilde{a}^{pe}}^l, \mathbf{V}_{\tilde{a}^{pe}}^l$: PE noisy action 的 K, V (单向读)

设计: MQ 只能 attend 自己 + PE noisy action,看不到 VLM/SQ。这是为了保持 motor prior 的纯度 (appendix A.4 解释了 VLM token 数量多,会让 MQ 被 scene features 淹没)。MQ 把 PE 每层的 motor representation 浓缩成 KV cache 供 Ada 使用。

### 3.6 Action Queries + AE noisy action 的 integrated attention (公式 6)

$$\mathbf{h}_{aq,\tilde{a}^{ae}}^{l+1} = \mathrm{Attn}(\mathbf{Q}_{aq,\tilde{a}^{ae}}^l, \mathbf{K}_{aq,\tilde{a}^{ae}}^l \| \mathbf{K}_{obs}^l \| \mathbf{K}_{sq}^l \| \mathbf{K}_{mq}^l, \mathbf{V}_{aq,\tilde{a}^{ae}}^l \| \mathbf{V}_{obs}^l \| \mathbf{V}_{sq}^l \| \mathbf{V}_{mq}^l)$$

变量解释:
- $\mathbf{h}_{aq,\tilde{a}^{ae}}^{l+1}$: AQ 和 AE noisy action tokens 在第 $l+1$ 层的 hidden state (合并的 token block)
- $\mathbf{Q}_{aq,\tilde{a}^{ae}}^l$: AQ + AE noisy action 联合的 query 投影
- $\mathbf{K}_{aq,\tilde{a}^{ae}}^l, \mathbf{V}_{aq,\tilde{a}^{ae}}^l$: AQ + AE noisy action 自身的 K, V (block 内 bidirectional)
- $\mathbf{K}_{obs}^l, \mathbf{V}_{obs}^l$: VLM 的 K, V (Ada 直接看 VLM)
- $\mathbf{K}_{sq}^l, \mathbf{V}_{sq}^l$: Scene Queries 的 K, V (Ada 通过 SQ 拿到 scene prior)
- $\mathbf{K}_{mq}^l, \mathbf{V}_{mq}^l$: Motor Queries 的 K, V (Ada 通过 MQ 拿到 motor prior)

**$\mathbf{K}_{\tilde{a}^{pe}}^l$ 和 $\mathbf{V}_{\tilde{a}^{pe}}^l$ 不出现**: 原始 PE noisy action 的 KV cache 被排除,motor 信息只通过 MQ 这条 interface 路由进来。作者发现直接访问会让训练不稳定,而且会让 Ada "抄近路" 复制 frozen state 而不是学会利用 prior。

### 3.7 训练目标

只用标准的 flow-matching MSE loss,作用在 Ada 的 denoising prediction 上:

$$\mathcal{L} = \mathbb{E}_{\tau, \tilde{\mathbf{a}}^\tau, \mathbf{a}^*} \left[ \| f_{Ada}^\tau - (\mathbf{a}^* - \tilde{\mathbf{a}}^\tau) \|^2 \right]$$

(论文没写显式形式,这是 flow matching 的标准 velocity 目标。)

- $\mathbf{a}^*$: ground-truth action chunk
- $f_{Ada}^\tau$: Ada 在 step $\tau$ 的 velocity 预测
- PE 的输出 never 进 loss, never 进 trajectory

Trainable 参数: 整个 Ada + 三组 Expert Queries + VLM vision encoder。其他 VLM 参数和 PE 全部 frozen。整体 ~25% of full fine-tuning 的参数量。

学习率分组 multiplier: SQ=2.0, MQ=4.0, AQ=4.0, 其他=1.0。MQ/AQ 学得快,因为它们是新初始化的 interface token,需要快速 align 到 frozen backbone 的 representation 尺度。

---

## 4. 实验数据详细解读

### 4.1 RoboTwin 2.0 (Table 1)

13 个双臂任务的 success rate,Easy=ID,Hard=OOD:

| Method | Easy avg | Hard avg |
|---|---|---|
| DP (Diffusion Policy) | 36 | 0 |
| RDT | 44 | 17 |
| π0 | 62 | 22 |
| π0.5 | 67 | 42 |
| **PriorVLA** | **77 (+10)** | **53 (+11)** |

注意几个 task 上的对比:
- Handover Mic: π0.5 Hard=13 (catastrophic),PriorVLA Hard=84 — 这种需要双臂协调的 task 上 prior forgetting 最严重,preservation 收益最大
- Lift Pot: π0.5 Hard=25, PriorVLA Hard=66 — 同理
- Pick Dual Bottles: Easy 75 vs 55, Hard 26 vs 17 — 双臂 pick 在 OOD 下极难,preservation 给了 9 个点

Diffusion Policy Hard 全 0,说明没有 pretraining 的 model 在 OOD 下基本废掉,这是 pretraining prior 价值的最强证据。

### 4.2 Data regime (Table 2 / Table 13)

| Method | Few Easy | Standard Easy | Large Easy | Few Hard | Standard Hard | Large Hard |
|---|---|---|---|---|---|---|
| π0.5 | 29% | 67% | 89% | 20% | 42% | 59% |
| PriorVLA | 41 (+12) | 77 (+10) | 88 (-1) | 31 (+11) | 53 (+11) | 65 (+6) |

非常重要的观察:
- **Large data 下 Easy 持平 (-1)**: 当 downstream data 充足时,ID 性能不再依赖 preserved prior,因为 data 自己能 cover ID 分布
- **Large data 下 Hard 仍 +6**: 即使 data 多到能 fit ID,OOD 仍然需要 preserved prior
- **Few-shot 下 Hard +11**: 10 个 demo 时,OOD 几乎完全靠 prior

这验证了核心论点: prior 的价值在 OOD 和 few-shot,不在 ID 大数据。

### 4.3 LIBERO (Table 3)

四个 suite, PriorVLA 99.1% avg, 打败 OpenVLA-OFT (97.1%), π0.5 (96.9%), MemoryVLA (96.7%), DD-VLA (96.3%)。在饱和的 benchmark 上还有 2 个点的提升,说明 preservation 不只是 OOD 的 trick,在 ID 上也帮得上 — 因为 LIBERO 也有跨 task 共享的 prior。

### 4.4 Real-world (Table 4 / Table 5)

Standard data (100-300 demos/task):
- ID: 81% (+12 over π0.5),OOD: 57% (+16 over π0.5)
- 注意 GR00T-N1.7 (NVIDIA 的模型) 在真机上 ID 只有 53%,OOD 只有 31%,显著弱于 π0.5,说明 base 选 π0.5 是对的

Few-shot (10 demos/task):
- ID: 48% (+24),OOD: 32% (+22)
- 10 demo 下 π0.5 ID 只有 24%,基本废了;PriorVLA 48% 接近 standard data 的水平 — 这是 prior-preserving 价值最直接的体现
- Place Ring 任务下 few-shot ID 75% vs π0.5 的 30% — 45 个点的差距,插入类任务特别依赖 motor prior

### 4.5 Compute (Table 9)

PriorVLA 训练 wall-clock 反而比 π0.5 短: RoboTwin 5.6h vs 6.8h,real-world 5.0h vs 6.5h。原因: trainable 参数少 (25%),frozen 部分用 bfloat16 不算梯度。Inference 时多跑 PE 是额外开销,但 chunked control 下可控。

---

## 5. Ablation 深入 (Table 6 / Table 15)

### 5.1 Prior Expert ablation (Table 6a)

| Variant | Params | Easy | Hard |
|---|---|---|---|
| w/o PE | 0.85B | 75 | 42 |
| Random PE | 0.85B | 75 | 43 |
| Trainable PE | 1.28B | 73 | 44 |
| **Full** | 0.85B | **77** | **49** |

解读:

- **w/o PE = w/o MQ**: 因为 PE 信息只通过 MQ 进 Ada,删 PE 等价于删 MQ 这条 prior 路径。Hard 从 49 → 42,说明 PE 贡献了约 7 个点的 OOD。
- **Random PE ≈ w/o PE**: 用随机初始化的 frozen branch 替换 PE,得到 43,基本等于 w/o PE。这是关键 control — 它证明收益来自 **pretrained motor prior**,不是 "多了一个 frozen branch" 这种 architectural regularization。
- **Trainable PE < Full**: 把 PE 也 fine-tune (1.28B params),Easy 73 / Hard 44 都比 frozen PE 差。这说明 "稳定 prior source" 比 "多一层 capacity" 重要 — PE 必须保持 read-only 才能 provide consistent prior。

### 5.2 Expert Queries ablation (Table 6b)

| SQ | MQ | AQ | Easy | Hard |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 61 | 28 |
| ✗ | ✓ | ✓ | 70 | 30 |
| ✓ | ✗ | ✓ | 75 | 42 |
| ✓ | ✓ | ✗ | 71 | 43 |
| **✓** | **✓** | **✓** | **77** | **49** |

关键观察:

- **w/o all EQ**: Hard 28,比 Full 低 21 个点 — frozen PE 一个人没用,必须有 learnable interface 才能被使用
- **w/o SQ** (Easy 75 / Hard 30): Hard 掉得最多 (-19),说明 **scene prior 是 OOD 的核心**。直觉是: OOD 下视觉变化最大,需要 VLM 通用 grounding 来 anchor; 没 SQ 就完全靠 trainable vision encoder,few-shot 下覆盖不到
- **w/o MQ** (Easy 75 / Hard 42): Hard -7,说明 motor prior 有用但不如 scene prior 关键
- **w/o AQ** (Easy 71 / Hard 43): AQ 缺了, Ada 还能直接 attend SQ/MQ,但仍 -6,说明专职的 "prior 整合 token" 比把整合任务塞进 action token 里更好

### 5.3 Frozen ViT (Table 15)

Frozen ViT: Easy 65 / Hard 37,明显比 Full 差。说明虽然 prior preservation 重要,vision encoder 仍然要 adapt 到下游 visual distribution — 完全 freeze 也不行。Trainable vision encoder + frozen VLM core + frozen PE 是最佳组合,这跟 Knowledge-Insulating VLA [16] 的发现一致。

### 5.4 LoRA baseline (Table 15)

Baseline-LoRA: Easy 53 / Hard 17 — 远差于 full PriorVLA。这是对 "PriorVLA 只是 parameter-efficient fine-tuning" 这种误读的反驳: LoRA 也只动 25% 参数,但 Hard 17 vs 49,差了 32 个点。**参数效率不是 PriorVLA 的核心,prior preservation 才是**。

---

## 6. 与其他工作的关联 (intuition building)

### 6.1 LoRA / Parameter-efficient fine-tuning

LoRA [24] 通过低秩 update 限制参数变化,但仍然在原 model 上 "推" prior。PriorVLA 不是参数效率 trick (虽然参数效率是 side benefit),核心是 **structural preservation**: 把 prior source 完全 freeze,只 train adaptation branch + interface。
https://arxiv.org/abs/2106.09685

### 6.2 Knowledge-Insulating VLA / VLA-Adapter

Driess et al. [16] (Knowledge-Insulating VLA) 提出 freeze VLM 训 action head; VLA-Adapter [17] 用 bridge attention 连接 frozen VL feature 和 action policy。这些方法 focus 在 "怎么 freeze / 怎么连接",PriorVLA 进一步问: **frozen 之后的 prior 怎么用?** — 答案是 learnable query interface (SQ/MQ/AQ)。
https://arxiv.org/abs/2505.23705
https://arxiv.org/abs/2509.09372

### 6.3 Q-Former (BLIP-2)

BLIP-2 的 Q-Former 用 learnable query 从 frozen image encoder 里抽 vision feature 给 LLM。PriorVLA 的 SQ/MQ/AQ 是 Q-Former 思想在 VLA 上的延伸,但有 important 区别: PriorVLA 有三组 query 分别服务不同 prior source (scene / motor / integration),而不是单一 Q-Former。
https://arxiv.org/abs/2301.12597

### 6.4 MoE 与 dual-expert 设计

DAE 像 MoE 但本质不同: MoE 是多个 expert 加权混合,PriorVLA 是 **PE 只读 + Ada 写入**,功能完全解耦。这更像 "teacher network in distillation" 但 teacher 的输出不进 loss,只有 internal representation 被 query 抽取。

### 6.5 Continual learning 与 catastrophic forgetting

PriorVLA 的 frozen PE 是 architectural 层面的 catastrophic forgetting 防护,跟 EWC [39] / GEM [40] 的 regularization-based 方法不同。EWC 在 loss 里加 Fisher information penalty, PriorVLA 直接 architectural 隔离 prior source。
https://arxiv.org/abs/1612.00796

### 6.6 Diffusion / Flow matching 中的 multi-expert denoising

π0 用 flow matching,DDPM 用 diffusion。PriorVLA 的两个 expert 共享同一条 noisy trajectory,在每步同步 forward,这是 flow matching 的特殊设计 — PE 看到的 noisy chunk 来自 Ada 的 update,所以 PE 给的 motor prior 跟 Ada 当前 denoising 状态一致。

### 6.7 MAPS / Robust fine-tuning via parameter merging

MAPS [18] 用 module-wise proximity scheduling 限制 VLM 表示漂移; Yadav et al. [19] 用 parameter merging (类似 WiSE-FT) 把 fine-tuned 模型跟 pretrained 模型 weight 平均。这些方法在 weight 空间做约束, PriorVLA 在 forward pass 空间做 preservation,更直接。
https://arxiv.org/abs/2511.19878

### 6.8 MemoryVLA / GR00T-N1 / UniVLA

MemoryVLA [46] 用 perceptual-cognitive memory; GR00T-N1 [35] 是 NVIDIA 的 foundation model; UniVLA [11] 用 task-centric latent action。PriorVLA 跟它们的关系是: 它们是不同的 base / 不同的 representation,PriorVLA 是一种 **adaptation framework**,理论上可以加到这些 model 上,论文里只在 π0.5 上验证。
https://arxiv.org/abs/2503.14734

### 6.9 OpenVLA-OFT

OpenVLA-OFT [12] 是 fine-tuning OpenVLA 的方法,优化 speed 和 success,通过调整 action representation 和 fine-tuning objective。PriorVLA 跟它对比在 LIBERO 上 (99.1 vs 97.1),说明 prior-preserving 比 action representation 调整更有效。
https://arxiv.org/abs/2505.09943 (OpenVLA-OFT)

---

## 7. Intuition: 为什么 PriorVLA 在 OOD / few-shot 上特别强

### 7.1 信息论视角

Few-shot downstream data $D_{few}$ 只覆盖了 task distribution $\mathcal{T}$ 的一个窄切片。Full fine-tune 让 $p_\theta(a|o,l,s) \to p_{D_{few}}(a|o,l,s)$,把 broad pretrained $p_{pre}(a|o,l,s)$ 推向 narrow distribution。

PriorVLA 改写了这个 optimization 目标。设 PE 的 forward 给出 motor prior $f_{PE}(\tilde{\mathbf{a}}|o,l,s)$ (一个 distribution over velocity field),VLM 的 forward 给出 scene prior $\phi_{VLM}(o,l)$。Ada 学的是:

$$\hat{f}_{Ada} = g_\theta(\phi_{VLM}, f_{PE}, \tilde{\mathbf{a}})$$

其中 $g_\theta$ 是 Ada + AQ 的 learnable integration function。**$g_\theta$ 学的是 "怎么用 prior",而不是 "把 prior 推成 downstream distribution"**。OOD 下,$\phi_{VLM}$ 仍然提供 general grounding (没被 shift),$f_{PE}$ 仍然提供 general motor regularity (没被 shift),$g_\theta$ 只学了一层 "适配",所以泛化更好。

### 7.2 Few-shot 下的 "prior 缺什么补什么"

Few-shot 下 $D_{few}$ 提供 "这个 task 的特殊 action 模式",但不提供 "通用 motor / scene structure"。PE 提供 motor structure, VLM + SQ 提供 scene structure,Ada 只学 task-specific 适配。所以 10 个 demo 就能让 PriorVLA 达到 48% ID / 32% OOD,而 π0.5 只能 24% / 10%。

### 7.3 OOD 下的 "prior 是不变量"

OOD perturbation 改变 lighting / background / object position / table height。这些变化不影响:
- VLM 的 grounding (VLM 在大规模 data 上学过各种 lighting / clutter)
- PE 的 motor prior (motor regularity 跟 scene appearance 弱相关)

所以 PE + SQ 在 OOD 下仍然能给出合理 prior,Ada 在其基础上做小修正就行。Full fine-tune 把 Ada 推成了 "ID 上的 overfit predictor",OOD 下就垮了。

### 7.4 VQA 探针的启示 (Section G / Figure 11)

Full fine-tune 后,模型的 VQA 能力完全废掉 (输出非语义内容)。PriorVLA 保留 VQA 能力 (能正确识别物体、数数、描述场景)。这说明 PriorVLA 的 VLM core 几乎没被破坏,只通过 SQ "借用" representation,而不是改写。这是个非常直观的 sanity check: **如果你的 fine-tune 把 VLM 的语言生成能力都打烂了,那 prior forgetting 一定发生了**。
https://priorvla.github.io/ (Figure 11)

---

## 8. Implementation 细节 (Appendix A/B)

### 8.1 Backbone

- VLM: SigLIP vision encoder + Gemma-2B language
- Action Expert: Gemma-300M transformer,flow-matching denoising
- 全部基于 π0.5 / OpenPI 代码库 (Apache-2.0)
https://github.com/Physical-Intelligence/openpi

### 8.2 关键超参

- AdamW, grad clip 1.0
- Trainable 用 float32, frozen 用 bfloat16 (省显存)
- Grouped LR multiplier: SQ=2.0, MQ=4.0, AQ=4.0, rest=1.0 — 新 init 的 query token 学得快
- RoboTwin: H=50, batch 32, peak LR 2.5e-5, 30k steps, EMA 0.99, 1k warmup, cosine decay
- LIBERO: H=10, batch 256, peak LR 5e-5, 30k steps (1M decay schedule), EMA 0.999 (跟 π0.5 LIBERO recipe 对齐)
- Real-world: H=50, batch 32, peak LR 2.5e-5, 30k steps, EMA 0.99
- 8 GPUs: H20 96GB (RoboTwin), A100 80GB (LIBERO / real-world)

### 8.3 推理

- 预测 H=50 的 action chunk,执行前 15 个 action,然后 replan
- OOD 测试 4 个维度联合 perturb: Light (暗), Background (杂物), Object Position (初始位置), Table Height (+2cm)
- Real-world 一个 task 20 ID + 20 OOD trials

### 8.4 RoboTwin 2.0 Easy vs Hard (Table 10)

Hard mode 开启: background randomization, table clutter (clean rate 0.02), table-height perturbation (0.03), lighting randomization (extreme lighting rate 0.02)。Easy mode 全部 off。同一个 robot (Aloha-AgileX),同样 camera (D435/D435),只差 domain randomization。
https://arxiv.org/abs/2506.18088 (RoboTwin 2.0)

### 8.5 LIBERO

四个 suite,每个 10 task,每个 task 50 demo,50 rollout evaluation,seed=7,policy 每 10 env step replan。
https://arxiv.org/abs/2306.03310 (LIBERO)

---

## 9. 一致性 / 显著性 (Appendix D.2)

Table 14 做 sign-test,看 PriorVLA 比 π0.5 在多少 (task, setting) pair 上更好:

| Setting | Non-tie pairs | Improved | Avg gain | Sign-test p |
|---|---|---|---|---|
| RoboTwin few-shot Easy | 13/13 | 13 | +12 | 1.2e-4 |
| RoboTwin few-shot Hard | 13/13 | 12 | +11 | 1.7e-3 |
| RoboTwin standard Easy | 11/13 | 11 | +10 | 4.9e-4 |
| RoboTwin standard Hard | 12/13 | 10 | +11 | 1.9e-2 |
| RoboTwin large Hard | 13/13 | 10 | +6 | 4.6e-2 |
| Real-world standard ID/OOD | 15/16 | 14 | +14 | 4.9e-4 |
| Real-world few-shot ID/OOD | 16/16 | 16 | +23 | 1.5e-5 |

Real-world few-shot 16/16 全胜,p=1.5e-5,这是非常强的统计证据。RoboTwin few-shot Easy 也是 13/13。说明收益不是被某个 outlier task 主导,是 systematic 的。

---

## 10. Limitations 与未来方向

1. **RoboTwin 只跑 13/50 task**: 每个 task 一个 model,跑全 50 task 太贵
2. **OOD 因子联合 perturb**: 不分别 ablate light / background / position / height 的影响,无法说清哪类 prior 对哪类 OOD 最关键
3. **Inference 开销**: PE 在每个 denoising step 都要 forward,虽然 chunked control 下可控,但理想情况下应该 cache PE 的 readout
4. **没有 finer-grained 分析**: scene prior 和 motor prior 在不同层、不同 denoising step 怎么 emerge / interact / evolve — 这是个 open question,作者明确说留给 future work

未来潜在方向 (我的联想):

- **Cache PE readout**: 既然 PE frozen,且只在 noisy chunk 上 forward,可以预先 cache 不同 noisy level 的 KV cache,或用小型 distillation 把 PE 蒸馏成 fixed-size motor prior embedding
- **多 task 共享 PE**: PE 在多 task adaptation 里可以共享 (因为它 freeze),只 Ada per-task train,这天然适合 multi-task / continual learning
- **Cross-embodiment PE**: 同一个 PE 给不同 robot embodiment 的 Ada 用,因为 PE 编码的是 abstract motor regularity
- **PE selection / routing**: 用 router 选择 PE 哪些层的 motor prior 对当前 task 最有用,类似 MoE 的 sparse routing
- **跟 World model 结合**: VQA probe 显示 PriorVLA 保留 VLM 的语言 / 视觉理解,可以跟 world model (DreamVLA [42], Genie Envisioner [45]) 结合做 imagination-augmented adaptation

---

## 11. 我对这篇 paper 的整体评价

**核心 insight 非常清晰且重要**: pretraining 是 forward-pass computation 不是 initialization。这个 reframing 简洁有力,把 "fine-tune 时的 prior forgetting" 问题从 weight 空间拉到 forward-pass 空间。

**架构设计 elegant**: DAE + EQ 的组合很干净,attention mask 设计充分考虑了 prior purity (MQ 不看 VLM) 和 prior usability (Ada 不直接看 PE raw action,只通过 MQ)。这种 "frozen source + learnable interface" 的范式应该可以推广到其他 adaptation 场景。

**实验扎实**: 三个 benchmark,两个 embodiment,few-shot / standard / large 三个 regime,OOD 四维 perturb,sign-test 统计显著性,Ablation 把 "random PE" / "trainable PE" / "LoRA baseline" 都做了 control,说服力强。

**唯一略遗憾的是**: 没有把 PE 跟 Ada 的 representation 做更细的可视化 / 相似度分析 (比如 CKA across layers),虽然作者在 limitations 里承认了。如果有这种分析,intuition 会更 solid。

整体看,这篇 paper 是 VLA adaptation 方向的一个重要的 conceptual contribution — 它把 "parameter-efficient fine-tuning" 这个 framing 升级成 "prior-preserving adaptation",并且给出了可操作的 architecture 实现。对后续 VLA adaptation 工作会有持续影响。

---

## Reference 链接汇总

- **PriorVLA project page**: https://priorvla.github.io/
- **π0 (Black et al.)**: https://arxiv.org/abs/2410.24164
- **π0.5 (Black et al.)**: https://arxiv.org/abs/2504.16054
- **OpenPI 代码库**: https://github.com/Physical-Intelligence/openpi
- **OpenVLA (Kim et al.)**: https://arxiv.org/abs/2406.09246
- **OpenVLA-OFT (Kim et al.)**: https://arxiv.org/abs/2505.09943
- **RT-2**: https://arxiv.org/abs/2307.15818
- **RoboTwin 2.0**: https://arxiv.org/abs/2506.18088
- **LIBERO**: https://arxiv.org/abs/2306.03310
- **Diffusion Policy**: https://arxiv.org/abs/2303.04137
- **RDT-1B**: https://arxiv.org/abs/2410.07864
- **GR00T-N1**: https://arxiv.org/abs/2503.14734
- **MemoryVLA**: https://arxiv.org/abs/2506.04817 (近似)
- **UniVLA**: https://arxiv.org/abs/2505.12292
- **Knowledge-Insulating VLA**: https://arxiv.org/abs/2505.23705
- **VLA-Adapter**: https://arxiv.org/abs/2509.09372
- **LoRA**: https://arxiv.org/abs/2106.09685
- **BLIP-2 (Q-Former)**: https://arxiv.org/abs/2301.12597
- **EWC (continual learning)**: https://arxiv.org/abs/1612.00796
- **GEM**: https://arxiv.org/abs/1703.08175
- **MAPS**: https://arxiv.org/abs/2511.19878
- **DreamVLA**: https://arxiv.org/abs/2507.04447
- **SmolVLA**: https://arxiv.org/abs/2506.01844
- **SpatialVLA**: https://arxiv.org/abs/2501.15830
- **CogACT**: https://arxiv.org/abs/2411.19650
- **FAST (π0-FAST)**: https://arxiv.org/abs/2501.09747
- **Gemini Robotics 1.5**: https://arxiv.org/abs/2510.03342
- **Octo**: https://arxiv.org/abs/2405.12213
- **BridgeData V2**: https://arxiv.org/abs/2308.12952
- **Open X-Embodiment**: https://arxiv.org/abs/2310.08864
- **DROID**: https://arxiv.org/abs/2403.12945
- **RoboCasa**: https://arxiv.org/abs/2406.02523
- **MimicGen**: https://arxiv.org/abs/2310.17596
- **RoboVerse**: https://arxiv.org/abs/2502.07484 (近似)
- **TinyVLA**: https://arxiv.org/abs/2409.12514
- **HAMLET**: https://arxiv.org/abs/2502.06177 (近似)
- **DD-VLA**: https://arxiv.org/abs/2508.20072
- **F1**: https://arxiv.org/abs/2509.06951
- **GR-2**: https://arxiv.org/abs/2410.06158
- **Agibot World Colosseo**: https://arxiv.org/abs/2503.06669
- **NVIDIA GR00T N1.7**: https://huggingface.co/nvidia/GR00T-N1.7-3B
