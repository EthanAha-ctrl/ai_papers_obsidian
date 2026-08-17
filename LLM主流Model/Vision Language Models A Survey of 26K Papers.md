---
source_pdf: Vision Language Models A Survey of 26K Papers.pdf
paper_sha256: e461cde4b37c99af703f8ca812d4745be3b6c53daf8e694e1dc889adc403e1f3
processed_at: '2026-08-13T01:23:25-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇 survey

---

## 一句话总结

**有人把 2023-2025 年 CVPR、ICLR、NeurIPS 一共 2.6 万篇论文的标题和摘要全爬下来，用脚本统计了一下大家到底在研究啥，结果发现 vision 社区已经被 LLM 范式吃掉了。**

---

## 怎么做的

没用什么 fancy 方法。就是写了个爬虫，把三个顶会的 paper title + abstract 全抓下来，normalize 文本（统一小写、去标点、把 "gaussian splatting" 这种多词短语当成一个 token 保护起来），然后拿 35 个手写的正则去匹配关键词。

一篇 paper 可以同时命中多个标签，最后算每年命中某个 label 的论文占比，再看这个占比随时间怎么变。

笨办法，但 transparent、可复现。缺点是只看 abstract，很多细节看不到；而且正则匹配 recall 有限，换个说法就漏了。

---

## 三个大趋势

### 1. VLM 爆发

2023 年大约 16% 的论文跟 vision-language 有关，到 2025 年涨到 40%。ICLR 2025 甚至 40.7%。也就是说，**现在每 5 篇顶会论文里有 2 篇是 VLM**。

这个数字本身就是最震撼的结论。整个 vision 社区在两年内被 LLM 范式重组了。

### 2. Diffusion 从"研究新模型"变成"研究怎么用好"

Diffusion 的论文占比从 8% 涨到 19%，但主题变了。2023 年大家还在发新的 U-Net 架构、新的采样器，到 2025 年大家关心的是：怎么控制生成（controllability）、怎么蒸馏加速（distillation）、怎么跑得快（speed）。

说白了，diffusion model 本身已经是 commodity 了，研究重心转向工程优化。

### 3. 3D 没死，但换了马甲

NeRF 的热度在降，Gaussian Splatting 在升。原因很简单：NeRF 渲染要一个一个 ray march 通过 MLP query，慢得要命；3D Gaussian Splatting 直接用一堆 3D Gaussian 做 alpha blending，可微 rasterize，训练 10 分钟出结果，NeRF 要 hours。

3D 整体占比没怎么变，但内部 representation 从 implicit MLP 切到了 explicit Gaussian。

---

## VLM 内部发生了什么

这部分是 paper 最有价值的细节。

### 模型谱系

时间线大致是这样：

- **2020-2021**：CLIP 和 ALIGN。两个 encoder，一个吃图一个吃文字，用 contrastive loss 对齐。简单粗暴，scale 大就行。
- **2021-2022**：BLIP、ALBEF 这些开始加 cross-attention 做 fusion，能做 understanding 也能做 generation。
- **2022**：Flamingo 把 visual feature 压成 64 个 token，通过 gated cross-attention 塞进 frozen LLM。
- **2023**：LLaVA 横空出世。CLIP encoder + 一个 linear projector + Vicuna LLM，用 GPT-4 生成的 instruction data 训。极简但有效。
- **2024-2025**：InternVL、Qwen-VL 系列开始做 native multimodal，vision token 直接进 LM 的 next-token prediction。

### 架构选择在 converge

2023 年大家还在各种 fusion 方式之间纠结——cross-attention、Q-Former、encoder-decoder。到 2025 年，community 基本收敛到两个组合：

- **frozen vision encoder + 一个轻量 projector + frozen/LoRA LLM**（LLaVA 路线）
- **cross-attention 注入**（Flamingo 路线，保留在需要更强 visual conditioning 的场景）

Q-Former 几乎没人提了，encoder-decoder 范式在下降，dual-encoder 也在退。LoRA 从 1.3% 涨到 4.1%，prompt tuning 保持高位。**参数高效微调成了标配。**

### Task 的 shift 最戏剧

Grounding/Referring 从 25.9% 跌到 12.9%，Reasoning/Instruction 从 13.5% 涨到 25.0%。

翻译成人话：以前大家发 paper 说"我在 RefCOCO 上做了 phrase grounding"，现在大家发 paper 说"我的 VLM 能 follow instruction 做 reasoning"，grounding 只是其中一个 sub-skill。

经典的 vision task（detection、segmentation、captioning、VQA）作为独立 contribution 在减少，它们变成了 instruction-tuned VLM 内部的一个 capability。

### Training 范式

最明显的信号：self-supervised / weak-supervised pretraining 从 9.6% 跌到 3.5%。SimCLR、MoCo 这些方法不再是 paper 的 main contribution，大家直接拿 CLIP 或 DINOv2 当 frozen backbone 用。

Instruction tuning 从 1.1% 涨到 5.0%。LoRA/Adapter 从 1.3% 涨到 4.1%。Pretrain+Finetune 从 11.6% 涨到 16.8%。

**整个社区从"训练 encoder"转向"微调 foundation model"。**

### Loss 也变了

Contrastive loss 从 10.8% 跌到 5.1%。因为新 paper 不从头训 CLIP 了，改用 next-token cross-entropy（instruction tuning）加 KL distillation。

### Dataset 提及在降

COCO 从 4.9% 降到 1.0%，ImageNet 从 3.1% 降到 1.6%。大家不再在 abstract 里 name-drop 数据集，改说"我们在 MMBench/MMMU 上做 broad evaluation"。老 benchmark 退化成 sanity check。

---

## 三个 venue 的差异

CVPR 仍然是 3D 最强的 venue（2025 年 23.1% 论文涉及 3D），也最多 diffusion。ICLR 在 VLM 上占比最高（40.7%），可能因为 ICLR 更接受 LLM-centric 的 methodology paper。NeurIPS 数据只到 2024，VLM 30.5%，ramp 稍慢。

---

## 给你的实操建议

1. **如果你做 classic perception**，考虑把它 reframe 成 instruction following。不要发"我做了个更好的 detector"，发"我的 VLM 通过 instruction tuning 能做 grounding"。

2. **如果你做 diffusion**，必须回答"为什么不直接调 API"。controllability、speed、distillation 是唯一能 justify 新 paper 的角度。

3. **Long-context video** 是上升趋势。能处理分钟级甚至小时级视频的 VLM 还很少。

4. **Efficiency 和 safety** 是跨 venue 的 acceptance signal。轻量推理、sparse attention、watermarking、jailbreak defense 这些主题在扩散。

5. **3D + VLM 是蓝海**。3D grounding、scene understanding、robotics VLM 在涨。如果你用 3D/depth/audio，在 abstract 里早点说，cross-modal grounding 越来越被重视。

6. **Instruction tuning + LoRA + projector 是 default recipe**。偏离这个 recipe 需要强理由。

---

## 我的直觉判断

这篇 survey 量化了一个大家已经感觉到但没数据证明的事情：**vision 社区在 2023-2025 完成了一次范式转移**。

driver 是四件套的组合——CLIP 提供了 vision encoder、LLaVA 证明了 instruction tuning 的可行性、LoRA 让微调 LLM 变便宜、GPT-4 能批量生成训练数据。

到 2025 年，从"训练 encoder"到"微调 foundation model"的转移基本完成。剩下的 question 是：**native multimodal（从头联合训 vision+text）会不会取代 frozen CLIP + LLM 的 modular 路线？** InternVL3、Qwen2.5-VL 在 push 这个方向，但 survey 的数据还没完全反映这个 shift，因为 abstract 有滞后性。

还有一个 survey 没覆盖的角度：**inference 时的效率**。VLM 的 visual token 太长导致 context window 爆炸，token compression、KV cache 优化、speculative decoding 这些工程问题在 2025 年下半年会越来越重要，但 abstract 里很难统计到。

总之，如果你要在 2026 年做 vision research，**不碰 LLM/instruction formulation 的纯 perception paper 会越来越难发**。要么拥抱 VLM 范式，要么找到一个 VLM 真的做不了的问题。

---

# Vision Language Models: A Survey of 26K Papers — 深度解读

## 1. 这篇 paper 在做什么（方法论层面）

这篇 survey 用的方法很特别：**不是**手动 review，**而是** lexicon-driven regex matching + TF-IDF 聚合。作者 Fengming Lin 从 CVPR/ICLR/NeurIPS 2023-2025 抓取了 26,104 篇 accepted paper 的 title + abstract，做 normalization（Unicode、lowercase、phrase protection，例如 "gaussian splatting" 当作单 token），然后用 35 个 hand-crafted regular expression category 打标签。

关键 metric 是 **prevalence**（某年匹配某 label 的 abstract 占比）和 **TF-IDF mass**（某方向在某年的聚合 TF-IDF 分数）。Linear slope（最小二乘）用来衡量 trajectory 的陡峭程度，单位 pp/yr（percentage points per year）。

这种方法的好处：transparent、reproducible、longitudinal signal 跨 venue 一致。坏处：abstract-only，recall 受限；多 label paper 百分比不 sum to 100%。

## 2. Macro trends — 三个 macro shift

### 2.1 VLM 从 16% → 40% (2023 → 2025)

这是最震撼的数字。看 Figure 1 的 trajectory，VLM/LLM 曲线几乎是 hockey stick。ICLR 2025 到 40.7%，CVPR 2025 到 39.5%，NeurIPS 2024 已经 30.5%。

**intuition**：到 2025 年，每 5 篇 CVPR/ICLR paper 就有 2 篇是 VLM-related。整个 vision 社区已经在 LLM 范式下重组。

### 2.2 Diffusion 从 8% → 19.2%

Diffusion 的研究主题从 "训练新 backbone" 转向 **controllability、distillation、speed**。这呼应了 Stable Diffusion 3、Flux、Rectified Flow、Consistency Models 这条线。

### 2.3 3D 稳定但 reconfigured：NeRF → Gaussian Splatting

NeRF 的 share 在降，Gaussian Splatting 在升。这是 2023 年 3DGS (SIGGRAPH) 出来之后的典型 transfer。背后的 intuition：3DGS 是 explicit representation（3D Gaussians 参数化 by 位置 μ、协方差 Σ、opacity α、color SH coefficients），比 NeRF 的 implicit MLP query 快得多，可微 rasterization 直接 backward。

### 2.4 一些 declining 的方向

Figure 2 显示：Self-supervised pretraining 在 2023 后 decline；Meta-learning、AutoML、weak/semi/few-shot 在降；GNN、Bayesian、Optimization、Theory 都在降。这些**不是**消失，**而是**变成了 foundation model pipeline 里的 module，不再是 primary focus。

---

## 3. VLM 的核心模型谱系（Section 4.1 深读）

作者把 VLM 划成几条 main line，我把它画成 timeline：

```
2020 ── CLIP/ALIGN (dual-encoder, contrastive)
       │
2021 ── ALBEF, SimVLM, CoCa, PaLI (encoder-decoder fusion)
       │
2022 ── BLIP/BLIP-2 (MED + Q-Former), Flamingo (gated cross-attn)
       │
2023 ── LLaVA, MiniGPT-4, InstructBLIP, Kosmos-2 (instruction tuning)
       │
2024 ── InternVL, Qwen-VL, PaLI-X
       │
2025 ── InternVL3.5, Qwen3-VL, Qwen3-Omni, LLaVA-OneVision-1.5
```

### 3.1 ALIGN 和 CLIP — dual-encoder contrastive 范式

ALIGN 用 ~1B noisy image-text pair，CLIP 用 ~400M pair，都走 dual-encoder + InfoNCE loss。

InfoNCE 的形式：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\mathbf{v}_i^\top \mathbf{t}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{v}_i^\top \mathbf{t}_j / \tau)}$$

- $\mathbf{v}_i \in \mathbb{R}^d$：第 $i$ 个 image embedding（来自 image encoder $f_v$）
- $\mathbf{t}_i \in \mathbb{R}^d$：第 $i$ 个 text embedding（来自 text encoder $f_t$）
- $\tau$：learnable temperature
- $N$：batch size（作为 in-batch negatives）

intuition：让正样本对的 cos similarity 远大于负样本对。CLIP 的关键 contribution 是 **scale** —— 400M pair 让 zero-shot ImageNet 达到 76.2%。

ALIGN 的关键 insight：**noisy alt-text 在 1B scale 下也能学到 alignment**，不需要精心 curate。这给了后来 web-scale pretraining 一个 license。

### 3.2 BLIP — MED + CapFilt

BLIP 的核心创新是 **Multimodal Mixture of Encoder-Decoder (MED)**，一个网络三种角色：
1. Unimodal image encoder（做 ITC，image-text contrastive）
2. Image-grounded text encoder（做 ITM，image-text matching，cross-attn 注入 image feature）
3. Image-grounded text decoder（做 LM，causal generation）

CapFilt 的 bootstrapping：

$$\mathcal{D}_{\text{clean}} = \{(\mathbf{x}, \mathbf{c}) \in \mathcal{D}_{\text{web}} : \text{Filter}(\mathbf{x}, \mathbf{c}) = 1\} \cup \{(\mathbf{x}, \mathbf{c}') : \mathbf{c}' = \text{Captioner}(\mathbf{x})\}$$

即 Captioner 生成新 caption，Filter 过滤 noisy pair，合成 cleaner dataset 再 pretrain。

### 3.3 Flamingo — gated cross-attention + Perceiver Resampler

Flamingo 是 LVLM 的 prototype。关键 architecture：

```
Image/Video → CNN/ViT features → Perceiver Resampler 
  → K visual tokens (e.g., 64)
  → insert into frozen LLM via Gated Cross-Attention layers
```

Gated Cross-Attention：

$$\mathbf{y} = \mathbf{x} + \tanh(\alpha) \cdot \text{CrossAttn}(\mathbf{x}, \mathbf{V})$$

其中 $\alpha$ 是 learnable gate，初始化为 0，保证训练开始时 LLM 行为不变（稳定初始化）。这是 PEFT-style 思想在 fusion 层的应用。

Perceiver Resampler 把变长 visual feature 压缩成固定数量的 latent token：

$$\mathbf{Q} = \text{LearnableLatents} \in \mathbb{R}^{K \times d}, \quad \mathbf{V}_{\text{out}} = \text{CrossAttn}(\mathbf{Q}, \mathbf{K}_{\text{img}}, \mathbf{V}_{\text{img}})$$

intuition：让 visual token 数量与 LLM context window 解耦。

### 3.4 LLaVA — visual instruction tuning 的 breakthrough

LLaVA 是 2023 年最有 impact 的工作之一，share 从 0.1% → 1.2% → 2.7%，slope +0.91 pp/yr。

Architecture 极简：

```
Image → CLIP ViT-L/14 → patches → linear projector W 
  → visual tokens → concat with text tokens → LLM (Vicuna)
```

$$\mathbf{H}_v = \mathbf{W} \cdot \mathbf{X}_v, \quad \mathbf{W} \in \mathbb{R}^{d_{\text{LLM}} \times d_{\text{CLIP}}}$$

训练两阶段：
1. **Stage 1 (feature alignment)**：freeze LLM 和 CLIP，只训 projector $W$，用 595K image-text pair
2. **Stage 2 (instruction tuning)**：freeze CLIP，训 projector + LLM (LoRA 或 full)，用 158K GPT-4 生成的 multimodal instruction data

LLaVA 的核心 insight：**用 GPT-4 把 image-caption 扩展成 instruction-style conversation**，把 VQA/captioning/grounding 这些任务统一成 instruction following。

### 3.5 DINO/DINOv2/DINOv3 — vision backbone 的进化

DINO 的 self-distillation：

$$\mathcal{L}_{\text{DINO}} = -\sum \mathbf{p}_s^\top \log \mathbf{p}_t$$

其中 $\mathbf{p}_s$ 是 student output（multi-crop small views），$\mathbf{p}_t$ 是 teacher output（momentum updated EMA, global views）。Centering + sharpening 防止 collapse。

DINOv2 加了 curated data、prototype loss、improved regularization。

DINOv3（2025, [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)）加了 **Gram anchoring**：

$$\mathcal{L}_{\text{Gram}} = \| \mathbf{G}_s - \mathbf{G}_t \|_F^2, \quad \mathbf{G} = \mathbf{F}^\top \mathbf{F}$$

其中 $\mathbf{G}$ 是 feature map 的 Gram matrix，$\mathbf{F}$ 是 flattened feature。这防止 dense feature 在 long schedule 下退化。DINOv3 在 VLM 里越来越常用作 vision encoder。

### 3.6 Grounding DINO — open-vocabulary detection

Grounding DINO 把 DINO detector + text encoder 联合训练，region-level visual feature 对齐 phrase-level text feature，loss 是 detection + phrase grounding 联合。share 从 0% → 0.2%，slope +0.06 pp/yr，是 VLM 的 grounding module 常用 plug-in。

### 3.7 MoE — sparse scaling

MoE 在 VLM 里从 0.6% → 1.3% (2023→2025)。Switch Transformer 的 gating：

$$\text{TopK}(\mathbf{g}(\mathbf{x})), \quad g_i(\mathbf{x}) = \text{softmax}(\mathbf{W}_g^\top \mathbf{x})_i$$

每个 token 只激活 top-1 或 top-2 expert MLP，参数量增大但 FLOPs 几乎不变。Qwen2.5-VL、Qwen3-VL 都用 MoE。

---

## 4. 架构演化趋势（Section 4.2 — Table 3 解析）

这是 paper 里最有 signal 的一张表。我把关键 trend 列出来：

| Mechanism | 2023 | 2024 | 2025 | Trend | 解读 |
|---|---|---|---|---|---|
| Prompt/Prefix Tuning | 13.0% | 16.4% | 14.3% | +1.3 pp | 主流 PEFT |
| Adapter/LoRA | 1.3% | 4.0% | 4.1% | +2.8 pp | 快速增长 |
| Cross/Co-attention | 1.7% | 2.2% | 2.2% | +0.5 pp | Flamingo-style 稳定 |
| Projector/MLP Head | 0.9% | 1.2% | 1.5% | +0.6 pp | LLaVA-style 上升 |
| Q-Former Bridge | 0.0% | 0.1% | 0.0% | +0.0 pp | 专业化使用 |
| Encoder-Decoder | 1.6% | 0.7% | 0.3% | -1.3 pp | OFA/PaLI 范式 decline |
| Dual-encoder/Two-tower | 0.3% | 0.2% | 0.1% | -0.1 pp | 纯 retrieval 衰退 |

**核心 insight**：community 在 converge 到 **"frozen backbone + lightweight bridge + instruction tuning"** 的设计。LoRA + projector 成了 default knob。

LoRA 的公式值得提：

$$\mathbf{W}' = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}, r \ll \min(d, k)$$

$\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ 是 frozen pre-trained weight，$\mathbf{A}$ 初始化为 Gaussian，$\mathbf{B}$ 初始化为 0，保证训练开始时 $\Delta \mathbf{W} = 0$。这给了 VLM community 一个**低成本 fine-tune LLM** 的标准工具。

---

## 5. Task 演化（Section 4.3 — Table 4 解析）

最戏剧性的 shift：

| Task | 2023 | 2024 | 2025 | Trend | Slope |
|---|---|---|---|---|---|
| Reasoning/Instruction | 13.5% | 22.3% | 25.0% | **+11.5 pp** | +5.71 pp/yr |
| Grounding/Referring | 25.9% | 14.5% | 12.9% | **-13.0 pp** | -8.36 pp/yr |
| Retrieval | 8.5% | 6.8% | 8.3% | -0.2 pp | +0.53 pp/yr |
| Captioning | 6.2% | 4.9% | 4.4% | -1.9 pp | -0.53 pp/yr |
| VQA | 2.4% | 2.0% | 1.9% | -0.5 pp | -0.05 pp/yr |

**直觉**：Grounding 从 "end task" 退化成 "sub-capability of instruction-tuned VLM"。RefCOCO/phrase-grounding 这种 task formulation 不再是 paper 的 main contribution，**而是**变成 LLaVA-style system 的 internal module。

Reasoning/Instruction 是 VLM 的 main growth engine，slope +5.71 pp/yr 是所有 task 里最快的。

---

## 6. Training Paradigm（Section 4.4 — Table 5）

| Paradigm | 2023 | 2024 | 2025 | Trend |
|---|---|---|---|---|
| Pretrain + Finetune | 11.6% | 16.9% | 16.8% | +5.2 pp |
| Prompt/Prefix | 13.0% | 16.4% | 14.3% | +1.3 pp |
| Self/Weak/Semi-sup | 9.6% | 2.8% | 3.5% | **-6.1 pp** |
| Distillation | 4.2% | 4.8% | 4.0% | -0.4 pp |
| Instruction Tuning | 1.1% | 4.2% | 5.0% | **+3.9 pp** |
| LoRA/Adapters | 1.3% | 4.0% | 4.1% | +2.8 pp |

**Self/weak/semi-sup 下降 6.1 pp 是最 dramatic 的信号**：community 不再从头 train encoder。SimCLR、MoCo、BYOL 这些方法被 foundation model 吸收，**不再是**独立 contribution。

Instruction tuning 增长 +3.9 pp 印证了 LLaVA 范式的胜利。典型 recipe 是 **CLIP encoder (frozen) + projector + LLM (LoRA) + GPT-4 generated instruction data**。

---

## 7. Loss Family（Section 4.5 — Table 6）

| Loss | 2023 | 2024 | 2025 | Trend |
|---|---|---|---|---|
| Contrastive/InfoNCE | 10.8% | 5.6% | 5.1% | **-5.7 pp** |
| KL/Distillation | 5.6% | 6.6% | 5.8% | +0.3 pp |
| Triplet/Ranking | 1.0% | 0.7% | 0.5% | -0.5 pp |
| Cross-Entropy/Focal | 0.8% | 0.3% | 0.6% | -0.1 pp |
| MSE/L1/L2 | 0.3% | 0.4% | 0.3% | -0.0 pp |

**Contrastive loss 从 10.8% 跌到 5.1%，slope -2.07 pp/yr**。intuition：CLIP-style contrastive pretraining 已经 commoditized，新 paper 不 build encoder from scratch，**而是**做 instruction tuning，loss 变成 CE on tokens + KL distillation。

LLaVA 的 loss 就是标准 next-token prediction：

$$\mathcal{L}_{\text{LLaVA}} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, \mathbf{X}_v, \mathbf{X}_q)$$

其中 $\mathbf{X}_v$ 是 visual tokens，$\mathbf{X}_q$ 是 instruction tokens，$y_t$ 是 target response token。Image tokens 不参与 loss（masked）。

---

## 8. Dataset 提及率（Section 4.6 — Table 7）

| Dataset | 2023 | 2024 | 2025 | Trend |
|---|---|---|---|---|
| MS-COCO | 4.9% | 2.1% | 1.0% | -3.0 pp |
| ImageNet | 3.1% | 2.4% | 1.6% | -1.5 pp |
| LAION | 0.6% | 0.8% | 0.2% | -0.5 pp |
| RefCOCO/g/+ | 0.6% | 0.6% | 0.3% | -0.3 pp |
| Flickr30k | 0.8% | 0.3% | 0.2% | -0.7 pp |
| VQA-v2/OK-VQA | 0.4% | 0.2% | 0.3% | -0.2 pp |

**Legacy benchmark 在 abstract 里被提及的频率持续下降**。这反映了一个转变：papers 不再 dataset-name-drop，**而是**强调在 multi-task instruction suite（MMBench、MMMU、MMStar 等）上的 broad evaluation。

不过作者也 caveat：abstract under-report training data，尤其 generalist LMM 用 private mixture。InternVL3.5、Qwen3-VL 这些 2025 年 model 的真实 training data 远超 abstract 透露的。

---

## 9. Cross-venue Comparison（Section 5）

| Venue | 2025 VLM share | 2025 3D share | 2025 Diffusion share |
|---|---|---|---|
| CVPR | 39.5% | 23.1% | 25.7% |
| ICLR | 40.7% | 7.8% | - |
| NeurIPS (2024) | 30.5% | - | 11.6% |

CVPR 仍然 dominate 3D 和 Diffusion（3D 23.1% vs ICLR 7.8%），这反映了 CVPR 的传统强项。ICLR 在 VLM 上略微领先，可能因为 ICLR 接受更多 LLM-centric/methodology paper。NeurIPS 的 VLM ramp 较慢（30.5% @ 2024），但仍在增长。

---

## 10. 给研究者 / 工程师的 Actionable Advice

基于 paper Section 5 的 advice + 我的扩展：

1. **Frame classic perception as instruction following**：把 detection/segmentation/tracking 包装成 VLM 的 grounding sub-task，加 instruction tuning data。LLaVA-style + SAM/GroundingDINO tool use 是热门 template。

2. **Diffusion paper 强调 controllability + speed + distillation**：纯 model 训练已经 saturated。Rectified Flow、Consistency Model、LCM、SDS-style distillation 是 trending。

3. **Long-context + Video**：minute/hour-scale video understanding + memory efficiency 是 rising 方向。LLaVA-OneVision、Qwen2.5-VL、InternVL3.5 都在做 long video。

4. **Efficiency + Safety 是 acceptance signal**：sparse attention、cache-aware inference、watermarking、jailbreak defense 这些 cross-cutting theme 在上升。

5. **3D + VLM 是蓝海**：3D grounding、scene graph、robotics VLM、embodied agent、digital twin 是 2025 的 growing edge。3D share +0.7 pp 的 slope 看起来不大，但这是从 VLM 角度 co-mention 的 3D。

---

## 11. 这篇 survey 的 limitations

作者诚实承认：
- Lexicon-driven，precision 高但 recall 有限
- Abstract-only scope，training detail、loss、dataset 都 under-report
- 多 label paper，百分比不 sum to 100%

我自己加的 observation：
- **没有区分 conference paper vs workshop paper**
- **没有审稿信号**（接收率、reviewer score）的影响
- **没有 author-level analysis**（group、country、industry vs academia）
- **没有 code release、model weight release 的统计** — 这在 VLM 时代是重要 signal
- **没有 compute scale 数据**（FLOPs、GPU hours）

---

## 12. 关联联想（相关 paper 和 trend）

### 12.1 VLM scaling law
[Karamcheti et al., "Prismatic VLMs"](https://arxiv.org/abs/2402.07857) 研究了 VLM 的 scaling：vision encoder scale vs LLM scale vs data scale 的 trade-off。结论：vision encoder 太小会 bottleneck，太大浪费 compute；7B LLM + SigLIP-SO400M 是 sweet spot。

### 12.2 Native multimodal pretraining
InternVL3、Qwen2.5-VL、GPT-4o 这些 2024-2025 工作 push "native multimodal" — **不是**用 frozen CLIP + LLM，**而是**从头联合训练 vision+text token，让 vision token 进入 LM 的 next-token prediction。这可能是 2025-2026 的主导 trend，paper 里 LoRA+instruction 的数据滞后于这个 shift。

### 12.3 RLAIF / DPO for VLM
LLaVA-NeXT、LLaVA-OneVision 引入了 preference optimization (DPO)：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

$y_w$ 是 preferred response，$y_l$ 是 dispreferred，$\pi_{\text{ref}}$ 是 reference policy，$\beta$ 是 KL penalty。Survey 里 KL/Distillation +0.3 pp 的小涨可能就是这波 RLAIF/DPO 趋势的早期 signal。

### 12.4 Video-LLM
LLaVA-Video、Video-LLaVA、Qwen2.5-VL 的 video branch、InternVL3.5 的 long video 都是 2025 的重点。Token compression（e.g., 2D spatial + 1D temporal pooling）+ RoPE 时间维度扩展是关键技术。Survey 里 Video QA/Captioning share flat (1.3% → 1.2%)，但这是 abstract-level label 偏保守，真实 video-LLM 活动远超这个数字。

### 12.5 Diffusion 的工程化
Survey 说 Diffusion 从 8% → 19.2%，主题集中在 controllability、distillation、speed。相关代表：
- **Consistency Models** ([Song et al.](https://arxiv.org/abs/2303.01469))：single-step generation
- **Rectified Flow** ([Liu et al.](https://arxiv.org/abs/2209.03003))：straighten trajectory，Flux/SD3 用
- **LCM** ([Luo et al.](https://arxiv.org/abs/2310.04391))：few-step distillation
- **Flow Matching** ([Lipman et al.](https://arxiv.org/abs/2210.02747))：generalization of diffusion

### 12.6 3D 的 reconfiguration
NeRF share 在降，Gaussian Splatting 在升。3DGS 的 rasterization：

$$\mathbf{C}(\mathbf{u}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j), \quad \alpha_i = \sigma_i \exp\left(-\frac{1}{2}(\mathbf{u}-\boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{u}-\boldsymbol{\mu}_i)\right)$$

$\boldsymbol{\mu}_i$ 是 Gaussian center，$\boldsymbol{\Sigma}_i$ 是 covariance，$\sigma_i$ 是 opacity，$\mathbf{c}_i$ 是 color（用 spherical harmonics 表示）。可微 rasterization 让 3DGS 训练只需 ~10 min vs NeRF 的 hours。

### 12.7 DINOv3 在 VLM 里的崛起
2025 年越来越多 VLM 用 DINOv3 替代 CLIP ViT 做 vision encoder，因为 DINOv3 的 dense feature 更好（Gram anchoring 保证），对 grounding、detection sub-task 更友好。Qwen3-VL、InternVL3.5 都考虑过 DINOv3 backbone。

---

## 13. 最后的 take-away

这篇 survey 用 quant 方法证实了 community 的 vibe：**VLM 已经是 vision research 的 organizing center**。40% 的 paper share 意味着每 5 篇就有 2 篇 VLM-related。

如果你在 2026 年做 vision research：
- **(a)** 不用 LLM/instruction formulation 的纯 perception paper，acceptance 难度上升
- **(b)** Diffusion paper 要回答 "why not just call a black-box diffusion API" — controllability + speed 是 must
- **(c)** 3D + VLM 是蓝海，但要在 abstract 里 explicitly state 3D/depth/audio modality（survey 发现这种 cross-modal 提及是 acceptance signal）
- **(d)** Instruction tuning + LoRA + projector 是 default recipe，偏离这个 recipe 需要强 justification
- **(e)** Long-context video + reasoning 是 trending edge，值得押注

**intuition 总结**：整个 vision community 在 2023-2025 完成了一次范式重组。从 "build encoder from scratch with contrastive loss on curated dataset" → "adapt frozen foundation model with instruction data and PEFT"。这次重组的 driver 是 LLaVA + GPT-4 + LoRA + CLIP 四件套，把 visual perception 纳入了 instruction following 的 LLM 范式。

---

## References (web links)

- [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [ALIGN (Jia et al., 2021)](https://arxiv.org/abs/2102.05918)
- [BLIP (Li et al., 2022)](https://arxiv.org/abs/2201.12086)
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)
- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [DINOv3 (Siméoni et al., 2025)](https://arxiv.org/abs/2508.10104)
- [Grounding DINO (Liu et al., 2024)](https://arxiv.org/abs/2303.05499)
- [MoE / Switch Transformer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [LoRA (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)
- [Prefix Tuning (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)
- [Prompt Tuning (Lester et al., 2021)](https://arxiv.org/abs/2104.08691)
- [InstructBLIP (Dai et al., 2023)](https://arxiv.org/abs/2305.06500)
- [MiniGPT-4 (Zhu et al., 2023)](https://arxiv.org/abs/2304.10592)
- [Kosmos-2 (Peng et al., 2023)](https://arxiv.org/abs/2306.14824)
- [OFA (Wang et al., 2022)](https://arxiv.org/abs/2202.03052)
- [PaLI (Chen et al., 2022)](https://arxiv.org/abs/2209.06794)
- [SAM (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)
- [Qwen2.5-VL (Bai et al., 2025)](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-Omni (2025)](https://arxiv.org/abs/2509.17765)
- [InternVL3 (Zhu et al., 2025)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (Wang et al., 2025)](https://arxiv.org/abs/2508.18265)
- [LLaVA-OneVision-1.5 (An et al., 2025)](https://arxiv.org/abs/2509.23661)
- [ImageBind (Girdhar et al., 2023)](https://arxiv.org/abs/2305.05665)
- [Prismatic VLMs (Karamcheti et al., 2024)](https://arxiv.org/abs/2402.07857)
- [Consistency Models (Song et al., 2023)](https://arxiv.org/abs/2303.01469)
- [Rectified Flow (Liu et al., 2023)](https://arxiv.org/abs/2209.03003)
- [Latent Consistency Models (Luo et al., 2023)](https://arxiv.org/abs/2310.04391)
- [Flow Matching (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [3D Gaussian Splatting (Kerbl et al., 2023)](https://arxiv.org/abs/2308.14737)
- [Original NeRF (Mildenhall et al., 2020)](https://arxiv.org/abs/2003.08934)
- [LAION-400M (Schuhmann et al., 2021)](https://arxiv.org/abs/2111.02114)
- [LAION-5B (Schuhmann et al., 2022)](https://arxiv.org/abs/2210.08402)
- [Visual Genome (Krishna et al., 2017)](https://arxiv.org/abs/1602.07361)
- [RefCOCO (Yu et al., 2016)](https://arxiv.org/abs/1608.00272)
- [SimCLR (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)
- [MoCo (He et al., 2020)](https://arxiv.org/abs/1911.05722)
- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)

需要我深入展开任何具体方向（比如 LLaVA 的 stage-wise training 细节、Q-Former 的 query token 学习机制、Gaussian Splatting 的 rasterization derivation、DPO 在 VLM 上的具体实现），随时告诉我。
