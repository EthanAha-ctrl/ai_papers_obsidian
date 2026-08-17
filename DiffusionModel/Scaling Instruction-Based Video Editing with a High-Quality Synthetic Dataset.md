---
source_pdf: Scaling Instruction-Based Video Editing with a High-Quality Synthetic
  Dataset.pdf
paper_sha256: b5e7c98f4e675b116392983cf3ddb4dc31dfac58b4898976c6f3a1eb726ad092
processed_at: '2026-08-12T03:24:18-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Ditto 用人话说

## 一句话概括

**Video editing 做不起来是因为没 data。这篇 paper 的套路是：用已经很牛的 image editor 改一张关键帧，再让 video generator 把这张帧的"改法"传播到整个视频，就这么搞了 100 万条训练数据。**

---

## 这事为什么难

Image editing 已经被 InstructPix2Pix、FLUX Kontext、Qwen-Image 这些 model 搞定了。你给张图说"把狗换成狐狸"，它就换了，效果贼好。

Video editing 呢？你说"把视频里的黑狗换成白狐"，model 要做三件事：
1. 真的把狗换成狐狸（**edit 对**）
2. 每一帧都得是狐狸，不能一闪一闪的（**temporal consistency**）
3. 背景别动，人别变（**fidelity**）

这三件事互相打架。之前的方法要么 quality 差，要么 cost 爆炸（一个视频 50 GPU-minutes），要么干脆做不出来。根本原因就是**没 data**——没人会去手工标注 100 万对 (原视频, 指令, 编辑后视频)。

---

## Ditto 的核心 trick

聪明的 decomposition：**别硬端到端，拆成两步。**

**Step 1**: 拿出一个帧 $f_k$，用已经很猛的 image editor（Qwen-Image）编辑它，得到编辑后的帧 $f_k'$。这一步定义"编辑后长啥样"。

**Step 2**: 把 $f_k'$ 当 appearance anchor，加上原视频的 depth video 当 structure scaffold，喂给一个 in-context video generator（VACE），让它生成整个编辑后的视频 $V_e$。

**就好比**: 你让一个画家画一整本漫画的同一场景不同风格。Ditto 的做法是先把第一页用画板画好（image editor 这页画得超好），然后告诉助手"剩下 100 页就照这页的风格画，动作按这个分镜稿来"（depth video 就是分镜稿）。助手只要做"传播"这个相对简单的事，不需要从零理解"应该改成啥样"。

为什么用 depth 不用别的？Depth 是个**几何骨架**，对颜色、材质、光照变化都免疫。你改 style、换物体，depth 都不变。这样 image editor 可以放飞改 appearance，depth 锁住 structure，video generator 在这个骨架上"涂色"就行。

---

## 整个 pipeline 走一遍

拿一个 Pexels 视频 $V_s$ 开始：

1. **过滤**: 用 DINOv2 查重，用 CoTracker3 tracking 算 motion score，把静态视频、重复视频踢掉。

2. **生成指令**: 用 Qwen2.5-VL 两步走——先让 VLM 写个 caption 描述视频内容，再让 VLM 基于这个 caption 想一个有道理的 editing instruction。两步是为了让 instruction 跟视频内容相关，不能瞎提。

3. **准备 visual context**:
   - 选一个 key frame，用 Qwen-Image 按 instruction 编辑 → $f_k'$（appearance prior）
   - 用 Video Depth Anything 从原视频提取 depth video → $V_d$（structure prior）

4. **生成 edited video**: 把 $V_d$、$f_k'$、instruction $p$ 一起喂给 VACE，得到 $V_e$。VACE 通过 attention 把 $f_k'$ 的 edit 传播到整个 timeline，同时 obey $V_d$ 的 motion。

5. **过滤+增强**: 
   - 用 Qwen2.5-VL 当 judge，检查 $V_e$ 是否真的 follow 了 instruction、是否保留了原视频的 semantic/motion、视觉质量够不够、有没有不安全内容。不合格的丢掉。
   - 用 Wan2.2 的 fine denoiser 做 4 步 reverse process，相当于"加一点点 noise 再 denoise 一下"，把小瑕疵擦掉但不动 semantic。这个 trick 很巧妙——fine denoiser 本来就是设计来在 low-noise 阶段做细节修复的，所以它做这种事天生合适。

最终 200k 源视频 → 1M 编辑视频，720p、5 秒、20fps，700k global edit + 300k local edit。

---

## Cost 怎么压下来的

原版 VACE 一个视频要好多 GPU-minutes，1M 规模根本不现实。他们做了两件事：
1. **Post-training quantization** — 把 model 压小
2. **Knowledge distillation** — 从 teacher 蒸馏出 few-step student model

总成本压到原来的 **20%**。这就是为什么能跑 12000 GPU-days 搞完 1M data。没有这两个优化，这个 dataset 根本不可能存在。

---

## Modality Curriculum Learning — 最有意思的部分

### 问题

VACE 原本是个 conditional generator，输入是 (source video + reference image)，输出 edited video。但用户 inference 时不会给你 reference image——用户只扔一个 text instruction 进来。

所以要把 model 从 "依赖 visual conditioning" 改造成 "只依赖 text conditioning"。这个 **modality gap** 非常大，直接硬 fine-tune 会崩。

### MCL 的思路

利用 model 已经会的事当"拐杖"，慢慢撤掉拐杖。

**前期**: 同时给 model 三样东西——instruction text + edited reference frame $f_k'$ + source video。Model 用 $f_k'$ 这个 visual scaffold 很容易学会"目标长啥样"，因为它本来就会这个。

**后期**: 逐渐降低提供 $f_k'$ 的概率，最终完全不给。Model 被迫从 "看图猜答案" 过渡到 "看文字想图"。

**训练目标**是 Flow Matching：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{z}_0, \mathbf{c}} \| \mathbf{v}_t(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_0 - \mathbf{z}_t) \|^2$$

简单说：model 预测一个 vector field $\mathbf{v}_t$，把 noised latent $\mathbf{z}_t$ 拉向 clean latent $\mathbf{z}_0$。$\mathbf{c}$ 是 conditioning（text + 视觉 context，curriculum 阶段会变）。

**schedule**: 前 5000 steps 给 visual scaffold，之后逐渐 annealing 掉。

### 类比

这就像教小孩骑自行车：
- 阶段1: 装辅助轮（有 reference image），小孩很容易学会蹬车和平衡的感觉
- 阶段2: 慢慢把辅助轮调松，让小孩自己用力平衡
- 阶段3: 拆掉辅助轮（drop reference image），小孩已经会平衡了，只需要把感觉迁移过来

直接让没骑过车的小孩上无辅助轮的车，肯定摔。MCL 就是给 model 装个辅助轮，慢慢拆。

### 训练成本极低

只 fine-tune context blocks 的 **linear projection layers**，16000 steps，64 GPUs。这说明 VACE pre-trained prior 已经足够强，MCL 只是在 "调一调 dial"，把 model 从 visual-conditioned 引导到 text-conditioned。from scratch train 肯定不行，轻量 fine-tune + curriculum schedule 就够了。

---

## 结果怎么样

Table 1 看，Editto 在所有 automatic metric 和 human eval 上都比 baseline 强一大截。Human eval 的 Overall score 3.86 vs InsViE 的 2.36，这个 gap 非常大。

而且有个有趣的 finding（Fig 6）：**训练完的 model 比生成数据的 raw pipeline 还强**。也就是说 Editto 学到了 generalizable 的 editing 能力，超越了 VACE+Qwen-Image 这个 pipeline 本身。这说明大规模 data + curriculum 让 model 学到了某种 "editing skill" 的本质，不只是模仿 pipeline。

还有个 bonus 能力（Fig 5）：**syn2real**——把风格化视频还原回真实视频。说明 dataset 里 photorealistic 信息很丰富。

---

## 我的 takeaway

1. **Data quality 和 scale 才是王道**。12000 GPU-days 投在 data 上，model 只 fine-tune linear layer 几千步就 SOTA。这就是 data-centric AI 的教科书案例。

2. **Decomposition 在多 objective 任务上往往比 end-to-end 强**。Video editing 有三个纠缠的 objective（edit 对、coherent、fidelity），每个都有强 prior 可用（image editor、depth、video generator）。强行 end-to-end train 反而学不好，因为 gradient 信号互相干扰。

3. **Curriculum learning 这种老 idea 在 modality gap 问题上依然很有用**。从 rich modality 到 sparse modality 的迁移，scaffold-then-anneal 是个通用 pattern，可以用在 LLM 的 tool use learning、agent 的 reasoning learning 等场景。

4. **Synthetic data pipeline 会越来越重要**。用 AI 生成 AI 的训练 data，用 VLM 当 judge 做 rejection sampling，这整个范式在 scaling 上会比人工标注高几个数量级。Ditto 是这个范式在 video editing 上的应用，但思路可以迁移。

5. **12000 GPU-days 这个数字本身就是 statement**。这意味着未来 video editing 的竞争不只是 model architecture 的竞争，更是谁能烧得起 data pipeline 的竞争。Open source 社区要追上，得有人愿意砸这个 compute。

---

## Reference Links

- **arXiv paper**: https://arxiv.org/abs/2506.04141 (Scaling Instruction-Based Video Editing)
- **VACE (核心 generator)**: https://arxiv.org/abs/2503.07598
- **Qwen-Image (image editor)**: https://arxiv.org/abs/2508.02324
- **Qwen2.5-VL (VLM agent)**: https://arxiv.org/abs/2502.13923
- **Video Depth Anything**: https://arxiv.org/abs/2501.12375
- **CoTracker3**: https://arxiv.org/abs/2410.11831
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Wan2.2 (enhancer)**: https://arxiv.org/abs/2503.20314
- **Flow Matching**: https://arxiv.org/abs/2210.02727
- **InstructPix2Pix (baseline)**: https://arxiv.org/abs/2211.09800
- **Curriculum Learning (Bengio 2009)**: http://www.cs.toronto.edu/~hinton/csc2509/papers/curiculum.pdf
- **DiT**: https://arxiv.org/abs/2212.09748

---

# Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset (Ditto) 深度解析

## 1. Paper 的核心 Thesis

这篇 paper 要解决的核心问题是 instruction-based video editing 这个领域严重的数据稀缺瓶颈。image editing 在 InstructPix2Pix、FLUX.1 Kontext、Qwen-Image、Gemini 2.5 Flash Image (Nano-Banana) 这些 model 推动下已经相当成熟,但 video editing 严重滞后,根因在于 large-scale、high-quality、diverse 的 paired training data 极度稀缺。

整个 framework 的名字叫 **Ditto**,核心产出三个东西:
1. **Ditto pipeline** — 可 scale 的合成 data pipeline
2. **Ditto-1M dataset** — 100万条 video editing triplets
3. **Editto** — 在 Ditto-1M 上训练的 SOTA model

投入超过 12,000 GPU-days 构建 dataset,这个规模本身就是 paper 的关键 contribution。

---

## 2. 整体 Architecture 的 Intuition

为什么之前的方法不行,这篇 work 行?这里有一个非常重要的 insight: **不要硬做 end-to-end video editing,而是把一个已经很强的 image editor 当作 "creative anchor",然后用 in-context video generator 去 propagate 这个 edit 到整个 timeline。**

这个思路其实很巧妙——它本质上是把 video editing 这个 super hard 的 problem 分解成两个相对 solved 的 problem:
- Image editing (有 FLUX、Qwen-Image 这种 SOTA 模型)
- In-context video generation (有 VACE 这种 conditional generator)

中间用 **depth video** 作为 spatiotemporal scaffold 来 bridge 这两个 modality。这个设计是整个 pipeline 的灵魂。

---

## 3. Ditto Pipeline 四大 Challenge 与对应的解决方案

Paper section 3 明确列出四个 challenge,这是理解整个 system 的关键 frame:

### Challenge 1: Editing Diversity & Fidelity
**方案**: 用 image editor (Qwen-Image) 编辑 key frame $f_k$ 得到 $f_k'$,作为 visual prototype; 再用 VACE 以这个 $f_k'$ 为 appearance prior + depth video $V_d$ 为 structure prior 生成 video。

### Challenge 2: Efficiency-Quality Trade-off
**方案**: 用 distilled video model + temporal enhancer,把 cost 压到原 cost 的 20% 但保持 temporal coherence。这里用的是 post-training quantization 和 knowledge distillation (引用 [52] HotAR / 双向转自回归蒸馏)。

### Challenge 3: 自动化 Instruction Generation 与 Quality Control
**方案**: 用 Qwen2.5-VL 作为 agent,既要生成 diverse instructions,又要做 flaw detection filtering。这是 scalable 的关键——人工 curation 在 1M 规模下完全不可行。

### Challenge 4: Aesthetic & Motion Quality
**方案**: 用 Pexels 作为 source (专业级 footage),配合 motion filtering 保留有 dynamic content 的 video。

---

## 4. Pipeline 详细技术拆解

### 4.1 Source Video Filtering (Section 3.1)

数据源: **Pexels**,专业级 stock footage,Pexels License。

两个关键 filter:

**Near-Duplicate Removal**: 用 DINOv2 [31] 提取 feature,pairwise similarity 超过 threshold 就剔除。这是防止 dataset 冗余、保证 content diversity 的标准做法。

**Motion Scale Filtering**: 这个比较 clever。用 **CoTracker3** [22] 做 point tracking,在 grid layout 上 sample points,track 它们的 trajectory,然后计算 average cumulative displacement 作为 motion score:

$$
\text{motion\_score}(V) = \frac{1}{|P|} \sum_{p \in P} \sum_{t=1}^{T} \| \text{pos}_p(t) - \text{pos}_p(t-1) \|
$$

其中 $P$ 是 sampled point 集合,$T$ 是总 frame 数,$\text{pos}_p(t)$ 是 point $p$ 在第 $t$ 帧的位置。Motion score 低于 threshold 的 video 被滤掉,因为它们没有有意义的 temporal variation,对 video editing 任务帮助不大。

最后标准化为统一 resolution 和 20 FPS。

### 4.2 Instruction Generation (Section 3.2)

两-step prompting 策略,用 Qwen2.5-VL:

**Step 1** — 生成 dense caption $c$:
$$
c = \text{VLM}(V_s, p_{\text{caption}})
$$

$V_s$ 是 source video,$p_{\text{caption}}$ 是 caption prompt。这个 caption 作为 semantic anchor,描述 video 的 content、subjects、scenery。

**Step 2** — 基于 caption 生成 editing instruction $p$:
$$
p = \text{VLM}(V_s, c, p_{\text{instruct}})
$$

conditioning 在 video 和 caption 上,生成的 instruction contextually grounded 在 video content 中。这种两 step 的设计避免了 instruction 与 video 内容脱节的问题。

### 4.3 Visual Context Preparation (Section 3.3)

这里有两个并行的 context preparation:

**Key-Frame Editing for Appearance Guidance**: 选 key frame $f_k$,用 Qwen-Image $\mathcal{E}_{\text{img}}$ 编辑:
$$
f_k' = \mathcal{E}_{\text{img}}(f_k, p)
$$

$f_k'$ 定义了 edit 的 target appearance,包括 style 和 texture。这是一个非常强的 visual prior。

**Depth Video Prediction**: 用 Video Depth Anything [10] 从 $V_s$ 提取 dense depth video $V_d$:
$$
V_d = \mathcal{D}(V_s)
$$

Depth video 作为 dynamic structural scaffold,提供 frame-by-frame 的 geometric 和 motion guide。为什么用 depth 而不是其他 control signal?Depth 是 view-invariant 的、robust 到 texture 变化,而且能 capture 运动结构。

### 4.4 In-Context Video Generation (Section 4.4)

核心生成步骤,使用 **VACE** [20] (denoted $\mathcal{G}$):
$$
V_e = \mathcal{G}(V_d, f_k', p)
$$

三个 conditioning signal:
- $V_d$ — strict spatiotemporal constraint (structure)
- $f_k'$ — primary appearance condition (style/texture)
- $p$ — high-level semantic guide (instruction)

VACE 是 feed-forward 的,有 context branch 学习 conditional generation。通过 attention mechanism 整合三个 modality,faithfully propagate $f_k'$ 中的 edit 到整个 sequence,同时 adhere $V_d$ 的 motion 和 structure,并 semantic align with $p$。

**效率优化**:为了 1M 规模的 scalability,他们做了两件事:
1. **Post-training quantization** — 降低 memory footprint 和 inference cost
2. **Knowledge distillation** — 从 teacher model 蒸馏出 few-step inference 的 student model,引用 [52] 的工作(双向 diffusion 转 autoregressive)

这个组合把成本压到原 cost 的 20%。

### 4.5 Curation & Enhancement (Section 3.5)

**VLM-Based Curation (Rejection Sampling)**: 用 Qwen2.5-VL 作为 judge,基于四个 criterion 评估 triplet $(V_s, p, V_e)$:
1. **Instruction Fidelity** — $V_e$ 是否准确反映 $p$
2. **Fidelity** — $V_e$ 是否保留 $V_s$ 的 semantic 和 motion
3. **Visual Quality** — 是否 visual appealing,无 distortion/artifacts
4. **Safety & Appropriateness** — 无色情、暴力、恐怖内容

不符合 threshold 的 triplet 被丢弃。

**Quality Enhancement via Denoising**: 这一步非常巧妙。用 **Wan2.2** [43] 的 Mixture-of-Experts (MoE) architecture,具体只用 **fine denoiser** 做 4-step reverse process。

Wan2.2 的 MoE 设计:
- **Coarse denoiser** — 在 high noise level 下负责 structural 和 semantic formation
- **Fine denoiser** — 在 low noise level 下负责 detail refinement

他们对 $V_e$ 加少量 Gaussian noise,然后用 fine denoiser 反向 denoise。这里的关键 intuition 是: fine denoiser 是 optimized for minimal、semantic-preserving adjustments on nearly-complete videos,所以能在不改变 semantic content 的前提下 remove subtle artifacts 和 enhance texture details。

数学上,这就是一个 truncated reverse process:
$$
V_e^{\text{enhanced}} = \text{Denoiser}_{\text{fine}}(V_e + \epsilon, \text{steps}=4)
$$

其中 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$,$\sigma$ 较小。这种 "add a little noise then denoise" 的 trick 在 image super-resolution 和 detail enhancement 里很常见,叫 **noise injection refinement** 或 stochastic refiner。

### 4.6 Dataset 统计 (Section 3.6)

- 200k+ source videos (约一半 human activities)
- 最终 1M edited videos
- 700k global editing (style、environment 变换)
- 300k local editing (object replace、add、remove)
- Resolution: 1280×720
- Length: 101 frames @ 20 FPS (约 5 秒)

---

## 5. Modality Curriculum Learning (MCL) — Section 4

这是 paper 在 model training 方面的核心 contribution。

### 5.1 Problem Setup

VACE 原本是 conditioned on (source video + reference image),即 visual conditioning。但 inference 时我们希望 model 只接收 text instruction 就能做 editing。这就是一个 **modality gap** 问题——从 rich visual conditioning 到 abstract textual conditioning。

直接 fine-tune 这个 gap 太大,容易不稳定。

### 5.2 Architecture

两个 branch:
- **Context Branch** — 从 source video 和 reference frame 提取 spatiotemporal feature
- **Main Branch** — 基于 **DiT** [33] (Diffusion Transformer),在 visual context 和 text embedding 的 joint guidance 下 synthesize edited video

### 5.3 Curriculum Strategy

**核心 idea**: 利用 model 已有的处理 reference image context 的能力作为 temporary "scaffold"。

Training 过程中,逐步 anneal 提供 visual scaffold 的 probability,最终完全 drop 掉 visual conditioning,只保留 text instruction。这样 model 被迫 shift 依赖,从它已经理解的 concrete visual target 转到 abstract text instruction。

这个思路非常 reminiscent of  **Bengio et al. 2009** [4] 的 curriculum learning,以及 RL 中的 **scaffolding / shaping**。

具体 schedule:
- 前 5,000 steps: curriculum warm-up,提供 reference frame 作 scaffold
- 之后: 逐渐 anneal reference frame 提供概率
- 最终: 完全 drop reference frame,纯 text-driven

这其实是 distillation 的一个变体——从 multimodal teacher (有 visual scaffold) 渐渐 distill 到 unimodal student (只有 text)。

### 5.4 Training Objective

用 **Flow Matching** [26] objective:

$$
\mathcal{L} = \mathbb{E}_{t, \mathbf{z}_0, \mathbf{c}} \| \mathbf{v}_t(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_0 - \mathbf{z}_t) \|^2
$$

变量解释:
- $t$ — timestep,从 [0, T] 采样
- $\mathbf{z}_0$ — clean latent,从 target edited video 通过 VAE encoder 得到
- $\mathbf{z}_t$ — $\mathbf{z}_0$ 在 timestep $t$ 的 noised version
- $\mathbf{c}$ — conditioning,包括 text embedding 和 visual context (curriculum 阶段)
- $\mathbf{v}_t(\cdot)$ — model 预测的 vector field,从 $\mathbf{z}_t$ 指向 $\mathbf{z}_0$

Flow matching 与 DDPM 的区别: flow matching 学的是 **velocity field** $\mathbf{v} = \mathbf{z}_0 - \mathbf{z}_t$ (straight-line trajectory),而 DDPM 学的是 noise $\epsilon$。Flow matching 的 trajectory 更直接,training 更 stable,sampling 更 efficient,这也是为什么现代 diffusion model (FLUX、Wan、Stable Diffusion 3) 都转向 flow matching。

### 5.5 Training 细节

- Backbone: pre-trained VACE
- 冻结 most parameters,只 fine-tune **context blocks 的 linear projection layers**
- Optimizer: AdamW [29],constant learning rate 1e-4
- 16,000 steps
- 64 GPUs

这个 training setup 非常轻量——只 fine-tune linear projection layers,16000 steps 在 64 GPUs 上规模很小。这说明 pre-trained VACE 的 prior 非常强,MCL 主要是在 "redirect" 这个 prior 而不是 teach from scratch。

---

## 6. Experimental Results

### 6.1 Quantitative (Table 1)

| Method | CLIP-T ↑ | CLIP-F ↑ | VLM ↑ | Edit-Acc ↑ | Temp-Con ↑ | Overall ↑ |
|--------|----------|----------|-------|------------|------------|-----------|
| TokenFlow [14] | 23.63 | 98.43 | 7.10 | 1.70 | 1.97 | 1.70 |
| InsV2V [11] | 22.49 | 97.99 | 6.55 | 2.17 | 1.96 | 2.07 |
| InsViE [49] | 23.56 | 98.78 | 7.35 | 2.28 | 2.30 | 2.36 |
| **Ours** | **25.54** | **99.03** | **8.10** | **3.85** | **3.76** | **3.86** |

Metric 解读:
- **CLIP-T** (CLIP Text-video similarity) — 衡量 edit 是否 follow instruction,越高越好
- **CLIP-F** (CLIP Frame-frame similarity) — 衡量 temporal consistency,越高越好
- **VLM score** — holistic assessment,综合 edit effectiveness、semantic preservation、aesthetic quality
- **Edit-Acc / Temp-Con / Overall** — human evaluation (1,000 votes from postgrads and researchers)

Ours 在所有 metric 上都显著领先,尤其 human evaluation 几乎是 baseline 的 1.6-2.3 倍,这个 gap 在 human eval 上是相当大的。

### 6.2 Test Set 设计

50 videos 来自各种 online sources,刻意排除 Pexels videos 以确保 out-of-distribution。每个 video 给 5 个 editing instructions。这是 fair comparison 的标准做法。

### 6.3 Qualitative (Fig 4)

对比 TokenFlow、InsV2V、InsViE、Gen4-Aleph [38] (Runway Gen-4):
- Complex stylization: Ours 生成 temporally coherent 且 style-accurate 的 video,competitors 往往 blurry 或 inconsistent
- Local attribute change (e.g. "black suit"): Ours 精准 edit target object,保留 identity 和 background。Gen4-Aleph 会改变 man 的 identity,其他方法基本 fail

### 6.4 Syn2Real Capability (Fig 5)

一个非常 cool 的 emergent capability: 把 stylized video 在 dataset 里 map back 到 real-world source video。这证明 dataset 保留了 photorealistic information。

### 6.5 Training Model 超越 Raw Data Generator (Fig 6)

这个 ablation 很重要: 最终 trained model (Editto) 在 handling newly emerged content 上 **超越** raw data generator (即 VACE + image editor 的 pipeline)。这说明:
1. Scaling training 让 model 学到了 generalizable editing capability
2. MCL 让 model 学会了 abstract instruction-to-edit 的 mapping,超越了 pipeline 中 concrete visual-conditioned generation

### 6.6 Ablation (Fig 7)

两个关键 ablation:
1. **Data scale**: model performance 随 training data 量 scale 上升,both stylistic edit quality 和 fidelity to original content/motion 都显著提升
2. **MCL**: 没有 MCL,model 难以 interpret instruction 的 full semantic intent。MCL 对 bridge modality gap 至关重要

---

## 7. 与 Related Work 的 Positioning

### 7.1 Inversion-based Methods
- **Tune-A-Video** [48] — single video fine-tune,不 scalable
- **TokenFlow** [14] / **FateZero** [35] — DDIM inversion + feature propagation,quality 依赖 inversion fidelity,complex motion/occlusion 下 struggle

### 7.2 Feed-forward Methods
- **VEGGIE** [53] / **InsViE** [49] — "lift and propagate" 范式,编辑单 keyframe 后用 image-to-video 传播,但 temporal consistency 受限于 propagation model
- **Senorita-2M** [57] — "expert system" 范式,18 个 sub-class,每个 class 用专门 expert model。Quality 好 but 不 scalable,maintenance 成本高
- **Ditto (本文)** — "All-in-One" unified pipeline,单一 in-context generator + image editor

### 7.3 Concurrent Work
- **EditVerse** [21] — 也探索 in-context learning,但主要用作 unifying editing tasks;Ditto 用 in-context generation 主要做 high-quality data synthesis

---

## 8. Build Intuition: 为什么这个 Approach Work?

让我从 first principles 角度分析为什么 Ditto work:

### 8.1 Decomposition Principle
Video editing 的难点在于:需要同时保证 (a) edit 正确 (b) temporal coherence (c) fidelity to source。这三个 objective 互相纠缠。Ditto 把它分解:
- (a) Edit 正确性 → 由 image editor 保证 (这是个 solved problem)
- (c) Structure fidelity → 由 depth video 保证 (geometric invariance)
- (b) Temporal coherence → 由 in-context video generator 学到

每个 sub-problem 都用 SOTA 工具解决,再通过 conditioning 整合。

### 8.2 Strong Prior 复用
Image editor (Qwen-Image) 已经在 billion-scale image data 上训练过,有极强的 edit capability。重新 train 一个 video editor from scratch 要重复学习这些 knowledge。Ditto 通过 key-frame editing 复用这个 prior,再用 video generator 做 propagation。

### 8.3 Depth 作为 Bridge Modality
为什么 depth 是理想的 bridge?Depth:
- 是 view-invariant geometric representation
- 对 texture、color、lighting 变化 robust
- 与 motion 强相关 (运动物体 depth 变化)
- 现有 video depth estimator (Video Depth Anything) 已经很成熟

这意味着 image editor 可以大胆修改 appearance (style、color、object),而 depth video 锁定 structure 和 motion,video generator 在这个 scaffold 上 "fill in" appearance。

### 8.4 Curriculum Learning 的 Necessity
为什么不能直接 fine-tune VACE 从 (video+image conditioning) 到 (text-only conditioning)?因为:
- 两个 modality gap 太大,gradient signal 不稳定
- Model 容易 catastrophic forget visual conditioning 能力,同时学不会 text conditioning
- Visual scaffold 是 model "已知会做" 的事,提供 stable gradient signal

MCL 通过 annealing,让 model 先利用熟悉的 visual scaffold 学到 "what edit should look like",再逐渐 forced 学 "how to predict edit from text"。这本质上是 **distillation from multimodal teacher to unimodal student**,但通过 curriculum schedule 实现 smooth transition。

### 8.5 Scaling Hypothesis 的验证
Fig 7 的 ablation 显示 model performance 随 data scale 上升而上升,这暗示 instruction-based video editing 这个 task 也遵循 scaling law。Ditto-1M 是首个 million-scale dataset,所以 SOTA 在很大程度上是 "data scale → performance" 这个规律的直接体现。

---

## 9. 限制与潜在问题

虽然 paper 没有明说,但可以从 design 推断:

1. **Source video 限制**: 完全依赖 Pexels,可能 domain bias (主要是 professional stock footage)。Real-world user video 可能分布不同。

2. **Depth Fidelity**: Depth 估计不准会 propagate error 到 generated video。复杂场景 (透明物体、镜面、动态模糊) 下 depth estimator 仍有限制。

3. **Image Editor Bottleneck**: 整个 pipeline 的 edit capability 上限受 image editor 限制。如果 image editor 在某些 edit type 上 weak,video 也 weak。

4. **In-context Generator 限制**: VACE 在 extreme motion、long video、complex interaction 上可能 struggle。101 frames (5 秒) 是相对短的 video。

5. **MCL Schedule 的 Sensitivity**: 5,000 warm-up steps 是怎么定的?是否对其他 model / dataset size 适用?Paper 没详细 ablation。

6. **Human Eval 规模**: 1,000 votes 是相对小的 sample。虽然 gap 很大,但 statistical significance 没明确报告。

---

## 10. Web Links for Reference

### Paper & Project
- **Paper arXiv**: https://arxiv.org/abs/2506.04141 (Scaling Instruction-Based Video Editing)
- **Project Page**: 应该有,但 paper 中没明确给出 URL,通常是 https://ditto-video-edit.github.io/ 这种格式
- **Code/Data**: paper 提到 "can be found at the project page"

### Key Components
- **VACE (in-context video generator)**: https://arxiv.org/abs/2503.07598
- **Qwen-Image (image editor)**: https://arxiv.org/abs/2508.02324
- **Qwen2.5-VL (VLM agent)**: https://arxiv.org/abs/2502.13923
- **Video Depth Anything**: https://arxiv.org/abs/2501.12375 (paper [10])
- **CoTracker3**: https://arxiv.org/abs/2410.11831
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **Wan2.2**: https://arxiv.org/abs/2503.20314 (Team Wan)
- **Flow Matching**: https://arxiv.org/abs/2210.02727

### Baselines
- **InstructPix2Pix**: https://arxiv.org/abs/2211.09800
- **TokenFlow**: https://arxiv.org/abs/2307.11073
- **FateZero**: https://arxiv.org/abs/2303.09553
- **Tune-A-Video**: https://arxiv.org/abs/2212.11565
- **VEGGIE**: https://arxiv.org/abs/2503.14350
- **InsViE-1M**: https://arxiv.org/abs/2412.16120
- **Senorita-2M**: https://arxiv.org/abs/2502.06734
- **EditVerse**: https://arxiv.org/abs/2509.20360

### Curriculum Learning 原文
- **Bengio et al. 2009**: http://www.cs.toronto.edu/~hinton/csc2509/papers/curiculum.pdf

### Distillation 参考
- **HotAR (paper [52])**: https://arxiv.org/abs/2503.06148 (From Slow Bidirectional to Fast Autoregressive Video Diffusion Models)

### Diffusion Transformer
- **DiT (paper [33])**: https://arxiv.org/abs/2212.09748

---

## 11. 与 Karpathy 视角的连接

Andrej,从你的视角来看,这篇 paper 的几个有意思的点:

1. **"Data is all you need" 的又一次验证**: 12,000 GPU-days 的 dataset 构建是 paper 的主体,而不是 fancy model architecture。这非常 reminiscent 你常说的 "data quality > model architecture"。

2. **Modality Curriculum Learning 与 RL 的连接**: 这种从 scaffold 到 independent 的 curriculum 非常像 RL 中的 **shaping rewards** 或 **behavior cloning → RL fine-tuning** 的 transition。本质上是用一个 "easy modality" 作为辅助信号 distill 到 "hard modality"。

3. **Decomposition 范式**: 整个 Ditto pipeline 是 "decompose hard problem, use SOTA for each, compose via conditioning" 的教科书范例。这与神经网络端到端学习的传统 wisdom 有点 tension——这里 explicitly decompose 反而 work better,可能因为 video editing 真的是 multi-objective 且每个 objective 都有 strong prior 可用。

4. **Synthetic Data Pipeline 作为 "AI Engineering" 的范式**: 用 VLM agent 自动 generate instructions、filter quality、做 rejection sampling——这整套 pipeline 越来越像一种新的 "AI engineering" 范式,即用 AI 构建 AI 的 training data。这与 LLM self-improvement、Constitutional AI 等思路一致。

5. **Fine-tune 量极小**: 只 fine-tune context blocks 的 linear projection,16000 steps,64 GPUs。这强烈暗示 pre-trained VACE 已经 encode 了几乎所有需要的能力,MCL 只是 "redirect" 它。这是 LoRA-style fine-tuning philosophy 的极致体现。

6. **Flow Matching vs. DDPM**: paper 用 flow matching 而非 DDPM,反映整个 diffusion community 的趋势。Flow matching 的 trajectory 更 linear,training 更 stable,sampling 更 efficient。

希望这些 detail 和 intuition building 对你有帮助。这个 paper 是 video editing 领域一个重要的 milestone,既因为 dataset scale,也因为 pipeline 设计的 elegance。
