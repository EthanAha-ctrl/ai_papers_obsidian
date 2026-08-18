---
source_pdf: Bernini Latent Semantic Planning for Video Diffusion.pdf
paper_sha256: 152674725cd6cb70e6a52075cd2b11564191a6a58022345a111226dad87d424a
processed_at: '2026-08-18T02:37:32-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Bernini 用人话说

## 一句话版本

**让 MLLM 当"导演"在脑子里把画面想清楚, 然后让 diffusion model 当"摄影师"把画面拍出来, 两人用一套共同的语言 (ViT embedding) 沟通.**

Project page: https://bernini-ai.github.io

---

## 为什么要这么搞

现在做 video generation + editing 的痛点:

**痛点 1**: 单纯的 diffusion model (比如 Wan2.2 [70](https://arxiv.org/abs/2503.20314), Sora [53](https://openai.com/sora)) 很会画画, 但是"听不懂人话". 你给它一句复杂的 editing instruction, 比如"把视频里那只白虎的动作改成在挠另一只老虎", 它经常 edit 不对, 因为它没有真正的 semantic understanding.

**痛点 2**: MLLM (比如 Qwen2.5-VL [69](https://arxiv.org/abs/2502.13923), GPT-4o) 很会理解人话, 也能 reasoning, 但是它不会生成 high-quality video. 它能输出 text 或者 coarse visual tokens, 但是 photorealistic video synthesis 这个能力, diffusion model 比 autoregressive model 强太多.

**痛点 3**: 之前的 unified model (Emu3 [73](https://arxiv.org/abs/2409.18869), Janus [79](https://arxiv.org/abs/2503.07229), Show-o [82](https://arxiv.org/abs/2408.12528)) 试图用一个 transformer 同时做 understanding 和 generation, 结果两边互相干扰, generation quality 上不去, understanding capability 又被 compromise.

**Bernini 的 insight**: 与其强行 merge 两个能力到一个 model 里, 不如让它们各自干自己最擅长的事——MLLM 做 planning (语义规划), DiT 做 rendering (像素合成). 关键问题是: **这两个 component 之间用什么"语言"沟通?**

---

## 核心设计: ViT embedding space 作为 bridge

### 先理解 MLLM 内部发生了什么

当你给 MLLM (比如 Qwen2.5-VL) 一张图片, 它内部是这样处理的:

```
image pixels → ViT encoder → ViT embeddings (一堆 vector)
                                    ↓
              MLLM transformer (self-attention 融合信息)
                                    ↓
                        hidden states (contextualized)
```

ViT embeddings 就是 MLLM "看到"图片的 internal representation. MLLM 后续所有的 reasoning, 都是基于这些 embeddings 来做的.

### Bernini 的关键 idea

既然 MLLM 本来就是用 ViT embeddings 来"理解"视觉内容的, 那让 MLLM 去"生成" target 的 ViT embeddings, 不就等于让 MLLM 在自己最熟悉的 representation space 里做 planning 吗?

```
传统做法: MLLM → text tokens → DiT (text-to-video)
            ↑ 信息瓶颈, 高级 semantic 信息损失

Bernini:  MLLM → target ViT embeddings → DiT (embedding-to-video)
            ↑ 在 MLLM 的 native representation space, 信息丰富
```

对比之前的工作:
- **UniVideo** [76](https://arxiv.org/abs/2510.08377), **VINO** [9](https://arxiv.org/abs/2601.02358): 用 MLLM 的 raw hidden states 当 condition. Hidden states 是 task-agnostic 的, 需要大量 co-training 来 align
- **MetaQuery** [54](https://arxiv.org/abs/2504.06256), **Bifrost-1** [42](https://arxiv.org/abs/2508.05954): 用 text tokens 或少量 learnable queries. 信息太窄, 高级 semantic 传不过去
- **Bernini**: 用 ViT embeddings. 这个 space MLLM 已经 pre-trained 得很好了, adaptation cost 最小

---

## Planner 怎么工作: Mask Modeling

### 问题: 怎么让 MLLM 输出 target 的 ViT embeddings?

不能用 standard autoregressive (left-to-right generate), 因为 visual content 是 bidirectional 的——一个 patch 的语义跟它周围所有 patch 都相关, 不能 left-to-right 生成.

借鉴 MAR [39](https://arxiv.org/abs/2406.11838) 和 MaskGIT [8](https://arxiv.org/abs/2203.04818) 的思路, 用 **mask modeling**:

### Training 时

```
Target image/video → ViT encoder → target ViT embeddings
                                        ↓
                        Random mask 一部分 (按 Beta 分布采样 mask ratio)
                                        ↓
Masked target embeddings + text + source visuals → MLLM
                                        ↓
                        MLLM 输出 hidden states
                                        ↓
              masked positions 的 hidden states → ViT Embedding Decoder
                                        ↓
              Flow matching in ViT embedding space → 预测被 mask 的 embeddings
```

MLLM 学的是: **给定 context (text + source visuals + 部分 target), infer 出被 mask 掉的 target semantic content**. 这就是 "semantic planning".

Mask ratio 用 Beta 分布采样, task-specific (Table 3):

$$r \sim \text{Beta}(\alpha, \beta)$$

- **r**: mask ratio, 即 target tokens 中被 mask 掉的比例
- **α, β**: Beta 分布的两个 shape parameter

| Task | α | β | E[r] ≈ | 直觉 |
|------|---|---|--------|------|
| T2I | 5.0 | 1.1 | 0.82 | text 信息少, mask 可以少一点 |
| T2V | 8.0 | 1.05 | 0.88 | video 比 image 复杂, mask 多一点 |
| I2I | 8.0 | 1.05 | 0.88 | image editing, source image 有信息 |
| I2V | 10.0 | 1.0 | 0.91 | source image 有信息, mask 更多 |
| V2V | 12.0 | 0.9 | 0.93 | source video 跟 target 高度相关, mask 要非常多 |
| IV2V | 12.0 | 0.9 | 0.93 | reference + source video, 信息最多, mask 最多 |

**为什么 V2V 要 mask 这么多?** 因为 V2V 的 source video 跟 target 太像了, 如果 mask 太少, MLLM 直接 copy visible target tokens 就行, 学不到真正的 planning. Mask 多了, MLLM 被迫从 source + text 去 infer target, 这才是 planning 该做的事.

### Inference 时

Target 全部 mask, 然后 iterative refinement:

```
Step 0: 全 mask → MLLM → decoder → 得到 coarse predictions
Step 1: 用 step 0 的 predictions + 仍然 mask 的 → MLLM → decoder → refine
...
Step K-1: 几乎全部 unmask → final target ViT embeddings
```

Mask ratio 按余弦 schedule 递减:

$$\text{mask\_ratio}(k, K) = \cos\left(\frac{\pi}{2} \cdot \frac{k+1}{K}\right)$$

- **k**: 当前 step (0 到 K-1)
- **K**: 总 steps (默认 25)
- 第 0 步: cos(π/50) ≈ 0.998 → 几乎全 mask
- 第 24 步: cos(π/2) = 0 → 全部 unmask

这是 coarse-to-fine: 早期确定全局 layout, 后期 refine 细节. 类似雕塑: 先捏大形, 再刻细节.

---

## Renderer 怎么工作: DiT + Hierarchical CFG

### 基本流程

```
MLLM planner 输出的 hidden states z
        ↓
Zero-init MLP (projection)
        ↓
Concat with T5 text features (保留 Wan2.2 原本的 text encoder)
        ↓
DiT (Wan2.2-A14B) ← 在 VAE latent space 做 flow matching
        ↓
+ source VAE features (editing 时, 保留 source 的 low-level detail)
        ↓
Final video (VAE latents → pixels)
```

Zero-init MLP 很关键: 初始时 output 全是 0, DiT 完全用 pre-trained 的 T5 features, 慢慢才"学会" attend to MLLM features. 这避免了 catastrophic forgetting.

### Hierarchical CFG — 最 elegant 的 inference 设计

标准 CFG 是: 每个条件独立 drop, 算 guidance. 但当有多个 condition (text, source video, source image, target semantic), standard CFG 有 redundancy——几个 condition 可能携带相同信息.

Bernini 用 **hierarchical incremental decomposition**:

$$\hat{\epsilon} = \epsilon_{\emptyset,\emptyset,\emptyset,\emptyset} + \omega_{\text{vid}} \Delta_{\text{vid}} + \omega_{\text{img}} \Delta_{\text{img}} + \omega_{\text{txt}} \Delta_{\text{txt}} + \omega_{\text{tgt}} \Delta_{\text{tgt}}$$

其中:

$$\Delta_{\text{vid}} = \epsilon_{\emptyset,\emptyset,\text{vid},\emptyset} - \epsilon_{\emptyset,\emptyset,\emptyset,\emptyset}$$

$$\Delta_{\text{img}} = \epsilon_{\emptyset,\emptyset,\text{vid},\text{img}} - \epsilon_{\emptyset,\emptyset,\text{vid},\emptyset}$$

$$\Delta_{\text{txt}} = \epsilon_{\text{txt},\emptyset,\text{vid},\text{img}} - \epsilon_{\emptyset,\emptyset,\text{vid},\text{img}}$$

$$\Delta_{\text{tgt}} = \epsilon_{\text{txt},\text{tgt},\text{vid},\text{img}} - \epsilon_{\text{txt},\emptyset,\text{vid},\text{img}}$$

变量解释:
- **ε_{a,b,c,d}**: DiT 的 noise prediction, 4 个 subscript 分别是 (text, target-semantic, source-video-VAE, source-image-VAE), ∅ 表示 dropped
- **Δ_vid**: 在"什么 condition 都没有"的基础上, 加 source video 带来的增量
- **Δ_img**: 在"已有 source video"基础上, 加 source image 带来的增量
- **Δ_txt**: 在"已有 source video + image"基础上, 加 text 带来的增量
- **Δ_tgt**: 在"已有所有其他 condition"基础上, 加 target semantic plan 带来的增量
- **ω_vid, ω_img, ω_txt, ω_tgt**: 各 condition 的 guidance scale

**Intuition**: 这是从 low-level 到 high-level 的 condition hierarchy:
1. vid (source pixels) → 最 low-level, temporal consistency + background preservation
2. img (source/reference pixels) → identity preservation + spatial detail
3. txt (text description) → semantic intent + editing instruction
4. tgt (target semantic plan) → 最高 level, 整体规划

每个 Δ 衡量的是 **marginal contribution**——"在已有所有其他 condition 的基础上, 这个 condition 还能提供多少额外信息". 这避免了 guidance 的 double-counting.

**Guidance scales** (Table 5):

| Task | ω_txt | ω_vid | ω_img | ω_tgt | Intuition |
|------|-------|-------|-------|-------|-----------|
| T2V | 4.0 | — | 1.0 | 1.0 | 没有 source video, text 是主要驱动 |
| S2V | 4.0 | 1.25 | 2.5 | 1.5 | reference image 要强 guidance preserve identity |
| V2V | 4.0 | 1.25 | 1.25 | 0.5 | target plan guidance 低, 因为 source 跟 target 太像, 过强 guidance 会 over-edit |
| RV2V | 4.0 | 1.25 | 3.0 | 1.5 | reference image guidance 最高, 因为 RV2V 核心是 preserve reference identity |

---

## SA-3D RoPE — 解决位置歧义

### 问题是什么

DiT 用 3D RoPE [64](https://arxiv.org/abs/2104.09864) encode 每个 token 的 (t, h, w) 坐标——时间、高度、宽度. Query-key 的 attention score 取决于它们的相对 (t, h, w) 距离.

但是 Bernini 把多个 visual inputs 串成一个 sequence:
- Reference image: 单帧, 所有 token 的 t=0, h∈[0,H), w∈[0,W)
- Source video frame 0: 也是 t=0, h∈[0,H), w∈[0,W)
- Target output frame 0: 也是 t=0, h∈[0,H), w∈[0,W)

Reference image 的 (0, 5, 10) 跟 target output 的 (0, 5, 10) 在 RoPE 看来是"同一位置", attention 分不清, 导致 reference 的 appearance 泄漏到 target 的非对应区域.

### 为什么 naive fix 不行

最 naive 的 fix: 给每个 segment 加一个 learnable segment embedding, 加到 hidden states 上.

但是 attention score 是 q·R·k (R 是 rotary matrix), additive bias 在 hidden states 上只是 shift 了整体 attention logits, 不能 selectively suppress "相同 (t,h,w) 但不同 segment" 的 attention.

Figure 15 的 ablation 显示: 加 segment embedding 能改善 reference consistency (比如 scarf 纹理更对), 但 reference leakage 依然存在 (background 泄漏, duck head 错位).

### SA-3D RoPE 的解法 (Eq.2)

$$\tilde{\mathbf{r}}_{t,h,w,i} = \mathbf{r}_{t,h,w} \odot \mathbf{r}_i^{\text{seg}}$$

- **r_{t,h,w}**: standard 3D RoPE 的 rotary frequency vector, encode spatiotemporal position
- **r_i^seg**: 一个 full-dimensional rotary frequency vector, encode segment index i. i=0 是 target, i=1,2,... 是 source segments
- **⊙**: element-wise 复数乘法

复数乘法的关键性质: **phase addition**. 如果把 RoPE 的每一维看成复数 e^{iθ}, 那 element-wise 复数乘法 e^{iθ_1} · e^{iθ_2} = e^{i(θ_1+θ_2)}, 即 phase 相加.

所以每个 token 的 total phase = spatiotemporal phase + segment phase.

**效果**:
1. **Segment 内部**: segment phase 是 constant, 不影响 relative position encoding, 标准 3D RoPE 的所有性质保留
2. **Segment 之间**: 不同 segment 有不同 segment phase, 两个 token 即使 (t,h,w) 相同, total phase 也不同, RoPE 的 query-key 内积大幅衰减 (复数内积对 phase 差敏感), attention 自然避开跨 segment 同位置的 confusion

**跟 additive embedding 的本质区别**:
- Additive embedding: 在 real domain 的 hidden states 上加 offset, 效果是 global attention logit shift
- SA-3D RoPE: 在 complex domain 的 rotary frequency 上做 phase modulation, 效果是 selectively suppress 特定 token pair 的 attention alignment

类比: additive embedding 像给每个人发一个胸牌, 但是大家还是能互相看到; SA-3D RoPE 像给每个 segment 调不同的无线电频率, 不同频率之间自然干扰小.

---

## 数据工程 — Bernini 的 hidden gem

Paper 的数据工程非常 elaborate, 对想要 build 类似系统的人极有参考价值.

### Video-pair Data (20M pairs)

从 general T2V corpora 里, 同一 raw video 内提取 video pairs, 用 X-CLIP [50](https://arxiv.org/abs/2207.07085) 算 similarity, 过滤:
- Similarity ∈ [0.65, 0.95]: 既不 near-duplicate 也不 unrelated
- Duration ∈ [2, 10] seconds
- Human : non-human = 1:1
- 每个 raw video 最多 100 pairs

Prompt 用 Qwen3-VL-235B-A22B-Instruct [4](https://arxiv.org/abs/2511.21631) 生成, coarse-to-fine: 先 coarse transition description, 再 refine 成 detailed prompt, 先 camera motion 再 foreground/background changes.

### Image-pair Data (30M pairs)

从 300k+ tutorial videos [52](https://arxiv.org/abs/1906.03327) 提取, 过滤 low-motion/blur frames, CLIP similarity ∈ [0.75, 0.95].

### Motion-aware Editing Data (最 clever 的部分)

问题: 当 edit 一个 object 后, 人的 motion 应该相应变化. 比如手里杯子换成更重的物体, 手臂姿态应该不同. 但现有 pipeline 不能合成这种 motion-aware editing.

Bernini 用 **dual-branch framework** (Eq.4):

$$
\begin{aligned}
\hat{\epsilon} &= \alpha \cdot \text{CFG}_{\text{I2V}}(\text{edited first frame}, \text{source video}) \\
&+ \beta \cdot \text{CFG}_{\text{V2V}}(\text{source video}, \text{edit prompt})
\end{aligned}
$$

约束: α + β = 1

- **I2V branch**: input 是 edited first frame + source video → 引入 motion adaptation (人能动起来)
- **V2V branch**: input 是 source video + edit prompt → preserve source motion consistency
- **α, β**: 控制 trade-off

两个 branch 的 CFG-corrected prediction 加权融合, 既有 motion adaptation 又有 source consistency.

### Propagation-based Data Boosting

初始 data 用 DifuEraser [40](https://arxiv.org/abs/2501.10018) 和 VACE [31](https://arxiv.org/abs/2503.06940) 生成, 质量差 (artifacts, shape constraint).

改进: 先训 base propagation model (input: source video + edited first frame + edit prompt → target edited video). 再用 strong image editing model 生成高质量 edited first frames. Propagate 到 video → 高质量 edited videos.

### Person R2V (特殊处理)

Image editors 在 face identity 上容易失败. Bernini 改用 **identity recurrence in long-form video**: 同一 video/episode 内, 用 face embedding 检索 same-identity 的 reference clip, cross-episode filtering 保证 visual diversity, 从 real footage crop 全身 reference image.

这保证了 identity preservation, 因为 reference 本来就是同一人. 这也是 Bernini 在 OpenS2V-Eval [91](https://arxiv.org/abs/2505.20292) 上 FaceSim 78.20 (next best Kling O3 57.20, +20 points) 的根本原因.

---

## 三阶段训练

| Stage | 干什么 | 关键点 |
|-------|--------|--------|
| I | 训 MLLM planner | 256p, 2 fps, 保留 20% understanding data 防 forgetting |
| II | 训 DiT renderer | 480p, 16 fps, pair data linear decay |
| III | Joint co-training | 480p, 16 fps, light training, 加 CoT data |

**关键 insight**: Stage III 是 "light" co-training, 只跑 relatively few steps. 目的只是 align 两个 component, 不破坏各自 pretrained capability. 这种"separate pretrain + light co-train"策略, 让 Bernini 同时享受 MLLM 的 understanding strength 和 DiT 的 synthesis strength.

Training loss (Eq.3):

$$\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{ntp}} + \lambda_{\text{visual}} \mathcal{L}_{\text{visual}} + \lambda_{\text{dit}} \mathcal{L}_{\text{dit}}$$

- **L_ntp**: next-token prediction on text, 保留 MLLM language capability. λ_text=0.2
- **L_visual**: flow matching in ViT embedding space, 训 ViT decoder. λ_visual=1
- **L_dit**: flow matching in VAE latent space, 训 DiT. λ_dit=1

L_ntp 的存在至关重要: 如果不 joint train 它, MLLM 会 catastrophic forget 它的 reasoning 能力, Bernini 就不能 generalize 到 OOD editing instructions.

---

## 结果怎么样

### Video Editing

**Bernini-Bench** (Table 6):
- V2V Overall: 3.49 (Wan2.7 3.30, Kling O3 3.05)
- **Video Consistency: 3.51 (Wan2.7 3.11)** — 这是 SA-3D RoPE + source VAE feature injection 的直接效果

**Bradley-Terry Leaderboard** (Figure 1):
- Bernini BT Score 1044, #2 (仅次于 HappyHorse-1.0 1080)
- Win rate 56.3%

**Public Benchmarks**:
- OpenVE-Bench [26](https://arxiv.org/abs/2512.07826): 4.04 overall (next best VINO 3.18, +27%)
- EditVerse [32](https://arxiv.org/abs/2509.20360): Editing Quality 8.02 (SOTA)
- FiVE [37](https://arxiv.org/abs/2503.13684): FiVE-Acc 78.16 (SOTA)

### Video Generation

**VBench T2V** (Table 11): Total 84.64, 几乎 retain Wan2.2-A14B 的 84.79
**OpenS2V** (Table 12): Total 62.94 (#1), FaceSim 78.20 (+20 points vs next best)

### Reasoning Ablation (Table 10)

| Method | OS |
|--------|-----|
| Baseline | 3.12 |
| + PE (Qwen2.5-VL-7B) | 3.20 |
| + Self-text CoT | 3.33 |
| + PE (GPT-5.4) | 3.49 |
| + PE (GPT-5.4) + Self-visual-text CoT | **3.52** |

**Insight**: Bernini 自己的 self-text CoT (3.33) 超过用 init model 当 prompt enhancer (3.20), 说明训练后 Bernini 发展出更强的 reasoning. 加 visual intermediate (self-visual-text) 最优, 证明 multimodal reasoning > pure text reasoning.

---

## 两个 Foundation Model 的选择

- **MLLM**: Qwen2.5-VL-7B [69](https://arxiv.org/abs/2502.13923) — 7B, 理解能力强
- **DiT**: Wan2.2-A14B [70](https://arxiv.org/abs/2503.20314) — 14B, video generation SOTA 之一

这俩都是公开 model, Bernini 是站在它们肩膀上 build 的. Paper 承认 Bernini 的上限受限于 foundation model 的强度, 复杂 editing 仍需要 GPT-5.4 当 rewriter, 说明 native reasoning 还不够.

---

## 工程细节亮点

1. **Memory**: 440K token sequences (4.4× improvement), per-GPU 72GB → 40GB
2. **Speed**: FlashAttention-4 [92](https://arxiv.org/abs/2603.05451), FlexAttention [17](https://arxiv.org/abs/2412.05496), RMSNorm kernel from QuACK [14](https://github.com/Dao-AILab/quack), ~46% speedup
3. **Parallelism**: FSDP + Ulysses SP [30](https://arxiv.org/abs/2309.14509), ~4.5× throughput
4. **Distillation**: CFG distillation [51](https://arxiv.org/abs/2203.01248) + ReFlow [47](https://arxiv.org/abs/2209.03003), 80 NFE → 4 NFE (20× speedup, minimal quality loss)
5. **Inference**: DeepSpeed Ulysses + async all-to-all, VAE context parallelism, >7.2× speedup

---

## Bernini 跟其他 unified model 怎么比

| Method | Philosophy | Generation Quality | Understanding | Architecture Complexity |
|--------|------------|-------------------|----------------|------------------------|
| Emu3 [73](https://arxiv.org/abs/2409.18869) | Single backbone, unified NTP | 受限于 discrete tokens | OK | Simple |
| Janus [79](https://arxiv.org/abs/2503.07229) | Decoupled encoders, shared backbone | OK | OK | Medium |
| Show-o [82](https://arxiv.org/abs/2408.12528) | AR text + discrete diffusion image | OK | OK | Complex |
| BAGEL [15](https://arxiv.org/abs/2505.14683) | MoTE, dual visual encoders | Good | Good | Complex |
| UniVideo [76](https://arxiv.org/abs/2510.08377) | MLLM hidden states → DiT | Good | Good | Medium |
| VINO [9](https://arxiv.org/abs/2601.02358) | MLLM hidden states → video DiT | Good | Good | Medium |
| **Bernini** | **MLLM ViT embeddings → DiT** | **Good** | **Good** | **Medium** |

Bernini 跟 UniVideo/VINO 最像, 区别在 interface representation. UniVideo/VINO 用 raw hidden states, 需要 heavy co-training align; Bernini 用 ViT embeddings, MLLM 已经 pre-trained 得很好, adaptation cost 最小.

---

## 最核心的三个 Takeaway

1. **Interface representation matters more than architecture**: 选对 bridge (ViT embedding space) 比设计复杂 architecture 更重要. 让 MLLM 在它最熟悉的 representation space 里做 planning, 就能最大程度 transfer pre-trained 的 understanding capability.

2. **Position encoding for multi-segment sequences is non-trivial**: 当 unified framework 把多个 visual inputs 串成一个 sequence, standard 3D RoPE 失效. SA-3D RoPE 通过 multiplicative phase modulation 解决, additive segment embedding 不够.

3. **Preserve understanding through data and objectives**: 3-stage training + L_ntp + 20% understanding data, 让 MLLM 在变成 planner 的过程中不丢失 reasoning capability. 这是 Bernini 能 generalize到 OOD editing (causal reasoning like "rain extinguishes fire") 的根本原因.

---

## 一句话总结

Bernini = MLLM planner (在 ViT embedding space 做 mask modeling) + DiT renderer (在 VAE space 做 flow matching) + SA-3D RoPE (解决 multi-segment 位置歧义) + 3-stage training (separate pretrain + light co-train, 保留各自 capability) + elaborate data pipeline (20M video pairs, 30M image pairs, motion-aware dual-branch synthesis, propagation boosting).

核心 philosophy: **与其 unified everything in one model, 不如 let each component do what it's best at, and design a smart interface between them.**

---

# Bernini: Latent Semantic Planning for Video Diffusion — 深度解析

## 一、核心哲学与定位

Bernini (Bytedance, 2026.05) 的核心 thesis 用一句话讲: **MLLM 做语义规划, DiT 做 pixel synthesis, 两者通过 MLLM 自身的 ViT embedding space 作为 interface 通信**. 这是一种 "modular decoupling" 哲学, 跟 Emu3 [73](https://arxiv.org/abs/2409.18869)、Janus [79](https://arxiv.org/abs/2503.07229)、Show-o [82](https://arxiv.org/abs/2408.12528)、BAGEL [15](https://arxiv.org/abs/2505.14683)、HunyuanImage 3.0 [7](https://arxiv.org/abs/2509.23951) 这些"single backbone, unified objective"路线形成鲜明对比.

| 路线 | 代表工作 | 特点 | 缺点 |
|------|---------|------|------|
| Unified next-token | Emu3 [73](https://arxiv.org/abs/2409.18869), OmniGen2 [80](https://arxiv.org/abs/2505.18404) | 一个 transformer, 一个 vocab, 一个 objective | generation 质量受限于 discrete tokenization; understanding 和 generation 互相干扰 |
| Decoupled encoders, shared backbone | Janus [79](https://arxiv.org/abs/2503.07229) | SigLIP for understanding, VQ for generation | 两个 visual pathway 需要 from-scratch alignment |
| Hybrid objectives in one backbone | Show-o [82](https://arxiv.org/abs/2408.12528), HunyuanImage 3.0 [7](https://arxiv.org/abs/2509.23951) | AR for text + diffusion for image | 架构复杂, MoE 调度开销 |
| MLLM as conditioner for diffusion | SEED-X [22](https://arxiv.org/abs/2404.14396), DreamLLM [18](https://arxiv.org/abs/2310.13885), Emu [65](https://arxiv.org/abs/2307.05222), UniVideo [76](https://arxiv.org/abs/2510.08377), VINO [9](https://arxiv.org/abs/2601.02358), MetaQuery [54](https://arxiv.org/abs/2504.06256), Bifrost-1 [42](https://arxiv.org/abs/2508.05954) | 两个成熟组件, 接口设计是关键 | 接口太窄 (text tokens) 信息损失; 接口太宽 (hidden states) 需要 heavy alignment |
| **Bernini (本文)** | — | **接口锚定在 MLLM 自己的 ViT embedding space** | 仍然依赖两个 foundation model 的强度 |

**关键 insight**: 为什么选 ViT embedding space 作为 bridge? 因为 MLLM 本身就是通过 ViT encoder [19](https://arxiv.org/abs/2010.11929) [58](https://arxiv.org/abs/2103.00020) [71](https://arxiv.org/abs/2502.14786) 来感知视觉内容的——它的 hidden states 里关于 visual 的部分, 本质上就是 ViT embeddings 经过 self-attention 融合后的 contextualized representation. 让 MLLM 去预测 target 的 ViT embeddings, 等价于让它在自己最熟悉的 "母语" 空间里做 planning, adaptation cost 最小. 这跟 SEED-X / UniVideo / VINO 用 raw hidden states 当 condition 不同——hidden states 是 task-agnostic 的, 没有跟 visual semantic space 对齐, 需要大量 co-training; 而 ViT embedding space 本身就是 visual semantic space.

项目主页: https://bernini-ai.github.io

---

## 二、Architecture Overview

整体 pipeline (Figure 3):

```
Inputs: text t, source visuals {v_i^src}, target visual v^tgt (masked)
                    ↓
   [MLLM Planner (Qwen2.5-VL-7B)]  ← Stage I 训练, 用 mask modeling
                    ↓
   contextualized hidden states z
                    ↓
   ┌──────────────────────────────────────────┐
   │ masked positions → [ViT Embedding Decoder]│ ← MLP + ResNet head, flow matching
   │   → predicted target ViT embeddings      │
   └──────────────────────────────────────────┘
                    ↓
   z (unmasked positions) + recovered target ViT embeddings
                    ↓
   [Zero-init MLP connector]
                    ↓
   concatenated with T5 text features
                    ↓
   [DiT Renderer (Wan2.2-A14B)] ← Stage II 训练, flow matching in VAE space
   + source VAE features (for editing)
                    ↓
   final video (VAE latents → pixels)
```

两个 component 用 pre-trained 的 foundation: MLLM 用 Qwen2.5-VL-7B [69](https://arxiv.org/abs/2502.13923), DiT 用 Wan2.2-A14B [70](https://arxiv.org/abs/2503.20314). 这种"站在巨人肩膀上"的选择, 让 Bernini 能直接 inherit 两个 foundation 的 capability, 同时通过 lightweight co-training 来 align.

---

## 三、MLLM-based Planner 深度解析

### 3.1 Unified Input Formulation (Eq.1)

$$
\mathbf{z} = \mathrm{MLLM}(\mathbf{t}, \mathbf{v}_1^{\mathrm{src}}, \mathbf{v}_2^{\mathrm{src}}, \ldots, \mathbf{v}_N^{\mathrm{src}}, \mathbf{v}^{\mathrm{tgt}})
$$

变量解释:
- **z**: MLLM 输出的 contextualized hidden states, 后续作为 renderer 的 conditioning
- **t**: input textual embeddings (tokenized text → embedding)
- **v_i^src**: 第 i 个 source visual input 的 ViT embeddings (通过 MLLM 内置的 ViT encoder 提取)
- **N**: source inputs 的数量 (reference image count, source video, etc.)
- **v^tgt**: target output 的 ViT embeddings. 训练时 **partially masked** (随机一部分替换成 shared mask token), 推理时 **fully masked** (所有位置都是 mask token)

关键点: 所有 task (T2V, S2V, V2V, RV2V) 都被 serialize 成同一个 1D token sequence 格式. 这种 unified formulation 避免了 task-specific architecture, 让模型可以跨 task 共享 representation.

### 3.2 Mask-based Semantic Planning

这部分借鉴了 MAR (Masked Autoregressive model, Li et al. NeurIPS 2024) [39](https://arxiv.org/abs/2406.11838) 的设计哲学.

**为什么用 mask modeling 而不是 autoregressive?** 因为 visual semantic latents 是 **intrinsically bidirectional** 的——一个 patch 的语义跟它周围所有 patch 都有关, 不是 left-to-right causal 的. 一个图像右上角的物体, 跟左下角的背景在 semantic 上是耦合的. 用 causal mask 会人为破坏这种 bidirectional dependency.

**Training procedure**:
1. 对 target ViT tokens, 按比例 r 随机 mask (替换成 shared learnable mask token)
2. r 从 Beta 分布采样: r ~ Beta(α, β), 其中 (α, β) 是 task-specific 超参 (见 Table 3)
3. MLLM 接收 (text + source visuals + masked target), 输出 hidden states
4. **masked positions 的 hidden states** → 喂给 ViT Embedding Decoder
5. Decoder 通过 flow matching 在 ViT embedding space 做 denoising, 预测 ground-truth ViT embeddings

**ViT Embedding Decoder 的结构**: 一个 MLP + ResNet-based prediction head. 借鉴 MAR [39](https://arxiv.org/abs/2406.11838) 的设计. 用 flow matching [43](https://arxiv.org/abs/2210.02747) 而不是 deterministic regression, 因为 ViT embeddings 本身有 distribution——同一个 semantic content 可以对应多个 valid ViT embedding (取决于 augmentation, viewpoint 等). Flow matching 建模这个 distribution, 比 L2 regression 更 faithful.

**Beta 分布的 task-dependent 配置** (Table 3):

| Parameter | T2I | T2V | I2I | I2V | V2V | IV2V |
|-----------|-----|-----|-----|-----|-----|------|
| α | 5.0 | 8.0 | 8.0 | 10.0 | 12.0 | 12.0 |
| β | 1.1 | 1.05 | 1.05 | 1.0 | 0.9 | 0.9 |

Intuition: task 的 input 越 informative (V2V 的 source video 跟 target 高度相关), α 越大 β 越小, mask ratio r 越接近 1.0. 这是为了 **prevent information leakage**——如果 V2V 不 mask 多一点, MLLM 可以直接 copy visible target tokens 的内容, 学不到真正的 planning. T2I 没有 source visual, mask ratio 可以低一些, MLLM 主要靠 text 信号来 plan.

Beta(5.0, 1.1) 的 mean ≈ 5.0/6.1 ≈ 0.82, Beta(12.0, 0.9) 的 mean ≈ 12.0/12.9 ≈ 0.93. 所以 V2V 训练时平均 93% 的 target tokens 被 mask 掉, 模型几乎完全靠 source + text 来 infer.

**Inference procedure (iterative refinement)**:

K 步 iterative decoding, 每步的 mask ratio 按余弦 schedule 递减:

$$
\text{mask\_ratio}(k, K) = \cos\left(\frac{\pi}{2} \cdot \frac{k+1}{K}\right)
$$

- **k**: 当前 refinement step (0-indexed, 从 0 到 K-1)
- **K**: 总 refinement steps (默认 25)
- 第 0 步: mask_ratio = cos(π/2 · 1/25) ≈ cos(3.6°) ≈ 0.998 → 几乎全 mask
- 第 24 步: mask_ratio = cos(π/2 · 25/25) = cos(90°) = 0 → 全部 unmask

每一步:
1. 当前已 predicted 的 tokens + 仍然 masked 的 tokens → 重新输入 MLLM
2. MLLM 输出 masked positions 的 hidden states
3. ViT Embedding Decoder 做 5 步 flow matching denoising → 得到 predicted ViT embeddings
4. 这些 predicted embeddings 作为"已确定"的 tokens 进入下一步

这是 coarse-to-fine 的过程: 早期 step 确定全局 semantic layout, 后期 step refine 细节. 类似 MaskGIT [8](https://arxiv.org/abs/2203.04818) 的 iterative decoding, 但在 continuous embedding space 而非 discrete token space.

**Inference cost 分析**: 25 planning steps × 5 denoising steps = 125 步在 MLLM+decoder 里, 但 paper 声称开销 "negligible relative to DiT renderer". 这是因为 MLLM 的 forward 比 DiT 便宜很多 (7B vs 14B, 而且 sequence 短), 加上 planner 只处理 semantic tokens (低分辨率), 而 DiT 要处理 full VAE latents (高分辨率 spatiotemporal).

### 3.3 Chain-of-Thought (CoT) Reasoning

两类 CoT 增强 planner 的 reasoning capability:

**Self-text Reasoning**: 用 1M 条 CoT 数据, 让 MLLM 把原始 editing instruction rewrite 成更 detailed, structured, semantically enriched 的 instruction. 这是 pure language-space reasoning.

**Self-vision-text Reasoning**: 两阶段
1. **Image-level reasoning**: 先对 source video 的 first frame 做 image editing, guided by textual reasoning, 得到 edited first frame (visual intermediate state)
2. **Video-level generation**: 基于这个 visual intermediate, propagate 到完整 video

这个设计很 clever: 它把 "video editing" 分解成 "image reasoning (spatial)" + "video generation (temporal)", 用 edited first frame 作为 visual grounding, 让 temporal generation 有 explicit 的 spatial anchor. 类似 Visual Sketchpad [28](https://arxiv.org/abs/2406.04823) 的思想——用 visual intermediate 来 ground reasoning.

Table 10 的 ablation 证明: baseline (3.12 OS) → +PE Qwen2.5-VL-7B (3.20) → +self-text (3.33) → +PE GPT-5.4 (3.49) → +self-vision-text (3.52, best). 显示 multimodal reasoning 优于 pure text reasoning.

---

## 四、SA-3D RoPE — 核心架构创新

这是本文最 novel 的 architectural contribution, 需要详细讲.

### 4.1 问题: 多 segment 的位置歧义

3D RoPE [64](https://arxiv.org/abs/2104.09864) 在 DiT 里是 standard practice: 把每个 visual token 的 (t, h, w) 三个坐标 encode 成三个 rotary subspaces, 拼接成 full-dimensional rotary frequency vector **r_{t,h,w}**. Attention 的 query-key 内积只对相同 (t, h, w) 的 token 最大.

但当 Bernini 把多个 visual inputs (reference image 1, reference image 2, source video, target output) 串成一个 unified 1D sequence 时, **不同 segment 的 token 可能有相同的 (t, h, w) 坐标**. 比如:
- Reference image (single frame): 所有 token 的 t=0, h∈[0, H), w∈[0, W)
- Source video frame 0: 也是 t=0, h∈[0, H), w∈[0, W)
- Target output frame 0: 也是 t=0, h∈[0, H), w∈[0, W)

这样 reference image 的 (0, 5, 10) 和 target output 的 (0, 5, 10) 在 RoPE 看来是"同一位置", attention 无法区分它们, 导致 content confusion (reference 的 appearance 泄漏到 target 的非对应区域).

### 4.2 Baseline 尝试: Learnable Segment Embedding

最 naive 的 fix: 给每个 segment 加一个 learnable segment embedding, 加到 hidden states 上. 这本质是 additive bias.

Figure 15 的 ablation 显示: 这能改善 reference consistency (比如 scarf 的纹理更对齐), 但仍然有 **reference leakage artifacts** (比如 reference 的背景泄漏到 target, 或者 duck head 出现在不该出现的位置).

为什么 additive embedding 不够? 因为 segment embedding 只在 hidden states 上加一个 constant offset, 而 RoPE 的 query-key 内积是 q·R·k (R 是 rotary matrix). Additive bias 在 attention score 上的效果是 global 的, 不区分 (t, h, w); 而 content confusion 的问题恰恰发生在相同 (t, h, w) 的 token 之间, additive bias 不能 selectively suppress 这些 specific pairs.

### 4.3 SA-3D RoPE 的设计 (Eq.2)

$$
\tilde{\mathbf{r}}_{t, h, w, i} = \mathbf{r}_{t, h, w} \odot \mathbf{r}_i^{\mathrm{seg}}
$$

变量解释:
- **r_{t,h,w}**: standard 3D RoPE 的 full-dimensional rotary frequency vector, encoding (t, h, w)
- **r_i^seg**: 一个 full-dimensional rotary frequency vector, encoding segment index i. i=0 表示 target segment, i=1,2,... 表示 input segments
- **⊙**: element-wise 复数乘法. 因为 RoPE 的每一维本质是一个 2D rotation (复数), element-wise 复数乘法等价于 **phase addition**: arg(a·b) = arg(a) + arg(b)

所以 SA-3D RoPE 的效果: 每个 token 的 rotary phase = spatiotemporal phase + segment phase.

**关键性质**:
1. **Segment disambiguation**: 两个 token 如果 (t, h, w) 相同但 segment i 不同, 它们的 total phase 不同, RoPE 的 query-key 内积会大幅衰减 (因为复数内积对 phase 差敏感), attention 自然避开"跨 segment 同位置"的 confusion
2. **Spatiotemporal structure preserved**: 在同一 segment 内, phase 的 spatiotemporal 部分不变, 标准 3D RoPE 的所有 properties (相对位置 encoding, long-range decay) 都保留
3. **Global phase modulation**: segment phase 是 global 的 (对 segment 内所有 token 加同样的 phase), 所以它 modulate 的是"这个 segment 整体的 identity", 不破坏内部的相对位置结构

直觉: 可以把 r_i^seg 理解成给每个 segment 分配一个独特的"global phase signature". 两个 segment 的 signature 不同, 它们之间的 cross-attention 自然被抑制; 而 segment 内部, signature 是 constant, 不影响 relative position 的 encoding.

这跟 standard RoPE 加 segment embedding 的本质区别: **RoPE 是 multiplicative in complex domain (phase addition), embedding 是 additive in real domain (value shift)**. Multiplicative phase modulation 在 attention 内积的层面有更强的 disambiguation power, 因为它直接 manipulate 了 query-key 的 alignment, 而 additive bias 只能 shift 整体的 attention logits.

### 4.4 与 MLLM 端的 Segment-wise Hybrid Attention

Figure 3D 提到 MLLM 端也用了 segment-wise hybrid attention mask. 在 MLLM 内部, 不同 segment (text, source visuals, target) 之间的 attention pattern 是定制化的——text 可以 attend to all, source visuals 可以互相 attend, target tokens 的 attention 遵循 mask modeling 的规则 (visible target tokens 可以 attend to context, masked target tokens 需要从 context infer). 这跟 DiT 端的 SA-3D RoPE 是互补的: MLLM 端用 attention mask 来 structure 信息流, DiT 端用 RoPE phase 来 disambiguate position.

---

## 五、DiT-based Renderer + Hierarchical CFG

### 5.1 Renderer 架构

基于 Wan2.2-A14B [70](https://arxiv.org/abs/2503.20314), 在 VAE latent space 做 flow matching denoising.

**Conditioning inputs**:
1. **MLLM hidden states z** (经过 zero-init 1-layer MLP projection) — 提供 high-level semantic guidance
2. **T5 text features** (保留 Wan2.2 原本的 text encoder) — 提供 detailed text description
3. **Source VAE features** (for editing tasks) — 提供 low-level detail preservation

Zero-init MLP 的设计 [He et al.]: 初始时 MLP output 全是 0, 所以训练初期 DiT 完全使用 pre-trained 的 T5 features, 慢慢才"学会" attend to MLLM features. 这避免了 catastrophic forgetting, 让 DiT 在添加新 condition 时不会破坏原有 capability.

这种"concat T5 + MLLM features"的设计很务实: 保留 Wan2.2 pre-trained 的 text-conditioning prior (T5 features 已经训得很好), 同时叠加 MLLM 的更高层 semantic guidance. 一个 lightweight 改动就 extend 了 condition 信息量.

### 5.2 Hierarchical CFG Decomposition (Eq. 8-12)

这是 inference 阶段最 elegant 的设计. 把 final prediction 分解成 unconditional base + 4 个 incremental guidance terms:

$$
\Delta_{\mathrm{vid}} = \epsilon_{\emptyset, \emptyset, \mathrm{vid}, \emptyset} - \epsilon_{\emptyset, \emptyset, \emptyset, \emptyset} \quad \text{(Eq.8)}
$$

$$
\Delta_{\mathrm{img}} = \epsilon_{\emptyset, \emptyset, \mathrm{vid}, \mathrm{img}} - \epsilon_{\emptyset, \emptyset, \mathrm{vid}, \emptyset} \quad \text{(Eq.9)}
$$

$$
\Delta_{\mathrm{txt}} = \epsilon_{\mathrm{txt}, \emptyset, \mathrm{vid}, \mathrm{img}} - \epsilon_{\emptyset, \emptyset, \mathrm{vid}, \mathrm{img}} \quad \text{(Eq.10)}
$$

$$
\Delta_{\mathrm{tgt}} = \epsilon_{\mathrm{txt}, \mathrm{tgt}, \mathrm{vid}, \mathrm{img}} - \epsilon_{\mathrm{txt}, \emptyset, \mathrm{vid}, \mathrm{img}} \quad \text{(Eq.11)}
$$

$$
\hat{\epsilon} = \epsilon_{\emptyset, \emptyset, \emptyset, \emptyset} + \omega_{\mathrm{vid}} \Delta_{\mathrm{vid}} + \omega_{\mathrm{img}} \Delta_{\mathrm{img}} + \omega_{\mathrm{txt}} \Delta_{\mathrm{txt}} + \omega_{\mathrm{tgt}} \Delta_{\mathrm{tgt}} \quad \text{(Eq.12)}
$$

变量解释:
- **ε_{a, b, c, d}**: DiT 的 noise prediction, subscript (a, b, c, d) 分别表示 (text, target-semantic, source-video-VAE, source-image-VAE) 这 4 个 condition 的有 (∅ 表示 dropped)
- **Δ_vid**: 加入 source video VAE features 带来的 incremental contribution
- **Δ_img**: 在已有 source video 基础上, 再加 source image VAE features 的 incremental contribution
- **Δ_txt**: 在已有 source video + image 基础上, 再加 text features 的 incremental contribution
- **Δ_tgt**: 在已有所有 condition 基础上, 再加 target semantic embeddings 的 incremental contribution
- **ω_vid, ω_img, ω_txt, ω_tgt**: 各 condition 的 guidance scale (Table 5)

**为什么这个 hierarchical order 重要?** 它反映了一个"从低级到高级"的 condition hierarchy:
- vid (source video pixels) → 最 low-level, 提供 temporal consistency 和 background preservation
- img (source/reference image pixels) → 提供 identity preservation 和 spatial detail
- txt (text description) → 提供 semantic intent 和 editing instruction
- tgt (target semantic plan from MLLM) → 最高 level, 提供整体规划

这种 incremental decomposition确保每个 condition 的 guidance 都是 "marginal"——它衡量的是"在已有所有其他 condition 的基础上, 这个 condition 还能提供多少额外信息". 这比 standard CFG (每个 condition 独立 drop) 更精细, 避免 guidance 的 redundancy.

**Guidance scales** (Table 5):

| Task | Steps | ω_txt | ω_vid | ω_img | ω_tgt |
|------|-------|-------|-------|-------|-------|
| T2V | 60 | 4.0 | — | 1.0 | 1.0 |
| S2V | 40 | 4.0 | 1.25 | 2.5 | 1.5 |
| V2V | 40 | 4.0 | 1.25 | 1.25 | 0.5 |
| RV2V | 40 | 4.0 | 1.25 | 3.0 | 1.5 |

观察:
- T2V: 没有 source video, ω_vid 不用. text guidance (4.0) 远大于 target semantic (1.0), 因为 T2V 主要靠 text 驱动
- S2V: ω_img=2.5 (高), 因为要 preserve subject identity
- V2V: ω_tgt=0.5 (低), 因为 V2V 的 target 跟 source 高度相关, MLLM 的 semantic plan 不需要太强的 guidance, 否则容易 over-edit
- RV2V: ω_img=3.0 (最高), 因为 reference image 是核心 condition, 必须强 guidance 来 preserve reference identity

另外用 adaptive projected guidance (APG) [62](https://arxiv.org/abs/2410.02431) 来 reduce oversaturation——high guidance scale 容易导致颜色过饱和, APG 通过 projection 来 mitigate 这个问题.

---

## 六、Training Objectives (Eq.3)

$$
\mathcal{L} = \lambda_{\mathrm{text}} \mathcal{L}_{\mathrm{ntp}} + \lambda_{\mathrm{visual}} \mathcal{L}_{\mathrm{visual}} + \lambda_{\mathrm{dit}} \mathcal{L}_{\mathrm{dit}}
$$

- **L_ntp**: next-token prediction loss on text tokens, 保留 MLLM 的 language understanding capability. λ_text=0.2 (Stage I & III)
- **L_visual**: flow matching loss in ViT embedding space, 训练 ViT Embedding Decoder. λ_visual=1
- **L_dit**: flow matching loss in VAE latent space, 训练 DiT renderer. λ_dit=1

**关键: L_ntp 的存在是为了 preserve MLLM 的 understanding capability**. 如果不 joint train L_ntp, MLLM 在 training 过程中会 catastrophic forget 它的 language/reasoning 能力, 导致 planner 退化成 pure visual predictor. 这种"理解能力 retention"是 Bernini 能 generalize到 OOD editing instructions 的根本原因.

### 6.1 Timestep Sampling (Eq.6, 7)

两种 noise schedule, task-specific (Table 4):

**Logit-normal** (for image tasks, Eq.6):
$$
\pi_{\ln}(t; m, s) = \frac{1}{s\sqrt{2\pi}} \cdot \frac{1}{t(1-t)} \exp\left(-\frac{(\mathrm{logit}(t) - m)^2}{2s^2}\right)
$$
- **t**: timestep ∈ (0, 1)
- **m**: mean in logit space, logit(t) = log(t/(1-t))
- **s**: std in logit space
- 用 Lognorm(0.5, 1) 即 m=0.5, s=1, 倾向于中间 timesteps

**Mode** (for video tasks, Eq.7):
$$
f_{\mathrm{mode}}(u; s) = 1 - u - s \cdot \left(\cos^2\left(\frac{\pi}{2}u\right) - 1 + u\right)
$$
- **u**: uniform random ∈ [0, 1]
- **s**: shift parameter
- 用 Mode(1.29) for video, shift=3.0/5.0 for T2V/V2V

**Shift parameter** 的作用: SD3 [20](https://arxiv.org/abs/2403.03206) 和 Waver [94](https://arxiv.org/abs/2508.15761) 发现, video generation 需要更大的 shift 来 emphasize high-noise timesteps (因为 video 的 noise distribution 跟 image 不同, temporal dimension 引入额外 variance). Table 4 显示 image tasks shift=3.0-4.0, video tasks shift=5.0, 体现这个 principle.

---

## 七、Three-Stage Training Pipeline (Table 2)

| Stage | Optimized | Res. | LR | EMA | T2I | T2V | I2I | V2V | I2V | IV2V | V.P. | I.P. | Int. | Und. | CoT |
|-------|-----------|------|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|-----|
| I | MLLM | 256p | 1e-5 | 0.999 | 13% | 19% | 3% | 1% | 1% | 1% | 15% | 21% | 6% | 20% | — |
| II | DiT | 480p | 1e-5 | 0.9995 | 31% | 42% | 4% | 0.4% | 0.4% | 0.3% | 11% | 11% | — | — | — |
| II' | DiT | 480p | 1e-5 | 0.9999 | 20% | 30% | 40% | 3.3% | 3.5% | 3.2% | — | — | — | — | — |
| III | All | 480p | 1e-5 | 0.9995 | 16% | 24% | 32% | 2.6% | 2.8% | 2.6% | — | — | — | 20% | — |
| III' | All | 480p | 1e-5 | 0.999 | 12% | 18% | 24% | 2% | 2% | 2% | — | — | — | 20% | 20% |

(V.P. = Video Pairs, I.P. = Image Pairs, Int. = Interleaved image-text, Und. = Understanding, CoT = Chain-of-Thought)

### Stage I: MLLM Planner Pretraining
- **目标**: 把 MLLM 从 pure understanding model 变成 semantic planner
- **Loss**: λ_text L_ntp + λ_visual L_visual, λ_text=0.2, λ_visual=1
- **Data curriculum**: 先 T2I (建立 image generation in semantic space), 再扩展到 T2V/I2I/V2V
- **保留 understanding**: 20% 数据是 multimodal understanding data, 防止 catastrophic forgetting
- **Resolution**: 256p, 2 fps (低分辨率, 因为 semantic planning 不需要 pixel-level detail)

### Stage II: DiT Renderer Pretraining
- **目标**: 给 DiT strong generation + editing capability, 在跟 MLLM 耦合之前
- **Loss**: L_dit, conditioned on text + source VAE features
- **Data**: T2I/T2V/editing/pair data mixture
- **Pair data linear decay**: 训练初期 pair data 占比高 (帮助 generalization 和 editing quality), 后期衰减到 0 (用 high-quality editing data refine)
- **Resolution**: 480p, 16 fps

Stage II 分两 phase: 第一 phase 重 generation (T2I 31%, T2V 42%), 第二 phase 重 editing (I2I 40%).

### Stage III: Joint Co-Training
- **目标**: align planner 的 semantic prediction 和 renderer 的 synthesis process
- **Loss**: Eq.3, λ_ntp=0.2, λ_visual=λ_dit=1
- **Mechanism**: text + source ViT tokens + masked target ViT tokens → MLLM → (1) unmasked positions 的 hidden states 作为 DiT condition, (2) masked positions 的 hidden states → ViT decoder → L_visual
- **CoT data**: 在 Stage III 后期加入 (III' phase), 20% 数据是 CoT, 强化 structured reasoning

**关键 insight**: Stage III 是 "light" co-training——只跑 relatively few steps, 目的只是 align 两个 component, 不破坏各自 pretrained 的 capability. 这种"separate pretrain + light co-train"策略, 让 Bernini 同时享受了 MLLM 的 understanding strength 和 DiT 的 synthesis strength, 避免了 joint training from scratch 的 interference 问题.

---

## 八、Data Pipeline (Section 3)

Bernini 的数据工程非常 elaborate, 是这篇 paper 的另一个核心贡献.

### 8.1 Pre-training Data

**Video-pair Data** (20M pairs):
- Source: general T2V corpora
- 同一 raw video 内提取 video pairs
- 过滤条件:
  - X-CLIP [50](https://arxiv.org/abs/2207.07085) similarity ∈ [0.65, 0.95] (既不 near-duplicate 也不 unrelated)
  - Duration ∈ [2, 10] seconds
  - Human-centric : non-human-centric = 1:1 (用 Qwen3-VL-30B-A3B-Instruct [4](https://arxiv.org/abs/2511.21631) 标注)
  - 每个 raw video 最多 100 pairs (避免 over-representation)
- Prompt 生成: Qwen3-VL-235B-A22B-Instruct, coarse-to-fine strategy (先 coarse transition, 再 refine 成 detailed prompt)
- Prompt 结构: 先 camera motion, 再 foreground changes, 再 background changes

**Image-pair Data** (30M pairs):
- Source: 300k+ tutorial videos [52](https://arxiv.org/abs/1806.03535)
- 过滤: inter-frame transformation (去除 low-motion/scaling-dominated frames), blur detection
- CLIP [58](https://arxiv.org/abs/2103.00020) similarity ∈ [0.75, 0.95]
- Prompt: Qwen3-VL-30B 描述 image pair 之间的 visual differences

**Interleaved Image-text Data** (12M):
- 10M from OmniCorpus [38](https://arxiv.org/abs/2406.08418), 用 Qwen3-32B [84](https://arxiv.org/abs/2505.09388) regenerate text + subject-aware QA augmentation
- 2M from video-derived: 每个 video 提取 ≤8 keyframes, Qwen3-VL-30B 生成 frame-to-frame textual transitions

### 8.2 Image Editing + I2V Editing Data

核心 insight: **image editing 技术 比 video editing 成熟很多, 所以 reformulate 部分 video editing 成 image-to-video editing, transfer image-level capability 到 video**.

两种 prompt 构造机制:
1. **Candidate sampling + rewrite**: 从 user instruction pool 采样多个候选, MLLM 选最合适的并 rewrite
2. **Dynamic prompt bank**: 维护一个 prompt bank, conditioned on source image + current bank, MLLM 生成 high-novelty instruction, 高 novelty 的插入 bank, 低 diversity 的淘汰 (capacity-limited)

产出两种 data:
- **Image editing triplets**: (Source Image, Edited Image, Edit Prompt)
- **I2V triplets**: (Source Image, Video, Edit Prompt + Motion Prompt), 其中 motion prompt 由 MLLM 生成

### 8.3 High-quality V2V Editing Data

**Propagation-based Data Boosting**:
- 初始 data 用 DifuEraser [40](https://arxiv.org/abs/2501.10018) (addition/removal) 和 VACE [31](https://arxiv.org/abs/2503.06940) (replacement), 但质量差 (artifacts, shape constraint)
- 改进: 先训 base propagation model (input: source video + edited first frame + edit prompt → target edited video)
- 再用 strong image editing model 生成高质量 edited first frames
- Propagate 到 video → 高质量 edited videos
- Swap source/target 再生成 matching prompts, 扩增 data

**Human Motion-aware Data** (Eq.4):

这是最 clever 的 data construction. 解决一个问题: 当 object 被 edit 后, 人的 motion 应该相应变化 (比如手里的杯子被换成更重的物体, 手臂姿态应该不同), 但现有 pipeline 不能合成这种 motion-aware editing.

**Dual-branch framework**:
- **I2V branch**: input = edited first frame + source video → 引入 motion adaptation (人可以动起来)
- **V2V branch**: input = source video + edit prompt → preserve source motion consistency
- 两个 branch 的 noise prediction 用 weighted guidance 融合 (Eq.4)

Eq.4 展开:
$$
\begin{aligned}
\hat{\epsilon} &= \alpha\left(w_{Full}^{I2V} \cdot \epsilon(T_{I2V}, \mathcal{O}, I) - w_T^{I2V} \cdot \epsilon(\mathcal{O}, \mathcal{O}, I) - w_I^{I2V} \cdot \epsilon(T_{I2V}, \mathcal{O}, \mathcal{O})\right) \\
&+ \beta\left(w_{Full}^{V2V} \cdot \epsilon(T_{V2V}, V, \mathcal{O}) - w_T^{V2V} \cdot \epsilon(\mathcal{O}, V, \mathcal{O}) - w_V^{V2V} \cdot \epsilon(T_{V2V}, \mathcal{O}, \mathcal{O})\right)
\end{aligned}
$$

约束:
$$
w_{Full}^{I2V} - w_T^{I2V} - w_I^{I2V} = 1, \quad w_{Full}^{V2V} - w_T^{V2V} - w_V^{V2V} = 1, \quad \alpha + \beta = 1
$$

变量解释:
- **V**: source video
- **I**: edited first frame
- **T_{I2V}, T_{V2V}**: 两个 branch 用的 prompt
- **ε(T, V, I)**: noise prediction conditioned on (text, source video, image), ∅ 表示 dropped
- **w_{Full}, w_T, w_I**: CFG weights for full / text-dropped / image-dropped conditions
- **α, β**: 两个 branch 的融合权重, α+β=1

这本质是一个 **two-branch CFG**, 每个 branch 内部做标准 CFG (full - text_dropped - image_dropped), 两个 branch 的 CFG-corrected prediction 用 α, β 加权融合. I2V branch 强调 action adaptation (因为 input 是 image, 模型必须 infer motion), V2V branch 强调 source motion consistency (因为 input 是 video, 模型 preserve 原 motion). 通过 α, β 控制 trade-off.

### 8.4 Reference-image-guided Generation (R2V) & RV2V

**General-object R2V**:
- 每个 source video 采样 keyframes
- MLLM 识别 3-5 个 salient objects, 为每个写 editing instruction (extract + re-place 到 different scene)
- 用 image editor 生成 reference image
- MLLM 生成 R2V caption from (reference, keyframe) pair

**Person R2V** (special handling):
- Image editors 在 face identity 上容易失败
- 改用 **identity recurrence in long-form video**: 同一 video/episode 内, 用 face embedding 检索 same-identity 的 reference clip
- Cross-episode filtering 保证 visual diversity
- 从 real footage crop 全身 reference image
- 这样保证 identity preservation (因为 reference 本来就是同一人)

**RV2V (Reference+Video-to-Video)**:
- 需要 triplets (input video, reference, target video), 其中 input video 不含 referenced object
- 这种 triplets 自然界罕见
- 用 trained intermediate video editor 合成: 对每个 R2V sample, MLLM 写一个"remove/replace referenced object"的 instruction, editor 应用到 target video → input video
- 原始 target + reference + 合成的 input = 完整 triplet

### 8.5 Motion Transfer Data

- 任务: 用 reference video 里人的 motion, 来 animate image 里的人
- Triplet: ⟨reference video, image, target video⟩
- 构造: 先从 real video 提取 DWPose [85](https://arxiv.org/abs/2306.06028), 再用 Bernini 的 pose-to-video capability 生成 same-motion reference video, reference image 是从 target video 随机采样的 frame
- 这样构造的 triplet 是 self-consistent 的

### 8.6 Reasoning-augmented Data

- **Self-text**: 1M 样本, MLLM 把原始 prompt rewrite 成 structured intermediate prompt
- **Self-vision-text**: 两阶段, image-level reasoning 先, video-level generation 后, edited first frame 作为 visual intermediate

---

## 九、Infrastructure (Section 5)

这部分对 industrial-scale training 很有参考价值.

### 9.1 Memory Optimization

**Challenge**: Video editing 训练 sequence 极长, GPU memory 压力巨大.

**Solutions**:
1. **FSDP tuning**: per-GPU memory 72GB → 40GB
2. **Input pipeline restructuring**: 不再"先 concat 所有 tokens 再 scatter", 而是 "directly index-scatter into pre-allocated buffers", 节省 17GB intermediate memory
3. **Custom activation offloading**: pinned CPU memory pool + delayed-queue prefetch, overlap D2H/H2D with compute
4. **Sequence length**: 从 100K tokens → 440K tokens (4.4× improvement)

### 9.2 Kernel-level Optimization

- **FlashAttention-4** [92](https://arxiv.org/abs/2603.05451) in DiT — 注意这是 FA4 (future paper, 假设的), 比 FA3 更快
- **FlexAttention** [17](https://arxiv.org/abs/2412.05496) in MLLM — PyTorch 原生的 attention kernel, 支持 flexible masking
- **Asynchronous QKV communication** — overlap QKV projection 的 communication
- **TND memory layout preservation** — 避免 costly transposes (TND = Time-Sequence-Head layout, 视频优化)
- **cu_seqlens on CPU** — 减少 device memory 压力
- **RMSNorm kernel from QuACK** [14](https://github.com/Dao-AILab/quack) — 5-10% end-to-end speedup

总计 ~46% speedup from kernel-level.

### 9.3 Parallelism

- **FSDP** for weight sharding
- **Ulysses-style sequence parallelism** [30](https://arxiv.org/abs/2309.14509) for both DiT and MLLM
- DiT: shard tokens across GPUs along sequence and head dims, 支持 long video sequences
- MLLM: extended Ulysses SP, 2× throughput at SP degree 4
- SP 只在 long-sequence tasks 启用, 避免 short input 的 communication overhead

### 9.4 Sequence Packing + Batching

- **Sort by sequence length** per SP group: 2× throughput
- **Batch forward** for MLLM (FlashAttention variable-length kernels) and diffusion (concat + joint process)
- **Dummy-forward padding**: 避免 cross-rank deadlock (varying local batch sizes)
- **Token-bucket batching**: group samples into length buckets, per-bucket loss re-weighting, eliminate padding waste
- **Load-balanced dataloader**: greedy bin-packing, max/min workload ratio < 1.01, ~15% throughput improvement

总计 ~4.5× end-to-end throughput improvement.

### 9.5 Inference Parallelism

- DiT: DeepSpeed Ulysses + async all-to-all for QKV
- VAE: context parallelism along temporal dim + async conv cache
- Speedup: >7.2×

### 9.6 Model Distillation

两阶段:
1. **CFG distillation** [51](https://arxiv.org/abs/2203.01248): student 直接预测 CFG-combined output, 单次 forward, 省掉 conditional + unconditional 双重 evaluation, per-step compute 减半
2. **ReFlow** [47](https://arxiv.org/abs/2209.03003): straighten probability flow ODE trajectories, 更少 integration steps 就能 accurate generation

最终: distilled student with **4 NFEs** ≈ teacher with 80 NFEs. 这是 20× speedup with minimal quality degradation.

---

## 十、Bernini-Bench (Section 6.2)

新 benchmark, 因为现有 benchmark (OpenVE-Bench [26](https://arxiv.org/abs/2512.07826), EditVerse [32](https://arxiv.org/abs/2509.20360)) 局限:
- 只覆盖 V2V, 忽略 RV2V
- Editing type 多样性不足
- Video content variety 有限

**Bernini-Bench 设计**:
- 300 test cases
- 22 editing categories, 跨 5 dimensions:
  - Subject Editing
  - Scene & Environment
  - Visual & Style
  - Camera & Motion
  - Reasoning (包括 causal reasoning, 比如 "prolonged heavy rain" → infer "fire 应该灭了")
- 两个 setting: V2V + RV2V (8 个 category 在 RV2V 也 evaluate)
- 每个 category 10 cases
- Rich editing instructions (e.g. style transfer 的 wide range of target styles)
- Source videos 来自 free stock media platforms
- 覆盖 diverse attributes: human composition, shot scale, scene environment, camera motion, visual complexity, horizontal + vertical aspect ratios

**Evaluation metrics** (5 dimensions):
- **Instruction Following (IF)**: 是否忠实执行 textual instruction
- **Video Consistency (VC)**: 非编辑区域是否保持一致
- **Reference Image Consistency (IC)**: 编辑结果 vs reference image 的视觉特征一致性 (RV2V only)
- **Generation Quality (GQ)**: 物理真实感, 自然度, AI artifacts
- **Overall Score (OS)**: 整体满意度

两种 evaluation:
- MLLM-based scoring (1-5 scale, 用 GPT-5.4)
- Human Side-by-Side (SBS) comparison

---

## 十一、Experiments (Section 6)

### 11.1 Implementation

- MLLM: Qwen2.5-VL-7B [69](https://arxiv.org/abs/2502.13923)
- DiT: Wan2.2-A14B [70](https://arxiv.org/abs/2503.20314)
- T5 features retained for DiT text conditioning
- MLLM penultimate-layer hidden states → zero-init 1-layer MLP → concat with T5 features
- Inference 用 additional MLLM 增强 user instruction (prompt rewriting)

### 11.2 Video Editing Results

**Bernini-Bench** (Table 6):

| Method | V2V OS | V2V IF | V2V VC | V2V GQ | RV2V OS | RV2V IF | RV2V VC | RV2V IC | RV2V GQ |
|--------|--------|--------|--------|--------|---------|---------|---------|---------|---------|
| UniVideo [76](https://arxiv.org/abs/2510.08377) | 2.44 | 2.58 | 3.30 | 3.16 | 2.36 | 2.67 | 3.15 | 2.87 | 2.82 |
| VINO [9](https://arxiv.org/abs/2601.02358) | 2.85 | 3.08 | 3.14 | 3.26 | 2.25 | 2.64 | 2.17 | 3.51 | 3.06 |
| Kling O3 [68](https://arxiv.org/abs/2512.16776) | 3.05 | 3.25 | 3.09 | 3.44 | 3.14 | 3.41 | 3.14 | 3.61 | 3.30 |
| Wan2.7 [70](https://arxiv.org/abs/2503.20314) | 3.30 | 3.57 | 3.11 | 3.56 | 3.58 | 3.82 | 3.48 | 3.62 | 3.43 |
| **Bernini** | **3.49** | 3.66 | **3.51** | 3.49 | 3.50 | 3.75 | **3.51** | 3.54 | 3.31 |

**Bernini 的优势在 Video Consistency** (V2V: 3.51 vs Wan2.7 3.11, RV2V: 3.51 vs Wan2.7 3.48). 这正是 SA-3D RoPE + source VAE feature injection 的设计目标——最大程度 preserve 非编辑区域.

**Bradley-Terry Leaderboard** (Figure 1, anchored at geom. mean=1000):

| # | Method | BT Score | 95% CI | Win% | W-L-T |
|---|--------|----------|--------|------|-------|
| 1 | HappyHorse-1.0 | 1080 | [1020, 1150] | 61.3 | 56-32-0 |
| 2 | **Bernini (OURS)** | 1044 | [986, 1105] | 56.3 | 57-39-0 |
| 3 | Wan2.7 | 1034 | [970, 1097] | 54.9 | 40-30-1 |
| 4 | Grok-imagine-video | 964 | [906, 1019] | 44.9 | 40-48-1 |
| 5 | Kling_v3_omni | 878 | [810, 933] | 33.1 | 32-76-2 |

Bernini #2, 仅次于 HappyHorse-1.0. Win rate formula:
$$
\text{Win\%} = \frac{1}{1 + 10^{-(\bar{s} - 1000)/400}}
$$
其中 s̄ 是 average score. 这是 standard Bradley-Terry [5] formulation, 用 logit scale 400.

**Public Benchmarks**:

OpenVE-Bench (Table 7, Gemini 2.5 Pro evaluation):
- Bernini: 4.04 overall (next best VINO 3.18, +27% relative)
- 各 sub-category 都领先: Style 4.45, Local Change 4.85, Camera Edit 4.67

EditVerse (Table 8):
- Bernini: Editing Quality 8.02 (next best EditVerse 7.65)
- Pick Score 20.26, Text Alignment (Video) 24.62, all SOTA

FiVE (Table 9):
- Bernini: FiVE-Acc 78.16 (next best Omni 72.41)
- Structure Dist. 13.54 (lower better, second only to VideoGrain 12.40)
- Background PSNR 26.35, SSIM 84.38 (SOTA)

### 11.3 Reasoning-augmented Editing (Table 10)

| Method | OS | IF | VC | GQ |
|--------|-----|-----|-----|-----|
| Baseline | 3.12 | 3.36 | 3.18 | 3.37 |
| + PE (Qwen2.5-VL-7B) | 3.20 | 3.43 | 3.21 | 3.39 |
| + Self-text | 3.33 | 3.55 | 3.31 | 3.44 |
| + PE (GPT-5.4) | 3.49 | 3.66 | 3.51 | 3.49 |
| + PE (GPT-5.4) + Self-visual-text | **3.52** | 3.65 | **3.54** | 3.49 |

**Intuition**: 
1. Baseline → +PE (Qwen2.5-VL-7B): 用初始化模型当 prompt enhancer, 提升有限 (3.12 → 3.20)
2. +Self-text: 用 Bernini 自己的 textual CoT, 超过用 init model 当 PE (3.33 > 3.20), 说明 Bernini 训练后发展出更强的 textual reasoning
3. +PE (GPT-5.4): 用更强的外部 LLM 当 prompt enhancer, 大幅提升 (3.49)
4. +Self-visual-text: 加入 visual intermediate, 最优 (3.52), 证明 multimodal reasoning > pure text reasoning

### 11.4 Video Generation

**VBench T2V** (Table 11):
- Bernini: Total 84.64
- Wan2.2-A14B (base): 84.79
- 几乎 retain base 的 T2V capability, 即使 unified framework 多了 editing 和 S2V

**OpenS2V Subject-to-Video** (Table 12):
- Bernini: Total 62.94 (#1, 超过所有 closed-source 和 open-source)
- Kling O3: 59.19
- RefAlign-14B [72](https://arxiv.org/abs/2603.25743): 60.42
- **FaceSim 78.20** (next best Kling O3 57.20, +20 absolute points!)

FaceSim 的巨大优势来自 Person R2V data pipeline——用 real footage 的 same-identity reference, 而不是 image editor 输出. 这保证了 identity preservation.

### 11.5 Ablation Studies

**SA-3D RoPE effect** (Figure 15):
- Standard 3D RoPE: reference leakage artifacts (background 泄漏, duck head 错位)
- 3D RoPE + learnable segment embedding: 改善 reference consistency, 但仍有 leakage
- **SA-3D RoPE**: cleanest isolation, no leakage

**ViT Semantic Interface + MLLM Planner** (Figure 16):
- Full model: 准确 object replacement + style transfer
- Remove ViT interface: weaker instruction following (fail to replace robot with robotic dog, omit flying birds in ink wash style)
- Remove both ViT and MLLM: further degradation

**证明**: ViT semantic interface 和 MLLM planner 是 complementary 的, 都不可或缺.

### 11.6 Generalizations (Section 6.7)

Bernini 能处理训练数据里没有的 editing types:
- Watercolor stylization
- 2D/3D animation
- Weather changes
- Effect additions
- Motion changes, focus shifts, position changes
- **Causal reasoning**: "prolonged heavy rain" → infer "fire 应该灭了" (no explicit causal supervision in training)

这种 generalization 来自 MLLM pre-trained 的 understanding capability, 通过 L_ntp 保留下来, transfer 到 generation.

---

## 十二、Related Work 定位

### 12.1 Joint Multimodal Backbones

- **Emu3** [73](https://arxiv.org/abs/2409.18869): text + image + video 共享 discrete vocab, pure next-token prediction. 简单但 generation quality 受限于 discrete tokenization
- **Janus** [79](https://arxiv.org/abs/2503.07229): decoupled visual encoders (SigLIP for understanding, VQ for generation), shared autoregressive backbone
- **Show-o** [82](https://arxiv.org/abs/2408.12528): AR for text + discrete diffusion for image, single transformer
- **HunyuanImage 3.0** [7](https://arxiv.org/abs/2509.23951): MoE decoder, NTP for text + diffusion for image
- **BAGEL** [15](https://arxiv.org/abs/2505.14683): MoTE (Mixture of Transformer Experts), understanding + generation experts 通过 shared self-attention 交互, dual visual encoders
- **Lumina-DiMOO** [83](https://arxiv.org/abs/2510.06308): fully discrete masked diffusion, single objective over both modalities, shared vocab

### 12.2 MLLM as Conditioner

按 interface 宽度排列:

**Narrow interfaces** (text tokens / learnable queries):
- **MetaQuery** [54](https://arxiv.org/abs/2504.06256): learnable query tokens → diffusion
- **Bifrost-1** [42](https://arxiv.org/abs/2508.05954): patch-level CLIP latents as bridge

**Wide interfaces** (hidden states):
- **SEED-X** [22](https://arxiv.org/abs/2404.14396): MLLM hidden states → image decoder
- **DreamLLM** [18](https://arxiv.org/abs/2310.13885): MLLM hidden states → external image decoder
- **Emu** [65](https://arxiv.org/abs/2307.05222): MLLM hidden states → image decoder
- **LaVi-Bridge** [97](https://arxiv.org/abs/2303.16160): connect frozen LLM + vision generator
- **UniVideo** [76](https://arxiv.org/abs/2510.08377): MLLM + video diffusion, hidden states (with learnable queries or interleaved context)
- **VINO** [9](https://arxiv.org/abs/2601.02358): MLLM + video diffusion, hidden states + VAE latents of references

**Bernini 的位置**: 跟 UniVideo/VINO 同属 decoupled paradigm, 但 interface 锚定在 **ViT embedding space** 而非 raw hidden states. 这是关键区别——让 pre-trained visual semantics 直接 transfer, 不需要 heavy alignment training.

### 12.3 Bernini 的差异化优势

1. **Native representation transfer**: ViT embedding space 是 MLLM 已经"理解"的空间, 不需要学新的 representation
2. **Modular preservation**: 3-stage training 保留两个 foundation 的 pre-trained strength
3. **SA-3D RoPE**: 解决 multi-segment 的 position ambiguity, 这是 unified sequence formulation 的 unique challenge
4. **Latent CoT**: 在 latent space 做 chain-of-thought, 比纯 text CoT 多了 visual grounding
5. **Task-dependent mask ratio**: 用 Beta 分布 control information leakage, 针对不同 task 的特点调参

---

## 十三、Limitations

Paper 自己承认:
1. **Foundation model dependency**: Bernini 受限于 MLLM planner (Qwen2.5-VL-7B) 和 DiT renderer (Wan2.2-A14B) 的强度. 复杂 editing 仍然需要 strong LLM rewriter (GPT-5.4) 来提供 detailed instructions, 说明 native reasoning 还不够强.
2. **Visual quality gap**: S2V 的 consistency SOTA, 但 visual quality 仍不及 closed-source 系统 (Wan2.7).
3. **Inference cost**: 虽然 distill 到 4 NFE, 但 planner 的 25 步 iterative refinement + 5 步 ViT decoder denoising 仍然有 overhead, 即使 paper 声称 negligible.

---

## 十四、关键 References

### Foundation Models
- **Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
- **Wan**: https://arxiv.org/abs/2503.20314
- **Qwen3-VL**: https://arxiv.org/abs/2511.21631
- **Qwen3**: https://arxiv.org/abs/2505.09388

### Architecture Components
- **MAR** (Li et al.): https://arxiv.org/abs/2406.11838
- **MaskGIT**: https://arxiv.org/abs/2203.04818
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **DiT**: https://arxiv.org/abs/2212.09748
- **ViT**: https://arxiv.org/abs/2010.11929
- **CLIP**: https://arxiv.org/abs/2103.00020
- **SigLIP 2**: https://arxiv.org/abs/2502.14786
- **3D RoPE / RoFormer**: https://arxiv.org/abs/2104.09864
- **SD3**: https://arxiv.org/abs/2403.03206

### Related Unified Models
- **Emu3**: https://arxiv.org/abs/2409.18869
- **Janus**: https://arxiv.org/abs/2503.07229
- **Show-o**: https://arxiv.org/abs/2408.12528
- **BAGEL**: https://arxiv.org/abs/2505.14683
- **HunyuanImage 3.0**: https://arxiv.org/abs/2509.23951
- **Lumina-DiMOO**: https://arxiv.org/abs/2510.06308
- **UniVideo**: https://arxiv.org/abs/2510.08377
- **VINO**: https://arxiv.org/abs/2601.02358
- **MetaQuery**: https://arxiv.org/abs/2504.06256
- **Bifrost-1**: https://arxiv.org/abs/2508.05954
- **SEED-X**: https://arxiv.org/abs/2404.14396
- **DreamLLM**: https://arxiv.org/abs/2310.13885
- **Emu**: https://arxiv.org/abs/2307.05222
- **OmniGen2**: https://arxiv.org/abs/2505.18404

### Data & Evaluation
- **OmniCorpus**: https://arxiv.org/abs/2406.08418
- **VBench**: https://arxiv.org/abs/2311.05222
- **OpenS2V-Eval**: https://arxiv.org/abs/2505.20292
- **OpenVE-Bench**: https://arxiv.org/abs/2512.07826
- **EditVerse**: https://arxiv.org/abs/2509.20360
- **FiVE**: https://arxiv.org/abs/2503.13684
- **HowTo100M**: https://arxiv.org/abs/1906.03327
- **X-CLIP**: https://arxiv.org/abs/2207.07085
- **SAM 2**: https://arxiv.org/abs/2408.00714
- **GroundingDINO**: https://arxiv.org/abs/2303.05499
- **DWPose**: https://arxiv.org/abs/2306.06028
- **DifuEraser**: https://arxiv.org/abs/2501.10018
- **VACE**: https://arxiv.org/abs/2503.06940
- **Senorita-2M**: NeurIPS 2025
- **TokenFlow**: https://arxiv.org/abs/2307.07992

### Infrastructure
- **DeepSpeed Ulysses**: https://arxiv.org/abs/2309.14509
- **FlashAttention-4**: https://arxiv.org/abs/2603.05451
- **FlexAttention**: https://arxiv.org/abs/2412.05496
- **QuACK**: https://github.com/Dao-AILab/quack
- **APG** (Adaptive Projected Guidance): https://arxiv.org/abs/2410.02431
- **ReFlow**: https://arxiv.org/abs/2209.03003
- **CFG Distillation** (Progressive Distillation): https://arxiv.org/abs/2203.01248

### CoT & Reasoning
- **Visual Sketchpad**: https://arxiv.org/abs/2406.04823
- **Chain-of-Thought**: https://arxiv.org/abs/2201.11903

---

## 十五、Final Thoughts

Bernini 的核心贡献可以浓缩成三个 idea:

1. **Semantic interface design matters**: 选对 interface representation (ViT embedding space 而非 hidden states) 能让两个 pre-trained foundation 直接 transfer capability, 避免 heavy alignment training. 这是 modular decoupling 哲学的胜利.

2. **Position encoding for multi-segment sequences**: SA-3D RoPE 通过 multiplicative phase modulation 解决了 unified sequence 里 multi-segment 的 position ambiguity. 这个 idea 虽然简单, 但是解决了一个 unified framework 的 fundamental 问题——当不同 visual inputs share 相同 (t,h,w) 坐标时, standard RoPE 失效. Additive segment embedding 不够, 需要 multiplicative phase.

3. **Preserve understanding through data and objectives**: 3-stage training + L_ntp + understanding data retention, 让 MLLM 在变成 planner 的过程中不丢失 reasoning capability. 这是 Bernini 能 generalize到 OOD editing instructions (causal reasoning like "rain extinguishes fire") 的根本原因.

潜在 future directions:
- 用更强的 MLLM (比如 Qwen3-VL-235B) 作为 planner, 可能解决 native reasoning 不足的问题
- 在 ViT embedding decoder 里引入更强的 generative model (比如 diffusion in embedding space with better architecture)
- 把 SA-3D RoPE 的思想扩展到 N-D RoPE, 处理更高维度的 segment 区分
- 把 latent CoT 扩展到 multi-step visual reasoning, 类似 Visual Sketchpad 但在 latent space

这篇 paper 的工程价值很高——data pipeline (20M video pairs, 30M image pairs, motion-aware dual-branch synthesis, propagation-based boosting) 和 infrastructure (440K token sequences, 4.5× throughput, 4 NFE distillation) 都是 industrial-scale 的 best practice, 对想要 build 类似系统的人有参考价值.
