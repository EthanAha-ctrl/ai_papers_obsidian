---
source_pdf: SLOWFAST-LLAVA.pdf
paper_sha256: 090288e101a44301e4e8ad51dce9e55489279f074f85fc035c3883bfeb8e0182
processed_at: '2026-08-12T07:51:55-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SlowFast-LLaVA

Paper: https://arxiv.org/abs/2407.15841

---

## 一句话版本

你有一段视频要喂给 LLM,但 LLM 的 context window 就那么大(4096~8192 token),你得算账:给每帧分多少 token,采多少帧。这篇 paper 说:**分两路走**——Slow pathway 少采几帧但每帧看清细节,Fast pathway 多采几帧但每帧看粗糙点,两路 concat 喂给 LLM。不训练,直接用 LLaVA-NeXT,性能逼近 SFT 模型。

---

## 生活中的类比

想象你给朋友讲一部两小时的电影。

你不会把每一帧都拍下来给他看(那是 216000 帧,看不过来),你也不会只挑 4 张图给他看(剧情线完全丢了)。

你会怎么做?你会 **挑几个关键场景仔细讲讲细节**(谁说了什么、穿了什么衣服、桌上有什么东西),然后 **把整个剧情的时间线快速过一遍**(开头是啥、中间是啥、结尾是啥)。

这就是 SlowFast。Slow = 关键场景的精讲,Fast = 整个剧情的快进扫描。

---

## 为什么这是个真问题

Video LLM 的核心矛盾就是 token budget 分配:

```
Total Token Budget = 4096 (or 8192 with RoPE)
                   = N_frames × Tokens_per_frame
```

- 想看清物体细节(object recognition, OCR, counting)→ 每帧多 token → 帧数少 → temporal context 差
- 想看懂时间变化(motion, event sequence)→ 多帧 → 每帧少 token → spatial detail 丢

之前 training-free 的方法(FreeVA 4 帧,IG-VLM 6 帧)都偏 spatial,所以长视频理解很烂。EgoSchema(3分钟第一人称视频)上 IG-VLM 只有 35.8%。

---

## SlowFast 的具体配比

默认配置(7B/34B 模型):

```
Input video
    │
    ▼ Uniform sample 50 frames
50 frames, each 24×24 = 576 tokens (from CLIP-L-14)
    │
    ┌────────────────────┴────────────────────┐
    ▼                                         ▼
Slow pathway                              Fast pathway
- 取 10 帧 (temporal stride 5)            - 取全部 50 帧
- Spatial pool 24×24 → 12×24              - Spatial pool 24×24 → 4×4
- 每帧 288 tokens                          - 每帧 16 tokens
- Total: 10×288 = 2880 tokens              - Total: 50×16 = 800 tokens
    │                                         │
    └─────────────────┬───────────────────────┘
                      ▼
              Concat → 3680 tokens
                      │
                      ▼
                  LLM (Vicuna-7B/34B)
```

Token 经济学:
- Slow 用 2880 token 买 10 帧的高 spatial detail
- Fast 用 800 token 买 50 帧的 dense temporal coverage
- 总共 3680 token,刚好 fit 在 context window 里

---

## 为什么这个比例 work

我前面看 ablation 时琢磨出来的 intuition:

### Intuition 1:LLM 能补 temporal,不能补 spatial

LLM 是 sequence modeler,你给它 50 个 frame embeddings 排成一列,加上 RoPE 的 position encoding,它自己能 infer 出"frame 1 到 frame 50 的顺序"和"frame 10 到 frame 20 发生了 motion"。这部分能力 LLM 本来就有。

但 spatial 不行。你把一张 24×24 token 的图 pool 成 4×4 token,物体细节全丢了,LLM 没法"脑补"出来。一张模糊的小图给 LLM,它就只能猜。

所以:**spatial 必须保留 resolution,temporal 可以用稀疏 token 标记**。

### Intuition 2:Fast pathway 主要是 temporal marker,不是 spatial content

Table 6 有个让我震惊的 ablation:把 Fast pathway 的每帧 pool 到 **1×1 token**(每帧只有 1 个 token),性能居然没崩!

| Fast pathway config | ANet-QA (7B) | EgoSchema (7B) |
|---|---|---|
| 50 frames × 8×8 = 3200 tokens | 54.9 | **48.6** |
| 50 frames × 4×4 = 800 tokens | 55.5 | 47.2 |
| 50 frames × 1×1 = 50 tokens | 54.4 | 47.1 |

1 个 token 等于告诉 LLM:"在这个时间点,视频大概长这样。"LLM 用 Slow pathway 的 10 帧 spatial detail 做 anchor,用 Fast pathway 的 50 个时间点做 temporal scaffold。

这暗示了:**对于 30 分钟以上的长视频,Fast pathway 应该用 1×1**,这样能 cover 几千帧。

### Intuition 3:Token 数等量时,spatial pooling 比 temporal pooling 好

Table 5 (7B model):

| Slow pathway config | Tokens | ANet-QA | EgoSchema |
|---|---|---|---|
| 10 frames × 24×24 | 5760 | 54.0 | 53.2 |
| 10 frames × 12×24 (spatial pool) | 2880 | **55.5** | 47.2 |
| 5 frames × 24×24 (temporal pool) | 2880 | 54.5 | 44.4 |

同样是 2880 token,spatial pool(保持 10 帧,每帧减半)比 temporal pool(减到 5 帧,每帧保持)好。尤其在 EgoSchema 上差距 2.8%。

直觉:LLM 用 RoPE 能自己 modeling temporal order,但减帧就是减帧,丢了就是丢了。spatial pool 反而让 LLM 多了更多时间锚点。

---

## 跟 SFT 对比的 surprise

这是 paper 最让我意外的点。

Table 1 (7B models):

| Method | Training | MSVD-QA | MSRVTT-QA | TGIF-QA | ANet-QA |
|---|---|---|---|---|---|
| PLLaVA | SFT on video | 76.6 | 62.0 | 77.5 | 56.3 |
| LLaVA-NeXT-Video-DPO | SFT + DPO | - | - | - | 60.2 |
| **SF-LLaVA** | **No training** | **79.1** | **65.8** | **78.7** | **55.5** |

SF-LLaVA 在 MSVD/MSRVTT/TGIF 上反超 SFT 的 PLLaVA,只有 ANet-QA 略低 0.8%。

这说明了什么?

**Video understanding 的瓶颈不在 LLM 能力,在 input representation**。

SFT 方法花大量 GPU 在 video instruction tuning 上,本质是教 LLM 怎么处理 video token sequence。但 SF-LLaVA 证明了:只要 representation 设计得对(SlowFast 双流),加上合适的 prompt,LLM 不用训练也能理解视频。

这跟 image LLM 的发展路径很像——早期的 image LLM 需要 visual instruction tuning,后来发现 LLaVA-NeXT 这种简单 linear connector + good data 就够强。Video 现在也走到这一步了。

---

## Prompt 是 training-free 的 "training"

Table 7 的 ablation 让我印象深刻。三个 prompt 组件分别 ablate:

| 去掉什么 | ANet-QA (7B) | EgoSchema (7B) |
|---|---|---|
| 什么都不去(baseline) | 54.9 | 47.2 |
| 去 task instruction | 55.5(+0.6) | 43.0(-4.2) |
| 去 input data prompt | 52.6(-2.3) | 44.8(-2.4) |
| 去 structured answer prompt | 52.7(-2.2) | 44.4(-2.8) |

Input data prompt "The input consists of a sequence of key frames from a video" 特别重要。原因:LLaVA-NeXT 训练时见的是 "image",你塞给它 3680 个 visual token 排成一列,它默认理解成"很多张图",需要明确告诉它"这是一个视频的帧序列"。

这个 prompt engineering 在 training-free 里替代了 SFT 的作用——给 LLM 提供 task context。

---

## 这个 paper 没解决的问题

作者自己在 4.6 节承认:

1. **Temporal grounding 不行**:不能精确定位"第 5 秒到第 8 秒发生了什么"。因为从没在 grounding 数据上训练过。LLM 能讲发生了什么,但说不出具体时刻。

2. **Uniform sampling 漏关键帧**:Fig. 5 的 Q3,一个快速动作"打开冰箱"正好在两个采样帧之间被漏掉了。这个 issue 在短视频上更严重(50 帧采 30 秒视频可能 OK,采 3 秒视频就太稀疏)。

3. **Long video 的 scaling**:30 分钟视频 50 帧根本不够。作者暗示 Fast pathway 应该用 1×1 token,这样能 cover 几千帧,但没实验。

---

## 我的 speculation 和 open questions

看完 paper 我有几个想法:

### 1. 为什么不加 special token 区分 Slow 和 Fast?

Paper 里 Eq 4 明确说:"we do not use any special tokens in F_v^aggr to separate the Slow and Fast pathways."

这意味着 LLM 看到 3680 个 visual token,需要自己 infer 出"前 2880 个是 high-res spatial,后 800 个是 low-res temporal"。

为什么 work?我猜是因为 LLM 见过 token 维度的 pattern——不同 token embedding 的 magnitude 和分布不同,LLM 能 implicitly 区分。但加 special token(比如 `<slow>` 和 `<fast>`)应该更好,paper 没试,是个 open question。

### 2. Fast pathway 的 4×4=16 token,信息量够吗?

4×4 pooling 等于把 24×24=576 token 压成 16 token,信息压缩率 36×。但 CLIP feature 本身就有 spatial redundancy(相邻 patch 高度相关),4×4 average pool 保留的是 global layout,正好是 motion 需要的信息(物体大致位置)。

这跟 SlowFast Networks 原始 paper 的 motivation 一致:Fast pathway 处理 low spatial frequency,Slow pathway 处理 high spatial frequency。

### 3. 这个 idea 能 extend 到其他 modality 吗?

SlowFast 的核心是 "token budget 分两条 pathway"。这个 idea 可以 extend:
- **Audio LLM**:Slow = 高采样率短片段,Fast = 低采样率长片段
- **Code LLM**:Slow = 完整函数定义,Fast = 整个文件的 import 和签名
- **Document LLM**:Slow = 关键段落全文,Fast = 所有段落的摘要

只要是 token-limited + multi-resolution information 的场景,SlowFast 都适用。

---

## 给你的 takeaway

如果只记一件事:

> **在 token budget 受限时,与其给每帧均匀分配 token,不如让一部分 token 专门买 spatial detail(Slow),另一部分专门买 temporal coverage(Fast)。spatial 信息 LLM 不能补,temporal 信息 LLM 能补。**

这个 inductive bias 来自生物视觉(ventral/dorsal pathway)和 SlowFast Networks,但在 LLM token 层面被重新实现了,而且不训练就 work。

说明 LLM 的 video understanding 能力其实够了,缺的是好的 input representation。未来的 Video LLM 工作应该更多关注 representation design,而不只是 scaling SFT data。

---

## 相关 links

- SlowFast Networks 原始 paper: https://arxiv.org/abs/1812.03982
- LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- LLaVA-NeXT-Video (SFT baseline): https://llava-vl.github.io/blog/2024-04-30-llava-next-video/
- IG-VLM: https://arxiv.org/abs/2403.18406
- FreeVA: https://arxiv.org/abs/2405.07798
- PLLaVA: https://arxiv.org/abs/2404.16994
- EgoSchema benchmark: https://arxiv.org/abs/2308.09116
- CLIP: https://arxiv.org/abs/2103.00020
- RoPE: https://arxiv.org/abs/2104.09864

---

# SlowFast-LLaVA: Training-Free Video LLM 的 SlowFast 双流设计

Paper 链接: https://arxiv.org/abs/2407.15841
Code: https://github.com/apple/ml-slowfast-llava
LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/

---

## 1. 核心动机:为什么需要 SlowFast?

### 1.1 问题本质

这篇 paper 要解决的核心问题是 **token budget allocation**。在 Video LLM 中,给定一个固定的 token budget (例如 LLaVA-NeXT 的 4096 token context window,加 RoPE scaling 后扩展到 8192),你需要在 **spatial resolution** (每帧多少 token) 和 **temporal coverage** (采多少帧) 之间做权衡。

- 想要 spatial detail → 每帧多 token → 帧数少 → temporal context 差
- 想要 temporal context → 多帧 → 每帧少 token → spatial detail 丢失

现有 training-free 方法 (如 FreeVA 用 4 帧,IG-VLM 用 6 帧) 都倾向前者,导致 long video understanding 很差。

### 1.2 SlowFast 灵感

灵感来自 SlowFast Networks (Feichtenhofer et al., 2019, ICCV) 和生物视觉的双流系统:
- **Ventral pathway (parvocellular)**: 慢,精细 spatial,识别 "what"
- **Dorsal pathway (magnocellular)**: 快,粗糙 spatial,敏感 motion,识别 "where/how"

这篇 paper 把这个 idea 搬到 LLM 的 token 层面:**用两条 pathway 分别承担 spatial 和 temporal,然后 concat 喂给 LLM**。

---

## 2. 架构详解

### 2.1 整体流程

```
Video V
   │
   ▼ Uniform sample N=50 frames
I = {I_1, ..., I_50}
   │
   ▼ CLIP-L-14 visual encoder (每帧独立)
F_v ∈ R^(N × H × W)  where H=W=24
   │
   ┌───────────┴───────────┐
   ▼                       ▼
Slow pathway            Fast pathway
N^slow=10               N^fast=50
pool stride 1×2         pool stride 6×6
→ F_v^slow ∈           → F_v^fast ∈
  R^(10×12×24)           R^(50×4×4)
   │                       │
   └─────────┬─────────────┘
             ▼ flatten + concat
F_v^aggr ∈ R^3680
             │
             ▼ LLM (Vicuna-7B/34B)
          Answer A
```

### 2.2 关键公式解析

**Eq 1** (一般 training-free Video LLM):
$$\mathbf{A} = \text{LLM}(\mathbf{Prompt}, \text{Aggregator}(\text{Visual}_{\text{enc}}(\mathbf{I})), \mathbf{Q})$$

- $\mathbf{A}$: 输出的 answer
- $\mathbf{Prompt}$: system/instruction prompt
- $\mathbf{I} = \{I_1, ..., I_N\}$: 采样的 N 帧
- $\text{Visual}_{\text{enc}}$: CLIP-L,每帧独立 encode (no temporal mixing at encoding stage)
- $\mathbf{Q}$: 用户 question
- $\text{Aggregator}$: pooling 操作,关键步骤

**Eq 2** (Slow pathway):
$$\mathbf{F}_v \in \mathbb{R}^{N \times H \times W} \xrightarrow[\text{temporal}]{\text{spatial pool}} \mathbf{F}_v^{\text{slow}} \in \mathbb{R}^{N^{\text{slow}} \times H^{\text{slow}} \times W^{\text{slow}}}$$

- $N^{\text{slow}}$: Slow pathway 用的帧数, $N^{\text{slow}} = 10 \ll N = 50$
- $H^{\text{slow}} = H / \sigma_h$, $W^{\text{slow}} = W / \sigma_w$
- 实际配置: $\sigma_h = 1, \sigma_w = 2$,所以 $H^{\text{slow}}=24, W^{\text{slow}}=12$,即每帧 12×24=288 token
- 注意:这里有两个采样,一个 temporal (从 50 选 10),一个 spatial (24×24 → 12×24,只在 W 方向 pool)

**Eq 3** (Fast pathway):
$$\mathbf{F}_v \in \mathbb{R}^{N \times H \times W} \xrightarrow{\text{spatial pool}} \mathbf{F}_v^{\text{fast}} \in \mathbb{R}^{N^{\text{fast}} \times H^{\text{fast}} \times W^{\text{fast}}}$$

- $N^{\text{fast}} = N = 50$:保留全部帧
- $H^{\text{fast}} = H / \gamma_h$, $W^{\text{fast}} = W / \gamma_w$
- 实际配置: $\gamma_h = \gamma_w = 6$,所以 $H^{\text{fast}} = W^{\text{fast}} = 4$,每帧只有 4×4=16 token
- 关键:Fast pathway 的"cheap"在于每个 frame 只用 16 token,所以可以覆盖 50 帧只用 800 token

**Eq 4** (总体):
$$\mathbf{A} = \text{LLM}(\mathbf{Prompt}, [\text{Slow}(\mathbf{F}_v), \text{Fast}(\mathbf{F}_v)], \mathbf{Q})$$

- $[\cdot, \cdot]$ 是 token 维度的 concatenation
- **注意:没有用 special token 分隔 Slow 和 Fast**,直接 concat。这点很有意思,意味着 LLM 需要自己 infer 出 token 的角色

### 2.3 Token 预算计算

默认 7B/34B 配置:
- Slow: $10 \times 12 \times 24 = 2880$ token
- Fast: $50 \times 4 \times 4 = 800$ token
- **Total: 3680 visual token**

加上 text prompt + question 大概几百 token,整个 sequence 在 4096~4500 左右,加 RoPE scaling (factor=2) 后 context 长度上限 8192,刚好 fit 34B 模型在 80GB GPU 上。

---

## 3. 设计直觉 (build your intuition)

### 3.1 为什么这个比例 (Slow 2880 : Fast 800) 而不是 50:50?

核心 insight:**spatial 信息丢失的代价 > temporal 信息稀疏化的代价**。

- Slow pathway 必须保持 spatial resolution,因为物体识别、OCR、counting 都依赖 local visual detail,LLM 无法"脑补"丢失的 spatial 信息。
- Fast pathway 可以激进 pool 到 4×4=16 token,因为:
  1. Motion 是粗粒度的(物体从 A 区移到 B 区,4×4 grid 够用)
  2. LLM 自己有 temporal attention,只要给它 frame sequence,它能 infer motion pattern
  3. 50 帧的 dense temporal sampling 弥补了每帧 spatial 的稀疏

### 3.2 为什么不只用 Slow,加更多帧?

Paper Table 4 做了这个 ablation。把 Fast pathway 从 50 帧加到 200 帧(去掉 Slow,纯 Fast),token 数 3200 vs SF-LLaVA 的 3104,基本等量。

| Method | Tokens | ANet-QA (7B) | EgoSchema (7B) |
|---|---|---|---|
| Fast only, N^fast=200 | 3200 | 49.7% | 37.0% |
| SF-LLaVA (N^slow=8, N^fast=50) | 3104 | 54.6% | 46.0% |

**等量 token 下 SlowFast 完胜**。说明这不是 token 数量问题,是 representation structure 问题。Slow 提供的 spatial anchor 让 LLM 能"读懂" Fast 流的 motion。

### 3.3 为什么不只用 Fast,加更多帧?

Table 5 (7B model):

| Slow tokens | ANet-QA | EgoSchema |
|---|---|---|
| 10×24×24 = 5760 | 54.0/3.3 | 53.2 |
| 10×12×24 = 2880 | **55.5/3.3** | 47.2 |
| 5×24×24 = 2880 | 54.5/3.3 | 44.4 |
| 5×12×12 = 720 | 51.4/3.2 | 36.4 |

关键观察:**在 token 数等量时,spatial pooling (10×12×24=2880) 优于 temporal pooling (5×24×24=2880)**,尤其在 EgoSchema (long video reasoning) 上差距明显(47.2 vs 44.4)。

直觉:LLM 自己能 modeling temporal order(尤其有 RoPE),但无法 recover spatial information。

### 3.4 Fast pathway 的 pooling 探索 (Table 6)

| Fast tokens | ANet-QA (7B) | EgoSchema (7B) |
|---|---|---|
| 50×8×8 = 3200 | 54.9/3.3 | **48.6** |
| 50×4×4 = 800 | 55.5/3.3 | 47.2 |
| 50×1×1 = 50 | 54.4/3.3 | 47.1 |

注意 **1×1 (每帧只有 1 token) 性能没崩!** 这是个非常重要的 insight:**Fast pathway 主要贡献是 temporal marker,不是 spatial content**。每帧 1 个 token 等于告诉 LLM "这里有一个时间点,内容大概是这样",LLM 自己去 aggregate。

这也提示了作者提到的一个方向:**对于 long video (30+ min),Fast pathway 应该用 1×1** 来 cover 更多帧。

### 3.5 Prompt 设计的重要性 (Table 7)

对 training-free 方法, prompt 是唯一的 "training signal"。

| Prompt config (7B) | ANet-QA | EgoSchema |
|---|---|---|
| Full (task + input data + structured answer) | 54.9/3.3 | **47.2** |
| 去掉 task instruction | 55.5/3.4 | 43.0 (-4.2) |
| 去掉 input data prompt | 52.6/3.2 | 44.8 (-2.4) |
| 去掉 structured answer prompt | 52.7/3.4 | 44.4 (-2.8) |

**Input data prompt** "The input consists of a sequence of key frames from a video" 至关重要。原因:LLaVA-NeXT 训练时见的是 "image",你塞给它一堆 token,它需要被明确告知这是 video frames sequence,而不是单张图。

---

## 4. 实验结果详解

### 4.1 Open-Ended VideoQA (Table 1a, 7B 级别)

| Method | Training | MSVD-QA | MSRVTT-QA | TGIF-QA | ANet-QA |
|---|---|---|---|---|---|
| Video-ChatGPT | SFT | 64.9/3.3 | 49.3/2.8 | 51.4/3.0 | 35.2/2.7 |
| PLLaVA | SFT | 76.6/4.1 | 62.0/3.5 | 77.5/4.1 | 56.3/3.5 |
| LLaVA-NeXT-Video-DPO | SFT | - | - | - | 60.2/3.5 |
| FreeVA | Training-free | 73.8/4.1 | 60.0/3.5 | - | 51.2/3.5 |
| IG-VLM | Training-free | 78.8/4.1 | 63.7/3.5 | 73.0/4.0 | 54.3/3.4 |
| **SF-LLaVA-7B** | **Training-free** | **79.1/4.1** | **65.8/3.6** | **78.7/4.2** | **55.5/3.4** |

- SF-LLaVA 在 7B training-free 中全面领先
- 跟 SFT 的 PLLaVA 比,在 MSVD/MSRVTT/TGIF 上 on-par,只有 ANet-QA 略低
- 跟 SFT 的 DPO 版本比,ANet-QA 还差 4.7% (55.5 vs 60.2)

### 4.2 Multiple Choice VideoQA (Table 2a, 7B)

| Method | NExTQA | EgoSchema | IntentQA |
|---|---|---|---|
| IG-VLM | 63.1 | 35.8 | 60.3 |
| **SF-LLaVA-7B** | **64.2** | **47.2** | 60.1 |

**EgoSchema 上 +11.4%** 是最大亮点。EgoSchema 是长视频 (3 min) egocentric reasoning benchmark,这正验证了 Fast pathway 的价值——long-form temporal context。

### 4.3 Text Generation (Table 3, 34B)

VCGBench 五个维度:CI (Correctness), DO (Detail), CU (Contextual), TU (Temporal), CO (Consistency)。

| Method | CI | DO | CU | TU | CO | Avg |
|---|---|---|---|---|---|---|
| LLaVA-NeXT-Image (training-free, 34B) | 3.29 | 3.23 | 3.83 | 2.51 | 3.47 | 3.27 |
| **SF-LLaVA-34B** | 3.48 | 2.96 | 3.84 | **2.77** | 3.57 | 3.32 |
| LLaVA-NeXT-Video-DPO (SFT) | 3.81 | 3.55 | 4.24 | 3.14 | 4.12 | 3.77 |

关键观察:
- **TU (Temporal Understanding) 在 training-free 中领先** (+0.26 over LLaVA-NeXT-Image),验证 SlowFast 的 temporal 建模能力
- **DO (Detail Orientation) 输给 LLaVA-NeXT-Image**,因为后者用 32 frames × 12×12 tokens = 4608 token 全 spatial,而 SF-LLaVA 只有 10 frames × 12×24 = 2880 spatial token
- 跟 SFT DPO 比仍有差距,但已经很小

---

## 5. 跟其他 training-free 方法的对比

### 5.1 vs FreeVA (Wu, 2024)
- FreeVA 用 4 帧,加 temporal pooling 聚合
- 缺点:frame 数太少,long video 不行
- SF-LLaVA 用 50 帧,通过 Fast pathway 实现 dense temporal coverage

### 5.2 vs IG-VLM (Kim, 2024)
- IG-VLM 把 6 帧 arrange 成 image grid,直接当 image 喂 LLaVA
- 缺点:frame 数受 grid layout 限制
- SF-LLaVA 用 independent frames,更灵活

### 5.3 vs LLaVA-NeXT-Image (Zhang, 2024b)
- 直接用 LLaVA-NeXT 处理 32 帧,每帧 12×12 = 4608 token
- 没有 SlowFast 区分,uniform 处理所有帧
- DO 略好(更多 spatial token),但 TU 差(没有 Fast pathway 的 dense temporal)

---

## 6. Limitations 和 future directions (作者自己指出)

1. **Temporal grounding 能力缺失**:SF-LLaVA 不能精确定位事件起止时间 (Q2 in Fig. 5)。因为没在 video grounding 数据上训练过。
   - Future: 加入 timestamp、subtitle、audio modality
   
2. **Uniform sampling 会漏关键帧**:Q3 的"开冰箱"瞬间被漏掉。
   - Future: dynamic sampling,关键帧检测

3. **Long video scaling**:30+ min 视频,50 帧不够。
   - Future: Fast pathway 用 1×1 token,可以 cover 1000+ 帧

---

## 7. 我的 Intuition 总结

这个 paper 的核心 contribution 很简单但很 effective:**用 SlowFast 的 inductive bias 来 allocate token budget**。

关键 insight 我提炼三条:

1. **LLM 是 sequence modeler,但不是 spatial completer**:你可以让它处理稀疏 temporal(50 帧每帧 16 token),但别让它处理稀疏 spatial(每帧 4 token 识别物体不行)。所以 Slow 要 spatial rich,Fast 可以 spatial poor。

2. **Token economics**:800 token (Fast) 买 50 帧的 temporal coverage,2880 token (Slow) 买 10 帧的 spatial detail。这个 exchange rate 是 sweet spot。

3. **Training-free 的 limit 来自 representation,不是 LLM**:同样的 LLM,只改 input representation 就能接近 SFT 模型,说明 LLM 的 video understanding 能力其实够了,关键是怎么把视频"翻译"成 LLM 能消化的 token sequence。Prompt engineering 在 training-free 里是真正的 "training"。

---

## 8. 相关 links

- SlowFast Networks (原始 inspiration): https://arxiv.org/abs/1812.03982
- LLaVA-NeXT: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- LLaVA-NeXT-Video (SFT baseline): https://llava-vl.github.io/blog/2024-04-30-llava-next-video/
- IG-VLM: https://arxiv.org/abs/2403.18406
- FreeVA: https://arxiv.org/abs/2405.07798
- PLLaVA: https://arxiv.org/abs/2404.16994
- EgoSchema benchmark: https://arxiv.org/abs/2308.09116
- CLIP: https://arxiv.org/abs/2103.00020
- RoPE: https://arxiv.org/abs/2104.09864

---

简单说,这篇 paper 给了我一个很 clean 的 takeaway:**在 token-limited 的 multimodal setting 下,与其均匀分配 token 给所有帧,不如让一部分 token 专门承担 spatial detail (slow),另一部分 token 专门承担 temporal coverage (fast)**。这个 inductive bias 来自生物视觉和 SlowFast Networks,但在 LLM token 层面被重新实现了。整个 paper 没有任何 training,完全靠 representation engineering 和 prompt engineering,却能逼近 SFT 模型,说明 video understanding 的 bottleneck 不在 LLM 能力,在 input representation。
