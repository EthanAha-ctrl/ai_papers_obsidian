---
source_pdf: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning.pdf
paper_sha256: 1637ed6b551ba33c3addf243894579a9858f02104ce7dbae5dc9b30168326e45
processed_at: '2026-08-12T19:41:23-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# UniPixel 论文的人话版本

好,刚才讲得太硬核了,咱们坐下来当朋友聊聊这篇 paper 到底想干嘛。

---

## 一句话总结

**现在 LMM 都只会"看整张图说话",不会"指着某个物体说话"。UniPixel 想让模型既能听懂"你手指的那个东西",又能"把那个东西抠出来",还能"基于抠出来的东西回答问题"。**

就这么简单。

---

## 为什么要搞这个?

你想想 ChatGPT-4o、Gemini 这些大模型,你问它"图里有什么",它能答。但你说"图里左边那个人手里拿的杯子是什么颜色",它就懵了。因为它根本不知道"左边那个人"是哪个像素。

之前有人尝试解决这个问题,比如 LISA。LISA 的做法是:模型输出一个特殊的 `<SEG>` token,触发 SAM 去抠图。但 LISA 有个很僵硬的地方——它只能在回答末尾吐一个 `<SEG>`,模板写死了("It's `<SEG>`")。你没法说"把图里的猫和狗都抠出来,然后告诉我它们在干嘛"。

更麻烦的是,你想让模型"先抠出物体再推理"这个动作,LISA 做不到。因为它的 `<SEG>` token 在 causal attention 下,只能看到前面的文字,看不到完整 context,所以抠出来的 mask 质量也不行。

UniPixel 想解决的就是:**让你随便指(点、框、mask 都行),模型自己抠图,然后把抠出来的东西"记住",再基于记住的东西来回答你的问题。**

---

## 核心创新:Object Memory Bank

这个就是整篇 paper 的灵魂。

想象你跟人聊天:

> 你:"图里左边那个人[1]和右边那个人[2],谁更高?"
> 朋友:(先找到这两个人,心里记住了他们的样子)"[1]更高,大概 1 米 8。"

Object Memory Bank 就是模拟这个过程:

**第一步:Pre-filling(记忆填充)**

模型看到你说的 `[1] <REF>` 和 `[2] <REF>`,先不急着回答。它先说:"让我看看 [1] 和 [2] 在哪",然后吐出 `[1] <SEG> [2] <SEG>`,触发 mask decoder 把这两个人的 mask 抠出来,存到一个 hashmap 里。

```
memory_bank = {
    1: [mask_frame1, mask_frame2, ..., mask_frameN],
    2: [mask_frame1, mask_frame2, ..., mask_frameN]
}
```

**第二步:Injection(记忆注入)**

模型把存好的 mask,通过 masked pooling 提取特征(就是把 mask 区域内的 visual feature 平均一下),变成几个 token,塞回 prompt 里。

原来的 prompt:
```
How does [1] differ from [2]?
```

注入后变成:
```
Here is a video. The highlighted regions:
[1]: <MEM_frame1> <MEM_frame2> ... <MEM_frameN>
[2]: <MEM_frame1> <MEM_frame2> ... <MEM_frameN>
How does [1] differ from [2]?
```

现在 LLM 推理时,每个 `<MEM>` token 里都装着对应 object 的 visual feature。它不再是"盲猜",而是"看着物体说话"。

---

## 为什么这个设计很妙?

你可能会问:为什么不直接把 `<SEG>` token 接在 `<REF>` 后面,抠完图直接把 mask feature 接上去?

论文做了 ablation(Table 11b),我给你翻译成人话:

| 方案 | J&F (mask 质量) | Acc (回答准确率) |
|---|---|---|
| 只用 `<REF>` token | 46.8 | 64.5 |
| `<REF>` + `<SEG>` 拼一起 | 47.8 | 64.9 |
| `<REF>` + `<SEG>` + pooling 注入 | 47.5 | 66.3 |
| Object Memory Bank(完整方案) | **49.0** | **68.5** |

看出门道了吗?

- 方案 2(mask 接在后面):mask 质量还行(47.8),但回答准确率才 64.9。因为 `<SEG>` token 在 causal attention 下看不到完整 context,抠图质量受限。
- 方案 3(加 pooling):回答准确率上来了(66.3),但 mask 质量反而掉了(47.5)。因为 pooling 注入干扰了 mask 预测。
- 方案 4(Memory Bank):两个都上去了(49.0 / 68.5)。

**关键在于"解耦"。** Pre-filling 阶段专心抠图,Injection 阶段专心推理。两件事不互相打架,各做各的,最后通过 memory bank 汇合。

这个思路很像编程里的"先算好存起来,用的时候再取"。不要 lazy evaluation,要 eager evaluation + cache。

---

## 架构三大件

### 1. Prompt Encoder - 你怎么"指"

你指物体有三种方式:point、box、mask。

**Point/Box**: 用 Fourier embedding 编码坐标。

啥是 Fourier embedding?简单说,坐标 $(x, y)$ 是个低维数字,直接喂给神经网络学不出高频细节。所以用一堆 sin/cos 把它"展开"成高维向量:

$$\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \ldots]$$

其中 $p$ 是归一化坐标 ∈ [0,1]。

Box 有两个角点,各编一个,concat 起来再过个 Linear 压回去。

**Temporal**: 关键!视频里你得说"第几帧的哪个位置",所以 frame index $t$ 也用 1D Fourier embedding 编码,跟 spatial embedding 拼一起。

Ablation 显示,去掉 temporal 编码,PixelQA 从 49.0 掉到 44.3。5 个点的差距,你说重不重要?

**Mask**: 直接 resize binary mask,在 visual encoder 输出上做 masked pooling:

$$F_{obj} = \frac{\sum_{i,j} m_{ij} \cdot F_{ij}}{\sum_{i,j} m_{ij}}$$

$m_{ij}$ 是 mask 值(0 或 1),$F_{ij}$ 是对应位置的 visual feature。就是把 mask 区域内的 feature 加权平均(权重就是 mask 值)。

### 2. Object Memory Bank - 上面讲过了

### 3. Mask Decoder - SAM 2.1

这里有个小细节:LLM 的 hidden state 维度很高(Qwen2.5-VL-3B 是 2048),SAM 2 的 prompt embedding 只有 256 维。直接降维会丢信息。

所以作者把 `<SEG>` token 的 hidden state reshape 成 **2 个 tokens** 再降维:

| Tokens 数 | ReVOS J&F | MeViS J&F |
|---|---|---|
| 1 | 61.6 | 59.2 |
| **2** | **62.1** | **59.7** |
| 4 | 61.9 | 59.9 |
| 8 | 61.8 | 59.6 |

1 个不够,2 个刚好,4 个以上没必要。信息瓶颈不在 token 数,而在 SAM 2 的处理能力。

---

## 训练策略:三段式

为什么不能直接一起训?因为三个模块(prompt encoder、LLM、mask decoder)表征空间不对齐,一起训会学崩。

| Stage 1 | Stage 2 | Stage 3 | ReVOS J&F |
|---|---|---|---|
| ✗ | ✗ | ✓ | 61.0 |
| ✓ | ✗ | ✓ | 61.2 |
| ✗ | ✓ | ✓ | 61.6 |
| ✓ | ✓ | ✓ | **62.1** |

- **Stage 1**: 训 prompt encoder。用 851K regional captioning 数据,让模型学会"你指哪,我看哪,然后描述一下"。
- **Stage 2**: 训 L→M projector。用 87K referring segmentation 数据,让 LLM 的 `<SEG>` hidden state 和 SAM 2 的 prompt space 对齐。
- **Stage 3**: 全部解冻(用 LoRA),用 2M 数据联合训。

有个有趣的发现:M→L projector(把 mask feature 投到 LLM space)直接复用 Qwen2.5-VL 原本的 V→L projector 权重,效果比额外 pre-training 还好。

这说明啥?V→L projector 学的是"怎么把 ViT feature 转成 LLM 能懂的 token",这个能力是通用的,不管输入是整张图的 feature 还是 mask 内的 feature,转换逻辑是一样的。

---

## 实验结果:数据说话

### ReVOS(需要 world knowledge 推理的视频分割)

| Method | Size | J&F |
|---|---|---|
| VISA | 13B | 50.9 |
| TrackGPT | 13B | 45.0 |
| GLUS | 7B | 54.9 |
| Sa2VA | 4B | 53.2 |
| **UniPixel** | **3B** | **62.1** |
| **UniPixel** | **7B** | **63.7** |

3B 模型打爆 13B,差 11 个点。Reasoning subset 更夸张:59.6 vs 44.3,差 15 个点。

为什么?因为 memory injection 让模型推理时能"看到"object feature,而不是纯靠文字 context 脑补。作者自己说:"UniPixel can be regarded as an object-centric test-time scaling approach"。

### Ref-SAV(长视频、大运动、严重遮挡)

| Method | Size | FT | J&F |
|---|---|---|---|
| Sa2VA | 8B | ✗ | 41.3 |
| Sa2VA | 8B | ✓ | 50.0 |
| **UniPixel** | **3B** | **✗** | **67.2** |

Sa2VA fine-tune 完才 50,UniPixel zero-shot 就 67.2。差 17 个点。

这说明 SAM 2 的 propagation 机制太强了。Ref-SAV 有 heavy occlusion、large camera motion,SAM 2 的 memory mechanism 天生适合处理这种场景。

### PixelQA(新任务)

这个任务要求:(1) 根据 point/box prompt 找到 object (2) 抠出 mask (3) 回答问题。

别的模型根本做不了。InternVL2-26B 和 Qwen2-VL-72B 只能用 set-of-mark(在 frame 上标数字),没法抠 mask。

| Method | Size | J&F | Acc |
|---|---|---|---|
| UniPixel-3B (point) | 3B | 57.2 | 70.8 |
| UniPixel-7B (point) | 7B | 42.3 | 71.4 |
| InternVL2 | 26B | - | 60.9 |
| Qwen2-VL | 72B | - | 69.0 |

UniPixel-3B 的 Acc 70.8 已经超过 Qwen2-VL-72B 的 69.0。

注意 7B 的 J&F 比 3B 低(42.3 vs 57.2),但 Acc 更高(71.4 vs 70.8)。这个 trade-off 很有意思:大模型更会推理,但 mask 质量反而差。可能因为 7B 模型把更多容量用在语言理解上,mask 预测的精度被稀释了。

---

## Ablation 里藏着的故事

### Task Unification 的 mutual reinforcement

| Refer | Segment | Memory | J&F | Acc |
|---|---|---|---|---|
| ✓ | ✓ | ✗ | 47.5 | 64.6 |
| ✓ | ✗ | ✓ | 48.2 | 67.4 |
| ✓ | ✓ | ✓ | **49.0** | **68.5** |

- 加 segmentation 数据:J&F +0.8(mask decoder 训得更充分)
- 加 memory 数据:Acc +2.9(reasoning 能力增强)
- 三个一起:两个都涨

这就是"mutual reinforcement"。referring 和 segmentation 不是互相抢资源,而是互相帮助。为啥?因为 segmentation 数据让模型更会抠图,抠出的图质量高,memory injection 注入的特征就更好,reasoning 自然更准。

### 为什么要 SAM 2 的 propagation?

| Mask Decoder 策略 | J&F | Acc |
|---|---|---|
| Independent(每帧独立) | 46.1 | 66.2 |
| **Propagation(SAM 2 原生)** | **49.0** | **68.5** |

差 3 个点。LLM 吐出的 `<SEG>` token 是基于整个视频 context 的,但它没法 capture 每一帧的 object 细节。所以把 tracking 交给 SAM 2 的 memory mechanism,让专业的做专业的。

---

## 我的一些直觉

### 1. 这本质是 Object-Centric RAG

你想想 RAG 是啥:先检索相关文档,再基于检索内容生成回答。

UniPixel 干的事:先 segment 相关 object,再基于 object feature 生成回答。

**Segmentation 就是视觉版的 retrieval**。你把 object 从视频里"检索"出来,inject 到 LLM 的 context 里,然后推理。

这个类比让整个架构变得很直观。Object memory bank 就是 vector database,pre-filling 就是 indexing,injection 就是 retrieval + augmentation。

### 2. 为什么不用 end-to-end 直接学?

因为 end-to-end 太难了。你要让 LLM 同时学:理解语言、理解图像、理解视觉 prompt、抠图、tracking、reasoning。这么多任务混在一起,梯度会互相打架。

UniPixel 的做法是:**模块化,各司其职**。

- Prompt encoder:学会"你指哪"
- LLM:学会"理解 + 推理"
- Mask decoder(SAM 2):学会"抠图 + tracking"
- Object memory bank:把三者连起来

每个模块用专门的数据训,最后通过 memory bank 桥接。这就是为什么 3B 模型能打爆 13B。

### 3. 为什么 temporal encoding 这么重要?

视频跟图像最大的区别就是有时间维度。你指"第 3 帧的 (100, 200) 位置",模型必须知道这是第 3 帧,不是第 1 帧。

没 temporal encoding,模型会把所有帧的 (100, 200) 位置混为一谈。Ablation 里去掉 temporal encoding,PixelQA 从 49.0 掉到 44.3,5 个点没了。

### 4. Qwen2.5-VL 的选择

为什么不用 LLaVA?因为 Qwen2.5-VL 有 **dynamic resolution**。不同 object 大小不一,固定 resolution 会丢信息。Qwen2.5-VL 还原生支持 **M-RoPE(Multimodal RoPE)**,时间维度编码是 built-in 的,跟 UniPixel 的 temporal prompt encoding 哲学一致。

### 5. PixelQA 任务的 7B vs 3B 反直觉

7B 模型的 mask 质量比 3B 差(J&F 42.3 vs 57.2),但回答准确率更高(Acc 71.4 vs 70.8)。

这说明大模型更倾向于"靠语言推理"而非"靠 mask 定位"。它学会了推理,但 mask 预测的"硬技能"反而退化了。

这个 trade-off 值得深思。如果要让 7B 在 mask 质量上也提升,可能需要更多 segmentation 数据,或者调整 loss 权重。

---

## 总结一句话

**UniPixel 用 object memory bank 把"指物体"和"抠物体"两件事解耦,让它们各自训练、各自变强,然后在推理时通过 memory injection 把抠出来的 object feature 塞回 LLM,实现真正的 pixel-level reasoning。3B 模型打爆 13B,靠的是架构设计,不是暴力 scale。**

---

相关 links:
- UniPixel: https://polyu-chenlab.github.io/unipixel/
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SAM 2: https://arxiv.org/abs/2408.00714
- LISA: https://arxiv.org/abs/2308.00647
- VISA: https://arxiv.org/abs/2407.02502
- Sa2VA: https://arxiv.org/abs/2501.04001
- VideoRefer: https://arxiv.org/abs/2501.00599
- GLUS: https://arxiv.org/abs/2504.07962
- ReVOS: https://arxiv.org/abs/2407.02502
- MeViS: https://arxiv.org/abs/2308.08444
- Fourier Features: https://arxiv.org/abs/2006.10739
- LoRA: https://arxiv.org/abs/2106.09685

---

# UniPixel 论文深度解析

这篇论文由 Hong Kong Polytechnic University 和 Tencent ARC Lab 合作完成,提出了一个统一 object referring 和 segmentation 的大型多模态模型,用于 pixel-level visual reasoning。作为 Andrej,你可能最关心的是 architecture 设计的 intuition 和 ablation 数据背后的故事,我会重点讲这些。

**项目主页**: https://polyu-chenlab.github.io/unipixel/
**Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
**SAM 2**: https://arxiv.org/abs/2408.00714
**LISA (前作)**: https://arxiv.org/abs/2308.00647
**VideoRefer**: https://arxiv.org/abs/2501.00599

---

## 1. 核心问题与动机

现有 LMM 在 pixel-level 理解上存在两个 fundamental 限制:

**Limitation 1 - Interaction 粗粒度化**: 用户只能通过 text 交互,无法用 point/box/mask 这种直观方式指定区域。早期工作如 Shikra, MiniGPT-v2 仅支持 box-level grounding。

**Limitation 2 - Reasoning 仍停留在 holistic level**: LMM 内部直接处理整个 image embedding,无法针对特定 object 进行 explicit reasoning,这导致 fine-grained 理解困难。

之前的方案如 LISA、VISA 采取的范式是"输出一个 `<SEG>` token 触发 SAM decoder",这样有两个问题:(1) input/output 模板僵硬(如 "It's `<SEG>`"),无法灵活组合 referring 和 segmentation;(2) 区域理解能力和 mask 预测能力耦合在一条 forward pass 里,互相干扰。

UniPixel 的核心洞察是:**通过 object memory bank 解耦 referring(指代)、segmentation(分割)、reasoning(推理)三个能力,让它们各自受益于专门的数据,最终通过 memory injection 实现协同。**

---

## 2. Architecture 详解

整体架构基于 Qwen2.5-VL(3B/7B)+ SAM 2.1 (Hiera Base+),引入三个新组件:

### 2.1 Prompt Encoder - Joint Positional & Temporal Encoding

这是处理用户输入 visual prompt 的模块。关键设计是**将 spatial embedding 扩展到 temporal domain**。

**Point prompt**: 表示为 $(x, y, t)$
**Box prompt**: 表示为 $(x_1, y_1, x_2, y_2, t)$

对于每个 spatial coordinate $(x_i, y_i)$,使用 **2D Fourier embedding**:
$$\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]$$

这里 $p$ 是归一化坐标 $\in [0,1]$, $L$ 是 Fourier 特征的阶数(超参,控制高频学习能力)。来自 Tancik et al. NeurIPS 2020 的工作,Fourier features 能让 MLP 学习高频函数,这里用于编码像素坐标。

每个位置再加上一个 **learnable type embedding** $e_{type} \in \{point, top\text{-}left, bottom\text{-}right\}$,所以每个 corner 的 embedding 为:
$$e_{corner}(x_i, y_i) = \gamma(x_i) \oplus \gamma(y_i) + e_{type}$$

Box prompt 通过 concatenation + linear projection 合并两个 corner:
$$e_{box} = W_{merge} \cdot [e_{corner}(x_1, y_1) \oplus e_{corner}(x_2, y_2)] + b_{merge}$$

Temporal index $t$ 用 **1D Fourier embedding** 类似编码,然后与 spatial embedding concatenation,再过 GELU → Linear 投影到 LLM embedding space(对 Qwen2.5-VL 是 hidden_size=2048 或 3584)。

**Mask prompt**(dense): 直接 resize binary mask 到 visual encoder 输出 resolution,然后做 **masked pooling**:
$$F_{obj} = \frac{\sum_{i,j} m_{ij} \cdot F_{ij}}{\sum_{i,j} m_{ij}}$$
其中 $F_{ij}$ 是 visual encoder 输出的第 $(i,j)$ 个 patch feature,$m_{ij} \in \{0,1\}$ 是 mask 值。

这个 masked-pooled feature 通过 M→L projector(Linear → GELU → Linear)映射到 LLM embedding space。

**关键差异(相比 SAM)**: (1) 增加了 temporal 编码,这是视频场景必需的;(2) 移除了 negative points(因为 LMM 场景下,用户用 ID 区分不同对象,不需要 negative prompt 来 disambiguate)。

### 2.2 Object Memory Bank - 这篇论文的核心创新

这是一个 **hashmap**:`{object_id: spatial-temporal mask}`,在每次对话开始时初始化为空,根据上下文动态更新。

设计两个操作:

**Memory Pre-filling**: 当 input prompt 中出现 `<REF>` token 时触发。模型自己分析需要关注哪些 object,在 response 中输出 object IDs 和 `<SEG>` token,然后用 mask decoder 预测对应 masks,存入 memory bank。

例如 prompt: "How does behavior of [1] `<REF>` differ from [2] `<REF>` and [3] `<REF>`?"

模型 pre-filling response:
"The relevant regions for this question are [1] `<SEG>` [2] `<SEG>` [3] `<SEG>` [4] `<SEG>`."

然后 4 个 object 的 masks 都存入 memory。

**Memory Injection**: 把存好的 object masks 通过 masked pooling 提取特征,每个 frame-level mask 压缩成一个 token,通过 M→L projector 替换 prompt 中的 `<MEM>` token。

Memory-injected prompt 形式:
```
Here is a video with 4 frames denoted as <1> to <4>. The highlighted regions are as follows:
[1]: <1> <MEM> <2> <MEM> <3> <MEM>   ← object 1 在 frame 4 看不到
[2]: <1> <MEM> <2> <MEM> <3> <MEM> <4> <MEM>
...
How does behavior of [1] differ from [2] and [3]?
```

**为什么不直接用 `<REF><SEG>` 拼接?** 这是关键的设计选择。论文给了两个理由:

**理由 A - Causal attention 限制**: `<SEG>` token 出现在 `<REF>` 之后,在 causal self-attention 下,`<SEG>` token 只能看到前面的 context,无法聚合完整的 prompt 语义,导致 mask 预测质量差。

**理由 B - 解耦训练数据**: Memory bank 让 referring 数据(如 RefCOCO)和 segmentation 数据(如 SAM 自动标注)可以独立训练,互不干扰,而在 inference 时通过 memory bank 桥接,实现"mutual reinforcement"。

这一点从 Table 11(b) 的 ablation 得到验证:
- ① 仅用 `<REF>` token: 46.8 J&F / 64.5 Acc
- ② `<REF><SEG>` 拼接: 47.8 J&F / 64.9 Acc (mask 质量受影响)
- ③ `<REF><SEG>` + pooling 注入: 47.5 J&F / 66.3 Acc (regional 理解提升)
- ④ Object Memory Bank(完整方案): 49.0 J&F / 68.5 Acc (最优)

可以看到方案 ② 比 ③ 的 J&F 更高但 Acc 更低,说明 segmentation token 直接拼接损害了 mask 质量,而 pooling 只解决 regional 理解;只有方案 ④ 通过 decoupling 让两者都得到优化。

### 2.3 Mask Decoder

采用 SAM 2.1 with Hiera Base+ backbone,固定 resolution 768×768。

**关键设计**: 对于每个 `<SEG>` token,提取其最后一层 hidden state,通过 **L→M projector**(Linear → GELU → Linear)降维到 SAM 2 的 prompt embedding space(256-dim),然后 **reshape 成 2 个 tokens** 而非 1 个。

为什么 2 个 tokens?从 Table 15 的 ablation:
- 1 token: 61.6 J&F (ReVOS), 59.2 (MeViS)
- 2 tokens: 62.1 J&F, 59.7 (最优)
- 4 tokens: 61.9 J&F, 59.9
- 8 tokens: 61.8 J&F, 59.6

LLM 的 hidden_size(2048 for 3B)和 SAM 2 的 prompt embedding dim(256)差距巨大,单 token 降维会丢失 object 信息;但增加到 4 或 8 个 token 之后边际收益递减,因为 SAM 2 的 prompt encoder 处理太多 token 反而会被稀释。

**Mask 预测流程**: SAM 2 decoder 先在第一帧预测 mask,然后用 memory mechanism propagate 到其他 frames。这就是 SAM 2 的 video segmentation 能力,UniPixel 直接复用。

Table 11(c) 最后一行 ablation 对比了 "Independent"(每个 frame 独立处理)vs "Propagation":
- Independent: 46.1 J&F / 66.2 Acc
- Propagation: 49.0 J&F / 68.5 Acc

差 3 个点,说明 `<SEG>` token 单独无法 capture 视频中所有 frames 的 object 信息,通过 SAM 2 的 propagation 机制 disentangle segmentation 和 tracking 能力是合理的。

### 2.4 训练 Loss

总 loss:
$$\mathcal{L} = \mathcal{L}_{LM} + 100 \cdot \mathcal{L}_{focal} + 5 \cdot \mathcal{L}_{dice} + 5 \cdot \mathcal{L}_{MAE}^{IoU} + 5 \cdot \mathcal{L}_{CE}^{obj}$$

- $\mathcal{L}_{LM}$: 标准 autoregressive cross-entropy
- $\mathcal{L}_{focal}$: focal loss,处理类别不平衡(前景 vs 背景 pixel)
- $\mathcal{L}_{dice}$: Dice loss = $\frac{2|P \cap G|}{|P| + |G|}$,处理前景背景比例失衡
- $\mathcal{L}_{MAE}^{IoU}$: 预测 IoU score 与真实 IoU 的 MAE,SAM 2 自带
- $\mathcal{L}_{CE}^{obj}$: objectness(是否该位置有 object)的 binary cross-entropy

100 倍的 focal loss 权重表明 mask 预测任务需要大幅放大才能与 LM loss 平衡,这点和 LISA 中的设计类似。

### 2.5 Three-Stage Training Recipe

从 Table 14 的 ablation 可以看出多阶段训练的必要性:

| Stage 1 (Prompt Encoder) | Stage 2 (L→M Projector) | Stage 3 (Joint) | ReVOS J&F | VideoRefer-Bench Q (Multi-Frame) |
|---|---|---|---|---|
| ✗ | ✗ | ✓ | 61.0 | 71.5 |
| ✓ | ✗ | ✓ | 61.2 | 72.3 |
| ✗ | ✓ | ✓ | 61.6 | 71.6 |
| ✓ | ✓ | ✓ | 62.1 | 72.8 |

**Stage 1**: 851K regional captioning 数据(Inst-IT + VideoRefer short caption),只训练 sparse prompt encoder。每个 sample 随机选择 ground truth mask 内的 point(50%)或 augmented box(50%)作为 prompt。目标:让 prompt encoder 学会理解 visual prompt 并生成合理的 region caption。

**Stage 2**: 87K referring segmentation 数据(RefCOCO/+/g + Ref-YouTube-VOS),只训练 L→M projector。目标:对齐 LLM 的 `<SEG>` hidden state 和 SAM 2 的 prompt embedding space。

**Stage 3**: 联合训练,使用 UniPixel-SFT-1M 数据 + LLaVA-1.5-Mix-665K + VideoGPT+ Instruct,总共约 2M samples。应用 LoRA(rank=128, alpha=256)到 visual encoder 和 LLM 的 QKVO layers。Learning rate: 5e-6 for mask decoder, 2e-5 for 其他参数。Global batch size=32,8x A6000 48G GPU。

**为什么不直接 stage 3?** 因为 prompt encoder, LLM, mask decoder 三个模块的表征空间未对齐,joint training from scratch 会陷入 sub-optimal 解。Stage 1 和 2 的 pre-alignment 大幅减轻了 stage 3 的学习负担。

Table 16 还有一个有趣的发现:M→L projector 直接复用 Qwen2.5-VL 的 V→L projector 权重(因为 object features 和 visual features 来自同一个 visual encoder,表征空间一致),**比额外 pre-training 效果更好**! 这暗示 V→L projector 学到的是通用的"如何把 ViT features 投影到 LLM space",而不是 image-specific 的能力。

---

## 3. 任务和实验数据深度解读

### 3.1 Reasoning Video Object Segmentation (ReVOS)

ReVOS 是最 challenging 的 benchmark,要求模型基于 implicit text query(需要 world knowledge推理)预测 mask。

**Table 1 关键数据**:
- UniPixel-3B: 62.1 J&F (59.7 J / 64.4 F)
- UniPixel-7B: 63.7 J&F (61.7 J / 65.7 F)
- VISA-13B: 50.9 J&F
- TrackGPT-13B: 45.0 J&F
- Sa2VA-4B: 53.2 J&F
- GLUS-7B: 54.9 J&F

**3B 模型超过 13B 的 VISA 11.2 个点**,这是 dramatic 的差距。更值得注意的是 Reasoning subset(需要 world knowledge): UniPixel-3B 59.6 vs VISA-13B 44.3,差 15.3 个点。

**为什么?** 因为 UniPixel 的 memory injection 机制让 LLM 在 reasoning 时能够"看到"object 的 features(类似 test-time scaling 中的 retrieval augmentation),而不是仅靠 text context 推理。这点在 paper Section 2 提到:"UniPixel can also be regarded as an object-centric test-time scaling approach"。

### 3.2 Referring Video Object Segmentation (Table 2)

**MeViS (val)**: UniPixel-3B 53.1 J&F vs GLUS-7B 51.3 (+1.8) vs VideoGLaMM-3.8B 45.2 (+7.9)

MeViS 强调 motion-based referring,所以 temporal 编码很关键。Table 11(c) 第一行的 ablation 显示,**移除 temporal encoding**后 PixelQA 性能从 49.0 掉到 44.3 J&F,验证了 temporal encoding 的重要性。

**Ref-YouTube-VOS**: UniPixel-3B 70.5 J&F vs VideoLISA-3.8B 63.7 (+6.8)
**Ref-DAVIS17**: UniPixel-3B 74.2 J&F vs VideoLISA 68.8 (+5.4)
**Ref-SAV (long video, large motion)**: UniPixel-3B **zero-shot 67.2 J&F**,超过 Sa2VA-8B fine-tuned 50.0 (+17.2!)

Ref-SAV 的差距非常夸张,说明 SAM 2 的 propagation 机制在 long video 场景下具有明显优势(因为 Ref-SAV 设计了 heavy occlusion, large camera motion)。

### 3.3 Image Referring Expression Segmentation (Table 3 & 6)

Table 3 是 co-trained 模型直接评估:
- RefCOCO val: UniPixel-3B 80.5 / 7B 80.8
- RefCOCO+ val: 74.3 / 75.3
- RefCOCOg val(U): 76.3 / 76.4
- ReasonSeg gIoU: 64.0 / 60.5; cIoU: 56.2 / 58.7

注意 ReasonSeg 数据集只有 239 个样本,在 2M 数据中被淹没,所以 co-trained 模型在 ReasonSeg 上表现并不最优(7B 反而比 3B 差,可能因为 7B 更容易 over-fit 到大量 RES 数据)。

Table 6 是 fine-tuned 版本(标准做法,在 RefCOCO/+/g 上 joint fine-tune):
- UniPixel-7B: RefCOCO val 83.0, testA 84.9, testB 80.4
- 超过 GLaMM-7B (79.5 / 83.2 / 76.9) 和 Sa2VA-4B (80.4)

### 3.4 Referring Expression Comprehension (Table 7)

REC 任务通过从 mask 提取 bounding box 实现(取 mask 的最小外接矩形),评估 IoU≥0.5 的 accuracy:
- UniPixel-7B: RefCOCO val 92.0, testA 94.4, testB 88.1
- 超过 Vitron-7B (90.9 / 93.2 / 89.3) 和 MiniGPT-v2-7B (88.7 / 91.6 / 85.3)

这个结果有意思:虽然 UniPixel 的主任务是 mask prediction,但 mask 质量足够高,直接转 box 也能达到 SOTA,说明 mask 预测的 localization 精度很高。

### 3.5 Referred Video Description & QA (Table 8 & 9)

**VideoRefer-Bench-D** (Description, 5 个维度: SC=Scene, AD=Action, TD=Temporal, HD=Hallucination, Avg):
UniPixel-3B Multi-Frame: SC 4.08, AD 3.13, TD 3.13, HD 3.42, Avg 3.44
对比 VideoRefer-7B: SC 4.44, AD 3.27, TD 3.10, HD 3.04, Avg 3.46

UniPixel 在 HD(Hallucination Detection)上明显更好(3.42 vs 3.04),因为 memory injection 让模型"看清"了 object,减少 hallucination。

**VideoRefer-Bench-Q** (QA, 5 类问题: BQ=Basic, SQ=Sequential, RQ=Relational, CQ=Causal, FP=Future Prediction):
UniPixel-3B Multi-Frame: 75.3 / 70.7 / 62.3 / 87.4 / 77.2, Avg 72.8
VideoRefer-7B Multi-Frame: 75.4 / 70.6 / 60.5 / 89.4 / 78.1, Avg 72.1

UniPixel 在 RQ(Relational)上有 1.8 点优势,因为 multi-object memory 可以同时存多个 object,有助于 reasoning 它们之间关系。

### 3.6 PixelQA - 新任务

这是论文的 novelty,基于 VideoRefer-Bench-Q 修改:把 mask prompt 替换成更 ambiguous 的 point 或 box prompt,要求模型 (1) 识别目标 object (2) 预测 mask (3) 回答 QA。

**Table 10 结果**:
- Baseline: InternVL2-26B 和 Qwen2-VL-72B 用 set-of-mark prompting(直接在 frame 上画数字标记)
- UniPixel-3B (point): 57.2 J&F / 70.8 Acc
- UniPixel-7B (point): 42.3 J&F / 71.4 Acc
- InternVL2-26B: 60.9 Acc (no mask)
- Qwen2-VL-72B: 69.0 Acc (no mask)

注意 7B 模型的 J&F 反而比 3B 低!这是因为 7B 模型 over-fit 到 mask quality 而非 reasoning,但 Acc 更高。这是个有趣的 trade-off。

混合 prompt(50% point + 50% box): UniPixel-7B 44.9 J&F / 71.4 Acc,显示 UniPixel 能灵活处理多种 prompt。

---

## 4. Ablation 深度分析

### 4.1 Task Unification (Table 11(a))

| Refer | Segment | Memory | J&F | Acc |
|---|---|---|---|---|
| ✓ | ✓ | ✗ | 47.5 | 64.6 |
| ✓ | ✗ | ✓ | 48.2 | 67.4 |
| ✓ | ✓ | ✓ | 49.0 | 68.5 |

- 第一行:仅 referring + segmentation(无 memory)
- 第二行:仅 referring + memory(无 segmentation 数据)
- 第三行:三者齐全

**关键发现**: segmentation 数据让 J&F 提升 0.8 个点(因为 mask decoder 训练更充分);memory 数据让 Acc 提升 2.9 个点(因为 reasoning 能力增强)。这是 **mutual reinforcement** 的直接证据。

### 4.2 Data Combination (Table 17)

| Regional | Segmentation | Memory | General | ReVOS J&F | VideoRefer-BenchQ (MF) |
|---|---|---|---|---|---|
| ✓ | | | | - | 72.1 |
| | ✓ | | | 61.4 | - |
| ✓ | ✓ | | | 61.5 | 72.6 |
| ✓ | ✓ | ✓ | | 62.1 | 72.5 |
| ✓ | ✓ | ✓ | ✓ | 62.1 | 72.8 |

加 General 数据(如 LLaVA-Mix-665K)对 pixel-level tasks 几乎无影响(62.1→62.1),但能保住 general video QA 能力(Table 13 显示 MVBench Avg 64.3,与专门的 video LMM 相当)。这是个 trade-off,但作者选择了多任务并存。

### 4.3 Number of Hidden Tokens (Table 15)

已经在 Section 2.3 分析过,2 tokens 是 sweet spot。这点很有 implementation 价值,因为增加 tokens 会增加 SAM 2 的 prompt encoder 计算量。

---

## 5. 与相关工作的关系

### 5.1 LISA (CVPR 2024) - https://arxiv.org/abs/2308.00647

LISA 是第一个将 LMM 用于 reasoning segmentation 的工作,核心是引入 `<SEG>` token 触发 SAM decoder。UniPixel 解决了 LISA 的两个问题:(1) 僵硬的 "It's `<SEG>`" 模板;(2) 无法同时支持 referring 和 segmentation。

### 5.2 Sa2VA (arXiv 2501) - https://arxiv.org/abs/2501.04001

Sa2VA 也是 SAM2 + LLaVA 的融合,但通过 iterative prompt-based interaction 实现。UniPixel 通过 object memory bank 实现 decoupling,在 Ref-SAV 上 zero-shot 67.2 vs Sa2VA fine-tuned 50.0,差距 17.2 个点。

### 5.3 VideoRefer (arXiv 2501) - https://arxiv.org/abs/2501.00599

VideoRefer 是 video region understanding 的工作,使用 mask prompt。UniPixel 在 VideoRefer-Bench-Q 上 multi-frame mode 达到 72.8 Avg,略超 VideoRefer-7B 的 72.1。关键是 UniPixel 还能用 point/box prompt,而 VideoRefer 只支持 mask prompt。

### 5.4 VISA (ECCV 2024) - https://arxiv.org/abs/2407.02502

VISA 专门做 reasoning video object segmentation,使用 frame sampler + LLM + SAM 的 pipeline。UniPixel 端到端集成,在 ReVOS 上 62.1 vs 50.9,差距 11.2 个点。

### 5.5 GLUS (arXiv 2504) - https://arxiv.org/abs/2504.07962

GLUS 是 global-local reasoning unified 的 video segmentation,7B 模型。UniPixel-3B 在 MeViS 上 53.1 vs GLUS-7B 51.3,3B 模型超过 7B。

---

## 6. 一些细节直觉

### 6.1 为什么 Qwen2.5-VL 而不是 LLaVA?

Qwen2.5-VL 支持 **dynamic resolution**,能处理不同尺寸的 image/video,这对 pixel-level 任务很重要(不同 object 大小需要不同 resolution)。Qwen2.5-VL 还有 native 的 video position encoding (M-RoPE),与 UniPixel 的 temporal prompt encoding 设计哲学一致。

### 6.2 为什么 SAM 2.1 而不是 SAM 1?

SAM 2 引入了 **memory bank + propagation** 机制,支持 video segmentation。SAM 1 只能处理单帧。UniPixel 复用 SAM 2 的 propagation 能力,避免了从零训练 video tracking 的成本。这也是 Ref-SAV (long video) 上 UniPixel 大幅领先的原因。

### 6.3 为什么用 LoRA 而不是 full fine-tune?

LoRA rank=128, alpha=256 应用在 visual encoder 和 LLM 的 QKVO layers。这保留了 pre-trained LMM 的 general 能力(Table 13 MVBench Avg 64.3,与 VideoChat2-7B 的 60.4 相当),同时让模型学习 pixel-level 任务。Full fine-tune 可能导致 catastrophic forgetting。

### 6.4 Frame Sampling

训练时随机采样 8 frames per video,每 frame resize 到 316²~448² 像素(对应 128~256 tokens)。这种 **varying resolution** 训练策略增强模型对 different video qualities 的 robustness。

Inference 时按 benchmark 规定采样。ReVOS 通常是 12-36 frames,Ref-DAVIS17 平均 60-85 frames。

### 6.5 Mask Decoder Resolution 固定 768×768

SAM 2 的 native resolution。即便输入 frame 是 448²,mask 仍在 768² 上预测,然后 resize 回原尺寸。这是为了利用 SAM 2 的 pre-trained 能力,避免 resolution mismatch。

---

## 7. Limitations 和 Future Work

作者承认:
1. **训练数据规模**: 由于计算资源限制(8x A6000),没有 scale 到 GCG (Grounded Caption Generation) 等更多 pixel-level 任务。
2. **Mask propagation 起始帧**: 当前固定从第一帧预测并 propagate,理论上可以选 best view frame 双向 propagate。

可能的扩展方向:
- **更高 resolution 的 visual encoder**: 当前 ViT 输出 128~256 tokens per frame,精细 object 可能被压缩。可以考虑 high-resolution adapter。
- **Multi-object interaction**: 当前 memory bank 处理多个独立 object,但 object 间关系(interaction)需要更复杂的 reasoning。可以引入 graph neural network。
- **3D pixel-level understanding**: 从 2D mask 扩展到 3D point cloud 或 mesh。
- **Long video scaling**: SAM 2 的 propagation memory 在超长视频(>1000 frames)上可能 memory 不足,需要 sparse memory mechanism。

---

## 8. 总结

UniPixel 的核心贡献是一个 **elegantly simple yet effective** 的设计:**用 object memory bank 解耦 referring/segmentation/reasoning 三个能力**。这种 decoupling 让每个能力都能从专门的训练数据中受益,最终通过 memory injection 协同。

实验数据说话:3B 模型在多个 benchmark 上超过 7B-13B 的 SOTA,验证了 architecture design 的有效性。新提出的 PixelQA 任务也为社区提供了一个 challenging benchmark,要求真正的 pixel-level reasoning。

直觉上,这个工作让我想到 test-time scaling 的思想:**先 segment 再 reason**,把 object-centric features 作为额外 context 注入 LLM,提升了 reasoning 质量。这和 RAG (Retrieval Augmented Generation) 在 NLP 中的思想类似,只是 retrieval 变成了 segmentation。

希望这个分析帮助你 build 起对 UniPixel 的 intuition。如果对某个组件还想深入(比如 SAM 2 的 memory mechanism,或者 Fourier embedding 的高频学习能力),可以继续讨论。

**相关 web links**:
- UniPixel 主页: https://polyu-chenlab.github.io/unipixel/
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- SAM 2: https://arxiv.org/abs/2408.00714
- LISA: https://arxiv.org/abs/2308.00647
- VideoRefer: https://arxiv.org/abs/2501.00599
- VISA: https://arxiv.org/abs/2407.02502
- Sa2VA: https://arxiv.org/abs/2501.04001
- GLUS: https://arxiv.org/abs/2504.07962
- MeViS: https://arxiv.org/abs/2308.08444
- ReVOS: https://arxiv.org/abs/2407.02502
- GroundMoRe: https://arxiv.org/abs/2411.09921
- Fourier Features: https://arxiv.org/abs/2006.10739
- LoRA: https://arxiv.org/abs/2106.09685
