---
source_pdf: Qwen3-VL Technical Report.pdf
paper_sha256: ee075d08e67de1148d6437c6c1d481f7894183b8793905a2deb5f62664f49380
processed_at: '2026-08-06T08:03:34-07:00'
target_folder: LLM主流Model/Qwen3
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Qwen3-VL 用人话讲

Andrej，好，我把那些公式和表格扔一边，直接说人话。

---

## 这是个啥

Qwen 团队把自家最强的 text model（Qwen3）和一个重新设计的 vision stack 拼在一起，做了 6 个 model——2B、4B、8B、32B dense，30B-A3B、235B-A22B MoE。核心目标就一句话：**让模型看图、看视频、看文档，同时不要把语言能力搞砸**。

听起来简单，但前几代 Qwen-VL 都栽在"加 vision 降 text"这个坑里。这次他们花大力气解决了。

---

## 三个架构改动，每个都解决一个具体痛点

### 痛点 1：长视频位置编码坏了

Qwen2-VL 搞了个叫 MRoPE 的东西，把 embedding 维度切成三段——一段管时间，一段管横坐标，一段管纵坐标。

问题在哪？**时间那段全挤在低频区**。低频意味着"只能感知很慢的变化"，但长视频里你需要同时知道"这是 3 秒前的事"和"这是 2 小时前的事"。低频维度不够用，位置信息就糊了。

**解法**：把时间、横、纵三个维度**交错打散**到整个频谱上，每个维度都能同时占低频和高频。

用人话讲：原来你把 RGB 三个通道分条带排列，现在改成 RGB 像素级 mosaic——任何一个空间时间尺度都不会瞎。

---

### 痛点 2：视觉信息传到 LLM 时太"抽象"了

传统 VLM 的做法：ViT 最后一层输出 → 一个 MLP projection → 塞给 LLM 当输入 token。

问题：ViT 最后一层已经把图像高度抽象化了，变成"这是一只猫，坐在桌子上"这种 semantic-level 表示。但 **DocVQA、ChartQA 这些任务需要 pixel-level 精度**——你得看清表格里某个小数字在哪。最后一层 ViT feature 已经把这种细节压没了。

**解法（DeepStack）**：从 ViT 的**浅层、中层、深层**各抽一份 feature，分别 inject 到 LLM 的前三层。

浅层 ViT feature 保留了边缘、纹理、局部结构；中层偏 mid-level pattern；深层是 semantic。LLM 在做推理之前，先"摸到"了完整的视觉细节，不只是拿到一个抽象总结。

**关键**：token 数量不增加。不是把三个 ViT feature 都当 input token 拼上去，而是用 residual add 把它们加到 LLM 早期层的 hidden state 上。context length 不变，计算开销小，但 fine-grained 能力涨了——InfoVQA +2.3，DocVQA +1.6，ChartQA +1.8。

---

### 痛点 3：视频时间戳编码方式炸了

Qwen2.5-VL 搞了个 T-RoPE——让 position ID 直接对应绝对秒数。一个 2 小时视频，时间 position ID 能到 7200，远超训练见过的范围，模型直接懵。

更糟的是，要让模型学会这种映射，你得在**各种 fps 上均匀采样训练数据**——数据构造成本爆炸。

**解法**：别把时间塞进 position encoding 了，**直接写成文本**。

```
<3.0 seconds> [frame tokens] <7.5 seconds> [frame tokens] ...
```

LLM 本来就是处理文本的大师，把时间戳当文本读，比让 RoPE 去拟合绝对秒数自然得多。训练时同时用 `<3.0 seconds>` 和 `<00:00:03>` 两种格式，模型两种都能懂。

代价：context length 多了一丢丢。收益：video grounding、dense captioning 这些时间敏感任务大幅提升，而且天然能 long-extrapolate。

---

## 训练流程：四阶段 pretrain + 三阶段 post-train

### Pretrain 四阶段

**S0（Alignment）**：只训 merger，冻住 ViT 和 LLM。为什么？merger 刚随机初始化，如果直接全参数训，random visual token 会冲垮 LLM 的 text 表征。先让 merger 学会合理投影，打地基。67B tokens，8K context。

**S1（Multimodal）**：解冻全部，1T tokens，VL + text 混合。8K context。

**S2（Long-Context）**：扩到 32K，1T tokens，加更多 video + agent data。

**S3（Ultra-Long）**：推到 256K，100B tokens，专门喂长视频、长文档。

**关键 trick：square-root reweighting**。传统有两种 loss 算法：
- Per-sample：每条数据权重一样，长文档被压扁
- Per-token：长文档权重正比于 token 数，长文档主导训练
- Square-root：权重正比于 $\sqrt{N}$，折中

这保证了 text 和 VL 数据的相对贡献不被某一边的长度分布主导。

---

### Post-train 三阶段

**SFT**：1.2M samples，1/3 text + 2/3 VL。两阶段 context：先 32K 1 epoch，再 256K 1 epoch。分 thinking 和 non-thinking 两条线。

**Strong-to-Weak Distillation**：用更强的 teacher model 蒸馏。关键决策——**只用 text-only 蒸馏 LLM backbone**。用 text reasoning teacher 提升 LLM 推理能力，间接传导到 multimodal。

**RL**：分 Reasoning RL 和 General RL。
- Reasoning RL：30K queries，math/code/logic/grounding 等可验证任务，用 SAPO 算法
- General RL：多任务 reward，特别引入**反直觉任务**来纠正 SFT 学到的错误 prior——比如反直觉的物体计数、复杂时钟识别。SFT 模型可能学到"看到 3 个东西就回答 3"，RL 用反直觉样本逼它真正去数

---

## 几个我觉得特别聪明的工程决策

### Multimodal Necessity Filtering

构造 VL 训练数据时，**如果 Qwen3-30B-nothink 不看图也能答对，就丢掉这条数据**。

为什么聪明？大量所谓的"视觉"数据集，其实 LLM 看 OCR'd text 就能解，视觉信息是冗余的。这种数据训了等于白训，还可能让模型学会"忽略图像走捷径"。这步过滤保证了训练信号真的需要视觉理解。

---

### Tool-Calling Reward 防 hacking

做 "thinking with images" 时，模型会调用 zoom-in 工具看图细节。早期实验发现，模型为了 hack accuracy reward，**只调一次工具就草率回答**——因为多调工具可能引入更多错误反馈。

解法：引入 Tool-Calling Reward，**对比实际 tool call 次数和 Qwen2.5-VL-72B 估计的合理次数**。如果模型明显 call 太少，惩罚。强制它根据任务复杂度 adaptive 地探索。

这是非常 practical 的 RL reward hacking 防御——发现模型偷懒，就专门设计一个 reward 来惩罚偷懒。

---

### Text-only Distillation 反哺 Multimodal

直觉上，VL model 的 multimodal 能力上限应该由 multimodal teacher 决定。但 Qwen3-VL 用 text-only distillation 提升 LLM backbone，结果 multimodal benchmark 也涨了。

为什么？multimodal reasoning 的瓶颈往往在 **reasoning 本身**，不在 vision。把 LLM 的数学推理、code 推理搞强了，视觉数学题自然也跟着强。这跟"VLM 在 AIME 上能超过自家 text-only 同尺寸 model"的现象一致——multimodal CoT 数据相当于额外的 reasoning 数据，反哺了 text 能力。

---

## 结果怎么样

**旗舰 235B-A22B**：
- MMMU 80.6（thinking），超过 Gemini 2.5 Pro 的 68.8
- MathVista-mini 85.8，SOTA
- OCRBench 920，SOTA
- MMLongBench-Doc 57.0，SOTA
- AIME-25 74.7，**超过自家 text-only Qwen3-235B 的 70.3**

**Video**：
- Needle-in-a-Haystack：256K context 100% accuracy，YaRN 外推到 1M token（2 小时视频）99.5%
- MLVU 84.3，LVBench 76.5

**Agent**：
- OSWorld-G 68.3，AndroidWorld 63.7，在 GUI agent 里属于第一梯队

**小模型也很强**：8B 在多数 benchmark 上超过 GPT-5-nano，4B 在 DynaMath 和 VisuLogic 上甚至超过 8B。

---

## 我的几个疑问

1. **Qwen3-ViT 的 ablation 主要在自家 OmniBench 上**，公平性存疑。你说 ViT 持续训练有用，但只在自家 benchmark 上验证，有点循环论证。

2. **Interleaved MRoPE 缺单独的 long-video ablation**。paper 说它好，但没直接对比 original MRoPE 在 long-video 任务上的差异。从 Needle-in-a-Haystack 看 long-context 检索没问题，但理解任务（LVBench 等）的提升是 architecture + data 共同贡献，分不开。

3. **Benchmark 对比公平性**。Gemini 2.5 Pro video eval 用 512 帧，GPT-5 用 256 帧，Claude 100 帧，Qwen3-VL 用 2048 帧——frame 数差这么多，video benchmark 的对比不太公平。当然这是 API 限制，不是 Qwen 的锅，但读结果时要留个心眼。

4. **MoE 的推理效率**。235B 总参 22B 激活，latency 接近 22B dense，但 routing overhead 和 KV cache 仍然是 235B 量级。实际部署时 latency 不会真的等于 22B dense，paper 没给详细 latency 数据。

---

## 一句话总结

Qwen3-VL 的核心 story：**架构上修了三个具体 bug（MRoPE 频谱、ViT feature 层级、时间编码），训练上用 square-root reweighting + text-only distillation + necessity filtering 三道防线保住 text 能力，数据上花了大量功夫做质量过滤和合成，最终在多数 multimodal benchmark 上达到 SOTA，同时 text 能力不降反升**。

不是什么颠覆性创新，但每个改动都有明确的工程逻辑，组合起来效果很强。典型的"系统集成大于单点突破"。

参考链接还是之前那些，不重复贴了。想深挖哪块再问我。

---

# Qwen3-VL Technical Report 深度解读

Andrej，这份 paper 信息密度极高，从架构、训练 recipe 到 eval 都有非常具体的工程决策。我把它拆成几层来 build intuition，重点讲那些"为什么这么干"的部分。

---

## 1. 整体定位：Qwen3-VL 在 VLM 演化谱系中的位置

Qwen3-VL 是 Qwen 系列从 Qwen-VL → Qwen2-VL → Qwen2.5-VL → Qwen3-VL 的第四代。它本质上是把 Qwen3 这个 strong text backbone 和一个 reworked vision stack 重新耦合，并试图解决前几代遗留下来的几个核心痛点：

- **MRoPE 的频谱不平衡**导致 long-video 退化
- **Vision-language alignment 太浅**（只靠 merger 层 + 最后一层 ViT feature）
- **T-RoPE 把绝对时间编码进位置 ID**，长视频位置 ID 爆炸
- **Text 能力在 VL 训练中被冲掉**

关键 reference：
- Qwen2-VL (MRoPE 首发): https://arxiv.org/abs/2409.12191
- Qwen2.5-VL (T-RoPE): https://arxiv.org/abs/2502.13923
- DeepStack 原始 paper: https://arxiv.org/abs/2411.15635
- SigLIP-2: https://arxiv.org/abs/2502.14786

---

## 2. 三大架构升级——这是这篇 paper 的灵魂

### 2.1 Interleaved MRoPE：从分块频率到交错频率

**原始 MRoPE 的设计**（Qwen2-VL）：
embedding 维度 $d$ 被切成三段，分别给 $t$ (temporal), $h$ (horizontal), $w$ (vertical):

$$\text{RoPE}(\theta_i) = \begin{cases}
e^{i m_t \theta_i} & i \in [0, d/3) \\
e^{i m_h \theta_i} & i \in [d/3, 2d/3) \\
e^{i m_w \theta_i} & i \in [2d/3, d)
\end{cases}$$

其中 $\theta_i = 10000^{-2i/d}$ 是 base frequency，$m_t, m_h, m_w$ 分别是时间、横纵坐标的 position id。

问题在哪？**低频段全给了 $t$，高频段全给了 $w$**。对于长视频，$m_t$ 可以非常大（几千、几万），但低频段旋转极慢，**信息容量塌缩**；而空间维度 $h, w$ 占了高频段，短距离分辨率高但长距离模糊。

Qwen3-VL 的解法是 **interleaved**：

$$\text{维度 } i \mod 3 = \begin{cases}
0 \Rightarrow t \text{ 频率} \\
1 \Rightarrow h \text{ 频率} \\
2 \Rightarrow w \text{ 频率}
\end{cases}$$

也就是把 $\{t, h, w\}$ 周期性地散布在 $[0, d)$ 整个频谱上。直觉上：**每个轴都能同时感知低频（长距离）和高频（短距离）信息**，频谱被均匀摊开，避免了某一轴独占低频或高频。

参考的类似工作：
- 3D-RoPE / Interleaved RoPE 在 video-LLM 中的应用: Huang et al. 2025 "Revisiting Multimodal Positional Encoding"
- TimeMarker: https://arxiv.org/abs/2411.18211

**Intuition**：这就像把三色通道均匀交织成 mosaic，而不是 RGB 分条带——任何一个空间-时间尺度都不会因为维度划分而丢分辨率。

---

### 2.2 DeepStack：多层级 ViT feature 注入 LLM 多层

原始 DeepStack（Meng et al. 2024）是把**多尺度视觉输入**堆叠进 LLM 多层。Qwen3-VL 做了一个聪明变体：**不是 multi-scale，是 multi-layer**——从 ViT 的三个不同深度抽 feature，分别投影到 LLM 的前三层。

架构上：

```
ViT Layer 6  →  Merger_1  →  add to LLM hidden layer 1
ViT Layer 12 →  Merger_2  →  add to LLM hidden layer 2  
ViT Layer 24 (final) →  Merger_3  →  add to LLM hidden layer 3 (主输入)
```

这里 ViT 用的是 SigLIP-2 SO-400M，约 24-27 层 transformer block。低层 ViT feature 偏 low-level（边缘、纹理、局部结构），高层偏 semantic（物体、关系、场景）。

**为什么有效（build intuition）**：
传统 VLM 只拿 ViT 最后一层 → LLM 的输入。问题是 ViT 最后一层已经高度抽象，**fine-grained spatial 信息被压缩掉了**。DocVQA、InfoVQA、ChartQA 这些需要 pixel-level 对齐的任务就吃亏。DeepStack 把多级 feature 直接 inject 到 LLM 早期层，相当于让 LLM 在抽象推理之前先"看到"更多 raw visual cue。

Ablation 数据（Table 12，15B-A2B LLM，200B tokens）：

| | AVG | AI2D | OCRB | TVQA | InfoVQA | ChartQA | DocVQA | MMMU | MMStar | RLWDQA | MMBEN | MMBCN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 74.7 | 81.8 | 81.0 | 80.6 | 71.9 | 81.5 | 89.5 | 52.9 | 55.5 | 67.7 | 81.0 | 78.1 |
| DeepStack | **76.0** | **83.2** | **83.6** | 80.5 | **74.2** | **83.3** | **91.1** | **54.1** | **57.7** | **68.1** | **81.2** | **78.5** |

注意 InfoVQA +2.3, DocVQA +1.6, ChartQA +1.8——**document-heavy 任务受益最显著**，这验证了 fine-grained 注入的直觉。

**Context length 不增加**——这是关键。不同于 multi-scale visual token 拼接（会爆 token budget），DeepStack 是 **residual add**，token 数量不变，只是 LLM 早期层的 hidden state 被多层 ViT feature 增广。

参考：
- DeepStack 原文: https://arxiv.org/abs/2411.15635
- 类似思路还有 LLaVA-NeXT 的 multi-layer fusion, CogVLM2 的 visual expert

---

### 2.3 Text-based Timestamp：从 T-RoPE 到显式文本

Qwen2.5-VL 的方案是 **T-RoPE**：让 temporal position id 直接对应绝对时间（秒）。问题：
1. 长视频（2 小时 = 7200 秒）产生巨大 sparse position id，远超训练时见过的范围
2. 训练数据需要在各种 fps 上均匀采样才能学到 robust 时间映射，数据构造极贵

Qwen3-VL 改为 **textual timestamp tokens**：

```
<3.0 seconds> [frame group 1 tokens] <7.5 seconds> [frame group 2 tokens] ...
```

每个 frame group 前面插一段格式化文本，表达这组帧的时间戳。训练时同时用 **seconds 格式** 和 **HMS 格式**（如 `<00:03:15>`），让模型两种时间码都能读懂。

**Intuition**：时间信息从**连续的、外部的 position encoding** 变成了**离散的、文本内部的 token**。LLM 本来就是处理文本的大师，把时间塞进文本流比塞进 RoPE 频段更自然、更可解释、更容易 long-extrapolate。代价是 context length 略增，但收益是 video grounding、dense captioning 这些时间敏感任务大幅提升。

参考 Chen et al. 2024b TimeMarker: https://arxiv.org/abs/2411.18211

---

## 3. Vision Encoder：SigLIP-2 + 持续训练

- 架构：**SigLIP-2-SO-400M** 作为默认 vision encoder（大模型用），2B/4B 小模型用 SigLIP2-Large (300M)
- 关键 trick：**dynamic resolution + 2D-RoPE + absolute position embedding 插值**，来自 CoMP (Chen et al. 2025): https://arxiv.org/abs/2503.18931
- 在官方 checkpoint 基础上**继续训练**，得到 Qwen3-ViT

Ablation（Table 11）：Qwen3-ViT vs SigLIP-2，在 CLIP zero-shot 上持平，但在 OmniBench（自家 holistic benchmark）上从 36.9 → 45.5 (+8.6)，下游 VLM（同 1.7B Qwen3 backbone, 1.5T tokens）InfoVQA 从 65.3 → 67.0，Omni 从 50.1 → 53.0。**说明持续训练在通用 vision 任务没退化，但在自家 evaluation suite 显著提升**——这是个有点"自圆其说"的 ablation，需谨慎解读，但方向是合理的。

---

## 4. Pretraining：四阶段从 alignment 到 256K

| Stage | Objective | 训练参数 | Token Budget | Seq Len |
|---|---|---|---|---|
| S0 | Vision-Language Alignment | **Only Merger** | 67B | 8K |
| S1 | Multimodal Pre-Training | All (ViT+Merger+LLM) | ~1T | 8K |
| S2 | Long-Context Pre-Training | All | ~1T | 32K |
| S3 | Ultra-Long-Context Adaptation | All | 100B | 256K |

**S0 的设计直觉**：刚接上 ViT 和 LLM 时，merger 是随机初始化的，LLM 是 strong text model。如果直接全参数训练，**LLM 会被 unaligned visual token 流冲垮**——gradient 来自 random projection，会污染 LLM 的 text 表示。S0 冻结 ViT 和 LLM，只让 merger 学到一个合理的视觉→文本空间映射，相当于先打地基。

S1 是真正的大规模 co-training，1T tokens，VL + text 混合。**关键 trick：square-root reweighting**——

传统 per-sample loss：每个 sample 算一次 cross-entropy，长样本和短样本权重一样。
传统 per-token loss：长样本权重正比于 token 数，长样本主导。
**Square-root reweighting**：

$$\mathcal{L} = \frac{1}{\sqrt{N_s}} \sum_{i=1}^{N_s} \ell_i$$

其中 $N_s$ 是 sample 的 token 数，$\ell_i$ 是第 $i$ 个 token 的 loss。

直觉：长文档/视频不要完全主导（per-token 的问题），但也别完全忽略（per-sample 的问题），用 $\sqrt{N}$ 折中——介于 $O(1)$ 和 $O(N)$ 之间。这种 reweighting 在 LLM 训练里也有类似思路（如 stable preference scaling）。

S2 扩到 32K，增加 video + agent instruction 数据比例。
S3 把 context 推到 256K，100B tokens，专门训练长视频、长文档。

**总训练量约 2.2T tokens**——这在当前 VLM 里属于第一梯队。

---

## 5. 数据管线：九大 modality

这是 paper 最庞杂的部分，但每块都透露了具体的工程取舍：

### 5.1 Image Caption
用 **fine-tuned Qwen2.5-VL-32B recaptioner** 对原始 alt-text 做 rewrite，丰富 object attributes / spatial layout / context。然后**只对 recaptioned text 做去重**（保留 visual diversity），再用 visual embedding 聚类找稀疏区域做定向增广。这个 pipeline 明显借鉴了 DCLM / DataComp-LM 的思路。

### 5.2 Interleaved Text-Image
来自中英文 web 文档，先用 **fine-tuned domain classifier** 过滤广告、推广、clickbait。book-scale 数据用 Qwen2.5-VL-7B 做 multimodal parsing，把文本和 figure 精确对齐。超长 context 子集把连续页拼成 256K token 序列，要求最低页数 + 最低图文比。

### 5.3 Knowledge
30+ semantic categories（animals, plants, landmarks, food, vehicles...）。**Importance-based sampling**：高频实体多采，低频少采但不丢弃——长尾分布的标准操作。

### 5.4 OCR / Document Parsing / Long Document
- OCR：30M 内部样本 + 29 种额外语言（Qwen2.5-VL 只支持 10 种），共 39 种语言
- Document parsing：3M PDF from Common Crawl × 10 类 + 4M internal；用 in-house layout model 预测 reading order 和 bbox，Qwen2.5-VL-72B 做区域识别；输出两种格式：
  - **QwenVL-HTML**：fine-grained element-level bbox
  - **QwenVL-Markdown**：只 localize images 和 tables，表格用 LaTeX
- Long Document：把单页 sample 拼成多页序列 + 跨页 VQA，覆盖 chart/table/figure/body text

### 5.5 Grounding & Counting
- Box grounding：COCO + Objects365 + OpenImages + RefCOCO/+/g，再用 Qwen2.5-VL + Grounding DINO 自动合成
- Point grounding：PixMo + 检测/分割数据 + 自家合成管线
- **坐标归一化到 [0, 1000]**（Qwen2.5-VL 是 [0, 1000] 没变？我重读一下…Qwen2.5-VL 用 [0, 1000]，Qwen3-VL "also" 用 [0, 1000]，paper 说 "Different from Qwen2.5-VL, we adopt a normalized coordinate system scaled to the range [0, 1000]"——这描述有点矛盾，可能 Qwen2.5-VL 用的是 [0, 100] 或 absolute pixel，Qwen3 改成 [0, 1000] 提高分辨率）

### 5.6 Spatial Understanding & 3D
- 2D Spatial：relational annotations ("cup to the left of laptop")、affordance labels ("graspable")、action-conditioned queries ("What should I move first to reach the book?")
- 3D Grounding：**9-DoF 3D bbox** JSON 格式 $[x_c, y_c, z_c, x_s, y_s, z_s, \text{roll}, \text{pitch}, \text{yaw}]$，统一到 Omni3D virtual camera coordinate system

### 5.7 Code
- Text-only code：复用 Qwen3 / Qwen3-Coder 的语料
- Multimodal code：UI screenshot → HTML/CSS、image → SVG、visual programming、StackOverflow with image、flowchart/diagram → code/LaTeX

### 5.8 Video
- **Short-to-long caption synthesis**：先短段 caption 再合成 holistic timestamp-interleaved story-level description
- **Spatio-temporal grounding**：object/action/person level annotation
- **Length-Adaptive Sampling**：根据 context 长度动态调 fps 和 max frames，避免信息损失

### 5.9 STEM
- 1M point-grounding samples + 2M perception VQA + 6M diagram captions（程序化渲染几何图）
- 60M+ K12/本科 level exercises（清洗 + reformulation + 中英互译）
- 12M+ long CoT multimodal reasoning samples，用 strong reasoning model 生成 rollout + rule check + model verify，reject ambiguous / code-switching

### 5.10 Agent
- GUI：跨平台（desktop/mobile/web）element description、dense captioning、dense grounding；self-evolving trajectory 生成 + 人工审核；CoT rationales 增强 planning/decision/reflection
- Function Calling：LLM with image 生成 query + function def → 采样 function call → 合成 response → 循环直到解决；不需要真正 implement executable function
- Search：online image search + text search tool，鼓励对陌生 entity 主动 search

---

## 6. Post-Training：三阶段

### 6.1 SFT（Cold Start）
- 总量 ~1.2M samples，1/3 text-only + 2/3 image-text/video-text
- 分两阶段：先 32K context 1 epoch，再 256K context 1 epoch（interleave long-context 和 32K 数据）
- **Thinking 和 Non-Thinking 两个分支**——same base model，不同 SFT data
- 数据过滤：Query Filtering（用 Qwen2.5-VL 过滤不可验证的）+ Response Filtering（rule-based + model-based，model 是 Qwen2.5-VL 系列 reward model）

### 6.2 Long-CoT Cold Start
- 1:1 text-only vs VL ratio
- **Difficulty curation**：只保留 baseline 低 pass rate 的样本
- **Multimodal Necessity Filtering**：关键 trick——如果 Qwen3-30B-nothink **不看图也能解**，丢弃！确保 VL 样本真的需要视觉信息，避免 cheating via text-only
- Response quality：去重 + 过滤 repetition / language mixing / guessing

### 6.3 Strong-to-Weak Distillation
两阶段：
- **Off-policy**：teacher 输出 + 学生 SFT，建立基础 reasoning
- **On-policy**：学生自己 sample → KL divergence 对齐 teacher logits

这里只对 **LLM backbone 用 text-only 蒸馏**——非常聪明的决策，用 strong text teacher 提升 LLM 推理，间接传导到 multimodal。

### 6.4 Reinforcement Learning

**Reasoning RL**：
- ~30K RL queries，每个 sample 16 次 rollout，pass rate > 90% 的丢弃（太简单）
- 任务：math, code, logic, visual grounding, visual puzzle，全部可 rule/code verify
- 用 **SAPO (Soft Adaptive Policy Optimization)**：smooth + adaptive policy gradient method。Gao et al. 2025: https://arxiv.org/abs/2511.20347
- Code-switching penalty：response 语言和 prompt 不一致就惩罚

**General RL**：
- 多任务 reward：VQA, captioning, OCR, parsing, grounding, clock recognition
- 两个维度：instruction following（复杂约束、JSON 等结构化输出）+ preference alignment
- **Counter-intuitive tasks 用来纠正 SFT 学到的错误 prior**——e.g. 反直觉的物体计数、复杂时钟识别
- Rule-based reward + Model-based reward（Qwen2.5-VL-72B / Qwen3 as judge）

---

## 7. Thinking with Images：Agentic 视觉推理

这是非常有意思的部分，借鉴了 DeepEyes / Mini-O3 等工作：
- Wu et al. 2025a MMSearch-R1: https://arxiv.org/abs/2506.20670
- Zheng et al. 2025 DeepEyes: https://arxiv.org/abs/2505.14362
- Lai et al. 2025 Mini-O3: https://arxiv.org/abs/2509.07969

两阶段训练：
1. **Stage 1**：10K grounding examples（简单 two-turn VQA，attribute detection），SFT 让 Qwen2.5-VL-32B 学会 think → act → analyze feedback → answer 的 agentic behavior，再 multi-turn tool-integrated RL
2. **Stage 2**：用 stage 1 trained agent 蒸馏出 120K 多轮 agentic 交互，覆盖更广任务；再做 cold start SFT + tool-integrated RL for Qwen3-VL

三个 reward：
- **Answer Accuracy Reward**（Qwen3-32B 评判）
- **Multi-Turn Reasoning Reward**（Qwen2.5-VL-72B 评判 step-by-step coherence）
- **Tool-Calling Reward**（实际 tool call 次数 vs Qwen2.5-VL-72B 估计的合理次数）

**关键观察**：早期模型倾向只 call 一次 tool 来 hack 前两个 reward——所以引入 Tool-Calling Reward 强制 adaptive tool exploration。这是典型的 RL reward hacking 防御。

---

## 8. 模型家族：Dense + MoE 双线

| 类型 | 模型 | 总参/激活 |
|---|---|---|
| Dense | Qwen3-VL-2B / 4B / 8B / 32B | 全 dense |
| MoE | Qwen3-VL-30B-A3B | 30B 总参 / 3B 激活 |
| MoE | Qwen3-VL-235B-A22B | 235B 总参 / 22B 激活 |

旗舰 235B-A22B 在多数 multimodal benchmark 上超越 Gemini 2.5 Pro (thinking budget-128) 和 GPT-5 (high thinking)。MoE 设计的好处是推理时只激活 22B，**inference latency 接近 22B dense 模型但能力接近 235B dense**。

---

## 9. 关键 Benchmark 结果解读

### 9.1 Multimodal Reasoning（Table 2）

Qwen3-VL-235B-A22B-Thinking 在 MathVista-mini 85.8、MathVision 74.6、MathVerse-mini 85.0、ZeroBench 4（这个 4 分说明 ZeroBench 真的是 impossible benchmark）、VisuLogic 57.2——多数 SOTA。

对比 Gemini 2.5 Pro thinking-128：MMMU 80.6 vs 68.8*、MathVision 74.6 vs 73.3、MathVista-mini 85.8 vs 82.7——Qwen3-VL 在 STEM 上明显领先。

### 9.2 Document Understanding

OCRBench 903（thinking）/ 920（instruct）、OCRBench_v2_en 875/920、CC-OCR 63.5/61.8、OmniDocBench_en 0.155/0.143（这个数值看起来是 error rate，越低越好）、MMLongBench-Doc 56.2/57.0。

特别 **MMLongBench-Doc SOTA**——验证了 DeepStack + 256K context 对长文档的价值。

### 9.3 Video Understanding

Video-MME w/o sub 79.0（thinking）/ 79.2（instruct）、MLVU 83.8/84.3、LVBench 75.2/76.5、Charades-STA mIoU 63.6/67.7。

**Needle-in-a-Haystack**（Figure 3）：30 分钟视频 256K context 100% accuracy；用 YaRN extrapolate 到 1M token（~2 小时视频）保持 99.5%。这是非常 strong 的 long-context video 结果。

### 9.4 Agent

OSWorld-G 68.3（thinking）/ 66.7（instruct）、AndroidWorld 62.0/63.7、OSWorld 38.1/31.6、WindowsAA 32.1/28.9。在 GUI agent 评估里属于 SOTA 梯队。

### 9.5 Text-Centric（Table 5/6）

这是最 interesting 的部分——VLM 的 text-only benchmark 跟 LLM 比：
- MMLU-Pro: 81.8 vs Qwen3-235B-A22B 83.0 vs DeepSeek V3 81.2 vs Claude-Opus-4 86.6
- AIME-25: 74.7 vs Qwen3 70.3 vs DeepSeek 46.6 vs Claude 33.9 ← Qwen3-VL **超过 text-only Qwen3**！
- LiveCodeBench v6: 54.3 vs Qwen3 51.8 vs DeepSeek 45.2 vs Claude 44.6

**关键 insight**：VLM 在 AIME/LiveCodeBench 上能超过自家 text-only 同尺寸 model——这说明 VL training 不仅没冲掉 text 能力，反而因为 multimodal reasoning data 的注入**反哺了 text reasoning**。这是非常反直觉但可解释：视觉数学题的 CoT 数据有大量结构化推理 trajectory，相当于额外的高质量 reasoning 数据。

---

## 10. 关键 Ablation 总结

### DeepStack（Table 12）
1.3 pt AVG gain，document-heavy task（InfoVQA +2.3, DocVQA +1.6, ChartQA +1.8）受益最大。

### Vision Encoder（Table 11）
Qwen3-ViT vs SigLIP-2：通用 CLIP benchmark 持平，OmniBench 大幅提升——但 OmniBench 是自家的，**这个 ablation 不能完全证明 Qwen3-ViT 的普适优越性**，需要谨慎解读。

### Needle-in-a-Haystack（Figure 3）
256K native 100%，YaRN 1M extrapolate 99.5%——long-context video retrieval 能力强。

---

## 11. 几个值得深挖的工程细节

### 11.1 Square-root reweighting 的数学
形式化：

$$\mathcal{L}_{\text{batch}} = \sum_{s \in B} \frac{1}{\sqrt{N_s}} \sum_{i=1}^{N_s} \ell_{s,i}$$

$B$ 是 batch，$s$ 是 sample，$N_s$ 是 sample $s$ 的 token 数，$\ell_{s,i}$ 是第 $i$ 个 token 的 per-token loss。等价于 per-token loss 乘以 $\sqrt{N_s}$ 的 weight。

对比：
- Per-sample: weight = 1（长样本被压扁）
- Per-token: weight = $N_s$（长样本主导）  
- Square-root: weight = $\sqrt{N_s}$（折中）

这跟 "normalized loss" 的思想一致，OpenAI 在 preference learning 里也用过类似 sqrt scaling。

### 11.2 Interleaved MRoPE 的实现
具体来说，如果 embedding dim $d = 128$，原来 Qwen2-VL：
- $[0, 42)$：t 的 RoPE
- $[42, 85)$：h 的 RoPE
- $[85, 128)$：w 的 RoPE

Qwen3-VL interleaved：
- $i \mod 3 = 0$：t
- $i \mod 3 = 1$：h  
- $i \mod 3 = 2$：w

每个轴均匀占有 $[0, d)$ 的低/中/高频段。这样 long-video 的 $m_t$ 既能在低频段表达"很久以前"，又能在高频段表达"刚刚发生"，不再因为低频段维度不够而退化。

### 11.3 9-DoF 3D bbox
$[x_c, y_c, z_c, x_s, y_s, z_s, \text{roll}, \text{pitch}, \text{yaw}]$：
- $(x_c, y_c, z_c)$：3D center
- $(x_s, y_s, z_s)$：3D size
- $(\text{roll}, \text{pitch}, \text{yaw})$：Euler 角表示朝向

模型直接输出这个 JSON，相当于把 3D detection 任务 reformulate 成 VQA——纯文本输出，不需要 head。

### 11.4 SAPO 算法
Soft Adaptive Policy Optimization，paper: https://arxiv.org/abs/2511.20347。核心是 smooth + adaptive 的 policy gradient，比 PPO 在多任务、多模态、多 model size 上稳定。具体细节我没读 paper，但从 Qwen3-VL 的使用看，它能同时 handle text reasoning、math、code、visual grounding 等异质 reward signal。

---

## 12. 我的几个观察 / 疑问

1. **Multimodal Necessity Filtering 是非常好的 trick**——很多 VL dataset 实际上不需要视觉就能解（LLM 看到 OCR'd text 就够了），这步过滤保证了 training signal 的有效性。值得所有 VLM 训练借鉴。

2. **Text-only distillation on LLM backbone** 反哺 multimodal 的思路很妙，但也意味着 multimodal reasoning 的"上限"被 text reasoning teacher 锁定。如果 teacher 不够强，student 上不去。

3. **Qwen3-ViT 的 ablation 只在自家 OmniBench 上显著**——这有 circular reasoning 嫌疑。建议未来在公开 benchmark 上做更公平对比。

4. **DeepStack 的 token budget 不增加**这点很关键——它不是 multi-scale visual token，是 multi-layer feature injection via residual add。如果 merger 是轻量 MLP（两层），额外计算开销很小。

5. **Thinking with Images 的 Tool-Calling Reward**——RL 中为了防 reward hacking 引入额外 reward signal 是非常 practical 的工程经验。Mini-O3 / DeepEyes 都遇到类似问题。

6. **Interleaved MRoPE 在 long-video 上的提升**——paper 没给单独的 ablation 直接对比 interleaved vs original MRoPE 在 long-video 上的效果。从 Needle-in-a-Haystack 看 long-context 检索没问题，但理解任务（LVBench 等）提升来自 architecture + data，难分贡献。

---

## 13. 相关参考链接汇总

**Qwen 系列**:
- Qwen3-VL GitHub: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL HF: https://huggingface.co/Qwen
- Qwen3-VL ModelScope: https://modelscope.cn/organization/qwen
- Qwen2-VL: https://arxiv.org/abs/2409.12191
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923

**Architecture components**:
- SigLIP-2: https://arxiv.org/abs/2502.14786
- DeepStack: https://arxiv.org/abs/2411.15635
- CoMP: https://arxiv.org/abs/2503.18931
- TimeMarker: https://arxiv.org/abs/2411.18211

**Benchmarks**:
- MMMU: https://arxiv.org/abs/2311.16502
- MathVista: https://arxiv.org/abs/2310.02255
- MathVision: https://arxiv.org/abs/2402.14804
- OCRBench: https://arxiv.org/abs/2405.10643
- OmniDocBench: https://arxiv.org/abs/2412.07626
- MLVU: https://arxiv.org/abs/2406.04264
- LVBench: https://arxiv.org/abs/2406.08035
- ZeroBench: https://arxiv.org/abs/2502.09696
- OSWorld: https://arxiv.org/abs/2411.18232
- AndroidWorld: https://arxiv.org/abs/2405.14573

**Reasoning / Agent**:
- SAPO: https://arxiv.org/abs/2511.20347
- DeepEyes: https://arxiv.org/abs/2505.14362
- Mini-O3: https://arxiv.org/abs/2509.07969
- MMSearch-R1: https://arxiv.org/abs/2506.20670

**Vision encoders / 3D**:
- Omni3D: https://arxiv.org/abs/2107.07504
- Grounding DINO: https://arxiv.org/abs/2303.05499
- PixMo: https://arxiv.org/abs/2409.17146

---

## 14. 总结：Qwen3-VL 的核心贡献与局限

**核心贡献**：
1. **Interleaved MRoPE** 解决频谱不平衡——简单但有效的工程改进
2. **DeepStack** 多层 ViT feature 注入 LLM——不增 token budget 提升 fine-grained 能力
3. **Text timestamp** 替代 T-RoPE——把时间感知从 position encoding 卸载到文本流
4. **Square-root reweighting**——平衡 text 和 VL loss
5. **Multimodal Necessity Filtering** + **Text-only distillation** + **Strong-to-Weak**——保证 text 能力不被冲掉，甚至反哺
6. **4-stage pretraining + 3-stage post-training**——明确分工，每阶段有具体目标
7. **完整 model family**（dense 2B-32B + MoE 30B-A3B / 235B-A22B）覆盖不同 latency-quality trade-off

**局限 / 可质疑点**：
1. Qwen3-ViT 的 ablation 主要在自家 benchmark，公平性不足
2. Interleaved MRoPE 缺单独 long-video ablation
3. Some benchmark 数字比 Gemini-2.5-Pro / GPT-5 略低，特别是 thinking budget 比较不公平（Gemini budget-128，Qwen 没有 budget limit）
4. Video eval 因 API 限制 input frame 数量不一致（Gemini 512 / GPT-5 256 / Claude 100），公平性受限

**对 VLM 领域的启示**：
- **不要让 vision 训练冲掉 text**——square-root reweighting + text-only distillation + necessity filtering 是三道防线
- **多层 feature 注入**比单层 feature + multi-scale 更高效（token budget 不变）
- **把时间信息塞进文本**比塞进 position encoding 更 long-extrapolatable
- **MoE for VLM** 在 latency-quality trade-off 上有显著优势

希望这个 walkthrough 帮你 build 了 Qwen3-VL 的 intuition。如果你想我深入某一节（比如 SAPO 的具体算法、DeepStack 的 merger 架构细节、或 RL reward design 的具体实现），告诉我，我再展开讲。
