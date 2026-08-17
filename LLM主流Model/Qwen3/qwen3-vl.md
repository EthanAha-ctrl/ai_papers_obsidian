---
source_pdf: qwen3-vl.pdf
paper_sha256: ee075d08e67de1148d6437c6c1d481f7894183b8793905a2deb5f62664f49380
processed_at: '2026-08-11T20:31:19-07:00'
target_folder: LLM主流Model/Qwen3
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Qwen3-VL 用人话讲讲

Andrej，这篇 paper 我读了几遍，整体感觉是 **engineering-heavy 但每一步都有 clear motivation**。不是那种炫技的 paper，每个 trick 都在解决一个具体的痛点。我从工程师视角讲讲。

---

## 一、这篇 paper 到底在解决什么 pain

你做 VLM 做久了肯定遇到几个心累的事：

1. **长视频一拉就崩**：模型看 1 分钟视频还行，看 30 分钟就忘了前面。为什么？因为之前的 positional encoding 在长视频上外推不过去。
2. **OCR / DocVQA 永远不够细**：ViT 最后一层给 LLM 的 visual feature 已经被"抽象化"了，细节丢了，所以小字、表格、图表永远是弱项。
3. **VLM 一训就忘 text**：vision 数据一多，LLM 原本的 reasoning 能力就掉。业界普遍做法是拼命补 text 数据，效果有限。
4. **Tool use 训不好**：让模型学会"什么时候 zoom in" 很难，模型要么不调用 tool，要么疯狂调用 tool 来 hack reward。

Qwen3-VL 这篇 paper 基本就是围绕这 4 个 pain 一一对应的解法。

---

## 二、三个架构 trick 用人话讲

### 1. Interleaved MRoPE：给 temporal 装上显微镜和望远镜

**老方法的问题**：Qwen2-VL 把 embedding 维度切成三份，一份给 temporal (t)，一份给 horizontal (h)，一份给 vertical (w)。具体说 $d/6$ 维给 t 的低频，$d/6$ 维给 t 的高频，h 和 w 同理。

但实际操作时 t 主要占低频 band。这造成一个尴尬的事：**temporal 维度只能感知"大局"，感知不到"细节"**。

类比一下：你戴了一副只能看远处的望远镜，能看到"这是早上 9 点"，但分不清"这是 9:01 还是 9:02"。对于短视频无所谓，对于 1 小时视频，10 秒粒度的 action 你就完全抓不住了。

**新方法**：把 t、h、w **interleaved 分布**到整个 frequency spectrum 上。第 0 维给 t 的低频，第 1 维给 h 的低频，第 2 维给 w 的低频，第 3 维给 t 的中频，第 4 维给 h 的中频……依此类推。

形式化一下，第 $i$ 个 frequency 对应的轴是 $\text{axis}(i) = i \mod 3$（0=t, 1=h, 2=w）。

**Intuition**：每个轴都均匀地拿到从低频到高频的全谱。t 既能感知"这是早上"（低频），也能感知"这是 9:01:30"（高频）。h、w 也一样，既能感知"图片整体在左上"（低频空间），也能感知"这个像素在 (234, 567)"（高频空间）。

这种思想其实和 RoFormer 原始论文里对纯文本的 RoPE 设计异曲同工——让每一层 attention 都能 attend 到 multiple scales。

**效果**：长视频理解能力大幅提升。Paper 里 Needle-in-a-Haystack 测试，30 分钟视频（256K token）100% accuracy，2 小时视频用 YaRN 外推到 1M token 还是 99.5%。这个数字我自己看到的时候是真的"啊？"。

### 2. DeepStack：让 LLM 早期层直接看到 raw 视觉

**老方法的问题**：传统 VLM 是 ViT 跑完所有层，把最后一层输出扔给 LLM。相当于 ViT 是个翻译官，先把图"消化"成抽象语义，再告诉 LLM "这张图里有个红色杯子在桌上"。

问题在哪？**OCR、表格、图表这种 fine-grained 任务需要细节，但细节在 ViT 深层被抽象掉了**。ViT 第 1 层看到的是 edge、texture，第 24 层看到的是 "cup"，但 OCR 需要的是字符级别的 stroke。

**新方法**：从 ViT 的 3 个中间层抽 features，分别通过 dedicated merger 投影，**直接加到 LLM 前 3 层的 hidden state 上**。

```
ViT Layer A → Merger_1 → add → LLM Layer 1 hidden
ViT Layer B → Merger_2 → add → LLM Layer 2 hidden
ViT Layer C → Merger_3 → add → LLM Layer 3 hidden
ViT Layer final → standard merger → LLM input (像传统 VLM)
```

**Intuition**：相当于 ViT 在 LLM 旁边递纸条："嘿，这是 raw edge 信息"，"嘿，这是 mid-level texture"。LLM 早期层就开始融合视觉，而不是等 ViT 把视觉"翻译"完。这有点像你在读论文时，旁边坐个专家随时给你补充背景知识，而不是等他看完论文再告诉你结论。

**为什么不增加 context length**：因为 DeepStack 是 hidden state 层面的 residual，不引入新 token。Visual token 数量不变，inference cost 不变。这点很关键——是免费午餐。

**效果**：Table 12 ablation，15B-A2B LLM 上 200B token 训练，OCR 从 81.0 → 83.6，InfoVQA 从 71.9 → 74.2，DocVQA 从 89.5 → 91.1。**全是 fine-grained 任务提升最大**，符合直觉。

### 3. Text-based Timestamp：把时间从"位置编码"降维成"文本"

**老方法的问题**：Qwen2.5-VL 用 T-RoPE，第 5 秒的帧，temporal position id 就是 5。听起来 natural，但有两个坑：

- **坑 1**：长视频 position id 跑到几千几万，RoPE 没见过这么大的 id，外推性能崩。
- **坑 2**：要让模型 robust，得在 1fps、2fps、4fps 各种帧率上均匀采样训练，数据构建成本爆炸。

**新方法**：在每个 frame group 前面**显式插入一个 text token**，比如 `<3.0 seconds>`。训练时同时用 seconds 和 HMS 两种格式（`<00:00:03.000>`），让模型学会读不同 timecode。

**Intuition**：把时间从 positional encoding 退化为 text understanding。LLM 处理文本是它的强项，所以这样更 robust、更灵活。代价是稍微多几个 token，但换来的是：
- 长视频不会 position id 爆表
- 不需要训各种 fps
- 时间表达灵活，可以做 "3.5s" 也可以做 "00:00:03.500"
- 模型可以输出 timestamp 做 video grounding 和 dense captioning

这其实是个很 elegant 的"hack"。我一开始觉得"这不算 architecture innovation 吧"，但仔细想，**把问题从硬编码（position）转化为软编码（text）**，让模型用自己擅长的能力去处理，这恰恰是 ML 里最高效的设计思路。

---

## 三、训练 trick 用人话讲

### 1. Square-root Reweighting：在 token 和 sample loss 之间找平衡

**问题背景**：训练 VLM 时一个 batch 里样本长度差异巨大。一张图的 caption 可能 50 token，一个长文档 QA 可能 100K token。

**Per-sample loss**（每个样本算平均 loss 再 batch 平均）：
$$\mathcal{L} = \frac{1}{N} \sum_i \frac{1}{T_i} \sum_t \ell_{i,t}$$
短样本被过度放大——50 token 的小 QA，每个 token 都被加权 $\frac{1}{N \cdot 50}$；100K token 的长文档，每个 token 被加权 $\frac{1}{N \cdot 100K}$。短样本 token 权重是长样本的 2000 倍。

**Per-token loss**（所有 token 平等）：
$$\mathcal{L} = \frac{1}{\sum T_i} \sum_{i,t} \ell_{i,t}$$
长样本主导——100K 文档有 100K 个 token，每个都贡献 loss，长文档"票数"太多。

**Square-root reweighting**：
$$\mathcal{L} = \frac{1}{\sum_i \sqrt{T_i}} \sum_i \frac{1}{\sqrt{T_i}} \sum_t \ell_{i,t}$$

**Intuition**：长样本权重 = $\sqrt{T_i}$，介于 1（per-sample）和 $T_i$（per-token）之间。长样本说话权大，但不是线性大，给短样本留存在感。

**效果**：Table 5/6 显示，Qwen3-VL 在 text benchmark（AIME-25、MMLU-Pro、LiveCodeBench）上和 Qwen3 text-only 持平甚至略胜。**VLM 训练不退化 text 能力**，这是 square-root + 大量 text 数据 + necessity filtering 共同作用的结果。

### 2. Multimodal Necessity Filtering：确保样本真的需要看图

**问题**：很多 multimodal dataset 里的 VQA 样本，**光看文本就能答对**。比如题目是"这张图里有几只猫？"，但选项里"3"和其他选项语义差距太大，模型从选项分布就能猜。

这种样本训 VLM 没用，反而让模型学到 shortcut——不看图也能蒙对。

**Qwen3-VL 的做法**：用 Qwen3-30B-nothink（不看图）跑一遍样本，能答对的全部 discard。只保留**真正需要看图才能答**的样本。

**Intuition**：这相当于 data curation 里的"hard negative mining"。把简单样本扔掉，只留模型必须用 multimodal 能力才能解的。我估计这个 filtering 过滤掉了 30-50% 的数据。

### 3. Tool-integrated RL：三个 reward 缺一不可

**问题**：让 VLM 学会 "thinking with images"——什么时候 zoom in、zoom in 哪、看到结果怎么 reason——很难。

Qwen3-VL 用三个 reward：
- **Answer Accuracy**：Qwen3-32B 评判最终答案对错
- **Multi-Turn Reasoning**：Qwen2.5-VL-72B 评判是否正确解读 tool feedback
- **Tool-Calling**：实际 tool 调用次数 vs Qwen2.5-VL-72B 估的"专家目标次数"

**关键发现**：只用前两个 reward，模型会 degenerate 到**只用一次 tool call** 来 hack——反正一次 tool call 也能拿到 partial 信息，答案可能对，推理过程可能 OK。引入第三个 reward 后，模型被迫**根据任务难度调整 tool 次数**。

**Intuition**：这其实是 RLHF 里经典的 reward hacking 问题。多 reward 互相制约，每个 reward 防 hack 的"漏洞"由另一个 reward 补上。这种 multi-objective RL 设计和 OpenAI o1 / R1 的思路一致。

### 4. Strong-to-Weak Distillation：text-only 蒸馏提升 reasoning

**做法**：用强大的 teacher model 在 text-only data 上 generate reasoning chain，然后让小 model 通过 off-policy + on-policy 两个阶段蒸馏。

- **Off-policy**：teacher 输出做 SFT，让 student 学到 teacher 的 reasoning pattern
- **On-policy**：student 自己 generate，然后 minimize KL divergence with teacher logits

**Intuition**：off-policy 让 student 学到"成品"，但 student 的 distribution 和 teacher 不同；on-policy 让 student 在自己 distribution 上对齐 teacher，解决 distribution mismatch。

这个 trick 让 Qwen3-VL 2B/4B/8B 这种小模型在 reasoning benchmark 上能打过比自己大几倍的 baseline。

---

## 四、评估里让我"啊？"的几个点

### 1. VLM 在 AIME-25 上超过 text-only LLM

Table 5：Qwen3-VL-235B-A22B-Instruct 在 AIME-25 上 74.7，DeepSeek V3 0324 只有 46.6。**VLM 数学竟然比纯 text LLM 还强**。

这说明 multimodal 训练不仅没退化数学，反而 enhance 了。我猜测原因是：
- Vision 数据里的几何、图表、公式训练让模型对 spatial reasoning 更敏感
- Square-root reweighting + 大量 text 数据保底
- Long-CoT 蒸馏从 reasoning teacher 那里继承了数学 chain

### 2. 8B 在 video 上能打 Qwen2.5-VL-72B

Paper 原文："Qwen3-VL 8B variant to achieve performance competitive with the significantly larger Qwen2.5-VL 72B model."

这是 9x 的 size compression。Interleaved MRoPE + text timestamp + dense video caption 三者结合的效果。

### 3. Tool > Size Scaling

Table 2 V* 上：Qwen3-VL-235B-Instruct 无 tool 是 85.9，加 tool 93.7。同 family 内，加 tool 一致带来 ~5 点提升。Paper 原文："the performance gains from integrating external tools consistently outweigh those from simply increasing model size."

**这个 observation 我觉得是这篇 paper 最重要的 takeaway**。在 multimodal fine-grained perception 上，**agentic tool use 比 scale 模型更高效**。这呼应了 o1 时代"reasoning + tool" 范式。

### 4. 1M Token Needle-in-a-Haystack 99.5%

256K native context 训练，用 YaRN 外推到 1M token，视频 needle-in-haystack 99.5%。这说明 RoPE frequency base 调整对长视频外推很有效，**不需要重新训练长 context**。

### 5. VisuLogic 大幅领先

Qwen3-VL-235B-Thinking 在 VisuLogic（visual logical reasoning）上 57.2，Gemini-2.5-Pro 31.6，GPT-5 28.5，Claude-Opus-4.1 27.9。**接近 2x 的领先**。这个 benchmark 是 visual logical reasoning，说明 Qwen3-VL 在 visual reasoning 上确实有优势，不只是 perception。

---

## 五、我自己读完后想再深挖的点

1. **DeepStack 用 3 层的依据**：Paper 没说为什么是 3 层不是 5 层或 2 层。更多层会更好还是 saturate？这值得做 ablation。

2. **Qwen3-ViT 的具体改动**：Table 11 显示 Qwen3-ViT 比 SigLIP-2 在 OmniBench 上 36.9 → 45.5，paper 只说"continuous training with dynamic resolutions"。具体改了什么？是数据变了还是 architecture 变了？

3. **Text timestamp 的 token overhead**：超长视频（10K frames）会增加多少 token？Paper 没量化。如果每帧一个 timestamp，10K 帧 = 10K timestamp tokens，相对于 visual token 总数占比多大？

4. **Multimodal Necessity Filtering 的副作用**：过滤掉"text 可解"的样本，会不会丢掉一些"text + vision synergy"的样本？比如某题 vision 提供 redundant cue，对 reasoning 有帮助但不必要。这种 sample 被过滤掉是不是 loss？

5. **Thinking mode 的 inference cost**：AIME-25 max output 81,920 token。生产中 80K thinking token 的 latency 和 cost 是什么概念？ROI 怎么算？

6. **VLM 不退化 text 的归因**：square-root reweighting、necessity filtering、strong-to-weak distillation 三个因素各贡献多少？如果做 ablation，哪个最关键？Paper 没拆开做。

7. **Tool > Size 的边界**：V*、HRBench 这种 fine-grained perception task 上 tool 大幅提升。但 reasoning task（如 AIME）上 tool use 是否同样有效？还是只在 perception 上有效？这是个开放问题。

---

## 六、一句话总结

Qwen3-VL 是一个 **engineering-focused 的 VLM 工作**，每个 trick 都在解决一个具体痛点：
- Interleaved MRoPE 解决长视频外推
- DeepStack 解决 fine-grained perception
- Text timestamp 解决时间表达
- Square-root reweighting 解决 text 退化
- Necessity filtering 解决 multimodal shortcut
- Multi-reward RL 解决 tool use hacking

最大的 takeaway 我觉得是 **"agentic tool use > scale"**——在 fine-grained multimodal 上，让模型学会用 tool 比堆参数更高效。这和 o1 时代的"reasoning is the new scaling" 思路一致。

Paper 链接：
- https://github.com/QwenLM/Qwen3-VL
- https://huggingface.co/Qwen

如果让我用一个词总结这篇 paper 的 design philosophy，是 **"elegant hacks"**——每个 trick 都不是惊天动地的 architecture innovation，但组合起来效果显著。这其实是工程界最珍贵的能力：**在已有 framework 里找到对的 intervention 点**。

---

# Qwen3-VL 技术报告深度解读

## 一、整体定位与核心贡献

Qwen3-VL 是 Qwen 系列中目前最强的 vision-language model，原生支持 256K token 的 interleaved context（文本+图像+视频），同时推出了 dense (2B/4B/8B/32B) 和 MoE (30B-A3B / 235B-A22B) 两种架构变体。三大核心支柱：

1. **Pure-text understanding 不退化**：在很多情况下甚至超过同规模的 text-only backbone
2. **Long-context comprehension**：原生 256K window，支持长文档和长视频的 faithful retention、retrieval、cross-referencing
3. **Advanced multimodal reasoning**：在 MMMU、MathVista、MathVision 等 benchmark 上达到 SOTA

架构上有三个关键升级：Interleaved MRoPE、DeepStack、Text-based timestamp alignment。训练侧引入了 square-root reweighting 来平衡 text 和 multimodal 的 loss。

参考链接：
- GitHub: https://github.com/QwenLM/Qwen3-VL
- HuggingFace: https://huggingface.co/Qwen

---

## 二、架构详解

整体沿用 Qwen2.5-VL 的三模块结构：Vision Encoder + MLP-based Vision-Language Merger + LLM。但每个模块都有重要改造。

### 2.1 Vision Encoder

采用 **SigLIP-2** 架构（Tschannen et al., 2025, https://arxiv.org/abs/2502.14786），并继续训练以支持 dynamic resolution。具体而言，遵循 CoMP（Chen et al., 2025, https://arxiv.org/abs/2503.18931）的方法，在 ViT 内部使用 **2D-RoPE**，并基于 input size 对 absolute position embeddings 做插值。

- 默认使用 **SigLIP2-SO-400M**（对于较大 LLM）
- 小规模 LLM（2B/4B）使用 **SigLIP2-Large (300M)**

**Intuition**：SigLIP-2 相比 SigLIP 在语义对齐、localization、dense features 上都有改进，这对后续 grounding、OCR 等 task 至关重要。2D-RoPE 让 ViT 天然支持任意分辨率输入，避免了传统 fixed-size patch 的 resolution bottleneck。

### 2.2 Interleaved MRoPE（核心创新 1）

**背景**：Qwen2-VL（Wang et al., 2024c, https://arxiv.org/abs/2409.12191）引入了 MRoPE（Multimodal RoPE），将 embedding 维度划分为三组：
- temporal (t)
- horizontal (h)
- vertical (w)

每组分配不同的 rotary frequencies。

**问题**：这种 partition 会导致 **frequency spectrum imbalance**。低频主要给 t，高频主要给 h/w。在长视频理解中，temporal 维度需要既能感知短时事件（高频）又能感知长时结构（低频），但原始 MRoPE 让 t 主要是低频，丢失了 short-term temporal precision。

**Qwen3-VL 的解法**（inspired by Huang et al., 2025）：将 t、h、w **interleaved** 地分布到 embedding 维度上，让每个轴都均匀地占据低频和高频 band。

**形式化描述**：设 embedding dimension 为 $d$，rotary frequency 第 $i$ 维为：
$$\theta_i = 10000^{-2i/d}, \quad i \in \{0, 1, \ldots, d/2 - 1\}$$

原始 MRoPE：
$$\text{axis}(i) = \begin{cases} t & i \in [0, d/6) \\ h & i \in [d/6, d/3) \\ w & i \in [d/3, d/2) \end{cases}$$

Interleaved MRoPE：
$$\text{axis}(i) = \begin{cases} t & i \mod 3 = 0 \\ h & i \mod 3 = 1 \\ w & i \mod 3 = 2 \end{cases}$$

**Intuition**：想象你在做 Fourier 分析，如果 t 只在低频 band，那么模型对 short-burst events（比如 1 秒内的 action）的位置感知会很差。Interleaved 之后，每个轴都有"细粒度"和"粗粒度"两种尺度，长视频理解能力显著提升。这其实和 RoPE 在纯文本中的 interleave frequency 思想类似——让每一层 attention 都能 attend 到不同尺度的位置关系。

### 2.3 DeepStack（核心创新 2）

借鉴 **DeepStack**（Meng et al., 2024, NeurIPS 2024, https://arxiv.org/abs/2411.16535），但做了重要改造。

**原始 DeepStack**：将 multi-scale visual inputs（不同分辨率）的 token 注入 LLM 不同层。

**Qwen3-VL 的改造**：从 ViT 的**不同中间层**抽取 features，分别通过 dedicated merger 投影到 visual token，然后**直接加到 LLM 前 3 层的 hidden states** 上。

**架构图解析**（Figure 1）：

```
ViT Layer 1 → Merger_1 → + → LLM Layer 1 hidden state
ViT Layer 2 → Merger_2 → + → LLM Layer 2 hidden state  
ViT Layer 3 → Merger_3 → + → LLM Layer 3 hidden state
ViT Layer 4 → ... → final visual tokens → (standard pathway to LLM input)
```

**Intuition**：ViT 的浅层捕获 low-level features（edges、textures），深层捕获 high-level semantics（object categories、scene gist）。传统 VLM 只把 ViT 最后一层输出喂给 LLM，相当于让 LLM 只看到"已经抽象过的"视觉信息。DeepStack 让 LLM 早期层就能接触到 raw 视觉细节，相当于在 LLM 内部做了一个 **hierarchical visual-textual fusion**。

这点对 OCR、DocVQA、InfoVQA 这种 fine-grained 任务特别重要——从 Table 12 的 ablation 可以看到，InfoVQA 从 71.9 → 74.2，DocVQA 从 89.5 → 91.1。

**为什么不增加 context length**：因为 DeepStack 是在 hidden state 层面做 residual connection，而不是引入新的 token 序列。visual token 数量保持不变。

### 2.4 Video Timestamp（核心创新 3）

**Qwen2.5-VL 的做法**：用 T-RoPE（time-synchronized MRoPE），将 temporal position ID 直接绑定到绝对时间。比如第 5 秒的帧，temporal position id = 5。

**两个问题**：
1. 长视频会产生**超大且稀疏的 temporal position ids**，比如 2 小时视频，最后一个帧的 position id 可能是 7200，而 RoPE 在外推到训练时未见过的 position id 时性能急剧下降
2. 训练需要在**各种 fps 上均匀采样**，数据构建成本极高

**Qwen3-VL 的做法**：用**显式的文本 timestamp token** 标记每个 frame group，比如 `<3.0 seconds>`。训练时同时使用 seconds 和 HMS (hours:minutes:seconds) 两种格式，让模型学会解读不同 timecode 表示。

**Intuition**：这其实是一个很 elegant 的"hack"——把 temporal 信息从 positional encoding 退化为 text token，相当于让 LLM 用自己最擅长的"语言理解"来处理时间。代价是稍微增加 context length，但换来的是：
- 时间信息**显式可读**，模型可以做 video grounding、dense captioning 等 task
- 不需要训练各种 fps，数据构建成本降低
- 时间表达方式更灵活（可以用 "3.5s" 也可以用 "00:00:03.500"）

这种做法和 Chen et al., 2024b 的 TimeMarker（https://arxiv.org/abs/2411.18211）思路一致。

### 2.5 Vision-Language Merger

沿用 Qwen2.5-VL 的设计：两层 MLP，将 2×2 的 visual feature 压缩成 1 个 visual token，对齐到 LLM hidden dim。但额外部署了**专门的 merger** 来支持 DeepStack 的多层 feature 投影。

---

## 三、Pre-Training 详解

### 3.1 四阶段训练（Table 1）

| Stage | Objective | Trainable | Token Budget | Seq Len |
|-------|-----------|----------|--------------|---------|
| S0 | Vision-Language Alignment | Merger only | 67B | 8,192 |
| S1 | Multimodal Pre-Training | All | ~1T | 8,192 |
| S2 | Long-Context Pre-Training | All | ~1T | 32,768 |
| S3 | Ultra-Long-Context Adaptation | All | 100B | 262,144 |

**S0 的 intuition**：先冻结 ViT 和 LLM，只训练 merger，让 projection 层学会把 visual feature 翻译到 LLM 的语义空间。这避免了早期 multimodal signal 破坏 LLM 已经学好的 text representation。

**S1-S3 的递进 context**：S1 在 8K 上建立基础能力，S2 扩展到 32K 让模型处理长文档和长视频，S3 在 256K 上做 ultra-long adaptation。这是典型的 **progressive context extension**，和 LLaMA、Qwen3 纯文本模型的 long-context training 一脉相承。

### 3.2 Square-Root Reweighting（关键 loss 改进）

从 per-sample loss 改为 **square-root-normalized per-token loss**。

形式化：设一个 batch 中有 $N$ 个样本，第 $i$ 个样本有 $T_i$ 个 token。原始 per-sample loss：
$$\mathcal{L}_{\text{sample}} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{t=1}^{T_i} \ell_{i,t}$$

Square-root reweighted loss：
$$\mathcal{L}_{\text{sqrt}} = \frac{1}{\sum_i \sqrt{T_i}} \sum_{i=1}^{N} \frac{\sqrt{T_i}}{T_i} \sum_{t=1}^{T_i} \ell_{i,t} = \frac{1}{\sum_i \sqrt{T_i}} \sum_{i=1}^{N} \frac{1}{\sqrt{T_i}} \sum_{t=1}^{T_i} \ell_{i,t}$$

**Intuition**：
- 纯 token-level loss（$\frac{1}{\sum T_i} \sum \ell_{i,t}$）会让长样本主导（长文档有 100K token，短 QA 只有 50 token）
- 纯 sample-level loss（$\frac{1}{N} \sum \frac{1}{T_i} \sum \ell_{i,t}$）会让短样本过度加权，每个 short sample 的 token 都被放大
- Square-root 是 $\sqrt{T}$ scaling，介于两者之间：长样本仍有更多权重，但不是线性，避免短样本被淹没

这种 reweighting 让 text（一般短）和 multimodal（图像 token 多）之间达到更好的平衡，同时不让 text 能力退化。从结果看（Table 5/6），Qwen3-VL 在纯 text benchmark 上和 Qwen3 text-only 持平甚至略胜，证明这个策略有效。

### 3.3 Pre-Training 数据（重要细节）

#### Image Caption & Interleaved Data
- 用 Qwen2.5-VL-32B fine-tune 一个 recaptioning model，基于原始 raw text 生成更 fine-grained 的 caption
- Deduplication **只在 recaptioned text 上做**（用 semantic similarity），保留 visual diversity
- 用 clustering（Johnson et al., 2019 FAISS, Douze et al., 2024, Diao et al., 2025 CLIP）在 visual embedding 空间找 sparse regions，定向 augmentation
- Interleaved data 来自 web，用 lightweight Qwen scorer 做 domain classification，排除广告、clickbait
- Book-scale interleaved：用 Qwen2.5-VL-7B 做 multimodal parsing，对齐 text 和 figures
- Ultra-long subset：合并连续页到 256K token 序列，要求 minimum page count 和 minimum image-to-text ratio

#### Knowledge
- 覆盖 12+ semantic categories：animals、plants、landmarks、food、vehicles、electronics、clothing 等
- **Importance-based sampling**：高 prominence entity 多采样，低 prominence 少量但保留
- 用 LLM 生成 rich description，包含 attributes、context、spatial layout、interactions

#### OCR & Document Parsing
- OCR：30M in-house samples，coarse-to-fine pipeline，从 10 种语言扩展到 39 种语言
- Document Parsing：3M PDFs from Common Crawl，10 种 document type 各 300K
- 两种表示：**QwenVL-HTML**（fine-grained element-level bounding box）和 **QwenVL-Markdown**（只有 images 和 tables localized，tables 用 LaTeX）

**Intuition**：双表示让模型既能做精确 layout 解析（HTML），又能做轻量级 markdown 输出。HTML 训练让模型学会 element-level 几何，Markdown 是它的"压缩版"，相当于内置了一种 distillation。

#### Grounding & Counting
- 坐标系归一化到 **[0, 1000]** 范围（Qwen2.5-VL 是 [0, 1000] 还是 absolute pixel？这里改进了 robustness）
- Box grounding：开源数据 + 自动合成 pipeline（Qwen2.5-VL 提候选 → Grounding DINO + Qwen2.5-VL 定位 → 质量过滤）
- Point grounding：PixMo + 检测数据 + fine-grained 合成
- Counting：direct、box-based、point-based 三种 task formulation

#### Spatial Understanding & 3D
- Spatial：relational annotation（"cup to the left of laptop"）、affordance（"graspable"、"pressable"）、action-conditioned query
- 3D Grounding：9-DoF 3D bounding box（x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw）
- 统一到 virtual camera coordinate system（Omni3D, Brazil et al., 2023, https://arxiv.org/abs/2207.10660）

#### Code
- Text-only code：复用 Qwen3-Coder 的 code corpus
- Multimodal code：UI screenshot → HTML/CSS、image → SVG、visual programming、flowchart/diagram → code、LaTeX equation 转写

#### Video
- **Dense Caption Synthesis**：short-to-long 策略生成 timestamp-interleaved 的 story-level description
- **Spatio-Temporal Video Grounding**：object、action、person 三层 annotation
- **Length-Adaptive Sampling**：动态调整 fps 和 max frame count，避免 sparse sampling 丢信息

#### STEM
- **Divide-and-conquer**：先独立训练 visual perception 和 linguistic reasoning，再 synergy
- 视觉感知：1M point-grounding + 2M perception VQA，6M diagram caption
- 多模态推理：60M K-12/undergrad exercises，12M long-CoT multimodal reasoning（用 strong reasoning model 的 rollout，rule + model 双重验证）
- 关键：**Multimodal Necessity Filtering**——如果 Qwen3-30B-nothink 不看图就能解，就 discard，确保样本真正需要 multimodal

#### Agent
- GUI：cross-platform (desktop/mobile/web) data，self-evolving trajectory production + human audit
- Function Calling：多模态 function calling trajectory 合成，迭代直到 query 解决
- Search：multimodal factual lookup trajectories with online image search + text search

---

## 四、Post-Training 详解

### 4.1 三阶段 post-training

1. **SFT**：32K context 第一阶段 → 256K context 第二阶段（聚焦长文档和长视频）。分叉为 non-thinking 和 thinking 两种数据格式
2. **Strong-to-Weak Distillation**：用 text-only data distill，提升 reasoning
3. **Reinforcement Learning**：Reasoning RL + General RL

### 4.2 SFT 数据

- 1.2M samples，1/3 text-only，2/3 image-text + video-text
- 多语言、single-turn + multi-turn、interleaved
- **两阶段 sequence length**：先 32K 一 epoch，再 256K 一 epoch（curriculum with 32K sampling）
- **两阶段 filtering**：Query Filtering（用 Qwen2.5-VL 识别 unverifiable query）+ Response Filtering（rule-based + model-based）

### 4.3 Long-CoT Cold Start Data

- 1:1 VL : text 比例
- 三个 filtering：
  1. **Difficulty Curation**：只保留 baseline 模型 pass rate 低或 response 长的样本
  2. **Multimodal Necessity Filtering**：discard Qwen3-30B-nothink 不看图就能解的题
  3. **Response Quality Control**：去除 repetition、language mixing、guessing

### 4.4 Strong-to-Weak Distillation

- **Off-policy Distillation**：teacher 输出做 response distillation
- **On-policy Distillation**：student 自己 generate，然后 minimize KL divergence with teacher logits

**Intuition**：off-policy 让 student 学到 teacher 的"成品"，on-policy 让 student 在自己的 distribution 上对齐 teacher，避免 distribution mismatch。这种两阶段 distillation 在 Qwen3 系列已经验证有效。

### 4.5 Reinforcement Learning

#### Reasoning RL
- 用 **SAPO**（Soft Adaptive Policy Optimization, Gao et al., 2025, https://arxiv.org/abs/2511.20347）
- 30K RL queries，每 query 采样 16 个 response，过滤 pass rate > 90% 的 easy query
- Task-specific format prompt，不靠 explicit format reward
- **Code-switching penalty**：response 语言和 prompt 语言不一致就惩罚

#### General RL
- 多 task RL：VQA、caption、OCR、document parsing、grounding、clock recognition
- 两个维度：Instruction Following（format、length、JSON）+ Preference Alignment
- **Corrective mechanism**：unlearn SFT 的错误 prior（counter-intuitive counting、complex clock）
- **Hybrid reward**：Rule-Based（高精度 verifiable task）+ Model-Based（Qwen2.5-VL-72B-Instruct 或 Qwen3 作为 judge）

### 4.6 Thinking with Images（创新点）

两阶段训练 paradigm：

**Stage 1**：合成 ~10k grounding examples，在 Qwen2.5-VL-32B 上做 SFT，emulate "think → act → analyze feedback → answer" 行为，再做 multi-turn tool-integrated RL

**Stage 2**：用 Stage 1 训好的 agent distill 出 120k 多轮 agentic interaction，对 Qwen3-VL 做 cold-start SFT + tool-integrated RL

**三种 reward**：
1. **Answer Accuracy Reward**：Qwen3-32B 评判最终答案对错
2. **Multi-Turn Reasoning Reward**：Qwen2.5-VL-72B 评判是否正确解读 tool feedback
3. **Tool-Calling Reward**：实际 tool call 次数 vs expert-estimated target

**关键观察**：早期模型会 degenerate 到只用一次 tool call 来 hack 前两个 reward，所以必须引入 tool-calling reward 来 promote adaptive tool exploration。

**Intuition**：这其实就是 OpenAI o1 / R1-style RL 的 multimodal 版本。让模型学会"什么时候 zoom in、zoom in 哪里、看到结果怎么 reason"，而不是无脑用 tool 或完全不用 tool。

---

## 五、评估结果关键发现

### 5.1 Multimodal Reasoning（Table 2）

Qwen3-VL-235B-A22B-Thinking 在 MathVista_mini、MathVision、MathVerse、ZeroBench、LogicVista、VisuLogic 上达到 SOTA。ZeroBench 是一个 "impossible visual benchmark"（Roberts et al., 2025, https://arxiv.org/abs/2502.09696），Qwen3-VL-235B-Thinking 拿到 4 分（vs 其他模型大多 1-3 分），这是相当显著的。

**VisuLogic**（Xu et al., 2025, https://arxiv.org/abs/2504.15279）上：Qwen3-VL-235B-Thinking 57.2，Gemini-2.5-Pro 31.6，GPT-5 28.5，Claude-Opus-4.1 27.9。这是 visual logical reasoning benchmark，差距悬殊。

### 5.2 Long Document（MMLongBench-Doc）

Qwen3-VL-235B-A22B-Instruct 57.0% / Thinking 56.2%，达到 SOTA。这验证了 256K context + 长文档数据 pipeline 的有效性。

### 5.3 3D Grounding（Table 2）

在 Omni3D 上（ARKitScenes、Hypersim、SUN RGB-D），Qwen3-VL-235B-A22B 显著超过 Gemini-2.5-Pro。SUN RGB-D 上 Thinking 39.4 vs Gemini 34.2，超 5.2 点。这说明 9-DoF 3D grounding 数据 + normalized coordinate 起作用了。

### 5.4 Fine-grained Perception with Tool

V*（Wu & Xie, 2024, https://arxiv.org/abs/2312.14135）上 Qwen3-VL-235B-Thinking + tool 拿 93.7，HRBench-4K 85.4，HRBench-8K 82.4。

**重要发现**：**tool 带来的提升 > 模型 size 提升**。在 Qwen3-VL family 内，加 tool 一致带来 ~5 点 V* 提升。这是 "scaling tool-integrated agentic learning is a highly promising path forward" 的实证。

### 5.5 Video Understanding

Qwen3-VL 8B 已经能和 Qwen2.5-VL 72B 竞争。这归功于 interleaved MRoPE + textual timestamp + temporally dense caption 三者结合。

Needle-in-a-Haystack（Figure 3）：30 分钟视频（256K token）100% accuracy，2 小时视频（1M token via YaRN extrapolation）99.5% accuracy。这非常 impressive。

### 5.6 Text-Centric（Table 5/6）

Qwen3-VL-235B-A22B-Instruct 在 AIME-25 74.7（超过 DeepSeek V3 0324 的 46.6），LiveCodeBench v6 54.3（超过 DeepSeek V3 的 45.2）。这意味着 VLM 已经在数学和代码上超过了同规模的纯 text LLM——**multimodal 训练没有 degrade text 能力，反而 enhance 了**。

Thinking 版本：AIME-25 89.7，HMMT-25 77.4，LiveCodeBench v6 70.1，超过 OpenAI o3 (medium) 的 88.9 / 77.5 / 58.6。这在 VLM 中是首次达到这种 reasoning level。

### 5.7 Ablation Study

#### Qwen3-ViT vs SigLIP-2（Table 11）
Qwen3-ViT 在 OmniBench（in-house holistic evaluation）上从 36.9 → 45.5，大幅提升。集成到 VLM 后，OCRB、AI2D、RLWRQA、InfoVQA、Omni 都提升。

#### DeepStack（Table 12）
15B-A2B LLM + 200B token pretraining，DeepStack 让 AVG 从 74.7 → 76.0。最大提升在 OCRB (81.0 → 83.6)、InfoVQA (71.9 → 74.2)、DocVQA (89.5 → 91.1)——全是 fine-grained 任务，符合 intuition。

---

## 六、关键 Intuition 总结

1. **Interleaved MRoPE 的本质**：让每个 spatial-temporal 轴都有"显微镜+望远镜"两种视角，长视频既不丢 short-term event，也不丢 long-term structure。

2. **DeepStack 的本质**：让 LLM 早期层就接触 raw 视觉细节，相当于在 LLM 内部做 hierarchical feature fusion，特别利好 OCR/DocVQA 这种需要 fine-grained 视觉的任务。

3. **Text-based timestamp 的本质**：把 temporal grounding 从 positional encoding 退化为 text understanding，利用 LLM 最强的能力处理时间，避免 RoPE 外推问题。

4. **Square-root reweighting 的本质**：在 token-level 和 sample-level loss 之间找平衡，既不让长样本主导，也不让短样本淹没。

5. **Multimodal Necessity Filtering 的本质**：确保训练样本真正需要看图，避免模型学到"通过文本 cue 就能猜答案"的 shortcut。

6. **Tool-integrated RL 的本质**：让模型学会 "when to zoom, where to zoom, how to reason about zoomed result"，三种 reward（answer、reasoning、tool count）共同防止 reward hacking。

7. **Strong-to-Weak Distillation 的本质**：用 text-only data distill 提升 reasoning，让小模型继承大模型的 reasoning chain，比单纯 SFT 更有效。

8. **VLM 不退化 text 的关键**：square-root reweighting + 保留大量 text-only 数据 + strong-to-weak distillation from text teacher。

---

## 七、个人观察与开放问题

1. **DeepStack 只用前 3 层**：为什么是 3 层？更多层会更好还是 over-saturate？这值得后续 ablation。

2. **Text timestamp 的 token overhead**：每个 frame group 一个 timestamp token，对超长视频（10K+ frames）会增加多少 context？这和 dense frame sampling 的 trade-off 是什么？

3. **Qwen3-ViT 的具体改动**：paper 没详细说 Qwen3-ViT 相比 SigLIP-2 改了什么，只说"continuous training with dynamic resolutions"。OmniBench 上从 36.9 → 45.5 是大跃升，背后机制值得挖掘。

4. **Thinking mode 的 inference cost**：max output length 设到 32K，AIME-25 等任务甚至 81,920 token。这种 thinking budget 在 production 中的 ROI 如何？

5. **Multimodal Necessity Filtering 的副作用**：可能过滤掉一些"虽然 text 可解但 multimodal 帮助 reasoning"的样本。这是一个 trade-off。

6. **256K native + 1M via YaRN**：YaRN 外推到 4x context 仍能保持 99.5% needle-in-haystack，这暗示 RoPE 的 frequency base 调整可能不需要重新训练。但 1M token 的 latency 和 cost 在 production 中是否可接受？

7. **Tool use > size scaling**：Table 2 中 Qwen3-VL-235B-Instruct + tool 在 V* 上 93.7，超过 Thinking 模式的 85.9。这暗示 agentic tool use 是比 model scaling 更高效的路径，呼应了 Sora / o1 时代的"reasoning + tool"范式。

---

## 八、参考链接汇总

- Qwen3-VL GitHub: https://github.com/QwenLM/Qwen3-VL
- HuggingFace Qwen: https://huggingface.co/Qwen
- ModelScope: https://modelscope.cn/organization/qwen
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- Qwen2-VL (MRoPE 原始): https://arxiv.org/abs/2409.12191
- SigLIP-2: https://arxiv.org/abs/2502.14786
- CoMP: https://arxiv.org/abs/2503.18931
- DeepStack: https://arxiv.org/abs/2411.16535
- TimeMarker: https://arxiv.org/abs/2411.18211
- SAPO: https://arxiv.org/abs/2511.20347
- Omni3D: https://arxiv.org/abs/2207.10660
- ZeroBench: https://arxiv.org/abs/2502.09696
- VisuLogic: https://arxiv.org/abs/2504.15279
- V*: https://arxiv.org/abs/2312.14135
- Charxiv: https://arxiv.org/abs/2406.18521
- MMLongBench-Doc: https://arxiv.org/abs/2502.04503

整体来看，Qwen3-VL 是一个 engineering-heavy 的工作，三个架构创新（Interleaved MRoPE、DeepStack、Text timestamp）都有清晰的理论 motivation 和 ablation 验证。最让我印象深刻的是 **multimodal training 不仅没退化 text，反而 enhance 了 reasoning**——这说明 text 和 vision 之间确实存在 synergy，square-root reweighting + necessity filtering 这两个"看似简单"的 data/loss 设计起了关键作用。
