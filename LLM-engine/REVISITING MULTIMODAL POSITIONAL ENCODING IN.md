---
source_pdf: REVISITING MULTIMODAL POSITIONAL ENCODING IN.pdf
paper_sha256: 1e2ccb95e0f30699dce66294d8b76fe17ebfb1e8762418f9e657da61a7c5d83b
processed_at: '2026-08-11T23:47:31-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，Andrej，我换一个更 conversational 的方式，把这篇 paper 的故事重新讲一遍。技术细节都在，但用更直觉的框架组织。

---

## 1. 这篇 paper 在纠结啥

VLM 里有个长期被忽视的小问题：**位置编码怎么处理多模态**。

LLM 用 RoPE 处理 1D text 序列，很成熟。但 VLM 同时要处理 1D text、2D image、3D video。RoPE 本质上是为 1D 设计的——每个 token 有一个标量 position id，每个 channel 对应一个频率。要扩展到多维，大家各显神通，搞出了一堆方法：MRoPE、VideoRoPE、HoPE、CircleRoPE、IL-RoPE、Omni-RoPE……每家都在自己关心的任务上刷点分，但没人系统比较过这些设计到底在干啥，为啥 work，为啥 fail。

这篇 paper 做的事情：把 multimodal RoPE 的设计空间拆成 3 个正交维度，系统做 ablation，发现 3 条 guidelines，落地成 2 个简单可用的变体（MHRoPE 和 MRoPE-I）。最有趣的 insight 是发现了一个叫 **spatial-reset** 的小技巧，几乎免费拿到一堆性能提升。

参考：
- 代码：https://github.com/JJJYmmm/Multimodal-RoPEs
- Qwen2-VL 原始 MRoPE paper：https://arxiv.org/abs/2409.12191
- RoPE 原始 RoFormer：https://arxiv.org/abs/2104.09864

---

## 2. 先回忆 RoPE 是干啥的

RoPE 的核心 idea：给每个 token 一个绝对 position id $m$，然后把 query/key 向量在 channel pair 维度做旋转，旋转角度是 $m \cdot \theta_i$。$\theta_i$ 是第 $i$ 对 channel 的频率，按几何级数从高到低排。

公式 1：
$$
S = (\mathcal{R}_m q)^\top (\mathcal{R}_n k) = q^\top \mathcal{R}_{n-m} k
$$

神奇之处在于：虽然编码时用的是绝对位置 $m$ 和 $n$，但 inner product 的正交性质让 attention score 只依赖相对距离 $n-m$。这是 RoPE 的"免费午餐"。

**频率谱的设计是关键**：channel 0 对应高频（捕捉近距离依赖），channel $d/2-1$ 对应低频（捕捉远距离依赖）。整个 channel 维度形成完整的频谱，覆盖多尺度依赖。这就是 RoPE 在 1D text 上成功的根本原因。

Intuition：你可以把 RoPE 想成给每个 token 一组"尺子"，从短尺子（高频）到长尺子（低频），测量它和其他 token 的距离。短尺子精细但测不远，长尺子测得远但精度低。一组尺子配合，覆盖各种距离尺度。

---

## 3. 多模态 RoPE 的三个 design axis

这篇 paper 把所有现有方法拆到三个正交维度上比较：

### Axis 1: Position Design（位置怎么分配）

**问题**：image 是 2D 的，video 是 3D 的，text 是 1D 的。怎么给这些 token 分配 position id？

#### 方案 A：拍扁成 1D（vanilla RoPE / V2PE）

最简单的做法——所有 token 排成一队，按顺序分配递增的 position id：
$$
m_i = m_{i-1} + s_{\text{mod}}
$$
- $m_i$：第 $i$ 个 token 的 position id
- $s_{\text{mod}}$：modality-specific 步长。vanilla RoPE 全部取 1，V2PE 给 visual token 取 $1, 1/2, \dots, 1/256$ 这种小步长

问题很明显：
1. image 的 2D 几何结构丢了——同一行的 token 和同一列的 token 在 1D 序列里距离完全不同，但 2D 上是对称的
2. 长视频下 position id 暴涨到几万，超出预训练 context window，模型 extrapolation 直接崩

#### 方案 B：保留 3D 结构（MRoPE 一族）

MRoPE 把 position id 从标量变成三元组 $(m^t, m^h, m^w)$，分别对应时间、高度、宽度。一个 visual block 内部的 token 按它的 spatio-temporal 位置分配坐标。

MRoPE 的更新规则（公式 2）：
$$
m_{\text{next}}^t = \max(m_{\text{prev}}^t, m_{\text{prev}}^h, m_{\text{prev}}^w) + 1
$$

意思是：下一个 modality block 的起点 = 前面所有维度坐标的最大值 + 1。这样保证 modality 之间 position id 不重叠。

#### 方案 B 的变种：Diagonal Layout（VideoRoPE / HoPE）

VideoRoPE 和 HoPE 觉得 MRoPE 不对称——视觉 frame 沿 t 堆叠，但 h/w 没动，他们想要 "inter-modal symmetry"，于是让 visual frame 沿 t/h/w 三个轴同时平移。

听起来 elegant，实际上是个坑。对于高分辨率文档（DocVQA、InfoVQA、ChartQA 这种），spatial 坐标可能延展到几千。这时后续生成的 text token 的 position id 会和前面 visual token 的 spatial 坐标 **重叠**。模型分不清"我现在生成的 token"和"前面 visual 的某个 spatial 位置"，导致 modalities confusion，failure mode 是无穷无尽的 text 重复（"1111..."）。

Table 4 ablation 直接证明：vanilla RoPE 在 DocVQA 上 82.94，加 diagonal layout 直接崩到 60.13。

#### 方案 B 的另一种变种：CircleRoPE

CircleRoPE 把 image token 排成环形，text 在线性轴上。好处：所有 visual token 离任意 text token 等距，attention 均匀。问题：
1. modality 之间间距太大，cross-modal 交互受阻
2. 没有 temporal axis，video frame 全部 collapse 到一个 ring 上，时间信息完全丢失

#### 本文的方案：MRoPE + Spatial-Reset

作者在分析 MRoPE 时发现一个有趣现象——**visual attention sink**：attention 集中在每张 image / video frame 的左上角（Figure 2）。这跟 LLM 中初始 token 吸引大量 attention 的现象是同构的。

LLM 中 attention sink 的本质：模型不是真的"喜欢"前几个 token，而是"喜欢"小 position id。预训练时 position id 从 0 开始，模型对"小 position id"形成了结构化偏好。

MRoPE 的更新规则让 visual block 内部的 spatial 坐标实际是 $t + h$ 这种形式（因为下一个 modality 起点用 $\max$ 推进，spatial 坐标被累加进 temporal 主轴）。后面的 visual content 起点坐标很大，远离 LLM 的"舒适区"。

**Spatial-reset 的 idea**：每个 visual content 的 spatial 坐标从 (0, 0) 重新开始，不再累加。

数学上看，公式 3 vs 公式 4：

MRoPE 下，物体在 $t_1$ 位于 $(h_1, w_1)$，在 $t_2$ 位于 $(h_2, w_2)$，绝对位置：
$$
m_1 = (t_1, t_1 + h_1, t_1 + w_1), \quad m_2 = (t_2, t_2 + h_2, t_2 + w_2)
$$

相对位置：
$$
m_{\text{rel}} = (t_2 - t_1, (t_2 - t_1) + (h_2 - h_1), (t_2 - t_1) + (w_2 - w_1))
$$

时间差"污染"了空间相对距离——三个轴纠缠在一起。

Spatial-reset 下：
$$
m_1 = (t_1, h_1, w_1), \quad m_2 = (t_2, h_2, w_2)
$$

相对位置：
$$
m_{\text{rel}}' = (t_2 - t_1, h_2 - h_1, w_2 - w_1)
$$

三个轴**完全解耦**。物体从 $(h_1, w_1)$ 移动到 $(h_2, w_2)$ 的相对位移是纯空间向量，时间差是纯时间向量，运动表示极其自然。

**Intuition**：spatial-reset 相当于给每个 visual content 一个"局部 attention sink"。LLM 喜欢小 position id，那我们就让每个 image / video frame 都从 (0, 0) 开始，每次都重新触发模型对小 position id 的偏好，相当于给每个 visual content 一个"重启"。

Table 7 直接验证：spatial-reset 后，layer 28 的 visual attention score 从 9.93% 升到 19.00%（MHRoPE）/ 23.23%（MRoPE-I），深层网络对视觉内容的关注度翻倍。

### Axis 2: Frequency Allocation（频率怎么分给各轴）

**问题**：RoPE 的 channel 维度有 $d/2$ 个 frequency band，从高频到低频。3 个 position 轴（t/h/w）怎么分这些 channel？

#### MRoPE 的简单分块

最直观的做法：把 $d$ 维 channel 等分成 3 块，前 $d/3$ 给 t，中间 $d/3$ 给 h，后 $d/3$ 给 w。

因为 $\theta_i = \text{base}^{-2i/d}$ 是 $i$ 的减函数：
- 前 $d/3$ 维是高频 → t 只占高频 → 时间 attention 随距离衰减过快 → 长视频理解差
- h 占中频，w 占低频 → 二者衰减率不同 → spatial 关系学习不对称
- 每个轴只有 $d/6$ 个 frequency band → 频率分辨率粗化 3 倍

Figure 4a 显示 MRoPE 三条曲线（t/h/w）衰减率完全不同，t 衰减最快。

#### VideoRoPE / HoPE 的"反向偏置"

为了修长视频问题，把时间轴挪到低频。代价：spatial 轴被挤到高频，无法捕捉多尺度空间关系，grounding 任务（RefCOCO 这种需要精细空间理解的）显著退化。

这就像跷跷板：压下视频一头，弹起 grounding 一头。

#### Long-Range Decay 的形式化（公式 5、6）

为啥 frequency allocation 这么重要？作者给了漂亮的理论推导。

RoPE dot product 用复数形式：
$$
(\mathcal{R}_m q)^\top (\mathcal{R}_n k) = \text{Re}\left[\sum_{i=0}^{d/2-1} (q_{[2i:2i+1]} \cdot k_{[2i:2i+1]}^*) e^{i(m-n)\theta_i}\right]
$$

- $q_{[2i:2i+1]}$：query 的第 $i$ 对 channel，视为复数
- $k_{[2i:2i+1]}^*$：key 第 $i$ 对 channel 的复共轭
- $h_i = q_{[2i:2i+1]} \cdot k_{[2i:2i+1]}^*$：content-dependent 项
- $S_j = \sum_{k=0}^{j-1} e^{i(m-n)\theta_k}$：position-dependent partial sum

用 **summation by parts**（阿贝尔变换）：
$$
\sum_{i=0}^{d/2-1} h_i e^{i(m-n)\theta_i} = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)
$$

边界条件 $S_0 = 0$, $h_{d/2} = 0$ 让两端项消失。取绝对值：
$$
\left|\sum h_i e^{i(m-n)\theta_i}\right| \leq \max |h_{i+1} - h_i| \cdot \sum |S_{i+1}|
$$

**关键**：attention score 的上界 = content-dependent scaling × position-dependent decay。**Long-range decay 性质完全由 $\sum |S_{i+1}|$ 决定**，与内容无关。

平均值 $\frac{1}{d/2}\sum_{i=1}^{d/2} |S_i|$ 作为 long-range decay 指标。MRoPE 把时间轴放在高频，意味着时间维度对应的 $S_j$ 项随 $|m-n|$ 增长迅速饱和（高频下 $e^{i(m-n)\theta}$ 翻转快，partial sum 不增长），导致时间 attention 快速衰减。

#### 本文方案 1：MHRoPE（Multi-Head Allocation）

灵感来自 partial RoPE（Barbero et al. 2025）：RoPE 在 channel 维度有冗余。本文假设 head 维度也有冗余。

设计：不同 attention head 分配给不同 position 轴。比如 head 0-7 编码 t，head 8-15 编码 h，head 16-23 编码 w。每个 head 内部所有 channel 用完整 frequency spectrum。

好处：
1. 每个轴都保留完整 frequency 分辨率
2. 可扩展——未来加新 axis 多分几个 head 就行
3. 衰减曲线对称（Figure 4b 三条曲线重合）

劣势：head-level partition 阻止不同轴在同一个 head 内做 cross-axis attention fusion，轻微性能损失。Table 5 显示 Multi-Head 64.63 vs Interleave 64.95。

#### 本文方案 2：MRoPE-I（Interleaved Allocation）

设计：channel 以 round-robin 方式交错分配。channel 0 给 t，channel 1 给 h，channel 2 给 w，channel 3 给 t，channel 4 给 h...每个轴都能用从 high 到 low 的完整 frequency spectrum。

Ablation Table 8 测了不同分配比例：
- t:h:w = 0:32:32（无 temporal）：image 66.42 / video 51.01 / grounding 76.02 / overall 64.48
- t:h:w = 24:20:20：image 66.65 / video 52.36 / grounding 75.85 / overall **64.95**（最优）
- t:h:w = 48:8:8（重 temporal）：image 65.06 / video 51.17 / grounding 72.87 / overall 63.03

t 占比适度高于 h/w 最好（temporal range 通常大），但过度倾斜会损害 spatial grounding。

好处：
1. 每个轴完整 frequency spectrum
2. YaRN/NTK-aware 兼容——这些 extrapolation 算法本质是 rescale frequency spectrum，interleave 让 rescaling 边界清晰
3. 实现简单，只是 channel index 重排

Table 6 验证 YaRN 兼容性：MRoPE + YaRN 在 LVBench 上甚至下降（41.5 → 41.2），MRoPE-I + YaRN 显著提升（42.0 → 43.6）。

**Intuition**：MRoPE 是 FDM（频分复用，每个轴独占一段频段），MHRoPE 是某种空间分集（每个 head 一个轴），MRoPE-I 是某种交错 TDM（时分复用）。interleave 让"分辨率降低"变成"采样步长增加"，对每个轴而言 frequency spectrum 仍然完整。

### Axis 3: Compatibility with Text-only RoPE

这条最容易被忽视。VLM 是从 LLM 初始化的，LLM 用 vanilla RoPE 训练了海量 text 数据。如果 VLM 中 text token 的 position encoding 偏离 vanilla RoPE，预训练知识传递就被破坏。

作者测试了两个"看似合理但实际有害"的修改：

**修改 1：Text spatial-reset**（IL-RoPE / Omni-RoPE 风格）
把 text token 的 spatial 坐标也设为 0，让 text "脱离"空间维度。Table 4 显示 image 从 65.69 降到 58.27，grounding 从 73.48 降到 68.20。

**修改 2：Scaling rotary base for spatial**
spatial 坐标范围比 temporal 小很多，直觉上应该用更小的 base（10000 而非 1000000）来更好编码 spatial。Table 4 显示 image 从 65.69 降到 60.15。

**Intuition**：LLM 在 base=1000000 下学到的 query/key 旋转模式是一种"知识"。任何偏离都让模型进入"未训练区域"。即便逻辑上更合理，pre-trained LLM 的 inductive bias 不能动。

这跟 LoRA、adapter 的精神一致——fine-tuning 大模型时，能不动 pre-trained 机制就不动，只增加新机制。

---

## 4. 两个方法的总结

### MHRoPE
- Position Design：MRoPE + spatial-reset
- Frequency Allocation：head-level partition（不同 head 分配给不同轴）
- Text Compatibility：保持 vanilla RoPE
- 适合：未来需要扩展更多 position axis（如 3D 场景、audio）

### MRoPE-I（作者更推荐）
- Position Design：MRoPE + spatial-reset
- Frequency Allocation：interleaved round-robin
- Text Compatibility：保持 vanilla RoPE
- 适合：当前 VLM 标准配置，实现简单，YaRN 兼容

两者都启用 spatial-reset。区别只在 frequency allocation：MHRoPE 按 head 切分，MRoPE-I 按 channel 交错。MRoPE-I 性能略优且实现更简单，作者更推荐。

---

## 5. 实验里最 striking 的几个点

### 5.1 主表（Table 2）的关键数字

MRoPE-I 相对 vanilla RoPE 的提升：
- MMMU +2.67（50.56 → 53.22）
- ChartQA +5.28（56.84 → 62.12）
- RefCOCO_val +3.27（77.67 → 80.94）
- Overall Image +1.62（62.17 → 63.79）
- Overall Grounding +2.37（73.48 → 75.85）

### 5.2 VideoRoPE / HoPE 在文档任务上的崩溃

Table 2 里 VideoRoPE 在 DocVQA 只有 60.13（vanilla RoPE 是 82.94），InfoVQA 只有 37.42（vanilla 是 58.85）。这就是 diagonal layout 的 position overlap 导致的 modalities confusion。文章定性分析了 failure mode——无穷无尽的 "1111..." 重复。

### 5.3 跨架构泛化（Table 3）

在 Qwen3-VL-4B 和 8B 上重复实验。Qwen3-VL 与 Qwen2.5-VL 有显著架构差异：移除 vision encoder 的 window attention、引入 DeepStack、加入 QK-Norm、4B 变体还有 weight tying。结果 MHRoPE / MRoPE-I 仍稳定领先，证明方法与具体架构解耦。

### 5.4 Visual Attention 翻倍（Table 7）

spatial-reset 后，layer 28 的 visual attention score 从 9.93% 升到 19.00%（MHRoPE），几乎翻倍。深层网络对视觉内容的关注度大幅提升。这直接验证了 spatial-reset 触发"局部 attention sink"的假设。

### 5.5 Long-context Extrapolation（Figure 5 + Table 6）

256K context 上：
- Vanilla RoPE 128K/256K 急剧退化（position id 增长过快）
- VideoRoPE / HoPE 在长视频上略好（低频分配给 temporal）
- MRoPE-I + YaRN 在 LVBench 上从 42.0 提升到 43.6，证明 interleave 设计与 YaRN 兼容

---

## 6. 我的几个 intuition

### 6.1 Spatial-Reset 的本质：给每个 visual content 一个"局部 sink"

LLM 预训练时 position id 从 0 开始，模型对"小 position id"有结构化偏好（attention sink）。MRoPE 的累加让后面的 visual content 起点坐标很大，远离 LLM 的"舒适区"。

spatial-reset 把每个 visual content 拉回 (0,0) 起点，相当于让每个 image / video frame 都有自己的"局部 sink"。这跟 StreamingLLM 中显式加 sink token 的思路是同构的——都是利用模型对小 position id 的偏好。

参考：
- StreamingLLM attention sink：https://arxiv.org/abs/2309.17453

### 6.2 Position Overlap 是隐性 namespace bug

Diagonal layout 的失败很 instructive。设计 position encoding 时容易陷入"几何 elegant"的陷阱，但忽略了一个朴素的事实：position id 是一个 namespace，必须有 unique 解析。

如果 visual 的 spatial 坐标和后续 text 的 temporal 坐标可能 overlap，模型就没有先验区分它们属于哪个 modality。这跟编程语言里 namespace pollution 类似——一个变量名在不同 scope 下指向不同对象是 OK 的，但同一个 scope 下必须 unique。

MRoPE 的 $\max$ update rule 保证了 namespace 唯一性，diagonal layout 破坏了它。

### 6.3 Pre-trained Prior 是一种"知识"

Text RoPE 兼容性这条 guideline 提醒我们：VLM 是 LLM 的"扩展"，不是从零训练。任何对 text encoding 的修改都在"撬动"已经学好的知识。

即便修改在逻辑上更合理（如更小的 rotary base 给 spatial），pre-trained LLM 的 inductive bias 是一种"知识"，逻辑优化与 inductive bias 冲突时，必须让位于 inductive bias。

这个 insight 跨任务普遍适用：fine-tuning 大模型时，能不动 pre-trained 机制就不动，只增加新机制。

### 6.4 还能改进啥

我自己看完后的几个 open question：

1. **Layer-wise 混合**：能否前几层用 MHRoPE（让 head 内部充分做 axis-specific attention），后几层切到 MRoPE-I（让 cross-axis fusion 更灵活）？

2. **Spatial-reset 与 long video**：spatial-reset 对每个 frame 都重置，同一物体在 frame 100 和 frame 200 的 spatial 坐标都是 $(h, w)$，相对位置 $(0, 0, 0)$——但实际时间差是 100。这是否会让模型忽略长时间静止物体的运动？Table 2 的 video benchmark 上 MRoPE-I 略低于 VideoRoPE/HoPE 似乎暗示这个问题。

3. **Attention sink 的更深利用**：既然 spatial-reset 触发了"局部 sink"，能否显式在 visual content 开头加一个 sink token？

4. **Frequency allocation 的 data-adaptive 版本**：当前 24:20:20 是固定比例。能否设计 learnable 的 allocation，让模型根据 input 自动调整？多图场景给 spatial 多分配，长视频场景给 temporal 多分配。

5. **与 3D scene / embodied AI 的衔接**：如果未来 VLM 要处理 3D 点云、深度信息、机器人 trajectory，position axis 会扩展到 6+ 个。MHRoPE 的 head-level partition 在这种场景下更 scalable。

参考：
- Llama 3 herd：https://arxiv.org/abs/2407.21783
- Qwen3 tech report：https://arxiv.org/abs/2505.09388
- Mogao (interleaved multimodal generation)：https://arxiv.org/abs/2505.05472
- Apollo (video VLM)：https://arxiv.org/abs/2412.10360
- Cambrian-1：https://arxiv.org/abs/2406.16860
- Jianlin Su 多模态位置编码博客：https://spaces.ac.cn/archives/10040
- Partial RoPE (Barbero et al.)：https://openreview.net/forum?id=GtvuNrk58a
- YaRN：https://arxiv.org/abs/2309.00071
- V2PE：https://arxiv.org/abs/2412.09616
- VideoRoPE：https://arxiv.org/abs/2502.11664
- HoPE：https://arxiv.org/abs/2505.20444
- CircleRoPE：https://arxiv.org/abs/2505.16416
- OmniGen2：https://arxiv.org/abs/2506.18871
- NTK-aware scaling Reddit 帖：https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

---

## 7. 一句话总结

这篇 paper 把 multimodal RoPE 的设计空间系统拆成 3 个正交 axis（position design / frequency allocation / text compatibility），用大量 ablation 提炼出 3 条 guidelines，并落地为 2 个 plug-and-play 变体。最有趣的 insight 是 spatial-reset——给每个 visual content 重置 spatial 坐标，触发 LLM 对小 position id 的偏好，相当于每个 image/video frame 都有自己的"局部 attention sink"。代码开源在 https://github.com/JJJYmmm/Multimodal-RoPEs ，可以直接 plug 进任何基于 Qwen2.5-VL / Qwen3-VL 的训练 pipeline。如果你在搭 VLM，**直接用 MRoPE-I + spatial-reset**，配置简单，跨架构泛化好，YaRN 兼容，全 benchmark 稳定提升。

---

# Revisiting Multimodal Positional Encoding in Vision-Language Models 深度解读

Andrej, 这篇 paper 来自 Qwen Team（Alibaba）+ ICT 联合工作，是首个系统性拆解 multimodal RoPE 设计的工作。我会把整个 paper 的逻辑链条、数学细节、实验证据、以及我自己对其底层 intuition 的思考一并铺开。

---

## 1. 这篇 paper 的核心问题

VLM 里 LLM backbone 几乎都基于 RoPE（rotary positional embedding），但 VLM 同时处理 1D text、2D image、3D video。如何让 RoPE 这种"标量位置 + 频率衰减"的设计 natural 地扩展到多模态、多维度？现有方案分成两派：

- **1D sequential 派**：vanilla RoPE / V2PE 直接 flatten，丢弃几何结构。
- **Multi-dimensional 派**：MRoPE / VideoRoPE / HoPE / CircleRoPE / IL-RoPE / Omni-RoPE 各自做 t-h-w 分块，但每家都偏一边——有的偏 video 长序列，有的偏 image grounding，有的偏 generation，缺乏统一 framework。

这篇文章做的事情是：把 multimodal RoPE 拆成 3 个正交 design axis（position design、frequency allocation、text compatibility），系统 ablation，提炼出 3 条 guidelines，并落地为 MHRoPE / MRoPE-I 两个 plug-and-play 变体。

参考链接：
- Paper arXiv（搜索关键词）：https://arxiv.org/abs/ (在 ICLR 2026 投稿)
- 代码仓库：https://github.com/JJJYmmm/Multimodal-RoPEs
- Qwen2.5-VL tech report：https://arxiv.org/abs/2502.13923
- Qwen2-VL MRoPE 原始 paper：https://arxiv.org/abs/2409.12191
- RoPE 原始论文 RoFormer：https://arxiv.org/abs/2104.09864
- V2PE：https://arxiv.org/abs/2412.09616
- VideoRoPE：https://arxiv.org/abs/2502.11664
- HoPE：https://arxiv.org/abs/2505.20444
- CircleRoPE：https://arxiv.org/abs/2505.16416
- YaRN：https://arxiv.org/abs/2309.00071
- NTK-aware scaling Reddit 帖：https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
- Partial RoPE (Barbero et al.)：https://openreview.net/forum?id=GtvuNrk58a

---

## 2. RoPE 的数学回顾（公式 1）

给定 query $\pmb{q}$ 在位置 $m$，key $\pmb{k}$ 在位置 $n$：

$$
S = (\pmb{\mathcal{R}}_m \pmb{q})^\top (\pmb{\mathcal{R}}_n \pmb{k}) = \pmb{q}^\top \pmb{\mathcal{R}}_m^\top \pmb{\mathcal{R}}_n \pmb{k} = \pmb{q}^\top \pmb{\mathcal{R}}_{n-m} \pmb{k}
$$

变量含义：
- $\pmb{\mathcal{R}}_m \in \mathbb{R}^{d \times d}$：block-diagonal rotation matrix，每个 2×2 block 是 $\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$。
- $\theta_i = \text{base}^{-2i/d}$，$i \in [0, d/2-1]$：第 $i$ 对 channel 的角频率，base 通常取 10000 或 1000000。$i$ 小 → $\theta_i$ 大（高频，捕捉短距离），$i$ 大 → $\theta_i$ 小（低频，捕捉长距离）。
- $d$：head dimension。
- $m$：绝对 position id。
- $n - m$：相对距离，最终 attention score 只依赖相对距离——这就是 RoPE 的"相对位置免费午餐"。

**Intuition**：RoPE 把绝对 position 编码进 query/key 的旋转中，但 inner product 的几何性质自动 collapse 出相对距离。channel 维度形成了一个完整的"频谱"（frequency spectrum），从高频到低频覆盖多尺度依赖。这是 vanilla RoPE 在 1D text 上成功的根本原因。

---

## 3. 三个 Design Axis 的系统拆解

### 3.1 Position Design（位置设计）

#### 3.1.1 1D Sequential（vanilla RoPE / V2PE）

$$
m_i = m_{i-1} + s_{\text{mod}}
$$

- $m_i$：第 $i$ 个 token 的 position id。
- $s_{\text{mod}}$：modality-specific 的步长。vanilla RoPE 取 $s = 1$，V2PE 取 $s_{\text{Visual}} \in \{1, 1/2, \dots, 1/256\}$。

缺点：
1. 丢弃 visual content 的 3D 几何结构（image 被拍扁成 token sequence，spatial 信息变成"离起点多远"）。
2. 长视频下 position id 暴涨，超出预训练 context length，extrapolation 性能急剧退化（见 Appendix D.4，vanilla RoPE 在 128K/256K 上崩盘）。

#### 3.1.2 Multi-dimensional（MRoPE / VideoRoPE / HoPE / CircleRoPE）

MRoPE 把 position identifier 从标量扩展为三元组：
$$
\pmb{m}_i = (m_i^t, m_i^h, m_i^w)
$$
- $m^t$：temporal（时间）坐标
- $m^h$：vertical（高度）坐标
- $m^w$：horizontal（宽度）坐标

MRoPE 的更新规则（公式 2）：
$$
m_{\text{next}}^t = \max(m_{\text{prev}}^t, m_{\text{prev}}^h, m_{\text{prev}}^w) + 1
$$

这个 $\max$ 操作确保 modality 之间没有 position overlap。下一个 modality 的起点是"前面所有维度最大值 +1"，相当于跳过整个 visual block。

#### 3.1.3 Diagonal Layout 的失败（VideoRoPE / HoPE）

VideoRoPE 和 HoPE 为了 inter-modal symmetry，把视觉 frame 沿 t/h/w 三个轴同时平移（diagonal layout）。看起来 elegant，但隐藏 bug：

对于高分辨率文档（如 DocVQA、InfoVQA、ChartQA），spatial coordinate 的 range 可能扩展到几千。当 visual content 占用的 spatial range 超出其 temporal id 时，后续生成的 text token 的 position id 会与前面 visual token 的 spatial coordinate **重叠**。模型无法区分"我现在生成的 token"和"前面 visual 的某个 spatial 位置"。

文章把这个现象命名为 **modalities confusion in generation**，对应的 failure mode 是无穷无尽的 text 重复，比如 "1111..."。Table 4 的 ablation 直接证明：
- vanilla RoPE DocVQA: 82.94
- + diagonal layout: 60.13（崩了 22 个点）
- + modality interval（强行拉大 modality 间距）: 70.43

#### 3.1.4 CircleRoPE 的限制

CircleRoPE 把 image token 排成环形，正交于 text 的线性轴。好处：所有 visual token 离任意 text token 等距，理论上 attention 均匀。问题：
1. modality interval 过大，cross-modal 交互被阻碍。
2. 没有 temporal axis，video frame 全部 collapse 到一个 ring 上，temporal 信息完全丢失。Table 2 里 CircleRoPE 在 MLVU/LVBench/VideoMME 都很差。

#### 3.1.5 Spatial-Reset——本文的关键 insight

作者发现 MRoPE 存在 **visual attention sink**：attention 集中在每张 image / video frame 的左上角（Figure 2）。这与 LLM 中初始 token 吸引大量 attention 的现象同构。

提出 **spatial-reset**：每个 visual content 的 spatial 坐标都从 (0, 0) 重新开始。

**数学动机**（公式 3 vs 公式 4）：

MRoPE 下，token 在 $t_1$ 时位于 $(h_1, w_1)$，在 $t_2$ 时位于 $(h_2, w_2)$。绝对 position 是：
$$
\pmb{m}_1 = (t_1, t_1 + h_1, t_1 + w_1), \quad \pmb{m}_2 = (t_2, t_2 + h_2, t_2 + w_2)
$$

注意这里 $m^h = t + h$ 是因为 MRoPE 更新规则中下一个 modality 的起点用 $\max$ 推进，所以视觉 block 内部的 spatial 坐标实际是 $t + h$ 这种"耦合"形式。

相对位置（公式 3）：
$$
\pmb{m}_{\text{rel}} = (t_2 - t_1, (t_2 - t_1) + (h_2 - h_1), (t_2 - t_1) + (w_2 - w_1))
$$

时间位移和空间位移纠缠在一起——temporal 距离"污染"了 spatial 相对距离。

spatial-reset 下：
$$
\pmb{m}_1 = (t_1, h_1, w_1), \quad \pmb{m}_2 = (t_2, h_2, w_2)
$$

相对位置（公式 4）：
$$
\pmb{m}_{\text{rel}}' = (t_2 - t_1, h_2 - h_1, w_2 - w_1)
$$

t、h、w 三个轴**完全解耦**。这对于"运动表示"极其自然——一个物体从 $(h_1, w_1)$ 移动到 $(h_2, w_2)$ 的相对位移是纯空间向量，时间差是纯时间向量。

**Intuition**：MRoPE 的 update rule 隐式把"前面 token 用过的 position id 范围"沿时间方向累加，导致 spatial 坐标实质上变成了 $t + h$。spatial-reset 强行截断这个累加，让每个 visual content 拥有独立的 spatial 坐标系。这同时利用了 LLM 的 "small position id 偏好"（attention sink），相当于给每个 visual content 一个"局部 sink"。

Table 7 直接验证：spatial-reset 后，layer 28 的 visual attention score 从 9.93% 提升到 19.00%（MHRoPE）/ 23.23%（MRoPE-I），深层网络对视觉内容的关注度翻倍。

### 3.2 Frequency Allocation（频率分配）

#### 3.2.1 Standard MRoPE 的偏置

MRoPE 把 $d$ 维 channel 简单分成 3 个 contiguous block：
- 前 $d/3$ 维给 $t$
- 中间 $d/3$ 维给 $h$
- 后 $d/3$ 维给 $w$

因为 $\theta_i = \text{base}^{-2i/d}$ 是 $i$ 的减函数，前 $d/3$ 维是高频，后 $d/3$ 维是低频。后果：
1. **时间轴只占高频** → 时间 attention 随距离衰减过快，长视频理解差。
2. **h 和 w 占不同频段** → 比如 h 占中频，w 占低频，二者衰减率不同，spatial 关系学习不对称（Figure 4a 显示 h 和 w 的 long-range decay 曲线分叉）。
3. **频率分辨率降低** → 原本 $d/2$ 个 frequency band 全分给 1 个轴，现在每个轴只有 $d/6$，分辨率粗化 3 倍。

#### 3.2.2 VideoRoPE / HoPE / IL-RoPE 的"反向偏置"

这些方法为了修长视频问题，把时间轴挪到低频。代价：spatial 轴被挤到高频，无法捕捉多尺度空间关系，grounding 任务（RefCOCO）显著退化。

#### 3.2.3 Long-Range Decay 的形式化推导（公式 5、6）

这是 paper 里很漂亮的推导。RoPE dot product 用复数形式表示：

$$
(\pmb{\mathcal{R}}_m \pmb{q})^\top (\pmb{\mathcal{R}}_n \pmb{k}) = \text{Re}\left[\sum_{i=0}^{d/2-1} (\pmb{q}_{[2i:2i+1]} \cdot \pmb{k}_{[2i:2i+1]}^*) e^{i(m-n)\pmb{\theta}_i}\right]
$$

- $\pmb{q}_{[2i:2i+1]}$：query 的第 $i$ 对 channel（视为复数 $q_{2i} + i \cdot q_{2i+1}$）。
- $\pmb{k}_{[2i:2i+1]}^*$：key 第 $i$ 对 channel 的复共轭。
- $e^{i(m-n)\theta_i}$：第 $i$ 对 channel 的旋转因子。
- $h_i = \pmb{q}_{[2i:2i+1]} \cdot \pmb{k}_{[2i:2i+1]}^*$：content-dependent 项。
- $S_j = \sum_{k=0}^{j-1} e^{i(m-n)\theta_k}$：position-dependent partial sum。

通过 **summation by parts**（阿贝尔变换）：
$$
\sum_{i=0}^{d/2-1} h_i e^{i(m-n)\theta_i} = -\sum_{i=0}^{d/2-1} S_{i+1}(h_{i+1} - h_i)
$$

边界条件 $S_0 = 0$, $h_{d/2} = 0$ 让两端项消失。

取绝对值，三角不等式：
$$
\left|\sum h_i e^{i(m-n)\theta_i}\right| \leq \max |h_{i+1} - h_i| \cdot \sum_{i=0}^{d/2-1} |S_{i+1}|
$$

**关键洞察**：attention score 的上界 = content-dependent scaling × position-dependent decay。**Long-range decay 性质完全由 $\sum |S_{i+1}|$ 决定**，与内容无关。

这就解释了为什么 frequency allocation 设计直接决定 long-range 行为：MRoPE 把时间轴放在高频 channels，意味着时间维度对应的 $S_j$ 项随 $|m-n|$ 增长迅速饱和（高频下 $e^{i(m-n)\theta}$ 翻转快，partial sum 不增长），导致时间 attention 快速衰减。

平均值 $\frac{1}{d/2}\sum_{i=1}^{d/2} |S_i|$ 作为 long-range decay 的指标。Figure 4a 显示 MRoPE 三条曲线（t/h/w）衰减率完全不同，Figure 4b 显示 MHRoPE 和 MRoPE-I 三条曲线**完美重合**且衰减更平缓。

#### 3.2.4 MHRoPE：Multi-Head Allocation

灵感来自 partial RoPE（Barbero et al. 2025）：RoPE 在 channel 维度有冗余。本文假设 head 维度也有冗余。

设计：
- 不同 attention head 分配给不同 position 轴
- 比如 head 0-7 编码 t，head 8-15 编码 h，head 16-23 编码 w
- 每个 head 内部所有 channel 都用完整 frequency spectrum

优势：
1. **每个轴都保留完整 frequency 分辨率**，没有 channel 分割带来的分辨率损失。
2. **可扩展**：未来 VLM 如果要加新 axis（比如 depth、audio time），多分几个 head 即可，不用挤占现有 channel。
3. **衰减曲线对称**（Figure 4b）：t/h/w 三条曲线重合，衰减率一致。

劣势：head-level partition 阻止不同轴在同一个 head 内做 cross-axis attention fusion，轻微性能损失（Table 5：Multi-Head 64.63 vs Interleave 64.95）。在分布式训练（tensor parallel）下实现也更复杂。

#### 3.2.5 MRoPE-I：Interleaved Allocation

设计：channel 以 round-robin 方式交错分配：
- channel 0 → t
- channel 1 → h
- channel 2 → w
- channel 3 → t
- channel 4 → h
- ...

每个轴都能用从 high 到 low 的完整 frequency spectrum。

Ablation Table 8 显示最优比例 t:h:w = 24:20:20（不是均匀的 21:21:21），因为 temporal 范围通常更大，多分一点 channel 给 t 有利。但过度倾斜（32:16:16、48:8:8）会损害 grounding——spatial 高频不足。

优势：
1. **完整 frequency spectrum** 对每个轴开放。
2. **YaRN/NTK-aware 兼容**：这些 extrapolation 算法本质是 rescale frequency spectrum，MRoPE-I 的均匀分布让 rescaling 边界清晰。Table 6 验证：MRoPE + YaRN 在 LVBench 上甚至下降（41.5 → 41.2），而 MRoPE-I + YaRN 显著提升（42.0 → 43.6）。
3. **实现简单**：相比 MHRoPE 的 head partition，interleave 只是 channel index 重排。

### 3.3 Compatibility with Text-only RoPE

这条 guideline 最容易被忽视。VLM 是从 LLM 初始化的，LLM 用 vanilla RoPE 训练了海量 text 数据。如果 VLM 中 text token 的 position encoding 偏离 vanilla RoPE，预训练知识传递就被破坏。

文章测试了两个"看似合理但实际有害"的修改：

1. **Text spatial-reset**（IL-RoPE / Omni-RoPE 风格）：把 text token 的 spatial 坐标也设为 0。Table 4 显示 image 性能从 65.69 降到 58.27，grounding 从 73.48 降到 68.20。

2. **Scaling rotary base for spatial**：因为 spatial 坐标范围比 temporal 小很多，直觉上应该用更小的 base（比如 10000 而非 1000000）来更好地编码 spatial。Table 4 显示 image 从 65.69 降到 60.15。

**Intuition**：这跟 LLM pre-training 的"先验分布"有关。LLM 在 base=1000000 下学到的 query/key 旋转模式是一种"知识"，任何偏离都让模型进入"未训练区域"。即便逻辑上更合理，pre-trained LLM 的 inductive bias 不能动。

---

## 4. 提出的方法总结

### MHRoPE
- Position Design：MRoPE + spatial-reset
- Frequency Allocation：head-level partition
- Text Compatibility：保持 vanilla RoPE
- 推荐场景：未来需要扩展更多 position axis 时（如 3D 场景、audio）

### MRoPE-I（作者更推荐）
- Position Design：MRoPE + spatial-reset
- Frequency Allocation：interleaved round-robin
- Text Compatibility：保持 vanilla RoPE
- 推荐场景：当前 VLM 标准配置，实现简单，YaRN 兼容

### Spatial-Reset
通用机制，独立于 frequency allocation 选择，每个 visual content 重置 spatial 坐标。两个变体都启用。

---

## 5. 实验细节

### 5.1 训练 setup
- Backbone：Qwen2.5 7B LLM
- Vision encoder：QwenViT（frozen）
- Connector：unfrozen
- Optimizer：AdamW，α=0.9，β=0.98，weight decay=0.05
- LR：cosine decay，1e-5 → 3e-6
- Batch size：128
- Context length：32K
- Rotary base：1000000
- 训练数据：2M SFT samples
- 算力：512 A100 GPU hours per experiment

### 5.2 主表（Table 2）关键发现

| Benchmark | Vanilla RoPE | MRoPE | VideoRoPE | MHRoPE | MRoPE-I | MRoPE-I 提升 |
|-----------|--------------|-------|-----------|--------|---------|--------------|
| MMMU | 50.56 | 50.22 | 49.89 | 53.00 | **53.22** | +2.67 |
| ChartQA | 56.84 | 63.56 | 54.88 | 62.44 | 62.12 | +5.28 |
| DocVQA | 82.94 | 81.49 | 60.13 | 81.32 | **83.72** | +0.78 |
| InfoVQA | 58.85 | 52.96 | 37.42 | 52.01 | **58.24** | -0.61 |
| RefCOCO_val | 77.67 | 78.35 | 77.95 | 79.87 | **80.94** | +3.27 |
| MLVU | 64.69 | 63.26 | 66.05 | 65.69 | 65.46 | +0.77 |
| Overall Image | 62.17 | 61.90 | 57.03 | 62.92 | **63.79** | +1.62 |
| Overall Video | 51.64 | 51.51 | 52.18 | 52.58 | 52.36 | +0.72 |
| Overall Grounding | 73.48 | 73.69 | 72.59 | 74.92 | **75.85** | +2.37 |

观察：
- VideoRoPE / HoPE 在 image/document benchmark 严重退化（diagonal layout 的 position overlap 病态）。
- MRoPE 整体偏弱，因为 frequency spectrum 分块导致每个轴分辨率粗。
- MHRoPE / MRoPE-I 在所有任务类别都稳定提升。

### 5.3 跨架构泛化（Table 3）

在 Qwen3-VL-4B 和 8B 上重复实验，这两个模型与 Qwen2.5-VL 有显著架构差异：
- 移除 vision encoder 的 window attention
- 引入 DeepStack 架构
- LLM backbone 加入 QK-Norm
- 4B 变体 embedding 与 LM head weight tying

结果：MHRoPE / MRoPE-I 仍稳定领先，证明方法与具体架构解耦。

### 5.4 Ablation 详解

#### Position Design（Table 4）

| Position Design | Image | Grounding | Video | DocVQA | InfoVQA | ChartQA |
|-----------------|-------|-----------|-------|--------|---------|---------|
| vanilla RoPE | 65.69 | 73.48 | 51.64 | 82.94 | 58.85 | 56.84 |
| + 3D structure | 65.87 | 74.40 | 51.29 | 82.33 | 57.24 | 61.44 |
| + 3D + spatial-reset | **66.65** | **75.85** | **52.36** | **83.72** | **58.24** | **62.12** |
| + diagonal layout | 61.20 | 72.33 | 52.51 | 60.13 | 37.42 | 54.88 |
| + modality interval | 62.80 | 73.19 | 50.88 | 70.43 | 42.18 | 51.28 |
| + text spatial-reset | 58.27 | 68.20 | 50.71 | 77.30 | 52.15 | 44.33 |
| + scaling rotary base | 60.15 | 74.13 | 52.11 | 80.44 | 52.16 | 58.80 |

关键观察：
- 3D structure 主要提升 grounding（73.48 → 74.40）和 ChartQA（56.84 → 61.44）。
- spatial-reset 在 3D 基础上再全面提升，特别 video（51.29 → 52.36）和 ChartQA（61.44 → 62.12）。
- diagonal layout 是灾难，InfoVQA 从 58.85 直接降到 37.42。
- text spatial-reset 破坏 text RoPE 兼容性，全面下降。

#### Frequency Allocation（Table 5）

| Allocation | Image | Video | Grounding | Overall |
|------------|-------|-------|-----------|---------|
| VideoRoPE-like | 65.33 | 52.11 | 72.50 | 63.31 |
| IL-RoPE-like | 65.26 | 51.15 | 72.80 | 63.07 |
| Multi-Head | 66.40 | 52.58 | 74.92 | 64.63 |
| Interleave | **66.65** | 52.36 | **75.85** | **64.95** |

Interleave 略优于 Multi-Head，但都显著优于部分频段分配的方法。

#### Allocation Ratio（Table 8）

| t:h:w | Image | Video | Grounding | Overall |
|-------|-------|-------|-----------|---------|
| 0:32:32 | 66.42 | 51.01 | 76.02 | 64.48 |
| 12:26:26 | 66.30 | 51.93 | 75.77 | 64.67 |
| **24:20:20** | **66.65** | **52.36** | 75.85 | **64.95** |
| 32:16:16 | 64.07 | 51.15 | 74.65 | 63.29 |
| 48:8:8 | 65.06 | 51.17 | 72.87 | 63.03 |

t:h:w = 24:20:20 最优。t 占比需要适度高于 h/w（因为 temporal range 通常大），但过度倾斜会损害 spatial grounding。

#### Temporal Stride（Table 9）

| Stride δ | MVBench | STAR | VideoMME | LVBench | MLVU | Charades | Overall |
|----------|---------|------|----------|---------|------|----------|---------|
| 0.5 | 56.55 | 57.90 | 58.96 | 38.99 | 62.37 | 31.88 | 51.11 |
| **1** | **57.05** | 57.79 | 58.96 | **40.54** | **65.46** | **34.36** | **52.36** |
| 2 | 55.70 | 58.13 | 58.15 | 38.02 | 63.11 | 33.51 | 51.10 |
| Dynamic | 56.28 | 57.93 | 58.74 | 41.12 | 63.75 | 32.99 | 51.80 |

δ=1 最简单也最优，V2PE 的 dynamic stride 没有显著收益。

#### Visual Attention 分析（Table 7）

| Method | Layer 4 | Layer 12 | Layer 20 | Layer 28 |
|--------|---------|----------|----------|----------|
| MHRoPE | 40.31 | 21.76 | 32.05 | 19.00 |
| w/o spatial-reset | 35.99 | 19.68 | 22.02 | 9.93 |
| MRoPE-I | 37.48 | 15.68 | 28.08 | 23.23 |
| w/o spatial-reset | 31.22 | 17.66 | 16.02 | 11.69 |

深层 layer（20、28）visual attention 几乎翻倍。spatial-reset 让模型在深层网络仍保持对视觉内容的关注，符合 LLM 中 attention sink 现象——小 position id 吸引更多 attention。

### 5.5 Long-context Extrapolation（Figure 5 + Table 6）

在 256K context 上：
- Vanilla RoPE 128K/256K 急剧退化（position id 增长过快）。
- VideoRoPE / HoPE 在长视频上略好（低频分配给 temporal）。
- MRoPE-I + YaRN 在 LVBench 上从 42.0 提升到 43.6，证明 interleave 设计与 YaRN 兼容。MRoPE + YaRN 反而略降，因为 partition spectrum 让 YaRN 的 rescaling 难以一致应用。

---

## 6. 我的 Intuition 和批判性思考

### 6.1 为什么 spatial-reset 这么有效？

这个发现的本质是 **position id 的"绝对值"对 LLM 有意义**。LLM 预训练时 position id 从 0 开始，前几个 token 的 attention sink 现象说明模型对"小 position id"有结构化偏好。MRoPE 把视觉 token 的 spatial 坐标累加到 $t + h + \cdots$，让后面的 visual content 起点坐标很大，远离模型的"舒适区"。

spatial-reset 把每个 visual content 拉回 (0,0) 起点，相当于让每个 visual content 都有自己的"局部 sink"。这跟 LLM 中观察到的 attention sink 高度同构——模型不是真的"喜欢"前几个 token，而是"喜欢"小 position id。这种"局部 sink"对每个 image / video frame 都重新触发一次，从而显著加速 visual adaptation。

### 6.2 Frequency Allocation 的本质：分辨率 vs 范围

Channel 分配存在 trade-off：
- 分配多 channel 给一个轴 → frequency 分辨率高，能精确编码该轴不同尺度的依赖。
- 但其他轴 channel 变少 → 分辨率降低。

MRoPE 的简单分块让每个轴分辨率粗化 3 倍。MHRoPE 在 head 维度做分割，每个轴保持完整 channel，但 head 之间不能 cross-axis fusion。MRoPE-I 用 interleave 在 channel 维度做"细粒度分块"——每个轴每隔 2 个 channel 出现一次，相当于把"分辨率降低"变成"采样步长增加"，对每个轴而言 frequency spectrum 仍然是完整的。

这让我想到 signal processing 里的 **frequency division multiplexing** vs **time division multiplexing**。MRoPE 是 FDM（每个轴独占一个频段），MHRoPE 是某种空间分集（每个 head 一个轴），MRoPE-I 是某种交错 TDM。

### 6.3 Position Overlap 是隐性 bug

Diagonal layout 的失败很 instructive。我们在设计 position encoding 时容易陷入"几何 elegant"的陷阱（如 inter-modal symmetry），但忽略了一个朴素的事实：**position id 是一个 namespace，必须有 unique 解析**。如果 visual 的 spatial 坐标和后续 text 的 temporal 坐标可能 overlap，模型就没有先验区分它们属于哪个 modality。

这跟编程语言里 namespace pollution 类似——一个变量名在不同 scope 下指向不同对象是 OK 的，但同一个 scope 下必须 unique。MRoPE 的 $\max$ update rule 保证了 namespace 唯一性，diagonal layout 破坏了它。

### 6.4 Pre-trained Prior 不可轻动

Text RoPE 兼容性这条 guideline 提醒我们：VLM 不是从零训练的，而是 LLM 的"扩展"。任何对 text encoding 的修改都在"撬动"已经学好的知识。即便修改在逻辑上更合理（如更小的 rotary base 给 spatial），pre-trained LLM 的 inductive bias 是一种"知识"，逻辑优化与 inductive bias 冲突时，必须让位于 inductive bias。

这个 insight 跨任务普遍适用：fine-tuning 大模型时，能不动 pre-trained 机制就不动，只增加新机制。这跟 LoRA、adapter 的精神一致。

### 6.5 还能改进什么？

我自己看完后的几个 open question：

1. **MHRoPE 与 MRoPE-I 的混合**：能否在前几层用 MHRoPE（让 head 内部充分做 axis-specific attention），后几层切到 MRoPE-I（让 cross-axis fusion 更灵活）？这种 layer-wise 混合可能优于纯单选。

2. **Spatial-reset 与 long video**：spatial-reset 对每个 frame 都重置，但视频里物体长时间停留同一位置是常见 pattern。Reset 后，同一物体在 frame 100 和 frame 200 的 spatial 坐标都是 $(h, w)$，相对位置 $(0, 0, 0)$——但实际时间差是 100。这是否会让模型忽略长时间静止物体的运动？Table 2 的 video benchmark 上 MRoPE-I 略低于 VideoRoPE/HoPE 似乎暗示这个问题。

3. **Attention sink 的更深利用**：既然 spatial-reset 触发了"局部 sink"，能否显式在 visual content 开头加一个 sink token？类似 StreamingLLM 的 sink token 思路。

4. **Frequency allocation 的 data-adaptive 版本**：当前 24:20:20 是固定比例。能否设计 learnable 的 allocation，让模型根据 input 自动调整？比如多图场景给 spatial 多分配，长视频场景给 temporal 多分配。

5. **与 3D scene / embodied AI 的衔接**：如果未来 VLM 要处理 3D 点云、深度信息、机器人 trajectory，position axis 会扩展到 6+ 个。MHRoPE 的 head-level partition 在这种场景下更 scalable，值得前瞻探索。

参考链接（扩展阅读）：
- StreamingLLM (attention sink)：https://arxiv.org/abs/2309.17453
- Llama 3 herd：https://arxiv.org/abs/2407.21783
- Qwen3 tech report：https://arxiv.org/abs/2505.09388
- Mogao (interleaved multimodal generation)：https://arxiv.org/abs/2505.05472
- OmniGen2：https://arxiv.org/abs/2506.18871
- Apollo (video VLM exploration)：https://arxiv.org/abs/2412.10360
- Cambrian-1：https://arxiv.org/abs/2406.16860
- Jianlin Su 多模态位置编码博客：https://spaces.ac.cn/archives/10040

---

## 7. 总结

这篇 paper 的价值在于它做了 VLM 社区长期缺失的"系统性 ablation"，把 multimodal RoPE 拆成 3 个正交 design axis，提炼出 3 条 guidelines（positional coherence、full frequency utilization、preservation of textual priors），并落地为两个 plug-and-play 变体 MHRoPE / MRoPE-I。spatial-reset 是被 ablation 发现的"小修改大收益"机制，与 LLM 的 attention sink 现象深度同构。

代码开源在 https://github.com/JJJYmmm/Multimodal-RoPEs ，可以直接 plug 进任何基于 Qwen2.5-VL / Qwen3-VL 的训练 pipeline。如果你在搭 VLM，**强烈建议直接用 MRoPE-I + spatial-reset**，配置简单，跨架构泛化好，YaRN 兼容，全 benchmark 稳定提升。
