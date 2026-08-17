---
source_pdf: TLA.pdf
paper_sha256: 674d92a2a97e54b429e00115d5e82573f2a33c32c25418969d170765a930599f
processed_at: '2026-08-12T16:28:17-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 TLA

## 一句话总结

让机器人手指头装上触觉传感器，把摸到的感觉变成图片，喂给大语言模型，大模型一边读文字指令一边看触觉图，然后告诉机器人手该怎么动，把柱子插进孔里。

## 为什么要搞这个事

现在的机器人 VLA 模型（比如 RT-2、OpenVLA）都靠摄像头看东西。但插柱子进孔这个活儿，精度要求 0.1mm 级别，摄像头根本看不清柱子跟孔差了多少。就好比你闭着眼睛把钥匙插进锁孔，得靠手感，不能靠眼睛看。

触觉传感器（GelStereo 2.0）就像给机器人手指头贴了一层有弹性的皮肤，碰到东西时皮肤会变形，这个变形被拍成图片。问题是：怎么让机器人理解这些触觉图片，然后做出正确的动作？

## 他们干了啥

### 第一步：造数据

在仿真里让机器人瞎插柱子，插歪了就记录下"手指摸到了啥"（8张触觉图，左右手各4个时间点）和"当时差了多少"（pose error）。

关键是 action label 的设计很聪明：不要求机器人一步到位把误差清零，只要把误差推到 clearance 一半以内就行。比如 clearance 是 2mm，那误差小于 1mm 就算成功。这样训练出来的 policy 更高效，不会动太大导致新的碰撞。

总共采集了 24k 条数据，全是 2.0mm clearance 的方形柱子。

### 第二步：把触觉图拼成一张大图

这是最 hacky 但也最实用的一招。8张触觉图（左右手×4帧）拼成一个 3×3 的九宫格，最后一格用白色填充，然后 resize 成一张 616×616 的图，直接喂给 Qwen2-VL 的 ViT。

为啥这么干？因为 Qwen2-VL 的 ViT 只认 2D 图片，改架构成本太高。把时间维度硬塞进空间维度，ViT 的 self-attention 照样能 attend 到不同时间帧的 patch。缺点是模型不知道这 8 格有先后顺序，但实验证明这个 hack 够用。

### 第三步：action 怎么编码

OpenVLA 的做法是把连续 action 离散成 256 个 bin，用 vocab 里最不常用的 token 替代。问题是这样丢掉了大模型预训练时学到的数字 sense。

TLA 的做法：先把 action 乘以一个 scale factor（比如 50），取整成 integer，比如 "0.024mm" 变成 "1"，"0.123mm" 变成 "6"。这样 action 就是 natural language 里的整数，大模型见过无数整数，数值 prior 完整保留。代价是 action sequence 变长（每个维度 3-4 个 token，总共 12 个 token），训练更难但 generalization 更好。

### 第四步：训练

冻结 ViT encoder，只对 Qwen2 7B 的 LLM 部分做 LoRA fine-tune。Loss 就是标准的 next token prediction。Inference 时 beam search 生成 action token，再转回浮点数。

## 结果怎么样

### 单柱子任务（训练分布内）

TLA 的 action 正确率比 BC 和 Diffusion Policy 高 20-50%，而且正确动作的精度高得多——x 方向 L1 error 0.079mm，比 DP 的 0.370mm 降了 78%。这意味着 TLA 不仅动作方向对，步长也精准，完成任务需要的步数更少。

### 多柱子任务（泛化测试）

训练时只见过方形和三角形柱子，测试时换成圆形和六边形。BC 的成功率从 18.4% 暴跌到 0.152%，TLA 从 18.4% 微降到 16.5%，几乎没掉。这就是 language grounding 的价值——大模型的 cross-modal prior 让模型对新形状更 robust。

### 真实 insertion 成功率

这是最关键的数字。训练只用 2.0mm clearance 数据，测试时换 1.0mm clearance（远 OOD）：
- BC: 18%
- Diffusion Policy: 20%
- TLA (单柱子训练): 74%
- TLA (多柱子训练): 90%

TLA 在 1.0mm 上还能 90%，这个 generalization 能力很 striking。

## 为什么能 work

我的几个 hypothesis：

**大模型的 cross-modal grounding prior**：Qwen2-VL 在 internet-scale 数据上学过"图片区域 ↔ 文字概念"的对齐。这个能力迁移到触觉上：触觉图里亮的地方（接触力大）↔ "碰撞"这个概念 ↔ "往反方向动"。三段式推理是 BC 学不到的。

**数值 prior**：大模型见过无数数字，知道 0.05 接近 0，1.23 比 1.22 大。这种 magnitude sense 让 action 预测更稳定。BC 用 ResNet，数字只是 feature，没有数值直觉。

**离散化的 regularization 效果**：peg-in-hole 这个任务，正确 action 的范围很窄，错误 action 的范围很宽。NTP 的离散化天然过滤掉 outlier，可能比 Diffusion Policy 的连续分布更适合这种"窄正确区间"的任务。

## 最 hacky 和最 clever 的地方

最 hacky：8张触觉图拼成九宫格喂 ViT。完全没处理时序，但 work 了。

最 clever：action label 加 clearance/2 的偏移。不要求消除全部误差，只要求推到容差以内。这个 task-aware label engineering 让 policy 更高效。

## 最大的 limitation

作者自己承认：时序编码不到位（spatial concat 不是真正的 temporal modeling）、触觉表示单一（只用 2D image）、sim-to-real 没验证。

我没看到但觉得最关键的缺失：没有 ablation 去掉 language instruction，只保留触觉。如果 language 在这里只是 task label（"这是方形柱子"），那大模型的价值就大打折扣。language 到底是 reasoning bridge 还是只是 task descriptor？这个没验证，是 paper 的一个 gap。

## 对未来的启发

这篇 paper 虽然规模小（24k data，7B model），但开了一个新方向：把 VLA paradigm 扩展到 tactile modality。下一个 breakthrough 可能是 vision + tactile + proprioception + language + action 的多模态 embodied foundation model，TLA 是第一步。

action decoder 还有提升空间——现在用 NTP + integer scaling，换成 flow matching（π₀ 的思路）或 diffusion head（RDT-1B 的思路），可能精度还能涨。tactile encoder 也可以换 3D point cloud 或 force/torque，解决 y 轴感知差的问题。

Project page: https://sites.google.com/view/tactile-language-action/

---

# TLA: Tactile-Language-Action Model for Contact-Rich Manipulation 深度解析

## 一、核心 Motivation 与定位

这篇 paper 解决的是 contact-rich manipulation 中一个长期被忽视的 gap: 现有 VLA (Vision-Language-Action) 模型如 RT-2 [1](https://arxiv.org/abs/2307.15818)、OpenVLA [2](https://arxiv.org/abs/2406.09246)、π₀ [3](https://arxiv.org/abs/2410.24164)、RDT-1B [4](https://arxiv.org/abs/2410.07864) 都把 visual modality 当作主输入, 这对 push/pull/pick-and-place 这类 free-space 任务够用, 但 peg-in-hole assembly 这类 sub-millimeter precision 的任务几乎完全依赖 tactile feedback. Vision 在 peg 与 hole 对齐误差 < 1mm 时基本失效, 而 visuotactile sensor (如 GelStereo 2.0 [5](https://ieeexplore.ieee.org/document/10309458)) 能捕捉到 contact normal force、surface deformation 这些 vision 看不到的物理量。

作者的 core insight: language model 已经在 vision-language 预训练中学到了 cross-modal grounding 能力, 如果把 tactile image 喂给 ViT encoder, 让 LLM 通过 language instruction 作为 bridge 来 ground tactile observation 到 action space, 就能复用 LLM 预训练的 reasoning prior, 而 BC/Diffusion Policy 这种 from-scratch 训练的方法则没有这个 prior。

## 二、Dataset 设计的关键 trick

### 2.1 仿真环境与 tactile rendering

数据采集在 NVIDIA Isaac Gym + Flex physics engine 中完成。FEM (Finite Element Method) 模拟 GelStereo 2.0 sensor 的 elastomer deformation, 然后用 [6](https://ieeexplore.ieee.org/document/10309458) 中的 tactile imprint rendering 方法生成 tactile image. 为缩小 sim-real gap, 用真实 sensor 拍摄的 image 做纹理映射, 而非手工 pattern, 这是一个 domain randomization 的变种思路。

### 2.2 Action label 的生成公式

这里很关键, action label 不是直接用 pose error, 而是 clip 后的"指导性动作"。设 peg-hole pose error 为 $(e_x, e_y, e_{rz})$, assembly clearance 为 $c$, 公式:

$$\Delta\hat{x} = \begin{cases} \mathbb{F}_{clip}(-e_x + c/2, -\delta, 0), & \text{if } e_x \geq 0 \\ \mathbb{F}_{clip}(-e_x - c/2, 0, \delta), & \text{if } e_x > 0 \end{cases}$$

变量解释:
- $e_x$: x 方向 pose error (peg 当前位置相对于正确 insertion 位置的偏差)
- $c$: assembly clearance (peg 与 hole 之间的间隙, 训练数据固定 2.0mm)
- $\delta$: action 上限, 设为 1mm, 防止单步动作过大导致新的 collision
- $\mathbb{F}_{clip}(\cdot, a, b)$: 将第一个参数 clip 到 $[a, b]$ 区间
- $\Delta\hat{x}$: 监督信号, 即 model 应该预测的 x 方向 action

为什么加 $c/2$? 因为只要 peg 在 clearance 范围内, 就算"对齐"了。直接用 $-e_x$ 会让 model 学到"完全消除误差"的 policy, 但实际上只要误差 $< c/2$ 就能成功 insertion。所以 label 是"把误差推到 clearance 一半以内的最小动作", 这样训练出来的 policy 更高效、step 更少。这是一个 task-aware label engineering 的典型例子。

y 方向同理, rz 方向:
$$\Delta\hat{rz} = \mathbb{F}_{clip}(-e_{rz}, -1.5°, 1.5°)$$

这里 rotation 限制在 $\pm 1.5°$, 避免 over-rotation。

### 2.3 Dataset 规模

24k pairs, 单 peg (square) 8k, 多 peg (square+triangle) 16k, 评估时额外采集 round/hexagon 8k 作为 OOD。每个 sample 包含 8 张 tactile images (左右 fingertip × 4 timestamps)。

## 三、TLA Architecture 详解

### 3.1 整体结构

Backbone 是 Qwen2-VL 7B [7](https://arxiv.org/abs/2409.12191), 包含:
- **Tactile Encoder**: Qwen2-VL 的 ViT, patch size 14
- **Language Model**: Qwen2 7B, 用 LoRA [8](https://arxiv.org/abs/2106.09685) fine-tune
- **Tactile Encoder freeze**, 只训 LoRA

### 3.2 Tactile Encoder 的核心设计: 时间→空间转换

这是这篇 paper 最有 idea 的部分。问题: 一次 contact action 产生 8 张时序 tactile images (left/right fingertip × 4 timestamps), 怎么喂给 ViT?

作者的解法: 把 8 张图拼成一张 3×3 grid (第 9 格用 white image 填充), resize 到 616×616, 作为单张图喂给 ViT。

这招很巧妙但也有明显 trade-off:

**优点**:
- 复用 ViT 的 2D spatial attention, 不需要改架构
- temporal information 通过 spatial arrangement 编码, ViT 能直接 attend 到不同 timestamp 的 patch

**缺点** (作者自己也承认):
- ViT 的 positional encoding 是 2D 的, 把 temporal 映射成 spatial 后, model 看不出"这 8 张图有先后顺序"
- 3×3 grid 的 spatial layout 是人工设计的, model 必须学会这个 layout 的语义

更直觉的解释: 这是一种"poor man's video transformer"。真正的 video ViT (如 ViViT [9](https://arxiv.org/abs/2103.15691)、TimeSformer [10](https://arxiv.org/abs/2102.05095)) 会用 factorized space-time attention, 但这里为了复用 Qwen2-VL 的预训练 weight, 选择了 spatial concat 的 hack。

ViT patch size = 14, 输入 616×616, 所以 patch grid 是 $44 \times 44 = 1936$ patches → 1936 tokens。然后再用 MLP 把 $2 \times 2$ 的 token 压成 1 个 token, 最终 $1936 / 4 = 484$ tactile tokens (注: paper 写 1936, 这里有点 ambiguity, 可能是压缩前)。

### 3.3 Action Tokenization

这里和 OpenVLA 的思路不同。OpenVLA 用 bin discretization, 把连续 action 映射到 256 个 bin, 然后用 vocab 中最不常用的 token 替代。这种做法的问题: 丢失了 LLM 预训练时学到的 numerical reasoning 能力。

TLA 的做法: 保留 Qwen2 原生的 numerical tokenizer (Qwen2 对数字是 digit-by-digit 编码的, 比如 "1.23" → ["1", ".", "2", "3"]), 但先对 action 做 scaling:

$$A_{gt} = A_{raw} \cdot s$$

其中 $A_{gt} \in \mathbb{R}^3$ 是 ground-truth action, $A_{raw} \in \mathbb{R}^3$ 是 raw action (单位 mm/degree), $s \in \mathbb{R}^3$ 是 scale factor (per-dimension). 然后取整成 integer, 这样 "1.23mm" 变成 "123", 用 3 个 token 表示。

这个设计的好处: action 是 natural language 中的 integer, LLM 预训练时见过无数 integer, 数值先验被保留。坏处: action sequence 变长 (每个 dimension 3-4 个 token, 总共 ~12 tokens), 而 bin discretization 只需 3 个 token。training 更难, 但 generalization 更好。

### 3.4 Training

Loss 是标准 NTP (Next Token Prediction):

$$\mathcal{L}_{NTP} = -\sum_{t=1}^{T} p(y_t) \log P(y_t | y_{<t}, x)$$

变量:
- $T$: action sequence 长度
- $y_t$: 第 $t$ 步的 ground-truth token
- $y_{<t}$: 前 $t-1$ 步的 token (训练时用 teacher forcing, 即用 ground-truth 替换)
- $x$: 输入, 包含 tactile tokens + text instruction tokens
- $p(y_t)$: ground-truth distribution (one-hot)
- $P(y_t | y_{<t}, x)$: model 预测的 distribution

Inference 用 beam search 生成 action token, 然后通过 Action-De-Tokenizer 映射回浮点数。

## 四、实验结果深度分析

### 4.1 Single-Peg (Table I)

| Method | GCR (%) | L1 x (mm) | L1 y (mm) | L1 rz (deg) |
|--------|---------|-----------|-----------|-------------|
| BC | 10.4 | 0.803 | 0.302 | 0.205 |
| DP | 8.5 | 0.370 | 0.382 | 0.568 |
| SP-TLA | 12.5 | 0.079 | 0.122 | 0.173 |

GCR (Goal Convergence Rate) 定义: 所有输出 action 在 x, y, rz 三个方向都正确的百分比。TLA 12.5% 看着不高, 但比 BC/DP 高了 20-50%。

更 striking 的是 L1 distance: x 方向 TLA 是 0.079mm, DP 是 0.370mm, 降了 78%。这说明 TLA 不仅"正确动作"更多, 而且正确动作的精度也高得多。这对 assembly 任务至关重要: 一个"方向对但步长过大"的动作会导致新的 collision, 反而更糟。

### 4.2 Multi-Peg (Table II)

| Method | ID GCR | ID L1 x | ID L1 y | ID L1 rz | OOD GCR | OOD L1 x | OOD L1 y | OOD L1 rz |
|--------|--------|---------|---------|----------|---------|----------|----------|-----------|
| BC | 18.4 | 0.260 | 0.655 | 0.186 | 0.152 | 0.286 | 0.722 | 0.246 |
| DP | 7.4 | 0.371 | 0.348 | 0.480 | 0.080 | 0.386 | 0.369 | 0.544 |
| MP-TLA | 18.4 | 0.102 | 0.114 | 0.135 | 0.165 | 0.121 | 0.102 | 0.184 |

注意 BC 的 GCR 在 ID 上和 TLA 一样 (18.4%), 但 OOD 上暴跌到 0.152%。TLA 从 18.4% → 16.5%, 几乎没掉。这就是 language grounding 的价值: LLM 的 cross-modal prior 让 model 对 peg shape 的变化更 robust。

### 4.3 Real Insertion Success Rate (Table III)

| Method | 2.0mm Suc | 2.0mm Step | 1.6mm Suc | 1.6mm Step | 1.0mm Suc | 1.1mm Step | Total Suc | Total Step |
|--------|-----------|------------|-----------|------------|-----------|------------|-----------|-----------|
| BC | 44 | 2.60 | 32 | 4.00 | 18 | 4.44 | 31 | 3.68 |
| DP | 58 | 2.45 | 43 | 4.35 | 20 | 4.70 | 40 | 3.83 |
| SP-TLA | 96 | 3.15 | 94 | 3.77 | 74 | 4.37 | 88 | 3.76 |
| MP-TLA | 94 | 3.04 | 86 | 3.30 | 90 | 4.35 | 90 | 3.56 |

这里才是真正的 money table。2.0mm clearance (训练分布内), TLA 96% success, DP 58%, BC 44%。更夸张的是 1.0mm clearance (远 OOD, 训练只用 2.0mm), SP-TLA 74%, MP-TLA 90%, 而 DP 掉到 20%, BC 掉到 18%。

为什么 MP-TLA 在 1.0mm 上比 SP-TLA 好 (90% vs 74%)? 作者解释: 多 peg 训练增强了 generalization。这暗示 LLM 的 reasoning 能力在 multi-task training 时被更好地激活, 类似 multi-task learning 中的 positive transfer。

### 4.4 Failure Case 分析 (Fig. 6)

Triangle peg 失败的原因很 insightful: triangle hole 在 x 方向允许偏差是 3d (d 是 clearance), 但 y 方向只有 2.3d。而 2D tactile image 对 y 方向 (垂直于 gripper) 的感知本来就差, 所以 triangle peg 在 y 方向更小的容差 + tactile 对 y 的弱感知 = 失败。

这暴露了一个根本问题: 2D tactile image 是 sensor surface 的 deformation 投影, 它对 in-plane (parallel to sensor) 的力敏感, 对 out-of-plane (perpendicular) 的力不敏感。要解决这个问题, 可能需要 3D tactile point cloud 或 force/torque sensor。

## 五、与相关工作对比

### 5.1 vs. FuSe [11](https://arxiv.org/abs/2501.04693)

FuSe (Jones et al., 2025) 也把 tactile 整合进 VLA, 但 TLA 的区别:
- FuSe 是 vision + tactile 双模态, TLA 是 tactile-only
- FuSe 做 pick-and-place, TLA 做 precision assembly
- FuSe 用 bin tokenization, TLA 保留 numerical tokenizer

### 5.2 vs. Touch-Vision-Language datasets [12](https://arxiv.org/abs/2402.13232) [13](https://arxiv.org/abs/2406.03813)

这些工作 (Fu et al., Cheng et al.) 只做 perception (material classification, texture description), 没有 action。TLA 是第一个 language-grounded tactile-action generation framework。

### 5.3 vs. π₀ [3](https://arxiv.org/abs/2410.24164)

π₀ 用 flow matching 做 action 生成, 比 NTP 更适合连续 action space。TLA 用 NTP + integer scaling, 是一种折衷。如果 TLA 换成 flow matching 或 diffusion head, 可能 action 精度还能提升, 但会失去 LLM 的 numerical reasoning prior。这是一个值得探索的方向。

## 六、Limitations 与未来方向

作者自己列了三个 limitation:

1. **Temporal encoding 不到位**: 用 spatial concat 代替真正的 temporal attention, 没充分利用时序信息。未来可用 3D ViT 或 temporal transformer。

2. **Tactile representation 单一**: 只用 2D image, 没用 contact depth map 或 3D point cloud。GelStereo 2.0 本身是 stereo sensor, 理论上能输出 depth, 但 paper 没用。

3. **Action detokenization 简单**: integer scaling + beam search 比较粗糙, 可以考虑 VAE 或 diffusion decoder。

我补充几个没提的 limitation:

4. **Sim-to-real 未验证**: 所有实验都在 simulation, 真实环境下的 sim-real gap 未知。GelStereo 2.0 的 FEM 模拟虽然 fidelity 高, 但 real sensor 的 noise、lighting、elastomer aging 都会引入误差。

5. **Language 的作用不清晰**: paper 用了 language instruction (如 "insert the square peg into the target hole"), 但 ablation 中没有"去掉 language 只用 tactile"的 baseline。language 在这里到底是 reasoning bridge 还是只是 task label? 如果只是 task label, 那 LLM 的价值就大打折扣。这是一个 critical ablation 缺失。

6. **单步 action, 非 closed-loop**: TLA 每次预测一个 $(\Delta x, \Delta y, \Delta rz)$, 然后重新感知。这是 discrete-step closed-loop, 不是 continuous control。对 high-frequency contact 任务 (如 compliant insertion) 可能不够。

## 七、Intuition Building: 为什么 TLA work?

回到 Karpathy 最关心的 intuition 层面。为什么把 tactile image 喂给预训练 VLM, 用 NTP fine-tune, 能比 BC/Diffusion Policy 好这么多?

**Hypothesis 1: LLM 的 cross-modal grounding prior**

Qwen2-VL 在 internet-scale image-text data 上预训练, 学到了"image region ↔ text concept"的对齐。这种对齐能力迁移到 tactile domain: tactile image 中的 "bright region" (高 contact force) ↔ language 中的 "collision" ↔ action 中的 "move away"。这个三段式推理是 BC (纯 supervised mapping) 学不到的。

**Hypothesis 2: Numerical reasoning prior**

Qwen2 见过无数数学表达式, 知道"1.23 比 1.22 大", "0.05 接近 0"。这种 magnitude sense 让 TLA 在 action scaling 时更稳定。BC 用 ResNet, 数值是纯 feature, 没有 magnitude prior。

**Hypothesis 3: Multi-task transfer**

MP-TLA 在 OOD 上比 SP-TLA 好, 说明 language instruction 作为 task descriptor, 让 model 学到了"peg shape → insertion strategy"的 abstract mapping。这和 multi-task LLM 的 in-context learning 机制类似。

**Hypothesis 4: Action distribution 的 smoothness**

NTP 学的是 token distribution, 而 Diffusion Policy 学的是 continuous action distribution via denoising。对于 peg-in-hole 这种"正确 action 很窄, 错误 action 很宽"的任务, NTP 的离散化反而是一种 regularization, 避免 model 被 outlier action 带偏。这可能解释为什么 TLA 比 DP 好。

## 八、对未来的启发

这篇 paper 给我几个 take-away:

1. **Tactile 是 VLA 的下一个 frontier**: vision-only VLA 已经饱和, contact-rich 任务必须引入 tactile。TLA 是第一个 proof-of-concept, 但远非最优。

2. **Language 作为 tactile-action 的 bridge**: 这个 framing 很重要。未来可能不需要 explicit language instruction, 但 cross-modal grounding 的机制会被保留。

3. **Action representation 很关键**: integer scaling + NTP 是一种简单但有效的方案, 但 flow matching (π₀) 或 diffusion (RDT-1B) 可能是更好的 action decoder。结合 TLA 的 tactile encoder + 这些 action head, 是一个 promising direction。

4. **Sim-to-real for tactile**: 这是整个 tactile robotics 的瓶颈。FEM 模拟虽然 fidelity 高, 但计算成本高, 难以 scale。需要 differentiable FEM 或 neural surrogate。

5. **Temporal tactile modeling**: TLA 用 spatial concat hack 了时序, 未来需要真正的 temporal tactile transformer, 可能借鉴 video LLM (如 Video-LLaMA [14](https://arxiv.org/abs/2305.15143)) 的设计。

Project page: https://sites.google.com/view/tactile-language-action/

这篇 paper 虽然规模不大 (24k data, 7B model), 但它 open 了一个新方向: 把 VLA 的 paradigm 扩展到 tactile modality。下一个 breakthrough 可能是 multi-modal embodied foundation model, 同时处理 vision + tactile + proprioception + language + action, 而 TLA 是这个方向的重要一步。
