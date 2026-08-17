---
source_pdf: SpatialBot Precise Spatial Understanding with Vision Language Models.pdf
paper_sha256: e95a6081620fe84ab92bffc009a64190b71005e400157842f9a320cf59359e34
processed_at: '2026-08-12T09:30:14-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 SpatialBot

Andrej, 我重新来一遍, 这次用更 intuitive 的讲法.

---

## 这篇 paper 在干嘛

一句话: **教 VLM 看懂 depth map**.

为什么这事重要? 你想, GPT-4o 这种模型看图片很厉害, 但你问它 "杯子在书前面还是后面" 或者 "机械爪碰到杯子没" — 它经常瞎猜. 原因很简单: 单张 RGB 图里, 远近信息是 ambiguous 的. 同一个像素值, 可能是远处的大杯子, 也可能是近处的小杯子. 这是 2D 视觉的 fundamental limitation.

那怎么办? 给它 depth map 啊. RGB-D 相机便宜得很, 机器人上标配. 问题在于: **VLM 没见过 depth map, 你直接喂给它, 它不知道这是啥**. 就像你给一个只学过英文的人一本中文书, 他能看见字, 但读不懂.

SpatialBot 的工作就是: 设计一套训练方法, 让 VLM 从零开始学会读 depth map, 然后用 depth 做空间推理.

---

## 三个核心 trick

### Trick 1: 把 depth map 编码成 "像 RGB 的样子"

depth map 本质上是个数值矩阵, 每个像素存 "这个点离相机多少毫米". 问题: 这个数值范围太大了. indoor 可能是 500mm, outdoor 可能是 50000mm. 你直接存成单通道图, VLM 的 vision encoder (SigLIP) 看不懂 — 它预训练时见过的都是 RGB 三通道 0-255 的 natural image.

作者的方案: 把 depth 值拆成三个 channel, 每个通道 0-255, 让它看起来像 RGB:

$$I_{h,w}^0 = (d_{h,w} // 2^{10}) \times 2^1$$
$$I_{h,w}^1 = (d_{h,w} // 2^5) \times 2^3$$
$$I_{h,w}^2 = (d_{h,w} \% 2^5) \times 2^3$$

变量解释:
- $d_{h,w}$: 像素 $(h,w)$ 处的真实 depth, 单位 mm
- $I_{h,w}^0, I_{h,w}^1, I_{h,w}^2$: 三个通道的像素值, 都是 0-255
- $//$: 整除, $\%$: 取余
- $2^{10} = 1024$mm ≈ 1m, 这是个关键分界: channel 0 编码 "几米" 量级
- $2^5 = 32$mm, channel 1 和 2 编码更精细的 mm 量级

直觉上, 三个 channel 分别管不同的尺度:
- **Channel 0**: 这东西大概在 1m, 2m, 还是 50m 开外? (粗粒度)
- **Channel 1**: 在 1m 这个量级里, 是 1000mm 还是 1050mm? (中粒度)
- **Channel 2**: 最后那点 mm 的零头 (细粒度)

为什么这么拆? 因为 VLM 的 vision encoder 预训练在 RGB 上, 它对 0-255 的三通道图像有天然的 inductive bias. 你把 depth 包装成 RGB-like, encoder 至少不会完全懵掉. 而 LLM 部分见过数字比较, 所以从三个 channel 解出真实 mm 值, 是个它能 fit 的函数.

对比一下 MiDaS / Depth Anything 那种 ordinal encoding: 只告诉你 "这个点比那个点近", 但不能说 "近多少 mm". ordinal 之间不能做数学运算. SpatialBot 的 encoding 保留了 metric 性质, 你可以直接做 `depth_A - depth_B = 50mm` 这种推理.

参考: 原 repo https://github.com/BAAI-DCAI/SpatialBot

---

### Trick 2: 像教小孩一样, 分三层 progressive 训练

直接让 VLM 从 depth map 做空间推理, 它学不会. 太难了. 作者设计了个 curriculum:

**第一层: 识字课**
教模型最基础的: "这个像素的 depth 值是多少". 
- Input: depth map + 坐标 (x, y)
- Output: 一个数字, 比如 "1240"
- 目的: 让 vision encoder 学会 encode depth map 的 token, 让 LLM 学会从 token 解出数值

还让它做: "只看 depth map, 猜图里有啥". 这逼模型理解 depth 的语义 — 比如看到一块平的区域有四个突起, 可能是桌子上有四个杯子.

**第二层: 对齐课**
RGB 和 depth 是配对的, 但模型得学会 "RGB 里的杯子" 对应 "depth 里的这块区域".
- 问: "杯子的 depth 是多少" → 模型先在 RGB 定位杯子, 再去 depth map 找对应区域
- 问: "杯子比盘子近还是远" → 模型比较两个 object 的 depth 值
- 用 max/min/mean/center 四个值描述一个 object 的 depth (用 95th/5th percentile 抗噪)

**第三层: 应用课**
有了基础能力, 做高级任务:
- counting (depth 提供 boundary, 帮模型分清物体)
- spatial relationship (left/right 用 real-world 坐标, 不是 image 坐标)
- robot manipulation: "拿最大的杯子放到左边的板上"

这个 curriculum 的妙处: 每一层都建立在前一层的能力上. 你不会走路就跑不了, 同理模型不会读 depth 就做不了 spatial reasoning.

参考数据集: https://huggingface.co/datasets/RussRobin/SpatialQA

---

### Trick 3: Depth API — 让模型可以 "查答案"

这个我觉得最聪明. VLM 即使学了读 depth map, 精度也有限 — 它毕竟是通过 token 来表示数字的, 不是直接做数值运算. 所以作者加了个 escape hatch:

模型可以输出 `Depth(point)` 这种特殊文本, 外部 API 收到后, 去查 depth map 里那个点的真实值, 然后把数值塞回 VLM 的 context, VLM 再继续推理.

举例:
1. 用户问: "杯子在书前面还是后面"
2. VLM 想: 我需要杯子和书的 depth 值
3. VLM 输出: "Depth(120, 80)" 和 "Depth(200, 150)" (杯子中心和书中心的坐标)
4. API 查完, 往 context 里塞: "Depth(120, 80) = 1240mm, Depth(200, 150) = 980mm"
5. VLM 看到: "1240 > 980, 所以杯子比书远, 书在前面"

这本质是 tool-use / function calling, 但用 in-context injection 实现, 不需要 fine-tune 特殊的 tool-use 机制.

关键工程细节: 训练时**只在 subset 数据上允许调 API**, 其余数据必须自己读 depth map. 这防止模型偷懒走 shortcut — 如果总是能调 API, 它就不学读 depth 了.

---

## 一个特别巧妙的设计: Illusion Task

SpatialQA-E 里有个 task: 把物体的照片打印出来放桌上. 看起来像真的, 但它是 flat 的. 模型得区分 "打印的假杯子" 和 "真杯子".

人眼可能都看不出区别 (打印质量好的话). 但 depth map 一看就懂: 真杯子 depth 在 1000-1100mm 范围, 打印的杯子 depth 几乎是常数 (一个平面).

这个 task 的意义: **逼模型必须用 depth, 任何 RGB-based shortcut 都失效**. RGB appearance 上真假杯子一样, 只有 depth 能区分. 这是检验 "模型是否真懂 depth" 的 gold standard.

参考: https://huggingface.co/datasets/RussRobin/SpatialQA-E

---

## 实验结果, 说人话

### GPT-4o 被 naive depth 反而搞差了

SpatialBench (他们自己标的 benchmark) 上:

| Model | Position | Counting | Reaching | Size |
|---|---|---|---|---|
| GPT-4o (只 RGB) | 70.6 | 84.5 | 51.7 | 43.3 |
| GPT-4o (RGB+Depth) | **61.8** ↓ | 85.2 | 51.7 | **40.0** ↓ |

注意: GPT-4o 给了 depth map, Position 和 Size 反而**下降**了! 这说明它根本不会用 depth, 把 depth 当 noise 输入, 干扰了原本的判断.

这跟你之前讲过的 modality integration 困难一致 — 给模型更多 modality 不代表它自动会用, 搞不好是 negative transfer.

### SpatialBot 教会了之后

| Model | Depth | Position | Counting | Reaching |
|---|---|---|---|---|
| SpatialBot-Phi2-3B (RGBD) | **>99** | 61.8 | 91.7 | 55.0 |

3B 小模型, Depth 任务接近满分 (靠 API), 其他任务能跟 GPT-4o 接近. 说明: **不是模型大小问题, 是训练方式问题**. GPT-4o 没在 depth-aware data 上训过, 所以不会用.

### General VLM benchmark 也涨了

这是反直觉的好结果. 在 MME, GQA, POPE 这种"看起来不需要 depth"的 benchmark 上, SpatialBot 比 Bunny baseline 还高:

| Model | MME-P | MME-C | GQA | POPE |
|---|---|---|---|---|
| Bunny-Phi2-3B | 1474 | 285 | 61.5 | 86.2 |
| SpatialBot-Phi2-3B | **1487** | **312** | **62.3** | **87.0** |

解释: depth map 提供了 object boundary prior, 帮模型做 grounding 和 counting. 同时 depth-aware 训练让模型学到更结构化的 scene representation, 对 spatial-language alignment 有正则化效果. 也就是说, depth 训练不只帮 depth 任务, 还帮 general visual understanding.

### 机械臂任务

RGBD variant 在 pick-and-place 上成功率显著高于 RGB, 尤其是含 spatial relation 的复杂任务. Figure 1 的核心 example: 判断 "gripper 是否碰到 rag" — 人眼看 RGB 都分不清, GPT-4o 看 RGBD 也错, SpatialBot 从 depth 直接读 gripper tip 和 rag surface 的 mm 级差, 判对了.

---

## 我觉得这篇 paper 的 take-away

1. **Modality integration 需要教, 不是塞就行**. GPT-4o 给了 depth 反而变差, 说明 naive modality fusion 有害. 要训.

2. **Encoding 设计很重要**. 把 depth 编成 RGB-like 三通道 metric encoding, 是个 pragmatic 且有效的小 trick. 保留了数学可运算性, 让 LLM 能 fit.

3. **Curriculum learning 对 VLM 仍然有效**. 三层 progressive QA, 从识字到对齐到应用, 每层建立在前一层能力上. 这跟人类学东西的方式一致.

4. **Tool-use as escape hatch**. Depth API 让模型在精度不够时可以 "查答案". 这跟 ReAct, Toolformer, 甚至你自己讲过的 "let the model think" 一脉相承.

5. **Illusion task 是检验真懂 depth 的好方法**. 任何 RGB shortcut 都失效, 逼模型用 depth.

---

## 对你的几个 follow-up question 的预测

你可能会问:

**Q: 为什么不直接用 point cloud?**
A: 作者提了一句, point cloud 难收集难处理, depth map 更便宜更通用. RGB-D 相机便宜, 而且可以用 MDE 模型 (ZoeDepth) 把 RGB 数据自动转成 RGBD. Scale 容易.

**Q: 为什么不专门训个 depth encoder?**
A: 作者选择 RGB 和 depth 共享 SigLIP encoder, 是为了保留 generality. 但这可能是个 limitation. 专门的 depth encoder (类似 DINOv2 self-supervised) 可能更好. Future work.

**Q: VLA 部分的 delta pose collapse issue 说明什么?**
A: Autoregressive LLM 直接做连续控制信号是脆弱的. 100 bins 离散化在某些数据上 loss 完全不降. 这暗示 VLA 的 action head 可能需要 non-autoregressive 设计 (diffusion policy, flow matching). 你在 Tesla 肯定也遇到过类似问题 — LLM token prediction 和 continuous control 是两种不同的 problem.

**Q: Scale 怎么扩大?**
A: 现在 50k GPT-4o generated QA + 2000 robot episodes, 是 pilot 规模. Scale up 需要自动化的 depth-aware QA 生成 pipeline. 可能用 GPT-4o + SAM + MDE 的组合, 全自动从任意 RGB 图生成 RGBD VQA.

---

## 关键 reference 链接

- SpatialBot repo: https://github.com/BAAI-DCAI/SpatialBot
- Bunny base VLM: https://github.com/BAAI-DCAI/Bunny
- SpatialQA dataset: https://huggingface.co/datasets/RussRobin/SpatialQA
- SpatialQA-E (embodiment): https://huggingface.co/datasets/RussRobin/SpatialQA-E
- SpatialBench: https://huggingface.co/datasets/RussRobin/SpatialBench
- ZoeDepth (MDE): https://arxiv.org/abs/2302.12288
- RT-X dataset: https://arxiv.org/abs/2310.08864
- OpenVLA: https://arxiv.org/abs/2406.09246

---

总结一句: 这篇 paper 告诉我们, **VLM 的 spatial intelligence 不是靠堆参数能解决的, 是靠精心的 modality encoding + curriculum data + tool API 一起 push 出来的**. 跟你做 nanoGPT 时强调 "data + architecture + training recipe 三位一体" 是一个道理.

---

# SpatialBot 深度解析

Andrej, 这篇 paper 触到了 VLM 的一个真实痛点: **2D 视觉做得再好, 对 spatial/3D 的理解依然崩**. 你在 Tesla 讲过类似的事—pure vision 不够, 需要 geometry prior. SpatialBot 的核心 bet 是: 把 depth map 当成 VLM 的"第二公民" modality, 通过精心设计的 progressive QA pipeline 教会模型读懂它. 我来拆解.

---

## 1. 核心问题与 motivation

VLM 在 spatial understanding 上弱的三个根因:

1. **Modality gap**: 现有 VLM 的 vision encoder (CLIP, SigLIP) 在 RGB image-text pair 上预训练, 完全没见过 depth map. 直接喂 depth 给它, encoder 输出的 tokens 是 noise.
2. **Data gap**: 没有 RGB-D VQA 数据集. LLaVA-style instruction tuning data 既没 depth map, 也没 depth-related task.
3. **Scale gap**: indoor manipulation 要 mm 精度 (depth 范围 0-2m), outdoor 要 m 级精度 (depth 范围 0-100m+). 用 ordinal encoding (MiDaS, Depth Anything) 无法做数学运算, 无法直接对比 "这个物体比那个深多少 mm".

GPT-4o 给 RGBD 反而比 RGB 更差 (SpatialBench Position: 61.8 vs 70.6), 这就证明 naive 地拼 depth 反而是有害的噪声.

---

## 2. Architecture overview

基于 Bunny VLM family (https://github.com/BAAI-DCAI/Bunny), 是经典的 **vision encoder + multimodal projector + LLM** 三段式:

```
[RGB image]   ─┐
               ├─► SigLIP (384×384) ─► projector ─► tokens ─┐
[Depth image] ─┘                                            ├─► LLM (Phi-2 / Phi-3 / QWen1.5 / Llama-3) ─► text
[Text] ───────────────────────────────────────────────────►┘
                                                              │
                                                              ▼
                                              text 可能含 Depth(point) 调用
                                                              │
                                                              ▼
                                              Depth API ── 查询 depth map ── 回填 token
```

关键点:
- RGB 和 depth 共享同一个 vision encoder (SigLIP). 没有 separate depth encoder, 因为想保留 generality.
- depth 是 optional input. 没有 depth map 时模型也能用 (退化为普通 VLM).
- **Depth API** 是这篇文章最有想法的设计: 模型可以输出特殊文本格式 `Depth(point)`, 外部 API 查询该 pixel 的 depth 值后, 把数值再喂回 VLM context. 类似 tool-use / function calling, 但用 in-context injection 实现. 训练时只在 subset 数据上允许 API 调用, 强迫模型大部分时候要自己读 depth map, 避免退化成 "总是调 API" 的 shortcut learner.

---

## 3. Depth Map Encoding — 这是核心工程 trick

论文放弃了 ordinal encoding, 改用 **uint24 (3-channel uint8) 直接存 metric depth**, 单位 mm, 范围 1 mm - 131071 mm (~131 m).

### 三通道 uint8 编码公式

对像素 $(h, w)$ 的真实 depth 值 $d_{h,w}$ (单位 mm), 三个 channel 的像素值为:

$$I_{h,w}^0 = (d_{h,w} \ // \ 2^{10}) \cdot 2^1$$
$$I_{h,w}^1 = (d_{h,w} \ // \ 2^5) \cdot 2^3$$
$$I_{h,w}^2 = (d_{h,w} \ \% \ 2^5) \cdot 2^3$$

变量解释:
- $d_{h,w}$: 在像素 $(h, w)$ 处的 metric depth, 单位 millimeter
- $//$: 整除运算 (floor division)
- $\%$: 取模运算
- $2^{10} = 1024$ mm ≈ 1 m, 这是 robotic desktop grasping 的典型 depth 上界
- $2^5 = 32$ mm, 精细粒度
- 乘子 $2^1, 2^3, 2^3$ 是 spread multiplier, 把有效 bit 范围在 uint8 [0, 255] 中均匀分布

直觉:
- **Channel 0** ($I^0$): 编码 "米" 量级. $d = 50\text{m} = 50000\text{mm}$, $50000 // 1024 = 48$, 乘 2 得 96. 一个 channel 编码 0-127 m.
- **Channel 1** ($I^1$): 编码 "32 mm 量级". $d // 32$ 取 "32 mm 块的索引", 一个 channel 编码 0-1024 mm 的精细结构 (5 bits).
- **Channel 2** ($I^2$): 编码 $d \mod 32$, 即 0-31 mm 的 sub-32 mm 残差. 5 bits 残差, 乘 8 映射到 0-248.

合起来, 三个 channel 给你 7+5+5 = 17 bits 的 metric depth 表达能力, 且每个 channel 上的值都在 uint8 中尽量 spread, 避免所有值集中在 [0, 10] 这种 VLM 学不到的退化分布.

这种 encoding 的妙处: 它**保留了 metric 数值可减可加可比较**的性质. VLM 在文本空间见过数字比较, 所以从三个 channel 解出真实 mm 值是一个 lightweight 函数, LLM 应该能 fit. 这跟 ordinal encoding (只保序不保距) 形成对比.

---

## 4. Relative depth → Metric depth 的修正

MDE 模型 (Depth Anything, MiDaS) 输出的是 relative depth $d_r \in [0, 1]$, 大值表示近. 一篇 prior 工作 [Li et al. 2024, ProximityQA] 直接用 $\frac{1}{d_r}$ 作为 metric depth 标签 — 作者指出这是**数学上错误的**:

$$A = \frac{1}{d_{min}} - \frac{1}{d_{max}}, \quad B = \frac{1}{d_{max}}, \quad d = \frac{1}{A \cdot d_r + B}$$

变量:
- $d_r \in [0, 1]$: MDE 输出的 relative depth (越大越近)
- $d_{min}, d_{max}$: 图像中真实的最近/最远 metric depth
- $d$: 反算出的 metric depth

**为什么 $\frac{1}{d_r}$ 直接用作 metric depth 是错的**: MiDaS 学的是 $\frac{1}{d_r} = A' \cdot \frac{1}{d_{\text{true}}} + B'$, 即 inverse depth 的 affine 变换. 只有当 $d_{max} \to \infty$ 时, $B \to 0$, $\frac{1}{d_r} \propto \frac{1}{d}$, 此时 $\frac{1}{d_r}$ 才是 $d$ 的 scale multiple. 有限 $d_{max}$ 下, $\frac{1}{d_r} = 0.4$ **不等于** $\frac{1}{d_r} = 0.2$ 的两倍 depth.

ZoeDepth (https://arxiv.org/abs/2302.12288) 直接输出 metric depth, 所以本文用 ZoeDepth 而不是 MiDaS / Depth Anything.

---

## 5. SpatialQA: 三层 progressive QA pipeline

这是数据集设计的精髓. 让 VLM 渐进式学习 depth 的能力, 而不是一上来就让它做 spatial reasoning:

### Low-level: 让模型读 depth map
- 问 "像素 (x, y) 的 depth 值是多少" -> 模型直接从 depth input 抽数值
- 描述 depth map 内容, 推断可能有什么物体 (只看 depth)
- 目的: 让 vision encoder + projector 学会 encode depth map 的 token

### Middle-level: RGB-D 对齐
- proximity: "哪个点离 camera 更近"
- object-level depth 描述: 用 max/min/mean/center 四个值刻画一个 object 的 depth (用 95th/5th percentile 代替 true max/min 抗噪)
- object proximity: "杯子比盘子近还是远"
- 目的: 让模型学会 "在 RGB 上定位 object → 在 depth 上找对应 depth"

### High-level: spatial reasoning
- counting/enumeration (depth 提供 boundary 帮助 grounding)
- spatial relationship: above/below, left/right (real-world 不是 image coordinate), inside/outside, touching/reaching
- 在 robot manipulation 场景中: "拿最大的杯子放到左边的板上"

数据规模 (Table 3, https://huggingface.co/datasets/RussRobin/SpatialQA):

| Source | Num | 用途 | Depth 来源 |
|---|---|---|---|
| Bunny695k | 695k | general VLM 能力 | - |
| VG, COCO | 20k | depthmap understanding | MDE (ZoeDepth) |
| KITTI | 1.75k | outdoor spatial | MDE (sensor sparse) |
| NYU Depth v2 | 1.5k | indoor spatial | sensor |
| RT-X | 7.5k | robot scene | sensor/MDE 混合 |
| SA-1B | 15k | spatial | MDE |
| 2D-3D-S | 2.9k | indoor spatial | sensor |

总共约 750k image-text 对. 用 GPT-4o 在 50k 张图上 prompt 生成 QA, prompts 见 Table 4. prompt 的关键约束: "用 real-world 坐标系, 不是 image 坐标系".

---

## 6. SpatialQA-E: Embodied version

2000 episodes, Franka Research 3 7-DoF 机械臂. 任务: pick and place teacups / balls / bananas.

包含三种关系层级:
- **Positional**: left/right/middle/up/down/on/in/inside/outside
- **Size**: tall/short/large/small/wide/thin/big/small (含 comparative -er, superlative -est)
- **Illusion**: 把物体的照片打印出来放桌上, 看起来像真的. 模型必须用 depth (打印物是 flat) 和 shadow 区分真假物体.

这个 illusion 设计非常聪明 — 它把 depth 的物理意义逼到极限, 模型不能用 2D appearance shortcut.

### VLA action 表示
- 7-DoF: $(\Delta X, \Delta Y, \Delta Z, \Delta R, \Delta P, \Delta Yaw, C)$
  - $\Delta X, \Delta Y, \Delta Z$: 末端执行器位置 delta
  - $\Delta R, \Delta P, \Delta Yaw$: roll, pitch, yaw 旋转 delta
  - $C$: gripper closure
- 每个维度离散化为 101 bins: $\{0, 0.01, 0.02, ..., 1.0\}$
- 模型直接输出文本: `"The robot should <0.17, 0.51, 0.44, 0.62, 0.83, 0.07, 1>"`
- 用 4 帧 history 预测当前帧的 delta pose

两个工程 trick:
1. **自然语言 wrapper** "The robot should ..." — 让模型既能回答 general QA 又能输出 action, 不退化成 robot-only 模型
2. **Delta pose > Target pose**: delta pose 控制更精细, 但 loss 下降更慢, 有时甚至 collapse (输出常数). 解法: 指数级增加训练数据

---

## 7. SpatialBench & 主要实验结果

### Table 1: SpatialBench (6 个维度)

最 striking 的对比:
- **GPT-4o-RGB**: Position 70.6, Existence 85.0, Counting 84.5, Reaching 51.7, Size 43.3
- **GPT-4o-RGBD**: Position **61.8** (反而降!), Existence 90.0, Counting 85.2, Reaching 51.7, Size **40.0** (也降!)

这数据告诉你: **naive RGBD 输入对 GPT-4o 是负向的**. GPT-4o 不知道怎么用 depth, 把它当 noise 干扰了 spatial reasoning.

- **SpatialBot-Phi2-3B-RGBD**: Depth **>99**, Position 61.8, Existence 80.0, Counting 91.7, Reaching 55.0, Size 26.7
- 3B 模型就能在多数指标上接近 GPT-4o, Depth 任务靠 API 几乎满分.

### Table 2: General VLM benchmarks

| Model | MME-P | MME-C | MMB-T | MMB-D | SEED-I | VQA-v2 | GQA | POPE |
|---|---|---|---|---|---|---|---|---|
| Bunny-Phi2-3B | 1472/1474 | 286/285 | 67.9 | 68.9 | 79.0 | 61.5 | 86.2 | - |
| SpatialBot-Phi2-3B | **1483/1487** | **310/312** | 70.1 | 68.6 | **79.8** | **62.3** | **87.0** | - |

加了 depth 训练, 在 MME Perception, MME Cognition, GQA, POPE 上**普遍提升**. 这是个反直觉结果: depth 训练帮助模型做"看起来不需要 depth"的 general VQA. 解释: depth map 提供 boundary prior, 帮 grounding 和 counting; 同时模型学到更结构化的 scene representation, 对 spatial-language alignment 有正则化作用.

---

## 8. Robot manipulation 成功率 (Figure 8)

RGBD variant 在 pick-and-place 上成功率显著高于 RGB. 具体:
- 简单任务 (单 object pick): RGB vs RGBD 接近
- 复杂任务 (含 spatial relation + illusion + obstacle avoidance): RGBD 大幅领先

depth 在 "gripper 是否已经接触 object" (Figure 1 的核心 example) 上至关重要 — 人眼看 RGB 都判断不了, 模型从 depth 直接读 gripper tip 和 object surface 的 mm 级差.

---

## 9. 我的几个 critical 观察

### Positive
1. **Depth API 设计**是真有想法. 它把 spatial intelligence 拆成 "vision encoder 读 depth" (软) + "API 查询精确值" (硬) 两条路, 用 in-context injection. 这跟 Toolformer / ReAct 一脉相承, 但用得非常精准.
2. **三通道 uint8 metric encoding** 保留了数学可运算性, 这是相对于 ordinal encoding 的关键改进. 值得 future work 直接学习这种编码方式的设计空间.
3. **Progressive QA curriculum** 非常合理: low → middle → high, 每层都建立在前一层的 capability 上. 类似你在 nanoGPT 之外讨论过的 curriculum 效应.
4. **Illusion task** 设计巧妙: 打印物 vs 真物体, 逼模型必须用 depth, 任何 RGB shortcut 都失效.

### Negative / Open questions
1. **MDE on single point/object 精度有限** (Table 1 中 SpatialBot-RGB 的 Depth 列只有 ~80-90%, 远不如 RGBD 版本的 >99). 论文 §5 也承认 "VLMs has not been fully prepared for MDE in a text-only output fashion". 也就是说 VLM 的 tokenizer + decoder 不适合输出 dense continuous 数值, 这是个 fundamental issue. 真 dense depth 还是需要专门的 depth decoder.
2. **MME-C cognition 大幅提升** (286 → 310) 但 absolute 值仍然很低 (310/400 max?), 说明 spatial reasoning 远未解决.
3. **Size comparison 性能普遍很低** (16-43 范围). GPT-4o 也只有 43.3. 说明从单视角 RGBD 推断 object 实际尺寸极难, 可能需要 multi-view 或 prior knowledge.
4. **Delta pose collapse issue** (§3.6) 暴露了 autoregressive LLM 直接做控制信号的脆弱性. 离散化 100 bins 在某些数据上 loss 完全不降, 输出常数. 这暗示 VLA 的 action decoder 可能需要 non-autoregressive head (类似 diffusion policy 或 flow matching).
5. **Scale**: 50k GPT-4o-generated QA + 2000 robot episodes 是个小规模 pilot. 想真正 scale, 需要自动化的 depth-aware QA 生成 pipeline.
6. **Sharing encoder**: RGB 和 depth 共享 SigLIP encoder 是个 simplifying choice, 但可能限制了 depth 表达. 专门 train 一个 depth encoder (类似 DINOv2 那种 self-supervised) 可能更好.

---

## 10. 跟相关工作对比

- **SpatialVLM** (https://arxiv.org/abs/2401.12168): 只从 RGB 推 spatial, 没有 depth input. 在 embodied 场景下精度不够.
- **Spatial-RGPT** (https://arxiv.org/abs/2407.07699): 同样 RGB-only, 但加 region token 做 grounded reasoning.
- **Depth Anything** (https://arxiv.org/abs/2401.10891): pure MDE, 输出 dense depth, 但不接入 language reasoning.
- **RT-X / OpenVLA / Octo** (https://arxiv.org/abs/2310.08864, https://arxiv.org/abs/2406.09246, https://arxiv.org/abs/2405.12213): VLA 模型, SpatialBot 借鉴了它们的结构, 但加了 depth modality.

SpatialBot 的位置: 在 MDE (专注 dense depth) 和 SpatialVLM (专注 RGB spatial reasoning) 之间搭桥, 通过 RGBD input + Depth API 把 metric depth 注入 language reasoning.

---

## 11. Build intuition 的核心 takeaway

如果让我总结一个 mental model:

> **VLM 是个强大的 reasoning engine, 但它对 2D image 之外的 modality (depth, audio, proprioception) 的"消化能力"取决于你怎么把 modality 翻译成它已经懂的语言 (text token + 可比较数值)**.

SpatialBot 做的三件事:
1. 把 depth map 编码成 RGB-like image (3-channel uint8) 让 vision encoder 能 ingest
2. 用 metric 数值 (mm) 让 LLM 能直接做数学推理
3. 用 tool-use API 把"看不清的细节"外包给 deterministic lookup

这是 VLM modality extension 的通用模式. 我相信你做 Tesla autonomy 的时候也碰到过类似问题: 怎么把 BEV / occupancy / lidar point cloud 喂进 LLM 类的 reasoning head. SpatialBot 给的答案: encode 成 image-like + 数值 token + tool API. 不一定最优, 但 pragmatic.

代码: https://github.com/BAAI-DCAI/SpatialBot  
数据: https://huggingface.co/datasets/RussRobin/SpatialQA  
Base VLM: https://github.com/BAAI-DCAI/Bunny

期待你后续在 eureka labs / 教育内容里也能用类似思路讲 VLM 的 modality integration — 这是个很好的 build-intuition 例子.
