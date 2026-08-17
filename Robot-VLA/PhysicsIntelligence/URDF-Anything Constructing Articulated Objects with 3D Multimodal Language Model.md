---
source_pdf: URDF-Anything Constructing Articulated Objects with 3D Multimodal Language
  Model.pdf
paper_sha256: 99edb34a3f652df3fb76be73523a1d3f84f85a735db1abd81403a17bba159877
processed_at: '2026-08-12T20:33:42-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 URDF-Anything

## 一句话总结

你给它一个柜子的照片，它吐给你一个能直接扔进 MuJoCo 里跑的 URDF 文件——每个抽屉、每扇门都切好了 mesh，每个 joint 的类型、转轴、位置都填好了，拖一下抽屉就能动。

---

## 为什么要做这件事

Sim-to-real 训练机器人，最大的痛点之一是场景搭建。你想让机器人学开抽屉，就得在模拟器里有一堆带活动抽屉的柜子。但 URDF 文件要手动写——每个 link 的 mesh、每个 joint 的 origin 和 axis、parent-child 关系……一个复杂柜子能写一下午。

所以大家一直想：能不能拍张照片，自动生成 URDF？

这事儿难在哪：

1. 一个 articulated object 本质是 **一棵 kinematic tree**：base link 是根，每个 joint 连接 parent 和 child link，joint 有 type（revolute / prismatic / fixed）、origin（3D 位置 + 姿态）、axis（运动方向）、limit（范围）。你要把这些全 infer 出来。

2. 你还得把 point cloud **切成几个 part**，每个 part 对应一个 link 的 mesh。

3. 这两件事是耦合的——如果你把抽屉的边界切错了，joint 的 origin 和 axis 也大概率会错。传统方法分两步做（先 segmentation 再 predict joint），error 会累积。

---

## 他们的做法

### 输入：先把图片变成 point cloud

- 有多张图（multi-view）→ 用 **DUSt3R** 重建 dense point cloud
- 只有一张图（single-view）→ 用 **LGM** 先 diffusion 出 multi-view，再重建 point cloud

不管哪种，输出都是 $P_{obj} \in \mathbb{R}^{N \times 6}$（XYZ + RGB），注意这个 point cloud 是 **整坨的**，没有 part 分割信息。

### 核心：3D MLLM 联合推理

Backbone 是 **ShapeLLM**（一个 3D MLLM，point cloud encoder + LLM）。

输入有两路：
- Point cloud → Uni3D encoder → geometric features $F_{pc} \in \mathbb{R}^{M \times d_{pc}}$
- Text instruction → LLM embedding → text features $F_{txt}$

然后 LLM autoregressive 地吐出一个 JSON。

JSON 长这样：

```json
{
  "joints": [
    {"type": "revolute", "parent": "base", "child": "link_0",
     "origin": {"xyz": [...], "rpy": [...]}, "axis": [0,1,0], ...},
    ...
  ],
  "links": {
    "link_0": "drawer[SEG]",
    "link_1": "door[SEG]",
    ...
  }
}
```

注意每个 link 名字后面跟了个 `[SEG]`。这个 `[SEG]` 不是摆设——它是个 **trigger signal**，一出现就启动一个 cross-attention 机制，去 point cloud 里把这个 link 对应的 region 抠出来。

### [SEG] token 是怎么工作的

这是全文最精巧的设计。

当 LLM 生成到 `drawer[SEG]` 的时候：

1. 取 `[SEG]` 这个 token 的 hidden state $h_{seg}$
2. 取它前面 `drawer` 这个 category token 的 hidden state $h_{category}$
3. 拼起来：$h_{combined} = [h_{category}; h_{seg}]$
4. 过一个 MLP 得到 query：$H_{query} = \text{MLP}_{query}(h_{combined})$
5. 另一路把 point cloud 重新 encode：$S_{pc} = F_{enc}(P_{obj})$，投影成 $F'_{pc} = \text{MLP}_{pc}(S_{pc})$
6. Cross-attention：$y_{mask} = \text{CrossAttn}(Q=H_{query}, K=F'_{pc}, V=F'_{pc})$
7. Sigmoid + threshold → binary mask

每个 `[SEG]` token 都跑一遍这个流程，就得到所有 part 的 mask。

为什么要把 $h_{category}$ 和 $h_{seg}$ 拼起来？因为 category token 告诉你"我在找抽屉"，`[SEG]` token 告诉你"现在要触发 segmentation 了"，两个信息合起来 query 更精准。消融实验（Table 12）证明：只用 $h_{seg}$ 得 mIoU 0.58，拼接后 0.63，涨 5 个点。

### 输出：拼成 URDF

- 每个 part 的 point cloud mask → ball-pivoting algorithm → mesh (.obj)
- JSON 里的 joint 参数 → URDF XML 的 `<joint>` 标签
- 每个 `<link>` 引用对应的 mesh 文件

完成，直接扔进 MuJoCo / Sapiens 就能跑。

---

## 为什么 joint training 比 separate training 好

这是论文最有 insight 的部分。

他们做了个 ablation（Table 5）：

| Model | Type Error | Axis Error | Origin Error | mIoU | Count Acc |
|-------|-----------|-----------|-------------|------|-----------|
| 只训 kinematics（no [SEG]） | 0.009 | 0.138 | 0.175 | — | — |
| 只训 segmentation | — | — | — | 0.61 | 0.89 |
| **联合训练** | **0.008** | **0.132** | **0.164** | **0.63** | **0.97** |

两个方向都受益。

**Kinematics 受益于 segmentation**：如果只让 LLM 预测 joint 参数，没有 segmentation 任务约束它，它可能会 "hallucinate" 一个文字上说得通但物理上对不上的 joint——比如把抽屉的 axis 放到了柜子外面。Segmentation loss 强制模型把抽象的 joint 参数锚定到 point cloud 的具体几何区域上，相当于一个 geometric regularizer。

**Segmentation 受益于 kinematics**：如果只做 segmentation，模型不知道这些 part 之间怎么连接、谁是 parent 谁是 child。预测完整 kinematic tree 强制模型学一个更 coherent 的 object 表示。Count Acc 从 0.89 飙到 0.97，说明模型更不容易漏 part 或多 part。

Figure 9 的 attention visualization 直接展示了这个效果：
- 联合训练的模型在生成 axis token 时，attention 集中在物理 hinge 区域
- 只训 kinematics 的模型 attention 是 diffuse 的，没找到该看哪儿

---

## 为什么必须用 3D point cloud 而非 2D image

Table 4 的 ablation 很说明问题：

| Variant | Type Error | Axis Error | Origin Error |
|---------|-----------|-----------|-------------|
| OBB（bounding box 简化） | 0.42 | 0.70 | 0.47 |
| Point Cloud only | 0.34 | 0.29 | 0.26 |
| Qwen2.5-VL-7B（image MLLM） | 0.57 | 0.85 | 0.23 |
| Qwen2.5-VL-7B + fine-tune | 0.38 | 0.81 | 0.18 |
| **Point Cloud + Text** | **0.008** | **0.132** | **0.164** |

即使是 fine-tuned 的 Qwen2.5-VL-7B（很强的 image MLLM），Axis Error 依然 0.81 radians——这接近 50 度的误差，joint 轴基本上是瞎猜的。

原因很直觉：joint axis 是 3D 空间里的方向向量，2D image 有投影 ambiguity。一张正面拍的照片，你根本看不出抽屉是往里推还是往外拉。Point cloud 直接给你 3D 几何，axis 方向就在数据里。

OBB 也不行（0.70 Axis Error），因为 bounding box 把几何细节抹掉了——一个抽屉的把手、滑轨形状都丢了，你拿什么推断 axis？

---

## 实验结果到底多强

### Part Segmentation（Table 1）

mIoU 0.63，比最好的 baseline（Uni3D w/ text 0.54）高 16.7%。OOD 物体上 0.62 vs 0.51，generalization 很强。

Count Accuracy 0.97，说明几乎不会漏 part 或多 part。这个对 URDF 很关键——少一个 part 整个 kinematic tree 就错了。

### Joint Parameter Prediction（Table 2）

Type Error 0.008（几乎不犯错），Axis Error 0.132 rad（约 7.5 度），Origin Error 0.164 米。

对比 Real2Code Oracle（给 GT segmentation 的 baseline）：Type 0.537，Axis 1.006，Origin 0.294。URDF-Anything 即使不给 GT segmentation 也比 Oracle baseline 强一个量级。

### Physical Executability（Table 3）

78% 的生成 URDF 能直接在 simulator 里加载并 actuate，比 Articulate-Anything（52%）高 50%。

OOD 物体上 71%，这意味着见到训练时完全没见过的类别（41 个 OOD 类别 vs 5 个 ID 类别），依然有 71% 的 URDF 能正常工作。

### Sim-to-Real（Table 11）

只在 PartNet-Mobility（synthetic）上训练，直接测 PARIS real-world dataset：
- Fridge: Type Error 0.0，Axis Error 0.335，Origin Error 0.256
- Storage: Type Error 0.0，Axis Error 0.362，Origin Error 0.349

Type Error 全对，axis 和 origin 误差比 sim 上大一些但合理。Zero-shot sim-to-real 跑通。

---

## 跟其他方法比定位

| 方法 | 核心思路 | 问题 |
|------|---------|------|
| **Real2Code** | OBB 简化 part + LLM 预测 joint code | OBB 丢几何细节，sequential pipeline 误差累积 |
| **Articulate-Anything** | Actor-critic iterative refinement + mesh retrieval | 依赖 mesh database，iterative process 容易 brittle |
| **URDFormer** | Image → pipeline → simulation scene | Hard-coded system 分配 kinematic 参数，fidelity 差 |
| **CARTO** | Feed-forward encoder-decoder | 只能 single-joint objects，不能 segmentation |
| **PARIS** | Per-instance optimization | 需要 start/end state 图像，>3min 一个物体 |
| **URDF-Anything** | 3D MLLM end-to-end joint prediction | 数值精度受 token 限制，不能预测 mass/inertia |

URDF-Anything 的独特定位：**第一个用 raw 3D point cloud 作为 MLLM 主输入、end-to-end 输出 URDF 的方法**。其他方法要么用 2D image（精度差），要么用 OBB（信息丢失），要么是 multi-stage pipeline（误差累积）。

---

## 我觉得最 clever 的几个点

1. **[SEG] token 把 segmentation 嵌进 autoregressive generation 流程里**。传统做法 segmentation 是个独立的 dense prediction 任务，这里它变成了 LLM 输出 sequence 的一部分。LLM 在 "说出" `drawer[SEG]` 的那一刻，就同时完成了"我知道这里有抽屉"和"我要把它的几何区域抠出来"两件事。这种 coupling 比 "先 segmentation 再 predict joint" 自然得多。

2. **Context fusion（$h_{category} + h_{seg}$）**。如果只用 `[SEG]` 的 hidden state 做 query，模型只知道"要分割"，但不知道"分割什么"。加上前面 category token 的 state，query 就有了语义方向。这是个很小但有效的 trick。

3. **Joint optimization 的 mutual regularization**。这个 insight 可以推广：任何需要同时做 dense prediction 和 structured symbolic output 的任务，都可能从 joint training 中受益。Segmentation 提供 geometric grounding，symbolic prediction 提供 structural prior，两者通过 shared representation 互相约束。这个思想对 general embodied AI 的 world model 构建也很有启发。

4. **ShapeLLM 的选择**。它天然能输出 3D bounding box 坐标，说明 backbone 已经有 3D 空间推理的 prior。在这个基础上 fine-tune 到 URDF prediction，比从头训一个模型效率高得多。2.5 hours on 1× A800 就训完了，cost 很低。

---

## Limitations 与未来方向

1. **数值精度受 token 限制**。LLM 输出的是 token，坐标值本质上是离散的。比如 origin xyz = [-0.079, -0.487, 0.0]，每个数字是独立 token。如果你想更精确，比如 [-0.07923, ...]，token 粒度可能不够。未来可能需要某种 continuous output head 或 pointer network 机制。

2. **不能预测 inertial properties**（mass, moment of inertia）。这对物理模拟很重要——没有质量，重力下行为就不对。这受 training data 限制，PartNet-Mobility 的 URDF 可能没标这些。

3. **Point-to-mesh 还是 external module**。Ball-pivoting algorithm 是个经典几何方法，但不是 end-to-end 的。未来可能用 3D Gaussian Splatting 或 occupancy network 直接从 point cloud 生成 mesh，完全打通 pipeline。

4. **Part 数量限制**（<8 parts）。复杂物体（比如一个有 20 个零件的机器）可能超出当前能力。这与 autoregressive 生成长度有关，LLM 输出太长的 JSON 容易出错。

5. **Sim-to-real gap 仍在**。Table 11 显示 axis/origin error 在 real data 上明显比 sim 大，说明 DUSt3R/LGM 重建的 point cloud 质量直接影响了下游精度。如果 point cloud 不好，再强的 MLLM 也救不回来。

---

## 更大的图景

这篇工作其实指向一个趋势：**3D MLLM 正在成为 embodied AI 的 core component**。

传统 pipeline：perception module → segmentation module → kinematic inference module → URDF writer，每个 module 独立训练，interface 靠 hand-crafted 规则连接。

URDF-Anything 展示的 paradigm：一个 3D MLLM 吃 raw 3D + language，吐 structured output（JSON + [SEG] token），end-to-end joint training。这跟 LLM 在 NLP 里取代 multi-stage pipeline 的趋势完全一致。

更远地想，如果 3D MLLM 能理解 articulated structure，下一步可能是：
- 理解 **affordance**（哪个 part 能怎么动，用来做什么）
- 理解 **interaction dynamics**（推抽屉需要多大力，门会怎么转）
- 直接从 visual observation **generate manipulation policy**

这就像从"静态识别物体"进化到"理解物体能怎么用"，而后者才是 robot learning 真正需要的 world model。

参考 ShapeLLM: https://arxiv.org/abs/2406.10218 | Uni3D: https://arxiv.org/abs/2310.06773 | LISA: https://arxiv.org/abs/2308.00692 | DUSt3R: https://arxiv.org/abs/2404.19764 | LGM: https://arxiv.org/abs/2402.06054 | PartNet-Mobility: https://arxiv.org/abs/1812.02719 | MuJoCo: https://arxiv.org/abs/2304.13648 | Sapiens: https://arxiv.org/abs/2408.12569 | LoRA: https://arxiv.org/abs/2106.09685 | AdamW: https://arxiv.org/abs/1711.05101 | Real2Code: https://arxiv.org/abs/2406.08588 | Articulate-Anything: https://arxiv.org/abs/2501.01836 | URDFormer: https://arxiv.org/abs/2406.09965 | CARTO: https://arxiv.org/abs/2306.07752 | PARIS: https://arxiv.org/abs/2210.09890 | Ball-pivoting: https://ieeexplore.ieee.org/document/790540 | Qwen2.5-VL: https://arxiv.org/abs/2502.13923

---

# URDF-Anything: 3D Multimodal LLM 重建铰接物体的端到端框架

## 1. 核心问题与 Motivation

铰接物体(articulated objects,如门、抽屉、剪刀)由多个 link 通过 joint 连接,既有几何属性又有运动学属性。传统方法需要手动建模或采用多阶段 pipeline(先分割 part,再单独预测 joint 参数),存在误差累积与几何信息丢失问题。URDF-Anything 提出用 3D MLLM 一次性完成 "geometric segmentation + kinematic parameter prediction" 的联合推理,直接输出可导入 MuJoCo / Sapiens 的 URDF 文件。

关键 insight: joint 参数(origin, axis)本质上是 3D 空间中的坐标与向量,3D MLLM 原生具备 3D 空间推理能力,比 2D image MLLM 更适合这个任务。

---

## 2. 完整 Pipeline 解析

### Stage 1: Input Representation (Section 3.2)

根据输入模态自适应生成 dense point cloud $P_{obj} \in \mathbb{R}^{N \times 6}$:

- **Multi-view input**: 用 **DUSt3R** ([paper](https://arxiv.org/abs/2404.19764)) 建立 dense 2D-to-3D correspondence,输出 dense point cloud
- **Single-view input**: 用 **LGM** ([paper](https://arxiv.org/abs/2402.06054)) 中的 diffusion model 合成 consistent multi-view 图像,重建为 3D Gaussian Splatting,再转换为 point cloud

关键点:输出 $P_{obj}$ 是 monolithic(整体)表示,没有 part-level segmentation,这部分留给 MLLM 推断。每个点包含 6 维 (XYZ + RGB)。

### Stage 2: Articulation Parsing with 3D MLLM (Section 3.3)

Backbone: **ShapeLLM** ([paper](https://arxiv.org/abs/2406.10218)),其架构为 `point cloud encoder + LLM`。

**Encoding 阶段**:
- Point cloud $P_{obj}$ 经过 **Uni3D** encoder ([paper](https://arxiv.org/abs/2310.06773)) 得到 dense geometric features:
  $$F_{pc} \in \mathbb{R}^{M \times d_{pc}}$$
  其中 $M$ 是点数(可能经过下采样),$d_{pc}$ 是 feature 维度。
- Text instruction $X_{txt}$ 经过 LLM 的 word embedding layer 得到:
  $$F_{txt} \in \mathbb{R}^{L \times d_{txt}}$$
  其中 $L$ 是 token 数,$d_{txt}$ 是 embedding 维度。

**Multimodal fusion 与 generation**:
$$Y_{output} = \text{MLLM}(F_{pc}, F_{txt})$$

MLLM 自回归生成 JSON 格式输出,同时:
- 预测 joint 参数(type, origin, axis, parent/child, limit)
- 在每个 link 描述后插入 `[SEG]` token(受 LISA [paper](https://arxiv.org/abs/2308.00692) 启发)

输出 JSON 结构示例(Faucet,4 parts):
```json
{
  "joints": [
    {
      "id": "joint_0",
      "type": "revolute",
      "parent": "base",
      "child": "link_0",
      "origin": {"xyz": [-0.079, -0.487, 0.0], "rpy": [1.5708, 0.0, 1.5708]},
      "axis": [0.0, 1.0, 0.0],
      "limit": {"lower": 0, "upper": 1.57}
    },
    ...
  ],
  "links": {
    "link_0": "switch[SEG]",
    "link_1": "switch[SEG]",
    "link_2": "spout[SEG]",
    "link_3": "faucet_base[SEG]"
  }
}
```

### Stage 3: Geometric Segmentation via [SEG] Token (Section 3.4)

这是论文最核心的创新。对每个 `[SEG]` token:

1. **Context fusion**: 取 `[SEG]` token 的 final hidden state $h_{seg}$,与它前面 category token 的 state $h_{category}$ 拼接:
   $$h_{combined} = [h_{category}; h_{seg}]$$

2. **Query 生成**:
   $$H_{query} = \text{MLP}_{query}(h_{combined})$$

3. **Point feature 投影**: 从 3D backbone 重新提取 point feature $S_{pc} = F_{enc}(P_{obj})$,然后投影:
   $$F'_{pc} = \text{MLP}_{pc}(S_{pc})$$

4. **Cross-attention 计算 per-point score**:
   $$y_{mask} = \text{CrossAttn}(Q=H_{query}, K=F'_{pc}, V=F'_{pc})$$

5. Sigmoid + threshold 得到该 part 的 binary mask。

对每个 `[SEG]` token 重复此过程,得到所有 part 的 mask。

**Context fusion 的作用**(Table 12 消融):相比仅用 $h_{seg}$(mIoU 0.58),用 $h_{combined} = [h_{category}; h_{seg}]$ 可达 mIoU 0.63,提升 5 个点。这说明 category token 的语义信息对 segmentation 起到关键 context 引导作用。

### Stage 4: Mesh Conversion & URDF Generation (Section 3.5)

- 分割后的 point cloud → mesh:用 ball-pivoting algorithm ([paper](https://ieeexplore.ieee.org/document/790540)) 和 alpha-shape 方法
- MLLM 输出的 JSON → URDF XML,每个 link 引用其 mesh,joint 填入预测参数
- 输出可直接导入 **MuJoCo** ([paper](https://arxiv.org/abs/2304.13648)) 或 **Sapiens** ([paper](https://arxiv.org/abs/2408.12569))

---

## 3. Training Objective 详解 (Section 3.6)

总 loss:
$$L = \lambda_{text} L_{text} + \lambda_{seg} \sum_{i=1}^{N} L_{i,seg}$$

变量解释:
- $\lambda_{text}, \lambda_{seg}$: 平衡两个 loss 的超参数
- $N$: object 的 part 数量
- $L_{text}$: language modeling loss(standard cross-entropy on token prediction)
- $L_{i,seg}$: 第 $i$ 个 part 的 segmentation loss

Segmentation loss 由 BCE + Dice 组成:
$$L_{seg} = \lambda_{bce} \text{BCE}(\hat{M}, M_{gt}) + \lambda_{dice} \text{DICE}(\hat{M}, M_{gt})$$

- $M_{gt} \in \{0,1\}^M$: 该 part 的 ground truth binary mask
- $\hat{M}$: 模型预测的 mask
- $\lambda_{bce}, \lambda_{dice}$: BCE 与 Dice loss 的权重
- **BCE** 负责像素级正确性,**Dice** 负责处理类别不平衡(small parts)

**Implementation details**:
- ShapeLLM-7B-general-v1.0 checkpoint
- LoRA rank = 8
- AdamW optimizer, lr = 3e-4, weight_decay = 0
- Cosine schedule with 0.03 warm-up ratio
- Batch size 2 per device, gradient accumulation = 10
- 训练时间: 2.5 hours on 1× NVIDIA A800 80GB

---

## 4. 实验数据深度解读

### 4.1 Part Segmentation (Table 1)

| Model | ALL mIoU | ID mIoU | OOD mIoU | ALL Count Acc | ID Count Acc | OOD Count Acc |
|-------|----------|---------|----------|---------------|--------------|----------------|
| Uni3D w/o text | 0.36 | 0.50 | 0.33 | 0.73 | 0.83 | 0.70 |
| Uni3D w/ text | 0.54 | 0.64 | 0.51 | 0.84 | 0.91 | 0.82 |
| **URDF-Anything** | **0.63 (+16.7%)** | **0.69** | **0.62** | **0.97 (+15.4%)** | **0.99** | **0.96** |

- ID (In-Distribution) 类别: Laptop, Box, Refrigerator, StorageFurniture, Table
- OOD (Out-of-Distribution) 类别: 其余 41 个类别
- Uni3D w/o text 弱,因为只靠 geometric features 缺少 semantic guidance
- Uni3D w/ text 比 w/o text 强很多,说明 text-aligned feature 有效
- URDF-Anything 全面碾压,OOD Count Acc 达 0.96,几乎不漏 part

**直觉解释**: [SEG] token 是 dynamic context-aware 的,每次出现都关联一个具体的 category token,使 segmentation 与 kinematic 结构推理互锁。Uni3D w/ text 用固定 prompt list,无法应对结构变化。

### 4.2 Joint Parameter Prediction (Table 2)

| Method | ALL Type↓ | ALL Axis↓ | ALL Origin↓ | OOD Type↓ | OOD Axis↓ | OOD Origin↓ |
|--------|-----------|----------|--------------|------------|------------|--------------|
| Real2Code Oracle | 0.537 | 1.006 | 0.294 | 0.576 | 0.937 | 0.272 |
| URDFormer Oracle | 0.556 | 0.374 | 0.581 | 0.609 | 0.643 | 0.513 |
| Articulate-Anything | 0.025 | 0.145 | 0.207 | 0.026 | 0.145 | 0.208 |
| **URDF-Anything** | **0.008** | **0.132** | **0.164** | **0.009** | **0.136** | **0.173** |

指标说明:
- **Type Error**: joint type 预测错误比例(revolute vs prismatic 等)
- **Axis Error**: 预测轴与 GT 轴的角度差(radians,归一化到 $[0, \pi]$)
- **Origin Error**: joint origin 的位置误差(meters)

"Oracle" 表示给 baseline 提供 GT segmentation,只评估 kinematic prediction 部分,这是为了公平比较(因为 baseline 本身做不了 joint segmentation)。URDF-Anything 即使不做 oracle 仍然全面胜过 oracle baseline,说明 joint prediction 本身就更强。

Articulate-Anything 在 Type Error 上表现接近 URDF-Anything(0.025 vs 0.008),但它的 axis/origin error 明显高,可能因为它依赖 iterative refinement 但仍受 mesh retrieval 限制。

### 4.3 Physical Executability (Table 3)

| Method | ALL | ID | OOD |
|--------|-----|-----|------|
| URDFormer Oracle | 24% | 34% | 15% |
| Real2Code Oracle | 41% | 49% | 23% |
| Articulate-Anything | 52% | 61% | 44% |
| **URDF-Anything** | **78% (+50%)** | **86%** | **71%** |

这个 metric 测试生成的 URDF 能否在 simulator 中正确加载并 actuate,不出现 parts flying off、joints freezing、unexpected rotations 等非物理行为。

OOD executability 达 71%,说明即使遇到训练时未见过的物体类别,模型依然能生成结构合理的 URDF。这是端到端 joint reasoning 的直接收益。

### 4.4 Input Modality Ablation (Table 4)

| Variant | Modality | Type↓ | Axis↓ | Origin↓ |
|---------|----------|--------|--------|----------|
| OBB | Text | 0.42 | 0.70 | 0.47 |
| Point Cloud only | Point Cloud | 0.34 | 0.29 | 0.26 |
| Qwen2.5-VL-7B | Image+Text | 0.57 | 0.85 | 0.23 |
| Qwen2.5-VL-7B + ft | Image+Text | 0.38 | 0.81 | 0.18 |
| **Point Cloud + Text (Ours)** | Point Cloud + Text | **0.008** | **0.132** | **0.164** |

重要观察:
- **Qwen2.5-VL-7B 即使 fine-tune 也只能做到 0.38 Type Error**,远差于 point cloud 方法。Image-based MLLM 难以精确推断 3D kinematic 参数,因为 2D 到 3D 的 ambiguity 太大
- **OBB 简化几何丢失关键信息**:0.42 Type Error,0.70 Axis Error。bounding box 抽象掉了几何细节,无法准确推断 axis 方向
- **Point Cloud only 也不够好**:0.34 Type Error,虽然比 OBB 强,但缺少语言引导
- **Point Cloud + Text 完胜**:language 提供对象类别、part 关系的 prior,与 point cloud 互锁

### 4.5 Joint Prediction Ablation (Table 5)

| Model | Loss | Type↓ | Axis↓ | Origin↓ | mIoU | Count Acc |
|-------|------|--------|--------|----------|-------|-----------|
| Kinematics-Only | $L_{text}$ | 0.009 | 0.138 | 0.175 | - | - |
| Segmentation-Only | $L_{seg}$ | - | - | - | 0.61 | 0.89 |
| **Joint** | $L_{text} + L_{seg}$ | **0.008** | **0.132** | **0.164** | **0.63** | **0.97** |

这是论文最重要的 ablation。两个方向都受益于联合训练:
- **Kinematics 受益于 segmentation**: segmentation 提供 geometric regularization,防止 LLM hallucinate 物理上不存在的 joint
- **Segmentation 受益于 kinematics**: 预测完整 kinematic tree 强制模型学习 object 的 coherent 内部表示,反过来提升 segmentation 精度。mIoU 0.61 → 0.63,Count Acc 0.89 → 0.97

**直觉**:单独做 segmentation 时,模型不知道 parts 之间如何连接,可能把不属于同一 kinematic unit 的几何区域合并或拆错。Joint prediction 提供结构性约束。

### 4.6 Shape Reconstruction Quality (Table 6, Chamfer Distance)

| Method | ALL | ID | OOD |
|--------|------|-----|------|
| CARTO | 1.24 | 0.88 | 1.27 |
| PARIS | 3.06 | 2.17 | 3.13 |
| **URDF-Anything** | 1.39 | **0.40** | 1.51 |

ID 类别的 CD 仅 0.40,远好于 CARTO(0.88)与 PARIS(2.17)。OOD 类别略高于 CARTO,但仍优于 PARIS。这说明在 distribution 内,我们方法的 mesh 重建质量极佳。

### 4.7 Failure Analysis (Table 7, Table 8)

总体 failure rate 22%,细分:
- **Incorrect joint parameters**: 21%
- **JSON Format Error**: 1%

MLLM 几乎不会生成语法错误的 JSON(只有 1%),主要失败原因是 joint 参数误差导致非物理行为。这说明提升 joint prediction 精度是未来工作重点。

按类别分(Table 8):
- Window: 15.5% 失败,JSON error 0%,Type error 0.03%
- Chair: 22.2% 失败
- Globe: 14.8% 失败

### 4.8 Speed Comparison (Table 9)

| Method | Methodology | Inference Time |
|--------|-------------|----------------|
| CARTO | Feed-forward Encoder-Decoder | 1s |
| PARIS | Per-instance Optimization | >3min |
| **URDF-Anything** | Feed-forward MLLM Inference | 13s |

URDF-Anything 是 feed-forward 推理,13s 远快于 PARIS 的 per-instance optimization。虽然比 CARTO 慢(13s vs 1s),但 CARTO 只能处理 single-joint objects,能力受限。

### 4.9 Sim-to-Real Zero-Shot Generalization (Table 11)

仅用 PartNet-Mobility 训练,直接在 PARIS real-world dataset 上测试:

| Category | mIoU | CD | Type Error | Axis Error | Origin Error |
|----------|-------|------|-------------|-------------|----------------|
| Fridge | 0.57 | 1.03 | 0.0 | 0.335 | 0.256 |
| Storage | 0.56 | 0.99 | 0.0 | 0.362 | 0.349 |

Type Error 为 0,说明模型即使在 real-world 上也能正确判断 joint 类型。mIoU ~0.57 与 CD ~1.0 表明 sim-to-real gap 仍存在但可接受。

---

## 5. Dataset 与训练细节 (Appendix A.1)

- PartNet-Mobility dataset ([paper](https://arxiv.org/abs/1812.02719)) 的 URDF annotations
- **Coordinate normalization & URDF regularization**: 每个 object 建立一致 `base` link 作为 kinematic tree root。非 base 的 joint 重新 parent 到 base,计算 base → child 的 transform 并填入 `<origin>` 标签
- **Mesh consolidation**: 原 URDF 中 link 可能有多个 `<visual>` 或 `<collision>` 元素,合并为单个 representative mesh,保留第一个 mesh 的 local transform
- **Part count filter**: 只保留 articulated parts 少于 8 的 objects
- **Rendering**: SAPIENS simulator 渲染,采用 equator plane 与 spherical 分布(用 minimum potential energy method 保证均匀覆盖)两种策略

---

## 6. Input/Output Design (Appendix A.3)

### Input Prompt Template
训练时使用多种模板,提供不同粒度的信息:

```
The articulated object [Object Category] consists of [Number of Parts] parts.
[Optional descriptive phrases]
Predict all joint parameters in JSON format, including type, origin, axis, parent, and child.
Segment each link in JSON format.
```

**关键设计**:部分模板包含 "number of parts" 作为 pedagogical signal,帮助模型学习 visual form 与 articulated structure 之间的 correlation。**Inference 时不提供 parts 数量**,所有 quantitative 评估使用不含 parts 数量的 generic prompt,确保公平比较。

### Output Format
JSON 包含两个 key:
- `"joints"`: list,每个 joint 含 id, type, parent, child, origin (xyz + rpy), axis, limit
- `"links"`: object,link name → category + `[SEG]`

`[SEG]` token 的位置就是触发 segmentation 的信号。

---

## 7. Geometric Regularization 理论解释 (Appendix A.9)

论文提出一个重要 theoretical insight:**Segmentation task acts as geometric regularization for kinematic prediction**。

**核心原理**:仅预测 kinematics 的模型可能 hallucinate 文本上合理但物理上无根据的 joint。Joint optimization 强制模型把抽象的 kinematic prediction 锚定到 point cloud 的具体几何 entity 上,约束参数搜索空间,引导模型走向物理一致的 articulation structure。

**Qualitative evidence**(Figure 9):可视化模型生成 kinematic token(如 axis)时的 self-attention map:
- 完整模型(joint training):attention 集中在物理相关的 joint region(hinge)
- 无 segmentation loss 的模型:attention 漫散,缺少精确 geometric grounding

这直接证明 joint optimization 让模型把抽象 kinematic concept 锚定到具体 geometric feature 上,而 degenerated model 则在 "想象" joint。

---

## 8. 整体架构图解读

```
[Single/Multi-view Image]
         │
         ├── Multi-view → DUSt3R → Dense Point Cloud P_obj ∈ R^(N×6)
         └── Single-view → LGM (diffusion + 3DGS) → P_obj
                            │
                            ▼
                  [Uni3D Encoder]
                            │
                            ▼
              F_pc ∈ R^(M × d_pc) ─────────┐
                                            │
[Text Instruction] → [Word Embedding] ──→ │
              F_txt ∈ R^(L × d_txt)         │
                                            ▼
                                  [ShapeLLM (LoRA)]
                                            │
                            ┌───────────────┴────────────────┐
                            ▼                                 ▼
                  JSON output (joints)              [SEG] tokens
                            │                                 │
                            │                                 ▼
                            │           h_combined = [h_category; h_seg]
                            │                                 │
                            │                                 ▼
                            │                        H_query = MLP_query(h_combined)
                            │                                 │
                            │                  S_pc = F_enc(P_obj) → F'_pc = MLP_pc(S_pc)
                            │                                 │
                            │                                 ▼
                            │                  y_mask = CrossAttn(Q=H_query, K=F'_pc, V=F'_pc)
                            │                                 │
                            │                                 ▼
                            │                       Binary Mask per part
                            │                                 │
                            ▼                                 ▼
              Parse JSON → Joint params        Point-to-Mesh (Ball-pivoting)
                            │                                 │
                            └─────────────┬─────────────────────┘
                                          ▼
                                [URDF XML File]
                                          │
                                          ▼
                            [MuJoCo / Sapiens Simulation]
```

---

## 9. 与 baseline 方法的对比定位

### vs. Real2Code ([paper](https://arxiv.org/abs/2406.08588))
- 用 OBB(Oriented Bounding Box)简化 part geometry,然后用 fine-tuned LLM 预测 joint 参数为 code
- 问题:OBB 丢失几何细节,且 sequential pipeline 易误差累积
- URDF-Anything 用原始 point cloud 保留几何精度,end-to-end 减少误差传播

### vs. Articulate-Anything ([paper](https://arxiv.org/abs/2501.01836))
- Actor-critic iterative refinement + mesh retrieval 机制
- 依赖 mesh database,iterative process 可能 brittle
- URDF-Anything 是 single-pass feed-forward,不依赖外部 asset

### vs. URDFormer ([paper](https://arxiv.org/abs/2406.09965))
- 直接从 real-world images 构建 simulation scene
- hard-coded system 根据 network 离散 part classification 分配 kinematic 参数,compromise 几何与运动学 fidelity
- URDF-Anything 用 MLLM 联合推理,避免 hard-coded assignment

### vs. CARTO ([paper](https://arxiv.org/abs/2306.07752))
- Feed-forward encoder-decoder,fast 但 only 支持 single-joint objects
- 无法做 part segmentation 或生成完整 URDF

### vs. PARIS ([paper](https://arxiv.org/abs/2210.09890))
- Optimization-based,需要 start/end state 图像
- Per-instance optimization >3min,impractical
- URDF-Anything feed-forward 13s

---

## 10. Limitations

1. **无法生成某些 URDF 属性**:mass、moment of inertia,受 training data 与 base model 约束
2. **Pipeline 不完全 end-to-end**:依赖 external point-to-mesh conversion module
3. **数值精度受 token-based generation 限制**:LLM 输出 token,数值精度有限制

---

## 11. 我的 Intuition 与思考

这篇论文的核心 insight 可以归纳为三点:

**第一**,3D point cloud 作为输入比 2D image 更适合 kinematic prediction。Table 4 的 ablation 直接证明,即使是 fine-tuned Qwen2.5-VL-7B 也无法从 2D image 精确推断 3D joint axis/origin,因为 2D-to-3D 的 inherent ambiguity 太大。Point cloud 直接保留 3D 几何信息,MLLM 在 3D 空间中推理坐标与向量。

**第二**,[SEG] token 机制巧妙地把 segmentation 与 autoregressive generation 耦合。传统做法是 segmentation 和 parameter prediction 分两步,先分割再预测。这里把它们放在同一个 autoregressive generation 流程中,让 LLM 在输出每个 link 描述时同时触发对应 point cloud region 的 segmentation。这种 tight coupling 使 segmentation 与 kinematic structure 一致,避免分两步时的不一致性。

**第三**,joint optimization 提供 mutual regularization。Table 5 的 ablation 显示,即使 segmentation-only 与 kinematics-only 分别能独立工作,联合训练使两者都受益。这个 mutual regularization 的机制是:segmentation 给 kinematic prediction 提供 geometric anchor,防止 hallucination;kinematic structure prediction 给 segmentation 提供 structural prior,使模型学到更 coherent 的 object representation。Figure 9 的 attention visualization 直接证明 joint training 让模型在生成 axis token 时 attention 集中在物理 joint region,而 degenerate model 的 attention 是 diffuse 的。

从 Karpathy 的视角看,这其实是一种特殊的 multi-task learning,两个 task 通过 shared representation 互相提供 inductive bias。Segmentation 是 dense prediction(每点一个 label),kinematic prediction 是 structured symbolic output(JSON),两者通过 [SEG] token 在 autoregressive sequence 中桥接。这种设计让人联想到 LISA 在 2D reasoning segmentation 中的思路,但这里扩展到 3D 且与 URDF 结构化输出深度耦合。

未来工作方向可能包括:
- 端到端的 point-to-mesh 生成(避免 external module)
- 加入 inertial properties(mass, moment of inertia)
- 处理更多 parts(>8)的复杂 object
- 用更大的 base model 提升数值精度

---

## 参考 Links

- **ShapeLLM**: https://arxiv.org/abs/2406.10218
- **Uni3D**: https://arxiv.org/abs/2310.06773
- **LISA**: https://arxiv.org/abs/2308.00692
- **DUSt3R**: https://arxiv.org/abs/2404.19764
- **LGM**: https://arxiv.org/abs/2402.06054
- **PartNet-Mobility**: https://arxiv.org/abs/1812.02719
- **MuJoCo**: https://arxiv.org/abs/2304.13648
- **Sapiens**: https://arxiv.org/abs/2408.12569
- **LoRA**: https://arxiv.org/abs/2106.09685
- **AdamW**: https://arxiv.org/abs/1711.05101
- **Real2Code**: https://arxiv.org/abs/2406.08588
- **Articulate-Anything**: https://arxiv.org/abs/2501.01836
- **URDFormer**: https://arxiv.org/abs/2406.09965
- **CARTO**: https://arxiv.org/abs/2306.07752
- **PARIS**: https://arxiv.org/abs/2210.09890
- **Ball-pivoting algorithm**: https://ieeexplore.ieee.org/document/790540
- **Qwen2.5-VL**: https://arxiv.org/abs/2502.13923
