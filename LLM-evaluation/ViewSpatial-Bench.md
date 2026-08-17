---
source_pdf: ViewSpatial-Bench.pdf
paper_sha256: 590331d7880a072cb30a7ebade16cd19d00e265054c68cbb2718fb0c90f2fc55
processed_at: '2026-08-13T01:12:19-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版: 这篇 paper 在搞啥

## 一句话总结

现在的 VLM(像 GPT-4o 这种多模态模型)看起来很聪明, 但其实有个很蠢的盲点: **它不会换位思考**。

你跟它说 "我右边的杯子", 它经常搞不清是"你的右边"还是"相机的右边"。GPT-4o 在这个 benchmark 上只答对 35%, 比瞎蒙(26%)高不到哪去。

---

## 这事儿有多严重?

想象一下你让机器人 "把右边的杯子递给我"。如果机器人理解成"相机画面的右边", 那它可能递错一个杯子。这就是当前所有 top VLM 的真实水平——哪怕参数堆到千亿、训练数据喂了几亿张图, 这个基本能力依然崩。

paper 里测了一堆 model:

- GPT-4o: 34.98%
- Gemini-2.0-Flash: 32.56%
- Qwen2.5-VL-7B: 36.85%
- Random baseline: 26.33%

差距小得可怜。说明这些 model 根本没学到"视角"这个 concept, 它们只是在做 pattern matching。

---

## 为什么 VLM 这么差?

作者的核心 hypothesis 很简单: **training data 没暴露这个 signal**。

web 上的 image-text pair 空间描述本来就稀疏, 而且几乎全是 implicit 的 camera 视角——"the cup on the left" 默认就是相机视角, 没人标注"从那个人角度看, 杯子在右边"。model 从来没见过 explicit 的 perspective label, 自然学不会。

---

## Benchmark 怎么设计的?

作者设计了 5 种 task, 核心是两个视角的对比:

**Camera 视角**(模型本来就该擅长的):
1. 杯子相对于桌子在哪?
2. 图里这个人朝哪个方向看?

**Person 视角**(要求换位思考):
3. 假设你是图里的人 A, 人 B 在你哪个方向?
4. 假设你站在那个人的位置, TA 在看哪边?
5. 假设你站在书架前面朝窗户, 沙发在哪? ——这个最狠, 要求 model 在脑子里做个 3D mental rotation

第 5 个 task 是精髓。它逼 model 真的"走到" object1 的位置, "转向" object2, 然后看 object3 在哪。这才是真正的 spatial intelligence。

---

## 数据怎么搞的?

两个数据源:
- **ScanNet**: 室内 3D 扫描, 有精确 3D 坐标, 用于空间方位计算
- **MS-CoCo**: 普通图片带人物 keypoints, 用于 gaze 方向估计

关键 trick 是用了一个叫 **Orient-Anything** 的 pretrained 模型, 从单张图估计物体的朝向角度。然后把人物切成 head 和 body 两块, 分别估角度, 算两者的角度差(Δ), 根据 Δ 落在哪个区间映射到 "front / front-right / right..." 这种离散方向。

角度差的公式 `Δ ← (az_head - az_body + 540) mod 360 − 180` 看着唬人, 其实就是 normalize 到 [-180°, 180°] 对称区间, 方便分段。

---

## 训练方案

作者拿了 Qwen2.5-VL-3B 当 base, freeze 视觉 encoder, 只 fine-tune language model 部分。用上面 pipeline 自动生成 43K 训练样本, 只跑了 **8.5 GPU hours**。

结果: 35.85% → **82.09%**, 涨了 46 个点。

---

## 几个反直觉的发现

1. **Person 视角平均分(35.7%)居然比 Camera 视角(33.2%)高一点**。理论上 egocentric 应该更简单, 但实际 model 在 allocentric 上表现反而略好。作者猜是因为 web 上 third-person 构图占多数, model 隐式学到了 allocentric bias。这是 data artifact, 不是 capability evidence。

2. **Task 和视角之间有诡异的交互**。Camera 视角下 orientation task 比 direction task 差, 但 Person 视角下反过来。说明 model 根本没有统一的 3D representation, 它把每个 (task, perspective) 组合当独立问题处理。

3. **GPT-4o 在同一个回答里会视角混乱**。一会儿用 camera frame, 一会儿用 person frame, 自相矛盾。这说明它内部根本没有 coherent 3D world model, 全是 surface correlation。

---

## 怎么验证 model 不是在作弊?

作者做了两个 ablation:

1. **Multiple choice vs Direct answer**: 把选择题改成直接生成方向词, 准确率从 82.09% 掉到 79.34%, 只掉 2.75 点。说明 model 真学到了 spatial reasoning, 不是靠排除选项作弊。

2. **换不同 base model**: Qwen2.5-VL-3B 涨 46 点, Qwen2.5-VL-7B 涨 46 点, InternVL2.5-2B 涨 41 点。跨架构一致提升, 说明这套方法不依赖某个 model 的 quirk。

---

## 迁移到别的 task 表现如何?

在 VSI-Bench(另一个 spatial benchmark)上:
- Object Relative Direction: +0.93(小涨)
- **Route Planning: +9.54**(大涨, 但 MVSM 从没训过 route planning!)

Route Planning 的提升是最强的 signal——说明 perspective-aware training 让 model 学到了 generalizable 的 3D spatial representation, 而不是 memorize 某个 task。

---

## 局限性

1. **Person-perspective 的 relative direction task 自动化失败**, 只能手动标 864 个, 训练数据只有 1K(其他 task 都是 8K-13K)
2. **ScanNet 只有室内**, 迁移到 outdoor 掉点严重(室内 +23, 室外只 +9)
3. **只测静态场景**, 物体不动相机不动。真实机器人场景需要 temporal reasoning
4. **假设 gaze = head orientation**, 但人可以只动眼珠不动头, 这个简化有 systematic bias

---

## 我(作为 Andrej)的直觉

这篇 paper 印证了一个我一直信的 thesis: **VLM 的 spatial reasoning 缺陷主要是 data 问题, 不是 architecture 问题**。

证据很硬: 43K 数据 + 8.5 GPU hours 就把 3B model 从 35% 拉到 82%。pretraining 阶段根本没给 model 暴露 perspective signal, fine-tuning 一注入, 能力就解锁了。这说明 perspective-taking 是 sample-efficient learnable 的能力, 只是 pretraining data 没覆盖到。

真正的 breakthrough 不会停在 static image + template QA。下一步显然是让 VLM 在 3D simulator 里实际移动、从多视角自己采集 spatial label, 像 ImagineNav 那种思路。或者更激进——在 LLM 里直接塞 3D scene token, 让它在 representation 层面就有 3D 结构, 而不是从 2D patch 硬 infer。

这篇 paper 给了我们一个干净的 benchmark 和一个 v1 解。v2/v3 才是真正 exciting 的地方。

---

# ViewSpatial-Bench 详解: VLM 的 Multi-perspective Spatial Localization 评估

Andrej, 这篇paper直击 VLM 一个被我长期关注的核心 blind spot: **perspective-taking**能力, 也就是从他人/他物的 reference frame 进行空间推理。我build up一套直觉给你。

---

## 1. 核心问题: Egocentric vs. Allocentric 的根本缺陷

当前 VLM(GPT-4o、Gemini、Qwen2.5-VL、InternVL 等)在 web-scale image-text pairs 上预训练, 空间标注稀疏且几乎全部 anchored 在 camera-centric frame。paper的核心 hypothesis 是: **VLM 的 cross-viewpoint 缺陷来自 training data 的 structural limitation**, 而不是 architecture 本身的 ceiling。

认知科学里这个能力叫 "perspective-taking", 在 embodied AI(机器人导航、human-robot interaction、multi-agent collaboration)中是 critical capability。例如人对机器人说 "把右边的杯子递给我", 机器人必须从 human frame 解析 "右边", 而不是从 camera frame。

paper 量化了这个 gap:
- GPT-4o overall: **34.98%**(random baseline 26.33%)
- Gemini-2.0-Flash: **32.56%**
- 几乎只比 random 高 8 个点, 说明 proprietary SOTA model 在 perspective transformation 上近乎失败。

---

## 2. ViewSpatial-Bench 任务体系: 5 个 task 的设计直觉

paper 将空间定位拆解成 **2 个 perspective × 5 个 task** 的笛卡尔结构, 这是关键的 conceptual contribution:

### Camera Perspective(egocentric, 相对相机)
1. **Cam-Rel. Dir.**(Object Relative Direction): 判断 object1 相对于 object2 的方位(直接从 image 像素读)
2. **Cam-Obj. Ori.**(Object View Orientation): 以 camera 为 front, 判断图中人物的 gaze direction

### Person/Human Perspective(allocentric, 视角转换)
3. **Per-Rel. Dir.**: 假设你是 person A, 判断 person B/object 在你的哪个方向
4. **Per-Obj. Ori.**: 假设你处于 person 的位置, 判断 TA 朝哪个方向看
5. **Per-Sce. Sim.**(Scene Simulation): 在连续 frames 中, 假设你站在 object1 朝 object2, object3 在哪——这本质上要求 model 在 mental space 中做 viewpoint transform

**我的直觉**: 5 个 task 的设计精妙之处在于把"视角转换"这个 latent ability 拆成可测的 explicit dimensions。Per-Sce. Sim. 尤其关键, 它逼 model 做一个 "imagine yourself at position X facing Y" 的 3D mental rotation, 这才是真正的 spatial intelligence。

---

## 3. Dataset Construction Pipeline: 自动化 3D 标注的核心

### 3.1 双数据源设计

| Source | 用于 task | 优势 |
|--------|-----------|------|
| ScanNet | Cam-Rel. Dir., Per-Sce. Sim. | 精确 3D voxel + camera parameter |
| MS-CoCo | Cam-Obj. Ori., Per-Obj. Ori., Per-Rel. Dir. | 多样人物 + keypoints |

### 3.2 Maximum Coverage Sampling(Algorithm 1)

ScanNet 视频帧冗余极高, paper 用一个 set cover 变体来选帧:

```
Require: F = {f_1, ..., f_n}, voxel sets V_k for each frame, budget K
Ensure: S ⊆ F maximizing voxel coverage
1: S ← ∅
2: U ← ∅  (已覆盖 voxel 集合)
3: while |S| < K:
4:     f* ← argmax_{f_k ∈ F\S} |V_k \ U|   ← 贪心选择边际覆盖最大的帧
5:     S ← S ∪ {f*}
6:     U ← U ∪ V_{f*}
7:     if Stop condition: break
8: return S
```

**直觉**: 这是经典 greedy submodular maximization, |V_k \ U| 是 marginal gain, set cover 问题的 greedy 近似有 (1-1/e) 近似比。这样保证 minimal frames 覆盖 maximal spatial information, 防 redundant capture。

### 3.3 Head-Body Orientation Offset(Algorithm 2)—最关键的技术细节

MS-CoCo 没有 3D 标注, 只能用 2D keypoints + Orient-Anything-Large 模型估计 orientation。算法逻辑:

```
Require: Image I, keypoints K, bbox B, Orient-Anything D
1: P ← Crop(I, B)
2: (L_x, L_y), (R_x, R_y) ← ExtractShoulders(K)   ← 左右肩 keypoints
3: if Visibility(L_y)=0 OR Visibility(R_y)=0: return False  ← 肩膀不可见则放弃
6: H ← min(L_y, R_y)   ← 取两肩较高点(y 较小)作为 head/body 分界
7: P_head ← P[0:H, :], P_body ← P[H:, :]   ← 上下切分
8: (az_head, conf_head) ← D(P_head)
9: (az_body, conf_body) ← D(P_body)
10: Δ ← (az_head - az_body + 540) mod 360 − 180
11: return direction based on Δ thresholds
```

**公式变量解析**:
- `az_head`: 头部 azimuth angle(方位角, 0-360°), 由 Orient-Anything-Large 输出
- `az_body`: 身体 azimuth angle
- `Δ`: head 相对 body 的相对旋转角
- `+540`: 等价于 `+180 + 360`, 先平移到正区间
- `mod 360`: 折叠到 [0, 360)
- `−180`: 最终映射到 [-180°, 180] 对称区间

**直觉**: 这个公式的目的是把任意 head-body 角差 normalize 到 [-180°, 180°], 然后用阈值分段映射到 8 个方向(left, front-left, front, front-right, right, back-right, back, back-left)。阈值是 45° 分段(或 22.5° 偏移以让正方向居中)。

Orient-Anything-Large 是个 pretrained 3D orientation estimator([arXiv:2412.18605](https://arxiv.org/abs/2412.18605)), 从 single image 估计 object 的 6-DOF pose 中 azimuth 分量。

### 3.4 角度 → 方向的离散化映射

paper 用规则化映射把连续角离散成 18 个方向类别(比 What'sUP 的 12 类、VSI-Bench 的 8 类更细):

| 角度范围(°) | 方向标签 |
|------------|---------|
| [337.5, 22.5) | front |
| [22.5, 67.5) | front-right |
| [67.5, 112.5) | right |
| [112.5, 157.5) | back-right |
| ... | ... |

### 3.5 Per-Sce. Sim. 的 3D 几何计算

这是最能体现 spatial reasoning 的 task。给定三个 object 的 3D 坐标, 要求 model 想象站在 o1 朝 o2, 判断 o3 方位:

```
Metadata: bookshelf(1.2, 0.5, 0), window(1.2, 3.5, 0), sofa(3.2, 1.5, 0)
1. v_front = o2 - o1 = (0, 3.0, 0)  ← 定义"前方"方向
2. v_target = o3 - o1 = (2.0, 1.0, 0)
3. angle = atan2(cross(v_front, v_target), dot(v_front, v_target))
        = atan2(0*1.0 - 3.0*2.0, 0*2.0 + 3.0*1.0) 
        = atan2(-6, 3) ≈ -63.43° (逆时针) → 顺时针 63.43°
4. 映射到 "front-right"
```

**变量直觉**:
- `v_front`: viewer 朝向的单位向量(从 o1 指向 o2)
- `v_target`: 从 o1 到 o3 的向量
- `cross(v_front, v_target)`: 2D cross product, z 分量决定顺/逆时针
- `dot`: 投影长度, 决定 angle 大小
- `atan2(cross, dot)`: robust atan, 处理象限

这个计算本质上是把全局坐标转换到 viewer 的 local frame(front 为 +y, right 为 +x), 然后用 atan2 求 polar angle。

### 3.6 Distractor 设计的妙处

paper 专门设计了一个 anti-aliasing distractor rule:
- 对单一方向(如 "front"), 干扰项**排除**含 "front" 的复合方向("front-left" 不选)
- 对复合方向(如 "front-left"), 干扰项**排除**其 constituent 单一方向("front" 或 "left" 不选)

**直觉**: 这是控制 question 难度的关键, 防 model 通过 partial keyword match 作弊。这把 benchmark 从"靠模糊语义匹配"逼回到"必须精确角度判断"。

---

## 4. MVSM(Multi-View Spatial Model)训练策略

### 4.1 训练数据生成

paper 用同样的 pipeline 从 ScanNet + MS-CoCo training set 自动生成 **43K samples**(test 集 5,712), 加上 Spatial-MM 的 Per-Rel. Dir. 数据(因为这个 task 全自动不准, 用 external dataset 补):

| Task | Train samples |
|------|---------------|
| Cam-Rel. Dir. | 13,644 |
| Cam-Obj. Ori. | 8,954 |
| Per-Obj. Ori. | 8,954 |
| Per-Rel. Dir. | 1,014 |
| Per-Sce. Sim. | 10,309 |
| **Total** | **42,875** |

注意 Per-Rel. Dir. 仅 1K 样本, 因为人物坐标 + 环境 context 复杂, 自动化失败率高(864 实例是 manual annotated)。

### 4.2 训练配置

- Base model: **Qwen2.5-VL-3B**([arXiv:2502.13923](https://arxiv.org/abs/2502.13923))
- **Freeze vision encoder + multi-modal projector**, 只训 language model 部分
- 3 epochs, effective batch size 16(gradient accumulation 4 × per-device batch 1 × 4 GPUs)
- 4 × NVIDIA A100 40GB
- **~8.5 GPU hours**——计算成本极低, 表明 spatial fine-tuning 是 sample-efficient 的

**架构直觉**: 只 fine-tune LLM 部分意味着 paper 假设视觉 perception 已经够用, 缺的是 reasoning 的"视角意识"。但 paper 没做 ablation 来验证这个假设是否最优——也许 unfreeze vision encoder 会更好。这是个 future work 信号。

### 4.3 Multi-Perspective Fine-Tuning 的核心思想

paper 没引入新 loss 或新 module, 而是用统一的 natural language template 把 5 个 task 的 spatial 关系表达出来, 让 model 在数据分布层面学到"视角"这个 latent dimension。QA 模板见 paper Table 4, 例如 Per-Sce. Sim. 用:

> "If you stand at {object1} facing {object2}, where is {object3}?"

这种 template-based instruction tuning 让 model 学到 perspective frame 是输入的一个 explicit variable, 而非固定的隐含假设。

---

## 5. 实验结果深度解析

### 5.1 主结果(Table 2)

让我重排关键数据:

| Model | Cam-Avg | Person-Avg | Overall | Δ(Person-Cam) |
|-------|---------|------------|---------|----------------|
| Random | 25.50 | 27.12 | 26.33 | +1.62 |
| GPT-4o | 33.57 | 36.29 | 34.98 | +2.72 |
| Gemini-2.0-Flash | 33.66 | 31.53 | 32.56 | -2.13 |
| InternVL2.5-8B | 46.48 | 40.20 | 43.24 | -6.28 |
| Qwen2.5-VL-7B | 40.56 | 33.37 | 36.85 | -7.19 |
| Kimi-VL-16B | 25.14 | 41.52 | 33.58 | +16.38 |
| **MVSM** | **85.05** | **79.31** | **82.09** | -5.74 |

**三个反直觉观察**:

1. **Person perspective 平均(35.7%) > Camera perspective(33.2%)**: 这违反直觉——理论上 egocentric 应该更简单。paper 解释为 web 训练数据中 third-person composition 占多数, model 隐式学到了 allocentric bias。这是 data distribution artifact 而非 model capability evidence。

2. **Task × Perspective 交互效应**: camera-perspective 下 Obj.Ori.(19-41%) < Rel.Dir.(25-54%); 而 person-perspective 下相反(Obj.Ori. 22-63% > Rel.Dir. 31-43%)。这表明 model 没有 unified 3D representation, 把每个 task-perspective 组合当独立问题。

3. **Kimi-VL 的极端 case**: Person(41.52) 远超 Camera(25.14), gap 16.38 个点。这是 data bias 的极致案例——可能 Kimi 训练数据中 first/third-person spatial question 比例失衡。

### 5.2 MVSM 提升分析

- Overall: 35.85% → 82.09%, **+46.24%**
- Cam-Obj. Ori.: 33.33% → 87.65%, **+54.32%**(最大)
- Per-Obj. Ori.: 39.16% → 90.16%, **+51.00%**
- Per-Sce. Sim.: 28.51% → 75.75%, **+47.24%**

**直觉**: orientation task 提升最大, 说明 perspective transformation 在 fine-tuning 后被显著激活。Sce. Sim.(mental simulation)也能涨 47 点说明 model 确实学到了可 transfer 的 3D spatial representation, 而非 memorize。

### 5.3 Anti-shortcut 验证(Table 6)

paper 怕 model 学到 multiple-choice 的 elimination shortcut, 做 ablation:
- Multiple Choice(MC) format: 82.09%
- Direct Answer(DA) format: 79.34%
- 差 2.75 个点

**直觉**: gap 这么小说明 model 不是靠选项结构作弊, 而是真学到了 spatial reasoning。但 2.75 点的 drop 也提示 MC 格式确实带来 marginal 便利, 这在 VLM benchmark 中是 universal phenomenon([What'sUP paper](https://arxiv.org/abs/2310.19785) 也讨论过)。

### 5.4 Multi-backbone 泛化(Table 6)

- Qwen2.5-VL-3B: +46.24%
- Qwen2.5-VL-7B: +46.16%
- InternVL2.5-2B: +41.47%

**直觉**: 提升跨架构一致, 说明 perspective-aware training 是 model-agnostic 的能力注入, 而非依赖某个 architecture quirk。

### 5.5 迁移到 VSI-Bench 和 VSI-App(Table 3)

| Model | VSI-Bench Rel.Dir | VSI-Bench Route Plan | VSI-App Indoor | VSI-App Outdoor |
|-------|-------------------|----------------------|----------------|----------------|
| GPT-4o | 41.30 | 31.50 | 34.00 | 27.00 |
| Qwen2.5-VL-3B | 46.00 | 21.90 | 18.00 | 27.00 |
| MVSM | 46.93(+0.93) | 31.44(+9.54) | 41.00(+23.00) | 36.00(+9.00) |

**关键观察**:
- **Route Planning +9.54**: MVSM 没显式训过 route planning, 但提升显著, 说明 perspective-aware training 间接帮助了 trajectory reasoning——这是 spatial representation generalization 的强证据。
- **VSI-App Indoor +23**: 比 Outdoor(+9) 强很多, 说明 paper 训练数据 indoor-dominant(ScanNet 是 indoor-only), 存在 indoor→outdoor domain gap。paper 在 Limitations 也承认这点。

---

## 6. 失败模式: Perspective Confusion

paper Figure 4 展示 GPT-4o 的典型失败: 在 single response 中 **alternating between human and camera perspective**。例如回答 "where is the cup from my perspective" 时, 一会儿用 camera frame, 一会儿用 person frame, 内部不一致。

**我的直觉**: 这说明 GPT-4o 没有 coherent internal 3D world model, 而是基于 surface correlation 做 pattern matching。MVSM 通过显式 perspective-label 训练, 学到了 "perspective 是一个需要 maintain consistency 的 latent variable"。

这呼应了 [VSI-Bench / Thinking in Space](https://arxiv.org/abs/2412.14171) 的发现: top model 在 spatial recall 上失败, 因为缺乏 persistent 3D scene representation。

---

## 7. 与现有 benchmark 的对比(Table 1)

ViewSpatial-Bench 是唯一一个同时覆盖:
- 18 directions(最多, 其他最多 12)
- 3D coordinates from ScanNet
- Camera + Person 双视角
- Person-target + Object-target 双查询目标

对比:
- [SpatialRGPT-Bench](https://arxiv.org/abs/2406.01584): 自动标注, 1,410 samples, 6 directions, 但无 person perspective
- [What'sUP](https://arxiv.org/abs/2310.19785): 820 manual samples, 12 directions, 无 3D
- [VSI-Bench](https://arxiv.org/abs/2412.14171): 3,672 multi-frame, 但 person perspective 评估有限
- [3DSRBench](https://arxiv.org/abs/2412.07825): 2,772 samples, 1,827 scenes, 但 single perspective
- [SPHERE](https://arxiv.org/abs/2412.12693): 2,285 samples, 7 directions, 部分 perspective

---

## 8. 代码 / 数据资源

- GitHub: [https://github.com/ZJU-REAL/ViewSpatial-Bench](https://github.com/ZJU-REAL/ViewSpatial-Bench)
- Project page: [https://zju-real.github.io/ViewSpatial-Page/](https://zju-real.github.io/ViewSpatial-Page/)
- Backbone: [Qwen2.5-VL](https://arxiv.org/abs/2502.13923)
- Orientation estimator: [Orient-Anything](https://arxiv.org/abs/2412.18605)
- 3D scene source: [ScanNet](https://arxiv.org/abs/1702.04405)
- Person source: [MS-CoCo](https://arxiv.org/abs/1405.0312)
- Related perspective work: [VSI-Bench / Thinking in Space](https://arxiv.org/abs/2412.14171), [What'sUP](https://arxiv.org/abs/2310.19785), [SpatialRGPT](https://arxiv.org/abs/2406.01584), [3DSRBench](https://arxiv.org/abs/2412.07825), [SPHERE](https://arxiv.org/abs/2412.12693), [EmbSpatial-Bench](https://arxiv.org/abs/2406.05756)

---

## 9. 我的整体直觉与批判

### 9.1 Strengths
1. **Task design 精妙**: 5-task 的 2×N 结构把"视角转换"这个 latent ability 拆解成可量化 dimensions, Per-Sce. Sim. 尤其是真正的 mental rotation test。
2. **自动化 pipeline 实用**: Maximum Coverage Sampling + Orient-Anything + 角度离散化是可扩展的 recipe, 43K training data 自动生成是亮点。
3. **Anti-shortcut 验证**: MC vs DA ablation 排除选择题作弊, multi-backbone 排除 architecture artifact, 实验设计严谨。
4. **VSI-Bench Route Planning 的迁移收益**: 没 explicit train 但涨 9.5 点, 是 spatial representation generalization 的强 evidence。

### 9.2 Weaknesses / Open Questions
1. **Per-Rel. Dir. 全自动失败**: 仅 1,014 training samples(其他 task 8K-13K), 且 864 manual labeled。这表明 person-perspective 在 cluttered scene 中自动化是 unsolved problem, MVSM 在这个 task 上(+42.52%)也可能 overfit 到 Spatial-MM 的 narrow distribution。
2. **Vision encoder frozen**: paper 没探索 unfreeze vision encoder 是否更好。如果 spatial 信息确实需要 vision-side adaptation(比如 depth 估计、perspective cue), freeze 可能 sub-optimal。
3. **Indoor-only ScanNet**: VSI-App outdoor 仅 +9 vs indoor +23, 暴露 domain gap。要解决 embodied AI(尤其是 outdoor robot), 需要 outdoor 3D 数据(如 [Matterport3D](https://arxiv.org/abs/1903.01906), [KITTI](https://arxiv.org/abs/1707.00463), [Waymo](https://arxiv.org/abs/1912.04838))。
4. **静态场景 only**: paper Limitations 承认没测 dynamic spatial reasoning(物体移动、相机移动)。Temporal extension 是显然的下一步, [STI-Bench](https://arxiv.org/abs/2503.23765) 在做这个方向。
5. **Gaze direction 假设**: MS-CoCo 标注假设 gaze direction = head orientation, 然后手动 filter 偏差大的 case。但实际 human gaze 可以独立于 head pose(eyes-only movement), 这个简化可能引入 systematic bias。
6. **18 directions 离散化损失**: 把连续角离散成 18 bins, 每个 bin 20° 宽, 在 boundary 附近的 case(如 65°, 在 front-right[22.5-67.5)边界)易引入 label noise。

### 9.3 对你(Andrej)的 research intuition

这篇 paper 印证了你的一个 long-standing thesis: **当前 VLM 的 spatial reasoning 不是 perception bottleneck, 而是 representation / training data bottleneck**。MVSM 只用 43K 数据 + 8.5 GPU hours, 就把 3B model 从 35% 拉到 82%, 说明 perspective-taking 能力是 sample-efficient learnable 的——只是 pretraining 数据没暴露这个 signal。

这与 [SpatialVLM(Boyuan Chen, 2024)](https://arxiv.org/abs/2401.12145) 的发现一致: VLM 空间能力的缺失是 data 问题, 显式注入 spatial 标注即可大幅提升。

下一步显然方向:
- **加入 temporal sequence**: 让 model 在 video 中 maintain consistent perspective across frames, [VSI-Bench](https://arxiv.org/abs/2412.14171) 已经在 push 这个。
- **加入 active inference**: 在 embodied sim 中让 model 实际移动相机, 从多视角采集 self-supervised spatial label, 类似 [ImagineNav](https://arxiv.org/abs/2410.09874)。
- **Vision encoder fine-tuning**: 探索 spatial-aware visual representation learning, 可能用 [Depth Anything V2](https://arxiv.org/abs/2406.09414) 或 metric depth prior 注入。
- **LLM-side 3D scene token**: 借鉴 [3D-LLM](https://arxiv.org/abs/2307.12981) 的做法, 在 LLM 内部引入 3D-aware token, 而非依赖 2D patch。

paper 给我们一个干净的 baseline 和 benchmark, MVSM 是 v1 解, 真正的 breakthrough 在 v2/v3: when we let VLM actually **move** in 3D space and learn perspective from embodied interaction, 而非从 static image caption 学。
