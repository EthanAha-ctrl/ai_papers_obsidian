---
source_pdf: Instruct2Act.pdf
paper_sha256: 7b121cff4987e35ada692ab5b6988f31dd00e31a534d61d107cdfe4b6e073900
processed_at: '2026-08-05T09:57:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Instruct2Act 的人话版

## 一句话总结

让 ChatGPT 当机器人的"大脑"，写 Python 代码去调用 SAM、CLIP 这些"眼睛"，最后把"手"动起来。整个 pipeline 是 zero-shot 的，不需要训练任何 robot policy。

GitHub: https://github.com/OpenGVLab/Instruct2Act
论文: https://arxiv.org/abs/2305.05498

---

## 1. 这篇 paper 到底想解决什么问题

机器人领域一直有个老大难问题：**用户给一句话指令，机器人怎么把它变成动作？**

历史上大致两条路线：

**路线 A（End-to-end learning）**：拿一个 transformer，把图像 + 语言塞进去，让它直接吐出 action。VIMA、Gato、Flamingo、RT-1/2 都是这路。缺点很明显：
- 要海量 demonstration 数据
- 换个没见过的物体就废了（distribution shift）
- 任务一长，action sequence 累积误差爆掉

**路线 B（LLM as planner）**：让 LLM 当高层大脑，输出 skill sequence 或 Python 代码，底层 skill 由预训练 controller 执行。SayCan、CaP、ProgPrompt 都是这路。缺点：
- Perception 通常还是 hand-crafted，或者只是个简单的 object list
- 遇到新物体就抓瞎
- 没有真正的 visual grounding

Instruct2Act 想做的事情很简单：**把路线 B 的 LLM planning 能力，和路线 A 用的那些 visual foundation models（SAM、CLIP）的 perception 能力，拼到一起**。LLM 不再是单纯写代码调用硬编码 API，而是写代码调用 SAM、CLIP 去做真正的 open-vocabulary perception。

Intuition 上就一句话：**LLM 当 orchestrator，foundation models 当 perception services，Python interpreter 当执行引擎**。这跟现在 ChatGPT 的 tool-use 思路完全一样，只不过把 "tools" 从 web search 换成了 SAM/CLIP/robot primitives。

---

## 2. Pipeline 长什么样

```
Instruction (text / image / 点屏幕)
         │
         ▼
┌─────────────────────────────┐
│  LLM (ChatGPT / davinci-003) │
│  输入：API 文档 + 5 个 example  │
│  输出：main() Python 代码字符串 │
└─────────────────────────────┘
         │  生成代码
         ▼
┌─────────────────────────────┐
│  Python Interpreter          │
│  执行 main()：               │
│   1. GetObsImage() 拿图       │
│   2. SAM(image) 切 mask      │
│   3. ImageCrop 裁 crop       │
│   4. CLIPRetrieval 选目标     │
│   5. Pixel2Loc 像素→坐标     │
│   6. PickPlace 生成 action   │
│   7. RobotExecution 执行     │
└─────────────────────────────┘
         │
         ▼
   PyBullet / 真实机器人
```

关键洞察：**LLM 输出的不是 action token，而是 Python 代码字符串**。这段代码被 Python interpreter 执行时，每行 API call 都对应一次 foundation model 推理或 robot control。

为什么这重要？
- Code 是 **declarative plan**，每一步可以独立 debug
- API 边界是天然的 error isolation
- Interpreter 是免费的 verifier（语法错就重试）
- Compositional generalization 来自 LLM 本身，不依赖 robot training data

---

## 3. API 的层级组织——这是个 design win

paper 里 API 不是平铺一坨，而是按 robot system 的层级分组：

```
Level 1: File IO          → GetObsImage(), SaveFailureImage()
Level 2: Core Modules
  ├─ Perception           → SAM(), CLIPRetrieval(), ImageCrop(), GetObjMatch()
  └─ Action               → Pixel2Loc(), PickPlace(), DistractorActions(), RearrangeActions()
Level 3: Robotic Hardware → RobotExecution(), SpeedSet()
```

这个 hierarchy **同时反映在 prompt 的注释里**，比如 prompt 里会写 `# Second Level: Core Modules` 然后再列 perception 和 action 的 API。

为什么这么做？给 LLM 一份"robot OS 的系统调用表"比给它一坨平铺函数更容易让它理解每个 API 的职责边界。这跟 ViperGPT 只列 function-level example 的做法不一样。

直觉上：**结构化的 prompt 让 LLM 知道"这个 API 属于哪一层，该在什么时候用"**，而不只是"这个 API 叫什么名字、参数是什么"。

---

## 4. Perception pipeline 的细节

这部分是 paper 里技术含量最高的。SAM 是 zero-shot segmentation，但直接拿来用效果很差。paper 加了一堆 processing module 把 SAM 的输出"洗"干净。

### 4.1 Image pre-processing（处理 shadow）

机器人桌面操作相机是 top-down 的，机械臂会投很大阴影到桌面。SAM 经常把阴影当成 object 边界切出来，整个 segmentation 就废了。

做法：
1. **Gray threshold filter**：把 RGB image 转 grayscale，按阈值 $\tau$ 二值化。
   $$
   I_{\text{gray}}(u,v) = \begin{cases} 255 & \text{if } 0.299R + 0.587G + 0.114B > \tau \\ 0 & \text{otherwise} \end{cases}
   $$
   系数 $0.299, 0.587, 0.114$ 是标准 NTSC grayscale 公式。$(u,v)$ 是 pixel 坐标，$R,G,B$ 是三通道值。$\tau$ 是预设定阈值。

2. **Morphological closing**：填掉二值化产生的小洞。
   $$
   A \bullet B = (A \oplus B) \ominus B
   $$
   - $A$：binary mask
   - $B$：structuring element（小方块）
   - $\oplus$：dilation（膨胀，把 mask 边界向外推）
   - $\ominus$：erosion（腐蚀，把 mask 边界向内收）
   - Closing = 先膨胀再腐蚀，能把小洞和窄 gap 填上但保持外形不变

为什么是 closing 不是 opening？因为 grayscale threshold 会留下细碎 gap，需要填，不是去颗粒。

### 4.2 SAM segmentation

$\{m_i\}_{i=1}^{N} = \text{SAM}(I_{\text{pre}})$

- $I_{\text{pre}}$：pre-processing 后的 image
- $m_i \in \{0,1\}^{H \times W}$：第 $i$ 个 binary mask
- $N$：候选 mask 数量

### 4.3 Mask post-processing（处理 SAM 的脏输出）

SAM 直接输出会有几个问题：
- 一个 object 切成多个碎片 mask
- mask 内部有空洞
- 桌面纹理被误识为 object

paper 的处理 pipeline：

**A. Size filtering**：去掉过大或过小的 mask。
$$
\text{keep } m_i \text{ iff } \tau_{\min} \leq |m_i| \leq \tau_{\max}
$$
$|m_i|$ 是 mask 面积（pixel 数），$\tau_{\min}, \tau_{\max}$ 是预设阈值。

**B. Morphological opening**：去掉小碎片、断开粘连。
$$
A \circ B = (A \ominus B) \oplus B
$$
- Opening = 先腐蚀再膨胀，跟 closing 反过来
- 效果：小颗粒去掉，粘连断开，大块边界保持

**C. NMS (Non-Maximum Suppression)**：多个重叠 mask 去重。
保留一个 mask，suppress 掉 IoU 大于阈值 $\theta_{\text{NMS}}$ 的其他 mask。
$$
\text{IoU}(m_i, m_j) = \frac{|m_i \cap m_j|}{|m_i \cup m_j|}
$$
SAM 输出没有 score，paper 用 area 作为 surrogate。

### 4.4 Crop + CLIP retrieval

每个 mask 算 bounding box，从原图裁出 crop $I_i$。

送 CLIP image encoder：
$$
F_{I_i} = \text{CLIP}_{\text{img}}(I_i) \in \mathbb{R}^d, \quad d=1024 \text{ (ViT-H)}
$$

Query 这边分两种情况：

**纯文本 query**：
$$
F_T = \text{CLIP}_{\text{txt}}(\text{query}) \in \mathbb{R}^d
$$

**图像 query**（多模态 instruction，比如 `{dragged_obj}` 是个 image crop）：
$$
F_I = \text{CLIP}_{\text{img}}(I_{\text{query}})
$$

Retrieval 公式：
$$
i^* = \arg\max_{i \in \{1,...,N\}} \frac{F_Q^\top F_{I_i}}{\|F_Q\|_2 \cdot \|F_{I_i}\|_2}
$$
- $F_Q$：query embedding，是 $F_T$ 或 $F_I$
- $F_{I_i}$：第 $i$ 个 candidate 的 CLIP image embedding
- $N$：候选 mask 数量
- $\top$：向量内积
- $\|\cdot\|_2$：L2 范数

本质就是把 open-vocabulary detection 拆成 segmentation + classification 两步：SAM 负责"哪里有东西"，CLIP 负责"这是不是我要的"。

### 4.5 Pixel 到 robot 坐标

$\begin{bmatrix} x_{\text{robot}} \\ y_{\text{robot}} \end{bmatrix} = T \begin{bmatrix} u_{\text{pixel}} \\ v_{\text{pixel}} \end{bmatrix} + b$

- $T \in \mathbb{R}^{2 \times 2}$：预标定的 affine 变换矩阵（scale + rotation）
- $b \in \mathbb{R}^2$：translation offset
- $(u_{\text{pixel}}, v_{\text{pixel}})$：mask centroid 的 pixel 坐标
- $(x_{\text{robot}}, y_{\text{robot}})$：robot base frame 坐标

然后 clamp 到 workspace bounds 防止机械臂撞边界。

### 4.6 Action

`PickPlace(pick=X_p, place=X_q, bounds=B, yaw_angle_degree=θ, tool="suction")`

- $X_p, X_q \in \mathbb{R}^2$：pick 和 place 的 robot 坐标
- $B$：workspace boundary
- $\theta$：旋转角度（degree）
- `tool`：suction cup 或 gripper

---

## 5. 三种 instruction modality 怎么统一处理

这是 paper 里另一个有意思的设计点。Instruct2Act 要支持三种输入：

### 5.1 Pure-Language
```
"Put the green and purple polka dot block into the green container"
```
LLM 解析出 "the green and purple polka dot block" 作为 query，走 CLIP text encoder。

### 5.2 Language-Visual（VIMA 的 multi-modal prompt 格式）
```
"Put {dragged_obj} into {base_obj}"
```
`{dragged_obj}` 是占位符，对应一张 object crop image，存在 environment cache `C` 里。LLM 生成的代码写 `query=templates['dragged_obj']`，运行时从 cache 取 image，走 CLIP image encoder。

### 5.3 Pointing-Language（人用鼠标点）
当物体描述太抽象、给 crop 又不实际时，用户直接点屏幕。点击坐标 $(x,y)$ 作为 SAM 的 point prompt：
$$
\{m_i\} = \text{SAM}(I, \text{point}=(x,y), \text{label}=1)
$$
- `label=1`：foreground point
- SAM 会根据这个点分割出包含该点的 object

这种模式下，人帮忙做了 grounding 的一部分。

### 5.4 Scene-level matching（Rearrange 任务）
任务："Rearrange to this {scene}"——把当前 arrangement 重排成 goal scene 的样子。

做法：
- 对 obs image 和 goal scene image 分别跑 SAM + CLIP，得到两组 features $\{F_{I_i}^{\text{obs}}\}$ 和 $\{F_{I_j}^{\text{goal}}\}$
- Cost matrix：$C_{ij} = -\cos(F_{I_i}^{\text{obs}}, F_{I_j}^{\text{goal}})$
- 用 Hungarian algorithm 求最优匹配：
$$
\sigma^* = \arg\min_{\sigma \in S_n} \sum_{i=1}^{n} C_{i, \sigma(i)}
$$
- $\sigma$：一种 permutation，把 obs object 映射到 goal object
- $S_n$：所有 permutation 的集合
- $\sigma^*$：cost 最低的最优匹配

这是个 $O(n^3)$ 的 assignment problem。匹配完之后算每个 (obs, goal) pair 的位移，生成一系列 PickPlace action。如果有 distractor（goal scene 里没有的），先用 DistractorActions 移走。

---

## 6. Prompt 工程——这是 LLM robotics 的真正手艺活

完整 prompt 包含四部分：

### 6.1 Third-party library imports
```python
from PIL import Image
import numpy as np
import scipy
import torch
import cv2
import math
from typing import Union
```
告诉 LLM 代码里能用这些库做计算。后面会看到，这让它能做角度换算之类的活。

### 6.2 API definitions
按 hierarchy 写的，每个 API 有 docstring + signature + 一两个 usage example。

### 6.3 In-context examples（5 个完整的 main() 函数）

| Example | 任务类型 | 教的 pattern |
|---|---|---|
| 1 | Put X into Y | 单步 pick-place，纯文本 query |
| 2 | Rotate {dragged_obj} 150° | 多模态 + yaw 控制 |
| 3 | Rearrange to {scene} then restore | Scene matching + 反向操作 |
| 4 | Put stripe obj in {scene} into orange obj | Scene 作为 context + retrieval |
| 5 | Sequential pick into A then B, restore | Multi-step planning |

这 5 个 example 覆盖了 6 个 meta-task 的所有 pattern。

注意 Example 5 里函数名故意打成 `mian_5()`，是 prompt 的"自然噪声"。LLM 居然能学到正确 pattern，不抄这个 typo。

### 6.4 Task instruction 前加 "Think step by step"
这是 Kojima et al. 的 Zero-shot CoT 技术（https://arxiv.org/abs/2205.11916）的迁移，让 LLM 在写代码前先做 reasoning。

---

## 7. Ablation 揭示的有趣 insight

### 7.1 Processing module 的作用（Table 2）

| Image Pre | Mask Post | Avg SR |
|---|---|---|
| ✗ | ✗ | 51.0% |
| ✓ | ✗ | 50.0% |
| ✗ | ✓ | 83.0% |
| ✓ | ✓ | **84.1%** |

关键发现：
- **Mask post-processing 是大杀器**：从 51% 跳到 83%，提升 32 个百分点
- **Image pre-processing 单独用反而轻微下降**（50.0 vs 51.0），因为 pre-defined 阈值没调好，shadow 滤除可能误伤目标纹理
- 两者结合 ≈ mask post 单独用

Intuition：mask post-processing 解决的是"误报"问题——多余的小 mask 会让 CLIP retrieval 把相似的小区域误选为 target。这个影响巨大。Image pre-processing 解决的是"漏报"——shadow 把 object 切成碎片，但 zero-shot 阈值难调，单独用收益不明显。

**这个 ablation 给做 foundation model 部署的人一个重要启示：raw foundation model 直接拿来用效果远不如加了 processing module 的版本。**

### 7.2 Foundation model 规模的 scaling（Table 3, 4）

SAM backbone：

| Backbone | SR(%) | Params(M) |
|---|---|---|
| Base | 75.1 | 93.7 |
| Large | 82.7 | 312.3 |
| Huge | 84.1 | 641.1 |

CLIP backbone：

| Backbone | SR(%) | Params(M) |
|---|---|---|
| B-16 | 70.0 | 149.6 |
| L-14 | 76.5 | 427.6 |
| H-14 | 84.1 | 986.1 |

清晰的 scaling law：**foundation model 越大，zero-shot 性能越好**。

这正是 modular 架构的红利——你不用重训 robot policy，只要把 SAM-B 换成 SAM-H，性能就涨 9 个百分点。End-to-end trained policy 想达到同样提升可能要重新跑几个月训练。

### 7.3 Prompt 元素的 ablation（Appendix A.5）

paper 给了非常细致的 case study：

**Only API definition**：LLM 写出 main 框架，但会 hallucinate 不必要的 action（DistractorActions、RearrangeActions），漏掉 `return info`。**但它会主动调用 `SaveFailureImage()` 处理 failure case，这是 examples 里没教过的 emergent behavior**。

**Only in-context examples**：输出更结构化，但缺 semantic comment，把 SAM 错误展开为 "Semantic Affinity Module"。变量命名 lack semantic info。

**API + examples**：human-readable、semantic-rich、结构良好的 Python 代码。

结论：
- API definition 给 LLM **flexibility + reasoning ability**（能自由组合、处理没见过的情况）
- In-context examples 给 LLM **structure + expert pattern**（代码风格、调用顺序、变量命名）
- 两者互补，缺一不可

---

## 8. 主实验结果的 surprise

| Model | Task 01 | Task 02 | Task 03 | Task 04 | Task 05 | Task 17 | Avg |
|---|---|---|---|---|---|---|---|
| VIMA-20M (trained) | 100 | 100 | 100 | 99.5 | 59.5 | 47.5 | 84.4 |
| Ours-Multi (zero-shot) | 91.3 | 81.4 | 98.2 | 78.5 | 72.0 | **85.2** | **84.4** |

几个反直觉的点：

1. **平均打平**：zero-shot 方法跟 trained 20M 参数 model 平均持平，已经是巨大胜利。
2. **Long-horizon 任务完胜**：Task 17（Pick in order then restore）85.2% vs 47.5%，差 38 个百分点。Task 05（Rearrange then restore）72.0% vs 59.5%。
3. **简单 task 反而略低**：Task 01 VIMA 100%，Ours 91.3%。

为什么会出现这种 pattern？

**End-to-end policy 在长 horizon 上 error accumulation 严重**——预测 6 步 action sequence，每步错一点，最后全崩。而 Instruct2Act 把 planning 交给 LLM（symbolic level，无累积误差），perception 每步重新做（closed-loop），所以 horizon 越长优势越大。

**简单 task 上 end-to-end ceiling 更高**——因为 VIMA 在训练分布内见过这种 task，已经 overfit 到 100%。Instruct2Act 是 zero-shot，CLIP retrieval 总有 borderline case 失败。

### L2/L3 generalization（Figure 5）

- L2 combinatorial：训练时见过的材料/object 重新组合
- L3 novel object：训练时没见过的 object

Learning-based 方法在 L2/L3 上性能显著下降（distribution shift），Instruct2Act 几乎不下降，因为 SAM 的 segmentation 和 CLIP 的 open-vocabulary classification 本来就是 zero-shot 设计。

**Foundation model based approach 的 distribution shift 成本几乎为零**——这是最根本的 architectural 优势。

---

## 9. Robustness 实验——LLM 的语言理解红利

paper 测了三种 robustness：

**Human Intervention**：
- "Put X into Y. I cancel this task. Stop!" → LLM 不生成执行代码
- "That instruction is wrong, use this one. [new]" → LLM 用新指令

**Missing Characteristics**：
- 随机删形容词、拼错单词 → 仍能正确 ground
- 比如 "polka dot block" 写成 "poka dot" 也能识别

**Synonym Replacement**：
- "Rotate" → "Spin" → 仍工作

这些 zero-shot robustness 来自 LLM 本身的语言理解能力，不需要在 robot data 上训练。这是 end-to-end policy 难以做到的——它们对指令改动很敏感。

### 第三方库扩展能力

paper 给了个巧妙 demo：所有 in-context example 用 degree，但用户 instruction 用 "0.5 radians"。LLM 知道 numpy 可用，于是生成：
```python
angle = 0.5  # in radians
yaw_angle = angle * 180 / np.pi  # convert to degrees
```

**这说明 LLM 能利用 prompt 里 import 的库做单位换算、几何计算**。Code policy 的表达能力远超 sequence-of-skills 范式。

---

## 10. LLaMA-Adapter 也能跑（Table 5）

| Task 01 | Direct | More Trials | Naive Filtering |
|---|---|---|---|
| SR(%) | 72.5 | 77.5 | 85.5 |

- Direct：LLaMA-Adapter 直接生成，72.5%
- More Trials：Python interpreter 抛 exception 时重新 generate（最多 N 次）→ 77.5%
- Naive Filtering：检测生成代码里没用到 environment cache（`templates[...]`）就重新生成 → 85.5%

启示：**LLM code generation 不需要完美，interpreter + retry 是 cheap error correction**。这跟 CodeT、Self-Debug 思路一致。开源 LLM 也能跑这个框架，性能差距不像想象那么大。

---

## 11. Limitations（paper 自己承认 + 我观察到的）

paper 自己提的：
1. **Compute cost**：SAM-H + CLIP-H + LLM API，每个 action step 要几秒，对 real-time robot 系统负担大
2. **Action primitive 限制**：只有 PickPlace / Rotate / Rearrange 几种，复杂 manipulation（双手协作、deformable object）需要扩展 API
3. **只测了 simulation**：PyBullet + top-view camera，没 real-world 验证

我观察到的：
4. **Coordinate calibration**：`Pixel2Loc` 依赖预标定矩阵 $T$ 和 offset $b$，换 camera setup 就要重标
5. **CLIP 在 fine-grained attribute 上的局限**：Table 1 Task 02 上 Instruct2Act 81.4%，还有 18.6% 失败，很可能是 CLIP 把相似的 polka dot / checkerboard 搞混了
6. **Open-loop code execution**：生成的 `main()` 一跑到底，中间出错不会自动 re-plan。Inner Monologue 那种 closed-loop verbal feedback 没集成进来
7. **In-context example 覆盖问题**：5 个 example 不可能覆盖所有 task pattern，新 task 可能需要 prompt engineering
8. **SAM 对透明 / 反光物体失败**：这点 paper 没提，但实际 deployment 肯定会遇到

---

## 12. 跟相关工作的对比——直觉版

**vs CaP (Code as Policies)**：
CaP 是这个方向的祖师爷，但 perception API 是硬编码的，比如 `get_object_position("mug")` 直接返回位置，不能 zero-shot 处理新物体。Instruct2Act 把 perception 升级成 SAM+CLIP，是真正的 open-vocabulary。

**vs VIMA**：
VIMA 是 end-to-end multimodal transformer，要训练 20M 参数。Long-horizon 任务上崩盘（Task 17 只有 47.5%），因为 end-to-end prediction 累积误差。Instruct2Act 在 long-horizon 上完胜。

**vs SayCan**：
SayCan 用 LLM 生成 next skill + affordance model 评估 grounding，是 sequential planner。但缺乏 closed-loop perception，skill 执行依赖预训练 controller。Instruct2Act 每步重新感知，更鲁棒。

**vs ViperGPT / VisProg**：
ViperGPT 在 vision QA 上做 LLM 生成 code 调 foundation model，但没 action loop。Instruct2Act 借用了 in-context example 思路，但加了 robotic API hierarchy 和 closed-loop execution + error handling。

**vs PaLM-E**：
PaLM-E 是 monolithic multimodal model（540B PaLM + 22B ViT），训练成本巨大。Instruct2Act 是 modular 架构，foundation models 之间解耦，任何一个升级都能受益。

**vs DALL-E-Bot**：
DALL-E-Bot 用 Stable Diffusion 生成 goal scene image，再 image matching 引导 action。但没 LLM 高层 reasoning，只能处理"目标视觉状态匹配"类任务。Instruct2Act 用 LLM 做任务理解 + 规划，更 general。

---

## 13. 我觉得这 paper 真正的启示

### 启示 1：Foundation model scaling 直接受益
SAM-H vs SAM-B 提升 9 个百分点，CLIP-H vs CLIP-B 提升 14 个百分点。**未来 GPT-4 level LLM + SAM-2 + CLIP-2 配合，性能会进一步大幅提升，而且不用重训任何 robot policy。**这是 modular 架构的根本优势。

### 启示 2：Processing module 是 zero-shot deployment 的关键
mask post-processing 贡献 32 个百分点提升，比换更大的 SAM 还多。**光调用 foundation model 不够，需要 domain-specific 的 post-processing 把"还行"变成"够用"。**这对所有想 deploy foundation model 的人都是重要教训。

### 启示 3：Code interpreter 是免费的 verifier
Python interpreter 天然验证 syntax error，runtime error 也能 retry。这是 LLM for code 在 robotics 上的独特红利——其他 modality 没这种免费验证。未来可以加 type checker、static analyzer、formal verifier 做 safety check。

### 启示 4：Long-horizon task 上 zero-shot 反超 trained policy
这个反直觉结果说明 planning 应该在 symbolic level 做，perception 应该 closed-loop 做，不要 end-to-end 一把梭。**End-to-end 在长 horizon 上的 error accumulation 是结构性问题，不是数据量能解决的。**

### 启示 5：Prompt engineering 在 robotics 上的手艺
API hierarchy、in-context example 覆盖、CoT 引导、third-party library 提示——这些 prompt 工程细节是 LLM robotics 真正的"know-how"。不是简单"给 LLM 一个任务描述它就能做机器人"。

### 启示 6：未来方向——closed-loop + error recovery
当前 retry 只针对 syntax error。可以扩展到 runtime error：CLIP retrieval 不置信时让 LLM 重新 query，PickPlace 失败时让 LLM 看 failure image 重新 plan。Inner Monologue + Instruct2Act 的结合是 obvious next step。

### 启示 7：Action primitive 扩展是真正瓶颈
当前只有 PickPlace / Rotate / Rearrange。要扩展到 Push、Pour、Cut、Fold 等 manipulation skill，需要 LLM 理解每个 skill 的 precondition 和 effect。这接近 PDDL planning 或 Behavior Tree 的思想，但更灵活。

### 启示 8：与 RT-2 等的 hybrid 是 future
高层用 Instruct2Act 做 task decomposition，底层每个 sub-task 用 RT-2 这种 learned policy 执行。Symbolic + neural 结合，既有 LLM 的 compositional generalization，又有 learned policy 的 low-level dexterity。

---

## 14. 一句话总结这 paper 的本质

**它把 LLM tool-use 范式从 "GPT 调用 web search" 推广到 "GPT 调用 SAM/CLIP + robot primitives"，并证明这在 zero-shot robotics 上 work，特别在 long-horizon 任务上 work 得比 end-to-end trained policy 还好。**

这个方向的未来想象空间巨大：未来 foundation model 升级、LLM 升级、prompt 工程成熟、closed-loop feedback 加入，每一步都能让这个框架进一步变强，而不用重新训练任何 robot-specific 参数。

---

参考链接汇总：
- Instruct2Act GitHub: https://github.com/OpenGVLab/Instruct2Act
- Instruct2Act 论文: https://arxiv.org/abs/2305.05498
- VIMA benchmark: https://arxiv.org/abs/2210.03094
- Code as Policies: https://arxiv.org/abs/2209.07753
- SAM: https://arxiv.org/abs/2304.02643
- CLIP: https://arxiv.org/abs/2103.00020
- ViperGPT: https://arxiv.org/abs/2303.08128
- VisProg: https://arxiv.org/abs/2211.11559
- SayCan: https://arxiv.org/abs/2204.01691
- PaLM-E: https://arxiv.org/abs/2303.03378
- Inner Monologue: https://arxiv.org/abs/2207.05608
- LLaMA-Adapter: https://arxiv.org/abs/2303.16199
- Zero-shot CoT: https://arxiv.org/abs/2205.11916
- GLIP: https://arxiv.org/abs/2112.03857
- DALL-E-Bot: https://arxiv.org/abs/2210.02438
- CLIPort: https://arxiv.org/abs/2109.12098
- PerAct: https://arxiv.org/abs/2209.05451
- Text2Motion: https://arxiv.org/abs/2303.12153
- TidyBot: https://arxiv.org/abs/2305.05658
- R3M: https://arxiv.org/abs/2203.12601
- Whisper: https://arxiv.org/abs/2212.04356
- Visual ChatGPT: https://arxiv.org/abs/2303.04671
- PyBullet: https://pybullet.org/
- RT-2: https://arxiv.org/abs/2307.15818
- RT-1: https://arxiv.org/abs/2212.06817
- Gato: https://arxiv.org/abs/2205.06175
- Flamingo: https://arxiv.org/abs/2304.02643
- ProgPrompt: https://arxiv.org/abs/2309.07469

---

# Instruct2Act: 深度解析

这篇 paper 由 Shanghai AI Lab、PKU、CUHK 等合作完成，核心 idea 是用 LLM 作为 orchestrator，把自然语言 / 多模态 instructions 翻译成可执行的 Python 代码，这段代码通过 API 调用 SAM、CLIP 等 foundation models 完成 perception，再调用 robotic primitives 完成 action。整个 pipeline 是 zero-shot 的，无需 fine-tuning。

GitHub repo: https://github.com/OpenGVLab/Instruct2Act
arXiv: https://arxiv.org/abs/2305.05498 (推测)
相关 VIMA benchmark: https://arxiv.org/abs/2210.03094
CaP (Code as Policies): https://arxiv.org/abs/2209.07753
SAM: https://arxiv.org/abs/2304.02643
CLIP: https://arxiv.org/abs/2103.00020
ViperGPT: https://arxiv.org/abs/2303.08128
VisProg: https://arxiv.org/abs/2211.11559

---

## 1. Motivation 与设计哲学

之前的 language-conditioned robotic 工作大致分两条路线：

**路线 A: End-to-end learning-based policy**——例如 CLIPort、PerAct、VIMA、Gato、Flamingo、RT-1/RT-2 等，把语言 + image 作为输入，直接回归 action。需要海量 demonstration data，泛化能力受限于 training distribution，且 task 越长越难学。

**路线 B: LLM-as-planner / Code-as-Policy**——例如 SayCan、Inner Monologue、CaP、ProgPrompt 等，把 LLM 当 planner，输出 sequence of skills 或 Python code。优点是 compositional generalization 强，缺点是 perception 部分通常依赖 hand-crafted detector 或者简单的 object list。

Instruct2Act 想要站在路线 B 的肩膀上，但是引入路线 A 中 visual foundation models 的 perception 能力——把 SAM 和 CLIP 当作"可调用的 perception service"，由 LLM 在生成的代码里 orchestrate。这本质上是 **neuro-symbolic** 思想在 robotics 上的落地，类似于 VisProg/ViperGPT 在 vision QA 上的做法，但加了 action loop 与 robot hardware 的层级。

Intuition 上可以这么理解：LLM 不是单独在"想"，它是"写程序去调用其他模型想"。LLM 输出的 Python 代码本质是一种 **declarative plan**，每个 API call 都对应一个 well-defined 的 perception/control 步骤，错误隔离在函数边界，便于 debug 和 verify。这与 ChatGPT 的 tool-use / ReAct 范式异曲同工，只不过这里是 embedding 在 robotics 的 perception-action loop 里。

---

## 2. 系统架构详解

整个系统的 data flow 可以画成：

```
Instruction (text / image / pointing)
    │
    ▼
┌────────────────────────────────────────────┐
│ LLM (text-davinci-003 / ChatGPT)            │
│  - Prompt: API definitions + in-context examples
│  - Output: main() Python function string    │
└────────────────────────────────────────────┘
    │  generated code (Python string)
    ▼
┌────────────────────────────────────────────┐
│ Python Interpreter (executes main())        │
│  ├─ GetObsImage()    → Image from camera    │
│  ├─ SAM(image)       → masks dict           │
│  ├─ ImageCrop(image, masks) → objs, masks   │
│  ├─ CLIPRetrieval(objs, query) → obj_idx    │
│  ├─ Pixel2Loc(obj, masks) → robot coords    │
│  ├─ PickPlace(pick, place, bounds) → action │
│  └─ RobotExecution(action) → result dict    │
└────────────────────────────────────────────┘
    │
    ▼
Low-level controller (PyBullet simulator / real robot)
```

### 2.1 API 的层级组织

这点是这篇 paper 的一个微妙但重要的设计。API 不是平铺列表，而是按 robotic system 的功能层级分组：

- **First Level: File IO** —— `GetObsImage()`, `SaveFailureImage()`，负责和外部环境通信。
- **Second Level: Core Modules**
  - **Perception**: `SAM()`, `CLIPRetrieval()`, `GetObjMatch()`, `ImageCrop()`
  - **Action**: `Pixel2Loc()`, `PickPlace()`, `DistractorActions()`, `RearrangeActions()`
- **Third Level: Robotic Hardware** —— `RobotExecution()`, `SpeedSet()`

这个 hierarchy 同时反映在 prompt 的结构里（`# Second Level: Core Modules` 这种注释会写进 prompt），让 LLM 理解每个 API 的"职责边界"。这与 ViperGPT 只给 function-level example 相比，更像给 LLM 一份"robot OS 的系统调用表"。

### 2.2 Perception pipeline

输入 image $I \in \mathbb{R}^{H \times W \times 3}$（top-view camera）：

**Step 1 — Image pre-processing**（针对 SAM 的 zero-shot 弱点）：
- Gray threshold filter：把 image 转成 grayscale，做 threshold $\tau$ 得到 binary mask $M_{\text{gray}}$，去除阴影。阈值 $\tau$ 是预设的。
- Morphological **closing** operation：$M_{\text{close}} = (M \oplus B) \ominus B$，先 dilation $\oplus$ 再 erosion $\ominus$，用 structuring element $B$，目的是填小洞。

为什么 closing 而不是 opening？因为 shadow 滤掉后会留下小 gap，需要 closing 去填。

**Step 2 — SAM segmentation**：
$\{m_i\}_{i=1}^{N} = \text{SAM}(I_{\text{pre}})$，输出一组 masks，每个 mask 是 $H \times W$ 的 binary array。

**Step 3 — Mask post-processing**：
- **Size filtering**：去掉 mask area 过大或过小的，比如 $|m_i| < \tau_{\min}$ 或 $> \tau_{\max}$ 的丢弃。
- **Morphological opening**：$M_{\text{open}} = (M \ominus B) \oplus B$，先 erosion 再 dilation。注意这和 pre-processing 用的 closing 相反——这里要去掉小的孤立碎片、连接断点。
- **NMS (Non-Maximum Suppression)**：多个 mask 重叠时按 IoU 阈值保留一个，公式上：保留 score 最高 mask，suppress 掉 IoU $> \theta_{\text{NMS}}$ 的 neighbors。这里 SAM 输出没有 score，应该用 area 或 random order 作为 surrogate。

**Step 4 — Image cropping**：
$\{I_i\} = \text{ImageCrop}(I, \{m_i\})$，根据每个 mask 的 bounding box 从原图裁出 crop。

**Step 5 — CLIP encoding + retrieval**：
图像 crop 进 CLIP image encoder 得 $F_{I_i} = \text{CLIP}_{\text{img}}(I_i) \in \mathbb{R}^d$，其中 $d=1024$（ViT-H）。

Query side：
- 纯文本：$F_T = \text{CLIP}_{\text{txt}}(\text{query}) \in \mathbb{R}^d$
- 图像 crop（multi-modal）：$F_I = \text{CLIP}_{\text{img}}(I_{\text{query}})$

Retrieval 通过 cosine similarity：
$$i^* = \arg\max_i \frac{F_Q \cdot F_{I_i}}{\|F_Q\| \cdot \|F_{I_i}\|}$$

其中 $F_Q$ 是 $F_T$ 或 $F_I$。这是一个 **open-vocabulary detection** 的标准做法，把 detection 退化为 segmentation + classification 两步。

**Step 6 — Pixel → Robot coordinate**：
$X_{\text{robot}} = T \cdot X_{\text{pixel}} + b$，其中 $T \in \mathbb{R}^{2 \times 2}$ 是预先标定的 homography/affine matrix，$b$ 是 offset。再加 boundary clamping 防止超 workspace。

**Step 7 — Action**：`PickPlace(pick, place, bounds, yaw_angle_degree=None, tool="suction")`，调用 low-level controller执行。

---

## 3. Multi-modality Instructions 的统一处理

这是 paper 里比较有 design 感的部分。Instruct2Act 把三种 instruction modality 都统一到一个 retrieval 框架里：

### 3.1 Pure-Language
```
Instruction: "Put the green and purple polka dot block into the green container"
```
LLM 解析出 query string "the green and purple polka dot block"，传给 `CLIPRetrieval` 的 text encoder。

### 3.2 Language-Visual
```
Instruction: "Put {dragged_obj} into {base_obj}"
```
`{dragged_obj}` 是占位符，对应一张目标 object 的 crop image，存放在 environment cache `C` 中（VIMA 用的是 multi-modal prompt format，每个 placeholder 对应一个 image token）。LLM 生成的代码里写 `query=templates['dragged_obj']`，运行时从这个 cache 取出 image crop，走 CLIP image encoder。

### 3.3 Pointing-Language
当 object 描述太抽象、给 crop image 又不实际时，用户用鼠标点击 image，cursor 坐标作为 SAM 的 point prompt（SAM 接受 point/box/mask prompt）：
$\{m_i\} = \text{SAM}(I, \text{point}=(x,y), \text{label}=1)$

`label=1` 表示 foreground point。这种模式下，相当于人在 loop 里做了 grounding 的一部分，系统只做 segmentation + classification。

### 3.4 Scene-level matching（Rearrange task）
任务："Rearrange to this {scene}"，需要把当前 observation 的 object arrangement 重排成 goal scene 的样子。

做法：
- 对 current observation $I_{\text{obs}}$ 和 goal scene $I_{\text{goal}}$ 分别跑 SAM + CLIP，得到两组 object features $\{F_{I_i}^{\text{obs}}\}$ 和 $\{F_{I_j}^{\text{goal}}\}$。
- 用 **Hungarian algorithm** 求最优匹配，cost matrix $C_{ij} = -\text{cosine\_sim}(F_{I_i}^{\text{obs}}, F_{I_j}^{\text{goal}})$。

Hungarian algorithm 解决的是 assignment problem：
$$\min_{\sigma \in S_n} \sum_i C_{i, \sigma(i)}$$
其中 $\sigma$ 是 permutation，$S_n$ 是所有排列。这是 $O(n^3)$ 算法。

匹配完之后，对每个 (obs_obj, goal_obj) pair 计算位移，生成一系列 `PickPlace` action。同时如果有 goal scene 里没有的 distractor object，用 `DistractorActions` 先移走。

---

## 4. Prompt 工程细节

paper 的 prompt 设计有几个关键元素，我在 ablation study 里能看到它们各自的作用：

### 4.1 Prompt 组成
1. **Third-party library imports**（`PIL.Image`, `numpy`, `scipy`, `torch`, `cv2`, `math`, `typing.Union`）—— 让 LLM 知道能在代码里用这些库做计算，比如 numpy 角度换算。
2. **API definitions**，按层级组织，每个 API 有 docstring + signature + example usage。
3. **In-context examples**：5 个完整的 `main()` 函数例子，覆盖 single pick-place、rotation、rearrange-restore、scene-conditioned、sequential multi-step 等场景。
4. **Task instruction** 末尾插入 "Please solve the following instruction step-by-step. You should implement the main() function and output in the Python-code style."

### 4.2 5 个 in-context examples 的功能覆盖

| Example | 任务 | 训练的关键 pattern |
|---|---|---|
| 1 | Put checkerboard round into polka dot pan | Single pick-place，text query |
| 2 | Rotate {dragged_obj} 150 degrees | Multi-modal + yaw 控制 |
| 3 | Rearrange to {scene} then restore | Scene matching + 反向操作 |
| 4 | Put yellow stripe obj in {scene} into orange obj | Scene 作为 context + retrieval |
| 5 | Sequential pick-place into {base_1} then {base_2}, restore | Multi-step planning |

注意 Example 5 故意有 typo `def mian_5()`，这是 prompt 的"自然噪声"——但实测 LLM 能学到正确的 pattern。

### 4.3 Chain-of-Thought 引导

paper 提到 "Think step by step to carry out the instruction" 加在 task instruction 前，这是 Kojima et al. 的 Zero-shot CoT 技术（https://arxiv.org/abs/2205.11916）的迁移。这让 LLM 在写代码前先做 reasoning。

### 4.4 Ablation 验证 prompt 元素的作用

paper 在 Appendix A.5 给了详尽的 case study，对比三种 prompt：
- **Only API definition**：LLM 能写出 main 框架，但会 hallucinate 一些不必要的 action（如 `DistractorActions`、`RearrangeActions`），可能漏掉 `return info`。但能主动调用 `SaveFailureImage()` 处理 failure case，这是 examples 里没教过的 emergent behavior。
- **Only in-context examples**：输出更结构化，但缺少 semantic comment，甚至把 SAM 错误展开为 "Semantic Affinity Module"。变量命名 lack semantic info。
- **API + examples**：生成 human-readable、semantic-rich、结构良好的 Python 代码。

结论：API definition 给 LLM **flexibility 和 reasoning ability**；in-context examples 给 LLM **structure 和 expert pattern**。两者互补。

---

## 5. 实验结果

### 5.1 主实验：L1 generalization

| Model | Task 01 | Task 02 | Task 03 | Task 04 | Task 05 | Task 17 | Avg |
|---|---|---|---|---|---|---|---|
| DT-20M | 60.5 | 64.0 | 50.5 | 44.0 | 41.0 | 2.5 | 43.8 |
| Gato-20M | 61.5 | 62.0 | 32.5 | 49.0 | 38.0 | 2.0 | 40.8 |
| Flamingo-20M | 63 | 61.5 | 55.0 | 50.0 | 42.5 | 1.0 | 45.5 |
| **VIMA-20M** | 100 | 100 | 100 | 99.5 | 59.5 | 47.5 | **84.4** |
| **Ours-Multi** | 91.3 | 81.4 | 98.2 | 78.5 | 72.0 | 85.2 | **84.4** |
| Ours-Single | 86.7 | – | 94.6 | – | – | 63.0 | – |

注意几个亮点：
- **Task 17**（Pick in order then restore，long-horizon）—— VIMA 只有 47.5%，Instruct2Act 多模态达到 **85.2%**。这是 38 个百分点的差距。
- **Task 05**（Rearrange then restore）—— VIMA 59.5% vs Ours 72.0%，多模态版本领先 12.5%。
- 但在 Task 01（简单 pick-place），VIMA 是 100%，Ours 是 91.3%，略低。这反映了 zero-shot 模型在简单 task 上的 ceiling 不如 end-to-end training。
- 平均 84.4% 与 VIMA 持平，但 Instruct2Act 是 **zero-shot、无任何训练数据**，VIMA 是 **20M 参数、trained on VIMA dataset**。

为什么 long-horizon task 上 zero-shot 反而更好？intuition：long-horizon 任务需要 compositional planning + robust perception。End-to-end policy 在长 horizon 上要预测完整 action sequence，error 累积严重；而 Instruct2Act 把 planning 交给 LLM（symbolic level，无累积误差），把 perception 交给 SAM+CLIP（每一步重新感知，closed-loop），所以 horizon 越长优势越明显。

### 5.2 L2 / L3 generalization（Figure 5）

L2 是 combinatorial generalization（新材料组合），L3 是 novel object generalization（训练时没见过的 object）。

- L1 → L2 → L3 性能下降对 learning-based 方法非常显著（distribution shift），但 Instruct2Act 在三个 level 上曲线相对 flat。
- Figure 5(d) 的 average 显示 Instruct2Act 在 L2、L3 上仍然保持竞争力，因为 SAM 的 segmentation 和 CLIP 的 open-vocabulary classification 都是 zero-shot 设计，不会因为 object 新就失效。

这是 foundation model based approach 的关键优势：**distribution shift 的成本几乎为零**。

### 5.3 Processing module 的 ablation（Table 2）

| Image Pre | Mask Post | Task 01 | Task 02 | Task 03 | Task 04 | Task 05 | Task 17 | Avg |
|---|---|---|---|---|---|---|---|---|
| ✗ | ✗ | 70.4 | 34.6 | 88.6 | 41.7 | 15.9 | 54.7 | 51.0 |
| ✓ | ✗ | 69.7 | 33.9 | 87.7 | 40.9 | 14.9 | 52.9 | 50.0 |
| ✗ | ✓ | 91.7 | 78.2 | 97.4 | 72.9 | 69.5 | 88.3 | 83.0 |
| ✓ | ✓ | 91.6 | 80.8 | 97.8 | 78.4 | 69.1 | 87.2 | **84.1** |

关键 insight：
- **Mask post-processing 是大杀器**：从 51.0% → 83.0%，提升 32 个百分点。这说明 SAM 的 raw mask 质量不够，NMS + size filter + morphological opening 大幅改善了下游 retrieval 的稳定性。
- **Image pre-processing 单独用反而轻微下降**（50.0% vs 51.0%）。原因：pre-defined 阈值没调，shadow 滤除可能误伤目标 object 的纹理。
- 两者结合基本 = mask post 单独用，说明 pre-processing 主要起到锦上添花作用。

Intuition：mask post-processing 解决的是"误报"问题（多余的小 mask 会让 CLIP retrieval 把相似的小区域误选为 target），所以提升巨大。Image pre-processing 解决的是"漏报"问题（shadow 把 object 切成碎片），但 zero-shot 阈值难调，所以单独用收益不明显。

### 5.4 Foundation model 规模 ablation（Table 3, 4）

SAM：

| Backbone | SR(%) | Params(M) |
|---|---|---|
| Base | 75.1 | 93.7 |
| Large | 82.7 | 312.3 |
| **Huge** | **84.1** | **641.1** |

CLIP：

| Backbone | SR(%) | Params(M) |
|---|---|---|
| B-16 | 70.0 | 149.6 |
| L-14 | 76.5 | 427.6 |
| **H-14** | **84.1** | **986.1** |

非常清晰的 **scaling law** 趋势：foundation model 越大，zero-shot 性能越好。这正是 Instruct2Act 的一个 architectural 红利——别人 finetune 一个 7B model 也不见得能打过 SAM-H + CLIP-H 这种组合的 zero-shot。

### 5.5 LLaMA-Adapter 实验（Table 5）

| Task 01 | Direct | More Trials | Naive Filtering |
|---|---|---|---|
| SR(%) | 72.5 | 77.5 | 85.5 |

- 直接用 LLaMA-Adapter 已经 72.5%，证明 framework 对 open-source LLM 也可用。
- "More Trials" 是 Python interpreter 抛 exception 时重新 generate（最多 N 次）→ 77.5%。
- "Naive Filtering" 是检测生成代码里没用到 environment cache（即 `templates[...]`）就重新生成 → 85.5%。这相当于一个轻量 self-consistency check。

这意味着 LLM 的 code generation 不需要完美，**interpreter + retry 是 cheap error correction**。这跟 CodeT、Self-Debug 思路一致。

### 5.6 Robustness experiments

- **Human Intervention**：在原 instruction 后加 "I cancel this task. Stop!" → LLM 不生成执行代码。加 "That instruction is wrong, use this one. [new]" → LLM 用新指令。
- **Missing Characteristics**：随机删一些形容词 / 拼错单词 → 仍能正确 ground。
- **Synonym Replacement**："Rotate" → "Spin" → 仍工作。

这些 zero-shot robustness 是因为 LLM 本身的语言理解能力，不是 robotics-specific 训练得到的。

### 5.7 Pointing-language 模式（Table 6）

| | Task 01 | Task 03 |
|---|---|---|
| Pointing-Language | 90.7 | 98.0 |

比 pure-language（86.7、94.6）更高。Click 操作给 SAM 提供了精确的 spatial prompt，省去了 CLIP retrieval 的语义歧义。

### 5.8 第三方库扩展能力

paper 给了一个巧妙 demo：所有 in-context example 用 degree 表示旋转角度，但用户 instruction 用 "0.5 radians"。LLM 知道 numpy 可用，于是生成：
```python
angle = 0.5  # in radians
yaw_angle = angle * 180 / np.pi  # convert to degrees
```
这说明 LLM 能利用 prompt 里 import 的库做单位换算、几何计算等，**policy code 的表达能力远超 sequence-of-skills**。

---

## 6. 与相关工作的细致对比

### 6.1 vs CaP (Code as Policies)
- CaP 直接用 LLM 生成 policy code，但 perception API 是 hand-crafted（比如 `get_object_position("mug")` 返回硬编码结果），不能 zero-shot 处理 novel object。
- Instruct2Act 的 perception 由 SAM+CLIP 提供，是真正的 open-vocabulary。
- CaP 缺乏 hierarchical API 组织，处理 long instruction 容易出错。
- Instruct2Act 通过 processing module 缓解 SAM zero-shot 的精度问题。

### 6.2 vs VIMA
- VIMA 是 end-to-end multimodal transformer，把 text+image prompt encode 后直接预测 action。
- 需要 20M 参数训练，泛化受限于训练 distribution。
- Long-horizon task 上 planning 容易 collapse。
- Instruct2Act 在 long-horizon 上完胜 VIMA。

### 6.3 vs SayCan
- SayCan 用 LLM 生成 next skill，再用 affordance model 评估 grounding 可能性，本质是 sequential planner。
- 缺乏 closed-loop perception，skill 执行依赖预训练的 controller。
- Instruct2Act 每一步都重新感知环境，更鲁棒。

### 6.4 vs ViperGPT / VisProg
- 同样是 LLM 生成 code 调用 foundation models，但 domain 是 vision QA 而非 robotics。
- 没有 action loop，没有 closed-loop execution。
- 只给 function-level example，没有 hierarchical API 组织。
- Instruct2Act 借用了 VisProg 的 in-context example 思路，但加了 robotic API hierarchy 和 closed-loop execution + error handling。

### 6.5 vs PaLM-E
- PaLM-E 把视觉 observation embedding 直接 inject 到 PaLM 的 token stream，是 monolithic multimodal model。
- 训练成本巨大（540B PaLM + 22B ViT）。
- Instruct2Act 用 modular API，foundation models 之间解耦，任何一个升级都能受益。

### 6.6 vs DALL-E-Bot
- DALL-E-Bot 用 Stable Diffusion 生成 goal scene image，再用 image matching 引导 action。
- 没有 LLM 做高层 reasoning，只能处理"目标视觉状态匹配"类任务。
- Instruct2Act 用 LLM 做任务理解 + 规划，更 general。

---

## 7. Limitations 与潜在问题

paper 自己提到的：
1. **Compute cost**：SAM-H + CLIP-H + LLM API 调用，对 real-time robotic 系统负担大。每个 action step 都要跑一遍 SAM，目前 inference 几秒级别。
2. **Action primitive 限制**：只有 PickPlace / Rearrange / Rotate 几种 basic primitives，复杂 manipulation（如双手协作、deformable object）需要扩展 API。
3. **只测了 simulation**：PyBullet + top-view camera，没有 real-world 验证。

我观察到的潜在问题：
4. **Coordinate calibration**：`Pixel2Loc` 依赖预标定的 $T$ 矩阵和 $b$ offset，换 camera setup 就要重标。
5. **CLIP 在 fine-grained attribute 上的局限**：例如 "polka dot block" 和 "checkerboard round" 这种纹理区分，CLIP H-14 的 zero-shot accuracy 不一定够。Table 1 Task 02 上 Instruct2Act 是 81.4%（多模态），还有 18.6% 失败。
6. **Open-loop code execution**：生成的 `main()` 一旦执行就跑到底，中间出错不会自动 re-plan（只有 retry-on-syntax-error）。Inner Monologue 那种 closed-loop feedback 还没集成进来。
7. **In-context example 的覆盖问题**：5 个 example 不可能覆盖所有 task pattern，新 task 出现时可能 need prompt engineering。

---

## 8. 对未来工作的启示

### 8.1 Scaling 思路
- 这篇 paper 证明了 foundation model 的 scaling benefit 能直接 transfer 到 robotics（Table 3, 4）。未来 GPT-4 level LLM + SAM-2 + CLIP-2 配合，性能会进一步提升。
- 这个架构本身是个"插槽式"设计，任何新 foundation model 都能插进来：把 SAM 换成 Grounding DINO + SAM-2，把 CLIP 换成 SigLIP 或 EVA-CLIP，性能大概率进一步提升。

### 8.2 Closed-loop + Error Recovery
- 当前 retry 只针对 syntax error。可以扩展到 runtime error（如 CLIP retrieval 不置信、PickPlace 失败），加入 Inner Monologue 式的 verbal feedback。
- 让 LLM 看到 failure image，写 debug 代码再 retry。

### 8.3 Real-world transfer
- 关键问题是 SAM 在 real image 上的 segmentation quality、camera 标定精度、grasping 的 robustness。
- 可能需要加入 tactile feedback 或 force-torque sensing 作为额外 API。

### 8.4 长期记忆与 multi-turn instruction
- 当前每个 instruction 是一次性的 `main()`。可以引入 session-level memory，让 LLM 跨 instruction 维护 state。

### 8.5 Action primitive 的扩展
- 把 action API 从 PickPlace 扩展到 Push, Pour, Pour, Cut, Fold 等 manipulation skills，需要 LLM 理解每个 skill 的 precondition 和 effect。
- 这接近 Behavior Tree 或 PDDL planning 的思想。

### 8.6 与 RT-2 / VPT 等的融合
- RT-2 把 VLM 直接 tokenize 到 action token，是 end-to-end 的路。可以想象一个 hybrid：高层用 Instruct2Act 做 task decomposition，底层每个 sub-task 用 RT-2 这种 learned policy 执行，结合 symbolic + neural 的优势。

### 8.7 Verifiable code generation
- Python interpreter 天然是 verifier。未来可以加 type checker、static analyzer、甚至 formal methods 验证生成代码符合 safety constraint（比如 collision check）。
- 这是 LLM for code 在 robotics 上的独特红利——其他 modality 没有 interpreter。

---

## 9. 公式总结与变量含义

为了清晰起见，把 paper 涉及的关键数学公式列一下：

**CLIP retrieval**：
$$i^* = \arg\max_{i \in \{1,...,N\}} \cos(F_Q, F_{I_i}) = \arg\max_i \frac{F_Q^\top F_{I_i}}{\|F_Q\|_2 \|F_{I_i}\|_2}$$

- $N$：SAM 输出的 candidate mask 数量
- $F_Q \in \mathbb{R}^d$：query embedding，纯文本时 $F_Q = \text{CLIP}_{\text{txt}}(\text{query})$，多模态时 $F_Q = \text{CLIP}_{\text{img}}(I_{\text{query}})$
- $F_{I_i} \in \mathbb{R}^d$：第 $i$ 个 crop 的 CLIP image embedding
- $d$：embedding 维度，CLIP ViT-H 是 1024

**Hungarian matching (Rearrange task)**：
$$\sigma^* = \arg\min_{\sigma \in S_n} \sum_{i=1}^{n} C_{i, \sigma(i)}$$

- $\sigma$：一种 permutation，把 obs object index 映射到 goal object index
- $S_n$：$n$ 个元素的对称群（所有 permutation）
- $C_{ij} = -\cos(F_{I_i}^{\text{obs}}, F_{I_j}^{\text{goal}})$：负 cosine similarity 作为 cost，求最小化等价于最大化 similarity
- $\sigma^*$：最优匹配

**Morphological operations**：
- Dilation: $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$
- Erosion: $A \ominus B = \{z \mid (B)_z \subseteq A\}$
- Closing: $A \bullet B = (A \oplus B) \ominus B$（填小洞，连接窄 gap）
- Opening: $A \circ B = (A \ominus B) \oplus B$（去小颗粒，分离粘连）

$A$ 是 binary mask，$B$ 是 structuring element（paper 里用 small square/disk）。

**Coordinate transformation**：
$$\begin{bmatrix} x_{\text{robot}} \\ y_{\text{robot}} \end{bmatrix} = T \begin{bmatrix} u_{\text{pixel}} \\ v_{\text{pixel}} \end{bmatrix} + b$$
然后 clamp 到 workspace bounds $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$。

- $T \in \mathbb{R}^{2 \times 2}$：标定的 affine matrix（包含 scale、rotation）
- $b \in \mathbb{R}^2$：translation offset
- $(u_{\text{pixel}}, v_{\text{pixel}})$：pixel 坐标，通常是 mask 的 centroid
- $(x_{\text{robot}}, y_{\text{robot}})$：robot base frame 坐标

**PickPlace action signature**（伪公式）：
$$a = \text{PickPlace}(\text{pick}=X_p, \text{place}=X_q, \text{bounds}=B, \theta=\theta_{\text{yaw}}, \text{tool}=\tau)$$

- $X_p, X_q \in \mathbb{R}^2$：pick 和 place 的 robot 坐标
- $B \in \mathbb{R}^4$：workspace boundary
- $\theta_{\text{yaw}} \in [0, 360)$：rotation 角度（degree）
- $\tau \in \{\text{suction}, \text{gripper}\}$：end-effector 类型

---

## 10. 整体评价

Instruct2Act 是一篇**架构性**的 paper，它的 contribution 不在某个 SOTA number，而在一个**清晰、可扩展、zero-shot** 的 neuro-symbolic robotics 框架。关键 takeaway：

1. **Foundation models + LLM code generation + hierarchical API** 是 robotics 的 promising 路线，尤其适合 long-horizon task。
2. **Processing module 是 zero-shot foundation model 落地的关键**——光调用 SAM 不够，需要 domain-specific 的 mask post-processing。
3. **Code interpreter 是天然的 verifier + error recovery**——比直接预测 action token 多了 symbolic debug 能力。
4. **Modular 架构受益于 foundation model 的 scaling**——不需要重训 robot policy 就能享受 SAM-H / CLIP-H 的提升。
5. **Long-horizon 任务上 zero-shot 反超 trained policy**——这是个反直觉的结论，原因是 planning 的 symbolic level 避免了 end-to-end 的 error accumulation。

限制也比较明显：实时性、action primitive 覆盖、real-world 验证、closed-loop feedback 缺失。这些都是后续工作的 obvious directions。

作为 LLM-driven robotics 的 baseline 设计，这篇 paper 的 prompt、API hierarchy、processing module 都值得作为模板借鉴。

---

参考链接汇总：
- Instruct2Act: https://github.com/OpenGVLab/Instruct2Act
- VIMA benchmark: https://arxiv.org/abs/2210.03094
- Code as Policies: https://arxiv.org/abs/2209.07753
- SAM: https://arxiv.org/abs/2304.02643
- CLIP: https://arxiv.org/abs/2103.00020
- ViperGPT: https://arxiv.org/abs/2303.08128
- VisProg: https://arxiv.org/abs/2211.11559
- SayCan: https://arxiv.org/abs/2204.01691
- PaLM-E: https://arxiv.org/abs/2303.03378
- Inner Monologue: https://arxiv.org/abs/2207.05608
- LLaMA-Adapter: https://arxiv.org/abs/2303.16199
- LLaMA-Adapter V2: https://arxiv.org/abs/2304.15010
- Zero-shot CoT: https://arxiv.org/abs/2205.11916
- Visual ChatGPT: https://arxiv.org/abs/2303.04671
- GLIP: https://arxiv.org/abs/2112.03857
- DALL-E-Bot: https://arxiv.org/abs/2210.02438
- CLIPort: https://arxiv.org/abs/2109.12098
- PerAct: https://arxiv.org/abs/2209.05451
- Text2Motion: https://arxiv.org/abs/2303.12153
- TidyBot: https://arxiv.org/abs/2305.05658
- R3M: https://arxiv.org/abs/2203.12601
- Whisper: https://arxiv.org/abs/2212.04356
- PyBullet: https://pybullet.org/
