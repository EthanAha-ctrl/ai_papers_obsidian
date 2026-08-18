---
source_pdf: A3VLM Actionable Articulation-Aware Vision Language Model.pdf
paper_sha256: 2da4d98866272f80c20953fb27789ef507e5a521b9b9c8c752ab232034c2257a
processed_at: '2026-08-17T23:38:55-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 A3VLM

好，我换个调子，跟 Karpathy 你坐下来喝咖啡那种感觉聊。

---

## 这个 paper 到底干了个啥事

robotics VLM 现在的问题很清楚：大家都在让 VLM 直接 output robot action。RT-2 这种，input 一张图 + 指令，output 就是 gripper 的 6D pose，token化成 text 输出。听起来很酷，但你要喂它 13 万条 robot demonstration data，而且换了 robot 就得重训一遍。

A3VLM 这帮人说：等等，我们想想这件事到底难在哪。

开个 drawer、拧个 faucet、打开 laptop——这些动作的难度其实不在 robot 本身，而是在 **object 的结构**。drawer 沿哪条轴拉、faucet 绕哪根轴转、laptop 的 hinge 在哪里，这些是 object 的属性，跟你是 Franka 还是 Kuka 没半毛钱关系。

所以 A3VLM 让 VLM output 的不是 robot action，而是 **object 的 articulation 描述**。拿到这个描述之后，随便什么 robot，接一个简单的 primitive 就能 translate 成自己的 action。

这个 framing 我觉得是 clean 的。

---

## 三元组 (B, A, S)：核心 representation

每个 movable part 用一个三元组描述：

**(B, A, S)**

- **B** = Bounding box，3D，8 个 vertex
- **A** = Axis，articulation 轴，2 个 3D 端点
- **S** = Semantic label，包含 joint type、link name、action type

### B 的具体公式

$$B = \{(x_i, y_i, z_i)\}_{i=1,\dots,8}$$

- $x_i, y_i$：第 $i$ 个 vertex 在 image plane 上的 pixel 坐标
- $z_i$：normalized depth，用 scene 的 max/min depth 归一化到 (0, 1)
- $i$ 是 vertex index，从 1 到 8

**关键 intuition**：为什么要 normalized depth 而不是 absolute 3D 坐标？因为 VLM 输出是 text token，absolute 3D 坐标数值范围大、不好 token化。normalized 到 (0,1) 后，2 位小数就能表达，比如 "0.42"。

代价是 deployment 时你得知道 scene 的 min/max depth，所以 real-world 还是需要 depth camera 做"反归一化"。这其实是个工程 trade-off，paper 没在 main text 强调，藏在 Appendix B 里。

### A 的公式

$$A = \{(\alpha_i, \beta_i, \gamma_i)\}_{i=1,2}$$

- $\alpha_i, \beta_i$：axis 端点的 2D 投影
- $\gamma_i$：normalized depth
- $i \in \{1, 2\}$：两个端点

revolute joint：A 就是 URDF 里的 rotation axis。
prismatic joint：A 方向是 URDF 的 prismatic direction，强制穿过 link 3D center（为了和 B 对齐）。

### S 包含三件事
1. joint type：`prismatic` | `revolute`
2. link name：比如 "handle", "lid", "door"
3. action type：比如 "open", "close", "rotate"

**和 GaPartNet 对比**：GaPartNet 用 9 类 prototype，A3VLM 砍到 2 类。simplification 的好处是 VLM vocabulary 小、token 准确率高；坏处是 lose 了一些 fine-grained 语义。但 action primitive 也只需要覆盖 2 种 joint type，工程上简单很多。

参考：GaPartNet https://arxiv.org/abs/2211.12144

---

## 为什么不一次性 output 全部 (B, A, S)

直觉上你可以让 VLM 一次性回答："这个 drawer 的 bbox、axis、type 是啥"。

paper 把它拆成 4 个 sub-task：

| Task | 输入 | 输出 | 样本数 |
|---|---|---|---|
| Detection | image | 所有 movable parts 的 B | 43K |
| REC-Link | image + "lid" | lid 的 B | 178K |
| REG-Joint | image + B | 该 part 的 A 和 S | 18K |
| REC-Action | image + "open storage" | actionable part 的 B + S | 15K |

总共 254K 样本。

**intuition**：如果让 VLM 一次吐出 B+A+S，cross-entropy loss 在长 output 上信号被稀释。拆开后每个 task 让 VLM 专注一个 concept，loss landscape 更 friendly。

这跟 LLM 训练里 chain-of-thought 的哲学类似——拆 step 比端到端 reasoning 更 stable。

Figure 1 展示了 sequential inference：先 REC-Action 问 actionable part 在哪，再 REG-Joint 问那个 part 的 articulation。两步 inference。

代价是慢——real-time control 受限。

---

## Dataset 怎么搞出来的

这是 paper 最 tricky 的工程部分，main text 一笔带过，但其实 trick 很多。

### 来源
PartNet-Mobility：2000+ objects，46 categories，URDF format。每 object 渲染 40 张图（随机 camera position / lighting / joint value）。

### Augmentation：ControlNet + ChatGPT
PartNet 原始 render 都是 plain gray（你看 Figure 3 全是灰色），sim2real gap 极大。他们用 ControlNet 增强：

- **主 condition**：depth map（保留 geometry + semantic）
- **fallback**：对 depth variance 小的 object，用 semantic segmentation
- **prompt diversity**：ChatGPT 生成多样化 texture description

Figure 9 对比 raw 和 augmented，效果显著。

**关键 intuition**：augmentation 必须保持 articulation 几何不变。所以 depth 作 condition 比直接 text-to-image 安全——你不会让 drawer 的形状变了，但可以让它有木头纹理、金属纹理。

### Affordance 标注的 hack
PartNet-Mobility **不提供 affordance**！只有 link name。怎么办？

作者用 GPT-4 从 RoboAgent 的 skill library 里选 skill，prompt 见 Appendix A。把 URDF 的 link name 喂给 GPT-4，让它 match 对应的 robot skill。

这其实是个聪明的 workaround——LLM 作 semantic bridge，把 URDF 的几何描述和 robot skill 库连起来。

参考：
- PartNet-Mobility https://sapien.ucsd.edu/
- ControlNet https://arxiv.org/abs/2302.05543
- RoboAgent https://arxiv.org/abs/2309.01918

---

## Architecture：三个 visual encoder 拼一起

基于 SPHINX-X，LLaMA2 作 language backbone。Visual 部分用了三个 encoder channel-wise concat：

1. **CLIP**：language-aligned semantic，400M image-text pair 预训练
2. **DINOv2**：self-supervised，patch-level detail 强，对 detection 任务重要
3. **Q-Former**（来自 Flamingo）：把任意数量 image token 压缩成固定数量，global feature summarization

**为什么三个一起用**：
- detection 任务（REC-Link）需要 dense local feature → DINOv2
- reasoning 任务需要 global context → Q-Former
- language alignment 让 LLM 能理解 visual token → CLIP

三个 features channel-wise concat 后过 projection layer 对齐到 LLM token dimension。

bbox value 归一化到 (0,1)，precision 2 位小数。这跟 Shikra 一样，是 VLM 输出 numeric 的 standard practice。

### 训练细节
- 8x A100 80GB
- 3 epochs，24 小时
- batch size 4，lr $2 \times 10^{-5}$
- visual encoder frozen，只 fine-tune projection + LLM
- Two-stage：先用 "This is a [OBJ]" naive caption 训 projection layer，再训全模型

参考：
- SPHINX-X https://arxiv.org/abs/2402.05935
- CLIP https://arxiv.org/abs/2103.00020
- DINOv2 https://arxiv.org/abs/2304.07193
- Flamingo https://arxiv.org/abs/2204.14198
- Shikra https://arxiv.org/abs/2306.15195

---

## Action primitive：从 (B, A, S) 翻译成 robot motion

定义 3 种 primitive：

```
if joint_type == "prismatic": use Slide
elif joint_type == "revolute":
    if link_name in {"bottle_cap", "scroll_button"}: use Scroll
    else: use Rotate
```

- **Slide**：沿 axis A 方向 linear motion
- **Rotate**：绕 axis A 的 circular arc
- **Scroll**：沿 axis A 的 helical motion（旋转 + 平移组合）

### Grasp point 选择
- Scroll：grasp pose 必须 overlap with axis A
- 其他：在 B 内随机选 grasp pose 作 contact point C

### 关键 design choice
A3VLM 不假设 gripper 类型。需要外部 grasp proposer（GPG / GraspNet / manual）。

paper real-world experiment 用 manual grasp pose list，简化实验。这是 honest 写法，但也暴露了 limitation——完整 pipeline 还需要 grasp detection 模块。

参考：
- GraspNet https://arxiv.org/abs/2003.08521
- GPG https://arxiv.org/abs/1603.01542

---

## 实验数字

### Simulation benchmark（Table 2）

Benchmark 来自 ManipLLM，Sapien simulator + PartNet-Mobility + Franka flying gripper with suction。

Success criterion：
$$d > \sigma, \quad \sigma = 0.01$$

$d$ 是被 manipulate 的 part 位移，$\sigma$ 是 threshold。

| Method | Training AVG (20 cat) | Testing AVG (10 cat) |
|---|---|---|
| Where2Act | 0.26 | 0.21 |
| UMPNet | 0.35 | 0.28 |
| FlowBot3D | 0.37 | 0.32 |
| Implicit3D | 0.46 | 0.41 |
| ManipLLM | 0.59 | 0.54 |
| **A3VLM** | **0.76** | **0.72** |

比 SOTA ManipLLM 高 17% / 18% absolute。generalization 几乎无损（0.76 → 0.72），而 ManipLLM 掉 5 pts。

### RGB vs Depth ablation（Table 4）

| Method | Training AVG | Testing AVG |
|---|---|---|
| A3VLM (RGB) | 0.76 | **0.72** |
| A3VLM-depth | 0.76 | 0.70 |

Training 上 depth 略好，但 testing 上 RGB 明显更好。

**intuition**：simulation depth 是 perfect geometry，real depth 充满 noise（reflective surface、transparent object），distribution shift 严重。RGB 的 appearance gap 被 ControlNet augmentation 缓解了，反而更 robust。

这也是 Figure 6 那张 real-world 推理图的 selling point——A3VLM 能处理 microwave（reflective）、pot（reflective）、coke bottle（transparent）这种 point cloud 方法会挂的 object。

### Real-world manipulation（Table 3）

5 个 object，每个 5 trials，Kuka + Robotiq 3-finger + RealSense D415：
- 4 个 object：5/5 success
- 1 个 object：4/5 success

24/25 overall，很 solid。

参考：
- ManipLLM benchmark https://arxiv.org/abs/2312.16217
- Sapien https://arxiv.org/abs/2003.08515

---

## Appendix D 的诚实记录

paper 在 Appendix 记录了 RGB-D 和 Point Cloud 输入的失败尝试。这种 honest reporting 在 robotics paper 里挺难得。

### RGB-D 失败原因
- visual foundation model 预训练在 RGB 上，depth 没 texture，feature extraction 失败
- token sequence 太长，LLM 处理不了

### Point Cloud 失败原因
- 用 PointBert + RECON encoder
- point cloud 没 visual texture，partial-level detection 困难
- point cloud model 参数量和训练数据都远小于 visual foundation model

**intuition**：这其实是整个 3D community 的痛点。3D foundation model 还没成熟到 RGB 同等水平。ShapeLLM、PointLLM 在 captioning 上还行，fine-grained detection 还不够。RGB 的 visual encoder 已经吃了 ImageNet + CLIP + LAION 几十亿图片的红利，3D 模型还在 Million 级 point cloud 上挣扎。

参考：
- ShapeLLM https://arxiv.org/abs/2402.17766
- PointLLM https://arxiv.org/abs/2308.16911
- PointBert https://arxiv.org/abs/2111.14819

---

## 我觉得值得批判的地方

### 1. Bounding box 精度限制
2 位小数归一化坐标，对精细 manipulation 不够。USB 插入这种 mm 级精度任务，2 位小数的 bbox 肯定不够。

### 2. Action primitive 只有 3 种
现实世界 articulation 远不止 slide / rotate / scroll。multi-DOF joint（比如球关节）、deformable object（布、绳子）都覆盖不到。

### 3. 依赖外部 grasp proposer
paper 用 manual grasp pose list，实际 deployment 接 GraspNet，pipeline 复杂度被隐藏了。claim "robot-agnostic" 是对的，但没告诉你还需要一个 grasp module。

### 4. Sequential inference 慢
两步 inference 才能拿到完整 (B, A, S)，real-time control 受限。这点 paper 没明说。

### 5. Sim2Real 的 SAM workaround
real-world 用 SAM 去 background，假设 object 是 salient foreground。复杂场景（cluttered kitchen）可能失效。

### 6. "RGB-only" claim 有点弱
deployment 还是需要 depth camera 做 normalized depth 的反归一化。所以"RGB-only"严格说是 inference 时 RGB-only，但 deployment pipeline 仍需要 depth。

---

## 跟你 Karpathy 视角的联系

你一直强调 **representation 比policy 重要**（Software 2.0 那篇）。A3VLM 本质上就是学一个 structured 的 object articulation representation，然后让 representation 足够 structured 以至于简单 primitive 就能 decode 成 action。这跟你 philosophy 一致。

但反过来可以问：为什么 articulation representation 一定要用 VLM 的 text output encode？

bbox 用 token化的 2 位小数其实是个 hack。如果你设计一个 **continuous output head**（VLM backbone + regression head 输出 bbox 和 axis），可能更精确。但这样就 lose 了 LLM 的 reasoning 和 in-context learning。

这是 LLM-based robotics 的根本 tension：**text interface 既是最强接口（compositional、interpretable、transferable）也是最大瓶颈（discrete、low precision）**。

你怎么看这个 trade-off？

参考：
- Karpathy Software 2.0 https://karpathy.medium.com/software-2-0-a64152b37c35

---

## 未来方向联想

1. **3D foundation model 成熟后**：如果 ShapeLLM 这种达到 RGB MLLM 水平，A3VLM 可以原生处理 point cloud
2. **Diffusion policy 接口**：(B, A, S) 作 diffusion policy 的 conditioning，结合 fine-grained trajectory generation
3. **Video input**：articulation 的 motion 在 video 里更明显，single image 其实丢了 motion 信息
4. **Tactile feedback**：完全 open-loop，加 tactile 做 failure detection 会大幅提升 robustness
5. **Articulation 的信息论分析**：(B, A, S) 是否 sufficient？friction coefficient、joint limit 这些没建模
6. **VLM 直接 output action token**：把 primitive 也 encode 进 vocabulary，end-to-end 训练

---

## 一句话总结

A3VLM 的核心 insight：**robot action 的复杂性来自 object articulation structure，把这部分从 robot action 里 decouple 出来，单独学一个 object-centric representation，既省 data 又 transferable**。

数字上：sim 比 SOTA 高 17%，real-world 24/25 success，RGB 在 testing generalization 上反直觉地优于 depth。

我觉得最值得 follow 的方向是 continuous output head 和 video input——前者解决 precision bottleneck，后者解决 single-frame ambiguity。

repo: https://github.com/changhaonan/A3VLM

---

# A3VLM: Actionable Articulation-Aware Vision Language Model 深度讲解

Andrej，这篇 paper 来自 SJTU / Shanghai AI Lab / Rutgers / PKU / CUHK 的合作，第一作者 Siyuan Huang 和 Haonan Chang。核心 idea 我先 build 你的 intuition，然后再拆 technical details。

---

## 1. Big Picture: 为什么这个工作重要

整个 robotics VLM 领域目前分三派：

**派别 A**：LLM 作 high-level planner，外部接 low-level API（SayCan, Code-as-Policies, Instruct2Act）。cheap 但 brittle，只能 pick-and-place。

**派别 B**：VLM 直接 output robot action（RT-1, RT-2, ManipLLM）。powerful 但 data-hungry——RT-2 需要 130k+ robot episodes，ManipLLM 靠 simulation 收 suction gripper 数据。这些 action representation 是 **robot-centric** 的，换一个 robot 就得重训。

**派别 C**：VLM output **中间表示**（VoxPoser 的 cost map, MOKA 的 constraint, ManipVQA 的 affordance），再用简单 primitives 翻译成 action。A3VLM 属于这一派。

A3VLM 的关键 insight：**robot action 的复杂性其实来自 object 的 articulation structure，而非 robot 本身**。一个 drawer 怎么开、一个 faucet 怎么拧，这是 object 的属性，与 Franka 还是 Kuka 无关。所以学习 object-centric representation 比 robot-centric representation 更 sample efficient、更 transferable。

参考文献：
- RT-2: https://arxiv.org/abs/2307.15818
- ManipLLM: https://arxiv.org/abs/2312.16217
- VoxPoser: https://arxiv.org/abs/2307.05973
- GaPartNet: https://arxiv.org/abs/2211.12144 (CVPR 2023)

---

## 2. Articulation Representation: 三元组 (B, A, S)

这是 paper 的核心设计。每个 movable part 用一个 triad 描述：

### 2.1 B (Bounding box)
3D bounding box，由 8 个 vertex 表示：
$$B = \{(x_i, y_i, z_i)\}_{i=1,\dots,8}$$

变量含义：
- $x_i, y_i$：vertex $i$ 在 image plane 上的 2D 投影坐标（pixel）
- $z_i$：normalized depth，用 scene 的 max/min depth 归一化到 (0, 1) 区间
- $i \in \{1, \dots, 8\}$：8 个 vertex index

**关键设计 trade-off**：用 2D + 1D normalized depth 而非 full 3D 坐标，因为 VLM 输出 text token，绝对 3D 坐标数值范围太大不好 token化。normalized depth 解决了这个问题，但牺牲了绝对尺度信息——这也是为什么 deployment 时需要 depth camera 反归一化（见 Appendix B）。

### 2.2 A (Axis)
Articulation axis，2 个 3D 点表示：
$$A = \{(\alpha_i, \beta_i, \gamma_i)\}_{i=1, 2}$$

- $\alpha_i, \beta_i$：axis 端点的 2D 投影
- $\gamma_i$：normalized depth
- $i \in \{1, 2\}$：两个端点

对 revolute joint：A 就是 URDF 里的 rotation axis。
对 prismatic joint：A 方向取 URDF 的 prismatic direction，但强制穿过 link 的 3D center（这是为了 bounding box 的对齐）。

### 2.3 S (Semantic label)
包含三件事：
1. Articulation type: `prismatic` | `revolute`
2. Link name（e.g., "handle", "lid", "door"）
3. Action type（e.g., "open", "close", "rotate"）

**和 GaPartNet 的对比**：GaPartNet 用 9 类 prototype（slider hinge, fixed hinge, round handle, etc.），A3VLM 简化到 2 类。Simplification 的代价是 lose 了一些 fine-grained 语义，但好处是：
- VLM 的输出 vocabulary 更小，token 准确率更高
- Action primitive 设计只需要覆盖 2 种 joint type

---

## 3. Instruction-Following Dataset 构建

这是 paper 最 tricky 的工程部分。用 PartNet-Mobility（2000+ objects, 46 categories, URDF format），每 object 渲染 40 张图（随机 camera / lighting / joint value），然后用 ControlNet 增强。

### 3.1 四类 sub-tasks（Table 1）

| Capability | Task | Template | Count |
|---|---|---|---|
| Partial Object Understanding | Detection | "Detect all manipulable object parts..." | 43K |
| Partial Object Understanding | REC-Link | "Provide 3D BBox of region described by: lid" | 178K |
| Articulation Understanding | REG-Joint | "Provide joint type and 3D axis for part: BBox B" | 18K |
| Action Grounding | REC-Action | "Execute: Open the storage" | 15K |

总计约 254K 样本。REC = Referring Expression Comprehension（text → bbox），REG = Referring Expression Generation（bbox → text）。这是借用 VQA 社区的标准任务模板，让 VLM 训练 pipeline 不需要大改。

**为什么拆成 4 个 sub-task 而不是 end-to-end**：单一 inference 让 VLM 同时输出 B、A、S 三个量，cross-entropy loss 在长输出序列上信号被稀释。拆开后每个 task 让 VLM 专注一个 concept，loss landscape 更 friendly。Figure 1 的 sequential inference 就是这样用的：先问 actionable part，再问 articulation。

### 3.2 Affordance 标注的 trick
PartNet-Mobility **不提供 affordance**！作者用 GPT-4 从 RoboAgent 的 skill library 里选 skill，prompt 见 Appendix A。这是个聪明的 workaround——LLM 作 semantic bridge 把 URDF 的 link name 和 robot skill 关联起来。

### 3.3 Data Augmentation: ControlNet + ChatGPT
PartNet-Mobility 的图都是 plain gray（Figure 3），sim2real gap 极大。augmentation 策略：
- **主信号**：depth map 作 ControlNet condition（geometric + semantic 信息都保留）
- **fallback**：对 depth variance 小的 object，用 semantic segmentation
- **prompt diversity**：ChatGPT 生成多样化 texture description（Listing 2）

Figure 9 对比 raw 和 augmented 图，效果显著。这其实是 sim2real 的一个 standard trick，但这里特别值得注意——**augmentation 必须保持 articulation 几何不变**，所以 depth condition 比直接 text-to-image 更安全（不会改变 link 的形状）。

参考：
- PartNet-Mobility: https://sapien.ucsd.edu/
- ControlNet: https://arxiv.org/abs/2302.05543
- RoboAgent: https://arxiv.org/abs/2309.01918

---

## 4. Architecture (Figure 4)

基于 SPHINX-X，LLaMA2 作 language backbone。

### 4.1 Visual encoder 设计
SPHINX 的 "any resolution" 思路：input image 切成 sub-image，分别过 visual encoder。

A3VLM 用了**三个** visual encoder，channel-wise concat：
1. **CLIP**：semantic features，pretrained on 400M image-text pairs
2. **DINOv2**：local semantic features，self-supervised，对 patch-level detail 强
3. **Q-Former**（来自 Flamingo）：global feature summarization，把任意数量 image token 压缩成固定数量

为什么三个一起用：
- CLIP 提供 language-aligned semantic
- DINOv2 提供 dense local feature（detection 任务需要）
- Q-Former 提供 global context（reasoning 任务需要）

三个 features channel-wise concat 后过 projection layer 对齐到 LLM 的 token dimension。

### 4.2 Bounding box 数值表示
Bbox values 归一化到 (0, 1)，precision 到 2 位小数。这是 VLM 输出 numeric 的 standard practice（Shikra 也是这么做的）。2 位小数意味着 100x100 的 grid 精度，对 224x224 或更大的 image 来说够用但不够 fine——这是个 limitation。

### 4.3 训练
- 8x A100 80GB
- 3 epochs, ~24 小时
- batch size 4, lr $2 \times 10^{-5}$
- Visual encoders frozen，只 fine-tune projection layers 和 LLM
- Two-stage: 先用 "This is a [OBJ]" 这种 naive caption 训 projection layer 对齐 visual feature，再训全模型

参考：
- SPHINX-X: https://arxiv.org/abs/2402.05935
- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193
- Flamingo (Q-Former 原始): https://arxiv.org/abs/2204.14198
- Shikra: https://arxiv.org/abs/2306.15195

---

## 5. Action Primitives (Figure 5)

A3VLM 输出 (B, A, S) 后，翻译成 robot trajectory。定义 3 种 primitive：

### 5.1 选择规则
```
if joint_type == "prismatic": use Slide
elif joint_type == "revolute":
    if link_name in {"bottle_cap", "scroll_button"}: use Scroll
    else: use Rotate
```

### 5.2 Grasp point 选择
- Scroll：grasp pose 必须 overlap with axis A（保证旋转轴对齐）
- 其他：在 B 内随机选 grasp pose 作 contact point C

### 5.3 Trajectory 生成
- Slide：沿 axis A 方向 linear motion
- Rotate：绕 axis A 的 circular arc
- Scroll：沿 axis A 的 helical motion（旋转 + 平移组合）

**这里有个重要 detail**：A3VLM 不假设 gripper 类型，需要外部 grasp proposer（GPG / GraspNet / manual）。Paper 里 real-world experiment 是 manual grasp pose list，简化实验。这是 honest 的写法，但也暴露了 limitation——end-to-end manipulation pipeline 还需要 grasp detection 模块。

参考：
- GraspNet: https://arxiv.org/abs/2003.08521
- GPG: https://arxiv.org/abs/1603.01542

---

## 6. Experiments

### 6.1 Simulation benchmark (Table 2)

Benchmark 来自 ManipLLM，用 Sapien simulator + PartNet-Mobility + Franka flying gripper with suction。

**Success criterion**：
$$d > \sigma, \quad \sigma = 0.01^2$$

$d$ 是被 manipulate 的 part 的位移，$\sigma$ 是 threshold。这里 $\sigma = 0.01^2$ 我怀疑是 paper 排版问题，实际应该是 $\sigma = 0.01$（ManipLLM 原文里也是 0.01）。

**结果**（Table 2 关键数字）：

| Method | Training AVG (20 cat) | Testing AVG (10 cat) |
|---|---|---|
| Where2Act | 0.26 | 0.21 |
| UMPNet | 0.35 | 0.28 |
| FlowBot3D | 0.37 | 0.32 |
| Implicit3D | 0.46 | 0.41 |
| ManipLLM | 0.59 | 0.54 |
| **A3VLM** | **0.76** | **0.72** |

A3VLM 比 SOTA ManipLLM 高 17% / 18% absolute。值得注意：
- Testing categories 上 generalization 几乎无损（0.76 → 0.72，掉 4 pts），而 ManipLLM 掉 5 pts（0.59 → 0.54）
- 有些 category A3VLM 远超 ManipLLM：e.g., category 11 (1.00 vs 0.53)、category 14 (0.97 vs 0.71)、category 17 (0.91 vs 0.44 in testing)
- 个别 category A3VLM 反而差：category 8 (0.35 vs 0.61)、category 16 (0.40 vs 0.64)

**为什么有些 category 差**：我推测是 action primitive 设计不全。比如 category 16 在 testing set，可能涉及某种特殊 articulation（multi-axis 或者 continuous rotation），A3VLM 的 2 类 prototype + 3 种 primitive 覆盖不到。

### 6.2 Real-world inference (Figure 6)
20 个 real-world object，包括 reflective（microwave, pot）和 transparent（coke bottle）surface。A3VLM 用 single RGB image 就能正确 detect movable parts 和 articulation。

这是 paper 的一个 **关键 selling point**：point cloud 方法在 reflective/transparent 上 depth 噪声大，A3VLM 用 RGB 完全绕开这个问题。

### 6.3 Real-world manipulation (Table 3, Figure 7)

5 个 object，每个 5 trials：
- 4/5 objects: 5/5 success
- 1 object: 4/5 success

Kuka + Robotiq 3-finger gripper + RealSense D415。

### 6.4 Ablation: RGB vs Depth (Table 4)

| Method | Training AVG | Testing AVG |
|---|---|---|
| A3VLM (RGB) | 0.76 | **0.72** |
| A3VLM-depth | 0.76 | 0.70 |

Training set 上 depth 略好（个别 category 高很多），但 **testing set 上 RGB 明显更好**。这印证了 sim2real gap 在 depth 上更大——simulation depth 是 perfect geometry，real depth 充满 noise，distribution shift 严重。RGB 的 appearance 虽然也有 gap，但 ControlNet augmentation 缓解了。

参考：
- ManipLLM benchmark: https://arxiv.org/abs/2312.16217
- Sapien: https://arxiv.org/abs/2003.08515

---

## 7. Appendix D 的诚实记录：失败的 modality 探索

Paper 在 Appendix D 记录了 RGB-D 和 Point Cloud 输入的失败尝试，这种 honest reporting 很难得。

### 7.1 RGB-D 失败原因
- Visual foundation models 预训练在 RGB 上，depth 没 texture，feature extraction 失败
- Token sequence 太长，LLM 处理不了

### 7.2 Point Cloud 失败原因
- 用 PointBert + RECON encoder
- Point cloud 没 visual texture，partial-level detection 困难
- Point cloud model 参数量和训练数据都远小于 visual foundation model

这其实是整个 3D community 的痛点：3D foundation model 还没成熟到 RGB 同等水平。ShapeLLM、PointLLM 在 captioning 上还行，但 fine-grained detection 还不够。

参考：
- ShapeLLM: https://arxiv.org/abs/2402.17766
- PointLLM: https://arxiv.org/abs/2308.16911
- PointBert: https://arxiv.org/abs/2111.14819
- RECON: https://arxiv.org/abs/2202.01113

---

## 8. 我的批判性分析

### 8.1 Strengths
1. **Object-centric representation** 这个 framing 很 clean，decoupling robot 和 object 是正确方向
2. **RGB-only** 的选择在 ablation 里被验证，且工程上简化了 real deployment
3. **4 sub-tasks** 拆分让 VLM 训练更 stable，符合 "divide and conquer" 的 LLM 训练哲学
4. **Honest ablation**——记录失败的 modality 探索

### 8.2 Limitations
1. **Bounding box 精度限制**：2 位小数归一化坐标对精细 manipulation 不够（比如 USB 插入这种 mm 级精度任务）
2. **Action primitive 只有 3 种**：现实世界 articulation 远不止 slide/rotate/scroll，比如 multi-DOF joint、deformable object
3. **依赖外部 grasp proposer**：paper 里 manual grasp pose，实际 deployment 需要接 GraspNet 等，整个 pipeline 复杂度被隐藏了
4. **Sequential inference 慢**：Figure 1 显示需要 2 步 inference（先 REC-Action 再 REG-Joint），real-time control 受限
5. **Sim2Real 的 SAM workaround**：用 SAM 去 background，意味着 real deployment 假设 object 是 salient foreground，复杂场景可能失效
6. **2D + normalized depth 的表示**：丢失了绝对尺度，需要 depth camera 反归一化，某种程度上 "RGB-only" 这个 claim 有点弱——实际还是需要 depth 信息做 deployment

### 8.3 与你 (Karpathy) 的视角的联系
你在 "Software 2.0" 和近期 LLM 训练的讨论里强调过：**好的 representation 比好的 policy 更重要**。A3VLM 本质上就是在学一个 object 的 articulation representation，然后让 representation 足够 structured 以至于简单 primitive 就能 decode 成 action。这跟你的 philosophy 一致。

但也可以反问：为什么 articulation representation 一定要用 VLM 的 text output 来 encode？bouding box 用 token 化的 2 位小数其实是个 hack。如果你设计一个 **continuous output head**（比如 VLM backbone + regression head 输出 bbox 和 axis），可能更精确。不过这样就失去了 LLM 的 reasoning 能力和 in-context learning 能力。这是 LLM-based robotics 的根本 tension：**text interface 既是最强接口（compositional, interpretable, transferable）也是最大瓶颈（discrete, low precision）**。

参考：
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35

---

## 9. 未来方向联想

1. **3D foundation model 成熟后**：如果 ShapeLLM 这类 3D MLLM 达到 RGB MLLM 水平，A3VLM 可以原生处理 point cloud，精度会大幅提升
2. **Diffusion policy 接口**：A3VLM 的 (B, A, S) 可以作 diffusion policy 的 conditioning，结合 fine-grained trajectory generation
3. **Video input**：articulation 的 motion 信息在 video 里更明显，A3VLM 目前 single image，扩展到 video 应该有提升
4. **Tactile feedback**：paper 完全 open-loop，加 tactile 作 failure detection 会大幅提升 real-world robustness
5. **Articulation representation 的信息论分析**：B + A + S 三元组的 information content 是否 sufficient？比如 friction coefficient、joint limit 这些 A3VLM 没建模
6. **VLM 直接 output action token**：把 action primitive 也 encode 进 LLM vocabulary，end-to-end 训练，省去 external grasp proposer

---

## 10. 总结

A3VLM 是一个 **well-engineered, honestly-reported** 的工作。核心贡献是把 robotics VLM 的 output 从 robot-centric action 转到 object-centric articulation representation，decoupling 了 robot 和 object。Trade-off 是损失了 end-to-end 的简洁性，换来了 sample efficiency 和 robot-agnostic transferability。

数字上：SAPIEN benchmark 比 SOTA ManipLLM 提升 17% absolute success rate，real-world 5 object 上 24/25 success。RGB-only 设计在 testing generalization 上反直觉地优于 depth input。

我认为最值得 follow 的方向是 **continuous output head** 和 **video input**——前者解决 precision bottleneck，后者解决 single-frame articulation ambiguity。

参考资源汇总：
- Paper repo: https://github.com/changhaonan/A3VLM
- SPHINX: https://github.com/Alpha-VLLM/LLaMA2-Accessory
- PartNet-Mobility: https://sapien.ucsd.edu/
- Sapien simulator: https://github.com/haosulab/SAPIEN
- ManipLLM repo: https://github.com/BLV-Lab/ManipLLM
