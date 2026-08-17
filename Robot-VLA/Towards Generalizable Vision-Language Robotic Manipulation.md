---
source_pdf: Towards Generalizable Vision-Language Robotic Manipulation.pdf
paper_sha256: c5c089286ee9ed759b292dda5439f61415530b8e8ce8aca9663891abf5a057de
processed_at: '2026-08-12T17:20:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej, 如果我们抛开论文里那些学术黑话，用最直白的“人话”来拆解这篇 paper，其实它讲的就是**怎么教机器人干家务，并且遇到没见过的活儿也能随机应变**。我来给你梳理一下这里的 intuition。

### 1. 机器人的痛点：背题库 vs 嘴强王者

目前教机器人干活（vision-language robotic manipulation）有两条极端的路线，都有致命缺陷：

- **End-to-end policy（比如 RVT-2, 3D Diffuser Actor）**：就像给学生背题库。你给它看 1000 次怎么抓红杯子，它抓红杯子贼溜。但你给它换个没见过的紫杯子，或者让它去开微波炉，它就懵了。因为它把视觉特征和动作死死绑在了一起，泛化能力极差。
- **Foundation model（比如 VoxPoser, SayCan）**：就像接了个 ChatGPT 的大脑。你让它开微波炉，它知道步骤，但它的“手”太笨。它能用 LLM 算出微波炉把手在 3D 空间的哪个坐标，然后调用传统的运动规划算法过去，但稍微复杂点需要精细操作的动作（比如拧螺丝、开合页）它就抓瞎了。

这篇 paper 的核心 intuition 极其简单：**那就把这两拨人组成一个团队呗**。让 LLM 当大脑规划，让 VLM 当眼睛找东西，让专门训练的 3D 模型当手去执行。这就是 3D-LOTUS++。

### 2. GemBench：怎么考机器人才公平？

作者吐槽以前的 benchmark 太水了。训练集是 18 个任务，测试集还是这 18 个，顶多换个光照。这测不出真实能力。于是他们搞了个 GemBench，把考试分成了 4 个难度等级，这个 level 的设计直觉非常清晰：

- **Level 1 (Novel Placements)**：杯子没见过，但放偏了点。这测的是空间泛化。
- **Level 2 (Novel Rigid Objects)**：练过抓方块，现在让你抓个没见过的星形玩具。这测的是形状泛化。
- **Level 3 (Novel Articulated Objects)**：练过开 3 层抽屉，现在让你开 4 层抽屉；练过开笔记本，现在让你开微波炉。这测的是对铰链、滑轨这些关节结构的泛化。
- **Level 4 (Long-Horizon)**：给你一串指令“打开抽屉，把红方块放进去，再把绿圆柱放进去，关上抽屉”。这测的是长链条组合能力。

实验结果非常打脸：所有 SOTA 的 end-to-end 模型，在 Level 4 的成功率全是 **0%**。只要中间错一步，后面全崩。这就引出了作者的 modular 解法。

### 3. 3D-LOTUS：一只极其高效的“机器手”

在搞复杂的 3D-LOTUS++ 之前，作者先做了一个纯 end-to-end 的底座模型叫 3D-LOTUS。它的直觉是：**预测动作不该是回归，而该是分类**。

以前的模型怎么预测机械手下一个去哪？用 MSE regression 去回归一个精确的 3D 坐标。问题是，抓杯子把手你可以从左边抓也可以从右边抓，动作分布是多峰的，回归一算平均，手就怼到杯子中间去了。

3D-LOTUS 的做法：把 3D 空间在每一个点云点周围切成一堆 1cm 的小方块，然后做分类——**“手下一个落点最有可能在这堆小方块的哪一个里？”** 
直觉上，这就相当于让场景里所有的物体点云一起“投票”，投票选出手该去哪。这不仅让多峰分布的问题解决了（选票最高的那个坑就行），而且计算极快。单卡 A100 训练 11 小时搞定，比以前的 SOTA 快了 30 倍，而且成功率还最高。

### 4. 3D-LOTUS++：流水线开工

有了厉害的“手”之后，怎么泛化？3D-LOTUS++ 把一个端到端网络硬拆成了三个流水线模块：

1. **大脑**：LLM (LLaMA3-8B)。你输入“开抽屉”，它输出伪代码步骤：`grasp("抽屉把手") -> move_grasped_object("往外拉") -> release()`。
2. **眼睛**：VLM (OWLv2 + SAM)。LLM 说要抓“抽屉把手”，眼睛就去 RGB 图里框出所有东西，算算哪个最像“抽屉把手”，用 SAM 抠出点云，交给手。
3. **手**：改装版 3D-LOTUS。拿到目标点云后，直接生成一连串动作轨迹去执行。

**这里有一个极绝的细节改动**：3D-LOTUS 在处理点云特征时，本来输入的是 RGB 颜色。但在 3D-LOTUS++ 里，它改成了输入“点的角色标签”（这是目标物体，这是障碍物，这是机械臂）。
为什么？因为你换了个没见过的紫杯子，用 RGB 它可能不认识。但如果你告诉它“不管这玩意啥颜色，这是你要抓的目标”，它就能把注意力全放在几何形状上，这就实现了对颜色和纹理的绝对泛化。

### 5. 实验里的 Insight：哪里在拖后腿？

作者做了一个极其漂亮的消融实验，人工把 LLM 的规划和 VLM 的识别全换成 Ground Truth（完美答案），看看系统哪里在掉链子。

- **在 Level 1 & 2（短任务）**：给完美的物体识别标签，成功率飙升。说明**瓶颈在眼睛（VLM grounding）**。VLM 经常认错东西，比如把金枪鱼罐头认成汤罐头，一认错全盘皆输。
- **在 Level 4（长任务）**：就算给完美的规划和完美的物体位置，成功率也只有 31.5%。说明**瓶颈在手本身**。
为什么？因为 3D-LOTUS 是用单步演示训练的。在长任务中，你做完第一步，机械臂的姿势就偏离了训练数据的分布，到了第二步，机械臂处于一个它从没见过的奇怪姿态，它就不会动了。这就像打台球，你规划了三杆，但第一杆擦边了一点，第二杆的球位就全乱了，后面自然打不进去。缺乏 closed-loop replanning 是死穴。

### 6. 给 Karpathy 的 Core Takeaway

这篇 paper 最大的贡献在于 methodological disentanglement：
**不要指望一个巨大的 transformer 把 perception、planning、control 全包揽了。**

它们的时间尺度和误差结构完全不同：
- Foundation model 适合处理 1Hz 的 semantic reasoning，数据来自整个互联网。
- 3D Policy 适合处理 100Hz 的 spatial geometry，数据来自少量 demo。

强行把它们揉在一起，不仅训练成本极高（几十上百个 GPU days），而且面对长链条任务极易崩溃。把它拆成 VLM 当眼睛、LLM 当大脑、3D-LOTUS 当手的 modular pipeline，虽然看起来不那么“end-to-end elegant”，但它在现实新任务里活下来了。特别是真机实验里，纯 end-to-end 遇到没见过的颜色物体成功率 0%，而 modular 流水线有 79%。

未来的方向太明确了：VLM 需要闭环重新识别，机械手需要能处理偏离分布的中间状态，Level 4 那个 17.4% 的成绩还有极大的提升空间。

---
**Reference links for further digging:**
*   Paper project page (含代码和模型): [https://www.di.ens.fr/willow/research/gembench/](https://www.di.ens.fr/willow/research/gembench/)
*   3D-LOTUS 用的 backbone PTv3: [https://arxiv.org/abs/2312.10071](https://arxiv.org/abs/2312.10071)
*   眼睛模块用的 OWLv2: [https://arxiv.org/abs/2306.09683](https://arxiv.org/abs/2306.09683)
*   眼睛模块用的 SAM: [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
*   RLBench simulator: [https://arxiv.org/abs/1909.12271](https://arxiv.org/abs/1909.12271)
*   竞品 benchmark Colosseum: [https://arxiv.org/abs/2402.08191](https://arxiv.org/abs/2402.08191)
*   竞品 benchmark VIMA-Bench: [https://arxiv.org/abs/2210.03094](https://arxiv.org/abs/2210.03094)
*   竞品 end-to-end SOTA 3D Diffuser Actor: [https://3d-diffuser-actor.github.io/](https://3d-diffuser-actor.github.io/)
*   竞品 RVT-2: [https://robotic-view-transformer.github.io/](https://robotic-view-transformer.github.io/)
*   纯 LLM 规划方法 VoxPoser: [https://voxposer.github.io/](https://voxposer.github.io/)
*   LLM 规划 SayCan: [https://say-can.github.io/](https://say-can.github.io/)

---

# GemBench与3D-LOTUS/3D-LOTUS++深度解析

Hey Andrej, 这篇paper来自Inria WILLOW团队（Shizhe Chen, Cordelia Schmid等），核心是想解决vision-language robotic manipulation中一个根本性的tension：**end-to-end policies在seen tasks上表现优异但在novel tasks上崩溃，而foundation models有generalization但缺乏precise control**。让我深入拆解。

## 1. Motivation与Problem Setting

### 1.1 Field的现状

当前robotic manipulation面临一个尴尬的trade-off：

- **End-to-end policies** (如RVT-2, 3D Diffuser Actor, Act3D)：通过imitation learning从demonstrations学习，在训练分布内表现极好，但generalization到新objects、新articulated parts、long-horizon组合时崩溃
- **Foundation model approaches** (如VoxPoser, Code as Policies)：利用LLM/VLM的commonsense reasoning实现zero-shot generalization，但缺乏precise low-level control，struggle with需要精细操作的任务如screw bulb

**这个paper的核心hypothesis**：能否将这两条路径的优势disentangle并组合？具体来说，让foundation models负责它们擅长的（task decomposition, object grounding），让learned 3D policy负责它擅长的（precise motion control）。

### 1.2 Benchmarking的缺失

作者指出当前benchmark的key limitation（Table I）：

| Benchmark | 关键问题 |
|-----------|----------|
| RLBench-18Task | 训练和测试用同一组tasks，不测generalization |
| VIMA-Bench | 只有pick-and-place，suction gripper |
| Colosseum | 评测environment perturbations（光照、相机角度），而非新任务 |
| Calvin | Long-horizon但只有34 tasks |

没有一个benchmark能**同时**评测：multi-skills + articulated objects + 多个generalization levels。

## 2. GemBench Benchmark设计

### 2.1 整体结构

GemBench建立在RLBench simulator之上，包含：

- **Training set**: 16 tasks, 31 variations
- **Test set**: 44 tasks, 92 variations

核心是**四个progressive generalization levels**，难度递增，这是一个非常clean的设计：

```
Level 1 (Novel Placements) → Level 2 (Novel Rigid Objects) 
    → Level 3 (Novel Articulated Objects) → Level 4 (Long-Horizon)
```

### 2.2 七个Action Primitives

Training set覆盖七个核心action primitives：

1. **press** (e.g., push button)
2. **pick** (e.g., pick up cup, pick and lift)
3. **push** (e.g., slide block, reach and drag)
4. **screw** (e.g., screw light bulb)
5. **close** (e.g., close fridge, close laptop lid)
6. **open** (e.g., open drawer, open door)
7. **stack/put** (e.g., stack blocks, put groceries)

这比VIMA-Bench只有pick-and-place要丰富得多，特别是包含了articulated objects操作和screw这种rotational action。

### 2.3 四个Generalization Levels的详细设计

**Level 1 - Novel Placements** (16 tasks, 31 variations)
- 同training tasks，但object placements重新采样
- 加入了不同colors的distractor objects
- 测试spatial generalization within distribution

**Level 2 - Novel Rigid Objects** (15 tasks, 28 variations)
两个sub-categories：
- **Novel object-color compositions** (20个): 训练时只见过yellow button + rose bulb，测试时遇到rose button
- **Novel object shapes** (8个): 训练时pick cube，测试时pick cylinder/star/moon/toy

**Level 3 - Novel Articulated Objects** (18 tasks, 21 variations)
三个sub-categories：
- **Novel action-part compositions** (8个): 训练open bottom drawer + put in middle shelf，测试open middle drawer
- **Novel instances** (11个): 训练3-drawer unit，测试4-drawer unit
- **Novel categories** (2个): 训练close laptop lid，测试close grill lid

**Level 4 - Novel Long-Horizon** (6 tasks, 12 variations)
- 组合多个training中学过的actions
- 例如"put items in drawer"：open drawer → pick cube → pick cylinder → pick moon → place them in order

这个level design的妙处在于：**每个level的failure mode可以被isolated分析**。如果Level 2崩了说明object grounding有问题；如果Level 3崩了说明articulated object理解有问题；如果Level 4崩了说明planning或compositional reasoning有问题。

## 3. 3D-LOTUS Policy Architecture

### 3.1 整体架构

3D-LOTUS = Point Cloud Preprocessing + Language-Conditioned PTv3 + Classification-based Action Head

### 3.2 Point Cloud Preprocessing

**Input**: K=4 cameras (front, left shoulder, right shoulder, wrist)的aligned RGB-D images，resolution 256×256

**Pipeline**:
1. 用已知的camera intrinsics/extrinsics将multi-view RGB-D投影到world coordinates的unified point cloud
2. **Voxel downsampling**: 1 point per 1cm³ voxel（这个resolution是关键设计选择，balance了精度和computational cost）
3. **Workspace filtering**: 排除workspace外的points
4. **Robot arm filtering**: 用CAD model + joint poses计算每个link的3D bounding box，排除这些box内的points（Figure 5展示了这个效果）

最终point cloud只包含objects和robot gripper，**显著减少points数量，提高speed而不损失performance**。

每个point $\nu_i \in V$ 包含：
- $\nu_i^p$: XYZ coordinates
- $\nu_i^o$: additional feature（RGB color + relative height to table）

### 3.3 Language-Conditioned Point Cloud Transformer

**Backbone**: Point Transformer V3 (PTv3)
- U-Net architecture with downsampling/upsampling blocks
- 每个block包含transformer layers
- 5个downsampling-upsampling blocks，hidden sizes: 64, 128, 256, 512, 768

**Language conditioning的两种variants**：

**Variant 1: Adaptive Normalization**
- Language instruction L通过CLIP text encoder得到word embeddings $(w_1, \cdots, w_L)$
- Weighted averaging得到global language embedding $\bar{w}$
- 用$\bar{w}$ regress出PTv3每个normalization layer的dimension-wise scale和shift参数
- 类似FiLM conditioning

**Variant 2: Cross-Attention**
- 在PTv3每个self-attention layer后加一个cross-attention layer
- 每个point attend to整个word embedding sequence $(w_1, \cdots, w_L)$
- Computation cost更高但expressiveness更强

**Ablation结果**（Table IV）显示Cross-Attention在所有levels都优于AdaptiveNorm，尤其是Level 1: 94.3 vs 90.8。这说明**token-level language grounding比global conditioning更有效**，符合预期因为manipulation需要细粒度的object-action binding。

### 3.4 Action Prediction - 关键创新

这是3D-LOTUS最重要的设计决策之一。作者argue for **classification-based action prediction**而非传统的regression。

**传统方法的problems**：
- Regression [17, 2, 8]: 直接回归position，收敛慢，对precision要求高的任务表现差
- Position classification over whole 3D workspace [18, 4]: 计算量太大（PerAct用100×100×100 voxel grid）

**3D-LOTUS的approach**: Per-point, per-axis classification

**Position Prediction**:

对每个point $\nu_i$ 和每个axis $k \in \{X, Y, Z\}$，定义sequential bins centered at $\nu_i^p$:
- $j \in [-m, m]$，其中$m = 15$，所以每个point有$2m+1 = 31$个bins
- Bin size $b = 1$ cm
- Bin $\nu_{i,k,j}$的position along k-axis: $\nu_{i,k}^p + b \times j$

**Training target**（公式1）：
$$\hat{p}_{t,k,i} = \begin{cases} 0, & \text{if } ||b_{t,k,i}^p - a_{t,k}^p||_2^2 > 0.01 \text{ or } b_{t,k,i}^p \in \mathbb{B} \\ 1, & ||b_{t,k,i}^p - a_{t,k}^p||_2^2, \text{otherwise} \end{cases}$$

变量解释：
- $\hat{p}_{t,k,i}$: timestep t, axis k, bin i的score
- $b_{t,k,i}^p$: bin i的position along axis k at timestep t
- $a_{t,k}^p$: groundtruth gripper position along axis k at timestep t
- $\mathbb{B}$: 属于robot arm和gripper的points集合
- $||\cdot||_2^2$: Euclidean distance的平方
- 阈值0.01: 大约1cm以内算"close"

**关键设计**：$\mathbb{B}$中points的score设为0，确保gripper position只基于objects预测，不被robot body干扰。然后通过L1 norm normalize得到probability distribution。

**Inference**: 对每个axis，concatenate所有points的bins形成global heatmap，选probability最高的bin确定该axis position。

**Rotation Prediction**:
- Discretize Euler angles into 72 bins (bin size 5°)
- One-hot label per axis
- 用max pooling over all points的final embedding $\nu_i^e$来预测

**Open State Prediction**:
- Binary classification (gripper open/closed)
- 同样用max pooling

**Loss**: 所有heads都用cross entropy loss

### 3.5 为什么Classification更好？

这个设计有几个深刻的好处：

1. **Multi-modal action distribution**: Regression假设unimodal输出，但manipulation中同一个state可能有多个valid next actions。Classification天然支持multi-modal distribution。

2. **Sharp predictions for precise tasks**: 对screw bulb这种需要精确到mm级的任务，regression容易产生模糊预测，而classification可以输出sharp argmax。

3. **Per-point bins的efficiency**: 相比PerAct的100³ global voxel grid，per-point bins只需要n×31×3个分类logits，n是points数量（通常几千），远小于10^6。

4. **Local geometry reasoning**: 每个point预测自己附近的bins，自然capture了local geometry information。

### 3.6 Training Details

- Batch size: 8
- Learning rate: 1e-4, linear decay
- 150k iterations
- Training time: ~11 hours on single A100 GPU
- Validation: 20 episodes per task variation on Level 1, every 10k iterations

**与baseline的训练efficiency对比**（Table II）：
- 3D-LOTUS: 2.23 V100 GPU days
- RVT-2: 6.6 days
- 3D Diffuser Actor: 67.6 days
- PerAct: 128.0 days

3D-LOTUS比3D Diffuser Actor快30倍，这是个impressive的efficiency gain。

## 4. 3D-LOTUS++ - Modular Framework

### 4.1 核心motivation

3D-LOTUS在seen tasks上SOTA，但在Table III的Level 2-4上急剧下降：
- L1: 94.3%, L2: 49.9%, L3: 38.1%, L4: 0.3%

为什么end-to-end policy泛化不好？作者的诊断是：**end-to-end policy把task planning、object grounding、motion control揉在一起，错误无法diagnose，且任何一个component的failure都会drag down整体**。

3D-LOTUS++的解法：**disentangle三个功能模块**，让每个模块用最适合的技术：
- LLM负责task planning（需要commonsense和compositional reasoning）
- VLM负责object grounding（需要open-vocabulary recognition）
- 3D-LOTUS负责motion control（需要precise spatial reasoning）

### 4.2 Task Planning with LLM

**Model**: LLaMa3-8B

**6 Action Primitives**:
1. `grasp(object)`: 抓取object
2. `move_grasped_object(target)`: 移动抓取的object到target
3. `push_down(object)`: 向下推object
4. `push_forward(object, target)`: 向前推object到target
5. `release()`: 释放gripper
6. `rotate_grasped_object()`: 旋转抓取的object

**In-context learning setup**:
- 用SentenceBERT计算query instruction与所有training instructions的sentence embedding相似度
- 选top-20最相似的training examples作为in-context examples
- Figure 6-7展示了prompt format和examples

**Example plan** for "open the door":
```
door_handle = grasp(object="door handle")
door_handle = rotate_grasped_object()
door_handle = push_forward(object=door_handle)
```

**Limitation**: LLM没有visual input，所以对"take shoes out of box"这种任务，无法判断box是open还是closed，可能产生wrong plans。

### 4.3 Object Grounding with VLMs

这是一个multi-stage pipeline：

**Step 1: Open-vocabulary detection**
- 用**OWLv2**检测每个RGB image中所有objectiveness score高的bounding boxes
- OWLv2同时生成每个bbox的semantic embedding，与CLIP text embedding aligned

**Step 2: Segmentation**
- 用**SAM** (Segment Anything Model)在每个bbox内分割出object mask

**Step 3: 3D point cloud extraction**
- 结合RGB-D image + segmentation mask，得到每个bbox的3D point cloud

**Step 4: Multi-camera merging**
- 比较不同camera观察到的同一object
- Merge条件：semantic embedding距离 < threshold AND point cloud距离 < threshold
- 得到object-centric representation: merged point cloud + averaged semantic embedding

**Step 5: Text-conditioned matching**
- 对plan中提到的object，用CLIP text encoder计算text embedding
- 与所有object semantic embeddings计算cosine similarity
- 选similarity最高的object作为match

**Articulated object的特殊处理**:
对于"bottom drawer" vs "top shelf"这种需要grounding到object的特定部分的任务，VLM只能detect整个drawer/safe，无法grounding到特定level。作者的workaround：让LLM预测target level的height range（Figure 8的prompt），结合VLM的overall object height，得到target的3D region。

### 4.4 Modified 3D-LOTUS for Motion Control

原始3D-LOTUS需要两处修改：

**Modification 1: Point feature变化**
原始3D-LOTUS用RGB color作为point feature。在3D-LOTUS++中，用object grounding的输出将points categorize为4 types：
- **goal object**: 要操作的object的points
- **goal target**: 操作target location的points
- **robot**: robot gripper的points
- **obstacle**: 其他所有points

用look-up table编码point label，作为新的point feature $\nu_i^o$。

**为什么这样设计？** 这让model focus on **geometry rather than textures**。对于novel textures的objects（Level 2的新shapes），RGB color信息可能misleading，但geometric role信息（这是要抓的，那是target）是task-invariant的。这是个很聪明的inductive bias。

**Modification 2: Trajectory prediction**
原始3D-LOTUS预测single action。3D-LOTUS++需要predict一个trajectory完成整个planned step。

实现方式：
- 引入timestep embedding $x_t$，$t \in \{1, \cdots, s\}$，$s=5$是最大trajectory length
- 将time embedding $x_t$与final point embedding $\nu_i^e$ concatenate
- Action prediction head在所有timesteps间shared
- 额外预测stop probability，indicate trajectory是否应该terminate at current step

## 5. Experimental Results深度分析

### 5.1 RLBench-18Task (Table II)

| Method | Avg. SR ↑ | Avg. Rank ↓ | Train time ↓ |
|--------|----------|-------------|---------------|
| PerAct | 49.4 | 6.2 | 128.0 |
| RVT | 62.9 | 4.4 | 8.0 |
| Act3D | 65.0 | 4.3 | 40.0 |
| RVT-2 | 81.4 | 2.4 | 6.6 |
| 3D Diffuser Actor | 81.3 | 2.3 | 67.6 |
| **3D-LOTUS** | **83.1±0.8** | **2.2** | **2.23** |

3D-LOTUS在success rate、rank、training time三个维度都达到SOTA。特别impressive的是training time只有2.23 GPU days，比3D Diffuser Actor快30倍。

Table IX的per-task breakdown显示3D-LOTUS在precision-demanding tasks上特别强：
- Insert Peg: 69.6% (vs RVT-2的40%)
- Stack Cups: 75.2% (vs RVT-2的69%)
- Place Cups: 40.8% (vs RVT-2的38%)

这验证了classification-based action prediction对precision tasks的优势。

### 5.2 GemBench Results (Table III) - 关键结果

| Method | L1 | L2 | L3 | L4 |
|--------|-----|-----|-----|-----|
| Hiveformer | 60.3 | 26.1 | 35.1 | 0.0 |
| PolarNet | 77.7 | 37.1 | 38.5 | 0.1 |
| 3D Diffuser Actor | 91.9 | 43.4 | 37.0 | 0.0 |
| RVT-2 | 89.1 | 51.0 | 36.0 | 0.0 |
| **3D-LOTUS** | **94.3** | 49.9 | 38.1 | 0.3 |
| **3D-LOTUS++** | 68.7 | **64.5** | **41.5** | **17.4** |

**几个关键observations**：

**Observation 1: Generalization cliff for end-to-end policies**
所有end-to-end methods（包括SOTA如RVT-2, 3D Diffuser Actor）在Level 4上都是0%。这不是它们"差"，而是long-horizon composition根本不是end-to-end imitation learning能学好的。

**Observation 2: 3D-LOTUS++在Level 1上反而比3D-LOTUS差**（68.7 vs 94.3）
这是个counter-intuitive但重要的结果。原因：3D-LOTUS++用zero-shot VLM grounding，对seen tasks中相似objects（如"tuna can" vs "soup can"）容易混淆。End-to-end policy在seen tasks上更好是因为它能learn task-specific features。

**Observation 3: Level 2和Level 3的gap不大**（64.5 vs 41.5）
这说明articulated objects的generalization比novel rigid objects难，但不像long-horizon那样有quantum jump。

**Observation 4: Level 4的17.4%仍然low**
即使是3D-LOTUS++，long-horizon composition仍然很难。Ablation study揭示了原因。

### 5.3 Ablation Studies

**3D-LOTUS components ablation** (Table IV):

| Action | Condition | L1 | L2 | L3 | L4 |
|--------|-----------|-----|-----|-----|-----|
| Regression | AdaptiveNorm | 83.3 | 29.3 | 34.5 | 0.0 |
| Classification | AdaptiveNorm | 90.8 | 47.8 | 37.9 | 0.0 |
| Classification | CrossAttn | **94.3** | **49.9** | **38.1** | **0.3** |

**Key takeaways**:
- Classification vs Regression: L1提升7.5%, L2提升18.5%。Classification对generalization帮助更大，可能是因为classification head学到的discrete distribution更transferable。
- CrossAttn vs AdaptiveNorm: 一致性提升，说明token-level language binding重要。

**3D-LOTUS++ modules ablation** (Table V):

| Task Planning | Object Grounding | L1 | L2 | L3 | L4 |
|---------------|------------------|-----|-----|-----|-----|
| GT | GT | **92.6** | **80.1** | **47.8** | **31.5** |
| GT | VLM | 71.0 | 66.3 | 46.0 | 19.4 |
| LLM | VLM | 68.7 | 64.5 | 41.5 | 17.4 |

这个ablation非常有informative：

**Level 1, 2: Object grounding是主要瓶颈**
- GT grounding vs VLM grounding: L1从92.6降到71.0（-21.6%），L2从80.1降到66.3（-13.8%）
- LLM vs GT planning: 只差2.3%和1.8%
- 说明VLM的zero-shot grounding在seen tasks和新rigid objects上还是不够robust

**Level 3, 4: 即使GT grounding也改善有限**
- Level 3: GT+GT 47.8%，但LLM+VLM只有41.5%
- Level 4: GT+GT 31.5%，但LLM+VLM只有17.4%
- 更重要的是，**即使给GT planning + GT grounding，Level 4也只有31.5%**

这说明**motion control policy本身在long-horizon上有limitation**。作者诊断：long-horizon tasks的initial robot configurations往往deviate substantially from training data，而3D-LOTUS是在single-step demonstrations上训练的，不能很好地handle中间states。

### 5.4 Real World Experiments

**Seen tasks** (Table VI):
- 3D-LOTUS: 8.1/10 avg
- PolarNet: 6.7/10 avg

**Unseen tasks** (Table VII):
- 3D-LOTUS: 0/10 on ALL unseen variations（完全失败！）
- 3D-LOTUS++: 7.9/10 avg

这个real world结果非常dramatic地验证了核心thesis：end-to-end policy在sim的seen tasks上表现好，但transfer到real world的novel variations时完全崩溃。而modular framework通过foundation models的generalization能力，实现了sim-to-real的novel task generalization。

注意unseen tasks包括：
- 新color combinations (stack red cup in yellow cup)
- 新objects (put lemon/banana in box)
- 新compositions (put tuna can then corn in box)

## 6. Critical Analysis与Intuition Building

### 6.1 为什么Modular Approach在Novel Tasks上更好？

这里有一个深层的intuition：**foundation models和learned policies的error structure不同**。

End-to-end policy的errors：
- 在distribution内：低，因为overfit到training demos
- Out of distribution：高且unpredictable，因为整个network都fail

Modular approach的errors：
- LLM planning: 大致robust，因为language understanding是foundation model的强项
- VLM grounding: 在常见objects上robust，在罕见objects上可能fail
- Motion control: 在geometrically similar configurations上robust

关键区别：modular approach的每个component的failure可以被reason about和partially compensate。例如VLM可能误grounding一个object，但motion control的geometry-based reasoning可能仍然产生合理trajectory。而end-to-end policy一旦input distribution shift，整个network的output就unpredictable。

### 6.2 为什么Level 4那么难？

从ablation看，即使GT planning + GT grounding，Level 4也只有31.5%。这揭示了几个深层问题：

**Problem 1: Distribution shift accumulation**
Long-horizon task中，每个step的robot state都depends on前面steps的execution。前面任何一步有小error，后面step的initial state就偏离training distribution。3D-LOTUS是在single-step demonstrations上训练的，没有见过这种"intermediate states"。

**Problem 2: Lack of closed-loop feedback**
3D-LOTUS++的plan是open-loop的：LLM生成plan后，按顺序执行，没有re-planning机制。如果中间step执行失败，后续steps仍按原plan执行，必然fail。

**Problem 3: Trajectory length limit**
Modified 3D-LOTUS的trajectory length s=5，对长任务来说每个primitive的trajectory可能不够。而且primitive之间的transition没有被explicitly model。

### 6.3 3D-LOTUS的Classification Head的Intuition

这是paper中最elegant的设计之一。让我深入讲讲intuition。

传统regression的问题：MSE loss在multi-modal target上会产生"averaging effect"。如果50%的demos去左边，50%去右边，regression会预测中间，但中间是invalid的。

3D-LOTUS的classification做法：对每个point $\nu_i$，预测其附近31个bins（±15cm范围）的概率。这等价于在每个point周围学习一个"local action distribution"。

为什么per-point？因为action的target通常near某个object surface point。通过per-point prediction，我们让每个object point"vote"它附近的target位置。最后通过concat所有points的predictions，得到global action distribution，这天然是multi-modal的。

公式1中的score设计也很巧妙：用Euclidean distance的平方作为score，threshold 0.01（即1cm²），这创造了一个soft label而非hard one-hot。这样训练时nearby bins也能得到gradient，更stable。

$\mathbb{B}$的masking：把robot arm和gripper的points score设为0，这是关键的inductive bias。如果不这么做，model可能学到"gripper现在在X位置，所以下一个gripper位置near X"，这是trivial shortcut。Masking掉robot points后，model必须从objects推理action，这才是真正的task reasoning。

### 6.4 Point Feature设计的Intuition

3D-LOTUS用RGB color + relative height。
3D-LOTUS++用point label (goal object / goal target / robot / obstacle)。

这个变化背后的intuition是：**对motion control来说，geometric role比appearance更重要**。

考虑Level 2的novel object shapes：训练时pick cube，测试时pick star。如果用RGB color，star的appearance可能induce错误的action pattern。但如果用point label（"这是goal object"），motion control只需要learn一个geometry-to-action mapping：如何approach一个object，如何move it to target，这些是shape-invariant的。

这是个经典的representation learning insight：**对downstream task有用的feature，而不是raw appearance**。

## 7. Limitations与Future Directions

### 7.1 Object Grounding Bottleneck

Table V显示VLM grounding是Level 1, 2的主要瓶颈。具体问题：
- OWLv2 + SAM的pipeline对similar objects（tuna can vs soup can）容易混淆
- Articulated object parts grounding依赖LLM的height prediction，不robust
- 没有closed-loop re-grounding，一旦initial grounding错误就无法recover

**Future direction**: 更强的open-vocabulary detection（如Grounding DINO）、part-level grounding models、或closed-loop visual grounding。

### 7.2 Motion Control的Long-Horizon Limitation

即使GT grounding，Level 4也只有31.5%。这表明motion control policy本身需要改进：
- 训练时应该包含multi-step trajectories，不只是single waypoints
- 需要closed-loop re-planning机制
- 可能需要hierarchical RL或trajectory optimization

### 7.3 LLM Planning without Vision

LLM task planning没有visual input，导致对state-dependent decisions无能为力。例如"take shoes out of box"需要先判断box是否closed。如果能integrate visual context（如GPT-4V），planning quality会提升。

### 7.4 Sim-to-Real Gap

虽然real world实验显示3D-LOTUS++能generalize到novel tasks，但只测了7 variations。更大规模的real world evaluation needed。

## 8. 与Related Work的Positioning

### 8.1 vs VoxPoser
VoxPoser用LLM构造3D voxel maps，再用经典motion planning。问题：只适合pick-and-place，对screw、articulated object操作困难。3D-LOTUS++通过learned motion control解决这个limitation。

### 8.2 vs SayCan / Code as Policies
SayCan用LLM + value functions of pretrained skills。CaP让LLM写code调用tools。两者都依赖predefined skills，限制applicability。3D-LOTUS++的motion control是learned的，更flexible。

### 8.3 vs RT-2 / OpenVLA
这些是end-to-end VLA models，直接从pixels+text到actions。它们有generalization但缺乏precise control。3D-LOTUS++通过modular design同时获得generalization（from foundation models）和precision（from learned 3D policy）。

### 8.4 vs 3D Diffuser Actor
3D Diffuser Actor用diffusion model在3D scene representations上做policy denoising，是end-to-end SOTA。但训练需要67.6 GPU days，且在novel tasks上generalization差。3D-LOTUS通过classification head + efficient PTv3 backbone，在2.23 GPU days达到更好performance。

## 9. Personal Takeaways

这篇paper有几个我认为非常重要的contributions：

1. **GemBench的level design**：四个progressive generalization levels是一个很好的benchmark design pattern，能isolate不同failure modes。这个idea可以推广到其他domains。

2. **Classification-based action prediction**：这个approach在manipulation中underused。3D-LOTUS的结果表明它比regression更sample-efficient和generalizable，特别是对precision tasks。

3. **Modular disentanglement的value**：3D-LOTUS++的核心贡献不是某个component的技术突破，而是证明了一个architectural principle——将planning、grounding、control disentangle后，可以让每个component用最适合的技术，并且enable更好的error analysis。

4. **Ablation study的depth**：Table V的GT vs learned ablation是这篇paper的亮点。它不仅show了method work，还show了**why it works and where it fails**。这种analysis对community非常有价值。

5. **Point feature的task-aware design**：从RGB color到point label的转变，是个小但深刻的design choice。它体现了一个重要原则：**representation should be aligned with downstream task**。

**Limitations方面**，我认为最关键的是：
- Level 4的17.4%说明modular approach对long-horizon仍然不够
- Object grounding的zero-shot limitation在real world deployment会是障碍
- 没有closed-loop re-planning，这在noisy real world中是必须的

**Future directions I'd explore**:
- 用closed-loop VLM grounding（每step重新ground）
- 用GPT-4V之类的multimodal LLM做visually-grounded planning
- 在3D-LOTUS训练中加入multi-step trajectories和intermediate states
- 探索diffusion-based motion control来handle multi-modal action distributions in long-horizon

---

**Reference links**:
- Project page: https://www.di.ens.fr/willow/research/gembench/
- PTv3 paper: https://arxiv.org/abs/2312.10071 (Wu et al., CVPR 2024)
- OWLv2: https://arxiv.org/abs/2306.09683
- SAM: https://arxiv.org/abs/2304.02643
- RLBench: https://arxiv.org/abs/1909.12271
- VIMA-Bench: https://arxiv.org/abs/2210.03094
- Colosseum: https://arxiv.org/abs/2402.08191
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- RVT-2: https://robotic-view-transformer.github.io/
- VoxPoser: https://voxposer.github.io/
- SayCan: https://say-can.github.io/
- Code as Policies: https://code-as-policies.github.io/
- CLIP: https://arxiv.org/abs/2103.00020
- LLaMA 3: https://ai.meta.com/blog/meta-llama-3/

这篇paper在benchmark设计、method architecture、experimental analysis三个维度都很solid，是vision-language robotic manipulation领域一个important contribution。特别是它的modular philosophy和ablation depth，对community的methodology有借鉴价值。
