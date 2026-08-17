---
source_pdf: ST4VLA.pdf
paper_sha256: 2c8887b96b7772519a29ea723f47b2da2e82be243a3cd3bc43f97609c09a0f7c
processed_at: '2026-08-12T10:24:59-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ST4VLA 人话版

## 一句话总结

给机器人装了个"空间感知脑"，让它别学动作时把看图识物的本事忘光。

## 问题出在哪

想象你雇了个博士毕业生当司机。博士脑子很好使，认识各种路标、地图看得明白、空间方位感超强。结果你让他去驾校学倒车入库，学了三天，好家伙，倒车是会了，但他连左右都分不清了，看红绿灯都得愣两秒。

这就是现在 VLA 领域的通病。

你拿一个预训练好的 VLM（比如 Qwen2.5-VL，这模型见过几亿张图，能指认物体、能定位、能描述空间关系），然后直接拿机器人动作数据 fine-tune 它，让它输出关节角度。训练完发现——动作倒是学会了，但 VLM 原来那些 spatial grounding 的本事全崩了。

论文里 Figure 3(a) 画得很清楚：训练 20k 步，RefCOCO-g 的 IoU 直接掉到 random 水平。相当于你那个博士倒车入库毕业了，但大脑前额叶被驾校训练 format 掉了。

**为什么会这样？** 想象 VLM 的参数空间是一间精心整理的图书馆。Action loss 的 gradient 像一头冲进来的野猪，横冲直撞把书架全撞翻了。Action 的 gradient 和 spatial grounding 的 gradient 方向不一致——论文用 PSS 这个指标量化，发现 naive co-training 时 PSS 只有 0.25，相当于两个优化方向几乎正交，互相打架。

## 他们怎么解决的

三个工程 trick 叠加，挺朴素但管用：

### Trick 1: 先给 VLM 补一堂"空间课"

正式学动作前，先拿 300 万条 spatial grounding 数据训练 VLM，包含四类：
- **Box QA**：给图里物体画 bounding box（87.9 万条）
- **Point QA**：指认物体上的关键点（83.2 万条）  
- **Trajectory QA**：预测 end-effector 的运动轨迹（68.4 万条）
- **General VQA**：维持一般视觉问答能力（63.7 万条）

这步的效果，相当于给图书馆每本书都贴上索引标签，书架按类目重新归位。后面 action training 再怎么折腾，至少 spatial 这个底子打牢了。

**有个有意思的数据**（Table 9）：从 0M 到 1M 数据，性能几乎没提升（61.4 → 63.4）。但 1M 到 2M 跳了 7.6 个点，2M 到 3M 又跳 6.9 个点。这暗示存在一个 "critical mass"——空间感知能力得积累到某个量级才会 emergent，跟 LLM 的 emergent ability 一个道理。

### Trick 2: 训练时加一句"灵魂 prompt"

训练 VLA 时，在 task instruction 后面加一句话：

> "Figure out how to execute it, then locate the key object needed."

就这一句话。

你可能会想：这是不是玄学？论文做了 ablation（Table 10），换成 "xxx, xxx, xxx" 这种无意义 padding，性能直接掉到 58.5%。换成 "请输出 bounding box 坐标" 这种硬约束，反而比不加 prompt 还差（76.6 vs 77.9）。只有这句软提示效果最好。

**直觉理解**：这句 prompt 像一个"开关"，激活 VLM 内部的 spatial attention 通路。你不用逼它真的输出坐标，只要让它的内部 representation "走一遍"空间推理流程，下游 action expert 拿到的 latent feature 就已经包含了空间信息。类似 chain-of-thought，但 reasoning 在 latent space 里完成，不强制 decode 成文字。

### Trick 3: 给 gradient 装个"减速带"

他们设计了一个 querying transformer（只有 8.7 MB）连接 VLM 和 action expert。关键 trick：action loss 反传到 VLM 时，gradient 乘以 0.5。

这是个挺精巧的设计。完全 freeze VLM 吧，VLM 适配不了 robotic 场景；完全 unfreeze 吧，spatial 知识被冲垮。0.5 这个衰减系数是中间路线——让 VLM 被 action signal 微调，但不至于被冲垮。

类比一下：RLHF 里用 KL penalty 防止 policy 偏离 reference model 太远。ST4VLA 的 gradient decay 是类似的 regularization 思路，只不过作用在 gradient magnitude 上。

## 架构长什么样

Dual-system 设计，借鉴 Kahneman 的 System 1 / System 2：

**System 2（慢思考）**：Qwen2.5-VL-3B，负责理解指令、推理空间关系，输出 latent planning tokens。这个模块"慢但靠谱"。

**System 1（快反应）**：DINOv2 视觉编码器 + DiT (Diffusion Transformer) 动作解码器，负责输出具体的关节控制信号。这个模块"快但需要指令"。

**桥梁**：那个 8.7 MB 的 querying transformer，用 cross-attention 把 VLM 的 variable-length tokens 压成固定数量的 query tokens，喂给 action expert。

整体数据流：
```
RGB image + instruction + spatial prompt
        ↓
   Qwen2.5-VL (System 2)
        ↓ latent spatial features
   Querying Transformer (gradient × 0.5)
        ↓ fixed query tokens
   DiT Action Expert (System 1)
        ↓
   8-dim action vector [7 joint deltas + 1 gripper]
   (chunk size = 16, via diffusion denoising)
```

## PSS 这个指标值得细说

Projection-Space Similarity 是这篇论文的方法论亮点。他们借用了 SVCCA（Raghu 2017）的思路，量化两个 loss 的 gradient 在参数空间中的"方向一致性"。

具体做法：
1. 选 VLM 最后一层 self-attention 的 q projection（2048×2048 矩阵）作为观测点
2. 固定一个 spatial data batch 和一个 action data batch
3. 分别计算两个 loss 对这个矩阵的 gradient
4. 用 SVD 分解出两个 gradient 的 column space
5. 计算两个 subspace 的 principal angles 的 cosine 平方均值

公式：
$$\text{PSS}(G_{spat}, G_{act}) = \frac{\text{tr}(P_{spat} P_{act})}{\min(r_{spat}, r_{act})}$$

其中 $P = G G^+$ 是投影矩阵，$G^+$ 是 Moore-Penrose 伪逆，$r$ 是矩阵的秩。

**人话翻译**：两个 gradient 矩阵各自张成一个子空间。如果两个子空间重合，PSS=1；如果正交，PSS=0。PSS 越高，说明 action 优化方向和 spatial grounding 优化方向越一致，两者不打架。

实验结果：
- Vanilla co-training: PSS = 0.25（打架严重）
- ST4VLA: PSS = 0.42（好多了）

虽然 0.42 听起来不高，但相比 0.25 是 68% 的相对提升，足以解释性能差异。

**为什么选 q projection 这一层？** 因为它是 VLM backbone 和 action expert 之间的"接口层"。两个 loss 的 gradient 在这里"相遇"，最能反映它们的相互作用。这个选择挺巧妙——选太靠前感受不到 action gradient，选太靠后感受不到 spatial gradient。

## 数字说话

### SimplerEnv（公开 benchmark）

Google Robot Visual Matching：
- Vanilla VLA：66.1
- 加 co-training：70.2（+4）
- ST4VLA：**84.6**（+18.5）

WidowX：
- Vanilla VLA：54.7
- ST4VLA：**73.2**（+18.5）

对比其他 SOTA：
- π0：Google VM 58.8，WidowX 27.1
- GR00T N1.5：Google VM 35.2，WidowX 61.9
- CogACT：Google VM 74.8，WidowX 51.3
- SpatialVLA：Google VM 75.1，WidowX 42.7

ST4VLA 在两个 benchmark 上都是 SOTA。

### LIBERO

平均 95.9%，超过 π0（94.2%）和 GR00T N1（93.9%）。特别是 long-horizon 任务 92.6%，比 π0 高 7.4 个点。

### Real-world（Franka 机器人）

In-distribution 92%，平均 65%（vs π0 31%，GR00T 48%）。在 unseen object orientation 这个维度上 72%，π0 只有 32%，差了 40 个点。

## 最 important 的几个 insight

**1. Spatial priors 存在 phase transition**

0M → 1M 数据几乎没用，1M → 3M 性能从 63.4 跳到 77.9。这跟 LLM 的 emergent ability 一个性质——空间知识需要积累到 critical mass 才能 generalize。

**2. Soft prompt > Hard constraint**

不强制 VLM 输出 box/point/trace 坐标，反而效果最好（77.9 vs 73.9-76.6）。Latent reasoning 比显式 decode 更高效。这是个挺反直觉的发现——我们直觉觉得"显式输出坐标应该更精准"，但强制输出会损失 latent space 的高维信息。

**3. Gradient alignment 比数据量更重要**

Vanilla co-training 用了同样的数据，但 PSS 只有 0.25，性能只有 70.2。ST4VLA 用 spatial prompting 把 PSS 提到 0.42，性能到 84.6。优化的"方向一致性"比单纯加数据更关键。

**4. Backbone 不是关键**

用弱得多的 Florence-2 替换 Qwen2.5-VL，ST4VLA 还是能 beat GR00T N1.5（67.9 vs 61.9）。这证明收益来自训练方法本身，不是堆参数量堆出来的。

**5. Loss ratio 1:10 是 sweet spot**

Spatial loss : action loss = 1 : 10 最优。1:1 时 VLM 一直在做 grounding 学不会动作，1:20 时 spatial 知识被 action 冲掉。这个 ratio 大约对应 action chunk length（16）和 VQA 平均 token 长度的比值——暗示 gradient budget 平衡的内在规律。

## 可以吐槽的地方

1. **Real-world 测试规模偏小**：300 rollouts，每个 setting 50 次。统计意义有限。

2. **没和 π0.5 在 SimplerEnv 上对比**：π0.5 是 Physical Intelligence 更新的版本，只在 LIBERO 上比了。

3. **Inference speed 没报**：DiT 的 diffusion denoising 通常需要多步，可能限制 control frequency。论文只说在 RTX 4080 上能跑，没给 Hz。

4. **Failure analysis 太浅**：只说"抓错"和"放错容器"，没分析 reasoning failure 的 case。

5. **Long-horizon 还是偏简单**：sandwich、drawer sorting 这些任务 step 数有限，离真正 long-horizon（几十步）还有距离。

## 我的 mental model

把 VLM 想象成一个"空间知识库"：
- Web pre-training 阶段它积累了海量 implicit spatial knowledge
- 但这些知识是 latent 的，没有"激活路径"
- Naive VLA training 像火灾，把知识库烧了
- ST4VLA 做了三件事：
  - Stage 1 给知识库装目录（spatial grounding pre-training）
  - Stage 2 训练时用 prompt 触发目录查询（spatial prompting）
  - Gradient decay 防火（不让 action gradient 冲垮 VLM）

为什么这个 design work？因为它尊重了 cognition 和 control 的不同 time scale。VLM 的 spatial reasoning 是慢思考，需要latent computation；action generation 是快反应，需要 reactive control。强行塞进一个网络，要么慢要么快，必然冲突。Dual-system 让两者各司其职，中间用 lightweight bridge 连接。

这个思路和人类的认知架构挺像——我们也是先用"System 2"想清楚要抓哪里，然后"System 1"自动执行抓取动作。你抓杯子时不会 consciously 计算每个关节角度，但你在抓之前会看一眼杯子在哪、判断一下怎么伸手。

## 对未来研究的启示

1. **Gradient analysis 会成为 VLA 标配**：PSS 这种工具应该成为 VLA 论文的 standard analysis。

2. **Spatial reasoning 是 universal substrate**：不止 pick-and-place，navigation、humanoid locomotion、dexterous manipulation 都可能受益。

3. **Critical mass 现象值得深究**：为什么是 2M 这个 threshold？跟 model size、task diversity 有什么关系？

4. **Soft prompting 范式可迁移**： latent space reasoning 比显式输出更高效，这个结论可以推广到其他 multi-modal 任务。

5. **Simulation data 是 scalable 关键**：他们用 GenManip + Isaac Sim 生成 244K pick-and-place 数据，planner 和 renderer 分离的设计很聪明——一次 planning 可以 render 多种视觉变体，data efficiency 倍增。

## Reference

- 项目主页：https://internrobotics.github.io/internvla-m1.github.io
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- DiT: https://arxiv.org/abs/2211.09748
- DINOv2: https://arxiv.org/abs/2304.07193
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- SVCCA (PSS 灵感来源): https://arxiv.org/abs/1706.05806
- SimplerEnv: https://simpler-env.github.io
- LIBERO: https://libero-project.github.io
- GenManip: https://arxiv.org/abs/2504.12636
- RefCOCO: https://arxiv.org/abs/1612.01558

总而言之，这篇论文的核心贡献就是把 VLA 训练中"spatial priors 崩塌"这个问题显式化了，给了一个简单可复用的解决方案（spatial prompting + gradient decay），并通过 PSS 量化分析让整个故事 self-consistent。Engineering execution 扎实，从 simulation 到 real-world 全链路打通，算是 VLA 领域一个 solid 的 milestone。

---

# ST4VLA 深度技术讲解

## 1. 论文核心动机与问题定义

这篇 paper 来自 Shanghai AI Lab，作者团队包括 Jiangmiao Pang (F1 系列作者), Yilun Chen 等。核心要解决的问题：当我们把一个预训练好的 VLM 直接 fine-tune 成 VLA (Vision-Language-Action) 模型时，**VLM 原本拥有的 spatial grounding 能力会发生 catastrophic collapse**。

这个观察非常关键。Figure 3(a) 显示，Vanilla VLA 在 20k steps 时，RefCOCO-g 的 IoU@0.5 已经掉到接近 random level。这意味着 action loss 的梯度在 backward 过程中破坏了 VLM 学到的视觉-空间表征。

**Intuition**: 可以这样想——VLM 在 web-scale 数据上学到的 spatial representation 是一个经过精心调教的"语言空间"中的几何概念，而 action loss 的梯度是一个完全不同方向的 force（低层 motor control 方向），两者在共享参数上发生 gradient conflict。

论文 webpage: https://internrobotics.github.io/internvla-m1.github.io

## 2. Dual-System 架构详解

### 2.1 架构组成

整个 framework 借鉴了 Kahneman 的 System 1 / System 2 思想，但他们做了一个工程上的精巧实现：

**System 2 (VLM Planner)**:
- Base: Qwen2.5-VL-3B-Instruct (https://arxiv.org/abs/2502.13923)
- 角色: slow but reliable，输出 latent planning tokens
- 输入: RGB image + task instruction + spatial prompt
- 输出: 内部 reasoning features（不强制输出显式文本格式）

**System 1 (Action Expert)**:
- Visual encoder: DINOv2 (https://arxiv.org/abs/2304.07193)
- Action decoder: DiT (Diffusion Transformer, https://arxiv.org/abs/2211.09748) 
- 角色: fast, embodiment-specific
- 输出: 8-dim continuous action vector (7-dim joint deltas + 1-dim gripper binary)
- Action chunk size: 16

**Bridge (Querying Transformer)**:
- 参数量仅 8.7 MB
- 结构: k-layer cross-attention (默认 k=1，只 attend 最后一层)
- 作用: 把 variable-length VLM tokens 压成 fixed set of learnable query tokens
- **关键 trick**: gradient decay factor = 0.5，即从 Action Expert 反传到 VLM 的梯度乘以 0.5

### 2.2 为什么需要 gradient decay?

参考 Driess et al. (π0.5, https://arxiv.org/abs/2504.16054) 和 Bjorck et al. (GR00T, https://arxiv.org/abs/2503.14734) 的发现：让 action gradient 直接流到 VLM 会 distort 多模态知识。ST4VLA 用一个 scalar decay 来"软化"这个 gradient flow，这是介于"完全 freeze VLM"和"完全 unfreeze"之间的中间方案。

**Intuition**: 这类似于 LoRA 或者 partial fine-tuning 的思想，但更精细——不是参数级别的 freeze，而是 gradient magnitude 层面的 attenuation。可以想象成给 VLM 一个 "防护罩"，让它既能被 action signal 微调，又不至于被冲垮。

## 3. 两阶段训练 Pipeline

### Stage 1: Spatial Grounding Pre-training

**目标**: 给 VLM 注入 transferable spatial priors。

**数据组成** (总共 3.032M):
- **General VQA** (637k): AOKVQA, ShareGPT4V, InternVL3 proprietary, COCOTextV2, VQAv2, TallyQA — 维持一般多模态能力
- **BBox-QA** (879k): RefCOCO (https://arxiv.org/abs/1612.01558), ASv2, COCO-ReM, RoboRefit, InternData-M1
- **Trajectory-QA** (684k): A0 ManiSkill (https://arxiv.org/abs/2504.12636), MolmoAct, InternData-M1 Traj
- **Point-QA** (832k): Pixmo-Points, RoboPoint (https://arxiv.org/abs/2406.10721), RefSpatial (https://arxiv.org/abs/2506.04308), InternData-M1 Point

所有数据被 reformat 成统一的 QA 格式，配合 Qwen2.5-VL 的 SmartResize 框架，坐标使用 JSON/XML 格式输出绝对坐标。

### Stage 2: Spatially Guided Action Post-training

**关键创新**: Spatial Prompting

在 task instruction 后面 append 一个统一的 prompt:
> "Figure out how to execute it, then locate the key object needed."

这个 prompt 不会强制 VLM 输出 bounding box 或 point，只是触发 VLM 内部的 spatial attention。

**Loss 设计**:
- Action data 上: VLM 做 next-token prediction + DiT 做 action diffusion
- 同时与 spatial grounding data co-train
- 最优 ratio: grounding : action = 1 : 10

为什么是 1:10? 论文的 hypothesis: 这个比例大约对应 action chunk length (16) 和 multimodal data 中 next-token prediction 平均长度的比值。

## 4. Projection-Space Similarity (PSS) 详解

这是 paper 的一个方法论亮点。他们借鉴了 SVCCA (Singular Vector Canonical Correlation Analysis, Raghu et al. 2017, https://arxiv.org/abs/1706.05806) 来量化两个 objective 之间的 alignment。

### 公式解析

设共享参数 θ ∈ ℝ^(d×n)，固定两个 probing mini-batches: B_spat (grounding) 和 B_act (action)。

**公式 (1)** - 计算两个 loss 对 θ 的梯度矩阵:
$$G_{spat} = \nabla_\theta \mathcal{L}_{spat}(\mathcal{B}_{spat}; \theta) \in \mathbb{R}^{d \times n}$$
$$G_{act} = \nabla_\theta \mathcal{L}_{act}(\mathcal{B}_{act}; \theta) \in \mathbb{R}^{d \times n}$$

变量含义:
- G_spat: spatial grounding loss 在参数 θ 上的 gradient matrix
- G_act: action policy loss 在参数 θ 上的 gradient matrix
- d × n: 参数矩阵的形状 (例如 2048 × 2048)

**公式 (2)** - 用 Moore-Penrose pseudoinverse 计算 orthogonal projector:
$$P_{spat} = G_{spat} G_{spat}^+$$
$$P_{act} = G_{act} G_{act}^+$$

变量含义:
- G^+: Moore-Penrose pseudoinverse，类似 (G^T G)^{-1} G^T 但对非方阵/奇异矩阵也适用
- P_spat, P_act: 投影到 G_spat 和 G_act 的 column space (range) 的 orthogonal projector
- P 是 idempotent (P² = P) 且 symmetric (P^T = P)

**公式 (3)** - 定义 PSS:
$$\text{PSS}(G_{spat}, G_{act}) = \frac{\text{tr}(P_{spat} P_{act})}{\min(r_{spat}, r_{act})} \in [0, 1]$$

变量含义:
- r_spat = rank(G_spat): G_spat 的秩 (column space 维度)
- r_act = rank(G_act): G_act 的秩
- min(r_spat, r_act): 归一化因子，保证 PSS ∈ [0,1]
- tr(P_spat P_act): 两个 projector 乘积的 trace

**几何意义**: tr(P_spat P_act) 等于两个 subspace 之间所有 principal angles 的 cosine 平方之和。除以 min rank 后，PSS 就是 mean squared cosine of principal angles。

- PSS = 1: 两个 gradient subspace 完全重合
- PSS = 0: 两个 gradient subspace 完全正交 (orthogonal)
- PSS 越高 → action 优化与 spatial representation 学习越 aligned

**实验 protocol**:
- 只在 Qwen2.5-VL 最后一层 self-attention 的 q projection 上计算 (2048 × 2048)
- 这个 layer 是 VLM backbone 和 action expert 之间的"接口"
- Probing batch size = 64 per data type

**结果** (Figure 3c):
- Vanilla co-training: PSS = 0.25 (gradient conflict 严重)
- Spatially guided training: PSS = 0.42 (alignment 显著提升)

**Intuition**: 想象两个优化方向是高维空间中的两个向量。如果它们 pointing to 大致相同方向，模型可以同时优化两个目标；如果 pointing to orthogonal 或相反方向，模型必须在两者间 trade-off。PSS 就是量化这种"方向一致性"的 metric。

## 5. 实验结果深度分析

### 5.1 SimplerEnv (https://simpler-env.github.io)

**Google Robot Visual Matching**:
- Vanilla VLA: 66.1
- Vanilla Co-training: 70.2
- ST4VLA: **84.6** (+18.5 over Vanilla VLA, +14.4 over co-training)

**Google Robot Visual Aggregation**:
- Vanilla VLA: 63.5
- ST4VLA: **75.9** (+12.4)

**WidowX Visual Matching**:
- Vanilla VLA: 54.7
- ST4VLA: **73.2** (+18.5)

vs. SOTA baselines:
- π0: 58.8 (Google VM), 27.1 (WidowX)
- GR00T N1.5: 35.2 (Google VM), 61.9 (WidowX)
- SpatialVLA: 75.1 (Google VM), 42.7 (WidowX)
- CogACT: 74.8 (Google VM), 51.3 (WidowX)

### 5.2 LIBERO (https://libero-project.github.io)

ST4VLA 在 LIBERO 上平均 95.9%，超过 π0 (94.2%) 和 GR00T N1 (93.9%)。特别值得注意的是:
- LIBERO-Long: ST4VLA 92.6% vs π0 85.2% (+7.4)
- LIBERO-Spatial: 98.0% (与 π0 持平)

### 5.3 Long-horizon Real-world

包括 5 类任务:
- Desktop sorting (5 物体类别 → 5 containers)
- Sorting items into drawers (open → place → close)
- Making sandwiches (5 种 recipe)
- Math calculation (按 button)
- Goods purchase (ARX LIFT2 dual-arm)

**Real-world result** (Table 4):
- In-distribution: 92%
- Unseen object (new instance): 62%
- Similar distractors: 49%
- New background: 63%
- Unseen object position: 52%
- Unseen object orientation: 72%
- Unseen instruction (by attribute): 73%
- Unseen instruction (by spatial): 61%
- Average: **65%** (vs π0 31%, GR00T 48%)

特别值得注意的是 **unseen object orientation** 上的表现 (72% vs π0 32%, GR00T 40%)，这归功于 large-scale simulation 数据 co-training 带来的 diverse grasp positions。

## 6. Ablation Studies 深度解读

### 6.1 Scaling Laws of Spatial Priors (Table 9)

| Pre-training Scale | Google VM | Google VA | WidowX | Average |
|---|---|---|---|---|
| 0M | 66.1 | 63.5 | 54.7 | 61.4 |
| 0.5M | 66.1 | 61.2 | 55.6 | 61.0 |
| 1.0M | 68.9 | 65.5 | 55.8 | 63.4 |
| 2.0M | 72.8 | 72.9 | 67.3 | 71.0 |
| 3.0M | 84.6 | 75.9 | 73.2 | **77.9** |

**关键洞察**: 这是一个非常明显的 **non-linear / phase transition** 行为。从 0M → 1M 几乎没有提升，但 1M → 2M 跳跃 +7.6，2M → 3M 再跳跃 +6.9。

**Intuition**: 这让我联想到 LLM 的 emergent abilities 现象。Spatial grounding 可能也存在 critical mass——当数据量低于某个阈值时，VLM 学到的 spatial representation 是碎片化的、无法 generalize；一旦超过阈值，representation 突然变得 coherent，能够 transfer 到 action 任务上。

### 6.2 Loss Weight Ratio (Table 7)

| Ratio (spatial : action) | WidowX VM |
|---|---|
| 1:1 | 47.2 |
| 1:5 | 58.3 |
| 1:10 | **71.7** |
| 1:15 | 71.8 |
| 1:20 | 68.3 |

**关键洞察**: 这是一个 **U-shaped** curve，最优在 1:10 附近。1:1 时 spatial loss 主导，action 学不好；1:20 时 action loss 主导，spatial grounding 被 wash out。

**Intuition**: 这与 multi-task learning 中的 task balancing 问题相关 (e.g., GradNorm, Uncertainty Weighting)。1:10 对应 action chunk length 16 与平均 next-token prediction 长度的比值，这暗示 gradient magnitude 平衡的内在规律。

### 6.3 Spatial Prompt Formulations (Table 10)

| Prompt Type | Average |
|---|---|
| Random Padding ("xxx, xxx...") | 58.5 |
| **Unified Prompting (default)** | **77.9** |
| Box Prompting | 76.6 |
| Point Prompting | 74.9 |
| Trace Prompting | 73.9 |

**两个关键洞察**:
1. **Semantic content matters**: Random padding 显著低于 unified prompting (58.5 vs 77.9)，证明收益来自 spatial semantic attention，不是 token length。
2. **Soft > Hard**: 不强制输出格式的 unified prompt 反而最好。强制 box/point/trace 输出会 constrain policy 灵活性。

**Intuition**: 这与 chain-of-thought 的设计哲学一致——给模型自由 reasoning 的空间，比强迫它输出固定格式更好。VLM 的 latent representation 本身就蕴含了 spatial 信息，强制 decode 成显式坐标反而损失信息。

### 6.4 Backbone-Agnostic (Table 8)

用 Florence-2 (https://arxiv.org/abs/2311.06642, weak VLM) 替换 Qwen2.5-VL:
- Florence-2 + ST4VLA: 67.9% (WidowX)
- Florence-2 + Vanilla Co-training: 46.1%
- GR00T N1.5 (with Eagle-2.5): 61.9%

**关键洞察**: 即使 backbone 弱很多，ST4VLA 仍然能 beat GR00T N1.5。这证明收益来自训练方法，不是 backbone capacity。

### 6.5 100k Steps Extended Training (Figure 6)

延长训练到 100k steps，baseline 仍然 saturate 在更低 plateau。这证明 spatial priors 提高的不仅是 convergence speed，更是 performance ceiling。

**Intuition**: 这与 critical learning periods (Achille et al.) 的概念相关。早期注入的 spatial inductive bias 会改变整个 optimization landscape，让模型 reach 一个不同的、更好的 basin。

## 7. 与相关工作的关系

### 7.1 与 Magma (https://arxiv.org/abs/2502.13130) 对比

Magma 也用 spatial pre-training，但没有 explicit spatial prompting 来 guide action generation。ST4VLA 的贡献在于第二阶段的 spatial prompting 设计。

### 7.2 与 π0 (https://arxiv.org/abs/2410.24164) 对比

π0 是 monolithic VLA，直接 map multimodal input 到 tokenized action。ST4VLA 通过 dual-system 分离 cognition 和 control，并 explicit align 两个 objective 的 gradient。

### 7.3 与 SpatialVLA (https://arxiv.org/abs/2501.15830) 对比

SpatialVLA 探索 spatial representations 但没有 explicit 的 gradient alignment 分析。ST4VLA 的 PSS 是首次量化 VLA 中 perception-action gradient conflict 的工作。

### 7.4 与 CogACT (https://arxiv.org/abs/2411.19650) 对比

CogACT 也是 dual-system 设计，但没有 spatial grounding pre-training stage。ST4VLA 在 CogACT 之上 Google VM 高出 9.8 points (84.6 vs 74.8)。

### 7.5 与 CoT-VLA (https://arxiv.org/abs/2502.05311) 对比

CoT-VLA 用 visual chain-of-thought (future frames) 作为 reasoning。ST4VLA 用 spatial latent reasoning，computational cost 更低。

### 7.6 与 RoboRefer (https://arxiv.org/abs/2506.04308) 对比

RoboRefer 用 RL 训练 fine-grained spatial grounding。ST4VLA 把 spatial grounding 作为 pre-training，更 scalable。

### 7.7 与 LLARVA (https://arxiv.org/abs/2406.11815) 对比

LLARVA 用 visual trace representations 来 align vision 和 action。ST4VLA 更进一步用 latent spatial reasoning 而非显式 trace。

## 8. 数据生成 Pipeline

### 8.1 GenManip (https://arxiv.org/abs/2504.12636)

基于 Isaac Sim (https://arxiv.org/abs/2108.10470) 构建的可扩展 simulation pipeline:
- 14K annotated objects
- 211 tables
- 1.6K textures
- 87 dome lights

**关键设计**: Planner 和 Renderer **decoupled**
- Planner: 记录 joint states, object positions, action info
- Renderer: 随机化 lighting, materials, viewpoints 重放
- 用 ArUco markers 标定 cameras，对齐 real-world camera 参数

这种 decoupling 让一次 planning 可以 render 多个视觉变体，极大提升 data efficiency。

### 8.2 Synthetic Action Post-Pre-training

244K closed-loop pick-and-place samples，用 GenManip pipeline 生成。每个 trajectory 都经过:
1. Scene graph solver 生成 layout
2. 基于 object mesh 计算 candidate grasps
3. Physics 执行验证
4. Scene-graph validator 检查 task goal 是否达成
5. 只有 fully successful 的 trajectory 才被接受

## 9. Failure Modes (Figure 23)

论文诚实地报告了 failure cases:
- Incorrect grasp execution
- Target container misidentification
- Sensor limitations 在 cluttered 环境

Future work 方向: 加入 depth sensing 和 proprioceptive feedback。

## 10. 整体 Intuition 总结

### 10.1 为什么 ST4VLA work?

我的理解是，ST4VLA 实际上解决了一个 **representation bottleneck** 问题:

1. **Pre-trained VLM 的 spatial representation 是 "frozen" 的**——它学到的是 web-scale 数据上的 geometric concepts，但没有 embodied grounding。

2. **直接 fine-tune 成 VLA 会 "wash out" 这些 representation**——action gradient 是 high-frequency signal，会 overwrite VLM 的 low-frequency spatial knowledge。

3. **ST4VLA 的两阶段策略**:
   - Stage 1: 把 spatial representation "enrich" 到更贴近 robotic task
   - Stage 2: 用 spatial prompting "anchor" 住这些 representation，让 action gradient 只能在其基础上 refine，而不是 destroy

### 10.2 为什么 1:10 ratio 最优?

这暗示了一个 **gradient budget** 的概念。每一步训练中，VLM 参数的更新可以看作一个有限的 budget。如果 spatial gradient 占比过高 (1:1)，VLM 一直在做 grounding，action 学不到 motor pattern；如果 action gradient 占比过高 (1:20)，spatial representation 被 overwrite。1:10 是一个 sweet spot，让 action gradient 主导 optimization 方向，但 spatial gradient 提供 regularization。

### 10.3 为什么 unified prompt > explicit format prompt?

VLM 的 latent space 是高维 continuous 的，而 explicit box/point/trace 是 low-dim discrete 的。强制 decode 成 explicit format 会 force information bottleneck。Unified prompt 让 VLM 在 latent space 中自由 reasoning，保留 full spatial information 给 action expert。这与 VAE 的 reparameterization trick 思想类似——保留 latent distribution 而不是 sample 一个 point。

### 10.4 与 LLM 中 RLHF 的相似性

ST4VLA 的 gradient decay factor (0.5) 类似于 RLHF 中的 KL penalty——防止 policy 偏离 reference model 太远。ST4VLA 是在 gradient magnitude 上做 decay，RLHF 是在 loss 上加 KL term，本质上都是 regularization。

## 11. 技术细节 Q&A

### Q1: 为什么选 Qwen2.5-VL 而不是 InternVL?
论文作者来自 Shanghai AI Lab，InternVL 也是他们的工作。但选 Qwen2.5-VL 可能因为:
- Qwen2.5-VL-3B 更 compact，适合 dual-system 设计
- SmartResize 框架对 coordinate prediction 支持更好
- 在 robotic task 上有 better 性能/参数 比

### Q2: DINOv2 在 Action Expert 中作用?
DINOv2 提供 embodiment-agnostic visual features，与 VLM 的 semantic features 互补。VLM 关注 "what and where"，DINOv2 关注 "how to grasp" 的 low-level visual cue。

### Q3: DiT Actor 的具体架构?
论文没详细说明，但参考 Peebles & Xie (https://arxiv.org/abs/2211.09748)，DiT 通过 adaptive layer norm 条件化 on VLM latent tokens。8-dim action vector 通过 diffusion process 生成，chunk size 16。

### Q4: 为什么 action chunk size = 16?
这和 ACT (Action Chunking with Transformers) 的发现一致。chunking 减少 compounding error，16 是一个经验 sweet spot，平衡 latency 和 smoothness。

### Q5: Real-world deployment speed?
论文提到 inference 在 RTX 4080 16GB VRAM 上运行，但没给具体 Hz。DiT 的 diffusion process 通常需要 multiple denoising steps，可能限制频率。

## 12. 对未来 VLA 研究的启示

1. **Gradient alignment 是新方向**: PSS 这种分析工具可以推广到其他 multi-modal multi-task 场景。

2. **Spatial reasoning 作为 substrate**: 论文核心 thesis 是 spatial reasoning 是连接 high-level semantic 和 low-level motor 的 universal substrate。

3. **Critical mass in data scaling**: 2M 是一个 interesting threshold，暗示 VLA 需要类似 LLM 的 emergent threshold 才能解锁 full capability。

4. **Soft prompting > Hard decoding**: 这对未来 VLA 设计有启示——不要强迫 VLM 输出显式 coordinate，让它在 latent space 中自由 reasoning。

5. **Decoupled simulation pipeline**: Planner/Renderer 分离是一个 scalable data 生成的重要 design pattern。

## 13. 可能的扩展方向

1. **加入 depth 和 tactile**: 论文 failure analysis 提到这。
2. **Mobile manipulation**: 当前局限 tabletop，可以扩展到 navigation + manipulation。
3. **Humanoid locomotion**: DiT Actor 可以替换成 humanoid joint space。
4. **3D spatial reasoning**: 当前主要是 2D + camera geometry，可以加入 NeRF / 3D Gaussian Splatting。
5. **Online adaptation**: 当前是 offline training，可以加入 RL fine-tuning。
6. **Multi-agent coordination**: ARX LIFT2 dual-arm 已经 hint 了这个方向。

## 14. 可能的局限与 Critique

1. **Long-horizon evaluation 规模**: 300 rollouts，每个 setting 至少 50 次，相对 small。
2. **Failure mode 分析不够深入**: 只提了 incorrect grasp 和 misidentification，没分析 reasoning failure。
3. **Computational cost 没报告**: 训练用 16 A100 GPU，50k steps，但没报告总训练时间。
4. **vs. π0.5 没对比**: π0.5 (https://arxiv.org/abs/2504.16054) 是更新的 SOTA，但只在 LIBERO 上对比，没在 SimplerEnv 上对比。
5. **Real-world generalization 仍有 gap**: 65% average vs 92% in-distribution，泛化仍有提升空间。
6. **PSS 的 robustness**: 只在 q projection 上计算，没验证在其他 layer 上的结果。

## 15. 个人 Intuition Building

读完这篇 paper，我形成的 mental model:

**VLM 是一个 "spatial library"**: 它在 web-scale 数据上学到了关于物体、空间、affordance 的大量 implicit knowledge，但这些知识是 "latent" 的，需要 "key" 来 unlock。

**Naive VLA fine-tuning 是 "library fire"**: action gradient 像一把火，烧掉了 library 的 index，让 VLM 无法检索 spatial knowledge。

**ST4VLA 是 "library catalog 系统"**:
- Stage 1 (spatial grounding pre-training) = 建立 catalog，给 library 加索引
- Stage 2 (spatial prompting) = 在 query 时 trigger catalog 查询，让 VLM 先 retrieve spatial info，再交给 action expert

**Dual-system 设计**:
- System 2 (VLM) = 图书管理员，slow 但 reliable
- System 1 (DiT) = 执行者，fast 但需要 instruction
- Querying transformer = 传话筒
- Gradient decay = 防止执行者对图书管理员大喊大叫，扰乱 library

这个 mental model 帮助我理解为什么 ST4VLA 比 monolithic 设计 work——它尊重了 cognition 和 control 的不同 time scale 和 representation 需求。

---

## Reference Links

- Paper: https://internrobotics.github.io/internvla-m1.github.io
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- DINOv2: https://arxiv.org/abs/2304.07193
- DiT (Peebles & Xie): https://arxiv.org/abs/2211.09748
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- OpenVLA: https://arxiv.org/abs/2406.09246
- CogACT: https://arxiv.org/abs/2411.19650
- SpatialVLA: https://arxiv.org/abs/2501.15830
- Magma: https://arxiv.org/abs/2502.13130
- CoT-VLA: https://arxiv.org/abs/2502.05311
- SVCCA (Raghu et al.): https://arxiv.org/abs/1706.05806
- RefCOCO: https://arxiv.org/abs/1612.01558
- SimplerEnv: https://simpler-env.github.io
- LIBERO: https://libero-project.github.io
- A0 (ManiSkill): https://arxiv.org/abs/2504.12636
- RoboPoint: https://arxiv.org/abs/2406.10721
- RoboRefer: https://arxiv.org/abs/2506.04308
- LLaRA: https://arxiv.org/abs/2406.20095
- LLARVA: https://arxiv.org/abs/2406.11815
- ATM (Any-point Trajectory): https://arxiv.org/abs/2401.00025
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Florence-2: https://arxiv.org/abs/2311.06642
- OXE Dataset: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- AnyGrasp: https://arxiv.org/abs/2304.00328

这篇 paper 我觉得最大的贡献有三点: (1) 系统性地观察并量化了 VLA 中 spatial priors collapse 现象 (via PSS); (2) 提出了一个简单但 effective 的解法 (spatial prompting + gradient decay); (3) 证明了 spatial reasoning 是 scalable robotic learning 的 universal substrate。整个工作的 engineering execution 非常 solid，从数据生成到 real-world deployment 都有完整 pipeline，这对未来 VLA 研究树立了一个 strong baseline。
