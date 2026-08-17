---
source_pdf: Sim-and-Real Co-Training.pdf
paper_sha256: fcaf870c0e325ae898154af0b093faec4ead174add91a6db60e9a21287f3e292
processed_at: '2026-08-12T06:14:25-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

Andrej，我换个说法，把这篇 paper 的核心 idea 用大白话拆解一下。

---

## 一句话说清楚

**你手头有 50 条 real robot demo，不够训出好 policy。你从 simulation 里白嫖 1 万条 demo，混在一起训，policy 性能直接翻倍。关键发现是：simulation 数据不需要跟 real 长得像，甚至 task 都可以不一样，只要语义搭点边就有用。**

---

## 为什么这件事 surprising？

传统 wisdom 说 simulation 跟 real 之间有个 "reality gap"——sim 里的物理引擎不准、渲染不真、摩擦系数不对、光照不对。所以 sim-to-real 一直是个 hard problem，大家花大量时间做 domain randomization（把 sim 参数乱调一通希望 policy 学到 invariance）、system identification（精调 sim 参数匹配 real）、digital twin（把 sim 建得跟 real 一模一样）。

这篇 paper 说：**别折腾了，直接把 sim data 和 real data 混一起训就行**。sim 长得不像 real 没关系，policy 网络自己会 figure out 哪些 feature 有用、哪些是 sim artifact 可以 ignore。

这就像你教小孩学骑车，传统做法是先在 sim 里学完美骑车姿势再 transfer 到 real bike。这篇 paper 说不如让小孩 real bike 骑 10 分钟、sim bike 骑 1000 分钟，混着来，他自己就学会了。

---

## 两种 Sim Data 都有用，这很反直觉

### 第一种：Task-Aware Digital Cousin (DC)
你知道 real task 是 "把 cup 从 counter 拿到 sink"，你在 sim 里也建一个 kitchen，也做 "把 cup 从 counter 拿到 sink"，但 cup 长得不一样、counter 纹理不一样、lighting 不一样、robot 起始位置稍微不一样。这种"差不多但不精确"的 sim 环境叫 digital cousin。

### 第二种：Task-Agnostic Prior
你直接拿别人已经做好的 sim dataset（RoboCasa 60k demos，24 个 task，100 个 scene），这些 task 里只有 3 个跟你 real task 语义对应，其他 21 个完全不搭边。你就这么直接用。

**结果**：两种都有用。DC 带来 35.8% 提升，Prior 带来 31.5% 提升。Prior 几乎追平 DC，这很 crazy。

这就像你想学做红烧肉，结果发现不仅看红烧肉教程有用，看糖醋排骨、宫保鸡丁、甚至西餐牛排教程都有用——因为它们都在教你 "怎么拿菜刀、怎么控制火候、怎么翻锅" 这些 transferable skill。

---

## 关键超参数 α：Sim 和 Real 的配比

这是整个 recipe 最 sensitive 的 knob。公式很简单：

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{sim}} + (1-\alpha) \cdot \mathcal{L}_{\text{real}}$$

- α=0.5：sim 和 real 一半一半，效果差（~60%）
- α=0.9：90% sim，10% real，效果好（~90%）
- α=0.99：99% sim，1% real，效果最好（95%）
- α=0.999：99.9% sim，0.1% real，效果崩了（60%）

**Intuition**：α 控制 "练习题 vs 标准答案" 的比例。你有 1 万道练习题（sim）和 50 道标准答案（real）。

- α=0.5 意味着你一半时间做练习题一半时间对答案，但 50 道答案被反复看，overfit 到这 50 道
- α=0.99 意味着你 99% 时间做练习题，1% 时间瞄一眼答案。练习题锻炼 skill，偶尔的答案做 calibration，告诉网络 "real 长这样"
- α=0.999 意味着答案几乎看不到，网络完全 fit 到 sim distribution，real 的 signal 被 drowned out

这个 pattern 跟 LLM pretraining + instruction tuning 一模一样：pretraining data 远大于 instruction data，但 instruction data 不能少到没有，否则 model 不 align 到 human preference。

---

## 哪些 Alignment 重要，哪些不重要？

他们系统消融了 6 个 factor：

| Factor | 重要吗？ | 备注 |
|---|---|---|
| Camera viewpoint | **Critical** | 不对齐直接掉 10-25% |
| Task semantic | **Important** | DC 对齐 task 比 Prior 不对齐好 |
| Sim data 数量 | **Critical** | 10k → 500 掉 14% |
| Object geometry | Less critical | DC 用不同 instance 仍有效 |
| Scene 纹理 | Less critical | Prior 完全不同 scene 仍有效 |
| Physics 参数 | **Not critical** | 调不调无区别（FAQ 里说的） |

**最反直觉的发现**：physics 参数不重要。你 sim 里摩擦系数、质量、惯性都跟 real 不一样，policy 照样 work。

**Intuition**：vision-based policy 是闭环控制。policy 看到 cup 偏左，就往右调一点。这个 feedback loop 把 physics error 吃掉了。policy 学的不是 "forward model of physics"，是 "visual servoing policy"——看到什么 visual pattern 就输出什么 action。只要 sim 和 real 的 visual pattern 有 overlap，policy 就能 transfer。

Camera 重要是因为 perspective 直接影响 spatial reasoning。sim 里 camera 在左边看，real 里 camera 在右边看，policy 学到的 "物体在画面左边 → 往右伸手" 在 real 里就完全反了。

---

## 最有意思的实验：CloseDoor

Real only 50 demos：10% success。
Real + DC：100% success。
Real only 100 demos：80% success。

这个 gap 巨大。为什么？CloseDoor 是个很 narrow 的 motion——robot 伸到 door handle 附近，推一下。50 个 real demo 不足以让 policy 学到 robust 的 reaching + pushing motion。但 sim 里 DexMimicGen 生成了 10000 个 door-closing trajectory，覆盖了各种 door angle、各种 reach 路径。Policy 从 sim 学到了 "怎么 reach and push a door" 的 motion primitive，real 50 demos 只需要告诉它 real door 长什么样。

这暗示 **sim data 的价值在于 motion skill coverage，real data 的价值在于 visual grounding**。两者 complementary。

---

## Generalization 的惊喜

### Unseen Objects
Real only 训了 9 种 object，遇到没见过的 carrot、ladle、lime，success rate 33%。
Real + DC，DC 里 randomize 了更多 object，success rate 50%。
GR-1 CupPnP 更夸张：Real only 10% → Real + DC 80%。

### Unseen Positions
Real only 训练时 object 都在 workspace 边缘，测试时放中间，policy 懵了（11% Panda，43% GR-1）。
Real + DC，DC 里 object 位置 uniform 分布，policy 学到 spatial invariance，success rate 翻倍（28% Panda，100% GR-1）。

**Intuition**：real data 贵，你只能收 50 demos，object 位置必然稀疏。Sim data 便宜，你可以 randomize 任何维度。这些 randomization 维度就是 policy 的 generalization axis。你在 sim 里 randomize 了 position，real policy 就 generalize 到新 position。你在 sim 里 randomize 了 object color，real policy 就 generalize 到新 color。**Simulation 是 generalization 的免费午餐**。

---

## 为什么这跟 LLM Pretraining 的 intuition 完全一致？

| Robotics Co-Training | LLM Training |
|---|---|
| Prior sim data (60k diverse tasks) | Web pretraining (trillions tokens) |
| DC sim data (task-aligned) | Domain pretraining (e.g. code, math) |
| Real data (50-100 demos) | Instruction tuning / SFT |
| α=0.99 | Pretraining loss >> SFT loss |
| Sim visual gap | Web data noise |
| Generalization to unseen objects | Generalization to unseen tasks |

这个 parallel 非常 clean。Robotics 正在 repeat NLP 的路径：
1. 早期：small task-specific dataset（BC on 50 demos ≈ small supervised dataset）
2. 现在：co-train with large sim data（≈ pretrain on web data + finetune on task）
3. 未来：robot foundation model on internet-scale data + co-training with sim + RL finetune（≈ GPT-4 pipeline）

这篇 paper 本质上是在 robotics 里 validate 了 "pretrain + finetune" paradigm 的有效性，只不过 pretraining data 来自 simulation 而不是 internet。

---

## Appendix 里的彩蛋：Vid2Vid

他们用 CogVideo-X 把 sim video 变得更 real-like（style transfer），发现在 low-data regime 特别有用：

| Setup | Real+DC | Real+DC w/ V2V |
|---|---|---|
| Real 20 + Sim 20 | 48% | 70% |
| Real 20 + Sim 1000 | 95% | 95% |

**Intuition**：sim data 少时，policy 没有 enough samples 来自己 learn visual invariance，Vid2Vid 帮忙把 sim visual 拉近 real。Sim data 多时，policy 自己就学会 ignore sim artifact，Vid2Vid 的 marginal benefit 趋零。

这指向一个未来 pipeline：**用 generative video model 直接生成 synthetic robot demos，完全绕过 physics simulator**。NVIDIA Cosmos、Genie、GAIA-1 都在往这个方向走。如果 video generation model 能 controllably 生成 action-labeled robot videos，reality gap 问题从根本上消失。

---

## Recipe 速查表（给 practitioner 的）

1. **有 real task + 有 prior sim data**：直接 co-train，α=0.9-0.99，prior data 开箱即用
2. **有 real task + 没有 prior sim data**：花 1-2 天建个 digital cousin（对齐 task semantic + camera，其他随意），用 MimicGen 生成 10k demos，co-train
3. **Sim data 数量**：越多越好，至少 10x real data，理想 100x
4. **必须对齐的**：camera viewpoint、task definition（success check + language）
5. **不用纠结的**：physics 参数、object 精确几何、scene 纹理
6. **想 boost generalization**：在 sim 里 randomize object 和 position

---

## Web Links

**核心方法**：
- Paper project：https://co-training.github.io/
- RoboCasa：https://robocasa.ai/
- MimicGen：https://mimicgen.github.io/
- DexMimicGen：https://dexmimicgen.github.io/
- Diffusion Policy：https://diffusion-policy.cs.columbia.edu/

**Foundation Model 方向**：
- π0：https://www.physicalintelligence.company/blog/pi0
- OpenVLA：https://openvla.github.io/
- GR00t N1：https://developer.nvidia.com/groot
- RT-2：https://robotics-transformer2.github.io/

**Generative World Model**：
- NVIDIA Cosmos：https://www.nvidia.com/en-us/ai/cosmos/
- Genie 2：https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- CogVideoX：https://github.com/THUDM/CogVideo

---

## 一句话 Takeaway

**别再纠结 sim-to-real gap 了，把 sim data 当 pretraining data、real data 当 finetuning data，混在一起训，α 调到 0.99，simulation 数据的量级和 diversity 会 surprise 你。这是 robotics 版的 "pretrain + finetune" paradigm，而且 sim data 的 cost 是 real data 的 1/100。**

---

# Sim-and-Real Co-Training 深度解析

Andrej，这篇 paper 我读完之后非常兴奋，因为它本质上回答了一个 robotics 社区长期纠结的问题：**到底 simulation 数据能不能"开箱即用"地帮 real-world policy**？答案是有条件的 yes，而且条件比想象中宽松得多。我来 build 一下你的 intuition。

---

## 1. Core Insight：Co-Training 比 Sim-to-Real Transfer 宽容得多

传统 sim-to-real 的哲学是 **"simulation 必须逼近 reality"**——要么通过 system identification 精调物理参数，要么通过 domain randomization 把 simulation 扰动得足够鲁棒。这两种路子的 human cost 都很高。

这篇 paper 提倡的 co-training 哲学是 **"simulation 只需要语义对齐，policy 自己会学到有用的 inductive bias"**。最震撼的发现是：即便是 task-agnostic 的 prior simulation 数据（与 real-world task 有大量 discrepancy），co-training 也能带来 **31.5% 的平均 success rate 提升**。这个发现颠覆了"必须先 align 再 transfer"的传统认知。

**Intuition**：policy 网络像一个 student，real data 是"标准答案"，sim data 是"大量练习题"。练习题的题型和真实考试不必完全一样，只要 exercise 了相关的 visuomotor skill（grasping、reaching、placing 的 motion primitive），网络就能学到 transferable representation。α 这个 ratio 就是在控制"练习题 vs 标准答案"的配比——99% sim 意味着 student 大部分时间在做练习，偶尔对一下标准答案校准。

---

## 2. 公式深入：Co-Training Loss 的形式与实现

$$\mathcal{L}_{\text{total}}(\theta; \mathcal{D}_{\text{real}}, \mathcal{D}_{\text{sim}}) = \alpha \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{sim}}) + (1 - \alpha) \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{real}})$$

其中 behavioral cloning loss：

$$\mathcal{L}(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(o_i, a_i) \in \mathcal{D}} -\log \pi_\theta(a_i | o_i)$$

变量解释：
- $\theta$：policy $\pi_\theta$ 的参数（Diffusion Policy 的 U-Net 或 Transformer 参数）
- $\mathcal{D}_{\text{real}} = \{\xi_i\}_{i=1}^{N}$：real-world trajectory 数据集，规模 $N$（实验中 20-100 demos）
- $\mathcal{D}_{\text{sim}} = \{\xi_i\}_{i=1}^{M}$：simulation trajectory 数据集，规模 $M$，且 $M \gg N$（实验中 1k-60k）
- $\alpha \in [0,1]$：co-training ratio，sim 数据的相对权重
- $o_i$：observation（RGB 图像 + proprioception）
- $a_i$：action（end-effector delta 或 joint position target）
- $\xi_i$：一条完整 trajectory $\{(o_t, a_t)\}_{t}$

**实现 trick**（Appendix VIII-G）：
他们没有显式加权 loss，而是用 **batch sampling 概率化** 实现。Batch size $B$ 时，每个 sample 被采到的概率：

$$P[(o_i, a_i) \in \mathcal{D}_{\text{sim}}] = \alpha, \quad P[(o_i, a_i) \in \mathcal{D}_{\text{real}}] = 1 - \alpha$$

具体做法是：先按 dataset size 归一化每个 sample 的 weight，再乘以 $\alpha$（sim）或 $1-\alpha$（real）。这等价于 weighted loss，但避免了对不同 dataset 做 oversampling 的 memory 浪费。

**Intuition for α**：Figure 5 显示 α=0.5（1:1 mix）suboptimal，α=0.99 最佳，α=0.999 反而 collapse。这说明：
- α 太小：sim 数据的 signal 被 real data 淹没，相当于浪费了大 pool 的练习题
- α 太大：real data 的 calibration signal 消失，policy 完全 fit sim distribution，无法 generalize 到 real visuals
- α=0.99：每 batch 256 个样本里 ~2.5 个来自 real，这极少的 real samples 起到 "anchor" 作用，把 sim 学到的 skill 校准到 real visual/动力学上

这和 LLM pretraining + few-shot instruction tuning 的动力学很像——大量 pretraining（sim）建立 representation，少量 instruction（real）做 alignment。

---

## 3. 两种 Simulation Data 的对比：Digital Cousins vs Prior

### Task-Aware Digital Cousins (DC)
定义借鉴了 Dai et al. 的 "digital cousin" 概念，但更精确：必须 preserve 四个要素：
1. Same robot + action space
2. Same task goal（success check + language instruction）
3. Same object categories（instances 可不同）
4. Same environmental fixture categories

**与 digital twin 的区别**：digital twin 要求几何、纹理、物理参数完全匹配，DC 只要求 semantic match。这降低了 human effort 几个数量级。

### Task-Agnostic Prior (Prior)
直接用 RoboCasa 的 60k 多任务数据。这些数据和 real-world task 有大量 mismatch：
- Robot base position 不同（[0, -20, -22] vs [0, -32, -4]，单位 cm，相对于 sink middle edge）
- Initial robot joints 不同（[0.09, -0.20, -0.02, -2.47, -0.01, 2.30, 0.85] vs [-0.02, -1.03, -0.02, -2.28, 0.04, 1.52, 0.70]）
- Object categories 范围更广（70 vs 9）
- Sampling region 更大（30×40 vs 27×23 cm）
- Task semantics 部分对应（RoboCasa 的 PickPlaceCounterToSink 对应 real 的 CounterToSinkPnP）

**关键处理**：他们只做了一件事——**re-rendering simulation demos 来 approximately match real camera pose**。其他全部 out-of-the-box。

---

## 4. 实验数据表精读

### Table I 主结果

| Data Composition | C2SPnP | C2CPnP | CloseDoor | CupPnP | MilkPnP | Pouring | Average |
|---|---|---|---|---|---|---|---|
| Real | 44% | 38% | 10% | 65% | 50% | 65% | 45.3% |
| Real + DC | 67% | 72% | 100% | 95% | 70% | 85% | 81.1% |
| Real + Prior | 58% | 53% | 100% | 80% | 80% | 70% | 76.8% |
| Real + DC + Prior | 72% | 72% | 100% | 85% | 80% | 90% | 83.2% |

几个值得注意的 pattern：

1. **CloseDoor** 的 Real only 只有 10%，但 co-training 后达到 100%。这个 gap 巨大。Appendix VIII-L 显示即便把 real demos 翻倍到 100，Real only 也只有 80%。这说明 CloseDoor 任务对 motion pattern 的依赖性极强，sim 的 DexMimicGen 数据提供了丰富的 door-closing motion primitive，real 50 demos 不足以覆盖。

2. **Pouring** 这种 non-prehensile 任务，co-training 也有效。这个任务涉及 deformable dynamics（液体或球的滚动），sim 物理 vs real 物理差异大，但 sim 数据还是带来了 20% 提升。这说明 visuomotor policy 学到的是 visual-conditioned motion skill，物理细节 mismatch 的影响被 visual feedback loop 吃掉了。

3. **Real + DC + Prior** 在多数任务上 ≥ Real + DC，说明 Prior 数据提供额外的 diversity bonus，但 CupPnP 上反而下降（95% → 85%），可能是 Prior 数据的 bimanual 行为干扰了 single-arm task（参考 Appendix VIII-M 的 bimanual 实验讨论）。

### Table II Generalization

| Data | Unseen Objects (Panda) | Unseen Objects (GR-1) | Unseen Positions (Panda) | Unseen Positions (GR-1) |
|---|---|---|---|---|
| Real | 33% | 10% | 11% | 43% |
| Real + DC | 50% | 80% | 28% | 100% |

GR-1 CupPnP 的 unseen objects 从 10% → 80% 是 8 倍提升。这个数字背后是 DC 中 object color 和 instance 的 randomization（Table IV 显示 CupPnP DC 虽然只有 1 个 object category，但 randomize color）。Real 20 demos 用固定红杯，policy overfit 到红色，DC 数据教会它 "cup 的 shape invariant"。

**Intuition**：simulation 的 diversity 是免费的。在 real 收集 100 种 cup 的 demo 要几周，在 sim 里 randomize color 一个 flag 就完成。这是 co-training 的核心杠杆。

### Figure 4：Data-Rich Setting

MultiTaskPnP 任务上，固定 4000 DC demos，变化 real demos 40 → 400：
- Real only：~30% → ~55%
- Real + DC：~55% → ~75%

Gap 在 400 real demos 时仍维持 20%。这回答了一个重要问题：**"当 real data 充足时 sim 还有用吗？"** 答案是有用，但 marginal benefit 减少。曲线趋势是收敛的，可能 1000+ real demos 后 gap 会 close。

### Figure 5：Co-Training Ratio α

α=0.5 → ~60%，α=0.9 → ~90%，α=0.99 → 95%（peak），α=0.999 → 60%。

这个非单调曲线很有意思。我的 interpretation：
- 左侧（α 小）：real data 被过度采样，sim data 不够 exercise visuomotor skill
- 右侧（α 大）：real data signal 太稀疏，policy collapse 到 sim distribution
- Peak 在 0.99：每 100 个 batch samples 有 1 个 real，刚好够 calibration

这和 mixed-precision training、contrastive learning 的 positive/negative ratio 调参有相似 flavor。

---

## 5. Data Composition Factors：哪些 alignment 重要？

Paper 定义了 6 个 factors：Task、Scene、Object、Initialization、Camera、Dynamics。系统消融结论：

**Critical**：
- Camera alignment（misaligned → 67% → 56% Panda，95% → 70% GR-1）
- Task semantic alignment（Prior 不对齐 task 仍有用，但 DC 对齐后更好）
- Simulation data 数量（10k → 500 导致 Panda 67% → 53%）

**Less Critical**：
- Dynamics alignment（Appendix VIII-M：调物理参数对 GR-1 CupPnP 无影响，95% → 95%）
- Perfect object geometry match（DC 用不同 instances 仍有效）
- Perfect scene match（Prior 完全不同 scene 仍有效）

**Intuition**：vision-based policy 主要从 visual feedback 学闭环控制，physics mismatch 被 visual servoing 吸收。Camera 必须对齐因为 visual perspective 直接影响 spatial reasoning——policy 看到物体在左边，但 real camera 在右边，spatial mapping 就乱了。

---

## 6. Recipe 总结（Section V-E）

Paper 给的 actionable recipe：

1. **Task/Scene**：尽量用 DC，但 Prior out-of-the-box 也有显著收益
2. **Object/Initialization**：sim 中尽量 diversify objects 和 positions，这直接 transfer 到 real 的 generalization
3. **Alignment**：task definition + camera viewpoint 要对齐，其他可松
4. **Hyperparameters**：sim data 要 orders of magnitude 多于 real，α 需要调（推荐 0.9-0.99）

这个 recipe 对 practitioner 极其友好。对比 sim-to-real 的传统 pipeline（system ID + domain randomization + retuning），co-training 的 setup 时间从几周降到几天。

---

## 7. Appendix 中的 Gem：Vid2Vid Enhancement

Appendix VIII-K 用 CogVideo-X 做 Vid2Vid style transfer，把 sim rendering 变得更 real-like。Noise strength 0.6 平衡 realism vs action label consistency。

Table VI 结果：
- Low-data regime（Real 20 + Sim 20）：48% → 70%（+22%）
- High-data regime（Real 20 + Sim 1000）：95% → 95%（+0%）

**Intuition**：visual realism 在 sim data 量少时重要（因为 policy 没有 enough sim samples 来 invariance-ize visual features）；sim data 量大时，policy 自己学到 visual invariance，realism boost 边际效益递减。

这暗示未来 **generative world model + co-training** 的 pipeline：用 video diffusion model 生成大规模 realistic synthetic demos，绕过 physics simulator 的 fidelity limit。NVIDIA Cosmos（引用 [66]）正是这个方向。

---

## 8. Architecture 细节（Appendix VIII-G）

两个 domain 用了不同的 Diffusion Policy variant：

**Panda Kitchen**：
- Transformer-based Diffusion Policy
- ResNet visual encoder
- Input：3 个 128×170 RGB views + proprioception（EE pose + gripper）
- Output：7-DoF action（delta EE + gripper）
- Batch size 256
- CLIP language encoder + FiLM conditioning（支持 multi-task）
- 默认 α=0.10 real / 0.90 sim（因为 sim demos 数量级远超 real）

**Humanoid Tabletop**：
- UMI-style Diffusion Policy
- ViT vision encoder
- U-Net diffusion backbone
- Input：1 个 first-person RGB + joint positions
- Output：arm + dexterous hand joint targets
- 默认 α=0.99（Figure 5 调参后发现 0.99 最佳）

**为什么两个 domain 用不同架构？** Panda 用 multi-task 训练（需要 language conditioning 区分任务），GR-1 单任务训练（不需要 language）。架构选择是 task-driven 的，co-training recipe 本身 architecture-agnostic。

---

## 9. 我的 Intuition & 联想

### 9.1 为什么 Co-Training 比 Sim-to-Real Transfer 宽容？
Sim-to-real transfer 训练时只有 sim data，policy 必须 generalize 到 unseen real distribution——这要求 sim 覆盖 real 的所有相关 axes（visual、dynamics、control）。Co-training 时 real data 持续在场，policy 在 training 中就 "see" 过 real distribution，sim data 只需要提供 auxiliary signal（motion skill、visual feature representation）。这相当于 **strong supervision + weak auxiliary** vs **only weak supervision**。

### 9.2 和 LLM Pretraining 的类比
- Prior sim data ≈ web-scale pretraining（diverse, noisy, 不对齐下游 task）
- DC sim data ≈ domain-specific pretraining（e.g. code pretraining for code LLM）
- Real data ≈ instruction tuning / RLHF
- α=0.99 ≈ pretraining loss weight >> instruction tuning loss weight

这个类比解释了为什么 α=0.99 最佳——大量 sim "pretraining" 建立 representation，少量 real "instruction" 做 alignment。OpenVLA、π0、GR00t N1 都在往这个方向走，这篇 paper 给了 sim data 角色的 quantitative study。

### 9.3 和 DROID、Open X-Embodiment 的关系
DROID（引用 [6]）是 large-scale real manipulation dataset，Open X-Embodiment（[65]）跨 embodiment 数据集。这些 dataset 收集成本极高。这篇 paper 暗示：**用 MimicGen 在 RoboCasa 里生成 100k demos 成本可能是 DROID 的 1/100**，加上少量 real demos co-train，能达到甚至超过 pure real 大数据训练的效果。这对 robot foundation model 的 data pipeline 设计有深远影响。

### 9.4 Limitation 中的 Open Question
Paper 提到 deformable objects 和 liquids 难以 simulate。但 Pouring 任务（球在杯里）仍然有效。这暗示只要 sim 能 approximate 关键 visual cue（球在杯里 vs 球倒出），policy 能用 visual feedback 弥补 physics gap。真正的 challenge 是 **visual cue 本身在 sim 中难以 render** 的情况（液体表面、布料褶皱）。

Vid2Vid 实验指向一个未来 path：**绕过 physics simulator，直接用 generative video model 生成 synthetic demos**。NVIDIA Cosmos、Genie（[67]）、GAIA-1（[68]）都是这个方向的基础设施。如果 video generation model 能 controllably 生成 action-labeled robot videos，simulation pipeline 就从 "physics + rendering" 变成 "video diffusion"，reality gap 问题从根本上消失。

### 9.5 Co-Training Ratio 的 Theoretical Interpretation
α 控制 effective sample size。如果 sim 和 real 来自不同 distribution $p_{\text{sim}}, p_{\text{real}}$，co-training 等价于在 mixture $\alpha \cdot p_{\text{sim}} + (1-\alpha) \cdot p_{\text{real}}$ 上做 MLE。Real deployment 的 target distribution 是 $p_{\text{real}}$。当 $p_{\text{sim}}$ 与 $p_{\text{real}}$ 有重叠且 sim 提供 informative prior 时，mixture MLE 在 real 上的 generalization 比 pure real MLE 好（因为 variance 减少 > bias 增加）。但 α 太大时 mixture 偏离 $p_{\text{real}}$ 太远，bias 主导，performance collapse。

这和 Bayesian inference 中 prior strength 的 tradeoff 完全同构。Sim data 是 prior，real data 是 likelihood，α 是 prior strength。Empirical Bayes 视角下，α 应该 tune 到 posterior 在 real test set 上 marginal likelihood 最大——这正好对应 paper 的 empirical tuning。

### 9.6 与 RLHF 的联系
如果考虑未来在 co-trained policy 上加 RL fine-tuning（e.g. 在 real 上做 PPO），co-trained policy 提供了一个更好的 initialization。这和 LLM 中 "pretrain → SFT → RLHF" pipeline 完全平行。Robot RL 的 sample efficiency 瓶颈可能通过 "co-trained initialization" 缓解。

---

## 10. Web Links for Reference

**核心工作**：
- Paper project page：https://co-training.github.io/
- RoboCasa (Nasiriany et al. RSS 2024)：https://robocasa.ai/
- MimicGen (Mandlekar et al. CoRL 2023)：https://mimicgen.github.io/
- DexMimicGen (Jiang et al. ICRA 2025)：https://dexmimicgen.github.io/
- Diffusion Policy (Chi et al. RSS 2023)：https://diffusion-policy.cs.columbia.edu/
- DROID (Khazatsky et al. RSS 2024)：https://droid-dataset.github.io/

**相关 Foundation Model 工作**：
- π0 (Physical Intelligence)：https://www.physicalintelligence.company/blog/pi0
- OpenVLA：https://openvla.github.io/
- GR00t N1 (NVIDIA)：https://developer.nvidia.com/groot
- RT-2 (Google DeepMind)：https://robotics-transformer2.github.io/
- Open X-Embodiment：https://robotics-transformer-x.github.io/

**Sim-to-Real 与 Digital Cousins**：
- Digital Cousins (Dai et al. 2024)：https://digital-cousins.github.io/
- Real2Sim2Real (Berkeley)：https://real2sim2real.github.io/
- Domain Randomization (OpenAI)：https://blog.openai.com/spinning-up-robotics/

**Generative World Models**：
- NVIDIA Cosmos：https://www.nvidia.com/en-us/ai/cosmos/
- Genie (DeepMind)：https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- GAIA-1 (Wayve)：https://wayve.ai/science/gaia-1/
- CogVideoX：https://github.com/THUDM/CogVideo

**Hardware & Tooling**：
- Fourier GR-1 Humanoid：https://www.fourierintelligence.com/gr-1
- UMI (Universal Manipulation Interface)：https://universal-manipulation-interface.github.io/
- Mink (MuJoCo IK)：https://github.com/kevinzakka/mink

---

## 11. 总结：这个工作的真正价值

这篇 paper 不是 algorithm paper，是 **empirical study paper**。它的价值在于：
1. **Quantify** 了 co-training 的 benefit（38% average improvement）
2. **Identify** 了哪些 alignment critical、哪些 optional
3. **Provide** actionable recipe 给 practitioner
4. **Surprise**：task-agnostic prior data 也有用，这降低 sim data 准备门槛

对 Andrej 你来说，这个工作和你的 "Software 2.0"、neural network as differentiable program 的视角很 align——policy 网络自己从数据中 learn invariance，sim 和 real 的 gap 由网络的 capacity 吸收，不需要 human engineer 手动设计 domain randomization 参数。这是 **data-centric AI 在 robotics 的具体体现**。

未来的研究方向（我猜测）：
- 自动 search α（用 meta-learning 或 BO）
- 用 VLM 自动 generate digital cousin scenes
- 把 Prior sim data 替换为 internet video data（cross-embodiment transfer）
- Hierarchical co-training：高层 plan 用 Prior，低层 control 用 DC

期待你之后的 tweet 或 lecture 谈这个方向！
