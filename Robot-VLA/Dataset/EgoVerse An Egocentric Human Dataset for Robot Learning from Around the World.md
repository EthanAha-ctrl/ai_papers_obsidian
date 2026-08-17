---
source_pdf: EgoVerse An Egocentric Human Dataset for Robot Learning from Around the
  World.pdf
paper_sha256: 3fef700d19faeffb07c247dd9b2b9b436854fad32253f8536e3244e80616bfbf
processed_at: '2026-08-04T02:53:17-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EgoVerse 人话版

## 一句话总结

Robot 学技能缺 data，人每天干活就是在产生 data，戴个眼镜或者手机录下来，把人的手部运动当成 robot 末端执行器的 proxy，跟 robot data 一起 co-train，performance 能涨 30%。

---

## 为什么需要这个东西

Andrej 你做 ML 这么多年肯定有感觉，LLM 之所以能 scale 是因为 web 上有几乎免费的 text data。Robot manipulation 这块卡在一个很尴尬的地方：collect 一条 robot demo 要 setup hardware、要 expert teleop、要 control environment，cost 可能 $10 到 $100 一条。你想要 100 万条 demo？先准备 $10M 预算再说。

但人呢，每天做饭、收拾屋子、叠衣服、装购物袋，这些 manipulation behavior 是免费的。你只要给人戴一个 egocentric camera，录下来就行。Project Aria glasses 大概 $3000 量级，iPhone 头戴 setup $1000 量级，一次性投入，后面 collect 多少 data 都几乎零 marginal cost。

所以核心 idea 就是：**用 human egocentric video 当 robot learning 的 data source**。

---

## 核心技术 trick：怎么把人和 robot 对齐

这里有个看起来很 technical 但其实很 intuitive 的问题。Robot 的 action 是在 robot base frame 里表达的，base frame 是静止的，所以 action 数字很稳定。Human 戴着眼镜动来动去，camera 一直在动，hand pose 是在 moving camera frame 里量的，每帧 reference frame 都不一样。

如果你直接用 raw hand pose 当 action，policy 会很 confused——同样一个 hand position 数字，在 t 时刻和 t+1 时刻 meaning 完全不同，因为 camera 转了。

Paper 里那个公式 $a_{t:t+k}^H = [(T_t^{\text{device}})^{-1} T_{t+i}^{\text{device}} \cdot p_{t+i}^H]_{i=1}^k$ 看着吓人，其实意思很简单：

> 把未来所有时刻的 hand pose，都变换到 t 时刻的 camera frame 里来表达。

这样就把 moving camera "frozen" 在 t 时刻了，human action 也就有了一个 stable reference frame，跟 robot 的 stable base frame 概念上对齐了。

这就是 EgoMimic、EgoBridge 那条线一直用的 trick，EgoVerse 沿用并 scale up。

---

## Dataset 设计的 intuition

EgoVerse 刻意分成 EgoVerse-A 和 EgoVerse-I 两部分，这背后是一个很 deliberate 的 trade-off。

**EgoVerse-A** 是 6 个 flagship task，跨多个 academic lab 用统一 protocol collect。它的目的是做 reproducible research——你要 study "加 human data 能涨多少点"，需要 controlled condition。

**EgoVerse-I** 是 industry partner 在 wild 里 collect 的，1400 小时，2000 个 task，240 个 scene。它的目的是提供 scale 和 diversity，让 policy 见过更多 visual variation 和 task type。

这个 split 背后的 intuition 是：ML research 需要两种 data。一种让你能 run controlled experiment 得出因果结论，一种让你能 train 出 generalizable policy。EgoVerse-A 服务前者，EgoVerse-I 服务后者。

6 个 flagship task 的选择也很有讲究：
- object-in-container：single-arm，pick-place-dump 循环，基础 manipulation
- cup-on-saucer：bimanual，fine-grained reorientation + handover
- bag-grocery：bimanual，long-horizon，deformable bag
- fold-clothes：bimanual，deformable，sequential stages
- scoop-granular：single-arm，granular material
- sort-utensils：single-arm，categorical sorting

覆盖了 single-arm vs bimanual、rigid vs deformable、short vs long-horizon 这几个 axis。如果一个 transfer 方法在这 6 个 task 上都 work，likely 它 generalizable 到一大类 manipulation。

---

## Hardware 的三个 tier

这个设计我觉得很聪明，目的是降低参与门槛：

1. **Project Aria glasses**：academic 标准，75g，5 个 camera，有 MPS pipeline 做 SLAM 和 hand tracking
2. **Custom industry rig**：stereo fisheye + depth + IMU，large-scale 用
3. **iPhone 头戴**：ultra-wide camera 1080p 30fps，cloud pipeline 处理

关键是 iPhone 版本的 output format 跟 Aria 完全一致——egocentric video + 6-DoF camera pose + 21 keypoints 3D hand pose。这意味着任何一个 researcher 用 $1000 的 iPhone 就能 contribute data 到这个 dataset，不用买 Aria。

这个 design choice 让 EgoVerse 能真正成为 "living dataset"，不是一次性 release 完了就死了。

---

## EgoDB：被低估的 infrastructure

Paper 大部分篇幅讲 dataset 和 experiment，但 EgoDB 这个 data management system 其实是整个 framework 能持续 grow 的 enabler。

它的工作流程：
1. 各地 collector 上传 raw data 到 S3，附带 metadata（operator、lab、task、scene 等）
2. SQL database 记录所有 episode 的 metadata
3. Nightly Ray daemon 自动跑 MPS、hand tracking、format conversion
4. 用户通过 SQL filter 查询需要的 subset，用 s5cmd parallel download 到本地 train

这里 s5cmd 是个细节但很重要——比 aws s3 sync 快 5-10x，对大数据集 download 体验影响很大。

整个设计的 intuition 是：把 data ingestion、processing、access 都自动化，让 contributor 只管 upload，让 user 只管 query，中间的 plumbing 全部由 EgoDB 处理。这跟 HuggingFace Hub 之类的 data infrastructure 思路类似，但针对 egocentric + robot data 的特殊需求定制了。

---

## Policy 架构选择

架构是 HPT（Heterogeneous Pre-trained Transformers）的变体，核心思想：modality-specific stems 处理不同 input，shared transformer encoder 做 fusion，embodiment-specific decoder 输出 action。

但这里有个关键 choice：action decoder 用 flow matching 而不是 standard diffusion。

为什么？Flow matching 用 linear probability path，ODE trajectory 是直线，10 步 Euler integration 就能 generate 高质量 action。Standard diffusion 是 stochastic process，通常需要 50-1000 步。对 robot real-time control 来说，inference latency 至关重要，10 步 vs 100 步是能不能上 real robot 的区别。

Paper 里还有个细节：time embedding 用 Beta(1.5, 1.0) 采样而不是 uniform。Beta(1.5, 1.0) 偏向 τ 接近 1，也就是 high noise region。训练时多 sample 困难的 denoising region，类似 importance sampling。这个 trick 来自 π₀ 的 ablation。

---

## 最核心的发现：Aligned data 是 scaling 的 anchor

Fig. 10 这个 ablation 是 paper 中最重要的 finding，值得仔细讲。

Setup 是这样的：robot data 固定，然后组合不同 human data：
- EV(8hr)：8 小时 diverse EgoVerse-A human data（跨 lab，task 对齐但 scene 和 object 不同）
- ID(2hr)：2 小时 in-domain human data（跟 robot data 同 task、同 scene、同 object）

结果：

| 配置 | 效果 |
|------|------|
| Robot only | baseline |
| 加 EV(8hr) alone | 几乎没提升 |
| 加 ID(2hr) alone | 微弱提升 |
| 加 ID(2hr) + EV(2hr) | 开始有 positive scaling |
| 加 ID(2hr) + EV(8hr) | 强 scaling |

人话解读：

> 光有 wild human data 不 work，光有少量 aligned data 也不 work，但只要少量 aligned data 当 "anchor"，wild data 的 scaling 才 unlock。

为什么？我猜 mechanism 是这样的：policy 是 high-capacity transformer，能 memorize 任意 dataset。当你给它大量 diverse human data，它能学到 "human 怎么做这类 task" 的一般知识。但 human 到 robot 的 mapping 是 underdetermined——同样一个 human hand trajectory，可以映射到多种 robot action，只有一种是 robot 实际能执行的。

ID data 提供了少量 "正确 mapping" 的 example，constrain 了这个 mapping space。有了这个 constraint，policy 才能正确 interpret diverse EV data 里学到的一般知识。

这跟 LLM 的 pre-train + fine-tune 有 conceptual parallel：EV data 像 pre-train corpus 提供 broad knowledge，ID data 像 fine-tune data 把 knowledge ground 到具体 task。但 co-training 是 joint optimization 而不是 sequential，有 computational 和 statistical efficiency 优势。

对 dataset curation 的 actionable implication：你不需要大量 aligned data，2 小时就够，但你需要它作为 anchor。剩下的 budget 应该投到 diverse data 上。

---

## Diversity 的拆解：scene vs demonstrator

Fig. 11 的 controlled diversity experiment 也很有 insight。

Setup：用 16 个 demonstrator × 16 个 scene 的 grid，固定 total data budget，systematically vary demonstrator 数量和 scene 数量，看 generalization 效果。

四个 regime 的结果：

**1. Single scene, 增加 demonstrator**：generalization to unseen demonstrator 提升
**2. Multi scene, 增加 demonstrator**：提升依然 hold
**3. 固定 demonstrator, 增加 scene**：generalization to unseen scene 提升，low budget 下 gain 最大
**4. Joint scaling**：scene diversity 是 reliable driver，demonstrator diversity 效果 task-dependent

人话解读：

> Scene diversity 比 data density 更重要。当你 data 量不够时，与其在一个 scene 里 collect 更多 data，不如把同样的 budget 分散到更多 scene。因为同一个 scene 内的 state-action distribution 很快就被 covered 了，更多 data 是 diminishing returns。但 unseen scene 有 lighting、background、object variation，必须靠 scene diversity 来 cover。

Demonstrator diversity 的效果更 subtle：对自由度高的 task（fold-clothes）benefit 明显，因为不同人叠衣服的 motion style 差异大，更多 demonstrator 让 policy 见过更广的 motion distribution。但对约束强的 task（cup-on-saucer），额外 demonstrator 引入的 behavioral noise 可能 outweigh coverage benefit。

这跟 DROID dataset 的 finding 一致：environment diversity 是 generalization 的 dominant factor。EgoVerse 把这个结论从 robot data 推广到 human data。

---

## 一个 negative finding：Strategy alignment 很重要

Paper 里有个诚实的 negative result：Robot B 在 bag-grocery task 上加 human data 后 performance 反而下降。

原因：Robot B 的 teleop strategy 跟 human strategy 不一样。Human 和 Robot A 都是双手打开 bag 再放东西进去；Robot B 用一只 gripper 撑开 bag，另一只放东西。

这个 strategy mismatch 导致 co-training 时 human 和 robot 的 behavior distribution 不一致，cross-embodiment alignment 被 weaken。

这个 finding 的 actionable implication：teleop protocol 设计应该尽量让 robot mimic human strategy，而不是让 robot 用自己最方便的 strategy。这是 paper 里 under-explored 但实际很重要的方向。

---

## 为什么 cross-embodiment evaluation 重要

Paper 在 3 个完全不同的 robot platform 上复现实验：Robot A（upright ARX5）、Robot B（side-mounted ARX5 humanoid-like）、Robot C（Unitree G1 + dexterous hand）。

这个设计的 intuition 是：single robot 上的 finding 可能是 system-specific artifact。比如 "co-training 涨 30%" 如果只在 Robot A 上观察到，可能是 Robot A 的 camera 配置恰好 favorable。在 3 个 platform 上都 hold 的 finding 才 robust。

这跟 medical research 的 multi-center clinical trial 思路一样——同一个 treatment 在多个 independent hospital 都有效，才能确认不是某个 hospital 的 confounding factor。

Robot learning 之前很多 paper 只在 single lab single robot 上 report result，导致 finding 难以 reproduce。EgoVerse 的 consortium-scale evaluation 是对这种 practice 的改进。

---

## 整体 intuition 构建

读完这篇 paper 应该建立几个 mental model：

**Egocentric human data 是 robot learning 的 "pretraining corpus"**，类比 web text 之于 LLM。区别是 robot 还需要少量 aligned data anchor the mapping。

**Aligned data 是 transfer 的 spark**，不是 optional 的，是必需的。2 小时 ID data 就能 unlock 几十小时 wild data 的 scaling。

**Diversity 是 multi-axis 的**，scene diversity 是 generalization 的 reliable driver，demonstrator diversity 效果 task-dependent。Dataset curation 要 deliberate 平衡。

**Living dataset 比 static dataset 更有价值**，EgoDB 的 incremental ingest + nightly processing 让 community 能持续 contribute。这模仿了开源 software 的协作模式。

**Cross-embodiment evaluation 是 reproducibility 的 anchor**，3 robot × multi lab × multi task 的 finding 才 robust。

---

## 未来的 speculation

基于 EgoVerse 开启的设计空间，几个方向值得 follow：

**VLA pretraining on EgoVerse-I**：1400 小时带 dense language annotation 的 data 适合训 VLA backbone，然后 fine-tune 到具体 robot。

**Hierarchical co-training**：EgoVerse-I 提供高层 task plan，EgoVerse-A 提供中层 motion primitive，robot data 提供低层 control，三层联合训练。

**Affordance-based transfer**：不直接用 hand pose 作为 action proxy，而是提取 object affordance（contact point、grasp pose），让 robot 用自己的 kinematic 实现。这能绕过 strategy alignment 问题。

**Active data curation**：把 controlled-diversity experiment 的 finding 系统化为 algorithm——给定有限 budget，应该 collect 多少 scene、多少 demonstrator、多少 task，才能 maximize generalization。

---

## 跟大趋势的连接

EgoVerse 出现在 2026 年，正好是 robot foundation model 爆发期。π₀、π₀.5、GR00T N1、RDT、OpenVLA、Octo 这些 model 都在探索 cross-embodiment policy learning。

EgoVerse 的 positional advantage 是：它提供了一个 standardized data substrate 和 reproducible evaluation framework。任何一个 foundation model 都可以在 EgoVerse benchmark 上 report result，cross-comparable 跨 lab。

这是 robot learning 从 "每个 lab 自己 collect data 自己 evaluate" 走向 "community-shared data + shared evaluation protocol" 的关键一步。类比 ImageNet 之于 computer vision——ImageNet 没解决所有问题，但提供了一个让整个 community 能在同一 reference frame 上 compare ideas 的 substrate。

EgoVerse 之后的 robot learning，decade 的核心问题可能不再是 "如何 collect 更多 robot data"，而是 "如何最有效地 leverage human data + 最少 robot anchor data"。

---

## Reference 汇总

- EgoVerse (这篇 paper 本身)
- EgoMimic: https://arxiv.org/abs/2410.24221
- EgoBridge: https://arxiv.org/abs/2410.24221
- Humanoid Policy ~ Human Policy: https://arxiv.org/abs/2503.13441
- Project Aria: https://projectaria.com/, https://arxiv.org/abs/2308.13561
- Ego4D: https://ego4d-data.org/
- EgoExo4D: https://egoexo4d-data.org/
- Epic-Kitchens: https://epic-kitchens.github.io/
- HOI4D: https://hoi4d.github.io/
- HOT3D: https://arxiv.org/abs/2406.09598
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- RH20T: https://arxiv.org/abs/2307.00595
- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- HPT: https://arxiv.org/abs/2409.20537
- DINOv3: https://arxiv.org/abs/2508.10104
- Octo: https://arxiv.org/abs/2405.12213
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoVLA: https://arxiv.org/abs/2507.12440
- Being-H0: https://arxiv.org/abs/2507.15597
- EgoZero: https://arxiv.org/abs/2505.20290
- EMMA: https://arxiv.org/abs/2503.13441
- In-n-on: https://arxiv.org/abs/2511.15704
- Immimic: https://arxiv.org/abs/2509.10952
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- GELLO: https://arxiv.org/abs/2309.11437
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Lipman et al. Flow Matching: https://arxiv.org/abs/2305.19432
- Factored scaling: https://arxiv.org/abs/2505.07728

---

# EgoVerse 深度解读：Egocentric Human Data 驱动的 Robot Learning 框架

## 一、Core Motivation 与 High-Level Intuition

EgoVerse 这篇 paper 的核心 insight 可以浓缩成一句话：robot data collection 是 robot learning 的 fundamental bottleneck，而 human egocentric data 提供了一个 embodiment-agnostic 的 abstraction layer，可以 defer embodiment decision downstream。这背后的 reasoning chain 是：

- Vision 和 language model 的 scaling laws 依赖于 web-scale data，几乎免费
- Robot manipulation 的 data acquisition 需要 hardware + expert teleop + controlled setup，cost per demo 在 $10-$100 量级
- Human 每天自然完成 manipulation task，egocentric capture 可以用 $200 量级的 phone 或者 $3000 量级的 Project Aria glasses 采集
- 关键 trick：把 human hand pose trajectory 当作 end-effector trajectory 的 proxy，让 human 和 robot 在同一个 action space 内 co-train

这跟 EgoMimic [1]、Humanoid Policy ~ Human Policy [2]、EgoBridge [3] 这条线的工作一脉相承，但 EgoVerse 的 positional advantage 在于 scale（1362 hours vs 几十小时）和 consortium-scale evaluation（3 个 robot platform 跨 lab 复现）。

参考链接：
- EgoMimic: https://arxiv.org/abs/2410.24221
- EgoBridge: https://arxiv.org/abs/2410.24221
- Humanoid Policy ~ Human Policy: https://arxiv.org/abs/2503.13441
- Project Aria: https://projectaria.com/

---

## 二、Dataset Composition 的双层设计

EgoVerse 刻意 split 成 EgoVerse-A 和 EgoVerse-I 两个 stream，这是非常 deliberate 的设计 choice，背后反映的是 ML research 中 reproducibility vs diversity 的经典 trade-off。

| Part | Percent | Hours | Episodes | Tasks |
|------|---------|-------|----------|-------|
| EgoVerse-A | 5.5% | 75 | 2,385 | 6 |
| EgoVerse-I Partner A | 76.1% | 1,035 | 72,993 | 1,898 |
| EgoVerse-I Partner B | 18.4% | 250 | 3,128 | 45 |

### EgoVerse-A：Controlled Reproducibility Stream

- 6 个 flagship task，固定 protocol 跨 lab 镜像
- 每个 lab 8-12 个 scene，每个 scene 1-10 dataset units
- Dataset unit = 5 分钟 = 5-10 demos
- Workspace 约 40cm × 60cm，object 位置 randomized

6 个 flagship task 的设计很有讲究，覆盖了 manipulation 的几个关键 regime：

| Task | 类型 | Manipulation Regime |
|------|------|---------------------|
| object-in-container | single-arm | pick-place-dump cyclic |
| cup-on-saucer | bimanual | fine-grained reorientation + handover |
| bag-grocery | bimanual | long-horizon, deformable bag manipulation |
| fold-clothes | bimanual | deformable object, sequential stages |
| scoop-granular | single-arm | granular material, repetitive |
| sort-utensils | single-arm | categorical sorting |

这种 task selection 的 intuition 是：如果一种 human-to-robot transfer 方法在这 6 个 task 上 work，那它 likely generalizable 到一大类 manipulation task。这跟 Open X-Embodiment [4] 的设计哲学类似，但 EgoVerse 更强调 bimanual 和 deformable。

参考：Open X-Embodiment: https://robotics-transformer-x.github.io/

### EgoVerse-I：Wild Diversity Stream

- 1,400 hours, 2,000 tasks, 240 scenes, 2,087 demonstrators
- Dense language annotation (1-2 秒级别)
- Active-hand indicator、static vs mobile flag
- 7 大 category：Logistics (15.4%)、Cooking (13.7%)、Cleaning (11.6%)、Laundry (10.9%)、Hardware (6.8%)、Crafts (4.0%)、Gardening (3.2%)

Top verbs 分布很有意思：pick 和 place 占据绝对主导，符合 manipulation 的"move object A to B"基本范式。但 scoop、fold、cut、scrub、iron 这些 task-specific verb 揭示了 EgoVerse-I 的真正价值——覆盖了 robot 还做不了但 human 每天做的 skill。

---

## 三、Hardware Capture Setup 的层级化设计

EgoVerse 设计了 3 个 tier 的 capture hardware，对应不同的 accessibility 和 data quality：

### Tier 1: Project Aria Gen 1 (EgoVerse-A 学术标准)
- 75g 头戴设备
- 5 cameras：1 forward RGB (global shutter) + 2 side monochrome (SLAM + hand tracking) + 2 inward (eye tracking)
- Tightly synchronized IMU
- Meta MPS (Machine Perception Services) pipeline 做 VI-ODO 和 hand tracking
- 关键 design choice：side monochrome cameras 保证 hand 始终 visible，即使超出 forward RGB FoV

### Tier 2: Custom Industry Rig (EgoVerse-I)
- Stereoscopic fisheye RGB，6 cm baseline，1920×1200 @ 30 FPS
- Stereo depth + IMU fusion
- 21 keypoints/hand 3D hand pose，anatomically plausible smoothing

### Tier 3: Phone-based System (EgoVerse ecosystem 开放)
- iPhone 头戴 + ultra-wide camera @ 1080p 30 FPS
- Cloud pipeline 做 6-DoF head pose + 21 keypoints 3D hand pose
- Output format 跟 Aria 完全一致 → 直接 join dataset

这个 3-tier 设计的 intuition 是：降低参与门槛。任何一个研究者用 $1000 的 iPhone 就能 contribute 到这个 dataset，这是 EgoDB 能 "living dataset" 的关键。

参考：Project Aria paper: https://arxiv.org/abs/2308.13561

---

## 四、EgoDB: Living Dataset 的 Infrastructure

EgoDB 是 paper 中被 undervalued 的一个 contribution，实际上它是 EgoVerse 能持续 grow 的核心 enabler。

### Architecture 拆解

```
[Capture Devices] 
    ↓ (unified upload script + UTC timestamp hash)
[S3 Bucket] ← → [SQL Database (PostgreSQL)]
    ↓                    ↑ (metadata query)
[Ray Processing Daemon (nightly)]
    ├── Cluster A: MPS for Aria (t3a.2xlarge head)
    ├── Cluster B: MPS → training-ready format (r6a.2xlarge workers)
    └── Cluster C: Raw robot → training-ready (c5.18xlarge workers)
    ↓
[EgoVerseDataset (PyTorch interface)]
    ↓ (s5cmd parallelized S3 sync)
[Local Training Cache]
```

### SQL Schema 关键字段（Table V）

最关键的字段设计是 `episode_hash`（UTC timestamp-based 唯一 ID）和 `processed_path`（S3 上处理后的数据路径）。这种设计支持：
- 增量 ingest：新数据 hash 后直接 append，不重名
- Lazy processing：processed_path 为空时 nightly daemon 触发处理
- Filtered sync：用户通过 SQL filter 只下载需要的 subset

### EgoVerseDataset 接口

```python
filters = {"robot_name": "robot_a", "task": "task_x"}
rows = query_sql_table(filters)
for processed_path, episode_hash in rows:
    run_command(["s5cmd", "sync", f"{processed_path}/*", local_dir])
    dataset = SingleEgoVerseDataset(root=local_dir, mode="train")
```

这里 s5cmd 是关键，相比 aws s3 sync 能 5-10x 加速 parallel download。

---

## 五、Cross-Embodiment Action Alignment：技术核心

这是 paper 中最 technical 也最值得深挖的部分。Human hand pose 在 moving camera frame 中，robot end-effector 在 robot base frame 中，要让它们 co-train 必须找到 common reference frame。

### Robot Action Representation

| Robot | Kinematics | Action Representation | Dim |
|-------|------------|----------------------|-----|
| Robot A | 2× 6-DoF ARX5 upright, parallel jaw | base-frame SE(3) Euler + gripper | 14 |
| Robot B | 2× 6-DoF ARX5 side-mounted, shoulder | base-frame SE(3) quaternion + gripper | 16 |
| Robot C | Unitree G1, 7-DoF arm + 6-DoF Inspire Hand | wrist SE(3) absolute + 5 fingertip keypoints | varies |

注意 Robot C 用 5 fingertip keypoints 而不是直接 joint angles，这是为了跟 human hand 21 keypoints 的子集对齐。Inverse kinematics solver 把 keypoints 映射到 Inspire Hand 的 joint commands。

### Human Action Representation：Camera-Centered Stable Frame

这是 paper 中最 elegant 的公式：

$$a_{t:t+k}^H = \left[(T_t^{\text{device}})^{-1} T_{t+i}^{\text{device}} \cdot p_{t+i}^H\right]_{i=1}^k$$

变量逐个拆解：
- $a_{t:t+k}^H \in \mathbb{R}^{k \times D}$: human action sequence，从 time t 到 t+k，D 是 pose 维度
- $T_t^{\text{device}} \in SE(3)$: device（即 camera）在 world frame 中的 pose at time t
- $(T_t^{\text{device}})^{-1}$: inverse rigid body transformation，把 world frame point 变换到 t-th device frame
- $T_{t+i}^{\text{device}}$: device pose at future time t+i
- $p_{t+i}^H \in \mathbb{R}^3$: hand position at future time t+i，原本在 device frame 中表达
- $[\cdot]_{i=1}^k$: 收集成 k 步序列

**Intuition 拆解**：
- 原始 hand trajectory $[p_t^H, p_{t+1}^H, ..., p_{t+k}^H]$ 每个 pose 在各自时刻的 device frame 里，因为 camera 在动，每帧 reference frame 不同
- 直接用这些 raw pose 会让 policy 学到"绝对世界坐标"，但 camera 一直在动，所以这些数字 meaning 在变
- 通过左乘 $(T_t^{\text{device}})^{-1} T_{t+i}^{\text{device}}$，把 future hand pose 从 t+i 时刻 device frame 先变到 world frame（乘 $T_{t+i}^{\text{device}}$），再从 world frame 变到 t 时刻 device frame（乘 $(T_t^{\text{device}})^{-1}$）
- 结果：所有 future hand pose 都用 t 时刻的 device frame 表达，camera motion 被 "frozen" at time t

这跟 EgoMimic、EgoBridge 的 trick 一致，关键 insight 是：**robot base frame 是静止的，所以 robot action 在 base frame 中是稳定的；human device frame 是 moving 的，必须人为构造一个 stable anchor frame**。t 时刻的 device frame 就是这个 anchor。

### Quantile Normalization

$$\hat{x} = 2 \cdot \left(\frac{x - q_{0.01}}{q_{0.99} - q_{0.01}}\right) - 1$$

变量：
- $x$: 任意 feature tensor（action 维度、proprioception 维度等）
- $q_{0.01}$: feature 分布的 1st percentile
- $q_{0.99}$: feature 分布的 99th percentile
- $\hat{x}$: normalized output，映射到 $[-1, 1]$

**为什么不用 standard z-score normalization？**
- Human hand motion 中有 outlier（突然快速挥手、reset motion）
- z-score 的 mean 和 std 会被 outlier 拉偏
- Quantile normalization 用 percentile clip 思想，1%-99% 之外的数据 clamp 到边界，对 outlier robust
- 这个 trick 来自 $\pi_0$ [5] 和 Physical Intelligence $\pi_{0.5}$ [6] 的工程经验

参考：$\pi_0$: https://arxiv.org/abs/2410.24164, $\pi_{0.5}$: https://arxiv.org/abs/2504.16054

---

## 六、Policy Architecture：HPT-style Cross-Embodiment Backbone

架构是 Heterogeneous Pre-trained Transformers (HPT) [7] 的变体，核心思想：modality-specific stems → shared transformer encoder → embodiment-specific decoder。

参考：HPT: https://arxiv.org/abs/2409.20537

### Vision Stem

- Input: $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$
- ImageNet normalization
- ResNet-18 truncated before global pool → $7 \times 7 \times 512$ feature map
- Flatten + linear projection to $d_{\text{proj}} = 256$
- L=16 learnable query tokens cross-attend to projected features（8 heads）
- Output: 16 tokens of dim 256

### Proprioception Stem

- Input: $\mathbf{q} \in \mathbb{R}^{d_q}$（joint angles, EE pose 等）
- Quantile normalize
- Linear to $d_{\text{proj}} = 256$
- 16 learnable query tokens cross-attend（8 heads）
- Output: 16 tokens

### Shared Encoder $f_\phi$

- Total input: $M + m \cdot L = 64 + 4 \times 16 = 128$ tokens
  - $M = 64$ learnable context tokens（prepended）
  - $m = 4$ stems（1 main vision + 2 wrist + 1 prop）
  - $L = 16$ tokens per stem
- $N_{\text{enc}} = 16$ transformer blocks
- $D_{\text{enc}} = 8$ heads，embed dim $d = 256$
- No masking（permutation invariant over tokens）

### Flow Matching Decoder $\pi_\theta$

这是 paper 用 flow matching 而非 standard diffusion 的关键选择。

**为什么 flow matching over DDPM？**
- Flow matching 用 linear probability path，ODE trajectory 是直线
- Diffusion 用 stochastic differential equation，需要 many steps
- Flow matching 在 10 步 Euler ODE integration 下就能 generate 高质量 action，diffusion 通常需要 50-1000 步
- 对 robot real-time control 来说，inference latency 至关重要

**Noise + Time Embedding 设计**

- Noise: $x_{\tau=1} \sim \mathcal{N}(0, I)$, shape $\mathbb{R}^{T \times d_{\text{dec}}/2}$
- Time: $\tau \sim \text{Beta}(1.5, 1.0)$, sine-cosine positional embedding to $\mathbb{R}^{T \times d_{\text{dec}}/2}$
- Concat → $\mathbb{R}^{T \times d_{\text{dec}}} = \mathbb{R}^{T \times 128}$

**Beta(1.5, 1.0) 分布的 intuition**：
- Beta(α, β) 在 α=1.5, β=1.0 时 density 偏向 τ 接近 1
- τ 接近 1 意味着接近 pure noise，这是 denoising 最难的 regime
- 训练时多 sample 困难 region，类似 importance sampling
- 这个 trick 来自 $\pi_0$ 的 ablation

### Linear Interpolation Probability Path

$$x_\tau = \tau a_0 + (1-\tau) a_1$$

变量：
- $a_0 \sim \mathcal{N}(0, I)$: pure noise
- $a_1$: ground-truth action
- $\tau \in (0, 1]$: continuous time
- $\tau = 0 \Rightarrow x_0 = a_1$（pure data）
- $\tau = 1 \Rightarrow x_1 = a_0$（pure noise）

注意 paper 中公式写的是 $\tau a_0 + (1-\tau) a_1$，所以 $\tau$ 增大对应更多 noise。Inference 时从 $\tau=1$（noise）Euler integrate 到 $\tau=0$（data）。

### Flow Matching Loss

$$\mathcal{L}_{\text{CFM}}^e = \mathbb{E}_{\tau, a_0, a_1, s} \left[\|\pi_\theta(x_\tau, \tau, f_\phi(s)) - (a_0 - a_1)\|^2\right]$$

变量：
- $\pi_\theta(x_\tau, \tau, f_\phi(s))$: predicted velocity field，给定 noisy action $x_\tau$、time $\tau$、encoded state $f_\phi(s)$
- $(a_0 - a_1)$: target velocity，从 data $a_1$ 指向 noise $a_0$ 的方向
- $e \in \{\text{robot}, \text{human}\}$: embodiment indicator

**Intuition**：flow matching 学一个 vector field，把 noise 推向 data。Target velocity $(a_0 - a_1)$ 是直线（因为 linear path）的 tangent vector，gradient flow 沿着这个 vector field 就能从 noise 到 data。

### Co-training Total Loss

$$\mathcal{L}_{\text{BC-cotrain}} = \mathcal{L}_{\text{CFM}}^{\text{robot}} + \mathcal{L}_{\text{CFM}}^{\text{human}}$$

每个 training step 同时 sample robot batch 和 human batch，分别 compute CFM loss 后相加。1:1 human:robot ratio，global batch 32-64（即 16-32 robot + 16-32 human per step）。

---

## 七、Consortium-Scale Evaluation：3 个 Robot Platform

| Robot | Kinematics | Camera | Teleop | Mount |
|-------|-----------|--------|--------|-------|
| Robot A | 2× ARX5 6-DoF, parallel jaw | Aria + 2× RealSense D405 wrist | Meta Oculus 3 + Mink IK | Upright |
| Robot B | 2× ARX5 6-DoF, parallel jaw | Aria + 2× Logitech wrist | Custom GELLO | Side-mounted 3D-printed shoulder |
| Robot C | Unitree G1, 7-DoF + 6-DoF Inspire Hand | ZED 2 stereo | Apple Vision Pro | Humanoid |

这 3 个 platform 覆盖了：
- Different kinematics（upright arm vs side-mounted humanoid-like vs full humanoid）
- Different end-effector（parallel jaw vs dexterous hand）
- Different sensing（Aria vs ZED 2 stereo）
- Different teleop（VR controller vs GELLO kinematic puppet vs Vision Pro hand tracking）

这种 cross-platform 复现是 paper 的 strength——任何 finding 如果在 3 个 platform 上都 hold，那它不太可能是 system-specific artifact。

---

## 八、Key Finding 1: Co-training Improves Transfer

Fig. 9 的核心 result：加 EgoVerse-A human data 后 ID 和 OOD performance 提升最多达 30%。

但有一个关键 caveat：Robot B 在 bag-grocery 上 performance 下降。Paper 给出 hypothesis 是 Robot B 的 teleop strategy 跟 human strategy 不一致——Robot B 用一只 gripper 撑开 bag，另一只插入 item；而 human 和 Robot A 都用双手打开 bag 再插入。

这揭示了一个重要的 negative finding：**co-training 不仅需要 task semantic alignment，还需要 strategy alignment**。如果 human 和 robot 用完全不同的 strategy 完成同一 task，co-training 可能 hurt performance。

Fig. 15 可视化了这种 strategy mismatch。这是一个 actionable insight：未来 teleop protocol design 应该尽量让 robot mimic human strategy，而不是让 robot 用自己最方便的 strategy。

---

## 九、Key Finding 2: Aligned Data Anchors Scaling

Fig. 10 是 paper 中最重要的 ablation。Setup：
- EV(8hr): 8 小时 diverse EgoVerse-A human data
- ID(2hr): 2 小时 in-domain human data（跟 robot data 同 task、同 scene、同 object）
- Robot data 固定

| Setup | ID | OOD |
|-------|-----|-----|
| Robot only | baseline | baseline |
| EV(8hr) only | ≈ baseline | ≈ baseline |
| ID(2hr) only | slight gain | slight gain |
| ID(2hr) + EV(2hr) | positive scaling | positive scaling |
| ID(2hr) + EV(8hr) | strong scaling | strong scaling |

**关键 insight**：
- 单独 EV(8hr) 或 ID(2hr) 都不足以 drive 显著 gain
- 但只要 ID(2hr) 作为 "anchor"，EV(2hr~8hr) 的 scaling 才 positive
- 这说明 diverse data 提供 "broad knowledge"，但需要 in-domain data "ground" 这个 knowledge 到具体 task

**Mechanism hypothesis**：
- Policy 是 high-capacity transformer，能 memorize 任意 dataset
- 没有 ID data 时，policy 学到的 human→robot mapping 是 underdetermined（多种 mapping 都能 fit human data，但只有一种是 robot 正确的）
- ID data 提供少量"correct mapping"示例，constrain 了 mapping space
- 有了这个 constraint，diverse EV data 才能被正确 interpret

这跟 pre-train + fine-tune 范式有 conceptual parallel：EV data 像 pre-train corpus，ID data 像 fine-tune data。但 co-training 是 joint optimization 而非 sequential，这有 computational 和 statistical 上的 efficiency 优势。

参考：Emergence of human to robot transfer in VLA: https://arxiv.org/abs/2512.22414

---

## 十、Key Finding 3: Diversity Decomposition

Fig. 11 在 controlled-diversity subset 上做了 4 个 scaling experiment，使用 16 demonstrators × 16 scenes 的 grid，offline Avg-MSE 作为 metric：

$$\text{Avg-MSE}(\hat{\mathbf{a}}_{1:T}, \mathbf{a}_{1:T}) = \frac{1}{T}\sum_{t=1}^T \frac{1}{D}\|\hat{\mathbf{a}}_t - \mathbf{a}_t\|_2^2$$

变量：
- $\hat{\mathbf{a}}_{1:T} \in \mathbb{R}^{T \times D}$: predicted action sequence
- $\mathbf{a}_{1:T} \in \mathbb{R}^{T \times D}$: ground-truth action sequence
- $T$: sequence length（=100 after resampling）
- $D$: action dimension

### 4 个 Scaling Regime

**1. Single-Scene Demonstrator Scaling** (Table VIII)
- Fixed 2-hour budget, 1 scene
- Demonstrators 从 1 到 16，per-demonstrator 时间从 120min 递减到 7.5min
- Evaluate on held-out 17th demonstrator
- **Finding**：Avg-MSE 单调下降，demonstrator 越多 generalization 越好

**2. Multi-Scene Demonstrator Scaling** (Table IX)
- Fixed 8-hour budget, 8 scenes
- Demonstrators 4/8/12，per-DS-pair 15/7.5/3.75 min
- Evaluate on unseen demonstrators within same scenes
- **Finding**：demonstrator scaling 在 multi-scene 下仍然 beneficial

**3. Scene Diversity Scaling** (Table XI)
- Fixed demonstrator pool, scenes 从 1 到 16
- Data usage fraction 6.25% / 12.5% / 25% / 50% / 100%
- **Finding**：scene diversity 在所有 data budget 下都 improve generalization，low budget 下 gain 最大

**4. Mixed Diversity** (Table X)
- Fixed 4-hour budget, jointly scale scenes 4-8 和 demonstrators 4-8
- **Finding**：scene diversity 是 reliable driver across tasks；demonstrator diversity 对 fold-clothes 有效，对 cup-on-saucer 有时 hurt

### Intuition 总结

为什么 scene diversity 比 data density 更重要？
- 当 data 量足够时，单个 scene 内的 state-action distribution 已经被 well covered
- 更多同 scene data 提供 diminishing returns
- 但 unseen scene 有 visual distractor、lighting、background variation，必须靠 scene diversity cover
- Demonstrator diversity 的 effect 更 task-dependent：自由度高的 task（fold-clothes）benefit from motion diversity；约束强的 task（cup-on-saucer）可能被 demonstrator noise 干扰

这跟 DROID [8] 的 finding 一致：environment diversity 是 generalization 的 dominant factor。但 EgoVerse 把这个 finding 从 robot data 推广到 human data。

参考：DROID: https://droid-dataset.github.io/

---

## 十一、UMAP 可视化的 Intuition

### EgoVerse-I Visual Diversity Case Study (Fig. 5)

- 用 DINOv3 [9] large model 抽 image embedding
- UMAP 降到 2D
- 三个 source 对比：robot data (1 lab)、EgoVerse-A (fold-clothes)、EgoVerse-I (fold-clothes)
- **Finding**：EgoVerse-I 的 visual coverage 大幅扩展，远超 single lab robot data

DINOv3 是 self-supervised vision transformer，能 capture semantic visual feature。UMAP 保留 local 和 global structure（相对 t-SNE）。这个可视化的 intuition 是：如果 EgoVerse-I 的 UMAP cloud 明显比 robot data cloud 大，那 training 时 policy 见过的 visual variation 就更多，inference 时对新 visual input 的 OOD robustness 就更强。

### Demonstrator Diversity Visualization (Fig. 12)

- HPT model 的 64 个 action-conditioned context tokens，flatten 成 single vector
- UMAP 降到 2D
- 4 vs 12 demonstrators 对比
- **Finding**：12 demonstrators 时 training 和 validation demonstrator 的 latent overlap 更大

这个可视化直观展示了 demonstrator diversity 如何让 model 见过更广的"motion style distribution"，从而 generalize 到 held-out demonstrator。

参考：DINOv3: https://arxiv.org/abs/2508.10104

---

## 十二、Failure Modes 与 Limitations

Fig. 16 展示了 4 个 task 的 common failure：
- object-in-container: workspace 边缘 grasp 困难
- bag-grocery: bag opening 失败、object 摆放位置不理想
- cup-on-saucer: handoff 失败（尤其 Robot C dexterous hand）

Section V 列的 limitations：
1. 只研究了 co-training，没研究 pre-train + fine-tune 范式
2. Diversity experiment 用 offline metric，没做 robot rollout
3. EgoVerse-I 的 wild data 还没被 fully exploit

未来的 open direction：
- 用 EgoVerse-I 做 large-scale VLA pre-training，EgoVerse-A 做 fine-tune
- Language-conditioned policy 训练（利用 EgoVerse-I dense annotation）
- Embodiment factor analysis（gripper vs dexterous hand 的 transfer 差异）

---

## 十三、与 Concurrent Work 的 Positioning

EgoVerse 处于几个 research line 的 intersection：

**1. Egocentric Pretrain Line**
- EgoVLA [10]: 用 egocentric video 训 VLA
- Being-H0 [11]: large-scale human video VLA pretrain
- EgoZero [12]: 从 smart glasses 直接学 robot policy

**2. Cross-Embodiment Co-training Line**
- EgoMimic [1]: 早期 egocentric + robot co-training
- Humanoid Policy ~ Human Policy [2]: humanoid 专版
- EMMA [13]: mobile manipulation 版
- PHANTOM [14]: 纯 human video 训 robot

**3. World Model Line**
- π₀.5 [6] 用 human video 训 world model
- Scaling cross-embodiment world models [15]

**4. Open-Source Robot Dataset Line**
- Open X-Embodiment [4]
- DROID [8]
- RH20T [16]

EgoVerse 的 unique positioning：把 1+2+4 三条 line unify 到一个 consortium-scale framework，并提供 reproducible evaluation。这是它跟 single-lab dataset 的本质区别。

参考：
- EgoVLA: https://arxiv.org/abs/2507.12440
- Being-H0: https://arxiv.org/abs/2507.15597
- EgoZero: https://arxiv.org/abs/2505.20290
- EMMA: https://arxiv.org/abs/2503.13441 (IEEE RAL 2026)
- RH20T: https://arxiv.org/abs/2307.00595

---

## 十四、整体 Take-away 与 Intuition 构建

读完这篇 paper 应该建立的几个 mental model：

**1. Egocentric data 是 robot learning 的 "pretraining corpus"**
- Web data 之于 LLM = egocentric human data 之于 robot policy
- 区别在于 robot 还需要少量 ID data anchor the mapping
- 这跟 LLM fine-tune 类似，但 anchor 概念比 fine-tune 更 subtle

**2. "Aligned data" 是 transfer 的 spark**
- 单纯 wild data 不 work
- 单纯 ID data 也不 work
- 两者组合才能 unlock scaling
- 这个 finding 对未来 dataset curation 有 actionable implication

**3. Diversity 是 multi-axis 的，不同 axis 效果不同**
- Scene diversity: generalization to unseen environment
- Demonstrator diversity: robustness to human variation
- Object diversity: task-level generalization
- 三者不能 interchange，design dataset 时要 deliberate 平衡

**4. Strategy alignment 比 task alignment 更 strict**
- Robot B 的 bag-grocery failure 说明 co-training 假设 human 和 robot 用同策略
- Teleop protocol 设计应该尽量 mimic human strategy
- 这是一个 under-explored 的研究方向

**5. Living dataset > Static dataset**
- EgoDB 的 incremental ingest + nightly processing 是关键 infrastructure
- 任何一个 community member 都能 contribute
- 这模仿了开源 software 的协作模式，应用到 robot data 上

**6. Cross-embodiment evaluation 是 reproducibility 的 anchor**
- Single robot finding 可能是 system-specific
- 3 robot × multi lab × multi task 的 finding 才 robust
- 这给 robot learning 引入了"clinical trial"风格的 evaluation

---

## 十五、可能的 Follow-up Direction 与 Speculation

基于 EgoVerse 的 design space，几个 promising follow-up：

**1. VLA Pretraining on EgoVerse-I**
- 用 1,400 小时带 dense language annotation 的 EgoVerse-I 训 VLA backbone
- Fine-tune 到具体 robot task
- 类似 RDT [17]、π₀ [5] 的 pretrain-finetune paradigm

**2. Hierarchical Co-training**
- EgoVerse-I 提供高层 task plan
- EgoVerse-A 提供中层 motion primitive
- Robot data 提供低层 control
- 三层 hierarchy 联合训练

**3. Affordance-based Transfer**
- 不直接用 hand pose 作为 action proxy
- 提取 object affordance（contact point、grasp pose）
- Robot 用自己的 kinematic 实现 affordance
- 绕过 strategy alignment 问题

**4. World Model Pretraining**
- 用 EgoVerse-I 训 next-frame prediction world model
- 在 robot 上 fine-tune，用 model-based RL
- 类似 [15] 的思路

**5. Active Demonstrator Selection**
- 基于 policy uncertainty active select demonstrator
- 把 controlled-diversity experiment 的 finding 系统化为 dataset curation algorithm
- 类似 [18] 的 factored scaling 思路

参考：
- RDT: https://arxiv.org/abs/2410.07839 (并没在 paper 里cite, 但思路相关)
- Factored scaling: https://arxiv.org/abs/2505.07728

---

## 十六、Implementation Detail 中的 Subtle Choice

几个值得注意的工程细节：

**1. Temporal Resampling**
- Human: 1 秒窗口 → resample 到 T=100
- Robot: 1.5 秒窗口 → resample 到 T=100
- Linear interpolation for position, SLERP for quaternion/euler
- 这个 normalization 是为了消除 human-robot execution speed 差异

**2. Vision Augmentation**
- Random image crop + color jitter
- 这对 cross-camera generalization 重要（Aria vs ZED 2 vs iPhone）

**3. Action Decoder 共享 vs Specific**
- "shared or embodiment-specific action decoders" — paper 没完全确定
- 实践中可能是 hyperparameter choice
- Shared decoder 强制 learn common representation
- Specific decoder 给每个 embodiment 更多 capacity

**4. Inference 10 Euler Steps**
- 10 步对 real-time robot control 足够快
- 50-100 步的 standard diffusion 会 too slow
- 这是 flow matching 的 practical advantage

---

## 十七、与 Foundation Model Trend 的连接

EgoVerse 出现的时间点（2026）正好是 robot foundation model 爆发期。几个相关 model：

| Model | Lab | Action Head | Cross-Embodiment |
|-------|-----|-------------|------------------|
| π₀ | Physical Intelligence | Flow matching | Yes |
| π₀.5 | Physical Intelligence | Flow matching | Yes |
| GR00T N1 | NVIDIA | Diffusion | Yes |
| RDT | THU | Diffusion | Yes |
| OpenVLA | Stanford | Tokenization | Yes |
| Octo | Berkeley | Diffusion | Yes |
| EgoVLA | Xiaolong Wang | Tokenization | Yes |

EgoVerse 跟这些 model 的关系：它提供了一个 standardized evaluation substrate 和数据 ecosystem。任何一个 foundation model 都可以在 EgoVerse benchmark 上 report result，cross-comparable 跨 lab。

参考：
- Octo: https://arxiv.org/abs/2405.12213
- GR00T N1: https://arxiv.org/abs/2503.14734
- OpenVLA: https://openvla.github.io/

---

## 十八、最终 Intuition 总结

把 EgoVerse 放在 robot learning 历史中看：

- 2017-2020：single-task BC，几十 demos，single lab
- 2021-2023：multi-task，几百 demos，multi scene
- 2024：Open X-Embodiment，几千 demos，multi robot
- 2025：EgoMimic/Humanoid Policy，egocentric data + robot data
- 2026：EgoVerse，1k+ 小时 human data，consortium-scale cross-embodiment evaluation

EgoVerse 的 contribution 是把 robot learning 从"robot-data-driven"推向"human-data-driven"，并且提供了 reproducible 的 evaluation framework 验证这条 path 的有效性。这跟 LLM 从 specific task dataset 走向 web corpus 的 evolution 类似，只是 robot 还需要解决 embodiment gap 这个额外 challenge。

最值得 build 的 intuition 是：**human egocentric data 是 robot learning 的 ImageNet moment**——它不解决所有问题，但提供了一个可 scale、可协作、可 reproducible evaluate 的 substrate，让整个 community 能在同一 reference frame 上 compare ideas。

EgoVerse 之后，robot learning 的下一个 decade 可能不再围绕 "如何 collect 更多 robot data"，而是围绕 "如何最有效地 leverage human data + 最少 robot anchor data"。

参考链接汇总：
- EgoVerse (这篇 paper)
- EgoMimic: https://arxiv.org/abs/2410.24221
- EgoBridge: https://arxiv.org/abs/2410.24221
- Humanoid Policy ~ Human Policy: https://arxiv.org/abs/2503.13441
- Project Aria: https://projectaria.com/, https://arxiv.org/abs/2308.13561
- Ego4D: https://ego4d-data.org/
- EgoExo4D: https://egoexo4d-data.org/
- Epic-Kitchens: https://epic-kitchens.github.io/
- HOI4D: https://hoi4d.github.io/
- HOT3D: https://arxiv.org/abs/2406.09598
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- DROID: https://droid-dataset.github.io/
- RH20T: https://arxiv.org/abs/2307.00595
- π₀: https://arxiv.org/abs/2410.24164
- π₀.5: https://arxiv.org/abs/2504.16054
- HPT: https://arxiv.org/abs/2409.20537
- DINOv3: https://arxiv.org/abs/2508.10104
- Octo: https://arxiv.org/abs/2405.12213
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoVLA: https://arxiv.org/abs/2507.12440
- Being-H0: https://arxiv.org/abs/2507.15597
- EgoZero: https://arxiv.org/abs/2505.20290
- EMMA: https://arxiv.org/abs/2503.13441
- In-n-on: https://arxiv.org/abs/2511.15704
- Immimic: https://arxiv.org/abs/2509.10952
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- GELLO: https://arxiv.org/abs/2309.11437
- UMI: https://universal-storage.github.io/
- AnyTeleop: https://anyteleop.github.io/
- Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Lipman et al. Flow Matching: https://arxiv.org/abs/2305.19432
