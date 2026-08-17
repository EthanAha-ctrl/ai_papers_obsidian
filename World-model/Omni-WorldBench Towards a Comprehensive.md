---
source_pdf: Omni-WorldBench Towards a Comprehensive.pdf
paper_sha256: e2165f404ea80f87be3a6561c957458d5ad21e11d9d2e23e7a40ea0e494032db
processed_at: '2026-08-05T23:06:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲 Omni-WorldBench

参考链接:
- VBench 项目主页: https://vchitect.github.io/VBench-project/
- WorldScore paper: https://arxiv.org/abs/2504.00983
- Wan 系列技术报告: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Cosmos (NVIDIA): https://arxiv.org/abs/2501.03575
- Genie 2 (DeepMind): https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

---

## 一句话讲完整个 paper

**大家都在喊 world model,但根本没人正经评测过 world model 该有的能力——这篇 paper 就来填这个坑。**

具体填法:整了 1068 个 prompt,按 interaction 复杂度分三层,又整了一套用 VLM 当裁判的打分系统,把市面 18 个模型拉过来跑一遍,结果发现这些号称 world model 的东西在"动作引起状态变化"这件事上其实很烂。

---

## 为什么要搞这个 benchmark?背景讲清楚

World model 这个词最近被滥用了。Sora 出来的时候 OpenAI 自己喊 world simulator,后续一堆 video diffusion 模型也跟着喊自己是 world model。但有一个尴尬的事实:

**FID、FVD 这类老 metric 只能告诉你"这视频看着像不像真的",告诉你不了"这个模型懂不懂物理世界怎么运转"。**

举个直觉例子:
- 给模型一个 prompt:"一个人把玻璃杯推下桌子,杯子摔碎"
- 模型 A 生成:杯子掉地上,没碎,但画面超美,FID 分爆高
- 模型 B 生成:杯子掉地上,碎了,但画面有点糊,FID 分一般

按老 metric,模型 A 赢。但按 world model 的逻辑,模型 B 才对——它懂"玻璃杯 + 摔落 = 碎"这个因果。

现有 benchmark 的问题:
- **VBench / VBench++**: 衡量 visual fidelity 和 text-video alignment,压根没碰 causality
- **WorldScore**: 前一阵刚出的 world model benchmark,但只考虑 camera motion 这一种"interaction",太窄。你给它个 robot 抓瓶子的 prompt 它测不出啥
- **WorldModelBench**: 也是类似 benchmark,但 coverage 远不如 Omni-WorldBench

所以这篇 paper 的 motivation 很直白:**world model 的核心是 interaction response,但没人正经测过这个能力,我来测**。

---

## Omni-WorldSuite:1068 个 prompt 怎么来的

### 三层 interaction——这是整个 benchmark 的灵魂

按"动作影响范围"分三层,这个分层简单到一句话能说明白,但 captures 了 world model 的 complexity ladder:

| Level | 大白话 | 例子 | 难在哪 |
|-------|--------|------|--------|
| **Level 1** | 动作只影响自己,不碰别人 | 看球往晶体球里看,沿河走 | 单 object 自己怎么动 |
| **Level 2** | A 动作直接作用于 B | 火堆加热金属棒,自动驾驶车与交通流交互 | 两 object 之间的 pairwise 因果 |
| **Level 3** | 动作引发连锁反应,改全局 | 掰意大利面、整理房间、机械臂抓瓶子给人 | 多 object 串成因果链 |

直觉:**Level 越高,模型要在 spatiotemporal 上 jointly 维护的东西越多**。Level 1 只要盯住一个 object,Level 3 要同时盯住多个 object 之间的连锁因果关系。

为什么 Level 2 prompt 数量最多? 因为 Level 1 太简单(现在 T2V 模型都能做),Level 3 太难(大多数模型直接崩),Level 2 是当前模型能力的 sweet spot,最能区分 model 之间差异。这是个非常 pragmatic 的 design choice。

### 数据从哪来?两条 pipeline

**Pipeline 1: 从现成 dataset 抽**
- Autonomous Driving: DriveLM 数据集抽第一帧 + 真实 camera trajectory
- Embodied Robotics: InternData-A1 抽机械臂操作任务
- Gaming: Sekai 数据集抽高动态场景
- 抽完用 Qwen-VL 生成 caption,人工 verify/refine

**Pipeline 2: 用 LLM 合成**
- 先建 prototype taxonomy: scene × object × action × interaction level
- ChatGPT-5.2 生成 textual prompt + camera trajectory
- Gemini 和 DeepSeek-R1 互相 cross-check
- 人工 refine
- FLUX.1-dev 生成 first frame,CFG scale=3.5,50 sampling steps,人工筛
- 不合格重写 prompt,必要时用 Qwen-Image 修 artifact

这里有个工程细节很关键:**第一帧的物理合理性**。如果第一帧里机械臂的关节角度就不对,后面模型生成啥都白搭——garbage in garbage out。所以他们对 first frame 做了严苛 QC:最低 1024×1024,物理合理,prompt 一致,interactive object 必须清晰可见。

### 还标注了一堆 auxiliary metadata

每个 prompt 还附带:
1. **Entity 分组**: 哪些 entity 是 affected set,哪些是 unaffected set,affected entity 标注预期运动方向和幅度
2. **Event timeline**: 按时序排好的关键事件 list
3. **Camera motion**(子集): 标注预期相机运动,还有"走一圈回原点"的 hard case

这些 metadata 是后面 metric 计算的 ground truth,没有它们就没办法自动打分。

---

## Omni-Metric:怎么自动打分

### 先把视频"结构化"

拿到模型生成的视频 $V \in \mathbb{R}^{T \times H \times W}$(T 帧,H 行,W 列):
- 用 GroundingDINO + SAM 提每个 entity 的 mask sequence,当 trajectory 用
- 用 RAFT 算 optical flow
- 用 optical flow 变化近似相邻帧的 camera motion

### 三个维度打分

#### 维度 1: Generated Video Quality(老 metric 凑一起)

借 VBench 的 imaging quality、temporal flickering、motion smoothness、dynamic degree,加 WorldScore 的 content alignment。这部分没啥新意,legacy metric 凑个场。

#### 维度 2: Camera-Object Controllability

**Camera Control**: 沿用 WorldScore 做法,把 trajectory error 拆成 rotation + translation,各自 normalize 后平均。

**Object Control**: 这里有改进。WorldScore 原来用 GroundingDINO 检测 + rule-based text matching,但 rule-based matching 对 synonymy 很脆弱——"cup" 和 "mug" 匹配不上,模型就吃亏。Omni-WorldBench 改成 VQA:

$$
\text{Object Control} = \frac{1}{K} \sum_{i=1}^{K} \hat{y}_i
$$

- $K$: prompt 里指定的 object 总数
- $\hat{y}_i \in \{0, 1\}$: VLM 对"第 $i$ 个 object 是否出现在视频里"的 binary 回答
- 等于算 VLM semantic recall

直觉:**rule-based matching 在 synonymy / compositional cue 上太脆,VLM 对这些天然 robust**。代价是 VLM 自己会 hallucination,但大模型 binary judgment 的方差比 brittle rule 小得多。

**Transitions Detect**: 用 PySceneDetect 检测 scene cut。如果视频里有一镜到底多个 scene cut,直接 0 分。直觉:很多 T2V 模型偷偷 cutscene 来规避长程一致性,这是 cheating。

#### 维度 3: Interaction Effect Fidelity(真正的贡献)

这是 Omni-WorldBench 区别于所有前作的灵魂部分,四个 sub-metric 各打一个 specific failure mode。

**InterStab-L (Long-horizon stability)**

针对场景: prompt 说相机走一圈回原点。revisit pair $\mathcal{R} = \{(t_a, t_b)\}$ 表示 $t_a$ 和 $t_b$ 应该看到相同的 world state。

对每对 $(i, j)$:
$$
s(i, j) = \frac{1}{2}\left(\text{SSIM}_{\text{gray}}(I_i, I_j) + \cos(\phi(I_i), \phi(I_j))\right)
$$

- $I_i, I_j$: 第 $i$ 帧和第 $j$ 帧
- $\text{SSIM}_{\text{gray}}$: 灰度 SSIM,量化低层 structural fidelity
- $\phi(\cdot)$: CLIP visual tower,把 frame 映到 semantic feature
- $\cos$: cosine similarity,量化高层 semantic 一致性

然后:
$$
\text{InterStab-L} = \frac{1}{|\mathcal{R}|} \sum_{(t_a, t_b) \in \mathcal{R}} s(i(t_a), i(t_b)) \cdot \mathbb{I}_{\text{dynamic}}
$$

**最关键的是 $\mathbb{I}_{\text{dynamic}}$ 这个 gating**。如果没有它,模型生成完全静止的视频就能刷高 SSIM 和 CLIP similarity。所以他们在视频 4 个 anchor 位置算相似度,如果平均超过 $\tau_{\text{static}}$,说明模型在 cheat,整个 metric 直接归零。

直觉:**anti-degeneracy gating**。就像 contrastive loss 里的 hard negative mining,防止模型走 trivial solution。

**InterStab-N (Non-target region stability)**

直觉超 sharp:**如果 prompt 要求抓一个瓶子,瓶子周围的桌子、背景墙应该几乎不动**。如果非 target region 也在剧烈变化,说明模型的 causal localization 失败了——interaction effect 没被正确"局部化",整个画面都在乱抖。

非 target region $\mathcal{N}$ 的 motion energy:
$$
E_{\text{non}}(s) = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{|\mathcal{N}|} \sum_{x \in \mathcal{N}} \|\text{Flow}_t(x)\|
$$

- $T$: 总帧数
- $|\mathcal{N}|$: 非 target region 像素数
- $\text{Flow}_t(x)$: 位置 $x$ 在帧 $t$ 的 optical flow vector
- $E_{\text{non}}$: 单位时间内非 target region 的平均 motion energy

映射到 $[0, 1]$:
$$
\text{InterStab-N}(s) = \exp\left(-\frac{E_{\text{non}}(s)}{\beta \cdot \min(H, W)}\right)
$$

- $\beta$: scaling factor
- $\min(H, W)$: 用视频短边做 resolution normalization
- $\exp(-x)$: motion energy 越小,分数越接近 1

为什么用 $\min(H, W)$ 而不是 $H \times W$? 因为 optical flow magnitude 是 pixel-level displacement,和 spatial dimension 是 linear scaling 关系(图片放大一倍 flow 也放大一倍),不是 quadratic。用短边是因为 camera motion 两个方向都有,取短更保守。

**InterCov (Interaction coverage)**

VLM-based semantic verifier 检查每个 target entity $o$ 行为是否符合 prompt 规定的 interaction logic:
$$
\text{InterCov} = \frac{1}{|\mathcal{O}|} \sum_{o \in \mathcal{O}} \mathbb{I}(v_o = 1)
$$

- $\mathcal{O}$: prompt 里所有 target entity 集合
- $v_o \in \{0, 1\}$: VLM 对 entity $o$ 是否符合 interaction logic 的 binary 判断
- $\mathbb{I}$: 指示函数

这就是 semantic recall。**它和 InterStab-N 互补**: InterStab-N 查"是不是动了不该动的",InterCov 查"该动的是不是动了"。

**InterOrder (Event ordering)**

prompt 描述因果链 event sequence $\mathcal{E} = \{e_i\}_{i=1}^K$,模型必须按时序生成。用 VLM 做 pairwise ordering check,对 $C(K, 2) = \frac{K(K-1)}{2}$ 个 event pair $(e_m, e_n)$ ($m < n$) 检查 generated sequence 中 $e_m$ 是否 precede $e_n$:

$$
\text{InterOrder} = \frac{2 K_s}{K(K - 1)}
$$

- $K$: 总事件数
- $K_s$: ordering 一致的 pair 数
- $\frac{K(K-1)}{2}$: 全部 pair 数

直觉:**因果一致性本质是 event chronology 的保持**。模型搞混"杯子掉下去 → 摔碎"和"摔碎 → 杯子掉下去"就是因果反转,即使单帧生成得再漂亮也是错的。

### AgenticScore: 怎么把 15 个 metric 合成一个总分

朴素做法是简单 average。问题:**不同 prompt 关注的 dimension 不同**。
- 纯 camera motion prompt 应该 weight Camera Control 高
- physical interaction prompt 应该 weight Interaction Effect Fidelity 高
- text-to-video prompt 没有 camera trajectory,Camera Control 应该 de-emphasize

Omni-WorldBench 的做法:
1. 三个 evaluation agent:
   - $A_I$ = (InterStab-L + InterStab-N + InterCov + InterOrder) / 4 (Interaction Effect Fidelity)
   - $A_G$ = Generated Video Quality 的 sub-metric 均值
   - $A_C$ = Camera-Object Controllability 的 sub-metric 均值
2. 一个 MLLM aggregation agent 分析 prompt 语义,输出三个维度的 importance ranking
3. Ranking 映射到 predefined weight 系数 $w_1, w_2, w_3$
4. Final score:
$$
\text{AgenticScore} = w_1 A_I + w_2 A_G + w_3 A_C
$$

直觉:**MLLM-as-aggregator**,weight 不是固定的,是 prompt-conditioned 的,可以处理 prompt diversity。这是现在越来越流行的 design pattern(参考 LLM-as-Judge 系列)。风险是 MLLM 的 ranking 不一定稳定,predefined weight 系数本身是个 hidden hyperparameter。

---

## 实验跑出来啥结果

### 18 个模型分类

- **T2V** (4个): Director3D, OpenSoraPlan, T2V-Turbo, HunyuanVideo
- **IT2V** (7个): Matrix-Game2.0, Wan2.1, Wan2.2, CogVideo, OpenSora, Cosmos, LargeVideoPlanner
- **Camera-conditioned** (7个): HunyuanWorld, HunyuanGameCraft, ViewCrafter, Gen3C, Lingbot, FantasyWorld, WonderWorld

### 总分排名

- **全场最高**: Wan2.2 (75.92%) → Cosmos (75.42%)
- **T2V 组最高**: HunyuanVideo (73.96%)
- **Camera 组最高**: HunyuanWorld (74.36%), WonderWorld 紧随 (74.02%)

### 几个关键发现

**发现 1: Visual quality metric 已经 saturate**

Temporal Flickering 和 Motion Smoothness 几乎所有模型都 > 95%。**真正的 differentiator 是 Dynamic Degree**,从 16.83 (OpenSoraPlan) 到 100 (ViewCrafter, WonderWorld) 都有。这说明 community 应该把注意力从 "smoothness" 转向 "dynamic responsiveness"。

**发现 2: WonderWorld 的诡异 trade-off**

WonderWorld 在 InterStab-L 拿 84.96%,但 InterStab-N 只有 24.89%。这个对比超 informative:

直觉解读:**3D-aware 生成模型可以维持 long-horizon spatial consistency(回到原位看到的场景一致),但在 camera sweep 过程中整个 scene 都在 "jitter"**。它的 stability 是"全局静态的",local dynamic 反而被压制了。3D priors 帮它维持 revisit 一致性,但同时压制了真正应该发生的局部 interaction dynamics。

这是 paper 中最有信息量的 single data point,指向一个 deep architectural issue:**3D-aware world model 牺牲了 local interaction dynamics 来换取 global spatial consistency**。

**发现 3: ViewCrafter 的 3D hallucination**

Fig. 6 qualitative comparison 里,ViewCrafter 在 camera sweep 过程中凭空生成一个 building。这是经典的 **3D prior hallucination**:基于 sparse 3D reconstruction 做新视角生成,当 camera 移到 training view 没覆盖的区域,reconstruction 稀疏,inpainting 模型就 invent 出不存在的 structure。

**发现 4: Matrix-Game2.0 在复杂 interaction 下崩**

Fig. 5 棒球投手案例。Wan2.2 成功合成完整投球动作,Matrix-Game2.0 最后几帧 human figure 完全 collapse 消失。本质:**autoregressive frame generation 在长序列下 drift,且没有 strong physical prior 来 anchor human pose**。

**发现 5: 主要 trade-off 对立关系**

| 强项 | 弱项 | 模型代表 |
|------|------|---------|
| Camera Control 强 | Interaction fidelity 弱 | WonderWorld, ViewCrafter |
| Long-horizon revisit 一致 | Local interaction dynamics 弱 | WonderWorld (InterStab-L 高 vs InterStab-N 低) |
| Visual quality 强 | Causal ordering 弱 | 大多数 T2V 模型 |
| Visual quality 强 | Object control 弱 | T2V-Turbo, OpenSoraPlan |

---

## 我对这篇 paper 的整体评价

### 强项

1. **Level 1/2/3 interaction hierarchy** 是一个 clean 的 complexity ladder,能做 curriculum learning,也能诊断 model 在哪个 complexity level 崩
2. **InterStab-L 的 dynamic gating** 是 metric 设计的关键 robustness trick,显示作者对 evaluator hacking 有警惕
3. **InterStab-N 用 non-target region motion energy** 检查 causal localization——大多数 evaluator 只看 "target 对不对",很少有人看 "非 target 应不应该动",这个 insight 非常 sharp
4. **VLM-as-aggregator** 解决了 fixed-weight aggregation 的 fragility,是合理的 design pattern

### 潜在 issue

1. **VLM-based metric 的 reliability**: InterCov 和 InterOrder 都依赖 VLM binary judgment,VLM 本身的 hallucination 会被 propagate 进 metric。文中提到做了 human alignment study,但只在 appendix
2. **AgenticScore 的 weight 系数**: MLLM 输出 ranking 映射到 "predefined weight coefficients" $w_1, w_2, w_3$,但没说具体数值,是个 hidden hyperparameter
3. **Camera motion estimation 精度**: 用 optical flow 变化近似 camera motion,在 dynamic scene 下误差很大,因为 flow 同时包含 ego-motion 和 object motion,separation 是 ill-posed
4. **Level 3 prompt 数量占比**: Level 2 > Level 3 > Level 1,但 Level 3 才是真正考验 multi-entity causal 的,数量可能不够
5. **Initial frame prior bias**: FLUX.1-dev + Qwen-Image 生成的 first frame 可能 carry FLUX 的 artifact distribution,被 evaluate 的 model 如果在 similar 分布上 train 过会有 unfair advantage

### 这篇 paper 揭示的 research direction

1. **Causal grounding for video diffusion**: diffusion 的去噪过程没有 explicit causal model,导致 long-horizon event ordering 弱。需要把 causal structure 注入 noise schedule 或 latent dynamics
2. **Decoupling camera motion from object dynamics**: WonderWorld 的灾难性 InterStab-N 说明 3D-aware generation 把 camera 和 object dynamics 强 coupling 了,需要 disentangle
3. **Causal localization**: InterStab-N 揭示模型不知道 interaction 应该 localized 在哪里,需要 explicit attention grounding 机制(类似 SAM 之于 segmentation)
4. **Long-horizon revisit vs local dynamics trade-off**: 3D prior model 的 inherent trade-off,可能需要 hybrid architecture(3D prior for global layout + 2D dynamics for local interaction)
5. **Dynamic Degree 成为 new bottleneck**: Temporal Smoothness 和 Flickering 已经 saturate,该把注意力从 "smoothness" 转向 "dynamic responsiveness"

---

## 一句话总结

**Omni-WorldBench 把 world model evaluation 从 "video generation quality" 推进到 "interaction-conditioned dynamics fidelity",实验结果证明当前模型 visual fidelity 接近 saturate 但 action-conditioned dynamics 和 causal coherence 仍是 bottleneck,WonderWorld 在 InterStab-L (84.96) vs InterStab-N (24.89) 的剧烈 trade-off 是最有信息量的 single data point,指向 3D-aware world model 牺牲了 local interaction dynamics 来换取 global spatial consistency 这个 deep architectural issue。**

如果想看 paper 原文,搜 arXiv "Omni-WorldBench" 或跟进作者 Meiqi Wu (CASIA) / Zhixin Cai (Beihang) 的 GitHub release。Abstract 里明确说会 publicly release,对 community 是好消息。

---

# Omni-WorldBench: 面向 World Model 的 Interaction-Centric 评测

下面我从动机、数据集设计、metric 公式细节、实验结果四个层面拆解这篇 paper,并夹杂一些我对设计选择的直觉判断。

参考链接:
- World Models (Ha & Schmidhuber, 2018): https://arxiv.org/abs/1803.10122
- VBench: https://vchitect.github.io/VBench-project/
- WorldScore: https://arxiv.org/abs/2504.00983
- Wan2.1/2.2: https://arxiv.org/abs/2503.20314
- HunyuanVideo: https://arxiv.org/abs/2412.03603
- Cosmos (NVIDIA): https://arxiv.org/abs/2501.03575
- ViewCrafter: https://arxiv.org/abs/2409.02048
- Gen3C: https://arxiv.org/abs/2504.18249
- GroundingDINO: https://arxiv.org/abs/2303.05499
- SAM: https://arxiv.org/abs/2304.02643
- RAFT: https://arxiv.org/abs/2003.12039

---

## 1. 这篇 paper 的核心问题

World model 的本质是:**给定 environment 的初始状态 $s_0$ 和一组交互动作序列 $\{a_t\}$,预测 environment 的状态演化轨迹 $\{s_1, s_2, \ldots, s_T\}$**。换句话说,world model 学的是一个 transition function $s_{t+1} = f(s_t, a_t)$。

但当前 community 有两个评测痛点:
1. **视频生成评测**(FID、FVD、VBench)只衡量 visual fidelity 和 text-video alignment,完全忽略 $a_t \to s_{t+1}$ 这个因果关系是否被模型正确捕获。
2. **WorldScore** 这类 world-model-aware benchmark 只考虑 camera motion 一种 interaction,过于狭窄。

Omni-WorldBench 的核心论点:**未来的 world modeling 是 4D generation**(space + time jointly),core capability 是 interactive response——即 interaction action 驱动 state transition 的忠实度。这个工作就是围绕这个 capability 设计 benchmark。

---

## 2. Omni-WorldSuite: 数据集设计

### 2.1 三层 Interaction 层级

这是整个 benchmark 设计的灵魂。他们把 interaction 按 effect scope 分成三层,这种分层本身就是一个非常合理的 complexity ladder:

| Level | Effect Scope | 例子 | 模型要求 |
|-------|-------------|------|---------|
| **Level 1** | 局限于 acting object 自身,不改变 environment | 看晶体球的光学折射,沿河岸行走 | object-centric dynamics |
| **Level 2** | 一个 object 直接作用于另一个 object | 火堆里加热金属棒,自动驾驶中车辆与动态交通流交互 | pairwise causal physics |
| **Level 3** | 多个 object + 整体 environment 改变 | 掰断意大利面、整理房间、机械臂抓瓶子递给人 | multi-entity causal chain |

这个分层的 intuition: Level 越高,模型需要 jointly 维护的 spatiotemporal consistency 维度越多,从 single-object localization 到 pairwise interaction 再到 global environment state propagation。

### 2.2 两条数据构建 pipeline

**Dataset-grounded Pipeline** (Fig. 2a):
1. 从 open-source dataset 抽 first-frame + camera trajectory
2. Qwen-VL 生成初始 caption
3. 人工 verify/refine,消除 spatial relation 错误
4. 最终 prompt = caption + initial frame + (optional) camera trajectory

覆盖三个 domain:
- **Autonomous Driving**: DriveLM dataset, ego-view + 真实 camera trajectory
- **Embodied Robotics**: InternData-A1, manipulation 任务
- **Gaming**: Sekai dataset, 高动态非真实感环境

**Concept-driven Pipeline** (Fig. 2b):
用 LLM/VLM 合成 prompt 而非从现有 dataset 抽,流程是 generate-verify-refine:
1. 构建 prototype concept taxonomy(scene × object × action × interaction level)
2. ChatGPT-5.2 生成 textual prompt + camera trajectory
3. Gemini + DeepSeek-R1 cross-check
4. 人工 refine
5. FLUX.1-dev 生成 3 个候选 first frame,CFG scale=3.5,50 sampling steps,人工筛 1 个
6. 不合格则重写 prompt,必要时用 Qwen-Image 修 artifact

这里有一个工程上的 key point: **initial frame 的物理合理性至关重要**。如果第一帧里一个机械臂的关节角度就不对,模型无论怎么生成都无法补救这个 prior 错误。所以他们对 first frame 做了非常严格的 QC(minimum 1024×1024, 物理合理, prompt 一致, interactive object 清晰可见)。

### 2.3 Statistics

- 总共 **1,068 个 evaluation prompts**(规模相当大)
- 多标签分布(每个 prompt 可同时命中多个 axis):
  - **Physics Principles (PP)** 出现最频繁 → Newtonian Mechanics (NM) 和 Fluid Mechanics (FM) 是大头
  - **Causality** 和 **Commonsense** 紧随其后
  - Causality 子类中 **C2B (Condition-to-Behavior)** 最多
  - Loop-closure consistency 中 ART (Axial Round-Trip) 和 ODC (Optical/Dynamic Consistency Closure) 居多
- Interaction Level 分布: **Level 2 > Level 3 > Level 1**

Level 2 占主导这个选择很聪明:Level 1 太简单(很多 T2V 模型已经能做),Level 3 太难(大多数模型直接崩),Level 2 是当前模型能力的 sweet spot,能最好地区分 model 差异。

### 2.4 Auxiliary Metadata

对每个 prompt,他们额外标注:
1. **Entity partition**: 把所有 entity 分成 affected set 和 unaffected set,affected entity 还标注预期 motion direction + magnitude
2. **Event timeline**: 按时序排列的关键事件 list
3. **Expected camera motion**: 子集 prompts 标注,还有 **return-to-origin** 设置(挑战 loop-closure 一致性)

这个 metadata 的存在是后续 metric 计算的关键,没有它就无法 ground truth 化 Interaction Effect Fidelity。

---

## 3. Omni-Metric: 评测框架

### 3.1 Structured Information Extraction

给定生成视频 $V \in \mathbb{R}^{T \times H \times W}$:
- **Entity Trajectories**: GroundingDINO + SAM 提取每个 entity 的 mask sequence $\{\text{traj}_k\}_{k=1}^N$,作为该 entity 的 trajectory 表示。这里 $\text{traj}_k \in \mathbb{R}^{T \times h_k \times w_k}$ 是第 $k$ 个 entity 的 per-frame mask。
- **Optical Flow**: RAFT 估计 dense flow field $F \in \mathbb{R}^{T \times H \times W \times 2}$,捕获 regional motion intensity
- **Relative Camera Motion**: 通过 optical flow 变化近似相邻帧的 relative camera motion(follow Li et al. CVPR 2021 的 online adaptation VO 思路)

### 3.2 Generated Video Quality

借用现成 metric:
- **Imaging Quality / Temporal Flickering / Motion Smoothness / Dynamic Degree**: VBench
- **Content Alignment**: WorldScore

这部分是 legacy metric,但放到一个 unified framework 里方便横向比较。

### 3.3 Camera-Object Controllability

#### Camera Control
沿用 WorldScore 的做法,把 camera trajectory error 拆成 rotational + translational 分量,各自 normalize,然后取平均得到 final score。

#### Object Control
**这是 Omni-WorldBench 的一个重要改进点**。WorldScore 原来的做法是用 GroundingDINO 检测 object,然后做 rule-based text matching——但 rule-based matching 对 synonymy 很脆弱(比如 "cup" vs "mug", "car" vs "vehicle")。

Omni-WorldBench 把它 reframe 成 VQA:
$$
\text{Object Control} = \frac{1}{K} \sum_{i=1}^{K} \hat{y}_i
$$

其中:
- $K$ 是 prompt 中指定的 object 总数
- $\mathcal{O} = \{o_i\}_{i=1}^K$ 是 object list
- $\hat{y}_i \in \{0, 1\}$ 是 VLM 对 uniformly sampled frames 的 binary 答案(目标 object 是否可见)

这个改造的 intuition:**rule-based matching 的脆弱性在于语义层级的语义错配,而 VLM 对 synonymy / compositional cue 天然鲁棒**。代价是 VLM 本身的 hallucination,但大模型 binary judgment 的方差比 brittle rule 小得多。

#### Transitions Detect
用 PySceneDetect 的 ContentDetector:在 HSV space 计算 frame-to-frame dissimilarity,超过阈值 $\tau$ 触发 boundary,同时有 minimum scene length $L$ 约束防止 spurious detection。

$$
s_{\text{trans}} = \begin{cases} 1, & N = 1 \\ 0, & N > 1 \end{cases}
$$

$N$ 是检测到的 scene 数。**Intuition**: 真实 world model 应该一镜到底,scene cut 意味着模型 generation 的 continuous latent 被打断了,这本质上是 cheating 的一种形式(很多 T2V 模型偷偷 cutscene 来规避长程一致性)。

### 3.4 Interaction Effect Fidelity(核心贡献)

这是 Omni-WorldBench 真正的差异化部分,四个 metric 各自解决一个 specific failure mode。

#### InterStab-L(Long-horizon temporal consistency)

针对的场景: prompt 要求相机"走一圈回到原点"。 revisit pair $\mathcal{R} = \{(t_a, t_b)\}$ 表示应该在 $t_a$ 和 $t_b$ 看到相同 world state。

对每对 $(i, j)$:
$$
s(i, j) = \frac{1}{2}\left(\text{SSIM}_{\text{gray}}(I_i, I_j) + \cos(\phi(I_i), \phi(I_j))\right)
$$

变量解释:
- $I_i, I_j$: 视频 $V$ 的第 $i$ 帧和第 $j$ 帧
- $\text{SSIM}_{\text{gray}}$: 灰度 SSIM,量化低层 structural fidelity(范围 $[-1, 1]$,实际通常 $[0, 1]$)
- $\phi(\cdot)$: 预训练 vision encoder(CLIP visual tower),把 frame 映射到 semantic feature
- $\cos(\cdot, \cdot)$: cosine similarity,量化高层 semantic consistency
- $\frac{1}{2}$: 等权平均 low-level + high-level

然后:
$$
\text{InterStab-L} = \frac{1}{|\mathcal{R}|} \sum_{(t_a, t_b) \in \mathcal{R}} s(i(t_a), i(t_b)) \cdot \mathbb{I}_{\text{dynamic}}
$$

- $\mathbb{I}_{\text{dynamic}}$ 是 dynamic gating indicator: 在视频的 4 个 canonical anchor 位置算相似度,如果平均超过 $\tau_{\text{static}}$,说明模型可能在生成完全静止的视频(刷分),整个 metric 被 penalize 到 0。

**这个 gating 是关键的设计 choice**。没有它的话,模型只要生成"什么都不动"的 static frame 就能刷高 SSIM 和 CLIP similarity。这个 anti-degeneracy 机制和 contrastive loss 里 negative sampling 的逻辑一样,都是防止 trivial solution。

#### InterStab-N(Non-target region stability)

直觉: **如果 prompt 要求抓起一个瓶子,那么瓶子周围的桌子、背景墙应该几乎不动**。如果非 target region 也在剧烈变化,说明模型的 causal localization 失败,interaction effect 没有被正确"局部化"。

给定非 target spatial region $\mathcal{N}$(把 target mask 从全图抠掉剩下的):
$$
E_{\text{non}}(s) = \frac{1}{T} \sum_{t=1}^{T} \frac{1}{|\mathcal{N}|} \sum_{x \in \mathcal{N}} \|\text{Flow}_t(x)\|
$$

- $T$: 总帧数
- $|\mathcal{N}|$: 非 target region 的像素数
- $\text{Flow}_t(x)$: 位置 $x$ 在帧 $t$ 的 optical flow vector
- $E_{\text{non}}$: 单位时间内非 target region 的平均 motion energy

然后映射到 $[0, 1]$:
$$
\text{InterStab-N}(s) = \exp\left(-\frac{E_{\text{non}}(s)}{\beta \cdot \min(H, W)}\right)
$$

- $\beta$: scaling factor
- $\min(H, W)$: 用视频短边做 resolution normalization
- $\exp(-x)$: 单调递减,motion energy 越小,分数越接近 1

**为什么用 $\min(H, W)$ 而不是 $H \times W$**: 因为 optical flow 的 magnitude 是 pixel-level displacement,它和 spatial dimension 是 linear scaling 关系(图片放大一倍,flow 也放大一倍),而不是 quadratic。用短边是因为 camera motion 一般在两个方向都有,取短的更保守。

#### InterCov(Interaction coverage)

VLM-based semantic verifier 检查每个 target entity $o \in \mathcal{O}$ 的行为是否符合 prompt 规定的 interaction logic:
$$
\text{InterCov} = \frac{1}{|\mathcal{O}|} \sum_{o \in \mathcal{O}} \mathbb{I}(v_o = 1)
$$

- $v_o \in \{0, 1\}$: VLM 对 entity $o$ 是否表现符合 interaction logic 的 binary 判断
- $\mathbb{I}(\cdot)$: 指示函数

这就是 semantic recall。**它和 InterStab-N 互补**: InterStab-N 是 metric-level 的能量检查,InterCov 是 semantic-level 的因果检查。前者查"是不是动了不该动的东西",后者查"该动的是不是动了"。

#### InterOrder(Event ordering)

针对的场景: prompt 描述的是因果链 event sequence $\mathcal{E} = \{e_i\}_{i=1}^K$,模型必须按时序顺序生成。

用 VLM 做 pairwise ordering check,对所有 $C(K, 2) = \frac{K(K-1)}{2}$ 个 event pair $(e_m, e_n)$ with $m < n$ 判断 generated sequence 中 $e_m$ 是否 precedes $e_n$。

$$
\text{InterOrder} = \frac{2 K_s}{K(K - 1)}
$$

- $K$: 总事件数
- $K_s$: ordering 一致的 event pair 数
- $\frac{K(K-1)}{2}$: 全部 pair 数,$\frac{2 K_s}{K(K-1)}$ 就是 consistent pair ratio

**Intuition**: world model 的因果一致性本质上是 event chronology 的保持。如果模型搞混了"杯子掉下去 → 摔碎"和"摔碎 → 杯子掉下去",那就是因果反转,即使单帧生成得再漂亮也是错的。

### 3.5 AgenticScore(Aggregation)

**这是整个 metric framework 最 interesting 的部分**。

朴素做法是把 15 个 sub-metric 简单 average。问题:**不同 prompt 语义关注的 dimension 不同**。例如:
- 纯 camera motion prompt 应该 weight Camera Control 高
- physical interaction prompt 应该 weight Interaction Effect Fidelity 高
- text-to-video prompt 没有 camera trajectory,Camera Control 应该 de-emphasize

Omni-WorldBench 用 agent-based aggregation:
1. 三个 evaluation agent:
   - $A_I$ = (InterStab-L + InterStab-N + InterCov + InterOrder) / 4  (Interaction Effect Fidelity)
   - $A_G$ = Generated Video Quality 的 sub-metric 均值
   - $A_C$ = Camera-Object Controllability 的 sub-metric 均值
2. 一个 MLLM aggregation agent 分析 prompt 语义,输出三个维度的 importance ranking
3. Ranking 映射到 predefined weight 系数 $w_1, w_2, w_3$
4. Final score:
$$
\text{AgenticScore} = w_1 A_I + w_2 A_G + w_3 A_C
$$

**这种 MLLM-as-aggregator 的设计 pattern 现在越来越流行**(参考 LLM-as-Judge 系列)。它的好处: weight 不是固定的,而是 prompt-conditioned 的,可以处理 prompt diversity。风险: MLLM 的 ranking 不一定稳定,且 predefined weight 系数本身是个超参。

---

## 4. 实验结果深度分析

### 4.1 评测的 18 个 model

| Paradigm | Models |
|----------|--------|
| **T2V** (4) | Director3D, OpenSoraPlan, T2V-Turbo, HunyuanVideo |
| **IT2V** (7) | Matrix-Game2.0, Wan2.1, Wan2.2, CogVideo, OpenSora, Cosmos, LargeVideoPlanner |
| **Camera-conditioned** (7) | HunyuanWorld, HunyuanGameCraft, ViewCrafter, Gen3C, Lingbot, FantasyWorld, WonderWorld |

测试规模: T2V 和 IT2V 用 410 个 prompts,camera-conditioned 用 120 个带 trajectory 的 prompts。

### 4.2 Table 1 关键发现

**整体 AgenticScore**:
- 全场最高: **Wan2.2 (75.92%)** → **Cosmos (75.42%)**
- T2V 组最高: **HunyuanVideo (73.96%)**
- Camera 组最高: **HunyuanWorld (74.36%)**,WonderWorld 紧随 (74.02%)

**Interaction Effect Fidelity 维度**:
- IT2V 组最稳定, Wan2.2 拿 67.34%
- Camera 组存在严重 trade-off: **WonderWorld 在 InterStab-L 拿 84.96%,但 InterStab-N 只有 24.89%**

这个 trade-off 的 intuition 非常关键: **WonderWorld 这种基于 3D-aware 生成的模型,可以维持 long-horizon spatial consistency(回到原位看到的场景一致),但在 camera sweep 过程中整个 scene 都在 "jitter"**。也就是说,它的 stability 是"全局静态的",不是"local dynamic 的"。3D priors 帮它维持 revisit 一致性,但同时压制了真正应该发生的局部 interaction dynamics。

**Generated Video Quality 维度**:
- Temporal Flickering 和 Motion Smoothness 几乎都 > 95%,已经 saturate
- **真正的 differentiator 是 Dynamic Degree**: ViewCrafter 和 WonderWorld 拿 100%(因为它们生成 3D-prior-driven 的大幅 camera motion)
- 其他模型在 Dynamic Degree 上差异巨大,从 16.83 (OpenSoraPlan) 到 100 不等

这意味着: **传统 video quality metric 已经无法区分 top-tier world model**,community 必须转向 interaction-aware metric。

**Camera-Object Controllability 维度**:
- WonderWorld 在 Camera Control 拿 96.12%, 压倒性领先(它有 explicit 3D geometry prior)
- Object Control 方面: Cosmos 94.90%, Wan2.2 94.01% 在 IT2V 组领先

### 4.3 关键 trade-off 观察

最 interesting 的发现可以总结成一组对立关系:

| 强项 | 弱项 | 模型代表 |
|------|------|---------|
| Camera Control 强 | Interaction fidelity 弱 | WonderWorld, ViewCrafter |
| Long-horizon revisit 一致 | Local interaction dynamics 弱 | WonderWorld (InterStab-L 高, InterStab-N 低) |
| Visual quality 强 | Causal ordering 弱 | 大多数 T2V 模型 |
| Visual quality 强 | Object control 弱 | T2V-Turbo, OpenSoraPlan |

这印证了 paper 的核心论点: **当前 world model 的 visual fidelity 已经接近饱和,真正的 bottleneck 是 action-conditioned dynamics 和 causal coherence**。

### 4.4 Qualitative 证据

**Fig. 5**(baseball player throw):
- Wan2.2 成功合成完整、anatomically reasonable 的投球动作,且全程维持 athlete 结构完整
- Matrix-Game2.0 严重失败:动作不完整,时序 degradation,最后几帧 human figure 完全 collapse 消失

这个例子非常 vivid。baseball throw 是 Level 2 interaction(hand → ball → 飞行轨迹),要求模型同时维护:
1. Human pose 序列的 temporal coherence
2. Hand-ball contact 时刻的 causal event
3. Ball 离手后的 projectile physics
4. 全程的 scene background stability

Matrix-Game2.0 失败的本质: 它的 autoregressive frame generation 在长序列下 drift,且没有 strong physical prior 来 anchor human pose。

**Fig. 6**(左 strafe camera):
- HunyuanWorld 全程稳定
- ViewCrafter 凭空生成一个 building,破坏 visual consistency

这里 ViewCrafter 的 failure mode 是经典的 **3D prior hallucination**: 它基于 sparse 3D reconstruction 做新视角生成,当 camera 移动到 training view 没覆盖的区域时,reconstruction 稀疏,inpainting model 就 invent 出不存在的 structure。

---

## 5. 我对这篇 paper 的 intuition 评价

### 5.1 强项
1. **Interaction hierarchy (Level 1/2/3)** 是一个 clean 的 complexity ladder,可以做 curriculum learning,也可以诊断 model 在哪个 complexity level 崩
2. **Anti-degeneracy gating** (InterStab-L 的 dynamic gating) 是 metric 设计中很关键的 robustness trick,显示作者对 evaluator hacking 有警惕
3. **VLM-as-aggregator** 解决了 fixed-weight aggregation 的 fragility,是合理的 design pattern
4. **InterStab-N 用 non-target region motion energy** 来检查 causal localization,这个 insight 非常 sharp——大多数 evaluator 只看 "target 对不对",很少有人看 "非 target 应不应该动"

### 5.2 潜在 issue
1. **VLM-based metric 的 reliability**: InterCov 和 InterOrder 都依赖 VLM binary judgment,VLM 本身的 hallucination 会被 propagate 进 metric。paper 中提到做了 human alignment study,但只在 appendix(本文件没包含)。如果 VLM 在某些物理常识上系统性 biased,metric 就会失真
2. **AgenticScore 的 weight 系数**: 文中说 MLLM 输出 ranking 映射到 "predefined weight coefficients" $w_1, w_2, w_3$,但没说具体数值。这是个 hidden hyperparameter,会影响 reproducibility
3. **Camera-relative motion estimation 的精度**: 用 optical flow 变化来近似 relative camera motion(following [62]),这个 approximation 在 dynamic scene 下误差很大,因为 flow 同时包含 ego-motion 和 object motion,separation 是 ill-posed
4. **Level 3 prompt 数量占比**: statistics 显示 Level 2 > Level 3 > Level 1,但 Level 3 才是真正考验 multi-entity causal 的,数量占比可能不足
5. **Initial frame 的 prior bias**: 用 FLUX.1-dev + Qwen-Image 生成的 first frame,本身可能 carry FLUX 的 artifact distribution,被 evaluate 的 model 如果在 similar 分布上 train 过,会 unfair advantage

### 5.3 这篇 paper 暴露的 research direction

1. **Causal grounding for video diffusion**: 当前 diffusion model 的去噪过程没有 explicit causal model,导致 long-horizon event ordering 弱。需要把 causal structure 注入 noise schedule 或 latent dynamics
2. **Decoupling camera motion from object dynamics**: WonderWorld 的 InterStab-N 灾难性低分说明,3D-aware generation 把 camera 和 object dynamics 强 coupling 了,需要 disentangle
3. **Causal localization**: InterStab-N 揭示了一个 fundamental 问题——模型不知道 interaction 应该 localized 在哪里。这暗示需要 explicit attention grounding 机制(类似 SAM 之于 segmentation)
4. **Long-horizon revisit consistency vs local dynamics**: 这是 3D prior model 的 inherent trade-off,可能需要 hybrid architecture(3D prior for global layout + 2D dynamics for local interaction)
5. **Dynamic Degree 已经成为 new bottleneck**: Temporal Smoothness 和 Flickering 已经 saturate,community 应该把注意力从 "smoothness" 转向 "dynamic responsiveness"

### 5.4 联想到的相关工作

- **WorldScore** (Duan et al., 2025, arXiv 2504.00983): Omni-WorldBench 的直接前驱,但只考虑 camera motion interaction
- **WorldModelBench** (Li et al., 2025, arXiv 2502.20694): 另一个 world model benchmark,但覆盖度不如 Omni-WorldBench
- **VBench / VBench++**: 视频生成 benchmark 的事实标准,Omni-WorldBench 借用了它的 sub-metric
- **Genie 2** (DeepMind): large-scale foundation world model,代表 game environment 下的 interactive generation 方向
- **Cosmos** (NVIDIA): physical AI 的 world foundation model,Table 1 中表现非常强(75.42%)
- **VMBench** (Ling et al., ICCV 2025): perception-aligned video motion benchmark,从 motion perception 角度切入了类似问题
- **FINGER** (Chen et al., ACM MM 2025): content-aware fine-grained evaluation with reasoning,可以看作 VLM-as-judge 在 video 上的应用

---

## 6. 总结

Omni-WorldBench 是一个 milestone 性的 benchmark,核心贡献是把 world model evaluation 从 "video generation quality" 推进到 "interaction-conditioned dynamics fidelity"。它的 Level 1/2/3 hierarchy、InterStab-N 的 causal localization insight、AgenticScore 的 prompt-conditioned aggregation,都是值得 community 学习的设计 pattern。

实验结果揭示的 **核心 finding**: 当前 top-tier video model 在 visual fidelity 上已经接近 saturate,真正的 gap 在 action-conditioned dynamics 和 causal coherence。WonderWorld 在 InterStab-L (84.96) vs InterStab-N (24.89) 的剧烈 trade-off 是 paper 中最有信息量的 single data point,它指向了一个 deep architectural issue: **3D-aware world model 牺牲了 local interaction dynamics 来换取 global spatial consistency**。

这条 research line 的 next step 我猜会是: (1) explicit causal module integrated into diffusion U-Net/transformer,(2) object-centric attention grounding for interaction localization,(3) hybrid 2D-3D architecture 来 disentangle camera motion from object dynamics。

如果想看 paper 原文: 可以直接搜 arXiv "Omni-WorldBench" 或者跟进作者 Meiqi Wu (CASIA) / Zhixin Cai (Beihang) 的 GitHub release。作者在 abstract 里明确说会 publicly release,这对 community 是好消息。
