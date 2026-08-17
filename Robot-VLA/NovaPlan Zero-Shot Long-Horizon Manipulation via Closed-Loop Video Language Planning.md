---
source_pdf: NovaPlan Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language
  Planning.pdf
paper_sha256: dd0f31a466ffb6c7ce652e3182a8f282b8d5f26c340b14b844e3d91f1f465f19
processed_at: '2026-08-05T22:47:01-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NovaPlan 用人话说:让 Robot 像 "看视频学做事" 的人一样干活

## 一、这篇 Paper 到底在解决什么问题?

想象一个场景:你给 robot 一个任务 "把抽屉里的蓝 block 找出来,放到另一个抽屉上面"。对人来说这很简单 - 走过去,打开抽屉看看,没有就开另一个,找到了抓起来放上去。但 robot 要做到这件事非常难,因为它:

1. **不知道怎么分解任务**:要先开抽屉还是先抓 block?
2. **不知道开抽屉长什么样**:需要想象 "我拉把手,抽屉会滑出来" 这个过程
3. **做错了不会补救**:如果 block 卡住了,robot 不知道怎么调整

NovaPlan 就是想解决这个问题:**给 robot 一个 "大脑" 让它能想象、能规划、能纠错,全程 zero-shot(不需要任何示范)**。

---

## 二、核心 Idea:让 Robot "看 YouTube 学做事"

人对着 YouTube 视频学做菜、学组装家具是常见的事。NovaPlan 借鉴这个思路,但更进一步 - **让 robot 自己生成 "教学视频" 然后照着做**。

整个流程用人话说:

### Step 1: VLM 当 "项目经理"

GPT-5.2 扮演项目经理角色。它看当前 scene 的照片 + 任务描述,然后说:"这个任务分 3 步:先抓 blue block,再抓 green block,最后抓 yellow block"。

这里有个聪明的设计:VLM 会自己判断任务是 "需要深思熟虑型" 还是 "走一步看一步型":

- **Strategic mode** (深谋远虑): 组装家具这种,装错顺序就完蛋了,VLM 会规划完整 $h = N$ 步 plan
- **Greedy mode** (走一步看一步): 找东西这种,不知道哪个抽屉有,先开一个看看再说,$h = 1$

公式表达就是 horizon $h \in \{1, N\}$,VLM 根据任务语义自动选。

### Step 2: Video Model 当 "演员"

VLM 说 "抓 blue block 放到 red block 上",video model (Wan 2.2 或 Veo 3.1) 就生成一段视频演这个过程。这里有个 trick - 生成不止一段,而是 $N_c \times L$ 段(比如 8 段),让 VLM 挑最好的。

为什么生成多段?因为 video model 会 hallucination - 可能 block 突然消失、可能 hand 穿过 block、可能方向反了。生成多个 candidate,再用 VLM 当评委挑选,能过滤掉大多数 nonsense。

### Step 3: VLM 当 "评委"

VLM 用 4 个标准打分(严格按顺序,前面挂了后面就不看):

1. **Target**: 动对 object 了吗?(抓错 block 直接 0 分)
2. **Physics**: 物理合理吗?(block 突然消失或穿透,最多 0.1 分)
3. **Motion**: 运动方向对吗?(flow 和语言指令 match 吗,最多 0.2 分)
4. **Result**: 最终状态对吗?(0.3-1.0 分)

实现上,把 8 个 candidate 的 flow image 和 final frame 拼成一张 grid 图,VLM 一次性 batch 评分,省 API call。

### Step 4: Geometric Grounding - "把视频翻译成 robot 动作"

这是最难的部分。视频里人手抓 block,但 robot 是 Franka arm + 2F-85 gripper,形态完全不同。怎么把 "视频里人手运动" 转成 "robot 关节运动"?

NovaPlan 用了两个 representation 并动态切换:

#### Object Flow (默认)

假设 block 在 video 里可见,用 TAPIP3D track block 上的 K 个 keypoint,得到 3D 轨迹 $\mathcal{F} = \{\mathbf{f}_i^t\}$。然后用 Kabsch algorithm 算出 block 每帧的 6-DoF pose:

$$\mathbf{R}^t = \operatorname*{argmin}_{\mathbf{R} \in SO(3)} \sum_{i=1}^{K} \| \mathbf{R}(\mathbf{f}_i^{t-1} - \mathbf{c}^{t-1}) - (\mathbf{f}_i^t - \mathbf{c}^t) \|^2$$

- $\mathbf{R}^t$: 相邻两帧间的旋转矩阵
- $\mathbf{f}_i^{t-1}, \mathbf{f}_i^t$: keypoint $i$ 在前帧和当前帧的 3D 位置
- $\mathbf{c}^{t-1}, \mathbf{c}^t$: 所有 keypoint 的中心点
- 这就是找最优旋转把两组点对齐

然后假设 block 抓住后和 gripper 刚性连接,block 的运动 = gripper 的运动。

#### Hand Flow (备用,object 被遮挡时用)

当 block 被 hand 遮住,或者 block 旋转太大($\theta > 45°$),object flow 就废了。这时候切换到 hand flow - 用 HaMeR 估计视频中人手的 MANO mesh,直接把人手运动当 robot gripper 运动。

但这里有个大坑:video model 生成的人手有 scale 问题 - 手和 block "飘着" 不接触,或者手靠近镜头时突然变大。NovaPlan 用 **dual-anchor calibration** 修这个问题:

**Anchor 1 (contact onset)**: 检测手刚碰到 block 的那帧 $t_{start}$,调整 scale 让 fingertip 正好贴在 block 表面:
$$s = \frac{\|P_c\|}{\|P_{tip}\|}$$
- $P_c$: block 上接触点的中心
- $P_{tip}$: fingertip 的 3D 位置
- $s$: scale 因子

**Anchor 2 (release)**: 检测手放开 block 的那帧 $t_{end}$,计算 drift:
$$\mathbf{d}_{err} = \mathbf{p}_{tip}^{t_{end}}(s_{end}) - \mathbf{p}_{tip}^{t_{end}}(s_{start})$$

然后在 $[t_{corr}, t_{end}]$ 区间用 linear ramp 补偿:
$$\mathbf{P}_{cal}^t = \mathbf{P}_{raw}^t + \alpha(t) \cdot \mathbf{d}_{err}$$
- $\alpha(t)$: 0 到 1 的线性递增
- 只在 release 附近补偿,不干扰抓取和运输阶段

**直觉**: 接触时 hand 和 block 物理上必须贴在一起,这个 constraint 让我们能从 noisy video 中恢复真实 scale。

### Step 5: Closed-Loop Verify & Recover - "做错了能补救"

每步执行后,VLM critic 比较 3 张图:
- $I_{real,t}$: 执行前
- $I_{real,t+1}$: 执行后
- $I_{target,t}$: 视频规划的最后一帧

如果 VLM 判定失败,有两种 recovery:

- **Grasp recovery**: 重新抓一次
- **Non-prehensile recovery**: 用指尖 poke/nudge 一下。这个特别 clever - recovery 视频用 first-last-frame conditioned generation,VLM 在 start image 上画红 star 标 contact point,然后让 video model 生成 "从红 star 推 block 到目标位置" 的视频。

---

## 三、为什么这个设计 Work?核心 Intuition

### Intuition 1: Video Model 是 Implicit Physics Engine

不用写 explicit physics simulation(那太复杂),video model 在互联网视频上训练,已经 implicitly 学会了 "block 不会穿桌"、"hand 抓住 block 会一起动" 这些物理常识。

VLM 负责 semantic reasoning(分解任务、判断对错),video model 负责 physics imagination(想象动作执行过程)。分工明确,各司其职。

类比: AlphaGo 里 MCTS 是 search,policy network 是 intuition。NovaPlan 里 VLM 是 reasoning,video model 是 imagination。

### Intuition 2: Human Hand 是 Robot Gripper 的 "Proxy"

人手和 robot gripper 虽然形态不同,但都是 "抓住物体移动" 的工具。Video model 生成的人手运动,实际上就是 "这个动作应该怎么执行" 的 visual demonstration。直接 map 到 robot gripper,比用 inverse dynamics model 更直接。

NovaPlan 的 hand flow 在 object flow 失效时(occlusion、大旋转)提供 backup,这是 paper 的 key contribution。

### Intuition 3: Contact 是 Scale 的 Anchor

Video model 生成的视频 scale 是 ambiguous 的 - 手可能比 block 大 10 倍或小 10 倍。但有一个硬约束:**接触时 hand 和 object 必须物理贴合**。

利用这个约束,dual-anchor calibration 从 noisy video 中 recover metric scale。这是 geometric grounding 的关键 insight。

### Intuition 4: Closed-Loop 是 Long-Horizon 的 Key

Open-loop 的致命问题:error accumulation。单步成功率 $p$,n 步成功率 $\approx (1-p)^n$:
- $p = 0.9, n = 4$ → 0.656
- $p = 0.9, n = 10$ → 0.349

Closed-loop 把 single-step failure 从 catastrophic 变 recoverable,这是 long-horizon success 的根本。

### Intuition 5: Re-grounding 每步都做

虽然 high-level plan 在初始 observation 生成,但执行一步后 scene 变了。直接 replay 预规划视频?Geometric mismatch 累积,catastrophic failure。

所以每步 execution 前都基于 updated observation 重新生成 video。Cost: ~30s/step for video gen。Trade-off: latency vs robustness。对 long-horizon,robustness 优先。

---

## 四、实验结果说了什么?

### Long-Horizon Task (Figure 6)

| Task | NovaPlan | NovaFlow | π 0.5 | MOKA |
|------|----------|----------|-------|------|
| 4-layer stacking | 7/10 | 3/10 (4th layer) | 最多 2 layer | - |
| Color sorting | 大部分 success | - | - | - |
| Hidden object search | 10/10 | 10/10 | drawer 好,抓取差 | drawer grasp 失败 |

**Key insight**: NovaPlan 在 4-layer stacking 上碾压 NovaFlow,正是因为第 4 层 rotation 大时自动切换到 hand flow。NovaFlow 纯 object flow 在大旋转下崩溃。

### FMB Assembly (最 challenging)

所有 VLA/VLM baseline 连一步都完成不了。NovaPlan 能 partial success,但遇到:
- Irregular shape (U-block) video generation 失败率高
- Recovery motion 太小,keypoint tracking 被 noise 主导
- GraspGen 在 failure scene 上 grasp proposal 差

**发现**: Non-prehensile recovery (指尖 poke) 比 grasp recovery 更 reliable,因为 poke motion 更 clean,hand pose estimation 更准。

---

## 五、这篇 Paper 的 Limitation

1. **Video model fidelity**: Wan 2.2 在 irregular object 上完全废掉,只有 Veo 3.1 能 work 且成功率低。Video model 是 bottleneck。

2. **Single fixed camera**: 真实场景中 active perception 可能 unlock 更 complex task。

3. **Single hand**: 当前限制单手,bimanual manipulation 未支持。

4. **Reorientation task**: 需要翻转 object 的 recovery 场景无法处理(Figure 8d)。

5. **Latency**: ~40s/step,real-time deployment 困难。

6. **Foundation model dependency**: 依赖 GPT-5.2、Veo 3.1、HaMeR、MoGe2、GraspGen 等多个 SOTA model 的 availability。

---

## 六、我的 Take

这篇 paper 最让我印象深刻的是 **modular composition of foundation models** 的思路。每个 component 都不是 novel(VLM planning、video gen、hand pose、Kabsch、RANSAC),但它们的 **interface design** 和 **组合方式** 是创新。

特别 clever 的几个点:

1. **Video model 当 implicit physics engine**: 避免写 explicit physics,leverage learned world dynamics。这和 Sora 当 world simulator 的思路一脉相承。

2. **Hand 作为 kinematic bridge**: 把 "video 里 human demonstration" 直接转 "robot action",比 inverse dynamics model 更直接。Morphological similarity 是关键。

3. **Contact 约束 recover scale**: 从 noisy generation 中 extract metric trajectory 的 elegant solution。

4. **Adaptive planning horizon**: 不同任务需要不同 look-ahead depth,VLM 自动选。反映 "什么时候 think hard,什么时候 act fast" 的 meta-reasoning。

5. **VLM critic 的 strict hierarchical scoring**: Target → Physics → Motion → Result,early rejection 省 compute。

但这套 system 也暴露了 foundation model 的 limitation。Video generation 在 irregular object 上不可靠,grasp gen 在 failure scene 上失效。这些 bottleneck 会随 foundation model 进步缓解,NovaPlan 的 modular design 让 component 升级容易。

值得思考的方向:

1. **End-to-end vs modular**: 能否把 closed-loop verification 和 recovery 内化到 VLA model?当前 NovaPlan modular 但慢(40s/step),end-to-end 可能快但 lack interpretability。经典 trade-off。

2. **Active perception**: robot 主动调整 camera view 可能 unlock 更 complex task。MIT-IBM 的 active vision for manipulation 方向值得关注。

3. **Bimanual extension**: 当前 single-hand constraint 严重限制 task scope。Human video 里 bimanual manipulation 很常见,extension 有 technical challenge 但方向明确。

4. **Foundation model co-training**: 能否把 video model 和 VLM 在 robotics data 上 co-train,让 video model 更懂 "可执行 action"?Google DeepMind 的 Gemini Robotics 方向在做这个。

5. **Sim-to-real via video**: Video model 能否替代 physics simulator 做 sim-to-real?如果可以,robotics 的 data bottleneck 可能被 fundamentally solve。

---

## 七、Reference

- NovaPlan Project: https://nova-plan.github.io/
- NovaFlow (前作): https://arxiv.org/abs/2410.22365
- FMB Benchmark: https://functional-manipulation-benchmark.github.io/
- HaMeR (hand pose): https://geommedia.github.io/hamer/
- MoGe2 (depth): https://wangrc.github.io/MoGe2/
- TAPIP3D (tracking): https://tapi3d.github.io/
- GraspGen: https://graspgen.github.io/
- Wan 2.2: https://github.com/Wan-Video/Wan2.2
- CoTracker3: https://co-tracker3.github.io/
- SAM 3: https://ai.meta.com/blog/segment-anything-3/
- DROID Dataset: https://droid-dataset.github.io/
- π 0.5: https://www.physicalintelligence.company/blog/pi05
- MOKA: https://moka-manipulation.github.io/
- MANO model: https://mano.is.tue.mpg.de/
- Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
- Consistent Video Depth: https://roxanneluo.github.io/Consistent-Video-Depth/
- MegaSaM: https://mega-sam.github.io/
- Veo 3.1: Google DeepMind 的 video generation model
- GPT-5.2: OpenAI 的 VLM,所有 reasoning task 用这个

用人话总结一句:NovaPlan 让 robot 通过 "自己生成教学视频 + 照着做 + 做错能补救" 的方式,zero-shot 完成长 horizon manipulation task。核心 trick 是把 VLM(语义)、video model(想象)、hand pose(运动先验)三个 foundation model 用 geometric grounding 串起来,形成 closed-loop system。Limitation 在 video model fidelity 和 foundation model dependency,但 modular design 让 future upgrade 容易。这反映了 robotics 从 "task-specific training" 到 "foundation model composition" 的 paradigm shift。

---

# NovaPlan 深度解析:Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning

## 一、Paper 的核心 motivation 与定位

NovaPlan 解决的是一个 long-standing 问题:如何让 robot 在 zero-shot setting 下完成 long-horizon manipulation task。当前的两大 paradigm 各有缺陷:

1. **VLA models** (如 π 0.5, OpenVLA, Octo):依赖大规模 robot trajectory data 训练, data scarcity 是 bottleneck,且缺乏 fine-grained 3D spatial awareness
2. **Video-based planning** (如 NovaFlow, Gen2Act, Track2Act):虽然能 "imagine" 未来状态,但 (a) temporal inconsistency 和 hallucination 累积, (b) occlusion/depth error 导致 tracking drift, (c) open-loop execution 无法 recovery from failure

NovaPlan 的关键 insight:把 video generation 当作 **dynamic query within a verify-and-recover loop**, 而不是 **static trajectory source**。同时利用 **human hand as kinematic prior** 来弥补 object-centric tracking 在 occlusion 下的 brittleness。

Project page: https://nova-plan.github.io/

## 二、System Architecture 总览

整个 pipeline 是 hierarchical closed-loop structure:

```
┌─────────────────────────────────────────────────────────┐
│  High-Level Planner (VLM = GPT-5.2)                    │
│  ┌───────────────┐    ┌──────────────┐                  │
│  │ Task          │───▶│ Video Rollout│ (Wan 2.2 + Veo)  │
│  │ Decomposition │    │ × N_c × L    │                  │
│  └───────────────┘    └──────────────┘                  │
│         │                    │                          │
│         ▼                    ▼                          │
│  ┌──────────────────────────────────┐                   │
│  │ Validation & Selection (4 metrics)│                  │
│  └──────────────────────────────────┘                   │
│         │                                               │
│         ▼                                               │
│  Action sequence A = {a_1, ..., a_h}                     │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼ Re-grounding at each step t
┌─────────────────────────────────────────────────────────┐
│  Low-Level Planner (Geometric Grounding)               │
│  ┌────────────┐         ┌──────────────┐                │
│  │ Object Flow│◀ switch │  Hand Flow   │                │
│  │ (Kabsch)   │ θ>θ_max │  (HaMeR+calib)│                │
│  └────────────┘         └──────────────┘                │
│         │                      │                         │
│         └──────────┬───────────┘                         │
│                    ▼                                     │
│            SE(3) trajectory → Robot action              │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼ Real-world execution
┌─────────────────────────────────────────────────────────┐
│  Closed-Loop Verify & Recover (VLM critic)             │
│  I_real,t → I_real,t+1 vs I_target,t                    │
│     success? ── yes ──▶ next step                        │
│     success? ── no  ──▶ recovery (grasp/non-prehensile) │
└─────────────────────────────────────────────────────────┘
```

## 三、Closed-Loop Video Language Planning 技术细节

### 3.1 Generate-then-Verify Tree Search

Plan formal definition:
$$A = \{a_1, a_2, \ldots, a_h\}$$
其中 $h$ 是 planning horizon。每一层 expansion 包含 4 步:

1. **Task Decomposition**: VLM propose $N_c$ 个 sub-goal candidates
2. **Video Rollout**: 对每个 candidate, video model 生成 $L$ 个 visual rollouts
3. **Validation and Selection**: VLM 按 4 个 metric 评分, 选 top-ranked
4. **Verify and Recover**: execution 后 VLM monitor, 失败时 trigger recovery

在 strategic mode,$N_c = 2$, $L = 4$, 所以每层产生 $2 \times 4 = 8$ 个 candidates, 通过 tree search (Algorithm 1) 在 $H$ 层上展开。Beam search 保留 top-$N_c$ 进入下一层。

**Key intuition**: 视频模型作为 "object dynamics verifier",因为 VLM 本身缺乏 physics simulation 能力, 而 video model 隐含 learned world dynamics。

### 3.2 Rollout Evaluation 的 4 个 Metric

VLM 按 strict hierarchical check 评分,stop at first failure:

| Metric | 检查内容 | 失败时 score |
|--------|----------|--------------|
| Target | 正确 object 是否移动 | ≤ 0.0 |
| Physics | motion 是否符合物理 (no melt, teleport, deform) | ≤ 0.1 |
| Motion | flow direction/magnitude 是否 match language | ≤ 0.2 |
| Result | final frame 是否符合 expected outcome | [0.3, 1.0] |

**Implementation trick**: 把 $N$ 个 candidate stitch 成 grid image,每 tile 上半部分是 flow image (2D object flow overlay on initial frame via CoTracker3), 下半部分是 final frame, 中间用 cyan divider 分隔。这样 VLM 一次 batch score 所有 candidate,减少 API call 成本。

### 3.3 Planning Horizon 自适应

$$h \in \{1, N\}$$

- **Greedy mode** ($h = 1$): 用于 exploratory task 或 partial observability (如 hidden object search)
- **Strategic mode** ($h = N$): 用于 coupled assembly task, 避免 irreversible dead-end

VLM 根据任务语义自主选择 mode。**这个设计非常 smart** - 它把 "什么时候 think hard" 和 "什么时候 act fast" 的 trade-off 自动化,而传统 planner 通常 fixed horizon。

### 3.4 Execution Re-grounding

这是关键的 engineering 决策:虽然 high-level plan $A$ 在初始 observation 下生成, 但每步 execution 前都基于 *updated* observation $I_{real,t}$ 重新生成 video $V_t$。

**为什么必须 re-grounding**: 
- Real-world execution 后 scene state 会变化
- Pre-planned trajectory 的几何与当前 workspace 不再 match
- Direct replay 会导致 geometric mismatch

**Cost**: 每 step 都要 video generation (~30s for 720P), 整个 pipeline ~40s end-to-end。这是 latency vs robustness 的 trade-off。

### 3.5 Verify-and-Recover 机制

VLM critic 比较 3 张图:
- $I_{real,t}$: execution 前 state
- $I_{real,t+1}$: execution 后 current state  
- $I_{target,t}$: generated video $V_t$ 的 last frame

如果 VLM 判定 failure, 进入 recovery:
- **Grasp recovery**: 走 standard planning pipeline
- **Non-prehensile recovery**: 用 fingertip poke/nudge, 生成 specialized recovery video

**Critical insight**: recovery 视频用 first-last-frame conditioned generation, 即给定 annotated start image $I_{t+1,anno}$ 和 target $I_{target}$:
$$V = \text{VideoModel}(I_{t+1,anno}, P, I_{target})$$

这里 $P$ 是结构化 prompt, $I_{t+1,anno}$ 是 VLM 在 start image 上画 red star 标记 contact point 后的 annotated image。这个 annotation 是关键 - 它 ground 了 video model 到 specific contact dynamics。

## 四、Low-Level Planner: Hybrid Flow Mechanism

### 4.1 Object Flow

**Definition**:
$$\mathcal{F} = \{\mathbf{f}_i^t \mid i = 1, \ldots, K; \; t = 1, \ldots, T\}$$
- $\mathbf{f}_i^t \in \mathbb{R}^3$: keypoint $i$ 在 frame $t$ 的 3D position
- $K$: 关键点数 (sampled on target object)
- $T$: video 总帧数

**Pipeline**:
1. **Depth recovery**: MoGe2 → CVD refinement (MegaSaM 假设 fixed camera)
2. **Scale calibration**: affine calibration via RANSAC
   $$D_{metric} = s_{depth} \cdot D_{gen} + t_{depth}$$
   $$\arg\max_{s,t} \sum_{p \in M} \mathbb{I}(|s \cdot D_{gen}(p) + t - D_{sensor}(p)| < \tau)$$
   - $M$: calibration mask (valid depth in both $D_{gen}$ 和 $D_{sensor}$)
   - $\tau = 0.15$: inlier threshold
   - RANSAC 1000 iterations, 移除 < 50 pixel connected components
   
3. **Mask + Keypoint**: SAM3 segmentation + TAPIP3D 3D dense point tracker
4. **6-DoF recovery via Kabsch algorithm**:
   $$\mathbf{R}^t = \operatorname*{argmin}_{\mathbf{R} \in SO(3)} \sum_{i=1}^{K} \| \mathbf{R}(\mathbf{f}_i^{t-1} - \mathbf{c}^{t-1}) - (\mathbf{f}_i^t - \mathbf{c}^t) \|^2$$
   - $\mathbf{R}^t$: 相邻 frame 间的 rotation matrix
   - $\mathbf{c}^{t-1}, \mathbf{c}^t$: 点云 centroids at prev/current timestep
   - $\bar{\mathbf{t}^t} = \mathbf{c}^t - \mathbf{R}^t \mathbf{c}^{t-1}$: translation

**Intuition**: Kabsch 求 best rotation 使两组点对齐,等价于 SVD on covariance matrix。这里假设 object 是 rigid body, keypoint 间相对位置不变。

### 4.2 Hand Flow 与 Dual-Anchor Calibration

**Definition**:
$$\mathcal{H} = \{\mathbf{H}^t \mid t = 1, \ldots, T\}$$
- $\mathbf{H}^t$: MANO mesh at frame $t$ (从 HaMeR 估计)

**Switching criterion** (Equation 2):
$$\exists i \in [1, T], \theta_i > \theta_{max}$$
- $\theta_i$: 相邻 frame 间 rotation magnitude (axis-angle form $\mathbf{R}_t = (\theta_t, \mathbf{u}_t)$)
- $\theta_{max} = 45°$: 当 rotation 过大时 object flow tracking 不可靠 (大旋转导致 Kabsch fit 退化), 切换到 hand flow

**Hand flow 的挑战**: 生成的视频有 geometric artifacts:
- Scale inaccuracy: hand 和 object 不接触 ("floating")
- Projective drift: hand 接近/远离 camera 时 scale 隐式变化

**Dual-anchor calibration** 三阶段:

**Stage 1: Detect interaction interval** (Equation 3)
$$t_{start} = \operatorname*{argmin}_t \left\{ \frac{|\mathbf{M}_{obj}^t \setminus \mathbf{M}_{obj}^{t_0}|}{|\mathbf{M}_{obj}^{t_0}|} \geq \epsilon \right\}$$
- $\mathbf{M}_{obj}^t$: frame $t$ 的 object mask
- $t_0$: 第一帧
- $\epsilon = 0.9$ (grasp mode) 或 $\epsilon = 0.95$ (non-prehensile mode, motion 更小)
- $|\cdot|$: mask area
- $t_{end}$: 类似定义 (最后一个满足条件的 frame)

**Intuition**: 通过 mask area 变化检测 contact onset/offset,假设 object 静止直到 hand 接近导致 mask 变形/occlusion。

**Stage 2: Recovering metric scale at contact**

对 grasp mode:
- 每个 fingertip $f \in \mathcal{F}_{contact}$: project 到 image plane (circle radius 15px)
- 找与 object mask 重叠的 object points
- 候选 scale $s_f$: snap fingertip 到 object point cloud center
- $s_{start} = \max_f s_f$: 取最大避免 under-scaling (occluded fingertip 会 under-estimate)
- 对应 fingertip 成为 designated contact finger

对 non-prehensile mode (Equation 7):
$$s = \frac{\|P_c\|}{\|P_{tip}\|}$$
- $P_c$: object contact points center
- $P_{tip}$: contact finger tip 3D position
- 额外 translation $\Delta t_{start}$ 使 $P_c = s \cdot P_{tip} + \Delta t_{start}$

**Stage 3: Compensating projective drift at release** (Equation 4 & 8 & 9)

Drift offset:
$$\mathbf{d}_{err} = \mathbf{p}_{tip}^{t_{end}}(s_{end}) - \mathbf{p}_{tip}^{t_{end}}(s_{start})$$
- $\mathbf{p}_{tip}^{t_{end}}(s)$: fingertip 在 scale $s$ 下的 3D position at $t_{end}$
- 比较 release 时 end-anchor scale 和 start-anchor scale 的差异

找到 ramp 起始点 (Equation 8):
$$t_{corr} = \operatorname*{argmin}_t \|\mathbf{p}_{tip}^t(s_{start}) - \mathbf{p}_{tip}^{t_{end}}(s_{start})\|_2 < \delta$$
- $\delta = 5$ cm (grasp) / $2$ cm (non-prehensile)
- 第一个 fingertip 接近 release 位置的 frame

Linear ramp (Equation 9):
$$\alpha(t) = \mathbb{I}(t \geq t_{corr}) \cdot \frac{t - t_{corr}}{t_{end} - t_{corr}}$$
$$\mathbf{P}_{cal}^t = \mathbf{P}_{raw}^t(s_{start}, \Delta t_{start}) + \alpha(t) \cdot \mathbf{d}_{err}$$
- $\alpha(t)$: 0 → 1 linear ramp 在 $[t_{corr}, t_{end}]$ 区间
- $\mathbf{P}_{raw}^t(s_{start}, \Delta t_{start})$: globally-scaled 位置
- 只在 release 附近应用 offset,避免 perturb approach/transport phase

**Intuition**: contact 阶段 hand 和 object 物理上接触, scale 应保持一致; 但 video model 隐式 drift 导致 release 时 scale 偏离。Anchor 1 在 contact 时固定 scale, Anchor 2 在 release 时补偿剩余 drift, 通过 ramp 平滑过渡。

### 4.3 Computing SE(3) Trajectory from Hand

假设 contact 期间 hand gesture fixed, 把 hand 当 rigid body:

对每 frame $t$:
- **Translation**: designated contact fingertip 的 calibrated 3D position
- **Rotation** $\mathbf{R}^t$: 从 palm frame 构建
  - $\mathbf{n}^t$: palm normal (fit plane to wrist + MCP joints)
  - $\mathbf{u}^t$: wrist-to-middle-MCP direction 投影到 plane
  - $\mathbf{v}^t = \mathbf{n}^t \times \mathbf{u}^t$: 完成 right-handed basis

得到 per-frame 6-DoF hand motion $(\mathbf{R}^t, \mathbf{t}^t)$, 与 object flow 的 6-DoF motion 等效。

### 4.4 Flow to Robot Action

**Object flow**:
1. GraspGen 生成 candidate grasps, 选 top-ranked
2. 建立 static transform $\mathbf{T}_{obj \leftarrow ee}$ between object frame 和 end-effector frame
3. 假设 post-grasp object fixed relative to gripper
4. 所有 object poses 通过此 fixed transform 转换为 end-effector poses

**Hand flow**:
1. 用 $t_{start}$ 时 first contact hand pose 引导 GraspGen
2. End-effector trajectory 直接 follow calibrated hand pose trajectory $(\mathbf{R}^t, \mathbf{t}^t)$

**关键假设**: hand 的 motion 直接 transfer 到 robot end-effector, 利用 hand 作为 kinematic prior。这避免了 object flow 在 heavy occlusion 下失效的问题。

## 五、Hand-Guided Grasp Selection (Appendix D)

GraspGen 生成 N=5000 candidate grasps, 三阶段 filter:

1. **Contact Filtering**: 
   - 计算 grasp "palm line" (gripper finger bases 连线) 与 contact fingers 的距离 $d_{prox}$
   - $d_{prox} > 5$ cm 的 grasp discarded
   
2. **Collision Avoidance** ($S_{collision}$): 
   - 对 scene point cloud (排除 target object) 验证 collision-free
   - threshold 1mm
   
3. **Palm Line Support** ($S_{support}$): 
   - 计算 object points 在 gripper opening 0.5cm 范围内的比例
   - 作为 soft filter

最终 score (Equation 10):
$$S_{total} = S_{conf} \cdot S_{support}$$
- $S_{conf}$: GraspGen raw confidence
- $S_{support}$: object surface 被覆盖的比例

## 六、Constraint-Aware VLM Prompting (Appendix A)

High-level planner 用 structured prompt,enforce 物理 constraint:

**Horizon-dependent urgency**:
- $T_{remain} \leq 2$: "Time is critical. Actions must be aggressive."
- 否则: "You have time. Prioritize precision and safety."

**Ordering constraint test** (mental simulation):
- "If I place A first, can I still place B?" 
- "If I place B first, can I still place A?"
- 如果 (A then B) works 但 (B then A) fails, A 是 prerequisite, 所有 proposal 必须以 A 开头

**State awareness**:
- 已在 final configuration 的 object 视为 completed, 不再 move
- Misplaced object 跨越 target zone 视为 obstruction

**Action format constraint**:
- "No telekinesis": 必须用 active voice, "Grasp the pot" 而非 "The pot moves"
- "Single hand only": robot 只有一只右手
- "Abstraction level": "Reach down and grasp the handle" 而非 "Move forward 10cm"

## 七、Experimental Results 深度分析

### 7.1 Runtime Breakdown (Table I)

| Model | Wan/Veo | MoGe2/+CVD | SAM3 | TAPIP3D | HaMeR |
|-------|---------|-----------|------|---------|-------|
| Time (s) | 30 | 3/90 | 3.5 | 3.5 | 0.8/frame |

- Video generation 主导: 30s
- CVD refinement 非常慢: 90s (但只对 long-horizon 用)
- 对 41-frame 720P video: end-to-end ~40s (parallelized)

### 7.2 Long-Horizon Tasks (Figure 6)

**Four-layer Block Stacking**:
- Task: stack 4 colored 2-inch blocks vertically in 3 steps
- NovaPlan: 7/10 success
- NovaFlow: 70% on 3 blocks, 30% on 4th (object flow instability)
- $\pi_{0.5}$: 最多 2 blocks
- **Key**: NovaPlan 在第 4 block 时切换 hand flow,而 NovaFlow 因 object flow 在 high rotation 下失效

**Color Sorting**:
- Task: sort 3 blocks into color-matched containers
- 挑战: yellow block tight fit, 需 precise vertical alignment
- 所有方法在 yellow block 上 drop
- NovaPlan failures: depth estimation error 导致 pose extraction 不准

**Hidden Object Search**:
- Task: find hidden object in closed drawers, place on other drawer top
- Partial observability, 2-3 steps depending on which drawer contains object
- $\pi_{0.5}$: drawer opening 好 (因 DROID dataset 分布), 但 object retrieval 差
- MOKA: 不能找到 drawer opening 的正确 grasp (horizontal handle)
- NovaPlan & NovaFlow: 全部 success

### 7.3 Short-Horizon Tasks (Figure 7)

Comparison with NovaFlow on: Block Insertion, Water Plant, Open Drawer
- NovaPlan 在所有 task 上 success rate 更高
- Hand flow 在 self-occlusion 场景下提供 stability

### 7.4 FMB Zero-Shot (Q3 - 最 challenging)

**Setup**: FMB Multi-Object Multi-Stage Assembly 1, Initial Layout 3
- 4-step assembly, 需 millimeter precision
- Irregular shapes (U-shaped blocks) unseen by foundation models
- Diverse failure modes

**Result**: 所有 VLA/VLM baselines 不能完成任何一步; 只有 NovaPlan 能 partial success

**Key challenges identified**:
1. **Single-view video generation for irregular objects**: Wan 2.2 fails, 只有 Veo 3.1 能生成 physically feasible video, 且 success rate 低
2. **Recovery motion 的 keypoints 噪声**: recovery 时 object displacement 和 hand motion 都小, keypoint flow 和 pose estimate 被 noise 主导
3. **GraspGen failure on irregular shapes**: 在 failure scene 下 grasp proposal 性能差

**Critical finding**: Non-prehensile recovery mode (fingertip push) 比 grasp mode 提供更 clean 的 trajectory,因 hand pose estimation 在这个 mode 下能 track 得更好。

### 7.5 Failure Cases (Figure 8)

- (c) Video model 不能生成 physically plausible recovery motion
- (d) Recovery 需 object reorientation, 当前 pipeline 无法处理

## 八、Non-Prehensile Correction 的特殊处理 (Section III-C)

当 prompt 是 "poke the object with the index finger" 时:
- Video model 会 distort hand shape, shift apparent contact geometry
- 标准 grounding 失效

**Extension**: 显式 enforce object contact 与 prompt 指定的 finger:
- Designate poke finger as contact finger
- 在 anchor frames $t_{start}$ 和 $t_{end}$ 求解 isotropic scale + translation correction $(s, \Delta \mathbf{t})$
- 使得 designated fingertip align with object surface in metric geometry
- Apply $(s_{start}, \Delta \mathbf{t}_{start})$ 到 full sequence
- Compensate residual drift near release via localized ramp

这保证 poke fingertip 在 anchor 处确实接触 object,即使 generated hand shape 偏离 normal human prior。

## 九、Related Work 的 positioning

| 方法 | Paradigm | Limitation |
|------|---------|-----------|
| VLA (π 0.5, OpenVLA, Octo) | End-to-end policy | Data scarcity, 缺 3D spatial awareness |
| MOKA | VLM-based symbolic planning | 缺 fine-grained 3D, dynamic reasoning |
| Gen2Act, Track2Act | Video → action via imitation | 需 task-specific demos |
| Robotic Manipulation by Imitating Videos (Patel et al.) | Object-centric 6D pose | Perception inaccuracy, tracking drift |
| NovaFlow | Object flow from video | Open-loop, occlusion brittleness |
| NovaPlan | Closed-loop VLM + hybrid hand/object flow | Video model fidelity, grasp gen on irregular shapes |

NovaPlan 的关键 differentiation:
1. **Closed-loop** 而非 open-loop (vs NovaFlow)
2. **Hybrid flow** 而非纯 object-centric
3. **VLM as critic** 而非 pure forward planning
4. **Geometric grounding** of human hand (而非 end-to-end policy)

## 十、Build Intuition: 为什么这套 design work?

### 10.1 为什么 VLM + Video Model 的组合?

VLM 擅长 semantic reasoning (task decomposition, failure analysis), 但 lack physics simulation。Video model 隐含 learned world dynamics (能 imagine physically plausible outcomes), 但 lack explicit symbolic reasoning。组合: VLM 提供高层结构, Video model 做 "physical rollout"。

类比: AlphaGo 中 MCTS (search) + Policy/Value Network (intuition)。VLM 是 symbolic reasoner, video model 是 implicit physics simulator。

### 10.2 为什么 hybrid flow 而非纯 object flow?

Object flow 在以下场景失效:
- Heavy occlusion (hand 完全遮挡 object)
- Large rotation (Kabsch fit 退化, $\theta > 45°$)
- Depth inaccuracy (object surface 反光/透明)

Hand flow 的优势:
- Hand 通常在 camera view 内 (visible)
- Hand pose estimation (HaMeR) 在 MANO prior 下 robust
- Hand 作为 kinematic prior: 直接 transfer 到 end-effector

**核心 insight**: 人类 hand 是 robot gripper 的 "analog"。Generated video 中的 human hand motion 提供了 robot 应该 execute 的 motion 的 visual demonstration。

### 10.3 为什么 dual-anchor calibration?

Video model 不是物理精确的 simulator,有 systematic distortions:
- **Global scale**: 不同 video 间 scale 不一致 → anchor 1 在 contact onset 固定 scale
- **Projective drift**: 同一 video 内 hand 相对 camera 距离变化导致 scale drift → anchor 2 在 release 补偿

**关键假设**: contact 期间 hand 和 object 物理接触, scale 应保持一致。这个 constraint 让我们能从 noisy video 中 recover metric scale。

### 10.4 为什么 closed-loop 而非 open-loop?

Long-horizon task 的 error accumulation 是 open-loop 的致命问题:
- Single-step failure rate $p$ → $n$-step success rate $\approx (1-p)^n$
- $p = 0.1, n = 4$ → success $\approx 0.65$
- $p = 0.1, n = 10$ → success $\approx 0.35$

Closed-loop recovery 把 single-step failure 从 catastrophic 变成 recoverable。**Verify-and-recover** 机制是 long-horizon success 的 key。

### 10.5 为什么 re-grounding at every step?

Pre-planned trajectory 在初始 observation 下生成, 但每步 execution 后 scene state 变化。如果直接 replay pre-planned trajectory:
- Geometric mismatch 累积
- Object pose 偏离 expected state
- 最终导致 catastrophic failure

Re-grounding cost: 每 step 一次 video generation (~30s)。Trade-off: latency vs robustness。对 long-horizon task, robustness 优先级高。

## 十一、Limitations 与 Future Directions

### 11.1 当前 bottlenecks

1. **Video generation fidelity**: Wan 2.2 在 irregular objects 上完全失败, 只有 Veo 3.1 能 work, 且 success rate 低
2. **Single-view limitation**: 缺 multi-view 或 moving-view video generation
3. **Depth estimation**: 在 transparent/specular surfaces 上不准
4. **GraspGen on irregular shapes**: failure scene 下 grasp proposal 性能差
5. **Reorientation tasks**: 当前 pipeline 无法 handle (Figure 8d)

### 11.2 Future directions

1. **Multi-view video generation**: 可能 unlock more complex assembly
2. **Better depth models**: 更多 camera views
3. **Specialized grasp gen for recovery scenes**
4. **Bimanual manipulation**: 当前 single-hand constraint 限制
5. **Active perception**: robot 主动调整 camera view

## 十二、Reference Links

- NovaPlan Project Page: https://nova-plan.github.io/
- NovaFlow (predecessor): https://arxiv.org/abs/2410.22365 (reference [19])
- FMB Benchmark: https://functional-manipulation-benchmark.github.io/
- HaMeR (hand pose): https://geommedia.github.io/hamer/
- MoGe2 (depth): https://wangrc.github.io/MoGe2/
- TAPIP3D (point tracking): https://tapi3d.github.io/
- GraspGen: https://graspgen.github.io/
- Veo 3.1: Google's video generation model
- Wan 2.2: https://github.com/Wan-Video/Wan2.2
- CoTracker3: https://co-tracker3.github.io/
- SAM 3: Meta's Segment Anything Model v3
- GPT-5.2: OpenAI's VLM used for all reasoning
- DROID Dataset: https://droid-dataset.github.io/
- π 0.5: https://www.physicalintelligence.company/blog/pi05
- MOKA: https://moka-manipulation.github.io/
- Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm
- MANO model: https://mano.is.tue.mpg.de/
- Consistent Video Depth: https://roxanneluo.github.io/Consistent-Video-Depth/
- MegaSaM: https://mega-sam.github.io/

## 十三、个人 Thoughts on Intuition

这篇 paper 的核心 contribution 在我看来是把多个 SOTA foundation model 组合成一个 robust 的 closed-loop system。每个 component 都不是 novel (VLM planning, video generation, hand pose, Kabsch, RANSAC calibration),但它们的 **组合方式** 和 **interface design** 是关键创新。

特别 clever 的几点:

1. **Video model 作为 implicit physics simulator**: 避免了 explicit physics engine 的复杂度, leverage 了 video model 的 learned world dynamics。这类似 LLM 作为 implicit world model 的思路。

2. **Hand 作为 kinematic bridge**: 把 "human demonstration in generated video" 转化为 "robot action"。这比 inverse dynamics model 更直接,因为 hand 和 gripper 有 morphological similarity。

3. **Dual-anchor calibration**: 简单但 elegant。利用 contact 物理约束 (hand 和 object 必须接触) 来 recover metric scale, 从 noisy generated video 中 extract 可执行 trajectory。

4. **VLM critic 的 strict hierarchical scoring**: Target → Physics → Motion → Result。这个顺序反映了 constraint 的 strictness, early rejection 减少 computation。

5. **Adaptive planning horizon**: greedy vs strategic 自适应。这反映了一个 deep insight: 不同任务需要不同 look-ahead depth, 没有一个 horizon 适合所有任务。

这篇 paper 也暴露了当前 foundation model 的 limitation: video generation 在 irregular objects 上不可靠, grasp generation 在 failure scene 上失效。这些 bottleneck 会随着 foundation model 进步而缓解, NovaPlan 的 modular design 让 component 升级变得容易。

值得思考的方向:能否把 closed-loop verification 和 recovery 内化到 end-to-end VLA model 中?当前 NovaPlan 的 modular design 虽然 interpretable,但 latency 高 (~40s/step)。End-to-end model 可能更 fast,但 lack interpretability 和 modularity。这是经典的 modularity vs end-to-end trade-off 在 robotics 上的体现。

另一个思考点:这篇 paper 假设 single fixed camera。在真实 robot deployment 中, active perception (robot 移动 camera) 可能 unlock 更 complex task。这和 MIT-IBM Watson Lab 的 recent work on active vision for manipulation 方向一致。

最后,这篇 paper 的 zero-shot claim 是 strong 的 - 不需要任何 task-specific demonstration 或 training。但依赖多个 SOTA foundation model (GPT-5.2, Veo 3.1, HaMeR, MoGe2, GraspGen, SAM3, TAPIP3D) 的 availability。这反映了一个 trend: future robotics system 会 increasingly rely on general foundation model composition,而非 task-specific training。
