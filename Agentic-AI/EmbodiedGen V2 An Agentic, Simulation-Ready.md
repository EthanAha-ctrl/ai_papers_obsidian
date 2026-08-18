---
source_pdf: EmbodiedGen V2 An Agentic, Simulation-Ready.pdf
paper_sha256: d18d8b6574ddf2009708c5a9192d3f9b1b34520a2d1b8256268a3efc7a1ae4e5
processed_at: '2026-08-18T10:49:30-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 EmbodiedGen V2

## 一句话概括

这篇 paper 干的事，就是**让 AI 自己造一个能跑机器人的虚拟世界**，造出来还不用人收拾，直接能给机器人拿去训练。

---

## 为什么要搞这个

现在的机器人 learning 面临一个尴尬局面：

你有一个 VLA policy（就是那种"看图说话还能动手"的 robot brain），想在 simulation 里训练它。但问题来了——**simulator 里那点 scene 哪来的？**

传统做法是人手搭 scene：工程师用 Blender 或 Maya 建模，手动标 collision mesh，手动调 physical properties，手动 export 成 simulator 格式。搭一个 scene 可能几天甚至几周。

所以你看 RLBench、ManiSkill3 这些 benchmark，scene 数量少得可怜，而且场景都是 hand-crafted 的，distribution 太窄。policy 在上面过拟合了，transfer 到 real robot 就拉胯。

那 generative 3D 呢？TRELLIS、Hunyuan3D 这些 model 确实能生成好看的 mesh。但它们生成的 mesh 有几个致命问题：

1. **没有真实 scale**：生成的杯子可能 2 米高，也可能 2 厘米高
2. **mesh 是坏的**：non-manifold faces、open surfaces，collision detection 直接爆炸
3. **没有 physical properties**：没 mass、没 friction、没 inertial
4. **没有 affordance**：policy 不知道这个杯子该抓哪里、哪里是 handle
5. **只支持一个 simulator**：你在 SAPIEN 里跑，想迁移到 MuJoCo 又得重搞

所以你生成的 3D asset 看着漂亮，但 simulator 根本不认。

**EmbodiedGen V2 要解决的就是：从 text 或 image 出发，生成一整个可以直接跑 policy 的 executable world，cross-simulator 兼容，还支持 natural language 编辑。**

---

## 怎么做的——用造房子打比方

想象你要造一栋能住人的房子，而且是那种"拎包入住"的房子。

### 第一层：造家具（Sim-Ready Asset Generation）

你说"我要一把椅子"，系统就给你造一把。

但这个过程不是简单跑一下 image-to-3D model 就完事。它像一个**流水线工厂**：

1. 先搞清楚你到底要什么椅子（input preparation）——如果是 text，先用 SD3.5 生成一张椅子图片；如果给的是 image，先用 Rembg 把背景抠掉
2. 用 TRELLIS 或 Hunyuan3D 把图片变成 3D（3D generation）——出来一个 Gaussian + 一个 mesh
3. 修 mesh 的毛病（geometry refinement）——non-manifold face 修好，simplify 一下
4. 把 Gaussian 的"皮"扒下来贴到 mesh 上（texture baking）——因为 Gaussian 不能直接当 simulation texture
5. **关键步骤**：VLM 看 multi-view rendering，猜这个椅子多大、多重、多滑（physical property recovery）。比如看到是木椅子，就猜 mass 大概 3-5kg，friction 0.3-0.5
6. **再关键**：用 CoACD 把 mesh 拆成几个 convex hull（convex decomposition）。为啥？因为 physics engine 算 collision 时，non-convex mesh 会产生 unstable contact，robot 抓不住。拆成 convex hull 后 grasp success 从 96.5% 到 98.6%
7. 打包成 URDF，然后自动转成 MuJoCo 的 MJCF、Isaac Sim 的 USD、SAPIEN 的格式

整个 pipeline 中间有 **quality gate**：VLM 检查生成的 image 对不对，检查 mesh 完不完整，最后还有 aesthetic score 打分。不合格就 retry。

这就像工厂流水线的质检——每个环节都过检，而不是最后才发现产品有问题。

实验数据说话：full pipeline 的 Human Acceptance 96.5%，Collision Success 98.6%，每个 asset 只花 2.6 分钟。

### 第二层：给家具贴标签（Affordance Autolabeling）

光有一把椅子还不够。robot 要知道：
- 哪里是 seat（可以坐）
- 哪里是 backrest（可以靠）
- 哪里是 leg（可以抓但没啥用）
- 哪里该抓才能稳定拎起来

所以 pipeline 给每个 asset 做 part-level affordance annotation：

1. 用 P3-SAM 把 mesh 切成 functionally meaningful parts（seat、backrest、4 个 leg）
2. 用 GPT-5.4 看着 multi-view RGB + part mask，给每个 part 打标签：name、graspability、functional role、appearance description
3. 用 GraspGen 生成一堆 6-DoF grasp candidates，在 SAPIEN 里真的抓一下试试——close、lift、perturb、lower，如果 object 在 gripper 里滑了超过 5cm 或 30°，这个 grasp 就被 discard

结果是每个 asset 都带着 queryable 的 part semantics 和 validated grasp set。policy 可以 query："这个 object 的 handle 在哪？"然后拿到一组经过物理验证的 grasp pose。

这层实验里有个反直觉发现：加了 VLM merging 之后，pipeline 反而更快了（94s vs 109s）。因为 VLM 把过碎的 part merge 了（平均从 5.3 个 part 降到 3.6 个），下游 grasp generation 和 validation 的工作量减少了。

### 第三层：搭场景（Task-Driven Interactive Worlds）

你说"把水果放到桌子上的盘子里"，系统就给你搭一个 scene。

这里有个很聪明的设计：**green-screen 思维**。

拍电影时，你不需要真的造一个完整的厨房——你只需要一个 green screen 当背景，然后把 actor 和关键道具放进去。后再把背景合成上去。

EmbodiedGen V2 用 LLM 把 task 分解成 5 个 role：

- **ROBOT**：什么机器人
- **BACKGROUND**：厨房？卧室？客厅？
- **CONTEXT**：主要 furniture（桌子、台面）
- **TARGETS**：robot 要操作的 object（水果、盘子）
- **DISTRACTORS**：合理但无关的道具（花瓶、书）

然后组织成一个 shallow rooted tree：background 是 root，context 和 robot 挂在 background 下，targets 和 distractors 挂在 context 下。

接下来是 **BFS spatial placement**。这步的核心公式是：

$$\mathbf{p}_c \in \mathcal{H}_p, \quad \text{Support}(\mathcal{B}_c(\mathbf{p}_c), \mathcal{H}_p) = 1, \quad \text{IoU}\left(\mathcal{B}_c(\mathbf{p}_c), \bigcup_{j \in \mathcal{P}_p} \mathcal{B}_j\right) = 0$$

人话翻译：

- $\mathbf{p}_c$：child object（比如盘子）要放在哪里
- $\mathcal{H}_p$：parent（桌子）的 support region，就是桌面那个矩形
- $\mathcal{B}_c(\mathbf{p}_c)$：child 在这个 pose 下的 footprint 投影
- $\mathcal{P}_p$：已经放在桌子上的其他东西
- 第一个条件：盘子必须完全在桌面上，不能悬空
- 第二个条件：IoU=0，就是盘子不能和已放的东西重叠
- 第三个条件（文中提到）：manipulated object 还要在 robot 的 reachable region 内

BFS 保证 parent 先放，child 后放。siblings 按 footprint 排序，大的先放。放完之后在 SAPIEN 里 gravity settle，消除 residual penetration。

最终结果：83.3% 的 generated world 可以直接拿去跑 simulation，不用人改。

### 第四层：造大房子（Large-Scale Scenes）

V1 只能造一个 room 的 panorama background，像一张贴纸贴在后面。robot 想从客厅走到卧室？没门，因为后面是 single mesh，没有 topology。

V2 解决这个：输出 $\mathbf{S} = (\mathcal{R}, \mathcal{F}, \mathcal{C})$

- $\mathcal{R}$：room topology graph，标注哪些 room 之间有门连通
- $\mathcal{F}$：每个 room 里 individually addressable 的 furniture set
- $\mathcal{C}$：house-level 全局坐标

用 Infinigen Indoors 作为 procedural generation backbone，但 reshape 成 simulation-oriented：去掉 physics engine 解析不了的 geometry，把 budget 重分配到 multi-room feasibility。

有意思的是 **complexity knob** $\ell \in \{\text{Minimalist, Simple, Medium, Detail}\}$。你可以控制房间有多乱——从空房间到满地杂物。这个 knob 直接控制 generation cost，让 cost 跟 task difficulty 匹配。

### 第五层：用嘴改场景（Vibe Coding）

这部分是最有意思的。叫 "Vibe Coding" 是因为它像你跟 AI coding assistant 对话写代码一样——你说话，系统改 scene。

但关键区别于 Chat-Edit-3D 或 Holodeck：那些每次 prompt 都 regenerate 整个 scene。Vibe Coding 是 **stateful** 的。

它维护一个 persistent world state：

$$S_t = (\mathcal{G}_t, \mathcal{A}_t, \mathcal{P}_t, \mathcal{H}_t)$$

- $\mathcal{G}_t$：当前 Scene Graph
- $\mathcal{A}_t$：当前 assets
- $\mathcal{P}_t$：当前 6-DoF poses
- $\mathcal{H}_t$：dialogue & edit history

每次 NL instruction 产生一个 **bounded delta** $\Delta S$。成功就 commit，失败就 return diagnostics 但 **不 mutate state**。

这就像 git commit 的 atomicity——每次改动要么全做要么全不做。

整个 editing loop 是：

1. **PARSE**："把那个杯子移到左边一点" → (skill=spatial-computing, args=("那个杯子", "左边"))
2. **GROUND**："那个杯子" → instance_key=42, room_id=1
3. **INVOKE**：执行 spatial-computing skill，在约束下算新 pose
4. **COMMIT**：如果成功，更新 $S_{t+1}$
5. 如果失败，return diagnostics，state 不变

Ground 这步特别重要。它要处理三种 reference：
- Category reference："the chair"
- Attribute reference："the largest piece of furniture"
- Historical anaphora："the apple I just placed" —— 这需要查 $\mathcal{H}_t$

低 confidence 时返回 top-k candidates 让用户 disambiguate。

---

## 最重磅的实验：Closed-Loop Validation

前面所有都是"造得好不好"，但真正的试金石是：**生成的 world 能不能用来 train policy？**

Choi et al. 做了这个实验。起点是 π0-style 的 VLA policy $\pi_{\text{pre}}$，在 BridgeV2 上 pretrain 的，sim success 只有 9.7%。

然后只用 EmbodiedGen V2 生成的 scenes 做 online RL fine-tuning：

| Setting | Before | After |
|---------|--------|-------|
| Sim success | 9.7% | **79.8%** |
| OOD success (N=1→50 scenes) | 53.2% | **77.9%** |
| ID-OOD gap | 41.1 points | **2.6 points** |
| Real-robot success | 21.7% | **75.0%** |
| Dynamics failure | 66.7% | 18.3% |

还有一个反讽实验：在 3 个 hand-built SimplerEnv scenes 上训练的 policy，在 SimplerEnv 上 96.7%，但 transfer 到 EmbodiedGen scenes 只有 36.0%。这说明 **hand-built scene 的 distribution 太窄，policy 过拟合了**。generative pipeline 的 diversity 是 essential 的。

这个实验的意义在于：它证明 generative 3D 不再只是 demo-grade content，而是能驱动 policy improvement 的 substrate。这 closes the loop between 3D generation 和 embodied learning。

---

## 用一句话总结各层

| Layer | 人话 |
|-------|------|
| Asset Generation | 给我一张图或一句话，我给你一个能进 physics engine 的物体 |
| Affordance | 告诉你哪个 part 能抓、抓哪里、怎么抓 |
| Task-Driven Worlds | 给我一个 task，我给你搭一个能跑的 scene |
| Large-Scale Scenes | 给我一个 house，我给你 multi-room 可导航的环境 |
| Vibe Coding | 你用嘴改 scene，我保证改完还能跑 |
| Closed-Loop Validation | 造出来的 world 真能 train 出更好的 policy，还能 transfer 到真机 |

---

## 最核心的 insight

这篇 paper 最大的 conceptual contribution：把 "sim-ready" 从一个 export step 提升为贯穿全 pipeline 的 contract。

V1 的思路：generate → 后处理 → export
V2 的思路：generate-verify-retry closed loop，每一步都 enforce sim-ready contract

这就像从 "先写代码再 debug" 进化到 "type-safe language with runtime checks"。contract baked into the pipeline，而非 append 在末端。

Scene Graph 的 factorization 也很 elegant：把 open-ended NL task 变成 bounded combinatorial problem，每个 pose 的求解是 constraint satisfaction，而非 end-to-end diffusion 生成 6-DoF。

Vibe Coding 的 statefulness 解决了一个真实痛点：3D scene editing 天然是 iterative 的，但 prior work 每次 prompt 都 regenerate 整个 scene。persistent world state + bounded delta + atomic commit 让 iterative editing 成为可能。

---

## 联想与 Open Questions

1. **Scale 的长尾**：128 categories 对日常 manipulation 够了，但 industrial、medical、outdoor 呢？foundation model 对这些 long-tail 的 physical property inference 准不准？

2. **Deformable body 的 affordance**：Figure 3 展示了 garments 做 cloth dynamics，但 affordance pipeline（P3-SAM、GraspGen）对 deformable body 的适用性没评估。抓一个 deformable object 和 rigid body 完全不一样。

3. **Long-horizon navigation 的 RL 验证**：Table 5 只有 tabletop manipulation。large-scale scenes generation 模块没有直接进入 closed-loop validation。multi-room navigation + mobile manipulation 的 RL 效果未知。

4. **Vibe Coding 的 user study**：没有定量评估 NL editing 的成功率、iteration 次数分布、user satisfaction。目前只有 qualitative examples。

5. **Compute cost 的 scalability**：单 world 47.7 min，单 background 25.5 min。要 scale 到 10000+ worlds 用于大规模 RL training，需要 offline asset library + parallelization。paper 提了 reuse offline library 的可能性，但没给具体数字。

6. **Sim-to-real 的 domain randomization 细节**：Table 5 说 "with domain randomization"，但没说 randomize 哪些 axis、什么 range。这关系到一个关键问题——generated world 和 real world 的 gap 有多大？randomization 是怎么 close 这个 gap 的？

7. **Skill harness 的通用性**：paper 说 harness 不 tied to particular agent，可以接 OpenAI Codex 和 Gemini CLI。但如果换成 Claude Code 或其他 agent framework，skill suite 的 trigger 和 argument parsing 是否 robust？

8. **和 World Model 的关系**：Embody4D 等 4D world model 也在做 generative world。EmbodiedGen V2 是 procedural + generative 混合，4D world model 是纯 learned。这两条路线未来会 converge 吗？生成 executable world 和生成 predictive world model 哪个更适合 policy learning？

---

## Related Links

- [EmbodiedGen V2 Project Page](https://horizonrobotics.github.io/EmbodiedGen)
- [EmbodiedGen V2 GitHub](https://github.com/HorizonRobotics/EmbodiedGen)
- [EmbodiedGen V1 arxiv](https://arxiv.org/abs/2506.10600)
- [Choi et al. Sim-to-Real RL for VLA](https://arxiv.org/abs/2603.18532)
- [TRELLIS](https://arxiv.org/abs/2412.01506)
- [Hunyuan3D 2.0](https://arxiv.org/abs/2501.12202)
- [SAM3D](https://arxiv.org/abs/2511.16624)
- [P3-SAM](https://arxiv.org/abs/2509.06784)
- [GraspGen](https://arxiv.org/abs/2507.13097)
- [CoACD](https://dl.acm.org/doi/10.1145/3528223.3530085)
- [SAPIEN](https://sapien.ucsd.edu/)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)
- [RoboVerse](https://arxiv.org/abs/2504.18904)
- [SimplerEnv](https://arxiv.org/abs/2405.05941)
- [π0](https://arxiv.org/abs/2410.24164)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [Infinigen Indoors](https://arxiv.org/abs/2406.12749)
- [Holodeck](https://arxiv.org/abs/2312.09067)
- [LayoutGPT](https://arxiv.org/abs/2305.09067)
- [Embody4D](https://arxiv.org/abs/2605.01799)
- [OpenAI Codex CLI](https://github.com/openai/codex)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)

---

# EmbodiedGen V2 深度解析

Andrej，这篇 paper 来自 Horizon Robotics，核心 contribution 是把 generative 3D 从 "visual plausible" 推进到 "sim-ready executable world"。我尽量把每个 module 的技术细节、公式、ablation 都拆开讲，build your intuition。

---

## 1. Problem framing：什么是 "sim-ready contract"

paper 定义了一个 output contract，包含四个 coupled requirements：

- **Metric geometry**：mesh 必须带真实世界 scale（米/厘米），而非 normalized unit cube
- **Simulation-compatible physical assets**：collision geometry、mass、friction、inertial parameters 都要显式化
- **Task-level semantics & affordances**：哪些 part 可抓、抓哪里、function 是什么
- **Standardized simulator interfaces**：URDF/MJCF/USD 多 backend 复用

这四个 requirements 在 V1 里是 disjoint 的 post-processing，V2 把它们 bake 进 generation pipeline 本身。这是整篇 paper 最核心的设计哲学——**simulation-oriented quality constraints 在每个 stage 都 enforce**，而非末端 export 时才处理。

V1 的致命问题：panorama back-projected single-mesh background，无法支持 long-horizon navigation 和 mobile manipulation，因为没有一个 explicit 的 room topology 和 traversable opening 概念。V2 用 multi-room Scene Graph + addressable furniture instances 解决。

---

## 2. Pipeline 总览（Figure 1 解析）

整个系统是一个 layered architecture：

```
Layer 1: Sim-Ready Asset Generation (Sec 2.2)
   ↓ 输出 URDF-bundled assets with collision proxy + physical params
Layer 2: Affordance Autolabeling (Sec 2.3)
   ↓ 给 asset 加 part-level semantics + validated grasps
Layer 3: Task-Driven Interactive Worlds (Sec 2.4)
   ↓ Scene Graph → BFS placement → physics settling
Layer 4: Large-Scale Scenes (Sec 2.5)
   ↓ Multi-room topology, 替代 V1 的 panorama mesh
Layer 5: Vibe Coding (Sec 2.6)
   ↓ Stateful agent-skill harness for NL editing
```

关键设计：**所有 layer 共享一个 unified world representation**，即 Scene Graph $S_t = (\mathcal{G}_t, \mathcal{A}_t, \mathcal{P}_t, \mathcal{H}_t)$，其中：
- $\mathcal{G}_t$：typed Scene Graph（节点是 assets，边是 spatial relations）
- $\mathcal{A}_t$：sim-ready assets 集合
- $\mathcal{P}_t$：每个 asset 的 6-DoF pose
- $\mathcal{H}_t$：dialogue & skill invocation history

这个 state 是 persistent 的，让 Vibe Coding 能做 bounded local edits，而非每次 regenerate 整个 scene。

---

## 3. Sim-Ready Asset Generation（Sec 2.2）

### 3.1 Pipeline 五阶段（Figure 2）

输入三种：text prompt、unoccluded image、partially occluded image。

**Stage (i) Input Preparation**
- Text → image：pluggable text-to-image model（SD3.5 [Esser et al. 2024](https://arxiv.org/abs/2403.03206) 或 Kolors [Kolors Team 2024](https://arxiv.org/abs/2404.18714)）
- Image input：foreground segmentation（Rembg / SAM [Kirillov et al. 2023](https://arxiv.org/abs/2304.02643) / RMBG）
- Partially occluded：3D-Fixer [Yin et al. 2026](https://arxiv.org/abs/2506.10600)，treats fragmented visible point cloud as spatial anchor，condition frozen TRELLIS backbone，coarse-to-fine completion，**避免显式 pose optimization**

**Stage (ii) 3D Generation**
- Pluggable image-to-3D：TRELLIS [Xiang et al. 2024](https://arxiv.org/abs/2412.01506)、SAM3D [SAM 3D Team 2025](https://arxiv.org/abs/2511.16624)、Hunyuan3D [Tencent 2025](https://arxiv.org/abs/2501.12202)
- 输出 dual representation：3D Gaussian [Kerbl et al. 2023](https://arxiv.org/abs/2308.04079) + mesh

**Stage (iii) Geometry Refinement & Texture Baking**
- Mesh topological repair & simplification
- Multi-view back-projection 把 Gaussian appearance bake 成 explicit texture map
- 原因：Gaussian representation 不适合做 simulation asset 的 texture carrier

**Stage (iv) Physical Property Recovery**
- VLM 从 multi-view renderings + object category 推断 real-world scale、mass、friction
- 用 estimated scale 同时校准 visual mesh、collision mesh、Gaussian representation

**Stage (v) Cross-Format Export**
- URDF 作为 canonical intermediate representation
- Converter 输出 MJCF（MuJoCo [Todorov et al. 2012](https://ieeexplore.ieee.org/document/6386109)）、USD（Isaac Sim）、SAPIEN/Bullet/Isaac Gym XML

### 3.2 Hierarchical Quality Gating

paper 强调 closed-loop generate-verify-retry：

| Stage | Check | 失败处理 |
|-------|-------|---------|
| Input | VLM validates semantic correctness & geometric completeness of segmentation | retry |
| 3D gen | Multi-view rendering 检查 geometric integrity，reject truncated geometry、duplicate bodies、attached elements | retry with different seed |
| Pipeline end | Aesthetic scoring model [Schuhmann 2025](http://captions.christoph-schuhmann.de/aesthetic_viz_laion_sac+logos+ava1-l14-linearMSE-en-2.37B.html) | filter below threshold |

所有 quality-check 结果作为 structured tags 写入 asset file，让下游 batch usage 可以 query 和 filter。

### 3.3 CoACD Convex Decomposition

paper 用 CoACD [Wei et al. 2022](https://dl.acm.org/doi/10.1145/3528223.3530085)（Approximate Convex Decomposition）把 non-manifold mesh 拆成一组 compact convex collision bodies。原因：physics engine（Bullet、MuJoCo）的 collision detection 在 non-convex mesh 上会产生 unstable contact 和 performance degradation。如果 decomposition 失败，fallback 到原始 mesh。

### 3.4 Deformable-Body Extension

Figure 3 展示了 12 个 text-conditioned garments，export 到 Genesis [Genesis Authors 2024](https://github.com/Genesis-Embodied-AI/Genesis) 作为 deformable meshes。Inset heatmap 显示 per-vertex displacement under cloth dynamics，证明 generated geometry fidelity 足够做 soft-body simulation。

---

## 4. Affordance Autolabeling（Sec 2.3）

### 4.1 为什么需要

embodied manipulation 需要的不是 object-level category，而是 **part-level interaction semantics**。policy 需要知道：
- 在哪里 contact
- contact 的 region 支持什么 function
- 在 geometric & physical constraints 下如何执行 contact

### 4.2 三阶段 pipeline（Figure 4）

**Stage 1: Functional Part Segmentation via P3-SAM**

[P3-SAM, Ma et al. 2025](https://arxiv.org/abs/2509.06784)：
1. 从 mesh 采样 point cloud
2. 在 normalized 3D space 推断 part structure
3. 把 predicted point-cloud masks 投影回原始 mesh faces，得到 face-level part segmentation map
4. Remap part identifiers 到 fixed color palette（让 VLM 阶段可以直接用 color name 引用）

**Post-processing 两层**：
- Geometry-consistent：merge smoothly connected face components、relabel small surrounded fragments，针对 projection noise，**不 collapse 真实 sharp part boundaries**
- VLM-guided merging：VLM checker 输入 object category + all part color names + RGB 2×3 multi-view grid + aligned part-mask grid。如果检测到 same functional part 被切成多个 region，自动 merge，iterate 直到 stable

**Stage 2: Part-wise Semantic Annotation via GPT-5.4**

每个 part 推断 structured attributes：
- `part_name`：语义名
- `graspability`：是否适合作为 contact target
- `task-conditioned grasp scenarios`：在哪些 task 下可抓
- `functional labels`：part 的 role
- `semantic description`：color、material、texture、shape、relative location

VLM checker 做 judge-and-revise。

**Stage 3: Grasp Generation & Physical Validation via GraspGen**

[GraspGen, Murali et al. 2025](https://arxiv.org/abs/2507.13097) 生成 confidence-scored 6-DoF grasp candidates，每个 grasp 映射到 contacted semantic parts，按 confidence 排序。

在 SAPIEN [Xiang et al. 2020](https://arxiv.org/abs/2003.08515) 中执行 simulated closing → lifting → perturbation → lowering，discard 掉 object-to-gripper slip > 5 cm 或 > 30° 的 grasp。

---

## 5. Task-Driven Interactive Worlds（Sec 2.4）

### 5.1 Problem formulation

输入：NL task（如 "Place the fruit onto the plate on the table"）
输出：
- (i) Scene Graph：rooted multiway tree，nodes = 3D assets，edges = spatial parent-child relations
- (ii) Composed interactive 3D world with real-scale geometry, physical properties, 6-DoF poses，directly loadable

paper 类比为 filmmaking 的 green-screen production：不 jointly generate 每个世界细节，而是 model 一个 interactive environment 作为 **background + minimal set of task-relevant interactive assets**。

### 5.2 Scene Decomposition（5 个 semantic categories）

LLM 把 task 拆成：
- **ROBOT**：robot type
- **BACKGROUND**：indoor environment
- **CONTEXT**：anchoring furniture（如 table、counter）
- **TARGETS**：robot 必须操作的 objects
- **DISTRACTORS**：plausible 但 task-unrelated props

约束：context 必须 plausibly belong to background（kitchen counter ∈ kitchen，∉ bedroom）。限制 rigid body。

### 5.3 Hierarchy Generation

第二次 LLM query 把 decomposed elements 组织成 shallow rooted Scene Graph：
- Background = root
- Context + Robot = children of background
- Manipulated + Distractor = children of context
- Edges encode：ON、INSIDE、FLOOR、IN
- Single-parent structure 减少 placement ambiguity

### 5.4 BFS Spatial Placement 公式

这是 paper 的核心数学。Equation (1)：

$$\mathbf{p}_c \in \mathcal{H}_p, \quad \text{Support}(\mathcal{B}_c(\mathbf{p}_c), \mathcal{H}_p) = 1, \quad \text{IoU}\left(\mathcal{B}_c(\mathbf{p}_c), \bigcup_{j \in \mathcal{P}_p} \mathcal{B}_j\right) = 0$$

变量解释：
- $\mathbf{p}_c$：child 的 candidate 6-DoF pose（待求解）
- $\mathcal{H}_p$：parent 的 support region（如 tabletop top surface）
- $\mathcal{B}_c(\mathbf{p}_c)$：child 在 candidate pose 下的 projected footprint（bounding box 投影到 parent support plane）
- $\mathcal{P}_p$：已经放在 parent $p$ 上的 siblings 集合
- $\mathcal{B}_j$：第 $j$ 个 sibling 的 footprint
- $\text{Support}(\cdot, \cdot)$：binary predicate，=1 当 child footprint 完全在 parent support region 内（防止悬挂）
- $\text{IoU}(\cdot, \cdot)$：Intersection over Union，=0 表示 no collision with already-placed siblings

物理含义：
1. Child pose 必须落在 parent 的 support region 内
2. Child 不能与已放置 siblings 重叠
3. Manipulated objects 还要满足 robot reachable & forward-facing 约束
4. 失败时 resample candidate 或 invoke relation-specific fallback

BFS traversal 保证 parent 先于 children 放置；siblings 按 footprint 排序 reserve support space。最后在 SAPIEN 中 gravity settle，resolve residual penetrations 和 floating artifacts。

---

## 6. Large-Scale Scenes Generation（Sec 2.5）

### 6.1 为什么 V1 不够

V1 的 panorama-back-projected single-mesh background：
- 没有 explicit room topology
- 没有 traversable openings 概念
- 没有 individually addressable furniture
- 无法支持 long-horizon navigation 和 mobile manipulation

### 6.2 Formal Output

输入 task $T$，输出 triple $\mathbf{S} = (\mathcal{R}, \mathcal{F}, \mathcal{C})$：
- $\mathcal{R}$：room topology graph，annotated with door & window connections
- $\mathcal{F}$：per-room individually addressable furniture set（每个带 visual mesh、collision proxy、physical params）
- $\mathcal{C}$：globally consistent house-level coordinate frame

### 6.3 三阶段 pipeline（Figure 7）

**Stage 1: Task-Conditioned Routing**

VLM 把 $T$ 映射到两个 discrete controls：
- Room scope：local task → 单 room category；cross-room/long-horizon → whole-house joint solve
- Complexity level $\ell \in \{\text{Minimalist, Simple, Medium, Detail}\}$：控制 furniture & clutter density

这个 $\ell$ 是一个 interpretable cost knob。

**Stage 2: Hierarchical Scene Solving**

基于 Infinigen Indoors [Raistrick et al. 2024](https://arxiv.org/abs/2406.12749)，但 reshape from render-oriented to simulation-oriented：

三个 semantic scale coarse-to-fine：
1. Skeleton-level furniture（bed、sofa、cabinet）—— 定义 room function
2. Mid-scale objects on supporting surfaces
3. Tabletop-scale clutter

Complexity tier $\ell$ 控制 activate 哪些 level。

关键改动：suppress unparseable-to-physics geometry & decorative-only geometry，把 solver budget 重分配到 multi-room feasibility。

**Stage 3: Simulator-Agnostic Canonicalization**

三个子步骤：
1. **Per-instance decomposition**：house-level geometry 沿 furniture & architectural units 拆成 individually loadable/replaceable instances。这样 background 可以作为 Sec 2.4 的 Background node，furniture instances 可被 foreground objects 替换或扩展
2. **Convex collision proxy batch**：对所有 furniture instances batch-apply CoACD，replace visual mesh with compact convex hulls
3. **Scene-level canonicalization**：house-level geometry centroid 对齐 world origin，消除 random-seed 导致的 global pose drift

输出 URDF + USD，复用 Sec 2.2 的 unified format converter。

---

## 7. Vibe Coding（Sec 2.6）

### 7.1 Motivation

"Vibe Coding" 概念：iteratively generate & edit sim-ready 3D worlds 通过 NL dialogue。User 表达 intent conversationally，deterministic physics-aware skill backends enforce feasibility。类比 developer vibes with AI coding assistant，compiler enforce type correctness。

痛点：
- Conventional 3D pipeline（modeling、physics annotation、format-specific export）不支持 NL interface
- Prompt-to-scene generators（LayoutGPT、Holodeck）每次 prompt 都 regenerate 整个 scene，无法做 state-preserving local edits
- Sec 2.2-2.5 的 generators 都是 single-shot
- Modern LLM agents 能 invoke domain skills through typed tool calls，但缺少 sim-ready 3D world 的 self-describing skill suite

### 7.2 Agent-Skill-Harness Architecture

**Agent**：LLM-based coordinator，负责 dialogue understanding、intent parsing、skill selection、argument completion、feedback explanation

**Skills**：self-contained capability units，每个 skill 暴露 NL description of usage、inputs、outputs、failure modes，backed by deterministic generators/solvers/exporters from Sec 2.2-2.5

**Harness**：runtime layer，维护 skill registry、dispatch logic、shared world state、failure loop、edit log

参考 Table 1，skill 分四大 abstraction：

| Abstraction | Skills | Role |
|------------|--------|------|
| Asset grounding | asset-creator, asset-retrieval, asset-process, asset-converter | Materialize object intent into sim-ready candidates；project to simulator-specific formats |
| World composition | background-creator, room-creator, layout-creator | Synthesize task-compatible background；produce structured room/house-level worlds |
| Stateful editing | spatial-computing | Commit bounded scene edits via collision-aware spatial constraints |
| Execution validation | sim-runner | Execute world state in simulation，return visual/policy feedback |

### 7.3 Algorithm 1：Parse-Ground-Invoke-Commit Loop

```
Require: dialogue stream {u_t}, initial world state S_0
1: for each instruction u_t do
2:   (ω, α_NL) ← PARSE(u_t, S_t)    # select skill and NL arguments
3:   α ← GROUND(α_NL, S_t)          # resolve typed world references
4:   ΔS ← INVOKE(ω, α, C(S_t))      # execute under constraints
5:   if ΔS = ⊥ then
6:     DIAGNOSE(ω, α, S_t)          # return diagnostics; no state mutation
7:     continue
8:   end if
9:   S_{t+1} ← COMMIT(S_t, ΔS)      # atomic state update
10:  Render(S_{t+1})                # refresh simulation preview
11: end for
```

关键性质：
- 失败时 no state mutation，return structured diagnostics for retry
- 每次成功 update preserves geometric & physical feasibility
- Edit log $\mathcal{H}_t$ 让 historical anaphora（"the apple I just placed"）可 resolve

### 7.4 Instance Grounding

Ground 处理三种 references：
- Category references（"the chair"）
- Attribute references（"the largest piece of furniture"）
- Historical anaphora（"the apple I just placed"）

对 spatial edits，Ground maps references 到 `instance_key` + `room_id`，dispatch 给 spatial-computing skill。该 skill 把 scene 暴露为 room-partitioned 2D floorplan of addressable instances（Figure 9），resolve ON/BESIDE/IN relations via reuse Eq. (1) 的 collision-IoU term，support test 从 object top-surface generalizes 到 room free-floor polygon。

---

## 8. 实验深度解读

### 8.1 Sim-Ready Pipeline Ablation（Table 2）

200 个 held-out assets，SAM3D 作为 image-to-3D model，单张 RTX 4090。

| Setting | Human Accept.↑ | Collision Success↑ | Time (min)↓ | Visual Mesh (MB)↓ | Collision Mesh (MB)↓ |
|---------|---------------|--------------------|--------------|--------------------|----------------------|
| Full pipeline | **96.5%** | **98.6%** | 2.6±0.4 | **1.43±0.63** | **0.29±0.21** |
| w/o Quality checker | 91.0% | 98.1% | **2.2±0.4** | 1.44±0.63 | 0.30±0.22 |
| w/o Mesh fixing | 95.5% | 98.3% | 21.3±22.8 | 51.63±25.87 | 0.31±0.26 |
| w/o Convex decomp. | 94.5% | 96.5% | 2.3±0.3 | 1.45±0.64 | 1.45±0.64 |

**Collision Success 测量协议**：scripted Franka Panda top-down grasp-and-lift in SAPIEN，4 trials per asset at evenly spaced yaw angles，trial successful if object lifted above adaptive height threshold proportional to bounding-box height。

关键 takeaways：
1. **Quality checker**：Human Acceptance 降 5.5 points，但只省 0.4 min。说明 checker 主要抓 perceptual defects，geometric properties 不变
2. **Mesh fixing**：runtime 增加 8×（2.6→21.3 min），visual mesh 大小增 36×（1.43→51.63 MB）。原因：raw generative outputs 含 redundant faces 和 topological defects，严重 slow UV unwrapping、texture baking、convex decomposition。>50 MB 的 object 在 batch-load hundreds of assets 的 simulator 中 impractical
3. **Convex decomposition**：collision mesh 0.29→1.45 MB（5×），Collision Success 98.6→96.5%。collision mesh 回退成 visual mesh，non-convex surface 导致 unstable contact

三个 components 互补地 address perceptual acceptance、deployment efficiency、contact reliability。

### 8.2 Affordance Pipeline Ablation（Table 3）

200 assets，cascaded protocol（每 stage 只测通过上一 stage 的 assets）。

| Setting | Seg. Pass Rate↑ | Semantic Validity↑ | Grasp Coverage↑ | Affordance Pass Rate↑ | Runtime (s)↓ |
|---------|-----------------|-------------------|------------------|------------------------|--------------|
| Baseline | 47.0% | 98.9% | 66.7% | 31.0% | 109±45 |
| +Post-process | 56.5% | 97.3% | 74.6% | 41.0% | 105±41 |
| +Post-process+VLM merging | **69.5%** | **99.3%** | 72.5% | **50.0%** | **94±30** |

**Grasp Coverage 测量**：whole object 至少有一个 simulation-validated candidate grasp pose；intrinsically 不适合 parallel-jaw grasping 的（如 large appliances）算 pass。

**Affordance Pass Rate** = Seg × Semantic × Grasp（三个 stage-wise rate 的 product）。

关键发现：
1. Post-processing 把 Seg Pass Rate 从 47% → 56.5%，conditional Grasp Coverage 66.7% → 74.6%
2. VLM merging 把 Seg Pass Rate 推到 69.5%，Semantic Validity 99.3%。VLM 主要增加 reliable part-level carriers 数量，preserve grasp coverage
3. **反直觉**：full variant 反而最快（94 vs 109 s）。VLM merging 把 average parts 数从 5.3 降到 3.6，减少下游 grasp generation 和 simulation validation 计算量，超过 segmentation 阶段的开销

### 8.3 Task-Driven Worlds（Table 4）

150 NL tasks，sequential online generation on single RTX 4090。

| Category | Metric | Value |
|----------|--------|-------|
| Task-to-Graph | Generated worlds | 150 |
| | Avg. interactive assets per world | 5.19 |
| | Distinct object categories | 128 |
| Asset Instantiation | Background instances | 150 |
| | Object instances | 778 |
| | Time per background | 25.5±3.5 min |
| | Time per object | 3.6±1.1 min |
| Asset QA | Semantic Appearance | 76.2% |
| | Mesh Geometry | 75.9% |
| | Cross-modal Text-to-3D Alignment | 91.0% |
| | Avg. attempts per valid asset | 1.35 |
| World-Level | Total time per world | 47.7±5.4 min |
| | Final environment acceptance | **83.3%** |

Background generation 是 dominant cost（25.5 of 47.7 min ≈ 53%）。但因为 Scene Graph 显式 separate backgrounds from interactive assets，可 reuse offline asset library，让 world generation 降到 minutes order。

三个 automated QA checkers：
1. **Semantic Appearance**：foreground image 是否 match target category & key visual attributes
2. **Mesh Geometry**：generated mesh 是否 complete & free of major geometric defects
3. **Cross-modal Text-to-3D Alignment**：最终 3D asset 是否与 original text description semantically consistent，capture semantic drift introduced during 3D generation

Generate-verify-retry 让每个 valid asset 只需 1.35 attempts。83.3% final worlds 无需 manual modification。

Failure cases（Figure 11）：object-scale mismatch、local geometry defects、imperfect initial spatial placement——通常 resampling 或 minor manual adjustment 可修复。

### 8.4 Downstream Closed-Loop Validation（Table 5）

这是最有说服力的实验。Choi et al. [2026](https://arxiv.org/abs/2603.18532) 用 EmbodiedGen V2-generated scenes 做 online RL of VLA policies。

起点：π0-style imitation policy $\pi_{\text{pre}}$ pretrained on BridgeV2 [Walke et al. 2023](https://arxiv.org/abs/2311.10833)。

| Validation Axis | Setting | Key Result |
|-----------------|---------|------------|
| Online trainability | Fine-tune $\pi_{\text{pre}}$ using only EmbodiedGen scenes | Sim success 9.7%→**79.8%**，avg completion time 10s→8s |
| Scene-distribution scaling | N=1→N=50 scenes | OOD success 53.2%→**77.9%**，ID-OOD gap 41.1→2.6 points |
| Hand-built comparison | Train on 3 SimplerEnv scenes [Li et al. 2024](https://arxiv.org/abs/2405.05941)，eval on EmbodiedGen | 96.7% on hand-built，**only 36.0% on EmbodiedGen** |
| Real-robot transfer | 12 real scenes, 240 trials | Real success 21.7%→**75.0%**，dynamics failure 66.7%→18.3% |

第二个结果（scene-distribution scaling）尤其有意思：从 1 个 scene 到 50 个 scene，ID-OOD gap 从 41.1 收敛到 2.6 points。这是 **scene diversity** 作为 policy generalization driver 的直接证据。

第三个结果（hand-built comparison）是反讽：hand-built SimplerEnv scenes 上训练的 policy 在 hand-built 上 96.7%，但 transfer 到 EmbodiedGen 只有 36.0%。说明 hand-built scenes 过拟合到 narrow distribution，generative pipeline 的 diversity 是 essential 而非 nice-to-have。

第四个结果：sim-to-real transfer 用 domain randomization，real-robot success 21.7%→75.0%，dynamics failure 66.7%→18.3%。这是 EmbodiedGen 作为 scalable generative simulation substrate 的最终证明。

Choi & Xu [2026](https://arxiv.org/abs/2605.11151) 进一步用 EmbodiedGen scenes 训练 cube-stacking sim-to-real VLA via offline-to-online RL（RankQ），real-world cube-stacking success 43.1%→88.9%（144 trials）。

---

## 9. Related Work 定位

### 9.1 Sim-Ready Asset Generation

相关工作谱系：
- DreamFusion [Poole et al. 2023](https://arxiv.org/abs/2209.14988)：score-distillation optimization
- Zero-1-to-3 [Liu et al. 2023](https://arxiv.org/abs/2306.09337)、LRM [Hong et al. 2024](https://arxiv.org/abs/2311.04400)：feed-forward paradigm
- TRELLIS、SAM3D、Hunyuan3D：SOTA end-to-end textured mesh generation

但这些 optimize visual fidelity，不 enforce full sim-ready contract。PhysX-3D [Cao et al. 2025](https://arxiv.org/abs/2505.01291)、PhysX-Anything [Cao et al. 2025](https://arxiv.org/abs/2511.13648)、PhysForge [Yang et al. 2026](https://arxiv.org/abs/2605.05163)、Gen2Sim [Katara et al. 2024](https://arxiv.org/abs/2310.02472) 部分解决物理属性，但缺少 quality-gated closed-loop generation 和 multi-simulator export。

### 9.2 3D Scene Layout

- LayoutGPT [Feng et al. 2023](https://arxiv.org/abs/2305.09067)：直接 predict object bounding-box coordinates
- Holodeck [Yang et al. 2024](https://arxiv.org/abs/2312.09067)：GPT-4 reasoning + Objaverse retrieval
- PhyScene [Yang et al. 2024](https://arxiv.org/abs/2312.09067)：diffusion + collision/layout/accessibility constraints
- Rein3D [Wang et al. 2026](https://arxiv.org/abs/2604.10578)：RL-refined panoramic diffusion
- Agentic 3D Scene [Liu et al. 2025](https://arxiv.org/abs/2505.20129)：VLM agents 做 spatial reasoning

这些 target scene plausibility or navigability，EmbodiedGen V2 从 embodied task 出发，explicit decompose 成 robot/background/context/targets/distractors 后 solve physical placement。

Infinigen Indoors 是 procedural indoor generation，但 collision proxies 非 convex-decomposed，不接受 NL control。

### 9.3 Affordance Labeling

- 3D AffordanceNet [Deng et al. 2021](https://arxiv.org/abs/2103.01523)：23 affordance categories benchmark
- Where2Act [Mo et al. 2021](https://arxiv.org/abs/2103.01523)：从 real robot trajectories 学 actionable regions
- P3-SAM、SegViGen [Li et al. 2026](https://arxiv.org/abs/2603.16869)、ManiTwin [Wang et al. 2026](https://arxiv.org/abs/2603.16866)

区别：prior methods 提供 labels around existing assets；EmbodiedGen V2 **co-produces** sim-ready geometry 和 structured part-level affordance，让 generated objects 进入 scene generation 时就带 queryable interaction semantics。

### 9.4 NL-Driven 3D Editing

- Chat-Edit-3D [Fang et al. 2024](https://arxiv.org/abs/2405.04817)：2D Hash-Atlas mechanism，不 maintain physically deployable 3D world state
- LayoutGPT、Holodeck：每次 prompt regenerate 整个 scene
- OpenAI Codex [2025](https://github.com/openai/codex)、Gemini CLI [Google 2025](https://github.com/google-gemini/gemini-cli)：general-purpose coding agents，缺 domain skills & persistent world state

### 9.5 Embodied Policy Learning

- VLA models：RT-2 [Brohan et al. 2023](https://arxiv.org/abs/2307.15818)、OpenVLA [Kim et al. 2024](https://arxiv.org/abs/2406.09246)、π0 [Black et al. 2024](https://arxiv.org/abs/2410.24164)、GigaBrain-0 [Ye et al. 2025](https://arxiv.org/abs/2510.19430)、HoloBrain-0 [Lin et al. 2026](https://arxiv.org/abs/2602.12062)
- Benchmarks：RLBench [James et al. 2020](https://arxiv.org/abs/1909.12271)、ManiSkill3 [Tao et al. 2024](https://arxiv.org/abs/2410.00425)
- Domain randomization [Tobin et al. 2017](https://arxiv.org/abs/1703.06907)
- Embody4D [Tu et al. 2026](https://arxiv.org/abs/2605.01799)：4D world models

---

## 10. 我的几个 intuition takeaways

### 10.1 "Sim-ready as output contract" 的哲学

paper 最大的 conceptual contribution 不是某个 module，而是把 "sim-ready" 定义为 multi-requirement contract 并 bake 进每个 stage。这类似于把 type system 内嵌到 language runtime 而非 linter。V1 把 sim compatibility 作为 export 末端 step，V2 改成 generate-verify-retry closed loop。

### 10.2 Scene Graph 作为 factorization primitive

task-driven decomposition 成 5 个 semantic roles（ROBOT/BACKGROUND/CONTEXT/TARGETS/DISTRACTORS）+ shallow rooted tree + BFS placement 是很 elegant 的设计。它把 open-ended NL task 转换成 bounded combinatorial problem，每个 object 的 pose 求解变成 constraint satisfaction：

$$\text{find } \mathbf{p}_c \text{ s.t. } \text{Support}=1 \text{ and } \text{IoU}=0 \text{ and } \text{Reachability}$$

这避免了 end-to-end diffusion 直接生成 6-DoF poses 的 instability。

### 10.3 Vibe Coding 的 statefulness

`S_t = (G_t, A_t, P_t, H_t)` 作为 persistent evolving world state 是关键。每次 NL instruction 产生 bounded delta $\Delta S$，failed calls 不 mutate state。这让 iteratively refine 一个 scene 成为可能，类似 git commit 的 atomicity。

Algorithm 1 的 Parse-Ground-Invoke-Commit 是一个 transactional editing loop：
- PARSE：NL → (skill, NL args)
- GROUND：NL args → typed world references（instance_key + room_id）
- INVOKE：执行 deterministic backend under constraints $\mathcal{C}(S_t)$
- COMMIT：atomic state update，update $\mathcal{H}_t$

failed INVOKE 返回 ⊥，no state mutation，return diagnostics。

### 10.4 Closed-loop validation 是真正的 milestone

9.7%→79.8% sim success 和 21.7%→75.0% real-robot success 是这 paper 的杀手级数据。它证明 generative 3D 不只是 demo-grade content，而是可驱动 policy improvement 的 substrate。尤其是 hand-built SimplerEnv scenes 上 96.7% 但 transfer 到 EmbodiedGen 只有 36.0% 的结果，是 diversity-driven generalization 的直接证据。

### 10.5 一些 limitations / open questions

paper 没有深入讨论的：
- **Asset diversity 的 long-tail**：128 categories 覆盖日常 manipulation，但 industrial、medical、outdoor 等 domain 未知
- **Deformable body 的 affordance**：Figure 3 只展示 garments 做 cloth dynamics，但 affordance pipeline（P3-SAM、GraspGen）对 deformable body 的适用性未评估
- **Long-horizon navigation 的 RL 验证**：large-scale scenes generation 模块没有直接出现在 Table 5 的 closed-loop validation 里，只有 tabletop manipulation
- **Vibe Coding 的 user study**：没有定量评估 NL editing 的成功率和 iteration 次数分布
- **Compute cost**：单 world 47.7 min，单 background 25.5 min。如果要 scale 到 1000+ worlds 用于 RL training，需要 offline asset library 和并行化
- **Sim-to-real gap 的 domain randomization 细节**：Table 5 提到 "with domain randomization"，但没说明 randomization 的 axis 和 range

### 10.6 联系到 broader trend

paper 处于几个 trend 的交汇点：
1. **VLA model 的 scaling**：π0、OpenVLA、HoloBrain-0 等 VLA 需要海量 diverse training environments
2. **Generative simulation**：Genesis、RoboVerse [Geng et al. 2025](https://arxiv.org/abs/2504.18904)、SimplerEnv 等 platform 的崛起
3. **Agentic skill harness**：OpenAI Codex CLI、Gemini CLI、Claude Code 等 coding agent 的 skill harness paradigm 扩展到 3D world editing
4. **Foundation model as semantic annotator**：VLM（GPT-5.4）做 part semantic annotation、physical property inference、quality checker，是 foundation model 在 robotics 的典型应用

可以预期 EmbodiedGen V3 可能会做：
- End-to-end trainable sim-ready asset generator（替代 modular pipeline）
- 4D / dynamic world generation（参考 Embody4D）
- Multi-agent embodied policy training 在 generated worlds
- Vibe Coding 的 hierarchical task decomposition（long-horizon mobile manipulation）

---

## 11. Web Links Reference 汇总

主要 references：
- [EmbodiedGen V1 arxiv](https://arxiv.org/abs/2506.10600)
- [EmbodiedGen V2 project page](https://horizonrobotics.github.io/EmbodiedGen)
- [EmbodiedGen V2 GitHub](https://github.com/HorizonRobotics/EmbodiedGen)
- [TRELLIS arxiv](https://arxiv.org/abs/2412.01506)
- [SAM3D arxiv](https://arxiv.org/abs/2511.16624)
- [Hunyuan3D 2.0 arxiv](https://arxiv.org/abs/2501.12202)
- [Hunyuan3D 2.1](https://arxiv.org/abs/2506.16504)
- [3D Gaussian Splatting arxiv](https://arxiv.org/abs/2308.04079)
- [SAM arxiv](https://arxiv.org/abs/2304.02643)
- [SD3.5 arxiv](https://arxiv.org/abs/2403.03206)
- [CoACD ACM TOG](https://dl.acm.org/doi/10.1145/3528223.3530085)
- [SAPIEN project page](https://sapien.ucsd.edu/)
- [MuJoCo](https://mujoco.org/)
- [Isaac Sim GitHub](https://github.com/isaac-sim/IsaacSim)
- [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [RoboVerse arxiv](https://arxiv.org/abs/2504.18904)
- [SimplerEnv arxiv](https://arxiv.org/abs/2405.05941)
- [BridgeData V2 arxiv](https://arxiv.org/abs/2309.11441)
- [P3-SAM arxiv](https://arxiv.org/abs/2509.06784)
- [GraspGen arxiv](https://arxiv.org/abs/2507.13097)
- [LayoutGPT arxiv](https://arxiv.org/abs/2305.09067)
- [Holodeck arxiv](https://arxiv.org/abs/2312.09067)
- [Infinigen Indoors arxiv](https://arxiv.org/abs/2406.12749)
- [3D AffordanceNet arxiv](https://arxiv.org/abs/2103.01523)
- [Where2Act arxiv](https://arxiv.org/abs/2104.14294)
- [DreamFusion arxiv](https://arxiv.org/abs/2209.14988)
- [Zero-1-to-3 arxiv](https://arxiv.org/abs/2306.09337)
- [LRM arxiv](https://arxiv.org/abs/2311.04400)
- [RT-2 arxiv](https://arxiv.org/abs/2307.15818)
- [OpenVLA arxiv](https://arxiv.org/abs/2406.09246)
- [π0 arxiv](https://arxiv.org/abs/2410.24164)
- [RLBench arxiv](https://arxiv.org/abs/1909.12271)
- [ManiSkill3 arxiv](https://arxiv.org/abs/2410.00425)
- [Domain Randomization arxiv](https://arxiv.org/abs/1703.06907)
- [Embody4D arxiv](https://arxiv.org/abs/2605.01799)
- [PhysX-3D arxiv](https://arxiv.org/abs/2505.01291)
- [PhysX-Anything arxiv](https://arxiv.org/abs/2511.13648)
- [PhysForge arxiv](https://arxiv.org/abs/2605.05163)
- [Gen2Sim arxiv](https://arxiv.org/abs/2310.02472)
- [Chat-Edit-3D arxiv](https://arxiv.org/abs/2405.04817)
- [Agentic 3D Scene arxiv](https://arxiv.org/abs/2505.20129)
- [Rein3D arxiv](https://arxiv.org/abs/2604.10578)
- [SegViGen arxiv](https://arxiv.org/abs/2603.16869)
- [ManiTwin arxiv](https://arxiv.org/abs/2603.16866)
- [GigaBrain-0 arxiv](https://arxiv.org/abs/2510.19430)
- [HoloBrain-0 arxiv](https://arxiv.org/abs/2602.12062)
- [OpenAI Codex CLI](https://github.com/openai/codex)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- [Blender](https://www.blender.org)
- [RankQ arxiv](https://arxiv.org/abs/2605.11151)
- [Scaling sim-to-real RL for VLA](https://arxiv.org/abs/2603.18532)

---

希望这个拆解能 build 你的 intuition。整体来说，这篇 paper 的核心 contribution 不是某个 novel algorithm，而是把 generative 3D 的整个 workflow 从 "asset generation" 重新 conceptualize 为 "executable world generation"，并用 Scene Graph + stateful editing + closed-loop validation 把它实现成一个 scalable simulation infrastructure。closed-loop RL 的 9.7%→79.8% sim success 和 21.7%→75.0% real success 数据是这个 conceptual shift 的最强证据。
