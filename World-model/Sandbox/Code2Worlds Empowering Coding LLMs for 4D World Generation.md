---
source_pdf: Code2Worlds Empowering Coding LLMs for 4D World Generation.pdf
paper_sha256: d3b3cf9537f35167af2725a7a6c58629b98bc7596ba9069d28e6712a05eb8fa6
processed_at: '2026-08-03T16:27:22-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Code2Worlds 人话版

好，我把之前那堆技术细节翻译成人话。

---

## 一句话总结

**让 LLM 写 Blender 脚本生成 4D 场景，然后用 VLM 看一眼渲染结果，发现问题就反馈回去改脚本，直到看起来对为止。**

就这么简单。剩下全是工程细节。

---

## 他们想解决什么问题

假设你跟 AI 说："秋天森林里刮风，叶子在飘。"

现有技术会怎么干：

**Video Diffusion (Sora 路线)**: 拿一堆视频训练一个神经网络，hope 它学会"刮风+落叶"长什么样。问题是你看 Sora 生成的视频，物理经常很奇怪——树叶会穿模、水会往上流、物体悬浮在空中。因为模型只是学会了"像素怎么变化"，完全不懂物理。

**Neural 4D (NeRF/Gaussian Splatting 路线)**: 优化一个神经场，让它从某个角度看画面 plausible。问题是你完全没法编辑它，而且物理一致性基本靠运气。

**Code Generation (3D-GPT/SceneCraft 路线)**: 让 LLM 写代码生成场景。这个思路不错，但只做 static 3D，没有动态。

Code2Worlds 想干的事：把 code generation 这条路扩展到 4D，而且要让生成的动态物理上合理。

---

## 两个核心痛点

### 痛点 1：一个 LLM 同时管细节和大局会精神分裂

你让 LLM 一次性生成"一片叶子"和"整个森林"，它很难两头兼顾。

管细节时：叶子的叶脉密度、midrib length、blade color HSV……这些参数一堆。
管大局时：地形、光照、季节、植被密度……又是另一堆。

如果让一个 monolithic forward pass 同时搞定，叶子往往会变成粗糙的方块，因为 LLM 把注意力花在森林布局上了。

**他们的解法**：拆成两条 stream。Object Stream 专门搞叶子，Scene Stream 专门搞森林。各管各的，互不干扰。

### 痛点 2：LLM 写物理代码是瞎子摸象

你让 LLM 写："风轻轻吹过，树叶微微摇摆。"

LLM 会写出一个 Blender 脚本，语法上完全正确。但运行起来你一看——树叶以 200 km/h 的速度狂甩，或者树根也在飞，或者叶子直接无视重力悬浮。

因为 LLM 不知道 "0.25 的 wind strength" 实际渲染出来是什么效果。它只是根据语义猜了个数字。

**他们的解法**：渲染出来，让 VLM 看一眼，如果不对就反馈回去改参数。这就是 closed-loop。

---

## 架构怎么运作

整个流程三个 phase：

### Phase 1：生成目标物体（Object Stream）

假设指令是"叶子在风中飘"。

**Step 1 - 选目标**：ObjSelect Agent 看指令，判断"叶子"需要做动态，把它标记成 target entity。

**Step 2 - 查参数手册**：系统有个 Procedural Parameters Library $\mathcal{L}_{\text{param}}$，里面写着 Infinigen 的所有参数定义。比如 LeafFactory 的参数包括 midrib_length、vein_density、blade_color_hsv 等等。Retrieve 出来 LeafFactory 的 schema。

**Step 3 - 生成参数**：ObjParam Agent 根据指令 + retrieved schema，生成具体参数值。比如"秋天的叶子"→ blade_color_hsv 设成偏黄的值。

**Step 4 - 查代码模板**：系统还有个 Reference Code Library $\mathcal{L}_{\text{code}}$，里面是验证过的 canonical implementations。Retrieve 出 LeafFactory 的代码模板。

**Step 5 - 生成代码**：ObjGenerate Agent 把参数填进模板，生成最终可执行的 Python 脚本。

**Step 6 - 自我反思**：渲染一张 2D 图片，让 VLM-Critic 看一眼——"这看起来像秋天的叶子吗？" 如果不像，VLM 给 feedback（比如"颜色太绿了，应该更黄"），feedback 传回 ObjParam Agent，重新生成参数。循环直到 VLM 说 OK。

### Phase 2：生成环境（Scene Stream）

同时进行，跟 Phase 1 并行。

**Step 1 - 语义分解**：Environment Planner 看指令"森林"，推断出：季节=秋天、天气=有雾、光照=昏暗、地形=有河流、植被密度=高。因为用户只说了"森林"，但真实 3D 世界需要这些 dense specification。Planner 用 LLM 的 world knowledge 做这个 inference。

**Step 2 - 参数具体化**：Parameter Resolver 把"高密度"翻译成 `max_tree_species = 5`。还会做 logical consistency check——如果是 rainforest，就强制 `snow_layer_chance = 0`，因为 rainforest 不会下雪。

**Step 3 - 场景实现**：Scene Realizer 把参数字典编译成 Infinigen 的 execution code，然后 invoke Infinigen 跑出来 3D scene。

### Phase 3：合成 + 加动态（PostProcess Agent）

**Step 1 - 合并**：把 Object Stream 生成的叶子和 Scene Stream 生成的森林合到一起。

**Step 2 - 推断物理参数**：PostProcess Agent 把"轻轻地"翻译成 wind_strength = 0.25，"叶子飘"翻译成 gravity + air resistance 参数。

**Step 3 - 写动画代码**：关键技巧——用 gradient mask。树根 mask=0（不动），树枝 mask=1（可以摇曳）。这样 force 只作用在枝叶上，不会让整棵树飞起来。还要 enforce collision constraints，让落叶会撞到地面停下。

**Step 4 - 渲染 + VLM-Motion 反思**：渲染出视频，让 VLM-Motion Critic 看。如果指令说"微风"但渲染出来树在狂甩，VLM 识别为 magnitude error，反馈回去把 wind_strength 调小。循环直到合理。

---

## 为什么这么做 work

### 1. 为什么不直接让 LLM 从零写代码？

因为 Infinigen 的 API 太复杂了，LLM 没有这个 domain knowledge。Ablation 显示，去掉 retrieval 后 SGS 从 61.4 暴跌到 23.5。

人话：你让一个没学过 Blender API 的人写 Blender 脚本，他肯定写不对。但如果你给他一个 working example，让他改参数，他就能改出来。

### 2. 为什么需要 VLM Critic？

因为 LLM 不知道它写的参数渲染出来是什么效果。它只知道"0.25 的 wind strength 听起来挺轻的"。

VLM 看了渲染结果才知道：哦，0.25 在这个 scene scale 下其实风力很大，树在狂甩。反馈回去调成 0.1。

这跟 RLHF 里 reward model 的角色一样——VLM 在充当 visual reward model。

### 3. 为什么不直接用 video diffusion？

Video diffusion 的 Failure Rate 是 30-70%，Code2Worlds 是 10%。

因为 video diffusion 没有物理引擎兜底，它只是学会了"像素怎么变化"。Code2Worlds 用 Blender 物理引擎保证基本物理（重力、碰撞），LLM 只负责语义到参数的翻译。

---

## 实验数据的人话解读

### Object Generation

他们的 SGS (Structural Goodness Score) = 61.4，第二名 ImmerseGen = 43.5。

提升 41%。主要来自 closed-loop refinement——VLM 看一眼不对就反馈回去改，改几轮质量自然就上去了。

### Scene Generation

Richness = 62.3，第二名 3D-GPT = 41.7。

提升 49%。因为 Scene Stream 会自动 populate understory elements（灌木、岩石、蘑菇），让场景更丰富。其他方法只生成主要物体，场景很空。

### Physics Failure Rate

他们的 10%，video diffusion 30-70%。

这是 physics grounding 的直接 payoff。Blender 物理引擎保证了基本物理一致性，VLM-Motion 再把语义层面的错误也修掉。

### 消融实验的关键发现

去掉 VLM-Motion → Failure Rate 从 10% 飙到 60%。

去掉 Retrieval → SGS 从 61.4 跌到 23.5。

这两个 component 最关键。

---

## 这个方法的局限

### 1. 完全依赖 Infinigen

Infinigen 支持什么，它就能生成什么。Infinigen 不支持的东西（比如复杂工业机械、电子产品），它也生成不了。

Expressiveness ceiling 受限于 procedural generator 的 coverage。

### 2. 慢

要跑 Blender 物理仿真，还要多次 VLM 反馈循环，一次生成可能要几分钟甚至几十分钟。没法 real-time。

### 3. Library 人工成本高

$\mathcal{L}_{\text{param}}$ 和 $\mathcal{L}_{\text{code}}$ 需要人工构建，每个 object category 都要写 schema 和 reference code。扩展到新 object 类别需要人工 effort。

### 4. 10% Failure Rate 仍然不算特别低

对 sim-to-real transfer 来说，10% 的物理错误可能还是会影响下游 training。理想情况应该降到 1% 以下。

---

## 更大的图景

这篇 paper 其实代表了 AI 生成领域两条路线的分歧：

**路线 A (Sora/Neural)**: 用大规模数据训练一个大神经网络，hope 它 emerge 出世界模型。优点是 expressiveness 高，什么都能生成。缺点是物理一致性差，controllability 弱。

**路线 B (Code2Worlds/Procedural)**: 用物理引擎做 substrate，LLM 做语义到代码的翻译，VLM 做质量保证。优点是物理正确、可控、可编辑。缺点是 expressiveness 受限于 procedural generator，速度慢。

这两条路线不互斥。未来可能是：
- Procedural method 做 skeleton（保证物理正确）
- Neural method 做 texture/detail（提升视觉质量）
- Hybrid approach 取两者之长

这跟机器人领域的 sim-to-real 思路很像——用 simulator 保证物理正确，用 neural network 处理 perception 和 high-level planning。

---

## 我的直觉

这篇 paper 最让我感兴趣的是 **VLM-as-reward-model** 这个 pattern。

之前 RLHF 用 human annotator 做 reward model。现在用 VLM 做 visual reward model。这个 idea 可以推广到很多地方：

- 机器人控制：LLM 写控制代码 → 执行 → VLM 看执行结果 → 反馈修正
- UI 生成：LLM 写前端代码 → 渲染 → VLM 看 UI → 反馈修正
- 游戏关卡设计：LLM 设计关卡 → 运行 → VLM 看是否好玩 → 反馈修正

本质上就是：**任何有 visual feedback loop 的 code generation 任务，都可以用 VLM 做 reward model 来 closed-loop refine**。

这个 pattern 比 Code2Worlds 本身更有 generalize 价值。

---

参考链接：
- Paper GitHub: https://github.com/AIGeeksGroup/Code2Worlds
- Project Page: https://aigeeksgroup.github.io/Code2Worlds
- Infinigen (上游依赖): https://arxiv.org/abs/2306.09310
- 3D-GPT (前作思路): https://arxiv.org/abs/2310.12945
- LATS (reflection 思路): https://arxiv.org/abs/2310.04406
- Voyager (code-as-policy 类比): https://arxiv.org/abs/2305.16291
- Eureka (LLM reward generation 类比): https://arxiv.org/abs/2310.12931

---

# Code2Worlds 深度解析

Andrej, 这篇paper的intuition其实非常clean：把4D world generation从neural latent representation转回**procedural code generation**，然后用closed-loop VLM critique来补足LLM blind scripting的缺陷。下面我尽量细节拆解，让你能build up直觉。

---

## 1. Core Thesis: 为什么Code-as-Simulation比Neural 4D更可控

这篇paper站的核心立场非常清晰：world simulator应该grounded in **causal physical laws**，并且 **executable programs** 提供对3D scene structure和semantics的rigorous control，这是black-box neural representations（如dynamic NeRF, 4D Gaussian Splatting）难以做到的。

它identified两个核心failure mode，这两个failure mode正好定义了整个architecture的设计动机：

**Failure Mode 1: Multi-scale context entanglement**
- Monolithic generation pass要在单次forward里同时resolve两个冲突granularity：
  - Local: 树皮的cortex纹理细节、leaf的midrib length、vein density
  - Global: atmospheric lighting、terrain layout、vegetation density
- 这种scale conflict导致prior方法（3D-GPT, SceneCraft, ImmerseGen）优先global coherence，target object变成coarse 3D structure，不适合后续fine-grained physical actuation
- **Intuition**: 想象让一个artist同时画整幅风景画和一片叶子的叶脉，他必然会在其中一个尺度上走神。Factorization是必要的。

**Failure Mode 2: Semantic-physical execution gap**
- 从"leaves trembling"到"vertex weights + turbulence force fields"是个巨大的abstraction leap
- Open-loop scripting让coding LLM像个"blind engineer without visual feedback"
- 导致physical hallucinations: rigid body distortion、particles ignoring gravity、刚体穿透
- **Intuition**: 没有render feedback的code generation就像不跑test就commit代码，错误会在下游放大。

---

## 2. Dual-Stream Architecture: Object Stream + Scene Stream

### 2.1 Object Stream: Retrieval-Augmented Parametric Generation

核心idea: 不要让LLM从scratch生成raw 3D structure，而是把它cast成**parameter space translation**问题。它capitalizes on Infinigen的procedural priors。

整个Object Stream包含4个agent，构成一个agent pipeline：

#### (a) ObjSelect Agent — Dynamic-Aware Target Selection

公式(1):
$$e_{\text{target}} = \arg\max_{e \in \mathcal{E}(\mathcal{I})} P_{\text{dyn}}(e \mid \mathcal{I})$$

变量解释:
- $e_{\text{target}}$: 最终选中的需要dynamic interaction的primary entity
- $\mathcal{E}(\mathcal{I})$: 从instruction $\mathcal{I}$里parse出的所有candidate entities
- $P_{\text{dyn}}(e \mid \mathcal{I})$: 给定instruction，entity $e$需要dynamic actuation的necessity probability
- $\arg\max$: 在所有candidate里选dynamic necessity最大的

**关键设计**: 全局环境变化（如sunlight shifting）bypass这个stream，defer到PostProcess Agent统一处理。这个factorization很关键，因为环境光变化是个global effect，不是单个object的attribute。

#### (b) Retrieval-Augmented Parameter Generation

公式(2):
$$S_{\text{ref}} = \text{Retrieve}(\mathcal{L}_{\text{param}}, e_{\text{target}})$$

- $\mathcal{L}_{\text{param}}$: **Procedural Parameters Library**，把Infinigen的复杂parameter schema总结成结构化文档
- $S_{\text{ref}}$: retrieved parameter definition for target entity

参数空间被disentangle成三个维度：
1. **Structural Shape**: 3D form参数（e.g., midrib length for LeafFactory）
2. **Surface Texture**: 外观attribute（e.g., vein density）
3. **Material Semantics**: rendering properties（e.g., blade_color_hsv）

公式(3):
$$\mathcal{S} = \text{ObjParam}(S_{\text{ref}}, \mathcal{I}, \mathcal{F}_{\text{obj}})$$

- $\mathcal{S}$: 最终生成的parameter set
- $\mathcal{F}_{\text{obj}}$: 来自Object Self-Reflection的feedback（closed-loop refinement signal）
- 关键trick: 用**semantic exemplars**配对natural language description和ground-truth parameter configuration，让LLM通过in-context analogical reasoning推断复杂参数组合

公式(4):
$$\mathcal{C}_{\text{ref}} = \text{Retrieve}(\mathcal{L}_{\text{code}}, e_{\text{target}})$$

- $\mathcal{L}_{\text{code}}$: **Reference Code Library**，indexed verified canonical implementations

公式(5):
$$\mathcal{C}_{\text{obj}} = \text{ObjGenerate}(\mathcal{C}_{\text{ref}}, \mathcal{S})$$

- $\mathcal{C}_{\text{obj}}$: 最终executable code

#### (c) Object Self-Reflection (closed-loop)

公式(6):
$$\mathcal{F}_{\text{obj}}, \mathcal{V} \leftarrow \text{VLM-Critic}(V_{\text{img}}, \mathcal{I})$$

- $V_{\text{img}}$: 渲染object的2D snapshot
- $\mathcal{V}$: validation signal (boolean)
- $\mathcal{F}_{\text{obj}}$: constructive natural language feedback

**Intuition**: 这就像用VLM做visual reward model。如果不满足要求，feedback propagate回ObjParam Agent触发regeneration。这是个iterative refinement loop，类似于RLHF里的reward model，只不过reward是来自VLM的semantic判断。

从Figure 6的JellyfishFactory参数示例可以看到参数空间的精细程度：
- `cap_thickness = 0.1` (bell厚度)
- `cap_inner_radius = 0.7` (bell开口)
- `cap_z_scale = 1.2` (Z轴scale，elongated bell)
- `cap_dent = 0.25` (dent深度，wavy edge)
- `anim_freq = 1/40` (bell动画频率，呼吸速度)
- `move_freq = 1/500` (运动频率，drift速度)

这种参数语义-数值的explicit mapping正是让LLM能grounded inference的关键。

### 2.2 Scene Stream: Hierarchical Environmental Orchestration

#### (a) Semantic Decomposition (Environment Planner)

公式(7):
$$\mathcal{M} = \text{Planner}(\mathcal{I})$$

- $\mathcal{M}$: execution manifest
- Planner是"Creative Extrapolation Brain"，通过LLM的world knowledge推断latent environmental context

它infer三个维度：
1. **Atmospheric Context**: "spooky forest" → infer "Autumn" season + "heavy fog" + "dim lighting"
2. **Terrain Morphology**: 自动instantiate隐含feature如river
3. **Vegetation Density**: enrich ecosystem with coherent understory elements

**Intuition**: 用户说"forest"时，其实信息极度sparse。真实3D world需要season、terrain、density等dense specification。Planner的工作就是做这个**information asymmetry的bridging**。

#### (b) Parameter Concretization (Parameter Resolver)

公式(8):
$$\mathcal{D} = \text{Resolver}(\mathcal{M})$$

- $\mathcal{D}$: scene parameter dictionary

关键设计：**enforce logical consistency**
- 如果是"rainforest"环境，force `snow_layer_chance = 0`（prune incompatible objects）
- Resolve parameter couplings: `air_density`和`dust_density` jointly calibrated

**Intuition**: 这是constraint satisfaction问题。LLM在这里扮演的不是生成器，而是个constraint solver。

#### (c) 3D Scene Realization

公式(9):
$$\mathcal{C}_{\text{env}} = \text{Realizer}(\mathcal{D})$$

- 把high-level logical flags（如`terrain.ground`）map到primitives（如`scene.ground_chance`）
- 然后invoke Infinigen程序执行这些codes，procedurally instantiate 3D scene

**关键设计**: 这里**decouple semantic planning from code execution**，避免free-form code generation的syntactic instability。

---

## 3. Physics-Aware 4D Scene Generation: PostProcess Agent

这是paper的核心创新点：把static scene变成dynamic 4D scene。

### 3.1 Dynamic Scene Integration

公式(10) - Parameter Inference:
$$\mathcal{P}_{\text{phys}} = \text{InferPhysics}(\mathcal{I}, \mathcal{F}_{\text{dyn}})$$

- $\mathcal{P}_{\text{phys}}$: quantitative physics parameters
- $\mathcal{I}$: user instruction
- $\mathcal{F}_{\text{dyn}}$: dynamic feedback from VLM-Motion Critic

例子: "peacefully" → wind strength coefficient = 0.25

公式(11) - Procedural Actuation:
$$\mathcal{W}_{\text{dyn}} = \text{Actuate}(\mathcal{W}_{\text{static}}, \mathcal{P}_{\text{phys}})$$

- $\mathcal{W}_{\text{static}}$: unified static geometry (object + scene merged)
- $\mathcal{W}_{\text{dyn}}$: final dynamic 4D scene

**关键技术细节**:
- Generate code applying **gradient masks** for structural deformation: anchor tree roots (mask=0)，allow branch sway (mask=1)
- **Enforce collision constraints**: particle-terrain interactions
- 这个mask gradient的设计非常elegant，避免了对整个mesh uniform施加force导致unphysical deformation

### 3.2 Dynamic Effects Self-Reflection (VLM-Motion Critic)

公式(12):
$$\mathcal{F}_{\text{dyn}}, \text{valid} = \text{VLM-Motion}(V_{\text{video}}, \mathcal{I})$$

- $V_{\text{video}}$: rendered video rollout of $\mathcal{W}_{\text{dyn}}$
- valid: boolean validation signal for refinement loop
- $\mathcal{F}_{\text{dyn}}$: constructive feedback

例子: instruction说"gentle breeze"，rendered footage显示trees thrashing violently → VLM识别为magnitude error → feedback调整wind force coefficient

**Intuition**: 这是从spatial domain的self-reflection扩展到temporal domain。VLM-Motion作为temporal critic，是个非常novel的设计——previous work多在static image做critique，这里是评估时序动态。

从Figure 23的VLM-Motion prompt看到它的评分维度：
1. **Physics Plausibility** (gravity, collision, inertia，是否有interpenetration或floating objects)
2. **Visual Aesthetics** (texture detail, lighting realism)
3. **Temporal Stability** (flickering, morphing textures, jittery motions)

评分scale:
- [0-40] Failure: severe physics violations
- [41-70] Mediocre: "floaty" or "stiff" physics
- [71-100] Cinematic/Realistic

---

## 4. Algorithm 1: Unified 4D Scene Generation Framework

```
Input: instruction I, libraries L_param, L_code
Output: 4D Scene W_4D

Phase 1: Object Stream (lines 3-12)
  e_target = argmax_e P_dyn(e|I)         # Target Selection
  S_ref, C_ref ← RETRIEVE(L_param, L_code, e_target)
  F_obj ← ∅
  repeat:
    S ← OBJPARAMS(S_ref, I, F_obj)
    C_obj ← OBJGENERATE(C_ref, S)
    V_img ← RENDER(C_obj)
    F_obj, valid ← VLM-CRITIC(V_img, I)
  until valid is true                    # Object Reflection Loop

Phase 2: Scene Stream (lines 13-16)
  M ← PLANNER(I)                          # Semantic Decomposition
  D ← RESOLVER(M)                         # Parameter Concretization
  C_env ← REALIZER(D)                     # 3D Scene Realization

Phase 3: 4D Scene Synthesis (lines 17-26)
  W_static ← UNIFY(C_obj, C_env)
  F_dyn ← ∅
  repeat:
    P_phys ← INFERPHYSICS(I, F_dyn)       # Grounding
    W_dyn ← ACTUATE(W_static, P_phys)
    V_video ← RENDER(W_dyn)
    F_dyn, valid ← VLM-MOTION(V_video, I)
  until valid is true                    # Dynamic Reflection Loop
  return W_4D ← W_dyn
```

**关键观察**: 两个reflection loop的位置很关键——Object loop在Phase 1内部（local），Dynamic loop在Phase 3（global）。这种locality separation非常重要：object的visual fidelity和scene的physical fidelity需要不同scale的critique。

---

## 5. Code4D Benchmark 和实验结果

### 5.1 Benchmark设计

Code4D覆盖三类物理现象：
- **Fluid dynamics** (water spill, rain)
- **Particle systems** (fire, smoke, steam, sand)
- **Rigid-body dynamics** (falling leaves, rolling bottle)
- **Soft-body/cloth simulation** (jellyfish, wind sway)
- **Atmospheric evolution** (10s time-lapse from sunrise to sunset)

从Table 6可以看到prompt设计非常dense且semantically rich，要求long-context reasoning和precise attribute binding。

### 5.2 Main Results (Table 2)

#### Object Generation对比:
| Method | O-CLIP↑ | SGS↑ | Style-CLIP↑ |
|---|---|---|---|
| MeshCoder | 0.2027 | 14.6 | 0.6406 |
| Infinigen | 0.2431 | 35.5 | 0.6671 |
| 3D-GPT | 0.2075 | 37.0 | 0.6178 |
| SceneCraft | 0.2411 | 34.6 | 0.6490 |
| ImmerseGen | 0.2417 | 43.5 | 0.5991 |
| **Code2Worlds** | **0.2655** | **61.4** | **0.6734** |

**关键观察**:
- SGS从43.5 (ImmerseGen) → 61.4，提升41%
- MeshCoder的SGS=14.6极低，说明point-cloud-to-script conversion鲁棒性问题严重
- Code2Worlds的巨大gap主要来自iterative closed-loop refinement

#### Scene Generation对比:
- S-CLIP: 0.2432 (best)
- Richness: 62.3 (vs ImmerseGen 35.5, 3D-GPT 41.7) — **49% improvement**
- HRS: 55.4
- **Physics Failure Rate: 10%**（其他baselines基本都× = N/A，因为不支持dynamics）

#### Video Generation对比（vs video diffusion models）:

| Method | Motion Smoothness↑ | Subject Consist.↑ | Failure Rate↓ | Temporal Flicker↑ |
|---|---|---|---|---|
| SVD | 0.9913 | 0.9312 | 50% | 0.9859 |
| AnimateDiff | 0.9833 | 0.9778 | 70% | 0.9743 |
| CogVideoX | 0.9912 | 0.9004 | 50% | 0.9746 |
| Hunyuan | 0.9925 | 0.9406 | 30% | 0.9893 |
| **Ours** | **0.9952** | 0.9415 | **10%** | **0.9949** |

**关键观察**:
- AnimateDiff的Subject Consistency=0.9778看起来很高，但其实是trade-off：它通过freezing pixel identity来maintain consistency，但缺乏3D structural representation，导致Failure Rate高达70%
- Code2Worlds通过deterministic 3D rendering避免了latent-space interpolation的stochastic noise
- Failure Rate从diffusion models的30-70%降到10%，这是**physics grounding的direct payoff**

### 5.3 Ablation Studies

#### Table 3: Object Generation Components Ablation

| Setting | O-CLIP↑ | SGS↑ | Style-CLIP↑ |
|---|---|---|---|
| w/o L_param | 0.2511 | 48.8 | 0.6535 |
| w/o Retrieve | 0.2221 | 23.5 | 0.6578 |
| w/o VLM-Critic | 0.2388 | 58.6 | 0.6591 |
| Ours | 0.2655 | 61.4 | 0.6734 |

**Intuition**:
- w/o Retrieve: SGS跌到23.5（暴跌62%），这是最大drop → retrieved reference script对proper initialization至关重要。LLM从scratch写procedural code极易syntactic error
- w/o L_param: SGS=48.8 → parameter space的explicit definition对LLM做semantic-to-quantitative mapping至关重要
- w/o VLM-Critic: SGS=58.6 → closed-loop visual feedback比open-loop提升约5%

#### Table 4: Self-Reflection Ablation

| Setting | O-CLIP↑ | Failure Rate↓ | SGS↑ | HRS↑ |
|---|---|---|---|---|
| w/o VLM-Critic | 0.2388 | — | 58.6 | — |
| w/o VLM-Motion | — | 60% | — | 47 |
| Ours | 0.2655 | 10% | 61.4 | 55.4 |

**关键发现**: 
- w/o VLM-Motion: Failure Rate从10% → 60%（6x恶化！），HRS从55.4 → 47
- 这证明**VLM-Motion Critic是physics grounding的关键**——没有它，LLM就是个blind physicist

#### Table 5: Scene Composition Ablation (Appendix B)

| Setting | S-CLIP↑ | Richness↑ |
|---|---|---|
| w/o Planner & Solver | 0.2251 | 50.9 |
| w/o Scene Stream | 0.2365 | 26.4 |
| Ours | 0.2432 | 62.3 |

**关键发现**: 
- w/o Scene Stream: Richness暴跌到26.4（跌58%）→ Scene Stream对environmental complexity至关重要
- w/o Planner & Solver: S-CLIP最低(0.2251) → 语义alignment需要explicit parameter reasoning

---

## 6. Implementation Details

- **Core reasoning engine**: Gemini 3（用于所有VLM-Critic、VLM-Motion、ObjSelect、ObjParam、ObjGenerate、Environmental Planner、Parameter Solver、Scene Realization、PostProcess Agent）
- **3D engine**: Blender 4.3 + bpy Python API
- **Renderer**: Cycles path-tracing engine (high-fidelity photorealism)
- **Nature scenes**: 1920×1080, 240 frames, 128 samples/frame
- **Indoor scenes**: 1920×1080, 120 frames, 196 samples/frame
- **Denoising**: OpenImageDenoise

**关键intuition**: 196 samples对indoor scenes来说相当高，说明他们真的在追求photorealism而不是fast prototyping。Cycles path-tracing是physically-based rendering，这跟paper整体"physics-grounded"的主题一致。

---

## 7. Intuition Building: 核心设计哲学

### 7.1 为什么Code-as-Simulation胜过Neural 4D?

对比MAV3D (Singer et al., 2023)用dynamic NeRF + video diffusion priors:
- **Computational cost**: NeRF优化极慢
- **Editability**: NeRF是black-box，无法精细控制单个object
- **Physical consistency**: NeRF的density field没有physics constraint，可以违反gravity

对比DreamGaussian4D (Ren et al., 2023):
- 虽然用4D Gaussian Splatting提升效率
- 仍然缺乏full-scale 4D scene的physical consistency

**Code2Worlds的approach**: 把generation问题转成**program synthesis**问题，physics engine（Blender）作为grounding substrate，natural language作为指令interface。

### 7.2 为什么Dual-Stream胜过Monolithic?

类比 transformer里的attention head specialization：
- Monolithic LLM在单次forward里同时处理local和global scale，必然有一个scale被compromise
- Dual-stream类似MoE (Mixture of Experts)的思想：让specialized stream处理对应scale
- Object Stream: focus on参数空间的fine-grained控制
- Scene Stream: focus on全局consistency和constraint satisfaction

### 7.3 为什么Closed-Loop胜过Open-Loop?

类比 self-refine / Reflexion / LATS (Zhou et al., 2024) 等agent work:
- Open-loop LLM coding像不跑test的coding → physical hallucinations
- Closed-loop通过VLM-Motion做visual verification，相当于把visual feedback作为implicit reward signal
- 这种closed-loop类似AlphaCodium的test-driven refinement，只不过test是visual fidelity而非unit test

### 7.4 为什么VLM Critic胜过LLM Self-Judge?

类比 RLHF里human reward model vs LLM self-evaluation:
- LLM无法直接evaluate visual quality，只能check textual coherence
- VLM能bridge modality gap：看rendered image/video直接judge physical plausibility
- VLM-Critic和VLM-Motion形成两个scale的critique：static (image) + dynamic (video)

### 7.5 Retrieval-Augmented Generation的必要性

类比RAG在document QA的作用:
- Infinigen的procedural API极其复杂，LLM缺乏domain-specific priors
- 直接从prompt生成script容易syntactic error (从ablation看SGS暴跌到23.5)
- 通过retrieve verified canonical implementations，LLM只需要adapt而不是create from scratch
- 这跟Voyager (Minecraft)用skill library、Eureka用reward examples的思路一致

---

## 8. 联想到的相关工作

### 8.1 Procedural Modeling方向

- **Infinigen** (Raistrick et al., 2023): https://arxiv.org/abs/2306.09310 — infinite photorealistic worlds via procedural generation，Code2Worlds的upstream dependency
- **Infinigen Indoors** (Raistrick et al., 2024): https://arxiv.org/abs/2406.11824 — 扩展到indoor scenes
- **3D-GPT** (Sun et al., 2023): https://arxiv.org/abs/2310.12945 — LLM procedural 3D modeling的pioneer
- **SceneCraft** (Hu et al., 2024): https://arxiv.org/abs/2403.01248 — LLM agent for Blender code synthesis
- **VULCAN** (Kuang et al., 2026): https://arxiv.org/abs/2512.22351 — tool-augmented multi-agents for 3D arrangement
- **LL3M** (Lu et al., 2025): https://arxiv.org/abs/2508.08228 — Large Language 3D Modelers

### 8.2 4D Generation方向

- **MAV3D** (Singer et al., 2023): https://arxiv.org/abs/2301.11280 — text-to-4D via dynamic NeRF
- **DreamGaussian4D** (Ren et al., 2023): https://arxiv.org/abs/2312.17142 — 4D Gaussian Splatting
- **SP-GS** (Wan et al., 2024): https://arxiv.org/abs/2406.03697 — Superpoint Gaussian Splatting for dynamic scenes

### 8.3 Multi-Agent + Reflection方向

- **LATS** (Zhou et al., 2024): https://arxiv.org/abs/2310.04406 — Language Agent Tree Search, unifies reasoning+acting+planning
- **Reflexion** (Shinn et al., 2023): https://arxiv.org/abs/2303.11366 — verbal reinforcement learning
- **Liu et al., 2023** (LLM visual instruction tuning): https://arxiv.org/abs/2304.08485

### 8.4 Video Generation (作为对比)

- **SVD** (Blattmann et al., 2023): https://arxiv.org/abs/2311.15127 — Stable Video Diffusion
- **AnimateDiff** (Guo et al., 2024): https://arxiv.org/abs/2307.04725
- **CogVideoX** (Yang et al., 2024): https://arxiv.org/abs/2408.06072
- **HunyuanVideo** (Kong et al., 2024): https://arxiv.org/abs/2412.03603

### 8.5 Code-as-Policy类工作 (强烈相关)

- **Voyager** (Wang et al., 2023): https://arxiv.org/abs/2305.16291 — LLM as Minecraft agent with skill library
- **Eureka** (Ma et al., 2023): https://arxiv.org/abs/2310.12931 — LLM-generated reward functions for RL
- **AlphaCodium** (Ridnik et al., 2024): https://arxiv.org/abs/2401.08500 — code generation with iterative test-driven reflection
- **Code as Policies** (Liang et al., 2022): https://arxiv.org/abs/2209.07753 — LLM for robot control

### 8.6 World Models方向 (philosophical related)

- **LeCun's JEPA** (2022): https://openreview.net/pdf?id=BZ5a1r-kVsf — paper里引用的spatial intelligence目标
- **Sora** (OpenAI, 2024): https://openai.com/research/sora — video diffusion as world simulator的alternative approach
- **Genie** (Bruna et al., 2024): https://arxiv.org/abs/2402.09184 — interactive world models

---

## 9. Critique 和 Limitations

### 9.1 Paper自己承认的局限
- **Fidelity-latency tradeoff**: physics engine + iterative VLM feedback导致computational bottleneck，无法real-time generation
- Future work: neural physics distillation加速simulation

### 9.2 我的观察

**优点**:
1. Dual-stream factorization非常elegant，把multi-scale entanglement问题拆解得很清晰
2. VLM-Motion Critic是个很novel的设计，把RLHF思想扩展到temporal domain
3. Code4D benchmark的prompt设计确实很有挑战性
4. Failure Rate从30-70%降到10%是很有说服力的硬指标

**潜在问题**:
1. **完全依赖Infinigen**: 这意味着方法bottleneck在procedural generator的coverage。如果Infinigen不支持某种object（如复杂工业机械），Code2Worlds也无力生成
2. **Gemini 3依赖**: 所有agent都用同一个LLM，没有modular LLM specialization（如用不同model做planning vs code generation）
3. **L_param和L_code的人工构建成本高**: 这两个library需要manual curation，scalability有问题
4. **Physics Failure Rate 10%仍然不算低**: 对sim-to-real transfer来说，10%的failure可能仍然影响downstream training
5. **没有跟Sora等最新video diffusion baseline对比**: Hunyuan的Failure Rate已经30%，跟Code2Worlds的10%比较接近，但Hunyuan的editability和controllability可能更强
6. **No real 4D video comparison with Gaussian splatting methods**: 没跟SP-GS, DreamGaussian4D直接对比4D scene generation
7. **Cost analysis缺失**: 没有报告每次generation需要多少秒、多少API calls、多少token consumption

---

## 10. 个人Intuition总结

这篇paper最深刻的贡献其实是**把world generation重新frame为program synthesis问题**，而不是neural representation问题。这背后的哲学跟LeCun的JEPA、Sora的latent diffusion走的是两条不同路径：

- **Sora路径**: 用大规模video data学习implicit world dynamics，hope model能emerge出physics understanding
- **Code2Worlds路径**: 用explicit physics engine作为substrate，LLM只负责semantic→code translation，physics correctness由Blender保证

Code2Worlds的优势是**compositional generalization**——可以通过组合不同procedural primitives生成训练时没见过的scene。劣势是**expressiveness ceiling**受限于Infinigen的API coverage。

VLM-Motion Critic是个非常promising的方向。这种**render-and-critique loop**跟RL里的model-based planning非常类似——VLM作为learned dynamics model的approximation，evaluate当前rollout的quality，然后propagate feedback回去调整action (physics parameters)。这跟AlphaGo的MCTS+value network有相似spirit，只不过在visual generation domain。

未来可能的extension方向：
1. **Neural physics distillation**: 把Blender physics engine distill成neural surrogate，加速10-100x
3. **Active learning for L_param**: 用LLM自动discover和annex新的parameter schema
4. **Cross-embodiment transfer**: 同一framework支持不同physics engine (Blender, Isaac Gym, MuJoCo)
5. **Interactive editing**: 用户可以局部修改4D scene（如"让风更强"）而不需要完全regenerate

代码和project page:
- GitHub: https://github.com/AIGeeksGroup/Code2Worlds
- Website: https://aigeeksgroup.github.io/Code2Worlds

---

## 11. 关键References链接汇总

主要相关papers:
- [Infinigen](https://arxiv.org/abs/2306.09310) — upstream procedural generator
- [3D-GPT](https://arxiv.org/abs/2310.12945) — LLM procedural 3D pioneer
- [SceneCraft](https://arxiv.org/abs/2403.01248) — Blender code synthesis agent
- [LATS](https://arxiv.org/abs/2310.04406) — Language Agent Tree Search
- [MAV3D](https://arxiv.org/abs/2301.11280) — text-to-4D NeRF
- [DreamGaussian4D](https://arxiv.org/abs/2312.17142) — 4D Gaussian Splatting
- [SVD](https://arxiv.org/abs/2311.15127) — Stable Video Diffusion
- [CogVideoX](https://arxiv.org/abs/2408.06072) — Text-to-video diffusion
- [HunyuanVideo](https://arxiv.org/abs/2412.03603) — Large video generative models
- [ImmerseGen](https://arxiv.org/abs/2506.14315) — Agent-guided immersive world generation
- [VULCAN](https://arxiv.org/abs/2512.22351) — Tool-augmented multi-agents for 3D
- [LL3M](https://arxiv.org/abs/2508.08228) — Large Language 3D Modelers
- [MeshCoder](https://arxiv.org/abs/2508.14879) — LLM-powered mesh code generation
- [LeCun JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf) — path towards autonomous machine intelligence
- [Voyager](https://arxiv.org/abs/2305.16291) — LLM agent with skill library
- [Eureka](https://arxiv.org/abs/2310.12931) — LLM reward generation
- [CLIP](https://arxiv.org/abs/2103.00020) — used in evaluation metrics
- [VBench](https://arxiv.org/abs/2311.17982) — video generation benchmark
- [GPT-4o](https://arxiv.org/abs/2410.21276) — used for SGS, Richness, HRS evaluation
- [DreamFusion](https://arxiv.org/abs/2209.14988) — text-to-3D via 2D diffusion
- [Magic3D](https://arxiv.org/abs/2211.10440) — high-resolution text-to-3D

希望这个分析帮你build up对procedural code generation for 4D worlds的intuition。最核心的take-away是：**在physics engine上做closed-loop LLM program synthesis可能比纯neural generation更适合需要physics grounding的任务**，而VLM-Motion Critic这种render-and-critique loop是连接semantic instruction和physical simulation的key bridge。
