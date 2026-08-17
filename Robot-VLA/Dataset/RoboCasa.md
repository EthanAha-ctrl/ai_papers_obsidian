---
source_pdf: RoboCasa.pdf
paper_sha256: 7aca93ca3f157e32a46e02d675b3b2c24e3073a7807c038f7f566b73db87bde3
processed_at: '2026-08-12T00:31:56-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboCasa 人话版

好，我换个调调，用更直白的方式讲。

## 这 paper 到底在干啥

robot learning 现在卡在一个尴尬的地方：你想 train 一个 generalist robot，就像 train GPT 一样需要海量 data，但 robot data 贵得离谱。Google 搞 RT-1 花了 17 个月、13 台 robot、几千小时才攒了 130K 条 trajectory。Open X-Embodiment 更是 22 个机构一起搞。

那怎么办？**simulation 呀**。simulation 几乎不要钱，你想生成多少 data 都行。问题在于，以前的 simulation 太 "塑料" 了 — 就那么几个 scene、几十个 object，policy 在里面学到的都是 "背地图"，换个 kitchen 就废。

RoboCasa 想干的事情就是：**把 simulation 做得跟 real world 一样 diverse**，让 policy 在里面学的不是 "这个特定 kitchen 怎么操作"，而是 "kitchen 这个东西一般怎么操作"。

## 他们具体做了啥

### Scene：120 个 kitchen 怎么来的

非常 simple 的组合数学：

```
10 种 floorplan (L 型、U 型、一字型、带 island 的...) 
   × 
12 种 style (北欧风、地中海风、工业风...)
   =
120 个 kitchen
```

然后每个 kitchen 还可以随机换 texture (墙壁、地板、台面、柜门各有 100 种 AI 生成的 texture)，相当于在 120 个基础 kitchen 上再叠一层 visual randomization。

**直觉**：这就像你训练 ResNet 时不只用 ImageNet 的 1.2M 张图，还要做 random crop、color jitter、horizontal flip 一样 — diversity 是防 overfit 的根本。

### Object：2500+ 个 3D 物体

这里他们用了一个聪明做法：

- **1592 个来自 Luma.ai** — 就是 text-to-3D 生成出来的，输入 "a ripe tomato" 给你一个 3D tomato
- **917 个来自 Objaverse** — 已经有的 3D dataset

153 个 category，从 apple 到 zucchini，从 mug 到 microwave。

**为什么这个重要**：以前 simulation 里的 object 都是手工建模的，做个 mug 就要一天。现在 text-to-3D 一秒钟给你一个，虽然 quality 参差，但架不住量大。这跟 LLM 的 "data quality 不够，data quantity 来凑" 思路一样。

### Appliance：会动的家具

microwave 有门可以开，stove 有 knob 可以拧，拧完 burner 还会 "亮" (state change)。这比静态 mesh 高一个 level — policy 要学会 "拧 knob" 这个 action 才能触发后续效果。

### Task：100 个，25 + 75

**25 个 atomic task**：最基本的 building block
- Pick & place (8 个变种：counter→cabinet, cabinet→counter, counter→sink...)
- Open / close door (4 个：单门、双门、开、关)
- Open / close drawer (2 个)
- Twist knob (2 个：开 stove、关 stove)
- Turn lever (3 个：sink faucet on/off/spout)
- Press button (3 个：coffee machine、microwave on/off)
- Insertion (2 个：往咖啡机塞 mug、取 mug)
- Navigation (1 个)

**75 个 composite task**：用 LLM 生成的

这部分最有意思。他们做了两步：

**Step 1**: 问 GPT-4 "给我 30 个常见 kitchen 活动"
→ 得到 brewing coffee、washing dishes、frying、baking... 人工筛出 20 个。

**Step 2**: 对每个活动，让 GPT-4 / Gemini 给出具体 task blueprint，格式是：
```
Task: PrepareCoffee
Goal: Place a mug under coffee machine and press start
Objects: mug
Fixtures: coffee machine, cabinet
Skills: Pick_up(mug), Place(mug, coffee_machine), Press(button)
Reasoning: 这是咖啡的标准准备流程
```

**但 LLM 会犯傻**，paper 里举了 3 个典型错误：

1. **幻觉 object**: 让 robot 用 blender，但 sim 里压根没 blender
2. **乱用 skill**: 让 robot "uncork wine" 的方式是按 coffee machine button — LLM 觉得 "press" 跟 "uncork" 语义上沾边就硬塞
3. **抓不该抓的**: 让 robot 抓 utensil，但 utensil 又扁又薄，sim 里根本抓不起来

所以最后还是得人 filter 一下，得到 75 个能 actually 跑的 task。

**我的 read**：这说明 LLM 的 "common sense" 是 surface level 的，真正落地到 physics / affordance constraint 还差一层。这跟你经常说的 "LLM 有 intelligence 但没有 grounding" 完全一致。

## Data 怎么来的：100K+ trajectories

这部分是 paper 最务实的部分。

### 人工采集：1250 条

4 个 operator 用 SpaceMouse，每个 atomic task 采 50 条。每次在 random kitchen scene 里采，所以天然有 diversity。

但 1250 条 train 出来的 policy success rate 只有 28.8% — 远远不够。

### MimicGen 自动生成：72K 条

这里用了一个叫 MimicGen 的工具 (NVIDIA 自己的 work)。核心 idea 很 elegant：

**人怎么 demo 的**：
1. 走到 object A 旁边
2. 抓住 A
3. 抬起来
4. 搬到 object B 旁边
5. 放下

**MimicGen 怎么做的**：
- 把这条 trajectory 按 "object reference frame" 切成 5 段
- 段 1-3 以 object A 为坐标原点
- 段 4-5 以 object B 为坐标原点
- 新场景里 A、B 位置变了，就把每段做 rigid transform 对齐到新位置
- stitch 起来，让 robot 跑一遍，成功就保留，失败就丢

这样 1 条 human demo 可以 "繁殖" 出几十条 new demo。RoboCasa 用 50 条 human demo 生成出 3000 条 / task，总共 72K 条。

**为什么这个 trick 行得通**：因为 atomic task 的结构都一样 (pick → place)，subtask 切分逻辑可以复用。pick-place 的 subtask sequence 写一次，所有 8 个 pick-place task 都能用。

## Policy 长啥样

### BC-Transformer

输入：
- 过去 10 步 observation (image + proprio)
- language goal (用 CLIP encode)

每个 observation 有 3 个 camera：eye-in-hand + 左右两个 workspace camera。每个 image 过一个 ResNet-18，然后用 **FiLM** 把 language 信息融合进去：

$$\text{FiLM}(x) = \gamma(\ell) \odot x + \beta(\ell)$$

变量解释：
- $x$ 是 visual feature
- $\ell$ 是 language embedding
- $\gamma, \beta$ 是 $\ell$ 经过 MLP 得到的 scale 和 shift 参数
- $\odot$ 是 element-wise 乘法

直觉：language 在 modulate visual feature — 看到 "pick up the mug" 时，mug 相关的 visual feature 被放大，其他被抑制。

Transformer 6 层，20M 参数，输出未来 10 步 action，执行 1 步就 replan。

### 为什么不用 Diffusion Policy

他们也试了，结果差很多 (BC-Transformer 56% vs Diffusion Policy 12% 在 single task 上)。

作者 hypothesis：Diffusion Policy 默认只用 2 步 history，而 BC-Transformer 用 10 步。mobile manipulator 需要 history 来推断 base motion 和 task 进度。

**我的吐槽**：这个 comparison 有点 unfair。Diffusion Policy 在 action distribution multi-modal 的时候才真正发光，RoboCasa 用 OSC + 6D pose action 本身已经比较 deterministic 了，Diffusion 的优势体现不出来。如果用 joint torque action 或者多物体场景 (抓 A 还是抓 B 都行)，结果可能反过来。

## 实验结果

### Atomic task 的 scaling

| 数据量 | Success rate |
|---|---|
| Human 50 demos/task | 28.8% |
| Generated 100 demos/task | 26.3% |
| Generated 300 demos/task | 35.0% |
| Generated 3000 demos/task | **47.6%** |

**这就是 paper 最核心的 figure** — scaling trend 在 robotic imitation learning 里出现了。

但要注意：
- **量少时 synthetic data 不如 human data** (Gen-100 26.3% < Human-50 28.8%)，说明 MimicGen 的 trajectory 质量确实比 human 差一些
- **量大了就反超**，说明 quantity 补 quality 这个逻辑在 robotics 里也成立

按 skill 分解看更有意思：
- **Pick-and-place 最难** (Gen-3000 avg ~24%)：object 太多太杂，affordance 各不相同
- **Drawer / button 最容易** (Gen-3000 avg ~85% / 75%)：geometry 受限，interaction 模式单一
- **Insertion 难** (~23%)：dexterity 要求高

**直觉**：这跟 LLM 里 "structured output 容易，creative writing 难" 类似。受约束的动作 space 容易学，open-ended 的 interaction 难学。

### Composite task：暴露了 BC 的短板

5 个 composite task，每个 50 human demos：
- Scratch: 0% / 0% / 0% / 0% / 2%
- Fine-tune from atomic: 0% / 2% / 4% / 6% / 12%

**结论**：BC 在 long-horizon 上 fundamental 挣扎。即使有 atomic pretraining，stage 之间的 transition 还是 hard。

**我的 read**：这其实印证了你需要 hierarchical policy 或者 RL fine-tuning。BC 学得到 "pick up mug" 这个 primitive，但学不到 "做完 step 1 之后要检查一下，然后 transition 到 step 2" 这个 meta-level 的 control。RT-2 用 chain-of-thought 来做这个，VLA 用 language 作为中间 representation，都是 trying to solve 这一层。

### Sim2Real 初步验证

最 exciting 的部分。real Franka + DROID hardware，3 个 pick-place task，每个 50 real demos：

| | Real only | Real + Sim |
|---|---|---|
| Seen object avg | 13.6% | **24.4%** |
| Unseen object avg | 2.6% | **9.3%** |

**关键 insight**：unseen object 的改善 (+258% relative) 比 seen object (+79% relative) 还大。这说明 sim data 主要贡献 generalization，不是让 policy "记住" 训练 object。

**但要 honest**：
- 这不是 zero-shot sim2real，还需要 50 条 real demo
- Real controller (15 Hz, no OSC) vs Sim (20 Hz, OSC) 有 gap，但能 transfer 说明 visual feature 占主导
- 总体 success rate 还是个位数到 20 几，离实用还远

## 和你视角的 connection

你之前在 Tesla 反复讲 "data is all you need for autonomy"。RoboCasa 印证了这个 thesis 的一半：**scale matters, diversity matters**。从 1250 human demo (28.8%) 到 72K synthetic demo (47.6%)，相对提升 65%。

但另一半 — **long-horizon composition** — 还没 scale 起来。Composite task 即使 fine-tune 也只有个位数。这说明光有 data scale 不够，还需要：
- 更好的 policy architecture (VLA? Hierarchical?)
- 更好的 learning paradigm (RL fine-tuning? Chain-of-thought?)
- 更好的 data quality (MimicGen 轨迹 jerky, 可以加 quality filter)

这跟你 Eureka Labs 的方向、跟你 Tesla Optimus 的 RL fine-tuning 思路都是呼应的。

## 几个我想跟你讨论的点

1. **Sim2real 还需要 real demo 这个事**：什么时候 sim data 能完全 zero-shot transfer？是不是需要 1M+ trajectories + 更 aggressive domain randomization？
2. **LLM 生成 task 还要人 filter**：什么时候 LLM 能自己写 MJCF code、自己 ground 到 affordance？GenSim / RoboGen 走得远一点但 quality 差。这跟 VLA 的 grounding 问题是同一回事吗？
3. **Diffusion Policy 在 RoboCasa 表现差**：你觉得是 history length 的问题，还是 action space 的问题，还是 benchmark 本身 bias?
4. **BC 的 long-horizon bottleneck**：你觉得 hierarchical policy、VLA、RL fine-tuning 哪条路最有希望？

## References

- RoboCasa 官网: https://robocasa.ai
- RoboCasa arXiv: https://arxiv.org/abs/2406.02523
- MimicGen (数据生成核心): https://arxiv.org/abs/2310.17596
- RoboMimic (BC-Transformer 基础): https://arxiv.org/abs/2103.04045
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Luma AI text-to-3D: https://lumalabs.ai
- Objaverse: https://objaverse.allenai.org
- DROID dataset: https://droid-dataset.github.io
- Open X-Embodiment: https://robotics-transformer-x.github.io
- RT-1: https://robotics-transformer1.github.io
- GenSim (LLM 生成 task): https://arxiv.org/abs/2310.01361
- RoboGen: https://arxiv.org/abs/2311.01455
- Behavior-1K: https://arxiv.org/abs/2303.13484
- Habitat 2.0: https://arxiv.org/abs/2106.14405
- ManiSkill 2: https://arxiv.org/abs/2302.04659
- LIBERO: https://arxiv.org/abs/2306.03310
- NVIDIA Omniverse: https://developer.nvidia.com/omniverse
- FiLM: https://arxiv.org/abs/1709.07871
- RoboSuite (基础框架): https://github.com/ARISE-Initiative/robosuite
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

要不要我展开讲讲 MimicGen 的 segment alignment 算法细节？那部分数学其实挺 elegant 的，我上面简化了很多。

---

# RoboCasa 深度技术解读

你好 Andrej，这篇 RoboCasa 是 UT Austin + NVIDIA Research 的工作，一作 Soroush Nasiriany，二作 Abhiram Maddukuri / Lance Zhang 共享。我下面从直觉到细节讲一下。

## 1. 一句话核心 thesis

RoboCasa 的核心 thesis 是：**用 Generative AI 作为 diversity engine，把 simulation 从 "少数 hand-crafted 场景" 扩展到 "几千 objects × 上百 scenes × 上百 tasks × 100K+ trajectories" 的规模**，从而在 robotic imitation learning 中复现出类似 LLM / CV 那种 scaling 的趋势。

Paper 的标题就叫 "Large-Scale Simulation of Everyday Tasks for Generalist Robots" — 关键词是 *generalist* + *everyday* + *large-scale*。

## 2. 为什么这篇 paper 重要 — 它在 robotic learning 的 "scaling 瓶颈" 上做了什么

你过去在 Tesla 反复强调 data is all you need。Robotics 的 data wall 在于：

- 真实世界采集成本高 (Open X-Embodiment、DROID、RT-1 dataset 都是巨资)
- 多样性受限 (几个 lab、几个 kitchen、几个 robot)
- 长程任务 failure 多，无法 bootstrapping

RoboCasa 的应对路径是 **simulation as data factory**，但 simulation 一直被诟病两点：(a) reality gap，(b) diversity 太低 (比如 LIBERO 5 scenes、ManiSkill2 4 scenes)。这篇 paper 的 contribution 主要就是解决 (b)：
- 用 text-to-3D (Luma.ai) + text-to-image (MidJourney) 把 asset pipeline 打通
- 用 LLM (GPT-4 + Gemini 1.5) 把 task definition pipeline 打通
- 用 MimicGen 把 trajectory pipeline 打通

这构成了 "three pillars of scaling"。

## 3. 仿真器架构 (Section III)

### 3.1 Base platform

继承自 RoboSuite (Zhu et al. 2020)：
- Physics engine: **MuJoCo** (deepmind 维护)
- 模型格式: **MJCF** (MuJoCo XML)
- Controller: Operational Space Control (OSC)，workspace end-effector control
- 控制 frequency: 20 Hz (仿真) vs 15 Hz (DROID real robot)
- 每一 timestep = 0.04 s (25 fps)

性能：在 PickPlaceCounterToCab 任务上，渲染 25.2 fps ≈ real-time；不渲染 31.9 fps。Reset 平均 9.5 s（场景随机化代价）。

### 3.2 Cross-embodiment

这是相对 RoboSuite 的关键扩展。RoboCasa 支持：
- Single-arm mobile platform (Franka + Omron base，类似 Omni-Frankie [Haviland 2022])
- Humanoid robots
- Quadruped with arms

模型来自 MuJoCo Menagerie + 各 robotics repo。这点对训练 generalist policy 很重要 — 你在 Eureka 时也强调过 cross-embodiment 是 foundation model 的必要条件。

### 3.3 Rendering

支持 NVIDIA Omniverse (photorealistic) + 原生 MuJoCo renderer (lightweight, 用于大规模数据生成)。Dataset 用 MuJoCo renderer 是为了速度。

## 4. Scene generation — 120 kitchens 是怎么来的 (Section III-B)

这是 paper 的一个 gem。它们用 **combinatorial 设计**：

$$N_{\text{scenes}} = N_{\text{floorplans}} \times N_{\text{styles}} = 10 \times 12 = 120$$

- $N_{\text{floorplans}}$ = 10，从建筑杂志整理出的 layout：one-wall / L-shape / U-shape w/ island / galley 等
- $N_{\text{styles}}$ = 12：Industrial / Scandinavian / Coastal / Modern / Traditional / Mediterranean / Rustic / ...

每个 style 是一组 design elements 的 bundle：
- **texture combination** (cabinet panel + counter + floor + wall)
- **appliance choice** (e.g. Mediterranean 用 ornate glass panel cabinet)
- **handle / knob style**

再叠加 **AI-generated textures** (MidJourney) 做 domain randomization：
- 100 wall textures
- 100 floor textures
- 100 counter textures
- 100 cabinet panel textures

实际训练时这些 texture 是随机采样的，相当于 $\text{style} \times \text{texture}$ 的 cartesian product。

**直觉**：这等于在 visual appearance 上做了一个 "structured randomization"。不是纯 random noise，而是有语义层次的 randomization (Scandinavian style 内部还是 consistent 的)。

## 5. Assets — 2509 个 objects 怎么来的 (Section III-C)

### 5.1 Sourcing pipeline

| 来源 | 数量 | 备注 |
|---|---|---|
| Luma.ai (text-to-3D) | 1,592 | 主要来源，生成式 |
| Objaverse [Deitke 2022] | ~917 | 已有 3D dataset |
| **Total** | **2,509** | **153 categories** |

### 5.2 Appliances 的 articulated 处理

把 appliances 拆成 articulated bodies：
- Microwave → door (hinge joint) + body
- Stove → knob (revolute joint) + burner (visual state)
- Coffee machine → button (linear joint) + nozzle

并实现 **state changes**：当 stove knob angle 超过阈值，对应 burner visual 切换到 "on" 状态。这对 sim2real 很重要，policy 要学会 knob twisting 才能触发下游 effects。

### 5.3 Filter pipeline

Luma.ai / Objaverse 出来的 mesh 经常有缺陷：non-manifold、duplicate vertices、bad UV。他们做 manual filtering + post-processing 转成 MJCF。

## 6. Task generation — 这是 paper 最 LLM-heavy 的部分 (Section IV-B)

### 6.1 Atomic tasks (25 个)

8 个 sensorimotor skills，每个 skill 对应若干 task variants：

| Skill | Example tasks | # |
|---|---|---|
| Pick-and-place | PickPlaceCounterToCabinet, ...ToSink, ...ToMicrowave, ...ToStove, reverse directions | 8 |
| Open/close doors | OpenSingleDoor, CloseSingleDoor, OpenDoubleDoor, CloseDoubleDoor | 4 |
| Open/close drawers | OpenDrawer, CloseDrawer | 2 |
| Twist knobs | TurnOnStove, TurnOffStove | 2 |
| Turn levers | TurnOnSinkFaucet, TurnOffSinkFaucet, TurnSinkSpout | 3 |
| Press buttons | CoffeePressButton, TurnOnMicrowave, TurnOffMicrowave | 3 |
| Insertion | CoffeeSetupMug, CoffeeServeMug | 2 |
| Navigation | NavigateKitchen | 1 |

这 8 个 skills 的选择是受到 manipulation affordance 启发 — 都是 kitchen 里高频出现的 action primitive。

### 6.2 Composite tasks (75 个) — LLM-driven generation pipeline

这是 paper 最有意思的 design。两阶段 prompt：

**Stage 1**: 让 GPT-4 列 high-level kitchen activities
> "Can you give me 30 simple everyday high-level kitchen activities? Each activity should be unique."

人工筛出 20 个 activity label:
brewing coffee, washing dishes, restocking supplies, chopping food, making toast, defrosting, boiling water, meat prep, setting table, clearing table, sanitizing, snack prep, tidying, washing fruits/veg, frying, reheating, mixing/blending, baking, serving food, steaming vegetables.

**Stage 2**: 对每个 activity，让 GPT-4 / Gemini 1.5 few-shot prompt 出 concrete task blueprint，包含：
- Task name
- Goal
- Objects (限制在 RoboCasa 已有 category)
- Fixtures (cabinet, stove, microwave, ...)
- Skills sequence (从 atomic skills 中组合)
- Reasoning

### 6.3 LLM 失败案例 — 这部分很 valuable

Paper Section VIII-B 列出 LLM 出错的 3 种典型 mode：

1. **Hallucinated object**: "Set Up Blending Station" 用了 blender，但 RoboCasa 没有 blender
2. **Misuse of skill**: "Wine Selection for Cooking" 让 robot 用 `Press(button_on_coffee_machine)` 来 "simulating uncorking" — LLM 把语义强行塞进不合适的 skill
3. **Grasping non-graspable**: "Retrieve Baking Utensils" 让 robot pick up utensils，但 utensils 在 sim 里 thin 且 affordance 差

**直觉**：这给了我们一个重要的 takeaway — LLM 出 task 的 "common sense" 是 surface-level 的，真正 deploy 到 sim 里需要 grounding 到 affordance / physics constraint。这是 RoboCasa 没完全自动化、需要 human filter 的原因。

## 7. Dataset 生成 — MimicGen (Section IV-C)

这是 dataset 达到 100K 的关键。

### 7.1 Human collection

- 4 个 operators，用 3D SpaceMouse
- 每个 atomic task 50 demos → 25 × 50 = 1,250 demos
- 每次在 random scene (random floorplan + style + textures) 中采集

### 7.2 MimicGen 的 mechanism

MimicGen (Mandlekar et al. 2023) 把每个 human demo 分解为 object-centric manipulation segments:

$$\tau_{\text{human}} = [s_1, s_2, \ldots, s_K]$$

每个 segment $s_k$ 对应一个 subtask (e.g. "approach object", "grasp", "lift", "transport to target", "release")，且 reference frame $F_k$ 是该 subtask 的 relevant object 的 frame。

对一个 new scene 配置 $\theta_{\text{new}}$ (新 object pose、新 robot init pose)，对每个 segment 做 rigid transform：

$$s_k^{\text{new}} = T(F_k^{\text{old}} \to F_k^{\text{new}}) \cdot s_k$$

然后 stitch 起来，让 robot 跟踪这条新 trajectory，用 sim 验证 task success；fail 则 reject (rejection sampling)。

### 7.3 RoboCasa 里如何 instantiate

关键 insight：atomic tasks 都 share 同一个 skill 结构，所以 subtask sequence 可以 per-skill 写一次，重用：

```python
# pseudo
subtask_sequence[Skill.PICK_PLACE] = [
    ("approach", obj_pick),
    ("grasp",    obj_pick),
    ("lift",     obj_pick),
    ("transport",obj_place),
    ("release",  obj_place),
]
end_condition[Skill.PICK_PLACE] = [
    lambda obs: gripper_close_to(obj_pick),
    lambda obs: gripper_closed and obj_in_gripper,
    ...
]
```

每个 segment 的 end-condition 用自动 metric 检测 (e.g. grasp success = gripper closed + object in gripper)，只在 skill 层面写一次。

### 7.4 生成结果

| Dataset | Demos per task | Total |
|---|---|---|
| Human-50 | 50 | 1,250 |
| Generated-100 | 100 | 2,400 |
| Generated-300 | 300 | 7,200 |
| **Generated-3000** | **3000** | **72,000** |

并行化 sim 进程加速生成。MimicGen 失败的 attempts 通过 rejection sampling 丢弃，所以 dataset 全是 task success 的 trajectory。

## 8. Policy architecture (Section IX)

### 8.1 BC-Transformer

输入：
- $\mathbf{o}_{t-9:t}$：过去 10 步 observation
- $\ell$：language goal (CLIP text encoder)

每个 observation 包含：
- Proprio: end-effector pose + mobile base pose
- 3 cameras: eye-in-hand + left workspace + right workspace

每个 image 用 dedicated **ResNet-18** encode，fuse 用 **FiLM** layers (Feature-wise Linear Modulation):

$$\text{FiLM}(\mathbf{x}) = \gamma(\ell) \odot \mathbf{x} + \beta(\ell)$$

其中 $\gamma, \beta$ 是 language embedding $\ell$ 经过 MLP 得到的 affine 参数，$\odot$ 是 element-wise product。这让 visual feature 被 language goal modulate，类似 CLIP-conditioned UNet 思路。

Transformer 6 层，~20M trainable params，输出未来 10 个 action，replan after first action (类似 Diffusion Policy 的 action chunking 思路)。

训练：500K gradient steps，lr = 1e-4，warmup。

### 8.2 Diffusion Policy 对比

也跑了 Diffusion Policy (Chi et al. 2023)：
- Same ResNet-18 + FiLM encoder
- History = 2 (default), pred horizon = 16, action horizon = 8
- DDIM: 100 train timesteps, 10 inference timesteps

结果：在 PickPlaceCounterToSink 上 BC-Transformer 56% vs Diffusion Policy 12%。

**作者 hypothesis**：history length 2 对 Diffusion Policy 太短，BC-Transformer 用 10。这可能是因为 mobile manipulator 需要 history 来推断 base motion 和 task progress。

**我的解读**：这其实是一个 mildly unfair comparison。Diffusion Policy 在更高维、更 multi-modal 的 action space 上优势明显，但 RoboCasa 里用的是 OSC + 6D pose action，本身已经比较 unimodal 了。

## 9. 实验结果

### 9.1 Atomic tasks scaling (Fig 13 完整表)

| Skill | Human-50 | Gen-100 | Gen-300 | Gen-3000 |
|---|---|---|---|---|
| **Pick & place avg** | ~3% | ~2% | ~8% | ~24% |
| **Door open/close avg** | ~40% | ~45% | ~53% | ~60% |
| **Drawer** | ~61% | ~59% | ~69% | ~85% |
| **Stove knob** | ~18% | ~25% | ~28% | ~35% |
| **Sink lever** | ~47% | ~41% | ~51% | ~67% |
| **Buttons** | ~60% | ~51% | ~60% | ~75% |
| **Insertion** | ~11% | ~7% | ~14% | ~23% |
| **Overall** | **28.8%** | **26.3%** | **35.0%** | **47.6%** |

观察：

1. **Clear scaling trend**: Human-50 (28.8%) → Gen-3000 (47.6%)，+18.8% absolute，+65% relative
2. **Gen-100 略低于 Human-50** (26.3 vs 28.8)：说明量太少时纯 synthetic data 不如 human data 质量
3. **Pick-and-place 最难**：object diversity 高 (dozens of categories)，affordance range 大
4. **Drawer / button 最容易**：geometry 简单，interaction 受限
5. **Insertion 难**：dexterity 要求高

**直觉**：scaling 曲线在不同 skill 上 slope 不同。Pick-place 这种 high-DOF interaction slope 大 (重 data)，button-press 这种受 geometry 限制大的 slope 小。这和你之前在 RT-2 / RT-X 上观察到的 "task 难度决定 scaling benefit" 一致。

### 9.2 Composite tasks (Fig 8)

5 个 composite task，比较 scratch (50 human demos) vs fine-tuning (从 atomic pretrained 模型 + 50 demos):

| Task | Scratch | Fine-tune |
|---|---|---|
| ArrangeVegetables | 2.0% | 12.0% |
| MicrowaveThawing | 0% | 2.0% |
| RestockPantry | 0% | 6.0% |
| PreSoakPan | 0% | 4.0% |
| PrepareCoffee | 0% | 0% |

- **Scratch 几乎全 fail**: 50 demos 学不会 multi-stage task，符合你常说的 "BC 在 long-horizon 上 struggle"
- **Fine-tune 略有改善**: atomic 阶段学到的 pick-and-place / door open 等 primitive transfer 过来，但 stage transition 还是困难
- **作者明确承认这是 weak result**: 留给 future work 探索更好的 policy arch、hierarchical learning、better fine-tuning

**我的解读**：这其实是 RoboCasa 的 honest moment。它证明了 **atomic skill 的 scaling 行得通，但 long-horizon composition 还没 scale 起来**。这呼应了你在 Eureka Labs 也提过的 "skill composition is the next frontier"。

### 9.3 Real-world transfer (Fig 10)

Real Franka + DROID hardware，3 个 pick-place 变种：
- Counter→Sink, Sink→Counter, Counter→Cabinet

每个 task 50 real demos，5 object categories，对比：
- **Real only**: 只用 50 real demos
- **Real + Sim**: 50 real + 全部 atomic MimicGen sim data

**Seen objects (5 类)**:
| Task | Real only | Real + Sim |
|---|---|---|
| Counter→Sink | 12.7 ± 2.5 | **22.0 ± 2.8** |
| Sink→Counter | 20.0 ± 5.9 | **29.3 ± 4.1** |
| Counter→Cabinet | 8.0 ± 1.6 | **22.0 ± 5.8** |
| **Average** | **13.6** | **24.4** (+79% rel) |

**Unseen objects (3 类)**:
| Task | Real only | Real + Sim |
|---|---|---|
| Counter→Sink | 3.3 | **8.9** |
| Sink→Counter | 1.1 | **7.8** |
| Counter→Cabinet | 3.3 | **11.1** |
| **Average** | **2.6** | **9.3** (+258% rel) |

**关键 takeaway**：
1. Real + Sim 的 unseen object 改善 (+258% rel) 比 seen (+79% rel) 还大 — 说明 sim data 主要贡献 generalization，而非 memorization
2. Real robot controller (15 Hz, no OSC) 和 sim (20 Hz, OSC) 不同 — 没做精细 system ID 也能 transfer，说明 visual feature 占主导
3. 这其实就是 "sim as data augmentation" 的一个 demo，并 **不是 zero-shot sim2real** — 真正 zero-shot transfer 还需要进一步 domain randomization / adaptation

## 10. Table I 解读 — 和其他 sim framework 对比

这是 paper 一张很重要的表，值得逐行看：

| Feature | RoboCasa | AI2-THOR | Habitat 2.0 | iGibson 2.0 | RLBench | Behavior-1K | ManiSkill 2 | OPTIMUS | LIBERO | MimicGen |
|---|---|---|---|---|---|---|---|---|---|---|
| Mobile Manip | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Room-Scale | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Realistic Physics | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AI-gen Tasks | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| AI-gen Assets | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Photoreal | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Cross-Embodiment | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Tasks | 100 | 3 | 6 | 100 | 1000 | 8 | 20 | 10 | 130 | 12 |
| Scenes | 120 | 1 | 15 | 1 | 50 | 3 | — | 4 | 20 | 1 |
| Object Cats | 153 | 46 | — | 28 | 1265 | — | — | — | — | — |
| Objects | 2509 | 3578 | 169 | 1217 | 28 | 5215 | 15 | 2144 | 72 | — | 40 |
| Human Data | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Machine Gen | ✓ | ✗ | ✗ | ✗ | ✗ | 0 | 6K | 30K | 245K | 5K | 50K |
| Total trajectories | 100K+ | — | — | — | — | — | — | — | — | — |

RoboCasa 的 unique 组合是 **同时拥有** AI-gen tasks + AI-gen assets + photoreal + cross-embodiment + 100K data。这是 RoboCasa claim 的 "唯一性"。

## 11. Limitations — paper 自己承认的

1. **Composite task 表现差**: fine-tune 后也只是 0-12%
2. **MimicGen 轨迹质量**: 成功但 jerky / 有 collision，没用 trajectory quality filter
3. **LLM task generation 还需 human code**: 没做到 fully autonomous
4. **Scope 限于 kitchen**: 没扩展到其他房间
5. **没有 dexterous / bimanual / deformable manipulation**
6. **没结合 real robot 大数据 + sim + internet video**

## 12. 和你 (Karpathy) 视角的连接

我觉得这篇 paper 的核心 punchline 用你过去的话讲是：**"simulation 在 robotics 里就是 LLM 的 internet"**。RoboCasa 做的事情就是把 simulation 这个 "internet" 做得足够 diverse：
- 用 generative AI 把 asset 端打通 (类似 web 抓 image/text)
- 用 LLM 把 task 端打通 (类似 instruction tuning 的 instruction 多样性)
- 用 MimicGen 把 trajectory 端打通 (类似 self-supervised pretraining 的样本量)

但 paper 也暴露了一个深层问题：**BC on imitation data 在 long-horizon 上不 scale**。Composite task 即使有 atomic pretraining 也只有个位数 success rate。这暗示：
- 要么需要 hierarchical / goal-conditioned policy (像 RT-2 的 chain-of-thought)
- 要么需要 RL fine-tuning (你 Eureka 工作的方向)
- 要么需要 VLA backbone (像 OpenVLA、π0)
- 要么需要更好的 data curation (MimicGen 轨迹质量)

## 13. 我的几个 open question 给你

1. **Sim2real 仍然依赖 50 real demos**: 这是不是说明 sim 还没真正 "replace" real data？是不是还需要更大 scale (比如 1M+ trajectories) 才能 zero-shot transfer？
2. **LLM task generation 还在 blueprint 层面**: 真正的 code (Python / MJCF) 还要 human 写。GenSim、RoboGen 走得更远一点 (LLM 直接写 code)，但 RoboCasa 选择了更 curated 的路径。这是 trade-off: diversity vs quality。
3. **Diffusion Policy 表现差**: 我觉得 history=2 是 cherry-picked default。如果 RoboCasa 自己 tune 一下 (history=10, bigger denoising steps) 应该会更强。这暗示 paper 这部分 experiment 可能不够 rigorous。

## References

- RoboCasa project page: https://robocasa.ai
- RoboCasa arXiv: https://arxiv.org/abs/2406.02523
- RoboSuite (基础框架): https://arxiv.org/abs/2009.12293, https://github.com/ARISE-Initiative/robosuite
- MimicGen (数据生成): https://arxiv.org/abs/2310.17596, https://github.com/NVlabs/MimicGen
- RoboMimic (BC-Transformer): https://arxiv.org/abs/2103.04045, https://github.com/ARISE-Initiative/robomimic
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DROID dataset: https://arxiv.org/abs/2403.12945, https://droid-dataset.github.io
- Objaverse: https://arxiv.org/abs/2212.08051, https://objaverse.allenai.org
- Luma AI text-to-3D: https://lumalabs.ai
- MidJourney (textures): https://www.midjourney.com
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- GenSim (LLM-generated tasks): https://arxiv.org/abs/2310.01361
- RoboGen: https://arxiv.org/abs/2311.01455
- NVIDIA Omniverse: https://developer.nvidia.com/omniverse
- Behavior-1K: https://arxiv.org/abs/2303.13484
- Habitat 2.0: https://arxiv.org/abs/2106.14405
- ManiSkill2: https://arxiv.org/abs/2302.04659
- LIBERO: https://arxiv.org/abs/2306.03310
- Omni-Frankie: https://arxiv.org/abs/2103.00982
- FiLM (Visual reasoning with language): https://arxiv.org/abs/1709.07871
- VIMA (multimodal prompts): https://arxiv.org/abs/2210.03094

如果你想要我 deep dive 哪一块 (比如 MimicGen 的 segment alignment 算法细节、BC-Transformer 的 FiLM fusion 具体、或 LLM prompt 工程)，告诉我，我可以再展开。
