---
source_pdf: sims.pdf
paper_sha256: 335e45ef59b5bb03d2ae8877e2a980711b71ea4610deac2c985dd64681ccfaf4
processed_at: '2026-08-12T06:39:25-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

## 用人话讲 SIMS

---

### 这个 paper 到底想干啥

你想想看, 给你一个 3D 的房间, 里面摆了沙发、床、桌子、书架, 然后给你一句话 "这个人今天心情不好, 在客厅里待了一下午", 你能不能自动生成一段 **物理上合理、又有情感风格、又是长序列** 的人物动作动画?

这件事难在哪:

- **Kinematic 方法** (以前主流): 就是直接从 mocap data 学一个 generative model, 生成动作。问题是你没经过物理引擎, 人物脚会穿地、会滑步、会悬浮, 看着就假
- **Physics 方法** (最近兴起): 把人扔进 Isaac Gym 里跑物理, 不会穿地了, 但之前的 work 都只关心 "坐到椅子上" 这种 task goal, 完全不管你是 **开心地坐** 还是 **抑郁地坐**, 而且 skill 种类少, story 也短

SIMS 想同时拿下三个东西: **physical plausibility + style diversity + long-term story**。这三个之前没人同时做到。

---

### 整体思路: 导演 + 演员

最直观的类比就是拍电影:

- **High-level = 导演**: 拿到 user 的 theme ("今天心情不好"), 先去翻一堆 pre-written 的 short scripts ("抑郁地坐在沙发上", "疲惫地走到床边", "tired 地躺下"), 挑出相关的几个, 然后拼成一个连贯的长 story, 输出一个 keyframe 序列
- **Low-level = 演员**: 拿到 keyframe 序列, 在物理引擎里实际演出来, 同时还要 match 文本描述的风格

导演用 LLM + RAG, 演员用 RL policy。中间用 FSM 把两个 level 串起来。

---

### Short Script Database: 你的"剧本素材库"

你先想一下, 如果直接让 LLM 从零写一个长 story, 会出什么问题? LLM 会偷懒, 写来写去就那几个套路, 而且 narrative 容易跑偏。

SIMS 的 trick 是 **先造一堆"乐高积木"**, 每块积木就是一个 short script, 大概长这样:

```
Summary: "The character feels stressed and seeks comfort in the living room"

Keyframes:
  - sit, stressed, armchair, "sitting with head bowed, hands resting on thighs"
  - touch, armchair
  - walk, stressed, sofa, "walking slowly, hands behind back"  
  - lie, stressed, sofa, "side-lie on left with left arm as pillow, legs bent"
```

每个 short script 有:
- 几个 keyframe, 每个 keyframe 说清 **干什么 (skill) + 拿什么情绪干 (style) + 跟谁交互 (object) + 具体动作细节 (caption)**
- 一句话 summary
- 一个 emotion 标签

这个数据库怎么造? 用 GPT-4, 给它 available skills、captions、styles、objects, 让它批量生成, 然后按 emotion 分 9 类 (happy, angry, hurried, tired, sad, stressed, drunk, relaxed, neutral) 存好。

**关键**: 每个 summary 用 CLIP encoder 算一个 embedding, 当 retrieval key。这就是 RAG 的 setup。

---

### RASG: 检索增强剧本生成

User 给一个 theme, 比如 "a depressed afternoon in the living room", 系统跑三步:

**Step 1**: LLM 先判断这个 theme 跟哪几个 emotion 最相关, 比如选了 sad + tired + stressed, 这就缩小了搜索范围

**Step 2**: 把 theme sentence 过 CLIP, 跟 database 里所有 summary embedding 算 cosine similarity, 每个 emotion style 检索 top-k 个 short script, 拿到 $M \times k$ 个候选

**Step 3**: 把这些 short script 的 summary 喂给 LLM, 让它根据 scene layout (3DFront 的房间布局) 挑合适的, 拼成一个连贯长 story

这里有个细节设计: 为了保证生成出来的 keyframe 序列是 **可执行的**, skill 必须按 tuple 组织, 比如 (sit, getup) 必须成对, (lie, getup) 必须成对, walk 可以插入任何两个 tuple 之间做 transition。这样你不会生成出 "sit 之后直接 lie" 这种物理上别扭的序列。

**Ablation 结果** (Table 6): 直接让 LLM 从零写, SBERT similarity 是 0.8167 (story 之间很像, 不 diverse), 耗时 12.2s; RASG 的 similarity 是 0.7759 (更 diverse), 耗时 7.32s (因为 LLM 只需要从现成素材里挑, 不用从头造)。

---

### Low-level: Multi-Condition Policy

现在拿到 keyframe 序列了, 比如:

```
[walk, tired, sofa, "head bowed while walking"] → 
[sit, sad, sofa, "sitting with head down, hands supporting head"] → 
[idle, sad, "", "bent over with hands on knees"] → 
[lie, tired, bed, "lying down, legs straight"]
```

要驱动一个物理引擎里的 humanoid 真正演出来。这里 SIMS 训了 7 个 policy:

$\pi_w$ (Walk), $\pi_i$ (Idle), $\pi_s$ (Sit), $\pi_l$ (Lie), $\pi_r$ (Reach), $\pi_g$ (GetUp), $\pi_c$ (Carry)

每个 policy 是一个 goal-conditioned RL agent, observation 是四个东西的拼装:

$$\pi(\mathbf{a}_t | \mathbf{s}_t, \mathbf{h}_t, \mathbf{g}_t, \mathbf{z})$$

- $\mathbf{s}_t$: **身体 proprioception**, 就是 humanoid 自己的 joint positions, velocities, root state 这些, 告诉 policy "我现在身体啥状态"
- $\mathbf{h}_t$: **egocentric heightmap**, 一个 $12 \times 12$ 的 grid, 相邻 0.15m, 围绕角色周围的地形高度。这个让 policy 知道 "我周围有没有椅子、床、桌子, 它们多高", 避免 collision
- $\mathbf{g}_t$: **task goal**, 比如 "走到这个 2D 坐标" 或 "让 pelvis contact 到这个 3D 点" 或 "把这个 box 搬到这个位置"
- $\mathbf{z}$: **language embedding**, 64-dim, 来自 CLIP text encoder 的 downgrade。这个 encoding 风格信息, 告诉 policy "你是 happy 地 walk 还是 tired 地 walk"

Reward 也分两部分:

$$r_t = \lambda^{style} r_t^{style} + \lambda^{task} r_t^{task}$$

- $r_t^{style}$: 来自一个 **text-conditioned motion discriminator** (AMP 的变种), 判断当前 motion 跟 text embedding z 是否 match, 如果 motion 风格跟 caption 一致就给高分
- $r_t^{task}$: 完成 task 的 reward, 比如 "pelvis 到达 target" 给 reward

这里有个 subtle 但关键的 design choice: UniHSI 之类的 tracking-based 方法需要 dense reference motion 作为输入, 也就是说你得有一个 "完美参照动作" 让 policy 一帧帧去 mimic, 结果就是生成的 motion 永远只跟 reference 一样, 没有 diversity (APD ≈ 1.14, 几乎 identical)。SIMS 用 discriminator 而是判 style 是否 match, 具体动作可以 vary, 所以 APD 能到 16.55, diversity 大了 14 倍。

---

### Reward 的具体形式 (Supp. Mat.)

为了让 7 个 skill 好训练, SIMS 把它们归到 3 个 template:

**Loco (Walk, Idle)**:
$$r_t^G = \begin{cases} 0.4 r_t^{near} + 0.5 r_t^{far} + 0, & \|x^* - x_t^{root}\|^2 > 0.5 \\ 0.4 r_t^{near} + 0.5 + 0.1 r_t^{still}, & \text{otherwise} \end{cases}$$

意思就是: 离 target 远的时候, 鼓励靠近 (far reward, 同时约束 velocity 和 facing direction); 到了 target 之后, 鼓励保持静止 (standstill reward)。

$$r_t^{far} = 0.6 \exp(-0.5 \|x^* - x_t^{root}\|^2) + 0.2 \exp(-2.0 \|g_t^{vel} - d_t^* \cdot \dot{x}_t^{root}\|^2) + 0.2 \|d_t^* \cdot d_t^{facing}\|^2$$

- $x^*$: target position
- $x_t^{root}$: 当前 root position
- $g_t^{vel}$: 目标 scalar velocity
- $d_t^*$: 目标 direction
- $\dot{x}_t^{root}$: 当前 root velocity
- $d_t^{facing}$: 当前 facing direction
- 三项分别是: 位置接近、速度 projection match、朝向对齐

$$r_t^{near} = \exp(-10.0 \|x^* - x_t^{root}\|^2)$$

一个 sharp Gaussian, 离 target 越近 reward 越大, 用 10.0 这个大系数让它在近距离时急剧上升, 类似 "last mile incentive"。

$$r_t^{still} = \exp(-2.0 \|\dot{x}_t^{root} - \dot{x}_{t-1}^{root}\|^2)$$

惩罚 root velocity 变化, 到了之后就别动了。

**HSI (Sit, Lie, Reach, GetUp)**:

$$r_t^G = \begin{cases} 0.7 r_t^{near} + 0.3 r_t^{far}, & \|x_t^* - x_t^{root}\|^2 > 0.5 \\ 0.7 r_t^{near} + 0.3, & \text{otherwise} \end{cases}$$

结构跟 Loco 类似, 但权重不同, 因为 HSI 的核心是 **contact**, 所以 near reward (contact 到 object surface) 占 0.7 大头。

- Sit/Lie/GetUp: 约束 pelvis joint
- Reach: 约束 left 或 right hand
- $x_t^*$: object interactable surface 上离 character 最近的 3D point

GetUp 是一个 step goal: 先要 contact (sit 或 lie 上去), contact 达成后切到 "把 pelvis 抬起来到站立" 的 reward。

**DOI (Carry)**:

$$r_t^G = \begin{cases} 0.3 r_t^{walk} + 0.5 r_t^{carry} + 0.2 r_t^{hand}, & \|x_t^{obj} - x_t^{goal}\|^2 > 0.5 \\ 0.3 r_t^{walk} + 0.5 r_t^{carry} + 0.2, & \text{otherwise} \end{cases}$$

这个最有意思, 因为它是三阶段:
- $r_t^{walk}$: 先走到 object 旁边
- $r_t^{hand}$: 手放到 object 上 (准备抓)
- $r_t^{carry}$: 把 object 搬到 target 位置

$$r_t^{walk} = 0.8 \exp(-10.0 \|x_t^{root} - x_t^{obj}\|^2) + 0.2 \exp(-2.0 \|v_t^{root} - v_t^{goal}\|^2)$$

$$r_t^{hand} = \exp(-0.5 \|x_t^{hand} - x_t^{obj}\|^2)$$

$$r_t^{carry} = 0.7 \exp(-10.0 \|x_t^{obj} - x_t^{goal}\|^2) + 0.3 \exp(-2.0 \|v_t^{obj} - v_t^{goal}\|^2)$$

- $x_t^{obj}$: object 当前位置
- $x_t^{goal}$: target 位置
- $v_t^{obj}, v_t^{goal}$: object 速度和目标速度

这个 template 很 general, push、throw 这类 dynamic object interaction 都可以套。

---

### Language Embedding 怎么训

这部分其实是 re-implement 了 MotionCLIP。目的是让 motion 和 text 对齐到一个 64-dim 的 latent space, 然后这个 latent code 喂给 policy 当 style condition。

架构:
- Motion encoder: bidirectional transformer, 输入 motion clip $\hat{\mathbf{m}} = (\hat{\mathbf{q}}_1, ..., \hat{\mathbf{q}}_n)$, 输出 $\mathbf{z} = \text{Enc}_m(\hat{\mathbf{m}})$, normalize 到 unit sphere
- Text encoder: CLIP text encoder (frozen) → $\text{MLP}_d$ (512→64) → 用于 alignment
- $\text{MLP}_u$ (64→512): 用于 reconstruction, 防止 $\text{MLP}_d$ 把 semantic 信息丢了

Loss:

$$\mathcal{L}_{AE} = \mathcal{L}_{recon}^m + \mathcal{L}_{align}^{m,t} + \mathcal{L}_{recon}^t$$

- $\mathcal{L}_{recon}^m = \text{MSE}(\mathbf{m}, \hat{\mathbf{m}})$: motion 重建
- $\mathcal{L}_{align}^{m,t} = 1 - d_{cos}(\text{Enc}_m(\hat{\mathbf{m}}), \text{MLP}_d(\text{Enc}_l(\mathbf{c})))$: motion-text 对齐
- $\mathcal{L}_{recon}^t = \|\text{MLP}_u(\text{MLP}_d(\text{Enc}_l(\mathbf{c}))) - \text{Enc}_l(\mathbf{c})\|_2$: text round-trip 重建

训好之后, 你给一个 text caption "walking happily with bouncy steps", 它的 embedding 跟真实 happy walking 的 motion embedding 就很近。Policy 在跑的时候, 把这个 64-dim embedding 喂进去, discriminator 就能判断 "你这个 motion 是不是真的 happy"。

---

### FSM: 怎么把 7 个 policy 串起来

Finite State Machine 的逻辑很简单: 当 character 的 root 跟当前 target 的 overlap time 超过 threshold, 就触发 next skill。

举个例子, 假设 long script 是:

```
[walk → sofa] → [sit on sofa, sad] → [getup from sofa] → [walk → bed] → [lie on bed, tired]
```

执行流程:
1. FSM 激活 $\pi_w$, target = sofa 位置, text z = "sad walking"
2. Walk policy 把 character 走到 sofa 附近, root overlap 达到 threshold
3. FSM 切换到 $\pi_s$, target = sofa sit surface, text z = "sitting with head down"
4. Sit policy 让 character 坐下, pelvis contact 到 sofa surface
5. FSM 检测到 contact 达成, 切换到 $\pi_g$
6. GetUp policy 让 character 站起来
7. FSM 切换回 $\pi_w$, target = bed 位置
8. ...

跟 InterScene 的 FSM 区别在于: InterScene 只有 task goal, 没有 text embedding 和 heightmap 这两个 condition, 所以它没法控制 style, 也对 scene geometry 的感知弱。

---

### Heightmap 怎么搞的

$12 \times 12$ grid, 相邻 0.15m, 围绕 humanoid root 的 egocentric heightmap。

做法:
1. 对 3DFront 的每个 object, 预生成 point cloud, 用 voxelizing mesh + normal vector segmentation 得到 affordance surface (能 sit/lie 的表面)
2. Runtime 时, 只有当 object 距离 humanoid root 足够近才更新 heightmap, 算 nearest object's pointclouds 的高度
3. Flatten 成 $144 \times 1$ vector, concatenate 到 observation

这个设计的 benefit: policy 不需要看到整个 scene 的全局 geometry, 只需要 local awareness 就能做 sit/lie/carry 这种 local interaction。而且 heightmap 对 unseen object shape 有 generalization, 因为它只采样 surface 高度, 不管 object 是什么类型。

Table 7 证明这点: 只在 3DFront 训练, 在 PartNet 上 test, Success Rate 几乎不掉 (Sit 98.7 vs 96.9, Lie 87.6 vs 89.7)。

---

### 数据集这一块

SIMS 混了 6 个 motion dataset:

| Dataset | 主要贡献 |
|---------|---------|
| SAMP | Sit, Lie, GetUp 的 stylized motion |
| COUCH | 更多 stylized Sit |
| Circles | Reach (touch) |
| 100Style | Walk 的 100 种 style |
| AMASS | Neutral Walk + 少量 Carry |
| **ViconStyle (新)** | Stylized Idle, Lie, Carry, GetUp, 71.6 min, 415 clips, 3 actors |

ViconStyle 是他们自己用 Vicon 光学 mocap 采的, 120fps, 用 SOMA fit SMPL, 标了 caption 和 style。Scene object 用 axis regression + bounding box 重建。

关键问题: 之前没有 stylized carry motion 的 dataset, ViconStyle 填了这个空白。Table 9 显示加了 ViconStyle 后, Carry 的 Success Rate 从 92.9 跳到 96.4, APD 从 14.36 到 14.92。

---

### 实验结果讲人话

**物理 task performance** (Table 3): SIMS 在 4096 个 random text condition 下测, Sit 98.1%, Lie 87.6%, Reach 95.2%, Carry 92.9%。UniHSI 这些没 style 的方法相当于 SIMS 的一个 specific case (no style condition), 所以 SIMS 比 SOTA 还好或持平。

**Motion diversity** (Table 4): 这是最 striking 的。UniHSI 的 APD 是 1.14, SIMS 是 16.55, 差 14 倍。说明 UniHSI 生成的 motion 几乎一模一样 (因为 dense tracking reference), 而 SIMS 因为用 discriminator 做 style prior, 可以 sample 出很多不同的 motion。

**User study** (Table 5): 30 个人打分 1-5, SIMS 在 Physical Realism (3.4 vs 2.6)、Motion Diversity (3.6 vs 2.9)、Plot Engagement (3.0 vs 2.4)、Emotional Resonance (3.8 vs 3.0) 全面胜出。Emotional Resonance 提升最大, 这是 RASG + text-conditioned policy 共同的功劳。

**Ablation** (Table 11): 去掉 text embedding, Carry 的 APD 从 14.92 掉到 12.41 (回到 UniHSI 水平), 证明 text condition 是 style diversity 的核心来源。去掉 heightmap, Sit 的 Success Rate 从 96.9 掉到 88.7, Lie 从 89.7 掉到 79.8, 证明 scene awareness 对 interaction task 至关重要。

---

### 这个工作的本质 insight

我用一句话总结: **physical plausibility 来自 physics simulator, style diversity 来自 text-conditioned discriminator, narrative coherence 来自 RAG, scalability 来自 template-based reward + FSM composition**。

四个东西分别解决四个问题, 然后 hierarchical framework 把它们拼起来。这个 decoupling 让每个 component 可以独立扩展:

- 想加新 skill? 按 Loco/HSI/DOI template 训一个新 policy, 加到 FSM 里
- 想加新 style? 录新 motion, 训 discriminator, 不用动 policy 结构
- 想加新 story pattern? 往 short script database 加新的乐高积木, RAG 自动会用

这种 decoupling 是整个 framework 的核心价值, 比 single monolithic model 好维护也好扩展。

---

### 局限性

我自己看下来的几个问题:

1. **Heightmap 太 local**: $12 \times 12$ grid 只覆盖 1.8m × 1.8m 区域, 长距离 navigation 还是靠 walk policy 自己探索, 没有 global path planning, 复杂场景可能卡住
2. **Walk 作为 universal transition 有点 crude**: 真人从 sit 到 walk 到 lie 中间的 transition motion 其实很复杂, 现在统一用 walk 可能会有 "uncanny valley" 的感觉
3. **Reward weight 是 hand-tuned**: Loco reward 里 0.4/0.5/0.1, HSI 里 0.7/0.3, DOI 里 0.3/0.5/0.2 这些系数都是 manual 调的, 换 skill 可能要重新调
4. **9 个 style 类别太少**: 真人 emotion 远比 happy/sad/angry 复杂, 而且 style 和 content 在 CLIP space 可能没完全 disentangle
5. **没有 articulated hand**: 现在 hand 是简化 model, 做 carry 还行, 做 "拿起茶杯喝一口" 这种 fine-grained 交互就不够了

---

### 我的 takeaway

如果你要从这篇 paper 学东西, 我觉得最有 transferable 的两个 idea 是:

1. **RAG for behavior generation**: 把 LLM 的 hallucination 问题用 retrieval 解决, 这个 pattern 在 robotics 和 embodied AI 里也能用。你不用让 LLM 从零生成 action sequence, 而是让它从 pre-built skill library 里检索组合。这比 pure LLM planning 稳定得多。

2. **Discriminator vs. tracker**: UniHSI 用 tracker (dense reference), 牺牲了 diversity; SIMS 用 discriminator (style prior), 保留了 diversity。这个 trade-off 在 imitation learning 领域是个经典话题, AMP paper 最早提出来, SIMS 把它用到 multi-condition HSI 上。如果你做 robotics manipulation, 这个 idea 也能用: 你不需要 mimic expert demo 一帧帧, 只要判 "你这个 motion 是不是 expert style" 就行。

Paper 的 limitation 主要在 data scale 和 reward engineering, 但 framework design 本身是 elegant 的, 可以扩展到更复杂的 embodied AI 任务。

---

# SIMS Paper 详解

Andrej, 这是一篇关于 physics-based stylized human-scene interaction 的工作。我先给你 build up 整体 intuition, 然后逐层拆解。

## 1. 整体架构 Intuition

SIMS 的核心 motivation 来自一个观察: 现有 HSI 工作分成两大派, 但都有缺陷:

- **Kinematics-based 方法** (NSM, SAMP, Humanise, AffordMotion, TesMo): 依赖 data-driven kinematic models, 容易出现 penetration, floating, sliding, 需要 post-processing, 无法实时
- **Physics-based 方法** (InterPhys, InterScene, UniHSI): 引入 physics simulator (Isaac Gym) 保证 physical plausibility, 但 skill 数量少, planning 简单 (chronological list 或 pure contact chain), 缺乏 style diversity

SIMS 的核心 idea 是做一个 **hierarchical framework**, 把 high-level script-driven intent 和 low-level control policy 串起来, 同时拿到 style diversity 和 physical plausibility。

整体 pipeline 分三层:

```
Layer 1: Short Script Database (offline)
  └─ LLM 生成 short scripts (keyframes + summary + style label)
  └─ CLIP encoding 用于 retrieval keys

Layer 2: Retrieval-Augmented Script Generation (RASG) (inference time)
  └─ user theme → CLIP feature → retrieve top-k short scripts
  └─ LLM 组合 short scripts 成 long-term story

Layer 3: Multi-Condition Controller + FSM (runtime)
  └─ FSM 解析 keyframe → task goal + language embedding + heightmap
  └─ 7 个 task-specific policies 执行 skills
```

参考链接:
- Project page: https://wenjiawang0312.github.io/projects/sims/
- RAG 原始 paper: https://arxiv.org/abs/2005.11401
- AMP (Adversarial Motion Priors): https://arxiv.org/abs/2104.02180
- UniHSI: https://arxiv.org/abs/2403.07905
- InterScene: https://arxiv.org/abs/2403.12028

---

## 2. Short Script Database Construction

这是整个系统的"原子单元"。每个 short script 定义为:

$$p = [\{f_0, f_1, ..., f_N\}, u, d]$$

变量说明:
- $p$: 一个 short script
- $\{f_0, f_1, ..., f_N\}$: keyframe 序列
- $u$: 一句话 summary, 封装 core style/emotion 和 interaction events
- $d$: 从 keyframe style labels 中提取的 distinctive style/emotion keyword

每个 keyframe:

$$f = (s, o, c, e)$$

- $s$: 要执行的 skill (walk, sit, lie, reach, getup, idle, carry)
- $o$: target object (sofa, bed, table, shelf, ...)
- $c$: caption 描述 motion attributes (e.g. "leaning back, legs straight, hands supporting head")
- $e$: emotion/style 标签 (happy, angry, hurried, tired, sad, stressed, drunk, relaxed, neutral)

**关键 insight**: 受 filmmaking 启发, short script 只用几个 keyframe 表达一个短交互段。把不同 emotion 分成 8 类 + neutral, 共 9 类, 便于 modular retrieval。

LLM (GPT-4) 被提示生成 wide range 的 short scripts, 输入包括 available skills, text captions, styles, objects。然后用 CLIP text encoder 提取 summary embeddings 作为 retrieval keys。

Table 13 里给了若干例子, 比如 "The character rushed anxiously through the living room" 这个 short script 包含:
- loco anxious → "rush anxiously forward"
- touch shelf
- idle anxious → "pace around nervously"
- loco hurried → "walk with large steps"

---

## 3. Retrieval-Augmented Script Generation (RASG)

直接用 LLM 生成长 script 的问题: redundancy, lack of diversity, 缺乏 coherent narrative guidance。UniHSI 就只生成有限 keyframes with minimal diversity。

RASG 的三个步骤:

### Step 1: Style narrowing
LLM 从 user theme 中识别 M 个最相关 styles, 缩小 retrieval scope。

### Step 2: Semantic Similarity Retrieval
- user theme sentence → CLIP feature (query)
- 计算 cosine distance between query 和 database keys
- 对每个 style retrieve top-k short scripts
- 总共 $M \times k$ 个 summaries 进入下一步

### Step 3: Summary Filtering + Long Script Creation
LLM 根据 scene layout, 选择并组合 summaries 成 coherent narrative, 通过 logically concatenating keyframes。

**关键设计**: 为了保证 executable permutation, skills 被组织成 tuples:
- (sit, getup)
- (lie, getup)
- (idle)
- (walk, carry)
- (walk, reach)
- ...

Walk skill 作为 transition motion, 可以在任何 tuple 之间插入, 实现 seamless connections。

### Ablation 实验 (Table 6)
对比 Direct LLM generation vs. RASG:

| Method | SBERT Similarity ↓ | Avg Generation Time (s) ↓ |
|--------|-------------------|--------------------------|
| LLM    | 0.8167            | 12.2                     |
| RASG   | 0.7759            | 7.32                     |

SBERT cosine similarity 越低, 说明 generated stories 之间越 diverse。RASG 不仅更 diverse, 还快了一倍 (因为 LLM 只需要 retrieve 4-5 个 short scripts ≈ 20 keyframes, 不需要从头生成)。

参考 SBERT: https://arxiv.org/abs/1908.10084

---

## 4. Multi-Condition Controller

### 4.1 MDP formulation

在每个 time step $t$, policy 接收:

$$\pi(\mathbf{a}_t | \mathbf{s}_t, \mathbf{h}_t, \mathbf{g}_t, \mathbf{z})$$

变量:
- $\mathbf{s}_t \in \mathcal{S}$: humanoid proprioception (joint positions, velocities, root state, ...)
- $\mathbf{h}_t \in \mathcal{H}$: egocentric heightmap, $12 \times 12$ grid, 相邻距离 0.15m, flatten 后 concatenate 到 observation
- $\mathbf{g}_t \in \mathcal{G}$: task-specific goal state
- $\mathbf{z} \in \mathcal{Z}$: language embedding (CLIP-based, 64-dim after downsizing)
- $\mathbf{a}_t \in \mathcal{A}$: action (target joint angles for PD controller)

优化目标 (expected discounted return):

$$J(\pi) = \mathbb{E}_{p(\tau|\pi)} \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right]$$

- $T$: horizon length
- $\gamma \in [0, 1]$: discount factor

### 4.2 Reward composition

总 reward:

$$r_t = \lambda^{style} r_t^{style} + \lambda^{task} r_t^{task}$$

- $r_t^{style}$: 来自 text-conditioned motion discriminator (AMP-style), 编码 stylized motion prior
- $r_t^{task}$: task-specific reward
- $\lambda^{style}, \lambda^{task}$: 加权系数

### 4.3 Three task templates

为了减少 7 个 skills 的开发开销, 把所有 interaction task 归到 3 个 templates:

**Loco (Walk, Idle)**:
- 目标: pelvis 到达 target 2D location $\mathbf{g} \in \mathbb{R}^2$
- Walk: $\geq 1m$ from initial position
- Idle: identical to current position (pacing in place)

**HSI (Sit, Lie, Reach, GetUp)**:
- 目标: 特定 body joint contact object surface
- Sit/Lie/GetUp: pelvis joint 约束
- Reach: left 或 right hand
- Target $\mathbf{g} \in \mathbb{R}^3$ 来自 object interactable surface 上最近的 3D point

**DOI (Carry)**:
- 不约束 body joints, 而是 object root 到达 target 3D location
- $\mathbf{g}^{bbox} \in \mathbb{R}^{3 \times 8}$: object bounding box 8 个顶点坐标
- $\mathbf{g}^{tar} \in \mathbb{R}^3$: target location
- $\mathbf{g} = \{\mathbf{g}^{bbox}, \mathbf{g}^{tar}\}$

### 4.4 详细的 Reward 公式 (Supp. Mat.)

#### Loco Reward (Eq. 1-4)

$$r_t^G = \begin{cases} 0.4 \cdot r_t^{near} + 0.5 \cdot r_t^{far} + 0, & \|x^* - x_t^{root}\|^2 > 0.5 \\ 0.4 \cdot r_t^{near} + 0.5 + 0.1 \cdot r_t^{still}, & \text{otherwise} \end{cases}$$

- $x^*$: target root position
- $x_t^{root}$: character root position at time $t$
- 第一项: far reward (尚未到达)
- 第二项: 已经到达, 加 standstill reward

$$r_t^{far} = 0.6 \exp(-0.5 \|x^* - x_t^{root}\|^2) + 0.2 \exp(-2.0 \|g_t^{vel} - d_t^* \cdot \dot{x}_t^{root}\|^2) + 0.2 \|d_t^* \cdot d_t^{facing}\|^2$$

- $g_t^{vel}$: target scalar velocity
- $d_t^*$: target direction
- $\dot{x}_t^{root}$: current root velocity
- $d_t^{facing}$: current facing direction
- 三个 term 分别权重: position, velocity projection, facing alignment

$$r_t^{near} = \exp(-10.0 \|x^* - x_t^{root}\|^2)$$

sharp Gaussian, 在近距离时 reward 急剧上升。

$$r_t^{still} = \exp(-2.0 \|\dot{x}_t^{root} - \dot{x}_{t-1}^{root}\|^2)$$

惩罚 root velocity 变化, 鼓励静止。

Walk 和 Idle 的主要区别: Idle 允许更大 distance threshold (3m), Walk 要尽可能贴近 target。

#### HSI Reward (Eq. 5-7)

$$r_t^G = \begin{cases} 0.7 \cdot r_t^{near} + 0.3 \cdot r_t^{far}, & \|x_t^* - x_t^{root}\|^2 > 0.5 \\ 0.7 \cdot r_t^{near} + 0.3, & \text{otherwise} \end{cases}$$

$$r_t^{far} = \exp(-2.0 \|g_t^{vel} - d_t^* \cdot \dot{x}_t^{root}\|^2)$$

$$r_t^{near} = \exp(-10.0 \|x_t^* - x_t^{root}\|^2)$$

GetUp 是 step goal: 先 contact (sit/lie) reward, 达到 contact 后切到 elevate pelvis to standing 的 reward。

#### DOI Reward (Eq. 8-11)

$$r_t^G = \begin{cases} 0.3 \cdot r_t^{walk} + 0.5 \cdot r_t^{carry} + 0.2 \cdot r_t^{hand}, & \|x_t^{obj} - x_t^{goal}\|^2 > 0.5 \\ 0.3 \cdot r_t^{walk} + 0.5 \cdot r_t^{carry} + 0.2, & \text{otherwise} \end{cases}$$

- $x_t^{obj}$: object root position
- $x_t^{goal}$: target goal position
- 3 个 phase: walk-to-object, hand-contact, carry-to-goal

$$r_t^{walk} = 0.8 \exp(-10.0 \|x_t^{root} - x_t^{obj}\|^2) + 0.2 \exp(-2.0 \|v_t^{root} - v_t^{goal}\|^2)$$

$$r_t^{hand} = \exp(-0.5 \|x_t^{hand} - x_t^{obj}\|^2)$$

$$r_t^{carry} = 0.7 \exp(-10.0 \|x_t^{obj} - x_t^{goal}\|^2) + 0.3 \exp(-2.0 \|v_t^{obj} - v_t^{goal}\|^2)$$

这个 template 可以扩展到 push, throw 等其他 DOI 任务。

### 4.5 FSM (Finite State Machine)

7 个 policies: $\pi_w$ (Walk), $\pi_i$ (Idle), $\pi_s$ (Sit), $\pi_l$ (Lie), $\pi_r$ (Reach), $\pi_g$ (GetUp), $\pi_c$ (Carry)

FSM 用 simple rule: 当 character root 和 target position 的 overlap time 超过 threshold, 触发 next skill。

与 InterScene 的区别: 这里的 FSM 包含 **per-frame egocentric heightmap** 和 **per-skill text embedding**, 保证 scene understanding 和 semantic control。

### 4.6 Language Conditioning (MotionCLIP re-implementation)

构建一个 motion-text aligned embedding space。

**Motion encoder**: bidirectional transformer, 把 motion clip $\hat{\mathbf{m}} = (\hat{\mathbf{q}}_1, ..., \hat{\mathbf{q}}_n)$ 映射到 embedding $\mathbf{z} = \text{Enc}_m(\hat{\mathbf{m}})$, normalize 到 unit sphere $\|\mathbf{z}\| = 1$, dim = 64。

**Text encoder**: 
1. CLIP text encoder $\text{Enc}_l$ 提取 512-dim feature
2. $\text{MLP}_d$ downsize 到 64-dim
3. $\text{MLP}_u$ upsample 回 512-dim (用于 reconstruction loss)
4. CLIP weights 冻结

**Training loss** (Eq. 12-14):

$$\mathcal{L}_{AE} = \mathcal{L}_{recon}^m + \mathcal{L}_{align}^{m,t} + \mathcal{L}_{recon}^t$$

$$\mathcal{L}_{recon}^m = \text{MSE}(\mathbf{m}, \hat{\mathbf{m}})$$

$$\mathcal{L}_{align}^{m,t} = 1 - d_{cos}(\text{Enc}_m(\hat{\mathbf{m}}), \text{MLP}_d(\text{Enc}_l(\mathbf{c})))$$

$$\mathcal{L}_{recon}^t = \|\text{MLP}_u(\text{MLP}_d(\text{Enc}_l(\mathbf{c}))) - \text{Enc}_l(\mathbf{c})\|_2$$

- $\mathcal{L}_{recon}^m$: motion reconstruction
- $\mathcal{L}_{align}^{m,t}$: motion-text alignment (cosine distance)
- $\mathcal{L}_{recon}^t$: text embedding round-trip reconstruction (防止 $\text{MLP}_d$ 丢失 semantic info)

**Sampling**: 30fps motion, sample 300 frames, >10s 的 clip 用 skip sampling 保证所有信息被包含。

### 4.7 Scene Conditioning (Heightmap)

$12 \times 12$ grid, 相邻距离 0.15m 的 egocentric heightmap。

与 UniHSI 类似, 预生成 scene point clouds。为了保留 surface intricacies (用于 sit/lie), 在 bounding box 范围内 voxelizing object meshes, 并基于 normal vectors 分割 point clouds 得到 affordance surface。

Heightmap 只在 object 距离 humanoid root 足够近时更新 (计算 nearest object's pointclouds), 节省计算。

---

## 5. Datasets

### 5.1 Stylized motion mixture (Table 2)

6 个 motion dataset:

| Dataset | Walk | Idle | Sit | Lie | GetUp | Reach | Carry |
|---------|------|------|-----|-----|-------|-------|-------|
| SAMP [12] | 20.6 | - | 35.2 | 14.8 | 11.2 | - | - |
| COUCH [54] | - | - | 36.4 | - | 23.4 | 1 | - |
| Circles [2] | - | - | - | - | - | 3.6 | - |
| 100Style [24] | 203.1 | - | - | - | 1 | - | - |
| AMASS [22] | 8.2 | - | - | - | - | - | 3.4 |
| ViconStyle | - | 12.0 | - | 21.9 | 11.7 | - | 26.0 |

(单位: minutes, 黑框数字表示 neutral style)

### 5.2 ViconStyle Dataset

新采集的数据集, 用 Vicon optical mocap, 120fps, 3 actors (2 male + 1 female, age 22-30), 71.6 minutes, 415 clips。

Pipeline:
1. 用 SOMA [10] fit SMPL body model
2. 标注 text descriptions ("hands on thighs", "lean back")
3. 标注 style/emotion labels
4. Scene object reconstruction:
   - Stage 1: 旋转 axis 回归最小化 marker-to-axis max distance, 得到 local coordinate, 算 scale
   - Stage 2: 后续 transformation 用 displacement + rotation 相对 initial frame

参考 SMPL: https://smpl.is.tue.mpg.de/
参考 SOMA: https://arxiv.org/abs/2107.04924

### 5.3 Style categories

8 + neutral = 9 类: happy, angry, hurried, tired, sad, stressed, drunk, relaxed, neutral。

每个 caption 配 5 个 synonymous sentences (LLM 生成), 左右翻转 augment, captions 按 body joint symmetry 翻转。

### 5.4 3D Scenes

3DFront [7] 用于 furniture 和 scene layouts, 没有 segmentation, 所以 voxelize meshes + 基于 normal vectors 分割 point clouds 得到 affordance surface。

参考 3DFront: https://arxiv.org/abs/2011.09127

---

## 6. Experiments

### 6.1 与 SOTA Physics-based Methods 对比 (Table 3)

| Method | Sit | Lie | Reach | Carry | Sit CE | Lie CE | Reach CE | Carry CE |
|--------|-----|-----|-------|-------|--------|-------|---------|----------|
| InterPhys [13] | 93.7 | 80.0 | - | 94.3 | 0.09 | 0.30 | - | 0.08 |
| InterScene [26] | 97.8 | - | - | - | 0.04 | - | - | - |
| UniHSI [47] | 94.3 | 81.5 | 97.5 | - | 0.032 | 0.061 | 0.016 | - |
| **SIMS** | 98.1 | 87.6 | 95.2 | 92.9 | 0.028 | 0.049 | 0.026 | 0.099 |
| **SIMS (+data)** | 98.4 | 89.6 | - | 96.4 | 0.033 | 0.048 | - | 0.085 |

- CE = Contact Error
- SIMS 在 4096 个随机 text conditions 下测试, 之前的方法相当于 SIMS 的一个 specific situation (没 style)
- 仅在 Reach 和 Carry 略低于 SOTA, 因为 InterPhys 没有开源, 用 AMASS 少量 carry motion 训练

### 6.2 Motion Diversity (Table 4)

| Method | FID Sit ↓ | FID Lie ↓ | FID Carry ↓ | APD Sit ↑ | APD Lie ↑ | APD Carry ↑ |
|--------|-----------|-----------|-------------|-----------|-----------|--------------|
| InterPhys* [13] | 153.84 | 211.22 | 81.0 | 1.14±0.01 | 1.35±0.02 | 12.41±0.19 |
| UniHSI [47] | - | - | - | 1.14 | 1.35 | 12.41 |
| **SIMS** | 125.66 | 171.24 | 65.14 | 16.55±0.54 | 16.40±0.94 | 14.36±0.12 |

**关键观察**: UniHSI 的 APD ≈ 1.14, 而 SIMS 的 APD ≈ 16.55, 差了 14 倍! UniHSI 生成的 motions 几乎 identical, 因为它依赖准确 reference motions 做 dense tracking, 失去了 diversity。SIMS 用 text-conditioned discriminator, 可以在 style space 里 sample 出 diverse motions。

FID 也大幅降低, 说明 motion distribution 更接近 reference。

### 6.3 User Study (Table 5)

30 participants, 1-5 scale:

| Metric | UniHSI | SIMS |
|--------|--------|------|
| Physical Realism ↑ | 2.6 | 3.4 |
| Motion Diversity ↑ | 2.9 | 3.6 |
| Plot Engagement ↑ | 2.4 | 3.0 |
| Emotional Resonance ↑ | 3.0 | 3.8 |

SIMS 全面胜出, 尤其在 Emotional Resonance (3.8 vs 3.0), 这正是 RASG + text-conditioned policy 带来的。

### 6.4 Generalization to Unseen Objects (Table 7)

| Dataset | Sit SR ↑ | Lie SR ↑ | Sit CE ↓ | Lie CE ↓ |
|---------|----------|----------|----------|----------|
| PartNet [25] | 98.7 | 87.6 | 0.028 | 0.065 |
| 3DFront [7] | 96.9 | 89.7 | 0.014 | 0.030 |

Policy 只在 3DFront 上训练, 在 unseen PartNet 上仍然表现良好, 证明 heightmap design 的 generalization 能力。

### 6.5 Dataset Ablation (Tables 8-10)

**Walk** (Table 8):

| Datasets | SR ↑ | APD ↑ |
|----------|------|-------|
| 100Style | 92.6 | 14.83±0.35 |
| AMASS + 100Style | 95.1 | 14.88±0.29 |

AMASS 提供 stable neutral walking, SR 上升; APD 变化小, 因为 100Style 也是 neutral style。

**Carry** (Table 9):

| Datasets | SR ↑ | APD ↑ |
|----------|------|-------|
| AMASS | 92.9 | 14.36±0.12 |
| AMASS + ViconStyle | 96.4 | 14.92±0.23 |

ViconStyle 是第一个包含 stylized carrying motion 的 dataset, 两个 metrics 都大幅上升。

**HSI (Sit/Lie)** (Table 10):

| Datasets | Sit SR ↑ | Lie SR ↑ | Sit APD ↑ | Lie APD ↑ |
|----------|----------|----------|-----------|-----------|
| SAMP | 95.5 | 86.9 | 16.43±0.90 | 16.40±0.94 |
| SAMP+COUCH | 96.9 | - | 16.52±0.47 | - |
| SAMP+COUCH+ViconStyle | - | 89.7 | - | 16.84±1.28 |

COUCH 提供 stylized sitting, ViconStyle 提供 stylized lying, 对应的 skill 都有提升。

### 6.6 Policy Setting Ablation (Table 11)

| Setting | Sit SR ↑ | Lie SR ↑ | Carry SR ↑ | Sit APD ↑ | Lie APD ↑ | Carry APD ↑ |
|---------|----------|----------|------------|-----------|-----------|--------------|
| w/o text | 89.7 | 89.6 | 92.4 | 16.29±0.22 | 16.59±0.28 | 12.41±0.19 |
| w/o htmp | 88.7 | 79.8 | - | 16.18±0.19 | 16.94±0.29 | - |
| **SIMS (full)** | 96.9 | 89.7 | 96.4 | 16.52±0.47 | 16.99±1.28 | 14.92±0.23 |

两个 ablation 都有 degraded performance:
- w/o heightmap: SR 明显下降, 因为缺少 environment awareness
- w/o text: APD 下降, 失去 style diversity (Carry APD 从 14.92 降到 12.41, 接近 UniHSI 的 12.41!)

---

## 7. 与 Related Works 的 Positioning

Table 1 给了完整 comparison matrix:

| Method | Phys-Plausible | Auto Planner | Style Diversity | Text-Aware | Scene-Aware Controller | Skill-Scalability | Walk | Sit | Lie | GetUp | Reach | Idle | Carry |
|--------|----------------|--------------|-----------------|------------|-------------------------|-------------------|------|-----|-----|-------|-------|------|-------|
| NSM [33] | ✗ | ✗ | ✗ | ✗ | √ | ✗ | √ | √ | ✗ | √ | ✗ | √ | √ |
| SAMP [12] | ✗ | ✗ | ✗ | ✗ | √ | ✗ | √ | √ | ✗ | - | ✗ | ✗ | ✗ |
| Humanise [44] | ✗ | ✗ | ✗ | √ | √ | ✗ | √ | √ | √ | ✗ | ✗ | ✗ | ✗ |
| AffordMotion [45] | ✗ | ✗ | ✗ | √ | √ | ✗ | √ | √ | √ | ✗ | ✗ | ✗ | ✗ |
| TesMo [50] | ✗ | ✗ | ✗ | √ | √ | ✗ | √ | √ | √ | ✗ | ✗ | ✗ | ✗ |
| InterScene [26] | √ | √ | ✗ | ✗ | √ | √ | √ | √ | √ | √ | ✗ | ✗ | ✗ |
| UniHSI [47] | √ | √ | ✗ | ✗ | √ | ✗ | √ | √ | √ | √ | √ | ✗ | ✗ |
| **SIMS (ours)** | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |

SIMS 是唯一一个 all-check 的方法, 而且覆盖 7 个 skills 全部。

---

## 8. Scalability (New Skills)

Supp. Mat. Sec. 9 (Fig. 6) 展示了 framework 的 easy scalability。添加新 skill + new style 的步骤:
1. 基于 3 个 task templates (Loco, HSI, DOI) 之一训练新 skill policy
2. 按照 Sec. 3.1 扩展 short script database
3. 更新 FSM

不需要 retrain 整个 controller, 这是相比之前 methods 的重要优势。

---

## 9. Init/Reset/Termination 细节

**Initialization**:
- HSI skills: reference state initialization [27] + default pose initialization with random global rotation and location nearby object
- Loco skills: 随机 sample 在 whole ground plane, 计算与 objects 的 collision
- DOI skills: 随机 sample target position on ground plane, 从 reference object motion 初始化 object in humanoid's hands
- 所有 skills 训练时都加入 Walk motion data 到 init reference state, 因为 Walk 用作 transition

**Termination**:
- 固定 episode length
- Fall detection
- Task accomplished for certain time
- Contact forces 极大

参考 DeepMimic: https://arxiv.org/abs/1804.02717

---

## 10. Limitations 和 Future Work

作者指出的方向:
1. **更多 stylized motion data**: 当前 dataset 仍不够丰富
2. **Articulated fingers**: 现在用简化 hand model
3. **Multi-agent HSI**: 当前只支持 single character

我 (作为分析者) 补充的几个潜在问题:
1. **Heightmap 的局部性**: $12 \times 12$ grid 只能 capture 局部 geometry, 长距离 navigation 可能需要 global planning
2. **Text embedding 的 entanglement**: CLIP text feature 可能没有完全 disentangle style 和 content
3. **FSM 的 hard switching**: skill 之间的 transition 通过 Walk 作为 bridge, 但在复杂场景下可能不够 smooth
4. **Reward engineering**: 三类 reward 模板有很多 hand-tuned weights (0.3, 0.5, 0.7 等), 可能需要 automatic reward shaping

---

## 11. 关键参考链接汇总

- **Project page**: https://wenjiawang0312.github.io/projects/sims/
- **RAG**: https://arxiv.org/abs/2005.11401
- **CLIP**: https://arxiv.org/abs/2103.00020
- **AMP**: https://arxiv.org/abs/2104.02180
- **ASE**: https://arxiv.org/abs/2205.01906
- **UniHSI**: https://arxiv.org/abs/2403.07905
- **InterScene**: https://arxiv.org/abs/2403.12028
- **InterPhys**: https://research.nvidia.com/labs/toronto-ai/synthesizing_physical_character_scene_interactions/
- **DeepMimic**: https://arxiv.org/abs/1804.02717
- **HumanML3D**: https://github.com/EricGuo5513/HumanML3D
- **SAMP**: https://research.nvidia.com/labs/toronto-ai/samp/
- **NSM**: https://arxiv.org/abs/1909.04228
- **COUCH**: https://virtualhumans.mpi-inf.mpg.de/couch/
- **3DFront**: https://arxiv.org/abs/2011.09127
- **PartNet**: https://arxiv.org/abs/1812.02746
- **AMASS**: https://amass.is.tue.mpg.de/
- **SMPL**: https://smpl.is.tue.mpg.de/
- **SOMA**: https://arxiv.org/abs/2107.04924
- **SBERT**: https://arxiv.org/abs/1908.10084
- **Isaac Gym**: https://arxiv.org/abs/2108.10470
- **100Style**: https://arxiv.org/abs/2203.08531
- **MotionCLIP**: https://arxiv.org/abs/2207.01596

---

## 12. Build Intuition 总结

SIMS 的核心 insight 可以浓缩成一句话: **把 LLM 当 high-level narrative director, 把 physics policy 当 low-level actor, 用 RAG 解决 narrative diversity, 用 text-conditioned discriminator 解决 style diversity, 用 FSM 解决 skill composition**。

整个系统的 elegant 之处在于:
1. **Decoupling**: planner 和 controller 解耦, 各自可扩展
2. **Retrieval-augmented**: 利用 pre-built short script database 作为"乐高积木", LLM 只负责组装, 避免从零生成的 hallucination 和 redundancy
3. **Template-based reward**: 3 个 reward template 覆盖 7 个 skills, 减少 engineering overhead
4. **Multi-condition policy**: 一个 policy 同时吃 scene geometry (heightmap) + task goal + text embedding (style), 实现真正 multi-modal control

这套 framework 的 limitation 主要在 data scarcity 和 reward engineering, 但 hierarchical + RAG 的设计 pattern 完全可以扩展到 robotics manipulation, embodied AI 等领域。
