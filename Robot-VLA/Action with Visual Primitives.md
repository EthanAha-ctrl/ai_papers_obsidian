---
source_pdf: Action with Visual Primitives.pdf
paper_sha256: c33dd922ac0636a1f83b9bd5b5d2e3e63c857c46300b54f7377b8c22d22efbf6
processed_at: '2026-08-17T23:48:59-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# AVP 用人话讲

好,我把学术腔调收一收,用大白话把这篇 paper 拆给你听。

---

## 1. 一句话说清楚 AVP 在干啥

现在的 VLA 模型 (π0、π0.5、OpenVLA 这些) 都是 **一个大网络,从 image + language 直接怼到 action**。AVP 说:这不对,中间应该有个"视觉提示"作为接口 — **让 VLM 先在图上"点一下"下一步要去哪儿,action expert 再照着这个点去执行**。

就这么简单。

---

## 2. 现有 VLA 到底哪里不行

打个比方。你给机器人一句指令:"把红车放到棋盘第 3 行第 5 列"。

现在的 VLA 网络长这样:

```
image + "放到第3行第5列" → [一个黑盒 mega-network] → 14维 action
```

问题在哪? 这个 mega-network 必须 *同时* 干三件事:
1. **理解** "第3行第5列"是啥意思 (semantic reasoning)
2. **找到** 这个位置在 image 像素坐标里对应哪儿 (spatial grounding)
3. **算出** 机械臂关节该怎么动才能到那儿 (motor control)

三件事揉在一个 loss 里。结果就是 action expert 必须从 raw VLM features 里 **重新学一遍** spatial reasoning — 但 VLM 本来就会 spatial reasoning 啊,你 pretrained 时已经花了几亿 image 学过了,现在又让 action expert 从头学,这不是白搞吗?

π0.5 [1] 想了个办法:把指令拆成 subtask description (语言层面的 plan),让 action expert 只学 atomic skill。但 **语言表达不了精细空间差异**。你跟机器人说"放在第 3 行第 5 列",跟"放在第 3 行第 6 列",语言上几乎没差别,但像素层面差了 50 个 pixel。语言这个接口太粗糙。

π0.7 [2] 和 world model 路线 [3] 干脆预测未来帧 (future image) 当 target。但未来帧信息冗余 — 你给 action expert 一张完整 future image,它还得自己从中 "挖"出哪部分才是 task-relevant。等于把难题往后推了一步。

Point-VLA [4]、TraceVLA [5] 这些 cascaded 方法走第三条路:调外部模型 (SAM [6]、Grounding DINO [7],或者直接调 Claude / Gemini API) 在 input image 上画个 box 或点。但这是 **流水线** — SAM 错了下游全错,而且调 API 慢得要命 (paper 里实测 Point-VLA 一步 37 秒,机器人早卡死了)。

AVP 的洞察: **VLM 和 action expert 之间缺一个 explicit 的、spatially grounded 的通信协议**。这个协议应该满足三条:
- Dense enough to be actionable (能指导 action)
- Sparse enough to be learnable (能被监督学习)
- End-to-end differentiable (能 co-train,不依赖外部模块)

**Visual primitive** (image plane 上的 point / box / mask) 就是这个协议。

参考:
- π0.5: https://arxiv.org/abs/2504.16054
- π0.7: https://arxiv.org/abs/2604.15483
- Point-VLA: https://arxiv.org/abs/2512.18933
- TraceVLA: https://arxiv.org/abs/2412.10345
- SAM 2: https://arxiv.org/abs/2408.00714

---

## 3. AVP 怎么搭的

三件套,全 end-to-end:

```
┌─────────────────────────────────────────────────────────┐
│  image + instruction                                     │
│         ↓                                               │
│  [VLM backbone]  ← PaliGemma-style, pretrained          │
│         ↓                                               │
│  multimodal context tokens                              │
│         ↓                                               │
│  [Visual-Primitive Decoder] ← autoregressive            │
│         ↓                                               │
│  p_t: discretized visual primitive (point/box tokens)   │
│         ↓                                               │
│  [Projection] ← 把 primitive 投影到 visual token space   │
│         ↓                                               │
│  z_t^vp (visual-primitive tokens)                      │
│         ↓                                               │
│  [fuse with original tokens] → z_t^aug                  │
│         ↓                                               │
│  [Flow-Matching Action Expert]                          │
│         ↓                                               │
│  a_{t:t+h}: 14维 action chunk (双臂)                    │
└─────────────────────────────────────────────────────────┘
```

公式版:

$$p_t = D_\psi(\text{VLM}(o_t, l))$$

$$z_t^{vp} = \text{Proj}(p_t, o_t)$$

$$a_{t:t+h} = \pi_\theta(z_t^{aug}, s_t)$$

人话翻译:
- $o_t$ = 当前看到的画面 (多视角 RGB)
- $l$ = 语言指令
- $p_t$ = VLM "画"出来的视觉 primitive (一组离散 token,可以解码成图上的点/框)
- $z_t^{vp}$ = primitive 投影到 visual embedding space 后的 tokens (上标 $vp$ = visual primitive)
- $z_t^{aug}$ = primitive tokens + 原始 tokens 融合后的 augmented representation
- $s_t$ = 机器人本体感觉 (关节角、gripper 开合)
- $a_{t:t+h}$ = 从 time $t$ 开始,horizon 长度 $h$ 的 action chunk
- $D_\psi$ = visual-primitive decoder,参数 $\psi$
- $\pi_\theta$ = action expert,参数 $\theta$

**直觉**: 这就像让 VLM 先在图上"指一下" next-stage target (画个框、点个点),action expert 看着这个"指"去执行。VLM 负责 "what + where",action expert 负责 "how"。分工明确。

**为啥不直接用 VLM 的 raw feature condition action expert**: 因为 raw feature 是 *distributed representation*,没有 explicit 的 spatial anchor。Primitive 把 spatial 信息 *bottleneck* 出来,变成 ~10 个 token,action expert 学起来容易得多。从 300k pixel → 14 DoF action 的 ill-posed mapping,被压成 10 primitive tokens → 14 DoF 的 well-posed mapping。

---

## 4. 监督信号哪来的 — 这是 paper 最聪明的地方

你现在肯定想问:OK,你让 VLM 学会"画 box",那 ground truth box 哪儿来? 让人标? 太贵。调 SAM? 那又回到 cascaded pipeline 了。

AVP 的答案: **直接从 robot 自己的运动学数据里推出来**。免费。

具体三步:

### Step 1: 找 keyframe

机器人抓东西、放东西,gripper 会突然闭合 / 张开。这是物理事件。我们监测 gripper state $g_t$ 的变化:

$$T_{key} = \{t \in [1, T] \mid |\Delta g_t| > \delta\}$$

变量:
- $T_{key}$ = keyframe 集合 (interaction 发生的关键时刻)
- $g_t$ = gripper 开合信号 (控制命令 vs 实际开合的差)
- $\Delta g_t$ = $g$ 在时间上的变化
- $\delta$ = 预设 threshold
- $T$ = 轨迹总长

人话: 找出机器人"刚刚抓了"或"刚刚放了"的那些时刻。

### Step 2: 拿到 3D 末端位置

每个 keyframe $t$,从 robot proprioception 里读出来 end-effector 的 3D 坐标:

$$P_t = [X_t, Y_t, Z_t]^T \in \mathbb{R}^3$$

变量:
- $P_t$ = end-effector 在 robot base frame 下的 3D 位置
- $X_t, Y_t, Z_t$ = 三个坐标分量

人话: 机器人自己知道自己的手在哪儿,直接读出来。

### Step 3: 投影到 image plane

用 pinhole camera model 把 3D 点投到 2D 像素:

$$z_c \begin{bmatrix} u_t \\ v_t \\ 1 \end{bmatrix} = K T_R^C \begin{bmatrix} P_t \\ 1 \end{bmatrix}$$

变量:
- $z_c$ = camera 坐标系下的 depth (z 轴值),做齐次坐标归一化用
- $u_t, v_t$ = 投影到 image plane 的 2D 像素坐标
- $K \in \mathbb{R}^{3\times3}$ = camera intrinsic matrix (内参,焦距 + 主点)
- $T_R^C \in SE(3)$ = robot base frame → camera frame 的 extrinsic (外参,旋转 + 平移)
- $[P_t; 1]$ = 齐次坐标

人话: 用标准几何把 3D 世界点投到 2D 图像上,得到一个 pixel 坐标 $(u_t, v_t)$。

把这个 2D anchor discretize 成 spatial grid (类似把 image 切成 patch),就是 visual primitive 的 ground truth label $y_t^{vp}$。

### 为啥这个设计很妙

1. **零额外标注**: teleop 时本来就要记录 robot state,投影是纯几何运算,几乎免费
2. **物理对齐**: 这个 anchor 就是机器人实际 interact 的位置,不是 SAM "觉得相关"的区域 — 在视觉相似的密集场景 (比如一整盘一样的中国象棋棋子) 里,SAM 容易分错,但 kinematics 永远准
3. **No semantic ambiguity**: 你不用纠结"这个 box 该包住整个棋子还是只包住顶部",因为 kinematics 告诉你 end-effector 接触的是哪儿,那就是哪儿
4. **Supervision alignment**: primitive label 和 action label 用的是 *同一个物理量* (end-effector pose),所以两个 loss 在 spatial 上 *永远 consistent*,不会出现"visual prompt 让你去 A,但 action demo 让你去 B"的矛盾

直觉: 这其实是 self-supervised signal — 你 teleop 时机器人已经在告诉你"我刚抓的是这个点",你只需要把它投回 image plane 当监督 label。

---

## 5. 训练目标

总 loss:

$$\mathcal{L} = \mathcal{L}_{act} + \lambda \mathcal{L}_{vp}$$

$$\mathcal{L}_{vp} = \mathcal{L}_{CE}(p_t, y_t^{vp})$$

变量:
- $\mathcal{L}_{act}$ = action prediction loss (flow-matching 的 denoising objective)
- $\mathcal{L}_{vp}$ = visual primitive prediction loss
- $\mathcal{L}_{CE}$ = cross-entropy loss
- $p_t$ = 预测的 primitive token
- $y_t^{vp}$ = 从 kinematics 投影得到的 GT primitive
- $\lambda$ = 平衡系数

**两阶段训练**:
- Phase 1 (chess 上 10k steps): 只训 visual-primitive decoder,只优化 $\mathcal{L}_{vp}$
- Phase 2 (chess 上 30k steps): joint train,两个 loss 一起

人话: 先让 decoder 学会"看图说话"(画出 target),再让 action expert 学会"照着 sketch 干活"。Curriculum 跟人类学精细运动一样 — 先用眼睛 fixate target,再练手眼协调。

---

## 6. 实验数据 — 看看账本

### 6.1 Chinese Chess Manipulation (主战场)

任务: 在 9×10 的密集棋盘上,把棋子从一个交叉点移到另一个,不能碰到周围棋子。72 条指令。

| Method | Instr. | Pick | Place | Avg. | Latency |
|--------|--------|------|-------|------|---------|
| π0 [8] | 62.50 | 45.83 | 25.00 | 44.44 | 0.16 s |
| π0.5 [1] | 75.00 | 63.89 | 20.83 | 53.24 | 0.16 s |
| Point-VLA [4] | 65.28 | 47.22 | 31.94 | 48.15 | **37.32 s** |
| DM0 [9] | 73.61 | 40.28 | 22.22 | 45.37 | 0.52 s |
| LDA [10] | 93.06 | 34.72 | 19.44 | 49.07 | 0.28 s |
| **AVP** | **98.61** | **90.28** | **81.94** | **90.28** | 0.27 s |

几个关键观察:

**Place success 是分水岭**。π0.5 Pick 还有 63.89%,但 Place 只有 20.83% — 抓得起来,放不下去。AVP Place 81.94%,提升 +61.11%。这告诉你 baseline 的 bottleneck 不是"抓",是"放准"。Visual primitive 在 place 阶段直接画出 target 交叉点,这个 bottleneck 当场解决。

**Point-VLA 的 latency 灾难**: 37.32 秒一步。这个数据值得单独拎出来 — 调 API 做 visual prompting 在真机上根本不可行。机器人早卡死了。这是 cascaded pipeline 的死穴。

**World model 路线 (DM0、LDA) 没赢**: DM0 avg 45.37%,LDA avg 49.07%,跟 π0.5 同档。说明 **pixel-level future prediction 信息密度太低** — 你给 action expert 一张 future image,它还得自己从中挖出 relevant region,难题没解决,只是平移了。AVP 用 sparse primitive 直接给 actionable target,信息密度刚好。

**LDA 的奇怪 pattern**: Instr. 93.06% (选对 arm),但 Pick 只有 34.72% — 它"听得懂"但"抓不起来"。这是 *localization gap*:VLM 理解了 high-level 指令,但 low-level spatial localization 没传到 action expert。AVP 通过 primitive 显式传递 spatial info,同时解决"理解"和"定位"。

**AVP latency 0.27s**: 比 π0.5 的 0.16s 慢一点 (因为 sequential two-stage:先 primitive 再 action),但 4 Hz 控制频率完全可用。

### 6.2 Domino Placement (双手高精度)

任务: 放多米诺骨牌,要求位置误差 < 一个骨牌厚度,角度误差 < 10°。

| Method | Pick | Place | Orien. | Avg. |
|--------|------|-------|--------|------|
| π0.5 | 87.50 | 64.58 | 93.75 | 81.94 |
| AVP | 100.00 | 64.58 | 100.00 | 88.19 |
| Imp. | +12.50 | 0.00 | +6.25 | +6.25 |

Place 持平,说明 domino 这个 task 的 place bottleneck 不在 spatial target (位置精度要求相对宽松),而在 orientation alignment。AVP 在 Orien. 上 +6.25% — visual primitive 的 box 长边方向给 action expert 提供了 orientation cue。

这个 gain 比 chess 小,合理 — domino 的 spatial reasoning 难度本来就比 chess 低,提升空间有限。

### 6.3 General Object Pick-and-Place (泛化)

任务: 抓放各种奇形怪状的物体 (眼影盘、塑料柚子、塑料橘子、塑料香蕉、玩具鸭)。

| Method | Instr. | Pick | Place | Avg. |
|--------|--------|------|-------|------|
| π0.5 | 100.00 | 71.79 | 23.08 | 64.96 |
| AVP | 100.00 | 90.24 | 68.29 | 86.18 |
| Imp. | 0.00 | +18.45 | +45.21 | +21.22 |

又是 Place 大幅提升 (+45.21%)。Pattern 一致 — **visual primitive 的最大价值在 placement**,因为 place 需要精确 spatial target,pick 相对容易。

### 6.4 Spatial-Compositional Generalization (最 striking 的实验)

训练数据全是 **Indirect** (A→C→B,经过 board 外 waypoint),测试时是 **Direct** (A→B,直接移动):

| Background | Task | π0.5 | AVP |
|------------|------|------|-----|
| Chessboard | Direct | 0/8 | **8/8** |
| Chessboard | Indirect | 7/8 | 8/8 |
| White cloth | Direct | 0/8 | **7/8** |
| White cloth | Indirect | 0/8 | **8/8** |

这个结果太 striking 了。

**π0.5 在 Direct 上完全失败 (0/8, 0/8)**。它在背 trajectory — 把 A→C→B 当 sequence memorization 来学,根本不会 compose 出没见过的 A→B。一旦中间 waypoint 没了,它就在 place 阶段卡住,超时失败。

**AVP 在 Direct 上 8/8**。它学到的不是 trajectory,是 *target 之间的关系*。Visual primitive 指定的是 next-stage target,不是完整路径。Pick 动作 + Place 动作可以独立 compose,因为 primitive 这个接口把它们解耦了。

**换 background (white cloth) AVP 依然 robust**。Visual primitive 把 task-relevant geometry 跟 background appearance 解耦 — 不管背景是棋盘还是白布,target 还是那个 target。

**Intuition**: 这就是 LLM 里 "memorize vs generalize" 的物理世界版本。π0.5 像在背剧本,AVP 像在学一个 spatial planning function。Karpathy 你写过 "Let's think step by step" 的工作 [11],这里 visual primitive decoder 本质上是 *visual chain-of-thought* — VLM 先"想"出 next-stage target (显式 visual output),再基于这个 thought 产生 action。结构同构,只是 token 是 spatial 而不是 language。

### 6.5 Cross-Domain Generalization

只在 chess 上训,直接 zero-shot transfer 到 45 个 unseen objects:

| Background | Task | π0.5 | AVP |
|------------|------|------|-----|
| Chessboard | Direct | 0/8 | 8/8 |
| Chessboard | Indirect | 7/8 | 8/8 |
| White cloth | Direct | 0/8 | 7/8 |
| White cloth | Indirect | 0/8 | 8/8 |

这就是 *in-context generalization* 的物理世界版本。AVP 学到的不是"chess piece 长啥样",而是"如何从 observation 里找出 next-stage target 并 ground 到 image plane"。Object appearance 怎么变都行,因为 primitive 是 *appearance-agnostic* 的 spatial target。

**为啥 π0.5 在 Chessboard + Indirect 上还能 7/8**: 这个 setting 相对简单,它可能 memorize 了"指令 → trajectory"的对应。但一旦 background 或 transition 结构变了,立刻崩盘。这是 *spurious correlation* 的典型表现 — 模型学到的是 surface pattern 而不是 underlying mechanism。

### 6.6 Visual Primitive Ablation

| Prompt | Instr. | Pick | Place | Avg. |
|--------|--------|------|-------|------|
| None | 100 | 70 | 64 | 78 |
| Box | 100 | 82 | 68 | 83 |
| Box + Mask | 100 | 86 | 70 | 85 |
| Box + Mask + Mem. | 100 | 94 | 78 | 91 |

最后一行 Memory 很有意思:把上一步 subtask 的 visual primitive 保留在图里 (用不同形式标记),给 action expert 提供 cross-step spatial context。Pick 从 86→94,Place 从 70→78。

人话: 给 action expert 一个"我刚才抓的是这个点"的 *spatial memory*,帮它在 place 阶段保持目标一致性。这跟 LLM 里 induction heads 的作用类似 [12],但这里是 *spatially grounded* 的 induction,不是 latent token 的 induction。

### 6.7 Primitive Type 对比

| Primitive Type | Pick | Place |
|----------------|------|-------|
| Raw (no prompt) | 70 | 64 |
| Point | 86 | 74 |
| Box | 82 | 68 |
| Box-mask | 86 | 70 |

**Point > Box** 在 Pick 上 (86 vs 82),有点反直觉。我的解读:Point 的 spatial precision 最高 (单一像素 anchor),Box 引入了"区域内哪里才是 true target"的 ambiguity。但 Box-mask (mask 掉区域外的 background) 反而追上 Point,说明 *suppressing distractor* 跟 *precise anchor* 同等重要。

### 6.8 Mask Opacity

| Opacity α | Pick | Place |
|-----------|------|-------|
| α = 0 (透明) | 82 | 68 |
| α = 0.7 | 86 | 70 |
| α = 0.9 (近黑) | 86 | 74 |

α 越大 (background 越被压暗),Place 越好 (74 vs 68)。Suppressing distractor 对 placement 精度关键。这跟 attention 机制里"focus on relevant token"的直觉完全一致。

---

## 7. 为啥 AVP work — 我的 mental model

把所有线索串起来:

**核心 insight**: VLA 的 bottleneck 不在 perception (VLM 已经很强),不在 control (flow-matching action expert 已经够 expressive),在 **interface**。Language 太 abstract,pixel 太 dense,external prompt 太 fragile。AVP 找到了一个 *恰好 dense enough to be actionable, 恰好 sparse enough to be learnable* 的中间态:visual primitive。

**三个深层原因**:

### 7.1 Information bottleneck

机器人 action 通常 7-14 DoF (这 paper 是 14 DoF bimanual)。Visual observation 是 ~300k pixel (640×480×多视角)。从 300k → 14 是巨大 dimensionality reduction。如果不给 explicit spatial target,action expert 必须从 300k pixel 里 *implicit* 学出 14 DoF mapping — 这是 ill-posed 的。

Visual primitive 把 300k pixel 压成 ~10 个 spatial token,dimensionality reduction 变成 10 → 14,well-posed。Action expert 的学习 burden 大幅降低。

### 7.2 Compositionality

learned motion skill 可以 reuse。Pick 动作在所有 task 里都类似 (gripper 闭合 + lift),差异只在 *where*。AVP 把 where 抽出来 (primitive),保留 how (action expert),所以 transfer 容易。

这跟 software engineering 里"逻辑跟数据分离"的原则同构 — 你不会把数据库 schema 焊死在 SQL 查询里,你也不会把 target location 焊死在 motor trajectory 里。

### 7.3 Supervision alignment

kinematics-derived primitive 跟 action 在物理上对齐 — 它们 share 同一个 end-effector pose。所以 visual primitive supervision 和 action supervision *never conflict*。

对比一下:如果你用 SAM 给的 mask 当监督,SAM 说"这个区域是 task-relevant",但 teleop 数据里 robot 实际抓的是另一个区域 — 两个 loss 打架。AVP 的设计从源头消除了这个问题。

---

## 8. 跟其他范式的人话对照

### 8.1 跟 π0.5 subgoal planning 对比

π0.5 把 instruction 拆成 subtask description (language space)。问题:language 表达不了 fine-grained spatial distinction ("第 3 行第 5 列" vs "第 3 行第 6 列" 在 token 层面几乎一样)。

AVP 把 subgoal 表达成 *spatial primitive*,天然 dense。等于把 π0.5 的 *language plan* 换成 *visual sketch plan*。

### 8.2 跟 π0.7 / world model 对比

π0.7 [2] 和 world-action models [3] 给 subgoal image 当 visual target。问题:subgoal image 信息冗余,action expert 还得自己 figure out image 里哪部分 task-relevant。

AVP 的 primitive 是 *sparse actionable* 的 — 只画 task-relevant 的点/box,其他全 mask。这跟 DETR [13] 用 sparse query 替代 dense detection head 的 motivation 同构。

### 8.3 跟 Visual Prompting (Point-VLA, TraceVLA) 对比

cascaded pipeline 的问题前面讲过。但 AVP 还有一个 *conceptual* 优势:visual primitive 是 *learned latent*,不是 *external prior*。可以跟 action co-adapt — 如果某个 spatial anchor 在某些 config 下对 action expert 不 helpful,gradient flow 回来调 decoder,让它产生更 actionable primitive。Cascaded pipeline 是 *frozen* 的,prompt 错了就错了。

### 8.4 跟 Chain-of-Thought 的对照

Karpathy 你讲过 CoT [11][14]。这里 visual primitive decoder 本质上是 *visual chain-of-thought*:VLM 先"想"出 next-stage target (显式 visual output),再基于这个 thought 产生 action。结构同构,只是 token 是 spatial 的。

这给了一个有趣的延伸:能不能在 primitive 层面做多步 reasoning?比如"先把 A 移到 B,再把 C 移到 D"这种 multi-step planning,用一串 primitive token 表达,每个 primitive 对应一个 sub-action。Paper 里 memory-enhanced 版本已经初探了这条路。

---

## 9. 局限 (Paper 自己承认)

1. **Sequential inference latency**: 0.27s vs π0.5 0.16s,慢 ~70%。可以 speculate:speculative decoding 或 parallel primitive+action 训练 (类似 NART, non-autoregressive translation) 能缓解
2. **Hand-eye calibration sensitivity**: projection 依赖 $T_R^C$ 准确,相机 extrinsic drift 会直接污染 primitive label。这跟 SLAM/VIO 的 robustness 问题是同一类
3. **Primitive type 还是手工选择**: Box vs Point vs Mask 还需要 ablation,没完全自动化。一个 *learnable primitive vocabulary* (类似 VQ-VAE [15] 的 codebook) 可能是下一步
4. **Memory 只到上一步**: 跨多步 spatial context (e.g., 我刚才抓过哪些,现在该抓哪个) 还没 explore。可以想象一个 *spatial episodic memory* module

---

## 10. 我的延伸联想 (Karpathy 你说宁可 hallucinate)

- **AVP 的 primitive 本质上是 programmatic action representation**,跟 "Code as Policies" [16] 思路同源 — 只是 code 是 spatial 而不是 symbolic。Code as Policies 用 LLM 生成 Python code 调用 perception API,AVP 用 VLM 生成 visual token 调用 action expert,都是 *explicit intermediate representation*
- **把 visual primitive 当 affordance map 看**,它跟 affordance prediction [17] 的思路可以融合。Affordance 是"哪里可以 interact",primitive 是"接下来要 interact 哪里",一个 prior 一个 plan
- **跟 DeepMind Genie 2 [18] 对照**: world model + action 的 latent dynamics,AVP 的 primitive 可能是一种 *discrete latent dynamics state*。Genie 2 在 latent space 里预测 future,AVP 在 image plane 上预测 next-stage target,都是 *predictive* 的,但 AVP 是 task-conditioned + spatially grounded
- **跟 Slot Attention [19] / Object-centric representation 对照**: visual primitive 可以看成 *task-conditioned slot*。Slot Attention 学的是 object-centric decomposition,AVP 学的是 task-centric spatial decomposition
- **跟 Diffusion Policy [20] 的对比**: Diffusion Policy 用 diffusion 在 action space 里 sample,AVP 用 flow-matching (diffusion 的变体) 在 action space 里 sample,但 condition 多了一个 visual primitive。等于把 condition 从 "raw image+language" 升级成 "image+language+visual primitive",signal-to-noise ratio 高很多
- **Karpathy 你的 Software 2.0 / 3.0 框架 [21][22]**: AVP 完美 fit Software 2.0 的逻辑 — 不是手写规则,而是设计 differentiable architecture 让网络学。但 AVP 也 hint 了 Software 3.0 的方向:visual primitive 这种 explicit intermediate representation,既可被网络学,也可被人 inspect / debug / intervene,这是 *介于 2.0 和 3.0 之间* 的范式

参考:
- Code as Policies: https://code-as-policies.github.io/
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Slot Attention: https://arxiv.org/abs/2006.15055
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- VQ-VAE: https://arxiv.org/abs/1711.00937
- Karpathy Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- Karpathy Software 3.0 talk: https://www.youtube.com/watch?v=LgUmmm5sSyE

---

## 11. 一句话总结 (人话版)

**AVP 让 VLM 先在图上"点一下"下一步要去哪儿,action expert 再照着这个点去执行**。监督信号直接从机器人自己的运动学数据推出来,不靠人工标注不靠 SAM 不靠 API。中国象棋上比 π0.5 提升 37.04%,而且展现出惊人的 spatial-compositional generalization (Direct 8/8 vs baseline 0/8) 和 cross-domain generalization。

**核心 takeaway for Karpathy**: interface design > raw scaling。VLA 不是只堆 VLM size 就能 work 的 — bottleneck 在 modality 之间的 information bottleneck design。这跟你 Software 2.0 / 3.0 的思考是同构的:不是让一个 mega-network 端到端学一切,而是设计好 *differentiable interfaces* 让不同 capability 模块协作。VLM 提供 reasoning,primitive 提供 spatial interface,action expert 提供 motor skill — 三层 stack,各自擅长,联合训练。

---

## 关键 Reference 链接汇总

**Baseline VLA models**:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- RDT-1B: https://arxiv.org/abs/2410.07864
- GR-3: https://arxiv.org/abs/2507.15493

**Cascaded visual prompting**:
- Point-VLA: https://arxiv.org/abs/2512.18933
- VP-VLA: https://arxiv.org/abs/2603.22003
- TraceVLA: https://arxiv.org/abs/2412.10345
- MOKA: https://arxiv.org/abs/2406.09187

**Action / flow-matching foundations**:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Flow Matching: https://arxiv.org/abs/2403.04125

**Datasets**:
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952

**World / spatial reasoning models**:
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

**Karpathy 你自己的相关思考**:
- Software 2.0: https://karpathy.medium.com/software-2-0-a64152b37c35
- Software 3.0 / AI Coding talk: https://www.youtube.com/watch?v=LgUmmm5sSyE
- State of GPT: https://www.youtube.com/watch?v=9qjy9zZdqW8
- Let's think step by step (CoT 论文): https://arxiv.org/abs/2201.11903

---

要不要我 drill-down 到某一个具体部分? 比如:
- Flow-matching action expert 的数学细节 (Lipman et al. 的 OT formulation [23])
- Visual primitive decoder 的 tokenization scheme (VQ-VAE style 还是直接 bin discretize?)
- Hand-eye calibration 的 sensitivity 分析 (paper 提了但没深入)
- Spatial-compositional generalization 的更深层机制 (为啥 baseline 完全不能 compose)

随时告诉我。

参考:
- Flow Matching original: https://arxiv.org/abs/2305.14886
- VQ-VAE: https://arxiv.org/abs/1711.00937

---

# AVP: Action with Visual Primitives 深度解析

这是 Karpathy 在看的"Action with Visual Primitives"论文,来自清华和 Anyverse Dynamics。我尽量 build 你的 intuition,从 high-level motivation 到技术细节、再到实验数据,系统讲解。

---

## 1. Motivation: VLA 的"接口塌陷"问题

当前主流 VLA (如 RT-2 [1], OpenVLA [2], π0 [3], π0.5 [4]) 都把 language、vision、action 塞进一个 single forward pass:

$$a_{t:t+h} = \pi_\theta(o_t, l, s_t)$$

变量含义:
- $a_{t:t+h}$: time step $t$ 开始,horizon 长度为 $h$ 的 action chunk (动作序列)
- $o_t$: 当前 visual observation (通常是多视角 RGB)
- $l$: language instruction
- $s_t$: proprioceptive state (本体感觉,如关节角度、gripper 状态)
- $\pi_\theta$: 参数为 $\theta$ 的 policy

**问题**: 这个 formulation 让 action expert 必须从 raw VLM features 里 *implicitly* 重新学习 spatial reasoning。VLM 预训练里已经有了的 spatial/semantic 能力,在 action expert 这一头被"压扁"了,等于浪费。

**AVP 的核心 insight**: VLM 和 action expert 之间应该有一个 *explicit 的、spatially grounded 的通信协议*。VLM 负责 "what + where" (推理 next-stage target + 它的 spatial location),action expert 负责 "how" (执行 motor skill)。这跟认知科学里 dorsal stream (where pathway) vs ventral stream (what pathway) 的分工有点像,也跟 Karpathy 你之前讲过 "system 1 vs system 2" 在 LLM 里的对照有异曲同工之处。

参考:
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 paper: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818

---

## 2. AVP 架构解析

AVP = VLM + Visual-Primitive Decoder + Flow-Matching Action Expert 三件套,全部 end-to-end。

### 2.1 三大组件

**Component 1: VLM backbone**
基于 π0.5 的 VLM (PaliGemma-style 视觉编码器 + LLM decoder),输入 $(o_t, l)$,产出 multimodal context tokens。这些 tokens 既包含语义理解,也包含空间上下文,但还没被"翻译"成 actionable 的 spatial target。

**Component 2: Visual-Primitive Decoder $D_\psi$**
一个 autoregressive decoder,从 VLM 的 context tokens 里 *自回归地* 预测离散化的 visual primitive $p_t$:

$$p_t = D_\psi(\text{VLM}(o_t, l))$$

变量:
- $p_t$: time step $t$ 的 visual primitive token 序列 (离散化,可以表示 point / box / mask 等)
- $D_\psi$: decoder,参数为 $\psi$
- $\text{VLM}(\cdot)$: VLM 的 forward pass

注意 $p_t$ 是 *discretized* 的,这意味着可以用 cross-entropy 监督 (后面讲)。

**Component 3: Projection + Action Expert**
预测出来的 $p_t$ 是 "language space" 里的离散 token,需要被 *project* 到 visual token space 才能 condition action expert:

$$z_t^{vp} = \text{Proj}(p_t, o_t)$$

变量:
- $z_t^{vp}$: visual-primitive tokens,投影到 visual embedding space 后的表示 (上标 $vp$ = visual primitive)
- $\text{Proj}(\cdot)$: projection module,以 $p_t$ 和 $o_t$ 为输入 (这里 $o_t$ 提供 image-plane 坐标的 spatial alignment)
- 融合: $z_t^{aug} = \text{concat/fuse}(z_t^{vp}, \text{original multimodal tokens})$

然后 action expert (flow-matching head) 基于 augmented representation 生成 action:

$$a_{t:t+h} = \pi_\theta(z_t^{aug}, s_t)$$

变量:
- $z_t^{aug}$: augmented multimodal tokens (上标 $aug$ = augmented)
- $\pi_\theta$: action expert (flow-matching policy),参数 $\theta$

**直觉**: 这相当于在 VLM 的"思考过程"和 action expert 的"肌肉控制"之间,加了一个 *spatial sketch pad*。VLM 先在图上"画"出来下一步要去哪儿,然后 action expert 照着这个 sketch 执行。这跟人类做精细操作时,眼睛先 fixate 到 target、手再跟过去的视觉-运动协同机制非常像。

参考 flow-matching:
- π0 用 flow matching: https://arxiv.org/abs/2410.24164
- Diffusion Policy (原始): https://arxiv.org/abs/2303.04137
- Flow Matching for policy: https://arxiv.org/abs/2403.04125

### 2.2 跟 cascaded visual prompting 的对比

cascaded visual prompting 方法 (Point-VLA [5], VP-VLA [6], TraceVLA [7]) 的 pipeline 是:

$$v_t = \mathcal{M}_{per}(o_t, l), \quad a_{t:t+h} = \pi_{act}(\mathcal{F}(o_t, v_t), l, s_t)$$

变量:
- $v_t$: visual prompt (坐标点 / bbox / mask / trajectory)
- $\mathcal{M}_{per}$: 外部 perception module (SAM, Grounding DINO, 或调 API 的 VLM)
- $\mathcal{F}$: visual composition operator (pixel-level overlay)
- $\pi_{act}$: 下游 action policy

**致命缺点**:
1. **Cascading error**: $\mathcal{M}_{per}$ 错了,下游 $\pi_{act}$ 无法纠正
2. **Latency**: Point-VLA 在 paper 里实测 37.32 s/step (调 Kimi API),完全不可用
3. **Heuristic dependency**: SAM 的 mask 需要针对 task 设计,不可 scale

AVP 把 $\mathcal{M}_{per}$ 内化到 VLM 里,end-to-end co-train,完全消除外部依赖。

参考:
- Point-VLA: https://arxiv.org/abs/2512.18933
- TraceVLA: https://arxiv.org/abs/2412.10345
- SAM 2: https://arxiv.org/abs/2408.00714

---

## 3. Action-Centric Visual-Primitive Supervision (关键创新)

这是 paper 里我认为最 elegant 的部分。**监督信号直接从 end-effector kinematics 派生**,无需人工标注、无需 SAM、无需调 API。

### 3.1 Pipeline 三步走

**Step 1: Kinematic Keyframe Extraction**
从 gripper state $g_t$ 里检测 keyframe:

$$T_{key} = \{t \in [1, T] \mid |\Delta g_t| > \delta\}$$

变量:
- $T_{key}$: keyframe set (下标 $key$ 表示 keyframe)
- $g_t$: gripper state signal (e.g., 控制命令 vs 实际开合度的差值)
- $\Delta g_t$: $g$ 在时间上的变化量
- $\delta$: 预设 threshold
- $T$: 轨迹总长度

直觉: 抓取开始 / 释放时刻,gripper 会突然闭合 / 张开,$|\Delta g_t|$ 出现尖峰,这就是"interaction 事件"发生的物理信号。

**Step 2: 3D End-Effector Pose**
在每个 keyframe $t \in T_{key}$,从 robot proprioception 里读 3D 末端位置:

$$P_t = [X_t, Y_t, Z_t]^T \in \mathbb{R}^3$$

变量:
- $P_t$: 3D 位置向量,在 robot base frame 下
- $X_t, Y_t, Z_t$: 三个坐标分量

**Step 3: Image-Plane Projection**
用标准 pinhole camera model 投影:

$$z_c \begin{bmatrix} u_t \\ v_t \\ 1 \end{bmatrix} = K T_R^C \begin{bmatrix} P_t \\ 1 \end{bmatrix}$$

变量:
- $z_c$: camera frame 下的 depth scaling factor (相机坐标系 z 轴的深度)
- $u_t, v_t$: 投影到 image plane 的 2D 像素坐标 (下标 $t$ 表示 time step)
- $K \in \mathbb{R}^{3 \times 3}$: camera intrinsic matrix (相机内参)
- $T_R^C \in SE(3)$: 从 robot base frame 到 camera frame 的 extrinsic transformation (上标 $C$ = camera,下标 $R$ = robot)
- $[P_t; 1]$: 齐次坐标形式

得到 2D anchor $m_t = (u_t, v_t)$,再 *discretize* 到 spatial grid 上 (类似 pixel tokenization),作为 visual primitive 的监督 label。

### 3.2 为什么这个设计 elegant

1. **零额外标注成本**: 你已经在 teleop 时记录 robot proprioception,投影是纯几何运算,几乎免费
2. **物理对齐**: 这个 anchor 就是 robot 实际 interact 的位置,而不是 SAM 分割出来的"语义上相关"但运动学上无关的区域
3. **Avoids semantic ambiguity**: 在 chess 这种视觉相似的密集场景里,SAM 给 bbox 很容易分错棋子,但 kinematics 永远准

**Intuition**: 这其实是一个 *self-supervised signal that happens to be spatially aligned with action*。它把 action 的"目标"和 perception 的"location"用同一个物理量 (end-effector pose) 统一起来,从而 visual primitive supervision 和 action supervision 在 spatial 上是 *consistent* 的。

---

## 4. 训练目标

$$\mathcal{L} = \mathcal{L}_{act} + \lambda \mathcal{L}_{vp}$$

$$\mathcal{L}_{vp} = \mathcal{L}_{CE}(p_t, y_t^{vp})$$

变量:
- $\mathcal{L}$: 总 loss
- $\mathcal{L}_{act}$: action prediction loss (flow-matching 的 denoising objective)
- $\mathcal{L}_{vp}$: visual primitive prediction loss
- $\mathcal{L}_{CE}$: cross-entropy loss
- $p_t$: predicted visual primitive (离散 token)
- $y_t^{vp}$: ground-truth visual primitive (上标 $vp$ = visual primitive,从 kinematics 投影 + 离散化得到)
- $\lambda$: 平衡系数 (paper 没明说数值,推测在 0.1-1.0 量级)

**两阶段训练**:
- Phase 1 (10k steps for chess): 只训 visual-primitive decoder,$\mathcal{L}_{vp}$ only
- Phase 2 (30k steps): joint train,$\mathcal{L}_{act} + \lambda \mathcal{L}_{vp}$

直觉: 先让 decoder 学会"看图说话" (画出 target),再让 action expert 学会"照着 sketch 干活"。这个 curriculum 跟人类学精细运动技能的方式有点像 — 先用眼睛 fixate target,再练手眼协调。

---

## 5. 实验数据深度解析

### 5.1 Chinese Chess Manipulation (主战场)

| Method | Instr. | Pick | Place | Avg. | Latency |
|--------|--------|------|-------|------|---------|
| π0 [2] | 62.50 | 45.83 | 25.00 | 44.44 | 0.16 s |
| π0.5 [3] | 75.00 | 63.89 | 20.83 | 53.24 | 0.16 s |
| Point-VLA [19] | 65.28 | 47.22 | 31.94 | 48.15 | **37.32 s** |
| DM0 [28] | 73.61 | 40.28 | 22.22 | 45.37 | 0.52 s |
| LDA [29] | 93.06 | 34.72 | 19.44 | 49.07 | 0.28 s |
| **AVP** | **98.61** | **90.28** | **81.94** | **90.28** | 0.27 s |

**关键观察**:
1. **Place success 是分水岭**: π0.5 只有 20.83%,AVP 81.94%,提升 +61.11%。这说明 baseline 的 bottleneck 不是"抓起来"(Pick 63.89% 还行),而是"放下去放得准"。AVP 的 visual primitive 在 place 阶段给出明确的 target intersection,直接解决了 spatial grounding 问题。
2. **Point-VLA 的 latency 灾难**: 37.32 s/step 完全不可用 — 调 API 的代价。这就是 cascaded pipeline 的死穴。
3. **World model 路线 (DM0, LDA) 没赢**: DM0 0.52s 但 Avg 45.37%,LDA 0.28s 但 Avg 49.07%。Pixel-level future prediction 在密集空间场景里 *信息密度太低*,action expert 还得自己从 future frame 里"挖"出 relevant region。这印证了 paper 的论点:dense pixel prediction ≠ sparse actionable primitive。
4. **AVP latency 0.27s**: 比 π0.5 的 0.16s 慢一点,因为两阶段 sequential inference (先 primitive 再 action),但完全可接受 (4 Hz 控制频率)。

**Karpathy 你可能感兴趣的**: LDA 在 Instr. (instruction following,选对 arm) 上 93.06% 很高,但 Pick 只有 34.72% — 这是个 *localization gap*:它能"听懂"要去哪边,但抓不起来。AVP 在两者都碾压,说明 visual primitive 同时解决了"理解"和"定位"。

### 5.2 Domino Placement (双手高精度)

| Method | Pick | Place | Orien. | Avg. |
|--------|------|-------|--------|------|
| π0.5 | 87.50 | 64.58 | 93.75 | 81.94 |
| AVP | 100.00 | 64.58 | 100.00 | 88.19 |
| Imp. | +12.50 | 0.00 | +6.25 | +6.25 |

Place 持平,说明 domino 的 placement bottleneck 不是 spatial target (位置精度要求相对宽松:one domino thickness 内),而是 orientation alignment。AVP 在 Orien. 上 +6.25%,从 visual primitive 提供的 orientation cue (bbox 的长边方向) 受益。这个 gain 比 chess 小,因为 domino task 的 spatial reasoning 难度本来就比 chess 低。

### 5.3 General Object Pick-and-Place (泛化)

| Method | Instr. | Pick | Place | Avg. |
|--------|--------|------|-------|------|
| π0.5 | 100.00 | 71.79 | 23.08 | 64.96 |
| AVP | 100.00 | 90.24 | 68.29 | 86.18 |
| Imp. | 0.00 | +18.45 | +45.21 | +21.22 |

这里 π0.5 Instr. 是 100% (因为是简单二选一),但 Place 只有 23.08% — 又是 place 问题。AVP Place 68.29%,提升 +45.21%,跟 chess 上的 pattern 一致。这强烈说明:**visual primitive 的最大价值在 placement 阶段**,因为 place 需要精确 spatial target,pick 相对容易。

### 5.4 Spatial-Compositional Generalization (最 exciting 的实验)

训练数据全是 Indirect (A→C→B,经过 external waypoint),测试时是 Direct (A→B,直接移动):

| Background | Task | π0.5 | AVP |
|------------|------|------|-----|
| Chessboard | Direct | 0/8 | 8/8 |
| Chessboard | Indirect | 7/8 | 8/8 |
| White cloth | Direct | 0/8 | 7/8 |
| White cloth | Indirect | 0/8 | 8/8 |

**这个结果太 striking 了**:
- π0.5 在 Direct 上 *完全失败* (0/8,0/8),说明它把 A→C→B 当成 *sequence memorization* 来学,无法 compose 出 A→B
- AVP 在 Direct 上 8/8,说明它学到的不是 trajectory 而是 *"target 之间的关系"*,因为 visual primitive 指定的是 next-stage target 而不是完整路径
- 即使换 background (white cloth),AVP 依然 robust,因为 visual primitive 把 task-relevant geometry 跟 background appearance 解耦了

**Intuition**: 这跟 LLM 里 "memorize vs generalize" 的对照很像。π0.5 像在背 trajectory,AVP 像在学一个 spatial planning function。Visual primitive 充当了一个 *compositional interface*:pick 和 place 可以独立 compose,因为 primitive 给的是"目标"而非"路径"。

### 5.5 Cross-Domain Generalization

只在 chess 上训,直接 transfer 到 45 个 unseen objects (eyeshadow palette, plastic pomelo, plastic tangerine, plastic banana, toy duck 等):

- 棋盘 + Direct: π0.5 0/8,AVP 8/8
- 棋盘 + Indirect: π0.5 7/8,AVP 8/8
- 白布 + Direct: π0.5 0/8,AVP 7/8
- 白布 + Indirect: π0.5 0/8,AVP 8/8

**Karpathy 你应该对这种现象很敏感**: 这就是 *in-context generalization* 的物理世界版本。AVP 学到的不是"chess piece 长啥样",而是"如何从 observation 里找出 next-stage target 并 ground 到 image plane"。这个能力一旦学到了,object appearance 怎么变都行,因为 visual primitive 是 *appearance-agnostic* 的 spatial target。

### 5.6 Visual Primitive Ablation

| Prompt | Instr. | Pick | Place | Avg. |
|--------|--------|------|-------|------|
| None | 100 | 70 | 64 | 78 |
| Box | 100 | 82 | 68 | 83 |
| Box + Mask | 100 | 86 | 70 | 85 |
| Box + Mask + Mem. | 100 | 94 | 78 | 91 |

**Memory 那行很有意思**: 把上一步 subtask 的 visual primitive 保留在 image 里 (distinct form),给 cross-step spatial context。Pick 从 86→94,Place 从 70→78。

直觉: 这相当于给 action expert 一个"我刚才抓的是这个点"的 *spatial memory*,帮它在 place 阶段保持目标 consistency。跟 RT-2 / OpenVLA 里 history token 的作用类似,但这里 memory 是 *spatially grounded* 的,而不是 abstract latent state。这有点 Karpathy 你在 nanoGPT 之后讲的 "induction heads" 在 vision-action 里的对应 — 不是 latent memory,而是 explicit spatial induction。

### 5.7 Primitive Type Ablation

| Primitive Type | Instr. | Pick | Place |
|----------------|--------|------|-------|
| Raw | 100 | 70 | 64 |
| Point | 100 | 86 | 74 |
| Box | 100 | 82 | 68 |
| Box-mask | 100 | 86 | 70 |

**Point > Box** 在 Pick 上 (86 vs 82),有点反直觉。我的解读:
- Point 的 spatial precision 最高 (单一像素 anchor)
- Box 提供了 region 但引入了"区域内哪里才是 true target"的 ambiguity
- 但 Box-mask (mask 掉 region 外的 background) 反而最好,因为抑制了 distractor

**Mask Opacity**:
| Opacity | Pick | Place |
|---------|------|-------|
| α=0 | 82 | 68 |
| α=0.7 | 86 | 70 |
| α=0.9 | 86 | 74 |

α 越大 (background 越黑),Place 越好 (74 vs 68),说明 *suppressing distractor* 对 placement 精度关键。这跟 attention 机制里"focus on relevant token"的直觉一致。

---

## 6. 跟其他范式的 intuition 对照

### 6.1 跟 π0.5 的 subgoal planning 对比

π0.5 把 instruction 拆成 subtask description (language space),让 action expert 学 atomic skill。问题:**language 太抽象**,无法表达 fine-grained spatial distinction ("放在第 3 列第 5 行" vs "放在第 3 列第 6 行")。

AVP 把 subgoal 表达为 *spatial primitive*,天然 dense。这相当于把 π0.5 的 *language plan* 换成了 *visual sketch plan*。

### 6.2 跟 world model / π0.7 对比

π0.7 [8] 和 world-action models [9] 给 subgoal image 作为 visual target。问题:**subgoal image 信息冗余**,action expert 还得自己 figure out image 里哪部分是 task-relevant。

AVP 的 visual primitive 是 *sparse actionable* 的 — 只画 task-relevant 的点/box,其他全部 mask 掉。这跟 DETR 用 sparse query 替代 dense detection head 的 motivation 很像。

### 6.3 跟 Visual Prompting (Point-VLA, TraceVLA) 对比

cascaded pipeline 的问题前面讲过。但 AVP 还有一个 *conceptual* 优势:它把 visual prompt 当成 *learned latent*,而不是 *external prior*。这意味着 visual primitive 可以 *跟 action co-adapt* — 如果某个 spatial anchor 在某些 configuration 下对 action expert 不 helpful,gradient 可以 flow 回来调整 decoder,让它产生更 actionable 的 primitive。Cascaded pipeline 是 *frozen* 的,prompt 错了就错了。

### 6.4 跟 Chain-of-Thought (CoT) 在 reasoning 里的对照

Karpathy 你写过 "Let's think step by step" 的工作。这里 AVP 的 visual primitive decoder 本质上是 *visual chain-of-thought*:VLM 先"想"出 next-stage target (显式 visual output),再基于这个 thought 产生 action。这跟 language CoT 的结构同构,但 token 是 spatial 的。

参考 Karpathy 的 CoT talk:
- https://www.youtube.com/watch?v=9qjy9zZdqW8 (state of GPT)
- https://karpathy.ai/ 

---

## 7. 我的 Intuition:为什么 AVP work

让我把所有线索串起来,build 一个完整 mental model:

**核心 insight**: VLA 的 bottleneck 不是 *perception* (VLM 已经很强),也不是 *control* (flow-matching action expert 已经够 expressive),而是 *interface*。Language 太 abstract,pixel 太 dense,external prompt 太 fragile。AVP 找到了一个 *恰好 dense enough to be actionable,恰好 sparse enough to be learnable* 的中间态: visual primitive。

**为什么 primitive work 的物理直觉**:
1. **Spatial 信息 bottleneck**: 机器人 action 通常是 7-14 DoF (这里 14 DoF bimanual),visual observation 是 ~300k pixel (640×480×多视角)。从 300k → 14 是巨大的 dimensionality reduction。如果不给 explicit spatial target,action expert 必须从 300k pixel 里 implicit 学出 14 DoF 的 mapping — 这是 ill-posed 的。Visual primitive 把 300k pixel 压成 ~10 个 spatial token,极大降低了 action expert 的 learning burden。
2. **Compositionality**: learned motion skill 可以 reuse。Pick 动作在所有 task 里都类似 (gripper 闭合 + lift),差异在 *where*。AVP 把 where 抽出来 (visual primitive),保留 how (action expert),所以 transfer 容易。
3. **Supervision alignment**: kinematics-derived primitive 跟 action 在物理上对齐 — 它们 share 同一个 end-effector pose。这意味着 visual primitive supervision 和 action supervision *never conflict*,不会出现"visual prompt 让你去 A,但 action 数据让你去 B"的 label inconsistency 问题。

**为什么 Place 比 Pick 提升大**: Pick 的 target 通常是当前 observation 里 *salient* 的物体,网络容易 attend 到。Place 的 target 是 *imagined* 的 (棋盘上某个空位),需要 spatial reasoning 而非 visual saliency。VLM 擅长 reasoning,把它的 reasoning 显式化为 primitive 后,Place 直接受益。

---

## 8. 局限与 Future Work (Paper 自己 acknowledge)

1. **Sequential inference latency**: 0.27s vs π0.5 0.16s,慢了 ~70%。可以 speculate:用 *speculative decoding* 或 *parallel primitive+action* 训练 (类似 non-autoregressive translation) 能缓解。
2. **Hand-eye calibration sensitivity**: projection 依赖 $T_R^C$ 准确,相机 extrinsic drift 会直接污染 primitive label。这跟 SLAM/VIO 的 robustness 问题是同一类。
3. **Primitive type 还是手工选择**: Box vs Point vs Mask 还需要 ablation,没完全自动化。一个 *learnable primitive vocabulary* (类似 VQ-VAE 的 codebook) 可能是下一步。
4. **Memory 只到上一步**: 跨多步的 spatial context (e.g., 我刚才抓过哪个,现在该抓哪个) 还没 explore。可以想象一个 *spatial episodic memory* module。

**我自己的延伸联想** (hallucination-prone 但 Karpathy 你说宁可联想):
- AVP 的 visual primitive 本质上是 *programmatic action representation*,跟 "Code as Policies" [10] 思路同源 — 只是 code 是 spatial 而不是 symbolic
- 把 visual primitive 当成 *affordance map* 看,它跟 NIE (Neural Information Engineering) 里 affordance prediction 的思路可以融合
- 跟 DeepMind 的 Genie [11] / Genie 2 [12] 对照:world model + action 的 latent dynamics,可能 AVP 的 primitive 是一种 *discrete latent dynamics state*
- 跟 Slot Attention [13] / Object-centric representation 对照:visual primitive 可以看成 *task-conditioned slot*

参考:
- Code as Policies: https://code-as-policies.github.io/
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/
- Slot Attention: https://arxiv.org/abs/2006.15055

---

## 9. 一句话总结

AVP 把 VLA 里 VLM 和 action expert 之间的 *implicit feature coupling* 替换为 *explicit visual primitive interface*,通过 kinematics-derived supervision 让 VLM 学会"画 target",让 action expert 学会"照 sketch 执行",在 chess、domino、general object 三个 task 上分别提升 37.04%、6.25%、21.22%,并展现出惊人的 spatial-compositional 和 cross-domain generalization。

**核心 takeaway for Karpathy**: interface design > raw scaling。VLA 不是只堆 VLM size 就能 work 的 — bottleneck 在 *modality 之间的 information bottleneck design*。这跟你在 Software 2.0 / 3.0 的思考是同构的:不是让一个 mega-network 端到端学一切,而是设计好 *differentiable interfaces* 让不同 capability 模块协作。

参考 Karpathy Software 2.0 / 3.0:
- https://karpathy.medium.com/software-2-0-a64152b37c35
- https://www.youtube.com/watch?v=LgUmmm5sSyE (Software 3.0 / AI Coding talk)

---

## 关键 Reference 链接汇总

**Baseline VLA models**:
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-2: https://arxiv.org/abs/2307.15818
- RDT-1B: https://arxiv.org/abs/2410.07864
- GR-3: https://arxiv.org/abs/2507.15493

**Cascaded visual prompting**:
- Point-VLA: https://arxiv.org/abs/2512.18933
- VP-VLA: https://arxiv.org/abs/2603.22003
- TraceVLA: https://arxiv.org/abs/2412.10345
- MOKA: https://arxiv.org/abs/2406.09187

**Action / flow-matching foundations**:
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Flow Matching: https://arxiv.org/abs/2403.04125

**Datasets**:
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952

**World / spatial reasoning models**:
- SAM 2: https://arxiv.org/abs/2408.00714
- Grounding DINO: https://arxiv.org/abs/2303.05499
- Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/

希望这个讲解 build 了你 对 AVP 的 intuition。如果你想 drill-down 到某一个部分 (比如 flow-matching action expert 的具体 formulation,或者 visual primitive decoder 的 tokenization scheme),告诉我,我可以再展开。
