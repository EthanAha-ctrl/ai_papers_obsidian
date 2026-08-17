---
source_pdf: RoboMME Benchmarking and Understanding.pdf
paper_sha256: 2bb260e6d5840ece4d1c350caef32f0d8908c32e774905ce7e0b58dfad97beb4
processed_at: '2026-08-12T01:17:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 RoboMME

Andrej，我把这篇 paper 当成一个故事讲给你听，少绕公式，多讲 intuition。

---

## 这 paper 在干嘛

现在的 robot policy，比如 π0.5，本质上是"看一眼当前画面 → 输出动作"。你给它一个任务"把三个蓝 cube 放进 bin"，它其实记不住自己已经放了几个。如果中间 cube 被遮挡一下，它就懵了。

**Memory 在 robotics 里一直是个 hand-wavy 的概念**。很多人加 memory，但每个人用自己的 task、自己的 backbone、自己的 eval protocol，然后说自己方法好。这 field 缺一个公平的擂台。

RoboMME 干了两件事：

1. **造了一个擂台**：16 个 task，每个 task 都故意设计成"光看当前画面不够用，必须记住过去"的类型
2. **在同一个 backbone (π0.5) 上，系统比较了 14 种 memory 方案**，看哪种在哪种 task 上管用

这就像做 ablation study，但把 ablation 做成了一个 benchmark。

Project page: https://robomme.github.io/

---

## Task 怎么设计的——四种"记不住就做不对"

作者从认知科学借了四个 memory 类型，对应四个 task suite：

### Counting（计数）—— 需要记住"我做到第几步了"

最典型的是 **PickXTimes**：让你"pick 一个 green cube 放到 target 上，重复 3 次，按 button 停止"。

问题在哪？你每次 pick 完放下后，画面看起来**一模一样**——桌上一个 green cube，一个 target。如果你没记住"我已经做了 2 次"，你就不知道这次是第 3 次还是第 1 次。

还有 **StopCube**：一个 cube 在一条线上来回 oscillate，你要在它"第 N 次经过 target"时按 button。这 task 不光要 count，还要 time-critical——按早了按晚了都不行。

### Permanence（持久性）—— 需要记住"东西原来在哪"

**VideoUnmask**：给你看一段 video，里面几个 cube 被容器盖住。然后让你揭开藏着 green cube 的那个容器。

你执行的时候，所有 cube 都被盖着，你**只能靠 memory** 知道哪个容器下面是 green。

**ButtonUnmaskSwap** 更狠：你按 button 的时候 cube 被 mask，而且容器之间还会 swap 位置。你得 track 这个 swap。

### Reference（指代）—— 需要记住"我说的是哪个"

**PickHighlight**：按 button 的时候，某些 cube 会被白光短暂 highlight 一下。光消失后，你要 pick 那些 highlight 过的 cube。

**VideoPlaceOrder** 最长（平均 1134 步）：看一段长 video，cube 被依次放到 4 个 target 上。然后 language instruction 说"把 red cube 放到 video 里的第三个 target"。你得记住"第三个 target 是哪个"。

### Imitation（模仿）—— 需要记住"demo 是怎么做的"

**MoveCube**：video 里 demo 用三种方式之一把 cube 移到 target——hooking（用钩子）、pushing（用夹爪推）、pick-and-place（直接抓）。你要复现相同方式。

**PatternLock**：video 里机器人拿着 stick 在 N×N grid 上画了一条连续轨迹，你要 retrace 同样的路径。

**RouteStick**：在障碍物之间 navigate，要匹配 demo 的 circling direction（顺时针还是逆时针绕障碍）。

---

## Benchmark 的关键 trick：强制 non-Markovian

很多 prior benchmark 号称测 memory，其实你光看当前画面就能做——因为 object state 在线更新，你看到 cube 在 target 上就知道"已经放过了"。

RoboMME 的设计哲学是：**让 current observation 故意不充分**。

手段包括：
- **Occlusion**：cube 被 mask 掉，你看不到
- **Video-conditioned reference**：关键信息只在过去 video frames 里，现在没了
- **Counting**：信息只在过去 actions 里，当前画面不告诉你做了几次
- **Swap**：东西位置变了，你得记住"它原来在哪"

这造出来的是真 non-Markovian task，不是装的。

---

## 14 种 memory 方案——三个维度的选择

所有方案都基于同一个 π0.5 backbone，memory budget 统一 512 tokens。变的是"memory 用什么表示"和"memory 怎么塞进 model"。

### 维度一：Memory 用什么表示

**Symbolic（符号化）**：把过去发生的事用 language 总结成 subgoal。

比如"pick up the green cube at [63, 152]"。这就像人做完一步在脑子里记一笔："好，我已经放了 2 个 blue cube"。

具体用 VLM (Qwen3-VL 或 Gemini) 看当前画面 + 之前的 subgoal 历史，预测下一个 subgoal。

**Perceptual（感知化）**：保留过去的 visual tokens。

两种选法：
- **FrameSamp**：均匀采样过去 32 帧，每帧 pool 到 16 tokens
- **TokenDrop**：用 RGB diff 找"变化大的 patch"，选 top-K

这就像人保留"过去的画面残影"，需要细节的时候用这个。

**Recurrent（循环化）**：把过去压缩成一个固定大小的 latent state。

两种选法：
- **RMT**：维护 B 个 memory slots，每来一帧用 attention 更新 slots
- **TTT**：用 fast weights，每帧 online SGD 更新一个小网络

这像人脑的"工作记忆"，容量有限但持续在线更新。

### 维度二：Memory 怎么塞进 model

**Memory-as-Context**：把 memory tokens 拼到输入前面，一起进 VLM expert。

最简单，不引入新参数。但改变了 input distribution，可能干扰预训练。

**Memory-as-Modulator**：用 AdaLN 把 memory 注入 action expert 的每一层。

具体做法：action expert 每层 MLP 之前，先用 cross-attention 从 memory 提取信息，投影成 scale/shift 参数，调制 action features。

**关键洞察**：π0.5 的 action expert 本来就用 AdaLN-Zero 接收 denoising timestep τ 的调制。Modulator **复用这个现成接口**，把 memory 当成"另一种 condition signal"。新参数只有 80M，surgical 改动，不破坏预训练 representation。

**Memory-as-Expert**：加一个独立的 memory expert（18 层 transformer，190M params），和 VLM expert 平行。

Action expert 同时 attend VLM features 和 memory features。但 VLM expert 和 memory expert **互不 attend**，各算各的。

这像 mixture-of-experts，给 memory 独立 capacity，但训练成本高。

---

## 实验结果——哪些方案赢，哪些输

### 总赢家：FrameSamp + Modulator (44.51%)

Perceptual memory + AdaLN modulation。性价比最高，overall 最强。

### 几个意外发现

**1. 没有 silver bullet**

不同 task 最优方案不同：
- **Counting task**（PickXTimes, BinFill）：Symbolic 最好。因为"做了几次"是离散事件，language subgoal 直接告诉你下一步干嘛最 efficient
- **Motion task**（PatternLock, RouteStick）：Perceptual 最好。因为要复现连续轨迹，visual tokens 信息密度高
- **Dynamic scene change**（ButtonUnmaskSwap）：MemER 最好。因为要持续维护 keyframe images 在线应对 swap
- **Time-critical**（StopCube）：Perceptual 最好。Symbolic 太离散，没法精确捕捉"cube 到 target 的那一瞬间"

**2. Symbolic 的瓶颈在 subgoal generator，不在 policy**

GroundSG + Oracle（用 simulator 给的 ground truth subgoal）达到 84.08%。换成 QwenVL 预测 subgoal，掉到 32.70%。换成 Gemini (prompt only)，掉到 11.56%。

**VLM subgoal 预测准不准，直接决定 symbolic 方案生死**。Policy 本身能力够，喂对 subgoal 就能做对。

**3. Recurrent methods 全面拉胯**

TTT 和 RMT 都只有 20% 出头。Paper 解释是浅层 recurrent layer fine-tune π0.5 训练不稳定。

我的猜想更深一层：π0.5 是 transformer-based，预训练里没有 recurrent inductive bias。RMT/TTT 是 from-scratch 的轻量模块，学不到长程依赖。如果换成 Mamba-based backbone，recurrent memory 可能翻身。

**4. TokenDrop < FrameSamp**

TokenDrop 用 RGB diff 选 informative patches，看起来聪明，但 StopCube 上惨败（5.33% vs 42.00%）。

原因：StopCube 需要判断 cube 和 target 的距离，这要 holistic view。TokenDrop 把 spatial context 打散，丢失了"全局空间关系"。FrameSamp 保留每帧完整（虽然 pool 到 4×4），全局信息还在。

**5. Modulator > Context > Expert**

Modulator 整体最强。Context 次之。Expert 最弱（除了 recurrent 那几个）。

我的理解：Modulator 是 surgical 改动，不动 VLM 主干，只在 action expert 每层插一个 lightweight AdaLN。预训练 representation 保留最好。Context 改变 input distribution。Expert 引入太多新参数（190M），fine-tune 学不充分。

---

## 效率-性能 trade-off

Memory budget 从 64 → 512 tokens：

- **FrameSamp+Modul**：性能稳步涨，计算开销温和（大部分计算在 visual tokens，不在 memory 整合）
- **GroundSG+QwenVL**：~3× π0.5 TFLOPs（VLM 推理贵）
- **MemER**：~5× π0.5（VLM + keyframe 维护）

实战中可以用 caching 减开销——subgoal 不每步都预测，reuse 几步。

---

## 人类也做不好

人类平均 90.5%，但 PatternLock (84%)、RouteStick (86%)、StopCube (78%)、SwingXTimes (80%) 都 fail。

Paper 解释：这些 task 要么 trajectory 长容易忘，要么 time-critical 按不准。**连人都 lose track**，说明这些 task 本质上对 memory 要求高，不是"模型不够强"的问题。

这给 benchmark 难度背书。

---

## Real-world 验证

在真机器人上跑了 4 个 task（PutFruits / TrackCube / RepickBlock / DrawPattern），趋势一致：

- PutFruits（counting 类）：Symbolic 强 (9/10)
- DrawPattern（motion 类）：Perceptual 强 (8/10)
- 总分：FrameSamp+Modul (25/40) > GroundSG+QwenVL (19/40) > π0.5 (4/40)

Sim→real 的**方法趋势**是 transferable 的，不只是 sim artifact。

---

## 大图景：这 paper 告诉我们什么

### 1. Memory 设计是 task-dependent 的

没有 universal best memory。Counting task 想要 symbolic，motion task 想要 perceptual，dynamic scene 想要 hybrid。

这解释了为什么 prior work 各说各话——每个人在自己 task 上 design memory，当然自己方法好。放到 RoboMME 上比，就露馅了。

### 2. Hybrid 是未来方向

MemER (42.38%) 靠"keyframe images + symbolic subgoals"的 hybrid 拿了第二。如果能把 perceptual + symbolic + recurrent 结合，可能突破 44.51% 的 SOTA。

这跟 LLM agent memory 领域（Mem0, A-Mem, Zep）的 hybrid 思路一致——episodic store + summary + reflection。

### 3. Pretrained backbone 的"标准接口"

AdaLN 是 pre-trained transformer 接收外部信号的标准接口。DiT 用它接 class label，Stable Diffusion 3 用它接 text，RoboMME 用它接 memory。

**直觉**：如果你想往 pre-trained model 里塞新 signal，AdaLN modulation 是最 surgical 的方式。不改主干，不破坏 representation，只加一个轻量 modulation layer。

### 4. Memory representation 要和 backbone inductive bias 匹配

Transformer backbone 配 perceptual/symbolic 好。SSM backbone 可能配 recurrent 好。

这给未来工作指方向：**不要光研究 memory representation，要研究"memory × backbone"的协同设计**。

---

## 一句话总结

RoboMME 是 robotics memory 领域第一个"公平擂台 + 系统比较"的工作。核心结论：**memory 设计是 task-dependent 的，perceptual + AdaLN-modulator 是 best general-purpose choice，symbolic 在 counting 上更高效，hybrid 是未来方向**。这 field 终于从"讲故事"走向"做 controlled study"。

---

## 关键 links

- Project: https://robomme.github.io/
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://proceedings.mlr.press/v305/black25a.html
- MemER: https://arxiv.org/abs/2510.20328
- DiT (AdaLN-Zero): https://arxiv.org/abs/2212.09748
- RMT: https://arxiv.org/abs/2304.11062
- TTT: https://arxiv.org/abs/2505.23884
- ManiSkill: https://github.com/haosulab/ManiSkill

需要我展开哪一块？比如 AdaLN 实现细节、某个具体 task 的 design 巧思、或者跟别的 VLA backbone 怎么对照，告诉我就行。

---

# RoboMME：为 Robotic Generalist Policies 构建一个 Memory 的系统化 Benchmark

Andrej, 这篇 paper 我读得很过瘾——它做了一件 field 里长期缺位的事：在 controlled 设定下系统比较 memory representations × integration mechanisms，把"为什么需要 memory"和"哪种 memory 在哪种 task 上有用"两件事拆开来看。下面我把技术骨架拆给你看，顺便聊一些直觉联想。

---

## 1. Why this paper matters：Benchmark + Controlled Study 的双层贡献

大部分 prior work（ContextVLA [22], MemER [36], MITL [47], RoboMamba [28]）各自设计自己的 memory 机制，在不同的 backbone 上、不同的 task 上、不同的 evaluation protocol 下做实验。这种不可比性让 field 停留在"看谁讲的故事漂亮"的阶段。

RoboMME 同时做了两件事：
- **Benchmark side**: 16 个 long-horizon、非 Markovian 的 task，1,600 demos、770k timesteps，覆盖四种认知 memory 类型
- **Method side**: 14 个基于同一 π0.5 backbone 的 MME-VLA variants，3 × 3 网格的 representation × integration

这种 "fix backbone, fix data, vary one axis" 的研究范式才是能给出科学结论的方式。

Project page: https://robomme.github.io/

---

## 2. Benchmark 设计：四类 cognitive memory → 四个 task suite

### 2.1 认知科学 motivation

作者从 Atkinson-Shifrin 的人记忆模型 [1] 出发，把 long-term memory 分成 declarative (episodic + semantic) 和 procedural。 episodic 进一步分成 temporal / spatial / object 三子类，对应 "when / where / what"，procedural 对应 "how"。

| Memory type | 对应认知维度 | RoboMME suite |
|---|---|---|
| Temporal | when | Counting |
| Spatial | where | Permanence |
| Object | what | Reference |
| Procedural | how | Imitation |

### 2.2 每个 task 的设计意图

**Counting suite**——temporal memory:
- **PickXTimes** (538 steps avg): 重复 pick-and-place 指定次数，按 button 终止。需要 count
- **BinFill** (604): cube 可能 streaming 出现，要把指定数量放 bin。需要 count + 对 dynamic scene 的应对
- **SwingXTimes** (435): swing 一个 cube 在两 target 间指定 cycle 数
- **StopCube** (317): 移动 cube 第 N 次到 target 时按 button。**Time-critical**

**Permanence suite**——spatial memory:
- **VideoUnmask** (217): 看 video 中 cubes 被 mask，事后揭开
- **ButtonUnmask** (267): 按 button 时 cubes 被 mask，事后揭开
- **VideoUnmaskSwap** (348): video + container 位置 swap
- **ButtonUnmaskSwap** (400): button + swap。最难的 spatial task

**Reference suite**——object memory:
- **PickHighlight** (346): 按 button 时某些 cube 短暂 highlight，事后 pick
- **VideoRepick** (687): 看 video 中 cube 被 pick，重复 pick 同一个 cube
- **VideoPlaceButton** (974): 长 video 中 interleaved placement + button press，根据 language 中 temporal reference 找 target
- **VideoPlaceOrder** (1134): 最长。根据 "third target" 等 ordinal reference 找 target

**Imitation suite**——procedural memory:
- **MoveCube** (394): 复现 demo 的 manipulation strategy (hooking / pushing / pick-and-place)
- **InsertPeg** (479): 复现 grasp end (head/tail) 和 insert side (left/right)
- **PatternLock** (208): N×N grid 上复现连续轨迹
- **RouteStick** (370): obstacles 间 navigate，匹配 circling direction (clockwise/counter-clockwise)

### 2.3 与 prior benchmarks 对比（Table 2）

| Dataset | Non-Markov | Partial Obs. | Dynamic | Vid | Lang | Subgoal | Keyframe | Memory Types | Avg Steps |
|---|---|---|---|---|---|---|---|---|---|
| RLBench18 | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | T | 137 |
| CALVIN | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | T | 584 |
| LIBERO | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | T | 162 |
| MemoryBench | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | T+S | 312 |
| MIKASA-robo | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | T+S+O | 72 |
| **RoboMME** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **T+S+O+P** | 481 |

RoboMME 是唯一一个四类 memory 都覆盖、且同时有 video-conditioned + language-conditioned + non-Markov + dynamic scene 的。

---

## 3. π0.5 Backbone：MME-VLA suite 的共同基底

要理解 14 个 variants，必须先吃透 π0.5。我把它拆给你看：

### 3.1 Tokenization (Eq 1-2)

```
ℓ = Tok_text(l) ∈ R^(N_ℓ × d)        # language tokens
o_t = Tok_img(I_t) ∈ R^(N_o × d)       # image tokens
```

- `l`: task instruction string
- `I_t = (I_t^1, I_t^2, ...)`: 多视角 RGB observations (front + wrist camera, 各 256×256)
- `d`: token embedding dimension (在实现里是 1024)
- `N_ℓ`, `N_o`: language 和 visual token 总数

### 3.2 Blockwise Causal Attention (Eq 3)

```
y = BlockAttn(x_1, x_2, ..., x_G)
```

每个 `x_i ∈ R^(n_i × d)` 是一个 block。Bidirectional within block，causal across blocks。block i 可以 attend blocks 1:i，不能 attend 未来 blocks。返回的是最后一个 block G 的更新后表示 `y ∈ R^(n_G × d)`。

这个设计让 π0.5 可以把 image 当一个 bidirectional block 处理（图像内部全双向），同时保持 action chunk 的自回归 causality。

### 3.3 VLM expert (Eq 4-6)

输入拼接:
```
u_t^0 = [o_t ; ℓ]    # 沿 sequence dimension concat
```

每层 (k = 1,...,L):
```
ũ_t^k = u_t^(k-1) + BlockAttn^k(Norm_vlm^k(u_t^(k-1)))
u_t^k = ũ_t^k + MLP_vlm^k(Norm_vlm^k(ũ_t^k))
```

只有单个 input block (因为 image + text 一起进)。Norm 用 RMSNorm。Attention 和 MLP 用各自的 norm 层（分开参数化）。

注意：作者在 paper 里说他们简化了原 π0.5——只用 VLM 当 feature extractor，不像原版还用它生成 low-level commands 做 task decomposition。

### 3.4 Action expert (Eq 7-8)——**这是 AdaLN-Zero 的核心**

设 `s_t^0 ∈ R^(N_s × d)` 是 action input (注入 noise)。每层:

```
s̃_t^k = s_t^(k-1) + g_τ ⊙ BlockAttn^k(Norm_vlm^k(u_t^(k-1)), Norm_act^k(s_t^(k-1), τ))
s_t^k  = s̃_t^k + g̃_τ ⊙ MLP_act^k(Norm_act^k(s̃_t^k, τ))
```

- `τ`: denoising timestep embedding (因为 π0.5 是 flow-matching)
- `Norm(·, τ)`: RMSNorm conditioned on τ 的 embedding
- `g_τ`, `g̃_τ`: gating functions conditioned on τ
- `⊙`: feature-wise multiplication

这套是 **AdaLN-Zero** [32, DiT 那篇]，初始化时 gate 输出 0 → 训练初期 action expert 是 identity → 等于在 action expert 侧做 conditional modulation，让 τ 控制 action 分布的 scale/shift。这非常重要，因为后面 memory-as-modulator 就是**借用**这套机制把 memory 也注入进来。

### 3.5 Flow matching objective (Eq 10-13)

预测 H 步 action chunk:
```
A_t = [a_t, a_(t+1), ..., a_(t+H-1)] ∈ R^(H×D)
```

采样:
```
A_t^data ~ p_data
A_t^noise ~ N(0, I)
τ ~ U(0, 1)
A_t^τ = (1-τ) A_t^data + τ A_t^noise     # 线性插值
dA_t^τ/dτ = A_t^noise - A_t^data          # constant velocity
```

训练 velocity field:
```
L_FM = E[ || v_θ(A_t^τ, τ, h_t) - (A_t^noise - A_t^data) ||² ]
```

`h_t` 是 context features（从 VLM expert 出来的）。这比 DDPM 那种离散 timestep 的去噪要简洁，linear interpolation 的 transport direction 是常数。

参考: 原始 π0 paper: https://arxiv.org/abs/2410.24164 ; π0.5 in CoRL 2025: https://proceedings.mlr.press/v305/black25a.html

---

## 4. MME-VLA Suite：3 × 3 网格的系统比较

### 4.1 Memory Representations

所有 memory-based 模型共享一个固定 memory budget `B = 512 tokens`。Memory 接口统一:
```
M_t ∈ R^(B × d)
```

#### 4.1.1 Symbolic Memory

```
M_t^sym = Tok_text(g(H_t)) ∈ R^(B × d)
```

`H_t = {(I_τ, a_τ)}_{τ<t}` 是 trajectory history。`g(·)` 是 VLM，给定 current image + prior subgoals，预测 next subgoal。两种 instantiation:

- **SimpleSG**: "pick up the green cube"
- **GroundSG**: "pick up the green cube at [63, 152]"——加了 grounding coordinates

这个对照很有意思：在 paper 里 GroundSG 几乎全面碾压 SimpleSG，**说明 grounding 信息让 symbolic 从"指挥"变成"指挥+点地图"**，policy 不用再做一遍 visual grounding。

VLM 选项: Gemini-2.5-Pro (prompt only) / Qwen3-VL-4B (fine-tuned) / Oracle (simulator ground truth)。

#### 4.1.2 Perceptual Memory

```
M_t^perc = Select({o_τ}_{τ≤t}, B)
```

两种 select 策略:

**FrameSamp (Eq 14-15)**: 均匀采样整帧
```
S_t = EvenSample({0,...,t}; N)
M_t^perc = [MaxPool(o_τ)]_{τ∈S_t}
```

`MaxPool` 把每帧从 SigLIP 的 16×16 = 256 tokens 压到 P=4×4=16 tokens per view。V 个 view 时 N = B / (P·V) = 512 / (16·1) = 32 帧。

**TokenDrop (Eq 16-18)**: 选 informative 的 spatial patches
```
s_(τ,i) = J(õ_τ,i, õ_(τ-K),i)         # temporal salience
K_t = TopK({(τ,i)}, B; s_(τ,i))
M_t^perc = [õ_(τ,i)]_{(τ,i)∈K_t}
```

- `i`: 索引 pooled spatial location
- `K`: 固定 temporal stride (= 8)
- `J(·,·)`: 平均 RGB patch difference，衡量 motion / appearance change

用 **M-ROPE** (Multi-modal Rotary Position Embedding) 给每个 token 一个 3D index `(time, height, width)` 保留时空结构。

直觉上 FrameSamp 保留了完整 spatial context（每帧 16 tokens 都在），TokenDrop 用 RGB diff 选 informative patches，但**会丢失全局 spatial layout**——这对需要 holistic view 的任务（如 StopCube，需要判断 cube 距 target 多远）不利。实验也证实了这一点。

参考 token dropping: TimeChat-Online [44] https://dl.acm.org/doi/10.1145/3757578

#### 4.1.3 Recurrent Memory

```
S_t = f_mem(S_(t-1), o_t)     # latent state online update
M_t^recu = S_t (or its last B tokens)
```

两种 instantiation:

**RMT (Eq 19)** [6, 7]: 持久 memory slots `S_init ∈ R^(B×d)`
```
o_t = [o_t^(1), ..., o_t^(M)],  o_t^(m) ∈ R^(N_C × d)    # 分段
S_t^(m) = Transformer(Q=S_t^(m-1), K=[o_t^(m); S_t^(m-1)], V=[o_t^(m); S_t^(m-1)])
S_t^(0) = S_init
M_t^recu = S_t^(M)
```

memory slots 当 queries，观察和上一时刻 memory 当 keys/values。segments 串行更新。

**TTT (Eq 20)** [37, 46]: 用 fast weights 而非 explicit tokens
```
W_t = W_(t-1) - η ∇_W ℓ_aux(W_(t-1); o_t)    # 在线 SGD 更新 fast weights
y_t = f(o_t; W_t)                              # 立即预测
M_t^recu = y_(-B:)                             # 取最后 B 个 token
```

`ℓ_aux` 是 self-supervised objective。TTT 把 memory 隐式存到参数空间，而不是 explicit tokens。

直觉: TTT/RMT 都依赖 recurrence 的稳定性，但 π0.5 是从大规模预训练来的 transformer，浅层 recurrent module (paper 里只 1 layer RMT，TTT 是 lightweight FFN) 难以学到稳定的长程依赖——这是 paper 里 recurrent 表现最差的原因之一。

参考 RMT: https://arxiv.org/abs/2304.11062 ; TTT: https://arxiv.org/abs/2505.23884

### 4.2 Integration Mechanisms

#### 4.2.1 Memory-as-Context

```
u_t = E_vlm([M_t ; o_t ; ℓ])
```

直接把 memory tokens 拼到 VLM expert 输入前面。**不引入任何新参数**。Memory 通过 self-attention 影响所有 VLM features，进而影响 action expert。

#### 4.2.2 Memory-as-Modulator (Eq 21-24)——**论文里表现最好**

在每个 action expert 的 MLP sublayer 注入 AdaLN-style modulation:

```
r_t^k  = Attn_mod^k(Q=s̃_t^k, K=M_t, V=M_t)            # cross-attention
(γ_t^k, β_t^k) = MLP_mod^k(r_t^k)                       # project to scale & shift
ŝ_t^k = γ_t^k ⊙ Norm_mod^k(s̃_t^k) + β_t^k              # adaptive LayerNorm
s_t^k = s̃_t^k + g̃_τ ⊙ MLP_act^k(Norm_act^k(ŝ_t^k, τ))
```

- `r_t^k`: layer-wise modulation feature
- `γ, β`: scale 和 shift 参数，从 memory-aware representation 投影而来
- `MLP_mod` 初始化为 γ=1, β=0 (identity modulation at fine-tune start)

这是个非常聪明的"借用 AdaLN-Zero 接口"的设计——既然 π0.5 的 action expert 已经用 τ 做 AdaLN 调制，那就**复用同一个调制接口**注入 memory。新参数只有 80M (一个 lightweight attention module，1 key-value head, head dim 256)。

为什么它最强? Paper 里解释: "lightweight, feature-wise conditioning largely preserves the original π0.5 architecture"——没动 VLM expert，没动 action expert 的主干，只是在每个 MLP 之前做一次 memory-conditioned AdaLN。预训练 representation 没被破坏。

#### 4.2.3 Memory-as-Expert (Eq 25-27)

引入独立的 memory expert E_mem (18-layer transformer, width 1024, MLP dim 2048, ~190M params):

```
v_t^0 = M_t
ṽ_t^k = v_t^(k-1) + BlockAttn^k(Norm_mem^k(v_t^(k-1)))
v_t^k = ṽ_t^k + MLP_mem^k(Norm_mem^k(ṽ_t^k))

# 然后 action expert attention 变成:
s̃_t^k = s_t^(k-1) + g_τ ⊙ BlockAttn^k(
    Norm_mem^k(v_t^(k-1)),
    Norm_vlm^k(u_t^(k-1)),
    Norm_act^k(s_t^(k-1), τ)
)
```

**Key 设计**: VLM expert 和 memory expert **不互相 attend** (都只用 self-attention)。只有 action expert 同时 attend 两者。

这个 block-wise causal pattern:
- Block 1 (VLM expert features): self-attention only
- Block 2 (Memory expert features): self-attention only
- Block 3 (Action features): attends blocks 1, 2, 3 (causal across)

目的: 限制 interference，preserve VLM pretrained behavior，给 memory 独立 capacity。

直觉: 这像是 mixture-of-experts 但 top-k=1 的固定 routing。Memory expert 专门处理 memory，不被 VLM 的语言/视觉任务带偏。但实验里它不如 Modulator，可能因为: (a) 190M 新参数 fine-tuning 学不充分; (b) Memory expert 是 from scratch 初始化，没有预训练 prior; (c) Block-wise causal 让 memory 和 VLM 不能 fuse 早期 features。

---

## 5. 实验结果：关键 takeaways

### 5.1 Main Results (Table 3)

挑几个亮点数据 (success rate %):

| Method | AVG | PickXTimes | StopCube | PatternLock | RouteStick | PickHighlight |
|---|---|---|---|---|---|---|
| Human | 90.50 | 100.0 | 78.00 | 84.00 | 86.00 | 92.00 |
| GroundSG+Oracle | 84.08 | 100.0 | 49.67 | 97.00 | 55.56 | 83.33 |
| GroundSG+QwenVL | 32.70 | 92.67 | 0.00 | 6.67 | 6.00 | 15.11 |
| **FrameSamp+Modul** | **44.51** | 87.33 | **42.00** | 53.56 | **66.67** | 22.89 |
| FrameSamp+Context | 30.68 | 72.00 | 13.67 | 15.22 | 19.67 | 17.67 |
| TokenDrop+Modul | 38.04 | 83.56 | 5.33 | 32.44 | 51.56 | 21.33 |
| TTT+Modul | 21.96 | 65.11 | 2.11 | 3.56 | 7.00 | 14.56 |
| RMT+Modul | 20.17 | 60.78 | 4.67 | 3.78 | 8.00 | 17.11 |
| π0.5 (baseline) | 17.93 | 42.89 | 6.67 | 2.89 | 4.67 | 11.33 |
| MemER | 42.38 | 79.33 | 0.00 | 16.67 | 12.00 | 70.67 |
| SAM2Act+ | 21.37 | 76.00 | 0.00 | 0.00 | 0.00 | 17.33 |

几个观察:

**1. FrameSamp+Modul 是 overall winner (44.51%)**，超过 MemER (42.38%)。Modulator 整合机制最强。

**2. Perceptual > Symbolic > Recurrent** (non-oracle):
- Perceptual best: 44.51 (FrameSamp+Modul)
- Symbolic best: 32.70 (GroundSG+QwenVL)
- Recurrent best: 22.35 (TTT+Expert)

**3. TokenDrop < FrameSamp**: aggressive spatial pooling 损失全局 context，尤其伤 StopCube (需要判断 cube-target 距离)。

**4. Recurrent methods 全面拉胯**，paper 解释是 fine-tuning π0.5 加 shallow recurrent layer 训练不稳定。这可能不是 recurrent 概念本身的问题，而是 π0.5 的 pretrain 没给 recurrent 接口留 capacity。

**5. Symbolic memory 在 ground truth oracle 下很强 (84.08%)**，但 QwenVL 实际预测 subgoal 后降到 32.70%。**说明 symbolic 的瓶颈在 subgoal 预测，不在 policy**。Gemini (prompt only) 更差，domain shift 严重。

### 5.2 Task dependency (Figure 3, Table 10-11)

按 functional requirement 重分组 (perceptual symbolic 各取最强代表):

| Category | Best | Score |
|---|---|---|
| Motion-Centric (PatternLock, RouteStick, SwingXTimes, InsertPeg) | FrameSamp+Modul | 54.95 |
| Time-Sensitive (StopCube) | FrameSamp+Modul | 42.00 |
| Short-Horizon Video Reasoning | GroundSG+QwenVL & MemER | 48.22 |
| Long-Horizon Video Reasoning | FrameSamp+Modul | 46.00 |
| Dynamic Online Scene-Change | MemER | 54.67 |
| Event-Salient (PickXTimes, BinFill, MoveCube) | SimpleSG+QwenVL | 84.96 |

直觉解读:
- **Motion-Centric**: 需要持续 trajectory tracking，visual tokens 比 language subgoal 信息更密 → perceptual 强
- **Time-Sensitive**: 需要精确 temporal coordination，subgoal 太离散 → perceptual 强
- **Short-Horizon Video Reasoning**: grounding 短 video 内 object → symbolic (GroundSG with coordinates) 强
- **Dynamic Scene-Change**: 需要 keyframe image 持续在线维护 → MemER 的 hybrid 设计强
- **Event-Salient**: 离散 salient events 标记 subtask 完成 → symbolic 直接给出 next subgoal 最 efficient

### 5.3 Efficiency (Figure 4, Q5)

Memory budget 64→512 的 scaling:

- **FrameSamp+Modul**: performance 稳步上升，TFLOPs 增长温和。原因: 主要计算在 visual tokens，memory integration overhead 小
- **GroundSG+QwenVL**: ~3× π0.5 的 TFLOPs (VLM 推理)
- **MemER**: ~5× π0.5 (需要 VLM + keyframe 维护)

实战意义: perceptual + modulator 是性价比最高的。

### 5.4 Human study (Q3, Table 8)

人类 90.5% average，但仍在 PatternLock (84), RouteStick (86), StopCube (78), SwingXTimes (80) 上 fail。Paper 解释这些 task 要么 trajectory 长，要么 time-critical，人类也会 "lose track"。

这给 benchmark 的难度背书: 不是"模型不够强"，是这些 task 本质上对 memory 要求高。

### 5.5 Real-world transfer (Table 4)

4 个 real-world tasks (PutFruits / TrackCube / RepickBlock / DrawPattern)，350 demos:

| Method | PutFruits | TrackCube | RepickBlock | DrawPattern | Total |
|---|---|---|---|---|---|
| π0.5 | 2/10 | 1/10 | 1/10 | 0/10 | 4/40 |
| GroundSG+QwenVL | 9/10 | 3/10 | 5/10 | 2/10 | 19/40 |
| FrameSamp+Modul | 6/10 | 5/10 | 6/10 | 8/10 | 25/40 |

**Trends 一致**: symbolic 强在 counting (PutFruits 9/10)，perceptual 强在 motion (DrawPattern 8/10)。说明 sim→real 的方法趋势是 transferable 的，不只是 sim artifact。

---

## 6. 联想与 Open Questions

### 6.1 Memory 的 hierarchy of abstractions

Paper 里的三种 representation 可以看作一个 abstraction spectrum:

```
Recurrent (最压缩)  ──  Perceptual (中等压缩)  ──  Symbolic (最抽象)
   latent state          visual tokens              language
```

直觉: 越抽象的 representation 越适合 high-level reasoning (count, decide next subgoal)，越保留 detail 的越适合 low-level motor imitation。这跟人类大脑的 dual-process 有点像——declarative memory 用语义，procedural 用 sensorimotor traces。

但 paper 也指出 **GroundSG+Oracle (84%) 远超所有 non-oracle**，说明 symbolic 的瓶颈是 subgoal generator，不是 policy。如果有个 perfect subgoal oracle，symbolic 会很强。这暗示未来工作可以专注提升 VLM subgoal 预测，或做 self-consistency / verifier-based subgoal refinement。

### 6.2 Modulator 为什么强？一个猜想

Memory-as-Modulator 用 AdaLN 注入 memory。这套机制在 DiT [32]、Stable Diffusion 3 里已经被验证——condition 信号 (class label, text) 通过 AdaLN 调制 diffusion transformer。RoboMME 把 memory 当成"另一种 condition signal"借用同一接口。

这给一个直觉: **AdaLN modulation 是 pre-trained transformer 的"标准 foreign key"**。任何外部信号想注入而 minimally disrupt pretrained weights，AdaLN 都是好选择。Memory-as-Context 会改变 input distribution，Memory-as-Expert 引入大量新参数，Modulator 是最 surgical 的。

### 6.3 Recurrent 为什么弱？另一个猜想

Paper 说 "fine-tuning π0.5 with shallow recurrent layers leads to unstable training"。但我认为深层原因是: π0.5 的 VLM expert 是 from-pretrained SigLIP2 + LLM backbone，没有 recurrent inductive bias。RMT/TTT 是 from scratch 的轻量 module，只能学到 "shallow summaries"，无法 capture 长程 temporal dynamics。

对比: Mamba [18] / RoboMamba [28] / MITL [47] 这些 state-space model 是 from scratch 训练的，有 recurrent prior。如果 RoboMME 在 Mamba-based VLA 上跑 recurrent memory，结果可能完全不同。

这给一个 open question: **memory 的设计应该和 backbone 的 inductive bias 匹配**。Transformer 用 perceptual/symbolic 好，SSM 用 recurrent 好。

### 6.4 与 LLM agent memory 工作的串联

RoboMME 的 symbolic ↔ perceptual 二分法，跟 LLM agent 领域的 "episodic memory (raw traces) vs. semantic memory (summarized facts)" 二分法完全对应。

- Mem0, A-Mem, Zep 这种 LLM agent memory framework 都是 hybrid: episodic store + summary + reflection
- MemER 在 RoboMME 里之所以强 (42.38%)，也是因为 hybrid: keyframe images (episodic) + symbolic subgoals (semantic)
- 这暗示 RoboMME 上 hybrid 是 promising direction

参考 MemER: https://arxiv.org/abs/2510.20328

### 6.5 跟 video-LLM 的关联

Perceptual memory 用 M-ROPE 给每个 video token 一个 3D index。这跟 VideoLLM-online [10], Long-context Diffusion Policies [39] 是同一思路。

Long-context Diffusion Policies by Torne et al.: https://arxiv.org/abs/2505.09561

一个开放方向: 直接用 video-LLM 处理 long history，把 robot action generation 当成 video-LLM 的一个下游 head。ContextVLA [22] 已经走这条路，但 RoboMME 没评估它。

### 6.6 关于 "非 Markovian" 的严格性

RoboMME 的 task 设计强调: identical observations can arise from different histories yet require different actions。这是真 non-Markovian。

但注意: 在很多 imitation learning setup 里，如果 state 是 fully observable（包括 robot proprioception + 所有 object positions），那 task 还是 Markovian 的——只要 policy 能 access full state。RoboMME 通过 occlusion (cube 被 mask)、video-conditioned reference (信息只在过去 frames)、counting (信息只在过去 actions) 等手段，强制让 current observation 不充分。

这是设计 benchmark 时的关键 trick: **要让 history 真的不可省略**，否则 policy 会学一个"忽略 memory"的 shortcut。

### 6.7 Failure modes 的观察

Paper Section 5.2 Q2 提到 GroundSG+Oracle 在 cluttered scene 里仍 fail (BinFill, PickHighlight)，原因是 "manipulating wrong objects and causing unintended collisions"。这说明: **symbolic subgoal 给的是 "what to do"，但执行层面的 perception (which exact pixel to grasp) 还是 policy 的活**。Grounding coordinate 帮助有限，因为 cube 位置在 clutter 里 ambiguous。

这暗示: **subgoal + low-level visuomotor policy 是 decoupled 的两层**。Symbolic 解决 "what"，perceptual 解决 "where + how"。这就是为什么 hybrid (MemER, GroundSG+modulator 混合) 在 long-horizon 上仍有 headroom。

### 6.8 Limitations 和 Future work

Paper 自己列了:
- 只 tabletop，没 mobile manipulation
- 只评估 π0.5 一个 backbone
- MemoryVLA [33] 没成功复现
- 没评估 memory-bank 方法

我会加几个:
- **Memory budget 固定 512**: 没探索 adaptive memory (按 task complexity 动态调整)
- **没有 multi-modal memory**: 比如把 proprioception、audio、force 都纳入 memory
- **没探索 cross-task memory transfer**: 在 task A 学的 memory 能 transfer 到 task B 吗
- **Pretraining for memory**: 应该有专门的 memory pretraining stage (类似 BERT 的 MLM)，而不是只在 downstream fine-tune recurrent layers

---

## 7. 公式变量总表 (便于 intuition building)

| Symbol | Meaning |
|---|---|
| `ℓ` | language tokens (instruction 编码后) |
| `o_t` | image tokens at time t (SigLIP2 编码后) |
| `d` | token embedding dimension (= 1024) |
| `N_ℓ`, `N_o`, `N_s` | language / image / action token 数 |
| `B` | memory budget (= 512 tokens) |
| `τ` | denoising timestep (in flow matching) |
| `g_τ`, `g̃_τ` | τ-conditioned gating functions (AdaLN-Zero) |
| `M_t` | memory state at time t, ∈ R^(B×d) |
| `S_t` | recurrent latent state |
| `u_t^k` | VLM expert output at layer k |
| `s_t^k` | action expert output at layer k |
| `v_t^k` | memory expert output at layer k |
| `γ_t^k`, `β_t^k` | scale & shift parameters in AdaLN |
| `H` | action chunk length (= 20 in implementation) |
| `D` | action dimension (7 for EEF, 8 for joint) |
| `η` | learning rate for TTT fast weights (= 0.01) |
| `K` | temporal stride for TokenDrop salience (= 8) |
| `P` | pooled tokens per view per frame (16 for FrameSamp, 64 for TokenDrop) |
| `V` | number of views (= 1 in main experiments) |

---

## 8. 一句话总结

RoboMME 用 cognitive science 的四类 memory 框架构造了一个真 non-Markovian 的 robotic manipulation benchmark，在固定 π0.5 backbone 上系统比较了 3 种 memory representation × 3 种 integration 机制。核心发现: **memory 的最优设计是 task-dependent 的**，perceptual + AdaLN-modulator 是 best general-purpose combination，但 symbolic 在 event-salient counting task 上更高效。这指向未来 hybrid memory 系统的方向。

---

## References / Web Links

- Project: https://robomme.github.io/
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 (CoRL 2025): https://proceedings.mlr.press/v305/black25a.html
- SAM2Act+ adaptation code: https://github.com/RoboMME/SAM2Act
- Diffusion Policy adaptation: https://github.com/RoboMME/DP
- MemoryVLA adaptation: https://github.com/RoboMME/MemoryVLA
- DiT (AdaLN-Zero): https://arxiv.org/abs/2212.09748
- RMT: https://arxiv.org/abs/2304.11062
- TTT: https://arxiv.org/abs/2505.23884
- MemER: https://arxiv.org/abs/2510.20328
- Long-context Diffusion Policies: https://arxiv.org/abs/2505.09561
- TimeChat-Online (token dropping): https://dl.acm.org/doi/10.1145/3757578
- ManiSkill: https://github.com/haosulab/ManiSkill
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2303.04137

如果你想要我更深入拆某一个 task suite 的具体 task semantics（比如 VideoPlaceButton 的 temporal reference 解析），或者 deep dive Modulator 的 AdaLN 实现细节、或对照其它 VLA backbone（如 Octo, OpenVLA）会发生什么，告诉我哪个角度，我继续展开。
