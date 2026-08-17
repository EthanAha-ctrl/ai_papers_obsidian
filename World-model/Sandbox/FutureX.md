---
source_pdf: FutureX.pdf
paper_sha256: b28220d77a8dd62cbb6560e9b990ece7618a4d37a0c6d75638d2350199c24724
processed_at: '2026-08-04T11:28:52-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 FutureX

## 一句话说清楚它在干嘛

现在的自动驾驶 model 都是 "看一眼路况，直接吐出一串方向盘操作"，跟考试时还没读完题就开始写答案一样。FutureX 给它加了个 "脑子"——遇到复杂场景先在 latent space 里 **脑补一下未来几秒会发生什么**，再决定怎么开。

---

## 它在解决什么痛点

你开车的时候，看到前面有辆大卡车挡着，你不会立刻打方向盘就冲。你会先想："我要是变道过去，旁边那车会不会加速？要是我减速，后面会不会追尾？" 这种 **"如果我这样干，世界会变成什么样"** 的 mental simulation，是老司机必备的。

但现在的 E2E driving model（UniAD、TransFuser 这些）根本没这个步骤。它们就是个 **giant feed-forward network**，sensor 进来，trajectory 出去，中间没有 thinking。结果就是：

- 前车突然倒车？撞了，因为 model 没 anticipate
- 要变道超车？只敢挪一点点，不敢果断变
- 急转弯？转得不够，直接卡在路口

根本原因：当前这一帧的 BEV feature 根本无法编码 "反事实信息"——"如果我这么做，未来会怎样"。这信息只能靠 **rollout** 拿到。

---

## 它怎么做的——三个核心 trick

### Trick 1：把 "思考" 这件事从文字搬进 latent space

之前 DriveVLM、EMMA 这些尝试过用 LLM 做 CoT，让 model 输出 "我看到前方有行人，应该减速..." 这种文字 rationale。问题在于：**这些文字和最终的方向盘控制是脱钩的**，model 说一套做一套，文字是装饰品。

FutureX 的 insight 是：CoT 的本质 **不是文字**，是 "一步一步把未来展开"。那么在 driving 里，一步 thought = 一次 latent world model 的 forward rollout。你不用写文字，直接在 latent feature 里把未来 K 步 scene 演化出来，这就是 thought。

### Trick 2：auto-think switch——什么时候该动脑子

人也不会每秒都在深思，喝口水、直道巡航这种场景用直觉就行。FutureX 学了一个小 switch network，输入当前 scene latent，输出一个 difficulty score。简单场景直接 forward 出 trajectory（叫 instant mode），复杂场景才启动 world model rollout（叫 thinking mode）。

这个 label 怎么来的？很 clever：离线先算一下 "thinking 后的 trajectory 比 thinking 前好多少"，如果相对提升 > 25%，就标成 "该 think"，否则标成 "不用 think"。然后用 BCE 训 switch。本质是学一个 **value of computation** 的 predictor——跟 OpenAI o1/o3 的 adaptive test-time compute 一回事。

实验里 FutureX-Auto 只在 75% 场景 think，剩下 25% 走 instant，但性能几乎不掉（89.2 vs 90.1），latency 平均才 9ms 左右，real-time 部署完全没问题。

### Trick 3：summarizer 把 K 步 thought 凝练成最终 trajectory

LLM 长篇 CoT 之后都要个 summary step 把中间推理凝练成 final answer，DeepSeek-R1、o1 都这样。FutureX 学了一个 summarizer network，输入 K 步 imagined future latents + 初始 trajectory，输出 **offsets**（残差修正），而不是 from scratch 重 decode。这是 ResNet 思想，让 model 聚焦在 "需要改的地方"。

---

## 数字说话

NAVSIM benchmark 上，TransFuser baseline 是 84.0 PDMS，FutureX-All 拉到 90.6，**+6.6 PDMS**。这个幅度在 autonomous driving 这种卷到死的领域算是非常猛的提升。

更值得注意的是 Table 3 的 CoT length 消融：

- K=1（不 think）：83.8
- K=2（think 2 次）：87.8
- K=4（think 4 次）：89.2

**K 翻倍，PDMS 大约涨 1.4，几乎线性**。这就是 test-time compute scaling law 在 driving 上的实证，跟 Snell et al. (https://arxiv.org/abs/2408.03314) 在 LLM 上看到的现象一模一样。如果 latency 预算允许 K=8 或 K=16，大概率还能继续涨。

CARLA Longest6 上 Town5 driving score 从 35.1 → 53.7，+18.6。Town5 是长程复杂场景，这种场景 CoT 价值最大，因为需要长程 reasoning。

---

## 为什么这个工作有意思

### 1. 它把 World Model 这个老 idea 重新做对了

Ha & Schmidhuber 2018 的 World Models (https://arxiv.org/abs/1803.10122) 早就提出 imagination-based planning。Dreamer (https://arxiv.org/abs/1912.01603) 在 RL 里把它做 work。但在 supervised E2E driving 里一直没做好，要么 pixel space 重建太慢（GAIA-1 (https://arxiv.org/abs/2309.17080)），要么只当 SSL pretrain（LAW (https://arxiv.org/abs/2406.08481)）。

FutureX 做对的点是：**纯 latent supervision + action-conditioned + trajectory refinement**。不重建 pixel，只对齐 latent；rollout 时 inject ego action；最后用 summarizer 把想象变成 action。整条链 differentiable，gradient 直接从 traj loss 流回 WM 每一层。这是 WM-policy co-training 的优雅形态。

### 2. 它是 System 1 / System 2 的 learned duality

你 Andrej 一直强调 "我们 model 需要 system 1 和 system 2 之间的可学习切换"。FutureX 的 auto-think switch 就是这个——简单场景 fast path，复杂场景 slow path，切换由 learned predictor 决定，不是 hardcode 的规则。这跟 Kahneman 的 dual process theory 在 model architecture 里 instantiate 了。

### 3. 它跟 o1 / DeepSeek-R1 的精神相通

o1 的本质是 inference time 多花 compute 换 accuracy。FutureX 把这个搬到 driving：K 越大，rollout 越深，PDMS 越高。auto-think switch 对应 o1 的 "什么时候该 think，什么时候不该 think" 的 router。summarizer 对应 o1 最后把长 CoT summary 成 final answer。

**整个 pipeline 就是 o1 范式在 continuous control driving 上的 instantiation**。

---

## 可能的 limitation 和我的胡乱联想

1. **Open-loop rollout 的误差累积**：W rollout 不接收真实未来 sensor，K 步后 latent 会 drift。author 用 L_lat 监督对齐真实未来 latent，但 distribution shift 在 closed-loop deployment 时可能爆。MPC-style closed-loop rollout 或 periodic replanning 可能更稳。

2. **CoT 边界是 evenly 切的**：每段 N=2 个 waypoint。但 driving 的 reasoning 粒度不该是均匀的——变道决策可能需要细化到 0.5s，直道巡航可以粗化到 4s。LLM CoT 也面临类似问题（thought 边界不自然）。可能可以学 adaptive segmentation。

3. **Auto-think label 有 bootstrap 依赖**：label 来自 "thinking 比 instant 好多少"，但 thinking 质量本身在训练过程中变化。早期 thinking 不准时 label 也不准，可能自我强化错误决策。可以用 curriculum 或 self-distillation 缓解。

4. **W 是确定性 transformer**：未来是 multimodal 的（前车可能左转也可能右转），确定性 rollout 会取 "平均" 的未来，可能 blur 掉关键 risk。换成 diffusion head（DiffusionDrive (https://arxiv.org/abs/2410.07781) 的思路）或 latent flow matching 可能更好。

5. **没做 latent beam search**：现在只在 ego 自己的初始 trajectory 上 rollout。其实可以 inject 多个 candidate action，对每个 rollout，然后用 summarizer + value head 选最好的。这就接近 MuZero (https://arxiv.org/abs/1911.08265) 的 latent MCTS 了。作者没往这个方向走，可能是 latency 考虑。

6. **W 换成 Mamba/SSM**：现在 W 是 transformer stack，每步都 full self-attention。换成 Mamba（DRAMA (https://arxiv.org/abs/2408.03601) 已经在用）可能省 latency，允许更大 K。

7. **Hierarchical CoT**：先 coarse 4 步（每步 2s），再 fine 8 步（每步 0.5s），类似 OpenAI o3 的 adaptive thought budget。这能更好 trade-off latency 和 reasoning depth。

8. **跟 RAG-like 机制结合**：复杂场景下检索历史类似场景的处理方式，inject 到 CoT context 里。像 Wayve 的 LINGO-2 (https://wayve.ai/thinking/lingo-natural-language-autonomous-driving/) 也在往这方向走，但都是文字层。FutureX 的 latent CoT 是更底层的好基底。

9. **CoT 可解释性**：latent thought 人看不懂。但可以训个 decoder 把 z_t^{(k)} 反解码成 "predicted future BEV"，做 visualization。这比 VLM 的文字 rationale 更 grounded——因为 thought 本身就是 driving decision 的直接 input，不存在说做脱节。

10. **Online RL fine-tuning**：现在纯 supervised。如果叠加 RL，让 W 在 closed-loop 里 finetune，可能解决 open-loop distribution shift。这跟 RLHF 之于 LLM 的角色类似。

---

## 一句话收尾

FutureX 这篇 paper 的核心贡献，是证明了 **"在 latent space 里做 CoT rollout 来补足当前观测缺失的反事实信息" 这个 idea 在 E2E driving 上 work，而且 cheap enough 可部署**。它把 LLM CoT、World Model、System 1/2 switching 这几个看似不相关的 concept 用一个统一 framework 串起来了，做得很干净，实验也 solid。在 NAVSIM 上 +6.2 PDMS 这种幅度的提升，配上 auto-think 的 efficiency trick，是有可能在工业界被实际采用的——不像很多 paper 只在 leaderboard 上好看。

References:
- FutureX paper 本身（附件）
- World Models (Ha & Schmidhuber 2018): https://arxiv.org/abs/1803.10122
- Dreamer: https://arxiv.org/abs/1912.01603
- MuZero: https://arxiv.org/abs/1911.08265
- CoT prompting (Wei et al.): https://arxiv.org/abs/2201.11903
- Test-time compute scaling (Snell): https://arxiv.org/abs/2408.03314
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OpenAI o1 system card: https://openai.com/index/openai-o1-system-card/
- NAVSIM: https://github.com/autonomousvision/navsim
- TransFuser: https://arxiv.org/abs/2205.15997
- LAW: https://arxiv.org/abs/2406.08481
- WoTE: https://arxiv.org/abs/2504.01941
- DiffusionDrive: https://arxiv.org/abs/2410.07781
- GAIA-1: https://arxiv.org/abs/2309.17080
- DRAMA (Mamba-based): https://arxiv.org/abs/2408.03601
- DriveVLM: https://arxiv.org/abs/2402.12227
- EMMA (Waymo): https://arxiv.org/abs/2410.23262
- LINGO-2 (Wayve): https://wayve.ai/thinking/lingo-natural-language-autonomous-driving/

---

# FutureX: Latent Chain-of-Thought World Model for E2E Autonomous Driving 深度解析

## 一、核心 idea 的 intuition build

这篇 paper 要解决一个很本质的问题：现有 E2E autonomous driving pipeline 都是 **one-shot feed-forward**——从当前 sensor 输入直接 decode 出一条 T 步的 trajectory，中间没有任何 "thinking"。这在 dynamic traffic 场景下会出问题，因为 ego vehicle 的 action 会改变未来 scene，当前 latent z_t 根本无法捕获 "如果我这样开，下一秒世界会变成什么样" 的反事实信息。

作者从 LLM 的 Chain-of-Thought (CoT) 里借鉴了一个很漂亮的 insight：CoT 的本质 **不在于它是文字**，而在于它能 **step-by-step 把未来展开**。把这一点迁移到 driving 里，"thought" 就不该是 VLM 那种生成 "I should slow down because..." 的文字 rationale（这些在 DriveLM、DriveVLM、EMMA 里都被证明是 descriptive commentary，和最终 control 脱节），而应该是 **latent space 里的 forward rollout**——每一步 thought = 一次 latent world model 的 forward + 一次 policy evaluation。

这其实就是把 Ha & Schmidhuber 2018 的 World Models (https://arxiv.org/abs/1803.10122) 里 "imagination-based planning" 的思想，和 LLM CoT 的 "intermediate reasoning step" 机制做了一个统一的形式化：**latent state evolution is the thought**。

Andrej 你在 https://karpathy.ai/ 一直强调的 "differentiable programming / system 2 thinking" 在这里被 instantiate 成了一个可学习、可 end-to-end backprop 的 WM-policy loop，这点我觉得是这篇 paper 最 elegant 的地方。

---

## 二、架构图解析 (Figure 2 逐模块拆解)

### 2.1 整体数据流

```
Raw sensor x_t 
    │
    ▼
Scene Encoder ──► z_t ∈ R^{L×C}   (BEV / latent scene feature)
    │
    ├──► Policy π(z_t) ──► w_t ∈ R^{T×3}  (initial trajectory, 一把梭)
    │
    ▼
Auto-think Switch G(z_t) ──► d_t ∈ [0,1]  (difficulty score)
    │
    ├── d_t < α  →  Instant mode: 直接输出 w_t
    │
    └── d_t ≥ α  →  Thinking mode:
                        │
                        ▼
              将 w_t 切成 K 段 w_t^{(1..K)}
                        │
                        ▼
              Latent World Model W (Transformer stack)
              循环 k=1..K:
                  z_t^{(k)} = W(z_t^{(k-1)}, w_t^{(k)})
                        │
                        ▼
              Z_CoT = {z_t^{(0)}, z_t^{(1)}, ..., z_t^{(K)}}
                        │
                        ▼
              Summarizer S(Z_CoT, w_t) ──► w_t^{ref}  (offsets)
                        │
                        ▼
              Final output: w_t^{ref}
```

### 2.2 Latent World Model W 的内部实现细节

paper 里 W 是一堆 Transformer layer 的 stack。输入构造方式：

1. trajectory encoder E_traj: R^{N×3} → R^{1×C}，把 N 个 waypoint 的 (x,y,θ) 压成一个 embedding token c_t^{(k)}。
2. 把 c_t^{(k)} 和 z_t^{(k-1)} ∈ R^{L×C} 沿 sequence dim concat → R^{(L+1)×C}。
3. 过 multi-head self-attention，让 c_t^{(k)}（query 信号 = "我接下来要这样走"）去 attend z_t^{(k-1)} 里所有 spatial-temporal token，产生 updated latent z_t^{(k)} ∈ R^{L×C}。

这里有个 **很重要的设计 choice**：c_t^{(k)} 是作为 *extra token* 拼进去，而不是 broadcast 加到所有 L 个 token 上。这意味着 attention 可以选择性地让某些 BEV location 受 trajectory intention 影响（比如 ego 前方区域 attention 权重大），这是 spatially-aware 的条件注入，比简单 FiLM 或 cross-attention 更灵活。

Appendix A 应该有更细节的 layer 数 / head 数，paper 正文没给，我猜测可能是 4-6 layer、8 head、C=256 这个量级，和 TransFuser 的 neck 对齐。

---

## 三、公式逐项拆解（变量、上下标全部讲清楚）

### Eq. (1) Initial Trajectory Proposal

$$\mathbf{w}_t = \pi(\mathbf{z}_t) = \{w_1, w_2, \ldots, w_T\} \in \mathbb{R}^{T \times 3}$$

- t：当前 timestep（驾驶帧）
- π：policy network，输入 latent，输出 trajectory
- z_t：scene encoder 输出的 latent，L 个 spatial token，每个 C 维
- w_i = (x_i, y_i, θ_i)：第 i 个未来 waypoint，在 **ego coordinate frame**（不是 global），x/y 是空间位置，θ 是 heading
- T：预测 horizon，NAVSIM 里 T=8

### Eq. (2) Chain-of-Thought Segment Construction

$$\mathbf{w}_t = [\mathbf{w}_t^{(1)}, \mathbf{w}_t^{(2)}, \ldots, \mathbf{w}_t^{(K)}] \in \mathbb{R}^{K \times N \times 3}$$
$$\mathbf{w}_t^{(k)} = \{w_{(k-1)N+1}, \ldots, w_{kN}\} \in \mathbb{R}^{N \times 3}$$

- 上标 (k)：第 k 个 CoT reasoning step，k ∈ {1, ..., K}
- N = T/K：每段 sub-trajectory 长度
- 这个切分是 **evenly** 的，意味着每个 thought 对应等长时间窗的未来
- NAVSIM 里 T=8，作者主推 K=4, N=2（Table 3 显示 N=2 比 N=4 好，因为更多 reasoning step）

这里 build intuition：K 越大，reasoning 越细，越像 LLM 的 long CoT，但 latency 线性增长。这是 **CoT length vs accuracy vs latency** 的三 way trade-off，跟 LLM test-time compute scaling (https://arxiv.org/abs/2408.03314) 思路一模一样。

### Eq. (3) CoT-guided Latent World Model Rollout

$$\mathcal{W}(\cdot, \cdot): \mathbb{R}^{L \times C} \times \mathbb{R}^{N \times 3} \mapsto \mathbb{R}^{L \times C}$$
$$\mathbf{z}_t^{(k)} = \mathcal{W}(\mathbf{z}_t^{(k-1)}, \mathbf{w}_t^{(k)}), \quad k=1,\ldots,K$$

- z_t^{(0)} = z_t：初始 latent，即当前观测编码
- z_t^{(k)}：执行第 k 段 sub-trajectory **后** 的 imagined latent，是 "如果我按 w_t^{(k)} 走，世界会变成什么样" 的 latent 反应
- W：参数共享的 transition function，所有 K 步都用同一个 W

**关键点**：W 是 **action-conditioned** 的 latent dynamics。这正是 World Model (Ha & Schmidhuber) 的核心定义：s_{t+1} = f(s_t, a_t)。这里 a_t = w_t^{(k)}（连续 action），s_t = z_t^{(k-1)}（latent state）。

### Eq. (4) Latent Reasoning Chain

$$\mathbf{Z}_{CoT} = \{\mathbf{z}_t^{(0)}, \mathbf{z}_t^{(1)}, \ldots, \mathbf{z}_t^{(K)}\}$$

- 这就是一个长度 K+1 的 latent state 序列，类比 LLM 的 thought sequence
- 每个 z_t^{(k)} 是一个 "thought"，对应一个 future time step 的 imagined scene
- 整个序列构成 latent CoT

### Eq. (5) Trajectory Refinement

$$\mathcal{S}(\cdot, \cdot): \mathbb{R}^{K \times L \times C} \times \mathbb{R}^{K \times N \times 3} \mapsto \mathbb{R}^{T \times C}$$
$$\mathbf{w}_t^{ref} = \mathcal{S}(\mathbf{Z}_t^{CoT}, \mathbf{w}_t)$$

- S：summarizer network
- 输入：reasoning chain Z_CoT + initial trajectory w_t
- 输出：refined trajectory w_t^{ref} = {w̃_1, ..., w̃_T}
- 实现上 S 是 **predict offsets**（w_t^{ref} = w_t + Δw），不是 from scratch 重新 decode。这跟 ResNet 残差思想一致，让模型聚焦在 "需要修正的地方"

这个 summarizer 的角色 **完全对应 LLM 的 "answer consolidation"**，比如 DeepSeek-R1 (https://arxiv.org/abs/2501.12948) 和 OpenAI o1 (https://openai.com/index/openai-o1-system-card/) 在长 CoT 之后都要 summary step 把中间 thought 凝练成 final answer。这里 author 显式引用了 [12, 29] 来类比。

### Eq. (6)(7)(8) Auto-think Switch

$$d_t = \mathcal{G}(\mathbf{z}_t), \quad d_t \in [0, 1]$$

- G：一个小的 head（可能就是 MLP 或一两层 transformer），从 z_t 预测 difficulty score
- d_t：当前场景的 planning 难度，0~1 之间

$$e_{init} = \|\mathbf{w}_t - \mathbf{w}_t^{gt}\|_1, \quad e_{ref} = \|\mathbf{w}_t^{ref} - \mathbf{w}_t^{gt}\|_1$$

- e_init, e_ref：initial / refined trajectory 与 ground truth 的 L1 error
- w_t^{gt}：人类专家 trajectory

$$r_t = \frac{e_{init} - e_{ref}}{e_{init} + \varepsilon}, \quad g_t = \mathbb{I}(r_t > \alpha)$$

- r_t：refinement gain ratio，衡量 "thinking 带来多少相对提升"
- ε：数值稳定项，避免 e_init=0 时除零
- α：thinking-mode 触发阈值，paper 设 α=0.25
- g_t：binary thinking flag，1 表示需要 thinking，0 表示直接 instant output
- I(·)：indicator function

**这个 label 构造非常聪明**：它把 "什么时候该 think" 变成一个可学习问题。简单场景 thinking 带不来收益 → 不 think；复杂场景 thinking 能显著降 error → think。这正是 ChatGPT5 引入 auto-reasoning trigger 的精神（author 显式 reference [30]）。跟 OpenAI o1/o3 的 "adaptive test-time compute" 思路完全一致。

### Eq. (9) Latent Consistency Loss

$$\mathcal{L}_{lat} = \frac{1}{K} \sum_{k=1}^{K} \|\hat{\mathbf{z}}_t^{(k)} - \mathbf{z}_t^{(k)}\|_1$$

- ẑ_t^{(k)}：**真实未来** 的 latent，由 scene encoder 对未来第 k 步的 sensor 输入 x_t^{(k)} 编码得到
- z_t^{(k)}：W rollout 预测的 latent
- L1 距离，对 K 步取平均

**这是 WM 训练的核心**：用 teacher-forcing 方式，让 imagined latent 对齐真实未来 latent。这跟 Dreamer (https://arxiv.org/abs/1912.01603)、PlaNet (https://arxiv.org/abs/1811.04579) 的 latent dynamics learning 是一脉相承的，区别在于 FutureX 不在 pixel space 重建，纯 latent supervision，所以非常 cheap。

### Eq. (10) Trajectory Loss

$$\mathcal{L}_{traj} = g_t \cdot e_{ref} + (1 - g_t) \cdot e_{init}$$

- 如果 g_t=1（thinking mode 激活），loss 用 refined trajectory 的 L1 error
- 如果 g_t=0（instant mode），loss 用 initial trajectory 的 L1 error
- 这是一个 **mixture**，让模型同时学好两种 mode

### Eq. (11) Auto-think Loss

$$\mathcal{L}_{auto} = -[y_t \log d_t + (1-y_t) \log(1 - d_t)]$$

- 标准 binary cross-entropy
- y_t：thinking flag 的 label，paper 这里写的是 y_t 但前面定义是 g_t，应该是同一个东西的 supervision target（可能 g_t 是 derived，y_t 是 smoothed label，paper 没完全说清楚）
- d_t：G 的输出 sigmoid score

### Eq. (12) Overall Objective

$$\min_{\Theta} \mathcal{L}_{traj} + \lambda_1 \mathcal{L}_{lat} + \lambda_2 \mathcal{L}_{auto}$$

- λ1 = 0.1, λ2 = 0.1：两个辅助 loss 权重，相对 traj loss 是 minor term
- 一起 end-to-end 反传，整个 pipeline 完全 differentiable（W 是 transformer，G 和 S 都是 head）

---

## 四、PDMS 评分公式拆解

$$PDMS = NC \times DAC \times \frac{5 \times EP + 5 \times TTC + 2 \times Comf.}{12}$$

- NC (No at-fault Collision)：无责任碰撞率
- DAC (Drivable Area Compliance)：可行驶区域合规率
- EP (Ego Progress)：自车进度（route 完成度）
- TTC (Time-To-Collision)：碰撞时间安全度
- Comf. (Comfort)：舒适度（jerk / accel 平滑度）
- 权重 5:5:2 加起来 12，归一化

**设计哲学**：NC 和 DAC 是 **multiplicative gate**（任何一个为 0 就全归 0，对应严重事故），EP/TTC/Comf 是 **additive quality**，符合 NAVSIM (https://arxiv.org/abs/2411.18726) 的设计哲学。FutureX 的提升主要来自 NC（从 97.4→99.6）和 EP（79.0→84.5），说明 thinking 真的能避免 collision 并让车开得更 "敢"。

---

## 五、实验数据表深度解读

### Table 1：NAVSIM Navtest 主结果

**Camera-only 对比**：
| Method | World Model | PDMS | 增量 |
|--------|-------------|------|------|
| UniAD | ✗ | 83.4 | baseline |
| VADv2 | ✗ | 83.0 | -0.4 |
| LTF | ✗ | 83.8 | +0.4 |
| LAW | ✓ | 84.6 | +1.2 |
| World4Drive | ✓ | 85.1 | +1.7 |
| **FutureX-Auto (on LTF)** | ✓ | **89.2** | **+5.4** |
| **FutureX-All (on LTF)** | ✓ | **90.1** | **+6.3** |

**Camera-LiDAR 对比**：
| Method | PDMS | 增量 |
|--------|------|------|
| TransFuser | 84.0 | baseline |
| DRAMA | 85.5 | +1.5 |
| Hydra-MDP | 86.5 | +2.5 |
| DiffusionDrive | 88.1 | +4.1 |
| WoTE | 88.3 | +4.3 |
| **FutureX-Auto (on TransFuser)** | **90.2** | **+6.2** |
| **FutureX-All (on TransFuser)** | **90.6** | **+6.6** |

**关键观察**：
1. FutureX 在 camera-only 上提升最大（+5.4），说明 latent CoT 补偿了 vision 缺 depth 的短板
2. 即使 FutureX-Auto 只在 75.5% 场景 think（24.5% instant），仍能拿到 FutureX-All 95% 的性能（89.2 vs 90.1）
3. 在 TransFuser 上 instant 占 13.7%，性能差只有 0.4 PDMS（90.2 vs 90.6），auto-think 学得相当准

### Table 3：CoT length 消融

| K | N | NC | DAC | EP | TTC | PDMS |
|---|---|----|-----|----|-----|------|
| 1 | 8 | 97.4 | 92.8 | 79.0 | 92.4 | 83.8 |
| 2 | 4 | 98.6 | 95.5 | 82.4 | 94.8 | 87.8 |
| 4 | 2 | 99.0 | 96.3 | 83.6 | 95.7 | 89.2 |

**这里有个很漂亮的 test-time compute scaling 现象**：K 翻倍，PDMS 提升 +1.4 / +1.4，几乎是线性的！这跟 Snell et al. 的 test-time compute scaling law (https://arxiv.org/abs/2408.03314) 高度一致。可以推测在 K=8 时可能还能继续涨，只是 latency 不可接受了。

### Table 4：Loss 消融

| L_lat | L_traj | NC | DAC | EP | TTC | PDMS |
|-------|--------|----|-----|----|-----|------|
| ✗ | ✗ | 97.4 | 92.8 | 79.0 | 92.4 | 83.8 |
| ✓ | ✗ | 99.7 | 95.1 | 82.8 | 97.0 | 88.7 (+4.9) |
| ✓ | ✓ | 99.6 | 96.6 | 84.5 | 96.5 | 90.1 (+6.3) |

**单加 L_lat 就涨 4.9 PDMS**！这非常值得注意：这意味着 **仅靠 latent CoT rollout 的 auxiliary supervision**（不 refine trajectory），就能让 scene representation 学得更好，连 instant mode 的初始 trajectory 也受益了。这跟 SimD (https://arxiv.org/abs/2410.14671)、DriveWorld (https://arxiv.org/abs/2401.16123) 等 "WM 作为 SSL pretrain" 的结论一致——WM 是个 powerful representation learner。

### Table 5：Latency

| Method | N | Latency | PDMS |
|--------|---|---------|------|
| LTF | 8 | 2.3 ms | 83.8 |
| FutureX-All | 4 | 17.0 ms | 87.8 |
| FutureX-All | 2 | 31.3 ms | 89.2 |

- 31.3 ms 对应 ~32 Hz，仍然 real-time
- FutureX-Auto 平均 latency 应该在 20 ms 量级（24.5% 场景走 31ms 分支，75.5% 走 2.3ms baseline → 0.245×31.3 + 0.755×2.3 ≈ 9.4ms）

### Table 2：CARLA Longest6

Town1 DS 从 73.3 → 84.3（+11.0），Town5 从 35.1 → 53.7（+18.6）。**Town5 这种长程复杂场景 FutureX 增益最大**，进一步印证 CoT 长程推理的价值。

---

## 六、和 related work 的精确定位

| 方法 | 思路 | FutureX 区别 |
|------|------|--------------|
| LAW (https://arxiv.org/abs/2406.08481) | latent WM 作为 SSL 提升 scene repr | FutureX 把 WM 当 CoT rollout，trajectory-conditioned |
| World4Drive (https://arxiv.org/abs/2411.14056) | intention-aware latent WM | FutureX 有 auto-think switch，不 always-on |
| WoTE (https://arxiv.org/abs/2504.01941) | 在 candidate trajectory 上做 BEV future predict 用于 evaluation | FutureX 在 latent 上 rollout，不评估 candidate set |
| GAIA-1/2 (https://arxiv.org/abs/2309.17080, https://arxiv.org/abs/2503.20523) | pixel-space generative WM | FutureX 纯 latent，deployable |
| DriveVLM (https://arxiv.org/abs/2402.12227), EMMA (https://arxiv.org/abs/2410.23262) | 文字 CoT | FutureX 是 latent CoT，"thought in action" |

---

## 七、Andrej 你可能会感兴趣的几个点

1. **System 1 / System 2 的可学习接口**：Auto-think switch 本质上是 learned duality——简单场景走 System 1（instant forward），复杂场景走 System 2（CoT rollout）。这跟你 tweet 过的 "we need a learnable switch between fast and slow thinking" 完全对应。

2. **Test-time compute scaling**：Table 3 显示 PDMS 随 K 近似线性提升。如果 author 把 K 推到 16 或加 self-consistency 多次 sampling + vote，可能能继续涨。这跟 o1-style inference scaling 完全可类比。

3. **Differentiable imagination**：W 是 transformer，S 是 head，整条 CoT 都可反传。不像 MCTS 那种 non-differentiable search，这里 gradient 直接从 final trajectory loss 流回 W 的每一层。这是 WM-policy co-training 的优雅形态。

4. **潜在 limitation**：
   - W rollout 是 **open-loop** 的（不接收真实未来 sensor），K 步后误差会累积。Author 用 L_lat 监督，但没讨论 distribution shift。MPC-style closed-loop rollout 在 deploy 时可能出问题。
   - trajectory 切分是 evenly 的，可能不是最优。LLM CoT 也面临类似问题（thought 边界不自然）。
   - Auto-think switch 的 label 依赖 w_t^{ref} 的离线 quality，bootstrap 一旦不准会自我强化。
   - 没报告 corner case 分析（什么场景被 switch 误判为 simple）。

5. **可能扩展**：
   - 把 W 换成 SSM/Mamba（DRAMA 已经在用），可能更省 latency
   - 加 hierarchical CoT：先 coarse 段，再 fine 段，类似 OpenAI o3 的 adaptive thought budget
   - 在 w_t^{(k)} 之外 inject 多个 candidate action，做 latent beam search（类似 MuZero https://arxiv.org/abs/1911.08265）
   - 和 DiffusionDrive (https://arxiv.org/abs/2410.07781) 结合：用 diffusion 而非确定性 transformer 做 W，捕获 multimodal future

6. **代码**：paper 说 code will be released，但目前没看到 repo。可关注作者主页 https://github.com/hongbin0112Lin 等待开源。

---

## 八、一句话总结

FutureX 把 LLM 的 CoT 重新诠释为 **latent world model 的 forward rollout**，用 auto-think switch 实现 system 1/2 的可学习切换，在 NAVSIM 上把 TransFuser 从 84 拉到 90.6 PDMS，latency 仍 real-time。本质上是把 "differentiable imagination" 这个老 idea 用 modern E2E AD 框架重新做了一遍，做得很干净。

References:
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- Dreamer: https://arxiv.org/abs/1912.01603
- MuZero: https://arxiv.org/abs/1911.08265
- CoT prompting: https://arxiv.org/abs/2201.11903
- Test-time compute scaling: https://arxiv.org/abs/2408.03314
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- NAVSIM: https://arxiv.org/abs/2411.18726 (官方 https://github.com/autonomousvision/navsim)
- TransFuser: https://arxiv.org/abs/2205.15997
- LAW: https://arxiv.org/abs/2406.08481
- WoTE: https://arxiv.org/abs/2504.01941
- DriveVLM: https://arxiv.org/abs/2402.12227
- EMMA: https://arxiv.org/abs/2410.23262
- DiffusionDrive: https://arxiv.org/abs/2410.07781
- GAIA-1: https://arxiv.org/abs/2309.17080
- OpenAI o1 system card: https://openai.com/index/openai-o1-system-card/
