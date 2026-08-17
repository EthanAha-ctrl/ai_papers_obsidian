---
source_pdf: Towards Practical World Model-based Reinforcement Learning for.pdf
paper_sha256: 52f8b7560bc7e9de4f7c1de4ec4379e9e1eba3bd3aa671999cde4cf0615d59d4
processed_at: '2026-08-12T17:27:54-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 VLA-MBPO

---

## 这篇paper到底想干啥

你训练了一个VLA model（比如 $\pi_0$），它能看图、听指令、输出动作。你用 Behavior Cloning 在专家数据上 fine-tune 了一下，效果还行，但总有些 corner case 处理不好——robot 偏了一下就不知道怎么恢复，或者遇到没见过的物体配置就傻了。

你想用 RL 继续 fine-tune，让 robot 自己试错、自己学习。但问题来了：**real-world RL 要 robot 真的去试错**。一个 $20000 的机械臂，你让它随机探索？撞坏了算谁的？就算不撞坏，收集几万次 rollout 的成本也扛不住。

所以自然的想法是：**能不能造一个"假世界"（world model），让 robot 在假世界里练习，练好了再去 real world？**

这就是 model-based RL 的思路。但把它用到 VLA 上，有三个大坑。

---

## 三个大坑

### 坑一：World model 得生成图片

传统的 model-based RL，state 是低维的——比如关节角度、位置速度，几个数字而已。World model 就是一个预测下一组数字的 function。

但 VLA 不一样。VLA 吃的是 **原始图片**。所以你的 world model 得生成 **未来的图片**。不是那种模糊的、大概像的图片，而是得足够清晰、足够准确，能让 VLA 从中提取出有用的视觉信息。

从零训练一个能生成图片的 world model？Data hungry 到爆炸，而且在 offline data 上容易 overfit。

### 坑二：得是多个视角的图片，还得一致

Fine-grained manipulation 光靠一个 head camera 不够。你得有 head view（看全局）+ wrist view（看细节）。

如果两个视角的预测不一致——比如 head view 里显示手在左边，wrist view 里显示手在右边——那 VLA 就 confused 了，不知道该信谁。

### 坑三：Sparse reward 下的 compounding error

这是最致命的。

World model 不可能完美。每一步预测有一点点 error，rollout 10 步后 error 累积起来，可能预测出一个跟现实完全不同的状态。

在 dense reward 的场景下还好——就算状态预测偏了一点，reward 大致趋势还是对的。但 manipulation task 的 reward 通常是 **binary 的**：成功=1，失败=0。

World model 预测 robot 差一毫米没插进插座——它觉得成功了，给了 reward=1。Reality 里其实失败了。Policy 学到了错误信号，越练越歪。

---

## VLA-MBPO 怎么填这三个坑

### 填坑一：用 UMM 当 world model

之前的做法是用两个 model：一个 video generation model 预测未来画面，一个 VLM 判断 reward。两个 model 两套工程，复杂得要命。

这篇 paper 说：**干嘛不用一个 unified multimodal model (UMM) 搞定？** 像 Bagel 这种模型，既能理解图片、又能生成图片、又能理解语言。稍加改造就能同时预测未来画面和 reward。

具体怎么改？Action 怎么输入 UMM？UMM 本来只吃 vision 和 language，不认识 action。

Trick 是：把 continuous action 离散化成 0-255 的整数，当成"文字 token"塞进去。比如一个 action chunk 是 `[111, 113, 127, ...]`，就当成一句话喂给 UMM。UMM 不需要改架构，不需要扩 vocabulary，直接用 pretrain 好的能力。

还有一个 efficiency trick：**frame-skipping**。传统 video world model 是一步步生成的——$s_t \rightarrow s_{t+1} \rightarrow s_{t+2} \rightarrow ... \rightarrow s_{t+k}$。UMM-World 直接从 $s_t$ 跳到 $s_{t+k}$，中间帧不生成。因为 VLA 本来就是 action chunking 的（一次输出 k 个 action），你只需要 k 步后的状态，中间帧没用。这个直接 2x speedup。

### 填坑二：Interleaved View Decoding

Naive 做法：head view 和 wrist view 各自独立预测。问题就是前面说的——可能不一致。

VLA-MBPO 的做法：**先预测 head view，再基于 head view 预测 wrist view**。

$$s_{t+k}^h \sim T_\theta(\cdot | s_t^h, s_t^w, a)$$
$$s_{t+k}^w \sim T_\theta(\cdot | s_t^w, s_{t+k}^h)$$

Head view 有全局信息，先确定"全局长什么样"。Wrist view 是局部细节，在全局确定的基础上再 predict。这样 wrist view 一定会跟 head view 一致。

实现上就是改 attention mask——让 wrist view 的 token 能 attend 到已经生成的 head view token，但 head view 不能 attend 到 wrist view（因为 head view 先生成）。

Paper 里有个很好的 qualitative example（Figure 12）：即使预测的 head view 跟 ground truth 不完全一样（比如右手位置有点偏），预测的 wrist view 跟预测的 head view 是一致的——手腕 view 里的手位置跟 head view 里看到的手位置吻合。**它保证了 internal consistency，即使整体跟 ground truth 有偏差**。这对 downstream policy 很重要，因为 policy 需要的是 consistent 的多视角信息，而不是每个视角都完美但彼此矛盾。

### 填坑三：Chunk-level Branched Rollout

这个是理论基础最扎实的部分。

**Compounding error 的本质**：world model 每步有 $\epsilon_m$ 的 error，rollout $n$ 步后 error 大约是 $n \cdot \epsilon_m$。Rollout 越长，error 越大。

**MBPO 的经典思路**（Janner 2019）：不从 initial state 开始 rollout 一整条 trajectory，而是从 offline dataset 里的 **任意中间状态** 开始，只 rollout 短短几步（branched rollout）。这样 error 累积有限。

**VLA-MBPO 的扩展**：因为 world model 在 chunk level 操作（一次跳 k 步），所以 branched rollout 的"长度"再除以 k。原来 rollout 20 步的 error 累积，现在只需要 rollout 2 个 chunk（= 20 步），但 model error 只在 2 个 macro-step 上累积，而不是 20 个 micro-step。

**理论分析**：

Theorem 4.1（传统方法）：在 step-level world model 上做 full-horizon rollout，value gap 里 model error 项是 $\frac{k \gamma^k}{1-\gamma^k} \epsilon_m$。注意那个 $k$——chunk size 直接放大了 model error。

Theorem 4.2（VLA-MBPO）：在 chunk-level world model 上做 n-chunk branched rollout，value gap 里 model error 项是 $n \epsilon_m^{k,n}$。线性增长，没有 $k$ 的放大。

**具体数字**（$\gamma=0.99, k=10, n=2$）：
- 传统方法：model error 项 $\approx 18916 \epsilon_m$
- VLA-MBPO：model error 项 $\approx 400 \epsilon_m^{k,n}$
- **47 倍的 reduction**

这个 reduction 不是 empirical 的 hand-wavy claim，是定理保证的。

---

## 但有个问题：short rollout 怎么学长 horizon 任务？

这是我看 paper 时最大的疑问。

你只 rollout 2 个 chunk（= 20 步），但 LIBERO-Long 的 task 可能要 100+ 步才能完成。Value function 怎么学到 long-horizon 的 return？

Paper 的回答（Section 5.2 + Figure 3）：**trajectory stitching**。

不同 branched rollout 从不同 states 开始。Value function 通过 GAE 的 bootstrap 项 $V_\phi(s_{t+k})$ 把 local estimates 串起来。State A 的 value 依赖 State B 的 value，State B 的 value 依赖 State C 的 value……这样虽然每次只 rollout 2 步，value function 还是能学到整条 trajectory 的 return。

Figure 3 展示了这个过程：训练初期 value model 的预测跟 ground-truth return 差很远，但随着训练进行，value model 逐渐对齐了 long-horizon return。**它学会了 cross-chunk 的 temporal dependency**。

这个 stitching 能力依赖于 offline dataset 的 coverage——你得有覆盖 trajectory 不同部分的数据，才能 stitch。如果 dataset 只有 short trajectories，stitch 就断了。这是这个方法的一个 implicit assumption。

---

## 整个 Algorithm 长什么样

```
1. 用 SFT 后的 VLA policy 在 real world 收集一些数据，放进 buffer D
   （50 条 rollout，不算多）

2. 用 D fine-tune UMM-World（7-8 小时，8×H100）

3. 在 UMM-World 里做 RL（4-6 小时）：
   for each RL iteration:
     a. 从 D 里随机采样一批 states
     b. 用 UMM-World 从这些 states 开始 branched rollout（2 chunks）
     c. 用这些 synthetic data 算 GAE advantage
     d. 用 Flow-Noise（PPO 的 flow matching variant）更新 policy 和 value
```

Real-world interaction 只在 step 1 发生，而且只需要 50 条 rollout。之后的训练全在 world model 里。

---

## 实验说了啥

### World model 本身

UMM-World 比 Ctrl-World（video generation model）在画面预测上更好（LPIPS 更低、PSNR 更高），还快 2 倍。Reward 预测跟专门的 Qwen3-VL 打平。

Interleaved view decoding 对 wrist view 至关重要——去掉后 wrist view 的 LPIPS 从 0.254 恶化到 0.454。

Pretraining 更关键——去掉后全面崩溃。

### LIBERO benchmark

VLA-MBPO 在所有 4 个 suite 上都比 baseline 好。Long-horizon suite 提升最大（+12.2），这正是 compounding error 最严重的场景，也是 chunk-level branched rollout 发挥最大价值的地方。

对比 online RL（$\pi_{RL}$）：$\pi_{RL}$ 也提升，但它需要 real-world interactions。VLA-MBPO 用同样的数据预算但不需要 real-world 试错。

### Real-world tasks

5 个 task，两个 robot（Arx-X5 bimanual + Galaxy-R1 whole-body 21 DoF）：

- Plug Cable（3mm socket，sub-centimeter precision）
- Fold Towel（deformable object）
- Pick Cup（viewpoint disturbance）
- Insert Pen（fine motor skill）
- Wipe Board（mobile manipulation + partial observability）

所有 task 都有提升，包括 seen 和 unseen configurations。

### Ablation

Rollout length：n=2 最优。n=1 太短限制 exploration，n=4 error 累积，full-horizon 直接崩。

Sample size：越大越好，monotonic improvement。

---

## 我的几点观察

### 1. Single hyperparameter set 是真的 practical

Table 5 里，除了 LIBERO-Long 需要更大的 sample size（1280 vs 512）和更多 update-to-data（50 vs 20），其他所有 hyperparameter 完全一致。这在 real-world deployment 中太重要了——你不想每换个 task 就重新调参。

### 2. Reward model 的 binary 限制

Prompt 强制 reward model 输出 "Yes" 或 "No"。对于 partial success（"差一点点就插进去了"），这种 binary signal 的 credit assignment granularity 不够。如果能有 graded reward（"离 goal 还有 2cm"），policy learning 可能更高效。但这也要求 reward model 有更细粒度的理解能力，可能需要更多 training。

### 3. Interleaved decoding 的 order dependency

先 predict head view 再 predict wrist view。如果 head view 预测错了，wrist view 也会被带偏。有没有可能做一个 iterative 的方案——先粗 predict 两个 view，再互相 refine？不过这会增加 inference 复杂度，paper 选了简单方案。

### 4. Failure cases 揭示的 fundamental limits

Figure 13 和 14 的 failure cases 很有启发：

- **Partial observability**：arm 移出 head camera 视野，world model 直接把 arm "弄丢了"。Wrist camera 有 local 信息但缺乏 global kinematic context。这说明 world model 没有真正的 3D spatial reasoning，还是在 2D pixel space 里做 pattern matching。

- **Large motions**：大幅运动导致 "motion collapse"——model 保守预测 minimal change。这是 diffusion/flow-based generation 的已知问题，在大幅 motion 上容易 mode collapse。

这两个 failure 跟 VLA-MBPO 的 framework 设计无关，是 world model 本身的能力上限。要解决可能需要 explicit 的 3D representation（比如 NeRF、3D Gaussian Splatting）或者 video diffusion model 的更好 motion modeling。

### 5. Value stitching 的 assumption

Trajectory stitching 假设 offline dataset 覆盖了 trajectory 的不同部分。如果你的 SFT policy 很差，收集的 data 都集中在 trajectory 开头（因为 policy 很快就 fail 了），那 stitching 就断了——你没法 stitch 到 success state 附近的数据，因为根本没有。

这意味着 VLA-MBPO 有一个 bootstrap 门槛：你的 SFT policy 得足够好，能收集到覆盖 trajectory 不同部分的数据。Paper 里用 $\pi_{0.5}$ 做 SFT，这个 base model 本身就挺强了。如果 base model 很弱，VLA-MBPO 可能不 work。

---

## 跟相关工作的关系

- **vs MBPO**：MBPO 是这个 framework 的精神祖先。核心 idea（branched rollout）一样，但 MBPO 在低维 state space，VLA-MBPO 扩展到 pixel + chunk level。

- **vs WMPO**：WMPO 也用 world model for VLA，但用 video generation model + full-horizon rollout。Paper 在 Theorem 4.1 里证明了这种方法在 long-horizon + sparse reward 下 value gap 会爆炸。VLA-MBPO 用 chunk-level branched rollout 解决这个问题。

- **vs $\pi_{RL}$**：$\pi_{RL}$ 是 online RL for flow matching VLA。效果好但需要 real-world interactions。VLA-MBPO 用 world model 避免 real-world interactions，在同样 data budget 下更 practical。

- **vs $\pi^*$ / Reinbot**：这些是 offline RL 方法，不用 world model。它们只能利用 static data，不能 imagine 新的 transitions。VLA-MBPO 通过 world model 能 generate synthetic experience，exploration 更充分。

---

## 一句话总结

VLA-MBPO = **UMM 当 world model（简单高效）+ interleaved view decoding（多视角一致）+ chunk-level branched rollout（理论保证 compounding error 可控）**

这三个 design choice 各自解决一个具体问题，合在一起形成一个 practical framework——single hyperparameter set across tasks，50 条 real rollout 就能 fine-tune，在 simulation 和 real-world 都 validate 了。理论分析（Theorem 4.1 vs 4.2）把每个 design choice 的贡献量化了，model error reduction 47 倍。这不是 empirical hand-wavy，是定理保证的。

---

# VLA-MBPO: Practical World Model-based RL for VLA Models 深度解析

Andrej, 这篇paper试图解决一个非常实际的问题：如何用world model-based RL来fine-tune VLA models，避免real-world interaction的高成本和安全性风险。让我从intuition出发，逐层拆解。

---

## 1. 问题动机：为什么需要这个framework

### 1.1 当前的困境

VLA models (如 $\pi_0$, OpenVLA, Gr00t-N1) 通过大规模pretraining获得了generalization，但fine-tuning阶段主要依赖Behavior Cloning (BC)。BC有两个根本问题：

1. **Distribution shift**: BC假设训练数据分布和测试分布一致，但expert demonstrations有限，无法覆盖所有可能的states
2. **No recovery behavior**: BC只在successful trajectories上训练，policy一旦偏离轨迹就不知道如何恢复

RL可以解决这些问题，但real-world RL面临：
- **Sample complexity**: 需要海量interactions
- **Safety**: exploration可能损坏robot或环境
- **Cost**: 每次real-world rollout都很昂贵

### 1.2 World Model作为虚拟环境的挑战

用learned world model作为simulator听起来很美好，但paper指出了三个核心challenge：

**Challenge 1: Pixel-level world modeling**
VLA models以原始图像作为input，world model必须生成高保真的future frames。从scratch训练data-hungry且容易overfit offline data。

**Challenge 2: Multi-view consistency**
Fine-grained manipulation需要多个camera views (head view + wrist view)。如果views不一致，policy会confused。

**Challenge 3: Compounding errors under sparse rewards**
这是最致命的。Manipulation tasks通常是sparse binary reward (success/fail)。即使world model有很小的prediction error，rollout几步后可能产生完全相反的reward signal。

---

## 2. VLA-MBPO的三个核心设计

### 2.1 UMM-based World Model

#### 2.1.1 核心insight

之前的approach (如Ctrl-World, WMPO)用两个separate models：
- Video generation model for dynamics
- VLM for reward prediction

这增加了系统复杂度。Paper的insight是：**Unified Multimodal Models (UMMs) 本身就能同时处理vision和language，可以joint prediction dynamics和reward**。

#### 2.1.2 Action representation

UMM原生处理vision和language，但VLA需要action input。Paper采用了一个聪明的trick：

$$\tilde{a}_t = (a_t, a_{t+1}, ..., a_{t+k-1}) \in \mathbb{R}^{k \times d}$$

其中：
- $k$: chunk size (通常10)
- $d$: action dimension (joint positions, gripper states等)

Continuous actions被discretized到 $[0, 256]$ 范围，映射为integer tokens，直接进入UMM的vocabulary。

**State transition**:
$$s_{t+k} \sim T_\theta(\cdot | s_t, \tilde{a}_t)$$

**Chunk-level reward**:
$$r_\theta(s_{t+k}, l) = \sum_{i=1}^{k} \gamma^{i-1} r(s_{t+i}, l)$$

这里 $\gamma$ 是discount factor，$r(s_{t+i}, l)$ 是step-level reward。这个formulation将k-step累计reward作为chunk-level reward，避免了step-by-step的reward prediction。

#### 2.1.3 Frame-skipping scheme

这是efficiency的关键。Figure 2展示的frame-skipping：

```
传统video world model: s_t -> s_{t+1} -> s_{t+2} -> ... -> s_{t+k}
UMM-World: s_t --[action chunk]--> s_{t+k}
```

不生成中间frames，直接预测k步后的state。这不仅2x faster (Table 1的Inf. Time: 10s vs 21s)，还避免了中间frames的compounding errors。

#### 2.1.4 为什么用Bagel作为base model

Bagel (Deng et al., 2025a)是一个unified multimodal model，原生支持：
- Image understanding (ViT features)
- Image generation (VAE features)  
- Text understanding

通过structured prompts (Appendix D)指导模型作为"physical simulator"：

```
"You are now acting as a world model that simulates robot manipulation 
task execution. Your task is to predict the next frame of visual 
observation, given: (1) Multiple current observation images; 
(2) An action sequence; (3) Optionally, next frame from head camera..."
```

### 2.2 Interleaved View Decoding (IVD)

#### 2.2.1 问题

Multi-view generation的naive approach是parallel generation：
```
s_{t+k}^h ~ T(·|s_t^h, s_t^w, a)   # 独立生成head view
s_{t+k}^w ~ T(·|s_t^h, s_t^w, a)   # 独立生成wrist view
```

问题：两个views可能physically inconsistent。比如head view显示arm在左边，wrist view显示arm在右边。

#### 2.2.2 Solution: Causal decomposition

$$s_{t+k}^h \sim T_\theta(\cdot | s_t^h, s_t^w, a_{t:t+k-1})$$
$$s_{t+k}^w \sim T_\theta(\cdot | s_t^w, s_{t+k}^h)$$

**Intuition**: Head view包含全局信息，wrist view包含local detail。先predict head view（全局布局），再conditioned on predicted head view predict wrist view（局部细节）。这建立了view间的causal dependency。

#### 2.2.3 Attention mask实现

Figure 10展示了causal attention mask的设计：

```
Token sequence: [ViT(s_t^h), VAE(s_t^h), ViT(s_t^w), VAE(s_t^w), 
                 Action_chunk, ViT(s_{t+k}^h), VAE(s_{t+k}^h), 
                 ViT(s_{t+k}^w), VAE(s_{t+k}^w)]
```

Attention matrix的关键设计：
- $s_{t+k}^h$ 可以attend to $s_t^h, s_t^w, a$ (所有历史信息)
- $s_{t+k}^w$ 可以attend to $s_t^w, s_{t+k}^h, a$ (包括已生成的head view)
- $s_{t+k}^w$ 不能attend to $s_{t+k}^h$ 的generation过程（避免reverse causality）

这种设计在UMM的transformer架构中很容易实现，只需要修改attention mask。

#### 2.2.4 Ablation验证

Table 1的ablation:
- **w/o IVD**: LPIPS wrist view从0.254恶化到0.454，SSIM从0.751降到0.559
- **w/o PT (pretraining)**: 全面崩溃，LPIPS head view从0.094到0.281

Figure 12的qualitative example很有意思：即使predicted head view偏离ground truth（right hand位置不同），predicted wrist view与predicted head view保持一致，而不是与ground truth一致。这证明了IVD确实enforce了cross-view consistency。

### 2.3 Chunk-level Branched Rollout

#### 2.3.1 Compounding error的本质

MBRL的经典问题：world model的prediction error随rollout length累积。

$$\epsilon_{total}(n) \approx n \cdot \epsilon_m$$

其中 $n$ 是rollout length，$\epsilon_m$ 是single-step model error。

在VLA的sparse reward setting下，这个问题更严重：
- Step 1-5: world model预测正确，robot接近object
- Step 6-10: 小的prediction error累积，world model预测robot已经grasp成功
- Reality: robot其实没grasp到

Policy在world model中看到success reward，但real-world执行失败。

#### 2.3.2 MBPO的classic idea + chunk-level extension

**Classic MBPO** (Janner et al., 2019): 从offline dataset中的任意state开始short branched rollout (length ~1-5 steps)，而不是从initial state开始full-horizon rollout。

**VLA-MBPO的extension**:
1. 从offline dataset的任意observation开始
2. World model在chunk level操作，所以rollout length = $n \cdot k$ (n chunks, k steps per chunk)
3. 相对于step-level，effective horizon减少 $1/k$

#### 2.3.3 Advantage estimation

Paper的Equation 4定义了chunk-level branched rollout的GAE：

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{i=0}^{n} (\gamma^k \lambda)^i \mathcal{T}_t^V$$

其中：
$$\mathcal{T}_t^V = \sum_{j=1}^{k} \gamma^{j-1} r(s_{t+j}, l) + \gamma^k V_\phi(s_{t+k}, l) - V_\phi(s_t, l)$$

**变量解释**:
- $n$: branched rollout length (chunks)
- $k$: chunk size (steps per chunk)
- $\gamma$: discount factor
- $\lambda$: GAE的bias-variance tradeoff
- $\mathcal{T}_t^V$: chunk-level TD error
- $V_\phi$: value function (MLP head on VLA)

**Intuition**: 
- $(\gamma^k)^i$ 而不是 $\gamma^i$，因为每个macro-step实际跨越k个micro-steps
- $\sum_{j=1}^k \gamma^{j-1} r(s_{t+j})$ 是chunk内累计reward
- Branched rollout限制model error累积在 $n$ chunks内

#### 2.3.4 Trajectory stitching

Figure 3展示了value model的学习dynamics。虽然training只用short branched rollouts (n=2 chunks)，value model逐渐学会预测full-horizon return。

**Mechanism**: 不同branched rollouts从不同states开始，覆盖trajectory的不同部分。Value model通过GAE的bootstrap (via $V_\phi(s_{t+k})$)将local estimates stitching成global estimates。这就是"generalizable credit assignment"。

---

## 3. 理论分析：Value Gap Bounds

### 3.1 Theorem 4.1: 传统方法的value gap

**Setting**: Chunk-level policy $\pi^k$ evaluated in **step-level** world model $\hat{T}$ with **full-horizon** rollout。

$$|V(\pi^k) - \hat{V}(\pi^k)| \leq \frac{2r_{max}}{1-\gamma} \left[ \frac{2\gamma^k}{1-\gamma^k}\epsilon_\pi^k + 2\epsilon_\pi^k + \frac{k\gamma^k}{1-\gamma^k}\epsilon_m \right]$$

**变量定义**:
- $V(\pi^k)$: true value of policy $\pi^k$
- $\hat{V}(\pi^k)$: value estimated in learned world model
- $r_{max}$: max reward
- $\gamma$: discount factor
- $k$: chunk size
- $\epsilon_\pi^k = \max_s D_{TV}(\pi_D^k(\tilde{a}|s) \| \pi^k(\tilde{a}|s))$: policy divergence from behavior policy
- $\epsilon_m = \max_t \mathbb{E}_{s \sim D^t}[D_{TV}(T(s'|s,a) \| \hat{T}(s'|s,a))]$: step-level model error
- $D_{TV}$: total variation distance

**三项含义**:
1. $\frac{2\gamma^k}{1-\gamma^k}\epsilon_\pi^k$: policy error通过chunk积累的效应
2. $2\epsilon_\pi^k$: 直接policy error
3. $\frac{k\gamma^k}{1-\gamma^k}\epsilon_m$: **model error被k放大** — 这是因为chunk-level policy在step-level model中evaluate时，每个chunk包含k个step-level transitions，model error累积k倍

### 3.2 Theorem 4.2: VLA-MBPO的value gap

**Setting**: Chunk-level policy $\pi^k$ evaluated in **chunk-level** world model $\hat{T}^k$ with **n-chunk branched rollout**。

$$|V(\pi^k) - \hat{V}^{branch}(\pi^k)| \leq \frac{2r_{max}}{1-\gamma} \left[ \frac{(\gamma^k)^{n+1}}{1-\gamma^k}\epsilon_\pi^k + (\gamma^k)^n \epsilon_\pi^k + n\epsilon_m^{k,n} \right]$$

**变量定义**:
- $\hat{V}^{branch}(\pi^k)$: value estimated via branched rollout in chunk-level world model
- $n$: branched rollout length (chunks)
- $\epsilon_m^{k,n} = \max_{t \leq n} \mathbb{E}_{s \sim d_{t,s_0 \sim D}^{\pi^k}}[D_{TV}(T^k(s'|s,\tilde{a}) \| \hat{T}^k(s'|s,\tilde{a}))]$: chunk-level model error在branched rollout内

**关键差异**:
1. Policy error项: $(\gamma^k)^{n+1}$ 而不是 $\gamma^k$ — **指数衰减**，因为branched rollout截断了long-horizon dependency
2. Model error项: $n\epsilon_m^{k,n}$ 而不是 $\frac{k\gamma^k}{1-\gamma^k}\epsilon_m$ — **线性增长**而不是被k放大

### 3.3 Case Study: 具体数字

$\gamma = 0.99, k = 10$:

**传统方法**:
$$\frac{2\gamma^k}{1-\gamma^k} = \frac{2 \times 0.99^{10}}{1 - 0.99^{10}} \approx \frac{2 \times 0.9044}{0.0956} \approx 18.93$$
$$\frac{k\gamma^k}{1-\gamma^k} = \frac{10 \times 0.9044}{0.0956} \approx 94.64$$

加上前面的系数 $\frac{2r_{max}}{1-\gamma} = \frac{2r_{max}}{0.01} = 200r_{max}$:

- Policy error: $200 \times (18.93 + 2) \approx 4186 \epsilon_\pi^k$ (paper说4183)
- Model error: $200 \times 94.64 \approx 18928 \epsilon_m$ (paper说18916)

**VLA-MBPO (n=2)**:
$$(\gamma^k)^3 = 0.9044^3 \approx 0.740$$
$$(\gamma^k)^2 = 0.9044^2 \approx 0.818$$

- Policy error: $200 \times (\frac{0.740}{0.0956} + 0.818) \approx 200 \times (7.74 + 0.818) \approx 1712 \epsilon_\pi^k$ (paper说1710)
- Model error: $200 \times 2 \times \epsilon_m^{k,n} = 400 \epsilon_m^{k,n}$

**对比**:
- Policy error: $4183 \rightarrow 1710$ (2.4x reduction)
- Model error: $18916 \rightarrow 400$ (**47x reduction!**)

Model error的reduction尤其dramatic，这正是chunk-level world model + branched rollout的核心价值。

### 3.4 证明的intuition

**Lemma A.1**: Joint distribution的TV distance ≤ marginal TV + conditional TV max
$$D_{TV}(\mathbb{P}_1(x,y) \| \mathbb{P}_2(x,y)) \leq D_{TV}(\mathbb{P}_1(x) \| \mathbb{P}_2(x)) + \max_x D_{TV}(\mathbb{P}_1(y|x) \| \mathbb{P}_2(y|x))$$

**Lemma A.2**: Rollout state distribution的TV distance线性增长
$$D_{TV}(\mathbb{P}_1(s_t) \| \mathbb{P}_2(s_t)) \leq t \delta$$

这是compounding error的数学根源。Branched rollout通过限制 $t \leq n$ 来bound这个growth。

**Lemma A.3**: Chunk-level的value divergence bound。通过reformulate为temporally-extended MDP $\mathcal{M}^k = (\mathcal{S}, \mathcal{A}^k, \tilde{T}, \tilde{r}, \gamma^k)$，利用discount factor $\gamma^k$ 的几何级数求和。

**Lemma A.4**: Branched rollout的value divergence。将trajectory分为pre-branch和post-branch，分别bound error。

---

## 4. Algorithm: VLA-MBPO

### 4.1 Algorithm 1 详解

```
Input: VLA Model {π_φ, V_φ}, World Model {T_θ, r_θ}, replay buffer D

1. Data Collection:
   Run π_φ in real environments → add to D
   
2. World Model Fine-tuning:
   Fine-tune {T_θ, r_θ} on D
   
3. Policy Optimization in World Model:
   for j = 1 to N_RL_iter:
     6. Sample {s_t}^M from D
     7. Generate chunk-level branched rollouts via T_θ, r_θ
     8. Run Flow-Noise to update {π_φ, V_φ} on synthetic data
```

### 4.2 Flow-Noise: PPO for flow matching policies

VLA models (如 $\pi_0$)用flow matching生成actions，不是传统的Gaussian policy。Flow-Noise是PPO的variant，处理flow matching的log-likelihood estimation。

**Log-likelihood decomposition** (Equation 5):
$$\log \pi_\phi(\mathcal{A}|s_t) = \log\left(\pi_\phi(A^0|s_t) \prod_{i=0}^{K-1} \pi_\phi(A^{\tau_{i+1}}|A^{\tau_i}, s_t)\right)$$

其中：
- $\mathcal{A} = (A^0, ..., A^1)$: action chunk的denoising sequence
- $A^0$: initial noise
- $A^1$: final action
- $A^{\tau_i}$: intermediate denoising step
- $K$: denoising steps (paper用3)

这是flow matching的概率密度分解，通过chain rule将joint distribution分解为conditional distributions。

### 4.3 为什么不需要conservative regularization

Paper的一个重要claim：VLA-MBPO不需要MOPO/MOBILE那样的conservative regularization。

**理由**:
1. UMM-World通过pretraining已经足够accurate，model bias小
2. Branched rollout本身限制了model exploitation
3. 简化了系统，hyperparameter不需要per-task tuning

---

## 5. 实验结果详解

### 5.1 World Model Evaluation (Table 1)

| Model | Head LPIPS↓ | Head PSNR↑ | Wrist LPIPS↓ | Wrist PSNR↑ | Inf Time↓ | Reward ACC↑ |
|-------|------------|-----------|-------------|------------|-----------|------------|
| Ctrl-World | 0.150 | 21.95 | 0.435 | 13.87 | 21 | - |
| Qwen3-VL-8B | - | - | - | - | - | 97.0 |
| UMM-World | **0.094** | **23.29** | **0.254** | **18.76** | **10** | **98.4** |
| w/o IVD | 0.116 | 21.71 | 0.454 | 13.38 | 8 | 98.5 |
| w/o PT | 0.281 | 19.26 | 0.579 | 12.80 | 10 | 94.5 |

**Key observations**:
1. UMM-World在dynamics prediction上全面优于video model (Ctrl-World)
2. Inference speed 2x faster (frame-skipping)
3. Reward prediction匹配specialist VLM (Qwen3-VL)
4. IVD对wrist view至关重要 (LPIPS 0.254 → 0.454)
5. Pretraining是foundation (w/o PT全面崩溃)

### 5.2 LIBERO Benchmark (Table 2)

| Model | Spatial | Object | Goal | Long | Avg |
|-------|---------|--------|------|------|-----|
| $\pi_{0.5}$ (SFT) | 78.2 | 88.6 | 85.0 | 54.6 | 76.8 |
| BC (WM) | 80.6 | 85.8 | 92.4 | 48.6 | 76.0 |
| $\pi_{RL}$ | 86.0 | 89.8 | 90.8 | 61.2 | 82.6 |
| IDQL | 79.0 | 92.4 | 86.4 | 52.2 | 77.5 |
| VLA-MBPO | **87.8** | **92.8** | **96.6** | **66.8** | **85.9** |

**Key observations**:
1. VLA-MBPO在所有suite上都improve
2. Long-horizon tasks (LIBERO-Long)提升最大 (+12.2)
3. Online RL ($\pi_{RL}$)虽然也improve但需要real-world interactions
4. BC(WM)在Long suite上反而degrade (48.6 vs 54.6)，说明full-horizon rollout在long-horizon上fail

### 5.3 Real-world Tasks (Figure 5)

5个tasks在两个robot platforms:

**Arx-X5 (bimanual, 14 DoF)**:
- Plug Cable: sub-centimeter precision, 3mm socket
- Fold Towel: deformable object manipulation

**Galaxy-R1 (whole-body, 21 DoF)**:
- Pick Cup: pick-and-place with viewpoint disturbance
- Insert Pen: fine motor skills
- Wipe Board: mobile manipulation with partial observability

每个task: 50 trajectories (30 seen + 20 unseen configurations)

VLA-MBPO在所有tasks上都show consistent improvement，特别是在unseen configurations上，证明generalization能力。

### 5.4 Ablation Studies

#### Rollout Scheme (Table 3)

| Rollout Scheme | 1 chunk | 2 chunks | 4 chunks | Full Horizon |
|--------------|---------|----------|----------|-------------|
| VLA-MBPO | 63.9 | **66.8** | 62.9 | 52.8 |

- n=1: too short, 限制exploration和trajectory stitching
- n=2: optimal balance
- n=4: model error累积过多
- Full horizon: 完全fail，compounding error不可控

#### Sample Size (Figure 6)

Success rate随generated sample size单调增加。更多synthetic data → 更accurate value estimation → 更stable policy improvement。

---

## 6. Failure Cases (Figure 13, 14)

### 6.1 Partial Observability

**Wipe Board task**: 当arm移出head camera视野，world model无法渲染arm。虽然wrist camera有local feedback，但缺乏kinematic history，无法infer global arm posture。

**Navigation phase**: robot旋转base面对whiteboard时，target object首次进入视野。World model无法从"nothing" hallucinate正确的geometry和texture。

### 6.2 Large Physical Movements

**Wipe Board**: 大幅end-effector位移导致"motion collapse" — model保守预测minimal change，arm看起来static。

**Fold Towel**: deformable object的大幅configuration变化，model succumb to hallucination，生成physically implausible的towel configurations。

---

## 7. Implementation Details

### 7.1 Hyperparameters (Table 4, 5)

**UMM-World**:
- lr: 2e-5
- Cross entropy weight: 0.01 (for text tokens)
- MSE weight: 1.0 (for image reconstruction)
- CFG interval: [0.4, 1.0] (classifier-free guidance)
- Text scale: 6, Image scale: 2

**RL**:
- Actor lr: 5e-6, Critic lr: 1e-4
- γ=0.99, λ=0.95, clip ratio ε=0.1
- Action chunk H=10, Denoise steps=3, Noise level=0.5
- **Sample size**: 512 (LIBERO Spatial/Object/Goal), 1280 (LIBERO-Long)
- **Update to data**: 20 (most), 50 (LIBERO-Long)

**Key insight**: 除了Long-horizon task需要更大sample size，其他hyperparameters完全一致。这是practical deployment的重要优势。

### 7.2 Computational Cost

- 8× NVIDIA H100 GPUs
- UMM-World training: 7-8 hours
- Policy optimization: 4-6 hours

### 7.3 Real-world Data Collection

- SFT: 50-100 expert trajectories per task (human teleoperation)
- RL: 50 self-collected on-policy trajectories per task
- Evaluation: 50 trials (30 seen + 20 unseen)

---

## 8. 与Related Work的对比

### 8.1 vs MBPO (Janner et al., 2019)

| Aspect | MBPO | VLA-MBPO |
|--------|------|----------|
| State space | Low-dimensional | Pixel-level |
| World model | Step-level | Chunk-level |
| Policy | Gaussian | Flow matching |
| Rollout | Step-level branched | Chunk-level branched |

### 8.2 vs WMPO (Zhu et al., 2025)

| Aspect | WMPO | VLA-MBPO |
|--------|------|----------|
| World model | Video generation model | UMM |
| Rollout | Full-horizon | Branched |
| Model error | Quadratic growth | Linear growth |
| Long-horizon | Vulnerable | Robust |

### 8.3 vs $\pi_{RL}$ (Chen et al., 2025a)

| Aspect | $\pi_{RL}$ | VLA-MBPO |
|--------|-----------|----------|
| Training | Online RL | World model RL |
| Real interactions | High | Low (50 rollouts) |
| Sample efficiency | Low | High |

---

## 9. Critical Analysis & Limitations

### 9.1 Strengths

1. **Practical**: 单一hyperparameter set across tasks
2. **Theoretical**: 清晰的value gap analysis，量化了每个design choice的贡献
3. **Empirical**: Simulation + real-world验证
4. **Efficient**: UMM的frame-skipping带来2x speedup

### 9.2 Limitations (paper承认的)

1. **Sample generation cost**: 虽然比video model快，但仍需大量compute生成synthetic data
2. **Pretraining gap**: UMM没有在action-labeled robotic data上pretrain，仍需少量数据fine-tune world model
3. **Partial observability**: 无法处理视野外的object generation
4. **Large motions**: Motion collapse问题未解决

### 9.3 My additional observations

1. **Reward model的binary limitation**: Reward prompt强制"Yes/No"输出，对于partial success无法提供graded feedback，可能limit credit assignment的granularity

2. **Chunk-level reward的discounting**: $r_\theta(s_{t+k}, l) = \sum_{i=1}^{k} \gamma^{i-1} r(s_{t+i}, l)$ 假设reward model能预测整个chunk的累计reward。如果reward只发生在chunk中间某一步，UMM是否能准确预测accumulated reward？

3. **Interleaved decoding的order dependency**: 先predict head view再predict wrist view。如果head view prediction error大，wrist view也会被contaminate。是否有alternative的joint prediction方案？

4. **Value function的stitching**: Figure 3显示value model能学习long-horizon dependencies，但这是否依赖于offline dataset的coverage？如果dataset只包含short trajectories，stitching还能work吗？

5. **Generalization to new embodiments**: World model在特定robot上fine-tune。换robot需要重新fine-tune world model，这限制了zero-shot transfer。

---

## 10. 未来方向

基于paper的limitations和我的分析：

1. **Action-conditioned pretraining**: 在大规模robotic data上pretrain UMM-World，实现zero-shot world modeling
2. **Hierarchical rollout**: Adaptive rollout length based on world model uncertainty
3. **Multi-modal rewards**: Beyond binary reward，支持graded success和intermediate progress
4. **Active data collection**: 用world model uncertainty指导real-world data collection
5. **Cross-embodiment transfer**: 用shared latent space实现world model的跨robot transfer

---

## References

- [VLA-MBPO Paper](https://arxiv.org/abs/2509.23014) - 本文
- [MBPO (Janner et al., 2019)](https://arxiv.org/abs/1906.08253) - Model-based policy optimization
- [MOPO (Yu et al., 2020)](https://arxiv.org/abs/2005.13239) - Offline model-based RL with uncertainty penalty
- [Bagel (Deng et al., 2025a)](https://arxiv.org/abs/2506.08225) - Unified multimodal pretraining
- [$\pi_0$ (Black et al., 2024)](https://arxiv.org/abs/2410.24164) - VLA flow model
- [$\pi_{RL}$ (Chen et al., 2025a)](https://arxiv.org/abs/2504.16057) - Online RL for flow-based VLA
- [LIBERO (Liu et al., 2023)](https://arxiv.org/abs/2306.03310) - Benchmark for lifelong robot learning
- [Ctrl-World (Guo et al., 2025)](https://arxiv.org/abs/2505.15464) - Controllable generative world model
- [WMPO (Zhu et al., 2025)](https://arxiv.org/abs/2505.19060) - World model-based policy optimization for VLA
- [Park et al., 2025](https://arxiv.org/abs/2505.11832) - Scalable offline MBRL with action chunks
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Lipman et al.
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) - Proximal policy optimization
- [GAE (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438) - Generalized advantage estimation

---

## 总结

VLA-MBPO的核心贡献是将三个insight cohesively结合：

1. **UMM as world model**: 利用pretrained multimodal model的vision+language能力，避免separate dynamics和reward model的complexity。Frame-skipping带来efficiency。

2. **Interleaved view decoding**: 通过causal decomposition (head→wrist) enforce multi-view consistency，这对fine-grained manipulation至关重要。

3. **Chunk-level branched rollout**: 将MBPO的classic idea扩展到chunk level，理论上将model error growth从quadratic (with k amplification) 降到linear (without k amplification)。

理论分析（Theorem 4.1 vs 4.2）清晰地量化了每个design choice的贡献，case study显示47x的model error reduction。实验在simulation和real-world上都validate了framework的有效性，特别是single hyperparameter set across tasks的practical advantage。

这为VLA models的post-training提供了一个scalable、safe、sample-efficient的path，避免了real-world RL的高成本和风险。
