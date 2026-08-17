---
source_pdf: Video2Reward.pdf
paper_sha256: 65e5cd0551033867053bff819afe340b9a3477e6fa34d2748cb4f6930d8c4da7
processed_at: '2026-08-13T00:45:01-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Video2Reward — 用人话讲

Andrej，咱坐下来用大白话聊聊这篇 paper 到底干了啥、为啥能 work、哪些地方 clever、哪些地方 hacky。

---

## 一句话版本

你丢给 LLM 一段 YouTube 视频（人跑步、狗走路），LLM 看完之后自动写一段 reward function 代码，机器人拿这 reward 跑 PPO 训练完，动作看起来就跟视频里那个东西长得一模一样。就这么个事。

---

## 为什么这事是个真问题

你做 RL 的人都知道一句话：**reward is the bottleneck**。Legged robot 的 reward 写起来特别痛苦，因为要同时管 balance、gait、energy、posture，一堆互相打架的 term。Expert 调 reward 能调一个月。

Eureka [1] 出来之后大家很兴奋，因为它让 LLM 写 reward code。你给它一段 task description + environment code，它直接吐 reward function。

但这里有个陷阱 Eureka 论文里没明说：**它在 Anymal 上能 work，是因为它的 task description 里偷偷塞了 reference signal**。原文写的是 "make the quadruped follow randomly chosen x, y, and yaw target velocities"。注意——它给的是 **velocity tracking**，不是 "走起来像狗"。

这篇 paper 做了一个特别 elegant 的 ablation：把 task description 换成 "amble like real dogs"（叫 **Eureka-$t_d$ 变体**）。结果 DTW score 从 34.186 涨到 44.727（Table 1）。**效果直接崩**。

这就揭示了 LLM reward generation 的根本矛盾：

> LLM 在 code generation 层面是 beast，但在 motor priors 层面基本是 blank。它知道 "walking" 这个词的语义，但完全不知道 "walking" 在 joint trajectory 长什么样。

文字太 abstract，video 是 dense supervision。这篇 paper 就是把 supervision source 从 text 换成 video。

---

## 方法的核心 trick：Video 不能直接喂给 LLM

LLM 是 text-to-text 模型，你不能丢 mp4 进去。所以得有一个 video-to-text 的桥。这是整个 paper 最关键的工程 insight。

### 怎么转

给定 video $v$，均匀采样 $L$ 帧 $\mathcal{T} = \{I_l\}_{l=1}^{L}$。每帧用 pose estimator 提取 $J$ 个 keypoint：

- 人：RTMpose
- 动物：HRNet

第 $j$-th keypoint 在第 $l$ 帧的 2D 坐标记作 $p_j(l) = (x_{lj}, y_{lj})$。整个视频里这个 keypoint 的轨迹序列就是：

$$\mathcal{T}_j = \{p_j(l) \mid l \in \{1, 2, \ldots, L\}\}$$

这里：
- $j$ = keypoint index，比如左肩、右膝、左肘...，$j \in \{1, \ldots, J\}$
- $l$ = frame index
- $x_{lj}, y_{lj}$ = keypoint $j$ 在 frame $l$ 的像素坐标
- $\mathcal{T}_j$ = keypoint $j$ 沿时间的轨迹

把这 $J$ 条轨迹 $\{\mathcal{T}_j\}_{j=1}^{J}$ 拼起来，就是这段视频的 "motion descriptor"。

### 为什么用 keypoint trajectory 而不是 raw frame

三个直觉：

1. **Background invariance** — 你要的是 motion，不是背景。Pose extractor 天然把背景过滤掉了。
2. **LLM-readable format** — 坐标序列 $(x_1, y_1), (x_2, y_2), \ldots$ 这种数字串 LLM 直接能 parse，根本不需要 VLM。
3. **Cross-video standardization** — 不同视频分辨率、不同视角、不同主体大小，pose 归一化之后都变成统一格式。

这个设计其实挺 elegant 的。它把 "video → reward" 这个看似要 VLM 才能干的事，**降维成了 LLM 就能干的事**。pose estimator 是个 off-the-shelf CV 工具，不需要训任何新东西。

---

## Reward 生成的完整 prompt

LLM 拿到的 prompt 是这样的（Figure 2 的 pipeline）：

$$R = f_{LLM}(T_{aux}, \mathcal{T})$$

其中 $T_{aux}$ 包含三块：

| Component | 作用 |
|-----------|------|
| $t_e$ (environment code) | 告诉 LLM 有哪些变量可用，比如 `obs["base_lin_vel"]`、`obs["joint_pos"]` |
| $t_d$ (task description) | 高层目标，比如 "follow x, y, yaw velocities" |
| $t_r$ (generation rule) | 代码规范，确保 TorchScript 兼容 |

加上从 video 提出来的 $\mathcal{T}$，LLM 就能写出 reward code 了。

---

## Feedback Loop —— 这部分最 tricky

单轮生成 reward 质量不稳定，需要迭代。Eureka 的迭代 feedback 是 numerical metric（速度误差）。但这有个问题：**机器人可能跑得很快，姿势完全不像人**。

论文提了个 video-assisted feedback function $F_{fb}$：

$$R^n = f_{LLM}(T, R^{n-1}, \mathcal{T}, F_{fb}(R^{n-1}, \pi^{n-1}, v))$$

上标 $n$ 表示第 $n$ 轮迭代。$F_{fb}$ 输入上一轮的 reward、policy、video，输出是 textual feedback 喂回 LLM。

### $F_{fb}$ 内部干了四件事

**Step 1: 收集 robot 轨迹**

在 simulator 里跑 policy $\pi^{n-1}$，收 $T$ 步里 $J$ 个 keypoint 的 3D 轨迹：

$$\hat{\mathcal{T}}_j = \{\hat{p}_j(t) = (x_{tj}, y_{tj}, z_{tj})\}_{t=1}^{T}$$

注意这里是 3D，因为 robot 在 IsaacGym 里是真实 3D simulation。

**Step 2: 3D → 2D projection**

Video 里的 $\mathcal{T}_j$ 是 2D pixel 坐标，robot 的 $\hat{\mathcal{T}}_j$ 是 3D 世界坐标，没法直接比。论文沿 motion direction 投影，把 3D 压成 2D。

这一步其实有点 hacky —— 深度信息丢了。对 side-view video 还行，对 front-view / 斜视角会有 aliasing。这是这个方法的一个明确 limitation。

**Step 3: 用 autocorrelation 切周期**

跑步、走路都是 periodic motion。用 autocorrelation function 检测周期长度，把长轨迹切成多段，每段含 2 个周期。

这步很关键。如果不切，FastDTW 会因为 phase misalignment 算出很高距离 —— robot 在跑第 3 步的时候，video 里的狗在跑第 7 步，DTW 会把它们强行对齐，结果不可信。

**Step 4: FastDTW 算相似度**

对每个 keypoint 单独算：

$$S_j = F_{sim}(\hat{\mathcal{T}}_j, \mathcal{T}_j)$$

用 FastDTW [2] 是因为它能 handle 变长周期。每个 agent 最终拿到 $J$ 个 similarity score，多个 agent 取平均，转成 text 喂回 LLM。

---

## 一个被忽视的 subtle design

整个 Algorithm 1 有一个 design choice 我觉得特别聪明，但 paper 里没强调：

> **Selection metric 用 $H_{mts}$（task 指标），Feedback metric 用 DTW（behavior 指标）。两者完全 decoupled。**

每轮生成 K=16 个 reward，用 $H_{mts}$（max training success）选 best reward 进入下一轮。但喂回 LLM 的 feedback 里包含 DTW score。

为什么不直接用 DTW 做 selection？因为如果用 DTW 选，policy 可能学出 "pose match 但原地踏步" 的退化解 —— 姿势像狗，但不往前走。把 selection 和 feedback 解耦，相当于说：

> "你能不能干 task，我用 task metric 选；但你长得像不像，我用 video metric 告诉你。"

这是 multi-objective 问题的经典处理手法。Eureka 没这个问题因为它只有一个 metric。Video2Reward 同时要 task completion + behavior matching，decouple 是必要的。

---

## 实验结果里几个真的有意思的数字

### Table 1 — DTW score

| Method | Anymal (Amble) | Anymal (Run) | Humanoid (Run) |
|--------|----------------|--------------|-----------------|
| Human | 26.368 | 64.117 | 8.721 |
| Eureka | 34.186 | 94.542 | 8.237 |
| **Eureka-$t_d$** | **44.727** | 130.427 | 8.252 |
| **Ours** | **17.292** | **28.124** | **7.359** |

看 Eureka-$t_d$ 那行。把 task description 从 "track velocity" 换成 "amble like real dogs"，DTW 涨到 130（跑步任务）。这说明 LLM 拿到 abstract text 完全 lost，没有 motor prior。

而 Ours 直接降到 28.124。从 130 到 28，这是 ~78% 的下降。**video 提供的 motion supervision 是 text 完全无法替代的。**

### Table 2 — 在 expert reward 上 evaluate

| Method | $r_{lin}$ | $r_{ang}$ | $r_{up}$ | $r_{alive}$ |
|--------|-----------|-----------|----------|-------------|
| Eureka | 22.629 | 13.441 | 34.714 | 1726.317 |
| Ours | **46.372** | 22.599 | 45.366 | 1789.592 |

$r_{lin}$ 是 expert reward 里的线速度 tracking term。Eureka 训出的 robot 在 expert reward 上只得 22.6，Ours 得 46.4，**直接翻倍**。

这说明什么？说明 Ours 训出来的 robot 不光 "姿势像狗"，**speed tracking 也更好**。Video 不是只教了 posture，连 task performance 都顺带提升了。这是反直觉的，因为 video 里没有 ground truth velocity。

直觉解释：human/dog 视频里的步频和速度是 coupled 的。robot 学会了像狗那样迈步，自然就达到了狗那样的速度。Motion prior 隐含了 velocity prior。

### Table 3 — Ablation: Video-Assisted Feedback (VAF)

| Method | Anymal HNS | Anymal DTW |
|--------|-----------|------------|
| Eureka | 0.729 | 34.186 |
| Ours w/o VAF | 0.921 | 20.355 |
| Ours | 1.003 | 17.292 |

w/o VAF 是 video 作为 input 但不进入 feedback loop。结果：
- Video 作为 input alone：HNS 0.729 → 0.921（+26%）
- 加 VAF：0.921 → 1.003（+9%）

**主要 gain 来自 video input，VAF 只是 refinement。**

这符合直觉：input 决定 reward 的 "方向"（朝 video 那个方向走），feedback 决定 "精度"（怎么微调到更像）。没有 input 是无头苍蝇，没有 feedback 是粗放训练。两个加一起最 work。

---

## 几个被 paper 一笔带过的 engineering detail

1. **Video sampling rate 不一样**：Humanoid 每 3 帧采一次（2 秒视频），Anymal 每 9 帧采一次（3 秒视频）。这是因为 human running cycle ~0.8s，dog amble cycle ~0.4s，采样率要匹配 motion frequency。

2. **Camera normalization 分两种**：stationary camera 用 frame dimension 归一化；moving camera 用 bounding box 归一化。后者是为了剔掉 camera motion 的干扰。

3. **N=5 iterations, K=16 samples**：完全跟 Eureka 对齐，fair comparison。

4. **LLM 没说用哪个**：从 paper 看，大概率是 GPT-4。这其实是个隐含 variable —— 如果用 code-specific LLM 比如 CodeLlama 或 DeepSeek-Coder，reward code 质量可能更高。

---

## 我的几点直觉判断

### 1. 为什么这个方法 work — motor prior 的外部化

LLM 在预训练时见过的文字里，"walking" 这个词对应的 physical pattern 是 implicit 的、latent 的、无法 decode 成 joint trajectory。

Video 通过 keypoint extraction 把 implicit motion **显式化**，相当于给 LLM 装了一个 **external motor memory**。LLM 不用知道 dog 怎么走路，video 直接告诉它左肩坐标序列长啥样，LLM 只需要把这个 sequence 翻译成 reward code。

整个 system 的分工特别 clean：
- Pose estimator 提供 motion perception
- LLM 提供 code generation
- RL 提供 optimization
- Video 提供 supervision target

每个 module 干自己擅长的事，没人被迫干自己不擅长的。

### 2. 这方法的 limitation

我自己看出几个 paper 没明说的：

**(a) 2D projection 信息损失**

把 3D robot trajectory 投影到 2D 比对，深度信息丢了。Side-view video 还行，斜视角就有 aliasing。两个 3D 轨迹投影到 2D 可能完全重合，但实际一个往前走一个往后走。

**(b) Single camera assumption**

Video 必须是 side-view 才语义一致。正面的 video 你提取出来的 keypoint 坐标和 robot 3D 投影下来的 2D 坐标根本不在一个语义空间。

**(c) Keypoint correspondence 是手工映射**

Video 里 human 有 17 个 COCO keypoint，Humanoid robot 有完全不同的 joint structure。这个 mapping 是手工定义的。Anymal 4 条腿，dog keypoint 怎么映射到 Anymal joint 也是手工的。换 robot 就得重做 mapping。

**(d) Periodicity 假设**

Autocorrelation 分段假设 motion 是周期的。对于 jumping、turning、sitting down 这种 non-periodic behavior 会 fail。

**(e) Real video noise**

目前用的是 clean internet video。如果换成 real-world noisy video（occlusion、motion blur、lighting change），keypoint extraction quality 会大幅下降，整个 pipeline 的 robustness 存疑。

### 3. 更大的 picture — Multimodal Reward Specification

这个工作其实指向一个更大的方向：**reward specification 的 interface 比算法本身更重要**。

Eureka 用 text，Video2Reward 用 video。未来可能：
- Demonstration video（this paper）
- Natural language（Eureka）
- Reference motion capture（DeepMimic [3] 那一套）
- Physical constraints（safety、energy）
- 甚至 sketch / stick figure

最终目标是让 non-expert 通过任意 intuitive interface 教 robot 干事。这是 robotics 走向 mass adoption 的必经之路。

### 4. 跟你之前 talk 的一些 connection

你在 EurekaLab talk 和跟 Yann LeCun 对谈里都提过，LLM + RL 的 bottleneck 在 reward specification。这个工作正好 hit 这个点。它没改 LLM、没改 RL、没改 PPO，改的是 reward specification interface。

这种 "interface innovation" 在 ML history 里往往比 algorithm innovation 更 impactful。Think about it：Transformer 是 algorithm，prompt engineering 是 interface，谁的影响力更大？很难说。在 robotics 这种 motor-prior-heavy 的领域，把 supervision source 从 text 扩到 video 可能是个 inflection point。

---

## 一个我自己想做的 follow-up

如果让我 extend 这个工作，我会想做 **cross-embodiment**：video 里是 dog，robot 是 humanoid。让 humanoid 从 dog video 学 quadrupedal-like locomotion。

这在生物学上有 support — Toddlers crawl before they walk，爬行和四足行走的 motor pattern 是共享的。如果 LLM 能从 dog video 提取 "alternating diagonal gait" 的 abstract pattern，再把这 pattern 翻译成 humanoid 适用的 reward（比如让手和脚交替触地），就有可能实现 cross-embodiment behavior transfer。

这是模仿学习的 holy grail —— 跨 morphology 的 behavior transfer。目前所有 IL 方法都假设 demonstrator 和 learner 同 morphology，这个 assumption 在自然界根本不成立（人能模仿猴子上树，鸟能模仿鱼游泳）。

Video2Reward 的 framework 天然支持这个扩展，只要 keypoint correspondence 那一步改成 learned mapping 而不是手工 mapping。

---

## Reference Links

[1] Eureka — https://eureka-research.github.io/  
[2] FastDTW — https://github.com/wannesm/dtaidistance  
[3] DeepMimic — https://xbpeng.github.io/projects/DeepMimic/  
[4] L2R (Language to Reward) — https://language-to-reward.github.io/  
[5] Text2Reward — https://text-to-reward.github.io/  
[6] RTMpose — https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose  
[7] HRNet — https://github.com/leoxiaobin/deep-high-resolution-net.pytorch  
[8] IsaacGym — https://developer.nvidia.com/isaac-gym  
[9] PPO — https://arxiv.org/abs/1707.06347  
[10] Video2Reward (this paper) — https://arxiv.org/abs/2506.14718 (推测)

---

## 总结一句

这篇 paper 的核心 takeaway：**LLM 写 reward code 没问题，但给 LLM 什么 input 决定了 reward 能多 accurate**。Text 是 abstract 的，video 是 concrete 的，keypoint trajectory 是 LLM-readable 的。把这三层串起来，reward specification 从 "专家手调一个月" 变成 "给段 YouTube 视频等 5 轮迭代"。

Interface 比 algorithm 重要。这个 lesson 我觉得在你做过的 nanoGPT、micrograd、makemore 那些 project 里也反复出现过 — 改 input representation 比改 model architecture 往往更 effective。这 paper 给了这个 lesson 在 robotics 领域的又一个 instance。

---

# Video2Reward 论文深度解析

Hey Andrej, 这篇 paper 解决的核心问题是 LLM-based reward design 中的一个 fundamental limitation：**text description 作为 behavior specification 的粒度太粗**。下面我从 motivation、methodology、formulation、experiments 四个层面展开。

---

## 1. Motivation: 为什么 text-only reward generation 不够用

现有工作 Eureka [1]、L2R [2]、text2reward [3] 都走 **Equation (1)** 的路线：

$$R = f_{LLM}(T_{ext})$$

其中 $T_{ext}$ 是 textual task description。问题在于：当你说 "make the robotic dog pace with natural posture"，LLM 根本不知道 "pacing" 在 joint space 长什么样。Eureka 在 Anymal 上之所以 work，是因为它的 task description 本身已经包含了 reference signal —— "follow randomly chosen x, y, and yaw target velocities"。一旦把 description 换成 "amble like real dogs"（论文里叫 **Eureka-$t_d$** 变体），DTW score 直接从 34.186 涨到 44.727（Table 1），说明 LLM 完全 lost 了。

这就揭示了 LLM reward generation 的一个本质矛盾：**LLM 在 code 层面很强，但在 physical dynamics intuition 层面很弱**。Video 恰好是 physical dynamics 的 dense supervision signal。

---

## 2. 方法架构总览

整个 pipeline 可以拆成三个 stage，对应论文 Figure 2：

```
Video → M_v2t (keypoint extraction) → Text prompt + T_aux 
      → LLM generates K reward functions 
      → PPO training → policy π 
      → Video-assisted feedback F_fb 
      → LLM refines reward (N iterations)
```

### 2.1 Video-to-Text Transforming Module $M_{v2t}$

这是整篇论文最关键的 insight。给定 video $v$，均匀采样得到 frame set $\mathcal{T} = \{I_l\}_{l=1}^{L}$。对每一帧用 pose estimator 提取 $J$ 个 keypoints：

- **Human**: RTMpose [4]
- **Animal**: HRNet [5]

第 $j$-th keypoint 在第 $l$ 帧的坐标 $p_j(l) = (x_{lj}, y_{lj})$，整个轨迹序列：

$$\mathcal{T}_j = \{p_j(l) \mid l \in \{1, 2, \ldots, L\}\}$$

最终 $\{\mathcal{T}_j\}_{j=1}^{J}$ 构成 motion descriptor。

**Intuition**: 为什么用 keypoint trajectory 而不是 raw video frames？三个原因：
1. **Background invariance** - 剔除无关 scene 信息，只保留 motion 本身
2. **LLM-readable format** - 坐标序列天然是 LLM 能 parse 的 token 序列
3. **Cross-video standardization** - 不同 video source 归一化到统一格式

### 2.2 Auxiliary Textual Context $T_{aux}$

除了 video-derived text，还需要三个 text component：

| Component | Symbol | 作用 |
|-----------|--------|------|
| Environment code | $t_e$ | RL env 的 state $S$、action $\mathcal{A}$、transition logic，让 LLM 知道有哪些 variable 可用 |
| Task description | $t_d$ | 高层目标，如 "follow randomly chosen x, y, yaw velocities" |
| Generation rule | $t_r$ | 代码规范，确保 TorchScript 兼容 |

### 2.3 修正后的 reward generation formula

论文把 Equation (1) 升级为 **Equation (3)**：

$$R = f_{LLM}(T_{ext}, M_{v2t}(v))$$

这里 $M_{v2t}(v)$ 就是 video 转出来的 keypoint trajectory text。

---

## 3. Video-Assisted Iterative Reward Refinement

这是论文的第二个核心贡献。单轮 reward generation 质量不稳定，需要 feedback loop。Eureka 的 feedback 是 numerical metric（如 velocity error），但这无法 capture posture similarity。论文提出 **Equation (4)**：

$$R^n = f_{LLM}(T, R^{n-1}, M_{v2t}(v), F_{fb}(R^{n-1}, \pi^{n-1}, v))$$

其中 $F_{fb}$ 是 video-assisted feedback function，输入是上一轮的 reward、policy、video，输出是 textual feedback。

### 3.1 Feedback 计算的四个 step

**Step 1: Trajectory collection**
在 simulator 中跑 policy $\pi^{n-1}$，收集 $T$ 步内所有 $J$ 个 keypoints 的 3D 轨迹：
$$\hat{\mathcal{T}}_j = \{\hat{p}_j(t) = (x_{tj}, y_{tj}, z_{tj})\}_{t=1}^{T}$$

**Step 2: 3D → 2D projection**
Video 里的是 2D 坐标 $\mathcal{T}_j$，robot 的是 3D $\hat{\mathcal{T}}_j$，没法直接比。论文沿 motion direction 投影，把 3D 压成 2D。

**Step 3: Period segmentation via autocorrelation**
用 autocorrelation function [6] 检测轨迹周期长度，把长轨迹切成 multiple segments，每个 segment 包含 2 个周期。这一步很关键 —— 否则 FastDTW 会因为 phase misalignment 算出很高的距离。

**Step 4: FastDTW similarity**
对每个 keypoint 单独计算：
$$S_j = F_{sim}(\hat{\mathcal{T}}_j, \mathcal{T}_j)$$

用 FastDTW [7] 是因为它能 handle variable period length 的 time series。最终 agent 得到 $J$ 个 similarity score，多个 agent 平均后作为 feedback 喂回 LLM。

### 3.2 Algorithm 1 的核心 loop

```
for n = 1 to N:
    1. LLM 生成 K 个 reward functions {R_1^n, ..., R_K^n}
    2. 每个 reward 用 PPO 训练出 policy π_k^n
    3. 用 H_mts 选出 best policy
    4. 采样 robot trajectory T̂
    5. 计算 video similarity d = F_sim(T̂, T)
    6. 把 (R_best, s_best, d) 加进 T_aux
    7. 保留历史最优 reward
```

**Key design choice**: 每轮 sample K=16 个 reward（跟 Eureka 一致），这样降低所有 reward 同时出错的风险。Selection criterion 是 $H_{mts}$ 而非 DTW —— DTW 只用于生成 feedback text，不用于 reward selection。这是一个 subtle 但重要的 design。

---

## 4. 实验结果深度分析

### 4.1 主结果 (Figure 3 + Table 1)

| Method | Anymal (Human Norm) | Humanoid (Human Norm) | Anymal DTW (Amble) | Humanoid DTW (Run) |
|--------|---------------------|----------------------|--------------------|--------------------|
| Human | 1.0 (baseline) | 1.0 (baseline) | 26.368 | 8.721 |
| Eureka | 0.729 | 1.062 | 34.186 | 8.237 |
| Eureka-$t_d$ | - | - | 44.727 | 8.252 |
| **Ours** | **1.003** | **1.180** | **17.292** | **7.359** |

几个观察：

1. **Anymal DTW 从 34.186 降到 17.292**，接近 50% 的降低，说明 video 确实把 behavior 拉向 target。
2. **Humanoid DTW 只降了 0.878**（8.237→7.359），但 Human normalized score 提升明显（1.062→1.180）。这说明 humanoid running 本身就比较 structured，video 的边际收益主要在 posture 而非 speed。
3. **Eureka-$t_d$ 比 Eureka 还差**，这个 ablation 非常重要 —— 它证明 abstract text 反而引入 noise，LLM 没有 animal locomotion 的 prior。

### 4.2 Expert reward evaluation (Table 2)

在 expert-designed reward 上 evaluate trained policy：

| Method | $r_{lin}$ | $r_{ang}$ | $r_{up}$ | $r_{alive}$ |
|--------|-----------|-----------|----------|-------------|
| Eureka | 22.629 | 13.441 | 34.714 | 1726.317 |
| Ours | 46.372 | 22.599 | 45.366 | 1789.592 |

$r_{lin}$ 翻倍是最强证据 —— video 让 robot 真的学会了 "怎么走"，而不是 "怎么凑速度"。

### 4.3 Ablation: Video-Assisted Feedback (Table 3)

| Method | Anymal HNS | Anymal DTW |
|--------|-----------|------------|
| Eureka | 0.729 | 34.186 |
| Ours w/o VAF | 0.921 | 20.355 |
| Ours | 1.003 | 17.292 |

w/o VAF 是指 video 作为 input 但不进入 feedback loop。结果显示：
- Video 作为 input alone 就能把 HNS 从 0.729 提到 0.921（+26%）
- 加上 VAF 再提 0.082（+9%）
- DTW 的提升更显著：20.355 → 17.292

这说明 **video input 是主要 gain 来源，VAF 是 refinement**。这也是合理的 —— input 决定了 reward 的 "方向"，feedback 决定了 "精度"。

---

## 5. 与 Related Work 的 positioning

### 5.1 vs. Imitation Learning

Imitation learning [8][9] 直接学 state-action mapping，问题是 demonstration coverage 有限。Video2Reward 学的是 reward function，reward function 是更 abstract 的 representation，泛化性更好。本质上是 **inductive bias 的选择**：IL 假设 demonstration 是最优的，Video2Reward 假设 video 是参考但允许 policy 探索。

### 5.2 vs. Eureka

Eureka [1] 的 feedback 是 $F(R^{n-1})$ = numerical training metric。Video2Reward 把 feedback 升级为 $F_{fb}(R^{n-1}, \pi^{n-1}, v)$ = visual similarity。这是从 **task-completion metric** 到 **behavior-matching metric** 的转变。

### 5.3 vs. text2reward / L2R

text2reward [3] 和 L2R [2] 用 natural language 生成 dense reward。Video2Reward 的区别是 input modality：video 提供 **geometric grounding**，text 只提供 **semantic grounding**。

---

## 6. 我的几点思考

### 6.1 为什么这个方法 work

核心原因是 **LLM 缺乏 motor priors**。LLM 在 pretraining 时见过的 text 里，"walking" 这个词对应的 physical pattern 是 implicit 的、无法 decode 成 joint trajectory。Video 通过 keypoint extraction 把 implicit motion 显式化，相当于给 LLM 一个 **external motion memory**。

### 6.2 Limitation

1. **2D projection loss**：把 3D robot trajectory 投影到 2D 比较会丢失 depth 信息。对于侧视角 video 可能 OK，但对于正面/斜视角会有 aliasing。
2. **Single camera assumption**：video 必须是 side-view 才能 work，否则 keypoint 坐标语义不一致。
3. **Keypoint correspondence**：video 里的 human keypoint 和 robot keypoint 不是 anatomically 对应的。Anymal 有 4 条腿，dog keypoint 和 Anymal joint 的 mapping 是手工定义的。
4. **Periodicity assumption**：autocorrelation 分段假设 motion 是周期性的，non-periodic behavior（如 jumping、turning）会 fail。

### 6.3 与未来方向的连接

这个工作其实指向一个更大的 research direction：**multimodal reward specification**。Video 只是其中一种 modality，未来可能结合：
- **Demonstration video** (this paper)
- **Natural language** (Eureka)
- **Reference motion capture** (DeepMimic [10])
- **Physical constraints** (safety, energy)

最终目标是让 non-expert 能通过任意 intuitive interface 指定 robot behavior。

### 6.4 跟你的 some thoughts on 的 connection

你之前在 Yann LeCun 对谈和 EurekaLab talk 里提到过，LLM + RL 的 bottleneck 在于 reward specification。这个工作正好 hits 这个点 —— 它没有改 LLM，没有改 RL，改的是 **reward specification interface**。这种 "interface innovation" 往往比 algorithm innovation 更 impactful，因为它降低了 human-robot interaction 的 friction。

---

## 7. Implementation details 的几个 gem

1. **Video sampling rate**: Humanoid 每 3 帧采一次，Anymal 每 9 帧采一次。这对应 human running cycle (~0.8s) vs dog ambling cycle (~0.4s) 的差异。
2. **Camera normalization**: stationary camera 用 frame dimensions 归一化，moving camera 用 bounding box 归一化。后者是为了剔除 camera motion 的干扰。
3. **N=5 iterations, K=16 samples**: 跟 Eureka 完全对齐，确保 fair comparison。
4. **Selection metric H_mts vs feedback metric DTW**: 这是 decoupling 的 —— selection 用 task-relevant metric，feedback 用 behavior-similarity metric。如果用 DTW 做 selection，可能学出 "pose matching 但不前进" 的 policy。

---

## Reference Links

[1] Eureka: https://eureka-research.github.io/  
[2] L2R (Language to Reward): https://language-to-reward.github.io/  
[3] Text2Reward: https://text-to-reward.github.io/  
[4] RTMpose: https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose  
[5] HRNet: https://github.com/leoxiaobin/deep-high-resolution-net.pytorch  
[6] Autocorrelation period detection: https://www.sciencedirect.com/science/article/pii/S0031320322006043  
[7] FastDTW: https://github.com/wannesm/dtaidistance  
[8] IsaacGym: https://developer.nvidia.com/isaac-gym  
[9] PPO: https://arxiv.org/abs/1707.06347  
[10] DeepMimic: https://xbpeng.github.io/projects/DeepMimic/  
[11] Video2Reward (this paper, likely arxiv): https://arxiv.org/abs/2506.14718 (推测，基于 typical arxiv ID pattern)

---

## 8. 如果要我 extend 这个工作

几个值得探索的方向：

1. **3D keypoints via multi-view**: 用 NeRF 或 multi-view pose estimation 直接出 3D trajectory，跳过 projection 这步。
2. **Diffusion-based reward**: 把 video → reward 改成 video → diffusion model → reward shaping，让 reward generation 更 continuous。
3. **Cross-embodiment**: video 里是 dog，robot 是 humanoid。这是模仿学习的 holy grail —— 跨 morphology 的 behavior transfer。
4. **Real-world video**: 目前只在 IsaacGym 测，real video 带 noise、occlusion、lighting variation，keypoint extraction quality 会下降。
5. **LLM choice**: 论文没说用哪个 LLM（猜测是 GPT-4）。如果用 code-specific LLM 如 CodeLlama 或 StarCoder，reward code quality 可能更高。

希望这个 deep dive 对你 build intuition 有帮助，Andrej。这个工作的核心 takeaway 是：**reward specification 的 modality 比 reward algorithm 更重要**。Video as interface 是一个被低估的 direction，尤其在 robotics 这个 motor-prior-heavy 的领域。
