---
source_pdf: VLASH Real-Time VLAs via Future-State-Aware Asynchronous Inference.pdf
paper_sha256: df0a64a49dd257c348f1053311248450095c187e0268dc97f8a18d2d28b27cec
processed_at: '2026-08-13T02:58:10-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VLASH 用人话说一遍

## 故事从哪开始

你看过那些 robot demo 视频吗?就是 VLA model 拿个杯子、叠个衣服那种。发在 Twitter 上看起来挺流畅,但你不知道的是 **大部分都被加速了 5 到 10 倍**。真实跑起来,机器人是 *走走停停* 的——动一下,愣一下,动一下,愣一下。那个"愣一下"就是在等 model inference。

这事儿在 lab demo 里还能糊弄过去,一到 real-time physical interaction 就彻底露馅。你让 robot 去打 ping-pong,球都飞过去了它还没反应过来。

为什么?因为现在所有 VLA 都跑在 **synchronous mode** 下:先 inference 完,再执行 action,执行完了再 inference,循环往复。inference 期间 robot 是 idle 的,在干等。$\pi_{0.5}$ 在 RTX 4090 上一次 inference 大概 100ms,执行 25 个 action step 大概 500ms,所以一个 cycle 是 600ms。一秒钟只能"看"世界不到 2 次。

球速 5m/s 的时候,600ms 它已经飞了 3 米。没救。

---

## 那 asynchronous inference 不是显然的解法吗

对,很显然。让 robot 执行当前 action chunk 的同时,后台偷偷 inference 下一个 chunk。执行完无缝切换,inference 时间被"藏"在执行背后了。理论上 cycle time 从 $T_{\text{infer}} + T_{\text{exec}}$ 降到 $\max(T_{\text{infer}}, T_{\text{exec}})$,reaction latency 从 ~600ms 降到 ~100ms。

这事儿大家早就想到了,SmolVLA 就实现了 naive async。但 *一用就翻车*。

翻车的原因是一个看起来很小、实际上很致命的 **timing misalignment**:

- Inference 开始时,robot 在 state $s_t$
- Inference 完成时(过了 $\Delta$ 步),robot 已经走到 $s_{t+\Delta}$
- 你 inference 出来的 action $A_t$ 本来是给 $s_t$ 准备的
- 但执行的时候机器人已经不在 $s_t$ 了

这就好比你给朋友指路,说"到你家路口左转",结果你话还没说完,朋友已经开过路口了。你的指令没错,但 *对象已经错了*。

naive async 直接 chunk switch,机器人就会在错误的位置执行原本正确的 action,看起来就是 jittery、unstable、laggy。

---

## 现有的 fix 都不够好

社区提过两个 fix:

**RTC (Real-Time Chunking)**: 把已经确定要执行的 action "freeze" 住,只 inpaint 剩下的部分。听起来合理,但 inpainting 本身要额外 forward pass,有 runtime overhead,而且大 delay 下能 inpaint 的空间太小,性能崩溃。

**A2C2**: 给 model 加一个 correction head,学一个 residual 来补偿 misalignment。要改架构,要 extra inference cost。

两个都有 adoption barrier。所以 async 在 VLA 社区一直没起来。

---

## VLASH 的 insight:一个尴尬的不对称

作者发现了一个 *trivially obvious but nobody exploited* 的事实:

在 inference 延迟 $\Delta$ 步期间,**环境你预测不了,但你自己的 robot state 完全能预测**。

为什么?因为 $\Delta$ 步内 robot 要执行的 action $a_{t:t+\Delta-1}$ 是 *上一个 chunk 已经生成好的*,已知。delta action 就是关节增量,所以:

$$s_{t+\Delta} = s_t + a_t + a_{t+1} + \ldots + a_{t+\Delta-1}$$

加法。完事。

那你 inference 的时候,不要 condition on $s_t$,condition on $s_{t+\Delta}$ 好了。observation 还是 $o_t$ (stale 的,没办法),但 state 是 *future* 的。这样 model 生成的 action 是给"执行时刻的 robot"准备的,misalignment 就消失了。

这就叫 **future-state-aware**。

---

## 这个 insight 美在哪

美在它和人类的认知机制一模一样。人有 ~200ms 的视觉反应延迟,但能打乒乓球。怎么做到的?大脑在发出运动指令的同时,复制一份(efference copy)给感觉皮层,用 forward model 预测"等视觉信号到了的时候,我的手应该在这个位置"。所以人不是用 *现在的视觉* 控制动作,是用 *过去的视觉 + 预测的本体状态* 控制动作。

VLASH 给 VLA 装上了 *极简版* 的同款机制。forward model 简单到就是加法,因为 action 已经是 delta 形式。但这就够了。

---

## 但有个问题:pretrained VLA 不会用 state

作者试了一下直接在 inference 时 feed roll-forward 的 future state 给 $\pi_{0.5}$,发现 model 根本不理它。性能没改善。

为什么?他们做了个对照实验(Table 1):在 LIBERO 上 fine-tune $\pi_{0.5}$,**不给 state 输入(visual only)反而比给 state 输入更好**(97.7% vs 96.8%)。

这说明现在的 VLA 训练完之后 *完全依赖 visual,忽略 state*。$\pi_{0.5}$ 的具体设计还火上浇油:它把 state 数值 tokenized 成 text token 拼到 language prompt 里。数值一过 text tokenizer,连续结构就毁了,model 学不到东西。

所以你喂一个 future state 进去,model 当没看见。VLASH 就废了。

---

## Training-time fix:Temporal Offset Augmentation

作者的解法特别 clever,完全不改架构,只改 data loader 怎么构造训练样本。

标准 fine-tuning 是这样的:给定 trajectory $(o_t, s_t, a_t)$,训练样本是
$$(o_t, s_t) \rightarrow A_{t:t+H-1}$$

VLASH 改成:随机采样一个 offset $\delta \in \{0, 1, \ldots, \Delta_{\max}\}$,构造
$$(o_t, s_{t+\delta}) \rightarrow A_{(t+\delta):(t+\delta+H-1)}$$

也就是说,**同一张图 $o_t$,对应不同 $\delta$ 下不同的 ground-truth action**。

模型不可能再偷懒只看 visual 就预测 action 了,因为同一张图对应多个答案。它被迫必须看 $s_{t+\delta}$ 才能区分。这就 *强迫* VLA 学会利用 state 信号。

而且 $\delta$ 是随机的,所以训练完的 model 对 *任何* delay 都鲁棒,部署时不管是 RTX 5090 还是 RTX 5070 都能用。$\delta=0$ 时退化为 standard training,所以同步性能不受损。

这其实顺带 *修复* 了 $\pi_{0.5}$ state tokenization 的 broken design——通过 data augmentation 让 model 重新学会读 state。Table 3 验证:VLASH 在同步 inference 下精度 96.6% ≈ Original 96.8%,完全无损。

---

## Efficient fine-tuning:一个 systems trick

上述 augmentation 有个 naive 实现的问题:对每个 $\delta$ 独立构造一个 training example,意味着同一个 $o_t$ 要被 vision encoder encode $N_\delta$ 次。vision encoder 是最贵的部分,这太浪费。

作者的 fix 是把多个 offset *pack* 到一个 sequence 里:

$$[o_t, (s_t, A_t), (s_{t+1}, A_{t+1}), \ldots, (s_{t+\Delta_{\max}}, A_{t+\Delta_{\max}})]$$

然后设计一个 block-sparse attention mask:
- Observation tokens 互相 attend
- 每个 offset branch 能 attend 到所有 observation tokens 和 branch 内的 tokens
- 不同 branch 之间 *互不 attend*

视觉只 encode 一次,5 个 offset branch 共享。Token 数从 750 增加到 950(+27%),但 effective trajectories 多了 5 倍。

实测训练 3.26× faster per step(Table 3)。这个 trick 在 PyTorch 2.x 上可以用 FlexAttention 实现,自动生成 fused kernel,不用手写 triton。

---

## Action Quantization:把 LLM 量化思想搬过来

Async 把 inference latency 隐藏了,现在 bottleneck 变成 robot 物理执行速度。SOTA VLA 训练数据是 50Hz teleop 录制,每步 delta 很小。

作者观察到:很多 micro-action 是冗余的,没必要执行那么细。类比 LLM 量化:FP16 权重精度高,但 INT8/INT4 量化后精度损失小,推理快很多。

公式很简单,对 delta action:
$$\hat{a}_i = a_{iq} + a_{iq+1} + \ldots + a_{(i+1)q-1}$$

把 $q$ 个连续 micro-action 加起来变成 1 个 macro-action,一次性走完原来 $q$ 步的位移。

实验结果很 striking:
- $q=1$: 1.12× speedup, 94% accuracy
- $q=2$: **2.03× speedup, 94% accuracy**(完全 free lunch!)
- $q=3$: 2.67× speedup, 89.3% accuracy

$q=2$ 时精度完全不掉,说明 50Hz teleop 数据本身就有 2× 冗余。

---

## 实验里的几个 striking 数字

**Kinetix (Figure 6)**: 高动态物理仿真,12 个 task,每点 1024 rollouts。Delay=4 时,VLASH 81.7% vs Naive Async 51.2%,绝对提升 **30.5%**。VLASH 几乎追平 Sync 上界。

**LIBERO (Table 1)**: $\pi_{0.5}$ 上,Delay=1, 2 时 VLASH *超过* Sync baseline(+0.3%, +0.4%)。这是个 surprising 的结果。我的猜测: roll-forward state 起到了 regularizer 作用,而且 async 让 transition 更平滑,减少了 chunk 边界的 discontinuity。

**Reaction Speed (Table 2)**: 
- Sync reaction latency: $T_{\text{exec}} + T_{\text{infer}} \approx 530$ms
- Async reaction latency: $T_{\text{infer}} \approx 30$ms (RTX 5090)
- **17.4× speedup**

这意味着 robot 从"看到变化到开始反应"的延迟从半秒降到 30ms。人类视觉反应延迟大概 200ms,所以这个 robot *比人反应还快*。

**Ping-pong demo**: 据我所知,这是 *第一个* VLA 能和人打 ping-pong rally 的 demonstration。Figure 1 的第三帧 robot 已经开始反应,说明 perception-to-action latency 极低。Synchronous inference 完全做不到。

---

## 为什么这个工作让我觉得"对"

我喜欢 VLASH 是因为它体现了几个 *正确的工程审美*:

**1. Insight 简单到 trivial,但没人这么做过**。"robot state 可以前向推演,environment 不行"——这是 *显然* 的事实,但社区一直在试图解决 *整个* misalignment 问题(RTC inpainting 整个 chunk、A2C2 加 correction head)。VLASH 只 fix 能 fix 的部分,接受不能 fix 的部分(stale observation),用神经网络的 implicit 能力补上剩下的事。这是 *partial solution with elegant boundary*。

**2. 零开销**。No extra inference cost, no architectural change, no runtime overhead。只改 data loader 和 inference 时加一行 state 加法。这种 *leverage* 极高的工作在 systems ML 里是稀少的。

**3. Training augmentation 解决了 inference 时的 architectural bug**。$\pi_{0.5}$ 的 state tokenization 是 broken 的,但作者没去改架构(那要重训 pretrain),而是用 offset augmentation 在 fine-tune 阶段 *teach* model 重新用 state。这是 *work with what you have* 的实用主义。

**4. Systems thinking 贯穿始终**。Shared observation packing、block-sparse attention、action quantization——这些都不是 ML 的新想法,但被准确应用在了 VLA 部署的 bottleneck 上。Song Han lab 一贯的风格:quantization、sparsity、efficient inference,这次搬到 robotics。

**5. Demo 比 benchmark 更有说服力**。一个 ping-pong rally 视频比 100 个 LIBERO success rate 数字更能传达"VLA 终于能实时了"这件事。这是好的 storytelling。

---

## 我的几个 broader thoughts

**VLA 的接口设计是 underestimated 的 bottleneck**。社区一直在卷 model scale、data scale,但 deployment 的真实瓶颈往往是 *temporal interface* (async alignment)、*granularity interface* (action quantization)、*modality interface* (state tokenization)。这类工作 leverage 比加 parameter 大得多。

**State signal 被严重低估**。Table 1 那个"visual only > with state"的结果 should be embarrassing for the community。它说明 SOTA VLA 实际上 *丢弃* 了一个关键的输入信号。VLASH 用 augmentation 修复了这个,但根本问题是 architecture。我预期后续工作会重新设计 state injection:continuous embedding、AdaRMSNorm conditioning、cross-attention,而不是 text tokenization。

**VLA 终于能做 physical interaction 了**。在 VLASH 之前,VLA 的 demo 都在"慢 manipulation"领域。VLASH 打开了 "fast physical interaction" 的大门:ping-pong、catch、juggling、contact-rich assembly、locomotion+manipulation 联合控制。这对 humanoid robot 的 commercial deployment 是关键一步——你不可能让一个 600ms 反应的 robot 在厨房帮你打鸡蛋。

**Async + future-state-aware 会成为 VLA serving stack 的标配**。就像 LLM serving 现在都有 KV cache + continuous batching 一样,未来 VLA serving 一定有这一层。VLASH 的实现非常 clean(见 GitHub: https://github.com/mit-han-lab/vlash),任何已有 VLA 都能低成本切换。

---

## 三个直觉性的 takeaways

**1. 任何 inference 有 latency 的 closed-loop control 系统,都可以问一句: "在 inference 期间,我有什么 state 是可以 roll-forward 的?"** 如果有,就 condition on future state,别 condition on stale state。这是 *generalizable pattern*,不限于 robot。

**2. 当 model 不用某个 input 信号时,不要急着改架构,先想想是不是 training data 让它"不需要"用**。$\pi_{0.5}$ 不用 state,不是 architecture 不行,是 standard fine-tuning data 让 model 视觉就能 fit。Offset augmentation 制造了一个 *必须用 state 才能解* 的 data distribution,强迫 model 学。这是 *data-driven architectural fix*。

**3. Quantization 的 philosophy 可以推广到任何 "过度精细的离散化"**。LLM 量化权重、robot 量化 action、diffusion model 量化 timestep——本质都是"当前的离散粒度比 task 需要的更细,合并几个 step 不影响精度"。这是个 *transferable principle*。

---

## 参考

- **VLASH GitHub**: https://github.com/mit-han-lab/vlash
- **$\pi_{0.5}$ paper**: https://arxiv.org/abs/2504.16054
- **SmolVLA**: https://arxiv.org/abs/2506.01844
- **RTC**: https://arxiv.org/abs/2506.07339
- **A2C2 (concurrent)**: https://arxiv.org/abs/2509.23224
- **LIBERO**: https://arxiv.org/abs/2306.03310
- **Gemini Robotics**: https://arxiv.org/abs/2503.20020
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **FlexAttention**: https://pytorch.org/blog/flexattention/
- **Speculative Decoding (类比思路)**: https://arxiv.org/abs/2302.01318
- **AWQ (Song Han lab)**: https://arxiv.org/abs/2306.00978

一句话总结:**VLASH 让 VLA 第一次能像人一样,在视觉延迟下用"我知道我的手在哪"来补偿,从而实时反应**。简单、干净、对的方向。

---

# VLASH: Future-State-Aware Asynchronous Inference for VLAs — 深度技术解析

## 1. Paper Overview & Core Insight

VLASH 由 MIT Han Lab 联合 NVIDIA、Berkeley、清华等团队提出,作者包括 Song Han、Zhijian Liu 等量化与系统优化的常客。它要解决一个非常具体、非常真实的痛点: **当前的 VLA models(如 $\pi_{0.5}$、Gemini Robotics、Gr00t)在真实部署时,演示视频往往被加速 5–10× 才看起来流畅**,根本原因是 synchronous inference paradigm 造成了 stop-and-go 的 motion stuttering。

核心 insight 一句话可以概括: **当我们执行 action chunk $A$ 时,机器人和环境继续在演化;等到 inference 完成时,环境已经"跑偏"了**。VLASH 的解法是利用 *已知的前一段 action chunk* 把 robot state "roll-forward" 到 inference 完成的那个时刻,然后 condition on 这个 future state 来生成下一组 action。这等价于让 policy 变成 *future-state-aware*。

Paper: https://arxiv.org/abs/2509.23224 (预印本链接猜测,以 GitHub repo 为准: https://github.com/mit-han-lab/vlash)

---

## 2. 问题背景:为什么 Synchronous Inference 在真实部署中是灾难

### 2.1 Action Chunking Policy 形式化

VLA policy 通常写成:
$$
\pi_\theta(A_t \mid o_t, s_t)
$$

其中变量含义:
- $\pi_\theta$: 参数为 $\theta$ 的 policy 网络(对 $\pi_{0.5}$ 是 PaliGemma 视觉编码器 + Gemma LLM + flow-matching action expert)
- $o_t$: 时刻 $t$ 的 environment observation(典型为两张 224×224 图像 + 语言指令,约 700 tokens)
- $s_t$: robot state(关节位置、gripper 状态等,对 7-DOF arm 约 14 维)
- $A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$: 一个 action chunk
- $H$: **prediction horizon**,$\pi_{0.5}$ 用 $H=50$
- $a_t$: 一个 single-step action(典型为 delta joint velocity 或 delta end-effector pose)

每个 chunk 中只执行前 $K \leq H$ 个 action,$K$ 称为 **execution horizon**($\pi_{0.5}$ 默认 $K=25$)。

### 2.2 Synchronous Inference 的时序问题

在 synchronous 模式下,每个 control cycle 的耗时为:

$$
T_{\text{sync}} = T_{\text{infer}} + T_{\text{exec}} = T_{\text{infer}} + \frac{K}{f_{\text{ctrl}}}
$$

- $T_{\text{infer}}$: 一次 forward pass 时间($\pi_{0.5}$ 在 RTX 4090 上约 103ms,2 图像输入)
- $f_{\text{ctrl}}$: control frequency(LIBERO 用 30Hz,real-world 50Hz)
- $T_{\text{exec}} = K/f_{\text{ctrl}}$: 执行 $K$ 个 action 的时间

当 $K=25$、$f_{\text{ctrl}}=50$Hz 时,$T_{\text{exec}}=500$ms,加上 $T_{\text{infer}}\approx 100$ms,总 cycle 约 600ms。这意味着 **policy 每秒只能"看到"世界不到 2 次**。在 ping-pong 这种球速 ~5 m/s 的任务里,600ms 内球已经飞了 3 米,完全不能反应。

### 2.3 Asynchronous Inference 的根本矛盾

Asynchronous 思路: robot 执行当前 chunk 的同时,在后台 inference 下一组 chunk。理论上消除了 $T_{\text{infer}}$ 的等待。

但产生了 **prediction-execution temporal misalignment**:
- Inference 开始时(timestep $t$),机器人处于 state $s_t$,policy 基于此预测 $A_t$
- Inference 完成时(经过 $\Delta$ 步),机器人已经走到了 $s_{t+\Delta}$
- $A_t$ 本来是为 $I_t^{\text{pred}} = [t, t+K)$ 这个区间准备的
- 但实际执行发生在 $I_t^\text{exec} = [t+\Delta, t+\Delta+K)$ 这个区间

数学上,$\Delta$ 是 inference latency 转换成 control step 的整数:
$$
\Delta = \left\lceil \frac{T_{\text{infer}}}{1/f_{\text{ctrl}}} \right\rceil
$$

当 $\Delta > 0$ 时, $A_t$ 中的第一个 action $a_t$ 实际上被应用到 $s_{t+\Delta}$,这个 state 与 $s_t$ 之间的差异就是 misalignment 的来源。Naive async 直接切换 chunk,造成 **第一帧执行 stale action 的"幽灵动作"**,后续动作也整体错位,出现 jittery、unstable 的运动。

---

## 3. VLASH 的核心机制:Future-State-Aware Roll-Forward

### 3.1 Key Insight:Robot State 可前向推演,Environment 不可

这是 paper 最关键的一段思考。在 $\Delta$ 步的 inference 延迟期间:
- Environment observation $o_{t+\Delta}$ 是 **未知的**(球已经飞到新位置了)
- 但 robot state $s_{t+\Delta}$ 是 **完全确定的**,因为它由当前 state $s_t$ 和已经被生成、且即将执行的 action $a_{t:t+\Delta-1}$ 唯一决定(假设 kinematics 是 deterministic 的)

所以 VLASH 用:
$$
s_{t+\Delta} = \text{FK}(\text{IK}(s_t, a_t), a_{t+1}) \ldots \text{(or simpler)}
$$

最简单情况(delta action 是关节增量):
$$
s_{t+\Delta} = s_t + \sum_{i=0}^{\Delta-1} a_{t+i}
$$

变量解释:
- $s_t$: 当前 robot state vector(关节角度向量 $\in \mathbb{R}^d$, $d$=DOF 数)
- $a_{t+i}$: 上一 chunk 中第 $i$ 个 action,如果是 delta-position 形式,直接累加
- $s_{t+\Delta}$: 预测的执行时刻 state

Figure 3(c) 展示的就是 $s_3 = s_1 + a_1 + a_2$,即在 inference delay $\Delta=2$ 的情况下,从 $s_1$ roll-forward 两步得到 $s_3$。

### 3.2 与人类认知的类比(我的解读)

人类也有 ~200ms 的视觉反应延迟,但接球、打乒乓球时表现良好,因为大脑使用 **efference copy**(运动指令副本)做 forward model,预测自己的肢体在动作发出后会在哪里。VLASH 的 roll-forward 本质上就是给 VLA 装上了一个 *极简版的 efference copy forward model*——简单到只用加法就能算,因为 action 已经是 delta 形式。

这与 Model Predictive Control (MPC) 有思想上的相似,但 VLASH 不解优化问题,只是做 *open-loop state prediction* 然后让神经网络 policy 自己处理剩下的判断。

### 3.3 推理时的实际流程

```
1. 在 timestep t,robot 处于 s_t,正在执行上一 chunk 的剩余部分 a_{t:t+Δ-1}
2. 同时在后台开始 inference:
   - 计算 s_{t+Δ} = roll_forward(s_t, a_{t:t+Δ-1})
   - Feed 给 VLA: A_{t+Δ} = π_θ(· | o_t, s_{t+Δ})   # 注意 o_t 是 stale 的,s_{t+Δ} 是 future 的
3. Inference 完成(实际到了 timestep t+Δ),robot 已在 s_{t+Δ}
4. 立即执行 A_{t+Δ},无缝切换
```

关键 trick: **observation 保持 stale,只 roll-forward state**。这是一种 "asymmetric prediction"——因为视觉信号无法预测,但本体感受可以。

---

## 4. 训练时的问题:Pretrained VLA 不会用 State

### 4.1 Empirical Observation

作者在 $\pi_{0.5}$ 上做了一个令人惊讶的实验(Table 1): 在 LIBERO 上 fine-tune 时,**不给 state 输入(visual only)反而比给 state 输入效果更好**(97.7% vs 96.8%)。这说明当前 SOTA VLA 在训练后 *几乎完全依赖 visual features,忽略了 state*。

为什么? 因为 $\pi_{0.5}$ 把 state 数值 tokenized 成文本 token 拼到 language prompt 里(很奇怪的设计)。这种 text-token 形式破坏了数值的连续结构,模型很难从中学到东西。

后果: 如果直接在 inference 时 feed 一个 roll-forward 的 state,模型根本不会用它,VLASH 就废了。

### 4.2 解法:Temporal Offset Augmentation

作者设计了一个训练时数据增强,把 "future state prediction" 的能力 *baked into* fine-tuning 阶段,完全不改架构,只改 sample 构造方式:

给定 trajectory $\{(o_t, s_t, a_t)\}$, standard fine-tuning 的训练样本是:
$$
(o_t, s_t) \rightarrow A_{t:t+H-1}
$$

VLASH 改为: 随机采样 offset $\delta \in \{0, 1, \ldots, \Delta_{\max}\}$,构造:
$$
(o_t, s_{t+\delta}) \rightarrow A_{(t+\delta):(t+\delta+H-1)}
$$

变量解释:
- $\delta$: temporal offset,代表 inference delay 的可能取值
- $\Delta_{\max}$: 最大考虑的 delay(实验中 $\Delta_{\max}=3$ 或 $4$)
- $o_t$: 固定使用 timestep $t$ 的原始 observation(不变)
- $s_{t+\delta}$: 从 trajectory 上取未来 $\delta$ 步的 state(ground truth 已知)
- $A_{(t+\delta):(t+\delta+H-1)}$: 对应的未来 action chunk

关键点: **同一个 $o_t$ 对应多个不同的 ground-truth action,取决于 $\delta$**。模型不可能再"作弊"只看 visual 就预测 action,被迫必须看 $s_{t+\delta}$ 才能区分不同 $\delta$ 下的正确 action。这就强制 VLA 学会利用 state 信号。

### 4.3 为什么随机采样 $\delta$ 而非固定 $\Delta$

这是一个 elegant 的设计:
- 实际部署时 inference delay $\Delta$ 取决于硬件(RTX 5090/4090/5070 各不同,见 Table 2)
- 训练时不知道部署 delay,所以训练一个 $\delta$ 的 *分布*,让 policy 对不同 delay 都鲁棒
- $\delta=0$ 时退化为 standard fine-tuning,所以不破坏 synchronous 性能(Table 3 验证)

---

## 5. Efficient Fine-tuning:Shared Observation Attention Pattern

### 5.1 Naive 实现的浪费

如果对每个 $\delta$ 独立构造一个 training example $(o_t, s_{t+\delta}, A_{t+\delta})$,那 $o_t$ 会被 vision encoder 编码 $N_\delta$ 次。对于 $\pi_{0.5}$:
- $o_t$ 编码产生 ~700 tokens
- $(s_{t+\delta}, A_{t+\delta})$ 产生 ~50 tokens

vision encoder 是整个 forward pass 中最贵的部分,这种重复是完全的浪费。

### 5.2 Packed Sequence with Block-Sparse Attention

作者的解法是 pack 成一个 sequence:
$$
[o_t, (s_t, A_t), (s_{t+1}, A_{t+1}), \ldots, (s_{t+\Delta_{\max}}, A_{t+\Delta_{\max}})]
$$

然后设计一个 **block-sparse attention mask**(参考 Figure 4):

| Block | $o_t$ | $(s_t, A_t)$ | $(s_{t+1}, A_{t+1})$ | ... | $(s_{t+\Delta_{\max}}, A_{t+\Delta_{\max}})$ |
|---|---|---|---|---|---|
| $o_t$ | ✓ | ✓ | ✓ | ... | ✓ |
| $(s_t, A_t)$ | ✓ | ✓ (within) | ✗ | ... | ✗ |
| $(s_{t+1}, A_{t+1})$ | ✓ | ✗ | ✓ (within) | ... | ✗ |
| ... | ✓ | ✗ | ✗ | ... | ✗ |

Mask 规则:
- Observation tokens 之间互相 attend(蓝色,standard)
- 每个 offset branch 的 $(s_{t+\delta}, A_{t+\delta})$ 可以 attend 到所有 observation tokens 和 *本 branch 内* 的 tokens
- 不同 offset branches 之间 **互不 attend**(灰色 mask)

Positional encoding: 每个 branch 的 $(s_{t+\delta}, A_{t+\delta})$ 都从同一 index 开始(等于 observation token 长度,如 700),这样不同 branches 在模型看来 *位置等价*。

### 5.3 实现上的技术 trick

这种 mask 在 PyTorch 2.x 上可以用 **FlexAttention**(Meta 提出的 programming model, ref https://pytorch.org/docs/stable/flex_attention.html) 高效实现,或用 block-diagonal mask 在 FlashAttention-2/3 上做。FlexAttention 可以让用户写 Python 表达式描述 mask,自动生成 fused kernel,不需要手写 triton。

Token 数计算: $N_\delta=5$ offsets,observation ~700 tokens,state-action ~50 tokens:
- 标准 forward: 700 + 50 = 750 tokens
- VLASH packed: 700 + 5×50 = 950 tokens(增加 ~27%)
- Effective trajectories: 5× → 训练样本效率提升 5×

但每个 step 的 wall-clock 时间因为只编码 $o_t$ 一次,反而 **3.26× faster**(Table 3: 420.99ms → 129.29ms per step)。

### 5.4 训练动态分析

Table 3 显示 VLASH 在 10K steps 时 87.1% < 94.1% Original(更慢收敛),但 30K steps 时追平(96.6% vs 96.8%)。这种现象的原因:
- Offset augmentation 增加了 task 难度(同一 observation 对应多个 action)
- 早期模型需要先学会"读 state"才能 fit 训练数据
- 一旦学会,数据多样性反而带来更好的 generalization

这很像 *curriculum learning* 的反面,先难后易:一开始任务更难,但学到的能力更强。

---

## 6. Action Quantization:把 LLM 量化思想搬到 Robot Control

### 6.1 Motivation

Asynchronous inference 把 $T_{\text{infer}}$ 隐藏到执行背后,所以瓶颈变成了 robot 物理执行速度。SOTA VLA 训练数据是 ~50Hz 的 teleoperation 录制,每步 delta 很小。

类比: LLM 中 FP16 权重精度太高,量化到 INT8/INT4 可以大幅加速,精度损失很小。VLASH 把这个思想用到 action sequence 上。

### 6.2 公式

给定细粒度 action sequence $\{a_0, a_1, \ldots, a_T\}$,量化 factor $q$,构造 macro-action sequence $\{\hat{a}_0, \hat{a}_1, \ldots\}$:
$$
\hat{a}_i = a_{iq} + a_{iq+1} + \ldots + a_{(i+1)q-1}
$$

变量解释:
- $q$: quantization factor,把每 $q$ 个连续 micro-action 合并成 1 个 macro-action
- $i$: macro-action 的索引
- $iq, iq+1, \ldots, (i+1)q-1$: 第 $i$ 个 macro-action 包含的 micro-action 索引范围
- $\hat{a}_i$: 第 $i$ 个 macro-action,等于 $q$ 个 micro-action 之和

例子: $q=3$ 时,$\hat{a}_0 = a_0 + a_1 + a_2$,执行 $\hat{a}_0$ 等于一次性走完原来 3 步的位移。

### 6.3 何时这种量化安全

关键 assumption: **robot 不需要精确访问每个中间 waypoint,只要从起点到终点的方式正确即可**。这对 ping-pong、whack-a-mole 这类需要快速大幅运动但中间轨迹不严格的任务特别合适。对于精密装配(peg-in-hole)可能就不行。

### 6.4 实验中的 Speed-Accuracy Trade-off

Table 1 和 Figure 7 显示:
- $q=1$ (no quantization): 1.12× speedup, 94% accuracy
- $q=2$: 2.03× speedup, 94% accuracy(无精度损失!)
- $q=3$: 2.67× speedup, 89.3% accuracy(损失 4.7%)

$q=2$ 完全免费午餐,这非常有意思。说明在 50Hz 的 teleop 数据中,确实有冗余的 micro-action,合并 2 个不影响精度。

---

## 7. 实验结果深度解析

### 7.1 Kinetix Simulation (Figure 6)

Kinetix 是高动态物理仿真,有 throwing、catching、balancing 等任务。模型是 4-layer MLP-Mixer,prediction horizon $H=8$。

关键结果:
- **Delay $\Delta=4$ 时**: VLASH 81.7% vs Naive Async 51.2%,提升 **30.5%** 绝对成功率
- RTC 在大 delay 下崩溃(因为 inpainting 在大 delay 下需要 freeze 太多 action,剩余可优化空间太小)
- VLASH 几乎追踪 Sync baseline 的上界,说明 future-state-aware 完全消除了 async 的代价

Intuition: Kinetix 中环境快速变化,所以 future env observation 的缺失会 hurt,但 future robot state 的精确预测已经足够稳定住大部分控制——这告诉我们 *robot state 在 closed-loop control 中的信息量被严重低估了*。

### 7.2 LIBERO Benchmark (Table 1)

LIBERO 是 4 个 sub-benchmark(Spatial, Object, Goal, LIBERO-10)各 10 个任务,环境变化缓慢。$\pi_{0.5}$ 上:

| Delay | SR | Time (s) | ΔSR | Speedup |
|---|---|---|---|---|
| Sync | 96.8 | 8.4 | - | - |
| Sync (no state) | 97.7 | 8.4 | +0.9 | - |
| VLASH (delay=1) | 97.2 | 7.2 | +0.4 | 1.17× |
| VLASH (delay=2) | 97.1 | 6.4 | +0.3 | 1.31× |
| VLASH (delay=3) | 94.6 | 5.7 | -2.2 | 1.47× |
| VLASH (delay=4) | 93.1 | 5.8 | -3.7 | 1.45× |

关键观察:
1. **Delay=1, 2 时 VLASH *超过* Sync baseline**,这是 surprising 的。我的猜测: roll-forward state 起到了 *regularizer* 作用,让模型学到更鲁棒的 state-to-action 映射;同时 action chunk 之间的 transition 更平滑。
2. **Delay=3, 4 时 accuracy 下降但仍然可用**,说明 LIBERO 任务对 delay 较不敏感(因为环境变化慢)。
3. Sync (no state) > Sync(with state) 这个结果很 striking,印证了 $\pi_{0.5}$ 的 state tokenization 设计有严重问题——state 通过 text tokenizer 后数值结构被破坏。VLASH 的 offset augmentation 实际上 *修复* 了这个,让 state 变成有用信号。

### 7.3 Real-World Reaction Speed (Table 2)

测量 $\pi_{0.5}$ 在不同 GPU 上的最大反应延迟。设置: $K=25$, 50Hz,$T_{\text{exec}}=500$ms。

| GPU | $T_{\text{infer}}$ (ms) | Sync reaction | Async reaction | Speedup |
|---|---|---|---|---|
| RTX 5090 | 30.4 | 530.4 | 30.4 | 17.4× |
| RTX 4090 | 36.1 | 536.1 | 36.1 | 14.9× |
| RTX 5070 | 64.1 | 564.1 | 64.1 | 8.8× |

公式:
- Sync reaction = $T_{\text{exec}} + T_{\text{infer}}$ (要等当前 chunk 执行完 + 推理完才能反应)
- Async reaction = $T_{\text{infer}}$ (推理时机器人继续执行,推理完立即反应)

17.4× 反应加速非常 dramatic。在 ping-pong 任务上,这意味着 robot 从"完全无法反应"到"可以打 rally"。

### 7.4 Fine-tuning Efficiency (Table 3)

| Method | Time/Step (ms) | 10K | 20K | 30K |
|---|---|---|---|---|
| Original | 420.99 | 94.1 | 97.1 | 96.8 |
| VLASH | 129.29 | 87.1 | 94.4 | 96.6 |
| Speedup | 3.26× | - | - | - |

3.26× per-step 加速来自 shared observation(只 encode 一次)。总训练成本: VLASH 在 30K steps 时实际 wall-clock 时间是 Original 的 30K×129.29 / (30K×420.99) = 0.307,即 *3.26× 更快* 达到几乎相同精度。

如果允许训练更久,VLASH 因为 5× effective trajectories 的数据增强,最终精度可能更高。

### 7.5 Real-World Manipulation (Figure 7)

三个任务: Pick-and-Place, Stacking, Sorting,各 16 rollouts。Score 用 2-point 系统(1 point 抓取成功,1 point 完成任务)。

- VLASH (q=1): 94% avg score, 18.8s, 1.12× speedup
- VLASH (q=2): 94% avg score, 2.03× speedup(精度不变!)
- VLASH (q=3): 89.3% avg score, 2.67× speedup(轻微精度损失)
- Sync: 83% avg score, 21s
- Naive Async: 89.7% avg score

VLASH 甚至在精度上 *超过* Sync baseline(94% vs 83%)。这部分归因于 Naive Async 的 prediction-execution misalignment 在某些 episode 中导致抓取失败,VLASH 通过 alignment 修复了。

---

## 8. Architectural Modifications & 通用性

### 8.1 Zero-Architecture-Change 是核心 selling point

VLASH 对所有 *接受 state 输入* 的 VLA 都适用。对 $\pi_0$ 和 SmolVLA 这种用 state projection layer(把 proprioceptive vector 通过 linear 投到 hidden dim,然后作为 continuous embedding 注入 transformer)的设计,VLASH out-of-the-box 工作。

### 8.2 对 $\pi_{0.5}$ 的特殊处理

$\pi_{0.5}$ 用了一个奇葩设计: state 数值通过 tokenizer 转成 text token 拼到 language prompt 里。这破坏了数值的连续结构。作者发现两个 fix(都在 supplementary §7.4):

**Option A: 加一个 lightweight state projection layer**
- Linear map: $\mathbb{R}^{\text{state\_dim}} \rightarrow \mathbb{R}^{\text{hidden\_dim}}$
- Zero-initialized(初始时不改变 pretrained 模型行为)
- 把得到的 embedding 注入到原来 state token 在 sequence 中的位置

**Option B: 用 AdaRMSNorm conditioning**
- 把 state embedding 加到 timestep embedding 上,作为 AdaRMSNorm 的 conditioning signal(类似 DiT/Flow Matching 的标准做法)

这两个 fix 都是 optional 的。VLASH 不加也能 work,但加了能进一步改善 smoothness。Zero-init 保证了不会破坏 pretrained 性能。

### 8.3 Generalization Across VLAs

Table 4 在 SmolVLA-450M 上验证:
- Delay=3 时 79.06% vs Sync 78.96% (+0.10%),1.35× speedup
- 完全 free lunch

SmolVLA 本身就是设计为支持 async 的 VLA,但 naive async 会 misalignment。VLASH 直接补上 future-state-awareness,完美适配。

---

## 9. Intuition Building:三个深层类比

### 9.1 与 Efference Copy / Forward Model 的类比

神经科学中,**efference copy** 是运动指令发出时大脑同时送一份副本给 sensory area,用来预测接下来感知会是什么样,从而区分"外界变化"和"自己动作造成的变化"。

VLASH 的 roll-forward 就是 *minimal efference copy*:
- Efference copy: $a_{t:t+\Delta-1}$
- Forward model: $s_{t+\Delta} = s_t + \sum a_{t+i}$
- 用预测的 state 来 condition 下一个 action,而不是用 stale 的 state

人类能用这种机制在视觉延迟 ~200ms 下打乒乓球,VLASH 给 VLA 装上同款机制,达到了类似效果。

### 9.2 与 MPC 的类比但更轻量

Model Predictive Control 每步解优化:
$$
\min_A \sum_{k=0}^{K} \|x_{t+k} - x_{\text{ref}}\|^2 \quad \text{s.t. } x_{t+k+1} = f(x_{t+k}, a_{t+k})
$$

VLASH 不解优化,而是让神经网络 policy 通过 fine-tune 学会 *隐式* 处理未来 state。这是一种 *learned controller + open-loop forward model* 的混合,计算成本远低于 MPC,但保留了 future-awareness。

### 9.3 与 Speculative Decoding 的类比

LLM 推理中 **Speculative Decoding**(Leviathan et al., 2023, https://arxiv.org/abs/2302.01318) 用一个小模型先 draft tokens,大模型并行 verify,实现加速。

VLASH 的 async pipeline 在 *time axis* 上做类似事:
- "Draft" phase: 用 stale observation $o_t$ + roll-forward state $s_{t+\Delta}$ 让 VLA 推测未来 action
- "Verify" phase: 不需要 verify,直接执行(因为 state 是 deterministic 的)
- 加速来自 *hiding* inference latency under execution

这种 *speculation without verification* 能 work,因为 robot state dynamics 是已知的,不需要 verify。

---

## 10. Limitations & 我的思考

### 10.1 Future Environment 的缺失

VLASH 假设 environment 在 $\Delta$ 步内变化可以忽略(对 LIBERO 成立),或对 state 而言 environment 是 *slow enough*。在 ping-pong 任务中,球已经飞了 ~30cm(在 50Hz、$\Delta=3$ 时),这其实相当显著,但模型仍能 rally 成功。

我的解读: VLA 的 visual encoder 在 single frame 中已经 encode 了 *motion implication*——即当前帧的球位置 + 模型隐式的"它会继续飞"的 prior,加上 robot state roll-forward,就够用了。如果球突然转向(对方击球),需要新一轮 inference 才能反应,这正是 async 的反应延迟下界 $T_{\text{infer}}$ 决定的。

### 10.2 Action Quantization 的边界

$q=2$ 在实验中是 free lunch,但 $q=3$ 开始掉点。这说明 50Hz teleop 数据本身大约有 2× 的冗余。如果任务是 peg-in-hole、surgical 这种,可能 $q=1$ 是上限。Action quantization 是 *task-dependent* 的。

### 10.3 训练 Distribution Mismatch

训练时 $\delta \sim \text{Uniform}\{0, \ldots, \Delta_{\max}\}$,部署时 $\Delta$ 是固定的。这是一个 *distribution shift*。但 Table 1 显示 delay=1, 2, 3, 4 都能 work,说明 policy 学到的是 *general state-to-action mapping*,对任意 future state 都能用,而非过拟合到某个 delay。

### 10.4 与 RTC、A2C2 的对比

| Method | 额外开销 | 架构改动 | 实现复杂度 | 性能 |
|---|---|---|---|---|
| Naive Async | 0 | 0 | 低 | 差 |
| RTC | 大(inpainting 每步) | 0 | 中(需要 inpainting 推理) | 中等,大 delay 崩溃 |
| A2C2 | 中(correction head) | 大(改架构) | 中 | 中等 |
| **VLASH** | **0** | **0(可选 small projection)** | **低** | **优** |

VLASH 在所有维度都占优,这是 paper 的强 selling point。

### 10.5 Open Questions

1. **Vision 也能 roll-forward 吗?** 如果有 world model(如 DreamerV3、Genie 2),可以预测 $o_{t+\Delta}$,但成本高。VLASH 选择 *不做* 这件事,因为 cost/benefit 不划算。
2. **Multi-modal actions 怎么办?** Flow matching 的 $\pi_{0.5}$ 可以输出 multi-modal distribution,但 roll-forward state 是 single point estimate。如果未来 state 在 multi-modal region,可能有问题。
3. **Non-deterministic dynamics 呢?** 真实 robot 有 friction、backlash,简单加法 roll-forward 会有误差。可以用 learned forward model 代替,但作者选择保持简单。
4. **Long-horizon tasks?** $\Delta$ 越大,roll-forward 误差累积越大。$T_{\text{infer}}$ 在更强 VLA 上会变大,可能 $\Delta=10$ 时就出问题。需要 hardware-software co-design 让 $T_{\text{infer}}$ 始终小于 $T_{\text{exec}}$。

---

## 11. 对 Robotics + VLA 社区的影响

### 11.1 Practical Adoption

VLASH 的实现非常轻量(见 GitHub: https://github.com/mit-han-lab/vlash),只需要改 fine-tune data loader 和 inference 时加一个 roll-forward 步骤。这意味着 *任何* 已经部署的 VLA 都能低成本切换到 async mode。这对社区的影响远大于论文本身。

### 11.2 重新审视 VLA 的 State 信号

Table 1 的 "Sync without state > Sync with state" 是一个 *should-be-embarrassing* 结果。它告诉我们当前 SOTA VLA 的 state tokenization 是 broken 的。VLASH 的 offset augmentation 是一个 *training-time fix*,但根本问题是 architecture 设计。我预期后续工作会:
- 用 *continuous state embedding* 而非 text token(像 $\pi_0$ 那样)
- 在 AdaRMSNorm / cross-attention 中注入 state
- 设计专门处理 state 的 adapter

### 11.3 VLA 走向 Real-Time Physical Interaction

VLASH 让 VLA 第一次能 *play ping-pong with human*。这是 VLA 从"慢 manipulation"走向"快 physical interaction"的关键一步。后续可能看到:
- VLA 玩 catch、juggling
- VLA 做 contact-rich assembly(需要快速 force feedback)
- VLA 做 locomotion + manipulation 联合( humanoid 整体控制)

### 11.4 与 NVIDIA GR00T / Gemini Robotics 的关系

GR00T N1(https://arxiv.org/abs/2503.14734) 和 Gemini Robotics(https://arxiv.org/abs/2503.20020)都用了 action chunking。VLASH 的方法对它们 *直接适用*。我预期 NVIDIA 和 Google DeepMind 内部已经在用类似思路,只是没公开。

---

## 12. 公式速查表

| 公式 | 含义 |
|---|---|
| $\pi_\theta(A_t \mid o_t, s_t)$ | policy,参数 $\theta$,给定 obs 和 state 输出 action chunk |
| $A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$ | action chunk,$H$ 是 prediction horizon |
| $I_t^{\text{pred}} = [t, t+K)$ | 预测区间,$K$ 是 execution horizon |
| $I_t^{\text{exec}} = [t+\Delta, t+\Delta+K)$ | 实际执行区间,$\Delta$ 是 inference delay |
| $s_{t+\Delta} = s_t + \sum_{i=0}^{\Delta-1} a_{t+i}$ | state roll-forward |
| $(o_t, s_{t+\delta}) \rightarrow A_{(t+\delta):(t+\delta+H-1)}$ | temporal offset augmentation 训练样本 |
| $\hat{a}_i = \sum_{j=0}^{q-1} a_{iq+j}$ | action quantization,$q$ 是 quant factor |
| $T_{\text{sync}} = T_{\text{infer}} + K/f_{\text{ctrl}}$ | synchronous cycle time |
| $T_{\text{async}} = \max(T_{\text{infer}}, K/f_{\text{ctrl}})$ | asynchronous cycle time(理想情况) |

---

## 13. 参考资源

- **VLASH GitHub**: https://github.com/mit-han-lab/vlash
- **$\pi_{0.5}$ paper**: https://arxiv.org/abs/2504.16054
- **$\pi_0$ paper**: https://arxiv.org/abs/2410.24164
- **SmolVLA**: https://arxiv.org/abs/2506.01844
- **RTC (Real-Time Chunking)**: https://arxiv.org/abs/2506.07339
- **A2C2 (concurrent work)**: https://arxiv.org/abs/2509.23224
- **LIBERO benchmark**: https://arxiv.org/abs/2306.03310
- **Kinetix**: https://arxiv.org/abs/2507.02341 (估计)
- **Gemini Robotics**: https://arxiv.org/abs/2503.20020
- **GR00T N1**: https://arxiv.org/abs/2503.14734
- **AWQ**: https://arxiv.org/abs/2306.00978
- **GPTQ**: https://arxiv.org/abs/2210.17323
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **FlashAttention-3**: https://arxiv.org/abs/2407.08608
- **FlexAttention**: https://pytorch.org/blog/flexattention/
- **PyTorch 2 (torch.compile)**: https://arxiv.org/abs/2401.15044
- **Speculative Decoding**: https://arxiv.org/abs/2302.01318
- **ACT (original action chunking)**: https://arxiv.org/abs/2304.13705
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **LeRobot**: https://github.com/huggingface/lerobot
- **Galaxea R1 Lite**: https://galaxea-ai.com/

---

## 14. 总结:VLASH 的 Elegance

VLASH 让我欣赏的地方:

1. **Insight 简单**: robot state 可以 roll-forward,environment 不可以。这种 asymmetric prediction 是一个 *trivially correct* 的 observation,但没人把它做得这么干净。
2. **Implementation 轻量**: 零架构改动,零 inference 开销,只改 data loader 和加一行 state 加法。
3. **Training augmentation 巧妙**: offset augmentation 通过 *forcing* 模型用 state,顺带修复了 $\pi_{0.5}$ state tokenization 的 broken design。
4. **Shared observation fine-tuning**: 用 block-sparse attention 把 5× 数据增强压缩到 1.27× token 增加,3.26× wall-clock 加速。这是 *systems thinking meets ML*。
5. **Action quantization**: 把 LLM 量化的 philosophy 转移到 robot action,得到 free-lunch speedup。
6. **Demo 极有冲击力**: VLA 打 ping-pong with human,这种 demo 比 100 个 LIBERO 数字更有说服力。

它让我想到一个 broader pattern: 很多 ML 系统的 bottleneck 不在 model expressivity,而在 *interface*——model 与 physical world 的接口设计。VLASH 修的是 temporal interface(async alignment),Action Quantization 修的是 granularity interface。这类工作往往比加 parameter 更有 leverage。

下一个前沿: 把 world model 嵌入 async pipeline,做 $o_{t+\Delta}$ 的 implicit prediction;或者把 action quantization 做成 learned(让模型自己决定哪些 micro-action 可以合并);再或者 extend 到 humanoid whole-body control,那里 state dimension 大得多,roll-forward 的收益更大。

 VLASH 给我一种"definitely the right direction"的感觉。未来的 VLA 部署 stack 一定会有这一层 async + future-state-aware 的 component,正如 LLM serving 现在都有 KV cache + continuous batching 一样。
