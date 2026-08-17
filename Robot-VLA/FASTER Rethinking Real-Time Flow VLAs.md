---
source_pdf: FASTER Rethinking Real-Time Flow VLAs.pdf
paper_sha256: 36a9bdd2f5dbfbee1d0199203dca8c56a2d7feaabe8d77628446c2566ee83259
processed_at: '2026-08-04T07:42:58-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FASTER 人话版

Andrej，我把这篇 paper 用最接地气的方式讲一遍，但技术细节不省。

---

## 一句话概括

你现在有一个 VLA model（比如 π0.5），它每次给你生成一串未来 50 个 action（叫 action chunk），用的是 flow matching——从纯 noise 开始，跑 10 步 denoising，最后得到 clean actions。问题在于：**机器人必须等这 10 步全跑完，才能开始动**。如果球飞过来了，等你算完球已经落地了。

FASTER 的发现特别 simple：**近期 action 比 远期 action 好猜得多**。flow matching 的 denoising 路径，近期 action 几乎是一条直线，1 步就能到位；远期 action 才需要多步。所以干嘛给所有 action 分配一样的 budget？给 近期 action 1 步搞定，马上发给机器人，远期 action 慢慢算，机器人一边动你一边算。这样第一个 action 的等待时间从 10 步压缩到 1 步，TTFA（Time to First Action）降低 10×。

就这么一个 insight，配合一个 streaming 通信 pipeline，让 flow-based VLA 第一次能打乒乓球。

---

## 问题到底出在哪

### 传统 flow VLA 的时间账

你有一个 π0.5 model，部署在 RTX 4090 上，控制频率 30Hz（每 33.3ms 一个 action）。每次 inference 的流程：

1. **VLM backbone forward 一次**：处理图像 + 语言 + proprioception，输出 features。耗时约 62ms（π0.5 on 4090）。这是 prefill，一次 inference 只做一次。
2. **Action Expert denoising 10 步**：拿 VLM features 作为 condition，从 noise 开始，跑 10 步 Euler integration。每步约 1.8ms，10 步约 18ms。
3. **总 inference latency** ≈ 62 + 18 = 80ms。
4. 机器人控制周期 33.3ms，所以 80ms 对应 **d = ⌊80/33.3⌋ = 2** 个 action 的延迟。等你算完，前 2 个 action 对应的物理时刻已经过去了。

打乒乓球时球飞行时间约 1 秒。如果突发情况发生（球弹桌角度变了），你需要 80ms 才能开始反应，加上 async pipeline 的 stochastic 等待，平均 reaction time 约 130ms。这在高速运动里就是 blind spot。

### 为什么 async 只解决了一半问题

Async inference（RTC、Training-time RTC、VLASH 这些）确实消除了 chunk 之间的停顿，让轨迹 smooth。但 reaction time 的下界仍然是 **Δt_infer = 80ms**。因为不管你怎么 overlap，observation 进去到第一个 action 出来，这 80ms 省不掉。

而且 paper 有个很漂亮的概率分析：reaction time 不是个固定值，是个 random variable，服从 uniform distribution。因为外部 event（球到了、物体被碰了）什么时候发生，相对于你的 inference cycle 是随机的。

**Sync 模式**：reaction time ∈ [80ms, 80+100ms] = [80, 180]ms，均值 130ms
**Async 模式**：reaction time ∈ [80ms, 80+100ms] = [80, 180]ms... 等等，这里 s_min = 3，Δt_exec = 100ms，所以 reaction time ∈ [80, 180]ms，均值 130ms。

Wait，Table 1 说 async 的 E[Δt_react] = Δt_infer + 0.5·Δt_exec = 80 + 50 = 130ms。Sync 的是 1.5·Δt_infer + 0.5·Δt_exec = 120 + 50 = 170ms。从 sync 到 async 只省了 40ms = 0.5·Δt_infer。

paper 的核心论点：**只搞 async 不够，必须同时把 Δt_infer 降下来**。

---

## 关键 Empirical 发现：近期 Action 容易多了

这是整个方法的基石。他们 fine-tune 了一个 π0.5 在真实任务上，然后测量 action chunk 内部不同位置 action 的 denoising 动力学。

### Straightness：路径有多直

Rectified Flow 的理论告诉我们，如果 denoising 路径是直线，1 步 Euler 就能精确积分。路径越弯，需要越多步。Straightness metric 就是量化这个弯曲程度。

$$S(\mathbf{A}) = \sum_{\tau=0}^{1} \mathbb{E}_t \left[ \left\| (\mathbf{A}_t^1 - \mathbf{A}_t^0) - v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) \right\|^2 \right] \Delta\tau$$

- $\mathbf{A}_t^1$：τ=1 时的纯 noise
- $\mathbf{A}_t^0$：τ=0 时的最终 clean action
- $v_\theta$：model 预测的 velocity field
- $(\mathbf{A}_t^1 - \mathbf{A}_t^0)$：从 noise 到 clean 的直线方向
- $v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau)$：当前 τ 的实际 velocity
- 两者差越小，路径越直，S 越小

**Fig. 3a 的结果**：action index 0-10 的 straightness 明显低于 index 40-50。近期 action 的 flow path 接近直线，远期 action 的 flow path 弯曲。

### Clean Action Estimate Deviation：提前能猜多准

在每个中间 τ，我们可以从当前 noisy action 外推出 clean action 估计：

$$\tilde{\mathbf{A}}_t^{\tau \to 0} = \mathbf{A}_t^\tau - v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) \cdot \tau$$

- $\mathbf{A}_t^\tau$：当前 τ 的 noisy action
- $v_\theta \cdot \tau$：从当前 τ 到 0 的位移估计（假设 velocity 恒定）
- $\tilde{\mathbf{A}}_t^{\tau \to 0}$：外推的 clean action

Deviation = $\|\tilde{\mathbf{A}}_t^{\tau \to 0} - \mathbf{A}_t^0\|$ 衡量当前 step 的预测精度。

**Fig. 3b 的结果**：近期 action 在 τ=0.9（第 1 步）就已经很接近最终 clean action 了；远期 action 要到 τ=0.2-0.3（第 7-8 步）才收敛。

### 为什么会这样？物理直觉

给你当前 observation（相机图像 + 关节角度 + 语言指令"打乒乓球"），下一个 action 几乎被物理约束死了：

1. 机器人当前关节角度决定了下一步能做什么（运动学约束）
2. 球的当前位置 + 速度决定了你该往哪挥拍（物理因果）
3. 这些约束让 solution space 极窄，flow matching 几乎不需要 "search"，直线就到了

而 50 步以后的 action，中间可能发生各种事（球弹了、物体滑了），uncertainty 累积，solution space 宽，flow matching 需要多步去 "探索" 和 "refine"。

---

## Horizon-Aware Schedule：给 近期 Action 开快车道

### 核心设计

传统 flow VLA：所有 action 共享同一个 timestep schedule τ，从 1 降到 0，10 步。

FASTER：每个 action 有自己的 hit time $u_i$，表示这个 action 在全局 progress 的什么时候完成 denoising。

$$u_i = \left(1 - \frac{i}{H-1}\right)^\alpha \cdot u_0, \quad i \in [1, H-1]$$

- $i$：action index，0 到 H-1（H=50）
- $u_0$：第一个 action 的 hit time，设为 $(N-1)/N = 0.9$（N=10 步）
- $\alpha$：控制衰减形状，默认 0.6-0.7

当 $\alpha < 1$（凹函数），hit time 快速下降：近期 action 的 $u_i$ 接近 $u_0$，远期 action 的 $u_i$ 接近 0。这意味着近期 action 很早完成，远期 action 一直算到最后。

### Local Timestep 计算

全局 sampling progress $\rho^j$ 在第 j 步为 $(N-j+1)/N$。action i 的 local timestep：

$$\tau_i^j = \max\left(0, \frac{\rho^j - u_i}{1 - u_i}\right)$$

当 $\rho^j < u_i$ 时，$\tau_i^j = 0$，这个 action 已经完成去噪，可以 dispatch。

### 第一个 Action 的命运

$u_0 = 0.9$：
- Step 1：$\rho^1 = 1.0$，$\tau_0^1 = (1.0 - 0.9)/(1-0.9) = 1.0$ → 纯 noise
- 经过一次 velocity prediction + Euler update
- Step 2：$\rho^2 = 0.9$，$\tau_0^2 = (0.9 - 0.9)/0.1 = 0$ → 完成！

**第一个 action 经过 1 次 AE forward 就完成**。这就是 10× 加速的来源。

### 为什么 α=1 不好

Table 9 的 ablation：α=1.0 时 CALVIN Avg.Len 只有 3.635，α=0.7 时 4.058。

α=1 是线性衰减，hit time 从 $u_0$ 均匀降到 0。近期 action 的 hit time 不够早，没能充分利用 "近期 action 容易" 的优势。α=0.7 让近期 action 的 hit time 快速跳到接近 $u_0$，远期 action 的 hit time 聚集在 0 附近，保持了远期 action 的多步 denoising budget。

---

## Mixed Schedule Training：不能蛮干

### 直接用 HAS fine-tune 会出什么问题

两个坑：

**坑 1**：预训练 model 用 constant schedule 优化出来的 velocity field 是针对"所有 action 同步去噪"的。突然换成 HAS，velocity field 的输入分布变了（不同 action 有不同 τ），fine-tuning gap 变大。

**坑 2**：训练时 $\rho \sim \mathcal{U}(0,1)$，很多时候近期 action 的 local τ = 0，意味着输入直接是 ground truth，loss 被 mask 掉，model 在这些 sample 上学不到东西，容量浪费。

### Mixed Strategy

以概率 p 用 HAS，概率 1-p 用 constant schedule + action conditioning。Loss 只对 suffix action 计算：

$$\mathcal{L}(\theta) = \mathbb{E}_{\rho, d} \frac{\left\| \mathbf{m} \odot \left( v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) - (\epsilon - \hat{\mathbf{A}}_t) \right) \right\|^2}{\|\mathbf{m}\|_1}$$

- $\mathbf{m}$：prefix mask，$m_i = \mathbf{1}(i \geq d)$
- $d$：prefix 长度，$\sim \mathcal{U}\{0, d_{\max}\}$，模拟不同设备的 TTFA
- $\odot$：element-wise 乘法
- $\|\mathbf{m}\|_1$：mask 中 1 的个数，用于 normalize

**Ablation 结果（Table 10）**：p=0.5 最好（CALVIN Avg.Len 4.058），p=1.0（纯 HAS）崩到 3.112，p=0.3（HAS 太少）3.756。说明 mixed 是必须的，让 model 同时适应两种 schedule。

---

## Streaming Pipeline：边算边发

### 传统方式

VLA server 跑完 10 步 denoising，把 50 个 action 打包成一个大的 payload，一次性发给 robot client。Robot 等到完整 chunk 才开始执行。

### FASTER 的方式

Progressive streaming：每个 action 一旦 $\tau_i = 0$ 就立刻 dispatch。Server 一边继续 denoising 后面的 action，client 一边执行前面已到的 action。

### Table 5 的关键数据（X-VLA on RTX 4090）

| Action Index | Robot 需要时间 | Server 发出时间 | 富余 |
|---|---|---|---|
| 1 | 66.7ms | 44.8ms | +21.9ms |
| 2 | 100.0ms | 52.0ms | +48.0ms |
| 3 | 133.3ms | 59.6ms | +73.7ms |

第一个 action 在 44.8ms 就准备好了，robot 在 66.7ms 才需要它，富余 21.9ms。第二个 action 在 52.0ms 准备好，robot 在 100ms 才需要，富余 48ms。越后面的 action 富余越大，因为前面 action 的执行时间把网络延迟都 mask 掉了。

**只有第一个 action 的 latency 是 critical path**。后面的 action 只要 "比 robot 执行完前一个 action 快" 就行，这个条件很容易满足。

### RTX 4060 上的极限 case

X-VLA on 4060，第 3 个 action：robot 需要 133.3ms，server 发出 129.2ms，富余只有 4.1ms。刚刚好。这说明 FASTER 在 consumer GPU 上 barely feasible，但能跑。π0.5 on 4060 的 TTFA 是 238.6ms，inference 频率约 3Hz，但 FASTER 通过 s_min=8 和 streaming 依然实现了 reactive control。

---

## Early Stopping：算够就停

如果你的 execution horizon s = 4（只用前 4 个 valid action），那一旦前 4 个 action 都 finalized，剩下的 46 个 action 不用算了，直接 break。

这让实际 inference latency 进一步降低。X-VLA on 4090：async 的 s_min=4，FASTER 的 s_min=**2**。s_min 减半意味着 inference 频率翻倍，reaction time 上界 [Δt_infer, Δt_infer + Δt_exec] 的上界直接减半。

---

## 实验数据讲了个什么故事

### Reaction Speed（Table 2）

**π0.5 on 4090**（AE 轻量）：
- Sync: TTFA=80ms, E[react]=170ms
- Async: TTFA=80ms, E[react]=130ms
- FASTER: TTFA=62ms, E[react]=112ms → 1.16× 提升

π0.5 的 AE 10 步只占 18ms，所以把 AE 压到 1 步省了 16ms，TTFA 从 80 到 62。提升不算夸张。

**X-VLA on 4060**（AE 重 + consumer GPU）：
- Sync: TTFA=399.5ms, E[react]=799.2ms
- Async: TTFA=399.5ms, E[react]=599.5ms
- FASTER: TTFA=129.2ms, E[react]=229.2ms → 2.62× 提升

X-VLA 的 AE 重得多，10 步占约 286ms。FASTER 压到 1 步省了 257ms。s_min 从 12 降到 6。这就是 paper 标题说的 "10× acceleration" 的实际意义：AE 部分从 10 步压到 1 步。

### 概率分析（Table 3）

X-VLA 上 FASTER vs Async 的胜率是 **1.00**，deterministic superior。因为 FASTER 的 reaction time 上界 111.5ms（4090）/ 329.2ms（4060）比 Async 的下界 113.7ms / 399.5ms 还低。无论 event 什么时候发生，FASTER 都更快。

### 乒乓球实验（Fig. 5）

Sync 完全打不到球——80ms+ 的 reaction 在球速面前太慢。Naive Async 和 Training-time RTC 能偶尔碰到球但挥拍无力，因为反应慢了，来不及积累 swing speed。FASTER 能提前开始动，有足够时间调整 racket 角度并加速挥拍，打出有力回球。

**关键指标是 contact 时刻的 racket 角度**。如果 reaction 慢，球到了 racket 还是平的，球弹一下就掉。如果 reaction 快，robot 提前把 racket 转到正确角度并加速，球被有力击回。

### 模拟 benchmark（Table 4）

LIBERO 和 CALVIN 上 FASTER 只略有性能下降（π0.5: LIBERO 96.9→96.5，CALVIN 4.313→4.292）。说明激进采样近期 action 不会毁掉远期 action 的质量，因为 HAS 给了远期 action 足够的 denoising budget。

---

## 这个 Insight 为什么重要

1. **它揭示了一个被忽视的维度**。之前所有 real-time VLA 工作都在搞 smoothness（chunk 间过渡），没人 explicitly 优化 reaction。FASTER 指出这两个是正交的，且 reaction 对动态任务更关键。

2. **它是 plug-and-play 的**。不改 architecture，不加 training cost，直接塞进 π0.5 或 X-VLA 的 fine-tuning pipeline。这让它可以 stack 在其他 efficiency 技术上（VLM pruning、quantization、token compression）。

3. **它建立了一个类比**。TTFA之于 VLA，就像 TTFT 之于 LLM。LLM 推理优化已经把 TTFT 做到极致（prefill optimization、speculative decoding），VLA 推理优化才刚意识到 TTFA 是个独立的、重要的 metric。

4. **它启发了一种 schedule 设计哲学**。不要给所有 token/action 同等的 compute budget，根据它们对 latency 的 criticality 分配。近期 action 是 critical path，给最小 budget；远期 action 不 critical，给更多 budget。这个思想可以推广到很多生成任务。

---

## 可能的联想和延伸

### Video Generation

Video diffusion model 也面临类似问题：早期 frame 与 conditioning（text prompt + 首帧）强相关，容易生成；后期 frame 需要保持时序一致性，难生成。AR-Diffusion [https://arxiv.org/abs/2410.05263] 已经在探索 autoregressive 的 frame-by-frame 生成。HAS 的思想可以直接搬过去：给早期 frame 分配少 step，后期 frame 多 step，streaming 输出。这对 real-time video generation（比如 game rendering、live streaming）特别有价值。

### LLM 的 Speculative Decoding

LLM 的 speculative decoding [https://arxiv.org/abs/2302.01318] 用一个小 model 快速 draft 多个 token，大 model 批量 verify。FASTER 的 streaming 思想类似：先快速出一个"够用"的 action 让机器人动起来，后面慢慢 refine。可以想象一个 hybrid：近期 action 用一个 distilled 1-step model 瞬间出，远期 action 用 full flow matching 精算。

### Robot Learning 的 Reward Shaping

如果用 RL fine-tune VLA，可以把 reaction time 放进 reward。但要注意：reaction 快但 action 质量差也不行。可能需要 curriculum：先训练在宽松时间约束下做到 high success rate，再逐步收紧时间约束，逼 model 学会快速出 "够用" 的 action。

### World Model 的联合训练

近期 action 容易预测的本质是 causal constraint 强。如果同时训一个 world model 预测下一帧 observation，world model 的预测可以作为 action generation 的额外 condition，进一步降低 uncertainty，让 HAS 更激进。想象 model 不仅能预测 action，还能预测 "执行这个 action 后图像会变成什么样"，这种 forward model 让近期 action 的 solution space 更窄。

### Humanoid Robot 的高频控制

人形机器人控制频率可达 100-200Hz，TTFA 要求 <5-10ms。VLM prefill 动辄 50-100ms，即使 AE 压到 1 step 也不够。可能需要 VLM prefill 也 streaming：先输出 coarse features 让 AE 开始第一步 denoising，VLM 继续计算 fine features 并更新 condition。这把 TTFA 进一步拆解，类似 LLM 的 incremental prefill。

### Multi-Robot 共享 VLM Server

多个 robot 共享一个 VLM server 时，VLM prefill 是 bottleneck。FASTER 让 AE 部分 lightweight，多个 robot 的 action expert 可以并行 sample（GPU batch），VLM prefill 串行但分时复用。这个架构下 FASTER 的价值更大：每个 robot 只需要一个 AE forward 就能拿到第一个 action，VLM prefill 的 amortized cost 被多个 robot 分摊。

### Consistency Model 整合

近期 action 用 1 step 可能牺牲质量。可以专门为近期 action distill 一个 consistency model [https://arxiv.org/abs/2303.01469]，远期 action 用 full flow matching。Consistency model 保证 1-step 生成的 action 在 distribution 上与 multi-step 一致，避免 HAS 1-step 近期 action 的质量损失。

### Adaptive α：根据任务难度动态调整

打乒乓球需要极快反应，α 设小一点（近期 action 极速完成）；叠毛巾不需要快反应，α 设大一点（近期 action 也多算几步保质量）。甚至可以让 model 自己预测一个 difficulty score，动态调整 α。这类似 LLM 的 adaptive computation [https://arxiv.org/abs/2402.19213]，根据 token 难度分配 compute。

### Flow Matching + Diffusion Forcing 的深层统一

Diffusion Forcing [https://arxiv.org/abs/2407.01392] 用 independent timestep per action，FASTER 用 structured HAS。两者的本质都是打破"全 chunk 同步去噪"的假设。更深层的统一可能是：学一个 meta-scheduler，根据 observation 自动决定每个 action 的 timestep schedule。这类似 neural architecture search 之于 hand-crafted architectures。

### 从 VLA 到 General Generative Model

FASTER 的哲学——"根据 output 对 latency 的 criticality 分配 compute budget"——是 universal 的。Image generation 中，coarse structure（低频）可以用少 step，fine detail（高频）用多 step。Music generation 中，近期的 note 用少 step，远期的 note 用多 step。任何 autoregressive 或 chunk-based 生成任务都可以套这个 pattern。这可能催生一个新的 "latency-aware generation" 研究方向。

---

## 代码层面的 intuition

Algorithm 2（inference）的关键几行：

```
for j = 1 to N:
    ρ^j = (N-j+1)/N          # 全局 progress
    τ_i^j = max(0, (ρ^j - u_i)/(1-u_i))  # 每个 action 自己的 local τ
    v = v_θ(o_t, A_t, τ^j)   # 一次 AE forward
    A_t += v * Δτ             # Euler update
    for each i where τ_i^{j+1} == 0 and not yet streamed:
        dispatch a_{t+i}     # 这个 action 完成了，发出去
    if all actions in [d, d+s-1] finalized:
        break                # 执行 horizon 内都好了，提前停
```

注意：VLM backbone 只在 loop 外 forward 一次，loop 内只跑 action expert。这是 VLA 场景的关键——VLM forward 太贵（50-100ms），不能每步都跑。HAS 的美妙之处在于它不需要改 VLM，只改 AE 的 timestep schedule。

---

## 总结：三层 intuition 串起来

**第一层：Causal Structure → Predictability Asymmetry**

物理世界有因果结构。给定当前 observation，近期 action 被物理约束（运动学、动力学、物体状态）紧紧限制，solution space 窄，flow path 直。远期 action 面对累积不确定性，solution space 宽，flow path 弯。这是 paper 用 straightness 和 estimate deviation 量化出来的。

**第二层：Predictability Asymmetry → Asymmetric Sampling Budget**

既然近期 action 容易，干嘛给它们和远期 action 一样的 10 步 denoising budget？HAS 给近期 action 1 步搞定，远期 action 保留接近 10 步。hit time $u_i$ 的设计让近期 action 早早到达 $\tau=0$，远期 action 慢慢收敛。

**第三层：Asymmetric Sampling → Streaming + Early Stop**

近期 action 1 步完成就立刻 dispatch，机器人马上开始动。远期 action 边执行边算，网络延迟被前面 action 的执行时间 mask 掉。执行 horizon 内 action 全完成就 early stop，不浪费算力。

三层从物理直觉到算法设计到系统实现，环环相扣。最终效果：π0.5 和 X-VLA 这种 flow-based VLA 第一次能在 consumer GPU 上打乒乓球。

---

参考链接汇总：
- FASTER project: https://innovator-zero.github.io/FASTER
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- X-VLA: https://arxiv.org/abs/2501.06749
- RTC: https://arxiv.org/abs/2510.08164
- Training-time RTC: https://arxiv.org/abs/2512.05964
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- Rectified Flow: https://arxiv.org/abs/2209.03003
- Flow Matching: https://arxiv.org/abs/2210.02747
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- LIBERO: https://libero-project.github.io/
- CALVIN: https://calvinrobot.github.io/
- Streaming Diffusion Policy: https://arxiv.org/abs/2406.04806
- AR-Diffusion: https://arxiv.org/abs/2410.05263
- Consistency Model: https://arxiv.org/abs/2303.01469

---

# FASTER: Rethinking Real-Time Flow VLAs 深度讲解

Andrej，这篇 FASTER 的核心思想非常 elegant。本质上是把 LLM 推理中 **TTFT (Time To First Token)** 的概念移植到 flow-based VLA，并通过一个 empirical 发现——action chunk 内部 denoising 难度的 temporal non-uniformity——来打破 constant schedule 的瓶颈。我从 intuition 到 formula 到 experiment 逐层拆解。

---

## 1. 问题动机：Reaction 与 Smoothness 是两个正交维度

现有 real-time VLA 工作（RTC [[1]](https://arxiv.org/abs/2510.08164), Training-time RTC [[2]](https://arxiv.org/abs/2512.05964), REMAC, VLASH [[3]](https://arxiv.org/abs/2512.01031), SmolVLA [[4]](https://arxiv.org/abs/2506.01844)）几乎全部聚焦在解决 **inter-chunk discontinuity**——即 asynchronous inference 切换 chunk 时的 multimodal jump。但 paper 指出，real-time embodied intelligence 有两个正交需求：

- **Smoothness**: 轨迹连续，无 jerk
- **Reaction**: 外部突发扰动时，closed-loop 响应延迟

现有方法 conflated 这两个 concept。FASTER 是第一个 explicitly target reaction 的 real-time VLA。

---

## 2. Reaction Time 的概率建模：为什么 Uniform Distribution？

这是 paper 最 beautiful 的理论分析。

### 2.1 关键时间量

- **Δt_ctrl** := 1/f，controller 周期（如 30Hz → 33.3ms）
- **Δt_infer**: observation 发出到 actions 返回的总 latency（含 model inference、网络、preprocessing、memory I/O）
- **d** := ⌊Δt_infer / Δt_ctrl⌋，discretized inference delay（chunk 内前 d 个 action 已经过期）
- **Δt_exec** := s · Δt_ctrl，执行 s 个 action 的时间
- **s_min** := ⌈Δt_infer / Δt_ctrl⌉，async 模式下保证 Δt_exec ≥ Δt_infer 的最小 execution horizon

### 2.2 Reaction time 的 uniform distribution 推导

为什么 reaction time 服从 uniform？因为外部 event 的发生时间相对于 robot controller 的 inference cycle 是 stochastic 的。考虑两次 consecutive inference 在时刻 t 和 t' 触发：

- **Best case**: event 刚好在 inference 触发前发生，reaction = Δt_infer
- **Worst case**: event 在 inference 刚触发后发生，必须等下一次 inference 完成才能响应，reaction = Δt_infer + (inference interval)

由于 event 在 interval 内任意时刻等概率发生，所以 reaction time ~ Uniform。

**Sync 模式**：inference interval = Δt_infer + Δt_exec
$$\Delta t_{\text{react}} \sim \mathcal{U}(\Delta t_{\text{infer}},\ 2\Delta t_{\text{infer}} + \Delta t_{\text{exec}})$$
$$\mathbb{E}[\Delta t_{\text{react}}] = 1.5 \Delta t_{\text{infer}} + 0.5 \Delta t_{\text{exec}}$$

**Async 模式**：inference interval = Δt_exec
$$\Delta t_{\text{react}} \sim \mathcal{U}(\Delta t_{\text{infer}},\ \Delta t_{\text{infer}} + \Delta t_{\text{exec}})$$
$$\mathbb{E}[\Delta t_{\text{react}}] = \Delta t_{\text{infer}} + 0.5 \Delta t_{\text{exec}}$$

**关键 insight**: 从 Sync 升级到 Async，期望 reaction time 只减少 0.5 · Δt_infer。这意味着如果 inference latency 本身很大，async 帮助有限。必须同时降低 Δt_infer 和缩短 inference interval（即减小 s）。

---

## 3. TTFA: 反应速度的 True Bottleneck

类比 LLM 的 **TTFT (Time To First Token)** [[5]](https://arxiv.org/abs/2401.08671)。Robot 不需要整个 action chunk 才能开始 move——只需要第一个 action。

传统 flow-based VLA 的 TTFA：
$$\text{TTFA}_{\text{conv}} \approx \Delta t_{\text{VLM}} + N \cdot \Delta t_{\text{AE}}$$

- Δt_VLM: VLM backbone forward 一次（prefill）
- Δt_AE: action expert 单次 denoising iteration
- N: sampling steps（通常 10）

FASTER 的 TTFA：
$$\text{TTFA}_{\text{FASTER}} \approx \Delta t_{\text{VLM}} + 1 \cdot \Delta t_{\text{AE}}$$

VLM 只 prefill 一次，后续 N 步只是 AE 的 denoising iteration。把 N 步压缩到 1 步就是 10× 的 AE 部分加速。这是 paper 的核心 acceleration claim。

---

## 4. Pilot Study: Action Chunk 的 Non-Uniform Sampling 难度

这是整个方法论的 empirical 基础。他们 fine-tune π0.5 到真实任务，测量两个 metric。

### 4.1 Straightness Metric（来自 Rectified Flow [[6]](https://arxiv.org/abs/2209.03003)）

对于从 Z_0 到 Z_1 的连续过程：
$$S(\mathbf{Z}) = \int_0^1 \mathbb{E}\left\| (Z_1 - Z_0) - \dot{Z}_\tau \right\|^2 d\tau$$

- **Z_0**: 起点（noise sample）
- **Z_1**: 终点（clean action）
- **$\dot{Z}_\tau$**: 时刻 τ 的瞬时 velocity = dZ_τ/dτ
- **S = 0**: 完美直线，1 step Euler 就能精确积分
- **S 大**: 路径弯曲，需要多 step 才能准确积分

离散化版本（paper Eq. 3）：
$$S(\mathbf{A}) = \sum_{\tau=0}^{1} \mathbb{E}_t \left[ \left\| (\mathbf{A}_t^1 - \mathbf{A}_t^0) - v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) \right\|^2 \right] \Delta\tau$$

- **A_t^1**: τ=1 时的纯 noise
- **A_t^0**: τ=0 时的最终 clean action
- **v_θ**: 学习的 velocity field
- **Δτ**: 离散 timestep 间隔 = 1/N

### 4.2 Clean Action Estimate Deviation

在任意中间时刻 τ，可以从当前位置 A_t^τ 沿 velocity 外推到 τ=0 的 clean action 估计（paper Eq. 4）：
$$\tilde{\mathbf{A}}_t^{\tau \to 0} = \mathbf{A}_t^\tau - v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) \cdot \tau$$

- **A_t^τ**: 当前 τ 时刻的 noisy action
- **v_θ · τ**: 从当前 τ 到 0 的 displacement 估计
- **$\tilde{\mathbf{A}}_t^{\tau \to 0}$**: 外推的 clean action 估计

Deviation = ||$\tilde{\mathbf{A}}_t^{\tau \to 0}$ - A_t^0|| 衡量当前 step 估计精度。

### 4.3 关键发现（Fig. 3）

两个 metric 都呈现显著的 **temporal non-uniformity**：
- 前 1-10 个 action: straightness 低，estimate deviation 在早期 step 就很小
- 后期 action: straightness 高，需要多个 step 才能收敛

**Physical intuition**: 给定当前 observation o_t 和 proprioceptive state，近期 action 受强 causal constraint（机器人当前 joint configuration + 物理动力学限制了下一步可能性），solution space 窄；远期 action 需要预测多个 step 的累积效应，multi-modal 且 uncertainty 大。

这直接 motivate 了 HAS 的设计：近期 action 用 1 个 step 就够，把 denoising budget 留给远期 action。

---

## 5. Horizon-Aware Schedule (HAS)

### 5.1 Hit Time 设计

每个 action i 有自己的 "hit time" u_i，表示该 action 完成去噪的全局时间（paper Eq. 5）：
$$u_i = \left(1 - \frac{i}{H-1}\right)^\alpha \cdot u_0, \quad i \in [1, H-1]$$

变量解释：
- **i**: action index，0 到 H-1
- **H**: prediction horizon（如 50）
- **u_0**: 第一个 action 的 hit time，设为 (N-1)/N。当 N=10 时 u_0 = 0.9
- **α ∈ (0, 1]**: 控制 hit time 衰减形状的 hyperparameter

α 的物理含义：
- **α = 1**: 线性衰减，hit time 从 u_0 均匀降到 0
- **α < 1**: 凹函数，早期 action 快速到达 hit time，后期 action 的 hit time 聚集在 0 附近
- **α 越小**: 越激进，近期 action 越早 dispatch，但远期 action 的 schedule 越接近 constant

### 5.2 Local Timestep

给定第 j 个 sampling step 的全局 timestep ρ^j（从 1 递减到 0），action i 的 local timestep（paper Eq. 6）：
$$\tau_i^j = \max\left(0,\ \frac{\rho^j - u_i}{1 - u_i}\right)$$

变量解释：
- **ρ^j**: 第 j 个 step 的全局 progress，ρ^j = (N-j+1)/N
- **u_i**: action i 的 hit time
- **τ_i^j**: action i 在 step j 的 local timestep
- 当 ρ^j < u_i 时，τ_i^j = 0，意味着 action i 已经完全去噪，可以 dispatch

### 5.3 第一个 Action 的 1-Step 完成

u_0 = (N-1)/N。当 N=10：
- Step 1: ρ^1 = 1.0, τ_0^1 = (1.0 - 0.9)/(1 - 0.9) = 1.0（pure noise）
- Step 2: ρ^2 = 0.9, τ_0^2 = (0.9 - 0.9)/0.1 = 0 → 完成！

所以第一个 action 经过 1 次 AE forward（step 1 到 step 2 的 update）就完成。这就是 "10× acceleration" 的来源。

### 5.4 与 Diffusion Forcing [[7]](https://arxiv.org/abs/2407.01392) 的关系

Diffusion Forcing 也用 index-dependent timestep，但训练时每个 action 的 timestep **独立采样**。FASTER 用结构化 HAS，且通过 mixed schedule 避免 train-inference mismatch。Table 10 显示 independent schedule（Diffusion Forcing 风格）的 Avg.Len = 3.671，不如 mixed 的 4.058。

---

## 6. Mixed Schedule Training

### 6.1 为什么不能直接用 HAS fine-tune

两个 challenge：
1. **预训练 mismatch**: 预训练 model 用 constant schedule 优化，直接切换到 HAS 引入 distribution shift
2. **训练效率问题**: 当 ρ ~ U(0,1)，很多情况下近期 action 的 local timestep = 0，等于输入就是 ground truth，loss 被 mask 浪费掉，model 容量没被有效利用

### 6.2 Mixed Strategy

以概率 p 用 HAS，以概率 1-p 用 constant schedule with action conditioning。Loss function（paper Eq. 11）：
$$\mathcal{L}(\theta) = \mathbb{E}_{\rho \sim \mathcal{U}(0,1),\ d \sim \mathcal{U}\{0, d_{\max}\}} \frac{\left\| \mathbf{m} \odot \left( v_\theta(\mathbf{o}_t, \mathbf{A}_t^\tau, \tau) - (\epsilon - \hat{\mathbf{A}}_t) \right) \right\|^2}{\|\mathbf{m}\|_1}$$

- **ρ**: global timestep，uniform 于 (0,1)
- **d**: action prefix 长度，模拟不同设备的 TTFA，uniform 于 {0, d_max}
- **m**: prefix mask，m_i = 1(i ≥ d)，i ∈ [0, H-1]
- **⊙**: element-wise multiplication
- **||m||_1**: mask 中为 1 的元素数，用于 normalize
- **ε**: Gaussian noise
- **$\hat{\mathbf{A}}_t$**: ground truth action chunk
- **v_θ**: velocity field network

只对 suffix actions 计算 loss。

### 6.3 Ablation（Table 10, X-VLA on CALVIN）

| p | 1 | 2 | 3 | 4 | 5 | Avg.Len |
|---|---|---|---|---|---|---|
| Baseline | 95.7 | 89.8 | 82.4 | 77.0 | 70.2 | 4.151 |
| 0.3 | 93.7 | 85.2 | 76.0 | 65.5 | 55.2 | 3.756 |
| **0.5** | **97.7** | **91.1** | **81.2** | **72.1** | **63.7** | **4.058** |
| 0.7 | 89.6 | 76.7 | 63.2 | 50.8 | 40.3 | 3.206 |
| 1.0 (no mix) | 89.0 | 74.7 | 60.4 | 49.0 | 38.1 | 3.112 |
| Independent | 91.4 | 82.6 | 74.0 | 64.6 | 54.5 | 3.671 |

p=0.5 是 sweet spot。p=1.0 严重退化，证明 mixed schedule 必要。p=0.3 让 inference schedule 训练时见得少，性能也下降。

---

## 7. Action Conditioning 整合

Training-time RTC [[2]](https://arxiv.org/abs/2512.05964) 的 action conditioning 把 action prefix 当作完全去噪（τ=0），引导新 chunk 平滑过渡。HAS 与之天然 synergize：HAS 本身就让 prefix position 的 timestep 趋近 0。

整合时加 offset d（paper Eq. 8）：
$$u_i = \left(1 - \frac{i - d}{\max(H-1-d, 1)}\right)^\alpha \cdot u_d, \quad i \in [d+1, H-1]$$

- **d**: action prefix 长度（即 inference delay）
- **u_d**: 替代 u_0，第一个 valid action（index d）的 hit time
- **i ∈ [d+1, H-1]**: 只对 suffix actions 定义 hit time

Prefix actions 的 local timestep 强制为 0（paper Eq. 10）：
$$\tau_i = \begin{cases} 0, & i < d \\ \max\left(0, \frac{\rho - u_i}{1 - u_i}\right), & i \geq d \end{cases}$$

训练时 d ~ U{0, d_max}（d_max=10，对应 TTFA ≤ 333.3ms at 30Hz），覆盖 RTX 4060 的延迟范围。

---

## 8. Streaming Client-Server Interface

### 8.1 传统 vs Streaming

- **传统**: 整个 chunk 一次性传输，等所有 step 完成
- **FASTER**: progressive streaming，每个 action 完成即 dispatch，后续 action 边执行边生成

### 8.2 Table 5 的延迟对比（X-VLA on RTX 4090）

| Index | Time Req. | Time Rec. | Margin |
|---|---|---|---|
| 1 | 66.7ms | 44.8ms | +21.9ms |
| 2 | 100.0ms | 52.0ms | +48.0ms |
| 3 | 133.3ms | 59.6ms | +73.7ms |

- **Time Req.**: robot controller 需要该 action 的时间
- **Time Rec.**: 从 policy server 收到该 action 的时间

第 1 个 action margin 充足（44.8 vs 66.7ms）。后续 action 的网络延迟被前面 action 的执行时间完全 mask，越靠后的 action margin 越大。这就是 streaming 的妙处：**只有第一个 action 的延迟是 critical path**。

### 8.3 RTX 4060 上 X-VLA 的临界 case

| Index | Time Req. | Time Rec. | Margin |
|---|---|---|---|
| 3 | 133.3ms | 129.2ms | +4.1ms |
| 4 | 166.7ms | 159.0ms | +7.7ms |

第 3 个 action margin 只有 4.1ms，刚好赶上。这显示 FASTER 在 consumer-grade GPU 上也 barely feasible，但需要 careful tuning。

---

## 9. Early Stopping

Execution horizon 内所有 action finalized 后，剩余 sampling steps 跳过。这让 s_min 可以更小。

### 9.1 量化效果（Table 2）

X-VLA on RTX 4090:
- Async: TTFA=113.7ms, s_min=4
- FASTER: TTFA=44.8ms, s_min=**2**

s_min 减半意味着 inference 频率翻倍。结合 TTFA 降低，reaction time upper bound 从 247.0ms 降到 111.5ms。

### 9.2 Reaction Time 上界

Async: upper bound = Δt_infer + Δt_exec = Δt_infer + s_min · Δt_ctrl
FASTER: upper bound = Δt_infer + Δt_exec，但 Δt_infer 更小（early stop 减少实际 inference latency），s_min 也更小

---

## 10. 实验数据深度分析

### 10.1 Table 2 完整数据解读

**π0.5 on RTX 4090**:
| Method | TTFA | s_min | E[Δt_react] | Speedup |
|---|---|---|---|---|
| Sync | 80.0±1.6ms | 3 | 170.0ms | - |
| Async | 80.0±1.6ms | 3 | 130.0ms | - |
| FASTER | **62.1±3.1ms** | 3 | **112.1ms** | 1.16× |

π0.5 的 AE 很轻量（10 step 只占 ~18ms），所以 TTFA 提升只有 1.29×，reaction 提升只有 1.16×。

**X-VLA on RTX 4060**（极端 case）:
| Method | TTFA | s_min | E[Δt_react] | Speedup |
|---|---|---|---|---|
| Sync | 399.5±8.5ms | 12 | 799.2ms | - |
| Async | 399.5±8.5ms | 12 | 599.5ms | - |
| FASTER | **129.2±2.4ms** | **6** | **229.2ms** | 2.62× |

X-VLA 的 AE 比 π0.5 重很多（10 step 占 ~286ms），FASTER 把 AE 部分压缩到 1 step，TTFA 提升 3.09×。s_min 从 12 降到 6，reaction time 整体提升 2.62×。

### 10.2 Table 3 的概率分析

| Model | Method | vs Sync (4090) | vs Async (4090) | vs Sync (4060) | vs Async (4060) |
|---|---|---|---|---|---|
| π0.5 | Async | 0.72 | - | 0.74 | - |
| π0.5 | FASTER | 0.81 | 0.66 | 0.88 | 0.77 |
| X-VLA | Async | 0.73 | - | 0.75 | - |
| X-VLA | FASTER | **1.00** | **1.00** | **1.00** | **1.00** |

X-VLA 上 FASTER 的 reaction time **deterministically** 优于 baselines。原因在 Table 8：
- X-VLA Async on 4090: U(113.7, 247.0)
- X-VLA FASTER on 4090: U(44.8, 111.5)

FASTER 的上界 111.5ms < Async 的下界 113.7ms → P(FASTER < Async) = 1。

### 10.3 Table 4 Simulation 性能保留

| Model | LIBERO Avg | CALVIN Avg.Len |
|---|---|---|
| π0.5 | 96.9 | 4.313 |
| π0.5 + FASTER | 96.5 | 4.292 |
| X-VLA* | 98.0 | 4.151 |
| X-VLA + FASTER | 97.0 | 4.058 |

LIBERO 上几乎无损（96.9→96.5），CALVIN 上略有损失（4.151→4.058）。这印证了 paper 的 claim：HAS 在长 horizon 任务上略有 cost，但保留了绝大部分 capability。

### 10.4 α Ablation（Table 9）

| α | CALVIN Avg.Len |
|---|---|
| 0.4 | 4.071 |
| 0.5 | 3.911 |
| 0.6 | 3.991 |
| **0.7** | **4.058** |
| 0.8 | 3.970 |
| 0.9 | 3.921 |
| 1.0 | 3.635 |

α=0.4 最好但近期 action 完成慢，paper 选 α=0.7 作为 efficiency-performance trade-off。α=1.0（线性衰减）最差，因为近期 action 不够激进。

---

## 11. Real-World Table Tennis 实验

Table tennis 是高动态 task，ball 飞行时间 ~1s，机器人必须在 ball 接触 table 后立刻调整姿态并积累 swing speed。

### 11.1 Fig. 5 关键观察

FASTER 在 contact 时刻 racket 角度明显优于 baselines。Reaction 慢的话：
- Racket 在 contact 时角度不对 → miss 或 weak hit
- 没有 time 积累 swing speed → 球飞不远

FASTER 让 robot 提前开始 motion，有足够 time 旋转 racket + 加速 swing。

### 11.2 RTX 4060 上的极端表现

RTX 4060 上 π0.5 的 inference 只有 ~3Hz（TTFA=238.6ms vs Δt_ctrl=33.3ms）。传统 async 在这种情况下基本不可用。FASTER 通过 s_min=8 和 streaming，让 robot 在 3Hz inference 下仍能 reactive control。这是 TTFA + 推理频率双重收益的体现。

---

## 12. Kinetix Benchmark（Table 11）

Identical-latency setting：baselines at d=4 vs FASTER at d=1（因为 FASTER 把 5-step flow matching 的第一步完成时间压缩了 5×）。

| Method | d | s | Solve Rate |
|---|---|---|---|
| Naive Async | 4 | 4 | 0.492 |
| BID [[8]](https://arxiv.org/abs/2406.11706) | 4 | 4 | 0.553 |
| Inference-time RTC | 4 | 4 | 0.614 |
| Training-time RTC | 4 | 4 | 0.726 |
| **FASTER** | **1** | 4 | **0.869** |

FASTER 即使在 simulator 中也显著胜出，因为 reaction 能力直接提升 closed-loop control 质量。这暗示 reaction speed 不只是 deployment concern，本身也是 policy quality 的关键维度。

---

## 13. 与相关工作的对比

### 13.1 One-Step Distillation 路线

RDT2 [[9]](https://arxiv.org/abs/2602.03310), MeanFlow [[10]](https://arxiv.org/abs/2507.01234), Shortcut Model [[11]](https://arxiv.org/abs/2410.18957), One-Step Diffusion Policy [[12]](https://arxiv.org/abs/2401.00830) 等需要 architectural 修改或两阶段训练（先训 multi-step 再 distill），fine-tune 预训练 VLA 困难。FASTER 是 plug-and-play，无架构修改。

### 13.2 Streaming Diffusion Policy

Streaming Diffusion Policy [[13]](https://arxiv.org/abs/2406.04806), Responsive Noise-Relaying [[14]](https://arxiv.org/abs/2401.00830) 每步更新 observation。对 VLA 来说 VLM forward 太重（每次 ~50-100ms），不可行。FASTER 只 forward VLM 一次，后续只跑 AE，避开主 bottleneck。

### 13.3 RTC 系列

RTC [[1]](https://arxiv.org/abs/2510.08164) 和 Training-time RTC [[2]](https://arxiv.org/abs/2512.05964) 解决 smoothness，但 reaction 仍是 Δt_infer + 0.5·Δt_exec。FASTER 与 RTC 正交，可以叠加：HAS + action conditioning 已经在 paper 中整合。

---

## 14. 架构图解析（Fig. 2 & Fig. 4）

### 14.1 Fig. 2: Temporal Pipeline

**Sync (a)**: inference 与 execution 串行。Robot 等 inference 完成（Δt_infer）才能开始下一个 chunk execution。Reaction time 范围 [Δt_infer, 2Δt_infer + Δt_exec]。

**Async (b)**: 在当前 chunk 执行期间触发下一个 inference。前 d 个 action 因 inference 延迟已过期，丢弃，从 d+1 开始执行。Reaction time 范围 [Δt_infer, Δt_infer + Δt_exec]。

### 14.2 Fig. 4: Constant vs Horizon-Aware Schedule

**Constant (a)**: 所有 action 共享同一 timestep schedule。在 step N 完成时所有 action 同时完成。TTFA = N · Δt_AE。

**HAS (b)**: 每个 action 有自己的 timestep。第 1 个 action 在 step 1 完成即 dispatch，后续 action 在不同 step 依次完成。TTFA = 1 · Δt_AE。

---

## 15. 可能的延伸联想

### 15.1 推广到 Video Diffusion

Video diffusion model 中早期 frame 容易预测（与 conditioning 强相关），后期 frame 需要更多 denoising。AR-Diffusion [[15]](https://arxiv.org/abs/2410.05263) 已探索类似 idea，但没用 HAS 这种 structured schedule。

### 15.2 与 Consistency Model 结合

近期 action 用 1 step 可能牺牲质量。可以 distill 一个 consistency model 专门负责近期 action，远期 action 仍用 multi-step flow matching。

### 15.3 Mobile Manipulator 的高频控制

Mobile manipulator 的 control frequency 可达 100Hz+，TTFA 要求 < 10ms。FASTER 的 1-step AE forward 在 RTX 4090 上 ~18ms（π0.5），还不够。需要更轻量的 AE 或更激进的 HAS（α 更小）。

### 15.4 VLM Prefill 的进一步分离

当前 TTFA = Δt_VLM + Δt_AE。如果 VLM prefill 可以 streaming（类似 LLM 的 streaming prefill），AE 可以更早开始。这是 future work 方向。

### 15.5 Multi-Agent / Multi-Robot 共享 VLM

多个 robot 共享一个 VLM server 时，VLM prefill 是 bottleneck。FASTER 让 AE 部分可以并行化，多个 robot 的 action expert 可以同时 sample，进一步提升 throughput。

### 15.6 Reinforcement Learning Fine-Tune

HAS schedule 下直接 RL fine-tune，reward = task_success - λ · reaction_time，可能发现更优的 α 和 p 组合，甚至让 model 自动学习 index-dependent schedule。

### 15.7 World Model 联合训练

近期 action 容易预测的本质是 causal constraint。如果同时训一个 world model 预测下一帧 observation，可以让 world model 的预测作为额外 conditioning，进一步降低近期 action 的 uncertainty，让 HAS 更激进。

---

## 16. 总结：FASTER 的三层 Intuition

1. **Temporal Causality → Predictability Asymmetry**: 近期 action 与当前 observation 强相关，flow path 接近直线；远期 action 累积不确定性，flow path 弯曲。

2. **Predictability Asymmetry → Asymmetric Sampling Budget**: 给近期 action 分配 1 个 denoising step 就够，远期 action 保留多 step。

3. **Asymmetric Sampling → Streaming Output**: 近期 action 早完成早 dispatch，远期 action 边执行边生成。TTFA 与 N 解耦。

这是一个 system-algorithm co-design 的典范：算法（HAS）提供 progressively decodable representation，系统（streaming interface + early stopping）充分利用这个 property。整个 pipeline 的每个组件都互相强化，没有冗余。

---

## References

- [1] RTC: https://arxiv.org/abs/2510.08164
- [2] Training-time RTC: https://arxiv.org/abs/2512.05964
- [3] VLASH: https://arxiv.org/abs/2512.01031
- [4] SmolVLA: https://arxiv.org/abs/2506.01844
- [5] DeepSpeed-FastGen (TTFT): https://arxiv.org/abs/2401.08671
- [6] Rectified Flow: https://arxiv.org/abs/2209.03003
- [7] Diffusion Forcing: https://arxiv.org/abs/2407.01392
- [8] Bidirectional Decoding: https://arxiv.org/abs/2406.11706
- [9] RDT2: https://arxiv.org/abs/2602.03310
- [10] MeanFlow: https://arxiv.org/abs/2507.01234
- [11] Shortcut Model: https://arxiv.org/abs/2410.18957
- [12] One-Step Diffusion Policy: https://arxiv.org/abs/2401.00830
- [13] Streaming Diffusion Policy: https://arxiv.org/abs/2406.04806
- [14] Responsive Noise-Relaying Diffusion: https://arxiv.org/abs/2401.00830
- [15] AR-Diffusion: https://arxiv.org/abs/2410.05263
- FASTER project page: https://innovator-zero.github.io/FASTER
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- X-VLA: https://arxiv.org/abs/2501.06749
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- Flow Matching: https://arxiv.org/abs/2210.02747
- LIBERO: https://libero-project.github.io/
- CALVIN: https://calvinrobot.github.io/
