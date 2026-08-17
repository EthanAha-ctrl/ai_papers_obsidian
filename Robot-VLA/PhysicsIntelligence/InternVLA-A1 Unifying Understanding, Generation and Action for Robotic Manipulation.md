---
source_pdf: InternVLA-A1 Unifying Understanding, Generation and Action for Robotic
  Manipulation.pdf
paper_sha256: a94c32fc45223a9b4eca28059961a196903d61d0c41ebf10acce68e2f6010c94
processed_at: '2026-08-05T10:17:28-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 InternVLA-A1

Andrej, 咱们把 academic 包装扒掉, 用大白话重新过一遍这篇 paper 的骨架.

---

## 一句话总结

现有 VLA 模型像个只会"看到啥就做啥"的 reactive 反应机器, 一遇到场景会动的 (传送带) 就傻. InternVLA-A1 的 idea 是: 给它装个"想象力", 让它先在脑子里模拟"下一秒场景会变成啥样", 再决定动作. 这个"想象力"是个 lightweight 的小模块, 不追求画面精美, 只追求 dynamics 趋势对. 结果在 dynamic 任务上比 π0.5 强 +26.7%.

---

## 1. 这帮人到底在 solve 什么 problem

机器人圈现在两派打架:

**派别 A: MLLM 派** (π0 [arxiv 2410.24164](https://arxiv.org/abs/2410.24164), π0.5 [arxiv 2504.16054](https://arxiv.org/abs/2504.16054), GR00T [arxiv 2503.14734](https://arxiv.org/abs/2503.14734))
- 思路: 把 vision + language 喂给 LLM, LLM 输出 action
- 优点: 语义理解强, 能听懂 "把那个红色的杯子拿过来"
- 缺点: text token 天生不擅长 modeling momentum, contact force 这种连续物理量. 你让 LLM "想"一个杯子从传送带掉下来的轨迹, 它其实就是在猜, 因为它的 token space 里压根没有这种 continuous dynamics 的 representation

**派别 B: World Model 派** (UniPi [NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023), VPP [ICML 2025](https://arxiv.org/abs/2412.14803), Genie Envisioner [arxiv 2508.05635](https://arxiv.org/abs/2508.05635))
- 思路: 先用 video generation model 预测未来视频, 再用 inverse dynamics 从视频反推 action
- 优点: 能 capture dynamics, 有 "想象力"
- 缺点: video generation 跟 task instruction 耦合松 (生成的视频可能跟 "我要抓杯子" 这个 instruction 没多大关系), 而且对 prediction error 超敏感 — 生成的杯子里中心点偏了 5 个像素, 反推出的 action 就全错

InternVLA-A1 想 reconcile 这两派: 把 MLLM 的 semantic reasoning 和 world model 的 dynamics prediction 揉进一个 unified architecture, 让它们 share context, 互相补台.

---

## 2. 架构 — 三个脑袋一个身体

整个 model 是个 Mixture-of-Transformers, 三个 decoder-only transformer 接力跑:

```
图像 + 文字 instruction
        ↓
 [Understanding Expert]  ← 相当于 MLLM, "我看懂了场景和任务"
        ↓ (通过 KV cache 传 context)
 [Generation Expert]     ← "我预测 15 帧后场景会变成这样"
        ↓ (再通过 KV cache 传 context)
 [Action Expert]         ← "综合上面两个, 我输出机器人动作"
        ↓
  连续 action chunk (用 flow matching 生成)
```

关键点: **三个 expert 共享同一个 attention context, 但信息流单向**. 用一个 blockwise mask 强制 understanding → generation → action 这个顺序. Earlier expert 的 token 作为 later expert 的 KV cache, later expert 可以 attend back 看 earlier, 但 earlier 不能 forward 看 later.

为什么这么设计? 直觉上, 你想抓一个运动中的杯子:
1. 你先 understand: "传送带上有杯子, 杯子朝右移动, 我要抓它"
2. 你 imagine: "0.5 秒后杯子会在我右手边 10cm 处"
3. 你 act: "那我手应该往右手边 10cm 伸, 而且要提前到达等它"

这个 cognitive pipeline 就是 understanding → generation → action. MLLM 派跳过了 step 2, world model 派 step 2 做了但和 step 1 耦合松散. InternVLA-A1 把三步强制连成一条 chain.

---

## 3. Generation Expert — 这篇 paper 的工程亮点

这部分最有意思, 因为它揭示了 paper 的真实 design philosophy.

### 3.1 为什么不用现成的大 video model

你可能会想: 直接拿 Sora 或者 Stable Video Diffusion [arxiv 2311.15127](https://arxiv.org/abs/2311.15127) 来做 foresight 不就完了? paper 里明确说不行, 给了具体数字:

- SANA-Sprint [arxiv 2503.09641](https://arxiv.org/abs/2503.09641): 生成一张图要 0.16s on RTX 4090, 最多 6Hz
- DreamZero (GEAR 2026): 38× 加速后还在 GB200 上才 7Hz

机器人控制要 10+Hz, 你 foresight 模块本身就 6-7Hz, 整个 control loop 就崩了. 所以必须自己 design 一个 lightweight generation module.

### 3.2 Decoupled visual encoding — 理解和生成用不同的眼睛

借鉴 Janus-Pro [arxiv 2501.17811](https://arxiv.org/abs/2501.17811) 的 insight:
- Understanding 任务需要 high-level 语义抽象 → ViT encoder 合适
- Generation 任务需要 pixel-level 空间结构保真 → VAE encoder 合适

所以 generation expert 用 COSMOS CI8×8 VAE tokenizer [arxiv 2501.03575](https://arxiv.org/abs/2501.03575), 而不复用 understanding expert 的 ViT. 这是一个 key engineering choice.

### 3.3 96 tokens 的疯狂压缩

输入是 6 张图 (3 视角 × 2 时间戳 $t$ 和 $t-15$), 每张 256×256.

COSMOS VAE 编码: 每张图变成 32×32 = 1024 个 latent tokens. 6 张图 = 6144 tokens.

直接喂 6144 tokens 进 transformer? 序列太长, 训练慢, 推理也慢. 他们用 8×8 conv 把 32×32 压成 4×4 = 16 tokens/图. 6 张图最终 96 tokens.

**压缩比 64×**. 这意味着每个 token 要"代言"原本 64 个 latent 位置的信息. 这么激进的压缩, 生成的 future frame 肯定糊. paper 里 Figure 10 的 visualization 确实糊. 但 paper 的态度是: "糊就糊吧, 我要的是 motion trend, 不是 photorealistic rendering".

这是整个 generation expert 的 design philosophy: **foresight 不必精确, 但必须 inform action**. action expert 需要的是"物体往哪边动"的信号, 不需要"物体表面纹理是什么".

### 3.4 Parallel Decoding — 不做 autoregressive

输出也是 96 tokens, 经过 temporal average pooling 把 2 个时间戳的信息 pool 成 1 个, 得到 48 tokens (每视角 16 个). 然后 projector + deconv 上采样回 32×32 grid, 再走 COSMOS VAE decoder 还原成 $t+15$ 的 future frame.

关键: **所有 future tokens 在 single forward pass 里同时出**, 不做 autoregressive next-token prediction. 这就避免了 LLM 那种"一个 token 一个 token 生成, KV cache 越来越长"的 latency 累积问题.

代价: 每个位置的 token 不能依赖同帧其他位置的生成结果. 但因为前面 transformer block 已经让 96 tokens 之间 fully bidirectional attend 过了, 信息已经在 latent 里 mix 好, 这一步只是"读出"已经算好的 latent.

整个 generation pipeline 在 RTX 4090 上, 配合 understanding expert 和 action expert, 整个 model 跑 13Hz. 这是为什么 dynamic manipulation (传送带这种) 能 work 的前提.

---

## 4. Loss Function — 一个 regression, 一个 flow matching

### 4.1 Generation Loss: 简单 MSE

$$\mathcal{L}_{\text{gen}} = \mathbb{E}_{\xi_1}\left[ \| f_{\text{gen}}(z_{t-m}, z_t; h_{\text{und}}) - \text{sg}[z_{t+m}] \|^2 \right]$$

人话: 给定历史帧 latent $z_{t-m}$ 和当前帧 latent $z_t$, 加上 understanding expert 给的 context $h_{\text{und}}$, 预测未来帧 latent $\hat{z}_{t+m}$, 用 MSE 跟 ground truth $z_{t+m}$ 比. $\text{sg}[\cdot]$ 是 stop-gradient, 让 loss 只 update generation expert, 不污染 ground truth latent 的 representation.

为什么用 MSE 而不用 diffusion? 直觉上, 物理世界在给定当前状态后, 未来相对 deterministic (只是 observation 有 noise). 而 action 是 multimodal 的 — 同一个 task 可以左手抓也可以右手抓. 所以 generation 用 regression 够了, action 要用 generative model.

### 4.2 Action Loss: Flow Matching

$$\mathcal{L}_{\text{act}} = \mathbb{E}_{\xi_2}\left[ \| \nu_\theta(q_t, a_{t:t+k}^\tau; h_{\text{und}}, h_{\text{gen}}) - (a_{t:t+k} - \epsilon) \|^2 \right]$$

人话翻译:
- $a_{t:t+k}$: ground truth action chunk (一段连续动作)
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $\tau \sim \text{Beta}(1.5, 1.0)$: 插值系数, Beta(1.5, 1.0) 偏向 τ→1 (接近 ground truth)
- $a^\tau = (1-\tau)\epsilon + \tau a$: 噪声和 ground truth 之间的插值点
- $\nu_\theta$: 神经网络学的 "速度场", 告诉你从当前插值点往 ground truth 走应该往哪个方向走
- target $(a - \epsilon)$: 从噪声指向 ground truth 的向量

训练就是让网络学这个 velocity field. 推理时 (公式 3):
$$a^{\tau + \Delta\tau} = a^\tau + \Delta\tau \cdot \nu_\theta(...)$$

从纯噪声 $\epsilon$ 出发, 走 K 步 Euler 法, 每步沿着 velocity field 走 $\Delta\tau = 1/K$, 最终到 $a^1 \approx \hat{a}$. 这就是 flow matching — 把 noise distribution "流" 到 action distribution.

为什么 action 要用 flow matching? 因为 action 是 multimodal distribution. 如果你用 MSE regression, 模型会学一个 "平均动作", 但平均动作往往是 invalid (左手抓和右手抓的平均可能是双手都伸一半, 啥也抓不到). Flow matching 让模型学一个 distribution, sample 出来的 action 始终是 valid 的 mode.

这个 idea 继承自 π0 [arxiv 2410.24164](https://arxiv.org/abs/2410.24164). π0 也是用 flow matching 输出 action. InternVLA-A1 在这基础上加了 generation expert 的 context $h_{\text{gen}}$ 作为额外 conditioning.

### 4.3 Total Loss

$$\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{act}}, \quad \lambda = 0.01$$

λ=0.01 这个数字很重要: action loss 是主导, generation loss 是 auxiliary. 这意味着 generation expert 不是为了"生成漂亮的 future frame", 而是给 action expert 提供一个 dynamics-aware 的 conditioning 信号. 即使 generation 预测的 latent 偏了, 只要它稳定地 capture "场景将怎么变" 的趋势, action expert 就能用上.

---

## 5. 数据 — 692M frames 的大杂烩

| Source | Type | Frames | Sampling Weight |
|---|---|---|---|
| InternData-A1 [arxiv 2511.16651](https://arxiv.org/abs/2511.16651) | Sim | 396M | 0.64 |
| RoboTwin [arxiv 2506.18088](https://arxiv.org/abs/2506.18088) | Sim | 17M | 0.08 |
| AgiBot-World [arxiv 2503.06669](https://arxiv.org/abs/2503.06669) | Real | 206M | 0.18 |
| RoboMind [arxiv 2412.13877](https://arxiv.org/abs/2412.13877) | Real | 5M | 0.02 |
| EgoDEx [arxiv 2505.11709](https://arxiv.org/abs/2505.11709) | Human video | 68M | 0.08 |

直觉: sim data 占绝对量优势 (~68%), 因为便宜多样; real data 防 sim-to-real gap; human video 给 generation expert 提供 manipulation 的 visual prior.

### 5.1 EgoDEx 的妙用

EgoDEx 是 829 小时第一人称人手操作视频. 关键: **pre-training 时不用 human action label**.

为什么? 人手有 27 个 DOF, 机器人 gripper 可能就 1-2 个 DOF, action 空间不兼容. 但 visual dynamics 是通用的 — 人怎么抓杯子、怎么翻面包, 这些 motion pattern 可以迁移给 generation expert 学 "future frame 长啥样".

所以 human video 只参与 $\mathcal{L}_{\text{gen}}$ 训练, 不参与 $\mathcal{L}_{\text{act}}$. 这是个聪明的设计: 你免费拿到了 68M frames 的 manipulation visual prior, 不用费劲去 cross-embodiment 地映射 action.

### 5.2 LPT — Load-balanced Parallel Training

692M frames heterogeneously 分布, naive 在每个 GPU worker 上 instantiate 全部数据会 OOM. LPT 用个简单 greedy algorithm: 把 dataset 按大小降序排, 每次把下一个 dataset 分给当前 load 最小的 worker.

$$\pi(i) = \arg\min_{k \in \{1,...,K\}} \sum_{j: \pi(j)=k} s_j$$

人话: 让每个 worker 拿到的数据量尽量平均. 这样避免了 "大 dataset 的 worker 跑得慢, 小 dataset 的 worker 跑完一遍又来一遍, 无形中大 dataset 采样权重被放大" 的 implicit re-weighting 问题.

---

## 6. 实验 — 重点看 dynamic task

### 6.1 Static Manipulation

10 个 real-world static task 平均成功率:

| Method | Params | Avg |
|---|---|---|
| GR00T N1.5 | - | 33.0% |
| π0 | 3.3B | 60.6% |
| π0.5 | - | 70.7% |
| InternVLA-A1 (2B) | 1.8B | 64.7% |
| **InternVLA-A1 (3B)** | 3.2B | **75.1%** |

2B 变体已经超过 3.3B 的 π0 (+4.1%), 说明 architecture + data 比 raw scale 更有效率. 3B vs π0.5 +4.4%, 在 long-horizon bimanual task (Make Sandwich) 上 +20%.

### 6.2 Dynamic Manipulation — 真正的 show case

两个 task: Express Sorting (传送带分拣) 和 In-motion Ingredient Picking (运动中抓食材).

| Method | Express Sorting | Ingredient Picking | Avg |
|---|---|---|---|
| GR00T N1.5 | 40.0% | 20.0% | 30.0% |
| π0 | 36.7% | 20.0% | 28.4% |
| π0.5 | 53.3% | 66.7% | 60.0% |
| **InternVLA-A1 (3B)** | **80.0%** | **93.3%** | **86.7%** |

vs π0.5 在 dynamic 上 **+26.7%**, 而 static 只有 +4.4%. 这个 gap 之大就是 paper 的核心 evidence: dynamic scene 需要 foresight, reactive policy 在传送带这种 momentum 主导的场景里就是不够.

Ingredient Picking 93.3% vs π0.5 的 66.7%, +26.6%. 这个 task 是双臂协作, 两台机器人协同组装三明治, 目标在传送带上移动. 没有 foresight, 你根本来不及协调时序.

### 6.3 RoboTwin 2.0 Simulation Benchmark

50 个 bimanual task, Easy/Hard 两档. vs π0.5 +2.6%, vs π0 +9.4%/10.1%.

sim benchmark 上提升小 (2.6%) 其实合理: sim 环境 dynamic 是 deterministic 的, 不需要那么强 foresight; real-world dynamic task 因为有摩擦、振动、传感器 noise 等 unmodeled dynamics, foresight 价值才显现.

---

## 7. Ablation — 验证每个设计

### 7.1 Pre-training 的影响

去掉 pre-training, success rate 从 77.0% 跌到 25.4% — drop 51.6%. 极端情况某些 task 直接 0%. pre-training on heterogeneous data 是 inductive prior, 没它模型根本学不到 generalizable manipulation policy.

### 7.2 Pre-training Data Mixture

| Pre-training Data | RoboTwin Easy | RoboTwin Hard | Place Flower (Real) | Sort Parts (Real) |
|---|---|---|---|---|
| Sim only | 88.3 | 88.5 | 53.3 | 33.3 |
| Sim + Human | 89.4 | 89.3 | 53.3 | 40.0 |
| **Sim + Real + Human** | 89.4 | 89.6 | **60.0** | **53.3** |

读这个表:
- Sim only 在 sim benchmark 强 (88.3/88.5), 但 real 弱 (53.3/33.3) — 经典 sim-to-real gap
- 加 human video 改善 sim 略升 real (Sort Parts 33.3→40.0)
- 加 real data 不太改善 sim, 但 real-world 大幅提升 (Place Flower 53.3→60.0, Sort Parts 40.0→53.3)

三者互补缺一不可. sim 给 scale + diversity, real 给 physical fidelity, human 给 visual prior.

### 7.3 Generation Expert 的影响

移除 generation expert, avg success 从 77.0% 降到 57.6% — drop 19.4%. 11/12 任务都降, dynamic task 降最狠.

注意: 移除 generation expert 后, action expert 只拿 $h_{\text{und}}$ 作 conditioning, 这相当于退化成 π0 的 architecture. 所以这个 ablation 本质是在比较 "MLLM + action" vs "MLLM + generation + action", 后者胜 19.4%. 这就是 unified understanding-generation-action 的价值量化.

---

## 8. 我的 intuition

把 academic 包装扒掉, InternVLA-A1 的核心 insight 是:

**Foresight 不必精确, 但必须 inform action.**

这个 idea 颠覆了 world model 派的常规思路. VPP, Genie Envisioner 那些方法在追求 "生成越逼真越好" 的未来视频, 但 InternVLA-A1 说: 我要的 future frame 糊一点没关系, 只要 motion trend 对, action expert 就能用. 所以他们敢用 64× 压缩的 96 tokens representation, 敢用 single forward parallel decoding 而不做 autoregressive, 敢用 MSE regression 而不用 diffusion. 所有这些 "不够 elegant" 的 engineering choice, 都是为了 inference speed — 13Hz 是 dynamic manipulation 能 work 的前提.

从你 Karpathy 经常说的 "neural network 是 circuit" 角度看, 这个架构像一个 information flow diagram:
- Understanding expert = perception encoder
- Generation expert = predictive model (imagination)
- Action expert = motor controller

三者通过 KV cache 共享 representation, attention mask 强制因果方向 (perception → imagination → motor). 这和 neuroscience 里的 "predictive coding" framework 呼应 — 大脑也在不断预测下一刻的感官输入, 用 prediction error 驱动 action. 见 Friston 的 free energy principle 相关工作 [arxiv 2205.04706](https://arxiv.org/abs/2205.04706) 之类.

dynamic task +26.7% 的提升是最有力的 evidence: 在 momentum/inertia 主导的场景里, reactive policy 在做 "看到 X 就做 Y" 的 lookup, InternVLA-A1 在做 "看到 X 预测 t+15 会变成 X', 针对 X' 做 Y" — 这是从 reactive 到 predictive 的 paradigm shift, 类似从 RLHF (reactive) 到 AlphaGo MCTS (predictive lookahead) 的跃迁. 见 Silver et al. 的 AlphaGo Zero [arxiv 1712.01815](https://arxiv.org/abs/1712.01815).

---

## 9. 局限与可能的下一步

paper 自己承认两个 limitation:
1. Understanding expert 没用大规模 VQA data 联合训练, semantic reasoning 弱. 这就是为什么 InternVLA-A1 没强调 chain-of-thought, 而 CoT-VLA [arxiv 2411.16610](https://arxiv.org/abs/2411.16610) 那种在 reasoning 部分强.
2. Generation expert 牺牲了图像保真度, future frame 模糊, 限制需要精确 spatial 信息的高精度任务.

我能想到的延伸:
- Hierarchical timescale: generation 预测 t+15, t+30, t+60, 给 action 多 horizon foresight
- Generation 用 latent diffusion 替代 MSE, 让 generation 也具备 multimodal prediction (一个动作可能有多种 future)
- 用 V-JEPA 2 [arxiv 2506.09985](https://arxiv.org/abs/2506.09985) 这类 self-supervised video model 替代 COSMOS VAE tokenizer, latent space 更 predictive
- Flow matching 的 K 步 inference 用 consistency distillation 压到 1-2 步, 进一步提 latency, 见 consistency model [arxiv 2303.01469](https://arxiv.org/abs/2303.01469)
- 把 understanding expert 的 pre-training 融入更多 VQA data, 让 model 既能做 manipulation 又能做 long-context reasoning

---

## References

- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- InternVL3: https://arxiv.org/abs/2504.10466
- Janus-Pro: https://arxiv.org/abs/2501.17811
- Cosmos VAE: https://arxiv.org/abs/2501.03575
- SANA-Sprint: https://arxiv.org/abs/2503.09641
- VPP: https://arxiv.org/abs/2412.14803
- Genie Envisioner: https://arxiv.org/abs/2508.05635
- InternData-A1: https://arxiv.org/abs/2511.16651
- EgoDex: https://arxiv.org/abs/2505.11709
- AgiBot-World: https://arxiv.org/abs/2503.06669
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- UniPi: https://papers.nips.cc/paper_files/paper/2023
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- F1: https://arxiv.org/abs/2509.06951
- AlphaGo Zero: https://arxiv.org/abs/1712.01815
- Consistency Models: https://arxiv.org/abs/2303.01469
- CoT-VLA: https://arxiv.org/abs/2411.16610
- Stable Video Diffusion: https://arxiv.org/abs/2311.15127
- Free Energy Principle survey: https://arxiv.org/abs/2205.04706

---

# InternVLA-A1 深度技术解析

Andrej, 这篇paper 的核心 thesis 是: 现有 VLA 模型在 reactive perception-to-action mapping 上很强, 但对 physical world dynamics 的 reasoning 是缺位的, 而 world model 路线 (VPP, Genie Envisioner) 缺乏 semantic grounding 且对 video prediction error 脆弱. InternVLA-A1 想把 semantic understanding + visual foresight + action execution 三件事 unify 到一个 Mixture-of-Transformers 里, 用统一的 masked self-attention 让三个 expert 协同. 下面按你的直觉习惯, 把每块拆开讲透.

---

## 1. 问题动机: 为什么 MLLM-based VLA 不够

现有 generalist policy 比如 π0 ([arxiv 2410.24164](https://arxiv.org/abs/2410.24164)), π0.5 ([arxiv 2504.16054](https://arxiv.org/abs/2504.16054)), GR00T N1/N1.5 ([arxiv 2503.14734](https://arxiv.org/abs/2503.14734)) 都把 visual data 映射到 text-based feature space. 这个设计继承了 MLLM 的语义先验, 但 text tokens 本质是离散符号, 不擅长建模 momentum, inertia, contact dynamics 这类连续物理量. 所以在静态任务 (叠衣服, 收桌子) 上 VLA 表现好, 一旦场景 dynamic (传送带上抓物体) 就崩.

另一个流派是 world model via video prediction, 比如 UniPi ([NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023)), UniSim ([ICLR 2024](https://arxiv.org/abs/2310.06114)), GR-1/GR-2 ([arxiv 2410.06158](https://arxiv.org/abs/2410.06158)), VPP ([ICML 2025](https://arxiv.org/abs/2412.14803)), Genie Envisioner ([arxiv 2508.05635](https://arxiv.org/abs/2508.05635)). 这些方法用 future video 作为 imagined observation 指导决策, 但有两个 pathology: (i) semantic grounding 弱 — 生成的视频和 task instruction 之间耦合松; (ii) 对 video prediction error 敏感 — 如果生成帧的物体位置偏了几像素, inverse dynamics 模型推出的 action 就错.

InternVLA-A1 的设计 hypothesis 是: 与其用一个独立的大 video model 然后再接 inverse dynamics, 不如让 generation 模块本身就 lightweight, 在 latent space 内预测, 并且和 understanding/action expert 共享同一个 attention context. 这样即使 generation 不精确, action expert 也能从 understanding expert 的语义上下文里得到 robust 的补救信号. 这点很关键 — 这就是为什么 ablation 里移除 generation expert 在 dynamic task 上的 drop 比 static task 更剧烈.

---

## 2. Mixture-of-Transformers 架构详解

### 2.1 三个 Expert 的角色

InternVLA-A1 借鉴了 unified multimodal model 里 MoT 的成功实践 (Deng et al. 2025, [arxiv 2505.14683](https://arxiv.org/abs/2505.14683)), 把三个 decoder-only transformer 拼起来:

| Expert | 作用 | Backbone (2B 变体) | Backbone (3B 变体) |
|---|---|---|---|
| Understanding Expert | encode scene context from image+text | InternVL3-1B ([arxiv 2504.10466](https://arxiv.org/abs/2504.10466)) | Qwen3-VL-2B |
| Generation Expert | predict future latent state | Qwen2.5 transformer blocks (0.36B) | Qwen3 blocks (0.44B) |
| Action Expert | flow matching 出连续 action chunk | Qwen2.5 blocks (0.36B) | Qwen3 blocks (0.44B) |

注意一个细节: generation 和 action expert 都是从 understanding expert 的 LLM backbone 复用 transformer block (但参数独立), 这样既保留了 LLM 的先验又能各自 specialize. 这和 π0 的 MoT 设计有相似之处, 但 π0 只有 MLLM + action expert 两路, InternVLA-A1 多了 generation 这一路.

### 2.2 Blockwise Masked Self-Attention — 信息流的关键

这是整个架构最巧妙的地方. 把三个 expert 的 token stream concat 起来, 用一个 cumulative segment mask 强制信息单向流:

```
[Understanding tokens] → [Generation tokens] → [Action tokens]
```

具体规则:
- Understanding expert 内部: tokens fully bidirectional attend (像 MLLM 一样)
- Generation expert 内部: tokens fully bidirectional attend, 同时 attend 到 understanding expert 的所有 tokens (用 understanding 的 K/V cache)
- Action expert 内部: 拆成 state token + action tokens. state token 只 attend 自己 + earlier blocks; action tokens attend state token + earlier blocks + 其他 action tokens
- 反向不可见: understanding 不能 attend generation/action, generation 不能 attend action

这个设计在数学上等价于一个 structured causal mask, 但比标准 LLM 的左到右 causal mask 更松 — 它允许每个 expert 内部 bidirectional, 跨 expert 才有方向性. 这对理解任务很重要, 因为 understanding 阶段你需要 bidirectional 把 image 和 instruction 综合起来.

直觉上: understanding expert 给出一个 "我现在看到的场景是什么 + 我被要求做什么" 的 context embedding $h_{\text{und}} = f_{\text{und}}(l, o_t)$. Generation expert 在这个 context 上做 conditional prediction, 产出 $h_{\text{gen}}$. Action expert 同时拿 $h_{\text{und}}$ 和 $h_{\text{gen}}$ 作为 conditioning, 加上 proprioception $q_t$, 输出 action chunk $\hat{a}_{t:t+k}$.

---

## 3. Generation Expert 的工程设计 (这是 paper 的工程亮点)

### 3.1 为什么不用预训练 video foundation model

paper 里有一段很重要的工程论证 (Section 3.2): 直接用大 video foundation model 做 foresight 在实时性上不可行. 举例:
- SANA-Sprint ([arxiv 2503.09641](https://arxiv.org/abs/2503.09641)): 0.16 s/image on RTX 4090, 最多 6Hz
- DreamZero (GEAR 2026): 即便 38× 优化加速, 在 GB200 上才 7Hz

而 VLA 任务 (尤其 dynamic manipulation) 需要 10+ Hz 的控制频率. 所以 InternVLA-A1 选择自己 design 一个 lightweight generation module, 而不是 reuse 现成的大 video model.

### 3.2 Input Tokenization — decoupled visual encoding

这个设计借鉴 Janus-Pro ([arxiv 2501.17811](https://arxiv.org/abs/2501.17811)) 的洞察: understanding 和 generation 对 visual encoder 的需求是 divergent 的.
- Understanding 需要 high-level semantic abstraction → ViT-based encoder 合适
- Generation 需要 pixel-level fidelity + fine-grained spatial structure → VAE-based encoder 合适

所以 generation expert 不复用 understanding expert 的 ViT, 而是用 COSMOS CI8×8 continuous VAE image tokenizer ([arxiv 2501.03575](https://arxiv.org/abs/2501.03575)).

输入是 6 张图: 3 个视角 (head + 2 wrist) × 2 个时间戳 ($t$ 和 $t-15$, m=15). 每张图 256×256, 编码后 32×32 latent grid = 1024 tokens.

### 3.3 Token Compression — 96 tokens 的 magic

直接 1024×6 = 6144 tokens 喂 transformer 序列太长. 他们用了一个 8×8 kernel 的 conv layer 做 downsample, 把 32×32 grid 压到 4×4 = 16 tokens/image. 6 张图最终 96 tokens.

这是一个激进的压缩 — 把 1024 个 latent 压到 16 个, 压缩比 64×. 直觉上这要求 generation expert 学到的 representation 必须 very abstract, 不可能保留高频视觉细节. 这其实和他们的设计哲学一致: foresight 模块不需要 photorealistic output, 只需要 capture motion trends 和 dynamics, 然后给 action expert 一个 instructive signal. Ablation 里他们也讨论了这点 (Section 5.6, Figure 10): 生成的 future frame 在视觉上模糊, 但 motion trend 准确.

### 3.4 Parallel Decoding — 一次性出全部 future tokens

这点是最有意思的工程选择. Generation expert 的输出还是 96 tokens, 经过 temporal average pooling 在时间轴聚合 (2 timestamps → 1), 得到 48 tokens (每视角 16). 然后 projector + deconv 上采样回 32×32 grid, 再走 COSMOS VAE decoder 还原出 $t+15$ 的 future frame.

关键点: 不是 autoregressive next-token prediction (像 LLaMA 那样逐 token 生成), 而是 single forward pass parallel decoding, 所有 future tokens 同时出. 这就避免了 KV cache 增长和 autoregressive 的 latency 累积. 代价是: 每个位置的 token 不能依赖同帧其他位置的生成结果, 但因为前面 transformer block 已经让 96 tokens 之间 fully bidirectional attend 过了, 信息已经在 latent 里 mix 好.

整体 generation expert pipeline:
```
6 images (256×256) 
  → COSMOS VAE encoder → 6 × (32×32) = 6144 latent tokens
  → 8×8 conv downsample → 6 × (4×4) = 96 tokens
  → projector → transformer blocks (attend to h_und via KV cache)
  → output 96 tokens
  → temporal avg pool → 48 tokens (3 views × 16)
  → projector + deconv → 3 × (32×32) = 3072 latent grid
  → COSMOS VAE decoder → 3 future frames at t+15
```

这就是为什么整个模型在 RTX 4090 上能跑到 13Hz.

### 3.5 为什么 m=15

历史帧 $t-15$ + 当前帧 $t$ + 预测帧 $t+15$, 对称的时间间隔. 15 帧在 ~30Hz 视频里大约是 0.5 秒, 在控制频率 ~13Hz 下大约是 1.15 秒. 这个 interval 既要足够长让 dynamic 有可观测变化, 又不能太长让 prediction 变成猜. 这个超参应该 empirically tune 过.

---

## 4. 优化目标的数学

### 4.1 Visual Foresight Generation Loss (公式 1)

$$\mathcal{L}_{\text{gen}} = \mathbb{E}_{\xi_1}\left[ \| f_{\text{gen}}(z_{t-m}, z_t; h_{\text{und}}) - \text{sg}[z_{t+m}] \|^2 \right]$$

变量解释:
- $\xi_1 = (o_{t-m}, o_t, o_{t+m}, l)$: training tuple, 无 action label, 这样 human video 也能用
- $z_t = \phi_{\text{cosmos}}(o_t)$: COSMOS VAE encoder 把图像编到 latent
- $f_{\text{gen}}$: generation expert 函数
- $h_{\text{und}}$: understanding expert 输出的 context embedding (作为 conditioning)
- $\text{sg}[\cdot]$: stop-gradient, 让 loss 只 backprop 到 generation expert, 不污染 ground-truth latent representation

这是 plain MSE 在 latent space, 不是 diffusion flow matching. 为什么 generation 用 MSE 而 action 用 flow matching? 直觉上: future frame 在给定当前帧和历史帧后, distribution 是相对 unimodal 的 (物理是 deterministic 的, 只是观察有 noise); 而 action distribution 是 multimodal 的 (同一个 task 可以有多种解法, 比如左手抓或右手抓). 所以 generation 用 regression, action 用 generative model.

### 4.2 Flow Matching Action Loss (公式 2)

$$\mathcal{L}_{\text{act}} = \mathbb{E}_{\xi_2}\left[ \| \nu_\theta(q_t, a_{t:t+k}^\tau; h_{\text{und}}, h_{\text{gen}}) - (a_{t:t+k} - \epsilon) \|^2 \right]$$

其中:
- $\xi_2 = (a_{t:t+k}, o_{t-m}, o_t, q_t, l)$: 带 action label 的 tuple
- $q_t$: proprioception (机器人 joint 状态)
- $a_{t:t+k}^\tau = (1-\tau)\epsilon + \tau a_{t:t+k}$: interpolated action chunk
- $\tau \sim \text{Beta}(1.5, 1.0)$: 时间采样, Beta(1.5, 1.0) 偏向 τ→1, 即更多采样接近 ground truth 的状态
- $\epsilon \sim \mathcal{N}(0, I)$: 高斯噪声
- $\nu_\theta$: 神经网络学的 velocity field, 预测从当前噪声态到 target 的 "速度方向"
- target 是 $(a - \epsilon)$, 即从噪声指向 ground truth 的向量

Flow matching 本质上是训练一个 ODE solver, 学习把 noise distribution transport 到 action distribution. 和 diffusion 的区别: diffusion 是 SDE (带随机性), flow matching 是确定性 ODE, 通常 sample 更快且 mode collapse 风险低. π0 也用的 flow matching, 这部分是继承.

Beta(1.5, 1.0) 的选择有意思: 它不是 uniform Beta(1,1), 而是 slightly 偏向大 τ. 直觉上, 训练时多见接近 target 的状态, 让网络在最终精修阶段 (τ→1) 学得更精细, 这对 action 这种 high-precision 任务有帮助.

### 4.3 Inference 的 Euler Update (公式 3)

$$a_{t:t+k}^{\tau + \Delta\tau} = a_{t:t+k}^\tau + \Delta\tau \cdot \nu_\theta(q_t, a_{t:t+k}^\tau; h_{\text{und}}, h_{\text{gen}})$$

- $\tau$ 从 0 走到 1, 一共 K 步, $\Delta\tau = 1/K$
- 从 $\epsilon \sim \mathcal{N}(0, I)$ 出发, 迭代 K 次 Euler 步, 最终得到 $a_{t:t+k}^1 \approx \hat{a}_{t:t+k}$

K 越大 sample 越精确但越慢. π0 paper 里 K=10, 这里 paper 没明确说 K, 但结合 13Hz 的 inference 速度, K 应该是 4-10 之间.

### 4.4 Total Loss (公式 4)

$$\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{act}}$$

$\lambda = 0.01$, 这个权重表明 action loss 是主导, generation loss 起的是 auxiliary 的作用. 这点很关键 — generation expert 不是为了让生成的 future frame 看起来好看, 而是给 action expert 提供一个 dynamics-aware 的 conditioning 信号. 即使 generation 预测的 latent 不准, 只要它能稳定地 capture "场景将怎么变" 的趋势, action expert 就能用上.

---

## 5. 数据配方: 692M Frames 的 Heterogeneous Mix

这是 paper 的另一大支柱. 数据 mixture 如 Table 3:

| Source | Type | Frames | Sampling Weight |
|---|---|---|---|
| InternData-A1 | Sim | 396M | 0.64 |
| RoboTwin | Sim | 17M | 0.08 |
| AgiBot-World (Beta) | Real | 206M | 0.18 |
| RoboMind | Real | 5M | 0.02 |
| EgoDEx | Human | 68M | 0.08 |

直觉: sim data 占绝对量优势 (413M frames / 601M frames total ≈ 68%), 因为 sim 数据便宜且多样; real data 提供物理 fidelity 防 sim-to-real gap; human video 给 generation expert 提供 manipulation 的 visual prior.

### 5.1 InternData-A1 — sim data 的 scale

InternData-A1 ([arxiv 2511.16651](https://arxiv.org/abs/2511.16651)) 是他们前作, 630k trajectories, 7433 hours, 覆盖 4 embodiments × 18 skills × 70 tasks × 227 scenes, 包含 rigid / articulated / deformable / fluid object 的 manipulation. 用 Nimbus ([arxiv 2601.21449](https://arxiv.org/abs/2601.21449)) 框架, 在 8 张 RTX 4090 上日产 209.7 小时仿真数据.

### 5.2 EgoDEx — human video 的 role

EgoDEx ([arxiv 2505.11709](https://arxiv.org/abs/2505.11709)) 是 829 小时第一人称灵巧操作视频, 200+ task. 关键: pre-training 时 **不用 human action label**. 所以 human video 只参与 $\mathcal{L}_{\text{gen}}$ 的训练, 不参与 $\mathcal{L}_{\text{act}}$.

这设计很聪明: human video 的 action label 和机器人不兼容 (人手和 gripper 自由度不一样), 但 visual dynamics 是通用的 — 人怎么抓杯子、怎么翻面包, 这些 motion pattern 可以迁移给 generation expert 当 visual prior.

### 5.3 Load-balanced Parallel Training (LPT)

这是工程上必须解决的问题. 692M frames heterogeneously distributed, naive 在每个 worker 上 instantiate 全部数据会 OOM + I/O 瓶颈. LPT 用 greedy load balancing 把 dataset 分配到 K 个 worker:

$$\pi(i) = \arg\min_{k \in \{1,...,K\}} \sum_{j: \pi(j)=k} s_j$$

按 $s_i$ 降序排, 每次把下一个 dataset 给当前 load 最小的 worker. 这是个简单的 greedy algorithm, 但有两个好处: (i) per-worker 内存压力小; (ii) 避免 implicit re-weighting — 如果不同 worker 跑完自己 dataset 的速度不一样, 大 dataset 的采样权重会被无意中放大. LPT 让所有 worker throughput 均衡, 近似 uniform sampling.

当 dataset 数少于 worker 数时, 允许复制 dataset, 但用不同的 random seed 和 load-aware placement, 避免某个 worker 被小 dataset 主导.

---

## 6. 实验结果 — 重点看 dynamic task

### 6.1 Static Manipulation (Table 4)

10 个 real-world task, 平均成功率:

| Method | Params | Avg Success |
|---|---|---|
| GR00T N1.5 | - | 33.0% |
| π0 | 3.3B | 60.6% |
| π0.5 | - | 70.7% |
| **InternVLA-A1 (2B)** | 1.8B | 64.7% |
| **InternVLA-A1 (3B)** | 3.2B | **75.1%** |

关键观察: 2B 变体已经超过 3.3B 的 π0 (+4.1%), 说明 architecture + data mixture 的设计比单纯 scale 参数更有效率. 3B 变体 vs π0.5 +4.4%, 在 Make Sandwich 任务上 93.3% vs π0.5 的 73.3% (+20%), 说明 long-horizon bimanual task 受益于 generation expert 的 foresight.

### 6.2 Dynamic Manipulation (Figure 6) — 这才是 paper 的 show case

两个 task: Express Sorting (传送带分拣) 和 In-motion Ingredient Picking (运动中抓食材).

| Method | Express Sorting | Ingredient Picking | Avg |
|---|---|---|---|
| GR00T N1.5 | 40.0% | 20.0% | 30.0% |
| π0 | 36.7% | 20.0% | 28.4% |
| π0.5 | 53.3% | 66.7% | 60.0% |
| InternVLA-A1 (2B) | 70.0% | - | - |
| **InternVLA-A1 (3B)** | **80.0%** | **93.3%** | **86.7%** |

vs π0.5 在 dynamic 上 +26.7%, 而 static 只有 +4.4%. 这个 gap 之大, 直接验证了 paper 的核心 thesis: dynamic scene 需要 foresight reasoning, 纯 reactive policy 在传送带这种 momentum 主导的场景里就是不够.

Ingredient Picking 93.3% vs π0.5 66.7% 这个 +26.6% 的提升尤其惊人 — 这个 task 是双臂协作, 两台机器人协同组装三明治 (两片面包 + 牛排 + 生菜), 目标在传送带上移动, 需要精确的时序协调. Generation expert 能预判物体在 t+15 时刻的位置, action expert 才能提前规划.

### 6.3 RoboTwin 2.0 Simulation Benchmark (Figure 7)

50 个 bimanual task, Easy / Hard (domain randomized) 两档. InternVLA-A1 (3B) vs π0.5 +2.6% (两档都是). vs π0 +9.4% / +10.1%.

sim benchmark 上提升不大 (2.6%), 这其实合理: sim 环境 dynamic 是 deterministic 的, 不需要那么强的 foresight reasoning; 而 real-world dynamic task 因为有摩擦、振动、传感器 noise 等 unmodeled dynamics, foresight 的价值才显现.

---

## 7. Ablation Studies — 验证每个设计

### 7.1 Pre-training 的影响 (Figure 8)

移除 pre-training, 从 scratch 训, success rate 从 77.0% 跌到 25.4% — drop **51.6%**. 极端情况下某些 task 完全失败 (0%). 这说明 pre-training on heterogeneous data 是 inductive prior, 没它模型根本学不到 generalizable manipulation policy.

### 7.2 Pre-training Data Mixture 的影响 (Table 5)

| Pre-training Data | RoboTwin Easy | RoboTwin Hard | Place Flower (Real) | Sort Parts (Real) |
|---|---|---|---|---|
| Sim only | 88.3 | 88.5 | 53.3 | 33.3 |
| Sim + Human | 89.4 | 89.3 | 53.3 | 40.0 |
| **Sim + Real + Human** | 89.4 | 89.6 | **60.0** | **53.3** |

关键观察:
- Sim only 在 sim benchmark 上很强 (88.3/88.5), 但 real-world 差 (53.3/33.3) — 经典 sim-to-real gap
- 加 human video 改善 sim (89.4/89.3) 且 real 略升 (Sort Parts 33.3→40.0)
- 加 real data 不太改善 sim (89.4 不变), 但 real-world 大幅提升 (Place Flower 53.3→60.0, Sort Parts 40.0→53.3)

这个 ablation 干净地说明: sim data 提供 scale + diversity, real data 提供 physical fidelity, human video 提供 visual prior, 三者互补缺一不可.

### 7.3 Generation Expert 的影响 (Figure 9)

移除 generation expert, avg success 从 77.0% 降到 57.6% — drop **19.4%**. 11/12 任务都有降, dynamic task 降得最厉害. 这直接证明 generation expert 不是 nice-to-have, 是 architecture 的 essential component.

但注意一个细节: 移除 generation expert 后, action expert 只拿 $h_{\text{und}}$ 作 conditioning, 这相当于退化为 π0 的 architecture. 所以这个 ablation 实际上是在比较 "MLLM + action expert" vs "MLLM + generation expert + action expert", 后者胜 19.4%. 这就是 unified understanding-generation-action 的价值量化.

---

## 8. 局限与未来方向

paper 自己承认两个 limitation:
1. Understanding expert 没有用大规模 multimodal VQA data 联合训练, 导致 general semantic reasoning 和 complex instruction following 弱. 这是为什么 InternVLA-A1 没强调 reasoning chain-of-thought 能力, 而其他 VLA 比如 CoT-VLA ([arxiv CVPR 2025](https://arxiv.org/abs/2411.16610)) 在这部分强.
2. Generation expert 为推理效率牺牲了图像保真度, future frame 视觉模糊. 这限制了 generation 用在需要精确 spatial 信息的高精度任务.

可以想到的延伸方向:
- 把 VQA data 融入 pre-training, 让 understanding expert 既懂 manipulation 又能做 long-context reasoning
- 用 hierarchical timescale — generation expert 预测 t+15, t+30, t+60, 给 action expert 多 horizon 的 foresight
- Generation 用 latent diffusion 替代 MSE regression, 让 generation 自己也具备 multimodal prediction 能力 (一个动作可能有多种 future)
- 用 V-JEPA 2 ([arxiv 2506.09985](https://arxiv.org/abs/2506.09985)) 这类 self-supervised video model 替代 COSMOS VAE tokenizer, 让 latent space 更 predictive
- 把 flow matching 的 K 步 inference 用 distillation 压到 1-2 步, 进一步提 latency

---

## 9. 我对这篇 paper 的整体直觉

InternVLA-A1 的核心 contribution 不在某个 single trick, 而在工程整合: 把 understanding (语义), generation (dynamics prediction), action (control) 三件之前分别由不同 model 做的事, 用 MoT + blockwise attention mask + shared training 拼成一个 unified system. 设计哲学是 "foresight 不必精确, 但必须 inform action" — 这就是为什么 generation 用 lightweight latent regression (而非大 video diffusion), λ=0.01 让 action loss 主导, 96 tokens 极度压缩 representation.

从 Karpathy 你经常强调的 "neural network 是 circuit" 的视角看, 这个架构像一个 information flow diagram: understanding 是 perception encoder, generation 是 predictive model (imagination), action 是 motor controller. 三者通过 KV cache 共享 representation, attention mask 强制因果方向 (perception → imagination → motor). 这和 neuroscience 里 "predictive coding" 的 framework 呼应 — 大脑也在不断预测下一刻的感官输入, 用 prediction error 驱动 action.

dynamic task 上 +26.7% 的提升是最有力的 evidence: 在 momentum / inertia 主导的场景里, reactive policy 本质上是在做 "看到 X 就做 Y" 的 lookup, 而 InternVLA-A1 在做 "看到 X 预测 t+15 会变成 X', 然后针对 X' 做 Y" — 这是从 reactive 到 predictive 的 paradigm shift, 类似从 OpenAI's RLHF (reactive) 到 AlphaGo 的 MCTS (predictive lookahead) 的跃迁.

---

## References

- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- InternVL3: https://arxiv.org/abs/2504.10466
- Janus-Pro: https://arxiv.org/abs/2501.17811
- Cosmos VAE: https://arxiv.org/abs/2501.03575
- SANA-Sprint: https://arxiv.org/abs/2503.09641
- VPP: https://arxiv.org/abs/2412.14803
- Genie Envisioner: https://arxiv.org/abs/2508.05635
- InternData-A1: https://arxiv.org/abs/2511.16651
- EgoDex: https://arxiv.org/abs/2505.11709
- AgiBot-World: https://arxiv.org/abs/2503.06669
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- UniPi: https://papers.nips.cc/paper_files/paper/2023
- UniSim: https://arxiv.org/abs/2310.06114
- GR-2: https://arxiv.org/abs/2410.06158
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- F1: https://arxiv.org/abs/2509.06951
- Nimbus: https://arxiv.org/abs/2601.21449
- MoT for unified multimodal: https://arxiv.org/abs/2505.14683
- OpenVLA: https://arxiv.org/abs/2406.09246
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
