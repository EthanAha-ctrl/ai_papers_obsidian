---
source_pdf: World Model for Robot Learning A Comprehensive Survey.pdf
paper_sha256: 1b035d79d5c135fd437e1bbe7625a7e284a405e0451a240e816ce5dcaf35f586
processed_at: '2026-08-13T05:01:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲一遍这篇 survey

Andrej 你让我用大白话讲，那我就剥掉学术包装，把这篇 paper 的"故事"和"直觉"讲清楚。

---

## 这篇 paper 到底在干嘛

robotics 社区现在有个尴尬局面：**"world model"这个词被用烂了**。Sora 生成个猫的视频叫 world model，Dreamer 的 latent dynamics 也叫 world model，VLA 内部隐含的 future prediction 也叫 world model。大家各说各的，没法比较。

这篇 survey 干了一件事：**给 world model 下一个 robot-centric 的功能定义，然后按"怎么和 policy 耦合"重新梳理整个领域**。

核心判断标准就一句话：**你的 model 生成的 future，能不能被 robot 拿来做决策**。Sora 生成的猫很漂亮，但猫不会因为 robot 的 action 改变轨迹，所以对 robotics 没用。IRASim 给一个 action sequence 预测未来 video，future 真的会跟着 action 变，这才是 world model。

---

## 为什么 robot 需要 world model

想象你闭着眼睛走路。纯 reactive VLA 就是闭着眼睛走 —— 只看当前 frame，直接输出 action。短距离没问题，长 horizon 就会 compound error，撞墙。

world model 给 robot 一双"想象的眼"。在真正 act 之前，先在脑子里 rollout 一下："如果我执行 action A，未来会变成什么样？" 然后选那个未来最好的 action。

这个 intuition 背后的数学很干净。你想要的是一个联合分布：

$$p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)$$

这个 joint 就是"给定当前观测和指令，未来观测和未来动作的联合分布"。你看，**所有 model 都是它的不同切片**：

- **Policy**：把 future observation 积分掉，只留 $p(a \mid o, l)$ → "给我 obs，我直接告诉你动作"
- **Passive world model**：把 action 积分掉，只留 $p(o_{t+1:t+k} \mid o_t, l)$ → "给我 obs 和指令，我告诉你未来长啥样，不管动作"
- **Controllable world model**：条件在 action 上，$p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k})$ → "给我 obs 和候选 action，我告诉你这个 action 会导致的未来"
- **Inverse dynamics**：$p(a \mid o_{t:t+k})$ → "给我一段观测序列，我反推中间的 action"

所以这四个东西**根本就是同一个物体的四个面**。policy 和 world model 不是两类 model，是同一类 joint predictive-control model 的不同 query 方式。

这个 view 一旦建立，后面整篇 survey 的 taxonomy 就顺理成章了。

---

## 五种把 world model 塞进 policy 的方式

从"最松的耦合"到"最紧的耦合"，一条光谱。

### 方式一：先预测再行动（IDM-style，decoupled）

最直觉的设计。**两阶段，两个独立模块**。

第一阶段：world model（通常是 video diffusion）预测未来视频。
$$\hat{\mathbf{o}}_{t+1:t+H} = \mathcal{W}(o_t, l)$$
- $\hat{\mathbf{o}}$：预测的未来 observation 序列
- $\mathcal{W}$：world model
- $H$：prediction horizon

第二阶段：一个 inverse dynamics 模块从"当前帧 → 预测帧"反推出 action。
$$\pi(a \mid o_t, l) = P(a \mid E_{img}(o_t), E_{text}(l), \Phi(\hat{\mathbf{o}}_{t+1:t+H}))$$
- $E_{img}, E_{text}$：image / text encoder
- $\Phi(\cdot)$：对 predicted future 做 feature extraction
- $P$：inverse dynamics decoder

**代表方法的演进路径**（这条线特别能看出 field 在往哪走）：

- **UniPi** (2023, https://arxiv.org/abs/2302.04994) — 最早这么干的。text → video generation → IDM 抠 action。想法很美，但 video 质量不够，action 误差大。
- **VidMan** (2024, https://arxiv.org/abs/2407.07743) — 加了 masked inverse dynamics，让 IDM 只关注 action-relevant 的区域。
- **Gen2Act** (2025, https://research.nvidia.com/labs/dir/gen2act/) — 换了个思路：不预测 robot video，预测 **human video**，再把 human motion 迁移到 robot。跨 embodiment 的 prior。
- **VPP** (2025, ICML, https://arxiv.org/abs/2412.14803) — 发现完全不需要生成 pixel，直接用 video diffusion 的 **latent feature** 注入 action head 就行。省掉 video 渲染开销。
- **MimicVideo** (2025, https://arxiv.org/abs/2512.15692) — 用 partially denoised 的 latent visual plan，进一步省 inference 成本。
- **TC-IDM** (2026, https://arxiv.org/abs/2601.18323) — 把 future 翻译成 tool-centric 的几何轨迹，更 execution-friendly。

**trend 一目了然**：future representation 从 raw pixel → latent feature → structured geometry，越来越抽象，越来越贴近 control 需要的量。

### 方式二：一个 backbone 同时生成 video 和 action（Single-backbone）

上面是两个模块。现在**把它们焊进一个 transformer**。

$$\hat{y} = f_\theta(\tilde{\mathbf{x}}_\tau, o_t, l, \tau), \quad \mathbf{x} = [z^v; z^a]$$
- $\tilde{\mathbf{x}}_\tau$：加噪后的输入，denoising step $\tau$
- $z^v$：visual latent token
- $z^a$：action latent token
- $[z^v; z^a]$：在 sequence 维度上拼接
- $f_\theta$：shared backbone

训练目标统一成一个：
$$\mathcal{L}_{unified} = \mathbb{E}[\ell(\hat{y}, y)]$$
$y$ 是 target —— 可能是 diffusion 的 noise、flow matching 的 velocity field、或 discrete token 的 mask。

**为什么要焊在一起**？paper 给的 argument 是：video diffusion backbone 的 pretraining objective 是"temporal prediction"，和 control 的需要天然对齐。VLM backbone 的 pretraining 是"image-text alignment"，更偏 semantic 对应，对 dynamics 没那么敏感。

代表方法：
- **UVA** (2025, https://arxiv.org/abs/2503.00200) — joint video-action latent，inference 时可以 bypass video branch 只取 action。
- **UWA** (2025, https://arxiv.org/abs/2504.02792) — single transformer，video 和 action 用不同的 diffusion timestep。
- **VideoVLA** (2025) — 把 Video DiT 扩展成 Video-Action DiT。
- **Cosmos Policy** (2026, https://arxiv.org/abs/2601.16163) — 很巧妙：action、future state、value 都当成 diffusion sequence 里的 "latent frame"。policy mode 只取 action output；planning mode 取 value 做 ranking。
- **DreamZero** (2026, https://arxiv.org/abs/2602.15922) — autoregressive flow-matching，chunk-wise joint denoising 限制 compounding error。

**但 paper 也诚实地说**：这是 suggestive evidence，同 scale 下 video backbone 是否稳赢 VLM backbone 仍是 open question。

### 方式三：保留专家分工，但深度交互（MoE / MoT）

方式二把所有东西塞一个 backbone，会不会丢掉 modality-specific 的 specialization？这派人认为：video prediction 和 action generation 的 temporal frequency、representational scale、optimization 需求都不一样，完全 share parameter 不一定最优。

所以保留**独立的 video expert 和 action expert**，但让它们在每一层深度交互：

$$
(\mathbf{h}_{\ell+1}^v, \mathbf{h}_{\ell+1}^a) = \mathcal{F}_\ell^{mix}(\mathbf{h}_\ell^v, \mathbf{h}_\ell^a; o_t, l)
$$
- $\ell$：layer index
- $\mathbf{h}_\ell^v, \mathbf{h}_\ell^a$：第 $\ell$ 层 video expert / action expert 的 hidden state
- $\mathcal{F}_\ell^{mix}$：交互算子，可以是 joint attention / cross-attention / shared attention

代表：
- **Motus** (2025, https://arxiv.org/abs/2512.13030) — 最直接的 MoT 实现，understanding / video / action 三个 expert。
- **LingBot-VA** (2026, https://arxiv.org/abs/2602.07322) — video 和 action token 交错排列进 autoregressive sequence，causal world modeling。
- **BagelVLA** (2026, https://arxiv.org/abs/2602.09849) — long-horizon manipulation，Residual Flow Guidance 让 visual foresight 用 single-step denoising 就能工作。
- **Fast-WAM** (2026, https://arxiv.org/abs/2603.16666) — **最值得注意的发现**：video co-training 在 training time 有用，但 inference time 不需要 future imagination。这是个对"imagination helps inference"假设的反例。
- **LDA-1B** (2026, https://arxiv.org/abs/2602.12215) — visual forecasting 移到 DINO latent space。

### 方式四：把预测内化进 VLA（Unified VLA）

这派**不单独搞个 video model**，而是在 VLA 内部加一个 future prediction 的 auxiliary objective。future 可以是 pixel、可以是 latent、可以是 structured knowledge。

代表：
- **GR-1** (2024, https://openreview.net/forum?id=At8pkPH2wfR) — GPT-style transformer，joint predict action + future image。
- **WorldVLA** (2025, https://arxiv.org/abs/2511.17502) — future image prediction 主要当 training signal，inference 时非必需。
- **DreamVLA** (2025, https://arxiv.org/abs/2503.14004) — 不预测 pixel，预测 "structured world knowledge"（dynamic + spatial + semantic cues）。
- **UniVLA** (2025, https://arxiv.org/abs/2506.19850) — post-training 在 native multimodal tokenization 上吸收 causal dynamics。
- **F1** (2025, https://arxiv.org/abs/2509.06951) — MoT 架构下 future visual state 当 planning target。

### 方式五：完全在 latent space 做，不生成 pixel（Latent-space WM）

最激进的一派。**赌 pixel-level prediction 是 overkill**，control 需要的"future 信息"是一个低维 manifold，在 embedding space 预测就够了。

和 LeCun 的 **JEPA** 一脉相承 (https://arxiv.org/abs/2301.08243)。

代表：
- **FLARE** (2025, CoRL, https://arxiv.org/abs/2411.04329) — "Future Latent Representation Alignment"，把 action denoising network 的 hidden feature 和 future observation 的 latent embedding 对齐。policy 隐式 anticipate 未来，不显式生成。
- **VLA-JEPA** (2026, https://arxiv.org/abs/2602.10098) — leakage-free state prediction，future frame 只用来产生 latent target 做监督，防止模型 shortcut。
- **JEPA-VLA** (2026, https://arxiv.org/abs/2602.11832) — 直接用 V-JEPA 2 的 predictive embedding 当 VLA backbone。
- **V-JEPA 2** (2025, https://arxiv.org/abs/2506.09985) — Meta 的大模型，能支持 zero-shot robot planning。
- **WoG** (2026, https://arxiv.org/abs/2602.22010) — 把 world modeling 塞进 action 的 **condition space**：不预测 future image，预测"future 里对 control 最有用的那部分条件"。

---

## World model 当 simulator 用

除了当 predictive module 塞进 policy，world model 还能**直接当环境用**。给当前 obs 和 candidate action，rollout 一条 imagined trajectory，在里面做 RL 或者评估 action 好坏。

### 当 RL 环境用

$(\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t) \sim p_\phi(\cdot \mid o_{\le t}, a_{\le t}, l)$

- $p_\phi$：world model，参数 $\phi$
- $\hat{o}_{t+1}$：imagined next observation
- $\hat{r}_t$：imagined reward
- $\hat{d}_t$：imagined done 信号

policy 在这个 imagined environment 里最大化 expected return：
$$J(\theta) = \mathbb{E}_{\hat\tau \sim (\pi_\theta, p_\phi)}\left[\sum_t \gamma^t \hat{r}_t\right]$$
- $\gamma$：discount factor
- $\hat\tau$：从 policy $\pi_\theta$ 和 world model $p_\phi$ 联合 rollout 出的 imagined trajectory

**两个层次**：

**Level 1：world model 固定，policy 在里面学**
- **World-Env** (2025, https://arxiv.org/abs/2509.24948) — VLA post-training 的 virtual environment。
- **VLA-RFT** (2025, https://arxiv.org/abs/2510.00406) — verified rewards 在 controllable simulator 里。
- **World-Gymnast** (2025) — 发现 RL inside video world model 比 supervised finetune 和 software simulator 都强。
- **WMPO** (2026, ICLR, https://arxiv.org/abs/2601.04320) — pixel-space imagination + on-policy GRPO。
- **ProphRL** (2025, https://arxiv.org/abs/2511.20633) — FA-GRPO + FlowScale 适配 flow-based action head。

**Level 2：world model 和 policy 一起迭代**
- **World-VLA-Loop** (2026, https://arxiv.org/abs/2602.06508)
- **VLAW** (2026, https://arxiv.org/abs/2602.12063) — iterative repair-and-improve。
- **WoVR** (2026, https://arxiv.org/abs/2602.13977) — Keyframe-Initialized Rollouts + 显式 co-evolution：

$$\phi^{k+1} \gets \text{UpdateWM}(\phi^k, D_{real} \cup D_{policy}(\pi_{\theta^k}))$$
$$\theta^{k+1} \gets \text{UpdatePolicy}(\theta^k, \hat{D}(\phi^{k+1}))$$
- $\phi^k$：第 k 轮 world model 参数
- $\theta^k$：第 k 轮 policy 参数
- $D_{real}$：真实数据
- $D_{policy}(\pi_{\theta^k})$：当前 policy rollout 的数据
- $\hat{D}(\phi^{k+1})$：用新 world model 生成的 imagined 数据

**直觉**：Level 1 是"在 world model 里做 RL"，Level 2 是"和 world model 一起做 RL" —— 承认 learned simulator 不 perfect，必须和 policy 共同进化。

### 当 evaluator 用

rollout 候选 action，看哪个 future 最好。

- **GPC** (2026) — frozen policy + world model 在 deployment 时 online rank 候选 action。
- **WorldEval** (2025, https://arxiv.org/abs/2505.19017) — 用 world model rank 不同 policy / checkpoint。
- **WorldArena** (2026, https://arxiv.org/abs/2602.08971) — benchmark 级别，明确把 policy evaluation 当 world model 的核心 downstream。
- **Gemini Robotics + Veo** (2025, https://arxiv.org/abs/2512.10675) — Google 用 Veo video simulator offline 评估 Gemini Robotics policy。

---

## Robotic video world model 的四个阶段

这是 paper 里第二条主线：把 robotic video generation 按 capability 分成四个阶段。

### 阶段一：当 imagination engine 造数据

把 video generation 当 supervision 来源，synthetic 未来当训练数据。

- **UniPi** (2023) — text-conditioned video generation 当 policy。
- **Dreamitate** (2024, CoRL) — fine-tune video diffusion on task-specific demos，生成新 scene 的 execution 当 visual plan。
- **RoboDreamer** (2024, ICML) — compositional world modeling，instruction 分解成 reusable primitives。
- **DreMa** (2025, ICLR, https://arxiv.org/abs/2410.15791) — Gaussian Splatting + physics simulator 当 manipulable digital twin。
- **DreamGen** (2025, CoRL, https://arxiv.org/abs/2505.01828) — strong video generator 适配 target embodiment，recover latent action。

### 阶段二：action-controllable

从"plausible future"到"action-faithful future"。future 必须真的响应 action。

- **IRASim** (2025, ICCV, https://arxiv.org/abs/2412.16138) — frame-level action conditioning，每帧对应一个 action。
- **RoboMaster** (2026, ICLR) — 分 phase 建模 robot arm + object 的耦合运动，处理 contact dynamics。
- **Ctrl-World** (2026, ICLR, https://arxiv.org/abs/2510.10125) — multi-view + frame-level action control + memory-based long-horizon。
- **EnerVerse-AC** (2025, https://arxiv.org/abs/2505.09723) — action-conditional multi-view generator。
- **EVA** (2026, https://arxiv.org/abs/2603.17808) — inverse dynamics reward 对齐 video world model 和 executable action，填"视觉 plausible 但物理不可执行"的 gap。

### 阶段三：引入 structure prior

光 action conditioning 不够，还要 mask / geometry / viewpoint / identity 这些中间结构。

- **Mask2IV** (2025, https://arxiv.org/abs/2510.03135) — 两阶段：先预测 actor + object 的 interaction trajectory，再 condition 生成 video。
- **TesserAct** (2025, https://arxiv.org/abs/2504.20995) — 4D embodied world model，RGB + depth + normal。
- **RoboVIP** (2026, https://arxiv.org/abs/2601.05241) — visual identity prompting 引导 multi-view 一致性。

### 阶段四：foundation-scale reusable world backbone

从 task-specific predictor 变成通用 platform。

- **Vid2World** (2026, ICLR) — 把 pretrained video diffusion 系统性改造成 interactive world model。
- **DreamDojo** (2026, https://arxiv.org/abs/2602.06949) — 大规模 human egocentric pretraining + continuous latent action 桥接 human / robot。
- **WoW** (2025, https://arxiv.org/abs/2509.22642) — 强调 physical intuition 必须 from interaction 数据，不能 from passive video。
- **Cosmos Predict 2.5** (2025, NVIDIA, https://arxiv.org/abs/2511.00062) — 大规模 video foundation world model。
- **GigaWorld-0** (2025, https://arxiv.org/abs/2511.19861) — controllable video branch + physically grounded 3D branch。

---

## Benchmark 的三个层次

paper 很强调：**评估 world model 不能只看 visual realism**。三层评估：

**Layer 1：open-loop predictive quality** — 给 (obs, action)，看生成的 future 是否 action-faithful。
- **RBench** (2026, https://arxiv.org/abs/2601.15282)
- **EWMBench** (2025, https://arxiv.org/abs/2505.09694) — 把评估分解成 scene consistency / motion correctness / semantic alignment。
- **EVA-Bench** (2025) — long-horizon + OOD robustness。

**Layer 2：closed-loop task utility** — world model 当 environment，看 policy 在里面 work 不 work。
- **WorldArena** (2026, https://arxiv.org/abs/2602.08971)
- **WorldEval** (2025, https://arxiv.org/abs/2505.19017) — rank consistency：world model 里 policy A 比 B 好，真实世界也是吗？
- **World-in-World** (2025, https://arxiv.org/abs/2510.18135) — closed-loop planning，最难，expose compounding error。

**Layer 3：physical consistency / executability diagnostics** — 生成的 future 能不能恢复出 executable action？
- **WorldSimBench** (2025, ICML, https://arxiv.org/abs/2410.06405) — IDM-based recovery test。
- **WoW-World-Eval** (2026, https://arxiv.org/abs/2601.04137) — IDM-based Turing Test：从生成的 video 抠出 action，执行回真实环境看对不对。
- **WM-ABench** (2025, ACL) — 把 world modeling 能力分解成 atomic capability：spatial / temporal / motion / mechanistic / counterfactual。

---

## 最关键的实验结果

Table 5 是 LIBERO 4-suite 的成绩。我挑几个有代表性的：

| Paradigm | Method | Spatial | Object | Goal | Long | Avg |
|----------|--------|---------|--------|------|------|-----|
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Unified VLA | RynnVLA-002 | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Latent-space | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Latent-space | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

**三个 takeaway**：

1. **多种 paradigm 都能打到 ~98%**。不存在"某一派碾压其他派"。说明 world model 的 utility 不绑定单一 architecture。

2. **Long suite 是 differentiator**。Spatial / Object 大家都高，Long（长 horizon）差距拉开。predictive structure 在长 horizon 才显出价值。

3. **VLA-JEPA 不生成 pixel 也能 97.2%**。这直接验证了 paper 的论点：**photorealistic video generation is not necessary for effective embodied control**。latent 预测够用。

---

## 六大 open challenge

1. **Causal conditioning gap** — world model 的 future 对 historical context 响应强，对 pending action 响应弱。future 可能"intention-consistent 但 action-inconsistent"。WorldVLA 用 implicit unified training 缓解。

2. **Efficiency** — 训练 + 推理都贵。video diffusion 的 iterative denoising 延迟高。MimicVideo / LingBot-VA 用 partial denoising；LeWorldModel 走 latent-only；Fast-WAM 干脆 inference 时抛弃 world model。

3. **Multi-modal perception** — 只有 vision + proprioception 不够，friction / stiffness / contact stability 需要 tactile / force。问题：tactile 高频低维 vs visual 低频高维，joint optimization 容易 visual dominance。代表：OmniVTA (https://arxiv.org/abs/2603.19201)、Visuo-tactile world models (https://arxiv.org/abs/2602.06001)。

4. **Classical control integration** — MPC 需要大量 rollout，real-time 难。Lyapunov stability / robust control 的 formal guarantee 和 neural expressivity 怎么 reconcile 是 frontier。

5. **Symbolic structure** — pixel rollout 长 horizon error accumulate。symbolic world model over predicates / relations 更 stable + compositional。Hybrid (perceptual + symbolic) 是 promising direction。代表：VisualPredicateator (2025, ICLR)、ExoPredicator (2026, ICLR)。

6. **Evaluation metric** — visual realism ≠ control utility。需要 function-aware metric：task success、policy-ranking fidelity、executability diagnostics。

---

## 一句话 build your intuition

**world model for robot learning 的核心赌注是：让 robot 在 act 之前先"想象"未来，这个想象能力能从大规模 video data 迁移过来，让 action generation 不再是 reactive 的 obs→action mapping，而是经过一个 predictive bottleneck 的 informed decision**。

但这个赌注有几个还没回答的关键问题：

- **pixel 还是 latent**？VLA-JEPA 和 Cosmos Policy 都接近 SOTA，可能 task-dependent。
- **inference 时真的需要 imagination 吗**？Fast-WAM 说不需要，Cosmos Policy 说需要。可能 training-time 的 predictive structure 才是本质，inference-time imagination 是 bonus。
- **world model 当 simulator 可靠吗**？WoVR 说必须 co-evolution，frozen simulator 不够。
- **长 horizon 怎么办**？pixel 会 accumulate error，latent 会丢 information，symbolic 需要 grounding。hybrid 是出路但还没人做出来。

这些就是 field 接下来 12-18 个月要打的仗。

---

参考资源：
- Survey 主页：https://ntumars.github.io/wm-robot-survey/
- GitHub：https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
- 通讯作者：Jianfei Yang (jianfei.yang@ntu.edu.sg)

---

# World Model for Robot Learning: A Comprehensive Survey — 深度技术讲解

这篇 paper 是 NTU、Berkeley、Stanford、Oxford、ETH、Princeton、Harvard 等机构合作的一篇 ~197KB 的 comprehensive survey，试图在 VLA (Vision-Language-Action) policy 浪潮的当下，把"world model"这个被滥用的概念重新约束到 robot learning 的语境中，并给出一套从 architecture → functional role → application domain 的分层 taxonomy。下面我从你的 intuition 出发，把整篇 paper 拆开讲。

---

## 1. 这篇 paper 想解决什么"概念混乱"

近两年 robotics 社区把 "world model" 这个词用得太宽：有人指 latent dynamics model (Dreamer 系)，有人指 video predictor (Sora 系)，有人指 VLA 内部那个 implicit 的 predictive structure。这篇 survey 的核心贡献是给出一个 **robot-learning-centered 的 functional definition**：

> A world model is a predictive model of agent-environment dynamics whose outputs support policy-related computation (control, planning, simulation, evaluation, data generation).

这个定义的关键约束是 **"actionable"** —— 一个 model 仅生成视觉上 plausible 的 future video，但 future 不响应 robot action，不算 world model。这条线把 Sora、Runway 这类纯 generative video 排除在外，但把 IRASim、Ctrl-World、Genie 这类 action-conditioned predictor 纳入。

Survey 主页：https://ntumars.github.io/wm-robot-survey/  
GitHub repo：https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy

---

## 2. 数学语言：把 policy / world model / IDM 统一到一个 joint distribution

这是 paper 最精彩的一节 (Sec 3.1)，从 probabilistic 角度把看似不同的 paradigm 统一起来。设当前 observation $o_t$，action $a_t$，language instruction $l$。理想的 predictive-control 联合分布是：

$$
p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)
$$

变量含义：
- $o_t$：t 时刻的 observation，可含 visual + proprioceptive
- $o_{t+1:t+k}$：未来 k 步的 observation 序列（上标表示时间索引范围）
- $a_{t+1:t+k}$：未来 k 步的 action 序列
- $l$：language instruction 或 task specification

从这个 joint 出发，四种 model 都是它的 marginal / conditional：

**Policy model** (Eq 5)：对 future observation 做 marginalization
$$
p(a_{t+1:t+k} \mid o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)\, d o
$$
这就是 RT-2、OpenVLA、π₀ 在做的事 —— 直接 regress / generate action，把 future observation 当 nuisance variable 积分掉。

**Passive world model** (Eq 6)：对 action 做 marginalization
$$
p(o_{t+1:t+k} \mid o_t, l) = \int p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)\, d a
$$
这就是 text-to-video 模型 (Sora、CogVideoX) 的目标，只关心"未来长什么样"，不关心 robot 怎么 act。

**Controllable world model** (Eq 7)：条件在 action 上
$$
p(o_{t+1:t+k} \mid o_t, a_{t+1:t+k})
$$
这是 IRASim、Ctrl-World、Cosmos Predict 这一类，给定候选 action，预测未来 observation。

**Inverse Dynamics Model** (Eq 8)：从 observation 轨迹反推 action
$$
p(a_{t+1:t+k} \mid o_{t:t+k})
$$
这是 UniPi、Gen2Act、MimicVideo 的核心 —— 先生成 future video，再用 IDM 把相邻 frame 之间的 action 抠出来。

**这个 unified view 的 intuition**：这四个 model 不是四个独立的 architecture，而是同一个 joint distribution 的四种 query 方式。所以 world model 和 policy 的耦合本质上是"引入 predictive structure 到 action generation 里"，让 action 不直接从当前 obs 蹦到未来，而经过一个"先想象未来、再决定 action"的中间 bottleneck。

---

## 3. Sec 3：World Model for Policy —— 五种 architectural paradigms

这是 paper 最重的一节，给出 Table 1 的 taxonomy。我按 coupling tightness 从松到紧讲。

### 3.1 IDM-style（decoupled predict-then-act）

公式 (Eq 9-12)：

$$
\hat{\mathbf{o}}_{t+1:t+H} = \mathcal{W}(o_t, l)
$$
- $\hat{\mathbf{o}}$：predicted future observation 序列
- $\mathcal{W}$：world model（通常是 video diffusion）
- $H$：prediction horizon

然后 policy 在 (current obs, predicted future) 上 condition：
$$
\pi(a_{t+1:t+H'} \mid o_t, l) = P\bigl(a_t \mid E_{img}(o_t), E_{text}(l), \Phi(\hat{\mathbf{o}}_{t+1:t+H})\bigr)
$$
- $E_{img}, E_{text}$：image / text encoder
- $\Phi(\cdot)$：feature extractor over predicted future
- $H'$：action chunk size（注意 $H$ 和 $H'$ 可以不同 —— world model 预测更长 horizon，policy 只输出短 chunk）

代表方法演进（按时间）：

1. **UniPi** (Du et al., 2023, NeurIPS) — https://arxiv.org/abs/2302.04994 — 最早把 text-conditioned video generation 当 policy 用，用 IDM 比较相邻 frame 得到 action。
2. **VidMan** (Wen et al., 2024, NeurIPS) — https://arxiv.org/abs/2407.07743 — 引入 masked inverse dynamics，强调 action-relevant region。
3. **Vidar** (Feng et al., 2025) — https://arxiv.org/abs/2507.12898 — embodied video diffusion，更 fine-grained。
4. **Gen2Act** (Bharadhwaj et al., 2025, CoRL) — https://research.nvidia.com/labs/dir/gen2act/ — 用 human video 而不是 robot video 当 future，跨 embodiment 迁移。
5. **VPP** (Hu et al., 2025, ICML) — https://arxiv.org/abs/2412.14803 — 不再 rollout 像素，直接用 video diffusion 的 latent feature 注入 action head，更紧凑。
6. **MimicVideo** (Pai et al., 2025) — https://arxiv.org/abs/2512.15692 — 用 partially denoised latent visual plan，避开完整 video 渲染开销。
7. **TC-IDM** (Mi et al., 2026) — https://arxiv.org/abs/2601.18323 — 把 future 翻译成 tool-centric geometric trajectory。
8. **Say-Dream-ACT** (Gu et al., 2026) — https://arxiv.org/abs/2602.10717 — 用 generated video 当 in-context visual guidance。

**关键 trend**：future representation 从 raw pixel → latent feature → structured geometric intermediate → in-context prompt，逐渐抽象化、execution-friendly化。

### 3.2 Single-backbone（joint video-action generation in one backbone）

Eq 13-14：
$$
\hat{y} = f_\theta(\tilde{\mathbf{x}}_\tau, o_t, l, \tau), \quad \mathbf{x} = [z^v; z^a]
$$
- $\tilde{\mathbf{x}}_\tau$：corrupted input at denoising step $\tau$
- $z^v$：visual latent token
- $z^a$：action latent token
- $[z^v; z^a]$：concatenation
- $f_\theta$：shared transformer backbone
- $\tau$：diffusion timestep

Loss：
$$
\mathcal{L}_{unified} = \mathbb{E}[\ell(\hat{y}, y)]
$$
$y$ 的具体形式取决于 instantiation —— diffusion 噪声 / flow matching velocity field / discrete masked token。

代表方法：
- **UVA** (Li et al., 2025c) — https://arxiv.org/abs/2503.00200 — joint video-action latent，推理时可 bypass video generation。
- **UWA** (Zhu et al., 2025a) — https://arxiv.org/abs/2504.02792 — single transformer 下 modality-specific timestep。
- **VideoVLA** (Shen et al., 2025) — 把 Video Diffusion Transformer 改造成 Video-Action Diffusion Transformer。
- **Cosmos Policy** (Kim et al., 2026) — https://arxiv.org/abs/2601.16163 — 把 action、future state、value 全部当 latent "frame" 塞进原始 diffusion sequence，policy mode 只取 action，planning mode 可用 value 做 rank。
- **DreamZero** (Ye et al., 2026b) — https://arxiv.org/abs/2602.15922 — autoregressive flow-matching + chunk-wise joint denoising，限制 compounding error。
- **UD-VLA** (Chen et al., 2026b) — 同步 denoising 的 discrete multimodal 设定。

**Intuition**：video diffusion backbone 的 inductive bias 是"temporally predictive"，比 VLM 的 image-text alignment objective 更接近 control 的需要。但 paper 也明确指出：这是 suggestive evidence，不是 definitive conclusion —— 同 scale 下 video backbone 是否稳赢 VLM backbone 还是 open question。

### 3.3 MoE / MoT-style（保留 expert specialization）

Eq 15：
$$
(\mathbf{h}_{\ell+1}^v, \mathbf{h}_{\ell+1}^a) = \mathcal{F}_\ell^{mix}(\mathbf{h}_\ell^v, \mathbf{h}_\ell^a; o_t, l)
$$
- $\ell$：layer index
- $\mathbf{h}_\ell^v, \mathbf{h}_\ell^a$：第 $\ell$ 层 video expert / action expert 的 hidden state
- $\mathcal{F}_\ell^{mix}$：layer-wise interaction operator，可以是 joint attention / cross-attention / shared-attention fusion

代表方法：
- **Motus** (Bi et al., 2025) — https://arxiv.org/abs/2512.13030 — Mixture-of-Transformers with dedicated experts。
- **LingBot-VA** (Li et al., 2026b) — https://arxiv.org/abs/2602.07322 — interleaved video-action token + dual-stream MoT，causal world modeling。
- **BagelVLA** (Hu et al., 2026) — https://arxiv.org/abs/2602.09849 — long-horizon manipulation + Residual Flow Guidance，single-step denoising。
- **DiT4DiT** (Ma et al., 2026) — https://arxiv.org/abs/2603.10448 — video branch 的 intermediate denoising feature 引导 action branch。
- **Fast-WAM** (Yuan et al., 2026) — https://arxiv.org/abs/2603.16666 — 关键发现：video co-training 在 training time 有用，inference time 不需要 future imagination。这是对该 paradigm 的一个反思性证据。
- **LDA-1B** (Lyu et al., 2026) — https://arxiv.org/abs/2602.12215 — 把 visual forecasting 移到 DINO latent space。

### 3.4 Unified VLA（internalize prediction in MLLM）

代表方法：
- **GR-1** (Wu et al., 2024) — https://openreview.net/forum?id=At8pkPH2wfR — GPT-style transformer joint predict action + future image。
- **WorldVLA** (Cen et al., 2025) — https://arxiv.org/abs/2511.17502 — future image prediction 主要当 joint training signal，inference 时非必需。
- **DreamVLA** (Zhang et al., 2025e) — https://arxiv.org/abs/2503.14004 — predict structured world knowledge (dynamic + spatial + semantic cues)。
- **UniVLA** (Wang et al., 2025) — https://arxiv.org/abs/2506.19850 — post-training 在 native multimodal tokenization 上吸收 causal dynamics。
- **F1** (Lv et al., 2025) — https://arxiv.org/abs/2509.06951 — MoT 架构下 future visual state 当 planning target。
- **InternVLA-A1** (Cai et al., 2026) — https://arxiv.org/abs/2601.02456 — lightweight latent visual foresight。

### 3.5 Latent-space world modeling（不生成 pixel，纯 latent prediction）

这一系和 **JEPA** (Assran et al., 2023, https://arxiv.org/abs/2301.08243) 一脉相承。代表方法：
- **FLARE** (Zheng et al., 2025, CoRL) — https://arxiv.org/abs/2411.04329 — Future Latent Representation Alignment，把 action denoising network 的 hidden feature 和 future observation 的 latent embedding 对齐。
- **VLA-JEPA** (Sun et al., 2026) — https://arxiv.org/abs/2602.10098 — leakage-free state prediction，future frame 只用来产生 latent target。
- **JEPA-VLA** (Miao et al., 2026) — https://arxiv.org/abs/2602.11832 — 用 V-JEPA 2 的 predictive embedding 直接当 VLA backbone。
- **V-JEPA 2** (Assran et al., 2025) — https://arxiv.org/abs/2506.09985 — Meta 的视频 JEPA 大模型，支持 zero-shot robot planning。
- **WoG** (Su et al., 2026) — https://arxiv.org/abs/2602.22010 — world modeling in action 的 condition space。

**Intuition**：这一路的核心赌注是 —— pixel-level prediction 是 overkill，control 需要的"future information"是一个非常低维的 manifold。JEPA 的 embedding-space predictive objective 更 sample efficient，也避开 generative decoding 的开销。但代价是失去了 video prior 的大量 semantic / spatial structure。

---

## 4. Sec 4：World Model as Simulator —— 两种功能角色

### 4.1 World Model for RL（learned simulator）

Eq 16：
$$
(\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t) \sim p_\phi(\cdot \mid o_{\le t}, a_{\le t}, l)
$$
- $p_\phi$：参数为 $\phi$ 的 world model
- $\hat{o}_{t+1}$：imagined next observation
- $\hat{r}_t$：imagined reward
- $\hat{d}_t$：imagined done signal
- $o_{\le t}, a_{\le t}$：历史 observation / action

Eq 17 是 RL 的 expected return：
$$
J(\theta) = \mathbb{E}_{\hat\tau \sim (\pi_\theta, p_\phi)}\left[\sum_t \gamma^t \hat{r}_t\right]
$$
- $\gamma$：discount factor
- $\hat\tau$：从 policy $\pi_\theta$ 和 world model $p_\phi$ 联合 rollout 出来的 imagined trajectory

Eq 18 是 GRPO-style objective：
$$
\mathcal{L}_{RL}(\theta) = -\mathbb{E}_t\left[\min\bigl(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\right]
$$
- $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$：importance ratio
- $\hat{A}_t$：advantage estimate
- $\epsilon$：clip range

代表方法分两个层次：

**Level 1（world model frozen / lightly adapted）**：
- **UniSim** (Yang et al., 2024) — https://arxiv.org/abs/2310.06147
- **World-Env** (Xiao et al., 2025) — https://arxiv.org/abs/2509.24948
- **VLA-RFT** (Li et al., 2025b) — https://arxiv.org/abs/2510.00406 — verified rewards 在 controllable simulator 里。
- **DiWA** (Chandra et al., 2025, CoRL)
- **World-Gymnast** (Quevedo et al., 2025) — RL inside video world model 优于 supervised finetune 和 software simulator。
- **PlayWorld** (Yin et al., 2026) — https://arxiv.org/abs/2603.09030 — autonomous play 数据训练。
- **RehearseVLA** (Xiao et al., 2026, CVPR)
- **WMPO** (Zhu et al., 2026, ICLR) — https://arxiv.org/abs/2601.04320 — pixel-space imagination + on-policy GRPO。
- **ProphRL** (Zhang et al., 2025b) — https://arxiv.org/abs/2511.20633 — FA-GRPO + FlowScale for flow-based action head。
- **RISE** (Yang et al., 2026b) — compositional dynamics + progress value estimation。
- **GigaBrain-0.5M\*** (Team et al., 2026) — 大规模 world-model RL。

**Level 2（world model 和 policy co-evolution）**：
- **World-VLA-Loop** (Liu et al., 2026b) — https://arxiv.org/abs/2602.06508
- **VLAW** (Guo et al., 2026a) — https://arxiv.org/abs/2602.12063 — iterative repair-and-improve。
- **WoVR** (Jiang et al., 2026) — https://arxiv.org/abs/2602.13977 — Keyframe-Initialized Rollouts + explicit co-evolution。

Eq 19 是 co-evolution 公式：
$$
\phi^{k+1} \gets \text{UpdateWM}(\phi^k, D_{real} \cup D_{policy}(\pi_{\theta^k}))
$$
$$
\theta^{k+1} \gets \text{UpdatePolicy}(\theta^k, \hat{D}(\phi^{k+1}))
$$
- $\phi^k$：第 k 轮的 world model 参数
- $\theta^k$：第 k 轮的 policy 参数
- $D_{real}$：真实数据
- $D_{policy}(\pi_{\theta^k})$：当前 policy rollout 出来的数据
- $\hat{D}(\phi^{k+1})$：用新 world model 生成的 imagined 数据

**Intuition**：Level 1 是 "RL in a world model"，Level 2 是 "RL with a world model" —— 后者承认 learned simulator 是 imperfect 的，必须和 policy 一起迭代改进。这是 paper 强调的核心 trend 之一。

### 4.2 World Model for Evaluation

三种 evaluation 形式：

**Rollout-based candidate assessment**：
- **GPC** (Qi et al., 2026) — https://arxiv.org/abs/2509.24948 — frozen policy + world model 在 deployment 时 online rank candidate action。
- **IRASim** (Zhu et al., 2025b, ICCV) — https://arxiv.org/abs/2412.16138
- **World-in-World** (Zhang et al., 2025a) — https://arxiv.org/abs/2510.18135 — closed-loop planning。
- **DreamPlan** (Jia et al., 2026a) — 用 world model rollout 构造 preference pair 训练 signal。

**MPC-style**：
- **TD-MPC2** (Hansen et al., 2024) — https://arxiv.org/abs/2310.16828
- **LeWorldModel** (Maes et al., 2026) — https://arxiv.org/abs/2603.19312 — end-to-end JEPA formulation。

**Whole-policy evaluator**：
- **Evaluating Gemini Robotics Policies in Veo World Simulator** (Team et al., 2025) — https://arxiv.org/abs/2512.10675
- **WorldEval** (Li et al., 2025e) — https://arxiv.org/abs/2505.19017
- **WorldArena** (Shang et al., 2026) — https://arxiv.org/abs/2602.08971

**带 explicit feedback head**：
- World-Env (reward + termination prediction)
- VLA-RFT (verified rewards)
- World-VLA-Loop (joint predict future obs + reward)
- RISE (progress value model)

---

## 5. Sec 5：Robotic Video World Models 的四个 capability 阶段

这是 paper 第二重的 section，给出 Fig 6 的四阶段 progression：

### Stage 1：Imagination for Policy Learning

把 video generation 当 imagination engine，synthesize future execution 当 supervision。
- **UniPi** (Du et al., 2023)
- **Video Language Planning** (Du et al., 2024, ICLR) — https://arxiv.org/abs/2310.10625
- **Dreamitate** (Liang et al., 2024, CoRL)
- **RoboDreamer** (Zhou et al., 2024, ICML) — compositional world modeling，把 instruction 分解成 reusable primitives。
- **ManipDreamer** (Li et al., 2025f) — action tree + depth + semantic guidance。
- **DreMa** (Barcellona et al., 2025, ICLR) — https://arxiv.org/abs/2410.15791 — Gaussian Splatting + physics simulator 当 digital twin。
- **PhysWorld** (Mao et al., 2025) — 把 generated motion 通过 object-centric residual RL 落地。
- **DreamGen** (Jang et al., 2025a, CoRL) — https://arxiv.org/abs/2505.01828 — strong video generator 适配 target embodiment，恢复 latent action。

### Stage 2：Action-Controllable Video World Models

从"plausible"到"action-faithful"。
- **IRASim** (Zhu et al., 2025b, ICCV) — frame-level action conditioning。
- **RoboEnvision** (Yang et al., 2025, IROS) — long-horizon 多任务。
- **RoboMaster** (Fu et al., 2026, ICLR) — collaborative trajectory control，分 phase 建模 robot arm + object 耦合运动。
- **Ctrl-World** (Guo et al., 2026b, ICLR) — https://arxiv.org/abs/2510.10125 — joint multi-view + frame-level action control + memory-based long-horizon。
- **EnerVerse-AC** (Jiang et al., 2025c) — https://arxiv.org/abs/2505.09723 — action-conditional multi-view generator。
- **Interactive World Simulator** (Wang et al., 2026c) — https://arxiv.org/abs/2603.08546
- **EVA** (Wang et al., 2026b) — https://arxiv.org/abs/2603.17808 — inverse dynamics reward 对齐 video world model 和 executable action。

### Stage 3：Structure-Aware Generation with Interaction / Geometry Priors

引入 mask / geometry / viewpoint / identity 当中间结构。
- **Mask2IV** (Li et al., 2025a) — https://arxiv.org/abs/2510.03135 — 两阶段：先预测 actor + object interaction trajectory，再条件生成 video。
- **TesserAct** (Zhen et al., 2025) — https://arxiv.org/abs/2504.20995 — 4D embodied world model over RGB + depth + normal。
- **RoboVIP** (Wang et al., 2026a) — visual identity prompting for multi-view。

### Stage 4：Foundation-Scale Video Backbones → Reusable World Models

- **Vid2World** (Huang et al., 2026, ICLR) — 把 pretrained video diffusion model 系统性改造成 interactive world model。
- **Genie Envisioner** (Liao et al., 2026, ICLR) — unified world foundation platform。
- **DreamDojo** (Gao et al., 2026a) — https://arxiv.org/abs/2602.06949 — 大规模 human egocentric pretraining + continuous latent action 桥接 human / robot。
- **WoW** (Chi et al., 2025c) — https://arxiv.org/abs/2509.22642 — 强调 physical intuition 必须 from interaction 数据，不能 from passive video。
- **UnifoLM-WMA-0** (Unitree, 2025)
- **Cosmos Predict 2.5** (Ali et al., 2025) — https://arxiv.org/abs/2511.00062 — NVIDIA 的大规模 video foundation world model。
- **GigaWorld-0** (Team et al., 2025b) — https://arxiv.org/abs/2511.19861 — controllable video branch + physically grounded 3D branch。
- **ABot-PhysWorld** (Chen et al., 2026d) — https://arxiv.org/abs/2603.23376

---

## 6. Sec 6：Navigation 和 Autonomous Driving

### 6.1 Navigation

World model 在 navigation 里的价值是把"看不见的空间"变成 predictive planning substrate。
- **Pathdreamer** (Koh et al., 2021, ICCV) — https://arxiv.org/abs/2103.04507 — 生成 360° RGB / depth / semantic 给 unvisited viewpoint。
- **VISTA** (Huang et al., 2025c) — imagine-and-align strategy。
- **VISTAv2** (Huang et al., 2025b) — https://arxiv.org/abs/2512.00041 — egocentric future rollout → online value map。
- **NWM** (Bar et al., 2025, CVPR) — https://arxiv.org/abs/2406.14873 — 把 controllable video generation 当 navigation world model。
- **SparseVideoNav** (Zhang et al., 2026) — sparse future generation 替代 dense long-horizon rollout。
- **EgoWM** (Bagchi et al., 2026) — https://arxiv.org/abs/2601.15284 — internet-scale video diffusion 适配 egocentric。

### 6.2 Autonomous Driving

Driving 的要求比 manipulation 更严：long-horizon forecasting、multi-agent interaction、structured geometry、safety-critical。
- **MILE** (Hu et al., 2022, NeurIPS) — https://arxiv.org/abs/2211.16462 — latent dynamics + geometric inductive bias。
- **OccWorld** (Zheng et al., 2024, ECCV) — https://arxiv.org/abs/2407.15028 — 3D occupancy space world model。
- **GAIA-1** (Hu et al., 2023) — https://arxiv.org/abs/2309.17080 — multimodal sequence over video + text + action tokens。
- **DriveDreamer** (Wang et al., 2024a, ECCV) — https://arxiv.org/abs/2309.09777 — diffusion + structural constraints。
- **Drive-WM** (Wang et al., 2024b, CVPR) — https://arxiv.org/abs/2311.13584 — multi-view future + image-based reward。
- **UniDWM** (Xiong et al., 2026) — https://arxiv.org/abs/2601.04453
- **DriveWorld-VLA** (Liu et al., 2026a) — https://arxiv.org/abs/2602.06521 — latent world state 当 planner decision state。
- **DriveVLA-W0** (Li et al., 2026c, ICLR) — future image prediction 提供 dense self-supervision。
- **SteerVLA** (Gao et al., 2026b) — https://arxiv.org/abs/2602.08440 — 高层 VLM 当 semantic world model 引导低层 VLA 处理 long-tail driving。

---

## 7. Sec 7：Benchmarks、Datasets、Results

### 7.1 三层 benchmark 体系

**Layer 1 — Open-loop predictive quality**：给定 (obs, action, instruction) 生成 future，看是否 action-faithful。
- **RBench** (Deng et al., 2026) — https://arxiv.org/abs/2601.15282
- **EWMBench** (Yue et al., 2025) — https://arxiv.org/abs/2505.09694 — factorized view：scene consistency / motion correctness / semantic alignment。
- **DreamGen Bench**
- **EVA-Bench** (Chi et al., 2025b) — long-horizon + OOD robustness。

**Layer 2 — Closed-loop task utility**：world model 当 environment simulator / policy evaluator。
- **WorldArena** (Shang et al., 2026) — https://arxiv.org/abs/2602.08971
- **WorldEval** (Li et al., 2025e) — rank consistency + value fidelity。
- **WorldGym** (Quevedo et al., 2025) — Monte Carlo evaluation。
- **World-in-World** — closed-loop planning benchmark，最难，expose compounding error。

**Layer 3 — Physical consistency / controllability / executability diagnostics**：
- **WorldSimBench** (Qin et al., 2025, ICML) — https://arxiv.org/abs/2410.06405 — manipulative evaluation + IDM-based recovery。
- **WoW-World-Eval** (Fan et al., 2026) — https://arxiv.org/abs/2601.04137 — IDM-based Turing Test for executability。
- **WM-ABench** (Gao et al., 2025b, ACL Findings) — atomic capability decomposition：spatial / temporal / motion / mechanistic / counterfactual。
- **DrivingGen** (Zhou et al., 2026) — https://arxiv.org/abs/2601.01528 — trajectory plausibility + temporal coherence。

### 7.2 Dataset 多轴视角

Table 3-4 给出 dataset 的多维度比较。关键 axes：
- **General trajectory**：Open X-Embodiment (OXE)、DROID、BridgeData V2、AgiBot World、RoboMIND 2.0
- **Cross-embodiment**：OXE、DROID、BridgeData V2、DexWild、MV-UMI
- **Human-to-robot prior**：UMI、MV-UMI、ActiveUMI、DexWild、EgoMimic、PHSD/In-N-On、UniHand 2.0、Action100M
- **Contact / physics**：RH20T、RH20T-P、Humanoid Visual-Tactile-Action、VTDexManip、Hoi!、FreeTacMan
- **Synthetic / recipe**：RoboTwin 2.0、UniHand 2.0、Action100M

Open X-Embodiment: https://robotics-transformer-x.github.io/  
DROID: https://droid-dataset.github.io/  
AgiBot World: https://agibot-world.com/  
UMI: https://universal- manipulation-interface.github.io/

### 7.3 LIBERO 上的代表结果

Table 5 (LIBERO 4-suite)：

| Group | Method | Spatial | Object | Goal | Long | Avg |
|-------|--------|---------|--------|------|------|-----|
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Unified VLA | RynnVLA-002 | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Latent-space | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |

**关键观察**：
1. 多个 paradigm 都能达到 ~98% avg —— 说明 world model 的 utility 不绑定到单一 architecture。
2. Long suite 是关键 differentiator —— Spatial/Object 高的方法不一定 Long 也高。
3. Latent-space (VLA-JEPA, JEPA-VLA) 不生成 pixel 也能接近 SOTA —— 验证 paper 论点："photorealistic video generation is not necessary for effective embodied control"。

Table 6 (RoboTwin / CALVIN / SIMPLER)：
- RoboTwin A/B（harder randomized env）的 gap 大（如 HALO：80.5 / 26.4），说明 cross-benchmark generalization 仍是 problem。
- CALVIN ABCD（C-D protocol）：UP-VLA 4.42，DreamVLA 4.44，Unified VLA 4.63 —— 这是 long-horizon 多任务 setting。
- SIMPLER Google Robot：VideoVLA 73.1 / 62.8（两 protocol variant）。

---

## 8. Sec 8：六大 Open Challenges

1. **Causal Conditioning Gaps** — world model 的 future 对 historical context / task intent 响应强，对 pending robot action 响应弱，导致 future "intention-consistent but not action-consistent"。WorldVLA 的 implicit unified training 是缓解方向。

2. **Efficiency Bottlenecks** — 训练 + 推理都贵。MimicVideo / LingBot-VA 用 partial denoising；LeWorldModel 走 latent-only；Fast-WAM 训练用 world model 推理抛弃。

3. **Multi-Modal Perception Bottlenecks** — vision + proprioception 不够，friction / stiffness / contact stability 需要 tactile / force feedback。问题：tactile 高频低维 vs visual 低频高维，joint latent optimization 容易 visual dominance。

4. **Classical Control Integration** — MPC 需要大量 rollout，real-time 部署难。Lyapunov stability / robust control 的 formal guarantee 和 neural expressivity 怎么 reconcile 是 frontier。

5. **Symbolic Structure Integration** — pixel-based rollout 长 horizon error accumulate。Symbolic world model over predicates / relations / occupancy map 更 stable + compositional。Hybrid (perceptual + symbolic) 是 promising direction。代表：VisualPredicateator (Liang et al., 2025c, ICLR), ExoPredicator (Liang et al., 2026, ICLR)。

6. **Open Challenges in Evaluation Metrics** — visual realism ≠ control utility。需要 function-aware metric：task success、policy-ranking fidelity、executability diagnostics。

---

## 9. 跨领域 intuition 总结

**Trend 1：从 decoupled 到 unified**。早期 UniPi 是"video model + IDM head"两阶段；现在 Cosmos Policy、DreamZero 把 video + action 塞进一个 diffusion sequence；再到 VLA-JEPA 完全在 latent space 做。耦合越来越紧，但 Fast-WAM 的发现是个反例：co-training 在 training 有用，inference 时 future imagination 可能不必要。

**Trend 2：从 plausible 到 actionable**。Sora-style 视觉真实度已经不是目标，action faithfulness / executability / closed-loop utility 才是。WorldSimBench、WoW-World-Eval 的 IDM Turing Test 是这个 trend 的 metric 体现。

**Trend 3：从 frozen simulator 到 co-evolving simulator**。Level 1 RL in world model 已经成熟，Level 2 (WoVR) 让 simulator 和 policy 一起迭代，因为"imagined RL 只和 simulator 一样 reliable"。

**Trend 4：从 task-specific video prediction 到 foundation world model**。Cosmos Predict 2.5、GigaWorld-0、DreamDojo 都在 build reusable world backbone。

**Trend 5：multi-modal beyond vision**。tactile / force / audio 正在进入 world model。Hoi!、VTDexManip、Humanoid Visual-Tactile-Action、OmniVTA 都是早期 signal。

**Trend 6：hybrid symbolic + neural**。pixel 和 latent 都不够 long-horizon reliable，symbolic abstraction 是 complementary direction。

---

## 10. 个人 commentary（build your intuition）

这篇 paper 在我看来最重要的 insight 是 Sec 3.1 那个 unified joint distribution view —— 它告诉我们：policy、passive world model、controllable world model、inverse dynamics 都是同一物体的不同投影。这有两个含义：

1. **架构设计 space 是连续的**。decoupled → single-backbone → MoT → unified VLA → latent WM 是这条连续谱上的点，不是互斥选项。WoG、DIAL 这种"把 world modeling 塞进 action 的 condition space"的设计就是谱上的新点。

2. **World model 的真正定义是 functional 的**，不是 architectural 的。一个 model 算不算 robot world model，取决于它的输出能不能被 downstream policy-related computation 用。这把 Sora 排除在外，把 VLA 内部那个 implicit future prediction 纳入。

对 robot learning 的实际意义：
- **Sample efficiency**：video pretraining 提供 spatiotemporal prior，在小 robot data 下增益明显（Gen2Act、DreamGen 的 evidence）。
- **Long-horizon reasoning**：reactive VLA 在 long horizon 上 compounding error 严重，predictive structure 是 mitigation。
- **Failure recovery**：world model 当 evaluator 可以 detect + reject bad action，但前提是 action faithfulness 足够（Ctrl-World 的 emphasis）。

Open question 我觉得最值得追：
- **Latent vs pixel 谁赢**？VLA-JEPA 在 LIBERO 接近 SOTA 不生成 pixel，但 Cosmos Policy 这种 pixel-generating 的方法也在最前面。可能答案和 task 有关 —— 短 horizon dense contact 的 latent 够用，long horizon sparse reward 的 pixel 更 informative。
- **Co-evolution 的稳定性**？WoVR 的 iterative update 会不会 collapse？这个方向缺乏理论分析。
- **Symbolic grounding 怎么学**？VisualPredicateator 用 VLM 落地到 predicate，但 predicate space 是 pre-defined 还是 learned？这个 interface 是大瓶颈。

参考链接：
- Survey 主页：https://ntumars.github.io/wm-robot-survey/
- GitHub：https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy
- 通讯作者：Jianfei Yang (jianfei.yang@ntu.edu.sg)
- NTU MARS lab：https://mars-ntu.github.io/

如果你想我把某个具体方法（比如 Cosmos Policy 的"action as latent frame"具体怎么实现、VLA-JEPA 的 leakage-free 究竟怎么做的、WoVR 的 Keyframe-Initialized Rollout 数学细节）拆开讲，告诉我，我可以再深入。
