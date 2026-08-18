---
source_pdf: ENPIRE Agentic Robot Policy Self-Improvement in the.pdf
paper_sha256: 7e91d8c80166286bd26a67c3b4ee1be1a7e059d25bcb8816aa769822c07bf549
processed_at: '2026-08-18T11:18:22-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ENPIRE 人话版

好的 Karpathy，我换种语气重讲一遍。

## 这 paper 到底干啥的

一句话：**让 coding agent 自己在 real robot 上做 research**。

之前你做的 [autoresearch](https://github.com/karpathy/autoresearch)，是 agent 在单卡 GPU 上跑 nanochat training，自己改超参、改架构、看 loss、再来一轮。整个过程 trial 几乎免费，因为 PyTorch forward 几秒钟。

ENPIRE 把这个 loop 搬到 physical world。trial 不再免费——每次 rollout 要 reset 场景，要 perception pipeline 验证成功失败，硬件还可能撞坏。所以 cost structure 完全变了，**robot hour 变成 binding resource，GPU 反而闲着**。

## 为什么不能直接搬

你可能会想：把 agent harness 套到 robot 上不就完了，reward 给个 success/fail 信号，让 agent 改 policy code。

问题在于：在 digital autoresearch 里，trial 免费意味着 agent 可以暴力 try 1000 个 idea 看哪个 work。在 physical autoresearch 里，每个 idea 的 trial 要：
- 等 reset（几秒到几十秒）
- 跑 rollout（几秒到几分钟）
- perception pipeline 验证（150ms 起步）
- 万一 damage 硬件（不可逆）

更要命的是 **credit assignment** 问题。如果 agent 同时改 reward function 又改 policy，policy 表现差你根本不知道是 policy 学错了还是 reward function 有 bug。在 sim 里这个还能忍受，在 real 上这种 ambiguity 是致命的。

## ENPIRE 的核心 design：把 loop 切两半

论文的 key insight 是：**先把 environment 冻死，再让 agent 改 policy**。

具体来说，Stage 1 是 human-in-loop 的一次性 setup。你给 agent 看几分钟的成功失败 demo，让它自己写 reward function、reset pipeline、safety constraint。写完之后这组 API 就冻住了，变成一个 immutable Gym interface。后续 Stage 2 agent 只能改 training code（policy 架构、超参、algorithm 选择），不能碰 environment。

这跟你 [autoresearch](https://github.com/karpathy/autoresearch) 里 fix 死 eval harness 的思路一脉相承——不让 agent 自己改 benchmark，否则它会 cheat。ENPIRE 把 environment 当 benchmark 冻死，agent 的 search space 就收窄到 policy code。

这步 design 的取舍很关键。好处是 credit assignment 干净了，坏处是 Stage 1 写错 reward function，Stage 2 永远学错的。论文里 zip-tie cutting 的 reward 用双视图几何测试，pin insertion 的 reward 是视觉对齐 + 插入深度 + 力感知三路 fuse——这些都是 agent 自己写的，human 只看了 sandboxed eval set 的准确率。

## Reset 这个环节其实最难

Paper 里 reset 没太强调，但实际上它是 physical autoresearch 的 silent bottleneck。

Sim 里 `env.reset()` 一行代码，instant。Real 里 reset 要：detect 物体位置（SAM3 segmentation）、估 6DoF pose（BundleSDF）、planning（cuRobo）、torque-verified grasp、pick-and-place、verify。

这串 pipeline 任何一个环节挂了——SAM3 mask 错了、cuRobo plan fail、grasp slip——整个 episode 的 data 就污染了。论文没 quantify reset failure rate，但我猜这是 long-horizon autoresearch 最大的 noise source。

一个聪明 design 是论文借用了 [CaP-X](https://arxiv.org/abs/2603.22435) 的思路：reset 不回到 episode 起点，而是回到"关键动作开始前"的瞬间。比如 pin insertion 不从 pickup pin 开始 reset，而是从 pin 已经 hover 在 hole 上方开始。这样 90% 的样本都集中在最难的 precision insertion 阶段，不浪费在 approach 上。

## Multi-agent 用 Git 协作

为了加速，他们搞了 8 个 robot station，每个 station 一个 coding agent。Agent 之间不直接通信，全部通过 Git。

每个 agent 持有一个 branch，push 自己的修改，pull 其他 branch 的 commit，可以 cherry-pick 任何人的 idea。没有 central coordinator。

这跟 [Voyager](https://voyager.on-firmament.io/) 在 Minecraft 里建 central skill library 思路类似，但 ENPIRE 是 P2P Git-based，更 robust（单 station 挂了不影响其他），但 coordination overhead 大。

这个 overhead 在 4 agent 之前还能扛住，到 8 agent 的时候 token cost 直接飙升。原因直观上就是：每个 agent 要 maintain 其他 7 个 agent 在干啥的 mental model。8 个 agent 就有 8×7 = 56 条 "peer summary" 的关系，所以 token cost 是 super-linear 涨的。

这是 paper 自己承认的 limitation，也是未来工作的一个 obvious direction。Hierarchical 多 agent（一个 leader summarize，其他 worker 只读 leader）应该能把 $O(N^2)$ 压到 $O(N)$。

## 两个新 metric：MRU 和 MTU

这是 paper 最有 lasting value 的 conceptual contribution。

**MRU** (Mean Robot Utilization)：robot 真正在干活的时间 / 总 wall-clock time。Agent 在 read logs、write code、debug、等 LLM 响应的时候，robot 是 idle 的。理想情况 MRU = 1。

**MTU** (Mean Token Utilization)：fleet 每分钟烧多少 token。用来量化 agent team 的 token burn rate。

**Token-to-Success**：拿到成功 policy 需要多少 token。这个 metric 在多 agent fleet scaling 时最有用——paper 显示 8 agent 比单 agent 快，但 token cost 涨得比 wall-clock 降得快，所以 trade-off 不一定划算。

这三个 metric 给 future physical autoresearch 一个 fair benchmark 维度。之前 autoresearch 的 benchmark 只看 capability 或 cost-per-paper，没有衡量 "scarce physical resource 用得多高效" 的维度。ENPIRE 这块填补了一个 gap。

## 几个有意思的 finding

### Finding 1: Sim 容易 Real 难

Paper Figure 5 显示，三个 frontier coding agent（Codex、Claude Code、Kimi Code）在 sim 的 Push-T 上都能 2 小时内跑到 95%。但 real Push-T 上只有 Codex 成功，另外两个失败。

原因是 sim 的 physics deterministic，contact friction、robot dynamics 都是 ideal。Real 里 friction 因灰尘变化、backlash 因 wear 变化、camera autofocus 微抖，整个 trial 的 variance 大得多。

**这告诉我们：physical autoresearch 的难度不是 task 难度的函数，是 task-determinism 的函数**。Push-T 涉及 contact，contact 是 high-variance source，所以难。如果 task 是 free-space motion，real 应该跟 sim 差距小很多。

### Finding 2: Pin insertion 100% success，50 consecutive

这个 result 最 impressive。4mm clearance（human hair 直径大概 0.1mm，所以 40 倍），要求 50 consecutive successes。Scaling 1→8 agents 把 wall-clock 从 1.5h 压到 40min。

注意 metric 是"8 retries 内完成 1 次成功"，不是 i.i.d. best-of-8。这 capture 的是 **in-context recovery**——policy 看到 failure 能不能 recover。这比 [pi0.5](https://www.physicalintelligence.company/blog)、[GR00T](https://github.com/NVIDIA/GR00T) 那种 i.i.d. evaluation 更接近真实部署场景。

部署时一定会 fail，关键是能不能 recover。这点 metric design 上 paper 是 ahead of 大部分 VLA 工作的。

### Finding 3: Native vision 不一定最好

Appendix C.3 做了个 ablation：Codex 有 native vision vs. 没有 native vision 但能 call 外部 image understanding module vs. 完全没 vision。

直觉上 native vision 应该最好，function-call vision 第二，no vision 最差。结果：native vision 最好，但 **no vision 比 function-call vision 快**。

Paper 的解释是 function call 有 IPC overhead。我换个角度：当 reward signal 设计得好（automated verification 准确），agent 从 reward 序列就能推断 failure mode，不需要看图。Logs 是高度压缩信号，反而让 agent 更 focus。这跟人类 researcher debug 时也经常只看 metric 不看 rollout video 是一致的。

### Finding 4: VLA + tool call 的 emergent synergy

Section 3.5 最有意思。Agent 发现 GR00T VLA 在 long-horizon 任务里 grasp 成功率低，所以自动加了一步：先 motion plan hover 到目标上方，再 grasp。这步 procedural tool call 让 VLA 的成功率大幅提升。

这是个 emergent behavior——agent 自己 discover 的 strategy。它揭示了 VLA + symbolic tool 的混合架构可能比纯 end-to-end VLA 更实用。这跟 [CaP-X](https://cap-x.github.io/) 的设计哲学一致，但 ENPIRE 是 agent autonomous discover 的。

## 硬件层的两个细节

### Torque-limited gripper 是无人值守的灵魂

Gripper 不控制 position（rigid target width），而是控制 bounded force。当 fingers 接触物体，电流上升，触顶就停。相当于 mechanical impedance control 的简化版。

好处：grasp 自动 conform 到物体形状，不需要精确知道物体尺寸；接触力 capped，不会 damage fixture；失败时 stall 而非 push（safe）。

这是 8 个 station 能跑 1.5 小时不动人的硬条件。如果用 position control，一个 bad grasp 就可能把 motherboard 撞坏。

### 三层 RL 系统

Deployment layer（robot machine）+ Learner（GPU host）+ Actor（Portal endpoint）三层解耦，用 disk-based contract 通信。这是 [SERL](https://serl-benchmark.github.io/) 标准设计。

关键 design 是 deployment layer 和 learner 之间用 disk 而不是 in-memory queue——这让两层可以完全 fail-and-restart 独立，符合 fleet fault-tolerant 要求。

数据 mixing 用 [RLPD](https://arxiv.org/abs/2306.09311) 风格：RL-generated transition 进 online replay buffer，human demo 进 demo buffer，每个 batch 都 sample 两边。这让 RL cold start 时有 demo bootstrap，是 SERL 在 real robot 上能 work 的关键之一。

## Transfer 是 markdown 不是 weight

Paper Section 3.4 提到 transfer。Pin insertion 跑完，把 insights 总结成 markdown 文档，传给 GPU insertion 的 agent。注意：是 markdown summary，不是 checkpoint weights。

这是 LLM-centric 的 transfer 方式。传统 robot learning 的 transfer 是 weight transfer（fine-tune from pretrained），ENPIRE 是 linguistic transfer（让 agent 读文档）。这有两个 implication：

1. **好处**：transfer 是 symbolic 的，可解释、可 audit。人类能读 markdown 检查 agent 学到了啥。
2. **坏处**：信息密度低。一个 BC regularization 的 insight 用 markdown 写出来就是几句话，但 weight 里的 implicit knowledge 远比 markdown 能表达的多。

未来一个 obvious direction 是 hybrid：markdown + LoRA adapter，让 agent 既读文档也载入 modular weight。

## 跟你 autoresearch 的对比

| 维度 | Karpathy autoresearch | ENPIRE |
|------|----------------------|--------|
| Substrate | Digital (GPU) | Physical (robots + GPUs) |
| Trial cost | 几分钟 | 几秒到几分钟 + reset time |
| Binding resource | GPU | Robot |
| Verification | LM eval harness | Agent-designed perception pipeline |
| Reset | `model.reset()` instant | Procedural tool calls，几秒到几十秒 |
| Safety | None needed | Hard joint limits + force-limited gripper |
| Multi-agent | Single agent | 8 stations，Git-based P2P |
| Failure mode | Bad checkpoint，throw away | Hardware damage，irreversible |
| Transfer | Checkpoint | Markdown summary |

核心 takeaway：**ENPIRE 的 design choice 几乎都是从"physical substrate 的 cost structure 不一样"推出来的**。Immutable Gym API 是为了 credit assignment；force-limited gripper 是为了无人值守 safe；Stage 1 human bootstrap 是因为 physical trial 太贵不能让 agent 自己 explore env；Git-based P2P 是为了 fault-tolerant。

这些 design 在 digital autoresearch 里都不需要。这给我一个 hypothesis：**autoresearch harness 的 design complexity 是 substrate cost 的单调函数**。未来如果 robot hardware 便宜到 GPU 那个量级，很多 ENPIRE 的 safeguard 可以去掉。短期看 robot 不会便宜到那个程度，所以 ENPIRE 这种"小心翼翼"的 harness design 是必要的。

## 我觉得最有 lasting value 的部分

1. **EN as immutable API abstraction**——让 physical autoresearch 的 credit assignment tractable。这个 idea 未来会被反复 reference。
2. **MRU/MTU metrics**——给未来 physical autoresearch benchmark 提供新维度。之前 autoresearch benchmark 只看 capability 或 cost-per-paper，没有衡量 physical resource utilization 的维度。
3. **Two-stage bootstrap**——human-in-loop 一次性，amortize 到所有后续 runs。这是 human-AI collaboration 的合理 trade-off。
4. **Git-based decentralized multi-agent**——虽然 super-linear，但是 de facto scalable 的 protocol，未来 hierarchical 改进空间大。

## 我觉得最 weak 的部分

1. **Reward hacking 没 address**。Agent 自己写 reward 又自己 optimize，sandboxed eval set mitigate 但 OOD robustness 没 guarantee。
2. **Coordination overhead 没 mitigate**。8-agent token 飙升只是 report 没解决。
3. **Reset robustness 没 quantify**。Reset pipeline failure rate 是 long-horizon autoresearch 的 silent killer。
4. **8-retry success metric 偏宽松**。best-of-8 不等于 1-shot success，部署时可能没那么 robust。

## 总评

ENPIRE 是 first credible attempt at physical autoresearch。它的 contribution 是 harness + metrics，不是 algorithm。跟你 autoresearch 项目的 contribution 性质一致——harness 而非 algorithm。

短期看，physical autoresearch 的 design complexity 会比 digital 高很多，因为 substrate cost 高。长期看，如果 robot hardware 按摩尔定律降价，ENPIRE 的 safeguard 可以逐步去掉，最终趋近 digital autoresearch 的简洁性。

ENPIRE 跟 [pi0.6](https://www.physicalintelligence.company/blog/pi06)、[GR00T](https://github.com/NVIDIA/GR00T)、[OpenVLA](https://openvla.github.io/) 这类 end-to-end VLA 路线是互补而非替代。VLA 提供 base capability，ENPIRE 提供 autonomous refinement loop。未来 likely 是 VLA + ENPIRE-style autoresearch 的组合，类似 LLM + tool use 的组合。

希望这个版本更直觉。如果你想再深入哪块（比如 RL data mixing 的细节、Git coordination 的形式化、或者 hardware 控制层的实现），我可以继续展开。

### 参考链接

- ENPIRE project page: https://research.nvidia.com/labs/gear/enpire
- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- Voyager: https://voyager.on-firmament.io/
- CaP-X: https://arxiv.org/abs/2603.22435
- GR00T: https://github.com/NVIDIA/GR00T
- SERL: https://serl-benchmark.github.io/
- RLPD: https://arxiv.org/abs/2306.09311
- pi0.5: https://www.physicalintelligence.company/blog
- OpenVLA: https://openvla.github.io/
- Eureka: https://eureka-research.github.io/
- DrEureka: https://eureka-research.github.io/dr-eureka
- Residual RL (PLD-RL): https://arxiv.org/abs/2511.00091

---

# ENPIRE 论文深度解读：从 Digital Autoresearch 到 Physical Autoresearch

Karpathy 你好，这篇 ENPIRE（Environment + Policy Improvement + Rollout + Evolution）我读了几遍，因为这正好落在我自己最有体感的那条轴上——你 2025 年 release 的 [autoresearch](https://github.com/karpathy/autoresearch) 项目跑的是 single-GPU nanochat training 的自动 ML 研究，本质上是在数字 substrate 上闭合 hypothesis → experiment → observation 的循环。ENPIRE 把这条 loop 试图搬到物理世界，而且 attempt 的是 manipulation 这种 contact-rich 的场景。整篇论文的核心 contribution 不是一个新 model，也不是一个新 RL algorithm，而是一个 **harness abstraction**，以及一组 **physical resource utilization metrics**。下面我尽量把 intuition 拉出来，并补一些论文里没有展开的技术细节。

---

## 1. 一句话 framing：ENPIRE 把"autoresearch"从 compute-bound 推到 robot-bound

Digital autoresearch（[AI Scientist](https://github.com/SakanaAI/AI-Scientist)、[MLE-bench](https://github.com/openai/mle-bench)、[SWE-bench](https://www.swebench.com/)、你自己的 [autoresearch](https://github.com/karpathy/autoresearch)）的 binding resource 是 GPU hour。每个 trial 几乎免费（PyTorch forward 几秒钟），所以可以暴力 hill-climb。

ENPIRE 切换 substrate 之后，binding resource 变成 **robot-access budget**：
- 每次 trial 要 reset 场景（哪怕自动 reset，物理上要花时间）
- 每次 verification 要 sensor 读数 + perception pipeline（150ms 的 reward inference 才能接近人类视觉 reactiveness）
- 一次 rollout 失败可能 damage 硬件，所以 safety constraint 必须硬编码

这就是论文里反复强调 "physical autoresearch as a distinct problem" 的根本原因。它并不是"把 ML autoresearch 搬到 robot 上"这么简单，而是整个 optimization substrate 的 cost structure 完全变了。

---

## 2. 四模块的 ENPIRE 闭环：架构解析

论文 Figure 2 画了一个 closed loop，但没把数据流讲清楚。我重新拆一下：

```
        Human feedback (Stage 1 only, one-time)
              │
              ▼
    ┌──────────────────────┐
    │   EN (Environment)    │  ←  safety constraints + automated verification
    │                       │     + automated reset；output：immutable Gym API
    └──────────────────────┘
              │  reward / done / truncated / obs
              ▼
    ┌──────────────────────┐
    │   R (Rollout)         │  ←  policy 在 N 个 robot station 上并行 rollout
    │   - 30Hz policy inf   │     log：video, proprioception, reward, action_source
    │   - 100Hz joint PD    │
    └──────────────────────┘
              │  rollout buffer directory（disk-based contract）
              ▼
    ┌──────────────────────┐
    │  PI (Policy Improve)  │  ←  coding agent 读 logs，改 training code
    │  - BC / online RL     │     write training code (RL / BC / hybrid)
    │  - offline→online RL  │
    └──────────────────────┘
              │  Git branch push / pull
              ▼
    ┌──────────────────────┐
    │   E (Evolution)       │  ←  N agent stations 异步并行，通过 Git 协作
    │   - cherry-pick ideas │     team-average best success rate 作为信号
    │   - merge branches    │
    └──────────────────────┘
              │
              └──── loop ────► 回到 R
```

### EN 模块最关键的设计：immutable API

EN 模块产出的是一组 "frozen Gym API"，相当于把 human 一开始的 effort 折成一个 deterministic transition function。这点很重要：它把 **Stage 1** 退化成 offline 优化，**Stage 2** 才是真正 autonomous 的 online hill-climbing。如果 Stage 1 写的 verification function 有 bug（比如 false positive 太多），Stage 2 的 agent 永远不会发现，因为它把 reward 当 ground truth。

这种 design choice 的 trade-off：
- **优点**：Stage 2 的 search space 收窄了（不用同时改 environment 和 policy）
- **缺点**：EN 错了就全错了，agent 没法 fix 它，只能学错的东西

这是典型的 "abstraction layer 锁定"策略，跟 compiler design 的分层思路一致——你不会希望 agent 同时改 reward 又改 policy，那样无法做 credit assignment。

### Hard safety constraints 的实现

论文 Section 2.1 只说了"restrict configuration space and kinematic behaviors"。从 Appendix B.3 推断，实现方式是：

$$\mathcal{C}_{\text{safe}} = \{ q \in \mathbb{R}^{14} \mid q_{\min} \preceq q \preceq q_{\max}, \, \| \dot{q} \|_\infty \leq \dot{q}_{\max}, \, \tau_{\text{grip}} \leq \tau_{\text{limit}} \}$$

其中：
- $q \in \mathbb{R}^{14}$：14 维 joint configuration（7 DoF × 2 arms，每 arm 是 6 关节 + 1 gripper）
- $q_{\min}, q_{\max}$：joint position limits，配置空间 hard bound
- $\dot{q}_{\max}$：velocity limit，防止 wild motion
- $\tau_{\text{grip}}$：gripper torque，由 force-limited mode 直接 cap，是无人值守 fleet 的关键 safeguard
- $\preceq$：element-wise inequality

违反任意一条立刻 trigger automated reset。这意味着 policy 的 action space 实际上是 $\mathcal{C}_{\text{safe}}$ 的子集，agent 写的 policy 必须在这个 manifold 上运作。

### Automated verification：从视频到 binary reward

论文给了一个有意思的例子：zip-tie insertion 的 reward。agent 自己设计了两视图 geometric test（Figure 4），从 top + side camera 同时确认 zip-tie strap 是否穿过 head。Inference latency < 150ms，对比 [Thorpe et al. 1996](https://www.nature.com/articles/381520a0) 的人类 visual system reaction time。

Pin insertion 的 reward 更复杂，是 hybrid：

$$r_t = \mathbb{1}\left[ \text{align}(o_t^{\text{img}}) < \epsilon_{\text{align}} \right] \wedge \mathbb{1}\left[ d_t^{\text{eef}} > d_{\text{depth}} \right] \wedge \mathbb{1}\left[ \tau_t^{\text{eef}} > \tau_{\text{contact}} \right]$$

变量：
- $o_t^{\text{img}}$：相机观测
- $\text{align}(\cdot)$：visual alignment score（pin tip 到 hole 中心的 pixel 距离）
- $d_t^{\text{eef}}$：end-effector insertion depth（从 proprioception 读出）
- $\tau_t^{\text{eef}}$：end-effector torque estimate
- $\epsilon_{\text{align}}, d_{\text{depth}}, \tau_{\text{contact}}$：thresholds，agent 自己 tune

这是论文里 reward design 的精髓：**agent 自己 fuse 多 modality 来定义 task success**，而非用 simulator oracle。这种 reward function 是 ambiguous 的（threshold 怎么定？align 怎么算？），但 agent 通过 sandboxed eval set 自己 hill-climb 验证。

### Automated reset：procedural tool calls

论文 Appendix A.1 提到 reset 用了 SAM3（open-vocabulary segmentation）+ BundleSDF（6-DoF pose tracking）+ cuRobo（GPU-accelerated collision-free trajectory optimization）。Code Snippet 2 描述 GPU insertion 的 reset pipeline：

```
1. SAM3 localization (motherboard + GPU slot)
2. RANSAC + OBB 3D bounding box estimation
3. Torque-verified grasping（gripper torque 作为 tactile surrogate）
4. cuRobo collision-free handover（pick GPU out, move to parking pose）
5. Camera-based pose verification
```

值得注意：CaP-X 的设计哲学被直接借用了——把 reset 跳到 "the onset of critical actions" 而非 episode 起点。这对 manipulation 学习是关键 design：如果 reset 永远从 episode 起点开始，agent 90% 的样本都浪费在 "approach" 阶段，而不是 "precision insertion" 阶段。

---

## 3. Two-Stage Bootstrap：为什么 EN 必须先 human-guided

这是论文最值得 build intuition 的地方。

### Argument

Physical autoresearch 的 credit assignment 问题远比 digital 严重。在 digital autoresearch 里，trial 是免费的，agent 可以暴力试 1000 种 code，看哪个跑通。在 physical autoresearch 里：
- 每次 trial 要 reset（时间成本）
- 每次 trial 可能 damage hardware（不可逆）
- Reward signal 本身需要 perception pipeline，可能 noisy

如果 agent 同时改 (environment, reward, policy)，credit assignment 是 ill-posed 的：policy 表现差，到底是 policy 的问题，还是 reward function 的问题，还是 reset 不够 robust？

ENPIRE 的解法是 **abstraction layering**：
- Stage 1 (EN)：human 提供 5 分钟 success/failure demo + specification，agent 写 reward/reset/safety。这个 stage 可以 offline verify，因为 human 还在 loop 里。
- Stage 2 (PIRE)：environment frozen，agent 只能改 training code（policy architecture, hyperparameters, algorithm choice）。这样 credit assignment 收窄到 "policy 是不是变好了"。

这跟你 [autoresearch](https://github.com/karpathy/autoresearch) 项目里的做法完全一致——你 fixed 了 evaluation harness（lm-eval-harness-style metrics），让 agent 只能改 training code，不能改 benchmark。否则 agent 可以 cheat。

### Hidden cost

Stage 1 的 human effort 论文里说 "one-time cost amortized across all subsequent runs"。这个 claim 我有点保留：每次新 task 都要重新做 Stage 1。论文 Section 3.4 提到 "Agentic Continue Learning"——把 pin insertion 的 insights 总结成 markdown，传给 GPU insertion。这个 transfer 是 markdown summary，不是 checkpoint。意味着 transfer 路径是 **linguistic abstraction**，不是 weight transfer。这是 LLM-centric 的 transfer 方式，跟传统 robot learning 的 weight transfer 完全不同。

---

## 4. MRU / MTU：physical resource accounting 的两个 metric

这是论文最有价值的概念贡献。我详细推一下。

### Mean Robot Utilization (MRU)

$$\text{MRU} = \frac{1}{N} \sum_{i=1}^{N} \frac{T_{\text{active},i}}{T_{\text{wall},i}}$$

变量：
- $N$：fleet size（1, 4, 8）
- $T_{\text{active},i}$：第 $i$ 个 robot 在 autoresearch 期间实际执行 experiment 的时间（rollout + reset）
- $T_{\text{wall},i}$：第 $i$ 个 station 的 wall-clock time，从 autoresearch 开始到结束

物理意义：robot 在 hardware 上花的有效 time fraction。agent 在 read logs / write code / debug / wait LLM 的时候，robot 是 idle 的。MRU = 1 表示 robot 永远在工作，agent 永远不让它闲下来。

### GPU Utilization

类似定义，但 measure GPU active time。注意：这里 GPU 既跑 policy inference 也跑 training，所以 GPU 高利用 ≠ robot 高利用。

### Mean Token Utilization (MTU)

$$\text{MTU} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{Tokens}_i}{T_{\text{wall},i}}$$

单位：tokens/min/fleet。用来量化 agent team 的 token burn rate。

### Token-to-Success

$$\text{Token-to-Success} = \frac{\sum_i \text{Tokens}_i}{\text{success rate}} \quad \text{或} \quad \frac{\text{total tokens until success}}{1}$$

论文 Section 4 报告 super-linear scaling：MTU 在 4 agent 之前接近 linear，8 agent 时 sharp rise。

### Intuition：为什么 super-linear

Figure 7b 里 8 agents 时 token cost 飙升，论文的解释是 "agents spend more time summarizing peer branches and less time actually operating the robot"。这里我深挖一下：

Git-based 协作意味着每个 agent 要 maintain 一个 local view of what other agents are doing。Agent 数量 $N$ 时，coordination overhead 至少是 $O(N)$（每个 agent 要 read $N-1$ 个 peer branches 的 commit log），甚至可能是 $O(N^2)$（如果 agent 之间互相 summarize 互相的 summary）。

形式化：每个 agent 每 step 消耗的 token 量 $T_i$ 满足：

$$T_i = T_{\text{base}} + \alpha \cdot \sum_{j \neq i} \text{summary}(B_j)$$

其中 $B_j$ 是 agent $j$ 的 branch state。Total token：

$$T_{\text{total}} = N \cdot T_{\text{base}} + \alpha \cdot N \cdot (N-1) \cdot \overline{\text{summary}}$$

第二项就是 $O(N^2)$ coordination cost。这是 super-linear 的根因。

### MRU 随 fleet 下降的 intuition

Figure 7a 显示 GPU utilization 随 $N$ 上升（合理，因为更多 GPU 在 fleet 里被使用），但 MRU 随 $N$ 下降。这意味着每个 robot 的 idle fraction 在上升。原因：agent 把时间花在 reading peer branches / summarizing / writing 合并 commits 上，对应的 robot 是 idle 的。

这给未来 physical autoresearch 一个 clear warning：**你不能只是 scale robot fleet，必须 design 让 robot 永远在工作 的 protocol**。一个简单的 fix 是让每个 agent 配多个 robot，agent 在等待 robot 时切到另一个 robot 继续 rollout。这跟 RL 的 asynchronous actor-learner 设计（[SERL](https://github.com/rail-berkeley/serl)、[RLPD](https://arxiv.org/abs/2306.09311)）思路一致。

---

## 5. Hardware & Control Hierarchy

### 控制层结构

```
30Hz ──────── policy inference (on RTX 5090)
                │
                ▼ action target q*
100Hz ─────── joint PD controller with gravity compensation (over CAN bus)
                │
                ▼ torque command τ
1kHz+ ─────── brushless motor driver (FOC)
                │
                ▼ current
                physical joint
```

PD + gravity compensation 的标准形式：

$$\tau = K_p (q^* - q) + K_d (\dot{q}^* - \dot{q}) + \tau_{\text{grav}}(q)$$

变量：
- $q^*, \dot{q}^*$：policy 输出的 target joint position 和 velocity
- $q, \dot{q}$：实际 joint state（encoder 读数）
- $K_p, K_d$：PD gains
- $\tau_{\text{grav}}(q)$：gravity compensation feedforward term，依赖当前 configuration

这个 design 让 PD gains 只需要 handle residual error，可以 small gains、更 stable，policy 输出也不需要 model full dynamics。

### Gripper：torque-limited compliant grasp

这是无人值守 fleet 的灵魂。Gripper 不控制 position（rigid target width），而是控制 **bounded force**：

$$\tau_{\text{grip}} = \min(\tau_{\text{cmd}}, \tau_{\text{limit}})$$

物理上：当 fingers 接触物体，电流上升，当 $\tau$ 触顶时停止。这相当于 mechanical impedance control 的一种简化形式。好处：
- Grasp 自动 conform 到 object shape（不需要精确知道 object 尺寸）
- 接触力 capped，防止 damage fixture / gripper / 物体
- 失败时 stall 而非 push（safe）

论文里有一句很关键："a bad contact results in a safe stall rather than a hardware-damaging push, with no human in the loop to intervene"。这是无人值守 8-station fleet 能跑 1.5 小时不动人的硬条件。

### YAM arm 规模

6-DoF + 1-DoF gripper × 2 = 14 actuated joints per station，8 stations 共 112 actuators。Brushless actuators over CAN bus。CAN bus 是 robotics 工业标准，bandwidth 有限（~1 Mbit/s），所以 100Hz 是合理的 upper bound。

### Compute

每 station 1× RTX 5090 32GB（注意是 5090 不是 4090，Blackwell 架构），Intel Core Ultra 9 285K 24 cores，128GB RAM。这配置差不多是 high-end workstation。**没有 cluster，没有 off-station compute**——这是 design choice，让 station 完全 decentralized，单个 station fail 不会影响其他。

---

## 6. Real-World RL System Integration

Appendix B.5 描述了三层架构（[SERL](https://serl-benchmark.github.io/) 风格）：

```
┌──────────────────────────────────┐
│  Deployment Layer (Robot machine)│  hardware orchestration, episode recording,
│  - FastAPI server               │  human-in-the-loop teleop
│    /start, /restart, /home      │
│    /avoid, /resume (Push-T)     │
└──────────────────────────────────┘
              │ disk-based contract
              │ rollout_buffer/<exp>/*.mp4 + obs tensor + action_label
              ▼
┌──────────────────────────────────┐
│  DiskBufferIngestor daemon       │  polls directory, parses episodes,
│                                  │  routes transitions by action_source
└──────────────────────────────────┘
              │ RLPD-style mixing
              ▼
┌──────────────────────────────────┐
│  Learner (GPU host)              │  RL agent (actor + critic),
│  - pixel obs → visual backbone   │  trains with online + demo buffer
│  - critic loss + actor loss      │
└──────────────────────────────────┘
              │ ZMQ msgpack
              ▼
┌──────────────────────────────────┐
│  Actor (Portal endpoint)         │  policy inference, served to controller
└──────────────────────────────────┘
```

这种三层解耦是 [SERL](https://arxiv.org/abs/2411.17739) 和 [PLD-RL](https://arxiv.org/abs/2511.00091) 的标准设计。关键 design 是 **disk-based contract** 而不是 in-memory queue——这意味着 deployment layer 和 learner 可以完全 fail-and-restart 独立，符合 fleet fault-tolerant 要求。

### RLPD data mixing

Action source label 区分 RL-generated transition 和 human demonstration。每个 training batch 是 mix：

$$\mathcal{B}_{\text{train}} = \text{sample}(\mathcal{R}_{\text{online}}, p_{\text{RL}}) \cup \text{sample}(\mathcal{R}_{\text{demo}}, p_{\text{demo}})$$

变量：
- $\mathcal{R}_{\text{online}}$：RL-generated transitions 的 replay buffer
- $\mathcal{R}_{\text{demo}}$：human demo transitions 的 buffer
- $p_{\text{RL}}, p_{\text{demo}}$：mixing ratio，agent 自己 tune

这个 mixing 让 RL 即使 cold start 也能 boot up（因为有 demo），是 [Ball et al. 2023](https://arxiv.org/abs/2306.09311) 的关键 insight，也是 SERL 在真实 robot 上能 work 的原因之一。

---

## 7. Multi-Agent Scaling：Git-Based Decentralized Collaboration

这是 ENPIRE 里 E 模块的核心。论文里 Figure 12 的 "idea tree" 给了可视化。

### Protocol

- 每个 agent station 持有独立 Git branch（fork from `autorl` baseline）
- Agent push 自己的修改，pull 其他 station 的 commits
- Agent 可以 cherry-pick 任何 branch 的 commit
- 没有 central coordinator，全部通过 Git history 共享

这跟 [Voyager](https://voyager.on-firmament.io/) 在 Minecraft 里的 skill library 思路类似，但 Voyager 是 central skill library，ENPIRE 是 P2P Git-based。后者更 robust（单 station fail 不影响其他），但 coordination overhead 更高。

### Idea Tree 分析（Figure 12）

从 Figure 12 可以读出几个有意思的点：

1. **BC regularization (I37, +10.8pp)** 是最大 single jump。这跟 [Residual RL](https://arxiv.org/abs/2511.00091) 的 finding 一致——real-world RL 容易 diverge，BC regularization 是 anchor。
2. **Batch-size tuning (I66, +0.9pp)** 和 **controller compensation (I76, +1.3pp)** 是 small incremental。说明 hill-climb 到 100% 附近时，idea 的 marginal value 急剧下降。
3. Idea tree 不是 tree，是 **DAG**：agent 可以 cherry-pick 多个 branch 的 idea 合并。论文里画成 horizontal line 表示 "related ideas"，但合并关系没画清楚。
4. 大部分 ideas 是 hollow nodes（evaluated but no gain）。这跟 [AI Scientist](https://arxiv.org/abs/2408.06292)、[Eureka](https://eureka-research.github.io/) 的经验一致：LLM-generated hypothesis 大部分 fail，关键是 cheap evaluation 让你能 fail fast。

### Git 作为 coordination substrate 的 trade-off

优点：
- Single source of truth
- 自带 versioning，可以 revert
- cherry-pick 是 well-defined operation

缺点：
- Merge conflict 需要 agent 自己 resolve（论文没说怎么 resolve）
- Token overhead（每次 git log / git diff 都要 LLM 解读）
- Latency（push-pull 周期决定 idea propagation 速度）

我感觉这里有个未探索的方向：用 vector embedding 做 idea retrieval 而非 Git text diff。Agent 写一个 commit message → embed → 其他 agent 用 semantic search 找相关 ideas。这可以大幅降低 coordination overhead。

---

## 8. 实验结果解读

### Push-T：sim vs real gap

Figure 5 显示所有三个 agent（Codex / Claude Code / Kimi Code）在 sim 都能 2 小时内达到 95%。但 real Push-T 里只有 Codex 成功，另外两个失败。

这是 physical autoresearch 的第一个 finding：**sim 的 deterministic physics 让 hypothesis testing 几乎免费，real 的 non-determinism 把 cost 拉爆**。具体原因：
- Robot dynamics（backlash, friction）varies across trials
- Contact friction（T block 和桌面）varies with dirt/dust/wear
- Camera autofocus 可能微变

这告诉我们一个 scaling law 的雏形：**physical autoresearch 的 difficulty 不是 task 难度的 function，而是 task-determinism 的 function**。Push-T 是视觉 servo + contact planning，contact 是 high-variance source。

### Pin Insertion：4mm clearance，50 consecutive successes

这是论文最 impressive 的结果。Pin insertion 4mm clearance（论文 Figure 2），要求 50 consecutive successes（不是 i.i.d. best-of-N，是允许 8 retries 内完成）。Scaling 1→8 agents，wall-clock 从 1.5h 降到 40min。

注意：metric 是 "8 retries 内 1 次成功"，不是 "8 次独立尝试"。这 capture 的是 **in-context recovery**——policy 看到 failure 之后能否 recover。这跟 [pi0.5](https://www.physicalintelligence.company/)、[OpenVLA](https://openvla.github.io/) 等 VLA 评测方式不同，那些是 i.i.d.。Real-world deployment 真正需要的是 recovery，因为部署时一定会 fail，关键是能不能 recover。

### GPU Insertion + Domain Randomization

Section 3.3 提到 multi-agent setting 下，agent 自动用 code-based policy 做 domain randomization。Domain randomization 的 variation 范围比 [Residual RL](https://arxiv.org/abs/2511.00091) 更宽。这是个有意思的 emergent behavior——agent 自己发现需要 robustify policy，所以加了 randomization。这跟 [DrEureka](https://eureka-research.github.io/dr-eureka) 的 LLM-guided domain randomization 思路一致，但是 autonomous 的。

### Zip-tie Cutting：VLA + Tool Synergy

Section 3.5 是我最喜欢的一节。Agent 自动发现 GR00T VLA 在 long-horizon 任务里 grasp 失败率高，所以加了一个 procedural tool call：先 motion plan hover 到目标上方，再 grasp。这是 **code-augmented VLA** 的 emergent pattern，跟 [CaP-X](https://cap-x.github.io/) 的设计哲学一致，但这里 agent 自己 discover 的。

 transferred to real：scissors + zip-tie cutting，agent 学到 hover-grasp-cut 的 sequence。这种 sim-to-real transfer 路径很特殊——它 transfer 的不是 weight，是 **strategy**。

### RoboCasa365 结果

Figure 6 显示 ENPIRE 在 RoboCasa365 超过 GR00T 和 CaP-X。但要注意 ablation：
- GR00T：end-to-end VLA，no tool use
- CaP-X：code-based policy，no autoresearch loop
- ENPIRE：code-based policy + autoresearch loop

所以 ENPIRE 的 win 来源是 **autoresearch loop**，不是 architecture。这呼应论文核心 thesis：loop 是 missing abstraction。

---

## 9. Simulation Benchmark: RoboCasa API Surface

Appendix D.1 给了一个非常详细的 API surface table（Table 2）。值得注意几个 design：

### Canonical script runtime 的 API gating

| API | Status | 含义 |
|-----|--------|------|
| `get_robot_state` | ✓ | Joint positions, EEF pose, gripper measurement, base/arm state |
| `set_gripper` / `open` / `close` | ✓ | Gripper 控制 |
| `move_with_curobo` | ✓ | cuRobo plan with collision avoidance |
| `move_with_pyroki` | ✓ | Pyroki IK without collision |
| `detect_object` | ✓ | Text-conditioned object detection |
| `segment_object_all` | ✓ | Multi-instance segmentation |
| `vlm_query` | ✓ | VLM query（用来 ground language） |
| `get_task_info` | C (CLI only) | Structured task info，runtime gated |
| `reset_env` | C (CLI only) | 防止 agent 自己 reset cheating |
| `get_oracle_targets` | O (Oracle only) | Ground-truth target pose，runtime gated |

最关键的设计：**`reset_env` 和 `get_oracle_targets` 在 canonical script runtime 被禁用**。这防止 agent 自己写 cheat code（"如果失败就 reset"或者"读 oracle 目标"）。

这跟你 autoresearch 项目里 fixed benchmark 的设计哲学完全一致：[eval harness must be immutable](https://github.com/karpathy/autoresearch)。

### SAM3 perception bottleneck

Figure 16/17/18 揭示一个有意思的 finding：limiting factor 不是 planning 也不是 control，是 **perception API reliability**。SAM3 在 small/ambiguous object 上 mask 失败。Agent 通过：
1. 提高分辨率（256×256 → 480×640）
2. 改写 object prompt（用更具体的描述）

来 mitigate。第二点是 LLM-style prompt engineering 在 perception API 上的应用，比较 novel。

---

## 10. Ablation：Native Vision Capability

Appendix C.3 ablation Codex 的 visual capability：
1. Codex with native vision（直接看 image tokens）
2. Codex without native vision but with image-understanding function call
3. Codex without any visual capability

结果：**1 最好，但 3 比 2 快**。这个反直觉 finding 论文解释为 "image function calls introduce additional overhead"。我换个角度解读：

- **Native vision**：agent 内化了 visual reasoning，token 流到 LLM backbone 里直接 process，no IPC overhead。
- **Function-call vision**：每次要看图就 call external module，返回 text description。Round-trip latency + token cost for description generation。
- **No vision**：agent 只能从 logs（reward, proprioception）推断。Logs 是 highly compressed signal，反而让 agent 更 focus 在 algorithm 上。

这给 future design 一个 hint：**visual grounding 对 manipulation autoresearch 是 useful 但非 necessary**。如果 reward signal 设计得好（automated verification 准），agent 可以从 reward 序列推断 failure mode，不需要看图。这跟人类 researcher debug 时也经常只看 metric 不看 rollout video 是一致的。

---

## 11. Limitations 的深层解读

论文 Section 4 列了两个 limit：resource underutilization + token cost super-linear。我加几个我自己观察到的：

1. **Reset robustness 是 hidden bottleneck**。Reset pipeline 用 SAM3 + cuRobo，如果 SAM3 mask 错了，reset 会把物体放到错位置，污染下一个 episode。论文没 quantify reset failure rate。这是 long-horizon autoresearch 的 silent killer——reset 错 1%，跑 1000 个 episode 就有 10 个 noisy data point。

2. **Reward hacking 没讨论**。Agent 自己设计 reward，然后自己 optimize policy 朝这个 reward 去。如果 reward function 有 bug（比如 false positive），agent 会 exploit 这个 bug 学到 spurious behavior。论文说 sandboxed eval set mitigate，但 eval set 也是 agent 自己 tune threshold 用的，没说 OOD robustness。

3. **Git coordination overhead 没形式化**。论文承认 8 agent 时 token 飙升，但没给 mitigation 方案。一个 obvious fix 是 hierarchical multi-agent（一个 "leader" agent summarize，其他 "worker" agent 只读 leader summary），把 $O(N^2)$ 降到 $O(N)$。

4. **No sim-to-real verification loop**。Stage 1 (EN) 在 real world 完成，agent 直接面对 non-determinism。如果先在 sim 上 EN，再 transfer 到 real，可能更 robust。论文 Push-T 的 sim variant 是用来 comparison 的，不是用来 bootstrap EN 的。

5. **Token economics 没考虑 cached tokens**。Appendix C.2 提到 token breakdown（input / cached input / fresh input / output / reasoning），但 cost model 没 separate。Cached tokens 通常便宜很多（[Anthropic prompt caching](https://www.anthropic.com/news/prompt-caching)），所以 raw token count 高估了 dollar cost。

---

## 12. 跟你 autoresearch 项目的对比

Karpathy 你的 [autoresearch](https://github.com/karpathy/autoresearch) 项目（references [22]）跑的是 single-GPU nanochat training，target 是 LM loss / benchmark metric。ENPIRE 是 multi-robot manipulation training，target 是 task success rate。下面是 axis-by-axis 对比：

| 维度 | Karpathy autoresearch | ENPIRE |
|------|----------------------|--------|
| Substrate | Digital (GPU) | Physical (robots + GPUs) |
| Trial cost | ~minutes (GPU hour) | ~minutes-seconds (robot + reset time) |
| Binding resource | GPU | Robot |
| Verification | LM eval harness（deterministic） | Agent-designed perception pipeline（noisy） |
| Reset | `model.reset()` instant | Procedural tool calls，几秒-几十秒 |
| Safety | None needed | Hard joint limits + force-limited gripper |
| Multi-agent | Single agent | 8 stations，Git-based P2P |
| Failure mode | Bad checkpoint，throw away | Hardware damage，irreversible |
| Transfer | Checkpoint | Markdown summary |

核心 insight：**ENPIRE 的 design choice 几乎都是从 "physical substrate 的 cost structure 不一样" 推出来的**。比如 immutable Gym API（防止 agent 同时改 env 和 policy）、force-limited gripper（无人值守必须 safe）、Stage 1 human bootstrap（physical trial 太贵不能让 agent 自己 explore env）。

这些 design 在 digital autoresearch 里都不需要，因为 digital 太便宜了，agent 可以暴力 explore。这给我一个 hypothesis：**autoresearch harness 的 design complexity 是 substrate cost 的单调函数**。未来如果 robot hardware 变便宜到 GPU 那个量级，很多 ENPIRE 的 safeguard 可以去掉。

---

## 13. Open Problems & 未来方向

基于 ENPIRE 的 limitation，我看到几个值得探索的方向：

### 13.1 Asymmetric multi-robot per agent

论文里 1 agent : 1 robot。如果改成 1 agent : K robots，agent 在 robot A rollout 时可以在 robot B 上 reset、在 robot C 上 evaluate 上一个 checkpoint。这把 MRU 推到 1 附近。这是 [SEED-RL](https://arxiv.org/abs/1910.06406) asynchronous actor 的思路。

### 13.2 Hierarchical agent team

$O(N^2)$ coordination cost 是 fleet scaling 的瓶颈。Leader-follower 结构：
- 1 leader agent：orchestrator，summarize 所有 worker branches
- N-1 worker agents：只 read leader summary，不互看

把 coordination cost 降到 $O(N)$。代价是 leader 是 single point of failure。

### 13.3 Automated reward verification 的 OOD test

Reward function 在 sandboxed eval set 上 tune，但 OOD robustness 没 guarantee。一个 fix：让 agent 同时 maintain 一个 adversarial test set（agent 故意构造 edge case 来 break reward function）。这跟 [adversarial training](https://arxiv.org/abs/1706.06083) 思路一致。

### 13.4 Sim-to-real bootstrap for EN

Stage 1 直接面对 real world，cost 高。如果先在 sim 上自动 build reward/reset（用 [Eureka](https://eureka-research.github.io/) style），再 transfer 到 real 上 fine-tune，可能更快。但 sim-real gap 在 verification function 上很大——sim 的 object detection 是 oracle，real 要 perception pipeline。所以这个方向不 trivial。

### 13.5 Weight transfer between tasks

论文只做 markdown transfer。如果 task family 相似（pin insertion → GPU insertion），policy weights 应该 transfer 更直接。但 agent 不知道怎么 transfer weights，因为 weight 是 opaque tensor。一个方向：让 agent 学会 use [LoRA](https://arxiv.org/abs/2106.09685) / [adapter](https://arxiv.org/abs/1902.00751) 这种 modular weight，方便 transfer。

### 13.6 Curriculum discovery

ENPIRE 是 single-task autoresearch。Multi-task curriculum 是 natural extension：agent 自动发现 "先学简单 task，再 transfer 到难 task" 的 sequence。这跟 [Voyager](https://voyager.on-firmament.io/) 的 automatic curriculum 类似，但在 physical substrate 上 cost 高很多。

---

## 14. 我的总评

ENPIRE 在我眼里是 **first credible attempt at physical autoresearch**。它的 contribution 是 harness + metrics，不是 algorithm。这跟你 autoresearch 项目的 contribution 性质一致——你也是 harness 而非 algorithm。

最有 lasting value 的部分我猜是：
1. **EN as immutable API abstraction**：让 credit assignment tractable
2. **MRU/MTU metrics**：给未来 physical autoresearch 一个 benchmark dimension
3. **Git-based decentralized multi-agent**：虽然 super-linear，但是 de facto scalable 的 protocol
4. **Two-stage bootstrap**：human-in-loop 一次性，amortize 到所有后续 runs

最 weak 的部分：
1. **Reward hacking 没 address**
2. **Coordination overhead 没 mitigate**
3. **Reset robustness 没 quantify**
4. **8 retries 内 success 的 metric 偏宽松**（best-of-8 ≠ 1-shot success）

最后一个观察：ENPIRE 跟 [PI0.6](https://www.physicalintelligence.company/blog/pi06)、[GR00T](https://github.com/NVIDIA/GR00T)、[OpenVLA](https://openvla.github.io/) 这类 end-to-end VLA 路线是 **互补**而非替代。VLA 提供 base capability，ENPIRE 提供 autonomous refinement loop。未来 likely 是 VLA + ENPIRE-style autoresearch 的组合，类似 LLM + tool use 的组合。

希望这个解读对你 build intuition 有帮助。如果你想深入任何一节（特别是 PD control 的实现、RL data mixing 的细节、或 Git coordination 的形式化），我可以继续展开。

### 参考链接

- ENPIRE project page: https://research.nvidia.com/labs/gear/enpire
- Karpathy autoresearch: https://github.com/karpathy/autoresearch
- Code as Policies: https://code-as-policies.github.io/
- Voyager: https://voyager.on-firmament.io/
- Eureka: https://eureka-research.github.io/
- DrEureka: https://eureka-research.github.io/dr-eureka
- CaP-X: https://arxiv.org/abs/2603.22435
- GR00T: https://github.com/NVIDIA/GR00T
- SERL: https://serl-benchmark.github.io/
- RoboCasa: https://robocasa.ai/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- AI Scientist: https://github.com/SakanaAI/AI-Scientist
- MLE-bench: https://github.com/openai/mle-bench
- SWE-bench: https://www.swebench.com/
- OpenAI Gym: https://gym.openai.com/
- Physical Intelligence (pi0/pi0.5/pi0.6): https://www.physicalintelligence.company/blog
- RLPD: https://arxiv.org/abs/2306.09311
- Residual RL (PLD-RL): https://arxiv.org/abs/2511.00091
- SEED-RL (asynchronous actor-learner): https://arxiv.org/abs/1910.06406
- Thorpe et al. 1996 (human visual system speed): https://www.nature.com/articles/381520a0
- Anthropic prompt caching: https://www.anthropic.com/news/prompt-caching
