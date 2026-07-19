---
source_pdf: AtlasVA Self-Evolving Visual Skill Memory for Teacher-Free VLM Agents.pdf
paper_sha256: b820fd463aeaecc95d4b93972551ec0fb45d57769f90a7202cdd59f23e481e51
processed_at: '2026-07-18T10:25:21-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# AtlasVA 深度解析:让 VLM agent 用 "原生视觉" 思考空间

## 0. 一句话 thesis (build your intuition)

这篇 paper 的核心立场:把 VLM agent 的 reusable experience 从 "text-only 的语言仓库" 升级为 "visually grounded 的层级记忆",用 self-evolving 的 spatial heatmaps 同时承担 **prompt-side 的 visual context** 和 **reward-side 的 potential function**,由此把 perception、memory、optimization 三个本来割裂的模块,统一到同一种 modality (RGB heatmap) 上,绕开 GPT-4o / Claude 这种 teacher LLM 的依赖。

这个设计的 elegance 在于:同一张 danger/affinity heatmap,在 forward pass 里是 VLM 的 visual token,在 backward pass 里是 PBRS (potential-based reward shaping) 的 Φ 函数。同一个数据结构服务两个计算路径,而且都是 VLM 的 native modality,不存在 lossy translation。

Homepage: https://wangpan-ustc.github.io/AtlasvaWeb/

---

## 1. Motivation:为什么 text-centric memory 在 spatial task 上崩掉

paper 提了三个具体痛点 (Figure 1):

**Pain 1 — 信息损失**: 把 2D layout 压成 1D text rule 会丢掉 topological 结构。dead end、corner trap、sub-goal region 这种几何先验,text 描述不出来。比如 Sokoban 里 "don't push box into corner" 这条 rule,对 VLM 来说只是一个 token 序列,它没法 ground 到具体像素位置的 corner 结构。

**Pain 2 — Teacher 依赖**: Reflexion、ExpeL、XSkill、SkillRL 这一套 (https://arxiv.org/abs/2303.11366, https://arxiv.org/abs/2308.10144) 都要反复 call GPT-4 去 summarize failure、merge skill、rewrite memory。计算成本高 + 不是真正 autonomous。这个痛点跟 RAG-based agent memory (Mem0: https://arxiv.org/abs/2504.19413) 的批评是一致的。

**Pain 3 — Feedback modality mismatch**: 文本 reward ("you shouldn't push the box to the corner") 是 abstract + delayed 的,RL 没法把它翻译成 dense coordinate-aligned gradient。credit assignment 问题在 long-horizon spatial task 上被放大。

**Karpathy 视角的联想**: 这三个 pain 本质上是同一个问题 —— **representation 和 gradient 必须同模态**。你在 perception 端用 pixel,在 memory 端用 text,在 reward 端用 text,这三个 representation translation step 每一步都引入 information bottleneck。这跟早期 deep learning 时代 hand-crafted feature 的问题一样 —— 你手工设计 feature,损失了 raw signal 的几何结构。AtlasVA 的做法是端到端:从 pixel observation 直接到 pixel memory 再到 pixel-shaped reward,中间没有 modality hop。

---

## 2. POMDP formulation:privileged state 的合法性边界

标准 POMDP: $\mathcal{M} := \langle S, \mathcal{O}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$

- $S$: hidden true state (unobservable)
- $\mathcal{O}$: observation space, $o_t \in \mathcal{O}$ 是 RGB frame + language task description
- $\mathcal{A}$: discrete action set
- $\mathcal{T}(s_{t+1} | s_t, a_t)$: transition dynamics
- $\mathcal{R}(s_t, a_t) \in \{0, 1\}$: sparse binary reward
- $\gamma$: discount factor
- $\pi_\theta(a_t | o_{\leq t})$: multimodal policy, parameterized by VLM

**关键设计选择 — GridState abstraction** $g_t$: 训练时从 simulator 内部 state 直接提取 2D coordinate $\mathbf{p}_t \in \mathbb{Z}^2$ (Sokoban 是 player_position + boxes_on_target;Navigation 是 AI2-THOR metadata;PrimitiveSkill 是 ManiSkill 的 _handle_info + object poses)。

这里有一个非常重要的 subtle point:**privileged access 仅用于 offline 的 atlas evolution 和 reward shaping,policy input 在 evaluation 时 100% 是 raw RGB + 渲染好的 heatmap**。这个 asymmetry 让 AtlasVA 既可以享受训练时的 dense gradient,又不破坏 zero-shot 评测的公平性。这跟 model-based RL 里 world model 的训练很像 —— 你可以训一个有 privileged access 的 world model,但 deploy 时只用它生成的 rollout 给 policy 看。

Table 4 列了具体的 simulator API access:
- Sokoban: `room_state, room_fixed, player_position, boxes_on_target`
- FrozenLake: `gym_env.s, gym_env.desc`
- Navigation: AI2-THOR metadata (agent poses, reachable positions, goal coordinates)
- PrimitiveSkill: ManiSkill `_handle_info, last_info, object poses`

---

## 3. Three-Layer Visual Skill Memory (VSM):三层互补的视觉知识梯度

augmented prompt:

$$\tilde{o}_t = [M_{danger}, M_{affinity}, \mathcal{E}_{vis}, S_{text}, o_t]$$

### Layer 1 — Spatial Heatmaps (dense perceptual priors)

两张图:
- **Danger map** $M_{danger}$: 死锁风险分布,red channel
- **Affinity map** $M_{affinity}$: 接近 task completion 的程度,green channel

rendering pipeline (Appendix C.1): 浮点 tensor $M \in [0,1]^{H \times W}$,通过 alpha-channel gradient mapping 转成 RGB。Danger → R=255, G=0, B=0,alpha 正比于 danger value。Affinity → R=0, G=255, B=0。

**关键技术细节**: heatmaps 不是 alpha-blend 在 $o_t$ 上,而是作为 **独立的 `<image>` token 注入**。作者在 Appendix 里讲了一个很有价值的 ablation insight —— 直接 alpha-blend 会 corrupt VLM 对 small foreground interactive objects 的识别。这个发现和 Visual Prompting (Shtedritski et al. 2023, https://arxiv.org/abs/2304.07999) 的结论一致:在 RGB image 上画 circle 会让 VLM attend 到那个 region,但会损失该 region 的原有细节。把 heatmap 作为 separate token,VLM 用 cross-attention 学到 "这个 spatial reference map 对应观察的哪个区域",保留了原 observation 的清晰度。

### Layer 2 — Visual Exemplars (episodic context)

从历史 rollout 里挖 "inflection point" keyframes —— irreversible deadlock 前一帧、sub-goal completion 那一帧。pool cap 是 6 (3 positive + 3 negative)。

retrieval 用 DINOv2 cosine similarity (而不是 CLIP):

$$\mathcal{E}_t = \arg\max_{e_i \in \mathcal{E}} \text{CosSim}(f_{\text{DINO}}(o_t), f_{\text{DINO}}(e_i))$$

其中 $f_{\text{DINO}}(\cdot)$ 是 frozen DINOv2 encoder。

**为什么 DINOv2 不用 CLIP**: 这是一个非常有 signal 的设计选择。CLIP 是 image-text contrastive 训练,global representation 强但 patch-level 空间对应弱。DINOv2 是 self-supervised ViT,patch token 保留了非常强的 spatial correspondence (Caron et al. 2021, https://arxiv.org/abs/2104.14294; Oquab et al. 2023, https://arxiv.org/abs/2304.07193)。在 Sokoban 这种需要 patch-level topology matching 的场景下,DINOv2 的 retrieval 显著更准。这跟 BackboneBench、Aguilar et al. 关于 dense feature 的研究是一个 intuition。

eviction 是 FIFO + DINOv2 distance 双重控制,pool 在前 40 步内填满,然后持续替换旧 frame (Figure 8 的 lifecycle visualization 非常漂亮)。

### Layer 3 — Symbolic Text Skills (semantic grounding)

这是最 thin 的一层,只保留 "General Principles / Push Strategies / Mistakes to Avoid" 这种 compact 文本。**关键 insight** —— 这些 text 不来自 GPT-4 summarize,直接来自 environment rulebook。比如 Sokoban 的 rule 直接从 gymnasium spec 抽出来,PrimitiveSkill 的 rule 从 ManiSkill task description 抽。

这一层的角色是 **cognitive scaffolding**: 文字定 logic (什么是 valid push),heatmap 和 exemplar 视觉 ground 这个 logic 在哪个 pixel 上 apply。三层形成一个 knowledge gradient:抽象 spatial map → 具体 layout example → symbolic logic。

**Karpathy 视角的联想**: 这套设计让我想到 Neural Turing Machine (Graves et al. 2014, https://arxiv.org/abs/1410.5401) 和 Sparse Distributed Memory —— memory 不是单一 modality,而是 hierarchy of representations at different abstraction levels。也像 LeCun 的 JEPA (https://openreview.net/forum?id=BZ5a1r-kVsf):不同 layer 处理不同 granularity 的 prediction。AtlasVA 的三层本质上是在做 "hierarchical abstraction of experience"。

---

## 4. Teacher-Free Atlas Evolution:从 trajectory statistics 直接 bootstrap spatial priors

这是 paper 最 hack-able 的部分。完全没有 LLM in the loop,纯统计 + EMA。

### Static heuristic branch $M_{heuristic}$

从 current layout 提取 topological features:
- BFS distance field 到 target objects (作为 affinity 的 base)
- corner-like deadlock regions (墙角、墙边、hole 邻域) (作为 danger 的 base)

### Trajectory statistics branch $M_{stat}$

batch-level danger map,从失败 trajectory 的 terminal position 累积:

$$M_{batch}^{danger}(\mathbf{p}) = \frac{1}{|\mathcal{T}_{fail}|} \sum_{\tau \in \mathcal{T}_{fail}} \mathbb{I}(\mathbf{p}_T = \mathbf{p})$$

变量解释:
- $\mathcal{T}_{fail}$: 当前 batch 中所有失败 trajectory 的集合
- $|\mathcal{T}_{fail}|$: 失败 trajectory 数量 (normalization)
- $\tau$: 单条 trajectory
- $\mathbf{p}_T$: trajectory $\tau$ 的 terminal 位置 (最终失败时的坐标)
- $\mathbb{I}(\mathbf{p}_T = \mathbf{p})$: indicator function,如果 terminal position 落在 grid cell $\mathbf{p}$ 上则为 1
- $\mathbf{p} = (x, y) \in \mathbb{Z}^2$: 2D grid coordinate

intuition: 如果某个 cell 经常成为失败终点,它很可能是 deadlock region (box 推不动的 corner、fatal hole)。

batch-level affinity map,从成功 trajectory 的 normalized visit frequency:

$$M_{batch}^{affinity}(\mathbf{p}) = \frac{1}{|\mathcal{T}_{succ}|} \sum_{\tau \in \mathcal{T}_{succ}} \sum_{t} \frac{1}{\text{length}(\tau)} \mathbb{I}(\mathbf{p}_t = \mathbf{p})$$

(Algorithm 1 line 28 的形式化) — 把成功路径上每个 visited coordinate 用 trajectory 长度做 normalization,然后跨 trajectory average。

### EMA blending (跨 batch 的时序平滑)

$$M_{stat} \leftarrow \alpha \, M_{stat} + (1 - \alpha) \, M_{batch}$$

变量:
- $\alpha \in [0, 1]$: EMA decay rate,论文用 0.85
- $M_{stat}$: 累积的历史统计 map (state variable,跨 batch 持久)
- $M_{batch}$: 当前 batch 的 fresh statistic

$\alpha = 0.85$ 意味着每 batch 保留 85% 的历史 + 15% 的新数据,大约对应 effective window ~6-7 个 batch (1/(1-α))。这个值选得不极端,既不会太快 forget 早期信号,也不会太慢 adapt 新策略产生的 trajectory。

### Schedule-based blend (heuristic vs experience)

$$M_{final} = (1 - \beta_k) M_{heuristic} + \beta_k M_{stat}$$

变量:
- $\beta_k \in [0, 1]$: scheduling coefficient,在训练 epoch $k$ 上从 0 anneal 到 1
- $M_{heuristic}$: static layout-based prior (BFS field, corner detection)
- $M_{stat}$: EMA-accumulated trajectory statistic
- $M_{final}$: 最终输出给 VLM 和 reward shaping 的 heatmap

intuition: 早期 (k 小) policy 还很烂,trajectory statistics 噪声大,主要 trust 静态 heuristic (BFS field 总是对的)。后期 (k 大) policy 改善,trajectory 信号变可信,平滑切换到 data-driven 之上。这个 annealing 跟 curriculum learning、warmup-with-fixed-target 的思路一脉相承。

**Karpathy 视角的联想**: 这套机制让我想到 DAgger (Dataset Aggregation, Ross et al. 2011, https://arxiv.org/abs/1011.0686) 的迭代更新,也像 AlphaZero 的 self-play bootstrapping —— policy 产生 data,data 改进 policy,policy 又产生更好的 data。AtlasVA 把这个 loop 落地到 VLM agent 上,但避开了 AlphaZero 那种 expensive MCTS,用 cheap 的 grid statistic + EMA 替代。这种 "bootstrap from own experience" 的设计在 RLHF、Constitutional AI 里都有 echo。

---

## 5. Atlas-Grounded Dense Visual Reward Shaping:把 heatmap 变成 potential function

这部分是数学上最 clean 的 contribution。把 sparse $\{0, 1\}$ reward 转成 dense per-step signal。

定义 augmented reward:

$$\tilde{r}_t = r_t^{env} + F(o_t, a_t, o_{t+1})$$

shaping term:

$$F(o_t, a_t, o_{t+1}) = \underbrace{\left[\Phi_{affinity}(o_{t+1}) - \Phi_{affinity}(o_t)\right]}_{\text{potential-based, policy-invariant}} - \underbrace{\beta \cdot \mathbb{I}(\text{enters\_danger}(o_{t+1}))}_{\text{heuristic safety constraint}}$$

变量:
- $r_t^{env}$: 环境 sparse reward, $\in \{0, 1\}$
- $F$: shaping term
- $\Phi_{affinity}(o_t)$: affinity potential function,evaluate 当前 position 在 affinity heatmap 上的值
- $\beta$: danger penalty scaling (Sokoban 0.05, PrimitiveSkill 0.3)
- $\mathbb{I}(\text{enters\_danger}(o_{t+1}))$: indicator,下一状态进入 danger region 则为 1

### 为什么 affinity 是 potential-based,danger 不是

这是 paper 一个非常 deliberate 的设计选择,跟 Ng et al. 1999 (https://www-cse.ucsd.edu/~gary/PAPER-SUGGESTIONS/ng-russell-shaping-icml1999.pdf) 的 PBRS theory 紧密对齐。

**PBRS 的核心定理**: 如果 shaping term 形式是 $F(s, s') = \gamma \Phi(s') - \Phi(s)$,那么 optimal policy 不变 — 只是 value function 偏移了 $\Phi(s)$。affinity 用 potential difference 形式 $\Phi(o_{t+1}) - \Phi(o_t)$,严格保持 policy invariance。agent 不会为了 "刷 affinity" 而走 shortcut,因为每个 trajectory 的 total shaping reward telescoping 到 $\Phi(s_T) - \Phi(s_0)$,只跟起点终点有关,跟路径无关。

**Danger 故意 break potential-based 形式**: $-\beta \cdot \mathbb{I}(\text{enters\_danger})$ 不是 potential difference,是 state-based penalty。这 **intentionally** 改变 optimal policy — 让 agent 偏好 safer trajectory,即使 hazardous shortcut 更短。这就像 constrained MDP (CMDP, Altman 1999) 的 Lagrangian 处理:你接受 suboptimal in reward 来换取 safety。

intuition: 如果 danger 也用 potential difference 形式,$F_{danger} = \Phi_{danger}(s') - \Phi_{danger}(s)$,那 agent 走过 danger 后离开时还会得到 positive reward,这违背 safety 的初衷。state-based penalty 保证 "进入 danger 一定扣分,离开不补分",把 danger region 变成一类的 absorbing negative basin。

### Affinity gain 的具体形式

$$r_{affinity} = M_{final}^{affinity}(\mathbf{p}_{t+1}) - M_{final}^{affinity}(\mathbf{p}_t)$$

因为 $M_{final}^{affinity}$ 继承了 $M_{heuristic}^{affinity}$ 的 BFS distance gradient,agent 每往 goal 走一步,affinity 值增加一个固定量,$r_{affinity} > 0$。每远离一步,$r_{affinity} < 0$。这给了一个非常稳定的 " compass signal ",credit assignment 从 terminal 拉回到每一步。

最终 reward: $r_{visual} = \lambda_{danger} \cdot r_{danger} + \lambda_{affinity} \cdot r_{affinity}$,然后 $\tilde{r}_t = r_t^{env} + r_{visual}$。

### Reward scale per environment (Table 3)

| Environment | Success | Failure | λ_danger | λ_affinity |
|---|---|---|---|---|
| Sokoban | +1.0 | -0.1 | 0.05 | 0.05 |
| FrozenLake | +1.0 | -0.1 | 0.05 | 0.05 |
| Navigation | +1.0 | -0.1 | 0.1 | 0.1 |
| PrimitiveSkill | +1.0 | 0.0 | 0.3 | 0.3 |

intuition: PrimitiveSkill 的 λ 大 (0.3),因为它需要 high-precision long-horizon coordinate manipulation,strong visual guidance 收益大。Sokoban 的 λ 小 (0.05),因为 grid-world 已经 discrete,policy 在每个 cell 的 action space 小,过强 shaping 会 over-constrain。

**Karpathy 视角的联想**: 这套 reward shaping 跟 DeepMind 的 Retro (https://arxiv.org/abs/2204.13046) 在精神上很像 —— 都用 trajectory-derived 的 prior 来 shape learning。也像 Hindsight Experience Replay (HER, Andrychowicz et al. 2017, https://arxiv.org/abs/1707.01495) 把 sparse reward 转成 dense 的思路,但 AtlasVA 用 spatial potential 而不是 goal relabeling。也跟 AlphaGo 的 reward shaping 有 echo —— 用 value network 当 potential,把 terminal reward 转 dense per-move。

---

## 6. Optimization loop & closed-loop dynamics

Algorithm 1 (Appendix A.4) 把整个流程形式化了。每个 PPO epoch 内:

**Phase 1 — Atlas-Grounded Visual Reward Shaping (online)**
1. 对 buffer $B$ 里每个 transition $(\mathbf{p}_t, a_t, \mathbf{p}_{t+1})$:
   - $\Phi_{affinity} \leftarrow M_{affinity}(\mathbf{p}_{t+1})$
   - $r_{affinity} \leftarrow \Phi_{affinity} - M_{affinity}(\mathbf{p}_t)$
   - $r_{danger} \leftarrow -\beta \cdot M_{danger}(\mathbf{p}_{t+1})$
   - $\tilde{r}_t \leftarrow r_t^{env} + r_{affinity} + r_{danger}$
2. Standard PPO update using $\tilde{r}_t$

**Phase 2 — Teacher-Free Atlas Evolution (offline)**
1. 从 $B$ 抽 $\mathcal{T}_{fail}$ 和 $\mathcal{T}_{succ}$
2. 更新 $M_{danger}$ via terminal position accumulation + EMA
3. 更新 $M_{affinity}$ via successful path visitation + EMA
4. 更新 exemplar pool via DINOv2 matching + FIFO eviction

off-policy evolution 形式化:

$$\mathcal{M}_{k+1}^{(heatmap)} \leftarrow \alpha \, \mathcal{M}_k^{(heatmap)} + (1 - \alpha) \cdot \text{Extract}(\tau_k)$$

其中 $\text{Extract}(\cdot)$ 是 non-parametric mapping (聚合 terminal failure 到 danger,聚合 successful visitation 到 affinity)。

**Closed loop 的关键**: policy π 改进 → 生成更高质量 trajectory τ → 更新 M → 提供更准 Φ 和 $\tilde{o}_t$ → 进一步改进 π。这是 self-bootstrapping cycle,完全 autonomous。

PPO 超参 (Table 5):
- Actor lr: $1 \times 10^{-6}$
- Critic lr: $1 \times 10^{-5}$ (critic 学得快 10×,因为要快速适应变化的 reward landscape)
- Batch size: 128 (rollout), 32 (mini-batch)
- KL control coef: 0.0 (完全 trust PPO clip,不加 KL soft constraint)
- 8 × RTX 6000 Ada GPUs
- vLLM async rollout, max prompt 5000-8000 tokens, max response 4000

---

## 7. Experimental results:3B 击败 GPT-5 的 breakdown

Table 2 是 main result。AtlasVA overall 0.93,大幅领先所有 baseline。

### vs Proprietary models
- GPT-5: 0.69
- o3: 0.71
- o4-mini: 0.60
- GPT-4o: 0.60
- Gemini 2.5 Pro: 0.51
- Claude Sonnet 4.5: 0.62

3B 参数的 AtlasVA 比 GPT-5 高 24 个点。这个 gap 主要来自 **spatial-intensive task 的本质**:这些 task 不是 reasoning 难,是 grounding 难 —— 大模型有再多的知识,如果没法 ground 到 pixel-level geometric structure,在 Sokoban 上还是不会做。这跟 BALROG benchmark (Paglieri et al. 2024, https://arxiv.org/abs/2411.13543) 的发现一致 —— 大模型在 game reasoning 上意外地差。

### vs Open-source models
- Qwen2.5-VL-72B: 0.55 (比 3B 强,但还是远不如 AtlasVA)
- Qwen2.5-VL-7B: 0.19
- Qwen2.5-VL-3B (base): 0.09
- VLM-R1-3B: 0.10
- VAGEN: 0.78 (前 SOTA)
- AtlasVA: 0.93

zero-shot Qwen2.5-VL-3B 在 Sokoban 上只有 0.14,AtlasVA 提升到 0.79 —— 一个数量级的提升。这个 gain 几乎完全来自 VSM 注入,因为 base model 一模一样。

### PrimitiveSkill 全 5 个 sub-task 都到 1.00

Place / Stack / Drawer / Align / Swap 全部 100%。Swap 是这篇 paper 新引入的 task (要求 agent 交换两个 cube 位置),Qwen2.5-VL-72B 在 Swap 上只有 0.33,GPT-5 只有 0.55。这非常 impressive,因为 Swap 需要 long-horizon multi-step manipulation (pick A → place buffer → pick B → place target A → pick buffer → place target B),典型 credit assignment hard case。

### Ablation studies (Figure 5)

- **w/o VSM**: Sokoban / FrozenLake 大幅退化,确认 visual grounding 不可替代
- **w/o Heatmap**: 退化,证明 Layer 1 是核心
- **w/o Exemplar**: 退化,证明 Layer 2 也有独立贡献
- **w/o Atlas Evolution**: 退化到 static rules,证明 trajectory-driven bootstrapping 是关键
- **w/o Dense Reward**: 卡在 local optima,证明 shaping 必要

### Learning efficiency (Figure 4)

text-only baseline (只有 Layer 3) 在 Sokoban 上 140 步内最高 0.25。AtlasVA 在 140 步内到 0.80。PrimitiveSkill 上 AtlasVA 快速收敛到 1.00,baseline 卡在 0.60。这个 efficiency gain 主要来自 dense visual gradient 解决 credit assignment。

### Heatmap evolution (Figure 6)

Step 0: heatmap 全空 (没信息)
Step 200: Danger map (top row) 明确 highlight structural hazards 和 dead-ends
Step 400: Affinity map (bottom row) trace 出 traversable sub-goal paths

这个过程视觉上非常 satisfying — 看着 agent 通过纯环境交互 "学到" 空间结构,完全不需要 teacher。

---

## 8. 3D → 2.5D projection (Appendix D.1)

paper 在 3D Navigation 和 PrimitiveSkill 上怎么处理 3D continuous space?

**Navigation**: 把 3D room 的 reachable surface 投影到 X-Z 平面,discretize 成 2D floor plan,0.25 meters/cell。Y 轴 (height) 在 navigation 任务里 mostly irrelevant (agent 在 floor 上走)。

**PrimitiveSkill**: tabletop workspace 投到 localized 2.5D grid,Z 轴 (height) 信息保留为 task metadata。Reward shaping 时,agent 的 continuous 3D coordinate 反向映射到 2.5D heatmap,lookup potential 值。

**Limitation (Section 5)**: 作者明确承认,这种 2.5D projection 在 highly occluded、ego-centric 3D robotics 上会失效。比如 robot 第一人称视角,depth occlusion 严重,2.5D 投影会丢掉很多 free space 信息。Future work 需要真正的 3D volumetric memory,可能要借 voxel grid 或 neural radiance field 的思路。

**Karpathy 视角的联想**: 这个 2.5D abstraction 让我想到经典 robotics 的 occupancy grid mapping (Thrun et al. 2005 的 Probabilistic Robotics) 和 2.5D heightfield in classical motion planning。也像 NeRF (https://arxiv.org/abs/2003.08934) 之前的 2.5D depth map rendering。3D robotics 真正的突破可能需要把 AtlasVA 的思路推广到 3D voxel (像 VoxFormer, https://arxiv.org/abs/2302.12251) 或 implicit neural field。

---

## 9. Prompt template (Appendix B, Figure 10)

```
[System] Sokoban rules and output format ...
## Spatial Skill Maps
Danger zones (red): <image>
Goal affinity (green): <image>
## Visual Exemplars (optional, when pool is non-empty)
Positive cases: <image>, <image>, ...
Negative cases: <image>, <image>, ...
## Learned Principles
### General Principles / Push Strategies / Mistakes to Avoid ...
[User] [Initial Observation]: <image>
Decide your next action(s).
```

这个 prompt 结构非常 careful:三层 VSM 在固定 anchor 位置注入 (## Spatial Skill Maps, ## Visual Exemplars, ## Learned Principles),current observation 放最后。这种 ordering 让 VLM 先看 abstract spatial context,再看 episodic exemplars,再看 symbolic rules,最后看 current state —— 模仿人类先看地图、再看类似经验、再看规则、最后看现状的决策顺序。

---

## 10. 跟相关工作的 positioning

### vs Reflexion (Shinn et al. 2023, https://arxiv.org/abs/2303.11366)
Reflexion 用 verbal self-critique 修 plan。AtlasVA 用 visual heatmap 替代 verbal critique,直接 ground 到 spatial coordinate。

### vs ExpeL (Zhao et al. 2024, https://arxiv.org/abs/2308.10144)
ExpeL distill trajectory 到 linguistic record。AtlasVA distill 到 spatial map,避开 language bottleneck。

### vs XSkill (https://arxiv.org/abs/2603.12056)
XSkill 用 Markdown/JSON 存 skill library。AtlasVA 用 RGB heatmap + RGB exemplar + minimal text。

### vs VAGEN (Wang et al. 2025, https://arxiv.org/abs/2510.16907)
VAGEN 是前 SOTA,用 world model reasoning + multi-turn VLM training。AtlasVA 在它基础上加了 VSM 和 dense visual shaping。AtlasVA 在 PrimitiveSkill 上的 Swap task 是 VAGEN 没评测的。

### vs RL-VLM-F (Wang et al. 2024, https://arxiv.org/abs/2402.03681)
RL-VLM-F 用 VLM 当 reward model 给 RL 提供反馈。AtlasVA 反过来 — 用 spatial statistics 给 VLM 提供训练 signal,避开了 VLM-as-reward-model 的 cost。

### vs SpatialRGPT (Cheng et al. 2024, https://arxiv.org/2406.15810)
SpatialRGPT 在 VLM 内做 grounded spatial reasoning。AtlasVA 在 VLM 外挂 spatial memory + reward shaping,是 agent-level 而非 model-level 的 solution。

### vs Thinking in Space (Yang et al. 2025, https://arxiv.org/2412.14171)
这篇是 CVPR 2025 paper,研究 VLM 如何 see、remember、recall space。AtlasVA 可以看作是这套 spatial cognition 的 operationalization。

---

## 11. Critical thoughts & limitations

### Strengths
1. **Same data structure, dual purpose**: heatmap 同时是 prompt 和 potential function。这个 elegance 在 RL+VLM 文献里很少见。
2. **Strict teacher-free**: 完全不依赖 GPT-4 / Claude,真正 autonomous。
3. **PBRS theory aligned**: affinity 严格 potential-based,保持 policy invariance。
4. **Strong empirical gain**: 3B 击败 GPT-5,数据 convincing。

### Concerns
1. **GridState privileged access**: 虽然只在 training 用,但 evaluator 可能质疑这破坏了 "纯 visual" 的 claim。不过作者在 Appendix C.5 明确说了 evaluation 时 VLM 只看 raw RGB + rendered heatmap,这个 demarcation 是清晰的。
2. **2.5D projection 的 scalability**: 高度 occluded 的 ego-centric 3D 会崩。作者承认。
3. **Generalization to unseen layout**: paper 说 zero-shot transfer 时主要靠 $M_{heuristic}$ 分支 (BFS field、corner detection 在新 layout 上 on-the-fly 算)。但 $M_{stat}$ 分支是 layout-specific 的,新 layout 上 stat 是空的,只能 rely heuristic。这意味着 generalization 上限是 heuristic 的 quality。
4. **Exemplar pool size 6 可能太小**: 对于 state space 大的任务 (比如 Navigation 的 multiple room),6 个 exemplar 可能不够 cover。作者没做 pool size 的 ablation。
5. **DINOv2 retrieval 是 frozen 的**: 不随 task adapt。如果 task 有 specific visual signature (比如特定颜色 cube),DINOv2 的 general feature 可能 suboptimal。

### Open questions for future work
1. AtlasVA 能不能 extend 到 3D voxel memory? 用 neural radiance field 当 potential function?
2. Heatmap 能不能 learn to render 而不是 hand-crafted colormap? 比如让 VLM 自己 learn 一个 heatmap decoder。
3. Affinity 和 danger 之外,能不能加更多 semantic channel (比如 "manipulable object"、"obstacle")?
4. EMA decay α=0.85 是 hyperparameter,能不能做成 adaptive 的?比如基于 trajectory quality 自动调整。
5. Exemplar retrieval 用 DINOv2,能不能用 VLM 自己的 visual encoder 做 retrieval?这样 retrieval 和 policy 用同一个 representation space。

---

## 12. 我的整体 take

这篇 paper 的核心 contribution 在我看来是 **modality alignment 的端到端化**。从 perception 到 memory 到 reward,全部用 VLM 的 native visual modality,没有 modality hop。这个设计 philosophy 跟 deep learning 的 end-to-end 训练精神一致 —— 你不要手工设计 intermediate representation,让整个 pipeline 用同一种 signal 训练。

3B 击败 GPT-5 的 result 一开始听起来 surprising,但仔细想是 inevitable 的 —— GPT-5 在 Sokoban 上再强,它的 spatial reasoning 都是 post-hoc 的 (从 image 推理到 text 再到 action)。AtlasVA 直接把 spatial prior 注入 visual context,跳过了 image → text → action 这个 lossy chain。这跟 Retrieval-Augmented Generation 在 NLP 上的成功是一个道理 —— 你不要让 model 把所有东西记住,你给它 external memory 直接 retrieve 进 context。

paper 最 valuable 的 technical insight 我觉得有三条:
1. **Heatmap 作为独立 image token 而非 alpha-blend** —— 这个 ablation finding 对整个 visual prompting 社区都有价值。
2. **Affinity potential-based + Danger non-potential-based 的混合** —— 这是对 PBRS theory 的精妙运用,既保持 policy invariance 又能 enforce safety constraint。
3. **Heuristic vs statistics 的 schedule blend** —— 解决 cold-start 问题,这个思路在任何 self-bootstrapping 系统里都适用。

Long-term 我觉得这个方向会跟 world model、model-based RL、active inference 走到一起 —— memory 不只是 retrieval 的 database,而是 agent 内化的 spatial/world model,既用于 planning 又用于 reward shaping。AtlasVA 是这条路上的一个 important milestone。

---

**Reference links**:
- AtlasVA homepage: https://wangpan-ustc.github.io/AtlasvaWeb/
- PBRS original paper (Ng et al. 1999): https://www-cse.ucsd.edu/~gary/PAPER-SUGGESTIONS/ng-russell-shaping-icml1999.pdf
- Reflexion: https://arxiv.org/abs/2303.11366
- ExpeL: https://arxiv.org/abs/2308.10144
- VAGEN: https://arxiv.org/abs/2510.16907
- DINOv2: https://arxiv.org/abs/2304.07193
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- ManiSkill3: https://arxiv.org/abs/2410.00425
- AI2-THOR: https://arxiv.org/abs/1712.05474
- BALROG: https://arxiv.org/abs/2411.13543
- Gymnasium: https://arxiv.org/abs/2407.17032
- SpatialRGPT: https://arxiv.org/abs/2406.15810
- Thinking in Space: https://arxiv.org/abs/2412.14171
- HER: https://arxiv.org/abs/1707.01495
- DAgger: https://arxiv.org/abs/1011.0686
- Visual Prompting: https://arxiv.org/abs/2304.07999
