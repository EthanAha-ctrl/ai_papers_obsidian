---
source_pdf: World-in-World World Models in a Closed-Loop World.pdf
paper_sha256: a862f4a4228619b49f484bade4ce25693aa2549dc056971fc001615aaff504df
processed_at: '2026-08-13T05:15:25-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好, 我换个频道, 用大白话跟你聊这篇 paper。

---

## 一句话总结

大家都在秀 video generator 出的片子多漂亮, 这篇 paper 说: **别看广告, 看疗效**。你把 Sora、Wan 这些模型塞进一个 robot 的决策 loop 里, 它能不能帮 robot 把活干了? 答案是: 看起来漂亮的 video 跟任务成功率几乎没关系, 真正管用的是 **你能不能精确控制这个模型按你说的 action 来生成**。

---

## 这篇 paper 在怼谁

怼的是整个 world model community 的评估方式。

现在大家发 paper 的套路是: 我训了个 video model, 在 VBench 上跑几个分数, temporal consistency 多少, aesthetic 多少, 然后说"看, 这就是 world simulator"。但问题来了 — Sora 出的片子确实震撼, 你让它生成"机器人往前推方块"的 video, 它给你一个很漂亮的推方块画面, 但方块的运动轨迹跟你的 action command 完全不对应。这种 model 你敢拿来做 planning 吗? 不敢, 因为它 rollout 出来的 future 是 hallucination, 你照着它走会撞墙。

所以这篇 paper 说: 咱们别再比 video 好不好看了, 咱们比 **agent 用了你的 model 之后任务成不成**。这就叫 closed-loop evaluation — 你的 model 在 loop 里, 真实环境也在 loop 里, agent 用你的 model 做 look-ahead planning, 然后在真实环境里执行, 再看新 observation, 再 plan, 循环往复。成不成, 用 success rate 说话。

---

## 他们做了什么

造了一个 benchmark, 叫 World-in-World, 四个 task:

1. **Active Recognition** — 房间里有个东西被挡住了, agent 要走过去找个好角度认出来
2. **ImageNav** — 给你一张目标照片, agent 要走到那个位置
3. **A-EQA** — 问你"厨房台面上有几个杯子", agent 要自己走过去数
4. **Robotic Manipulation** — 控制机械臂推按钮、滑方块、插钉子、叠杯子

每个 task 里, agent 的策略都一样: 
- 先想几条候选行动方案
- 让 world model 把每条方案"演"一遍 (生成 future video)
- 看哪条演出来效果最好, 就执行那条
- 执行完看新画面, 重新来

这个就是 **proposal → simulate → revise → execute** 的循环。本质上是把 world model 当成 agent 脑子里的"想象器", 让它在行动前先在脑子里彩排一下。

---

## 他们测了哪些 model

十几个, 大致分两类:

**Zero-shot 派** — 直接拿现成 video generator 用, 不给它看 embodied 数据:
- Wan2.1, Wan2.2, Hunyuan, LTX-Video, Cosmos-Predict2 — 这些吃 text prompt
- SVD — 吃 image
- PathDreamer, SE3DS — 吃 viewpoint / camera trajectory
- NWM — 吃 trajectory
- Runway Gen4 — proprietary, 只在 AR 上测了

**Post-trained 派** — 拿上面几个 model, 用 40K 条 Habitat 里采的 (observation, action) pair 做 fine-tune, 让它学会"给定 action, 生成对应 future"。加个 † 标记。

post-training 的成本: SVD 29 个 H100 小时, Wan2.1 74 个 H100 小时。跟 pretrain 一个 video model (动辄几万 H100 小时) 比, 是九牛一毛。

---

## 三个核心发现

### 发现 1: 好看没用, 好控才有用

把所有 model 的 visual quality 分数 (aesthetic + image quality) 跟 task success rate 画散点图, **没有相关性**。一个生成出来的视频可以美得像电影, 但 task 成功率跟抛硬币似的。

但如果你把 controllability (就是 model 生成的画面跟 ground-truth 画面的 LPIPS 距离的倒数) 跟 success rate 画, **正相关**。意思就是: 你的 model 能不能听话, 按 action 给的动作来生成画面, 这才决定它对 agent 有没有用。

Post-training 之所以有效, 就是因为它把 text-prompt 控制换成了 action 控制, controllability 大幅上升, SR 跟着涨。

**人话翻译**: 你让 text-to-video model 演"机器人往前走", 它可能给你一个镜头横摇的画面, 因为网上这种视频多, 它的 prior 觉得这样好看。但你的 action 是 "forward 0.2m", 镜头应该直着往前推。这个错配就是 text conditioning 的硬伤。Action-conditioned post-training 就是来修这个的。

### 发现 2: 花小钱 post-train 比花大钱 pretrain 划算

Figure 6 是数据 scaling 曲线:
- Wan2.1† (14B) 从 400 条数据到 80K 条, AR SR 从 60.25% 涨到 63.34%, 还没饱和
- SVD† (1.5B) 涨到 40K 条就饱和了 (56.80% → 60.98%)
- Wan2.2† A14B 是更大的 base model, web pretraining 比 Wan2.1 多不少, 但 zero-shot 时并不比 Wan2.1 强, 直到 post-training 40K 条数据才追上

**人话翻译**: 你想把一个 video generator 变成 embodied agent 的 brain, 与其无脑堆 pretrain compute (让它看更多网上的视频), 不如花 1% 的钱搞一批 action-observation pair 做 fine-tune。这跟 LLM 领域的故事一模一样 — base model 看再多网页, 也不如 instruction tuning 来得管用, 因为 instruction tuning 把模型跟"任务接口"对齐了。Action-conditioned post-training 就是 embodied 版的 instruction tuning。

### 发现 3: Inference 时多算一会儿也管用

agent 每一步多 generate 几条 candidate plan (从 3 条到 11 条), SR 从 53.36% 涨到 60.98%。这就是 test-time compute scaling, 跟 o1 在 reasoning 上多想一会儿是同一个道理, 只不过这里是在 visual + action space 上做。

**人话翻译**: 你的 world model 不用变, 只要 agent 每次决策前多模拟几条路线, 比一比再走, 成功率就上去了。这就是为什么 model-based planning 比纯 policy 好 — model 让你有了"后悔药", 可以在脑子里试错。

---

## 哪些 task WM 帮助大, 哪些帮助小

帮助最大: **AR** 和 **ImageNav** — 因为这些 task 主要是 viewpoint 变化, world model 只要会"挪视角生成新画面"就够了, 这个相对简单。

帮助中等: **A-EQA** — 因为问题是 open-ended, agent 要 long-horizon 探索, world model 的"想象"在长 horizon 上衰减快。

帮助最小: **Manipulation** — 因为机械臂跟物体接触的物理太复杂了, compliance、friction、articulated joint 这些, video model 根本学不到。最好的 SVD† 也只从 44.5% 提到 46.5%。**但是** — 当 base policy 很弱时 (3D Diffuser Actor, 24% SR), 加 WM 跳到 44.7%, **涨了 20 个点**。所以 WM 对弱 policy 的帮助更大, 这跟 model-based RL 的经典结论一致: 弱 policy + 好 model > 强 policy alone, 尤其在 long horizon 任务上。

---

## 我觉得这篇 paper 的真正价值

不是那些数字, 而是它给 community 划了一条线:

> 你的 world model 要 claim "world simulator", 就别只发 video clip 了, 把它接到这个 benchmark 上跑一遍 closed-loop success rate, 亮出来。漂亮不等于有用, 有用不等于漂亮。

这跟当年 ImageNet 之于 image classification 是一样的角色 — ImageNet 之前大家比手动设计 feature, ImageNet 之后大家比 end-to-end 在真实世界的泛化。World-in-World 可能就是 world model 的那个 ImageNet moment。

---

## 最值得记住的三条

1. **Controllability > Visual quality** — 你的 model 能不能按 action 来生成, 比生成得漂不漂亮重要 10 倍
2. **Post-training data scaling 比 pretraining scaling 划算** — 40K 条 action-observation pair, 74 个 H100 小时, 就能让 14B model 在 embodied task 上脱胎换骨
3. **Inference-time compute scaling 在 embodied 上也成立** — agent 多想几条路再走, 成功率稳步上升

如果这三条记住了, 这篇 paper 的精华你就拿到了。

---

# World-in-World: 把 World Model 放进 Closed-Loop 里去烤一烤

Andrej, 这篇 paper 我读下来觉得它其实是在戳破 community 的一个 illusion —— 大家看到 Sora、Wan、Hunyuan 这些 video generator 出的 clip 越来越漂亮, 就默认它们已经接近 "world simulator" 了。但真正的问题是: 当你把这些模型插到一个 embodied agent 的 loop 里, 让它用 rollout 来做 planning, 它到底 deliver 不 deliver? 这篇 paper 给的答案是 — **visual quality 几乎不预测 task success, controllability 才预测; 而 action-conditioned post-training 的 data scaling 比升级 base generator 更划算; inference-time compute scaling 也能显著抬升 closed-loop 性能**。这三条 finding 我觉得对下一波 world model 的设计方向是有 steer 作用的。

项目主页: https://world-in-world.github.io  
Code (承诺开源): https://github.com/World-In-World

---

## 1. 为什么这个 benchmark 必须存在

先看 motivation。目前的 WM benchmark 分几类:
- **VBench** (https://github.com/Vchistar/VBench) — 纯 video quality, temporal consistency、subject consistency 这些
- **WorldModelBench** (Li et al. 2025a, https://arxiv.org/abs/2502.20694) — 判 video gen 是不是 "world model", 但还是 open-loop visual plausibility
- **WorldScore** (Duan et al. 2025, https://arxiv.org/abs/2504.00983) — image + camera trajectory 输入, 评估 spatial/temoral/physics consistency, 但仍然 open-loop

没有一个 benchmark 让 WM 真正进入 agent-environment 的 closed loop, 去做 perception → plan → execute → replan。这就是 World-in-World 要填的 gap。它把 "world model 有没有用" 从 "video 漂不漂亮" 切换到 "agent 任务成功率", 这其实跟你在 Tesla AI Day 讲的 "世界模型用于规划" 的精神是一致的, 但更系统化。

---

## 2. Closed-Loop Online Planning: 一个 policy-guided beam search

这是整个 framework 的核心。Figure 3 里画的循环是: proposal → simulation → revision → execute → 下一个 observation。

### 2.1 形式化

设 agent 在时间步 $t$ 的 egocentric observation 是 $\mathbf{o}_t$。定义 horizon $L$ 的 future action sequence 为:

$$\hat{\mathbf{A}}_t = [\hat{a}_{t+1}, \hat{a}_{t+2}, \dots, \hat{a}_{t+L}]$$

其中每个 elementary action $\hat{a} \in \mathcal{V}$, $\mathcal{V}$ 是 action primitive set (continuous 或 discrete)。

**Proposal step** — proposal policy $\pi_{\mathrm{proposal}}$ 采样 $M$ 个候选 action sequences:

$$\hat{\mathbf{A}}_t^{(m)} \sim \pi_{\mathrm{proposal}}(\mathbf{A} \mid \mathbf{o}_t, g), \quad m = 1, \ldots, M$$

这里 $g$ 是 task goal, $M$ 是 beam width。$\pi_{\mathrm{proposal}}$ 可以是 VLM、diffusion policy、甚至 rule-based heuristic —— paper 里三种都试了。

**Action API transformation** — 因为不同 WM 接受不同的 control interface, 所以需要一个统一映射:

$$I_t^{(m)} = C(\hat{\mathbf{A}}_t^{(m)})$$

$I_t^{(m)}$ 可以是 text prompt、camera trajectory $[(x_k, y_k, \phi_k)]_{k=1}^K$, 或者 low-level action sequence $\mathbf{A}_{\mathrm{world}}$。这是让 heterogeneous WM 能 plug-and-play 的关键。

**Simulation step** — world model $g_\theta$ 做 counterfactual rollout:

$$\hat{\mathbf{O}}_t^{(m)} \sim g_\theta\!\left(\mathbf{O} \mid \mathbf{o}_t, I_t^{(m)}\right), \quad \hat{\mathbf{O}}_t^{(m)} = [\hat{\mathbf{o}}_{t+1}^{(m)}, \hat{\mathbf{o}}_{t+2}^{(m)}, \dots, \hat{\mathbf{o}}_{t+L}^{(m)}]$$

注意 $\mathbf{o}_t$ 是 **真实** observation, 不是 model 自己生成的; 这是 anchor, 防止 rollout 漂走。这跟 Dreamer / TD-MPC2 (https://github.com/nicklashansen/dreamerv3) 里的 latent imagination 有差别, 这里 imagination 是在 **pixel space** 做的, 没有压到 latent —— 因为他们用的是现成的 video generator, 不一定要为它配套 encoder/decoder。

**Revision step** — revision policy $\pi_{\mathrm{revision}}$ 综合所有候选 plan 和它们的 rollout, 输出最终决策:

$$\mathbf{D}_t^* = \pi_{\mathrm{revision}}\!\left(\{(\hat{\mathbf{A}}_t^{(m)}, \hat{\mathbf{O}}_t^{(m)})\}_{m=1}^M, \mathbf{o}_t, \mathbf{g}\right)$$

一个常见 instantiation 是 score-and-select:

$$m^* = \arg\max_{m \in \{1,\dots,M\}} S\!\left(\hat{\mathbf{A}}_t^{(m)}, \hat{\mathbf{O}}_t^{(m)} \mid \mathbf{o}_t, \mathbf{g}\right), \quad \mathbf{D}_t^* = \hat{\mathbf{A}}_t^{(m^*)}$$

$S(\cdot)$ 是 task-specific scoring function, 可以是 VLM 做 reward model, 也可以是 heuristic。这一步比经典 MPC (Morari & Lee 1999, https://www.sciencedirect.com/science/article/pii/S0098135498002431) 更 general —— MPC 只在 action sequence space 优化, 这里 $\mathbf{D}_t^*$ 可以是 high-level answer (AR/A-EQA)、recognition result、或 refined action sequence。

### 2.2 这个框架跟邻近工作的关系

- **Dyna-style MBRL** (Sutton 1991): 早期 model-based RL 也是用 model 做 rollout 再 update policy。区别是这里 model 是 pixel-level video generator, policy 是 VLM, 不做 gradient update, 是 inference-time planning。
- **Video Language Planning** (Du et al. 2024, https://arxiv.org/abs/2310.02535): 类似 idea, 但没标准化 action interface, 没 closed-loop benchmark。
- **Generative World Explorer** (Lu et al. 2025, https://arxiv.org/abs/2506.19565): 他们自己组里的前作, 也是 mental exploration 思路, 这篇把它扩展到多 task 多 model。
- **VLM as reward model** (Rocamonde et al. 2023, https://arxiv.org/abs/2310.12921): revision policy 的 score function 本质上是这个思路。

---

## 3. Unified Action API: 让任何 WM 都能被 plug 进来

我觉得这是这篇 paper 最容易被人忽略但工程上最重要的部分。Table 7 里列了 12 个 WM, 控制接口五花八门:
- **Text prompt**: LTX-Video, Hunyuan, Wan2.1/2.2, Cosmos-Predict2, Runway Gen4
- **Camera trajectory / viewpoint**: PathDreamer, SE3DS, NWM
- **Image conditioning** (无显式 action): SVD
- **Action-conditioned (post-trained)**: SVD†, Wan2.1†, Wan2.2†, Cosmos-Predict2†, LTX-Video†

把 text-prompt 控制的 WM 用到 embodied task 上有个尴尬: 你想把 "前进 0.2m 然后左转 22.5°" 这个 action 喂给 Wan2.1, 它原生只吃 text。Action API 就用一个 template 把 primitive action 翻译成 phrase, 比如 `["forward", "turn_left"]` → `"The camera moves forward, then turns left."`。这种 mapping 是 lossy 的, 这也是为什么 zero-shot text-conditioned WM 在 closed-loop 上表现一般 —— **controllability 瓶颈** 就在这。Camera trajectory 控制稍微好一点, 但也只是粗粒度 spatial control, 没法精细到物体级 interaction。Low-level action conditioning 是最理想的, 但需要 post-training 才能拿到。

---

## 4. 四个 Task: 覆盖 perception / navigation / manipulation

Table 里四个 task 设计上有 deliberate 的层次:

### 4.1 Active Recognition (AR)
- **Setup**: Habitat-Sim + Matterport3D, 551 episodes, 29 scenes。Target object 被 occlusion 或 extreme viewpoint 挡住。Agent 有 $K=10$ decision step。
- **WM 的两个用途**:
  1. **Answering**: WM 生成 synthetic future views, 作为辅助 evidence 帮 VLM 识别 (跟 amodal recognition 思路, Aydemir 2013, https://ieeexplore.ieee.org/document/6519273 接近)
  2. **Navigation**: 用 rollout 来选 informative 的 path
- **Configuration**: $M=2$ candidates, $L=4$ horizon
- **Bounding box tracking**: SAM2 (https://github.com/facebookresearch/sam2) seeded by ground-truth box on $\mathbf{o}_t$, 沿着 rollout frames 做 segmentation propagation

### 4.2 ImageNav (Image-Goal Navigation)
- **Setup**: HM3D, 144 episodes, 87 scenes。Goal 是一张参考图, agent 要走到那个 viewpoint 0.5m 半径内。
- **Budget**: $K=20$, $L=5$, 执行前 $L-2=3$ 步再 replan (经典 receding horizon)
- **WM 只用于 planning**, 不用于 perception
- **Metrics**: SR / Mean Traj / SPL

SPL 的公式讲一下, 因为后面会反复出现:

$$\mathrm{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{L_i^*}{\max(L_i, L_i^*)} \times 100\%$$

- $N$: episode 数
- $S_i \in \{0, 1\}$: episode $i$ 是否成功
- $L_i^*$: 从起点到 goal 的最短路径长度 (geodesic)
- $L_i$: agent 实际走过的路径长度

### 4.3 A-EQA (Active Embodied Question Answering)
- **Setup**: OpenEQA + HM3D, 184 questions, 54 scenes。Open-ended question, 要 active explore 后回答。
- **Two-level policy**:
  - High-level planner: 发 textual instruction 或 landmark index (用 YOLO-World + SAM2 + Set-of-Marks prompting, https://arxiv.org/abs/2304.06468)
  - Low-level controller: 用 depth + pathfinder 执行
- **Budget**: 250 low-level actions per episode
- **WM 角色**: 只 strengthen high-level planner, $M=3$ candidates, $L=14$, 但只 return terminal observation 给 planner score
- **A-EQA SPL**:

$$\mathrm{SPL}_{\text{A-EQA}} = \frac{1}{N} \sum_{i=1}^N \left(\frac{\sigma_i - 1}{4}\right) \frac{L_i^*}{\max(L_i, L_i^*)} \times 100\%$$

- $\sigma_i \in [1, 5]$: GPT-4o 给的 raw answer score
- $(\sigma_i - 1)/4$: 把 [1,5] 映射到 [0,1], 作为 success weight

### 4.4 Robotic Manipulation
- **Setup**: RLBench (https://github.com/stepjam/RLBench), 4 个 task (Push Buttons, Slide Block to Color Target, Insert onto Square Peg, Stack Cups), 每个 50 episodes
- **Action**: 7-DoF end-effector $[x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper}]$
- **两种 base policy**:
  - VLM (Qwen2.5-VL-72B-AWQ, https://github.com/QwenLM/Qwen2.5-VL): 离散化 action (position 100 bins, orientation 120 bins), $K=15$, $L=5$
  - 3D Diffuser Actor (Ke et al. 2024, https://arxiv.org/abs/2410.21135): continuous action, $K=8$, $L=50$
- **WM 角色**: $M=5$ candidates, linear interpolate 或 uniform sample 来匹配 WM 需要的 conditioning length

---

## 5. Post-Training Recipe: 把 video generator 变成 action-conditioned WM

### 5.1 Formulation

给定初始帧 $\mathbf{x}_1 \in \mathbb{R}^{3 \times H \times W}$, 要学 conditional distribution:

$$p_\theta(\mathbf{X} \mid \mathbf{x}_1, C(\mathbf{A}))$$

其中 $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \in \mathbb{R}^{3 \times H \times W \times N}$, $C(\mathbf{A})$ 是 action API 输出。

**Action alignment 细节**:
- Habitat-Sim: action 是 relative transformation, $\mathbf{x}_{i-1} \to \mathbf{x}_i$, 所以 prepend 一个 Null token 让 action 和 frame 一一对齐: $a_1 = a_{\mathrm{Null}}$
- Manipulation: action 是 absolute end-effector pose in world frame, 自然一一对齐

### 5.2 数据集构造 (Appendix D)

这是 paper 里我觉得最 underappreciated 的工程贡献。他们从 HM3D + Matterport3D 的 train split 里采了 **763,724 panorama RGB frames, 439,213 action trajectories, 858 scenes** (Table 9)。

Algorithm 1 三阶段:
1. **Waypoint selection**: 对面积 $S$ 的 scene, 采 $N_{\mathrm{wp}} = \max(1400, \lfloor \rho S \rfloor)$ 个 navigable 点, $\rho = 4 \, \mathrm{m}^{-2}$。每个点算 leaf score:

$$s(i) = \mathrm{ecc}(i) + \alpha \bar{d}(i)$$

- $\mathrm{ecc}(i) := \max_j D_{ij}$: eccentricity, 点 $i$ 到所有其他点的最大 geodesic distance
- $\bar{d}(i) = (|\mathcal{P}| - 1)^{-1} \sum_j D_{ij}$: 平均 geodesic distance
- $\alpha = 1.7$: 平衡权重

按 $s(i)$ 降序排, greedy 建一个 minimum spacing $r_f = 3 \mathrm{m}$ 的 waypoint set $\mathcal{W}$。这保证了 sample 点偏向 **peripheral 区域** (卧室、角落), 不在走廊里 redundant 采样。

2. **Path generation**: nearest-unvisited greedy, 用 Habitat path-finder 算 collision-free shortest path, 沿途记录 panoramic RGB-D。
3. **Dynamic update**: 每段轨迹后, 半径 $r_f$ 内的 waypoint 标记为 visited, 重算 $s(\cdot)$, refresh unvisited list $\mathcal{U} \leftarrow \mathcal{W}[:N_{\mathrm{leaf}}]$。这步保证 peripheral 区域优先被覆盖。

**Generalization check**: 所有 Habitat 训练 scene 都 disjoint from evaluation scene, 所以 SR 提升不能怪 memorization。

### 5.3 计算成本

Table 8 给了 post-training 在 ~40K clips 上的资源:
- SVD (1.5B): 84GB peak, 29 H100-hours
- LTX-Video (2B): 61GB, 5 H100-hours
- Wan2.1 (14B): 57GB (LoRA), 74 H100-hours
- Cosmos-Predict2 (2B): 71GB, 15 H100-hours

这相比从零 pretrain 一个 video model (动辄上万 H100-hours) 是 **几个数量级** 便宜。这是 "post-training scaling 比 pretraining scaling 划算" 这条 finding 的成本基础。

---

## 6. 主要结果 — 我把数字读给你听

### 6.1 AR + ImageNav (Table 1)

| 设置 | AR SR↑ | AR Traj↓ | ImageNav SR↑ | ImageNav Traj↓ | ImageNav SPL↑ |
|---|---|---|---|---|---|
| Heuristic (no WM) | 39.02 | 8.81 | 2.08 | 59.6 | 0.63 |
| VLM (no WM) | 50.27 | 6.24 | 35.42 | 47.5 | 25.88 |
| Wan2.1 (zero-shot, text) | 58.26 | 5.24 | 38.19 | 48.2 | 25.92 |
| Wan2.2 A14B (zero-shot, text) | 59.53 | 4.91 | 43.05 | 45.8 | 31.46 |
| Runway Gen4 (proprietary) | 64.79 | 4.06 | — | — | — |
| **Wan2.1†** (post-trained, action) | **62.61** | 4.73 | **45.14** | 45.8 | 32.10 |
| Wan2.2† A14B (post-trained, action) | 62.43 | 4.67 | **46.53** | 44.6 | **34.61** |
| SVD† (post-trained, action) | 60.98 | 5.02 | 43.05 | 46.0 | 30.96 |

**几个 readout**:
1. WM 一致地 lift base policy: VLM 50.27 → 64.79 (+14.5 pt AR), 35.42 → 46.53 (+11.1 pt ImageNav)
2. Post-training 提升: Wan2.1 58.26 → Wan2.1† 62.61 (AR), 38.19 → 45.14 (ImageNav)
3. Heuristic base policy + WM (SVD†): 60.62 SR on AR —— 即使 base policy 极弱, 也能被 WM 救起来, 这跟 model-based RL 的经典观察一致 (poor policy + good model > good policy alone, 在 long horizon 任务上)
4. Runway Gen4 proprietary 拿到 64.79 SR, 但 paper 没法做 post-training, 不知道 action-conditioned 之后会不会更高

### 6.2 A-EQA (Table 2)

Ans. Score 提升幅度小: VLM 45.7 → Wan2.1† 48.2 (+2.5), Wan2.2† A14B 48.4 (+2.7)。这个 task 难, 因为 question 涉及 spatial reasoning、functional reasoning, WM 给的"mental preview"信息量在长 horizon 上衰减很快。LTX-Video† 和 Wan2.2† A14B 是最好的, 48.6 / 48.4。

### 6.3 Manipulation (Table 3)

这是最弱的 link:
- VLM base 44.5 SR, +SVD 44.0, +SVD† 46.5 (+2 pt), +Cosmos-P2† 45.0 (+0.5 pt)
- 3D Diffuser Actor base 24.0 SR, +SVD† 44.7 (+20.7 pt!), +Cosmos-P2† 38.0 (+14 pt)

注意 3D-DP 的 base policy 起点很低 (24%), 加 WM 之后跳到 44.7%, 说明 **base policy 越弱, WM 增益越大** —— 这跟 AR 上 heuristic 的现象一致。但 contact-rich manipulation 的物理细节 (compliance、friction、articulation) WM 还抓不住, 所以天花板低。这跟 Kang et al. 2024 (https://arxiv.org/abs/2411.02385) "How far is video generation from world model" 的物理分析是吻合的。

---

## 7. 三条 Key Findings — 这是 paper 的真正 contribution

### 7.1 Finding 1: Visual quality 不预测 task success, controllability 才预测

Figure 5(a) 是 SR vs generation quality (aesthetic + image quality 平均), 散点几乎 random。Figure 5(b) 是 SR vs controllability, controllability 定义为:

$$\text{controllability} := 1 - \mathrm{LPIPS}(\mathbf{o}_{\mathrm{gt}}, \hat{\mathbf{o}})$$

(ground-truth observation vs WM prediction 的 LPIPS 距离, 越小 controllability 越高)

Post-training 之后 controllability 大幅提升, SR 跟着涨。这告诉我们: **video generator 的 "世界知识" 是被封在 text-prompt 语义层, 而不是 low-level action 层。你想要 embodied agent 用, 必须把 action 接进来做 post-training, 哪怕只有 40K clip。**

这也呼应了 PhysDreamer (https://arxiv.org/abs/2404.13026)、Force Prompting (https://arxiv.org/abs/2505.19386) 这条线 —— 物理信号必须显式 condition 进去。

### 7.2 Finding 2: Post-training data scaling > Pretrained model scaling

Figure 6 是 SR vs post-training examples (400 → 80K):
- Wan2.1† (14B): 60.25% → 63.34% (saturate 慢)
- SVD† (1.5B): 56.80% → 60.98% (saturate 早)
- Wan2.2† A14B: 起点 58%, 40K 后追上 Wan2.1†

Wan2.2 A14B 的 web-video pretraining 比 Wan2.1 14B 大不少, 但在 embodied task 上并没有跑赢 —— 直到 post-training data 上来才 match。这说明 **pretraining 知识对 embodied transfer 的边际效用递减, action-conditioned post-training 的边际效用还在涨**。

这对下一波世界模型设计的 implication: 与其无限 scale pretraining compute, 不如花小钱 (LoRA 5-74 H100-hours) 做 action-conditioned post-training。这跟 LLM 领域 instruction tuning / RLHF 的故事结构很像。

### 7.3 Finding 3: Inference-time scaling 显著有效

Figure 7: 平均 WM inferences per episode 从 3 → 11, SVD† 的 AR SR 从 53.36% → 60.98%。这等价于在 closed-loop 里做 inference-time search —— 更多 candidate plan → 更 informative rollout → 更好的 revision 决策。

这跟 OpenAI o1 (https://openai.com/o1/)、DeepSeek R1 在 LLM reasoning 上的 test-time compute scaling 是同构现象, 但在 embodied + visual 模态上第一次系统化报告。budget 怎么分配? Paper 用的是 beam search over candidate plans, 可以扩展到 MCTS、beam search with branching, 跟 AlphaGo (https://www.nature.com/articles/nature16961) 思路一样。

---

## 8. Ablation: Panorama vs Front View (Table 4)

这组实验我挺喜欢的, 因为它揭示了 input context 的一个 tradeoff:
- Panorama 给 360° 全局信息, 但要做 panorama → perspective 转换, 损失 resolution
- Front view 高分辨率, 但 FOV 有限

结果:
- AR: SVD† panorama 60.98 vs front 57.89, +3 pt
- AR: Wan2.1† pano 62.61 vs front 62.25, +0.4 pt
- ImageNav: SVD† pano 43.05 vs front 38.19, +4.86 pt
- ImageNav: Wan2.1† pano 45.14 vs front 48.61, **-3.47 pt** (反过来了)

这说明 tradeoff 是 task-dependent 的, 没有银弹。Wan2.1† 在 ImageNav 上 front view 更好, 大概是因为 navigation 需要细节匹配 goal image, resolution loss 反而成了主导。这指向一个未来方向: **hierarchical context encoding** (Zhou et al. 2025b 3D persistent WM, https://arxiv.org/abs/2505.05495; Xiao et al. 2025 WorldMem, https://arxiv.org/abs/2506.18903)。

---

## 9. 失败案例分析 (Appendix E, Figure 13-14)

这部分 visual quality 高的 WM 经常 hallucinate。例子: action sequence 全是 "Forward", 但 zero-shot Wan2.1 / Hunyuan / LTX-Video 的 rollout 里相机自己转起来了 —— 因为 text prompt 没法强制约束 motion, 模型 fallback 到 web video prior (那些 clip 里 camera 经常动)。这种 "priors override action" 现象在 unseen scene 上特别明显, 这就是 paper Discussion 里说的 "generalization 是 practical use 的关键"。

这个观察让我想到 Sutton 在 "The Bitter Lesson" 里的论点: web video prior 是 general knowledge, 但 embodied task 需要 specific action-conditioned knowledge —— 你必须用搜索 + compute 来叠加 specific knowledge, 不能指望 prior 自动 generalize。

---

## 10. Discussion 部分 paper 提的四个未来方向, 我点评一下

1. **Generalization to novel environment** — 他们建议 unified action representation (Wang 2025b https://arxiv.org/abs/2508.13104, Zhi 2025 https://arxiv.org/abs/2506.06199)、curriculum data。我觉得更根本的可能是 action tokenizer 的设计, 类似 RT-2 (https://arxiv.org/abs/2307.15818) 把 action 变成 text token。

2. **Long-horizon planning** — Panorama 没解决, 真正需要的是 **spatial memory + episodic memory**。VMem (https://arxiv.org/abs/2506.18903) 用 surfel-indexed memory, Mixture of Contexts (https://arxiv.org/abs/2508.21058) 做长 video, 都是候选。

3. **Precise interaction/dynamics modeling** — Manipulation 的瓶颈。PhysDreamer、InterDyn (https://arxiv.org/abs/2505.13211)、Force Prompting 都在试, 但都没到能解决 contact-rich 的程度。可能需要 hybrid: video diffusion 出 appearance, physics engine 出 dynamics, 两者耦合。

4. **Stronger proposal/revision policy** — Performance floor 由 base policy 决定, ceiling 由 WM 决定。Neary et al. 2025 (https://arxiv.org/abs/2508.12211) 探索 VLA + model-based search, 是这个方向。

---

## 11. 我的整体 takeaway

这篇 paper 真正的价值是把 world model community 从 "video fidelity 竞赛" 拽到 "closed-loop utility 竞赛"。三条 finding:
- **Fidelity ≠ Utility** (controllability 才是)
- **Post-training data scaling > Pretraining scaling** (在 embodied setting)
- **Inference-time compute scaling 有效** (test-time planning)

这三条结合起来, 对下一波 world model 的设计有一个清晰的 steer: **base model 用 web video pretrain (获取 general visual prior), post-training 用 action-observation pair 对齐 task domain + action space, inference 时用 beam search 做 look-ahead planning**。这跟 LLM 的 pretrain → SFT → RLHF → test-time search 的 recipe 几乎是同构的, 只是把 token 换成 pixel + action。

需要警惕的地方:
- Manipulation 上 WM 增益小, 说明接触式物理建模是 hard wall
- Web video prior 在 unseen scene 会 override action control, generalize 仍是 open
- Long-horizon 上 single-frame anchor + 短 horizon rollout 信息衰减快, 需要记忆机制

如果让我赌下一步: 我觉得 **action-conditioned video diffusion + memory-augmented long context + physics-aware conditioning** 这三件事会汇合, 出现一个真正能跑 manipulation 的 world model。当前 World-in-World 这个 benchmark 会是衡量这条线进展的标尺。

---

## Web Links 汇总

- 项目主页: https://world-in-world.github.io
- Code: https://github.com/World-In-World
- 相关 world models:
  - Wan2.1/2.2: https://github.com/Wan-Video/Wan2.1
  - Cosmos-Predict2: https://github.com/nvidia-cosmos/cosmos-predict2
  - SVD: https://github.com/Stability-AI/generative-models
  - LTX-Video: https://github.com/Lightricks/LTX-Video
  - HunyuanVideo: https://github.com/Tencent/HunyuanVideo
  - NWM: https://arxiv.org/abs/2506.06040
  - PathDreamer: https://arxiv.org/abs/2105.01235
  - SE3DS: https://arxiv.org/abs/2304.09308
  - Runway Gen4: https://runwayml.com/research/introducing-runway-gen-4
- Benchmark / Simulators:
  - Habitat: https://github.com/facebookresearch/habitat-sim
  - HM3D: https://arxiv.org/abs/2109.08238
  - Matterport3D: https://github.com/niessner/Matterport
  - RLBench: https://github.com/stepjam/RLBench
  - OpenEQA: https://github.com/facebookresearch/open-eqa
- 相关方法:
  - SAM2: https://github.com/facebookresearch/sam2
  - YOLO-World: https://github.com/AILab-CVC/YOLO-World
  - Set-of-Marks: https://arxiv.org/abs/2304.06468
  - Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
  - 3D Diffuser Actor: https://arxiv.org/abs/2410.21135
  - Video Language Planning: https://arxiv.org/abs/2310.02535
  - Generative World Explorer: https://arxiv.org/abs/2506.19565
- 物理感知生成:
  - PhysDreamer: https://arxiv.org/abs/2404.13026
  - Force Prompting: https://arxiv.org/abs/2505.19386
  - InterDyn: https://arxiv.org/abs/2505.13211
- 长上下文 / memory:
  - WorldMem: https://arxiv.org/abs/2506.18903
  - Mixture of Contexts: https://arxiv.org/abs/2508.21058
  - 3D Persistent WM: https://arxiv.org/abs/2505.05495
- 相关 analysis:
  - How far is video gen from world model: https://arxiv.org/abs/2411.02385
  - WorldScore: https://arxiv.org/abs/2504.00983
  - WorldModelBench: https://arxiv.org/abs/2502.20694
  - VBench: https://github.com/Vchistar/VBench

需要我 deep dive 哪一块的话, 比如 post-training 的 trajectory sampling 算法、或者 revision policy 的 scoring function 具体怎么用 VLM 实现, 或者某个 task 的 prompt template, 都可以再展开。
