---
source_pdf: HAMSTER.pdf
paper_sha256: 896385bcdc12ec558ad4118a823de897d3e90064ee8b598d3399ca8f343481e7
processed_at: '2026-08-04T23:25:47-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HAMSTER 人话版 — 用最直白的话讲清楚

## 核心故事就一句话

Robot 学习最大的痛点是**数据太贵**。你在 Franka 上 teleop 收数据，一天撑死几百条，跟 LLM 吃的 web text 量级差了十万八千里。所以大家就想: 能不能用便宜的 data (simulation、YouTube video、别人 robot 的 data) 来训练 real robot policy?

HAMSTER 说: 能, 但你得换个**中间接口**。别让 VLM 直接吐 action, 让它吐一个 **2D path** (robot 末端在 image 上的粗略轨迹 + 什么时候开关 gripper)。这个 path 可以从任何 video 里抽出来, 跟 robot 长什么样无关, 跟物理参数无关, 所以 simulation data、别人 robot 的 data、甚至人手 video 都能拿来训练 VLM。然后一个小的 low-level policy 负责把这个 path 在 real robot 上 ground 成精确 action。

就这么简单。**VLM 负责"想", 小 policy 负责"做", 中间用 2D path 传话。**

Project page: https://hamster-robot.github.io/

---

## 为什么直接预测 action 行不通

OpenVLA、RT-2、π0 这些 monolithic VLA 的 recipe 是: 拿一个 VLM, fine-tune 它直接输出 robot action (joint angles 或 end-effector delta), token 化成 text 输出。听起来很 elegant, 实际上有几个硬伤:

**第一, action label 跟 embodiment 绑死**。Franka 的 7-DoF joint angle 跟 WidowX 的完全不一样, 你在 WidowX 上收集的数据没法直接给 Franka 用。所以 Open X-Embodiment 虽然号称 cross-embodiment, 实际 transfer 效果一般 (https://robotics-transformer-x.github.io/)。

**第二, action label 跟 dynamics 绑死**。仿真器里的 friction、inertia、control frequency 跟 real world 不一样, 你在 RLBench 上 fine-tune 的 action policy 拿到 real 基本要重新训。这就是 sim-to-real gap 的本质 (https://arxiv.org/abs/2405.05941)。

**第三, VLM 推理慢**。OpenVLA 7B 在 RTX 4090 上 6 Hz, 做 contact-rich manipulation 不够用。而且你想 scale 到 70B, control frequency 更低, 根本没法做 dynamic task。

**第四, VLM 只能吃 RGB + text**。但精细 manipulation 需要 depth、point cloud、proprioception history, 这些 VLM 原生不支持。RVT-2、3D Diffuser Actor 这种 3D policy 能吃这些, 但它们又缺少 semantic reasoning 和 generalization。

所以 monolithic VLA 把**两件本来该分开的事**绑在了一起: task-level reasoning (这个指令是什么意思、该抓哪个物体、按什么顺序) 和 motion-level control (怎么规划轨迹、怎么避障、怎么精确 grasp)。这两件事需要的数据类型、model capacity、inference frequency 完全不同, 硬塞一个 model 里两头不讨好。

HAMSTER 的解法: 拆开。VLM 只管 reasoning, 输出一个 coarse plan; 小 policy 只管 control, 吃 plan 执行。中间的 plan 就是 2D path。

---

## 2D Path 长什么样

形式化定义:

$$p = [(x_t, y_t, \text{gripper\_open}_t)]_t$$

- $x_t, y_t \in [0,1]$: normalized pixel coordinate, 就是 robot 末端在 image 上的位置, 归一化到 0-1
- $\text{gripper\_open}_t \in \{0, 1\}$: gripper 是开还是关, 二值
- $t$: path 上的离散 step index, **不是** control timestep, 只是 waypoint 编号

实际上经过 RDP 算法简化 (https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) 后, 一个 task 的 path 通常只有 **3-5 个点**。比如 "pick up the cup and put it in the bowl" 的 path 可能是:

```
[(0.3, 0.7), (0.3, 0.5), <Close Gripper>, (0.3, 0.5), (0.6, 0.4), <Open Gripper>]
```

人话翻译: 先到 (0.3, 0.7) 附近 (杯子位置), 下移到 (0.3, 0.5) 抓住, 闭合 gripper, 移动到 (0.6, 0.4) (碗位置), 松开 gripper。

就这么简单。没有任何 velocity、force、3D depth 信息, 纯粹是 image plane 上的 coarse spatial plan。

---

## 为什么 2D Path 是"对的" representation

这是 paper 最核心的 insight, 我详细讲下 intuition。

### 三个 criteria

作者明确说了, 一个好的 intermediate representation 必须满足:

1. **能从 image sequence 自动抽取** — 不需要 action label, 任何 video 都能产生
2. **跟 embodiment 无关** — Franka、WidowX、人手都能产生同一种 representation
3. **跟 dynamics 无关** — sim 和 real 的 friction、inertia 差异不影响它

2D path 完美满足这三条:

**Criterion 1**: 给一段 video, 用 TAP (TAPIR https://arxiv.org/abs/2306.08637, CoTracker https://arxiv.org/abs/2307.07663) 在 gripper 或人手上 track 一个 point, 抽出来就是 path。Paper 里 simulation 用 forward kinematics + camera projection 算, real robot 用 proprioception + PnP 估计 extrinsics。

**Criterion 2**: pixel coordinate 只跟 camera viewpoint 有关, 跟 robot joint 结构无关。Franka 末端投到 image 是 (0.3, 0.4), WidowX 也是 (0.3, 0.4), 人手也是 (0.3, 0.4)。这把不同 embodiment 的数据 normalize 到同一个 space。

**Criterion 3**: path 不 encode velocity、acceleration、force, 只 encode "下一步去哪 + 何时开关 gripper"。所以仿真 50 Hz Franka、real 20 Hz Franka、WidowX 30 Hz 的 path 长得一样, 只是 low-level policy 自己 adapt timing。

### 跟其他 representation 对比

| Representation | 例子 | Bandwidth | Embodiment-agnostic? | 问题 |
|---|---|---|---|---|
| Keypoint affordance | MOKA, RT-Affordance | 低 (1-3 points) | Yes | 只能说"去这里", 不能说"怎么绕" |
| 2D path | HAMSTER, RT-Trajectory | 中 (3-5 pts) | Yes | sweet spot |
| Dense flow | APTM, Track2Act | 高 | Yes | VLM 输出 token 太多, reasoning 难 |
| Direct action | OpenVLA, RT-2 | 低 | **No** | embodiment 绑死, dynamics 绑死 |

**Intuition**: representation 的选择本质是 bandwidth 和 transferability 的 trade-off。Direct action bandwidth 低但 transferability 也低 (跨 embodiment 不通用); dense flow bandwidth 高 transferability 也高, 但 VLM 生成困难。2D path 是中间的 sweet spot, bandwidth 够表达 task structure, transferability 又够 cross-domain。

---

## 架构怎么搭

### High-level VLM

**Base**: VILA-1.5-13B (https://arxiv.org/abs/2312.07533), 13B parameter 的 VLM

**Input**: 一张 RGB image + language instruction

**Output**: text 格式的 path, 大概长这样:
```
<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), 
<action>Open Gripper</action>, (0.74, 0.21), 
<action>Close Gripper</action>]</ans>
```

注意 gripper action 用 language token (`<action>Open Gripper</action>`) 而不是数值, 这保留了 VLM 的 text generation inductive bias。

**Finetuning loss**:
$$\mathcal{L}_{\text{VLM}} = -\mathbb{E}_{(\text{img}_i, z_i, \text{ans}_i) \sim \mathcal{D}_{\text{off}}} \log \text{VLM}(\text{ans}_i \mid \text{img}_i, z_i)$$

标准 next-token prediction NLL, 没什么花活。整个 model 包括 vision encoder 都 full fine-tune。

**Training data**: 关键在这里。VLM 完全**没见过** deployment environment 的数据, 全部是 off-domain:

| 数据源 | 数量 | 提供什么 |
|---|---|---|
| RoboPoint pixel prediction | 770k | "what" — 物体位置 grounding |
| RLBench simulation paths | ~320k | "what + how" — 完整 path, 视觉跟 real 完全不同 |
| Bridge + DROID real robot | ~110k | "what + how" — real visual, 但不同 embodiment |
| General VQA | 660k | 防止 catastrophic forget web knowledge |

**Co-training with VQA 是关键**: 单独 train path prediction 会 forget 掉 VLM 预训练的 world knowledge, 混 660k VQA 起 regularization 作用。这跟 RLHF 阶段混 SFT data 一个思路。

**Training setup**: 8× A100, 30 小时, batch size 256, lr $1 \times 10^{-5}$。

### Low-level Policy

两个选择, 都是 3D-aware 的 state-of-the-art:

**(a) RVT-2** (https://arxiv.org/abs/2406.13845):
- 吃 RGB + depth, 做 multi-view virtual scene re-projection
- Transformer 预测 3D next keyframe action
- 跟 HAMSTER 集成时去掉 language (path 已经 encode 语义)

**(b) 3D Diffuser Actor** (https://3d-diffuser-actor.github.io/):
- 吃 RGB + point cloud
- Diffusion policy over 3D action sequence
- 保留 language 反而更好 (它的 CLIP cross-attention 机制依赖 language)

**Path 怎么喂给 policy**: 两种方式

1. **Overlay**: 把 path 画在 RGB image 上, color gradient (蓝→红) 表示时间推进, 实心圆圈表示 gripper state change。优点: 不改架构, 通用。缺点: RVT-2 的 re-projection 会把 2D drawing 切碎。

2. **Concat channel**: path-only image 作为额外 3 channel concat, 变成 6-channel input。优点: 信息保真, Table 2 显示 camera view invariance 从 0.83 → 1.00。缺点: 跟 pre-trained 3-channel image encoder 不兼容, 3D-DA 用不了。

**Low-level training loss**:
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}_{(s_i, o_i, z_i, p_i, a_i) \sim \mathcal{D}_{\text{path}}} \log \pi_\theta(a_i \mid s_i, o_i, z_i, p_i)$$

- $s_i$: proprioception (joint position, gripper state)
- $o_i = (\text{img}, \text{point cloud})$: 多模态感知
- $z_i$: language (RVT-2 case 省略)
- $p_i$: **oracle** path (训练时用 ground truth, 不是 VLM 的 noisy 输出)
- $a_i$: action

**Robustness trick**: 训练时给 $p_i$ 的 $(x, y)$ 加 $\mathcal{N}(0, 0.01)$ 噪声, 模拟 VLM 推理误差, 让 policy 对 imperfect path 鲁棒。

### Inference flow

1. Episode 开始调一次 VLM: $\hat{p} \sim \text{VLM}(\text{img}, z)$, 大概 1 秒
2. $\hat{p}$ 画到 image 上
3. Low-level policy 每个 control step query: $a_t \sim \pi_\theta(\cdot | s_t, o_t, z, \hat{p})$, 10-30 Hz
4. 跑到结束

**频率解耦**: VLM 慢 (1 Hz) 但只调一次, policy 快 (30 Hz) 每步都调。这跟 OpenVLA 6 Hz 完全不同量级。你把 VLM 换成 70B 也不影响 control freq, 这是 hierarchical 结构的巨大优势。

---

## 实验数字怎么读

### 主结果 (Table 6, 真机)

| Task Type | RVT2 (base) | 3D-DA (base) | OpenVLA | HAMSTER+RVT2 | HAMSTER+3D-DA |
|---|---|---|---|---|---|
| Pick & place | 0.28 | 0.19 | 0.46 | **0.79** | 0.78 |
| Press button | 0.13 | 0.16 | 0.25 | 0.50 | **0.63** |
| Knock down | 0.17 | 0.03 | 0.41 | 0.47 | **0.66** |

**人话解读**:
- HAMSTER 平均比 OpenVLA 高 20% absolute, 相对提升 50%
- 比 base 3D policy 高 2-4x, 说明 path guidance 对 base policy 是巨大 unlock
- 3D-DA 单独只有 0.19, 加 HAMSTER 到 0.78 — **4 倍提升**。这说明 base policy 本身能力够 (3D + diffusion), 但缺 high-level task grounding, path 把它 unlock 了

### Camera view invariance (Table 2)

| Method | 原 camera success | 新 camera success | 掉多少 |
|---|---|---|---|
| OpenVLA | 0.60 | 0.23 | **崩盘 62%** |
| HAMSTER+RVT2 (overlay) | 0.83 | 0.73 | 掉 12% |
| HAMSTER+RVT2 (concat) | 1.00 | 0.98 | **几乎不掉** |

**人话**: OpenVLA overfit 到训练 camera 角度, 换个角度就崩。HAMSTER 的 VLM 在 RLBench + Bridge + DROID 上见过 diverse viewpoint, 所以 robust。Concat channel 比 overlay 更好, 因为 path 信息不被 re-projection 切碎。

### Data efficiency (Table 1)

| Method | Success |
|---|---|
| 3D-DA (100% data) | 0.18 |
| HAMSTER+3D-DA (50% data) | 0.36 |
| HAMSTER+3D-DA (100% data) | 0.43 |

**人话**: HAMSTER 用一半数据就达到 baseline 2 倍效果。因为 path 把 task 分解了, low-level 只需要学 "follow the line in 3D", 不用学 "which object to pick up", 样本效率自然高。

### 仿真 data 真的有用吗 (Table 5)

Human ranking (1=best, 4=worst) on **real-world** images:

| Method | Rank |
|---|---|
| RT-Trajectory + GPT-4o zero-shot | 3.47 |
| RT-Trajectory + GPT-4o + Code-as-Policies | 3.41 |
| HAMSTER VILA (no RLBench sim data) | 2.13 |
| HAMSTER VILA (with RLBench sim data) | **1.40** |

**这是 paper 最反直觉的发现**: 加 RLBench 这种视觉上跟 real 完全不一样的 simulation data, real-world performance **反而更好** (rank 从 2.13 降到 1.40)。

**为什么**: 2D path 是 appearance-agnostic 的, 仿真器再 ugly, 它产生的 path label 跟 real-world 的 path label 在同一个分布。VLM 学的是 "task → spatial plan" 的 mapping, 这个 mapping 跟 renderer 无关。这跟 monolithic VLA 完全相反 — OpenVLA 加 RLBench fine-tune 几乎没收益 (0.54 vs 0.58), 因为 action label 跟 real 不兼容。

**这才是 HAMSTER 最深刻的贡献**: 证明了 hierarchical + 2D path 这个组合能 unblock simulation data, 这是 monolithic VLA 结构性做不到的。

---

## 失败模式 (Appendix E) — 很有启发

作者把失败分三类:

**(1) VLM 预测错 path**
- 没理解 language goal (训练集缺类似 task)
- 预测错物体或方向
- 环境 dynamic 变化, path 一次生成没法 re-plan

**(2) Low-level 没跟住 path**
- 3D ambiguity: 2D path 在 pixel (0.5, 0.5), policy 不知道这是 "物体前面" 还是 "物体上方"
- Policy 没 hard constraint 必须 follow path, 可能 drift

**(3) 执行错**
- Grasp 角度错, 物体滑掉
- Contact-rich 精细控制失败

**Failure distribution** (Figure 15):
- RVT-2: 72% adherence failure, 28% execution failure
- 3D-DA: 10% adherence failure, 90% execution failure

**为什么差这么多**: RVT-2 有 virtual view re-projection, 把 2D drawing 投到 3D 时会碎片化, policy 难 decode path; 3D-DA 直接在原 2D image 上 CLIP feature, path 信息保留完整, 但 execution 本身精细控制不如 RVT-2。

**Intuition**: 不同 low-level architecture 对 path representation 的兼容性不同, 这影响 hierarchy 的 effective bandwidth。选 low-level 时要考虑它怎么 consume path input。

---

## 跟相关工作什么关系

最接近的是 **RT-Trajectory** (https://arxiv.org/abs/2311.00899), 也用 2D sketch condition low-level policy。但 RT-Trajectory 的 sketch 来自 human 或 zero-shot GPT-4o, **没有 fine-tune VLM**。HAMSTER 证明 fine-tune VLM on off-domain data 显著 better (Table 5), 而且 sim data 能 transfer 到 real, RT-Trajectory 没验证这点。

**LLARVA** (https://arxiv.org/abs/2411.02807) 也预测 end-effector trajectory, 但只作为 auxiliary task 帮 action prediction, 仍然 monolithic, 仍然要 on-robot action data。HAMSTER 把 trajectory 从 auxiliary 升级为 primary interface, 这是结构性差异。

**MOKA** (https://arxiv.org/abs/2403.03174) 用 mark-based keypoints 当 interface, 表达力不如 2D path (只能 说"去这里", 不能说"怎么绕过去")。

**SayCan** (https://arxiv.org/abs/2204.01691) 也是 hierarchical, 但 high-level 只输出 discrete skill, 不输出 continuous plan, 表达力弱很多。

---

## 我的几个直觉

### 2D path 就是 robot 的 chain-of-thought

LLM 里 CoT (https://arxiv.org/abs/2201.11903) 把 complex reasoning 拆成 explicit intermediate steps, 显著提升 performance。HAMSTER 把 complex manipulation 拆成 explicit 2D plan, low-level 跟 plan 不跟 raw instruction。这是同一个 pattern: **把 implicit reasoning 变成 explicit intermediate representation**, 让下游 module 有更 structured 的 signal 可用。

### Representation choice 决定 data scalability

这是 paper 最深层的 takeaway。Direct action representation 把你锁死在 on-robot teleop data; 2D path representation 解锁 simulation、internet video、cross-embodiment data。**不是 model 不够大, 是 representation 把 data source 限住了**。

这跟 LLM 领域 "data quality > model size" 趋势一致 (Chinchilla https://arxiv.org/abs/2203.15556, Phi https://arxiv.org/abs/2306.11644)。Robotics field 也在经历同样的 pivot: 从 "scale up model" 到 "find the right representation to unlock more data"。

### Open-loop high-level 是当前最大局限

VLM 一次推理整个 episode, 不 re-plan。Long-horizon task 或 dynamic env 会断。如果能 integrate world model (DreamerV3 https://arxiv.org/abs/2301.04104, JEPA https://arxiv.org/abs/2301.08243), VLM 可以 rollout 多个 candidate paths 选最优的。这相当于把 model-based RL 的 planner 替换成 VLM, 用 2D path 当 plan representation。

### Future direction: learnable interface

2D path 是 hand-designed, bandwidth 有限, 无法表达 force、rotation、velocity。未来可以 learn 一个 optimal intermediate representation (类似 VQ-VAE codebook https://arxiv.org/abs/1711.00937), end-to-end trainable between VLM and policy, 同时保持 cross-domain transferability。这是 HAMSTER 自然延伸方向。

### Cross-embodiment 的更优雅解法

Open X-Embodiment (https://robotics-transformer-x.github.io/) 用 action tokenization 实现 cross-embodiment, 但效果一般。HAMSTER 提供另一条路: 把 action 替换成 embodiment-agnostic 的 2D path, 不同 embodiment 数据自然 normalize。High-level 共享, low-level embodiment-specific, 这可能是 cross-embodiment scaling 的更优雅方案。

---

## 一句话总结

HAMSTER 告诉我们: **robot learning 的瓶颈不在 model size, 在 representation choice**。2D path 这个看似简单的 interface, 因为它 embodiment-agnostic + dynamics-robust + cheaply-obtainable, 解锁了 simulation、internet video、cross-embodiment 这些便宜数据源。VLM 在这些 data 上 fine-tune 后, 能 transfer 到完全没见过的 real robot scene, 这是 monolithic VLA 结构性做不到的。

用一句话概括整个故事: **好的 representation 让你用得起便宜数据, 便宜数据才是 scaling 的真正燃料。**

如果想继续 dive deeper:
- RT-Trajectory: https://arxiv.org/abs/2311.00899
- RoboPoint: https://robopoint.github.io/
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- RVT-2: https://arxiv.org/abs/2406.13845
- VILA: https://arxiv.org/abs/2312.07533
- Colosseum benchmark: https://arxiv.org/abs/2402.08191
- OpenVLA: https://openvla.github.io/
- TAP video (TAPIR): https://arxiv.org/abs/2306.08637
- CoTracker: https://arxiv.org/abs/2307.07663
- Open X-Embodiment: https://robotics-transformer-x.github.io/

---

# HAMSTER 深度解读 — Hierarchical Action Models for Open-World Robot Manipulation

## 1. 一句话核心

HAMSTER 把 monolithic VLA 拆成两层: high-level VLM 只输出 **2D path** (coarse end-effector trajectory in image plane + gripper open/close markers), low-level 3D policy (RVT-2 / 3D Diffuser Actor) 负责把这个 path 在 RGB-D + proprioception 下 ground 成精确 actions. 关键收益: VLM 可以在 **off-domain data** (RLBench simulation, Bridge, DROID, RoboPoint VQA) 上 fine-tune, 因为 2D path 这个 intermediate interface **embodiment-agnostic, dynamics-robust, 容易从任何 video 抽取**, 跨 domain transfer 比直接预测 action 强很多。

Project page: https://hamster-robot.github.io/
arXiv (相关作者): https://arxiv.org/abs/2410.24164 (π0), https://arxiv.org/abs/2406.09246 (OpenVLA)

---

## 2. Motivation: 为什么 monolithic VLA 不够好

当前 VLA 主流 recipe 是把 Prismatic / PaliGemma / VILA 这种 VLM 在 observation-action pair 上 fine-tune, 把 action token 化输出 (RT-2, OpenVLA, π0)。这套思路有几个结构性问题:

**问题 A — 数据稀缺且昂贵**。On-robot teleop data 规模远小于 web-scale image-text。Open X-Embodiment 才 ~1M episodes, 跟 LLM/VLM 的预训练 corpus 完全不在一个量级。参见 https://arxiv.org/abs/2310.08864。

**问题 B — Action token 频率瓶颈**。OpenVLA 7B 在 RTX 4090 上 6 Hz, 做 dynamic / contact-rich manipulation 不够。RT-2 类似。这种 intrinsic latency 跟 model size 强绑定。

**问题 C — Cross-domain transfer 差**。Action label 是 embodiment-specific (joint angles, end-effector deltas), 也是 dynamics-specific. 仿真器里 fine-tune 的 action policy 很难 transfer 到真机 (Li et al. 2024, Mandlekar et al. 2021 都有讨论 sim-to-real gap)。

**问题 D — Sensory modality 受限**。VLM 输入基本是 RGB + text, 没法原生 consume depth / point cloud / tactile / multi-step proprioception, 但 low-level 精确 manipulation 又恰恰需要这些 (RVT-2, 3D-DA, Diffusion Policy 都依赖 3D 输入)。

HAMSTER 的核心 claim: 这些问题不是 VLA 不行, 而是 **monolithic 结构把 task-level reasoning 和 motion-level control 绑死在同一个 model + 同一种 data format 上**。Hierarchical 解耦后, high-level 享受 VLM 的 semantic generalization, low-level 享受 3D policy 的 spatial precision, 中间用 2D path 当 interface。

---

## 3. 关键 Insight: 为什么 2D Path 是好的 Intermediate Representation

这一节是 paper 的灵魂, 我展开讲下 intuition。

### 3.1 设计 criteria

Paper Section 4 明确给出 3 个 criteria 一个好 intermediate representation 必须满足:

1. **Easily obtainable from image sequences** — 能从 action-free video 自动抽出来, 不需要 action label
2. **Largely embodiment agnostic** — 跟 robot morphology 解耦, WidowX / Franka / 仿真机器人 / 人手都能产生同一种 representation
3. **Sufficiently robust to subtle dynamics changes** — 不 encode physics 细节, 所以 sim 和 real 之间的 friction / inertia / control frequency 差异不影响它

**2D path** 形式:
$$p = [(x_t, y_t, \text{gripper\_open}_t)]_t$$

- $x_t, y_t \in [0,1]$: normalized pixel location of end-effector (or hand) at step $t$
- $\text{gripper\_open}_t \in \{0, 1\}$: binary gripper state

**注意上下标语义**: $t$ 是 path 上离散 step index, 跟 low-level control timestep 不一一对应 (VLM 只输出 sparse waypoints, low-level policy 在它们之间 interpolate / react)。这是 RDP (Ramer-Douglas-Peucker) 简化后典型的 3-5 个点, 不是 dense trajectory。

### 3.2 为什么这个 representation 满足三个 criteria

**Criterion 1 (cheaply obtainable)**: 给一段 video, 用 TAP (TAPIR, CoTracker, https://arxiv.org/abs/2306.08637) 在 gripper / hand 上 track 一个 point, 抽出来就是 path。Paper 里 simulation 用 forward kinematics + camera params 投影; Bridge / DROID 用 proprioception projection + PnP 估计 extrinsics (因为 Bridge 没标定好的 camera extrinsics)。

**Criterion 2 (embodiment agnostic)**: pixel coordinate 只跟 camera viewpoint 有关, 跟 robot joint 结构无关。Franka 的 end-effector 投到 image 是 (0.3, 0.4), 人手抓东西也是 (0.3, 0.4), WidowX 也是 (0.3, 0.4)。这是 path representation 最关键的性质, 它把不同 embodiment 的数据 normalize 到一个 common space。

**Criterion 3 (dynamics robust)**: Path 不 encode velocity / acceleration / force, 只 encode "下一步去哪 + 何时开关 gripper"。所以仿真里 50 Hz Franka 跟真机 20 Hz Franka 跟 WidowX 30 Hz 的 path 长得一样, 只是 low-level policy 自己 adapt timing。

### 3.3 与其他 intermediate representation 的对比

Paper Section 2 把这个谱系列得很清楚:

| Representation | 来源 | Bandwidth | Embodiment-agnostic? | 论文 |
|---|---|---|---|---|
| Keypoint affordance | 检测器 / VLM prompting | 低 (1-3 points) | Yes (mostly) | MOKA, RT-Affordance, KITE |
| 2D path (HAMSTER) | TAP / sketch / FK projection | 中 (3-5 pts + gripper) | Yes | HAMSTER, RT-Trajectory |
| Object trajectory | TAP on object | 中 | Yes | Track2Act, General Flow |
| Dense grid flow | fixed grid points | 高 | Yes | APTM |
| Direct action | proprioception | 低 | No | OpenVLA, RT-2, π0 |

**关键 trade-off**: Keypoint affordance 表达力不够 (只能说 "去这里", 不能说 "怎么绕过去"); dense flow 太冗余 (VLM 输出 token 多, 不利于 reasoning); 2D path 是 sweet spot。

LLARVA (https://arxiv.org/abs/2411.02807, Niu et al. 2024) 是最接近的工作: 它也预测 end-effector trajectory, 但只作为 **auxiliary task** 帮助 action prediction, 仍然 monolithic, 仍然要 on-robot action data。HAMSTER 把 trajectory 从 auxiliary 升级为 **primary interface**, 这是结构性差异。

---

## 4. 架构详解

### 4.1 High-level VLM

**Base model**: VILA-1.5-13B (Lin et al. 2024, https://arxiv.org/abs/2312.07533)。13B 是经过 ablation 选的, 3B 在 path 表示上不够 robust (paper Appendix G Figure 17 显示 VILA-1.5-3B 在 fixed-20-point representation 下表现差, 13B 才能驾驭)。

**Input**: single RGB image + language instruction $z$
**Output**: text-formatted path tokens

Prompt 格式 (paper Figure 10):
```
In the image, please execute the command described in <quest>{quest}</quest>.
Provide a sequence of points denoting the trajectory of a robot gripper...
Format your answer as a list of tuples enclosed by <ans> and </ans> tags.
For example: <ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), 
<action>Open Gripper</action>, (0.74, 0.21), 
<action>Close Gripper</action>, ...]</ans>
```

**关键设计**: gripper action 单独作为 language token (`<action>Open Gripper</action>` / `<action>Close Gripper</action>`), 不写成数值。这保留了 VLM 的 token-based generation inductive bias, 类似 RT-2 的 action token 思路, 但只针对 discrete 的 gripper state。

**Finetuning loss** (Section 4.1.1):
$$\mathcal{L}_{\text{VLM}} = -\mathbb{E}_{(\text{img}_i, z_i, \text{ans}_i) \sim \mathcal{D}_{\text{off}}} \log \text{VLM}(\text{ans}_i \mid \text{img}_i, z_i)$$

标准 next-token prediction negative log-likelihood, 没有任何 special trick。**整个 model 包括 vision encoder 都更新** (full fine-tune, 不 freeze ViT)。

**Training data mixture** $\mathcal{D}_{\text{off}}$:

| 数据源 | 数量 | 用途 |
|---|---|---|
| RoboPoint pixel prediction | 770k | "what" — 物体位置 grounding |
| RLBench 2D paths (81 tasks × 1000 episodes × ~4 instructions) | ~320k | "what + how" — 完整 path |
| Bridge (10k) + DROID (45k) real robot | ~110k | "what + how" — real visual |
| General VQA (Liu et al. 2024c) | 660k | Preserve world knowledge |

**Co-training with VQA** 是关键技巧: 单独 train 在 path prediction 上会 catastrophic forget web knowledge (类似 LLM fine-tune 损失 generalization)。660k VQA 起到 regularization 作用。这跟 InstructGPT RLHF 阶段混 SFT data 一个套路。

**Path simplification**: RDP algorithm (Ramer 1972, Douglas & Peucker 1973, https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm), tolerance $\epsilon = 0.05$, 把 100+ step 的 path 简化到 2-5 points。RDP 的精髓: 递归地保留 deviation 最大的点, 直到所有点离直线距离都 < $\epsilon$。这样 path 只保留 "key turning points", 去掉 redundant waypoints, VLM 容易学, low-level policy 也容易 generalize。

**Training setup**: 8× A100, 30 小时, effective batch size 256, learning rate $1 \times 10^{-5}$, ~65 GB GPU mem per card。

### 4.2 Low-level Policy

**Two instantiations**:

**(a) RVT-2** (Goyal et al. 2024, https://arxiv.org/abs/2406.13845):
- 输入 RGB + depth → multi-view virtual scene re-projection (NeRF-like idea)
- Transformer predicts next keyframe action in 3D
- 跟 HAMSTER 集成时去掉 language input (path 已经 encode task semantics)

**(b) 3D Diffuser Actor** (Ke et al. 2024, https://3d-diffuser-actor.github.io/):
- 输入 RGB + point cloud
- Diffusion policy over 3D action sequence
- CLIP language tokens cross-attend with CLIP visual features (所以保留 language 反而更好)

**Path conditioning 的两种实现**:

1. **Overlay**: 把 2D path 直接画在 RGB image 上, color gradient 表示 temporal progression (蓝→红), 实心圆圈表示 gripper state change (蓝=close, 红=open)。优点: 不改架构, 通用。缺点: RVT-2 的 virtual re-projection 会把 2D drawing 切碎, 信息丢失。

2. **Concat channel**: 把 path-only image 作为额外 3 channel concat 到 RGB, 变成 6-channel input。优点: path 信息保真, Table 2 显示在 camera view invariance 实验里从 0.83 → 1.00。缺点: 需要 modify input layer, 跟 pre-trained image encoder (要求 3 channel) 不兼容, 所以 3D-DA 用不了。

**Low-level training loss**:
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}_{(s_i, o_i, z_i, p_i, a_i) \sim \mathcal{D}_{\text{path}}} \log \pi_\theta(a_i \mid s_i, o_i, z_i, p_i)$$

变量:
- $s_i$: proprioceptive state (joint position, gripper state)
- $o_i = (\text{img}, \text{point cloud})$: 多模态感知输入
- $z_i$: language instruction (RVT-2 case 下省略)
- $p_i$: **oracle** 2D path (训练时用 proprioception projection 构造 ground truth, 而不是 VLM 输出的 noisy path)
- $a_i$: action

**Robustness trick**: 训练时给 $p_i$ 的 $(x, y)$ 加 Gaussian noise $\mathcal{N}(0, 0.01)$, 模拟 VLM 推理时的 path 误差, 让 policy 对 imperfect path 鲁棒。这是 sim2real / domain transfer 的经典做法 (DAgger, dataset augmentation 思路)。

### 4.3 Inference flow

1. Episode 开始, 调一次 VLM (or 几次): $\hat{p} \sim \text{VLM}(\text{img}, z)$
2. $\hat{p}$ 画到 image 上
3. Low-level policy 每个 control step query 一次: $a_t \sim \pi_\theta(\cdot | s_t, o_t, z, \hat{p})$
4. 跑到 episode 结束 or 显式重新规划

**频率解耦**: VLM 一次查询 ~1s (13B VLM), low-level policy 10-30 Hz。这跟 OpenVLA 6 Hz 完全不在一个量级。可扩展性极强, 把 VLM 换成 70B 也不影响 control freq。这一点跟 SayCan / RT-2 / RT-X 思路类似, 但 SayCan 只输出离散 action, HAMSTER 输出 continuous path, 表达力强很多。

---

## 5. 实验数据深度解读

### 5.1 主结果 (Table 6, 真机)

按 task type 分组的 success rate:

| Task Type | RVT2 | 3D-DA | OpenVLA | HAMSTER+RVT2 | HAMSTER+3D-DA |
|---|---|---|---|---|---|
| Pick & place | 0.28 | 0.19 | 0.46 | **0.79** | 0.78 |
| Press button | 0.13 | 0.16 | 0.25 | 0.50 | **0.63** |
| Knock down | 0.17 | 0.03 | 0.41 | 0.47 | **0.66** |

观察:
- HAMSTER 平均比 OpenVLA 高 20% absolute (50% relative) — abstract 数字
- 比 base 3D policy (RVT-2 / 3D-DA) 高 2-3x, 说明 path guidance 巨大提升
- 3D-DA 单独 0.19 在 pick & place, 加 HAMSTER 到 0.78 — **4x 提升**, 这是惊人的。说明 base policy 本身能力够 (3D + diffusion), 但缺少 high-level task grounding, path 把它 unlock 了。

### 5.2 Generalization axes (Figure 4)

7 个 axes:
1. **obj and goal** — unseen object-goal combos
2. **visual** — table texture, lighting, distractor
3. **language** — "candy" → "sweet object", 需要世界知识
4. **spatial** — unseen spatial relationships
5. **novel object** — 完全没见过的物体
6. **multiple** — 多种 variation 叠加
7. **camera view** (Table 2 独立测) — 改 camera 角度

**Camera view invariance** (Table 2) 特别有意思:
- OpenVLA: 原 camera 0.60 success, 新 camera 0.23, **崩盘 60%**
- HAMSTER+RVT2 (overlay): 0.83 → 0.73
- HAMSTER+RVT2 (concat): 1.00 → 0.98 — **几乎不掉**

这说明 VLM 在 RLBench + Bridge + DROID 上学到的 viewpoint 是 diverse 的, 而 OpenVLA 因为 fine-tune data 单一 camera, overfit。

### 5.3 Data efficiency (Table 1, Colosseum simulation)

| Method | Success |
|---|---|
| 3D-DA (100% data) | 0.18 ± 0.10 |
| HAMSTER+3D-DA (50% data) | 0.36 ± 0.04 |
| HAMSTER+3D-DA (100% data) | 0.43 ± 0.05 |

**50% data 已经超过 100% data baseline 的 2x**。Intuition: path 把 task 分解了, low-level policy 只需要学 "follow the line in 3D", 不用学 "which object to pick up", 样本效率自然高。

### 5.4 Ablation: VLM 用 RLBench 仿真 fine-tune 有用吗 (Table 5)

Human ranking (1=best, 4=worst) on real-world images:

| Method | Rank |
|---|---|
| RT-Trajectory + GPT-4o zero-shot | 3.47 |
| RT-Trajectory + GPT-4o + Code-as-Policies | 3.41 |
| HAMSTER VILA (no RLBench sim data) | 2.13 |
| HAMSTER VILA (with RLBench sim data) | **1.40** |

**关键发现**: 加 RLBench 这种视觉上跟 real 完全不一样的 simulation data, real-world performance 反而**更好** (rank 从 2.13 降到 1.40)。这跟 monolithic VLA 直觉相反 — OpenVLA 加 RLBench fine-tune 几乎没收益 (paper Section 5.1 提到 0.54 vs 0.58)。

**我的解读**: 2D path 是 dynamics-agnostic & appearance-agnostic 的, 所以仿真器再 ugly, 它产生的 path label 跟 real-world 的 path label 在同一个分布。VLM 学到的是 "task → spatial plan" 的 mapping, 这个 mapping 跟 renderer 无关。这是 paper 最深刻的发现, 也是 hierarchical 设计 unblock simulation data 的核心机制。

### 5.5 Colosseum simulation variations (Table 3)

15 个 visual variations (background texture, camera position, distractor, lighting, manip object color/size/texture, recipient object, table color/texture, etc.):

平均 success: 3D-DA 0.35 → HAMSTER+3D-DA 0.46 (+31%)。每个 axis 都有提升, 最显著的是 `rlb var` (RLBench scene variation): 0.45 → 0.58。

---

## 6. 失败模式分析 (Appendix E) — 这节特别精彩

Paper 把 failure 分成三类:

**(1) Trajectory prediction failures** (VLM 错)
- 没理解 language goal (training set 缺类似 task)
- 预测错物体 / 方向
- 环境 dynamic 变化 (path 一次生成, 没法 re-plan) — 这是 closed-loop feedback 缺失的代价

**(2) Trajectory adherence failures** (low-level 没跟住 path)
- 3D ambiguity: 2D path 在 pixel (0.5, 0.5), 但 low-level 不知道这是 "物体前面" 还是 "物体上方"
- Policy 没 hard constraint 必须跟 path, 可能 drift 到错物体

**(3) Action execution failures** (跟对了也执行错)
- Grasp 角度错, 物体滑掉
- Contact-rich 操作的精细 control 失败

**Failure distribution** (Figure 15):
- RVT-2: 72% adherence failure, 28% execution failure
- 3D-DA: 10% adherence failure, 90% execution failure

**Why the discrepancy**: RVT-2 有 virtual view re-projection, 把 2D drawing 投到 3D scene 时会碎片化, policy 难 decode; 3D-DA 直接在原 2D image 上 CLIP feature, path 信息保留完整, 但 execution 本身 3D-DA 的精细控制不如 RVT-2 (这就是为啥 3D-DA base 比 RVT-2 base 弱, 但加 path 后反超)。

**Intuition**: 不同 low-level architecture 对 path representation 的兼容性不同, 这影响 hierarchy 的 effective bandwidth。HAMSTER 框架是 architecture-agnostic 的, 但实际部署要 pick 兼容的 low-level。

---

## 7. 跟相关工作位置

| 工作 | Hierarchical? | Intermediate rep | Off-domain data? |
|---|---|---|---|
| RT-2 / OpenVLA / π0 | No | None (direct action) | Limited (need action labels) |
| SayCan / Inner Monologue | Yes (LLM only) | Discrete skill | Yes |
| VoxPosPoser / Eureka | Yes | 3D value map / reward code | Yes |
| RT-Affordance | Yes | Keypoint affordance | Limited |
| MOKA | Yes | Mark-based keypoints | Yes |
| RT-Trajectory | Partially | 2D sketch (specified by human/VLM) | N/A (no VLM finetune) |
| LLARVA | No (auxiliary) | Trajectory as auxiliary | No |
| **HAMSTER** | **Yes** | **2D path (VLM-finetuned)** | **Yes (sim + video + cross-embodiment)** |

最接近的是 **RT-Trajectory** (Gu et al. 2023, https://arxiv.org/abs/2311.00899), 它也用 2D sketch condition low-level policy, 但 RT-Trajectory 的 sketch 来自 **human 或 zero-shot GPT-4o**, 没有 fine-tune VLM。HAMSTER 的 contribution 是证明 fine-tune VLM on off-domain data 显著 better (Table 5), 而且 RLBench sim data 能 transfer 到 real (RT-Trajectory 没验证这点)。

---

## 8. Limitations & Future directions (作者自己承认)

1. **2D, 不是 3D**: VLM 没真正 3D 理解, depth ambiguity 靠 low-level policy 自己 resolve。未来可能预测 3D path (NeRF-style coordinates) 或者 multi-view 2D paths。
2. **Bandwidth limited**: 2D path 无法表达 force, rotation, velocity。比如拧瓶盖需要 torque feedback, 这种 task HAMSTER 表达不了。
3. **Open-loop high-level**: VLM 一次推理整个 episode, 不 re-plan。Long-horizon task 或 dynamic env 会断。
4. **No learnable interface**: Path 是 hand-designed, 未来可以 learn 一个 optimal intermediate representation (类似 VQ-VAE codebook, 但 end-to-end trainable between VLM and policy)。

---

## 9. 我的 intuition 和联想

### 9.1 跟 LLM chain-of-thought 的对应

HAMSTER 的 2D path 本质是 **robot control 的 chain-of-thought**。LLM 里 CoT 把 complex reasoning 拆成 explicit intermediate steps, 显著提升 performance; HAMSTER 把 complex manipulation 拆成 explicit 2D plan, low-level policy 跟 plan 不跟 raw instruction。这跟 Toolformer, ReAct 思路也类似 — 把 implicit reasoning 变成 explicit intermediate representation。

参考: https://arxiv.org/abs/2201.11903 (CoT), https://arxiv.org/abs/2210.03629 (ReAct)

### 9.2 跟 Diffusion Policy 的关系

Diffusion Policy (Chi et al. 2023, https://diffusion-policy.cs.columbia.edu/) 把 action 生成建模成 diffusion, 用 noise schedule 表达 multimodality。HAMSTER 的 2D path 也是一种 "coarsened action representation", 但更 high-level, 更 semantic。两者可以结合: VLM 输出 2D path → diffusion policy 在 path 周围 sample 精细 action, 这样既有 semantic grounding 又有 action multimodality。其实 3D Diffuser Actor 已经是这个思路的雏形。

### 9.3 跟 RT-2 action tokenization 的对比

RT-2 把 6-DoF action 离散化成 256 bins × 6 tokens, 让 VLM 当 language 输出。问题是 action space 是 continuous 高维的, token 化损失精度, 且 VLM 在 token level 推理 action 不擅长 (VLM 训练时没见过 action token distribution)。HAMSTER 把 action 推理替换成 spatial reasoning (2D 坐标), VLM 本来就在 image-text 上训过 spatial grounding (coco, refcoco 类数据), 所以 transfer 更顺。

### 9.4 跟 VLA scaling law 的关系

当前 VLA 困境: data 是 bottleneck, 不是 model size。OpenVLA 7B 已经 near saturation on Open X-Embodiment data, scale 到 70B 不会 magically 出现 emergent capability。HAMSTER 的角度: 不 enlarge model, 而 **enlarge data** — 通过 cheap interface (2D path) 解锁 simulation, internet video, cross-embodiment 数据。这是 alternative scaling axis, 跟 LLM 的 "data quality > model size" 趋势 (Chinchilla, Phi) 一致。

参考: https://arxiv.org/abs/2203.15556 (Chinchilla), https://arxiv.org/abs/2306.11644 (Phi-1)

### 9.5 Path 作为 "robotic language"

更深一层: 2D path 是一种 **visual language**, 让 VLM 用 spatial coordinates 而非 natural language 来表达 plan。这跟 Scratchpads, Program-of-Thoughts (PoT) 思路一致 — 让 model 用更结构化的 output format 表达 reasoning。未来可能演化出更丰富的 robotic language: path + waypoints + force profile + contact points, 类似一个 DSL。

参考: https://arxiv.org/abs/2211.12588 (PoT), https://arxiv.org/abs/2406.03061 (Voyager, Minecraft 里的 skill library)

### 9.6 跟 World Model 的潜在结合

HAMSTER 是 open-loop (VLM 推一次), 但如果 integrate world model (DreamerV3, JEPA, https://arxiv.org/abs/2301.04104), VLM 可以 rollout 多个 candidate paths, 选 expected reward 最高的。这相当于把 model-based RL 的 planner 替换成 VLM, 用 2D path 当 plan representation。

### 9.7 Inverse dynamics 视角

从 RL 角度看, HAMSTER 把 policy 拆成:
- **High-level**: $\pi_{\text{high}}(p | s, z)$ — 一个 goal-conditioned planner
- **Low-level**: $\pi_{\text{low}}(a | s, p)$ — 一个 path-following inverse dynamics

这跟 UniPi (https://arxiv.org/abs/2303.04107), SuSIE (https://arxiv.org/abs/2310.10647) 这类 video planning 工作结构类似, 但 HAMSTER 用 2D path 而非 full video 当 interface, 大幅降低 bandwidth 和 learning 难度。

### 9.8 In-context learning 可能性

HAMSTER 当前是 fine-tune VLM。但 VILA / GPT-4o / Gemini 都有 in-context learning 能力。如果 prompt 里塞几个 (image, instruction, path) 例子, 也许 zero-shot 也能 work。Paper Table 5 显示 GPT-4o zero-shot 排 3.47/4, 比 fine-tune VILA 差, 但说明 ICL 有信号, future work 可以做 better prompting (chain-of-visual-thought, self-consistency over paths)。

### 9.9 Cross-embodiment 的根本意义

Open X-Embodiment 论文核心 claim: cross-embodiment generalization 通过 action tokenization 实现。但实际效果一般。HAMSTER 提供另一种 path: 把 action 替换成 embodiment-agnostic 的 2D path, 不同 embodiment 数据自然 normalize。这可能是 cross-embodiment scaling 的更优雅解法 — 让 high-level 共享, low-level embodiment-specific。

### 9.10 跟 Foundation Model in Robotics 的更宏大叙事

NVIDIA 这篇 + π0 (Physical Intelligence) + RT-2 + Octo + OpenVLA 一起看, 整个 field 正在分化成两个流派:
- **Monolithic foundation model**: 一个大 VLM 直接输出 action, 期望 emergent scaling (RT-2, OpenVLA, π0)
- **Hierarchical foundation model**: 大 VLM 做 reasoning, 小 specialist model 做 control, 中间 interface (HAMSTER, RT-Trajectory, SayCan)

HAMSTER 给 hierarchical 派提供了关键 empirical evidence: 2D path interface 能 unblock off-domain data, 这是 monolithic 派做不到的。如果 field 最终走 hierarchical 路线, HAMSTER 这种 "find the right interface" 的思路会反复出现。

---

## 10. 总结

HAMSTER 的 contribution 三句话:
1. **Structural**: VLM + 3D policy 中间用 2D path 连, decouple semantic reasoning 和 motion control
2. **Empirical**: 7 个 generalization axis 平均超 OpenVLA 20% absolute, data efficiency 2x
3. **Methodological**: 证明 2D path 这个 interface 让 VLM 能从 cheap off-domain data (sim, video) transfer 到 real robot, 这是 monolithic VLA 做不到的

最大 takeaway: **representation choice 决定 data scalability**。2D path 不是最 expressive 的 interface, 但它的 embodiment-agnostic + dynamics-robust 性质让它能 harvest internet-scale data, 这是真正的 scaling unlock。Future work 沿这条线找更 expressive 同时仍 cross-domain-transferable 的 interface (3D path? waypoint + affordance? learnable tokens?) 是 promising direction。

如果想 dive deeper, 推荐看:
- RT-Trajectory: https://arxiv.org/abs/2311.00899
- RoboPoint: https://robopoint.github.io/
- 3D Diffuser Actor: https://3d-diffuser-actor.github.io/
- RVT-2: https://arxiv.org/abs/2406.13845
- VILA: https://arxiv.org/abs/2312.07533
- Colosseum benchmark: https://arxiv.org/abs/2402.08191
- OpenVLA: https://openvla.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
