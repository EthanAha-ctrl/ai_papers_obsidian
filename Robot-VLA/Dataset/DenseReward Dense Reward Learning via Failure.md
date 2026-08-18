---
source_pdf: DenseReward Dense Reward Learning via Failure.pdf
paper_sha256: 115651988624eed90e76b1b7e3c58cf8d1af4bc2d44a51ea4c59479f66961256
processed_at: '2026-08-18T05:12:32-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DenseReward 的人话版本

好，Karpathy，我换个讲法，完全从直觉出发，把这篇 paper 拆成"为什么需要它"、"它做了什么"、"为什么 work"三层。

---

## 一句话总结

**在 simulator 里故意制造各种真实的机器人失误，让一个 VLM 学会看着画面判断"任务完成到百分之几了"，然后把这个判断当 reward 去训 policy。**

就这么简单。剩下的全是 engineering detail。

---

## 为什么需要这个东西

假设你在用 $\pi_0$ 或者 OpenVLA 这种 VLA model 做 robot manipulation。这些 model 是用 imitation learning 训的——supervised learning on demonstration data。 imitation learning 有个天花板：**policy 的上限就是 demonstration 的上限**。demonstration 里的 human operator 技术多好，policy 最多就学到那么好。

RL 理论上能突破这个天花板——让 robot 自己试错，比 demonstration 做得更好。但 RL 在 robot 上一直没大规模 work，原因很朴素：

**你没法定义 reward。**

传统 robot RL 的 reward 设计大概两条路：

### 路线 A：手写 reward function

```python
def reward(state):
    r = 0
    if gripper_close_to_object:
        r += 0.1
    if object_lifted:
        r += 0.3
    if object_at_target:
        r += 1.0
    return r
```

这种 reward 需要你 access simulator state（object position, gripper position 等），real-world 上根本拿不到精确值。而且每个 task 都要手写，完全不 scalable。

### 路线 B：用 VLM 当 reward model

既然 VLM 能看图能理解语言，那就让 VLM 看着 robot 的画面，判断"任务完成了没有"。RoboReward、Robometer、VLAC 这些工作都是这个思路。

问题出在两个地方：

**问题 1：VLM 没见过 failure。**

你拿 GPT-4V 或者 Qwen-VL 去看一张"机器人抓空了"的图，它大概率会说"机器人正在操作物体"——因为它的训练数据里失败案例太少。现有的 robot dataset（DROID、Open X-Embodiment、BridgeData V2）几乎全是成功 demonstration。之前的工作试图用 data augmentation 造假 failure——比如把一条成功 trajectory 从中间截断，当成 failure。这种 pseudo-failure 的物理动态完全是错的：真实 failure 是 gripper 撞到 object、object 从 gripper 滑落、gripper 抓空之后继续移动。截断一条成功 trajectory 根本模拟不出这些物理现象。

**问题 2：reward 只在 episode 结尾给一个 0 或 1。**

假设一个 episode 有 50 步 action。你只在第 50 步告诉 policy"你失败了"。policy 完全不知道是第 10 步的 grasp 出了问题，还是第 40 步的 place 出了问题。这就是 credit assignment problem——**稀疏 reward 让 gradient signal 几乎为零**。

DenseReward 同时解决这两个问题。

---

## 它做了什么

### Step 1：把任务切成 5 个 phase

作者观察到，几乎所有 pick-and-place 类 manipulation 任务都有相同的几何结构：

```
Reach → Grasp → Lift → Move → Place
```

- **Reach**：end-effector 从初始位置移向 object
- **Grasp**：gripper 闭合抓住 object
- **Lift**：object 离开桌面
- **Move**：object 被运到 target 位置
- **Place**：gripper 释放 object

这 5 个 phase 的 boundary 可以从 simulator state 自动检测：
- Grasp 开始 = gripper 接触 object
- Lift 开始 = object 离开桌面
- Move 开始 = end-effector 进入 target 周围 $d_{place}$ 半径
- Place 开始 = end-effector 在 target 位置释放

**完全不需要 human annotation。**

这个 decomposition 的真正价值在于：它让 reward 曲线有了结构。在成功 trajectory 里，reward 从 0 单调上升到 1，每个 phase 占大约 0.2 的区间。这样 reward 就从一个"任意的 scalar"变成了一个"关于 task progress 的几何函数"。

### Step 2：在 simulator 里故意制造 6 种 failure

这是整篇 paper 最聪明的地方。作者在 Isaac Sim 和 RoboSuite 里，通过**在 pipeline 的特定阶段做 targeted perturbation**来制造真实的物理 failure：

| Failure Mode | 怎么制造 | 物理上发生了什么 |
|---|---|---|
| **Success** | 不做任何 perturbation | 正常完成 |
| **Collision** | 关掉 CuRobo 的 collision avoidance | robot 撞到 object 或 table |
| **Miss** | 在 grasp pose 上加 offset | gripper 在空气中闭合，没抓住 object |
| **Fall** | 在 Move phase 加 random rotation | object 在运输中从 gripper 滑落 |
| **Smooth** | 每步注入 Gaussian joint noise | 抖抖嗦嗦地完成任务，suboptimal motion |
| **Recover** | 先 collision，再让 planner replan | 撞了之后重新规划路径，最终成功 |

每种 failure mode 对应一种**特定的 reward 曲线形状**：

- Success：单调上升 `↗`
- Collision：先上升，collision 时达到 peak，然后下降 `↗↘`（山形）
- Miss：同上，山形
- Fall：同上，山形
- Smooth：上升但被 penalty 压低，最终到 1 但曲线偏低
- Recover：先上升，collision 时下降，恢复后重新上升 `↗↘↗`

**这个 reward curve shape design 是核心 insight。** 它让 model 学到的 reward 不只是一个数字，而是一个"故事"——任务进展到哪里了、有没有出错、出错后有没有恢复。

### Step 3：Validity filtering

Perturbation 不保证产生预期 failure。比如 grasp pose 被 offset 了 3cm，但 gripper 可能还是碰巧抓住了 object。所以每种 failure mode 都有 validity check：

- Collision trajectory：robot 必须真的撞了东西，且 object 没被成功 lift
- Miss trajectory：grasp 后 object 位置几乎不变
- Fall trajectory：object 必须先被 lift 起来（证明 grasp 成功了），然后在 transport 中掉落
- Recover trajectory：必须先失败再成功 replan

通不过 check 的 trajectory 直接丢弃，重新 generate。

### Step 4：训一个 VLM 预测 dense reward

Base model：Qwen3-VL-4B-Instruct
输入：task instruction + current frame + 2 frame history
输出：一个 0.000 到 1.000 的 float（3 位小数）
Training：LoRA rank 16，8×H100，10 epochs

System prompt 里明确告诉 model 6 个 phase 的定义和 reward 上升/下降规则，强制它只输出一个 float，不许输出任何解释文字。

最终 dataset：**26,579 个 episode，约 7.56M 个 frame-level sample**。涵盖 DROID（real-world）、Isaac Sim、RoboSuite、LIBERO 四个 source。

---

## 为什么它 work

### Intuition 1：Failure data 比 model size 重要

看 Table 1 的数据：

| Model | MAE |
|---|---|
| Qwen3-VL-8B（8B 参数，无 failure 训练） | 0.293 |
| RoboReward-8B（8B，pseudo-failure） | 0.230 |
| DenseReward（4B，real failure） | **0.081** |

8B model 比 4B model 差 3.6 倍。这说明 reward prediction 的 bottleneck 根本不是 model capacity，是 training data 的质量。你给 4B model 喂真实的 failure trajectory，它就能学到"抓空之后 reward 该下降"这种物理直觉。你给 8B model 喂截断的成功 trajectory，它学到的只是"trajectory 短了就算 failure"——这种 signal 在 real-world 上完全没用。

这和 Language Model 的 scaling law 是一个完全不同的 dimension。LLM 的 scaling 是 "more parameters + more tokens = better"。Reward model 的 scaling 是 "better data structure > more parameters"。

### Intuition 2：Dense supervision 释放了 VLM 的 grounding 能力

VLM 本来就有 visual grounding 能力——它能看出图里 gripper 离 object 多远、object 有没有被抓住。但这些能力在 sparse binary reward 下被浪费了，因为 model 只需要回答"成功 or 失败"，不需要判断精细 progress。

Dense reward 强迫 model 去做精细判断：gripper 离 object 5cm 时 reward 是 0.15，离 2cm 时是 0.25，碰到 object 时是 0.35。这种 fine-grained supervision 把 VLM 的 grounding 能力真正转化成了 reward signal。

### Intuition 3：Phase decomposition 是个极强的 inductive bias

如果没有 phase decomposition，reward 就是一个 free-form regression——model 要从 raw image 学会一个任意形状的 reward 函数。这非常难。

有了 phase decomposition，reward 变成 piecewise monotonic——每个 phase 内单调上升，phase 之间有 transition。model 只需要学会"当前在哪个 phase"+"这个 phase 内的 progress 百分比"。学习难度大幅下降。

这和你在 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmAuSZE) 里讲 transformer 时强调的"positional encoding 给 model 一个 structure prior"是完全类似的思路。Phase decomposition 就是 reward model 的 positional encoding。

### Intuition 4：Mountain-shaped reward curve 教会 model 什么是"退化"

这个设计特别精妙。在 Collision / Miss / Fall 这三种 failure 里，reward 先上升再下降。这意味着 model 学到的 reward function 不只是"越接近目标越高"，还包括"如果出了不可逆的物理事故，reward 要掉下来"。

这种"reward 可以下降"的特性，是 pseudo-failure（截断成功 trajectory）永远学不到的。截断的 reward curve 只会停在某个值，不会下降。而真实 failure 里，robot 在 collision 之后可能继续执行 motion plan，看起来在做 progress，但实际上任务已经搞砸了——reward 必须反映这一点。

### Intuition 5：Recover curve 教会 model "失败不是终点"

Recover trajectory 的 reward 曲线是 `↗↘↗`——先上升，collision 时下降，恢复后重新上升。这教会 model 一个非常重要的概念：**temporary failure 不等于 terminal failure**。

在 real-world RL 中，这个概念极其重要。Robot 在执行中经常会碰到小 collision 或者 grasp 不稳，但只要 recover 了，任务还是可以完成。如果 reward model 把这些 temporary setback 都判成 0，policy 就会过度保守，不敢试错。Recover curve 让 reward model 理解"退一步进两步"是合理的。

---

## 实验结果讲什么

### Dense Reward Prediction（Table 1）

DenseReward 的 MAE 是 0.081，最强 baseline RoboReward-8B 是 0.230。差距接近 3 倍。

值得注意的是 **DROID（real-world data）上的 MAE 是 0.259**，远高于 simulated data 上的 0.04-0.08。这说明 sim-to-real gap 在 reward model 上依然存在——simulator 里生成的 failure 物理动态和 real-world 的 failure 还是有差异。比如 real-world 的 object drop 可能因为 friction、object shape、gripper compliance 等因素表现得和 sim 里不一样。

### MPC 实验（Table 2）

这个实验很直觉：在每个 step，sample 28 个 candidate action（3×3×3 的 3D grid + 1 个 gripper toggle），用 reward model 给每个 candidate 打分，选分最高的执行。

Action 表示：$\boldsymbol{a} = [d_x, d_y, d_z, g]$
- $d_x, d_y, d_z \in \{-0.05, 0, +0.05\}$（米）：end-effector 在 x/y/z 方向的平移
- $g \in \{0, 1\}$：gripper open/close

Metric 是 end-effector 到 object 的最小 3D 距离：
$$d_{min} = \min_t \| p_t^{ee} - p_t^{obj} \|_2$$

- $p_t^{ee}$：timestep $t$ 时 end-effector 的 3D 位置
- $p_t^{obj}$：timestep $t$ 时 object 的 3D 位置
- $\|\cdot\|_2$：L2 norm（欧几里得距离）
- $\min_t$：在整个 episode 里取最小值

DenseReward 平均距离 0.229m，最好 baseline RoboReward-4B 是 0.267m。差距不大但 consistent。这说明 dense reward 的精细判断确实能指导 action selection。

### PPO on LIBERO（Figure 5）

Base policy 是 $\pi_0$（SFT on LIBERO），用 PPO finetune。Reward integration：

$$r_t = \alpha \cdot r_t^{sim} + \beta \cdot r_t^{model}$$

- $r_t^{sim}$：simulator 的 sparse success signal（只在 episode 结束时非零）
- $r_t^{model}$：DenseReward 给的 dense reward，$\in [0, 1]$
- $\alpha = 1.0$：sparse reward 权重
- $\beta = C / T_{max}$：dense reward 权重
- $C = 5$：action chunk size（$\pi_0$ 一次输出 5 个 action）
- $T_{max}$：最大 episode length

为什么 $\beta = C / T_{max}$？算一下：整个 episode 最多有 $T_{max}/C$ 个 chunk，每个 chunk 贡献 $\beta = C/T_{max}$ 的 dense reward，累积最多 $(T_{max}/C) \times (C/T_{max}) = 1$。这和 sparse success signal（成功时 = 1）量级匹配。这是个 reward scale normalization trick——让 dense reward 不会 overwhelm sparse reward。

结果：DenseReward 在 LIBERO-Spatial 和 LIBERO-10 上比 sparse PPO 高，LIBERO-Object 持平。Proof of concept 成立。

### Real-world DSRL（Figure 6）

这是最有说服力的实验。

Setup：
- Franka Research 3 arm + Robotiq 2F-85 gripper
- 两个 task：stack the cups（精细操作）、put ball in basket（OOD object）
- Base policy：frozen $\pi_0$
- RL algorithm：DSRL（在 diffusion policy 的 latent noise space 上做 RL，不 fine-tune policy weights）

DSRL reward integration：
$$r_t = -1 + r_t^{model}$$

- $-1$：每步 step penalty（鼓励快完成）
- $r_t^{model} \in [0, 1]$：DenseReward 给的 progress reward
- Terminal step 特殊：$r_T = r_T^{model}$（如果完成，不加 step penalty）

Training budget：
- stack the cups：20k steps ≈ 20 次 real-world rollout
- put ball in basket：10k steps ≈ 10 次 real-world rollout

结果：
- stack the cups：40% → 80%（+40%）
- put ball in basket：30% → 70%（+40%）

**10-20 次 real-world rollout 就能把 success rate 翻倍。** 这就是 dense reward 的价值——每次 rollout 的每一步都提供 gradient information，sample efficiency 极高。

DSRL 的关键设计：不 fine-tune policy weights，而是学习 steer diffusion action head 的 latent noise。这让 policy 改进时仍 close to demonstration prior，避免 catastrophic forgetting。在 real-world 里，policy 退化一次可能就要重新 calibration，代价极大。DSRL + DenseReward 的组合让改进"安全"且"高效"。

### Ablation：Failure data 有多重要

| Setting | MAE |
|---|---|
| 有 failure data | 0.0809 |
| 无 failure data | 0.1312 |

去掉 failure data，MAE 从 0.08 涨到 0.13。这证明 failure trajectory 是 critical signal——没有 failure，model 不知道"task progress 可以退化"，不知道 mountain-shaped curve 长什么样。

### Ablation：History frame 数量

| History frames | MAE |
|---|---|
| 0 | 0.096 |
| 1 | 0.088 |
| 2 | 0.081 |
| 3 | 0.086 |

0→1→2 持续提升，2→3 反而变差。这很符合 VLM 的 temporal reasoning 能力——给它一点 temporal context 有助于判断 motion direction 和 phase transition，但给太多会引入 visual noise，model 处理不了。

---

## 更深层的联想

### 和 AlphaGo value network 的类比

[AlphaGo](https://www.nature.com/articles/nature16961) 有 policy network + value network。Value network 估计当前棋局有多好——本质就是 reward model。DenseReward 在做类似的事：估计当前 robot state 的 task progress。

但区别在于：AlphaGo 的 value 是从 terminal reward（胜负）通过 self-play 倒推出来的。DenseReward 的 reward 是从 phase geometry + failure mode **直接合成**的。合成的好处是 reward shape 可控、可解释；坏处是可能不完全 align 真实任务的成功标准。

这指向一个更深的 question：**reward 应该是 learned from experience 还是 designed from structure？** AlphaGo 走了 learned 路线（self-play），DenseReward 走了 designed 路线（phase decomposition + perturbation）。两条路线在 robot learning 里可能需要融合——用 designed reward 做 bootstrap，用 experience 做 refine。

### 和 RLHF 的类比

[RLHF](https://arxiv.org/abs/2203.02155) 的 reward model 是从 human preference data 训的——human 标注"response A 比 response B 好"。Robometer 在 robot 上做了类似的事（preference pair from suboptimal rollout）。但 DenseReward 走了一条完全不同的路：**用 simulator 的物理 ground truth 直接生成 dense label**。

这其实是 robot learning 相对于 NLP 的一个结构性优势——robot 有 simulator，可以 access ground truth state。NLP 没有 simulator，只能靠 human judgment。所以 robot reward model 理论上可以比 NLP reward model 更精确、更 scalable。

### 和 Synthetic Data Scaling 的关系

DenseReward 本质上是在做 **synthetic data scaling for reward learning**。它不依赖 human annotation，不依赖 real robot rollout，完全在 simulator 里 scale。这和 [Constitutional AI](https://arxiv.org/abs/2212.08073) 用 AI generate feedback data 是同一个 pattern，只不过 DenseReward 用 physics simulator 代替了 language model 来 generate ground truth。

这个 pattern 的 power 在于：**只要 simulator 足够好，data 就可以无限 scale**。而 simulator 的 quality 是可以独立提升的——Isaac Sim 越来越真实，sim-to-real gap 在缩小。这意味着 DenseReward 这种方法会随着 simulator 进步自动变强。

### 和 World Model 路线的对比

另一条做 dense reward 的路线是 world model——predict future observation，然后从 predicted future 计算 reward。[Dreamer](https://arxiv.org/abs/1912.01603)、[Visual Foresight](https://arxiv.org/abs/1812.00568)、[CTRL-World](https://arxiv.org/abs/2510.10125) 走的是这条路。

World model 路线的优势：reward 是 emergent 的，不需要 hand-design。劣势：contact-rich manipulation（抓取、放置）的 future prediction 极不可靠——物体接触瞬间的物理动态太复杂，video prediction model 很难准确预测。

DenseReward 绕过了这个问题：它不 predict future，直接从 current observation 判断 progress。这更 robust，因为不需要预测物理接触的结果，只需要判断当前状态。代价是需要 explicit 的 reward label（通过 phase decomposition + failure synthesis 提供）。

两条路线本质是 **density of supervision** 和 **reliability of prediction** 的 trade-off：
- World model：dense but unreliable（contact-rich 场景下）
- Direct reward model：reliable but needs explicit label

DenseReward 选了第二条，并用 synthetic data pipeline 解决了 label 获取问题。

### Reward Hacking 的隐患

DenseReward 没有显式讨论 reward hacking。当 policy 用 learned reward model 做 RL 时，policy 可能找到 reward model 的盲区——某些视觉上看起来像 progress 但实际不是 task completion 的 action sequence。

这在 [Inverse Scaling in RL](https://arxiv.org/abs/2305.18730) 等 work 里有讨论。DenseReward 的 PPO 实验里用 $\alpha \cdot r_t^{sim} + \beta \cdot r_t^{model}$ 的混合 reward 来 mitigate 这个问题——simulator 的 sparse success signal 作为 ground truth anchor，dense reward 只做 shaping。这是个合理的 design，但 long-horizon task 上 reward hacking 仍可能是个问题。

### 5-Phase Decomposition 的泛化性

Reach-Grasp-Lift-Move-Place 这个 decomposition 对 pick-and-place 任务很自然，但对其他任务呢？

- **Tool use**（用锤子钉钉子）：没有明显的 Lift-Move-Place 结构
- **Long-horizon**（做三明治）：多个 pick-and-place 串起来，需要 hierarchical phase decomposition
- **Bimanual**（双手协作）：两个 end-effector 各自有 phase，还需要协调
- **Articulated object**（开柜门、拉抽屉）：phase 结构完全不同

paper 在 limitations 里承认了这个。但我觉得 phase decomposition 这个 idea 本身是可扩展的——只是具体 phase 定义需要 per-task 设计。未来可能用 LLM 自动 generate phase decomposition，就像 [Code as Policies](https://code-as-policies.github.io/) 那样。

### 和 $\pi_0$ / $\pi^*$ 的关系

Physical Intelligence 最近发了 [$\pi^*$](https://arxiv.org/abs/2511.14759)，声称 VLA 可以 "learn from experience"。DenseReward 提供的 dense reward signal 正是这种 "learn from experience" 的关键 enabler。$\pi^*$ 如果只用 sparse success reward，sample efficiency 会很差。加上 DenseReward 这种 dense reward model，real-world RL 的 rollout budget 可能从几百次降到几十次。

这指向一个可能的 future direction：**VLA + dense reward model + DSRL 的组合**，成为 real-world robot RL 的标准 stack。VLA 提供 strong behavioral prior，dense reward model 提供 fine-grained feedback，DSRL 在 latent space 上做 safe adaptation。

---

## 最终直觉

DenseReward 教给我三件事：

1. **Data structure 比 model size 重要**。4B + structured failure data 吊打 8B + unstructured success data。这和你反复强调的 "data quality > model size" 完全一致。

2. **Phase decomposition 是个被低估的 inductive bias**。它把 reward modeling 从 free-form regression 变成 piecewise monotonic regression，结构性降低学习难度。这个 idea 可以迁移到很多其他 sequential decision making 问题。

3. **Synthetic data + simulator 是 robot learning 的结构性优势**。NLP 只能靠 human judgment or model judgment 来 generate reward data。Robot 有 physics simulator，可以 access ground truth state，可以无限 scale synthetic data。DenseReward 是这个优势的一个具体体现。

paper 代码和数据都开源了：https://dense-reward.github.io/

如果你想 reproduce，我建议从 Appendix C 的 MPC experiment 开始——它最简单，不需要训 policy，只需要 reward model + Isaac Lab。跑通 MPC 之后再去搞 PPO/DSRL，learning curve 会平滑很多。

## References

- DenseReward: https://dense-reward.github.io/
- GraspNet: https://graspnet.net/
- CuRobo: https://curobo.org/
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- DROID dataset: https://droid-dataset.github.io/
- LIBERO: https://lifelong-robot-learning.github.io/libero/
- $\pi_0$: https://www.physicalintelligence.company/blog/pi0
- $\pi^*$: https://arxiv.org/abs/2511.14759
- DSRL: https://arxiv.org/abs/2506.15799
- PPO: https://arxiv.org/abs/1707.06347
- SAC: https://arxiv.org/abs/1801.01290
- AlphaGo: https://www.nature.com/articles/nature16961
- RLHF: https://arxiv.org/abs/2203.02155
- Constitutional AI: https://arxiv.org/abs/2212.08073
- Dreamer: https://arxiv.org/abs/1912.01603
- Visual Foresight: https://arxiv.org/abs/1812.00568
- CTRL-World: https://arxiv.org/abs/2510.10125
- Code as Policies: https://code-as-policies.github.io/
- Karpathy - Let's build GPT: https://www.youtube.com/watch?v=kCc8FmAuSZE
- Karpathy - Intro to LLMs: https://www.youtube.com/watch?v=zjkBMFhXj6Y
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- BridgeData V2: https://github.com/rail-berkeley/bridge_data_v2
- Molmo: https://arxiv.org/abs/2409.12146
- ms-swift: https://github.com/modelscope/swift

---

# DenseReward：通过 Failure Synthesis 学习 Dense Reward

Karpathy 你好，这篇 paper 解决的是 robotic manipulation RL 中一个非常具体的 bottleneck：**怎么搞到一个又 dense 又 informative 的 reward signal**。它的核心 insight 其实很简单——**真正的 failure 数据比 pseudo-failure 重要得多，dense per-timestep supervision 比单纯的 sparse success label 信息量大得多**。让我把整篇 paper 拆开讲。

---

## 1. 这篇 paper 想解决什么问题

当前 robot manipulation 的 RL pipeline 有两个被忽视但极关键的问题：

### Problem 1: Failure data 稀缺且"假"

现有 VLA dataset（Open X-Embodiment, DROID, BridgeData V2）几乎全是 successful demonstration。最近的工作（RoboReward, Robometer）试图通过 data augmentation 解决，但本质上是把 successful trajectory 截断或 relabel 当成 failure——这种 pseudo-failure 在物理上根本不真实。真实的 robot failure 包含 collision、missed grasp、object drop、recovery，这些是 augmentation 抓不到的。

### Problem 2: Reward 太 sparse

很多 reward model 在 trajectory 结束时才输出一个 binary $p \in [0, 1]$。这导致严重的 **credit assignment problem**：policy 在 50 步 action 之后才知道这一整段 trajectory 是成功还是失败，完全不知道是哪一步的哪个 action 把事情搞砸的。

DenseReward 同时攻击这两点。

---

## 2. 方法：Phase Decomposition + Failure Synthesis

### 2.1 Dense Reward 的数学形式

传统 setup：trajectory $\tau$ 结束时给一个 scalar $p \in [0,1]$。

DenseReward 的 setup：对每个 trajectory $\tau = \{l, \mathbf{o}_{1:T}, \mathbf{r}_{1:T}\}$，其中
- $l$ 是 language instruction
- $\mathbf{o}_{1:T} = (o_1, o_2, ..., o_T)$ 是 image observation sequence
- $\mathbf{r}_{1:T} = (r_1, r_2, ..., r_T)$ 是 per-timestep dense reward，每个 $r_t \in [0, 1]$

下标 $t \in \{1, ..., T\}$ 表示 timestep，$T$ 是 episode length。每个 $r_t$ 反映这一帧时刻的 task progress。

### 2.2 Five-Phase Decomposition

作者把 manipulation 任务硬切成 5 个 phase，这是整个 pipeline 的几何骨架：

| Phase | 含义 | Phase boundary 判定 |
|---|---|---|
| 1. Reach | end-effector 移向目标 object | 初始状态自动定义 |
| 2. Grasp | gripper 闭合抓 object | gripper 接触 object 时 |
| 3. Lift | object 离开 table | object 离开桌面时 |
| 4. Move | 运输 object 到目标位置 | end-effector 进入 target 周围 $d_{place}$ 半径 |
| 5. Place | 释放 object 到目标 pose | end-effector 在目标位置释放 |

这个 decomposition 让 reward 的曲线变成 piecewise monotonic——每个 phase 内 reward 单调上升，phase 之间有清晰的 transition。这非常关键，因为它把"task progress"从一个模糊的语义概念变成一个可量化的几何量。

### 2.3 Automated Pipeline（成功轨迹怎么生成）

这个 pipeline 是 fully automatic 的，**不需要任何 human annotation**：

1. **Scene initialization**: 随机摆放 object 和 container，确保 spatial configuration 多样
2. **Grasp pose prediction**: 用 GraspNet 从 multi-view RGB-D 中预测最多 $N=50$ 个 grasp pose candidate
3. **Motion planning**: 把 candidate 传给 CuRobo（NVIDIA 的 collision-aware motion planner），选一个 feasible candidate
4. **Execution**: 执行 6 个 motion segment 对应 5 个 phase
5. **Phase boundary detection**: 从 simulator state 自动检测，不需要人工标注

这里 GraspNet 提供 grasp candidate diversity，CuRobo 提供 collision-aware feasibility check。整个 pipeline 是 failure-agnostic 的——它生成的是 unperturbed success trajectory。

### 2.4 Failure Synthesis（失败轨迹怎么生成）

这是 paper 的核心 contribution。作者定义了 6 种 failure mode，每种通过在 pipeline 的特定阶段做 targeted perturbation 实现：

| Failure Mode | Perturbation 方法 | Reward 曲线形状 |
|---|---|---|
| 1. Success | 无 perturbation | 单调上升 |
| 2. Collision | 关闭 collision avoidance，强制走 infeasible path | 山形：上升→collision 事件→下降 |
| 3. Miss | 在 grasp target pose 上加 offset，gripper 在空气中闭合 | 山形：上升→miss 事件→下降 |
| 4. Fall | 在 Move phase 加 random rotation perturbation，object 在运输中掉落 | 山形：上升→peak→drop→下降 |
| 5. Smooth | 每个 timestep 注入 small Gaussian joint noise | scaled penalized reward，suboptimal motion |
| 6. Recover | 先 collision，再让 motion planner replan 一条 clear path | 下降→恢复后继续上升 |

这里的 reward 曲线设计非常关键：
- Mountain-shaped curve 反映"partial progress that is ultimately unsuccessful"——机器人确实做了一些 progress，但被某个 failure event 打断
- Recovery 曲线特别有意思：reward 在 collision 时 drop，然后**恢复后继续上升**，这捕捉了"failure 不是 irreversible 的"

### 2.5 Validity Filtering

Perturbation 不一定保证产生预期的 failure——比如 grasp pose 被 offset 了，但 gripper 可能还是偶然抓住了 object。所以需要 validity check：

| Check | 适用 mode | 条件 |
|---|---|---|
| Planning | All | 必须能找到 feasible grasp 或 motion plan |
| Grasp and lift | Success, Recover | grasp 后 object 必须高于 table 一个 threshold |
| Holding | Fall | 运输时 object 必须高于更严格的 threshold（确保是稳定 holding 而非 dragging） |
| Collision | Collision | robot 必须真的 displace object 或撞到 scene，且 object 没被成功 lift |
| Miss | Miss | grasp 后 object 位置和姿态几乎不变 |
| Final placement | Success, Recover | object 最终在 target container 一个 distance threshold 内 |
| Recovery | Recover | 必须先 failed，再成功 replan |

这一步很重要——否则 mislabeled failure 会污染训练数据。

---

## 3. Dataset Statistics

最终生成的 dataset 统计：

| Source | Episodes |
|---|---|
| DROID (real) | 2,986 (1500 success + 1486 failure) |
| Isaac Sim | 12,481 (2303 success + 2511 collision + 2603 miss + 2295 fall + 2514 smooth + 255 recover) |
| RoboSuite | 9,287 (3366 success + 5921 failure) |
| LIBERO | 1,825 (across Spatial/Object/Goal/10) |
| **Total** | **26,579 episodes**，约 7.56M frame-level samples |

注意 Isaac Sim 中 Recover 只有 255 个——这种 trajectory 需要先失败再成功 replan，相对难自动生成。

---

## 4. DenseReward Model

### 4.1 架构

基于 **Qwen3-VL-4B-Instruct** fine-tune。输入：
- Task instruction $l$（text）
- Current frame $o_t$
- Historical frames $o_{t-1}, o_{t-2}$（default 2 frames history）

输出：scalar reward $r_t \in [0, 1]$，**保留 3 位小数**（这个细节很重要，让 model 学到 fine-grained progress difference）

### 4.2 Training

- Framework: ms-swift
- Method: LoRA fine-tuning, rank = 16
- 8 张 H100 GPU
- batch size = 32
- 10 epochs

### 4.3 System Prompt 设计

paper 附录里给出了完整的 system prompt，强制 model 只输出一个 float 值（3 位小数），明确告诉它 6 个 subtask phase 的语义定义、reward 上升/下降的语义规则。这是一个很典型的 VLM-as-scalar-predictor 的 setup——把 reward prediction 当成 instruction following 任务，但 ground truth 是连续值。

---

## 5. 实验

### 5.1 Dense Reward Prediction（核心 benchmark）

Metric: MAE (Mean Absolute Error)

| Model | Overall | DROID | Isaac Sim | RoboSuite | LIBERO |
|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | 0.289 | 0.532 | 0.285 | 0.195 | 0.478 |
| Qwen3-VL-8B-Instruct | 0.293 | 0.538 | 0.305 | 0.180 | 0.502 |
| Molmo2-4B | 0.282 | 0.506 | 0.282 | 0.187 | 0.478 |
| Molmo2-8B | 0.335 | 0.480 | 0.307 | 0.303 | 0.455 |
| RoboReward-4B | 0.275 | 0.534 | 0.269 | 0.179 | 0.470 |
| RoboReward-8B | 0.230 | 0.484 | 0.185 | 0.172 | 0.431 |
| Robometer | 0.366 | 0.521 | 0.328 | 0.345 | 0.468 |
| **DenseReward (Ours)** | **0.081** | **0.259** | **0.081** | **0.051** | **0.044** |

**Intuition**: 注意几个现象——
- 更大的 model（4B → 8B）没有提升，反而 Qwen3-VL-8B 在某些 source 上更差。这说明 reward prediction 不是 model size 问题，是 supervision signal 问题
- Robometer（用 preference pair 训练）反而最差（0.366），说明 pseudo-failure 不仅没用还有害
- DenseReward 在 simulated data（Isaac, RoboSuite, LIBERO）上几乎完美（0.04-0.08），但在 real-world DROID 上仍有 0.259——sim-to-real gap 在 reward model 上依然存在

### 5.2 Model Predictive Control (MPC)

Setup: 在每个 decision step，sample 28 个 candidate action（27 个 spatial direction + 1 个 gripper toggle），用 reward model 给每个 candidate transition 打分，选 highest score 的 action 执行。

Action 表示：$\boldsymbol{a} = [d_x, d_y, d_z, g]$
- $d_x, d_y, d_z$: end-effector translation offset，grid $\{-d, 0, +d\}$ 其中 $d = 0.05$ m
- $g \in \{0, 1\}$: gripper open/close

Metric: 最小 end-effector-to-object 距离
$$d_{min} = \min_t \| p_t^{ee} - p_t^{obj} \|_2$$
其中 $p_t^{ee}$ 是 timestep $t$ 时 end-effector 3D 位置，$p_t^{obj}$ 是 object 3D 位置。下标 2 是 L2 norm。$\min_t$ 在整个 episode 上取最小。

| Model | Can | Cup | Lemon | Avg |
|---|---|---|---|---|
| RoboReward-4B | 0.199 | 0.307 | 0.295 | 0.267 |
| RoboReward-8B | 0.314 | 0.270 | 0.317 | 0.300 |
| VLAC-2B | 0.316 | 0.346 | 0.380 | 0.347 |
| VLAC-8B | 0.351 | 0.360 | 0.363 | 0.358 |
| **DenseReward** | **0.219** | **0.181** | **0.288** | **0.229** |

**Intuition**: 这里有个有趣现象——RoboReward-4B 在 Can 上是 0.199，比 DenseReward 的 0.219 还好。但 DenseReward 在另外两个 task 上明显胜出。这暗示 reward model 的 task-generalization 不是均匀的，dense supervision 让 model 学到了更通用的 task progress 概念。

### 5.3 PPO Fine-tuning on LIBERO

Base policy: $\pi_0$（已 supervised fine-tune 在 LIBERO 上）
Algorithm: PPO
DenseReward integration:

$$r_t = \alpha \cdot r_t^{sim} + \beta \cdot r_t^{model}$$

变量含义：
- $r_t^{sim}$: simulator 提供的 sparse reward（episode 末尾才非零）
- $r_t^{model}$: DenseReward 在 chunk 末尾给出的 dense reward，$\in [0, 1]$
- $\alpha = 1.0$: sparse reward 权重
- $\beta = C / T_{max}$: dense reward 权重
- $C = 5$: action chunk size
- $T_{max}$: maximum episode length

为什么 $\beta = C/T_{max}$？这是为了让 dense reward 在整个 episode 上的累积贡献（最多 $T_{max}/C$ 个 chunk × 每个 $\beta$ = $T_{max}/C \cdot C/T_{max} = 1$）与 episode-level sparse success signal 同量级。这是一个 reward shaping trick，保持 reward scale comparable。

结果（图 5）：DenseReward 在 LIBERO-Spatial 和 LIBERO-10 上比 sparse PPO baseline 高，在 LIBERO-Object 上持平。这是 RL finetuning 的一个 proof-of-concept。

### 5.4 Real-world RL with DSRL

这是最有说服力的实验。Setup:
- Franka Research 3 arm + Robotiq 2F-85 gripper
- ZED 2i exterior camera + ZED mini wrist camera
- 两个 task: (1) stack the cups（精细 object interaction）(2) put ball in basket（OOD object）
- $\pi_0$ 作为 frozen base policy，DSRL 学习在 diffusion action head 的 latent noise space 上做 steer

DSRL reward integration:
$$r_t = -1 + r_t^{model}$$
其中 $-1$ 是 DSRL 的 step penalty。Terminal step 特殊处理：$r_T = r_T^{model}$（如果任务完成则没有 step penalty）。

Training budget:
- stack the cups: 20k steps ≈ 20 real-world rollout
- put ball in basket: 10k steps ≈ 10 real-world rollout

结果：
- stack the cups: 40% → 80% success rate（+40%）
- put ball in basket: 30% → 70% success rate（+40%）

**Intuition**: 这是 dense reward 真正起作用的地方。real-world RL 的 bottleneck 是 sample efficiency——每个 rollout 都极贵（机器人时间 + 物理磨损）。dense reward 让每一步 rollout 都提供梯度信息，而不只是末尾一个 binary 信号。$-1$ 的 step penalty 配合 dense reward 让 agent 同时学会"快"和"对"。

DSRL 的关键 trick：不 fine-tune policy weights，而是在 diffusion policy 的 latent noise space 上做 RL。这让 policy 改进时仍 close to demonstration prior，避免 catastrophic forgetting——这在 real-world 中特别重要，因为 policy 退化一次就要重新整个 calibration。

---

## 6. Ablation Studies

### 6.1 Failure Data 的作用

| Setting | MAE |
|---|---|
| w/ Failure Data | 0.0809 |
| w/o Failure Data | 0.1312 |

去掉 failure data 后 MAE 几乎翻倍。这证明 failure trajectory 是 dense reward supervision 的 critical signal——没有 failure 的 model 不知道什么是"task progress 退化"，不知道 mountain-shaped curve 该长什么样。

### 6.2 Historical Frames 数量

| # History Frames | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| MAE | 0.096 | 0.088 | 0.081 | 0.086 |

- 0 → 1 frame: -0.008（-temporal context 重要）
- 1 → 2 frames: -0.007（继续提升）
- 2 → 3 frames: +0.005（反而变差，过度 history 引入 noise）

这是个很经典的 U-curve 现象。2 frame 是 performance/cost 的 sweet spot。

---

## 7. 我的 Intuition 与相关联想

### 7.1 为什么 DenseReward Work

Karpathy 你可能会问：为什么 phase decomposition + failure synthesis 比单纯用 VLM 直接打分好这么多？我的理解是：

1. **VLM 没见过失败轨迹**：现有 VLM 训练数据主要是 web image + text，其中"机器人把东西搞砸"的图极少。failure 是个 long-tail distribution，必须 explicit 注入。
2. **Dense supervision 释放了 VLM 的 in-context learning 能力**：从 system prompt 看，paper 显式告诉 model 6 个 phase 和 reward 上升下降规则。VLM 实际是在做 in-context reasoning + dense regression。
3. **Phase boundary 给 reward 一个几何 prior**：reward 不再是任意 scalar，而是关于 3D 空间中 end-effector 和 object 关系的几何函数。这让 VLM 的 grounding 能力真正发挥作用。

### 7.2 与相关工作的对比

- **RoboReward (2026)**：trajectory-level binary label，靠 truncation 造 failure——本质上是 success trajectory 的子前缀，没有真正的物理 failure
- **Robometer (2026)**：preference pair from suboptimal rollout——比 RoboReward 强，但还是 pseudo-failure
- **VLAC (2025)**：vision-language-action-critic，end-to-end actor-critic with VLM——更接近 online RL，但 reward signal 不显式
- **RoboDopamine (2025)**：process reward modeling for high-precision manipulation——概念上最接近，但侧重点不同
- **SARM (2025)**：stage-aware reward modeling for long-horizon——和 phase decomposition 类似思路

### 7.3 与 Foundation Model 思路的联系

Karpathy 你在 [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhXj6Y) 里讲过 model 的 "first principle" 是 next token prediction。DenseReward 实际上是把 reward modeling 当成一个 regression-style next-token-prediction——让 VLM 输出一个 float。这其实是一种"reward as language"的思路。

更广义地，这与 [Constitutional AI](https://arxiv.org/abs/2212.08073) 中 RLAIF 的思路相通——把人类无法直接 supervise 的 reward 用 model 来 generate。但 DenseReward 走得更远：它不只 generate reward，还 generate 了 training data（failure trajectory + dense label）本身。这其实是 **synthetic data scaling for robot reward learning** 的一个范例。

### 7.4 与 World Model 的关系

paper 在 related work 里提到了 [Dreamer](https://arxiv.org/abs/1912.01603)、[Visual Foresight](https://arxiv.org/abs/1812.00568)、[CTRL-World](https://arxiv.org/abs/2510.10125) 这些 world model 方法。World model 路线是：predict future observation → 计算 reward。DenseReward 反过来：directly predict reward from current observation。这两条路线本质是 **density of supervision** 的 trade-off：
- World model：dense supervision 但 contact-rich 任务下 future prediction 不可靠
- Direct reward model：direct supervision 但需要 explicit failure data

DenseReward 选了第二条，并 explicit 解决了 failure data 缺失问题。

### 7.5 与 AlphaGo 类比

这让我想到 [AlphaGo](https://www.nature.com/articles/nature16961) 的 two-network 设计：policy network + value network。Value network 本质就是 estimate 当前局面有多好。DenseReward 在某种意义上是 robotic manipulation 的 value network——但它不 estimate expected return，而是 estimate 当前的 task progress。

但区别也很明显：AlphaGo 的 value 是从一个明确的胜负 terminal reward 倒推的，而 DenseReward 的 reward 是从 phase geometry + failure mode 直接合成的。这是一种 **synthetic value**——synthetic 的好处是可以精确控制 reward shape，坏处是可能不完全 align 真实 task 的成功标准。

### 7.6 Open Questions

读完后我有几个 question：

1. **Reward shape 的泛化性**：phase decomposition 假设 task 是 Reach-Grasp-Lift-Move-Place。但对 long-horizon、tool use、bimanual 任务，这个 5-phase 模型是否还成立？paper 在 limitations 里提到了这个。
2. **Real-world DROID MAE = 0.259**：尽管 sim data 上接近完美，real-world 仍有显著 error。是否需要 real-world failure data 来 close gap？作者用 real-world failure (1486 episodes from DROID) 训练，但仍然不够。
3. **Reward Hacking 风险**：dense reward model 是 learned，policy 可能找到 reward model 的盲区。paper 没有显式讨论 reward hacking。这一点 [Inverse Scaling in RL](https://arxiv.org/abs/2305.18730) 之类的工作有相关讨论。
4. **Recovery trajectory 只有 255 个**：从 dataset statistics 看，Isaac 中 Recovery 只有 255 个，远少于其他 mode。这可能让 model 对 recovery 行为的 estimate 不够鲁棒——而 real-world 中 recovery 是常见场景。Fig 8 显示 DenseReward 能 capture recovery，但 sample 量限制值得注意。
5. **History frame 的 noise 增长**：3 frame 反而比 2 frame 差。是否是因为 VLM 对 longer temporal reasoning 能力不足？这和 [Long Video Understanding](https://arxiv.org/abs/2506.10247) 中的 observation 类似——VLM 在长 video 上很快 degrade。

---

## 8. 我的 Verdict

这篇 paper 的 contribution 是**一个具体的、可复现的、scalable 的 dense reward data pipeline**。它不是理论突破，是 engineering + data 上的"组合拳"。它说服我两件事：

1. **Failure data 比 model size 重要**——8B VLM + pseudo-failure < 4B VLM + real-failure
2. **Phase decomposition 是个被低估的 inductive bias**——它让 reward modeling 从一个 free-form regression 变成 piecewise monotonic regression，结构性大幅降低学习难度

对于做 robot VLA RL 的人，这是一篇 must-read。对做 foundation model 的人，它提示了 **synthetic data + structured supervision** 这个 pattern 的力量。

---

## References

- DenseReward 项目主页: https://dense-reward.github.io/
- GraspNet (grasp pose prediction): https://graspnet.net/
- CuRobo (NVIDIA motion planner): https://curobo.org/
- Qwen3-VL (base model): https://arxiv.org/abs/2511.21631
- DROID dataset: https://droid-dataset.github.io/
- LIBERO benchmark: https://lifelong-robot-learning.github.io/libero/
- RoboSuite: https://robosuite.ai/
- $\pi_0$ VLA model: https://www.physicalintelligence.company/blog/pi0
- DSRL (Diffusion Steering RL): https://arxiv.org/abs/2506.15799
- PPO: https://arxiv.org/abs/1707.06347
- SAC (Soft Actor-Critic, DSRL-SAC 用的 base): https://arxiv.org/abs/1801.01290
- ms-swift (fine-tuning framework): https://github.com/modelscope/swift
- AlphaGo: https://www.nature.com/articles/nature16961
- Constitutional AI (RLAIF): https://arxiv.org/abs/2212.08073
- Dreamer (world model RL): https://arxiv.org/abs/1912.01603
- Visual Foresight: https://arxiv.org/abs/1812.00568
- Karpathy - Intro to LLMs: https://www.youtube.com/watch?v=zjkBMFhXj6Y
- Karpathy - Let's build GPT: https://www.youtube.com/watch?v=kCc8FmAuSZE
- RLHF original paper: https://arxiv.org/abs/2203.02155
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- BridgeData V2: https://github.com/TFJ-NTU/bridgedata-v2
- Molmo (vision-language model): https://arxiv.org/abs/2409.12146
- AlphaZero (related value/policy decomposition): https://www.nature.com/articles/nature24270

---

如果你想 dive deeper，我特别建议看 Appendix C（MPC 细节）和 Appendix E（DSRL config，包括 $\gamma = 0.995$、target entropy = 0、4 critics 这些 SAC 经典超参）。real-world DSRL setup 是个工程范本，可以直接拿来做自己的实验。
