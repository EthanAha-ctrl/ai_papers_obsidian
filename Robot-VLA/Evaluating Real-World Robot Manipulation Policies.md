---
source_pdf: Evaluating Real-World Robot Manipulation Policies.pdf
paper_sha256: 6a9481f3c0fc43fa6736e5f5fc8ddb72ebf05aad7dbd0ff31721e4d6bffec9e5
processed_at: '2026-08-04T05:33:01-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SIMPLER 用人话版

## 这 paper 想干啥

你训了个 robot policy, 比如 RT-1, 现在要测它到底行不行。

Real robot 上跑 100 个 trial, 几个 PhD 蹲在 robot 旁边两星期, 花好几千度电, 最后得到一个 success rate 数字。下个 checkpoint 又来一遍。下个 paper 又来一遍。这套玩法在 RT-1 时代还撑得住, 等到 Open-X-Embodiment 这种百万 episode 数据集, 上百个 task, 你根本 eval 不完。

那 sim 里跑不就行了? 问题是 sim 长得跟 real 不一样, 你拿 real 训的 policy 扔进 sim, 它就懵了, success rate 一塌糊涂, 根本没法用来比较 checkpoint。

这 paper 说: 其实你只要把 sim 调到"刚好够像", real 训的 policy 在 sim 里跑出来的 ranking, 就能跟 real 里的 ranking 强相关。这样 sim 就变成一个 cheap proxy, 你可以一天跑 1000 个 trial, 用来筛选 checkpoint, 偶尔去 real 上做 ground truth 验证。

项目主页: https://simpler-env.github.io

---

## 为啥这事难

sim 和 real 之间有两个 gap, 都是 first-order 的, 都会直接把 success rate 搞砸:

### Control gap

Real robot 上你说"end-effector 往前挪 5cm", controller 用某套 PD 参数, 实际走出来的轨迹是 A。Sim 里你用同一套 PD 参数, 因为 PhysX 的 contact model、friction、solver 都不一样, 走出来的轨迹是 B。A 和 B 差个几厘米, 就 grasp 不到 coke can, 任务直接挂。

Figure 4 里那张图特别直观: SysID 之前 sim 里的 robot 手伸到 can 旁边就停了, 没抓到; SysID 之后一模一样的 action sequence, 顺利抓起来。

### Visual gap

Policy 是吃 image 的。你 sim 里 render 出来的图, 跟 real 的图光照不同、背景不同、object 的 texture 不同。Real 训的 policy 看到这种 OOD image, 行为就跑偏。这不是小事, Table VIII 里 Octo-Base 换个 robot arm 颜色, success rate 从 29% 直接掉到 0%。

---

## 他们怎么处理 control gap

做法简单粗暴: 从已有的 demonstration dataset (不用采新数据) 里拿一段 action trajectory, 在 real 里对应有一段 end-effector pose trajectory。Sim 里 open-loop replay 同样的 action, 看走的 pose 轨迹差多少, 然后用 simulated annealing 调 PD 参数让差距最小。

损失函数:

$$
\mathcal{L}_{\text{sysid}} = \underbrace{\frac{1}{T}\sum_{i=1}^T \|\mathbf{x}_i - \mathbf{x}_i'\|_2}_{\text{平移误差}} + \underbrace{\frac{1}{T}\sum_{i=1}^T \arcsin\left(\frac{1}{2\sqrt{2}}\|R_i - R_i'\|_F\right)}_{\text{旋转误差}}
$$

变量意思:
- $\mathbf{x}_i \in \mathbb{R}^3$: real robot 第 $i$ 步 end-effector 的位置
- $R_i \in \text{SO}(3)$: real robot 第 $i$ 步的 3×3 旋转矩阵
- $\mathbf{x}_i', R_i'$: sim 里同样的量
- $\|\cdot\|_F$: Frobenius norm, 矩阵所有元素平方和开根号
- 那个 $\arcsin$ 项: 两个旋转之间 geodesic distance 的标准近似, 单位是弧度, 越小代表两个朝向越接近

为啥不用梯度下降? 因为 PD 参数到轨迹误差的关系不光滑, 而且 simulator 是黑盒, 求不了导。Simulated annealing 这种 zero-order 优化刚好合适, 三轮 coarse-to-fine 就能收敛。

效果: Table II 里 control loss 从 0.267 降到 0.131, MMRV 从 0.070 降到 0.031。把 PD 参数搞对, correlation 直接翻倍。

---

## 他们怎么处理 visual gap

两部分:

### Green screening

你把 real video 第一帧拿来, 用 cleanup.pictures 把 robot 和 foreground objects 擦掉, 得到一张干净 background。Sim 里渲染时, 前景用 sim render (保证物理一致), 背景直接贴 real 的图。一行公式:

$$
I' = M \odot I_{\text{sim}} + (1-M) \odot I_{\text{real}}
$$

$M$ 是 sim 里 query 出来的 foreground mask (SAPIEN 直接给你), $\odot$ 是逐元素乘。前景用 sim, 背景用 real, 合起来。

这招在 navigation 里也有人用过, 但 manipulation 这篇是第一个系统化的。好处是 sim 渲染不用去 reconstruct 整个房间的几何, 只需要前景几个 object 的几何, 大大降低 asset 制作成本。

### Texture matching

只换背景不够, 前景的 object 和 robot arm 也得像。流程:

1. 用 SAM (Segment Anything) 从 real image 把 object抠出来
2. 把 sim object 大致 align 到 real 的位置
3. 用 Nvdiffrast 可微渲染精细优化 pose, 让 sim 的 segment mask 跟 real 完全对齐
4. 把 real 的 RGB "unproject" 到 sim 的 mesh 上当 texture
5. 看不到的角度可以用 Zero123++ 这种 diffusion model 生成补全

robot arm 更简单, 直接用 GIMP 的 bucket-paint 把颜色刷成 real 里采的。因为 arm 在运动中颜色会变 (光照角度变了), 他们从 task 不同阶段采多个颜色, eval 时取平均。

### 关键 ablation: 必须一起做

Table III 是这篇 paper 最反直觉的发现之一:

| GreenScreen | Drawer tune | Robot tune | MMRV↓ |
|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | 0.087 |
| ✗ | ✓ | ✗ | 0.087 |
| ✗ | ✗ | ✓ | 0.087 |
| ✗ | ✓ | ✓ | 0.087 |
| ✓ | ✗ | ✗ | 0.087 |
| ✓ | ✓ | ✗ | 0.142 (更差!) |
| ✓ | ✓ | ✓ | **0.050** |

几个 takeaway:
- 只 tune 一项完全没用, MMRV 都是 0.087
- tune 了 foreground 但没 green screen, 反而更差 (0.142)。因为 foreground 变 real 了, background 还是 sim 的, 整体不一致, policy 看着更别扭
- 只有 foreground + background 全都搞一致, 才会从 0.087 跳到 0.050

直觉: **policy 对 scene 的整体 consistency 比局部 realism 更敏感**。这跟 LLM 里 chain-of-thought self-consistency、image generation 里全局 harmonization 比 local detail 更影响 perception 是一回事。

---

## MMRV: 一个我觉得很 elegant 的 metric

假设你有 6 个 policy, real 里 success rate 分别是 0.9, 0.85, 0.8, 0.75, 0.3, 0.1。Sim 里 ranking 应该跟 real 一致。

Pearson $r$ 的问题:
1. 它假设线性关系。如果 sim 和 real 是单调但非线性的 (比如 saturating curve), $r$ 会偏低, 但 ranking 完全正确, 实际上这个 sim pipeline 是好用的。
2. 它对 narrow performance range 的噪声过敏。如果 6 个 policy real success rate 都在 0.8±0.02, real 一点 noise 就让 $r$ 大幅波动。

Spearman rank correlation 解决问题 1, 但忽略 margin。Figure 3 左边和中间左边两个 case, 都只有 1 个 rank violation, Spearman 给相同的分数, 但左边的 violation 是 5% real margin (噪声范围内, 可接受), 中左是 25% real margin (真正的 ranking 错误)。Spearman 分不出这两个。

作者提的 MMRV:

$$
\text{MMRV} = \frac{1}{N} \sum_{i=1}^N \max_{1 \le j \le N} \left[|R_i - R_j| \cdot \mathbf{1}[(R_{S,i} < R_{S,j}) \neq (R_i < R_j)]\right]
$$

人话翻译: 对每个 policy $i$, 找出它和所有其他 policy $j$ 中, ranking 翻车最严重的那一对, 翻车的"严重程度"用 real 里这两个 policy 的 success rate 差来衡量。然后对所有 $i$ 取平均。

- $N$: policy 数量
- $R_i, R_j$: real 里 $\pi_i, \pi_j$ 的 success rate
- $R_{S,i}, R_{S,j}$: sim 里对应的
- $\mathbf{1}[\cdot]$: 如果 sim ranking 和 real ranking 矛盾就是 1, 否则 0
- $\max_j$: 找最严重的那个 violation
- 外层 $\frac{1}{N}\sum$: 平均所有 policy 的最坏情况

直觉: 一个好的 sim pipeline, ranking 错了应该只发生在 real performance 差距很小 (噪声范围内) 的 policies 之间。如果它把 real 差距 25% 的两个 policy 搞反了, 那这个 sim pipeline 真的不行。MMRV 把"错没错"和"错的代价多大"绑在一起。

这个 metric 我觉得可以推广到很多场景:
- LLM eval set 设计: cheap eval set 的 model ranking 是否跟 expensive eval set 一致
- Hyperparameter tuning: proxy objective 和真实目标的 ranking alignment
- Benchmark design: 一个新 benchmark 能否正确区分 model 能力

---

## 主要实验结果

### Real vs Sim 强相关

Google Robot Pick Coke Can (Visual Matching):

| Policy | Real | Sim |
|:-:|:-:|:-:|
| RT-1 (Converged) | 0.853 | 0.857 |
| RT-1 (15%) | 0.920 | 0.710 |
| RT-1-X | 0.760 | 0.567 |
| RT-2-X | 0.907 | 0.787 |
| Octo-Base | 0.293 | 0.170 |
| RT-1 (Begin) | 0.133 | 0.027 |

Pearson r = 0.976, MMRV = 0.031。绝对数字也对得上, ranking 也对。

WidowX + Bridge 的 4 个 task, MMRV 平均 0.028, Pearson r 平均 0.85。

### Validation MSE 几乎没用

这是我觉得最应该让 imitation learning 圈子认真读的数字:

| Method | Pick Coke Can MMRV | Pearson r |
|:-:|:-:|:-:|
| Validation action MSE | 0.412 | 0.464 |
| SIMPLER Visual Matching | 0.031 | 0.976 |

差 13 倍。Validation MSE 在 action space 上衡量误差, 跟 task success 中间隔着 dynamics + perception + closed-loop feedback 三层。Sim rollout 成本只比 MSE 高一点, 但 signal 质量是数量级的差别。

以后看到 imitation learning paper 只报 validation loss 选 checkpoint, 可以直接质疑。

### Sim 能预测 real 的 OOD 行为

他们测了 5 种 distribution shift: background, lighting, distractors, table texture, camera pose。Sim 里的发现跟 real 完全一致:
- Camera pose 和 table texture 影响最大 (掉 40%+ success rate)
- Lighting 和 distractor 影响很小 (掉 5-10%)

更厉害的: 他们在 sim 里发现 Octo-Base 对 robot arm texture 特别敏感 (untuned arm 0%, tuned arm 29.3%), RT-1-X 不敏感。为了验证这个 sim 里的观察在 real 成立, 他们在 real robot 上用礼品包装纸包住 arm 做全新 shift 实验。结果 sim 预测完全成立: Octo-Base real OOD 从 29% 掉到 0%, RT-1-X 从 76% 掉到 52%。

Sim 可以当"policy behavior microscope"用, 不只是 ranking。

### Robustness 对物理参数

把 coke can mass 从 10g 扫到 80g, gripper friction 从 0.25 扫到 2.0, Pearson r 一直 > 0.96。Cabinet joint friction 也类似。说明不用精确定位每个物体的 mass 和 friction, 粗略估个值就够 ranking 用。

### 跨 simulator

在 Isaac Sim 上复现 Google Robot 实验, MMRV 0.064 vs SAPIEN 的 0.031, Pearson r 0.973 vs 0.976。差距很小, 说明这套方法不依赖 SAPIEN 特性, 可以搬到任何 physics simulator。

---

## 一些更深的联想

### Sim-to-real 和 real-to-sim 是同一个 gap 的两个方向

Sim-to-real training 用 domain randomization 让 policy robust 到 real variation; real-to-sim eval 用 Visual Matching 让 sim 视觉接近 real, 让 real-trained policy 在 sim 里能 perform。两者其实在解决同一个 gap, 只是方向反了。

这暗示一个 closed loop: 用 real data 训 policy → 用 sim eval 选 checkpoint → 用 sim 生成更多 data → 混进 real data 训下一版 policy → 再 eval。有点 AlphaGo self-play 的味道, 只是 sim 不能完全替代 real, 需要周期性校准。

### 评估基础设施的范式

这篇 paper 给了"在 sim 里评估 real policy"的范式。如果社区接受, robot policy 论文的 reporting 格式可能从"我们跑了 100 个 real trial"变成"100 个 real trial + 1000 个 sim trial", 前者是 ground truth, 后者提供更多统计信号。这跟 ML 里 ImageNet 之于图像分类差不多, 大家共享一个可复现的 sim benchmark。

### Foundation model 评估的启示

RT-2, OpenVLA, π0 这些 VLA 模型越来越大, real eval 越来越贵。SIMPLER 的思路 — 构造少量 sim benchmark envs, 用 ranking correlation 代理 real eval — 可以推广到 VLA 评估。这和 LLM eval 里用 benchmark 代理真实部署是一回事, 但 robot 多了 physical grounding 这个约束, 所以需要 SysID 和 Visual Matching 这种 grounding 技术。

### Visual consistency > visual realism

Table III 那个 ablation 我觉得值得反复琢磨。Tune drawer 但没 green screen, MMRV 反而升到 0.142, 比啥都不 tune 还差。这说明 policy 感知的不是孤立的 pixel statistics, 而是整个 scene 的"协调感"。一旦 foreground 变 real 了, background 还是 sim 的, 反而比全部都是 sim 的更刺眼。

这和 image generation 里 global harmonization 比 local detail 重要是一回事, 和 LLM 里 context consistency 比 single token accuracy 重要也是一回事。Policy 不是在判别单张 image, 而是在整套 sensory input 上做决策, 整体 distribution 的 shift 比局部 distribution 的 shift 更致命。

### 为什么 control gap 是 first-order

我以前低估过这点。Table II 把 PD 参数随便扰动一下, control loss 从 0.131 升到 0.267 (2x 差), MMRV 从 0.031 升到 0.070 (2x 差)。说明对 contact-rich task, controller 跟踪误差直接放大到 success rate。Grasping 一个 coke can, end-effector 偏 1cm 就 grasp 不到, 偏 5 度就 align 不上 drawer handle。"control gap 是 visual gap 的二阶效应"这种直觉是错的。

### Real eval 不能完全被取代

作者也强调, sim 永远是不完美的 proxy。他们的目标是给 practitioner 一个 cheap 信号用来 iterate, 而不是完全替代 real eval。在论文最终 reporting 时, real eval 还是 ground truth。这点要 keep in mind, 别过度解读成"以后不用 real robot 了"。

---

## 总结成一句话

**用 sim 评估 real-trained robot policy 是可行的, 只要你 (a) 用 system identification 把 controller 调到轨迹对得上, (b) 用 green screening + texture matching 把视觉调到一致, 然后 sim 的 ranking 就能强代理 real ranking, 还能预测 policy 对 novel distribution shift 的行为模式。**

References:
- Paper 主页: https://simpler-env.github.io
- Code: https://github.com/simpler-env/SIMPLER
- GeTex (texture matching 工具): https://github.com/Jiayuan-Gu/GeTex
- SAPIEN: https://sapien.ucsd.edu
- Octo: https://octo-models.github.io
- Open-X-Embodiment: https://arxiv.org/abs/2310.08864
- RT-1: https://robotics-transformer.github.io
- BridgeData V2: https://github.com/rail-berkeley/bridge_data_v2
- Xie et al. 2023 (generalization gap): https://arxiv.org/abs/2307.03659
- SAM: https://segment-anything.com
- Objaverse: https://objaverse.allenai.org
- CoACD: https://github.com/wkzheng/CoACD
- One-2-3-45++: https://one2345plus.github.io
- Ruckig: https://github.com/pantor/ruckig
- Ditto: https://ditto3d.github.io
- NVIDIA Isaac Sim: https://developer.nvidia.com/isaac-sim

---

# SIMPLER: 用 Simulation 评估 Real-World Robot Manipulation Policies

这篇 paper 解决了一个在 robot learning 研究里越来越尖锐的痛点: 当我们训练出 RT-1, RT-2, Octo 这类 generalist manipulation policies 时, real-world evaluation 既慢又贵又不可复现, 而 validation MSE 又靠不住。作者们提出一个反直觉的方案: 把在 real data 上训练的 policy 放进 purpose-built 的 simulation 里跑, 用 sim 的 success rate 来代理 real 的 success rate。这听起来像 sim-to-real 的镜像问题 (real-to-sim evaluation), 但其实它的难点完全不同。

项目主页: https://simpler-env.github.io
代码: https://github.com/simpler-env/SIMPLER

---

## 1. Motivation 和 Core Insight

Generalist robot policies 的能力越宽, 需要评估的任务越多。RT-X 系列要 evaluate 几十个任务, 每个 task 几十上百 trials, 一个 lab 几周才能跑完一轮。Digital twin 是 navigation 和 autonomous driving 里常用的方案, 但 manipulation 的难点在于: dynamic objects, articulated objects, friction, contact — 这些都很难高保真地复刻。作者的核心 insight 是:

> 我们不需要一个 pixel-perfect 的 digital twin, 我们只需要一个"刚好够用"的 simulation, 使得 sim 和 real 的 policy relative performance 强相关。

也就是, 给两个 policy $\pi_a, \pi_b$, 如果 real-world $R_a > R_b$, 那么 sim 里也应该 $R_{S,a} > R_{S,b}$。这个 relative ranking 的保真度, 比绝对 success rate 的匹配更重要, 因为 practitioners 真正关心的是"哪个 checkpoint 更好"这个 ranking signal, 而不是 absolute number。

---

## 2. MMRV Metric: 为什么 Pearson r 不够用

这节是 paper 里我觉得最 elegant 的部分之一。Pearson $r$ 有两个 well-known 的缺点:

1. **Linear fit 假设过强**: 如图 3 中间右所示, 如果 sim 和 real 是 monotonic 但 non-linear 的关系 (比如 saturating curve), Pearson $r$ 会偏低, 但 ranking 仍然正确。
2. **对 noise 在 narrow performance range 上过敏**: 如果一组 policies 的 real performance 都差不多, 一点 real-world noise 就会让 $r$ 大幅变化。

Spearman rank correlation 解决问题 1, 但忽略 magnitude。Figure 3 左 vs 中左 两个 case 都有一个 rank violation, 但左边的 violation 是 5% 的 real margin (噪声), 中左是 25% 的 real margin (真正的失败)。Spearman 给两个相同的分数, 但显然左边更好。

所以作者提出 **Mean Maximum Rank Violation (MMRV)**:

$$
\text{RankViolation}(i, j) = |R_i - R_j| \cdot \mathbf{1}[(R_{S,i} < R_{S,j}) \neq (R_i < R_j)]
$$

$$
\text{MMRV}(R, R_S) = \frac{1}{N} \sum_{i=1}^{N} \max_{1 \le j \le N} \text{RankViolation}(i, j)
$$

变量解释:
- $N$: policies 数量
- $R_i, R_j$: policy $\pi_i, \pi_j$ 的 real-world success rate
- $R_{S,i}, R_{S,j}$: 对应的 simulated success rate
- $\mathbf{1}[\cdot]$: indicator function, 当 sim 的 ranking 和 real 的 ranking 矛盾时为 1, 否则为 0
- $\max_j$: 对每个 policy $i$, 找出它和所有其他 policies 的 rank violation 中最严重的一个
- 整体平均: 用 $\frac{1}{N}$ 把每个 policy 的 worst violation 平均起来

直觉: 一个好的 sim pipeline, ranking 错了应该只发生在 real performance 差距很小 (噪声范围内) 的 policies 之间。MMRV 把"ranking 错了"和"错的代价有多大"绑在一起。MMRV $\in [0, 1]$, 越低越好; 0 表示 sim ranking 完全正确, 或者只在 real margin 为 0 的地方错。

这个 metric 我觉得可以推广到很多 ML benchmark 评估场景, 比如 LLM leaderboard 上某个 eval set 是否能正确反映 model ranking。

---

## 3. Mitigating Control Gap: System Identification

Real robot 的 PD controller 参数 (stiffness $\mathbf{p}$, damping $\mathbf{d}$) 直接拿过来用在 SAPIEN 里通常轨迹对不上。原因包括: simulator 的 contact model, friction, solver iteration count, simulation timestep 都和 real 不同。作者做法很直接 — 从已有 demonstration dataset 里取一段 action trajectory $\{a_i\}_{i=1}^T$ 和对应的 real end-effector pose trajectory $\{(\mathbf{x}_i, R_i)\}$, 在 sim 里 open-loop replay 同样的 actions, 然后最小化 trajectory 距离:

$$
\mathcal{L}_{\text{transl}}(\mathbf{p}, \mathbf{d}) = \frac{1}{T} \sum_{i=1}^T \|\mathbf{x}_i - \mathbf{x}_i'\|_2
$$

$$
\mathcal{L}_{\text{rot}}(\mathbf{p}, \mathbf{d}) = \frac{1}{T} \sum_{i=1}^T \arcsin\left(\frac{1}{2\sqrt{2}} \|R_i - R_i'\|_F\right)
$$

$$
\mathcal{L}_{\text{sysid}}(\mathbf{p}, \mathbf{d}) = \mathcal{L}_{\text{transl}} + \mathcal{L}_{\text{rot}}
$$

变量解释:
- $\mathbf{x}_i \in \mathbb{R}^3$: real robot 在第 $i$ 步的 end-effector 平移
- $R_i \in \text{SO}(3)$: real robot 在第 $i$ 步的 end-effector 旋转矩阵 (3×3 正交矩阵)
- $\mathbf{x}_i', R_i'$: sim 里对应步骤的 pose
- $\|\cdot\|_F$: Frobenius norm, $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$
- $\arcsin\left(\frac{1}{2\sqrt{2}}\|R_i - R_i'\|_F\right)$: 这是两个旋转矩阵之间 geodesic distance 的常用近似, 当 $R_i \approx R_i'$ 时它等于 $\frac{1}{2}\|\log(R_i R_i'^{-T})\|$, 单位是弧度
- $\mathbf{p}, \mathbf{d}$: 要优化的 PD 参数 (stiffness 和 damping), 通常每个 joint 一个分量

优化用 **simulated annealing** [Pincus 1970], 一共跑 3 轮, 每轮把搜索范围归一化到 $[0, 1]$, 然后选 $\mathcal{L}_{\text{sysid}}$ 最低的参数 $(\mathbf{p}_1, \mathbf{d}_1)$ 作为下一轮的起点, 同时缩小搜索范围。这种 coarse-to-fine simulated annealing 对 non-convex loss 很合适, 因为 PD 参数和轨迹误差之间没有简单的解析关系。

Table II 的 ablation 很 striking:
- Setting 1 (随机扰动的 PD): control loss 0.267, MMRV 0.070
- Setting 2 (更差扰动): control loss 0.432, MMRV 0.100
- SIMPLER SysID: control loss 0.131, MMRV 0.031

也就是说, controller 跟踪误差从 0.267 降到 0.131 (2x 改善), MMRV 也相应从 0.070 降到 0.031。这说明 control gap 是 real-to-sim correlation 的一个真实 bottleneck, 而不是被 visual gap 掩盖的二阶效应。

---

## 4. Mitigating Visual Gap: Green Screening + Texture Matching

Visual gap 的处理分两部分。

### 4.1 Green Screening (背景替换)

最简单粗暴但效果惊人的 trick:
1. 从 real evaluation video 第一帧, 用 https://cleanup.pictures/ 这种 inpainting 工具把 robot 和 foreground objects 擦掉, 得到 clean background $I_{\text{real}}$。
2. 在 sim 里 query ground truth segmentation mask $M$ (SAPIEN 提供), 拿到 foreground mask。
3. 合成: $I' = M \odot I_{\text{sim}} + (1-M) \odot I_{\text{real}}$

$\odot$ 是 element-wise multiplication。简单理解: 前景用 sim render (保证物理一致性), 背景用 real image (保证 visual realism)。这等价于把 sim 当作一个"前景绿幕", 然后合成到 real 背景上。

### 4.2 Texture Matching (前景物体)

只换背景不够, 因为 policy 对前景 object 和 robot arm 的 texture 也敏感。两种策略:

**对一般 object** (project real texture to sim mesh):
1. 用 SAM [Kirillov et al. 2023] 从 real image 里 crop 出 object
2. 把 sim object 的 pose 粗调到和 real 重合
3. 用 Nvdiffrast 做可微渲染, 精细优化 sim object pose 使其 segment mask 和 real 对齐
4. 把 real RGB 值"unproject"到 sim mesh 上
5. (optional) 用 Zero123++ [Shi et al. 2023] 这种 diffusion model 生成剩余视角的 texture, 再 unproject 上去

**对 robot arm** (color copy-paste):
- robot link 的 visual mesh 本来就有 texture map, 直接用 GIMP bucket-paint 把 real 视频里采样的颜色刷上去就行
- 因为 arm 在 motion 中颜色会变 (光照、视角), 作者从 task 不同阶段采多个颜色, 跑 eval 时平均

### 4.3 Visual Matching 的 ablation (Table III)

| GreenScreen | Drawer Matching | Robot Matching | MMRV↓ | Real-Sim Gap↓ |
|:-:|:-:|:-:|:-:|:-:|
| ✗ | ✗ | ✗ | 0.087 | 0.272 |
| ✗ | ✓ | ✗ | 0.087 | 0.266 |
| ✗ | ✗ | ✓ | 0.087 | 0.272 |
| ✗ | ✓ | ✓ | 0.087 | 0.328 |
| ✓ | ✗ | ✗ | 0.087 | 0.198 |
| ✓ | ✓ | ✗ | 0.142 | 0.253 |
| ✓ | ✓ | ✓ | **0.050** | **0.136** |

非常值得注意的两个现象:
1. 只 tune drawer 不 tune robot, 或者只 tune robot 不 tune drawer, **完全没改善** (MMRV 仍然是 0.087)。这暗示 visual gap 的"水桶效应": 场景里只要有一处不一致, policy 就会感到 distribution shift。
2. 4 行 (drawer+robot 都 tune 但没 green screen) MMRV 反而升到 0.142 — tuning foreground 但保留 sim background 反而引入了 inconsistency, 因为 foreground 变 real 了, background 还是 sim 的, 整体上更不协调。

这个 takeaway 我觉得很有意思: visual consistency 比 visual realism 更重要。这和 LLM 里 chain-of-thought 的 self-consistency 类比有点像, "整体一致性"比"局部精度"更影响下游行为。

### 4.4 Variant Aggregation 作为 baseline

Domain randomization 在 sim-to-real training 里很常用, 作者把它对称地用到 real-to-sim eval: 在 background, lighting, distractor, table texture 4 个 axis 上各构造 2 个 variant, 跑完后取平均。Table I 显示 Variant Aggregation 平均 MMRV 0.143, Visual Matching 0.056, 差了快 3 倍。原因是 Variant Aggregation 仍然把 policy 暴露在 visual distribution shift 下, 而 Visual Matching 直接消除 shift。

---

## 5. SIMPLER 环境的构建流程

环境构造流程:
1. **Robot URDF**: Google Robot 从公开 MuJoCo .mjcf 转成 URDF; WidowX 从 Interbotix repo 用 ROS export。Google Robot base link 的 collision mesh 需要重新 refine 避免 mesh penetration。PhysX 默认 Projected Gauss-Seidel solver 在 grasp 时会穿透, 改成 Temporal Gauss-Seidel 才正常。
2. **Object assets**: 
   - 从 [Objaverse](https://objaverse.allenai.org/) 拿常见物体 (cans, apples)
   - 不常见的通过 3D 扫描
   - 或者用 [One-2-3-45++](https://one2345plus.github.io/) 从单张图生成 mesh
3. **Articulated objects** (cabinet): 手工建模 + texture baking。作者说这是最耗时的部分, 建议未来用 [Ditto](https://ditto3d.github.io/) 这种 multi-view articulated object generation 加速。
4. **Collision mesh**: 用 [CoACD](https://github.com/wkzheng/CoACD) 做 convex decomposition, 得到 watertight + locally convex collision mesh
5. **物理参数**: density 用 GPT-4 查询 material 的常见 density; 也可以查 mass 除以 volume。Friction 按 material 常识赋值。
6. **Camera alignment**: 调 sim 的 robot 和 camera pose, 让 fixed objects (table edge, cabinet edge) 和 initial gripper 位置大致对齐。

性能: 4090 GPU + 640×512 渲染, 一个环境跑 3.5k sim steps/s, 500Hz sim 频率下相当于 real eval 7× speedup。

---

## 6. 实验结果

### 6.1 Main result: Real vs. Sim 强相关

Table IV (Google Robot Pick Coke Can, Visual Matching):
- RT-1 (Converged): real 0.853, sim 0.857
- RT-1 (15%): real 0.920, sim 0.710
- RT-1-X: real 0.760, sim 0.567
- RT-2-X: real 0.907, sim 0.787
- Octo-Base: real 0.293, sim 0.170
- RT-1 (Begin): real 0.133, sim 0.027

Pearson r = 0.976, MMRV = 0.031。这非常强。

Table V (WidowX + Bridge):
- Put Spoon on Towel: MMRV 0.000, r 0.827
- Put Carrot on Plate: MMRV 0.111, r 0.575
- Stack Block: MMRV 0.000, r 1.000
- Put Eggplant in Basket: MMRV 0.000, r 0.990

只有一个 task (Put Carrot on Plate) 表现略弱, 可能是因为 carrot 这种细长形状的 grasp 物理敏感度高。

### 6.2 vs. Validation MSE

Table XII 给了一个我想强调的对比:
- Validation action MSE: Pick Coke Can MMRV 0.412, Pearson r 0.464
- SIMPLER Visual Matching: Pick Coke Can MMRV 0.031, Pearson r 0.976

这是 imitation learning practitioner 应该认真读的数字。validation MSE 几乎完全没用, 而 sim rollout 成本只比 MSE 略高, 但 signal 质量是数量级的差别。原因是 MSE 在 action space 上衡量, 和 task success 之间隔着 dynamics + perception + closed-loop feedback 的鸿沟。

### 6.3 Distribution Shift Robustness

作者用 5 个 axis: background, lighting, distractors, table texture, camera pose, 测 RT-1 with/without image augmentation。Fig. 8 的核心发现:

| Shift | RT-1 w/o Aug $\Delta$ Success | RT-1 w/ Aug $\Delta$ Success |
|:-:|:-:|:-:|
| Background | 0.048 | 0.123 |
| Lighting | 0.057 | 0.075 |
| Distractors | 0.080 | 0.059 |
| Table Texture | 0.144 | 0.189 |
| Camera Pose | 0.473 | 0.394 |

两个非常一致的发现:
1. **Camera pose 和 table texture 影响最大**, lighting 和 distractor 影响小。这和 Xie et al. 2023 (Google 的 generalization gap 论文) 在 real 上得到的结论一致。Sim 上的 ranking 完美匹配。
2. **细粒度预测也对**: 真实实验里换 unseen solid color table 影响是 4%, 换 unseen patterned texture 是 25%; sim 里分别是 2% 和 24%。这暗示 sim pipeline 可以用来预测"我的 policy 对哪种 OOD 类型敏感"。

### 6.4 预测未见过的 distribution shift

更 cool 的实验: 在 sim 里他们观察到 Octo-Base 对 robot arm texture 很敏感 — 用未 tune 的 arm, success rate 是 0%, 用 tuned 的 arm 是 29.3%。RT-1-X 在同一个 Open-X-Embodiment dataset 上训练, 但 robustness 高很多。

为了验证 sim 里的这个观察是否在 real world 成立, 他们在 real robot 上用礼品包装纸包住 arm 做了一个全新 distribution shift 实验 (Fig. 9)。结果:

| Policy | Sim Success Range | Real Orig | Real OOD |
|:-:|:-:|:-:|:-:|
| RT-1-X | [0.507, 0.653] | 0.760 | 0.520 |
| Octo-Base | [0.000, 0.293] | 0.293 | 0.000 |

Real 验证了 sim 预测: Octo-Base 对 arm texture 远比 RT-1-X 敏感。这是 sim 评估 pipeline 可以预测 policy 在 real-world novel shift 下的行为的直接证据, 这个能力比单纯 ranking policy 强很多。

### 6.5 Physics Robustness 和 Simulator Transferability

Table X(a): 把 coke can mass 从 10g 扫到 80g, gripper friction 从 0.25 扫到 2.0, Pearson r 一直在 0.96 以上, MMRV 一直是 0.031。说明 physical property 的 imprecision 对 ranking 影响很小。

Table X(b): cabinet joint friction 从 0.0125 到 0.20, MMRV 一直 0.055 (除了一个 outlier), Pearson r 在 0.90-0.94 之间。同样稳健。

Table XI: 在 Isaac Sim 上复现 Google Robot Pick Coke Can 的实验, MMRV 0.064, Pearson r 0.973, 和 SAPIEN 上的 0.031/0.976 接近。说明这套方法不依赖于 SAPIEN 特性, 可以迁移到其他 physics simulator。

---

## 7. Algorithm 1: Google Robot Controller in Sim

Appendix 里的 Algorithm 1 我觉得值得细看, 因为它揭示了 sim controller 设计的一些细节:

```
Input: action (x_a, R_a), sensed q_arm, v_arm, g_a, q_grip, v_grip
       H_sim = 501, H_ctrl = 3 (Google Robot 的控制频率)
       L_arm = (1.5, 2.0, 50.0)  # velocity, acceleration, jerk limits
       L_grip = (1.0, 7.0, 50.0)
1. (x, R) = FK(q_arm)
2. (x_goal, R_goal) = (x_a + x, R_a · R)
3. (q_goal, v_goal) = (IK(x_goal, R_goal, q_arm), 0.0)
4. ArmPlan = Ruckig(q_goal, v_goal, q_arm, v_arm, L_arm)
5. if T == 0: q_lastplan_grip = q_grip; q_lastgoal_grip = q_grip
6. if |g_a| < 0.01: q_goal_grip = q_lastgoal_grip  # small action filter
   else: q_goal_grip = q_lastplan_grip + g_a
7. GripPlan = Ruckig(q_goal_grip, 0, q_lastplan_grip, v_lastplan_grip, L_grip)
8. for i in 1..(H_sim/H_ctrl):  # 每个 control step 跑 H_sim/H_ctrl 个 sim step
     t = i / H_sim
     SetArmJointPosTarget(ArmPlan(t))
     SetGripperJointPosTarget(GripPlan(t))
     SetGripperJointVelTarget(...)
9. q_lastgoal_grip = q_goal_grip; T += 1
```

几个值得注意的细节:
- **Ruckig**: time-optimal joint trajectory planner, 在 velocity/acceleration/jerk 三重约束下生成轨迹。这个比简单的 cubic spline 更符合 real robot 行为。
- **Small action filtering**: 当 $|g_a| < 0.01$ 时不更新 gripper goal, 避免数值噪声导致 gripper 抖动。这是 real robot controller 上常见 trick, sim 里也必须复现, 否则 sim-real 控制行为不一致。
- **Goal pose composition**: $(x_{\text{goal}}, R_{\text{goal}}) = (x_a + x, R_a \cdot R)$, 其中 $x, R$ 是当前 FK 结果。这是 action 作为 end-effector frame 下的 delta 的写法, 不是 base frame delta。

WidowX 的 Algorithm 2 略不同, 用 4×4 SE(3) 矩阵做 composition, 而且 control frequency 是 5Hz。

---

## 8. Kruskal-Wallis Test (Appendix G-B)

Table XIV 用了另一个角度: 对每个 policy 单独做 Kruskal-Wallis test, 看 sim 和 real 的 trial-level success distribution 是否有显著差异 (p<0.05)。Visual Matching 下 Google Robot Pick Coke Can 大多数 policy 的 p 都 ≥ 0.05 (有 2 个 < 0.05), 说明 absolute success distribution 也大致保持。

这给了 MMRV 一个补充: MMRV 衡量 relative ranking, Kruskal-Wallis 衡量 absolute distribution。两个 metric 互补。

---

## 9. Single-Task Policy 也 Work (Table XIII)

作者 train 了一个只在 "Pick Coke Can" 数据上的 RT-1, real 0.680, sim 0.403。加进去之后 7 个 policy 的 MMRV 0.027, Pearson r 0.959。这个实验挺重要, 因为 single-task policy 的 dataset 小, 容易 overfit, 对 sim-real gap 更敏感。SIMPLER 在这个 regime 下仍然 work, 说明它不仅对 Open-X-Embodiment 这种大规模 multi-task data 训出来的 policy 有效。

---

## 10. Limitations 和我的一些联想

作者明确列了几个 limitation:
- 只做 rigid body, 没做 soft body / deformable
- Green screening 只支持 fixed camera, 不处理 shadow 和 fine visual detail
- 环境构造还需要手工 asset curation, 没完全自动化

我自己额外的几点思考:

**关于 sim-to-real 与 real-to-sim 的对称性**: 这篇 paper 间接证实了一个哲学观点 — sim 和 real 之间的 gap 是双向的。sim-to-real 训练用 domain randomization 让 policy robust 到 real 的 variation; real-to-sim eval 用 Visual Matching 让 sim 视觉上接近 real, 让 real-trained policy 在 sim 里能 perform。两者其实在解决同一个 gap, 只是方向反了。这意味着 sim-to-real 训练 + real-to-sim eval 可以形成一个 closed loop: 用 real data 训 policy, 用 sim eval 选 checkpoint, 再用 sim 训练的 data 增强 real data (比如 sim 自动生成演示), 再训下一版 policy。这有点像 AlphaGo 的 policy network + self-play 思路。

**关于 generalist policy 的 scaling**: 当 Open-X-Embodiment 扩到 100k+ hours, 用 real eval 完全不现实。SIMPLER 这种方法会成为标配。我预期未来一两年里会有 SIMPLER v2, 覆盖 100+ tasks, 自动 asset acquisition, 用 NeRF/3DGS 做 scene reconstruction 而不是手工建模。

**关于 action space 的影响**: 这篇 paper 测的都是 end-effector pose control 的 policy (Google Robot 是 6D + gripper, WidowX 也是)。如果是 joint torque control 或者 delta joint control, control gap 处理方式会很不同。SysID loss 需要改成 joint trajectory tracking 而不是 EE pose tracking。

**关于 visual gap 处理的 future direction**: 现在 Visual Matching 还需要手工 SAM + Nvdiffrast 对齐。如果能用 GS-LRM 或者 NeRF-based relightable reconstruction, 直接从 real video 重建 object mesh + PBR material, 然后塞进 sim, 整个 pipeline 就更接近全自动。这和 TensoIR [Jin et al. 2023] 那条线很相关。

**关于 manipulation benchmark 的标准化**: SIMPLER 给了一个"在 sim 里评估 real policy"的范式。如果社区接受, 这可能让 robot policy 的论文报告范式从 "we ran 100 trials on real robot" 变成 "we ran 100 trials on real robot + 1000 trials in SIMPLER", 后者提供更多统计信号, 前者保留 ground truth。

**关于 foundation model 评估的启示**: 这个方法其实可以推广到 VLA 模型评估。RT-2, OpenVLA, π0 这些模型越来越大, real eval 越来越贵。SIMPLER 的思路就是: 构造少量 sim benchmark envs, 用 ranking correlation 来代理 real eval。这和 LLM 评估里"用 benchmark 代理真实部署"的思路类似, 但 robot 多了 physical grounding 这个约束。

**关于为什么 control gap 这么重要**: 我自己以前也低估过这点。Table II 里把 PD 参数随机扰动一下, control loss 从 0.131 升到 0.267 (2x 差), MMRV 从 0.031 升到 0.070 (2x 差)。这说明对 grasping 这种 contact-rich task, controller 跟踪误差直接放大到 success rate 上。原因可能是 end-effector 偏 1cm 就 grasp 不到 can, 偏 5 度就 align 不上 drawer handle。所以"control gap 是 visual gap 的二阶效应"这种直觉是错的, 它是一阶的。

**关于 MMRV 的推广**: 我觉得 MMRV 这个 metric 可以推广到很多场景:
- LLM eval set 比较: 用 sim proxy eval (比如一个 cheap eval set) ranking 是否和 real eval (expensive eval set) ranking 一致
- Model selection: validation loss ranking vs. test performance ranking
- Hyperparameter tuning: 在 cheap proxy 上 tuning, 用 MMRV 衡量 proxy 和真实目标的对齐度

---

## 11. 关键引用和资源链接

- **项目主页**: https://simpler-env.github.io
- **SAPIEN simulator**: https://sapien.ucsd.edu/ (Stanford/UCSD 的 part-based 仿真器)
- **NVIDIA Isaac Sim**: https://developer.nvidia.com/isaac-sim
- **Octo policy**: https://octo-models.github.io (开源 generalist policy)
- **Open-X-Embodiment**: https://arxiv.org/abs/2310.08864 (RT-X 论文)
- **RT-1**: https://robotics-transformer.github.io/
- **BridgeData V2**: https://github.com/rail-berkeley/bridge_data_v2
- **Objaverse**: https://objaverse.allenai.org/ (3D asset 库)
- **CoACD**: https://github.com/wkzheng/CoACD (convex decomposition)
- **SAM (Segment Anything)**: https://segment-anything.com/
- **One-2-3-45++**: https://one2345plus.github.io/ (单图 3D 重建)
- **Ruckig**: https://github.com/pantor/ruckig (time-optimal trajectory planner)
- **Ditto (articulated object generation)**: https://ditto3d.github.io/
- **Xie et al. 2023 (generalization gap)**: https://arxiv.org/abs/2307.03659
- **GeTex (作者 release 的 texture matching 工具)**: https://github.com/Jiayuan-Gu/GeTex

---

## 12. 总结 Takeaway

如果让我把这篇 paper 压成几条 takeaway 给 Karpathy 这种人:

1. **Real-to-sim evaluation 是可行的**, 在 Google Robot 和 WidowX 两个 setup, RT-1/RT-1-X/RT-2-X/Octo 4 个 policy family 上, Pearson r 在 0.9 以上, MMRV < 0.06。这意味着 lab 可以省下大量 real eval 时间。
2. **Control gap 和 visual gap 都是 first-order**, 都需要 explicit 处理。SysID 用 simulated annealing 在 PD 参数上 minimize trajectory tracking error; Visual Matching 用 green screen + texture projection 双管齐下。两者必须一起做, 缺任何一个 correlation 都掉。
3. **MMRV 是个 elegant metric**, 比 Pearson r 和 Spearman 都更适合"评估的评估"场景, 因为它对 noise-allowed violation 和 true-failure violation 区别对待。
4. **Validation MSE 在 imitation learning 里几乎没用**, 实验数据上 MMRV 0.4 vs SIMPLER 0.03, 差 13 倍。这个 takeaway 应该进入所有 imitation learning 论文的 reviewing checklist。
5. **SIMPLER 不仅 ranking policy, 还能预测 policy 对 novel distribution shift 的 robustness**, 这点比单纯 performance ranking 更有价值, 因为它让 sim 成为一种"policy behavior microscope"。

我个人觉得这篇 paper 的思想会沿着几个方向被 follow up: 自动 asset acquisition pipeline、coverage 扩展到 deformable/articulated/soft-body、和 sim-to-real 训练形成 closed loop、以及把 MMRV 推广成 ML 评估的通用 metric。它给 robot learning community 提供了一个"评估基础设施"的范式, 类似 ML 里 ImageNet 之于图像分类 — 你不需要在每次新论文都重做 eval, 而是大家共享一个可复现的 sim benchmark。
