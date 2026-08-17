---
source_pdf: ReSim.pdf
paper_sha256: cf4a3aa09fe8c3264f11e6c0abffff332e613c7800d74768669f8457683d3351
processed_at: '2026-08-11T22:58:02-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ReSim 人话版

## 1. 这 paper 到底在讲啥

现在 driving world model 有个尴尬: 你让它生成"未来 4 秒的视频", 它生成的画面挺漂亮, 但你一旦给它一个 action 信号(比如"向右急转冲下路基"), 它生成的视频还是老老实实往前开, 完全无视你的指令。

为啥? 训练数据全是人类安全驾驶的录像, 根本没见过"冲下路基"长什么样。模型没这个数据, 它就 hallucinate 成它见过的最常见的样子——安全直行。这就像一个只看过驾校教学视频的人, 你让他脑补"闯红灯撞车会怎样", 他真脑补不出来, 因为驾校视频里没有这种画面。

ReSim 的招数特别直接: 既然现实世界不能随便撞车采集数据, 那就用 CARLA simulator 去撞, 把这些"危险动作 + 后果"的录像跟真实人类驾驶录像混在一起喂给 video 生成模型。模型同时学到"真实场景长这样"和"危险动作的后果长这样", 两边就拼上了。

这个思路的底层逻辑其实跟 Karpathy 你自己常说的 "data is the bottleneck" 完全一致——ReSim 的核心 contribution 不是新架构, 是 data curation 策略。架构上它就是 CogVideoX + Vista 的组合拳, 真正让它 work 的是把 CARLA 的 non-expert data 塞进训练集这个决定。

参考: ReSim 项目页 https://opendrivelab.com/ReSim

## 2. 三类数据各自干啥

用大白话说, 三类数据像三个角色配合:

| 数据源 | 角色 | 提供什么 |
|---|---|---|
| OpenDV (4M clips, 1700h 网络爬的 driving 视频) | "见多识广的老司机" | 场景多样性, 视觉泛化能力, 让模型见过各种天气、路况、camera 风格 |
| NAVSIM (85K clips, 专业标注) | "驾校教练" | 严格标注的 expert action(未来 waypoint), 让模型学会"听 trajectory 指令" |
| CARLA (88K clips, 一半 expert 一半 non-expert) | "赛车游戏玩家" | 危险动作 coverage + infraction score(reward 监督信号) |

CARLA 里的 non-expert data 是怎么来的? agent 从一组预设的 steering angle、throttle、behavior pattern 里 **random sample**, 然后执行。这等价于在 action space 里做 random exploration, 跟 rule-based planner 的 narrow 分布完全不同。这一步很关键——如果 non-expert data 只是"稍微偏离车道", 模型还是学不到真正的碰撞后果。它必须覆盖"急转向冲出路面"这种极端 action。

数据规模配比: OpenDV 4M vs NAVSIM 85K vs CARLA 88K。OpenDV 占绝对多数是为了保证 visual fidelity 和 generalization 不被 sim data 拉偏。CARLA 虽然量小, 但覆盖的是 real data 缺失的关键维度, 所以影响远超数量比例。

参考: OpenDV/GenAD https://github.com/OpenDriveLab/GenAD, NAVSIM https://github.com/autonomousvision/navsim, Bench2Drive https://github.com/Thinklab-SJTU/Bench2Drive

## 3. 模型架构, 用大白话拆解

ReSim 在 CogVideoX (https://github.com/THUDM/CogVideo) 上改。CogVideoX 本身是个 2B 参数的 text-to-video diffusion transformer, 原本只听 text 指令。ReSim 加了三个 conditioning 入口:

**输入端**:
- History frames(9 帧 @ 10Hz): 过去 0.9 秒的视频, 用 clean latent(不加 noise), 作为 conditioning
- Text command("Turning left" / "Moving forward" / "Turning right"): 高层意图, T5 encoder 编码
- Trajectory(8 个 waypoint @ 2Hz, 覆盖未来 4s): 精细 action, 通过专门的 trajectory encoder 投影到 DiT input space

**为什么 history 用 clean latent**: 这是 Vista 的做法。如果把 history 和 future 一起 denoise, 两者会耦合, history 里的 noise 会污染 future 预测。用 clean latent 让 history 变成纯粹的 conditioning signal, future 的 denoise 只关注"未来怎么生成", 职责清晰。

**输出端**: 4 秒 @ 10Hz, 512×896 分辨率的 driving video。

## 4. 三个技术 trick 的人话解释

### 4.1 Dynamics Consistency Loss (DCL) — "别只看每一帧, 看帧之间的变化"

普通 video diffusion loss 是这样:

$$
\mathcal{L}_{\mathrm{diffusion}} = \mathbb{E}_{x, \epsilon, t} \left[ \| x^{k:} - D_\theta(x_t; t, h, c, a)^{k:} \|^2 \right]
$$

变量解释:
- $x$: clean video latent(VAE 压缩后的 video 表示)
- $\epsilon$: Gaussian noise
- $t$: diffusion timestep, 范围 $[0, 1000]$, 越大加的 noise 越多
- $x_t$: 加了 noise 的 latent, $x_t = \sqrt{\bar\alpha_t} x + \sqrt{1-\bar\alpha_t} \epsilon$
- $D_\theta$: 参数为 $\theta$ 的 diffusion transformer
- $h$: history frames 的 clean latent
- $c$: text command embedding
- $a$: action(未来 waypoint sequence)
- $k$: 从第 $k$ 帧开始算 loss, history 帧不算

这个 loss 的问题是: 它 **逐帧独立监督**。模型可以偷懒——在低 noise timestep 时, 把相邻两帧 average 一下就能 minimize loss, 根本不需要学"车在往前开"这个 motion 概念。这就是为什么很多 video diffusion 生成的视频看起来"每帧都 OK, 但动起来很奇怪"。

ReSim 的 DCL:

$$
\mathcal{L}_{\mathrm{dynamics}} = \mathbb{E}_{x, \epsilon, t} \left[ \sum_{j=1}^{K} \sum_{i=1}^{N-j} \frac{1}{s} \| (d^{i+j} - d^i) - (x^{i+j} - x^i) \|^2 \right]
$$

变量解释:
- $x$: ground-truth latent
- $d = D_\theta(\cdots)$: 模型预测的 denoised latent
- $i$: frame index(latent space 里, VAE 时间压缩 4×, 所以 latent frame 数 < raw frame 数)
- $j$: 时间间隔, 从 1 到 $K=4$
- $K$: 最大间隔, 实验最佳值 4, 对应 latent space 里约 1.6 秒
- $N$: latent 总帧数
- $s$: normalization factor, 每个间隔下 $|\text{motion}|$ 的平均值, 稳定 loss 量级

直觉: 这个 loss 强制模型预测的 **"第 $i$ 帧和第 $i+j$ 帧的差值"** 等于 ground-truth 的 **"第 $i$ 帧和第 $i+j$ 帧的差值"**。$j=1$ 监督短期 motion(0.4s), $j=4$ 监督长期 motion(1.6s, 覆盖"刹车让行"这种慢动作)。模型再也不能靠相邻帧 average 偷懒了, 因为 loss 直接惩罚"运动量"的误差。

这跟 RL 里的 temporal difference learning 精神同源: 监督 state 的变化而非 state 本身, 强迫 capture dynamics。总 loss 是 $\mathcal{L} = \mathcal{L}_{\mathrm{diffusion}} + 0.1 \cdot \mathcal{L}_{\mathrm{dynamics}}$。

Figure 10 的 ablation 很直观: 不加 DCL, 预测的车会"漂"; 加了 DCL $K=4$, 车的运动连贯、场景稳定。

### 4.2 Unbalanced noise sampling — "别让模型在简单题上刷分"

CogVideoX 默认 uniform sample timestep $t \in [0, 1000]$。问题: $t$ 小的时候(低 noise), input 几乎是 clean video, 模型只要做轻微 denoise 就行, 这时候"相邻帧 average"的偷懒策略特别有效。模型把大量 capacity 浪费在这些简单 case 上, 真正学 motion 的 high-noise case 训练不够。

ReSim 的做法: 把 $t \in [500, 1000]$ 的采样频率从 1/2 提高到 2/3。等价于 **强迫模型在严重 corrupt 的 input 上多训练**, 这时 local average 完全没用, 模型必须靠理解场景 structure 才能还原。

这跟 Min-SNR weighting (https://arxiv.org/abs/2303.09556) 和 EDM (https://arxiv.org/abs/2206.00364) 的精神相通: high-noise timestep 携带更多 "structure" 信息, 应该被重点训练。ReSim 的做法更 brute force——直接 bias sampling distribution, 不改 loss 权重。在 driving domain 上 work, 说明这个 domain 下 high-noise timestep 确实最关键。

Figure 9 的 ablation: uniform sampling 下, 预测视频里车辆 motion 不连贯, 场景 layout 会漂移; unbalanced sampling 下, motion 和 layout 都 consistent。

### 4.3 Multi-stage training — "先学看, 再学听, 最后精修"

| Stage | DiT | LoRA | Traj Enc | 数据 | 分辨率 | Steps |
|---|---|---|---|---|---|---|
| 1 | 全训 | - | - | OpenDV | 512×896 | 20K |
| 2 | 冻结 | 训 | 训 | OpenDV+NAVSIM+CARLA | 256×448 | 80K |
| 3 | 全训 | 训 | 训 | OpenDV+NAVSIM+CARLA | 512×896 | 50K |

Stage 1: 让 CogVideoX 学会"看懂 driving 场景, 预测未来"。只喂 OpenDV, 没有 action, 纯视觉学习。

Stage 2: 引入 action condition。冻结 DiT 主干(省钱), 只训 trajectory encoder + LoRA adapter。低分辨率加速。这一步本质是让 trajectory encoder 学会"怎么把 waypoint 翻译成 DiT 能理解的 visual motion 指令"。注意 NAVSIM 的 trajectory 50% dropout(支持 action-free prediction), 但 CARLA trajectory **完全不 drop**——因为 hazardous action 无法从 visual context infer, 必须显式 condition。这个不对称设计很关键: 让模型知道"CARLA 的危险动作必须听 trajectory, NAVSIM 的安全动作可以自己推断"。

Stage 3: 全主干 high-res fine-tune, 让模型最终收敛到 production quality。

40× A100, 14 天, 不算便宜, 但相比从头训 CogVideoX 省太多了。

## 5. Video2Reward (V2R) — 用大白话讲

问题: world model 生成视频后, 怎么判断这个视频里的 driving 是好是坏? 需要 reward, 但 real-world driving 没有现成的 reward 信号(你不可能在真实路上撞车来收集"这是坏 driving"的 label)。

ReSim 的 insight: **CARLA 里有现成的 infraction score**(撞没撞、闯没闯红灯、冲没冲出路面、速度是否过低), 而且是 continuous 的 scalar。用这个当监督信号训一个"从视频预测 reward"的模型。

V2R 结构:
- Backbone: **frozen DINOv2** (https://github.com/facebookresearch/dinov2), self-supervised ViT, 在 142M 自然图像上 pretrain
- Head: 两个 spatio-temporal attention block + MLP, aggregate 所有 frame feature → scalar reward
- 训练: 35K CARLA sample, 20 epochs, AdamW lr 1e-3, 输入 224×224

**为什么能 generalize 到 real world**: 这是最 elegant 的部分。逻辑链:

1. CARLA 的 infraction score 是 ground truth reward, 涵盖 expert + non-expert 全谱
2. DINOv2 是 self-supervised, 学到的是 "物体、运动、场景的通用 visual concept", 不依赖特定 domain
3. ReSim 生成的 video 是 real-world 风格的(因为 Stage 1+3 在 OpenDV/NAVSIM 上 train)
4. V2R 在 CARLA video 上学的是 "撞车看起来长什么样" 这个 **视觉概念**, 通过 DINOv2 的通用 feature space, 这个概念 transfer 到 real-style video 上

等价于: **reward 的"概念"在 CARLA 学, 但"视觉接口"是 foundation model**。这跟 robotics 里用 VIP (https://arxiv.org/abs/2210.04498)、LIV (https://arxiv.org/abs/2306.13200) 这种 pretrained visual representation 当 reward 是同一个思路: 用 foundation model 的通用 feature 当 sim 和 real 之间的 "翻译层"。

这其实是 foundation model 时代 sim2real 的新范式: 不再做传统的 domain randomization 或 sim appearance 渲染逼真, 而是用 foundation model 的 semantic feature space 做 bridge。只要 sim 和 real 在 DINOv2 feature space 里对齐了, 在 sim 里学的 reward function 就能 transfer。

## 6. 三个 application, 用大白话说

### 6.1 Video prediction-based policy — "让它自己想象怎么开, 然后翻译成动作"

ReSim 在 action-free 模式下(不给 trajectory, 只给 history + command)生成未来视频, 然后用 Inverse Dynamics Model (IDM) 把视频"翻译"成 ego trajectory。

IDM 架构: XVO backbone (https://github.com/lostxine/XVO) + attention head, 输出 8 个 2Hz waypoint。在 NAVSIM navtrain 上训 100 epochs。

NAVSIM navtest PDMS 结果(Table 4):

| Method | 输入 | PDMS ↑ |
|---|---|---|
| VO planner (XVO 单独) | front video only | 78.4 |
| UniAD | multi-camera + ego status + extra anno | 83.4 |
| Transfuser | multi-camera + ego status + extra anno | 84.0 |
| DrivingGPT | past trajectory | 82.4 |
| LAW | multi-camera | 84.6 |
| GT Future + IDM (oracle) | - | 90.8 |
| **ReSim + IDM** | **front video only, no action** | **86.6** |

关键 insight: VO planner(78.4) 和 ReSim+IDM(86.6) 共享同一个 IDM, 差距 8.2 PDMS **完全来自 ReSim 生成的 video quality**。这是在 ablate"一个好的 generative world model 当 policy"的 value。

更狠的: ReSim+IDM 只用 front-view video + command, 没用 multi-camera、ego status、past trajectory、extra annotation, 却超过了 Transfuser(全传感器 fusion)。这说明 **video prediction 已经隐式 encode 了 driving policy 所需的几乎所有信息**——其他 agent 的意图、场景 structure、可行驶区域, 都在 video 里。

这跟 robotics 里 GR-1/GR-2 (https://arxiv.org/abs/2410.06158) 的思路同源: video prediction pretraining → inverse dynamics → action。也跟 Du et al. universal policies via text-guided video generation (https://arxiv.org/abs/2212.11985) 同构。

### 6.2 Reward-guided policy selection — "让多个 policy 出方案, ReSim 当评委选最好的"

场景: 有 Transfuser 和 LTF 两个 policy, 对每个 scenario 各出一条 trajectory。怎么选?

ReSim 的方案: 每条 trajectory 用 ReSim 生成 4s 视频, 用 V2R 打分, 选分高的执行。

Table 3(NAVSIM 300 个 challenging scenario):

| Method | PDMS ↑ |
|---|---|
| Transfuser alone | 47.7 |
| LTF alone | 47.2 |
| Uniform average ensemble | 66.8 |
| Vista reward | 59.2 |
| ReSim w/o sim + V2R | 69.7 |
| **ReSim + V2R** | **74.1** |
| Oracle (GT PDMS 选) | 94.2 |

ReSim + V2R 比 average ensemble 高 7.3, 比 Vista reward 高 14.9, 离 oracle 只差 20。这证明 V2R 的 reward 确实 capture 了"哪条 trajectory 更安全"的信号。

这个 pipeline 跟 LLM 里的 **Best-of-N sampling with reward model** 完全同构: generator 生成多个 candidate, reward model rerank, 选最好的。AV 里 ReSim 是第一个把这个 pipeline 做完整的。

注意 "w/o sim" 版本(69.7)比完整版(74.1)低 4.4, 说明 CARLA non-expert data 对 reward 判断能力有直接贡献——没见过危险场景的 model 不知道怎么 penalize 危险 trajectory。

### 6.3 Closed-loop visual simulation — "把 ReSim 当游戏引擎, policy 在里面跑"

ReSim 当 environment:
1. Policy 输出 4s trajectory
2. ReSim 生成 4s 视频
3. 取最后 9 帧当新 context
4. Policy 再决策
5. 循环

这是把 ReSim 当成 **learned simulator**, 跟 Genie (https://arxiv.org/abs/2406.19102)、GameNGen (https://arxiv.org/abs/2408.14846)、DIAMOND (https://arxiv.org/abs/2405.12358) 同类, 但是 driving domain。

Figure 8 展示 ReSim 把 VO-based policy 推到 pre-recorded dataset 里 never 出现过的 state——比如 policy 决策失误导致车偏到路边, 下一步 ReSim 生成的就是"从路边继续开"的视角, 这是真实数据集里永远没有的。这等于把 open-loop benchmark 升级成 closed-loop, 能 expose policy 的 error accumulation 问题。

## 7. 实验数据的直觉解读

### 7.1 Action controllability (Table 1, Waymo zero-shot)

| Method | Action-free ↓ | Expert Action ↓ |
|---|---|---|
| GT Future | 0.58 | 0.58 |
| Vista | 5.68 | 1.89 |
| ReSim w/o sim | 1.47 | 1.18 |
| **ReSim** | **1.13** | **0.86** |

Metric 是 Trajectory Difference(IDM 从生成视频估出 trajectory, 跟 GT trajectory 算 L2), 越小越好。

ReSim 在 expert action 下比 Vista 好 54%, 在 action-free 下好 80%。"w/o sim"版本仍然远好于 Vista, 说明 multi-stage + DCL + unbalanced sampling 本身就有效; 加上 CARLA 再提升一个台阶, 尤其在 non-expert action 上(Figure 4 人类评估, ReSim 几乎垄断了"trajectory following for non-expert"的 preference)。

### 7.2 Visual fidelity (Table 2, nuScenes zero-shot)

ReSim FID 5.2, FVD 50.4, 是 SOTA, 而且 **没在 nuScenes 上 train 过**, 比 in-distribution 的 Vista(6.9/89.4) 还好。这归功于 OpenDV 4M clips 的规模 + CogVideoX 的 visual prior。

这说明一个反直觉的事: **data scale > in-distribution training**。用 4M 网络视频预训练 + zero-shot eval, 居然比专门在 nuScenes 上 train 的模型还好。这是 foundation model 时代的典型现象——大 scale pretrain 的 generalization 超过小 scale specialized training。

### 7.3 Reward correlation (Figure 7)

250 对 (expert, non-expert) trajectory pair, V2R 给 expert 更高 reward 的比例在 CARLA 和 NAVSIM 上都显著高于 baseline。这是 reward model 能区分好坏 behavior 的直接 evidence。

### 7.4 DCL $K$ 值 ablation (Figure 10)

$K=4$ 是 sweet spot。$K=1$ 只监督 1-step motion, 太局部; $K$ 太大引入 noise。这个值对应 latent space 约 1.6s, 恰好覆盖"yielding"这种慢动作的关键时间窗口。

## 8. 放到 research landscape 里看

ReSim 踩在好几条 active research line 的交叉点上:

### 8.1 World model 谱系
- **Classical latent**: World Models (Ha & Schmidhuber 2018, https://arxiv.org/abs/1806.01922) → Dreamer (https://arxiv.org/abs/1912.01603) → DreamerV3 (https://arxiv.org/abs/2301.04104)。全在 latent space 做 dynamics + reward + policy。
- **JEPA**: LeCun (https://openreview.net/pdf?id=BZ5a1r-kVsf), V-JEPA (https://arxiv.org/abs/2301.08243)。Latent predictive learning, 不生成 pixel, 更 sample efficient。
- **Diffusion world model**: DIAMOND (https://arxiv.org/abs/2405.12358, Atari), Genie (https://arxiv.org/abs/2406.19102, platformer), GameNGen (https://arxiv.org/abs/2408.14846, DOOM)。ReSim 属于这一支, 但 driving 是 visual complexity 最高的 domain。

### 8.2 Driving world model 谱系
- **Video generation**: GAIA-1 (https://wayve.ai/), GenAD (https://github.com/OpenDriveLab/GenAD), DriveDreamer (https://github.com/fudangzhang/DriveDreamer), Vista (https://github.com/OpenDriveLab/Vista), Drive-WM (https://github.com/OpenDriveLab/Driving-with-World-Model)。
- **BEV/occupancy**: OccWorld (https://arxiv.org/abs/2404.08346), GaussianWorld, BEVWorld。
- **Autoregressive**: DrivingGPT (https://arxiv.org/abs/2412.18607), LAW (ICLR 2025)。
- **Sim-conditioned**: SimGen (https://arxiv.org/abs/2411.04983), Bench2Drive-R (https://arxiv.org/abs/2412.09647)。

ReSim 的独特定位: **第一个把 simulator data 混进 real-world video world model 训练, 并用 video 接口做 reward**。Vista 解决了 visual fidelity + versatile control, 但 action coverage 受限; SimGen 用 sim condition 生成 real-style scene, 但是 generation 不是 controllable world model; ReSim 合并两边。

### 8.3 Foundation model as sim2real bridge
- **传统 sim2real**: domain randomization (Tobin et al. IROS 2017), sim appearance rendering 逼真。
- **Foundation model 时代**: 用 DINOv2 / CLIP / VLM 当 sim 和 real 之间的 semantic bridge。只要两边在 foundation feature space 对齐, sim 里学的 function 能 transfer 到 real。VIP (https://arxiv.org/abs/2210.04498), LIV (https://arxiv.org/abs/2306.13200), ELLA (https://arxiv.org/abs/2310.12931) 在 robotics 里做这个; ReSim 在 driving 里做这个。

这是 sim2real 在 foundation model 时代的新 formulation, 比 domain randomization 优雅得多。

## 9. 跟 Karpathy 你常讲的方向的连接

你在多个场合提过几个观点, ReSim 都踩在点上:

1. **"Generative model is world model in disguise"**: ReSim 直接把 video diffusion model 当 world model, 加 action conditioning + reward head 就从"漂亮生成"变成"可决策"。

2. **"Software 3.0 = prompt + neural net"**: ReSim 的 high-level command ("Turning left") 就是 prompt, 同一个 neural net 在不同 prompt 下 behave differently。

3. **"Big model + small adapter"**: Stage 2 frozen DiT + LoRA + 新 trajectory encoder, 跟你在 LLM 上推的 PEFT 思路一致。

4. **"Data is the bottleneck, not algorithm"**: ReSim 的核心 contribution 是 data curation(混 CARLA), 架构上是 CogVideoX + Vista 组合。

5. **"Intelligence = predictive learning + interaction"**: ReSim 把 predictive(video diffusion) + interaction(action conditioning + reward)在真实 driving task 上拼齐了。

6. **你最近常提的 "micro-LLM on device"**: ReSim 的 limitation 正是推理太慢(4s video 要 2 min on A100), 这跟所有 diffusion world model 一样。你提到过 consistency model、rectified flow distillation 是解决方向, ReSim 的 future work 也指向这条路。

## 10. Limitations 和我的额外观察

Paper 自己承认:
- **Inference 太慢**: 4s video 在 A100 上 2 分钟, onboard 不现实。潜在解法: consistency model (https://arxiv.org/abs/2303.01958)、rectified flow distillation (https://arxiv.org/abs/2209.03003)、single-step policy distillation (Video Prediction Policy, https://arxiv.org/abs/2412.14803)。
- **Front-view only**: 跟 UniAD / VAD 多 camera 系统不兼容, 限制 closed-loop benchmark。
- **No quantitative closed-loop metric**: 现在 closed-loop 是 qualitative demo。

我的额外观察:

- **V2R reward 维度被 CARLA metric 限定了**: infraction score 只覆盖 collision / red light / off-road / low speed, 无法 capture "driving comfort"、"social compliance" 这种细维度。未来可以 multi-head reward 或 VLM-based reward(直接问 VLM"这个 driving 看起来 safe 吗")。

- **DCL 在 latent space 做**: 没在 pixel space 做(计算成本)。但 latent motion 不完全等价 pixel motion, VAE 压缩可能丢 fine-grained motion。如果 VAE 把小物体的 motion 压没了, DCL 就监督不到。

- **Unbalanced noise sampling 是 hack**: 长远看 Min-SNR-style loss reweighting 更 principled, 但 ReSim 的 brute force 在 driving 上 work, 说明这个 domain 下 high-noise timestep 最关键。

- **Trajectory conditioning granularity**: 8 个 2Hz waypoint 对 4s 来说 granularity 不够。要 capture "微秒级 steering 调整" 需要 10Hz continuous control conditioning, 但训练 stability 会下降。

- **Non-expert data diversity**: CARLA random exploration 不一定覆盖所有 real-world hazardous mode——高速爆胎 drift、湿滑路面 understeer, 这些 CARLA 默认 physics 不一定准。

- **Sim2real gap 在 reward 维度**: V2R 在 CARLA 学的"撞车 visual concept" transfer 到 real video 依赖 DINOv2 feature 的 domain invariant 性。如果 real-world 撞车的 visual pattern 跟 CARLA 差异太大(比如真实撞车有碎片、烟雾, CARLA 没有), transfer 会有 gap。

## 11. 未来方向猜测

顺着 ReSim logic:

1. **在 ReSim 里 RL 训 policy**: Dreamer-style, 在 ReSim imagination 里 rollout, V2R 当 reward, backprop 通过 diffusion(需要 score-based gradient 或 value function approximation)。Paper 的 future work 明确提到这个方向。

2. **Multi-view ReSim**: 扩展到 6-camera, 跟 BEVFormer/VAD 兼容。需要 multi-view consistency loss(WoVoGen, https://arxiv.org/abs/2312.08051 有探索)。

3. **Real-time ReSim**: 蒸馏到 consistency model 或 single-step predictor, 跑 10Hz onboard。这是 onboard deployment 的硬性要求。

4. **VLM-based reward**: 把 V2R 换成 VLM 直接 judge video("这个 driving safe 吗?"), reward 信号更丰富、更 transferable。参考 Eureka (https://arxiv.org/abs/2310.12931) 在 robotics 里的成功。

5. **Long-horizon planning**: ReSim 已能 rollout 30+s(Figure S.17), 可做 longer-horizon planning, 比如 10s trajectory optimization。

6. **Closed-loop benchmark**: 定量 metric 评估 policy 在 ReSim 里的 closed-loop performance。这需要一个公平的 protocol, 类似 CARLA Leaderboard 但是在 ReSim 的 visual space 里。

## 12. 一句话 intuition

ReSim 的核心 thesis, 用大白话: **driving world model 的可靠性取决于它见过多少种 action, real-world data 全是安全驾驶所以 action 维度不够; 用 simulator 补 action 维度, 用 real video 补 visual 维度, 用 foundation model 的通用 feature 当两边对齐的桥梁**。

这个 thesis 比 ReSim 本身更 general——它适用于所有"real data 安全但 narrow, sim data 自由但 ugly"的 domain。Robotics、medical、safety-critical control 都能用同样的 heterogeneous data + foundation model bridge 范式。这其实是 foundation model 时代 sim2real 的新 formulation, 比 domain randomization 优雅得多, 也比纯 sim training 更接近 real-world deployment。

参考链接汇总:
- ReSim: https://opendrivelab.com/ReSim
- Vista: https://github.com/OpenDriveLab/Vista
- CogVideoX: https://github.com/THUDM/CogVideo
- DINOv2: https://github.com/facebookresearch/dinov2
- CARLA: https://carla.org/
- NAVSIM: https://github.com/autonomousvision/navsim
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- OpenDV/GenAD: https://github.com/OpenDriveLab/GenAD
- Dreamer: https://danijar.com/project/dreamer/
- DIAMOND: https://github.com/eloialonso/diamond
- Genie: https://arxiv.org/abs/2406.19102
- GameNGen: https://arxiv.org/abs/2408.14846
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- V-JEPA: https://arxiv.org/abs/2301.08243
- Min-SNR: https://arxiv.org/abs/2303.09556
- EDM: https://arxiv.org/abs/2206.00364
- VIP (visual reward): https://arxiv.org/abs/2210.04498
- Eureka (LLM reward): https://arxiv.org/abs/2310.12931
- GR-2: https://arxiv.org/abs/2410.06158
- DrivingGPT: https://arxiv.org/abs/2412.18607
- XVO: https://github.com/lostxine/XVO
- Curse of rarity: https://www.nature.com/articles/s41467-024-49134-4
- Bench2Drive-R: https://arxiv.org/abs/2412.09647
- SimGen: https://arxiv.org/abs/2411.04983
- Du et al. universal policies: https://arxiv.org/abs/2212.11985
- Video Prediction Policy: https://arxiv.org/abs/2412.14803
- Consistency model: https://arxiv.org/abs/2303.01958
- Rectified flow: https://arxiv.org/abs/2209.03003

---

# ReSim: 把 simulator 的"自由度"嫁接到 real-world world model 上

## 1. 核心问题与 motivation 的直觉

Driving world model 当前的核心 bottleneck 在于 **action coverage**。Real-world driving data(OpenDV、nuScenes、Waymo、NAVSIM)几乎全是 expert trajectories, state-action space 被 safety / regulation 强约束, 严重 underrepresent 了 hazardous 事件(collisions、off-road、red-light violations)。当 policy 把这种 world model 拿来做 evaluation 或 reward estimation 时, 一旦 action 落到 OOD 区域, world model 会 hallucinate 成 "依然安全向前开", 完全 fail to follow 指定的 trajectory。

这个问题本质上是 **data distribution 不覆盖 action space** 的问题, 在 AV 领域被称为 "curse of rarity" (Liu & Feng, Nature Communications 2024, https://www.nature.com/articles/s41467-024-49134-4)。ReSim 的 key move 是: 既然 real world 不让做危险动作, 就从 CARLA 里采集 non-expert data, 然后用 video 这种 unified interface 把 heterogeneous data 灌进同一个 diffusion world model。

这跟机器人里的 sim2real + real2sim 思路是同源的: real video 提供 visual fidelity 和 generalization, sim 提供 action coverage 和 reward signal。ReSim 把这两个 axis 在一个 controllable video generator 里 fuse 起来。

参考链接:
- ReSim 项目页: https://opendrivelab.com/ReSim
- Curse of rarity: https://www.nature.com/articles/s41467-024-49134-4
- 类似思路在 robotics 里的对应 SimGen (Zhou et al., NeurIPS 2024): https://arxiv.org/abs/2411.04983

## 2. Heterogeneous data compilation 的细节

三类数据被 mix 到一起:

| Source | 规模 | 作用 | Annotation |
|---|---|---|---|
| OpenDV (https://github.com/OpenDriveLab/GenAD) | 4M clips, 1700h | 视觉 generalization, scenario coverage | pseudo-labeled command (turning left / forward / right), 无 trajectory label |
| NAVSIM navtrain (https://github.com/autonomousvision/navsim) | 85K clips | 严格标注的 expert action (future waypoints), 用于学 action conditioning | ego trajectory + command |
| CARLA via Bench2Drive (https://github.com/Thinklab-SJTU/Bench2Drive) | 88K clips, 一半 expert 一半 non-expert | 危险 action 的 coverage + reward supervision | ego trajectory + infraction score |

CARLA 里 non-expert 数据的采集方式很关键: agent 从 predefined set 里 **randomly sample steering angle + throttle + behavior pattern**。这意味着 action space 是被 deliberately 扩张过的, 不是 PDM-Lite 这种 rule-based planner 的 narrow 分布。这一点决定了 ReSim 能不能 follow 像 "急转向冲出路外" 这种 hazardous action。

每个 video sample 4.9s @ 10Hz, 前 9 帧作为 context, 后 40 帧作为 prediction target。频率 10Hz 但 action waypoint 是 2Hz(8 个 waypoint 覆盖 4s)——这跟大部分端到端 driving 的 waypoint 设定一致(Transfuser、VAD 都是这个频率)。

## 3. Architecture 深度解析

ReSim 的 backbone 是 **CogVideoX** (https://github.com/THUDM/CogVideo), 一个 2B 参数的 text-to-video diffusion transformer, 原本只 conditioned on text。ReSim 在它上面做了几个改造:

### 3.1 整体结构

- **3D Causal VAE**: 把 raw video (T×H×W×3) 压成 compact latent (T'×H'×W'×C), CogVideoX 原配, time 维度 4× 压缩, space 8× 压缩, channel 16。所以 49 帧 512×896 → 13 帧 64×112×16 latent。
- **T5 encoder**: 把 high-level command 编码成 text embedding("This video depicts a realistic view from the driver's perspective of a car driving on the road. Turning left.")。
- **Trajectory encoder**: 两个 attention block + linear head, 把 8 个 2Hz waypoint 投影到 DiT 的 input space, 跟 video latent token 拼在一起。
- **DiT denoiser**: 2B 参数, expert transformer 结构(CogVideoX 的设计, 是一个 T5-structured 3D attention, 用 expert splitting 让 temporal 和 spatial attention 解耦)。

借鉴 Vista (https://github.com/OpenDriveLab/Vista, NeurIPS 2024) 的做法: 历史帧的 latent input 用 clean version 而不是 noised version, 这样 historical context 不参与 denoising, 只作为 conditioning。这避免了 history 和 future 一起 denoise 造成的耦合问题。

### 3.2 Video diffusion loss (Eq. 1)

$$
\mathcal{L}_{\mathrm{diffusion}} = \mathbb{E}_{x, \epsilon, t} \left[ \| x^{k:} - D_\theta(x_t; t, h, c[, a])^{k:} \|^2 \right]
$$

变量含义:
- $x$: clean video latent(经过 VAE 压缩后的)
- $\epsilon$: 从 $\mathcal{N}(0, I)$ 采样的 noise
- $t$: diffusion timestep, $\in [0, 1000]$
- $x_t$: 在 $x$ 上加 noise $\epsilon$ 得到的 noised latent, $x_t = \sqrt{\bar\alpha_t} x + \sqrt{1-\bar\alpha_t} \epsilon$
- $D_\theta$: 参数为 $\theta$ 的 diffusion transformer(denoiser)
- $h$: 历史帧的 clean latent(conditioning)
- $c$: text command 的 embedding
- $a$: action, 即未来 ego waypoints sequence(optional)
- $k$: frame index, loss 只从第 $k$ 帧开始算, 不监督历史帧本身
- $[\cdot]$: 表示 $a$ 是 optional, 训练时 50% drop (classifier-free guidance)

这个 loss 的关键设计是 $x^{k:}$ 这个 slice——历史帧不参与 denoise loss, 这样 history 是 pure conditioning 而非 reconstruction target。

### 3.3 Dynamics Consistency Loss (Eq. 2) — 这是最重要的创新

标准 video diffusion loss 是逐 frame 独立监督的, 类似 image diffusion 的 frame-wise MSE 加和。这种 supervision **丢失了 temporal correlation**——模型可以在低 noise timestep 上"抄近路", 通过对相邻 frame 做 averaging 就能 minimize loss, 完全不需要学真正的 motion pattern。这种现象在 video generation 里被广泛讨论(Track4Gen CVPR 2025, MotiF CVPR 2025, https://arxiv.org/abs/2411.02385)。

ReSim 的做法: 在 latent space 上显式监督 **"latent motion"**, 也就是 frame 之间的 latent 差值:

$$
\mathcal{L}_{\mathrm{dynamics}} = \mathbb{E}_{x, \epsilon, t} \left[ \sum_{j=1}^{K} \sum_{i=1}^{N-j} \frac{1}{s} \| (d^{i+j} - d^i) - (x^{i+j} - x^i) \|^2 \right]
$$

变量含义:
- $x$: ground-truth video latent
- $d = D_\theta(x_t; \cdots)$: 模型预测的 denoised latent
- $i$: frame index in latent space(注意是 VAE 压缩后的 frame, 不是 raw video frame)
- $j$: 时间间隔 offset, 从 1 到 $K$
- $K$: 最大考虑的时间间隔, 实验 set to **4**
- $N$: latent 的总 frame 数
- $s$: normalization factor, 是每个 interval 下 |motion disparity| 的平均值, 用来稳定 loss magnitude

直觉: 这个 loss 强制模型预测的 **运动差分** 匹配 ground-truth 的 **运动差分**, 而且在多个时间尺度上同时监督($j=1$ 是短期 motion, $j=4$ 是较长期 motion, 比如 yielding 这种慢动作)。$K=4$ 不是随便选的——它对应 latent space 里大约 1.6 秒(因为 VAE 时间压缩 4×, 4 个 latent frame = 16 raw frames @ 10Hz = 1.6s), 恰好覆盖 "刹车让行" 这种动作的关键时间窗口。

这个 loss 跟 **temporal difference learning** 在 RL 里的思路是同源的: 监督 state 的变化而不是 state 本身, 强迫模型 capture dynamics 而非 memorize appearance。也跟 flow matching / rectified flow 里直接预测 velocity field 的精神一致(https://arxiv.org/abs/2209.03003, https://arxiv.org/abs/2403.03206)。

总 loss: $\mathcal{L} = \mathcal{L}_{\mathrm{diffusion}} + \lambda \mathcal{L}_{\mathrm{dynamics}}$, $\lambda = 0.1$。

### 3.4 Unbalanced noise sampling — 反直觉但很关键

CogVideoX 默认用 uniform timestep sampling。ReSim 发现 uniform sampling 在 driving 这种有 complex dynamics 的 domain 上会出问题: 模型在低 noise timestep 上可以通过简单 averaging 相邻 frame 来 reconstruct, 不学真正的 motion。

解决方法: 把 timestep $t \in [500, 1000]$ 的采样频率从 1/2 提高到 2/3。也就是 **强迫模型在更多高 noise 的 step 上训练**, 这时 input 被严重 corrupt, 模型无法靠 local averaging 走捷径, 必须 learn 真正的 motion structure 才能 recover。

这个 insight 跟 **Min-SNR weighting** (Hang et al., ICCV 2023, https://arxiv.org/abs/2303.09556) 和 EDM (Karras et al., NeurIPS 2022, https://arxiv.org/abs/2206.00364) 的精神相通: 不是所有 timestep 同等重要, 高 noise timestep 携带更多 "structure" 信息。但 ReSim 的做法更 brute force——直接 bias sampling distribution, 而不是 reweight loss。

Figure 9 的 ablation 视觉效果很明显: uniform sampling 下车辆 motion 不连贯, 场景 layout 会漂移; unbalanced sampling 下 motion 和 layout 都更 consistent。

### 3.5 Progressive multi-stage learning

三 stage curriculum:

| Stage | DiT | LoRA | Traj Enc | Data | Res | BS | LR | Steps |
|---|---|---|---|---|---|---|---|---|
| 1 | Trainable | - | - | OpenDV | 512×896 | 80 | 1e-5 | 20K |
| 2 | Frozen | Trainable | Trainable | OpenDV+NAVSIM+CARLA | 256×448 | 160 | 5e-5 | 80K |
| 3 | Trainable | Trainable | Trainable | OpenDV+NAVSIM+CARLA | 512×896 | 80 | 5e-5 | 50K |

直觉:
- **Stage 1**: 先让 CogVideoX 学会 "在 driving domain 里做 visual context + text-conditioned 未来预测"。这一步只跟 visual fidelity 有关, 还没引入 action。
- **Stage 2**: 引入 action condition。但为了高效, 冻结 DiT 主干, 只训 trajectory encoder + LoRA adapter。低分辨率(256×448)加速训练。这一步其实是让 trajectory encoder 学会 "怎么把 waypoint 翻译成 DiT 能理解的 visual motion instruction"。NAVSIM 的 trajectory 50% dropout 是为了支持 action-free prediction(后面 video prediction-based policy 要用), 但 CARLA 的 trajectory **完全不 drop**——因为 hazardous action 无法从 visual context infers, 必须显式 condition。
- **Stage 3**: 全主干 fine-tune, 高分辨率, 让 model 最终收敛到 512×896 的 driving video quality。

40× A100, 14 天, 这是个不小的 compute budget。

### 3.6 Conditioning augmentation

借鉴 Genie (https://arxiv.org/abs/2406.19102) 和 Cascaded Diffusion (Ho et al. 2022): 对 history latent 也加 noise, 但用独立采样的 timestep $t\text{-aug}$, 推理时设为 0。这缓解 long-rollout 的 error accumulation——一个迭代式 world model 最怕的就是 context 里的 noise 在多步 rollout 后被放大。

## 4. Video2Reward (V2R) — 把 video 翻译成 scalar reward

### 4.1 设计

V2R 结构很简单:
- Backbone: **frozen DINOv2** (https://github.com/facebookresearch/dinov2), self-supervised ViT, 输出 per-frame feature
- Head: 两个 spatio-temporal attention block + MLP, aggregate 所有 frame feature → scalar reward
- 训练 supervision: **CARLA infraction score**(collisions、red light、off-road、unreasonable low speed 等综合惩罚)

### 4.2 为什么这样设计能 generalize 到 real world

这是整篇 paper 最 elegant 的部分。直觉链路:
1. CARLA 里的 infraction score 是 ground truth reward signal, 涵盖 expert + non-expert 全谱。
2. DINOv2 是 self-supervised, 在 142M 自然图像上 train, 学到的是 **物体、运动、场景的通用 visual feature**。
3. ReSim 生成的 video 是 real-world 风格的(因为 stage 1 + 3 在 OpenDV/NAVSIM 上 train), 不是 CARLA 风格的。
4. V2R 在 CARLA video 上学的 "what does a bad outcome look like visually" 通过 DINOv2 的通用 feature space, **transfer 到 ReSim 生成的 real-style video 上**。

这等价于: **reward function 的 "概念" 在 CARLA 学, 但 "视觉接口" 是通用 foundation model**。这跟 Yann LeCun 一直推的 JEPA 思路(https://openreview.net/pdf?id=BZ5a1r-kVsf)、以及 robot learning 里用 VIP / LIV (https://sites.google.com/view/robot-rewards) 这种 pretrained visual representation 当 reward 是同一个 family 的 idea。

参考:
- DINOv2: https://arxiv.org/abs/2304.07193
- VIP (visual reward): https://arxiv.org/abs/2210.04498
- 整个 generative reward / VLM-as-reward 方向: https://arxiv.org/abs/2310.12931

### 4.3 训练配置

- 数据: 35K CARLA sample, 20 epochs
- Optimizer: AdamW, lr 1e-3
- Input: 224×224 frames(很小, 因为只做 reward regression, 不需要高分辨率 detail)
- Loss: 应该是 MSE 或 L1 on infraction score(paper 没明说, 但从 setup 看是 regression)

## 5. Applications — 把 world model 真的用起来

### 5.1 Video prediction-based policy

ReSim 在 action-free 模式下(不给 $a$, 只给 history + command)生成 future video, 然后用 **Inverse Dynamics Model (IDM)** 把 video 转成 ego trajectory。

IDM 架构: XVO backbone (https://github.com/lostxine/XVO) + attention head, 输出 8 个 2Hz waypoint。在 NAVSIM navtrain 上训 100 epochs, 前 50 epochs lr 1e-4, 后 50 epochs 1e-5。

NAVSIM navtest PDMS 结果(Table 4):

| Method | Type | PDMS ↑ |
|---|---|---|
| VO planner (XVO 单独) | E2E | 78.4 |
| UniAD | E2E | 83.4 |
| Transfuser | E2E | 84.0 |
| DrivingGPT | WM+E2E | 82.4 |
| LAW | WM+E2E | 84.6 |
| GT Future + IDM (oracle) | - | 90.8 |
| **ReSim + IDM (ours)** | **WM + IDM** | **86.6** |

关键 insight: VO planner(78.4) 和 ReSim+IDM(86.6) 共享同一个 IDM, 差距 8.2 PDMS 完全来自 ReSim 生成的 video quality。这等于在 ablate "一个好的 generative world model 当 policy" 的 value。

这跟 robot learning 里 **GR-1 / GR-2** (https://arxiv.org/abs/2410.06158) 的思路一致: video prediction pretraining → 转成 action via inverse dynamics。也跟 Du et al. 的 "universal policies via text-guided video generation" (NeurIPS 2023, https://arxiv.org/abs/2212.11985) 同源。

### 5.2 Reward-guided policy selection

给定两个 candidate policy(Transfuser + LTF), 对每个 scenario 让两个 policy 各出一条 trajectory, ReSim 把每条 trajectory render 成 video, V2R 给 reward, 选 reward 高的执行。

Table 3:
- Transfuser alone: 47.7
- LTF alone: 47.2
- Uniform average ensemble: 66.8
- Vista reward: 59.2
- **ReSim + V2R: 74.1**
- ReSim w/o sim + V2R: 69.7
- Oracle (用 GT PDMS 选): 94.2

ReSim + V2R 比 average ensemble 高 7.3 PDMS, 比 Vista reward 高 14.9, 离 oracle 只有 20 的 gap。这证明 V2R 的 reward 信号确实 capture 了"哪条 trajectory 更安全"的信息。

这个范式其实跟 LLM 里的 **Best-of-N sampling with reward model** 完全同构: 用 generator 生成多个 candidate, 用 reward model rerank, 选最好的。在 AV 里是第一次把这个 pipeline 做得这么完整。

### 5.3 Closed-loop visual simulation

把 ReSim 当 environment: policy 输出 4s trajectory → ReSim 生成 4s video → 把最后 9 帧当新 context → policy 再决策 → 循环。

这是把 ReSim 当成一个 **learned simulator**, 跟 Genie (DeepMind, https://arxiv.org/abs/2406.19102)、GameNGen (https://arxiv.org/abs/2408.14846)、DIAMOND (https://arxiv.org/abs/2405.12358) 是同一类工作, 但是 driving domain。

Figure 8 展示了 ReSim 把一个 VO-based policy 推到 pre-recorded dataset 里 never 出现过的 state, 这是 open-loop benchmark 永远做不到的。

## 6. 实验结果细节解读

### 6.1 Action controllability (Table 1, Waymo zero-shot)

| Method | Action-free ↓ | Expert Action ↓ |
|---|---|---|
| GT Future | 0.58 | 0.58 |
| Vista | 5.68 | 1.89 |
| ReSim w/o sim | 1.47 | 1.18 |
| **ReSim** | **1.13** | **0.86** |

ReSim 在 expert action 下比 Vista 好 54%, 在 action-free(纯 visual imagination)下好 80%。注意 "w/o sim" 版本(去掉 CARLA 数据)仍然显著好于 Vista, 说明 multi-stage training + DCL + unbalanced sampling 本身就有效, 但 **加上 CARLA 数据再额外提升一个台阶**, 尤其在 non-expert action 上更明显(看 Figure 4 的人类评估, ReSim 在 "trajectory following for non-expert action" 上几乎垄断了 preference)。

### 6.2 Visual fidelity (Table 2, nuScenes zero-shot)

ReSim FID 5.2, FVD 50.4, 是 SOTA, 而且 **没在 nuScenes 上 train 过**。比 in-distribution 的 Vista(6.9/89.4) 还好。这归功于 OpenDV 的规模(4M clips, 1700h)和 CogVideoX backbone 的强 visual prior。

### 6.3 Reward correlation (Figure 7)

在 250 对 (expert, non-expert) trajectory pair 上, V2R 在 CARLA 和 NAVSIM 上都给了 expert 更高 reward 的比例显著高于 baseline。这是 reward model 区分 "好" "坏" behavior 的直接 evidence。

### 6.4 Ablation: DCL 的 $K$ 选择 (Figure 10)

$K=4$ 是 sweet spot。$K=1$ 只监督 1-step motion, 太局部; $K$ 太大引入 noise。这个值跟前面算的"1.6s 时间窗口覆盖 yielding 动作"对应。

## 7. 在更广 research landscape 里的位置

ReSim 横跨了好几个 active research direction, 我把它放进几条 lineage 里:

### 7.1 World model lineage
- **Classical**: World Models (Ha & Schmidhuber, NeurIPS 2018, https://arxiv.org/abs/1806.01922) → Dreamer (Hafner et al., ICLR 2020, https://arxiv.org/abs/1912.01603) → DreamerV3 (https://arxiv.org/abs/2301.04104)。Latent dynamics model + reward + policy 全在 latent space。
- **JEPA branch**: LeCun (https://openreview.net/pdf?id=BZ5a1r-kVsf), V-JEPA (https://arxiv.org/abs/2301.08243), I-JEPA (https://arxiv.org/abs/2301.08243)。Predictive learning in latent space, 不生成 pixel。
- **Diffusion branch**: DIAMOND (https://arxiv.org/abs/2405.12358, Atari), Genie (https://arxiv.org/abs/2406.19102, platformer), Diffusion Forcer (https://arxiv.org/abs/2402.00751), GameNGen (DOOM, https://arxiv.org/abs/2408.14846)。ReSim 属于这一支, 但在 driving 这个 open-domain、real-world 视觉复杂度最高的场景。

### 7.2 Driving world model lineage
- **Generative**: GAIA-1 (Wayve, https://wayve.ai/), GenAD (CVPR 2024, https://github.com/OpenDriveLab/GenAD), DriveDreamer (ECCV 2024, https://github.com/fudangzhang/DriveDreamer), DriveDreamer-2, GEM (CVPR 2025), Vista (NeurIPS 2024, https://github.com/OpenDriveLab/Vista), Drive-WM (CVPR 2024, https://github.com/OpenDriveLab/Driving-with-World-Model)。
- **BEV / occupancy**: OccWorld (ECCV 2024, https://arxiv.org/abs/2404.08346), GaussianWorld, BEVWorld, Visual Point Cloud Forecasting (CVPR 2024)。
- **Autoregressive**: DrivingGPT (https://arxiv.org/abs/2412.18607), LAW (ICLR 2025)。
- **Closed-loop with sim**: Bench2Drive-R (https://arxiv.org/abs/2412.09647), SimGen (NeurIPS 2024, https://arxiv.org/abs/2411.04983)。

ReSim 的独特定位: 它是 **第一个把 simulator 数据混进 real-world video world model 训练**, 并用 video 接口做 reward 的。Vista 解决了 visual fidelity + versatile control, 但 action coverage 仍受限; SimGen 用 sim condition 生成 real-style scene, 但不是 controllable world model; ReSim 把两边合并。

### 7.3 Reward from video / VLM-as-reward
- **Visual reward in robotics**: VIP (https://arxiv.org/abs/2210.04498), LIV (https://arxiv.org/abs/2306.13200), ELLA (https://arxiv.org/abs/2310.12931)。
- **VLM-as-reward**: RLAIF (https://arxiv.org/abs/2309.00267), Text2Reward, Eureka (https://arxiv.org/abs/2310.12931)。
- **Driving 里的 reward**: Drive-WM 用 rule-based reward + 3D perception, 对 sensor config 敏感; Vista 用 uncertainty-based reward, 难以分辨具体 behavior 类型(off-route、collision 区分不开)。ReSim 的 V2R 通过 simulator supervision + DINOv2 feature, 直接从 video regress 一个综合 score, 是个更优雅的接口。

### 7.4 Video prediction → policy
- **Robotics**: GR-1 (https://arxiv.org/abs/2312.06595), GR-2 (https://arxiv.org/abs/2410.06158), Video Prediction Policy (https://arxiv.org/abs/2412.14803), Du et al. universal policies (https://arxiv.org/abs/2212.11985)。
- **Driving**: 视觉预测 → IDM → trajectory 这个 pipeline 在 ReSim 之前不是主流, 主流是 end-to-end direct regression(UniAD、Transfuser、VAD、DriveTransformer)或者 latent world model(LAW、DrivingGPT)。ReSim 用纯 video 接口做到 86.6 PDMS 是个很强的 evidence, 说明 **video prediction 已经隐式 encode 了 driving policy 所需的几乎所有信息**。

### 7.5 Heterogeneous data for sim2real
- **Domain randomization**: 早期 robotics sim2real 的支柱(Tobin et al., IROS 2017)。
- **Real2sim2real**: SimGen, Bench2Drive-R, ReSim 属于这一波。核心 idea: 用 real data 提供 visual prior, 用 sim data 提供 action / reward coverage, 再把学到的 model 用到 real。
- **Foundation model as bridge**: 用 DINOv2 / CLIP / VLM 这种通用 visual encoder 当 sim 和 real 之间的 "通用语义接口", 这是 sim2real 在 foundation model 时代的新范式。

## 8. 跟 Karpathy 你自己常讲的方向的连接

你在多个场合(e.g., Stanford CS231N 之后的 lecture、Yann LeCun 对话、podcast)都强调过几个观点, ReSim 都踩在这些点上:

1. **"Generative model 是 world model in disguise"**: ReSim 直接把 video diffusion model 当 world model 用, action conditioning + reward head 就是把它从 "漂亮生成" 变成 "可决策" 的关键改造。
2. **"Software 3.0 = prompt + neural net"**: ReSim 的 high-level command ("Turning left") 就是 prompt, 让同一个 neural net 在不同 prompt 下 behave differently。
3. **"Big model + small adapter"**: Stage 2 用 frozen DiT + LoRA + 新 trajectory encoder, 跟你在 LLM 上推的 PEFT 思路一致。
4. **"Data is the bottleneck, not algorithm"**: ReSim 的核心 contribution 是 **data curation**(混 CARLA), 而不是新架构。架构上几乎是 CogVideoX + Vista 的组合, 真正让 work 的是 data。
5. **"Intelligence = predictive learning + interaction"**: ReSim 把 predictive(video diffusion)+ interaction(action conditioning + reward)在 driving 这个真实 task 上拼齐了。

## 9. Limitations 和 open questions

Paper 自己承认的:
- **Inference latency**: 4s video 在 A100 上要 2 分钟, 完全 onboard 不现实。这是所有 diffusion world model 的通病。潜在解法: consistency model、rectified flow distillation、single-step policy distillation(像 Du et al. 在 robot 上做的, https://arxiv.org/abs/2412.14803)。Karpathy 你在 recent talk 里也提到过, diffusion model 推理慢是它进入 closed-loop control 的最大障碍。
- **Front-view only**: 跟 UniAD / VAD 这种 multi-camera setup 不兼容, 限制 closed-loop benchmark 的 policy 选择。
- **No true closed-loop benchmark**: 现在 closed-loop visual simulation 是 qualitative demo, 没有定量 metric。

我的额外观察:
- **V2R 的 reward 信号仍来自 CARLA metric 定义**: 这意味着 reward 维度被 CARLA 的 infraction score 限定了, 无法 capture 比如 "driving comfort"、"social compliance" 这种更细的 reward dimension。未来可以 multi-head reward 或 VLM-based reward。
- **DCL 在 latent space 做**: 没在 pixel space 做, 原因是计算成本。但 latent motion 不完全等价于 pixel motion, VAE 压缩可能丢掉一些 fine-grained motion。
- **Unbalanced noise sampling 是 hack**: 长远看, Min-SNR-style 的 loss reweighting 应该更 principled, 但 ReSim 的 brute force bias 在 driving 上 work, 说明这个 domain 下 high-noise timestep 确实携带了关键的 structure 信息。
- **Trajectory conditioning 是 waypoint, 不是 continuous control**: 8 个 2Hz waypoint 对 4s 来说 granularity 不够。如果要 capture "微秒级 steering 调整" 这种 action, 需要 10Hz continuous control conditioning, 但训练数据 stability 会下降。
- **Non-expert data 的 diversity**: CARLA random exploration 是否覆盖了所有 real-world hazardous mode? 比如高速爆胎后的 drift、湿滑路面的 understeer——这些 CARLA 默认 physics model 不一定 simulate 得准。

## 10. 未来方向(我的猜测)

顺着 ReSim 的 logic, 下一步可能是:

1. **在 ReSim 里 RL 训 policy**: Dreamer-style, 在 ReSim 的 imagination 里 rollout policy, 用 V2R 当 reward, 反向传播通过 diffusion(需要 score-based gradient 或者 value function approximation)。
2. **Multi-view ReSim**: 扩展到 6-camera, 跟 BEVFormer / VAD 兼容。需要 multi-view consistency loss(参考 WoVoGen, https://arxiv.org/abs/2312.08051)。
3. **Real-time ReSim**: 蒸馏到 consistency model 或 single-step predictor, 让它能跑 10Hz onboard。
4. **VLM-based reward**: 把 V2R 换成 VLM 直接 judge video("这个 driving 看起来 safe 吗?"), reward 信号会更丰富, 更 transferable。
5. **Long-horizon planning**: ReSim 已经能 rollout 30+s(Figure S.17), 可以做 longer-horizon planning, 比如 10s trajectory optimization。

## 11. 一个直觉总结

ReSim 的核心 thesis, 一句话: **driving world model 的 reliability 取决于 action coverage, 而 real-world data 在 action 维度上是不够的; 用 simulator 把 action 维度补全, 用 real video 把 visual 维度做精, 用 foundation visual feature 把两边在 semantic space 里对齐**。

这个 thesis 比看上去更 general——它适用于所有 "real data 安全但 narrow, sim data 自由但 ugly" 的 domain。Robotics、medical、safety-critical control 都可以用同样的 heterogeneous data + foundation model bridge 的范式。这其实是个 sim2real 在 foundation model 时代的新 formulation。

参考链接汇总:
- ReSim: https://opendrivelab.com/ReSim
- Vista: https://github.com/OpenDriveLab/Vista
- CogVideoX: https://github.com/THUDM/CogVideo
- DINOv2: https://github.com/facebookresearch/dinov2
- CARLA: https://carla.org/
- NAVSIM: https://github.com/autonomousvision/navsim
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- OpenDV / GenAD: https://github.com/OpenDriveLab/GenAD
- Dreamer: https://danijar.com/project/dreamer/
- DIAMOND: https://github.com/eloialonso/diamond
- Genie: https://arxiv.org/abs/2406.19102
- GameNGen: https://arxiv.org/abs/2408.14846
- LeCun JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- V-JEPA: https://arxiv.org/abs/2301.08243
- Min-SNR: https://arxiv.org/abs/2303.09556
- EDM: https://arxiv.org/abs/2206.00364
- VIP (visual reward): https://arxiv.org/abs/2210.04498
- Eureka (LLM reward): https://arxiv.org/abs/2310.12931
- GR-2: https://arxiv.org/abs/2410.06158
- DrivingGPT: https://arxiv.org/abs/2412.18607
- XVO: https://github.com/lostxine/XVO
- DriveLM: https://github.com/OpenDriveLab/DriveLM
- Curse of rarity: https://www.nature.com/articles/s41467-024-49134-4
- Bench2Drive-R: https://arxiv.org/abs/2412.09647
- SimGen: https://arxiv.org/abs/2411.04983
- Video as policy (Du et al.): https://arxiv.org/abs/2212.11985
- Video Prediction Policy: https://arxiv.org/abs/2412.14803
