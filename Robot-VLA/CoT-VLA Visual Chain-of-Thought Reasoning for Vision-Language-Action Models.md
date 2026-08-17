---
source_pdf: CoT-VLA Visual Chain-of-Thought Reasoning for Vision-Language-Action Models.pdf
paper_sha256: 20081e8c7efca4e4bf7f8d6e83ee69793a30961b5069975c0980dddb2cf211f0
processed_at: '2026-08-03T17:36:06-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 CoT-VLA

---

## 一句话说清楚

现在大多数 robot model 是这样工作的：给它看一张图，给它一句话，它直接吐出一个动作。**像让人蒙着眼睛只看一眼就做决定，没有"想一下"的过程**。

CoT-VLA 说：先等会儿，让我想想我接下来要去什么状态，画张图给自己看，然后再决定怎么动。**就像人做复杂事情会先在脑子里想象一下结果**。

---

## 为什么这件事重要

先回到一个朴素的问题：教 robot 做事为什么难？

难在 robot data 太少。互联网上有无限多的视频、图片、文字，但真正带"robot 应该怎么动"标注的数据只有那么几万到几十万条。Open X-Embodiment 这种 dataset 听起来很大，但跟 internet video 比就是九牛一毛。

那能不能用 internet video 教 robot？难点是视频没有 action label —— 你看到一个人切菜，但不知道每时每刻 robot joint torque 应该是多少。

之前 VLA model 完全没法用这些数据，因为它们只学"observation → action"这个 mapping，没 action label 就没法训。

CoT-VLA 的 trick：**把 task 拆成两步**。

第一步：看当前画面 + 听指令，想象接下来的画面。这一步只需要 video，不需要 action label。所以 EPIC-KITCHENS 这种人做饭视频、Something-Something 这种手跟物体互动视频都能用。

第二步：从"当前画面 + 想象的画面"生成具体动作。这一步才需要 robot demo。

这样 internet video 终于能进 VLA 训练 pipeline 了。**这是第一次**。

---

## 这个 idea 跟 LLM chain-of-thought 是同构的

LLM 里 chain-of-thought 大家都熟。问 GPT 一个数学题，它直接给答案容易错，但你让它"let's think step by step"它会写出推理过程，最后答案就对了。

为什么 CoT work？因为 $x \to y$ 这个直接 mapping 难学，但 $x \to z \to y$ 这种带中间步骤的 mapping 好学，每个 sub-mapping 都更简单。

robot 这边完全一样。$\text{observation} \to \text{action}$ 这个 mapping 难，因为 7 维 action vector 信息容量太小，所有 reasoning 都要压扁到 7 维里。

拆成 $\text{observation} \to \text{subgoal image} \to \text{action}$ 就好学了。中间那个 subgoal image 是一张 256×256×3 的图，信息容量比 7 维 action 大几个量级，相当于给 model 一个"宽跑道"去想。

**直觉**：让 model 在高维空间里 think，在低维空间里 act。

---

## 怎么生成 subgoal image

这里有个 architecture choice 很关键。CoT-VLA 不是用 Stable Diffusion 那种 diffusion model 生成图，而是用 **autoregressive next-token prediction**。

具体怎么做的？他们用 VILA-U 这个 base model。VILA-U 把 image 压成 discrete tokens，跟 text token 用同一个 vocab，这样 LLM 可以"说话一样地生成图"。

每张 256×256 的图被压成 16×16×4 个 token：
- 16×16 = 256 个空间位置
- 每个位置 4 层 residual token（用 RQ-VAE，residual quantization）

为什么 4 层？第一层 token 抓 coarse info（大致是什么颜色、什么形状），后 3 层抓 high-frequency detail。**像 JPEG 从粗到细的编码**。

LLM 先输出 256 个 spatial position 的"锚定" embedding，然后一个叫 depth transformer 的小网络 autoregressively 生成每个 position 的 4 层 token。

这跟 LLM 生成 text token 在数学上完全一样 —— 都是 next-token prediction，只是 vocab 不一样。这就让 image 和 text 在同一个 transformer 里统一了。

---

## Action 怎么生成

这部分有个 clever design。

vanilla VLA 用 causal attention 逐 token 生成 action。但 action 是结构化的 7 维 vector，逐维生成意味着第 1 维看不到第 7 维的预测，action vector 内部不一致。

CoT-VLA 的 trick：**action 用 full attention**。

text 和 image 用 causal attention（标准 next-token），但 action token 用 full attention —— 所有 action dim 互相能看见，类似 parallel decoding，一次性出 7×10=70 个 action token（7 维 × 10 步 chunk）。

这叫 **hybrid attention**。同一个 transformer 里两种 attention pattern。

Action discretization 是 OpenVLA 的标准做法：每维 action 用 256 bins 离散化，bin 宽由 training data 的 1st-99th percentile 决定。然后复用 text tokenizer 里最不常用的 256 个 token 当 action bin token。

---

## 训练数据怎么配

paper 用了三类数据混在一起 pretrain：

**Robot demos（带 action）**：
- Bridge Data V2（24.14% 权重）
- RT-1（6.90%）
- TOTO（10.34%）
- VIOLA（10.34%）
- RoboTurk（10.34%）
- Jaco Play（10.34%）
- Berkeley Autolab UR5（10.34%）
- Berkeley Fanuc（10.34%）

**Action-less videos**：
- Something-Something V2（3.45%）
- EPIC-KITCHENS-100（3.45%）

action-less video 权重小是因为他们怕 video domain gap 大，先保守用。但 Table 3 那个实验证明 video data 的潜力还没挖出来。

每个 dataset 配不同的 subgoal horizon $[n_l, n_u]$。比如 Bridge 是 [5, 10]，TOTO 是 [20, 24]，RoboTurk 是 [1, 2]。**轨迹短的设小 horizon，不然 subgoal 直接跳过 task 终点**。

训练时每次从 $[n_l, n_u]$ uniform sample 一个 n，让 model 见到不同 lookahead distance，避免过拟合到单一 horizon。

---

## 测试时怎么跑

闭环 control。Algorithm 1 伪代码：

```
拿到初始 observation 和 instruction
while True:
    生成 subgoal image（看现在的图，想 n 步后的图）
    生成 10 步 action chunk（从现在到 subgoal 怎么走）
    执行这 10 步
    重新拿 observation
    重新生成 subgoal + action chunk
    重复
```

这跟 MPC / receding horizon control 是一回事。每 10 步重新 plan，subgoal 可以被纠正。

跟 SUSIE 那种 two-stage open-loop 方法不同，SUSIE 是一次生成 subgoal 然后跟着走，错了不能纠正。CoT-VLA 是 closed-loop。

---

## 实验里最 striking 的数字

**LIBERO Long-horizon**：CoT-VLA 69.0% vs OpenVLA 53.7%。

Long-horizon suite 需要组合多个 sub-task 完成 long task。CoT 提升最大（+15.3% absolute）。这是 visual CoT hypothesis 最强 evidence —— 任务越长，planning 越重要，subgoal 越有用。

**Franka-Tabletop**：CoT-VLA 平均比 OpenVLA 高 17%。这个 setup 是 base model pretraining 时没见过的，10-150 demos fine-tune。证明 visual reasoning capability 能 transfer。

**Ground-truth vs generated goal experiment**（Table 3，最 striking）：

两个 OOD long-horizon task，比较"用 model 生成的 subgoal"vs"用人类 demo 的 ground-truth subgoal"：
- generated: 20%, 0%
- ground-truth: 60%, 40%

**直接给 ground-truth subgoal 提升 +40% absolute**。

这说明什么？**action prediction module 已经很强了，瓶颈在 subgoal generation**。如果未来 video generation model 更强（Sora 级别），robot performance 会跟着免费提升。

隐含论点：**robotics 的下一个 scaling law 在 video generation，不在 robot data 本身**。

---

## Ablation 三件套

Figure 6 把 contribution 拆开看：

1. **Action chunking**：single-step → 10-step chunk。提升来自减少 compounding error，mitigate Markov violation（真实世界不完全是 Markovian）。

2. **Hybrid attention**：action 用 full attention。提升来自 action dim 互相 condition，减少 incoherent action vectors。

3. **Visual CoT**：加 subgoal generation。提升最大，尤其在 long-horizon。

三个 component 都单调提升 performance。**没有 trade-off，全是 free lunch**。

---

## Limitations paper 自己承认的

**Inference 慢**：要先生成 256 个 image token 才能生成 action token。action chunk size=10 下平均 7× slowdown。image generation 是主要 bottleneck。Consistency model / speculative decoding / fast image gen 可能能解决。

**Image generation 质量不如 diffusion**：autoregressive pixel gen 画质不如 Stable Diffusion / Emu3 这种。但 action 更准。这是质量-精度的 trade-off。SUSIE 反过来，image 漂亮但 action 不准。

**Action chunking 的副作用**：chunk 之间 action 不连续，缺高频 feedback。Diffusion Policy 用 per-step prediction 解决这个，但 compounding error 上升。paper 建议用 temporal smoothing。

---

## 我的几个联想

### 联想一：信息 bottleneck 的角度

vanilla VLA 把 perception-planning-control 全压进 7-DoF action vector。所有 reasoning 都要压缩到 7 维。

CoT-VLA 把 bottleneck 换成 256×256×3 的 pixel subgoal，capacity 暴涨几个量级。这是"think in high-capacity space, act in low-capacity space"。

类比 LLM CoT：直接答数学题 vs 先写出推理步骤再答。中间步骤是在高维 token space 里 think，最终答案在低维 space 里 act。

### 联想二：robotics scaling law 转移

action prediction module 只能从 robot data 学，bounded by robot data scale。subgoal generation 可以从 internet video 学，解锁第二个 scaling axis。

两个 axis 独立 scaling 意味着：可以用 video data 补 robot data 的不足。Table 3 证明 video axis 远没 saturate。

类比 LLM：pretrain 用 internet text，fine-tune 用 task-specific data。robot 这边可能变成 pretrain 用 internet video（学 dynamics），fine-tune 用 robot data（学 control）。

### 联想三：跟 model-based RL 的关系

CoT-VLA 的两阶段结构类似 model-based RL 的 world model + policy：
- subgoal generator ≈ world model（在 pixel space 模拟 dynamics）
- action predictor ≈ policy（给 state 生成 action）

但都用 supervised learning 不用 RL。更 sample-efficient，更容易 scale。

差别：world model 通常在 latent space 模拟，CoT-VLA 在 pixel space 模拟。pixel space 更 interpretable，但可能 less efficient。

### 联想四：跟 LLM CoT 的细微差别

LLM CoT 中间步骤是 unsupervised 的 —— 让 model 自己 discover reasoning structure。

CoT-VLA 中间 subgoal 是 supervised 的 —— 来自 demo 视频。

supervised CoT 更可控但 less general。如果 model 从未见过的 task，它没法 generate 合理 subgoal。Table 3 就显示这个 limitation —— OOD task generated subgoal 质量差导致 performance 跌。

未来方向可能是 unsupervised visual CoT —— model 自己想象 subgoal 不靠 demo。这需要 world model 能力。

### 联想五：test-time compute scaling

现在 subgoal generation 是一次 forward pass。如果改成 sample N 个 subgoal 然后用 value model 选最好的，就是 test-time compute scaling。

类似 OpenAI o1 在 reasoning token 上花更多 compute。robot 这边可以在 subgoal image 上花更多 compute。

### 联想六：hierarchical subgoal

paper 只用单层 subgoal（一个 n 步后的图）。如果做多层级 —— 远期 subgoal (n=50) + 中期 (n=20) + 近期 (n=10)，就是 temporal abstraction。

人做 long task 也是这样想：先想最终目标，再想下个 milestone，再想眼前动作。hierarchical planning。

### 联想七：跟 LLM agent 的同构

CoT-VLA 的 subgoal image 类似 LLM agent 的 plan / scratchpad。

LLM agent: observation → plan text → action
CoT-VLA: observation → subgoal image → action

都是"先想后做"，只是中间 representation 不同。LLM 用 text，robot 用 image（因为 robot state 是 visual 的）。

这暗示一个 universal principle：**intermediate reasoning 应该在 state space 的 native representation 里**。LLM 处理 text 所以 reasoning 用 text。Robot 处理 visual 所以 reasoning 用 image。Audio agent 可能用 spectrogram reasoning。

---

## 一些 web links

- Project page: https://cot-vla.github.io/
- VILA-U base model: https://arxiv.org/abs/2409.04429
- OpenVLA baseline: https://github.com/openvla/openvla
- Octo baseline: https://github.com/octo-models/octo
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- BridgeData V2: https://arxiv.org/abs/2308.12952
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- SUSIE: https://arxiv.org/abs/2310.10639
- Chain-of-Thought original paper: https://arxiv.org/abs/2201.11903
- RQ-VAE: https://arxiv.org/abs/2203.01894
- GR-2 (相关 video-language-action): https://arxiv.org/abs/2410.06158
- 3D-VLA: https://arxiv.org/abs/2403.09631
- Gen2Act: https://arxiv.org/abs/2409.16283
- ECoT (Embodied CoT for robotics): https://arxiv.org/abs/2407.08693
- Emu3 (next-token prediction for multimodal): https://arxiv.org/abs/2409.18869
- Janus (decoupled visual encoding): https://arxiv.org/abs/2410.13848
- Show-O (unified understanding + generation): https://arxiv.org/abs/2408.12528
- Transfusion (predict token + diffuse): https://arxiv.org/abs/2408.11039

---

## 一句话总结

CoT-VLA 把"先想后做"这个 LLM 里证明有效的 principle 搬到 robot。中间 representation 用 pixel-space subgoal image 而不是 text 或 abstract state，因为 robot 的 native state 是 visual 的。这个 decomposition 顺便让 internet video 能进 VLA 训练 pipeline，解锁第二个 scaling axis。实验证明 visual reasoning quality 直接 cap 住 robot performance，意味着 robot 的下一个 scaling law 可能在 video generation 而不在 robot data。

要不要展开哪个方向？比如 RQ-VAE 的 codebook 设计细节、hybrid attention 的工程实现、跟 GR-2/3D-VLA 的具体对比、或者 ground-truth vs generated subgoal 那个实验的更多 implication？

---

# CoT-VLA 深度解析

paper来自 NVIDIA + Stanford + MIT 团队 (Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin)，核心 idea 是把 **visual chain-of-thought reasoning** 引入 VLA (Vision-Language-Action) 模型。具体 project page: https://cot-vla.github.io/

---

## 1. Motivation 与核心 insight

vanilla VLA 的 mapping 形式：

$$\hat{\mathbf{a}}_t \sim P_\theta(\mathbf{a}_t \mid \mathbf{s}_t, l) \tag{1}$$

变量：
- $\mathbf{s}_t$：当前 frame 的 visual observation（image）
- $l$：language instruction
- $\hat{\mathbf{a}}_t$：预测的 single-step action
- $\theta$：model parameters

这个 formulation 把 perception → planning → control 整段 pipeline 全部压缩到一个 end-to-end mapping 里。 Karpathy 你肯定熟悉这种"superposition of abstractions"的问题：当一个 mapping 同时要承担 perception grounding + temporal planning + low-level control，模型只能学会 shortcut，难学到真正的 reasoning。

paper 的诊断很 sharp：在 LIBERO-Spatial 里，OpenVLA / Octo / Diffusion Policy 这些 baseline 会在 visually-similar initial states 上做错任务（明明 instruction 是 task A，但 state 像 task B 就去执行 B），这说明 baseline 没有真正 ground 语言，只是从 visual cue shortcut 到 action。

CoT-VLA 的核心 move：在 action 之前插入一个 **subgoal image generation** 作为 intermediate reasoning step，让 model "先想清楚要去哪"，再"想怎么去"。这跟 NLP 里 chain-of-thought 的本质完全同构 —— 只是把 text reasoning 换成了 pixel-space subgoal。

为什么 subgoal image 是好的中间表示？
- naturally available 在 robot demonstration 视频里，零额外标注
- 同样可以来自 action-less video (EPIC-KITCHENS, Something-Something V2)，解锁 internet-scale video data
- pixel-space 的 high-capacity bottleneck，比 bounding box / keypoint 这种 abstracted state 保留更多 planning 信息
- interpretable：可以直接看 model "在想什么"
- 闭环执行时每次 chunk 完重新 obs + 重新 plan，错误可纠正

---

## 2. 方法形式化

两类数据：
- $D_r = \{(l, \mathbf{a}_{1..T}, \mathbf{s}_{1..T})\}$：robot demonstrations，带 action
- $D_v = \{(l, \mathbf{s}_{1..T})\}$：action-less videos，只有 caption + frames

CoT-VLA 的两阶段 mapping：

$$\hat{\mathbf{s}}_{t+n} \sim P_\theta(\mathbf{s}_{t+n} \mid \mathbf{s}_t, l) \tag{2}$$

$$\{\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}\} \sim P_\theta(\{\mathbf{a}_t, ..., \mathbf{a}_{t+m}\} \mid \mathbf{s}_t, l, \mathbf{s}_{t+n}) \tag{3}$$

变量与上下标：
- $\mathbf{s}_t$：当前 observation
- $\hat{\mathbf{s}}_{t+n}$：generated subgoal image，**n 帧之后**的状态。$n$ 是 subgoal horizon
- $l$：language instruction
- $m$：action chunk size（paper 设 m=10）
- $\hat{\mathbf{a}}_t, ..., \hat{\mathbf{a}}_{t+m}$：m 步 action chunk
- $\theta$：model 参数

**关键设计 choice**：subgoal horizon $n$ 从 $[n_l, n_u]$ uniform sample。这给 model exposure 到不同 lookahead distance，避免过拟合到单一 horizon。 Table 4 给了每个 dataset 不同的 $[n_l, n_u]$：
- Bridge: [5, 10]
- RT-1: [5, 10]
- TOTO: [20, 24]  ← long-horizon trajectory
- VIOLA: [15, 20]
- RoboTurk: [1, 2]  ← 短轨迹
- Jaco Play: [10, 15]
- Berkeley Autolab UR5: [5, 10]
- Berkeley Fanuc: [10, 15]
- Something-Something V2: [5, 7]
- EPIC-KITCHENS-100: [5, 7]

dataset-specific range 反映 trajectory length scale。 short trajectory 设小 horizon 否则 subgoal 直接跳过 task 完成点。

哪个 step 用哪个数据 train？
- Eq (2) 视觉推理：用 $D_r$ + $D_v$ 都行（不需要 action annotation）
- Eq (3) action 生成：只能用 $D_r$（需要 action label）

这是 paper 最 elegant 的部分 —— action-less video 第一次能直接进入 VLA 训练 pipeline，且只是通过 visual reasoning pathway 进入，物理一致。

---

## 3. 架构细节

base model 用 **VILA-U** (Yecheng Wu et al., 2024, https://arxiv.org/abs/2409.04429)，一个 unified multimodal foundation model。

### VILA-U 核心：

- **Unified vision tower**：把 image 编码成 discrete tokens，与 text token space 对齐 —— 这样可以做 autoregressive image generation
- **Residual quantization (RQ-VAE)**：每张 256×256 image 被压缩成 $16 \times 16 \times 4$ tokens
  - $16 \times 16 = 256$ 个 spatial positions
  - 每个 spatial position 4 层 residual token（depth D=4）
  - 总共 256×4 = 1024 个 visual tokens 一张 image
- **Depth transformer $P_\delta$**：autoregressively 预测每个 spatial position 的 D 层 residual token
- **LLM backbone**：处理 projected visual + text features

为什么用 RQ-VAE 而不是 VQ-VAE？VQ 单层 codebook 容量有限，RQ 用多层 residual 让信息 capacity 指数增长，autoregressive image generation 质量更高。这个 choice 直接决定了后面 image gen 的 quality。

### Hybrid Attention 机制（Figure 3）：

这是 paper 的 architecture novelty。在同一个 transformer 里用两种 attention pattern：

| Modality | Attention type | Why |
|---|---|---|
| Text generation | Causal | 标准 next-token prediction |
| Image token generation | Causal | autoregressive pixel generation |
| Action token generation | **Full attention** | parallel decoding，所有 action dim 互相可见 |

为什么 action 用 full attention？
- Action 是结构化的 7-DoF vector，每维 discretize 成 256 bins
- Causal attention 逐 token 生成会慢，且让早期 dim 无法看到后期 dim 的预测
- Full attention 让所有 action token 一次性互相 condition，类似非自回归 parallel decoding

Action 表示细节：
- 每个 action $\mathbf{a}_i$ → 7 tokens（7 个 DoF）
- 每个 DoF 用 256 bins discretize
- bin width 由 training data action distribution 的 1st 和 99th percentile uniform 切分
- repurpose text tokenizer 中最少用的 256 个 token 作为 action bin token（避免 collision）

action chunk size = 10 → 每次 generate 7 × 10 = 70 个 action token，parallel decoding 全部一次出。

paper 还在 [x], [θ], [g] 位置加 special token 做 parallel decoding 控制（Figure 3 里的特殊 marker）。

---

## 4. 训练目标

### Visual loss (Eq 4)：

$$\mathcal{L}_{\text{visual}} = -\sum_j \sum_{d=1}^D \log P_\delta(k_{jd} \mid k_{j,<d}) \tag{4}$$

变量与上下标：
- $j$：包含 visual token 的 position index（遍历 256 个 spatial positions）
- $d$：residual depth index，从 1 到 D（D=4）
- $k_{jd}$：第 $j$ 个 spatial position 的第 $d$ 层 residual token
- $k_{j,<d} = (k_{j,1}, ..., k_{j,d-1})$：同 position 前 d-1 层 token
- $P_\delta$：depth transformer，参数 $\delta$
- 隐含 condition：还 conditioned on LLM 生成的 code embedding $h_j$（公式里省略）

直觉：每个 spatial position，LLM 先生成一个 code embedding $h_j$，然后 depth transformer 用 $h_j$ 当 anchor，autoregressively 生成 D 层 residual token 重建该 position 的 image patch。第 1 层 token 抓 coarse info，第 2~4 层抓 high-frequency details。

### Action loss (Eq 5)：

$$\mathcal{L}_{\text{action}} = -\sum_{i=1}^m \log P_\theta(\mathbf{a}_t ... \mathbf{a}_{t+m} \mid l, s_t, s_{t+n}) \tag{5}$$

标准 cross-entropy over action tokens。变量：
- $i$：action chunk 内的 step index，1 到 m
- $m$：action chunk size = 10
- $\theta$：整个 model（含 LLM backbone + projector + depth transformer）参数

### Total objective (Eq 6)：

$$\mathcal{L} = \mathcal{L}_{\text{action}} + \mathcal{L}_{\text{visual}} \tag{6}$$

简单相加，没 weighting —— 因为 image token 和 action token 都在同一个 token stream 里，next-token prediction 自带自然 weighting。

### 训练流程：

1. **Pretraining**：base VILA-U 7B + OpenX (robot demos) + EPIC-KITCHENS-100 + Something-Something V2 (action-less videos)
   - 12 个 A100 node × 8 GPU = 96 GPUs
   - 11K A100 GPU hours total
   - Learning rate 1e-4, cosine decay, batch 2048
   - 10 epochs
   - Vision tower frozen，训练 LLM backbone + projector + depth transformer
2. **Adaptation fine-tuning**：在下游 robot setup（LIBERO, Bridge-V2, Franka-Tabletop）上继续 fine-tune
   - LR 1e-5 constant，150 epochs
   - 单 A100 node 10-24 小时

---

## 5. Test-time 算法（Algorithm 1）

```
Require: P_θ, s_0^obs, l
t ← 0
while True:
    sample ŝ_{t+n} ~ P_θ(s_{t+n} | l, s_t^obs)         # 视觉 CoT 推理
    sample [â_t, ..., â_{t+m}] ~ P_θ(a_t..a_{t+m} | l, s_t^obs, s_{t+n})   # action chunk
    for j = 0 to m:
        execute â_{t+j}
    t ← t + m + 1
    s_t^obs ← new observation
```

闭环 control：每执行一个 chunk（m=10 步），重新获取 observation，重新生成 subgoal + action chunk。这是 receding horizon control 思路，跟 MPC 类似。

---

## 6. 实验

### LIBERO benchmark (Table 1)

LIBERO 4 个 suite：Spatial（spatial reasoning）, Object（object interaction）, Goal（task objective）, Long（long-horizon）。每个 suite 10 个 task，每个 task 50 demos，3 seeds × 500 episodes。

| Method | Avg ↑ | Spatial ↑ | Object ↑ | Goal ↑ | Long ↑ |
|---|---|---|---|---|---|
| Diffusion Policy | 72.4±0.7 | 78.3±1.1 | **92.5±0.7** | 68.3±1.2 | 50.5±1.3 |
| Octo fine-tuned | 75.1±0.6 | 78.9±1.0 | 85.7±0.9 | 84.6±0.9 | 51.1±1.3 |
| OpenVLA fine-tuned | 76.5±0.6 | 84.7±0.9 | 88.4±0.8 | 79.2±1.0 | 53.7±1.3 |
| **CoT-VLA-7B (ours)** | **81.13±0.6** | **87.5±1.4** | **91.6±0.5** | **87.6±0.6** | **69.0±0.8** |

关键观察：
- Long-horizon suite: CoT-VLA 69.0% vs OpenVLA 53.7% → **+15.3% absolute**，相对 +28.5%
- Goal suite: +8.4% absolute
- Average: +4.6% absolute

**Long-horizon 提升最大** —— 这是 visual CoT hypothesis 的最强 evidence。Long-horizon 任务需要 planning across multiple sub-tasks，subgoal image generation 正好补足这个能力。

Diffusion Policy 在 Object suite 上反而最好（92.5%），可能因为 Object suite 主要是 fine-grained manipulation，不需要太多 reasoning，DP 的 explicit action diffusion 更直接。

### Bridge-V2 (Table 2)

WidowX 6-DoF，45k trajectories。4 个 generalization category，每个 10 trials：

| Category | SUSIE | Octo | OpenVLA | CoT-VLA |
|---|---|---|---|---|
| Visual (cluttered env) | 30% | 35% | 75% | 65% |
| Motion (height variation) | 10% | 10% | 45% | **60%** |
| Semantic (unseen concept) | 20% | 0% | 40% | **50%** |
| Language (instruction grounding) | 40% | 40% | 75% | 70% |

CoT-VLA 在 Visual 和 Language 上略输 OpenVLA 10% 和 5%，paper 归因为 **action chunking 导致的 discontinuous action + grasping failure**（chunk 之间不平滑，高频 feedback 缺失）。Motion 和 Semantic 上明显领先。

SUSIE 是 image-editing diffusion 生成 goal + goal-conditioned policy 的 two-stage 方法，在 Language 上反而最高（除 OpenVLA）但 Visual 上 30%——image gen quality 好但 action execution 不行。这印证了 paper 的 limitation 分析：autoregressive image gen quality 比 diffusion 差但 action 准。

### Franka-Tabletop (Figure 4)

Franka Emika Panda 7-DoF，10-150 demos/task，6 个 task（3 single-instruction + 3 multi-instruction）。CoT-VLA average 比 OpenVLA 高 **17%**。

Diffusion Policy 在 single-instruction 上 top（"put corn in bowl" 这种 narrow task），但 multi-instruction 大幅 degrade。OpenX-pretrained 模型在 multi-instruction language grounding 上明显占优。CoT-VLA 在两种场景都最强。

---

## 7. Ablation Study (Figure 6)

### Component ablation (LIBERO-Spatial & LIBERO-Goal)：

四个 variant：
1. **VLA**：vanilla VLA，VILA-U backbone，single-step action prediction，causal attention
2. **+ action chunking**：single-step → m-step
3. **+ hybrid attention**：action 用 full attention
4. **+ CoT (full)**：完整 visual CoT

每个 component 都单调提升 performance。CoT 提升最大（最后一步）。

直觉：
- Action chunking 提升：减小 compounding error，mitigate Markov violation
- Hybrid attention 提升：action dim 互相 condition，减少 incoherent action vectors
- CoT 提升：planning 显式化，compositional generalization

### Pretraining ablation (Franka-Tabletop)：

- 直接 fine-tune VILA-U → 53.7%
- 加 OpenX + action-less video pretraining → **78.8%**（+25.1% absolute，相对 +46.7%）

预训练 stage 是巨大的 lever。意味着 base model 的 visual reasoning capability 直接决定下游 performance。

### Better visual reasoning helps (Table 3) —— **这是最 striking 的实验**

OOD long-horizon tasks（组合两个未见 sub-task）：
- "move green scallion to apple-covered book"
- "move green cauliflower to bear-covered book"

每个 task 给 1 demo 拿 ground-truth goal image，比较：
| | Sub-task 1 | Sub-task 2 |
|---|---|---|
| Generated goal images | 20% | 0% |
| **Ground-truth goal images** | **60%** | **40%** |

→ ground-truth goal image 直接给 +40% absolute！这意味着：**action prediction module 已经很强了，瓶颈在 subgoal generation**。如果 video generation scaling 能进一步突破（Sora-like world model、diffusion-based image gen），robot performance 会跟着 free 提升。

这是 paper 的一个隐含论点：robotics 的下一个 scaling law 在 video generation，不在 robot data 本身。

---

## 8. Limitations

paper 自陈三大限制：

1. **Inference 速度**：生成 256 个 image token 才能 generate action tokens。action chunk size=10 下平均 **7× slowdown** vs direct action gen。image generation 是主要 bottleneck。Mitigation: consistency model / speculative decoding / fast image gen ([7, 31, 33, 57, 73])。

2. **Image generation 质量**：autoregressive pixel gen 不如 state-of-the-art diffusion ([61, 65, 69, 79])。Emu3, Janus, Show-o, Transfusion 这些 unified multimodal model 可能能改进。

3. **Action chunking 的副作用**：chunk 之间 action 不连续，缺高频 feedback。Diffusion Policy 用 per-step prediction 解决（但 compounding error 上升）。paper 建议 temporal smoothing 或 hybrid per-step + chunked prediction。

---

## 9. 给 Karpathy 的 intuition building

这个工作的 deep insight 有几层：

### 第一层：信息瓶颈视角

vanilla VLA 把整个 perception-planning-control pipeline 压进 $\mathbf{s}_t \to \mathbf{a}_t$ 的 single-step mapping，**bottleneck 是 7-DoF action vector**。所有 planning 必须压缩到 7 维。CoT-VLA 把 bottleneck 换成 $256 \times 256 \times 3$ 的 pixel-space subgoal，capacity 暴涨几个量级。 这是 "let the model think in high-capacity space, act in low-capacity space" 的 instantiation。

### 第二层：Compositional generalization 视角

action prediction module 只能从 $D_r$ 学，generalization bounded by robot data scale。subgoal generation 可以从 $D_v$ 学（internet video），解锁了**两个 scaling axis**：
- Robot data → action quality
- Video data → visual reasoning / dynamics understanding

这两个 axis 独立 scaling，意味着可以用 video data 补 robot data 的不足。 Table 3 的 ground-truth vs generated gap 直接证明这条 axis 还远没 saturate。

### 第三层：跟 LLM CoT 同构

LLM CoT 的本质：把 hard mapping $x \to y$ decompose 成 $x \to z \to y$，其中 $z$ 在同 output space (text)，且每个 sub-mapping 比 end-to-end 更易学。CoT-VLA 完全同构：$s_t \to a_t$ decompose 成 $s_t \to s_{t+n} \to a_t$，中间 $s_{t+n}$ 在 pixel space。

唯一差异：LLM CoT 的中间 $z$ 不监督，让 model 自己 discover reasoning structure。CoT-VLA 的中间 $s_{t+n}$ 是 supervised（来自 demo 视频）。这跟 process reward model / verification 的方向有连接 ([Zelikman et al. STAR](https://arxiv.org/abs/2203.14465)) —— supervised CoT 比 unsupervised CoT 更可控但 less general。

### 第四层：闭环 error correction

Algorithm 1 每 m 步重新 plan，subgoal 可被纠正。这是 receding horizon / MPC 思路。 vs OpenVLA 等单步 VLA 每步重新预测但缺 planning horizon。 vs SUSIE 等 two-stage open-loop 方法 subgoal 一次生成不可纠正。CoT-VLA 是 "closed-loop planning with visual subgoal"。

### 第五层：物理一致性

subgoal generation 从 video 学 dynamics（在 visual space 模拟 world model），action generation 从 robot data 学 control。这种分解类似 model-based RL 的 world model + policy，但都在 pixel space 而不是 latent space，且都用 supervised learning 不用 RL。 这条路可能比 model-based RL 更 sample-efficient 且更容易 scale。

---

## 10. 联想到的开放方向

1. **Diffusion-based subgoal generation**：替换 autoregressive image gen，质量 + 速度双赢。Janus / Emu3 / Show-o 路线。
2. **Hierarchical subgoal**：远期 subgoal (n=50) + 中期 subgoal (n=20) + 近期 subgoal (n=10)，类似 temporal abstraction。
3. **World model 集成**：让 subgoal generator 学 physical dynamics，生成的 subgoal 既要 language-grounded 又要 physically plausible。Sora-like video model ([72](https://arxiv.org/abs/2402.17139))。
4. **Value function / reward model**：subgoal generator 同时输出 value estimate，做 best-of-N subgoal selection 或 RL fine-tune。
5. **Per-step action prediction with temporal smoothing**：解决 chunk discontinuity，借鉴 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 的 receding horizon。
6. **Test-time compute scaling**：subgoal 上花更多 inference time（multi-sample + select），类似 reasoning model。
7. **3D-aware subgoal**：当前 subgoal 是 2D image，[3D-VLA](https://arxiv.org/abs/2403.09631) 路线用 3D 表示，更几何 consistent。

---

## 参考 links

- Project page: https://cot-vla.github.io/
- VILA-U paper: https://arxiv.org/abs/2409.04429
- OpenVLA paper: https://arxiv.org/abs/2406.09246
- OpenVLA repo: https://github.com/openvla/openvla
- Octo repo: https://github.com/octo-models/octo
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- BridgeData V2: https://arxiv.org/abs/2308.12952
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- SUSIE: https://arxiv.org/abs/2310.10639
- Chain-of-Thought prompting: https://arxiv.org/abs/2201.11903
- VQ-VAE / RQ-VAE: https://arxiv.org/abs/2203.01894
- GR-2 (video-language-action): https://arxiv.org/abs/2410.06158
- 3D-VLA: https://arxiv.org/abs/2403.09631
- Gen2Act: https://arxiv.org/abs/2409.16283
- Robotic Control via Embodied CoT (ECoT): https://arxiv.org/abs/2407.08693

---

如果你想让我深挖某个具体方向（比如 hybrid attention 的实现细节、RQ-VAE 的 codebook 设计、CoT 跟 RL 结合的 potential、或者跟 GR-2/3D-VLA/SUSIE 这些 follow-up work 的对比），告诉我，可以继续展开。
