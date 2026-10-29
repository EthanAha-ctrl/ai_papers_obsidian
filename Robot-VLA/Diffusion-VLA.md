---
source_pdf: Diffusion-VLA.pdf
paper_sha256: f036c44e762cd5fcdc0a32dd12f9294e390dafd7eb3233f839717c587ed0129d
processed_at: '2026-08-03T21:54:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiVLA 用人话讲

## 1. 一句话先概括

让机器人先在脑子里"想一下"要干嘛，然后把"想法"直接塞进动作生成器里，这样动作既精准又有脑子。

---

## 2. 为啥要做这个？背景痛点

现在教机器人干活有两种主流套路，各自都有硬伤：

### 套路 A: 把动作当文字来"说" (NTP-based VLA)

代表是 RT-2 和 OpenVLA。思路很直白：既然 GPT 能一个个 token 预测文字，那我把机器人动作也切成一个个小 token，让模型像说话一样把动作"说"出来。

**问题在哪？**

第一个问题：动作是连续的浮点数（比如关节角度 0.3141 rad），你硬要把它切成离散 token，就像让人用乐高积木拼一条平滑曲线——怎么拼都不够丝滑。精细操作（比如拧螺丝）就容易翻车。

第二个问题：慢。GPT 生成 100 个 token 要一个个来，生成 16 步动作就要 16 次 forward pass。OpenVLA-7B 只能跑 5Hz，机器人一秒只能"想"5次，对快节奏任务完全不够。

第三个问题：同一个场景下可能有多种合理动作（比如抓杯子可以从左边抓也可以从右边抓），NTP 取的是 token 概率最大的，容易取一个"平均"的动作，既不是左边也不是右边，直接抓空。

参考：
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/

### 套路 B: 用 Diffusion 生成动作 (Diffusion Policy)

代表是 Chi et al. 2023 的 Diffusion Policy。思路：像 Stable Diffusion 生成图片那样，从一团噪声开始，一步步去噪，最后蹦出一串连贯动作。

**优点**：
- 连续值，精度高
- 一次 forward 就能生成一整段 16 步动作 chunk，快
- 天然能表达 multimodal distribution（同时有"抓左边"和"抓右边"两种 mode）

**问题**：它是个黑盒。你给它图和指令，它吐动作，但你不知道它"想"了啥。换个没见过的物体它就傻眼，因为它没推理能力，全靠 pattern matching。

更关键的是：如果它失败了，你完全不知道为啥失败。是看错物体了？是动作选错了？是规划错了？没法诊断。

参考：
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 所以 DiVLA 想干嘛？

**把两者的长处拼一起**：用 VLM 做"大脑"负责想（reasoning），用 diffusion 做"小脑+肌肉"负责做（action generation）。并且让"想法"直接灌进动作生成器，而不是像以前那样先想完再分开做。

---

## 3. 怎么拼起来的？架构直觉

### 3.1 数据流人话版

想象一个机器人看到场景，脑子里发生这样的事：

1. **眼睛**：SigLIP 把多个相机视角的图编码成 visual token
2. **大脑皮层 (VLM)**：Qwen2-VL 看图和指令，开始"自言自语"生成 reasoning（"我看到一个红杯子在左边，我应该从左侧接近..."）
3. **大脑皮层发信号给小脑**：取 reasoning 最后一层的 hidden embedding（可以理解为"想法的脑波"），通过一个 MLP 投影到 diffusion 能理解的维度
4. **小脑用 FiLM 调制肌肉**：diffusion policy 在去噪的每一层，都用 reasoning embedding 做 affine 变换（放大某些 feature、缩小某些 feature），让动作生成被"想法"引导
5. **肌肉输出**：最后通过一个 MLP 预测 joint-space action chunk

**关键洞察**：reasoning 不用"说出口再听回去"，而是在 VLM 内部生成 reasoning 的同时，把 hidden representation 直接抓出来用。这就是为啥推理速度不下降。

### 3.2 用类比理解 FiLM

FiLM (Feature-wise Linear Modulation) 用人话讲就是"调音台"。

想象 diffusion policy 是一支乐队在演奏（生成动作），reasoning 是指挥。指挥不直接演奏乐器，而是通过手势告诉每个乐手："这里强一点，那里弱一点，节奏快一些"。

数学上就是：
$$\text{FiLM}(\mathbf{x}) = \gamma(\mathbf{r}) \odot \mathbf{x} + \beta(\mathbf{r})$$

变量解释：
- $\mathbf{x} \in \mathbb{R}^d$：diffusion 某一层的 feature（乐手演奏的声音）
- $\mathbf{r} \in \mathbb{R}^{d_r}$：reasoning embedding（指挥的脑波）
- $\gamma(\cdot) \in \mathbb{R}^d$：MLP 输出的 scale（音量旋钮）
- $\beta(\cdot) \in \mathbb{R}^d$：MLP 输出的 shift（音调旋钮）
- $\odot$：element-wise 乘法

每个 $\gamma, \beta$ 都是 d 维向量，所以对每个 feature 维度都能独立调。这比 concat 然后让 network 自己学好理解强得多，因为这种 affine modulation 是显式的、element-wise 的、每一层都发生。

参考：
- FiLM 原文: https://arxiv.org/abs/1709.07871

---

## 4. 为啥这样做 work？三个直觉

### 4.1 Reasoning = 任务分解

考虑 "把 cube 放进带盖子的盒子里" 这个任务。没 reasoning 的话，模型直接 map pixel → 16 步动作，但这个任务有 4 个阶段：approach 盖子 → 开盖 → approach cube → 抓 cube → 放进去。每个阶段的动作分布完全不同，硬要让一个 model 学会所有阶段，很容易混乱。

有 reasoning 的话：VLM 先生成 "我看到一个闭合的盒子，先打开盖子" → FiLM 让 diffusion 知道"现在是开盖阶段" → 动作分布被大幅缩窄 → 容易生成正确动作。

这就是为啥 ablation 里 Task 5 (这个 multi-step 任务) 从 90.9% 掉到 27.3%。

### 4.2 Reasoning = 视觉注意力过滤器

"把所有物体分类放到对应区域" 这个任务，桌子上堆了 10 个物体。没 reasoning 的 model 看到所有 pixel 都参与决策，干扰极大。

有 reasoning 的 model 先生成 "这个是 toy car，那个是 hex key" → FiLM 调制让 diffusion denoiser 关注 toy car 那块区域，其他区域 feature 被压低 → 动作更精准。

这解释了 visual generalization 性能 (57.8% vs OpenVLA 26.7%)。

### 4.3 Diffusion = 多模态动作采样

bin picking 任务中物体大小形状千差万别，同一个物体也可能有多种抓取姿态。NTP-based 容易取"平均"动作抓空，diffusion 能从 multimodal distribution 里采样出合理动作。

这就是 bin picking DiVLA 完爆 OpenVLA 的原因（63.7% vs 28.4%）。

---

## 5. 一些关键设计决策的"人话"解释

### 5.1 为啥用 Qwen2-VL 而不是 LLaVA？

Qwen2-VL 支持 dynamic resolution（不同图片不同 token 数），且开放 2B/7B/72B 三个尺寸，方便做 scaling law 实验。论文里明确说 architecture 是 decoupled 的，未来可以换任何 VLM。

参考：
- Qwen2-VL: https://qwenlm.github.io/blog/qwen2-vl/

### 5.2 为啥要 LoRA 而不是全量 fine-tune？

LoRA 只 tune 一小部分参数（通常 < 1%），好处：
1. **保留 VLM 的 pretrain 知识**：包括 conversational 能力（论文 Table 11 测了 VQA，DiVLA 还能聊天）
2. **显存友好**：72B 模型也能 fine-tune
3. **避免 catastrophic forgetting**：robot 数据少，全量 fine-tune 容易把 VLM 的 visual prior 忘光

参考：
- LoRA: https://arxiv.org/abs/2106.09685

### 5.3 为啥多视角用 concat 而不是 cross-view attention？

简单粗暴但 work。三个视角（2 个外部 Zed + 1 个 wrist Realsense）各自过 SigLIP，token 直接 concat 喂给 VLM。VLM 的 positional encoding 会隐式区分视角。

论文 Table 7 显示 OpenVLA 单视角从 45.3% 掉到 12.7%，证明多视角对 manipulation 至关重要。

### 5.4 为啥换机器人只要重新 init 一层 MLP？

Octo 那种做法是给每个 embodiment 复制一整个 action decoder，参数浪费严重。DiVLA 的 action decoder 共享，只在最后一层换 MLP 来适配不同 action dimension（Franka 8 维 vs AgileX bimanual 14 维）。

这样 pretrain 的动作 prior 被保留，新 embodiment 只需要少量数据 fine-tune 最后一层。

参考：
- Octo: https://octo-models.github.io/

### 5.5 为啥 α = 10？

Loss 是 $L = L_{\text{diff}} + \alpha \cdot L_{\text{ntp}}$。

论文 observation 是 $L_{\text{ntp}}$ 量级比 $L_{\text{diff}}$ 小约 10 倍（diffusion MSE 是 O(1)，NTP cross-entropy 是 O(0.1)）。不调权重的话 NTP 会被忽略，VLM 不学 reasoning。所以 α=10 让两边贡献平衡。

这是工程细节，但 Karpathy 你应该 appreciate 这种 scale balancing 的直觉——和你当年在 nanoGPT 里强调 learning rate schedule 一样，small detail matters。

---

## 6. 数据来源：Reasoning 是怎么搞出来的？

Droid 数据集只有 (image, action, language instruction) 三元组，没有 reasoning text。DiVLA 用 GPT-4o 自动 augment：

给 GPT-4o 看 image + instruction + action trajectory，让它生成 "为什么这么动" 的 reasoning text。比如：

```
Instruction: "Pick up the red cube"
Action trajectory: [approach left, grasp, lift up]
GPT-4o generated reasoning: "I see a red cube on the left side. 
  I need to approach it from the left, close the gripper, 
  then lift it up."
```

然后用这种 (image, instruction, reasoning, action) 四元组训练。VLM 学会 generate reasoning，diffusion 学会从 reasoning embedding 生成 action。

**这是关键**：训练时 reasoning 是 GPT-4o 生成的，但 inference 时是 model 自己 generate reasoning（self-generated reasoning）。

参考：
- Droid: https://droid-dataset.github.io/

---

## 7. 实验结果人话解读

### 7.1 Multi-Task Learning (Table 1)

DiVLA-2B 用 39K trajectory 打了用 970K trajectory 的 OpenVLA-7B。**用 1/25 的数据，性能翻倍**。

人话解释：
- Reasoning injection 让 model "看懂"任务结构，不需要海量数据来覆盖各种 case
- VLM pretrain 知识被充分利用（commonsense reasoning 来自大规模预训练）
- Diffusion 的连续 action 表达高效

### 7.2 Factory Sorting (Figure 3)

最难的 cluttered mixed scenario，DiVLA 维持 60%，Diffusion Policy 掉到 9.2%。

人话：一堆没见过的东西混在一起，没脑子的 policy 直接傻眼。DiVLA 一个个识别物体类别再决定放哪个 sector，所以能维持。

### 7.3 Zero-Shot Bin Picking (Figure 4)

102 个完全没见过的物体，各种 size 和 texture。DiVLA 63.7%，OpenVLA 28.4%。

人话：没见过的物体怎么办？有 reasoning 的 model 知道"这就是个物体，抓起来放过去"，而没 reasoning 的 model 看到没见过的特征就懵了。

### 7.4 Bimanual Table Bussing (Table 2)

OpenVLA 直接 0%。这说明 NTP-based VLA 换 embodiment 完全失效——离散 action token 学的是 Franka 单臂的分布，换 bimanual 14-DoF 直接不能工作。

DiVLA 只要换最后一层 MLP 就能 fast adapt，pretrain 知识保留。

### 7.5 Speed (Table 5)

DiVLA-2B 82Hz，DiVLA-7B 42Hz，OpenVLA-7B 5Hz。

人话：diffusion 一次 forward 生成一整段 action chunk（16步），NTP 要逐 token 生成。加上 vLLM 加速，DiVLA 比 OpenVLA 快 16 倍。

82Hz 意味着 robot 可以做高频反馈控制，对动态任务（比如接球、快响应交互）至关重要。

### 7.6 Scaling Law (Table 10)

从 2B → 7B → 72B：
- Sorting: 66.2% → 74.9% → 82.4%
- Bin Picking: 63.7% → 66.7% → 75.9%

人话：模型越大越聪明。这跟 LLM 的 scaling law 一致，但 robot 领域这是第一次明确展示。

**为什么大模型重要？** Reasoning 质量直接决定 action 质量。小模型 reasoning 不准 → FiLM inject 错信号 → action 错。72B 模型 reasoning 准，所以 action 也准。

### 7.7 Reasoning Injection Ablation (Table 8)

去掉 reasoning injection：avg 50.3% vs 83.6%。掉 33.3 个百分点。

人话：这是论文的"灵魂 ablation"。证明 reasoning 不是花架子，是性能来源。尤其是 multi-step task（Task 5）从 90.9% 掉到 27.3%，证明 reasoning 的任务分解能力是关键。

### 7.8 Novel Instruction (Table 9)

复杂指令"Watermelon → Blue Paper Trash → Lemonade"：
- OpenVLA: 0/3
- DiVLA: 2/3

人话：OpenVLA 只会单步指令，DiVLA 能 follow sequence。这就是 reasoning 的 task decomposition 能力——把 long-horizon task 拆成 sub-task。

### 7.9 View Shifting (Table 6)

完全换相机位置：
- DP: 0%
- OpenVLA: 0%
- DiVLA: 60%

人话：Qwen2-VL 在大规模 image-text pair 上预训练过，visual prior 强，换视角也能 robust。Diffusion Policy 和 OpenVLA 没 pretrain 知识，直接死。

### 7.10 VQA (Table 11)

DiVLA 没专门 co-train vision-language data，但还能聊天、识别颜色、描述场景。

人话：LoRA fine-tune 保留 VLM 的 conversational 能力。这一点比 RT-2/ECoT 那种需要 co-training 的方案更优雅。

---

## 8. 和其他工作的关系

### 8.1 和 π₀ (Physical Intelligence)

π₀ 用 flow matching 而非 DDPM diffusion，3B 参数，同样 multi-embodiment。但 π₀ 没 reasoning injection，是个高性能"黑盒"。

DiVLA 借鉴了 π₀ 的 fine-tune 思路，但加了 reasoning。可以说 DiVLA = π₀ + reasoning injection。

参考：
- π₀: https://arxiv.org/abs/2410.24164

### 8.2 和 TinyVLA

同一作者团队的前作。TinyVLA = 小 VLM + diffusion，没 reasoning。DiVLA = TinyVLA + reasoning injection + scaling。

可以理解为 DiVLA 是 TinyVLA 的"加 thinking 版"。

参考：
- TinyVLA: https://arxiv.org/abs/2409.12514

### 8.3 和 ECoT

ECoT (Embodied Chain-of-Thought) 是先想再做：VLM 生成 reasoning → 喂回 VLM → 生成 action。问题是两次 forward，慢。

DiVLA 的 reasoning injection 是"想的同时就把想法塞进去"，一次 forward 完成。这是效率优势。

参考：
- ECoT: https://ecot-site.github.io/

### 8.4 和 YAY (Yell At Your Robot)

YAY 用 FiLM 把 language correction 注入 policy。但 YAY 注入的是外部指令，DiVLA 注入的是 self-generated reasoning。可以理解为 DiVLA 把 YAY 的"外部 correction"升级成"内部 reasoning"。

参考：
- YAY: https://yay-robot.github.io/

### 8.5 和 Transfusion / Show-O

这些是 unified understanding + generation 工作。Transfusion 一个 model 同时 NTP text + diffusion image。

DiVLA 借了这个思路但用在 robot：NTP for reasoning + diffusion for action。这是 robot 领域的"transfusion"。

参考：
- Transfusion: https://arxiv.org/abs/2408.11039
- Show-O: https://arxiv.org/abs/2408.12528

---

## 9. 局限性：论文没说但确实存在的问题

### 9.1 Reasoning 错了怎么办？

FiLM 注入的 reasoning embedding 是 VLM 自己 generate 的。如果 VLM reasoning 错了（比如 Table 11 把 toy dragon 认成 toy tiger），错误信号会被 FiLM 放大，导致 action 错。

这个 paper 没解决。可能需要 reasoning verification 或 RLHF 来 penalize 错误 reasoning。

### 9.2 第一次 inference 的 latency

82Hz 是稳态频率。但第一次 inference 时 VLM 要 autoregressive 生成 reasoning text（可能 50-100ms），之后 diffusion 部分 12ms 生成 action chunk。

实际 robot control 需要稳定低延迟。如果 reasoning generation 阻塞，第一帧延迟高。论文没细说这个。

### 9.3 FiLM 表达能力有限

FiLM 是 element-wise affine，对 reasoning-action 的 complex interaction 表达能力比 cross-attention 弱。如果 reasoning 要描述 trajectory shape（不只是分类），FiLM 可能不够。

### 9.4 GPT-4o bias

训练数据 reasoning 来自 GPT-4o，所以 reasoning style 受 GPT-4o 限制。如果 GPT-4o 倾向某种 reasoning 模式，DiVLA 学到的也是那种模式。

### 9.5 Multi-Embodiment 不是真 zero-shot

换机器人还是要 fine-tune 最后一层 MLP。真 zero-shot 跨 embodiment 还做不到。这是 robot foundation model 的共同挑战。

### 9.6 没 sim-to-real

所有实验都是 real robot。不知道 sim pretrain 能否 transfer 到 real。这是 robot learning 大问题，论文没涉及。

---

## 10. 最有想象力的方向

### 10.1 Test-Time Compute Scaling

reasoning chain 可以更长。复杂任务让 VLM generate 长 reasoning chain（类似 OpenAI o1 / DeepSeek R1）。这能让 robot 做 test-time compute scaling。

想象 robot 遇到难题，自己"想 30 秒"再动作。

参考：
- DeepSeek R1: https://arxiv.org/abs/2501.12948

### 10.2 Online Reasoning Correction

如果 action 失败（比如 grasp 没抓住），让 VLM 观察到失败后 regenerate reasoning，re-inject 到 diffusion。这是 "online reasoning correction"。

类似 YAY 的思路但自动化。

### 10.3 Reasoning + RL

如果 reasoning 错导致 action 错，用 RL reward 来 penalize 错误 reasoning。让 reasoning 越来越准确。

这是把 LLM 的 RLHF 思路搬到 robot reasoning。

### 10.4 Video Reasoning Pretrain

Droid 数据用 GPT-4o augment reasoning。更好可能用 video-language model (Video-LLaVA) 给 trajectory generate reasoning，能更准确捕捉时序信息。

参考：
- Video-LLaVA: https://arxiv.org/abs/2311.10122

### 10.5 Hierarchical Reasoning

现在 reasoning 是平的（一句话）。可以做 hierarchical reasoning：high-level planning ("先开盖再放 cube") + low-level reasoning ("抓盖子边缘")。每层 inject 到 diffusion 不同层。

### 10.6 Multi-Agent Reasoning

bimanual robot 可以让 VLM 为每只手 generate 独立 reasoning，分别 FiLM inject 到各自的 action decoder。实现"双手各想各的"。

---

## 11. Karpathy 视角的 takeaway

从你"State of GPT"演讲里 System 1 / System 2 的二分法看：

- **Diffusion Policy 是 System 1**：快速、直觉、自动的 action 生成
- **VLM Reasoning 是 System 2**：慢速、deliberative、逻辑的 reasoning
- **FiLM Injection 是 System 2 调制 System 1 的桥梁**：reasoning 不直接生成 action，而是通过 affine modulation 调整 System 1 的 feature space

这和人类大脑类似：前额叶皮层（reasoning）不直接控制肌肉，而是通过运动皮层（action generation）的 gain modulation 来影响动作。

DiVLA 是 robot 领域向 cognitive architecture 迈进的标志性工作。它证明：
1. **Reasoning 和 Action 应该 decouple**：共享 representation 会互相限制
2. **Reasoning 应该 deep inject**：浅层 conditioning 不够，要每层都 inject
3. **Self-generated reasoning > External reasoning**：模型自己产生的 reasoning embedding 和它的 action head 更 compatible
4. **Scaling Law 在 robot 领域成立**：2B → 72B 持续提升

这预示着 robot foundation model 可以像 LLM 那样 scale up。未来 robot GPT 时代可能就在这种 paradigm 下诞生。

---

## 12. 一句话再总结

DiVLA 让机器人先"想"再做，把"想法"用 FiLM 直接灌进 diffusion 动作生成器的每一层，实现了 reasoning 和 action 的解耦但深度耦合。推理快、泛化强、可解释、能 scale，是 robot foundation model 的一个新 paradigm。

---

## 13. 所有参考链接汇总

**核心 paper**：
- DiVLA (假设链接): https://arxiv.org/abs/2506.03963
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- π₀: https://arxiv.org/abs/2410.24164
- TinyVLA: https://arxiv.org/abs/2409.12514
- ECoT: https://ecot-site.github.io/
- Octo: https://octo-models.github.io/

**方法基础**：
- FiLM: https://arxiv.org/abs/1709.07871
- RT-1: https://arxiv.org/abs/2212.06817
- YAY: https://yay-robot.github.io/
- LoRA: https://arxiv.org/abs/2106.09685
- vLLM: https://arxiv.org/abs/2309.06180

**模型 backbone**：
- Qwen2-VL: https://qwenlm.github.io/blog/qwen2-vl/
- SigLIP: https://github.com/google-research/big_vision
- LLaVA: https://llava-vl.github.io/

**数据集**：
- Droid: https://droid-dataset.github.io/
- OXE: https://robotics-transformer-x.github.io/

**unified model 趋势**：
- Transfusion: https://arxiv.org/abs/2408.11039
- Show-O: https://arxiv.org/abs/2408.12528
- Janus: https://arxiv.org/abs/2410.13848
- CogACT: https://arxiv.org/abs/2411.19650

**未来方向相关**：
- DeepSeek R1: https://arxiv.org/abs/2501.12948
- Video-LLaVA: https://arxiv.org/abs/2311.10122
- LLaDA: https://arxiv.org/abs/2502.09992

---

## 14. 给你的 build intuition 提示

Karpathy 你如果要 build 对这个方向的 intuition，建议：

1. **看 FiLM 原文**：理解为什么 element-wise affine 比 concat 强，这是 reasoning injection 的数学基础
2. **跑 Diffusion Policy 官方 repo**：感受 diffusion 生成 action chunk 的速度，对比 NTP 的逐 token 生成
3. **看 Qwen2-VL 的 dynamic resolution 设计**：理解为什么多视角 concat work
4. **看 YAY repo**：理解 language correction injection 的工程实现，DiVLA 是这个思路的升级
5. **看 π₀ paper**：理解 flow matching vs DDPM 的差异，DiVLA 选了 DDPM 是为了和 Diffusion Policy 兼容
6. **重点看 ablation Table 8**：这是整个 paper 的核心证据，证明 reasoning injection 是性能来源
7. **关注 scaling Table 10**：这是 robot foundation model 的 scaling law 证据，未来 robot GPT 的希望

这个 paper 的 elegance 在于：把 LLM 领域的 reasoning + diffusion 领域的 action generation 用一个简单的 FiLM 模块连起来，且计算高效、可解释、可 scale。这是 robot foundation model 设计的一个新 paradigm。

---

# Diffusion-VLA (DiVLA) 深度讲解

## 1. Big Picture: 这篇paper要解决什么问题

robot learning 领域的 VLA (Vision-Language-Action) model 现在有两条主流路线，它们各有硬伤：

**Route A: Autoregressive VLA (RT-2, OpenVLA, ECoT)**
- 把 action 离散化成 token，用 next-token prediction (NTP) 来训
- 优点：自然继承 LLM 的 reasoning 能力，能做 chain-of-thought
- 缺点 1：连续 action 被离散化会损失精度，对精细操作不友好
- 缺点 2：autoregressive 生成 action token 慢，实时控制频率低（OpenVLA-7B 才 5Hz）
- 缺点 3：multimodal action distribution 表达能力弱

**Route B: Diffusion Policy (Chi et al., 2023)**
- 把 action sequence 生成建模成 noise → denoise 过程
- 优点：天然处理 multimodal action distribution，生成速度快
- 缺点 1：完全 black-box，没有显式 reasoning
- 缺点 2：泛化到 novel object / novel instruction 能力差
- 缺点 3：失败后无法诊断

DiVLA 的核心 insight：**把两者 unification 起来，autoregressive 负责 "talk" (reasoning)，diffusion 负责 "act" (action generation)**，并且通过一个 *reasoning injection module* 把 reasoning 的 representation 直接 embed 进 diffusion policy，避免传统 ECoT 那种 recursive 的两阶段 pipeline 带来的额外 inference latency。

参考链接：
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ECoT: https://ecot-site.github.io/

---

## 2. Architecture 详解

### 2.1 整体数据流

```
Image (multi-view) + Text instruction
        ↓
   SigLIP encoder (shared across views)
        ↓
   Concatenate visual tokens
        ↓
   Qwen2-VL backbone (2B / 8B / 72B)
        ↓
   ├── [NTP head] → Reasoning text tokens (autoregressive)
   ↓ (final embedding of reasoning tokens)
   Projection MLP (2层 + LayerNorm)
        ↓
   Diffusion Policy (action decoder, standard DP design)
        ↓
   FiLM injection (用 reasoning embedding 调制 diffusion denoiser)
        ↓
   Joint-space action chunk
```

### 2.2 视觉编码的细节

- **SigLIP** (Zhai et al., 2023) 用于编码图像。和 CLIP 的 contrastive loss 不同，SigLIP 用 sigmoid loss，对 batch size 要求低，更适合大规模 pretrain。
- 输入 N 张图（多视角），共享 backbone，输出固定数量 N 个 visual tokens，然后 **concatenate**。这一点比 token-level attention 更简单粗暴，但实验证明 work。
- 每个相机视角独立 encode，然后 concat，这样 VLM 的 positional encoding 实际上隐式承担了视角区分功能。
- 论文里用的是 **2 个 Zed external camera + 1 个 Realsense 435i wrist camera**，三视角设计类似于 π₀ 和 OpenX-Embodiment 的常见配置。

### 2.3 VLM Backbone 选择

Qwen2-VL (Wang et al., 2024b) 是 SOTA 之一，关键优势：
- 支持 dynamic resolution (Naive Dynamic Resolution)
- 三个 size: 2B / 8B / 72B，方便做 scaling law 实验
- 已经预训练过，可以直接拿 weights 来用

论文明确说 architecture 是 **decoupled** 的，VLM 和 action head 解耦，所以未来可以换任何 VLM (比如 InternVL, LLaVA-OneVision, Gemma-3 等)。这点比 OpenVLA 把 action token 嵌在 VLM 内部要灵活。

### 2.4 Action Token Projection

VLM 最后一层 embedding 出来后，要 bridge 到 diffusion model 的输入维度。这里用一个 **2 层 MLP + LayerNorm**，类比 LLaVA 的 visual projection 设计：

```
h_vlm ∈ R^(d_vlm) → MLP → h_action ∈ R^(d_diff)
```

这里 d_vlm 是 VLM 的 hidden size (Qwen2-VL-2B 是 1536)，d_diff 是 diffusion policy 的 conditioning dimension。

### 2.5 Diffusion Policy Head

这部分是标准的 Diffusion Policy (Chi et al., 2023) 设计：
- 1D temporal U-Net 或 Transformer-based denoiser
- 输入：noisy action chunk $a_t^k$ + observation conditioning
- 输出：noise prediction $\epsilon_\theta(a_t^k, t, \text{conditioning})$
- Action chunk size 一般是 16 步 (horizon)，预测未来 16 步的 joint-space action
- 训练用 DDPM noise schedule，inference 可以用 DDIM 加速

**关键设计 choice**：在 action decoder 最底层挂一个 MLP 来 predict robot 的 joint space。如果换 embodiment，不需要复制一整个 action decoder（像 Octo 那样），只需要 **重新 init 一个新的 MLP layer**，这样 pretrain 的知识能保留。

参考：
- Qwen2-VL: https://qwenlm.github.io/blog/qwen2-vl/
- LLaVA: https://llava-vl.github.io/
- Octo: https://octo-models.github.io/

---

## 3. Reasoning Injection Module: 这篇 paper 的核心创新

### 3.1 传统 ECoT 的痛点

ECoT (Zawalski et al., 2024) 的做法是：
1. VLM 先生成一段 reasoning text ("I see a red cube on the left, I should grab it...")
2. 把这段 text 作为新的 input 喂回 VLM
3. VLM 再基于 reasoning 生成 action

这有两个问题：
- **递归调用** = inference latency 翻倍
- Reasoning 和 action 之间还是有个 gap，因为只是把 reasoning 当 input，没有直接进入 policy 的内部 computation

### 3.2 DiVLA 的做法

DiVLA 不递归调用，而是 **复用 reasoning 阶段产生的 hidden representation**，直接 inject 到 diffusion policy 内部。

具体实现用 **FiLM (Feature-wise Linear Modulation)** (Perez et al., 2018)：

FiLM 的原始公式：

$$\text{FiLM}(\mathbf{x}) = \gamma(\mathbf{r}) \odot \mathbf{x} + \beta(\mathbf{r})$$

变量解释：
- $\mathbf{x} \in \mathbb{R}^d$：被 modulate 的 feature（这里是 diffusion denoiser 某一层的 hidden state）
- $\mathbf{r} \in \mathbb{R}^{d_r}$：conditioning signal（这里是 VLM 的 reasoning token embedding）
- $\gamma, \beta: \mathbb{R}^{d_r} \to \mathbb{R}^d$：两个 MLP，分别输出 scale 和 shift
- $\odot$：element-wise multiplication

直觉：reasoning embedding 不是一个简单的 "concat 后忘掉" 的 conditioning，而是通过 **affine transformation** 主动 reshape diffusion denoiser 的 feature space。每一层都被 reasoning 引导，所以 reasoning 信号是 deep 的、持续的。

这种设计灵感来自：
- **RT-1** (Brohan et al., 2022) 用 FiLM 把 language token 注入到 Transformer policy
- **YAY** (Yell At Your Robot, Shi et al., 2024) 用 FiLM 做 language correction injection

### 3.3 为什么这比 cross-attention 好？

可以设想几种 inject 方式：
1. **Cross-attention**：reasoning token 当 key/value，diffusion feature 当 query。问题：reasoning token 太多时 attention 计算贵，且 reasoning signal 容易被稀释
2. **Concatenation**：reasoning token concat 到 diffusion input。问题：浅层 conditioning，深层可能丢失
3. **FiLM (DiVLA 的选择)**：reasoning 变成全局的 affine transformation，每层都调制。**计算便宜**（只有两个小 MLP），**signal 不被稀释**（直接 multiplicative + additive）

### 3.4 "Injection" 的含义

论文里专门强调：
> "We refer to this process as 'injection' because, in our design, the policy network focuses primarily on action-specific tokens, while the reasoning module functions as an auxiliary enhancement"

意思是：action 的 main signal 还是来自 diffusion denoiser 自己的 U-Net/Transformer，reasoning 只是 **辅助调制**，不主导决策流。这点很重要，因为如果 reasoning 主导，模型容易 hallucinate 不合理的 action。

### 3.5 Reasoning 是怎么生成的？

这部分论文没讲特别细，但可以推断：
1. 训练数据用 **GPT-4o 自动 augment** Droid dataset，给原始 (image, action) pair 加 reasoning text
2. 推理时 VLM autoregressive 生成 reasoning text
3. 取 reasoning text 最后一个或多个 token 的 final embedding 作为 FiLM 的 conditioning

这意味着 reasoning 不需要外部 LLM 生成，模型自己产生 reasoning embedding，然后 inject 给自己。这就是 "self-generated reasoning" 的含义。

参考：
- FiLM: https://arxiv.org/abs/1709.07871
- RT-1: https://arxiv.org/abs/2212.06817
- YAY: https://yay-robot.github.io/
- ECoT: https://arxiv.org/abs/2407.08693

---

## 4. Training Objective 数学推导

### 4.1 总 loss

$$L = L_{\text{diff}} + \alpha \cdot L_{\text{ntp}}$$

变量解释：
- $L_{\text{diff}}$：diffusion policy 的 noise prediction MSE loss
- $L_{\text{ntp}}$：VLM 的 next-token prediction cross-entropy loss（针对 reasoning text token）
- $\alpha$：balance hyperparameter，论文里设 **α = 10**

### 4.2 Diffusion Loss 细节

标准 DDPM loss：

$$L_{\text{diff}} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{c}_{\text{obs}}, \mathbf{c}_{\text{reason}}) \|^2 \right]$$

变量：
- $t \sim \mathcal{U}(0, T)$：diffusion timestep
- $\mathbf{a}_0 \in \mathbb{R}^{H \times D_a}$：ground-truth action chunk（horizon $H$，action dimension $D_a$，比如 7-DoF Franka 是 7 + 1 gripper = 8）
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$：采样的高斯 noise
- $\mathbf{a}_t = \sqrt{\bar{\alpha}_t} \mathbf{a}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$：forward diffusion 加噪
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$：cumulative noise schedule
- $\mathbf{c}_{\text{obs}}$：observation conditioning（来自 VLM processed visual token）
- $\mathbf{c}_{\text{reason}}$：reasoning embedding，通过 FiLM 注入

### 4.3 Next-Token Prediction Loss

标准 cross-entropy：

$$L_{\text{ntp}} = -\sum_{i=1}^{N_{\text{reason}}} \log p_\theta(y_i \mid y_{<i}, \mathbf{x}_{\text{img}}, \mathbf{x}_{\text{lang}})$$

变量：
- $y_i$：reasoning text 的第 i 个 token
- $y_{<i}$：前 i-1 个 token
- $\mathbf{x}_{\text{img}}$：图像 token
- $\mathbf{x}_{\text{lang}}$：language instruction token

### 4.4 为什么 α = 10？

论文说 observation："$L_{\text{ntp}}$ consistently remains about ten times smaller than $L_{\text{diff}}$"。

这是 numerical scale 问题：
- Diffusion MSE loss 在 action 是 normalized 后的连续值，loss 量级在 O(1)
- NTP cross-entropy 经过 vocab softmax，典型 loss 在 0.1-1.0
- 两者量级失衡，小的 loss 会被忽略
- 用 α=10 让它们贡献相当

这是工程细节，但很重要，Karpathy 你应该 appreciate 这种 scale balancing 的工程直觉。

### 4.5 训练 setup

- **Pretrain**: Droid dataset (Khazatsky et al., 2024) 用于 2B/7B；Droid + OXE 用于 72B
- **Fine-tune**: 各个 task 自己的数据，LoRA (Hu et al., 2021) on VLM
- **Freeze**: SigLIP visual encoder + VLM backbone（除了 LoRA adapter）
- **Learning rate**: 2e-5（和 OpenVLA 一样）
- **LoRA**: 比全 fine-tune 参数高效，且保留 VLM 的 conversational 能力

参考：
- Droid: https://droid-dataset.github.io/
- OXE: https://robotics-transformer-x.github.io/
- LoRA: https://arxiv.org/abs/2106.09685

---

## 5. 实验结果深度分析

### 5.1 Multi-Task Learning (Table 1)

5 个 task，3 个 setting (in-distribution, visual generalization)。最有意思的对比：

| Model | Pretrain Data | In-Dist Avg | Visual Gen Avg |
|-------|---------------|-------------|----------------|
| Diffusion Policy | 0 | 27.9% | 8.9% |
| TinyVLA | 0 | 45.5% | 28.9% |
| Octo | 970K | 24.3% | 17.8% |
| OpenVLA-7B | 970K | 39.4% | 26.7% |
| **DiVLA-2B** | **39K** | **83.6%** | **57.8%** |

**惊人结论**：DiVLA 用 39K trajectory（25 倍少于 OpenVLA 的 970K），in-distribution 性能翻倍，visual generalization 也是 2 倍以上。

可能的解释：
1. **Reasoning injection 让模型在 task decomposition 阶段就过滤掉 irrelevant visual feature**
2. **Diffusion head 处理 multimodal action distribution 更好**，避免了 NTP 那种 average-of-modes 问题
3. **Pretrained VLM 的 commonsense knowledge 被充分利用**，因为 reasoning path 用的就是 VLM 的语言能力
4. **LoRA fine-tune** 保留 VLM 的 visual prior，只 tune task-specific adapter

### 5.2 Factory Sorting (Figure 3)

任务：把 4 类物体（toy car, knit glove, stuffed toy, hex key）sort 到 box 的 4 个 sector。

四个 difficulty：
- Seen (训练数据里见过的物体)
- Mixed (seen + unseen)
- Cluttered Seen (5+ 物体重叠)
- Cluttered Mixed (6-11 物体)

DiVLA 在最难的 Cluttered Mixed 仍维持 60% 成功率，而 Diffusion Policy 掉到 9.2%。**这是 reasoning 的力量**：reasoning 让模型先识别每个物体类别再决定 sort 哪个 sector，而不是直接 end-to-end map pixel → action。

### 5.3 Zero-Shot Bin Picking (Figure 4)

102 个 unseen object，各种 size / texture / deformability。

| Method | Success Rate |
|--------|--------------|
| Diffusion Policy | 8.9% |
| Octo | 19.6% |
| TinyVLA | 23.5% |
| OpenVLA | 28.4% |
| **DiVLA-2B** | **63.7%** |
| DiVLA-7B | 66.7% |
| DiVLA-72B | 75.9% |

这里 **DiVLA-2B 就超过 OpenVLA 7B 一倍多**。原因分析：
- Reasoning 让模型 "知道" 这是个 grasping task，"any object on right → left basket"
- Diffusion 的连续 action 表达对各种 size 的物体 robust
- VLM 对 novel object 有 semantic prior，能识别 "这是个物体"

### 5.4 Table Bussing (Bimanual) (Table 2)

AgileX bimanual robot，把餐具放左边 panel，垃圾放右边 bin。

| Scenario | Diffusion Policy | OpenVLA | DiVLA-2B |
|----------|------------------|---------|----------|
| Seen | 45.8% | 0% | **72.9%** |
| Mixed | 31.2% | 0% | **70.8%** |

OpenVLA 在 bimanual 上完全失败（0%），这印证了 NTP-based VLA 在 embodiment transfer 上的劣势——离散 action token 学的是 Franka 单臂的 distribution，换到 bimanual 完全失效。

DiVLA 只需要重新 init 最后一层 MLP（adapt 到新 action dimension），pretrain 知识保留，所以 fast adaptation。

### 5.5 Inference Speed (Table 5)

| Model | Control Frequency |
|-------|---------------------|
| OpenVLA-7B | 5 Hz |
| DiVLA-7B (no vLLM) | 30 Hz |
| DiVLA-7B (vLLM) | **42 Hz** |
| DiVLA-2B (no vLLM) | 74 Hz |
| DiVLA-2B (vLLM) | **82 Hz** |

为什么这么快？
1. **Diffusion 一次 forward 生成整个 action chunk**（16步），而 NTP 要逐 token 生成
2. **vLLM 的 PagedAttention** (Kwon et al., 2023) 优化 VLM 的 KV cache memory
3. **Reasoning 不需要递归调用**，因为 FiLM injection 是 zero-cost（一次 forward 就完成）
4. **2B 模型小，inference 快**

82Hz 对 real-time robot control 来说绰绰有余（人类肌肉控制也就 ~10Hz 主动控制，但视觉反馈需要 ~30Hz）。

参考：
- vLLM: https://arxiv.org/abs/2309.06180

### 5.6 Model Scaling (Table 10)

| Task | DiVLA-2B | DiVLA-7B | DiVLA-72B |
|------|----------|----------|-----------|
| Sorting | 66.2% | 74.9% | 82.4% |
| Bin Picking | 63.7% | 66.7% | 75.9% |

这是 robot foundation model 的 scaling law 证据。从 2B → 72B（36x），sorting +16.2%，bin picking +12.2%。

可以联系到：
- **Chinchilla scaling law** (Hoffmann et al., 2022)：模型和数据需要按比例 scale
- **π₀** (Black et al., 2024)：3B 参数，flow matching
- **RT-X** paper 证明 model size 增加带来泛化提升
- LLaMA 3 也展示了类似 scaling behavior

72B 是关键，因为小模型可能 reasoning 能力不足，reasoning 不准 → FiLM inject 的 signal 不对 → action 也错。这能解释为什么 72B 比 2B 提升明显。

### 5.7 Ablation: Reasoning Injection (Table 8)

| Model | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg |
|-------|--------|--------|--------|--------|--------|-----|
| DiVLA-2B | 100 | 100 | 63.6 | 63.6 | 90.9 | 83.6 |
| w/o reasoning injection | 66.7 | 66.7 | 45.5 | 45.5 | 27.3 | **50.3** |

去掉 reasoning injection 掉了 33.3%。这是论文最关键的 ablation，证明 reasoning 不是可有可无的装饰，而是性能来源。

Task 5 (place cube inside box with closed lid) 从 90.9% → 27.3%，跌幅最大。这是个 multi-step task，需要先开盖再放 cube，没 reasoning 就没法分解。

### 5.8 Multi-View Ablation (Table 7)

OpenVLA 在 sorting 上：
- 3 views: 45.3%
- 1 view: 12.7%

这印证了 multi-view 对 manipulation task 极其重要，单视角深度估计差，遮挡问题严重。

### 5.9 Novel Instruction Following (Table 9)

测试复杂指令如 "Watermelon → Blue Paper Trash → Lemonade"：
- OpenVLA: 0/3
- DiVLA-2B: 2/3

DiVLA 能 follow sequential instruction，OpenVLA 不行。**这是 reasoning 的 decomposition 能力**——把 long-horizon task 分解成 sub-task。

### 5.10 View Shifting Generalization (Table 6)

完全换相机位置后：
- DP: 0%
- OpenVLA: 0%
- DiVLA-2B: 60%

DiVLA 的 robustness 来自 VLM 的 strong visual prior（Qwen2-VL 在大规模 image-text pair 上 pretrain 过）。

### 5.11 VQA Capability (Table 11)

虽然没 co-train vision-language data，DiVLA 仍能做 visual question answering，比如识别 color、spatial relation。

这是 LoRA fine-tune 的副作用——backbone 大部分 frozen，conversational 能力保留。这一点比 RT-2/ECoT 那种需要 co-training 的方案更简洁。

---

## 6. 与其他方法的 Connection

### 6.1 和 π₀ 的对比

π₀ (Black et al., 2024) 是 Physical Intelligence 的工作：
- 用 **flow matching** 而非 DDPM diffusion
- 3B 参数 PaliGemma + flow matching action head
- 同样 multi-embodiment（fine-tune 到 bimanual）
- 没有 reasoning injection

DiVLA 借鉴了 π₀ 的 fine-tune 到新 embodiment 思路，但加了 reasoning，且用 DDPM 而非 flow matching。

### 6.2 和 TinyVLA 的对比

TinyVLA (Wen et al., 2024) 也是同一作者团队：
- 小 VLM (1B-3B) + diffusion policy
- 但没有 reasoning injection
- 性能比 DiVLA 差

可以说 DiVLA 是 TinyVLA 的进化版，加上了 reasoning 这个关键 component。

### 6.3 和 CogACT 的对比

CogACT (Li et al., 2024a) 也很类似：
- VLM backbone + action token
- 用 adaptor 而非 diffusion
- DiVLA 用 diffusion + FiLM，结构更明确

### 6.4 和 Transfusion / Show-O / Janus 的对比

这些是 unified understanding + generation 的工作：
- Transfusion (Zhou et al., 2024)：一个 model 同时 NTP text + diffusion image
- Show-O (Xie et al., 2024)：single Transformer 统一两种 mode
- Janus (Wu et al., 2024a)：decoupled visual encoding

DiVLA 借用了这个 idea，但用在 robot：NTP for reasoning + diffusion for action。

### 6.5 和 YAY 的对比

YAY (Yell At Your Robot, Shi et al., 2024)：
- 用 language correction 改进 policy
- FiLM 把 correction 注入 policy
- 但是是 post-hoc correction，不是 self-generated reasoning

DiVLA 把 FiLM 注入 mechanism 升级：注入的不是外部指令，而是模型自己 generated 的 reasoning。

参考：
- π₀: https://arxiv.org/abs/2410.24164
- TinyVLA: https://arxiv.org/abs/2409.12514
- CogACT: https://arxiv.org/abs/2411.19650
- Transfusion: https://arxiv.org/abs/2408.11039
- Show-O: https://arxiv.org/abs/2408.12528
- Janus: https://arxiv.org/abs/2410.13848

---

## 7. Architecture Diagram 文字版

```
┌──────────────────────────────────────────────────────────────┐
│                  Input: multi-view images + text              │
└────────────────────────────┬─────────────────────────────────┘
                             ↓
                  ┌─────────────────────┐
                  │  SigLIP (frozen)    │
                  │  per-view encode    │
                  └──────────┬──────────┘
                             ↓ concat
                  ┌─────────────────────┐
                  │  Qwen2-VL backbone  │  ← LoRA adapters
                  │  (2B/7B/72B)        │
                  └──────────┬──────────┘
                             ↓
              ┌──────────────┴───────────────┐
              ↓                              ↓
    ┌──────────────────┐         ┌────────────────────┐
    │ NTP head         │         │ Reasoning embedding│
    │ (reasoning text) │         │ (last layer hidden) │
    └──────────────────┘         └──────────┬─────────┘
                                            ↓
                                  ┌──────────────────┐
                                  │ Projection MLP   │
                                  │ (2 layers + LN)   │
                                  └──────────┬───────┘
                                             ↓
                                    ┌────────────────┐
                                    │ FiLM γ, β MLPs │
                                    └────────┬───────┘
                                             ↓
                  ┌──────────────────────────────────────┐
                  │  Diffusion Policy Denoiser           │
                  │  (U-Net / Transformer-based)         │
                  │                                      │
                  │  For each layer l:                  │
                  │    h_l = γ_l(r) ⊙ h_l + β_l(r)      │  ← FiLM modulation
                  │                                      │
                  │  Input: noisy action chunk a_t       │
                  │  Output: ε prediction                │
                  └──────────────────┬───────────────────┘
                                     ↓
                         ┌────────────────────┐
                         │ Final action MLP   │
                         │ (per-embodiment)   │
                         └────────────────────┘
                                     ↓
                          Joint-space action chunk
```

---

## 8. Intuition Building: 为什么这个设计 work

### 8.1 Reasoning as Task Decomposition

考虑 "place cube inside box with closed lid" 这个 task：
- 没有 reasoning 的 model：直接 map pixel → 14-DoF action chunk。这是非常 ill-posed 的 mapping，因为不同子阶段（approach lid → grasp lid → open → approach cube → grasp cube → place）的 action distribution 完全不同。
- 有 reasoning 的 model：先生成 "I see a closed box. I should open the lid first." → FiLM 注入这个信号 → diffusion policy 知道现在在 "open lid" 子阶段，action distribution 被大幅 narrowed down。

这就是为什么 multi-step task 提升最明显（90.9% vs 27.3%）。

### 8.2 Reasoning as Visual Attention Filter

"Sort all items into corresponding areas" 这个 task：
- 没有 reasoning：model 看到 cluttered scene，所有 pixel 都参与 action decision，干扰大
- 有 reasoning：model 先生成 "this is a toy car, that is a hex key" → FiLM 让 diffusion denoiser 关注 "toy car" 那块区域，hex key 区域的 visual feature 被抑制

这解释了 visual generalization 性能（57.8% vs OpenVLA 26.7%）。

### 8.3 Diffusion as Multimodal Action Sampler

考虑 bin picking 各种 size 的物体：
- NTP-based：action 被离散化成 bin，物体 size 变化大时离散 bin 可能不够精细
- Diffusion：连续 action，multimodal distribution（同样物体可以多种 grasp pose）都能采样出来

这就是 bin picking 上 DiVLA 完爆 OpenVLA 的原因。

### 8.4 Decoupling Reasoning from Action

传统 end-to-end VLA 把 reasoning 和 action 绑在一起：
- Image → VLM → action token (NTP)
- Reasoning 和 action 共享同一个 representation space
- Action 的精度受 vocab size 限制

DiVLA 解耦：
- Image → VLM → reasoning text + reasoning embedding
- Reasoning embedding → FiLM → diffusion
- Action 由 diffusion 独立生成，不受 vocab 约束

这种 decoupling 让两个 module 各司其职。

---

## 9. 局限性和潜在问题

论文没明说但可以推断的 limitation：

### 9.1 Reasoning Quality 上限

FiLM 注入的 reasoning embedding 是 VLM autoregressive 生成的。如果 VLM reasoning 错了（比如 misidentify 物体），错误的 reasoning embedding 会通过 FiLM 误导 diffusion policy。论文 Table 11 VQA 实验里就观察到 DiVLA 把 toy dragon 识别成 toy tiger，这种错误在 reasoning-driven action generation 里会被放大。

### 9.2 Inference 时 Reasoning 还是要 Generate

虽然 FiLM injection 本身 zero-cost，但 reasoning text 还是要 autoregressive 生成。82Hz 这个数字可能是：先 VLM 生成 reasoning (~100ms)，然后 diffusion 部分 12ms 内完成 action chunk。如果 reasoning token 数量多，第一帧 latency 会高。

实际 robot control 需要 consistent low latency，所以 reasoning generation 时间是个隐藏 cost。

### 9.3 FiLM 表达能力有限

FiLM 是 element-wise affine，表达能力比 cross-attention 弱。如果 reasoning 需要和 action 有 complex interaction（比如 reasoning 描述一个 trajectory shape），FiLM 可能不够。

### 9.4 Pretrain Data 依赖

DiVLA 依赖 GPT-4o 给 Droid 数据 augment reasoning。这意味着 reasoning 的 style 受 GPT-4o 限制，且可能引入 GPT-4o 的 bias。

### 9.5 Multi-Embodiment 还是要单独 Fine-tune

虽然 re-init 一层 MLP 比 re-init 整个 action decoder 轻量，但仍然需要 task-specific data fine-tune。真正 zero-shot 跨 embodiment 还做不到。

### 9.6 没测 Sim-to-Real

所有实验都是 real robot。没看到 sim-to-real transfer 实验，不知道 sim pretrain 能否 transfer 到 real。

---

## 10. Future Directions 推演

基于 DiVLA 的 framework，可以想象几个 future direction：

### 10.1 在线 Reasoning Refinement

如果 robot action 失败（比如 grasp 没抓住），可以让 VLM 观察到失败后 regenerate reasoning，然后 re-inject 到 diffusion。这是一种 "online reasoning correction"。

### 10.2 Reasoning from Video Pretrain

Droid 数据 augment reasoning 用 GPT-4o，但更好可能用 video-language pretrain 的 model (比如 Video-LLaVA) 给 trajectory generate reasoning，能更准确捕捉时序信息。

### 10.3 Action Token via Diffusion 的反向

现在 DiVLA 是 reasoning (NTP) + action (diffusion)。也可以反过来：text generation 用 diffusion (像 LLaDA / Plaid 那样)，action 用 NTP。但 robot 应用里 text generation 不需要太快，所以当前设计更合理。

### 10.4 Test-Time Compute Scaling

reasoning chain 可以更长。如果任务复杂，可以 generate 长 reasoning chain (类似 OpenAI o1 / DeepSeek R1)。这能让 robot 做 test-time compute scaling。

### 10.5 Reasoning + RLHF

如果 reasoning 错了导致 action 错，可以用 RLHF reward model 来 penalize 错误的 reasoning。这能让 reasoning 越来越准确。

参考：
- LLaDA: https://arxiv.org/abs/2502.09992
- Video-LLaVA: https://arxiv.org/abs/2311.10122
- DeepSeek R1: https://arxiv.org/abs/2501.12948

---

## 11. 对比表总结

| 维度 | OpenVLA | π₀ | TinyVLA | DiVLA |
|------|---------|-----|---------|-------|
| Action head | NTP | Flow matching | Diffusion | Diffusion |
| Reasoning | ECoT variant | 无 | 无 | FiLM injection |
| VLM | Prismatic-7B | PaliGemma-3B | Qwen2-VL-2B | Qwen2-VL-2B/7B/72B |
| Multi-view | 否 (1 view) | 是 | 是 | 是 |
| Bimanual | 困难 | 是 | 未知 | 是 (re-init MLP) |
| Inference | 5Hz (7B) | 未知 | 较快 | 82Hz (2B) / 42Hz (7B) |
| Pretrain data | OXE 970K | 10K hours | Droid | Droid (39K) |
| Scaling | 单一 size | 单一 size | 单一 size | 2B → 72B |
| Visual gen | 中 | 中 | 中 | 强 |
| Novel instruction | 弱 | 中 | 弱 | 强 |

---

## 12. 论文的 Significance

DiVLA 的 significance 不仅是 SOTA，更在于它给 VLA 设计一个新 paradigm：

**Reasoning 和 Action 应该 decouple，但 reasoning 应该 deep inject 到 action 而非仅仅作为 input context**

这一点对后续 VLA 设计影响大。可以想象：
- 接下来会有 cross-attention injection 版本
- 会有 reasoning + RL 的 VLA
- 会有 reasoning chain 更长 + 验证机制的工作
- 会有 VLM 在 robot 数据上 continue pretrain + reasoning inject 的工作

从 Karpathy 你的视角，这个 work 印证了你之前在 "State of GPT" 演讲里讲到的：**system 1 (fast, intuitive) + system 2 (slow, deliberative)** 的二分法。Diffusion 是 system 1，reasoning 是 system 2，FiLM 是 system 2 调制 system 1 的桥梁。

这是 robot 领域向 cognitive architecture 迈进的一步。

---

## 13. 复现和资源

参考链接汇总：
- **DiVLA 论文**: https://arxiv.org/abs/2506.03963 (假设)
- **Diffusion Policy**: https://github.com/real-stanford/diffusion_policy
- **OpenVLA**: https://github.com/openvla/openvla
- **Qwen2-VL**: https://github.com/QwenLM/Qwen2-VL
- **SigLIP**: https://github.com/google-research/big_vision
- **Droid Dataset**: https://github.com/droid-dataset/droid
- **OXE**: https://github.com/google-deepmind/open_x_embodiment
- **vLLM**: https://github.com/vllm-project/vllm
- **LoRA**: https://github.com/microsoft/LoRA
- **FiLM 原文**: https://arxiv.org/abs/1709.07871

---

## 14. 一句话总结

DiVLA 通过 FiLM 把 autoregressive VLM 的 reasoning embedding deep inject 到 diffusion policy，实现了 reasoning + action 的统一，且不增加 inference 成本。这是 robot foundation model 从 "端到端黑盒" 走向 "可解释 + 可泛化 + 可控" 的关键一步，并且 scaling law 从 2B → 72B 都成立，预示着 robot GPT 时代的可能性。
