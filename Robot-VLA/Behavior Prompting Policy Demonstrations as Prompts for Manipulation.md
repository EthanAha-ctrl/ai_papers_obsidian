---
source_pdf: Behavior Prompting Policy Demonstrations as Prompts for Manipulation.pdf
paper_sha256: e457d48683ce022eee6ad8814367e580adeae9054752a89df6afd02a1f4a592a
processed_at: '2026-08-18T02:24:26-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 到底在搞啥

Andrej，我用最朴素的话再过一遍，同时保留技术细节和公式。

## 一句话总结

**给机器人看一段人类示范视频（带动作），它就能当场学会干新活，不用 fine-tune。** 这段示范就叫 "behavior prompt"——跟 LLM 里给 few-shot example 一个道理，只不过这里的 example 是一整条 sensorimotor trajectory，而不是文本。

Paper: https://behavior-prompting.github.io

## 为什么这个 idea reasonable

robot learning 一直有个尴尬：你要让它干新活，传统套路是 retrain 或 fine-tune，成本高。VLA 模型（RT-2、π0、π0.5）虽然能 zero-shot 处理新 object / 新环境，但本质上还是**"语言说啥我做啥"**，对低层 action 的细微变化（比如"抓杯子用哪个角度"、"画画下一笔往哪走"）完全无能为力。

Language 这种 task descriptor 有个根本缺陷：**信息密度太低**。你说"fold the left sleeve"，机器人知道了"做什么"，但完全不知道"怎么抓、抓哪儿、什么时候换手"。

Goal image 也一样废：只给最终状态，过程全靠脑补。

Behavior prompt 把这个 information gap 直接补上：

$$\mathcal{P} = \{(o_i, q_i, a_{i:i+\Delta t})\}_{i=0}^{n}$$

变量含义：
- $o_i$：第 $i$ 段的 observation（图像）
- $q_i$：proprioception（gripper 位姿、宽度）
- $a_{i:i+\Delta t}$：接下来 $\Delta t$ 步的 action chunk
- $n$：prompt 总段数

一段 demonstration 同时编码了 **what + how + when**，这就是它的 expressive power。

## BPP 架构怎么用这个 prompt

核心 split 成两个 module，这点是和前作 ICRT 最大的区别。

### Module 1：Prompt Encoder

干一件事：**给定当前画面，去 prompt 里"查表"，看现在最像 prompt 的哪一段**。

流程：
1. 把 prompt 切成 chunk，每个 chunk 通过 attention pooling 压成单 embedding $p_i$
2. 当前 observation tokenize 后，对 $P = [p_0, p_1, ..., p_n]$ 做 cross-attention

公式上就是标准 attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

其中 $Q$ 来自 current obs，$K, V$ 来自 prompt chunks，$d_k$ 是 key 维度。

物理意义：**policy 学会了 "prompt lookup" 这个 meta-procedure**。Figure 5 的可视化特别直观——画画任务里 attention 沿着 prompt 连续滑动；manipulation 任务里 attention 跳到下一个 milestone。这跟 LLM in-context learning 学到的 retrieval 能力是一个味道。

### Module 2：Action Decoder

输入：current obs + prompt encoder 提取的相关信息 + diffusion step $k$

输出：用 diffusion 跑 $K$ 步 denoise 出 action。

Forward process 加噪：

$$q(a_t | a_0) = \mathcal{N}\left(a_t; \sqrt{\bar{\alpha}_t}\, a_0, (1 - \bar{\alpha}_t) I\right)$$

变量：
- $a_0$：clean action
- $a_t$：第 $t$ 步加噪版本
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：累积 noise schedule
- $\alpha_s$：每步保留比例（超参）

Reverse process 学个 $\epsilon_\theta$ 预测 noise。用 diffusion 不用 L1，是因为 manipulation action 经常 multimodal（一个物体多种抓法），diffusion policy 在这上面表现好。参考 Chi et al. 的 Diffusion Policy: https://arxiv.org/abs/2303.04137

## 为什么是双 module 不是单 transformer

这是相对 ICRT 最重要的设计 choice。ICRT 把 prompt + rollout history 全塞进一个 causal transformer，问题是：

1. **Inference 慢**：每次前向都 reference 整个 context（虽然靠 KV cache 缓解）
2. **OOD 隐患**：rollout history 越长，spurious correlation 越多，长 context 容易爆掉（参考 Torne et al. Long-context Diffusion Policies）

BPP 把 prompt understanding 和 action generation **解耦**：
- Prompt embedding 一次性算好，整个 rollout 复用
- Prompt encoder 每 step 调一次
- Action diffusion 跑 $K$ 步 denoise，每步只看 obs + 提取的 prompt info，不看原 prompt

固定长度 history 这个设计也避免了 ICRT 那种"跑到一半 context 满了"的问题。

## iPhUMI：让 behavior prompting 真的能用

光有 algorithm 还不够，你得有办法**在现场快速采 prompt 喂给 robot**。原版 UMI 用 GoPro，需要 SfM mapping，setup 几小时。iPhUMI 用 iPhone 15 Pro + ARKit SLAM，**零 mapping time**。

数据模态：
- Main camera：1920×1440 @ 60Hz
- Ultrawide：640×480 @ 10Hz
- LiDAR depth：256×192 @ 60Hz（这篇没用）
- Gripper pose：60Hz（ARKit 输出）
- Gripper width：10Hz（ultrawide 检测 ArUco tag）

测试时流程：iPhone 采一段 demonstration → 无线传 desktop → 处理成 prompt → condition BPP。这实现了"现场演示、立即执行"的闭环。

参考 UMI: https://arxiv.org/abs/2402.10329

## Benchmark：专门设计来测 test-time adaptation

现有 benchmark 测的都是"新 object、新环境"，没有专门测"新 action sequence"的。paper 提了两个：

### DrawAnything

画画任务，完美 fit behavior prompting，因为：
- 每一步都是低层 action 的新组合（continuous action adaptation）
- Goal image 完全没用，必须看过程
- 可以 procedural 生成到 2000 个 task

**DrawAnything-Sim**：2000 个 procedural drawing（1-6 个 part 组合），每个 5 demos，50 个 unseen eval。
**DrawAnything-Real**：ARX arm + iPhone wrist camera + marker + spring（compliance），1000 个 training task（200 human + 800 scripted），10 个 eval。Metric 是 Chamfer distance。

### LIBERO-Gen

扩展 LIBERO benchmark。
- **LIBERO-Gen Combination**：174 个 task（原 10 + 164 新），两个相同 bowl 之间选一个 pick + 放到 9 个 location 之一。held out 10 个 (pick, place) 组合——单独都见过，joint 没见过。
- **LIBERO-Gen Chain**：321 个 task（原 10 + 311 新），两步 chain。first step: open drawer / push plate / turn on stove / pick-place；second step 都是 pick-place。held out 10 个两步组合——每步 primitive 单独见过，组合没见过。

这两个 benchmark 都很巧妙：测的是**"见过的 primitive 重组"**的 generalization，而不是"全新 primitive"。这正好是 in-context learning 的 sweet spot。

## 实验数据

主结果（Figure 4）：

| Benchmark | Metric | Goal-Image | ICRT | BPP | π0.5 |
|---|---|---|---|---|---|
| DrawAnything-Sim (unseen) | Chamfer ↓ | 基准 | -33.3% vs BPP 之前 | **减 80.7% vs Goal-Image** | - |
| LIBERO-Gen Combination | Success % | 较低 | - | **最高** | 接近 BPP |
| LIBERO-Gen Chain | Success % | 较低 | - | **最高** | 接近 BPP |

最 impressive 的对比是：**BPP 没有 foundation pretraining，居然 rival π0.5**（π0.5 有大规模预训练 + 100K LoRA fine-tune steps）。这说明 in-context demonstration 是个非常强的 condition signal。

## Ablation 几个关键发现

**Prompt 里该放啥**（Figure 6a-c）：
- Observation 必须——anchor prompt lookup
- Action 必须——提供 temporal transition
- Proprioception 没用——已经 visual 在 obs 里了
- Attention pooling 比 separate tokens 好（temporal binding + 序列变短）

**数据怎么配**（Figure 6d-f，最 actionable 的部分）：
- **Task diversity > per-task quantity**：固定 demo 预算下，"many tasks × few demos" 完胜 "few tasks × many demos"
- 5 demos/task 情况下，task 数量持续提升 unseen 性能
- 训练 task 要够复杂（4-6 parts 比 1-3 parts 好）

这个 finding 跟 LLM in-context learning scaling 规律完全呼应：**task 数量是 in-context 能力涌现的关键 driver**。

## 一个 negative result 很说明问题

Laundry folding（§4.2, Appendix A）：三个 task，每个 ~150 demos。结果 BPP 反而比 Language baseline 差：

| Task | Language | BPP |
|---|---|---|
| Fold left arm | 96% | 76% |
| Fold right arm | 100% | 100% |
| Fold bottom up | 100% | 60% |

**为什么**：task 太少时，prompt 里丰富的 spatial/temporal 信息反而成 noise（duration 变化、configuration 变化），policy 容易 overfit 到 spurious cue（背景差异）。Language 这种简洁 identifier 反而更鲁棒。

**Implication**：behavior prompting 是 high task diversity regime 的工具。低多样性场景下，老老实实用 language。

## 我的 intuition 总结

1. **Task diversity 是 in-context learning 的 fuel**。LLM 里被验证过，robotics 里再次验证。要 emergent in-context 能力，task 覆盖广度比单 task 数据量重要得多。

2. **Demonstration 是最 expressive 的 prompt format**，但代价是 representation 复杂、需要 cross-attention 推理。它的 sweet spot 是"语言难以描述的低层行为"——画画、folding preference、特定 grasp 策略。

3. **双 module 架构（prompt encoder + action decoder）是好工程选择**：inference 高效、history 固定避免 OOD、复用 action diffusion 的 multimodal 能力。

4. **In-context learning 在 robotics 的门槛比 LLM 高得多**。LLM internet-scale 文本预训练就 emerge 了；robot 这里 2000 个 task 才看到明显效果，且还没到完全 generalizable 的程度。

5. **Hardware interface 是 paradigm enabler**。iPhUMI 让"现场演示→立即执行"真的能用，这种交互闭环是 behavior prompting 能落地的关键。

6. **Next step 显然是和 VLA foundation model 结合**。BPP 无预训练就 rival π0.5，加上预训练大概率更强。sim-to-real prompting（sim 里 prompt、real 里 execute）也值得做。

7. **Laundry folding 的 negative result 很重要**。它告诉我们 in-context learning 不是万能药，需要足够 task diversity 才能 emerge。少 task 场景下，language 这种 compact descriptor 反而更合适。

参考链接汇总：
- 项目主页: https://behavior-prompting.github.io
- ICRT: https://arxiv.org/abs/2408.15980
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- UMI: https://arxiv.org/abs/2402.10329
- LIBERO: https://arxiv.org/abs/2304.06752
- π0.5: https://arxiv.org/abs/2410.24164
- RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html
- Octo: https://octo-models.github.io/
- VIMA: https://vimalabs.github.io/
- Vid2Robot: https://rss24.github.io/
- One-shot imitation (Duan 2017): https://arxiv.org/abs/1703.07318
- FiLM conditioning: https://arxiv.org/abs/1709.07871
- Attention sinks (StreamingLLM): https://arxiv.org/abs/2309.17453

希望这版"人话"讲清楚了。核心 intuition 就一句话：**把 LLM 的 in-context learning 范式搬到 robotics，用完整 demonstration 当 prompt，task diversity 是涌现的关键，prompt lookup 是学到的 meta-procedure**。

---

# Behavior Prompting Policy 深度讲解

Andrej，这篇 paper 我觉得非常有意思，它在 robotics 领域做了一个相当有信念的 bet：把 LLM 里的 in-context learning 范式直接搬到 manipulation 里，但是用一整条 sensorimotor demonstration 当作 prompt，而不是 language 或 goal image。下面我从 motivation、架构、data、benchmark、实验、limitation 六个维度展开，尽量把直觉和细节都讲清楚。

## 1. 核心动机：为什么 behavior prompting 是一个 reasonable bet

robot 学习新 skill 传统上要么 fine-tune，要么靠 language/goal image condition。问题是 language embedding（像 CLIP/RT-2 那种）只能给你"做什么"，goal image 只能给你"最终状态"，两者都缺**spatial 和 temporal 的 how 信息**。

paper 的核心 insight 在 §3.2 写得很清楚：

> A behavior prompt additionally provides **spatial and temporal information** that inform the policy how to complete the task.

也就是说，一条 demonstration $\mathcal{D} = \{(o_t, q_t, a_t)\}_{t=1}^{T}$ 同时编码了：
- **what**：要完成什么任务（grasping、folding 哪个袖子）
- **how**：用什么 manipulation strategy（从哪个角度 grasp、push 还是 pull）
- **when**：时序上的 sub-goal 切换点

这其实就是 LLM in-context learning 的延伸：在 LLM 里 few-shot examples 提供输入输出的对应关系，policy 学会了 "induction"，在 robot 这里 demonstration 提供了 observation-action 的对应关系，policy 学会了模仿。

直观上，behavior prompt 的 expressive power 大致是 goal image < language < demonstration 的层级关系，paper 在 §4.1 总结里也提到："the benefits of more temporally-rich task descriptors (goal image → language → behavior prompt) are more pronounced as the temporal task complexity increases"。这个观察我觉得很关键，因为它告诉我们 behavior prompting 不是在任何 task 上都赢，而是在**temporal complexity 高的 task**（drawing、chained manipulation）上赢。

参考链接：
- 原始 paper: https://behavior-prompting.github.io
- ICRT (前序工作): https://arxiv.org/abs/2408.15980
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- UMI: https://arxiv.org/abs/2402.10329

## 2. Behavior Prompt 的形式化定义

§3.1 给的定义：

> A behavior prompt is a single demonstration of a desired task (at a different environment configuration) consisting of a sequence of observations, proprioception, and actions in the same sensorimotor space as the robot's execution.

写成符号：

$$\mathcal{P} = \{(o_i, q_i, a_{i:i+\Delta t})\}_{i=0}^{n}$$

其中：
- $o_i \in \mathbb{R}^{H \times W \times C}$：第 $i$ 个 chunk 对应的 visual observation（RGB image）
- $q_i \in \mathbb{R}^{d_q}$：proprioception（gripper pose、width 等）
- $a_{i:i+\Delta t} \in \mathbb{R}^{\Delta t \times d_a}$：从第 $i$ 步到 $i + \Delta t$ 步的 action chunk（保留原始频率，不做 temporal downsample）
- $n$：随 prompt 长度变化的 chunk 数
- $\Delta t$：chunk 步长（observation 下采样到 ~1Hz，action 不下采样）

这个设计有几个 subtle 但重要的点：

1. **observation 降频到 1Hz 是为了计算效率**，但 action 不降频以保留完整的 fine-grained behavior 序列。这是一个 asymmetry 的设计，因为 action 的高频信息对模仿至关重要，而 observation 在 1Hz 下已经能 anchor 时序位置。
2. **prompt 在不同的 environment configuration 下采集**，这就强制 policy 必须学会 spatial alignment（prompt 里的物体位置和 rollout 时不一样）和 temporal correspondence（什么时候 prompt 进展到哪一步）。

## 3. BPP 架构详解

BPP 分两个 module：Prompt Encoder 和 Action Decoder。这是相对 ICRT 最大的架构差异——ICRT 是一个 autoregressive causal transformer，把 prompt 和 rollout history 全塞进 context；BPP 是 decoupled 的双 module 设计。

### 3.1 Prompt Encoder（Figure 2b）

**Step 1 - Attention Pooling**：每个 chunk 内部的 $\{o_i, q_i, a_{i:i+\Delta t}\}$ 通过 attention pooling merge 成单一 embedding：

$$p_i = \text{AttnPool}(o_i, q_i, a_{i:i+\Delta t})$$

得到 prompt 序列 $P = [p_0, p_1, ..., p_n]$。

ablation（Figure 6c）显示 attention pooling 相比 separate tokens per modality 有显著提升，原因有二：
- **temporal association**：同一 timestep 的不同 modality 应该 binding 在一起，attention pooling 让它们互相 attend
- **sequence length reduction**：每个 chunk 变成单个 token，避免 prompt 序列过长

**Step 2 - Cross-Attention**：当前 observation $o_{\text{cur}}$ 先 tokenize（每个 history timestep 一个 token），然后和 $P$ 做 cross-attention：

$$\text{Attn}(Q = W_Q o_{\text{cur}}, \ K = W_K P, \ V = W_V P)$$

其中 $Q, K, V$ 是标准的 attention 矩阵，learned positional embedding 加在 prompt 和 current observation 上。

prompt encoder 本身是一个 6-layer transformer decoder（参数量 57M，hidden size 768，8 heads，见 Table 4）。

这个 cross-attention 的物理意义是：**给定当前 observation，去 prompt 里找最相似的那一段 chunk，提取出来作为 sub-goal 信息**。Figure 5 的 attention visualization 验证了这一点——对于 DrawAnything 这种 dense drawing task，attention 是连续滑动跟踪的；对于 LIBERO-Gen 这种 discrete manipulation task，attention 是跳到下一个 "milestone"。

### 3.2 Action Decoder（Figure 2c）

输入：current observation $o_{\text{cur}}$ + 从 prompt encoder 输出的 relevant prompt info + diffusion timestep $k$

输出：通过 $K$ 步迭代 denoise 出 action

用 CNN U-Net action diffusion architecture（来自 Diffusion Policy, Chi et al.），151M 参数。FiLM conditioning 把 prompt info 和 diffusion step embedding inject 到 U-Net 各层。

diffusion 的 forward process：

$$q(a_t | a_0) = \mathcal{N}(a_t; \sqrt{\bar{\alpha}_t} a_0, (1 - \bar{\alpha}_t) I)$$

reverse process 学习 $\epsilon_\theta(a_t, t, o_{\text{cur}}, \text{prompt})$ 去预测 noise。变量含义：
- $a_0$：clean action chunk
- $a_t$：第 $t$ 步加噪后的 action
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$：cumulative noise schedule
- $\epsilon_\theta$：要学习的 noise predictor

为什么用 diffusion 而不是 L1（ICRT 那种）？因为 diffusion policy 在 multimodal action distribution 上表现好，manipulation task 经常有多个合理 action（比如 grasp 一个物体的不同角度）。

### 3.3 Training 和 Inference 的解耦

**Training**：每个 step
1. 从训练集 sample 一个 demonstration 作为 prompt
2. 从同 task 的其他 demonstration 采样 batch 个 (observation, action) pairs
3. End-to-end 学，**不要求 prompt 和 training pairs 之间有显式的 spatial/temporal correspondence**

这点很关键：policy 自己学怎么 leverage prompt，不需要人为标注 "prompt 的第 5 步对应 training pair 的第 3 步"。

**Inference**：
1. 每个 rollout 选一个 prompt（不是每步换）
2. Prompt chunk embeddings $P$ 一次性 cache，整个 rollout 复用
3. Prompt encoder 每个 inference step 调用一次，输出 relevant prompt info
4. Action decoder 跑 $K$ 步 denoising，每步不需要再 reference 整个 prompt

这个设计相对 ICRT 的好处是 ICRT 每个前向 pass 都要 reference 整个 prompt + 整个 rollout history（虽然用了 KV cache 缓解），而 BPP 把 prompt understanding 和 action generation 解耦，inference 效率更高。

### 3.4 与 ICRT 的对比（Appendix G）

| 维度 | BPP | ICRT |
|---|---|---|
| 架构 | 双 module：prompt encoder + action decoder | 单个 causal transformer decoder |
| Attention 类型 | cross-attention（obs → prompt） | causal self-attention |
| Action 生成 | diffusion | L1 loss |
| History | 固定长度 | 整条 rollout history（在 context 里） |
| Inference 效率 | prompt 一次 cache，denoising 不 reference prompt | 每次 forward 都 reference 全部 context，靠 KV cache |

ICRT 把 rollout history 全塞进 context 的设计有个隐患——paper 在 §4.1 提到："making it susceptible to OOD due to spurious correlations [32]"，引用的正是 Torne et al. 的 Long-context Diffusion Policies via Past-token Prediction，那篇也是关于这个问题的。BPP 用固定长度 history 规避了这个问题。

## 4. iPhUMI：硬件接口

UMI 原版用 GoPro，需要 SfM 做 mapping，setup 慢。iPhUMI 用 iPhone 15 Pro + ARKit 做 real-time SLAM，**zero mapping time**。

数据采集模态（Figure 15）：
- Main camera：1920×1440 @ 60Hz
- Ultrawide camera：640×480 @ 10Hz
- LiDAR depth：256×192 @ 60Hz（paper 里没用到）
- Gripper pose：60Hz（ARKit SLAM 输出，shared world frame）
- Gripper width：10Hz（通过 ultrawide camera 检测 ArUco tag）

还支持 bimanual + head-mounted 三设备共享一个 ARKit world frame，这个是和 HoMMI [35] 协同开发的。

测试时的 prompt 传输流程：iPhone 采集 → 无线传到 desktop → desktop 处理成 behavior prompt → condition BPP。这就实现了 "user 在现场演示一次，robot 立刻能执行" 的交互闭环。

## 5. Benchmark 设计

paper 引入了两个新 benchmark，都是为了**测试 test-time adaptation to unseen tasks**（不是 unseen objects/environments，而是 unseen action sequences）。

### 5.1 DrawAnything

**DrawAnything-Sim**：
- 2000 个 procedurally generated drawing tasks，每个 5 demos
- drawing 由 1-6 个 part 组合（line、Bezier curve、partial/full oval、free space）
- board orientation 随机（$-\pi$ 到 $\pi$），speed 变化，trajectory 有 noise
- 50 个 hand-collected unseen evaluation tasks

**DrawAnything-Real**：
- ARX robot arm + iPhone wrist camera + marker + spring（compliance）
- 1000 training tasks：200 human-collected (iPhUMI, 5 demos/task) + 800 scripted (6 demos/task)
- 10 evaluation tasks（4 training + 6 unseen）
- 6 DoF action space（sim 是 2D）
- 评估 metric：Chamfer distance（pixel 单位）

drawing 这个 domain 选得很好，因为：
1. **Continuous action adaptation**：每一步都是新的低层 action，不像 pick-place 那种离散 primitive
2. **必须 reference prompt**：goal image 完全不够用，必须知道"接下来画什么"
3. **Procedural generation 容易**：可以 scale 到 2000 个 task

### 5.2 LIBERO-Gen

**LIBERO-Gen Combination**：扩展 LIBERO Spatial（10 tasks → 174 tasks），同一个环境里两个相同的 bowl，task 是 pick 哪个 + 放到 9 个 location 中的哪个。held out 10 个 (pick, place) combination，其中 pick 和 place 都单独在 training 里见过，但 jointly 没见过。

**LIBERO-Gen Chain**：扩展 LIBERO Goal（10 tasks → 321 tasks），两步 chained task。first step 包括：open middle/top drawer、push plate、turn on stove、pick-place；second step 都是 pick-place。held out 10 个两步 chain，每一步的 primitive 单独见过，但组合没见过。

这两个 benchmark 都设计了很有意思的 generalization gap：**individual primitives 见过，combination 没见过**，正好测试 test-time adaptation 而不是全新 primitive 学习。

## 6. 实验结果与关键发现

### 6.1 主结果（Figure 4）

**DrawAnything-Sim**（unseen drawings）：
- BPP 相比 Goal-Image 减 80.7% error
- BPP 相比 ICRT 减 33.3% error
- Goal-Image 在 training drawing 上还行，unseen 完全崩盘（因为只能 reconstruct 最终形状，没有 step-by-step 信息）

**LIBERO-Gen Combination / Chain**：
- BPP 在两个 benchmark 都超过 Language 和 Goal-Image baselines
- BPP rival π0.5（虽然 BPP 没有 foundation pretraining，π0.5 有大规模预训练 + 100K LoRA fine-tune steps）
- 这个对比非常 impressive，说明 in-context demonstration 是个很强的 condition signal

### 6.2 Ablation（Figure 6，全在 DrawAnything-Sim 上）

**Q: Prompt 应该包含什么？**
- **必须包含 observation**：anchor prompt lookup，告诉 policy "现在 prompt 进展到哪"
- **必须包含 action**：提供 temporal transition 信息
- **proprioception 不重要**（因为已经 visual shown 在 observation 里）
- **observation 降频不能太狠**（<1Hz 性能掉）
- **attention pooling 比 separate tokens 好**

**Q: 训练数据该怎么配？**（这是 paper 最 actionable 的发现之一）
- **Task diversity > per-task quantity**：固定 demo 预算下，"many tasks × few demos/task" 比 "few tasks × many demos/task" 强很多
- Figure 6e 显示：只 5 demos/task，增加 task 数量持续提升 unseen task 性能
- Figure 6f 显示：训练 task 要够复杂（4-6 parts 比 1-3 parts 好）

这个 finding 跟 LLM 里的 in-context learning scaling 规律呼应：task diversity 是 in-context 能力涌现的关键 driver。

### 6.3 Attention 可视化（Figure 5）

这个 figure 是我最喜欢的部分，因为它给了一个 mechanistic interpretation：

- **DrawAnything**：attention 沿着 task progression 连续滑动跟踪，说明 BPP 学到了 "prompt lookup"——找到 prompt 中和当前 observation 最接近的那一段，然后从那里读取 upcoming action。这是一个 dense sub-goal conditioning。
- **LIBERO-Gen**：attention 跳到 discrete 的 milestone（task transition、下一个 object、placement location），是 sparse sub-goal conditioning。

也就是说，BPP 不是单纯 memorize training tasks，而是学了一个 **"prompt lookup + sub-goal extraction"** 的 meta-procedure，类似 LLM 学到的 in-context learning algorithm。

### 6.4 Laundry Folding Case Study（§4.2, Appendix A）

这是个 negative result，但很有启发性。

三个 task：fold left arm、fold right arm、fold bottom up（双臂）。每个 task ~150 demos。结果（Table 1）：

| Task | Language | BPP |
|---|---|---|
| Fold left arm | 96% | 76% (有 4% 做成了 fold right arm) |
| Fold right arm | 100% | 100% |
| Fold bottom up | 100% | 60% (有 24% 做成了 fold right arm) |

**Why BPP underperform**：低 task diversity 下，prompt 里丰富的 spatial/temporal 信息反而成了 noise source（duration 不同、spatial configuration 不同），导致 BPP 容易 overfit 到 spurious cue（比如 background variation）。Language 这种简洁的 task identifier 反而更鲁棒。

**Implication**：behavior prompting 是 high task diversity regime 的工具，不是万能药。当 task 数量少、task 之间区分度高时，language 反而更合适。

## 7. Limitations（§5）

paper 自己列了几条，我补充一些思考：

1. **Goal image / language 在简单 task 上够用**，没必要花成本采 prompt。所以理想系统应该 support 多种 task descriptor（其实 Vid2Robot [18] 和 VIMA [17] 都尝试过 multimodal prompt）。
2. **Tabletop manipulation 还没看到能 adapt 到完全 new action primitive**。LIBERO-Gen 都是 pick-place、push、open drawer 这些 training 见过的 primitive 重组，还没测试 "训练里从没见过 push，test time 给个 push demo" 这种。
3. **低 task diversity 下 BPP 弱于 language**——这是个 fundamental limitation，因为 in-context learning 本来就是 emergent ability，需要 task diversity 来 emerge。
4. **prompt 和 execution 还是 same environment**，跨环境（比如 sim 里 prompt、real 里 execute）还没测。
5. **没和 foundation model 结合**——paper 提到未来可以 apply behavior prompting 到 foundation-level pretrained model，这是个很自然的 next step（比如把 π0.5 + behavior prompting 结合）。

## 8. 跟相关工作的 positioning

paper 在 §2 把自己定位在 in-context few-shot learning 这一类，区别是 prompt representation 选了 sensorimotor demonstration。几个最相关的工作：

- **ICRT [23]**（Fu et al.）：直接前序，causal autoregressive transformer，BPP 在它基础上改进了架构（decoupled modules）和规模（task 数量从 29 到 2000+）
- **Vid2Robot [18]**：用 human video 当 prompt，cross-attention，BPP 架构借鉴了它的双 module 设计
- **One-shot imitation [15]**（Duan et al., 2017）：早期 one-shot 工作，用 attention over demonstration
- **Task parameterized methods [12]**（Calinon）：用 explicit reference frame transformation，强 prior 但弱 generalization
- **VIMA [17]**：multimodal prompt（text + image），但是不是用完整 demonstration
- **π0.5 [5]**：VLA foundation model，做 baseline，BPP 在 LIBERO-Gen 上 rival 它

## 9. 我的 takeaways 和 intuition

1. **Task diversity 是 in-context learning 的 fuel**。这个 finding 在 LLM 里已经被验证（task 数量多才有 in-context learning），现在 robotics 也呼应了。这意味着 robotics 要走向 generalist，关键不是单 task 的 demo 数量，而是 task 的覆盖广度。

2. **Demonstration 是最 expressive 的 prompt format**。Language 是 compressed task descriptor，丢失了 how 和 when；goal image 是 final state，丢失了过程；demonstration 保留了全部信息。代价是 representation 复杂、对 policy 容量要求高、需要 cross-attention 推理。

3. **In-context learning 在 robotics 里需要的"任务多样性"门槛比 LLM 高得多**。LLM 用 internet-scale 文本预训练就 emerge 了 in-context learning；robot 这里要 2000+ 个 task 才看到明显效果，而且还没到完全 generalizable。

4. **Decoupled 架构（prompt encoder + action decoder）相比 unified causal transformer 有几个 advantage**：inference 效率高、可以固定 history 长度避免 OOD、可以复用 action diffusion 的 multimodal 能力。这是个好工程选择。

5. **Behavior prompting 的 sweet spot 是 "task descriptor 难以 compressed 表达" 的场景**：drawing、folding preference、特定 grasp 策略。这些 task 用 language 描述很别扭，用 demonstration 反而自然。

6. **Future direction**：把 behavior prompting 和 foundation VLA model 结合是个显然的下一步。现在 BPP 没用预训练就 rival π0.5，如果加上预训练，可能进一步提升。还有 sim-to-real prompting（sim 里 prompt、real 里 execute）也很有意思。

7. **数据接口很重要**。iPhUMI 这种"现场演示一次就传 prompt"的设计让 behavior prompting 真正可用。Robot learning 里 hardware interface 经常被低估，但这篇 paper 显示好的接口能 unlock 整个 paradigm。

参考链接：
- Project page: https://behavior-prompting.github.io
- LIBERO benchmark: https://arxiv.org/abs/2304.06752
- UMI: https://arxiv.org/abs/2402.10329
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ICRT: https://arxiv.org/abs/2408.15980
- π0.5: https://arxiv.org/abs/2410.24164 (π0) 以及后续 π0.5 CoRL 2025
- RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html
- Octo: https://octo-models.github.io/
- VIMA: https://vimalabs.github.io/
- Vid2Robot: https://rss24.github.io/
- One-shot imitation (Duan 2017): https://arxiv.org/abs/1703.07318

希望这个讲解能帮你 build 起对 behavior prompting paradigm 的 intuition。这个方向我觉得还有很大空间——尤其是把 behavior prompt 和 VLA foundation model pretraining 结合，以及解决跨环境（sim→real）prompting 的 robustness 问题。
