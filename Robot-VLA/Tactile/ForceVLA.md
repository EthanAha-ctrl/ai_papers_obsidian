---
source_pdf: ForceVLA.pdf
paper_sha256: 1327311916c59b84a866b2b2fd5849bd27bbbf68ccebe302560abfcb9e52e3ca
processed_at: '2026-08-04T09:51:20-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ForceVLA 用人话讲

Andrej，我把刚才那些公式和表格剥离掉，用大白话把这篇 paper 的核心直觉讲一遍。

---

## 痛点是什么

现在 VLA 模型（OpenVLA, π₀, RT-2）做 pick-and-place 挺溜，但是遇到**需要接触、需要用力**的任务就傻眼了——插 USB、拧瓶盖、削黄瓜皮、擦白板。这些任务有个共同特征：**一旦 gripper 碰上物体，视觉就基本废了**。

为什么废了？三个原因：
1. **物理遮挡**：gripper 自己把 camera 视线挡住了，你看不到 plug 到底对没对齐 socket
2. **精度天花板**：camera 的 depth sensing 噪声在毫米级，但是 insertion 要的是微米级调整
3. **信号缺失**：摩擦力、滑不滑、卡没卡住——这些物理量 vision 根本感知不到，就像你闭着眼睛摸黑走路，脚踩到什么感觉都没有

人类做这些任务靠什么？**手感**。你闭着眼也能把钥匙插进锁孔，靠的就是手上的 force feedback 在引导你做 compliant motion——歪了就转一下，卡了就退一点。

所以核心问题就是：**VLA 模型缺了"手感"这个 modality**。

---

## ForceVLA 的核心 insight

一句话：**把 force/torque 当成一等公民 modality，别当成 state 的附属维度**。

什么叫"一等公民"？现有方法（π₀ w/ F）的做法是把 6 维 force/torque 拼到 proprioception state 后面，和 TCP pose、gripper width 一起扔进 network。这就像你给一个人手上贴了个力传感器，但大脑根本不知道怎么处理这个信号——它淹没在一堆 positional encoding 里了。

ForceVLA 的做法是：**force 单独走一条路，用专门的 MoE 模块和 vision-language 做 fusion**。这就像大脑里有一个专门的"体感皮层"负责处理触觉，而不是把触觉信号随便塞进视觉皮层。

---

## 架构直觉

整个 pipeline 可以类比为**大脑的感觉处理层次**：

1. **VLM（PaliGemma）= 初级视觉皮层 + 语言皮层**：先把 image 和 instruction 编码成 abstract representation。这一步 force 完全不参与。

2. **FVLMoE = 顶叶联合区**：这里是 vision、language、force 三路信号汇合的地方。Force 经过一个 linear projection 变成一个 token，append 到 VL token 序列后面，然后一起过 Transformer encoder 做 self-attention，再过一个 sparse MoE 层。

3. **Flow head = 运动皮层**：融合后的 feature 作为 guidance，引导 π₀ 的 flow matching 去噪生成 action trajectory。

关键设计点：**force 是在 VLM 输出之后才介入的**，这叫 late fusion。

---

## 最反直觉的发现

整个 paper 里最 striking 的实验结果是 Table 3 的 ablation：

- Early fusion（force 在 VLM input 阶段混进去）：**MoE 版直接 0% 成功率**
- Late fusion + MoE（ForceVLA）：**80%**

0% 这个数字太刺眼了。为什么 early fusion 会彻底崩？直觉解释是：**pretrained VLM 的 feature distribution 是 fragile 的**。PaliGemma 经过大规模预训练，它的 token embedding 收敛在一个特定的 manifold 上。你往 input 里塞一个 statistics 完全不同的 force token，就像往一个精密调校过的交响乐团里突然塞进一个电钻——整个 harmony 就乱了。

这和 LLM 的 instruction tuning 逻辑类似：你不能随便往底层 embedding 里加新东西，得在 output 端做 adapter。

另一个反直觉点：**naive concatenation 几乎没用**。π₀-base 加了 force 只从 37.3% 涨到 40.2%，才 2.9 个点。但是 ForceVLA 的 MoE fusion 涨了 23.2 个点。

这说明：**关键不在于有没有 force 信号，而在于怎么 fuse**。Force 信号本身是好的，但是 network 需要一个专门的机制去学会什么时候用它、怎么用它。MoE 的 router 正好干这个活——它学会了 phase-aware routing，不同任务阶段激活不同 expert。

---

## MoE 为什么 work

这个要 build intuition 的话，可以这么想：contact-rich 任务有**不同的 phase**，每个 phase 的 force dynamics 完全不一样。

以 plug insertion 为例：
- **Approach phase**：free-space motion，force ≈ 0，主要靠 vision 引导
- **Contact onset**：突然撞上 socket，force spike，需要立刻减速
- **Search phase**：在 socket 表面滑动找孔，lateral force + friction
- **Insertion phase**：对准后压进去，axial force 主导

一个 monolithic network 很难同时学好这四种 mode。MoE 的好处是 router 可以学会**根据当前 force profile 自动切换 expert**——approach 时走 Expert A，contact onset 走 Expert B，以此类推。

Appendix C 的 expert load 分析证实了这点：不同任务、不同 phase 下 expert 激活 pattern 确实不同。Expert 0 像个 generalist 占了一半 load，其他 expert 做 phase-specific 的 specialization。

---

## 最强的 evidence

Table 2 的 Visual Occlusion setting：**ForceVLA 90%，baseline 60%**。

这个实验最能说明问题。当 vision 被人为遮挡时，force feedback 直接接管了 perception 的角色。这和人类行为完全一致——你闭着眼睛也能完成 insertion，因为你手上有感觉。

反过来，Unstable Socket 只有 20%，说明 force feedback 在 environment 本身 dynamic 变化时也有局限。6-axis F/T 只能感知施加在 TCP 上的力，感知不了 socket 在哪。这可能需要 tactile sensing（指尖的 spatial 分布式触觉）来补。

---

## 局限和我的联想

**1. Force 是单 token，没有时间维度**

当前设计把 6 维 F/T 投影成 1 个 token。但是做 insertion 时，你关心的不只是当前这一刻的力，还有**力的变化趋势**——刚才在往左滑还是往右滑？卡住了还是进展了？单 token 丢失了 temporal context。

改进方向：维护一个 force history buffer，比如最近 10 步的 force，project 成 10 个 token，让 attention 学 temporal pattern。这就像 RNN 的 hidden state，但是用 attention 来做。

**2. Estimated wrench 精度有限**

Paper 用的是 Flexiv Rizon 的 joint torque 估计 external wrench，不是外置 F/T sensor。精度大概 0.5-1N 级别。但是 USB insertion 需要感知 <0.1N 的力变化。这可能解释了为什么 USB insertion multi-task 只有 10% 成功率。

**3. Action chunking 和 reactive control 的矛盾**

π₀ 输出 50 步 action chunk，相当于 open-loop 执行 50 步才 re-plan。但是 force feedback 的核心价值是实时 reactive——撞上了立刻退。如果 chunk 太长，force 的 reactive 价值被稀释了。

真正的 contact-rich 可能需要**双层控制**：outer loop 做 trajectory planning（VLA 负责），inner loop 做 impedance control / force-position hybrid control（传统控制负责）。Adaptive Compliance Policy (https://arxiv.org/abs/2410.09309) 就是这个思路。

**4. 数据量太小**

244 trajectories、5 个 task。这个规模训练 4-expert MoE 勉强够，但是要 scale 到更多 task、更多 expert 肯定不够。未来需要类似 Open X-Embodiment 规模的 force-aware dataset。

**5. Force encoder 太简单**

Vision 有 SigLIP 预训练，language 有 Gemma 预训练，force 呢？一个 linear projection。这不公平。Force 也需要一个 pretrained encoder——用大规模 force data 预训练一个 force foundation model，类似 SigLIP 之于 vision。这个方向大有可为。

---

## 一句话总结

ForceVLA 告诉我们：**机器人要学手感，光给数据不够，还得给大脑一个专门的回路去处理它**。Force 信号不能随便塞进 vision-language 的 soup 里，得有 dedicated 的 fusion mechanism。MoE 的 router 正好学会了"什么时候用力、怎么用力"这个 meta-skill。

这个 lesson 对未来 multimodal robot learning 是通用的——tactile、audio、IMU 这些新 modality 都不能 naive concat，需要 dedicated fusion。这是 VLA 从"看听说"走向"全感官"的第一步。

---

## References

- ForceVLA: https://forcevla.github.io/
- π₀: https://arxiv.org/abs/2410.24164
- Forceful Foundation Models Survey: https://arxiv.org/abs/2504.11827
- Adaptive Compliance Policy: https://arxiv.org/abs/2410.09309
- ForceMimic: https://arxiv.org/abs/2410.07554
- Foar: https://arxiv.org/abs/2411.15753
- LIMOE: https://arxiv.org/abs/2205.13709
- Switch Transformer: https://arxiv.org/abs/2101.03961
- TLA: https://arxiv.org/abs/2503.08548
- TacMan: https://ieeexplore.ieee.org/document/10409920

Andrej，这个直觉版希望更对胃口。如果你想聊 force foundation model 的预训练范式，或者 MoE routing 的 interpretability 怎么做，我可以继续展开。

---

# ForceVLA: Force-aware MoE for Contact-rich Manipulation 深度解析

Hi Andrej，这篇 ForceVLA 我觉得挺有意思的——它处理的是 VLA 社区长期存在的一个痛点：contact-rich 任务里 vision 完全 blind 的那部分。我来详细拆解一下架构、数学和实验直觉。

---

## 1. Core Problem & Motivation

当前 VLA models（OpenVLA, π₀, RT-2）主要依赖 vision-language 作为输入。对于 pick-and-place 这种 free-space manipulation 还可以，但是遇到 **insertion, peg-in-hole, tool use, surface wiping** 这类任务，视觉信息在 contact 发生瞬间就**饱和甚至失效**了：

- **Visual occlusion**：plug 进入 socket 后被 gripper 和 hand 遮挡，camera 看不到 alignment 细节
- **Sub-millimeter precision**：vision 的 depth sensing 噪声通常在 mm 量级，无法支撑 insertion 阶段所需的 μm-level 调整
- **Dynamic contact**：friction, compliance, slip 这些物理量 vision 完全感知不到

而人类做这类任务时，**proprioception + tactile feedback** 是主导信号——你闭着眼睛也能把 USB 插进去，靠的就是 force feedback 引导的 compliant motion。ForceVLA 的核心 insight 是：**force/torque 不应该作为 state 的简单拼接维度，而应该作为 first-class modality 在 action decoding 阶段被 dedicated fusion module 处理**。

这里的关键 reference 是 William Xie & Correll 的 forceful foundation models survey：https://arxiv.org/abs/2504.11827

---

## 2. Problem Formulation 深入

Observation space 定义：

$$O_t = \{V_t^b, V_t^h, s_t, f_t\}$$

其中：
- $V_t^b$：base/third-person camera RGB（D435, 1280×720, 30 FPS）
- $V_t^h$：hand/wrist camera RGB（D415, 640×480, 30 FPS）
- $s_t \in \mathbb{R}^7$：proprioception = TCP pose $(x, y, z, \alpha, \beta, \gamma)$ + gripper width
- $f_t \in \mathbb{R}^6$：6-axis external wrench = $\{f_{tx}, f_{ty}, f_{tz}, m_{tx}, m_{ty}, m_{tz}\}$（force + moment，expressed in world frame）

Policy 输出 action chunk：

$$A_t = \{a_t, a_{t+1}, ..., a_{t+H-1}\}$$

每个 $a$ 是 target TCP pose + gripper width，horizon $H$ 由 π₀ 的 flow matching head 决定（通常 H=50 左右）。

注意 $f_t$ 是 **estimated external wrench**——不是直接从 F/T sensor 读的 raw signal，而是经过 robot dynamics 补偿后估计的"外界施加在 TCP 上的 wrench"。这是 paper 的 limitation 之一，但也意味着 Flexiv Rizon 这种 arm 本身的 joint torque sensing 已经够用，不需要 external F/T sensor。这点和 ForceMimic (https://arxiv.org/abs/2410.07554) 的 force-motion capture system 思路不同。

---

## 3. FVLMoE 架构详解

这是整篇 paper 的核心。我用 token flow 来描述：

### 3.1 为什么 Late Fusion？

这是 ablation study (Table 3) 里最重要的发现：

| Model | Success Rate |
|-------|--------------|
| baseline (π₀) | 45% |
| linear before VLM | 55% |
| **MoE before VLM** | **0%** |
| concate after VLM | 60% |
| ForceVLA (MoE after VLM) | **80%** |

**MoE before VLM 直接 0%** 这点很 striking。原因是：pretrained VLM（这里是 PaliGemma + SigLIP）的 token embedding 分布已经收敛在一个特定的 manifold 上，如果你在 VLM input 阶段就把 force token 混进去，会破坏 vision-language alignment 的 feature distribution。这和 LLM 的 instruction tuning 类似——你不能随便往底层 embedding 里塞新 modality。

而 late fusion 的逻辑类似于**大脑皮层的高级联合区**（比如 intraparietal sulcus 处理 visuomotor integration）：低级感觉皮层先把各自的信号 process 成 abstract representation，再送到联合区做 modality binding。

### 3.2 FVLMoE 的 Token Flow

**Stage 1: VLM Encoding**

PaliGemma 把 $V_t^b, V_t^h, L$ 编码成 VL embedding：

$$E_{VL} \in \mathbb{R}^{N_{VL} \times D_{model}}$$

其中 $N_{VL}$ 是 VL token 数量（PaliGemma 通常输出几百个 token），$D_{model} = 2048$（与 Gemma-2B hidden dim 对齐）。

**Stage 2: Force Token Projection**

6-axis force 经过一个 linear projection $\phi_F$：

$$E_F = \phi_F(f_{raw}) \in \mathbb{R}^{D_{model}}$$

注意这里 **force 是单一 token**（不是 sequence）。因为 6 维 force/torque 是 instantaneous scalar reading，没有 temporal context。这个 design choice 我觉得有讨论空间——后面我会展开。

**Stage 3: Concatenation**

$$E_{in} = [E_{VL}; E_F] \in \mathbb{R}^{(N_{VL}+1) \times D_{model}}$$

Force token 被 append 到 VL 序列末尾。

**Stage 4: Pre-MoE Encoder**

一个标准 Transformer encoder block（multi-head self-attention + FFN），让 force token 和所有 VL token 互相 attend：

$$E_{enc} = \text{TransformerEncoder}(E_{in})$$

参数：$N_H = 8$ heads, $D_h = 256$ per head, MLP expansion factor 1（这个比较奇怪，通常 FFN expansion 是 4x，这里 1x 可能是为了控制参数量）。

**Stage 5: Sparse MoE Layer**

这是核心。4 个 expert（每个是独立 MLP：$D_{model} \rightarrow D_{model}$），router 是一个 linear layer $G: \mathbb{R}^{D_{model}} \rightarrow \mathbb{R}^E$。

标准 MoE 公式：

$$y(x) = \sum_{i \in \text{TopK}(G(x))} g_i(x) E_i(x)$$

变量解释：
- $x$：input token embedding
- $G(x) \in \mathbb{R}^E$：router logits，$E=4$
- $\text{TopK}(G(x))$：取 logits 最大的 $k$ 个 expert index，这里 $k=1$
- $g_i(x)$：softmax 后的第 $i$ 个 expert 的 gating weight
- $E_i(x)$：第 $i$ 个 expert 的输出

**Top-k=1** 是非常 sparse 的选择。对比 Switch Transformer 也是 top-1，但 GShard 用 top-2。这里 top-1 的好处是 routing 决策非常 sharp，每个 token 明确走一个 expert，方便 interpretability（见 Appendix C 的 expert load 分析）。

加上 residual connection：

$$E_{fused} = E_{enc} + \text{MoE}(E_{enc})$$

**Stage 6: Output Projection**

Linear：$\mathbb{R}^{D_{model}} \rightarrow \mathbb{R}^{D_{act\_e}}$，$D_{act\_e} = 1024$。

### 3.3 注入 Action Flow Head

这里和 π₀ 的 flow matching 对接很巧妙。π₀ 的 action expert 接收两个输入：
- $S_{suffix}$：当前 proprioception state $s_t$ 经过 projection 后的 embedding，shape $\mathbb{R}^{H_{action} \times D_a}$
- $a_t^\tau$：noisy action trajectory at denoising step $\tau$，shape $\mathbb{R}^{H_{action} \times D_a}$

ForceVLA 从 FVLMoE 输出里**抽取最后 $H_{action}$ 个 token**：

$$G_{\text{FVLMoE}} \in \mathbb{R}^{H_{action} \times D_a}$$

然后**element-wise addition**：

$$\text{Action Expert Input} = S_{suffix} + G_{\text{FVLMoE}}$$

这个 additive injection 的设计直觉是：force 信号作为 **residual guidance** 调制 action trajectory，而不是完全 override proprioception。Flow matching 的 iterative denoising 过程中，每一步都会重新 attend 这个 fused guidance。

Flow matching 本身的公式（来自 π₀ / Flow Matching for Generative Modeling, https://arxiv.org/abs/2210.02747）：

$$v_\theta(a^\tau, \tau | c) = -[a^\tau - a^0] / \tau$$

其中 $c$ 是 conditioning（VL + force + state），$\tau$ 是 denoising timestep。这个 velocity field 被 ForceVLA 的 fused feature 直接 condition。

---

## 4. Dataset: ForceVLA-Data

数据规模其实不大：**244 trajectories, 140K timesteps, 5 tasks**。每个 task 约 50 demos。这个规模比 Open X-Embodiment 小两个数量级，但**任务特异性高很多**——所有任务都是 contact-rich。

Hardware setup：
- Robot: Flexiv Rizon 7-DOF（这是阻抗控制 arm，force sensing 是内嵌的）
- Gripper: Dahuan adaptive gripper
- Cameras: RealSense D435 (third-person) + D415 (wrist)
- Teleop: Quest3 VR interface

5 个 task 的 contact dynamics 非常 diverse：

| Task | Contact Type | Force Modality |
|------|--------------|----------------|
| Bottle Pumping | Vertical pressing | Transient impact + sustained normal force |
| Plug Insertion | Peg-in-hole | Lateral friction + alignment torque |
| USB Insertion | Peg-in-hole (asymmetric) | 高 precision + insertion torque |
| Whiteboard Wiping | Surface sliding | Sustained normal force + friction |
| Cucumber Peeling | Tool-surface contact | Continuous force modulation + contour following |

这个 task suite 设计得不错——覆盖了 transient / sustained / continuous 三类 force interaction。对比 Forge (https://arxiv.org/abs/2504.11827) 或 Foar (https://arxiv.org/abs/2411.15753) 这些 force-centric 方法，ForceVLA 的 task diversity 明显更广。

---

## 5. 实验结果深度分析

### 5.1 Main Results

Figure 5 的核心数字：

| Model | Avg Success Rate |
|-------|------------------|
| π₀-base w/o F | 37.3% |
| π₀-base w/ F | 40.2% |
| π₀-fast w/o F | 31.0% |
| π₀-fast w/ F | 14.2% |
| **ForceVLA** | **60.5%** |

几个值得注意的点：

1. **Naive force concatenation 收益有限**：π₀-base w/o F → w/ F 只提升 2.9%。这说明单纯把 6 维 force 拼到 state 后面，VLA 学不会怎么用这个信号。
2. **π₀-fast w/ F 反而崩溃**：31.0% → 14.2%。Paper 解释是 fast 的 token space 高度优化和 compact，naive force token 会 disrupt 它。这个观察很重要——**action tokenization 的 compactness 和 modality extensibility 之间存在 trade-off**。
3. **ForceVLA 的 23.2% 提升** 主要来自 FVLMoE 的 adaptive fusion，而不是 force 信号本身。

### 5.2 Cucumber Peeling 的细节指标

Table 1 很有信息量：

| Model | Avg Peel Length | Min Strokes |
|-------|-----------------|-------------|
| π₀-base w/o F | 10.27 cm | 14 |
| π₀-base w/ F | 13.17 cm | 10 |
| ForceVLA | **14.12 cm** | **7** |

Peeling 这种 continuous surface contact 任务，force 的作用非常直接——peeler 需要维持稳定 normal force 否则会跳刀或卡住。ForceVLA 的 stroke efficiency 提升 2x，说明 force feedback 让 policy 能 sustain contact 而不是反复 lift-off。

### 5.3 Generalization

Table 2 的 5 个 generalization setting 设计得很好：

1. **Object Gen. 1**：换 bottle shape → test force profile 的 transferability
2. **Object Gen. 2**：换 plug geometry → test insertion force 的 adaptability
3. **Height Gen.**：变 socket 高度 → test compliance under torque limit
4. **Visual Occlusion**：遮挡 → 直接 test force-only 的 robustness
5. **Unstable Socket**：socket 会晃 → test reactive force control

ForceVLA 在 Visual Occlusion 上达到 90%（baseline 60%），这是**最强 evidence**——当 vision 失效时，force 接管了 perception 的角色。这和人类行为完全一致。

不过 **Unstable Socket 只有 20%**，说明 force feedback 在 dynamic environment（socket 本身在动）下还是有限制。这可能是因为 6-axis F/T 只能感知施加在 TCP 上的力，无法直接感知环境物体的运动。这个 limitation 很有意思——可能需要 tactile sensing（比如 TacMan, https://arxiv.org/abs/2401.09658）或者 visual tracking of environment 来补充。

### 5.4 Multi-task Joint Training

Table 5 的 joint training 结果更惊人：

| Model | Avg SR |
|-------|--------|
| π₀-fast w/o F | 0.0% |
| π₀-base w/o F | 5.0% |
| π₀-base w/ F | 42.5% |
| ForceVLA | **67.5%** |

π₀-fast 在 multi-task 下**完全失败**（0%），这暴露了 FAST tokenizer 在 multi-task setting 下的 limitation——它的 action vocabulary 是 task-specific 的，混在一起会 collapse。

---

## 6. Expert Routing 分析（Appendix C）

这是 paper 最 fascinating 的部分。Figure 9 展示了 4 个 expert 在不同 task completion percentage 下的 load distribution。

Key observations：

1. **Expert 0 dominates ~50% tokens across all tasks**——它是一个 general-purpose expert，处理 modality fusion 和 routine control。
2. **Insert Plug / Peel Cucumber 有 temporal specialization**——某些 expert 只在特定 phase 激活。这和人类 motor cortex 的 phase-specific neuron firing 类似。
3. **Wipe Board 倾向于 single expert throughout**——因为这个 task 的 force profile 比较稳定（constant normal force + sliding）。

这种 emergent specialization 是 MoE 的最大价值——**router 自己学会了 phase-aware 和 modality-aware 的 routing**。Paper 没有明确说哪个 expert 对应哪个 modality，但从 routing pattern 推测：
- Expert 0: VL-Force fusion（general）
- Expert 1: Contact onset（high force gradient）
- Expert 2: Free-space motion（low force）
- Expert 3: Sustained contact（stable force）

这个 hypothesis 可以通过 ablation expert 来验证，paper 没做，但值得 follow-up。

---

## 7. 与相关工作的定位

### 7.1 vs π₀

ForceVLA 本质是 π₀ + force modality。π₀ 用 PaliGemma + flow matching，ForceVLA 在两者之间插入 FVLMoE。架构上非常 minimal 的改动，但效果显著。参考 π₀ paper: https://arxiv.org/abs/2410.24164

### 7.2 vs TacDiffusion / Foar / ForceMimic

这些 force-centric 方法通常用专门的 force-conditioned policy，没有 VLA 的 semantic understanding。ForceVLA 的优势是**保留 VLA 的 generalization，同时补上 force 的 precision**。

- TacDiffusion (https://arxiv.org/abs/2409.11047): force-domain diffusion
- Foar (https://arxiv.org/abs/2411.15753): force-aware reactive policy
- ForceMimic (https://arxiv.org/abs/2410.07554): force-motion capture

### 7.3 vs TLA / Tac-Man

TLA (https://arxiv.org/abs/2503.08548) 和 Tac-Man 用 tactile sensing 而不是 F/T。Tactile 的空间分辨率高但 temporal bandwidth 低，F/T 反过来。ForceVLA 选 F/T 是 pragmatic 选择——更容易部署。

### 7.4 vs MoE-VLA works

LIMOE (https://arxiv.org/abs/2205.13709) 是 vision-language MoE，但没有 force。MORE (https://arxiv.org/abs/2503.08007) 是 quadruped VLA MoE。ForceVLA 是第一个把 force 作为 explicit modality 加入 MoE routing 的 VLA。

---

## 8. 我的 Intuition & 批判性思考

### 8.1 Force Token 设计的局限

ForceVLA 把 6-axis F/T 投影成**单一 token**。这个 design 简单但 suboptimal：

- **No temporal context**：当前时刻的 F/T 没有 history。人类做 insertion 时，会记住刚才的 force profile 来判断 alignment 进展。
- **No spatial grounding**：force token 没有 spatial position embedding，无法和 vision token 做 spatial correspondence。

更好的做法可能是：
1. **Force history**：维护一个 force buffer $\{f_{t-K}, ..., f_t\}$，project 成 K 个 token，让 attention 学 temporal pattern。
2. **Force-Vision cross-attention**：explicitly 让 force token attend 到 vision token 的 contact region（比如 gripper tip 的 vision feature），建立 spatial binding。

### 8.2 MoE 容量的 trade-off

4 个 expert + top-1 太 sparse 了。每个 token 只走一个 expert，意味着 modality fusion 的 capacity 受限。对比 LIMOE 用几十个 expert。ForceVLA 的 4-expert 设计可能是数据量限制（244 trajectories 不够训大 MoE），但理论上 force VL fusion 可能需要更多 expert 来覆盖不同的 contact phase × modality combination。

### 8.3 Estimated Wrench 的精度问题

Paper limitation 里提到用的是 estimated external wrench，不是 direct F/T sensor。Flexiv Rizon 的 joint torque estimation 精度大概在 0.5-1N 量级，对于 fine insertion（USB 需要 <0.1N 的 sensitivity）可能不够。这解释了为什么 USB insertion 只有 10%（Table 5）。

### 8.4 Action Chunking 和 Force Feedback 的矛盾

π₀ 的 action chunking 输出 H≈50 步的 trajectory，这意味着 policy 是 open-loop 执行 50 步才 re-plan。但是 force feedback 的价值在于**实时 reactive control**。这里有一个 fundamental tension：

- 如果 H 太大，force feedback 的 update rate 太慢，失去 reactive 价值。
- 如果 H 太小（比如 H=1），flow matching 的 trajectory prior 丧失。

ForceVLA 的做法是 chunking + denoising 时 condition force，相当于把 force 作为 planning prior。但是真正 reactive 的部分可能需要 **inner-loop impedance control**（参考 Adaptive Compliance Policy, https://arxiv.org/abs/2410.09309）来 complement。

### 8.5 为什么 Late Fusion 比 Early Fusion 好？

这个 ablation 结果（Table 3）的深层原因我觉得是：

1. **Pretrained VLM 的 feature manifold 是 fragile 的**。任何 input perturbation 都会破坏 VL alignment。
2. **Force 信号的 statistics 和 VL 完全不同**。Vision 是 dense spatial tensor，force 是 low-dim scalar。混在一起会让 attention 机制 confuse。
3. **Late fusion 让 VLM 先做 semantic understanding，再让 force 做 physical refinement**。这是 hierarchical processing，符合 neuroscience 的原理。

但是 late fusion 也有 cost——force 信号无法影响 VLM 的 attention。如果 contact region 在 vision 里被遮挡，VLM 的 attention 可能 focus 错地方。一个可能的改进是 **iterative fusion**：VLM 先输出 coarse attention，force 提示后 refine attention，再输出 final embedding。

---

## 9. 对未来工作的启发

1. **Force as cross-modal attention prior**：与其把 force 作为 token，不如用 force gradient 来 modulate vision token 的 attention weights。
2. **Tactile + F/T fusion**：tactile 给 spatial distribution，F/T 给 global wrench。两者互补。
3. **Force-conditioned action chunking horizon**：检测到 contact onset 时自动缩短 H，提高 reactive rate。
4. **Sim-to-real with force**：Isaac Gym / MuJoCo 的 force simulation 精度不够，这是 sim-to-real for contact-rich 的瓶颈。Forge 和 ForceMimic 都在尝试解决这个。
5. **Foundation model for force**：类似 SigLIP 之于 vision，需要一个 pretrained force encoder。当前 ForceVLA 用 linear projection 太简单。

---

## 10. 总结

ForceVLA 的核心贡献是把 force 从"state 的一个维度"提升为"first-class modality"，并用 MoE 做 modality-aware fusion。架构上很 minimal（一个 FVLMoE block），但效果显著（+23.2% avg, +35% on insertion under occlusion）。

最重要的 takeaway 是 **fusion strategy > modality presence**——naive concatenation 几乎没用，adaptive MoE fusion 才是关键。这个 lesson 对未来 multimodal robot learning 通用——任何新 modality（tactile, audio, IMU）都不能简单 concat，需要 dedicated fusion mechanism。

Paper 的 limitation 也很明确：data scale 小、estimated wrench 精度有限、single force token 无 temporal context。这些都是 follow-up 的明确方向。

---

## References

- ForceVLA 主页（待 release）：https://forcevla.github.io/
- π₀: https://arxiv.org/abs/2410.24164
- OpenVLA: https://arxiv.org/abs/2406.09246
- PaliGemma: https://arxiv.org/abs/2407.07726
- Flow Matching: https://arxiv.org/abs/2210.02727
- Rectified Flow: https://arxiv.org/abs/2209.14577
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- Forceful Foundation Models Survey: https://arxiv.org/abs/2504.11827
- ForceMimic: https://arxiv.org/abs/2410.07554
- Foar: https://arxiv.org/abs/2411.15753
- TacDiffusion: https://arxiv.org/abs/2409.11047
- Adaptive Compliance Policy: https://arxiv.org/abs/2410.09309
- TLA: https://arxiv.org/abs/2503.08548
- TacMan: https://ieeexplore.ieee.org/document/10409920
- Forge: https://arxiv.org/abs/2501.06418
- LIMOE: https://arxiv.org/abs/2205.13709
- Switch Transformer: https://arxiv.org/abs/2101.03961
- GShard: https://arxiv.org/abs/2006.16668
- ST-MoE: https://arxiv.org/abs/2202.08906
- SigLIP: https://arxiv.org/abs/2303.15343
- Multi-gate MoE (MMoE): https://dl.acm.org/doi/10.1145/3219819.3220007
- Impact (VLM contact planning): https://arxiv.org/abs/2503.10110
- Gr00t N1: https://arxiv.org/abs/2503.14734
- HybridVLA: https://arxiv.org/abs/2503.10631
- DexVLA: https://arxiv.org/abs/2502.05855
- CoT-VLA: https://arxiv.org/abs/2503.22020
- PointVLA: https://arxiv.org/abs/2503.07511
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DROID dataset: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- BridgeData V2: https://arxiv.org/abs/2308.12952
- RT-1: https://arxiv.org/abs/2212.06817
- RT-2: https://arxiv.org/abs/2307.15818
- Octo: https://arxiv.org/abs/2405.12213

希望这些分析对你 build intuition 有帮助，Andrej。如果你对 force token 的 temporal modeling 或者 expert routing 的 interpretability 想深入聊，我很 happy 继续。
