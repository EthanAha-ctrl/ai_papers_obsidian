---
source_pdf: MoLe-VLA.pdf
paper_sha256: da1d8ca001d27d8b694c4010df03981d5a3c2d64bd822ff22a05955460590c1a
processed_at: '2026-08-05T19:57:17-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用最直白的话讲。

## 一句话版本

7B VLA model太慢了，但中间层90%都在说废话——我们让一个router根据当前画面和指令，动态决定哪些层值得跑、哪些直接跳过，然后用一个"完整版teacher"告诉"瘦身版student"该关注哪些token，结果算力砍半，效果反而更好。

## 为什么这件事make sense

你训一个7B LLM去做robot action prediction，其实大部分layer的output跟上一层几乎一样（cosine sim > 0.9）。这些层在"空转"。

但直接砍掉后面几层（early exit）也不对——最后几层恰恰是semantic information最密集的地方，砍了就变傻。

所以正确做法是 **selective skipping**：中间有些层该跑就跑，该跳就跳，具体取决于当前input。

## 三个关键决策

**1. 怎么决定跳哪些层？**

搞一个router。普通MoE的router是个linear layer，但机器人任务的input有两类：image（spatial）和text（temporal），混在一起会丢inductive bias。所以STAR把两条路分开走：image过MLP提spatial feature，text过Transformer提temporal feature，再融合成layer selection。

**2. 跳了层会变笨怎么办？**

用full model当teacher，skip model当student。但别naively对齐所有token——visual token和task-relevant token重要性不同。引入一个learnable cognition token，它学到"聚合vision+language的任务理解"，然后用它去identify哪些token是关键token（ToIs），只distill这部分。这叫CogKD。

**3. Teacher怎么训？**

不单独训。用EMA，teacher = 0.999 × old_teacher + 0.001 × student。便宜。

## 结果

RLBench 10个task，FLOPs减半：

- CogAct baseline: 57.2%
- MoLe (skip 50% layers): **60.8%** (+3.6%)

skip越多效果越好，直到skip 30层（95% FLOPs reduction）才崩。

INT8量化后 4.11 Hz，是原版CogAct的1.8x，但离Franka的50Hz还远。

## 直觉上的takeaway

skip掉一半layer反而涨点，这个现象很有意思。可能的解释：冗余层不只是"无害的空转"，它们可能在引入noise或over-smoothing。跳过它们相当于一种 **structured regularization**，让信号路径更干净。

这也呼应了Shallow Brain Hypothesis——大脑不是纯粹深的，而是深浅pathway动态切换。MoLe把每层当expert，用router选pathway，本质上是在做vertical方向的conditional computation。

## 诚实地说limitations

- 距离real-time control (50-1000Hz) 还差一个数量级
- 只测了single-step/short-horizon task，long-horizon的layer selection是否需要temporal consistency未验证
- 大模型scale up后收益递减（Small +2.7%, Large +1.5%）
- Router的out-of-distribution generalization没测

整体来说，这是一个well-executed的efficiency paper，idea不复杂但每个组件都有道理，实验也自洽。它的价值在于 **plug-in设计**——可以叠到任何VLA backbone上，不需要换架构。

---

# MoLe-VLA 深度技术解析

Andrej，这篇paper做的事情其实非常优雅。让我一层一层剥开来讲，帮你build up the intuition。

## 1. Motivation: VLA部署的核心矛盾

**问题本质**：把7B级别的VLA model (比如 OpenVLA, CogAct) 部署到 Franka 机器人上，RTX 4090 上推理频率只有 5–12 Hz，但 Franka 控制器需要 50–1000 Hz。这个 gap 是 10x–100x 量级，根本不实用。

paper的key observation在 Fig.1(A) 中：作者测量了 OpenVLA 在 RLBench 上 consecutive layer outputs 之间的 cosine similarity，发现 **相邻层之间相似度超过 90%**，但 **第一层和最后一层之间差异显著**。这暗示：
- LLM中间层存在大量冗余计算
- 简单的 early-exit (DeeR, SkipDecode 那一类) 会丢掉 deep layer 中关键的 semantic信息
- 需要的不是"提前退出"，而是"动态跳过"

## 2. 核心Idea: 从 Shallow Brain Hypothesis 到 Mixture-of-Layers

paper引用了 Suzuki et al. 2023 在 *Nature Reviews Neuroscience* 上的 **Shallow Brain Hypothesis (SBH)** [1]：人脑并不是纯粹深度hierarchical的，而是同时存在 deep hierarchical cortex 和 shallow parallel cortico-subcortical loops，根据认知需求动态切换信号路径。

这个神经科学启发转化为一个architecture设计原则：**把每一层 LLM 当作一个expert**，用一个router动态决定哪些layer被激活。这本质上是把 MoE 的 "horizontal expert-wise activation" 扩展为 "vertical layer-wise activation"。

### 与 MoD (Mixture-of-Depth) 的本质区别

Raposo et al. 2024 的 MoD [2] 是 token-level routing：每个token可以选择经过block还是走residual。问题是 **不同token经过的layer深度不同，导致perception inconsistency** —— 有些token的语义被深层加工，另一些被浅层加工，这在robotic action prediction里很危险，因为action chunks要求一致的semantic grounding。

MoLe是 **layer-level holistic skipping**：要么整层激活，要么整层跳过，所有token的深度处理保持一致。

公式 (7) 是核心：

$$\pmb{h}_k = G_k \cdot \pi_k(\pmb{h}_{k-1}) + (1 - G_k) \cdot \pmb{h}_{k-1}$$

变量解释：
- $\pmb{h}_k \in \mathbb{R}^{b \times n \times d}$：第 $k$ 层的hidden state，$b$ 是batch size，$n$ 是sequence length，$d$ 是hidden dim
- $G_k \in \{0, 1\}$：第 $k$ 层的binary gate，通过top-k选择得到
- $\pi_k(\cdot)$：第 $k$ 个Transformer layer的forward function
- 当 $G_k = 1$ 时正常计算这一层；$G_k = 0$ 时直接 residual pass-through

这个公式看着简单，但关键在于 **gate是sample-dependent的**：不同的robot state + language instruction会激活不同的layer组合，这就实现了SBH所说的"动态深度-并行平衡"。

## 3. STAR Router: 为什么spatial-temporal分开处理

paper的第二个贡献是 **Spatial-Temporal Aware Router (STAR)**。普通的MoE router就是一个linear layer + softmax，但在机器人任务里这远远不够。

### 问题诊断

机器人任务的input有两类异构信息：
- **Visual** $v_t \in \mathbb{R}^{b \times n_{img} \times d}$：spatial structure dominant
- **Textual** $l \in \mathbb{R}^{b \times n_{text} \times d}$：temporal/causal dependency dominant

一个简单的 `concat → linear` 会把这两类信息混在一起，丢失各自的inductive bias。

### STAR架构详解

Step 1: Modality Projection (公式8)
$$h_{img} = v_t \cdot \mathbf{W}_p, \quad h_{text} = l \cdot \mathbf{W}_p$$
其中 $\mathbf{W}_p \in \mathbb{R}^{d \times d_1}$ 是共享投影矩阵，把两个modality映射到同一个latent space。

Step 2: Spatial Routing (公式9) — 用一个2层MLP处理视觉特征：
$$\mathbf{S} = \mathbf{W}_s^{(2)} \cdot \varphi(\mathbf{W}_s^{(1)} \cdot h_{img} + \mathbf{b}_s^{(1)})$$
- $\mathbf{W}_s^{(1)} \in \mathbb{R}^{d_1 \times d_2'}$, $\mathbf{W}_s^{(2)} \in \mathbb{R}^{d_2' \times N_e}$
- $\varphi$ 是 GELU activation
- $\mathbf{S} \in \mathbb{R}^{b \times N_e}$：每个样本对 $N_e$ 个"layer-expert"的spatial routing score

Step 3: Temporal Routing (公式10) — 用Transformer + pooling处理文本：
$$\mathbf{T} = \mathbf{W}_t \cdot \Phi(\text{Transformer}(h_{text}))$$
这里用Transformer而不是MLP，因为language instruction有sequential dependency ("pick up the cup and pour into the bowl"这种)，需要self-attention建模。$\Phi$ 是average pooling把token维度压到1。

Step 4: Dynamic Temperature (公式11) — 这步很关键：
$$\alpha = \sigma(\mathbf{W}_\tau^\top \cdot h_{text}^{[CLS]} + b_\tau)$$
- $\sigma$ 是sigmoid
- $h_{text}^{[CLS]}$ 是文本的CLS token（语义压缩）
- $\alpha \in [0,1]$ 控制routing sharpness：instruction越复杂，$\alpha$越大，routing越deterministic；简单instruction则更soft

Step 5: Gumbel-Softmax融合 (公式12)：
$$\mathbf{G} = \tau(\alpha \cdot (\mathbf{S} + \mathbf{T}), \tau=1.0)$$
用Gumbel-Softmax让离散选择可微分，这是标准技巧 [3]。

### 计算复杂度对比

paper强调STAR的FLOPs是 $\mathcal{O}(N_e(d_2' + N_{text}^2))$ vs 标准MoE的 $\mathcal{O}(N_e \cdot d)$，因为 $d \gg N_{text}, d_2'$（hidden dim远大于text长度和spatial MLP中间维度）。所以STAR几乎是免费的。

## 4. CogKD: 解决layer-skipping的认知塌缩

这是paper最subtle的部分。Skip掉一些layer必然会损失cognitive expressiveness，但朴素的token-wise mimic distillation（公式13）：
$$\mathcal{L}_{\text{mimic}} = \frac{1}{N} \|\pmb{f}^{(t)} - \mu(\pmb{f}^{(s)})\|_2^2$$
对所有token一视同仁，这是错的——因为visual token和task-relevant的token重要性不同。

### Cognition Token的设计

借鉴 CogAct [4]，引入一个 **learnable cognition token** $e_t^c \in \mathbb{R}^{1 \times d}$，插入到input sequence最底层。它的作用是 **聚合vision和language信息**，形成一个"任务理解"的表示。

Teacher model有 $e^{c,(t)}$，Student model有 $e^{c,(s)}$，各自独立学习。

### Tokens of Interest (ToIs) 提取 (公式14)

$$M^{(i)} = \eta(e^{c,(i)} \cdot \pmb{f}^{(s)T}), \quad i \in \{s, t\}$$
- $e^{c,(i)} \in \mathbb{R}^{1 \times d}$ 是cognition token
- $\pmb{f}^{(s)} \in \mathbb{R}^{n \times d}$ 是student的输出features
- $e^{c,(i)} \cdot \pmb{f}^{(s)T} \in \mathbb{R}^{1 \times n}$ 是cognition token和每个student token的相似度
- $\eta$ 是sigmoid，得到attention mask $M^{(i)} \in \mathbb{R}^{1 \times n}$

然后取intersection：$M = M^{(t)} \odot M^{(s)}$，要求"teacher和student都认为是重要的token"才被distill。

### 加权mimic loss (公式15)

$$\mathcal{L}_{\text{cog-mimic}} = \frac{1}{N} \|M \odot \pmb{f}^{(t)} - \mu(M \odot \pmb{f}^{(s)})\|_2^2$$
- $\odot$ 是element-wise multiplication（broadcasting到$d$维）
- $\mu$ 是linear projection对齐student到teacher维度

### Reverse-KL (公式16)

$$\mathcal{L}_{\text{cog-reversekl}} = (M \odot \pmb{f}^{(s)}) \log\left(\frac{M \odot \pmb{f}^{(s)}}{M \odot \pmb{f}^{(t)}}\right)$$

为什么用Reverse-KL而不是forward KL？参考 MiniLLM [5] 的工作：forward KL会让student在teacher低概率区域过发散，而reverse KL迫使student的support严格落在teacher的support内，更适合distillation场景。

### 最终CogKD loss (公式17)

$$\mathcal{L}_{\text{cog}} = (1-\lambda_1)\mathcal{L}_{\text{cog-mimic}} + \lambda_1 \mathcal{L}_{\text{cog-reversekl}}$$
$\lambda_1 = 0.5$ 平衡两项。

### Teacher更新策略 (公式18)

用EMA更新teacher：
$$\pi_t^{(t)} = \alpha \cdot \pi_{t-1}^{(t)} + (1-\alpha) \cdot \pi_t^{(s)}$$
$\alpha = 0.999$，参考 Mean Teacher [6]。这避免了teacher需要单独训练的成本。

## 5. 总训练目标 (公式19)

$$\mathcal{L}_{\text{MoLe}} = \mathcal{L}_{\text{task}} + \lambda_2 \mathcal{L}_{\text{cog}} + \lambda_3 \mathcal{L}_{\text{lb}}$$

- $\mathcal{L}_{\text{task}}$：action prediction MSE (公式6，diffusion head的noise prediction)
- $\mathcal{L}_{\text{cog}}$：CogKD distillation
- $\mathcal{L}_{\text{lb}}$：layer-level load balance loss（公式3的扩展，防止router总是跳同一组layer）
- $\lambda_2 = 0.5, \lambda_3 = 0.1$

## 6. 实验数据深度分析

### Table 1: RLBench 10 tasks 主结果

| Method | Mean Acc | FLOPs (G) | 备注 |
|--------|----------|-----------|------|
| OpenVLA | 45.4% | 1930.0 | baseline |
| CogAct | 57.2% | 1935.8 | diffusion head |
| RoboMamba | 43.6% | 826.3 | Mamba替代transformer |
| Random-skip-CogAct | 51.2% (-6.0%) | 984.3 | 随机skip |
| MoD-CogAct | 56.4% (-0.8%) | 985.8 | token-wise routing |
| DeeR-CogAct | 59.2% (+2.0%) | 997.4 | early-exit |
| **MoLe-OpenVLA** | **55.6% (+10.2%)** | **981.5** | layer-skip + CogKD |
| **MoLe-CogAct** | **60.8% (+3.6%)** | **985.8** | layer-skip + CogKD |

Key observations:
- MoLe相对CogAct 在FLOPs减半的情况下 **success rate反而提升3.6%** —— 这是反直觉的，skip掉一半layer反而更好
- Random-skip直接掉6%，说明router的选择性是必要的
- MoD只掉0.8%，但MoLe是正向的，说明layer-level holistic > token-level routing
- MoLe-OpenVLA提升10.2%非常显著，说明weak base model反而能从distillation中获益更多

### Table 2: 推理时间分析

| Method | Inference time | FLOPs | Mean |
|--------|---------------|-------|------|
| CogAct | 0.434 s | 1935.8 G | 57.2% |
| DeeR | 0.337 s | 997.4 G | 59.2% |
| MoLe | **0.309 s** | 985.8 G | **60.8%** |

推理时间从0.434s降到0.309s，约30%加速，但FLOPs减半没换来2x加速，说明存在memory bandwidth瓶颈——layer-skip减少了compute但attention的KV cache和token routing overhead还在。

### Table 3: 量化效果

| Method | Precision | Freq | Memory | Mean |
|--------|-----------|------|--------|------|
| CogAct | FP16 | 2.30 Hz | 16055 MB | 57.2% |
| MoLe | INT8 | **4.11 Hz** | **8887 MB** | 58.8% |

INT8量化后MoLe达到4.11Hz，约CogAct的1.8x，但仍未达到Franka的50Hz。这表明 **仅靠layer-skip还不足以做real-time control**，需要配合action chunking或者asynchronous policy execution。

### Table 4: Scalability

| Model Size | CogAct | MoLe | Delta |
|------------|--------|------|-------|
| Small | 47.2% | 49.9% | +2.7% |
| Base | 57.2% | 60.8% | +3.6% |
| Large | 70.0% | 71.5% | +1.5% |

Scale越大，improvement越小（+2.7 → +3.6 → +1.5%），这说明大模型本身的冗余度更低，layer-skip的收益空间在缩小。

### Table 5: Ablation Study

| Exp | STAR | Cognition | MSE | KL | Reserve-KL | Mean |
|-----|------|-----------|-----|-----|-----------|------|
| Ex0 (baseline) | × | × | × | × | × | 57.2% |
| Ex1-1 | × | × | ✓ | × | × | 56.3% |
| Ex1-2 | × | × | × | ✓ | × | 54.8% |
| Ex2-1 | ✓ | ✓ | × | × | × | 58.3% |
| Ex2-2 | ✓ | ✓ | ✓ | × | × | 57.7% |
| Ex2-3 | ✓ | ✓ | × | × | ✓ | 59.4% |
| **Ex2-4** | ✓ | ✓ | ✓ | × | ✓ | **60.8%** |

观察：
- 纯token-wise mimic (Ex1-1) 反而掉到56.3%，证明uniform distillation有害
- 加STAR+cognition token (Ex2-1) 立刻提升1.1%
- Reserve-KL (Ex2-3, 59.4%) 远好于普通KL (Ex1-2, 54.8%)，验证MiniLLM的insight
- MSE+Reserve-KL组合 (Ex2-4) 达到最优，说明token reconstruction和distribution matching互补

### Table 10: Skip层数分析（附录E）

| Skip Layers | FLOPs reduction | Mean |
|-------------|-----------------|------|
| 2 | ~6% | 65.2% |
| 6 | ~19% | 63.2% |
| 8 | ~25% | 61.6% |
| 12 | ~38% | 62.4% |
| 20 | ~63% | 55.2% |
| 24 | ~75% | 53.2% |
| 26 | ~81% | 54.0% |
| 30 | ~95% | 38.4% |

Skip 2 layers时反而 **超过full model 5%**，这是个非常有趣的现象——可能是因为去掉冗余layer起到了regularization作用，类似Dropout。Skip到24层还能保持53.2%，超过原始CogAct的57.2%只差一点；但skip到30层（95%FLOPs reduction）就崩了。

## 7. 真实世界实验 (Table 6)

| Method | Detach charger | Pull drawer | Pour water | Mean |
|--------|---------------|-------------|------------|------|
| CogAct | 60.0% | 60.0% | 80.0% | 66.7% |
| MoLe | 70.0% | 60.0% | 80.0% | 70.0% |

Pour water这种需要精确3D rotation的任务，MoLe保持80% success，证明layer-skip没有破坏精细spatial reasoning。

## 8. Failure Cases 分析 (附录G)

作者诚实地列了4类失败：
1. **Loss of control**: 物体重量变化、gripper打滑
2. **Rotational prediction errors**: pour water任务里的累积误差
3. **Pose超出物理极限**: Franka机械臂workspace constraint
4. **Workspace unreachable**: detach charger任务

这暗示layer-skip主要损害的是 **fine-grained motor control**而非high-level cognition。

## 9. 我的几点思考和联想

### 9.1 与 LayerSkip / DepthAdaptive Inference 的关系

LayerSkip [7] 在LLM推理时用layer-wise early-exit + shared policy，但只用于language modeling。MoLe的差异化在于把layer-skip搬到多模态+action prediction场景，并且引入了STAR这种spatial-temporal router——这是必要的，因为robotic task的input不是纯token序列，而是异构的vision+language。

### 9.2 与 Conditional Computation in Robotics 的联系

最近一两年robotics领域涌现了类似工作：
- **RoboMamba** [8]：换架构（Mamba替代Transformer）
- **DeeR-VLA** [9]：multi-exit early termination
- **π0** (Physical Intelligence) [10]：用flow matching + smaller VLM

MoLe走的是另一种路径：保留Transformer架构的compatibility，但让inference path变浅。这种"plug-in"设计使得它可以叠加到任何VLA backbone上——这是它的工程价值。

### 9.3 关于 SBH 的科学性

Shallow Brain Hypothesis [1] 在neuroscience还有争议。人脑的cortico-subcortical loops (比如 basal ganglia → thalamus → cortex) 确实提供shortcut，但这些loop的功能是motor gating和reward-driven action selection，并不是high-level cognition。MoLe借用这个比喻做architectural inspiration是合理的，但要小心：**neuroscience-inspired ≠ neuroscience-validated**。

### 9.4 仍未解决的问题

1. **Router的ood generalization**：STAR在训练任务上work，但unseen task上router的选择是否还合理？paper没测试task generalization。
2. **Action chunking的兼容性**：CogAct预测的是action chunks（多个未来step），MoLe的layer-skip是基于当前observation的，但action chunk的执行是开环的，这可能与layer-skip的adaptive特性冲突。
3. **Long-horizon tasks**：所有实验都是single-step或short-horizon，长期task的layer选择是否需要temporal consistency约束？

### 9.5 与最近的VLA scaling trend的对比

最近的VLA发展倾向于 **bigger model + more data**（π0 3B, Open-X-Embodiment 55B dataset），而MoLe走的是 **same model + dynamic compute**。这两种哲学其实是互补的：理论上可以用MoLe加速π0这种large VLA，paper Table 4的scalability实验（CogAct-Large +1.5%）暗示大模型上收益会减少，但这是在7B规模，到了3B+可能需要重新设计router。

### 9.6 对训练cost的思考

paper训练用8×A800，1.5小时跑1k iterations。这个训练成本相对低，因为：
1. 基于pretrained weights finetune
2. RLBench的100 trajectories/task × 10 tasks = 1000 trajectories，数据量小
3. FSDP + constant LR = 2e-5

但EMA teacher的额外forward pass相当于训student两倍的forward cost，这是hidden cost。在更大规模训练时这个开销需要考虑。

## 10. 总结：MoLe-VLA的intuition

一句话总结：**让VLA model像大脑一样，简单任务走浅路径，复杂任务走深路径，并且用teacher的"完整思考过程"指导student的"走捷径"**。

三个key insight:
1. **Layer-as-Expert**：把vertical depth做成horizontal MoE，通过router实现sample-adaptive depth
2. **Spatial-Temporal Routing**：vision和language用不同的inductive bias建模，再融合成layer selection
3. **Cognitive Distillation**：用cognition token识别task-critical tokens，做asymmetric distillation，避免简单token mimic导致的"cognitive collapse"

这个工作的真正贡献不在单点创新，而在于把多个idea (MoE + dynamic depth + knowledge distillation + spatial-temporal routing) 有机组合成一个可plug-in的VLA efficiency framework，并且在simulation和real-world都验证了consistent improvement。

## References (Web Links)

- [1] Shallow Brain Hypothesis (Suzuki et al., Nature Reviews Neuroscience 2023): https://www.nature.com/articles/s41583-023-00732-x  
- [2] Mixture-of-Depths (Raposo et al., 2024): https://arxiv.org/abs/2404.02258  
- [3] Gumbel-Softmax (Jang et al., 2017): https://arxiv.org/abs/1611.01144  
- [4] CogAct (Li et al., 2024): https://arxiv.org/abs/2411.19650  
- [5] MiniLLM (Gu et al., 2023): https://arxiv.org/abs/2306.08543  
- [6] Mean Teacher (Tarvainen & Valpola, 2017): https://arxiv.org/abs/1703.01780  
- [7] LayerSkip (Elhoushi et al., 2024): https://arxiv.org/abs/2404.16710  
- [8] RoboMamba (Liu et al., NeurIPS 2024): https://arxiv.org/abs/2406.04639  
- [9] DeeR-VLA (Yue et al., NeurIPS 2024): https://arxiv.org/abs/2411.04865  
- [10] π0 (Physical Intelligence, 2024): https://www.physicalintelligence.company/blog/pi0  
- [11] OpenVLA (Kim et al., CoRL 2024): https://openvla.github.io/  
- [12] RLBench (James et al., RAL 2020): https://sites.google.com/view/rlbench  
- [13] MoE-LLaVA (Lin et al., 2024): https://github.com/PKU-YuanGroup/MoE-LLaVA  
- [14] Shazeer et al. Sparsely-Gated MoE (2017): https://arxiv.org/abs/1701.06538  
- [15] DeepSeek-MoE (Dai et al., 2024): https://arxiv.org/abs/2401.06066  
- [16] Project page (MoLe-VLA): https://MoLe-VLA-Web (链接在paper中标注，具体URL未提供，建议搜索arXiv ID)  

如果这篇paper最终能挂到arXiv，搜 "MoLe-VLA" 应该能直接找到 project page 和代码。从作者列表看，这是Peking University + NJU + HK PolyU的合作工作，通讯作者是Shanghang Zhang，她的实验室近期在VLA efficiency方向很活跃 (RoboMamba也是她组的)。
