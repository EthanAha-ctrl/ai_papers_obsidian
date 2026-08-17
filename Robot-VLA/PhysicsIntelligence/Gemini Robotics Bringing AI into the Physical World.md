---
source_pdf: Gemini Robotics Bringing AI into the Physical World.pdf
paper_sha256: f2c1ab4e73a76013b9ebf5a4ff90bb3d291eee0730304d3ebf4531256a204b78
processed_at: '2026-08-04T13:09:13-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Gemini Robotics

Hey Andrej，好的我用更口语化的方式再讲一遍，但该有的技术细节还是给你保留。

## 一句话总结

Google 把 Gemini 2.0 这个 huge VLM fine-tune 了一下，让它既能"看懂"物理世界（3D、trajectory、affordance），又能直接输出 robot action 控制机械臂，做到 50Hz 控制 + generalist + cross-embodiment。核心 thesis 很简单：**robotics 的 scaling law 跟 LLM 一样，大模型 + 多样数据 = 强 generalist robot**。

## 这篇 paper 在解决什么问题

Robotics 过去几十年的痛点：每个 task 要单独训一个 policy，换个场景就崩。RT-2、OpenVLA、π0 这些 VLA 模型想搞 generalist，但 backbone 太小（PaliGemma 3B），language understanding 不够，generalization 很拉胯。

Google 的赌注：**与其搞 specialist，不如直接把 frontier VLM (Gemini 2.0, 估计几百B参数) 拿来 ground 到物理世界**。问题是 Gemini 本身只会输出 text，不会输出 robot action，latency 也太高（秒级，robot 要 ms 级）。

## 他们的方案：两层架构

### 第一层：Gemini Robotics-ER（让 Gemini "看懂"世界）

这个 model 不输出 action，只输出"理解信号"：
- **2D pointing**：给个语言描述，输出 `[y, x]` 坐标（归一化到 0-1000）
- **2D bounding box**：`[y_0, x_0, y_1, x_1]`
- **Top-down grasp**：`[y, x, θ]`，θ 是 gripper 旋转角度 [-90, 90]度
- **3D bounding box**：`[x, y, z, w, h, l, r_1, r_2, r_3]`，其中 r_1/r_2/r_3 是 Euler angles

这里有个很 hack 的 trick：**Euler angles 用 text token 表示**，截断到 2 位小数。为什么不搞个 regression head？因为 VLM 的 text head 训练得最充分，用语言"念出"数字反而比专门加个数值输出头更准。这跟 RT-2 用 token 表示 action 是一个思路。

关键数据（Table 3, pointing benchmark）：
- Paco-LVIS：Gemini Robotics-ER **71.3** vs Molmo 72B 47.1 vs GPT-4o 16.2
- Pixmo-Point：Gemini Robotics-ER **49.5** vs Molmo 72B 12.5

这个差距巨大。说明 ER 训练阶段往 Gemini 里灌了大量 pointing/spatial 数据，把这部分能力专门 boost 了。

3D detection（Table 4, SUN-RGBD AP@15）：
- Gemini Robotics-ER: **48.3**（新 SOTA）
- ImVoxelNet (expert model): 43.7
- Gemini 2.0 Flash: 30.7

而且 Gemini 支持 open-vocabulary，expert model 是 closed-set。这是 VLM 范式对 specialist 的降维打击。

### 第二层：Gemini Robotics（让 Gemini "会动"）

这部分是 VLA model，直接输出 robot action。架构上有个很关键的设计：

**Cloud-Edge 双组件**：
- **Backbone（云端）**：distilled Robotics-ER，latency 压到 <160ms
- **Decoder（robot 本地）**：补偿 backbone latency
- 端到端 ≈ 250ms，配合 action chunking 做到 **50Hz effective control**

为什么这么搞？因为几百 B 的模型不可能 onboard 跑。Cloud 推理 latency 不稳定（网络抖动），local decoder 起到"预测 + 平滑"作用。Action chunking 就是每次预测未来 K 步 action，执行前几个再 re-plan。

形式化一下，给定 observation history $(o_{t-H}, ..., o_t)$ 和 instruction $\ell$：

$$\hat{a}_{t:t+K} = f_\theta(o_{t-H:t}, \ell)$$

- $o_t$: 时刻 t 的 observation（多视角图像 + robot state）
- $\ell$: language instruction
- $H$: history length
- $K$: chunk size
- $f_\theta$: VLA policy

执行用 receding horizon，每步执行 chunk 第一个 action 或前几个。50Hz 就是这么来的——不是真的每 20ms 推理一次，是 chunk + local decoder 的组合。

这个架构 pattern 我觉得会成为未来大 VLA 部署的标准。RT-2 是 1-3Hz，π0 大概 5Hz，这里 50Hz，差了一个数量级。

## 训练数据

- **数千小时** ALOHA 2 teleoperation 数据（12 个月采集）
- **数千个** diverse tasks
- 混了 web documents、code、multimodal content、ER/VQA 数据

这个 mixture 很关键。Section 4.1 有个 ablation：用同样架构、同样数据量，**从 scratch 训 specialist → 0% success**。但从 generalist checkpoint fine-tune → 能学 origami。

这说明 robot action data 训出来的 representation 里有 transferable 的 "physical common sense"，不仅仅是 task-specific skill。跟 LLM 的 in-context learning 类似——pre-training 学到的是通用 prior，fine-tune 只是激活。

## 几个让我"啊哈"的实验

### 1. CoT 让 Flash 超过 Pro（Table 2）

| Prompt | Gemini 2.0 Flash | Gemini 2.0 Pro |
|--------|------------------|----------------|
| w/o CoT | 46.3 | 48.3 |
| w/ CoT | **50.3** | 54.8 |

Flash + CoT (50.3) > Pro w/o CoT (48.3)。

Insight：VLM 内部其实"知道"答案，但直接预测会被 shortcut learning 带偏。CoT 强制模型先 grounding 到 visual evidence 再下结论。这跟纯 text LLM 的 CoT 机制类似，但这里 grounding 到的是空间概念。

### 2. Multilingual 上 baseline 全崩（Fig. 21）

85 个 generalization tasks，其中 instruction generalization 包含 multilingual（Spanish）。

结果：**π0 re-implement 和 diffusion baseline 在 Spanish instruction 上 0% success**，Gemini Robotics 仍有 non-zero performance。

为什么？PaliGemma (3B) 的 language encoder 太弱，根本不懂 Spanish。Gemini 2.0 是多语言训练的，天然支持。

这直接证明了 **VLM backbone 规模 = robotics generalization 能力**。Robotics 不再是"小模型 + 大数据"的游戏。

### 3. Origami Fox（Section 4.1）

这个任务要 4 次精确折叠，每次要 align/bend/pinch/crease，paper 层数递增，bi-arm coordination 要求极高，一个小错就 irrecoverable。

Gemini Robotics specialist 解决了，baseline 全崩。

关键：用 2000-5000 episodes fine-tune generalist checkpoint。从 scratch 训同架构 → 0%。

这是 foundation model 范式在 robotics 上的最强证据。跟 LLM 的 few-shot learning 一个故事：pre-training 学通用 representation，fine-tune 激活特定能力。

### 4. Cross-Embodiment（Section 4.4）

训练数据全是 ALOHA 2，fine-tune 到：
- **Bi-arm Franka**（不同 form factor）
- **Apollo humanoid**（Apptronik，5-fingered dexterous hands，完全不同的 action space）

在 Franka 上不仅 in-distribution 任务能做，**visual & action generalization 也显著超过 single-task diffusion baseline**。

这说明 Gemini 学到的 representation 是 embodiment-agnostic 的，generalization 能力可以 transfer。

参考 Apollo: https://www.apptronik.com/

### 5. Reasoning-Enhanced Variant（Section 4.2）

Vanilla VLA 在 OOD reasoning 任务上掉点严重。比如 "sort the bottom right mouse into the matching pile"——要按颜色分类，但 training data 里没有 sorting 任务。

他们的解法：用 re-labeled action dataset fine-tune，让 action prediction 中间对齐 trajectory understanding 表示。Local decoder 扩展为把 reasoning intermediates 转成 continuous actions。

效果（Fig. 24）：
- One-step Reasoning：vanilla 60% → enhanced 90%
- Semantic Generalization：vanilla 50% → enhanced 80%
- Spatial Understanding：vanilla 40% → enhanced 70%

而且输出可解释的 keypoint trajectories（Fig. 25），相当于 spatial chain-of-thought。

Insight：VLM 的 reasoning 能力不会自动传导到 action，需要专门的 alignment training。这是未来工作的关键方向。

## Zero-shot vs Few-shot 的对比

Table 5 & 6 很有说服力：

**Simulation（Table 5）**：
- Zero-shot avg: 53%（Robotics-ER）
- ICL avg: 65%（Robotics-ER + 10 demos）

**Real world（Table 6）**：
- Zero-shot avg: 25%
- ICL avg: 65%

ICL 在 dexterous 任务上提升巨大（Fold Dress: 0% → 56%），但在简单任务上提升有限。

Insight：**demonstrations 主要帮助的是"精细动作"而非"语义理解"**。语义理解来自 VLM pre-training，精细动作来自 demonstration。这两个维度是 orthogonal 的。

## Fast Adaptation（Section 4.3）

8 个 subtasks，用 5/20/100 episodes fine-tune：
- 7/8 任务在 ≤100 demos 下达到 >70% success
- 2 个任务 100%
- "Origami first fold" 和 lunch-box 上 baseline 在 100 demos 时仍很低

100 demos ≈ 15 分钟到 1 小时 teleoperation。这是 foundation model 的核心价值——**少量数据快速学新任务**。

但离 human-level（1-3 demos）还远。这是未来要攻克的。

## Safety 部分

这块 paper 讲得比较轻，但思路值得注意。

三层安全：
1. **Content safety**：继承 Gemini 的 safety training
2. **Pointing safety**：bias-inducing queries rejection rate 20% → 96%（通过 SFT）
3. **Semantic action safety**：ASIMOV benchmark 评估

ASIMOV 分两个子集：
- **ASIMOV-Multimodal**：visual safety QA
- **ASIMOV-Injury**：基于 NEISS 真实伤害记录

用 Constitutional AI 方法 post-train。对抗 prompt 下性能也能保持。

但 semantic action safety 是个 long tail 问题。"不能把毛绒玩具放热炉子上"这种常识有无限多个。Constitutional AI 在 text domain work，但物理世界 consequences 更难 enumerate。

参考：
- ASIMOV: https://arxiv.org/abs/2503.08663
- Constitutional AI: https://arxiv.org/abs/2212.08073

## 跟其他 VLA 对比

| 维度 | RT-2 | OpenVLA | π0 | Gemini Robotics |
|------|------|---------|-----|-----------------|
| Backbone | PaLI-X (55B) | PaliGemma (3B) | PaliGemma (3B) | Gemini 2.0 (~?) |
| Action freq | 1-3 Hz | ~5 Hz | ~5 Hz | **50 Hz** |
| Architecture | VLM token | VLM token | Flow matching | Cloud VLA + local decoder |
| Cross-embodiment | limited | limited | limited | ✓ (Franka, Apollo) |
| Long-horizon | ✗ | ✗ | limited | ✓ (origami, lunch-box) |
| ER capabilities | ✗ | ✗ | ✗ | ✓ (pointing, 3D, trajectory) |

参考：
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- ALOHA Unleashed: https://proceedings.mlr.press/v270/zhao25b.html

## 我的几点思考

### 1. Robotics scaling law = LLM scaling law

最 striking 的发现：PaliGemma (3B) 在 multilingual 上 0%，Gemini (~数百B) non-zero。这跟 LLM scaling 完全一致。Robotics 不再是"小模型 + 大数据"的游戏，而是"大模型 + 多样数据"的游戏。

### 2. Cloud-Edge 是务实选择

直接把 100B+ 模型部署到 robot 上不现实。Google 选择 cloud backbone + local decoder，用 action chunking 弥补 latency。这个 pattern 会成为未来 VLA 部署的标准。

### 3. Physical Common Sense 是 Emergent 的

From-scratch ablation 极有说服力：同架构、同数据量，从 generalist checkpoint fine-tune 能学 origami，从 scratch 训 0%。这说明 robot action data 训出来的 representation 里有 transferable 的 physical understanding。

### 4. Reasoning-Action Gap 仍存在

Vanilla VLA 在 OOD reasoning 上掉点严重。即使有 Gemini backbone，把 reasoning 传导到 action 仍需专门 alignment training。这是未来关键方向。

### 5. 数据效率离 human-level 还远

100 demos 学新任务虽然好，但人 1-3 demos 就行。这个 gap 意味着 sample efficiency 还有很大提升空间，可能要靠 simulation 或者 better representation learning。

## Limitations

Paper 自己承认：
1. 长 video 上 spatial grounding 仍弱
2. 数值预测精度不够 fine-grained control
3. Multi-step reasoning + precise dexterity 结合仍难
4. Cross-embodiment 还要 fine-tuning，没实现 zero-shot transfer

我额外几点：
- **Contact-rich tasks** 仍是难点（Fold Dress zero-shot 0%）
- **Evaluation 规模**：20 trials per task 统计 power 有限
- **Sim-to-real** 没充分探索，paper 提到未来要 lean on simulation

## 总结

这篇 paper 的核心贡献：把 frontier VLM 的 reasoning 能力成功 ground 到物理世界，通过 ER 增强 + VLA fine-tuning + specialization recipe + cross-embodiment transfer，把 generalist robot 推到了新高度。

关键 take-away：**robotics 的 scaling law 跟 LLM 一致——更大的 VLM backbone + 更多样的数据 = 更强的 generalist robot**。

对未来工作的启示：
- VLA 应该 build 在 frontier VLM 上
- Cloud-edge 架构是部署大 VLA 的可行路径
- Generalist pre-training + specialist fine-tuning 范式在 robotics 上 work
- Reasoning 与 action 的对齐仍是 open problem

主要参考链接：
- ERQA: https://github.com/embodiedreasoning/ERQA
- ALOHA 2: https://arxiv.org/abs/2405.02292
- ASIMOV: https://arxiv.org/abs/2503.08663
- π0: https://arxiv.org/abs/2410.24164
- ALOHA Unleashed: https://proceedings.mlr.press/v270/zhao25b.html
- Di Palo & Johns (keypoint action tokens): https://arxiv.org/abs/2403.19578
- Code as Policies: https://arxiv.org/abs/2209.07753
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- Apollo robot: https://www.apptronik.com/
- Constitutional AI: https://arxiv.org/abs/2212.08073

希望这个版本更 readable 但 still technically meaty。如果想 deep dive 某个具体部分（比如 local decoder 怎么设计的、CoT 在 spatial reasoning 上的具体机制、或者 cross-embodiment transfer 的 loss function），随时聊。

---

# Gemini Robotics: Bringing AI into the Physical World 深度技术解读

Hey Andrej! 这篇 paper 是 Google DeepMind 在 2025 年初发布的一个重量级 robotics foundation model 工作。核心 thesis 是：把 frontier VLM (Gemini 2.0) 的 multimodal reasoning 能力 ground 到物理世界，让 robot 既能"看懂"世界又能"动"得精准。下面我从技术细节角度拆解。

## 1. 整体架构与设计哲学

### 1.1 两层模型结构

Gemini Robotics 家族包含两个层次模型：

**Gemini Robotics-ER (Embodied Reasoning)**:
- 本质是 Gemini 2.0 Flash 的 enhanced 版本
- 输出模态扩展到 pointing、trajectory、3D bounding box、grasp pose
- 无需 robot action 数据即可使用（zero-shot / few-shot）

**Gemini Robotics (VLA)**:
- 在 Robotics-ER 基础上 fine-tune，直接输出 robot action
- 双组件架构：
  - **Gemini Robotics backbone**（云端）：distilled Robotics-ER，query-to-response latency 从秒级压缩到 **<160ms**
  - **Gemini Robotics decoder**（robot 本地）：补偿 backbone latency
- 端到端 latency ≈ 250ms，配合 action chunking（Zhao et al., 2023, ACT）实现 **50Hz effective control frequency**

这里有一个关键的工程 insight：backbone 在云端跑导致 latency 不稳定，local decoder 起到了"预测+平滑"的作用。这种 cloud-edge 分离架构让我想起 RT-2 的思路但更进一步——RT-2 是 1-3Hz，这里做到 50Hz，差了一个数量级。

参考链接：
- ACT paper: https://tonyzhaozh.github.io/aloha/
- π0 paper: https://arxiv.org/abs/2410.24164

### 1.2 训练数据组成

Gemini Robotics 的训练 mixture 包括：
- **数千小时** ALOHA 2 teleoperation 数据（12个月采集）
- **数千个** diverse tasks
- 非 action 数据：web documents, code, multimodal content (image/audio/video)
- ER 和 VQA 数据

这种 mixture 很关键——单纯 robot action 数据训出来的 specialist 在 OOD 上崩（见 Section 4.1 的 from-scratch 实验，0% success），而 web data 提供 "physical common sense" 的 prior。

## 2. ERQA Benchmark 与 Embodied Reasoning

### 2.1 ERQA 设计

ERQA 包含 400 道 multiple-choice VQA 题，覆盖 7 个类别：
- spatial reasoning
- trajectory reasoning
- action reasoning
- state estimation
- pointing
- multi-view reasoning
- task reasoning

28% 的题包含多张图片（需要 cross-frame correspondence）。

### 2.2 关键实验数据

Table 1 显示 Gemini 2.0 Pro Experimental 在 ERQA 上达到 **48.3%**，超过 GPT-4o (47.0%) 和 Claude 3.5 Sonnet (35.5%)。

Table 2 的 Chain-of-Thought (CoT) 实验非常有意思：

| Prompt | Gemini 2.0 Flash | Gemini 2.0 Pro Exp | GPT-4o | Claude 3.5 |
|--------|------------------|--------------------|--------|-------------|
| w/o CoT | 46.3 | 48.3 | 47.0 | 35.5 |
| w/ CoT | 50.3 | 54.8 | 50.5 | 45.8 |

**Gemini 2.0 Flash + CoT (50.3) > Gemini 2.0 Pro Exp w/o CoT (48.3)**

这个结果对 build intuition 很重要：VLM 内部其实"知道"答案，但没有显式 reasoning trace 时会被 shortcut learning 带偏。CoT 强制模型先 grounding 到 visual evidence 再下结论。

## 3. Embodied Reasoning 能力详解

### 3.1 2D 输出表示约定

**2D Bounding Box**: 表示为 `[y_0, x_0, y_1, x_1]`
- y: 垂直轴
- x: 水平轴  
- (y_0, x_0): 左上角
- (y_1, x_1): 右下角
- 坐标范围归一化到 [0, 1000] 整数

**2D Point**: 表示为 `[y, x]` tuple

**Top-down Grasp**: 表示为 `[y, x, θ]`
- θ: rotation angle，整数度数范围 [-90, 90]
- θ=0 时 gripper fingers 与水平轴对齐

**3D Bounding Box**: 表示为 `[x, y, z, w, h, l, r_1, r_2, r_3]`
- (x, y, z): 中心坐标
- w, h, l: width/height/length
- r_1, r_2, r_3: Euler angles，截断到 2 位小数的 text tokens

把 Euler angle 用 text token 表示这个 trick 很有意思——避免了 regression head 的精度问题，让 VLM 用语言"念出"角度。

### 3.2 Pointing 性能对比 (Table 3)

| Benchmark | Robotics-ER | 2.0 Flash | Molmo 72B |
|-----------|-------------|-----------|-----------|
| Paco-LVIS | **71.3** | 46.1 | 47.1 |
| Pixmo-Point | **49.5** | 25.8 | 12.5 |
| Where2Place | 45.0 | 33.8 | 63.8 |

注意 Pixmo-Point 上 Robotics-ER (49.5) 几乎是 Molmo 72B (12.5) 的 4 倍。这表明在 Robotics-ER 训练阶段对 pointing 能力的增强非常显著。

### 3.3 3D Detection (Table 4)

在 SUN-RGBD 上：
- Gemini Robotics-ER: AP@15 = **48.3**（新 SOTA）
- ImVoxelNet (expert): 43.7
- Gemini 2.0 Flash: 30.7

而且 Gemini 支持 open-vocabulary，而 baseline 是 closed-set。这是 VLM 范式相对于 specialist model 的根本优势。

## 4. Zero-shot & Few-shot Robot Control

### 4.1 Zero-shot via Code Generation

Gemini 2.0 通过一个 robot API 实现零样本控制：
- API 提供 `move_gripper_to`, `open/close_gripper`, `detect_objects`, `get_grasp_position_and_euler_orientation` 等
- Gemini 2.0 自己做 perception（不调用外部模型）
- Iterative loop：observe → generate code → execute → get feedback → replan

Table 5 仿真结果：
- Gemini 2.0 Flash: avg 27%
- **Gemini Robotics-ER: avg 53%**（几乎翻倍）

Table 6 真实世界：
- Zero-shot avg 25%（Fold Dress 0%）
- ICL avg 65%（Fold Dress 56%）

ICL 在 dexterous 任务上提升巨大，说明 demonstrations 主要帮助的是"精细动作"而非"语义理解"。

### 4.2 Few-shot via In-Context Learning

借鉴 Di Palo & Johns (2024) 的 keypoint action tokens 方法：
- 把 N 条 teleoperated trajectories 转成 object list + end-effector poses
- tokenize 成文本加入 prompt
- 关键改进：用 Robotics-ER 自己提取 keypoints，不依赖外部模型

参考：https://arxiv.org/abs/2403.19578

## 5. Gemini Robotics VLA 模型核心实验

### 5.1 Out-of-the-box 多任务性能 (Fig. 16)

20 个 short-horizon dexterous tasks 评测：
- Gemini Robotics 在一半任务上 >80% success
- 在 deformable object manipulation（fold cloth, wrap wire）上显著优于 baseline
- "wrap the wire around the headphone" 任务上，**Gemini Robotics 是唯一非零 success 的方法**

Baseline 对比：
- **π0 re-implement**: 用 PaliGemma + diffusion transformer action expert，在 Google 自己的数据 mixture 上训练，比原版 openpi checkpoint 还强
- **Multi-task diffusion**: ALOHA Unleashed 改造版 + CLIP text encoder

### 5.2 Language Instruction Following (Fig. 17)

25 个 instructions × 5 scenes。关键发现：
- π0 re-implement 在 in-distribution 场景上还行，但在 **descriptive language attributes**（"top black container", "blue clip"）和 unseen objects 上崩
- 这暴露了 PaliGemma 这种小 VLM backbone 的根本局限——language encoder 不够强
- Gemini Robotics 的优势来自更强大的 vision-language backbone

### 5.3 三维 Generalization (Fig. 21)

85 个 tasks 评测，分四类：
- In-distribution (20%)
- Visual generalization (28%): distractors, background, lighting
- Instruction generalization (28%): typos, multilingual, rephrasing, descriptive
- Action generalization (24%): initial positions, object instances

**关键发现**：在 multilingual instruction 上，baselines **catastrophic failure（0%）**，Gemini Robotics 仍有 non-zero performance。这直接证明了 VLM backbone 规模的价值。

## 6. Specialization 与 Adaptation

### 6.1 Long-horizon Dexterity (Section 4.1)

6 个极具挑战的任务：
1. **Make an origami fox**：4 次精确折叠，bi-arm coordination
2. **Pack a lunch-box**：多步骤，2分钟+完成，**100% success rate**
3. **Spelling board game**：视觉识别 + 拼字，hand-drawn sketch 上 60% accuracy
4. **Play a game of cards**：fine-grained card manipulation
5. **Add snap peas to salad**（用 tongs）
6. **Add nuts to salad**（用 spoon）

每个任务用 2000-5000 episodes fine-tune。**关键 ablation**：从 scratch 训 Gemini Robotics specialist → 0% success across the board。这说明 generalist checkpoint 提供的 "physical common sense representation" 是必要的。

### 6.2 Reasoning-Enhanced Variant (Section 4.2)

这个实验很有意思：用 re-labeled action dataset fine-tune，让 action prediction 跟 trajectory understanding 中间表示对齐。Local decoder 扩展为把 reasoning intermediates 转成 continuous actions。

效果（Fig. 24）：
- One-step Reasoning：vanilla 60% → enhanced 90%
- Semantic Generalization：vanilla 50% → enhanced 80%
- Spatial Understanding：vanilla 40% → enhanced 70%

而且输出可解释的 keypoint trajectories（Fig. 25），类似 chain-of-thought 但是 spatial 的。

### 6.3 Fast Adaptation (Section 4.3)

8 个 subtasks，用 5/20/100 episodes fine-tune：
- 7/8 任务在 ≤100 demos 下达到 >70% success
- 2 个任务达到 100%
- "Origami first fold" 和 lunch-box 任务上，baselines 在 100 demos 时仍很低

这是 foundation model 的核心价值：**15分钟-1小时** demos 就能学新任务。

### 6.4 Cross-Embodiment (Section 4.4)

适应两种新平台：
- **Bi-arm Franka**：4 个 industrial tasks，avg 63% success
- **Apollo humanoid**（Apptronik）：5-fingered dexterous hands

在 Franka 上还测了 generalization（Fig. 28）：adapted Gemini Robotics 在 visual & action generalization 上**显著超过 single-task diffusion baseline**，说明 generalization 能力可以 cross-embodiment transfer。

## 7. Safety 与 Responsible AI

### 7.1 三层安全

1. **Content safety**：继承 Gemini 的 safety training
2. **Pointing safety**：针对 bias-inducing queries，SFT 后 rejection rate 从 20% → 96%
3. **Semantic action safety**：通过 ASIMOV benchmark 评估

### 7.2 ASIMOV Benchmark

两个子集：
- **ASIMOV-Multimodal**：visual safety QA
- **ASIMOV-Injury**：基于 NEISS 真实伤害记录

用 Constitutional AI 方法（Bai et al., 2022）post-train，对抗 prompt 下性能也能保持。

参考：
- ASIMOV: https://arxiv.org/abs/2503.08663
- SciFi-Bench: https://arxiv.org/abs/2503.10706
- Constitutional AI: https://arxiv.org/abs/2212.08073

## 8. 核心公式与数学直觉

这篇 paper 没有太多显式公式，但几个关键概念可以形式化：

### 8.1 Action Chunking

给定 observation $o_t$ 和 instruction $\ell$，模型输出 action chunk：

$$\hat{a}_{t:t+K} = f_\theta(o_t, o_{t-1}, ..., o_{t-H}, \ell)$$

其中：
- $o_t$: 时刻 t 的 observation（多视角图像 + robot state）
- $\ell$: language instruction
- $H$: history length
- $K$: chunk size（predict未来 K 步 action）
- $f_\theta$: VLA policy 参数化 by $\theta$

执行时用 **receding horizon**：每步执行 chunk 的第一个 action，或者执行前几个再 re-plan。50Hz effective frequency 就是靠 chunk size 实现的。

### 8.2 Diffusion Policy 目标（baseline）

Multi-task diffusion baseline 的训练目标：

$$\mathcal{L}_{diff} = \mathbb{E}_{t, \epsilon, a_0, K}\left[\|\epsilon - \epsilon_\theta(a_t, t, c)\|^2\right]$$

其中：
- $a_0$: ground truth action
- $a_t$: 加噪后的 action，$a_t = \sqrt{\bar{\alpha}_t}a_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$
- $t$: diffusion timestep
- $\epsilon \sim \mathcal{N}(0, I)$: noise
- $\epsilon_\theta$: noise prediction network
- $c$: conditioning（CLIP-encoded language + visual latents）
- $\bar{\alpha}_t$: cumulative noise schedule

### 8.3 π0 的 Flow Matching（baseline）

π0 用 flow matching 而非 DDPM：

$$\mathcal{L}_{FM} = \mathbb{E}_{t, a_0, a_1}\left[\|v_\theta(a_t, t, c) - (a_1 - a_0)\|^2\right]$$

其中：
- $a_0 \sim \mathcal{N}(0, I)$: 起点（noise）
- $a_1$: 目标 action
- $a_t = (1-t)a_0 + t a_1$: 线性插值
- $v_\theta$: velocity field
- $t \in [0, 1]$

Flow matching 比 DDPM 训练更稳定，采样步数更少。

### 8.4 Progress Score

为了更细粒度评测长 horizon 任务，定义 continuous progress：

$$\text{progress}(s) = \sum_{i} w_i \cdot \mathbb{I}[\text{subtask}_i \text{ completed}(s)]$$

其中 $s$ 是 rollout state，$w_i$ 是 subtask $i$ 的权重。例如 "Pack a lunch-box"：
- bread in ziploc: 0.1
- ziploc zipped: 0.25
- ziploc in lunch-box: 0.5
- all items in: 0.75
- lunch-box fully zipped: 1.0

## 9. 我对这篇 paper 的几点思考

### 9.1 VLM 规模就是 Robotics 规模

最 striking 的发现：**PaliGemma (3B) 在 multilingual instruction 上 0% success，Gemini (估计数百B) non-zero**。这跟 LLM scaling 的故事完全一致。Robotics 不再是"小模型+大数据"的游戏，而是"大模型+多样数据"的游戏。

### 9.2 Cloud-Edge 架构是务实选择

直接把 100B+ 模型部署到 robot 上不现实。Google 选择 cloud backbone + local decoder 的双组件方案，用 action chunking 弥补 latency。这个设计 pattern 我觉得会成为未来 VLA 部署的标准。

### 9.3 Physical Common Sense 是 Emergent 的

Section 4.1 的 from-scratch ablation 极有说服力：同样架构、同样数据量，从 generalist checkpoint fine-tune 能学 origami，从 scratch 训 0%。这说明 robot action data 训出来的 representation 里有 transferable 的 physical understanding，不仅仅是 task-specific skill。

### 9.4 Reasoning-Action Gap 仍然存在

Section 4.2 的 reasoning-enhanced variant 暴露了一个问题：vanilla VLA 在 OOD reasoning 上掉点严重。即使有 Gemini backbone，把 reasoning 能力"传导"到 action 仍需要专门的 re-labeling + fine-tuning。这是未来工作的关键方向。

### 9.5 安全维度被低估

ASIMOV benchmark 是个很好的开始，但 semantic action safety 是个 long tail 问题。"不能把毛绒玩具放热炉子上"这种常识有无限多个。Constitutional AI 在 text domain 工作良好，但物理世界 consequences 更难 enumerate。

## 10. 与相关工作对比

| 维度 | RT-2 | OpenVLA | π0 | Gemini Robotics |
|------|------|---------|-----|-----------------|
| Backbone | PaLI-X (55B) | PaliGemma (3B) | PaliGemma (3B) | Gemini 2.0 (~?) |
| Action freq | 1-3 Hz | ~5 Hz | ~5 Hz | **50 Hz** |
| Architecture | VLM token | VLM token | Flow matching | Cloud VLA + local decoder |
| Multi-task | ✓ | ✓ | ✓ | ✓ |
| Cross-embodiment | limited | limited | limited | ✓ (Franka, Apollo) |
| Long-horizon | ✗ | ✗ | limited | ✓ (origami, lunch-box) |
| ER capabilities | ✗ | ✗ | ✗ | ✓ (pointing, 3D, trajectory) |

参考：
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- π0: https://arxiv.org/abs/2410.24164
- ALOHA Unleashed: https://proceedings.mlr.press/v270/zhao25b.html

## 11. Limitations 与未来方向

Paper 自己承认的 limitations：
1. 长 video 上的 spatial grounding 仍弱
2. 数值预测（points, boxes）精度不够 fine-grained control
3. Multi-step reasoning + precise dexterity 结合仍难
4. Cross-embodiment 还需要 fine-tuning，未实现 zero-shot transfer

我额外几点观察：
- **数据效率仍待提升**：100 demos 学新任务虽然好，但离 human-level（1-3 demos）还远
- **Sim-to-real 没充分探索**：Paper 提到未来要 lean on simulation
- **Contact-rich tasks** 仍是难点：grasp prediction 在 fold dress 上 zero-shot 失败
- **Evaluation 规模**：20 trials per task 在 robotics 里算不少，但统计 power 仍有限

## 12. 总结

这篇 paper 的核心贡献是把 frontier VLM 的 reasoning 能力成功 ground 到物理世界，通过：
1. **ER 增强**：让 Gemini "看懂"3D、trajectory、affordance
2. **VLA fine-tuning**：让 Gemini "会动"，50Hz 控制
3. **Specialization recipe**：少量数据 specialize 到 extreme dexterity
4. **Cross-embodiment**：transfer 到 humanoid

关键 take-away：**robotics 的 scaling law 跟 LLM 一致——更大的 VLM backbone + 更多样的数据 = 更强的 generalist robot**。Google 用 Gemini 2.0 (~数百B参数) 把这个 thesis 推到了新高度。

对未来工作的启示：
- VLA 模型应该 build 在 frontier VLM 上，而不是 small VLM
- Cloud-edge 架构是部署大 VLA 的可行路径
- Generalist pre-training + specialist fine-tuning 的范式在 robotics 上 work
- Reasoning 与 action 的对齐仍是 open problem

希望这个解读对你 build intuition 有帮助！如果你想 deep dive 某个具体方面（比如 local decoder 的具体设计、CoT 在 ER 上的机制、或者 cross-embodiment transfer 的细节），可以继续聊。

主要参考链接：
- Paper 本身（arXiv 应该会发）: https://storage.googleapis.com/deepmind-media/gemini-robotics/
- ERQA: https://github.com/embodiedreasoning/ERQA
- ALOHA 2: https://arxiv.org/abs/2405.02292
- ASIMOV: https://arxiv.org/abs/2503.08663
- Google AI Principles: https://ai.google/static/documents/ai-responsibility-update-published-february-2025.pdf
