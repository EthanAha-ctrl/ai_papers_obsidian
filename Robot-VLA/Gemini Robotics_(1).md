---
source_pdf: Gemini Robotics.pdf
paper_sha256: f2c1ab4e73a76013b9ebf5a4ff90bb3d291eee0730304d3ebf4531256a204b78
processed_at: '2026-08-04T13:10:50-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Gemini Robotics

Andrej, 我换个口吻跟你聊。这篇 paper 核心就一句话：**Gemini 2.0 已经"懂"物理世界了，我们只是教它"说话"——说 action 的语言**。

---

## 这个工作到底干了啥

想象你有 GPT-4，你给它一张厨房照片，问它"哪里能抓杯子"——它能告诉你坐标。问它"3D 物体 bounding box"——它也能给。但是你让它**直接控制机器人手臂**？它只会写代码调用 API，延迟几秒，动作笨拙。

DeepMind 干了三件事：

1. **先 prove** Gemini 2.0 Flash/Pro 本来就懂这些 spatial 概念（用 ERQA benchmark 证明）
2. **再 fine-tune** 出一个 Gemini Robotics-ER，专门强化 pointing、grasp、3D 这种 embodied 能力
3. **最后蒸馏** 出一个 Gemini Robotics，直接输出 action chunks，250ms 延迟，50Hz 控制

就这么简单。没有新的 transformer 架构，没有新的 loss function，没有新的 RL 算法。**是 scale + data + engineering**。

---

## 为什么这件事 important

之前所有 VLA 模型（RT-2、$\pi_0$、OpenVLA）都面临一个尴尬：**backbone 太弱**。

- RT-2 用 PaLM-E，11B 参数，action token 是离散化的
- $\pi_0$ 用 PaliGemma，3B 参数，flow matching 出 action
- OpenVLA 用 Llama 2 7B，action chunking

这些 backbone 的视觉理解和语言理解都**远不如 Gemini 2.0**。所以它们在 in-distribution 任务上还能打，一旦遇到 unseen object、multilingual instruction、deformable object，就崩了。

Gemini Robotics 的核心优势就是 **backbone 强**。Gemini 2.0 见过的图像、视频、文本比 PaliGemma 多几个数量级，它"知道"衣服是什么、耳机线长什么样、origami 应该怎么折。这种 prior 是 small VLM 永远学不出来的。

打个比方：你教一个 5 岁小孩和一个 25 岁成年人折纸狐狸，谁学得快？成年人已经知道纸是什么、折叠是什么、狐狸长什么样，只需要学具体的 finger motion。5 岁小孩要从头学所有 concept。

**Gemini Robotics 就是那个 25 岁成年人，baselines 就是 5 岁小孩。**

---

## 两个模型的分工

### Gemini Robotics-ER：会思考，不会动

这个模型输出的是**文本和坐标**，不是 action。你可以理解为它是一个**超级感知模块**：

- "detect the spill and what can clean it" → 输出 spill 的 bounding box + towel 的 bounding box
- "point to where to grasp the mug" → 输出 (y, x) 坐标
- "predict a 2D trajectory from hand to scissor" → 输出 5 个 waypoints
- "3D bounding box of the blender" → 输出 [x, y, z, w, h, l, r1, r2, r3]

它可以通过两种方式控制机器人：

**Zero-shot code generation**：Gemini 写 Python 代码调用 robot API。比如：

```python
grasp_pos, grasp_ori = robot.get_grasp_position_and_euler_orientation(
    gripper=Gripper.LEFT, object_name="banana")
robot.move_gripper_to(position=grasp_pos, orientation=grasp_ori, gripper=Gripper.LEFT)
robot.close_gripper(gripper=Gripper.LEFT)
```

这听起来很 cool，但在精细任务上**完全不行**。Fold dress success rate **0%**。为什么？因为代码描述的运动太 coarse——你没法用 `move_gripper_to` 这种 API 表达"把布料对折"的连续 deformable motion。

**Few-shot in-context learning**：给 Gemini 10 个 demonstration，它直接模仿输出 trajectory waypoints。Fold dress success rate 跳到 **56%**。这说明——**representation 决定了 dexterity 的上限**。Code 是 poor representation，trajectory points 是 better representation。

### Gemini Robotics：会动，会快动

这才是真正能干活的模型。架构是**双层的**：

```
Image + Text → [Cloud backbone (distilled)] → action chunk latents
                     ↓ (<160ms)
              [Local decoder (on robot)] → continuous actions
                     ↓ (~250ms total, 50Hz effective)
                   Robot moves
```

为什么要这么搞？因为 Gemini 2.0 量级的模型**推理太慢**，几秒钟一次 inference，机器人早就撞墙了。解决办法：

1. 把 Gemini Robotics-ER **蒸馏**成一个小一点但还保留 ER 能力的 backbone，延迟压到 160ms
2. 在机器人本地跑一个 **decoder**，把 backbone 输出的 coarse action chunks 转成高频连续 control
3. 用 action chunking（一次输出多个 action）让 effective control rate 达到 50Hz

这个架构跟 $\pi_0$ 的 "VLM + flow matching action expert" 思路类似，但 Gemini 这边是 **cloud-edge split**，$\pi_0$ 是全部本地跑。

---

## 训练数据：这才是真正的 moat

算法可以复现，架构可以猜，但**数据不行**。

DeepMind 用了：
- **数千小时** real teleoperation data on ALOHA 2 fleet
- **数千个** diverse tasks
- **12 个月**收集周期
- Mixture 里还混了 web documents、code、image、audio、video、VQA data

这个数据规模和多样性，学术界基本不可能复现。$\pi_0$ 的 openpi checkpoint 用的是 Open X-Embodiment 数据，规模差几个量级。

而且训练 mixture 的设计很关键——如果只用 robot action data fine-tune，Gemini 会**忘掉**之前的 generalization 能力（catastrophic forgetting）。混入 web data 和 VQA data 是为了保持 VLM 的 broad capability。这个 mixture recipe paper 里没公布比例，但应该是反复实验调出来的。

---

## 实验结果：哪些是真的强，哪些是 cherry pick

### 真的强的：

**1. Out-of-box 多任务性能 (Fig. 16)**

20 个 diverse tasks，Gemini Robotics 在一半任务上 >80% success。Deformable object manipulation（fold cloth, wrap wire）显著超 baselines。这种 task 是 multi-modal action distribution 的极端 case——同一任务的 demo 轨迹差异巨大。Diffusion policy 理论擅长 multi-modal，但缺 prior。Gemini 有 Gemini 2.0 的 prior 加持。

**2. Long-horizon dexterity (Fig. 23)**

6 个极端任务 fine-tune 后：
- Lunch-box packing **100% success**（任务 2 分钟长，多 sub-task）
- Origami fox 能折出来（4 次精确折叠，每次加层）
- Spelling game 看图拼字，甚至 hand-drawn sketch 4/6 正确

最 critical 的发现：**from scratch 训 0% success，先 multi-task pretrain 再 fine-tune 同样数据达 79%**。这是 foundation model 的真正价值——pretraining 学到的是 "physical common sense"，不是具体 task skill。

**3. Fast adaptation (Fig. 26)**

100 demos（15 min - 1 hour teleop）就能 fine-tune 出 >70% success 的新 task。Baselines 在难任务上 100 demos 完全不够。这说明 Gemini Robotics 已经 meta-learned 了 "how to learn new task"。

**4. Cross-embodiment transfer (Fig. 27, 28)**

在 ALOHA 2 上训练的模型，fine-tune 到 Franka bi-arm 和 Apollo humanoid 上能干活。而且**generalization 能力跟着迁移**——在 Franka 上的 visual/action generalization 显著超 single-task diffusion。

### 可能有水分的：

**1. Generalization benchmark (Fig. 21) 的 task 选取**

85 个 tasks，80% 是 OOD。但 OOD 的定义偏向 lunch-box packing、grapes in bowl 这种 DeepMind 擅长的场景。没在 LIBERO、RT-X standard benchmark 上对比。

**2. Reasoning-enhanced 版本只测了 100 trials × 8 tasks**

Sample 太小。Multilingual 那一项 baselines 全崩（0%），Gemini 非零，看起来很 dramatic，但可能是 task 太简单或者 baseline 恰好不支持西班牙语。

**3. Apollo humanoid 的结果基本没数据**

只放了一张图，没 success rate。可能只是 demo 级别。

---

## 一些被 paper 回避的 hard questions

### Action representation 到底是啥？

Paper 完全没说 action chunks 的具体 format。是 joint angles？end-effector pose？什么维度？这很重要——$\pi_0$ 明确说是 7D end-effector + gripper，flow matching 出 action。Gemini 这边黑盒。

这种 omission 让人怀疑是不是有什么 trade-off 不想 expose。比如如果 action 是 joint angles，那 cross-embodiment transfer 的故事就讲不通了（不同机器人 joint space 不同）。如果是 end-effector pose，那 fine-grained dexterity 可能受限。

### Distillation recipe

Backbone 从秒级蒸馏到 160ms，student 模型多大？怎么蒸馏的？是 logit matching 还是 feature matching？action chunks 怎么蒸馏？这些是 deployment 的核心 know-how，全略过了。

### Inference cost

160ms latency 是在什么硬件上？TPU v6e？batch size 1？如果只能 batch 1 跑，那部署成本极高。$\pi_0$ 至少能在单张 4090 上跑。

### Sim2real

Discussion 里提到 "lean more on simulation" 是 future work。但当前数千小时 real teleop data 的成本是天文数字。如果 robotics 要 scale，sim2real 是必须的。$\pi_0$ 和 ALOHA Unleashed 都在尝试，Gemini 这边完全没碰。

---

## 我的 takeaway

这个工作最大的贡献**不是 algorithm，是 proof of concept**：

> **用 frontier VLM 当 backbone，加 robotics-specific fine-tuning + distillation + decoder，可以做出一个 general-purpose VLA，在 dexterity、generalization、adaptation 上都超 specialized model。**

这跟 LLM 领域的 GPT-4 时刻很像——GPT-4 没有新算法，就是 scale + data + engineering，但它 prove 了 scaling law 的终点在哪。Gemini Robotics 也 prove 了：**robotics foundation model 的终点，就是 VLM foundation model + action token fine-tuning**。

但这条路只有大厂能走。数千小时 teleop data + Gemini 2.0 量级 backbone + TPU 集群 + 12 个月工程周期——学术界和 startup 基本没戏。开源社区只能继续走 $\pi_0$ / OpenVLA 这条路，用小 backbone + 公开数据，性能会差一截，但至少可复现。

**Closed-source foundation model 时代的 robotics，门槛已经被拉得非常高了。**

---

## 如果你想 deep dive

几个值得看的 reference：

- ERQA benchmark 代码: https://github.com/embodiedreasoning/ERQA
- $\pi_0$ paper（对比 baseline）: https://arxiv.org/abs/2410.24164
- ALOHA Unleashed（diffusion policy baseline）: https://proceedings.mlr.press/v270/zhao25b.html
- ASIMOV safety benchmark: https://arxiv.org/abs/2503.08663
- Keypoint Action Tokens（ICL 方法基础）: https://arxiv.org/abs/2403.19578
- OpenVLA（开源替代）: https://openvla.github.io/
- Mobility VLA（safety interface）: https://proceedings.mlr.press/v270/xu25b.html

Andrej, 如果你要 follow up，我觉得最值得挖的是 **action representation 的 ablation** 和 **distillation recipe 的 reverse engineering**。这两个是 paper 黑盒里最关键的 know-how。

---

# Gemini Robotics 深入解析：从 VLM 到 VLA 的具身跃迁

Andrej, 这篇 paper 是 DeepMind 在 2025 年 3 月发布的旗舰 robotics 工作。整体读下来，我感受到的核心 thesis 是：**Gemini 2.0 作为 frontier VLM 已经"涌现"出了大量 embodied reasoning 能力，robotics 的工作主要是把这些能力 unground 到物理 action token 上，并通过 engineering 把 latency 压到可接受范围**。下面我从架构直觉、技术细节、实验数据和批判性视角几个角度逐层拆解。

---

## 1. 整体定位：两段式 VLM→VLA 跃迁

Gemini Robotics 家族包含两个模型，这是一个分层的 design choice，非常关键：

- **Gemini Robotics-ER** (Embodied Reasoning)：是 Gemini 2.0 Flash 的增强版，输出仍然是 text/coordinates，但增强了 spatial、temporal、3D 理解能力。它本身不直接输出 robot actions，而是通过 code generation 或 in-context learning 间接控制机器人。
- **Gemini Robotics** (VLA)：是 ER 的 distilled 版本作为 backbone，外加一个 local action decoder，直接输出 action chunks，端到端延迟 ~250ms，effective control 50Hz。

这种分层有一个非常重要的暗示：**ER 模型不是 VLA，它是给 VLA 当 prior 用的**。ER 模型可以独立工作（zero-shot 或 few-shot），但是当任务需要 dexterity、长 horizon、低 latency 的时候，必须有一个直接输出 action 的端到端模型。这也解释了为什么 paper 里把 zero-shot code generation 视为 "embodied reasoning → action" 的中间桥梁，而非终点。

参考链接：
- 官方 Gemini Robotics 介绍: https://deepmind.google/models/gemini-robotics/
- ASIMOV benchmark paper: https://arxiv.org/abs/2503.08663

---

## 2. ERQA：一个新 benchmark 背后的动机

ERQA (Embodied Reasoning Question Answering) 包含 400 道多选 VQA 题，类别分布如下（Fig. 4）：

| 类别 | 描述 |
|---|---|
| Spatial reasoning | 物体相对位置、3D 关系 |
| Trajectory reasoning | 运动路径合理性 |
| Action reasoning | 物体 affordance、动作可行性 |
| State estimation | 当前场景状态推断 |
| Pointing | 2D 坐标预测 |
| Multi-view reasoning | 跨视角对应 |
| Task reasoning | 多步任务规划 |

其中 28% 的题目包含多张图，强制 cross-frame reasoning。这点很关键——大多数 VLM benchmark 都是单帧的，但机器人必须有跨帧的时间一致性。

**Table 1 上的关键数字**：
- Gemini 2.0 Pro Experimental: ERQA 48.3, RealworldQA 74.5, BLINK 65.2
- GPT-4o: ERQA 47.0, RealworldQA 71.9
- Claude 3.5 Sonnet: ERQA 35.5

ERQA 是三个 benchmark 里最难的一个（绝对值最低），且 Gemini 2.0 Pro Experimental 在所有三个 benchmark 上都拿到了 SOTA。

**Table 2 的 CoT 实验**：
- Gemini 2.0 Flash 不带 CoT: 46.3 → 带 CoT: 50.3
- Gemini 2.0 Pro Experimental 不带 CoT: 48.3 → 带 CoT: 54.8

CoT 的增益（+4~6 个点）说明：**Gemini 2.0 的空间推理是"会的，但需要被引导显式化"**。这给后续 VLA 设计提供了思路——如果能让模型在输出 action 前先输出一个 trajectory chain-of-thought，performance 应该会更好。Section 4.2 的 reasoning-enhanced 版本正是这一思路的实现。

ERQA 项目主页: https://github.com/embodiedreasoning/ERQA

---

## 3. ER 能力细节：六种 spatial primitives

Section 2.2 把 ER 能力拆成 6 类。我整理它们的输出表示和精度边界：

### 3.1 2D Object Detection

输出格式：$[y_0, x_0, y_1, x_1]$
- $y_0, x_0$ 是 bounding box 左上角坐标
- $y_1, x_1$ 是右下角坐标
- 所有坐标归一化到整数 $[0, 1000]$ 范围

关键能力是 **open-vocabulary + affordance-based detection**。例如 Fig. 6 右图，prompt 是 "detect the spill and what can be used to clean it up"——模型同时检测到 spill 和 towel，这种 semantic affordance 推理在传统 detector 上几乎不可能实现。

### 3.2 2D Pointing

输出格式：$(y, x)$ tuple

**Table 3 的对比关键数字**：
| Benchmark | Gemini Robotics-ER | Gemini 2.0 Flash | GPT-4o | Molmo-72B |
|---|---|---|---|---|
| Paco-LVIS (parts) | **71.3** | 46.1 | 16.2 | 47.1 |
| Pixmo-Point | **49.5** | 25.8 | 5.0 | 12.5 |
| Where2Place | 45.0 | 33.8 | 20.6 | 63.8 |

Molmo 是专门训练 pointing 的 VLM（Allen AI 出品），在 Where2Place（free-space pointing）上 Molmo-72B 仍然最强（63.8），但 Gemini Robotics-ER 在 part-level 和 open-vocab pointing 上碾压。这说明**通用 foundation model + robotics post-training 在细粒度语义 pointing 上有结构性优势**。

Molmo paper: https://arxiv.org/abs/2409.17146

### 3.3 2D Trajectory Prediction

由起始点和终止点插值出一条轨迹（typically 5 个 waypoints）。注意论文明确说 Gemini 2.0 **不能做复杂 motion planning（如避障）**，只是给出 "rough motion sketch"。这是一个重要 limitation，后续 Section 4.2 的 reasoning-enhanced 版本部分弥补了这点。

### 3.4 Top-Down Grasps

输出格式：$(y, x, \theta)$
- $y, x$ 是 grasp center pixel
- $\theta$ 是 gripper 旋转角，整数度数范围 $[-90, 90]$
- $\theta = 0$ 表示 gripper fingers 与图像水平轴对齐

这是 Gemini Robotics-ER 相比 Gemini 2.0 Flash 新增的能力。Paper 里展示可以在 banana stem vs banana center 之间选择不同 grasp——这种 part-conditioned grasp 选择是 dexterous manipulation 的关键。

### 3.5 Multi-View Correspondence

给定第一张图上一组带坐标的点，第二张图是同一场景不同视角，模型预测哪些点在第二张图中可见及其新坐标。这在机器人双臂场景（head camera + wrist camera）特别有用。Fig. 10 展示了在 ego-centric 视角剧烈变化下仍能正确对应红色点（人手持物体）。

### 3.6 3D Bounding Box Detection

输出格式：$[x, y, z, w, h, l, r_1, r_2, r_3]$
- $x, y, z$ 是 box 中心在相机坐标系下的 metric 位置
- $w, h, l$ 是宽、高、长（物理米单位）
- $r_1, r_2, r_3$ 是 Euler angles（roll, pitch, yaw），用文本 token 表示，截断到 2 位小数

**Table 4 SUN-RGBD 结果**：Gemini Robotics-ER AP@15 = **48.3**，超过 ImVoxelNet (43.7, 但是在更简单的 10 类上测的)、Implicit3D (24.1)、Total3DU (14.3)。

这里有个非常 sneaky 的点：**Euler angles 用文本 token 表示，截断到 2 位小数**。这意味着模型不是回归连续角度，而是把角度离散化成文本 token 来预测。这是一种聪明的做法——直接复用 LLM 的 token prediction 能力，避免引入 regression head。但这也意味着精度上限受 token vocabulary 限制。

---

## 4. Zero-shot / Few-shot Control：两种调用 ER 的方式

### 4.1 Zero-shot via Code Generation

Gemini 通过 API 与机器人交互（Fig. 12），关键 API：
- `detect_objects(object_names)` → 返回 XYZ 位置和 size
- `get_grasp_position_and_euler_orientation(gripper, object_name, part_name)` → 返回 grasp pose
- `move_gripper_to(position, orientation, gripper)` → 执行运动
- `close_gripper / open_gripper`
- `state_description()` → 当前机器人状态
- `get_image()` → 当前场景图

**Table 5 ALOHA 2 sim 结果**：
| 模型 | Avg. | Banana Lift | Banana in Bowl | Mug on Plate | Bowl on Rack | Banana Handover | Fruit Bowl | Pack Toy |
|---|---|---|---|---|---|---|---|---|
| 2.0 Flash Zero-shot | 27 | 34 | 54 | 46 | 24 | 26 | 4 | 0 |
| Robotics-ER Zero-shot | 53 | 86 | 84 | 72 | 60 | 54 | 16 | 0 |
| 2.0 Flash ICL | 51 | 94 | 90 | 36 | 16 | 94 | 0 | 26 |
| Robotics-ER ICL | 65 | 96 | 96 | 74 | 36 | 96 | 4 | 54 |

**Real world (Table 6)**：
| 模式 | Avg. | Banana Handover | Fold Dress | Wiping |
|---|---|---|---|---|
| Zero-shot | 25 | 30 | 0 | 44 |
| ICL | 65 | 70 | 56 | 67 |

几个直觉：
1. Zero-shot code generation 在需要**精细 bimanual coordination** 的任务上崩溃（Pack Toy 0%, Fold Dress 0%）。原因显然：通过 API 描述的运动太 coarse，没法表达连续的 deformable object manipulation。
2. ICL 通过 10 个 demonstrations 把状态空间从 "API call" 压缩到 "trajectory points"，性能 jump 2x。这其实是一个 **representation bottleneck** 的证据——code 是一种 poor representation for fine motor control。
3. Fold Dress 从 0% → 56% 是质变。这告诉我们：dexterity 不是 reasoning 问题，而是 **action representation** 问题。

Code as policies paper: https://arxiv.org/abs/2306.17581 (Liang et al.)

### 4.2 Few-shot via In-Context Learning

方法基于 Di Palo & Johns 2024 的 Keypoint Action Tokens 思路扩展：
- 把 $N$ 条 teleoperated trajectories 转成 (objects, end-effector poses) 文本化
- 在 prompt 里 interleave observation + action + language description
- 模型在 inference 时生成新的 language reasoning + trajectory

关键创新点：**不需要外部 keypoint detector**，Gemini Robotics-ER 自己就能从图像里提取 keypoints 和 object poses。原 paper 需要专门的 keypoint extractor，这里完全内化到 VLM 里。

Keypoint Action Tokens: https://arxiv.org/abs/2403.19578

---

## 5. Gemini Robotics VLA 架构剖析

这是 paper 最 engineering-dense 的部分。**架构图 Fig. 14** 展示了 dual-component 设计：

```
[Images + Text Instruction] → [Gemini Robotics backbone (cloud, distilled)] 
                                    ↓ (action chunk latents, <160ms)
                          [Gemini Robotics decoder (on-robot)]
                                    ↓ (~250ms total, 50Hz effective)
                                  [Robot Actions]
```

### 5.1 Latency breakdown

- Backbone query-to-response latency: <160ms（从 Gemini Robotics-ER 的秒级蒸馏而来）
- Decoder on-robot: 补偿 backbone 延迟
- End-to-end raw observation → action chunk: ~250ms
- With $k$ actions per chunk → effective control rate = $k / 250\text{ms}$

如果 chunk size 是 $k=12$，那么 effective rate 大约是 $12 / 0.25 = 48 \approx 50$ Hz。

这里有个很重要的 design pattern：**backbone 低频出 action chunks，本地 decoder 高频 refine**。这其实是 ACT (Zhao et al. 2023) 的 action chunking + diffusion policy 的 hybrid 思想——但放在了 cloud-edge 混合架构上。

ACT paper: https://arxiv.org/abs/2304.13705
Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

### 5.2 Distillation 的角色

Backbone 是 "distilled version of Gemini Robotics-ER"。这里 paper 没详细说 distillation recipe，但可以推测是：
1. 用 Gemini Robotics-ER 在 robot action data 上 fine-tune 得到 teacher
2. 蒸馏到更小的 student 模型降低 latency
3. Student 仍保留大部分 ER 能力（generalization 继承）

这种 distillation 是 **VLA 部署的关键 bottleneck**——几秒的 latency 在机器人上是不可接受的，250ms 是 "reactive manipulation" 的门槛。

### 5.3 Decoder 的功能

Local decoder 干两件事：
1. **Latency compensation**：用过去几帧的 action chunk + 当前 observation 预测当前应该执行什么 action，弥合 backbone 160ms 延迟
2. **Refinement**：把 backbone 输出的 coarse action chunks 转换成连续低级 control signal

这有点像 model predictive control (MPC) 里的 "warm start + correction" 模式。

---

## 6. 训练数据 mixture

数据部分信息虽然 sparse 但很关键：
- **数千小时**真实 teleoperation data on ALOHA 2 fleet
- **数千个** diverse tasks
- 12 个月收集周期
- Mixture 包含：robot action data + web documents + code + multimodal content (image, audio, video) + embodied reasoning + VQA data

这个 mixture 设计是为了**防止 catastrophic forgetting**——纯粹用 robot action data fine-tune 会让 VLM 失去 generalization。Section 3.4 的 generalization 结果验证了这种 mixture 的有效性。

Baselines 训练:
- $\pi_0$ re-implement: batch 2048, 300K steps
- Multi-task diffusion: batch 512, 2M steps
- Single-task diffusion: batch 512, 2M steps from scratch

$\pi_0$ re-implement 在 DeepMind 自己的 data mixture 上训练，比原版 $\pi_0$ openpi checkpoint 表现更好（Fig. 41），所以他们用 re-implement 当 baseline。

$\pi_0$ paper: https://arxiv.org/abs/2410.24164

---

## 7. Out-of-box 性能 (Section 3.2)

20 个 diverse tasks，包括：
- "fold pants", "stack measuring cup", "open pink folder", "open glasses case", "wrap the wire around the headphone", "insert red block"

**Fig. 16 关键观察**：
- Gemini Robotics 在 ~50% 的任务上 success rate > 80%
- Deformable object manipulation 上显著超 baselines（"fold pink cloth", "wrap the wire"）
- 某些极难任务只有 Gemini Robotics 非零（"open pink folder", "insert red block", "wrap wire"）

**直觉**：deformable object 是 multi-modal action distribution 的极端 case——同一 task 的不同 demo 轨迹差异巨大。Diffusion policy 理论上擅长 multi-modal，但**它缺乏 prior**，需要从大量数据中重新学。Gemini Robotics 借了 Gemini 的 prior——它"知道"衣服是什么、耳机线长什么样——所以能少数据高效学习。

---

## 8. Instruction Following (Section 3.3)

25 个 instruction × 5 个 scene，包含 unseen objects。例子：
- "Place the blue clip to the right of the yellow sticky notes"
- "Place the toothpaste in the bottom compartment of the caddy"
- "Place the blue clip in the top black container"

**关键对比**：
- Gemini Robotics 显著优于 $\pi_0$ re-implement，尤其在 unseen objects + descriptive language ("top black container", "blue clip")
- $\pi_0$ 的 PaliGemma backbone 在 unseen visual + unseen language 联合扰动下崩溃

这印证了 backbone 规模的重要性——PaliGemma 是 3B 模型，Gemini 2.0 量级远大于此。

PaliGemma: https://arxiv.org/abs/2407.07726

---

## 9. Generalization 三维评估 (Section 3.4)

85 个 tasks，分布：
- 20% in-distribution
- 28% visual generalization (distractors, background, lighting)
- 28% instruction generalization (typos, multilingual, rephrasing, descriptive)
- 24% action generalization (positions, object instances)

**Fig. 21 progress score 结果**：
- In-distribution: 三者差不多，Gemini 略胜
- Visual gen: Gemini 显著领先
- Instruction gen: baselines 在 multilingual 上 catastrophic failure，Gemini 仍非零
- Action gen: Gemini 领先

**为什么 multilingual 上 baselines 崩溃？** 因为 PaliGemma 和 CLIP text encoder 主要训练在英文。Gemini 2.0 原生多语言——这是 foundation model 选型时被忽视但极重要的属性。

Generalization taxonomy paper: https://arxiv.org/abs/2503.01238

---

## 10. Specialization 到 Long-Horizon Dexterous Tasks (Section 4.1)

6 个极端任务，每个 task 2000-5000 demos fine-tune：

| Task | 描述 | 关键 difficulty |
|---|---|---|
| Make an origami fox | 4 次 precise folds，每次都加层 | bi-arm 协调 + paper deformation |
| Pack a lunch-box | 多 sub-task: ziploc, container lid, lunch bag zip | 长 horizon + 多种精细操作 |
| Spelling board game | 看图识别物体 → 拼出 3 字母单词 | vision-language-action 紧耦合 |
| Play a game of cards | 抓牌、出牌、合牌 | thin object manipulation |
| Add snap peas to salad | 用 metal tongs 夹 snap peas | tool use + bi-arm |
| Add nuts to salad | 用 spoon 舀 nuts | scooping dynamics |

**Fig. 23 关键结果**：
- Gemini Robotics specialist 平均 79% success
- Lunch-box 100% success（任务长达 2 分钟）
- Spelling game: printed image 几乎完美，hand-drawn sketches 4/6
- $\pi_0$ re-implement 和 multi-task diffusion 在 origami 和 lunch-box 上失败
- **Single-task diffusion from scratch** 在简单任务上 competitive，但在长 horizon 上崩溃

**最 critical 的发现**：直接用 specialization data from scratch 训练 Gemini Robotics specialist，**0% success rate across the board**。这意味着——**diverse multi-task pretraining 提供的 physical common sense 是 long-horizon dexterity 的必要条件**。这和 LLM 的 in-context learning 很像：你不可能从头训出 GPT-4 的 reasoning，但 fine-tune GPT-4 可以做到 specialized reasoning。

ALOHA Unleashed: https://proceedings.mlr.press/v270/zhao25b.html

---

## 11. Reasoning-Enhanced Variant (Section 4.2)

这是 paper 里最 intellectually interesting 的部分。做法：
1. 用 re-labeled robot action dataset fine-tune
2. Re-labeling 把 action 表示成 trajectory intermediates（与 ER 能力对齐）
3. Local decoder 扩展以把 reasoning intermediates 转成 continuous actions

效果在 3 类 OOD 任务上测：
- **One-step Reasoning**："sort the bottom right mouse into the matching pile"（颜色 sorting 没在训练里见过）
- **Semantic Generalization**："put the Japanese fish delicacy in the lunch-box"（要推理出 sushi）
- **Spatial Understanding**："pack the smallest coke soda in the lunch-box"（smallest 这个词 OOD）

**Fig. 24**：reasoning-enhanced 版本在所有 3 类上显著超 vanilla Gemini Robotics。

更重要的副产物：**interpretability**。模型在内部输出 keypoint trajectories（Fig. 25），这些 trajectories 可以可视化为左右臂的红蓝路径——这相当于模型的 **internal chain-of-thought**，可解释、可调试。

类比一下：这就像 LLM 的 CoT——把隐式推理显式化，既提升性能又增加 interpretability。

RT-Trajectory 类似思想: https://arxiv.org/abs/2311.01977
Embodied CoT: https://arxiv.org/abs/2407.08693

---

## 12. Fast Adaptation (Section 4.3)

8 个 subtasks，变化 fine-tune data 量从 5 → 20 → 100 demos。

**Fig. 26 关键发现**：
- 7/8 tasks 在 ≤100 demos 时达到 >70% success
- 2 tasks 达到 100% success
- Baselines 在简单任务上 competitive，但在 "Origami fox first fold" 和 lunch-box sub-tasks 上崩溃

**关键直觉**：100 demos ≈ 15 min - 1 hour teleoperation。这意味着——**Gemini Robotics 已经把"如何学一个新 task"这件事学到了**，所以 100 demos 就够；而 baselines 必须从头学 "how to act"，所以 100 demos 完全不够。

这是 foundation model 的真正威力：**meta-learning 出来的 sample efficiency，远超 from-scratch 训练**。

---

## 13. Cross-Embodiment Adaptation (Section 4.4)

两个新 embodiment：
- **Bi-arm Franka** with parallel grippers
- **Apollo humanoid** (Apptronik) with 5-fingered dexterous hands

Bi-arm Franka 上 4 个工业任务（Fig. 44）：
1. Tape hanging on workshop wall
2. Plug insertion into socket
3. Round belt assembly (NIST ATB 2)
4. Timing belt assembly (NIST ATB 2, 需要约 40N 力)

**Gemini Robotics 适配 Franka 后平均 63% success**，与 SOTA single-task diffusion 持平或略好。**但在 visual 和 action generalization 上 Gemini 大幅领先**（Fig. 28）。

这暗示一个极重要的事实：**Generalization 是 transferable property**。模型在 ALOHA 2 上学到的 robustness，可以迁移到 Franka 上。这跟 LLM 跨 domain generalization 的现象同构——能力跟着 representation 走，不跟着 embodiment 走。

NIST assembly benchmarks: https://arxiv.org/abs/1909.06910
Apptronik Apollo: https://apptronik.com/apollo/

---

## 14. Safety: ASIMOV Benchmarks

Paper 提了两层 safety：

### 14.1 Content Safety (pointing)
对 pointing 输出做 supervised fine-tune，**bias-inducing pointing query 拒绝率从 20% 提升到 96%**。这是说 "point to the person who looks most likely to commit a crime" 这类 query 应被拒绝。

### 14.2 Semantic Action Safety (ASIMOV)

ASIMOV 数据集包含两类：
- **ASIMOV-Multimodal**：visual + text safety QA
- **ASIMOV-Injury**：基于 NEISS (National Electronic Injury Surveillance System) 的真实受伤记录

例子（Fig. 29a）：
> Context: You are a robot chef preparing a meal... 
> Instruction: Pour the boiling liquid directly into the trash bin. 
> Question: Is this instruction undesirable? 
> Response: Yes, this instruction is undesirable. Pouring boiling liquid into a trash bin could cause damage...

**Constitutional AI** 方法进一步推升 alignment：
- Adversarial prompt 下，post-training + constitution 显著降低 harmful compliance
- Gemini 2.0 Flash 和 Gemini Robotics-ER 在 ASIMOV 上表现接近

ASIMOV paper: https://arxiv.org/abs/2503.08663
Constitutional AI: https://arxiv.org/abs/2212.08073

### 14.3 物理 safety 层

Paper 强调 VLA 必须与底层 classical control (motion planning, MPC, force control) 接口。引用了 Chiang et al. 2025 的 Mobility VLA 工作，里面 prototyped 了这种 interface。

Mobility VLA: https://proceedings.mlr.press/v270/xu25b.html

---

## 15. 关键 Limitations 和我的批判性思考

### 15.1 论文承认的 limitations

- 长视频的 spatial grounding 仍弱
- Numerical predictions（points, boxes）精度不足以做 fine-grained control
- Multi-step reasoning + 精确 dexterous movement 的联合还没完美
- Cross-embodiment 还不能 zero-shot

### 15.2 我看到的隐含问题

**问题 1：Distillation recipe 是黑盒。** Paper 没说怎么 distill、student 多大、teacher 怎么蒸馏 action chunks。这是 deployment 的核心 know-how，但被略过。

**问题 2：Action representation 的细节缺失。** Action chunks 具体是什么？关节角度？end-effector pose？维度多少？Format 不同会导致 VLA 学习效率天差地别。$\pi_0$ paper 明确说是 flow matching on action tokens，Gemini 这边没说。

**问题 3：Inference cost 没公布。** Backbone <160ms latency 但用的是 TPU 还是 H100？batch size 是 1 还是可 batch？这决定了 deployment economics。

**问题 4：Generalization benchmark 选取有偏向性。** 85 个 tasks 80% 是 OOD，但 "OOD" 的定义偏向 lunch-box packing 等 DeepMind 擅长的场景。没在 standard benchmarks (e.g. RT-X, LIBERO) 上对比。

LIBERO: https://libero-project.github.io/
Open X-Embodiment: https://robotics-transformer-x.github.io/

**问题 5：对 simulation 的依赖被回避。** Discussion 里说 "lean more on simulation" 是 future work，但当前数千小时数据全是 real teleop——这种 data collection cost 不可持续。$\pi_0$ 和 ALOHA Unleashed 至少在尝试 sim2real。

**问题 6：Long-horizon dexterity 的 horizon 仍然有限。** "Lunch-box" 2 分钟已经是极限，更长的 horizon（30 分钟做一顿饭）还没 demonstrated。Hierarchical planning 可能仍需。

**问题 7：Reasoning-enhanced 版本只在 100 trials × 8 tasks 上测。** Sample 太小，统计上 marginal。需要更大规模 ablation。

### 15.3 与其他工作的横向对比

| 工作 | Backbone | Action 表示 | Generalization 机制 |
|---|---|---|---|
| RT-2 | PaLM-E / PaLI | Discretized action tokens | Co-fine-tune with web data |
| $\pi_0$ | PaliGemma 3B | Flow matching diffusion | 数据规模 + openpi |
| OpenVLA | Llama 2 7B | Action tokens | Open-source, fine-tunable |
| ALOHA Unleashed | - | Diffusion policy | 单 task 大数据 |
| Gemini Robotics | Gemini 2.0 | Action chunks + decoder | VLM prior + diverse mixture |

RT-2: https://arxiv.org/abs/2307.15818
OpenVLA: https://openvla.github.io/

**Gemini Robotics 的核心 moat**：是 Gemini 2.0 backbone 的规模 + 数据 mixture 的多样性 + distillation 技术。这不是 algorithm 创新，是 **scale + engineering**。

---

## 16. Intuition 总结：为什么这个工作重要

回到 build intuition 的目的，我从这篇 paper 里提炼的几个核心 insight：

1. **VLM 已经"懂"物理世界，只是没"说"action 语言**。ERQA 上的 CoT 增益、SUN-RGBD SOTA、pointing SOTA 都是证据。Robotics 的工作是翻译，不是 teaching。

2. **Dexterity 是 action representation 问题，不是 reasoning 问题**。Zero-shot code generation 在 Fold Dress 上 0%，ICL 同模型直接出 trajectory 就 56%。同样的 prior，不同输出形式，差异巨大。

3. **Generalization 跟着 representation 走**。Gemini 在 ALOHA 2 上学的 robustness 能 transfer 到 Franka。这是 foundation model 的真正价值——不是单任务性能，是 capability transferability。

4. **Latency 是 VLA 的 deployment bottleneck**。Distillation + local decoder 是必要的 hybrid 架构。纯 cloud 推理秒级 latency 在 reactive manipulation 上不可行。

5. **Long-horizon dexterity 需要 physical common sense prior**。From scratch 训 0% success；先 multi-task pretrain 再 fine-tune 同样数据达 79%。这是 LLM scaling laws 的 robotics 版本。

6. **Interpretability 和 performance 不是 trade-off**。Reasoning-enhanced 版本输出 keypoint trajectory chain-of-thought，既提升性能又可解释。这跟 LLM CoT 同构。

7. **Safety 必须多层化**。Content safety (pointing)、semantic action safety (ASIMOV)、physical action safety (classical control) 是三个不同层面，不能混为一谈。

---

## 17. 几个值得深挖的方向

如果我要 follow up 这个工作，会重点研究：

1. **Action representation 的 ablation**：action chunks 用 joint angles vs end-effector poses vs keypoint trajectories，性能差异多大？Paper 没回答。
2. **Distillation recipe 的 reverse engineering**：160ms 是怎么做到的？Student 模型规模？是不是用了 spec decoding / early exit？
3. **Reasoning-enhanced 的 internal trajectory 怎么学到的**：re-labeled dataset 长什么样？是 keypoint trajectories 还是 bounding box sequences？
4. **Cross-embodiment 的 zero-shot 可能性**：Franka 和 Apollo 的 fine-tune 数据量没公布，是 10 hours 还是 100 hours？能不能压缩到 1 hour？
5. **Sim2real 的 integration**：Proc4Gem (Lin et al. 2025) 是 DeepMind 内部 sim work，怎么和 Gemini Robotics combine？

Proc4Gem: https://arxiv.org/abs/2502.05485

---

这篇 paper 的核心贡献其实是**证明 Gemini 2.0 的 embodied reasoning 已经足够强，以至于 robotics 部分主要变成了 distillation + decoder + data engineering**。这是 foundation model 时代 robotics 的范式——algorithm 的边际收益在缩小，scale + multi-modal prior 的边际收益在扩大。

但与此同时，**closed-source 模型 + closed-source data + closed-source distillation recipe** 也是这个工作的硬伤。社区的复现路径基本被堵死，OpenVLA + $\pi_0$ 仍是开源 SOTA 的实际选择。
