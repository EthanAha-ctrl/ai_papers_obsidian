---
source_pdf: GR-2.pdf
paper_sha256: 8b9bf7c3de182ce263ca65d103fc3a84aa0c70d13aa88f6b8dd5cdf5df2f5f3a
processed_at: '2026-08-04T22:16:15-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GR-2 的人话版本

## 1. 一句话讲清楚它在干嘛

**先让机器人看 3800 万个 YouTube 视频学"世界是怎么运作的",再给它几十条示范就能学会一项新技能。**

这跟人差不多——一个从来没见过开罐器的成年人,只要看到一次别人怎么用,就能上手;但一个婴儿学不会,因为他没有"世界是怎么运作的"先验。GR-2 干的就是先灌这个先验,再 fine-tune 具体技能。

---

## 2. 为什么 video 能教机器人?

这个是整个 paper 最关键的 intuition,我展开讲。

假设你要让机器人"把可乐从桌上拿到篮子里"。传统做法是:

- 收集几千条 teleoperation demo
- 训个 policy 直接学 (image → action)

问题来了:数据太少,泛化不了。换个背景、换个可乐罐子形状,就完蛋。

GR-2 的思路是:**人类互联网上已经有海量"手抓东西"的视频**(Howto100M、Ego4D、EPIC-KITCHENS 这些 first-person 视频)。机器人不需要动作 label,只要学会"预测下一帧长什么样",就能 implicitly 学会:

- 物体被抓住会怎么动
- 手靠近物体时视角怎么变
- 桌上的东西被移动后相对位置怎么变化
- "倒水"这个 verb 在视觉上是什么样

这些都编码进 transformer 的 hidden state 里。等 fine-tune 时,只要告诉模型"你要做的动作长这样",它就把这个 prior 直接拿来用。

参考 UniPi (https://arxiv.org/abs/2302.00118) 最早提出这个想法,GR-1/GR-2 把它工程化做大。

---

## 3. 训练两步走,跟 LLM 一模一样

### Step 1:Video Pre-training(38M clips, 50B tokens)

输入:`(文字描述, 当前帧)`  
目标:`预测下一帧、下下一帧...`

这里的"帧"是被 VQGAN 压成 discrete token 的图,跟 LLM 吃 text token 是同一个流程。所以作者可以**直接复用所有 LLM 训练基础设施**(FlashAttention、ZeRO、FSDP、KV-cache、autoregressive sampling)。这是为什么能 scale 到 719M 参数。

Loss 就是 next-token cross-entropy:

$$\mathcal{L}_{\text{pre}} = -\sum_{i} \log p_\theta(x_i^{\text{img}} \mid x_{<i}^{\text{img}}, l)$$

变量:
- $x_i^{\text{img}}$ — 第 $i$ 个 image token(VQGAN codebook 里的一个 index)
- $x_{<i}^{\text{img}}$ — 在它之前的所有 image tokens
- $l$ — language condition
- $\theta$ — transformer 参数

### Step 2:Robot Data Fine-tuning(40K + 94K trajectories)

输入:`(文字, 多视角图像序列, 机器人状态)`  
输出:`(未来图像序列, 动作轨迹)`

注意输出是**双 head**:一个继续生成 video,一个出 action。Video head 留着不是浪费,它是 auxiliary task,强迫 hidden state 保持"world model"能力。同时 inference 时可以看到模型"想象"的未来画面,debug 友好。

---

## 4. Action 怎么出?cVAE 的作用

这一块技术细节比较多,但 intuition 很简单。

**问题**:同一个状态可以对应多个合理动作。比如从 A 到 B,绕左绕右都行。如果用 MSE regression 训练,模型会输出"绕左和绕右的平均值",结果就是直直撞中间障碍物——不可执行。

**解法**:用 conditional VAE,引入一个 latent variable $z$。$z$ 采样不同,model 输出不同 mode。

cVAE 的 ELBO:

$$\mathcal{L}_{\text{act}} = \underbrace{\mathbb{E}_{q_\phi(z \mid \mathbf{a}, \mathbf{c})}\left[\log p_\theta(\mathbf{a} \mid z, \mathbf{c})\right]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\left(q_\phi(z \mid \mathbf{a}, \mathbf{c}) \,\|\, p(z)\right)}_{\text{regularizer}}$$

变量:
- $\mathbf{a} \in \mathbb{R}^{k \times d_a}$ — 一段长度 $k$ 的 action chunk,每步 $d_a$ 维(比如 7 维:xyz position + 旋转 + gripper)
- $\mathbf{c}$ — conditioning,即 transformer 在 action 位置的 hidden state
- $z$ — latent,通常 $\mathcal{N}(0, I)$ prior
- $\phi$ — encoder,$\theta$ — decoder

推理时 $z \sim \mathcal{N}(0, I)$,sample 出来的 action 就是合理的 multi-modal 轨迹。这个设计跟 ACT (https://arxiv.org/abs/2304.13705) 是同源思路。

**为什么是 trajectory chunk 而不是单步?** 单步 autoregressive 会有高频抖动(相邻 step 不连续),而且每步都要 forward pass,inference 慢。chunk 一次性出 $k$ 步,既平滑又快。

---

## 5. 模型大小与参数冻结策略

Default GR-2 是 230M total params,其中 **95M trainable**,其余 frozen。这是个很关键的工程细节。

| 组件 | 是否 frozen | 为什么 |
|---|---|---|
| CLIP text encoder | Frozen | 文本语义稳定,没必要 retrain |
| VQGAN image tokenizer | Frozen | codebook 不能动,否则 pretrain token 语义失效 |
| GPT backbone 的早期层 | Frozen | 保住 web-scale 知识,防止 catastrophic forgetting |
| GPT backbone 的后期层 + state proj + action head | Trainable | 适应 robot domain 与 action output |

这种 partial freezing 是 foundation model 微调的常见做法,但在这里尤其重要——因为 robot data 量小(40K trajectories),如果全参数 fine-tune,pretrain 学到的 world dynamics 会被 wash 掉。

参考 GR-1 (https://arxiv.org/abs/2312.13139),作者在 GR-2 里强调 "lossless knowledge transfer" 就是指这个冻结策略。

---

## 6. 部署:Policy 出 Cartesian,WBC 翻译成 joint

GR-2 直接输出的是**末端执行器在 Cartesian 空间的轨迹**:$\mathbf{a}_{t:t+k} = \{(x_t, R_t, g_t), ..., (x_{t+k}, R_{t+k}, g_{t+k})\}$,其中 $x$ 是 position、$R$ 是 rotation、$g$ 是 gripper state。

但 Kinova Gen3 是 7-DoF arm,关节空间是 joint angles $\mathbf{q} \in \mathbb{R}^7$。需要 IK 把 Cartesian → joint。

WBC (Whole-Body Control) 算法做了三件事:

1. **Trajectory optimization**:对 raw Cartesian chunk 做 minimum-jerk 平滑,保证速度/加速度连续
2. **Inverse kinematics + constraint**:
   - collision avoidance(自碰、环境碰)
   - manipulability index $\mathcal{M}(\mathbf{q}) = \sqrt{\det(J(\mathbf{q}) J(\mathbf{q})^\top)}$ 最大化,避免接近 singularity
3. **200Hz 下发 joint command**

这里 $J(\mathbf{q})$ 是 Jacobian 矩阵,描述 end-effector velocity 与 joint velocity 的线性关系。$\det(J J^\top)$ 越大,机械臂在当前姿态下越能灵活响应各方向运动需求。

这一层把"高层 policy + 低层 control"解耦,让 GR-2 专心做规划,WBC 专心做 kinematics + safety。参考 MOMA-Force (https://arxiv.org/abs/2304.01934)。

---

## 7. 数据增强:用 Diffusion + SAM 做"换背景、加物体"

这是 GR-2 把 GR-1 甩开的另一个工程杠杆。

原来的 teleoperation 数据都是在一个固定桌面、固定背景拍的,泛化到新厨房必死。作者做了一整套 augmentation pipeline:

### 7.1 加新物体

训练一个 diffusion model (DDPM, https://arxiv.org/abs/2006.11239),用 self-collected + Open Images (https://arxiv.org/abs/1811.00982) 物体图训练。给定一个 frame 的指定 region,diffusion model 把这个物体"贴"进去,保证 lighting/阴影合理。

### 7.2 换背景

SAM (https://arxiv.org/abs/2304.02688) 抠出前景(机器人 + 桌上物体),背景区域留空。

### 7.3 Video inpainting 保持运动一致

只换第一帧还不够,后面帧物体怎么动、机器人怎么动都要跟着改。用 Latte (https://arxiv.org/abs/2401.03048) 这个 video diffusion transformer,以原 video + inpainted first frame 作 condition,生成 augmented video。**关键是 robot motion 保持不变**,因为 action label 还是原 teleoperation 的 action。

这等价于一个 "synthetic data engine",把 40K trajectory 几乎免费扩充成数十万条 background/object 多样化的 trajectory。结果(图 6):
- Unseen Environment 成功率从 71.7% → **87.0%**

---

## 8. 多任务学习的几个数字

GR-2 在 105 个 table-top 任务上:

| Setting | 成功率 |
|---|---|
| Simple(训练分布) | **97.7%** |
| Distractor(加干扰物) | 接近 Simple |
| Unseen Backgrounds | 71.4% |
| Unseen Environments | 71.7% |
| Unseen Manipulation | 55.8% |

而用 1/8 数据(50 trajectories/task)训练:

| Setting | 成功率 |
|---|---|
| Simple | 73.9% |

**50 条/task 就能在训练分布内拿到 73.9%**——这是真正惊人的数据效率。对比 RT-1 (https://arxiv.org/abs/2212.06817) 用 130K demos 训单任务;GR-2 用 50 × 105 = 5250 demos 训 105 任务,平均每任务数据量低一个量级以上。

---

## 9. Bin Picking:工业场景的真正考验

这个实验我觉得是 paper 里最重磅的。

### 设置

- 两个 basket:source(右)、target(左)
- 指令固定:`move any object from the right basket to the left basket`
- 94K 训练轨迹,55 个训练物体
- 评测 122 个物体(55 seen + 67 unseen)
- 4 个 setting:Seen / Unseen / Cluttered Seen / Cluttered Unseen(Cluttered = 物体数量翻倍到 12-17 个)

### 结果

| 模型 | 平均成功率 |
|---|---|
| GR-1 | 33.3% |
| GR-2 | **79.0%** |

提升 45.7 个百分点。

更关键的:**GR-2 在 Unseen / Cluttered 上的成功率与 Seen 几乎持平**,这是真正的工业可用性指标。GR-1 在 unseen/cluttered 上崩盘,说明 pre-training 数据没到 critical mass;GR-2 用 38M video 把这个 threshold 跨过去了。

图 8 还展示 GR-2 能抓透明塑料杯、毛绒玩具、反光金属罐——这些是传统 grasp planner(model-based 方法)的盲区,因为没法建 mesh / 算 geometry。end-to-end learning 直接从 pixel 学就绕开了这个限制。

---

## 10. CALVIN:long-horizon benchmark

CALVIN (https://arxiv.org/abs/2112.03227) 是 simulated benchmark,要连续完成 5 个 language instruction 的 chain。1000 个 chain 评测。

| Metric | GR-1 | GR-2 |
|---|---|---|
| 1-task SR | 94.9% | **98.6%** |
| 5-task SR | 73.1% | **85.9%** |
| Avg length | 4.21 | **4.64** |

平均能完成 4.64 个,接近满分 5。这是当前 SOTA。

但 5-task SR 不到 100% 说明长 horizon 还是有 drift,模型在第 4、5 个任务上开始失手。这是个值得攻的方向——可能需要 hierarchical planning。

---

## 11. Scaling law 验证

图 11 给出四个 size 的实验:

| 模型 | Trainable Params | Pretrain Val Loss (↓) | Real-Robot SR (↑) |
|---|---|---|---|
| GR-2-S | 30M | 高 | 低 |
| GR-2-B | 95M | 中 | 中 |
| GR-2-L | 312M | 低 | 高 |
| GR-2-XL | 719M | 最低 | 最高 |

**pre-training val loss 单调下降,real-robot SR 单调上升**。这意味着还能继续 scale——把 backbone 推到 7B 或更大,理论上能继续吃 38M video 数据红利。这是 foundation model 路线的核心 KPI:loss 与下游 metric 一起随 scale 改善。

跟 Sora (https://openai.com/research/video-generation-models-as-world-simulators) 的 scaling law 观察一致。

---

## 12. 两个 mental model

### Model A:GR-2 是个会"想象"的机器人

```
看到当前帧 + 听到指令
       ↓
   在脑内"放映"接下来几秒画面(predicted video)
       ↓
   生成能"演完"这个画面的动作(action trajectory)
       ↓
   WBC 翻译成关节运动
       ↓
   机械臂执行
```

图 12-17 的可视化证明这个"想象"非常准——predicted frames 与 GT rollout 高度对齐。这说明 transformer 的 hidden state 里真的编码了"下一步世界长什么样"的 plan,action head 只是在执行这个 plan。

### Model B:GR-2 是个"先理解世界、再学技能"的小孩

```
Stage 1: 看了 3800 万个 YouTube 视频(无监督)
         → 理解物体怎么动、手怎么抓、语言对应什么视觉事件
         
Stage 2: 看几十次示范就能学会一项新技能
         → 因为"技能"本质是"在已有世界模型上读出一个 action"
```

类比 LLM:GPT 看完整个互联网文本后,给个 few-shot prompt 就能做新任务。GR-2 是同一个范式,只是从 text token 换成了 VQGAN image token + action token。

---

## 13. 架构总图(再画一遍,更口语)

```
       "pick up the yellow bottle"
              ↓
       [CLIP text encoder,frozen]
              ↓
         text tokens: [pick, up, the, yellow, bottle]
              
   当前帧(head cam)  →  [VQGAN,frozen]  →  image tokens
   当前帧(hand cam)  →  [VQGAN,frozen]  →  image tokens
   robot state       →  [Linear,trainable]  →  state tokens
              ↓
   全部拼成一条长序列,送进 GPT transformer
              ↓
   ┌──────────────────────────────────────┐
   │  GPT-style causal transformer        │
   │  前 80% 层 frozen,后 20% 层 trainable │
   └──────────────────────────────────────┘
              ↓
        ┌─────┴─────┐
        ↓           ↓
   未来图像 token   action latent z ~ N(0,I)
        ↓           ↓
   VQGAN decode   cVAE decoder
        ↓           ↓
   预测的未来视频   Cartesian action chunk
                    ↓
              ┌─────────┐
              │   WBC    │
              │ 平滑+IK+ │
              │ collision │
              └─────────┘
                    ↓
              关节指令 @ 200Hz
```

---

## 14. 我的直觉判断

### 这条路线是对的

robot manipulation 的 bottleneck 一直是数据,而不是算法。GR-2 通过 video generative pre-training 把"数据"从稀缺资源(robot demo)换成海量资源(web video),并且验证了 scaling law 成立。这是 foundation model 范式在 manipulation 上的正名。

### 但还有几个缺口

1. **VQGAN 是个 bottleneck**。codebook 重建质量上限会限制对接触细节的捕捉。下一步可能换 FSQ (https://arxiv.org/abs/2309.15505) 或 MAGVIT-v2 的 LFQ (https://arxiv.org/abs/2404.06665),codebook collapse 风险更低,重建质量更高。

2. **Unseen Manipulation 55.8% 是真瓶颈**。web video 学到的 affordance 仍然是"见过物体类别"的 affordance,对真正 novel shape 泛化不够。可能需要加 3D representation(参考 Act3D https://arxiv.org/abs/2304.03559 或 3D Diffuser Actor https://arxiv.org/abs/2402.10885)补 geometry prior。

3. **cVAE 的 multi-modal 能力弱于 diffusion**。Diffusion Policy (https://arxiv.org/abs/2303.04137) 已经证明 diffusion head 在 multi-modal action 上更强。如果 action head 换成 diffusion,可能进一步提升。

4. **长 horizon drift**。CALVIN 5-task SR 85.9% 而非 100%,暗示需要 hierarchical 结构。可以让上层显式生成 sub-goal token,下层解码为 primitive action(类似 HULC https://arxiv.org/abs/2201.05919)。

5. **Latency 数据缺失**。论文没给 inference latency,但 WBC 跑 200Hz 说明 receding horizon 是必须的。如果要做 dynamic environment(物体被人推动),replan 频率要 ≥10Hz,对 719M 模型是个 nontrivial 的算力要求。

6. **50 trajectories/task 的极限**。能不能再降到 5 demo/task?这关系到 one-shot / few-shot manipulation 的工业诉求。可能需要 in-context trajectory 设计,类比 LLM 的 in-context learning。

---

## 15. 一句话总结

**GR-2 = Sora 的"video as world simulator"思路 + robot manipulation 的 fine-tune 范式 + cVAE 出 multi-modal action + WBC 做低层 control**。它证明了 web-scale video pre-training 不只是 demo 漂亮,而是真的能在 100+ 真实任务上把数据效率提一个数量级、把 unseen 场景成功率翻倍。这是 manipulation foundation model 路线上一个里程碑式的数据点。

项目主页:https://gr2-manipulation.github.io  
GR-1 论文:https://arxiv.org/abs/2312.13139

---

# GR-2:Generative Video-Language-Action Model 深度解析

## 0. TL;DR 与核心 intuition

GR-2 的核心论点可以浓缩成一句话:**video generation 是 world dynamics 的"通用先验",把它当作 robot policy 的 pre-training 目标,可以让 policy 在 fine-tune 阶段用极少数据高效学会上百种 manipulation skill**。这跟 LLM 在 text token 上的 next-token prediction 是同一个范式,只是 token space 换成了 VQGAN-discretized 的 image tokens,再额外并联一个 action head。

更深层的设计哲学:robot policy 的难点不在拟合 (state, action) 对,而在于 **counterfactual reasoning**——"如果我这样做,世界会怎么变"。video generation pre-training 直接强迫模型去 internalize 这个 forward model $p(o_{t+1:t+k} \mid o_{t-h:t}, l)$,于是 transformer 的 hidden state 不仅是 perceptual feature,而是 latent world state。action head 等于是在这个 latent world state 上读 out 一个 inverse dynamics $p(a_{t:t+k} \mid o_{t-h:t}, o_{t+1:t+k}, s_{t-h:t})$。这就是为什么 GR-2 能用 50 trajectories/task 学会新任务。

项目主页:https://gr2-manipulation.github.io
GR-1 原文:https://arxiv.org/abs/2312.13139

---

## 1. Pre-training 数据的工程化

GR-2 把 GR-1 的 0.8M videos 直接拉到 **38M video clips / 50B tokens**,这是关键的 scaling 杠杆。

数据来源组成:
- Howto100M (Miech et al. ICCV 2019): https://arxiv.org/abs/1906.10070
- Ego4D (Grauman et al. CVPR 2022): https://arxiv.org/abs/2110.07058
- Something-Something V2 (Goyal et al. ICCV 2017)
- EPIC-KITCHENS (Damen et al. ECCV 2018)
- Kinetics-700 (Carreira et al. 2019)
- RT-1 dataset (Brohan et al. 2022): https://arxiv.org/abs/2212.06817
- BridgeData V2 (Walke et al. CoRL 2023): https://arxiv.org/abs/2308.12952

**关键的 curation pipeline**:
1. MediaPipe (Lugaresi et al. 2019) 做 hand filtering,保留含 hand-object interaction 的 clip → 直接筛选出与 manipulation 相关的子集
2. Open-Sora (Zheng et al. 2024) 重新 caption,以提升 text-video alignment 质量
3. 与 in-domain robot data 联合混入

这里有一个很重要的细节——pre-training 数据并不需要严格是 robot data。作者依赖的是 "human activity video 包含足够多 object affordance 和物理 dynamics" 这一假设。这与 RT-2 (https://arxiv.org/abs/2307.15818) 的 "co-train with web data" 哲学相近,但 GR-2 走的是 "generative pre-training + fine-tune" 而不是 "joint co-training",pretrain 阶段不需要 action label,这是它能在 38M 量级 scaling 的关键——action label 是稀缺资源,video 不是。

---

## 2. 模型架构:从 input 到 output 的全流程

### 2.1 输入侧

定义输入序列(论文 Eq. 1):

$$\mathbf{a}_{t:t+k} = \pi(l,\, \mathbf{o}_{t-h:t},\, \mathbf{s}_{t-h:t})$$

变量含义:
- $l$ — language instruction,token 数量取决于 CLIP text encoder
- $\mathbf{o}_{t-h:t}$ — observation history,$h$ 是历史窗口长度,每帧来自 head camera + hand camera 两个 view
- $\mathbf{s}_{t-h:t}$ — robot state history,包含 end-effector 的 position (3D) + rotation (常见 6D rotation 或 quaternion) + binary gripper state
- $\mathbf{a}_{t:t+k}$ — action trajectory chunk,长度 $k$,在 Cartesian 空间

### 2.2 编码模块

| 模块 | 实现 | 是否 frozen |
|---|---|---|
| Text encoder | CLIP ViT-L/14 (Radford et al. ICML 2021, https://arxiv.org/abs/2103.00020) | Frozen |
| Image tokenizer | VQGAN (Esser et al. CVPR 2021, https://arxiv.org/abs/2012.09841) | Frozen,在大规模 web + robot data 上预训练 |
| State encoder | Linear projection | Trainable (仅在 fine-tune 阶段) |
| Backbone | GPT-style causal transformer (decoder-only) | Trainable |

**为什么 VQGAN 而不是 continuous patch embedding?** VQGAN 把 image 压成 discrete tokens 后,整个 pre-training 就退化成标准 LLM 的 next-token prediction,可以直接复用 LLM 训练 infrastructure (FlashAttention, FSDP, ZeRO, etc.),并且 KV-cache、autoregressive sampling 等工具全部可用。代价是 reconstruction quality 上限受 codebook size 限制,但作者额外在 robot data 上训练了 VQGAN,弥合 domain gap。

### 2.3 Transformer 主体

输入 token 序列排布(直觉上的伪图):

```
[ CLIP_text_tokens ] [ VQGAN_image_tokens_view1_t-h ... t ] 
[ VQGAN_image_tokens_view2_t-h ... t ] 
[ state_tokens_t-h ... t ] 
[ future_image_tokens_to_predict ] 
[ action_tokens_to_predict (cVAE) ]
```

注意这里有个 architecture 关键改进点(论文 Sec. 1 第三条 bullet):"a novel model architecture that allows the knowledge gathered from pre-training to seamlessly transfer to downstream fine-tuning in a **lossless** way"。结合 GR-1 的经验,这通常意味着:pre-training 时所有 transformer 参数都是 trainable,而 fine-tune 时只让一小部分 head (state projection、action cVAE decoder) trainable,backbone 的早期 layer 可以 frozen(论文 3.5 提到 default 230M 参数里 95M trainable,印证了这一点)。这种 partial freezing 既保住了 web-scale 知识不被 catastrophic forgetting,又减少了 fine-tune 的 gradient cost。

### 2.4 Action head:conditional VAE

action trajectory 通过 cVAE (Sohn et al. NeurIPS 2015, https://papers.nips.cc/paper/2015/hash/8d55a249e6baa441fe224c2f914292c8) 生成:

cVAE 的标准 ELBO:
$$\mathcal{L}_{\text{act}} = \mathbb{E}_{q_\phi(z \mid \mathbf{a}, \mathbf{c})}\left[\log p_\theta(\mathbf{a} \mid z, \mathbf{c})\right] - D_{\mathrm{KL}}\left(q_\phi(z \mid \mathbf{a}, \mathbf{c}) \,\|\, p(z)\right)$$

变量:
- $\mathbf{a} \in \mathbb{R}^{k \times d_a}$ — action trajectory chunk,$k$ 步、每步 $d_a$ 维
- $\mathbf{c}$ — conditioning context,即 transformer 在 action token 位置的 final hidden state
- $z$ — latent variable,通常 $\sim \mathcal{N}(0, I)$ prior
- $\phi$ — encoder 参数,$\theta$ — decoder 参数

**为什么用 cVAE 而不是 diffuser actor / flow matching?** 关键是 multi-modality:同一个 (state, obs, lang) 可以对应多个合理 action trajectory (比如从左或右绕过障碍物)。MSE regression 会让 model 输出多 mode 的平均值 → 不可执行。cVAE 通过 latent $z$ 显式建模 mode,采样时 $z \sim p(z)$ 即可生成不同 mode。这一点和 ACT (Zhao et al. 2023, https://arxiv.org/abs/2304.13705) 是同一个动机。

**为什么 trajectory 而不是 single step?** 论文原话:"generating action trajectories rather than single-step actions is crucial for both trajectory smoothing and real-time performance。" 这个直觉是:autoregressive 单步生成会产生高频抖动(相邻 step 之间不连续),而 chunk 生成相当于做了 temporal smoothing;同时一次 forward pass 输出 $k$ 步,可降低 inference latency,直接关系到 200Hz control 是否可行。

### 2.5 整体 fine-tune objective

$$\pi(l, \mathbf{o}_{t-h:t}, \mathbf{s}_{t-h:t}) \to \mathbf{o}_{t+1:t+k+1},\, \mathbf{a}_{t:t+k}$$

Loss 可写为:
$$\mathcal{L}_{\text{total}} = \underbrace{-\sum_{i}\log p_\theta(x_i^{\text{img}} \mid x_{<i}, l, \mathbf{s})}_{\text{video generation loss}} + \lambda \underbrace{\mathcal{L}_{\text{act}}^{\text{cVAE}}}_{\text{action loss}}$$

其中 $x_i^{\text{img}}$ 是 future frame 的 VQGAN tokens,$\lambda$ 是 action loss 的权重。video loss 保留下来——这一点很重要:它既是 auxiliary regularizer(迫使 hidden state 保持 world-modeling 能力),又让我们能在 inference 时可视化模型"想象"的未来。

---

## 3. 训练流程的两阶段图

```
Stage 1: Generative Pre-training (web-scale)
   ┌──────────────────────────────────────────────┐
   │  38M clips, 50B tokens                        │
   │  Input:  (text, frame_t)                      │
   │  Target: frame_{t+1}, ..., frame_{t+k}        │
   │  Loss:   next-token CE on VQGAN tokens        │
   │  Frozen: CLIP, VQGAN                          │
   │  Trainable: GPT backbone                      │
   └──────────────────────────────────────────────┘
                        ↓
Stage 2: Robot-data Fine-tuning
   ┌──────────────────────────────────────────────┐
   │  ~40K trajectories (multi-task)               │
   │  ~94K trajectories (bin-picking)              │
   │  Input:  (text, multi-view frames, states)    │
   │  Target: future multi-view frames + action   │
   │  Loss:   video CE + cVAE ELBO                 │
   │  Frozen: CLIP, VQGAN, 部分 backbone           │
   │  Trainable: state proj, action head, 部分     │
   └──────────────────────────────────────────────┘
                        ↓
            Inference: WBC @ 200Hz
```

---

## 4. Real-Robot 部署:Whole-Body Control

这是论文里容易被忽视但 industrial 落地关键的一环。

**硬件**:
- 7-DoF Kinova Gen3 arm
- Robotiq 2F-85 gripper
- 两台 camera:head (static, workspace overview) + hand (wrist-mounted, contact close-up)

**WBC pipeline**:
1. GR-2 输出 Cartesian space trajectory $\mathbf{a}_{t:t+k}$
2. Trajectory optimization:对 raw trajectory 做平滑性/连续性优化(常见做法是 minimum-jerk 或 quadratic programming with smoothness regularizer)
3. Cartesian → Joint space 转换(逆运动学 / IK),同时显式加入:
   - collision constraint (避免自碰、与环境碰)
   - manipulability constraint (Yoshikawa manipulability index $\mathcal{M}(q) = \sqrt{\det(J(q) J(q)^\top)}$)
4. 关节级指令以 **200Hz** 下发到 real robot

这种"policy 出 Cartesian chunk → 优化器出 joint command"的分层控制非常工程化,等价于把传统 trajectory optimization 当成 safety filter + smoothness filter,允许上游 policy 关注 high-level planning 而非 low-level kinematics。MOMA-Force (Yang et al. IROS 2023, https://arxiv.org/abs/2304.01934) 是同 group 的前序工作。

---

## 5. Multi-Task 实验:数据与设置

### 5.1 任务谱

105 个 table-top task,覆盖 8 类 primitive skills:
- picking / placing / uncapping / capping / opening / closing / pressing / pouring

每 task 平均 400 条 demo,总约 40,000 trajectories。低数据 setting 下用 1/8 = 50 条/task。

### 5.2 评测场景(图 3)

| Setting | 描述 |
|---|---|
| Simple | 训练分布同构 |
| Distractor | 加入相似 color/shape 干扰物 |
| Unseen Backgrounds | 两条 unseen tablecloth |
| Unseen Environments | 两个 unseen kitchen,含 scene distractor |
| Unseen Manipulation | unseen 物体类别 + unseen instance + unseen instruction |

### 5.3 Data Augmentation

这是 GR-2 在 generalization 上能甩开 GR-1 的关键工程操作:

1. **Object insertion**: 训练一个 diffusion model (DDPM, Ho et al. NeurIPS 2020, https://arxiv.org/abs/2006.11239),用 self-collected + Open Images (Kuznetsova et al. IJCV 2020) 训练,可指定 region 插入特定物体
2. **Background swap**: SAM (Kirillov et al. ICCV 2023, https://arxiv.org/abs/2304.02688) 抠出 background region
3. **Video inpainting**: Latte (Ma et al. 2024, https://arxiv.org/abs/2401.03048) 以原 video + inpainted first frame 为 condition,生成 augmented video,**保持 robot motion 不变**

这一套流程本质上是把 diffusion + SAM 当成 robot data 的 "synthetic data engine"。区别于传统 domain randomization(改光照、改颜色),这里直接做几何/语义级别的合成,与 RT-X (https://arxiv.org/abs/2310.08864) 的思路一致。

### 5.4 主要结果(图 6)

| 模型 | Simple | Unseen BG | Unseen Env | Unseen Manip |
|---|---|---|---|---|
| GR-1 (400/task) | ~? | ~35% | ~35% | — |
| GR-2 (400/task) | **97.7%** | 71.4% | 71.7% | 55.8% |
| GR-2 w/ DA | — | — | **87.0%** | — |
| GR-2 (50/task) | 73.9% | > GR-1 | > GR-1 | — |

平均 generalization 74.7%。值得注意:

- **GR-2 vs GR-1 在 unseen scenarios 上几乎翻倍**(71.4 vs 35 左右)——这是 pre-training 数据从 0.8M 到 38M 的直接 payoff
- **50 trajectories/task 即可在 Simple 上达到 73.9%** → 数据效率高到可怕,这从侧面证明 pre-training 已经"理解"了大多数 manipulation primitive
- **Unseen Manipulation 55.8%** 是当前 bottleneck,failure mode 主要是 (a) novel shape object 抓取失败,(b) 在 unseen instruction 下选错 object——这说明 VQGAN + language grounding 在 unseen composition 上还是受限

---

## 6. End-to-End Bin Picking

更接近工业场景的设定。

### 6.1 数据

- 94,000 pick-and-place trajectories,55 objects 训练
- 评测总共 122 objects(55 seen + 67 unseen)
- 4 个 setting:Seen, Unseen, Cluttered Seen, Cluttered Unseen
- Cluttered = 物体数量翻倍(12-17 个)
- 指令固定:"move any object from the right basket to the left basket"

### 6.2 结果(图 9)

| 模型 | 平均成功率 |
|---|---|
| GR-1 | 33.3% |
| GR-2 | **79.0%** |

差距 45.7 个百分点。而且 GR-2 在 Unseen 与 Cluttered 设置上的成功率与 Seen 接近——这是 industrial 落地真正需要的"鲁棒 plateau"。GR-1 在 unseen/cluttered 设置上 degrade 严重,说明 web-scale pre-training 数据量没到 critical mass。

### 6.3 透明、可形变、反光物体的处理

图 8 显示 GR-2 可以 handle transparent cup、deformable plush toy、reflective metal can 等 model-based 方法(传统 grasp planner)几乎无法处理的物体。这是 end-to-end learning 的天然优势——不需要 mesh/几何先验,policy 直接从 pixel→action 学。

---

## 7. CALVIN Benchmark

CALVIN (Mees et al. RA-L 2022, https://arxiv.org/abs/2112.03227):simulated 长程任务,34 个 task,language-conditioned,ABCD-D split,~20,000 expert demos。

评测:1000 个 instruction chain,每个 chain 5 个任务连续完成。指标是 1/2/3/4/5-task success rate + average length。

### 7.1 Baselines

- RT-1 (Brohan et al. 2022) — FiLM-conditioned transformer
- MT-ACT (Bharadhwaj et al. 2023, https://arxiv.org/abs/2309.01918) — FiLM + action chunking
- HULC (Mees et al. RA-L 2022) — hierarchical latent plan
- RoboFlamingo (Li et al. 2023, https://arxiv.org/abs/2311.01378) — VLM fine-tune
- GR-1 (Wu et al. 2023)

### 7.2 GR-2 结果

| Metric | GR-1 | GR-2 |
|---|---|---|
| 1-task SR | 94.9% | **98.6%** |
| 5-task SR | 73.1% | **85.9%** |
| Avg length | 4.21 | **4.64** |

平均长度 4.64 意味着在 5 任务连续指令下,平均能完成 4.64 个——这是当前 CALVIN ABCD-D 的 SOTA。

---

## 8. Scaling 实验(图 11)

四个尺寸:

| 变体 | Trainable Params |
|---|---|
| GR-2-S | 30M |
| GR-2-B (default) | 95M |
| GR-2-L | 312M |
| GR-2-XL | 719M |

观察:

1. Pre-training validation loss(Ego4D、RT-1、in-domain robot data三个验证集)随 model size 单调下降 → **video generation 的 scaling law 成立**,这与 Sora (Brooks et al. 2024, https://openai.com/research/video-generation-models-as-world-simulators) 的发现一致
2. Fine-tune 后的 real-robot success rate 也随 size 单调上升

这非常重要——它意味着这条路还有 headroom。如果 GR-2-XL 的 719M 已经在 unseen manipulation 上给出 55.8%,那么把 backbone 推到 B 级 (~7B) 或更高,理论上还能继续吃 pre-training 数据红利。这是 foundation model 范式的核心 KPI:loss 与下游 metric 都随 scale 改善。

---

## 9. Autoregressive Video Generation 的可视化(图 12-17)

这是论文里我个人觉得最有 intuition-building 价值的部分。GR-2 在 inference 时同时输出:
- predicted future frames (Pred)
- action trajectory

可视化显示 Pred 与 GT rollout 高度对齐。这意味着 transformer 的 hidden state 里已经隐式编码了 "下一步环境长什么样" 的 plan,action head 只是在执行这个 plan。

这给了一条改进路线:**iterative refinement**——先 refine video generator(更多数据、更高 resolution、更长 horizon),再让 action head 跟上。这与 Sora 的 "video as world simulator" 主张一脉相承,但 GR-2 把它落到了 robot manipulation 上。

与之相关的前序工作:
- VIPER (Escontrela et al. NeurIPS 2024, https://arxiv.org/abs/2310.09148):video prediction 作 RL reward
- UniPi (Du et al. NeurIPS 2024, https://arxiv.org/abs/2302.00118):text-guided video generation 出 policy
- Video Language Planning (Du et al. 2023, https://arxiv.org/abs/2310.10625)
- MaskViT (Gupta et al. 2022, https://arxiv.org/abs/2206.11894)

---

## 10. 与其他路线的对比

| 方向 | 代表 | 与 GR-2 的差别 |
|---|---|---|
| VLM fine-tune | RT-2, OpenVLA (https://arxiv.org/abs/2406.09246) | 复用 vision-language pretrain,但 action 是 regression head;不显式建模 world dynamics |
| Masked pre-train | RPT (Radosavovic et al. CoRL 2023, https://arxiv.org/abs/2306.10007), MVP | 学 representation,不学 generative dynamics |
| Contrastive | R3M (Nair et al. 2022, https://arxiv.org/abs/2203.12601), VC-1 | 同上,representation-only |
| World model + RL | DreamerV3 (Hafner et al. 2023, https://arxiv.org/abs/2301.04104) | latent world model,在 RL 内部学习;GR-2 用 supervised generative pretrain + imitation |
| Inverse dynamics + unlabeled video | VPT (Baker et al. NeurIPS 2022, https://arxiv.org/abs/2206.11376) | 在 Minecraft 用 inverse dynamics 给 web video 打 action label,再训 policy;GR-2 直接在 web video 上做 generative pretrain,不需要 inverse dynamics 标注 |
| Diffusion policy | Diffuser Actor (Ke et al. 2024, https://arxiv.org/abs/2402.10885), 3D Diffuser Actor | 用 diffusion 而非 cVAE 出 action;不和 video generation 联合 |
| Goal-image conditioning | RoboCat (Bousmalis et al. 2023), Transporter family | 不用 language |

GR-2 在这条 spectrum 上的独特定位:**用 generative video pretrain 作为 world-modeling prior + cVAE 作 multi-modal action head + multi-view token fusion**。它既不是纯 VLM 路线(因为显式做 video generation),也不是纯 world-model RL 路线(因为没有 RL)。

---

## 11. 我的几个批评性直觉与可探索方向

### 11.1 VQGAN bottleneck

VQGAN 的 codebook size 是重建质量与 token sequence 长度的 trade-off。对 manipulation 来说,精细的 contact 区域(指尖-物体接触点)对 action 预测至关重要,但 VQGAN 往往会模糊掉这些 high-frequency 区域。一个可能的改进是用 FSQ (Finite Scalar Quantization, Google 2023, https://arxiv.org/abs/2309.15505) 或 MAGVIT-v2 (https://arxiv.org/abs/2404.06665) 的 LFQ,codebook collapse 风险更低。

### 11.2 Action 与 video 的对齐方式

论文里没明确说 action tokens 在 transformer 序列中如何 attend 到 video tokens(全 causal?cross-attention?)。如果 action tokens 在最后位置且 fully attend 到所有 prior video tokens,那它实际上做的是 "given predicted plan, output action"。这与 UniPi 的 "video → inverse dynamics → action" pipeline 在功能上等价,但 GR-2 把两步合并到一个 forward pass 里。

### 11.3 长 horizon planning 的缺失

CALVIN 5-task chain 的平均长度 4.64 已经很好,但 5 步 chain 的 SR 是 85.9% 而非接近 100%。这暗示 model 在长 horizon 上还是有 drift。一个直接改法是引入 hierarchical structure,让上层显式生成 sub-goal token(类似 HULC 的 latent plan),下层解码为 primitive action。这相当于在 GR-2 的 transformer 之外再加一层 abstraction。

### 11.4 Unseen Manipulation 的瓶颈

55.8% 在 unseen manipulation 上提示 web-scale pretrain 学到的主要是 "见过物体类别的 affordance",对 true novel shape 还没泛化。可能的解法:
- 加入 3D-aware representation(参考 Act3D https://arxiv.org/abs/2304.03559、3D Diffuser Actor),用 NeRF/3DGS 提供 geometry prior
- 用 part-level affordance 而非 whole-object,提升组合泛化

### 11.5 50 trajectories/task 的极限

GR-2 用 50 demo/task 就能在 Simple 上拿 73.9%,这是数据效率的高水位。但 50 这个数能否继续降到 5?这关系到 one-shot / few-shot manipulation 的工业诉求。可能需要 meta-learning 或 in-context trajectory 的设计(类比 in-context learning in LLM)。

### 11.6 Closed-loop与 open-loop 的张力

GR-2 一次出 $k$ 步 trajectory,如果 $k$ 太大,open-loop error 会累积;如果 $k$ 太小,实时性受影响。论文没给 $k$ 的具体值,也没讨论 replan frequency。一个常见的折中是 receding horizon / MPC-style,执行 $k_{\text{exec}} < k$ 步后重新 infer,这可以在 WBC 那一层加。

### 11.7 cVAE vs Diffusion Action

cVAE 的 multi-modality 表达能力其实弱于 flow matching / diffusion(参考 Diffusion Policy, Chi et al. RSS 2023, https://arxiv.org/abs/2303.04137)。如果 GR-2 把 action head 换成 diffusion head(与 video 生成部分共享 denoising transformer),可能能进一步提升 multi-modal action 采样质量。

### 11.8 Latency 数据缺失

论文没给 inference latency,但提到 WBC 跑 200Hz。GR-2 在 GPU 上的 forward pass 估计是几十到几百 ms 级别,因此 receding horizon 是必须的。如果要做 dynamic environment(物体被人推动),replan 频率需要 ≥10Hz 才能跟上,这对 719M 参数的 XL 模型是 nontrivial 的算力需求。

---

## 12. 一张 mental model 总结

把 GR-2 当作下面这个流水线:

```
   Language ───► CLIP ───► [text tokens]
                                │
   Multi-view frames ─► VQGAN ─► [image tokens]
                                │
   Robot states ──► Linear ──► [state tokens]
                                │
                                ▼
                   ┌────────────────────────┐
                   │  GPT-style Transformer │  ← pre-trained on 38M videos
                   │  (causal, decoder-only)│     → next-token on VQGAN
                   └────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
       [future image tokens]            [action tokens]
                │                                │
                ▼                                ▼
           VQGAN decode                    cVAE decode
                │                                │
                ▼                                ▼
        Predicted future video         Cartesian action chunk
                                                 │
                                                 ▼
                                ┌───────────────────────────┐
                                │ WBC: trajectory optimize │
                                │ + IK with manipulability │
                                │ + collision constraint   │
                                └───────────────────────────┘
                                                 │
                                                 ▼
                                  Joint commands @ 200Hz
```

---

## 13. 参考链接汇总

- GR-2 项目主页:https://gr2-manipulation.github.io
- GR-1 论文:https://arxiv.org/abs/2312.13139
- CLIP:https://arxiv.org/abs/2103.00020
- VQGAN:https://arxiv.org/abs/2012.09841
- RT-1:https://arxiv.org/abs/2212.06817
- RT-2:https://arxiv.org/abs/2307.15818
- OpenVLA:https://arxiv.org/abs/2406.09246
- BridgeData V2:https://arxiv.org/abs/2308.12952
- Open X-Embodiment:https://arxiv.org/abs/2310.08864
- CALVIN benchmark:https://arxiv.org/abs/2112.03227
- RoboFlamingo:https://arxiv.org/abs/2311.01378
- ACT / ALOHA:https://arxiv.org/abs/2304.13705
- Diffusion Policy:https://arxiv.org/abs/2303.04137
- DDPM:https://arxiv.org/abs/2006.11239
- SAM:https://arxiv.org/abs/2304.02688
- Latte (video diffusion transformer):https://arxiv.org/abs/2401.03048
- Open-Sora:https://github.com/hpcaitech/Open-Sora
- Howto100M:https://arxiv.org/abs/1906.10070
- Ego4D:https://arxiv.org/abs/2110.07058
- EPIC-KITCHENS:https://arxiv.org/abs/1804.02748
- DreamerV3:https://arxiv.org/abs/2301.04104
- VPT:https://arxiv.org/abs/2206.11376
- UniPi:https://arxiv.org/abs/2302.00118
- VIPER:https://arxiv.org/abs/2310.09148
- MaskViT:https://arxiv.org/abs/2206.11894
- RPT:https://arxiv.org/abs/2306.10007
- R3M:https://arxiv.org/abs/2203.12601
- MOMA-Force:https://arxiv.org/abs/2304.01934
- MT-ACT (RoboAgent):https://arxiv.org/abs/2309.01918
- HULC:https://arxiv.org/abs/2201.05919
- 3D Diffuser Actor:https://arxiv.org/abs/2402.10885
- Act3D:https://arxiv.org/abs/2304.03559
- Sora:https://openai.com/research/video-generation-models-as-world-simulators
- MediaPipe:https://arxiv.org/abs/1906.08172

---

如果让我用一句话回答 "为什么 GR-2 重要":它把 Sora 式的 video-as-world-simulator 思路落地到真实机器人,证明 web-scale generative pre-training 不只是 demo 漂亮,而是可以在 100+ 真实 manipulation task 上把 data efficiency 提升一个数量级、在 unseen environment 上把成功率翻倍。这是 manipulation foundation model 路线上一个里程碑式的数据点。
