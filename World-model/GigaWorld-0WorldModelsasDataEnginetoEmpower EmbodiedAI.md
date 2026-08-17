---
source_pdf: GigaWorld-0WorldModelsasDataEnginetoEmpower EmbodiedAI.pdf
paper_sha256: c0995727d03d71a4f3d8a8be17704b9ab9cac434563d601c23ba281468b5879d
processed_at: '2026-08-04T21:43:45-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GigaWorld-0 用人话讲

Andrej, 好嘞, 我把学术腔调脱掉, 用聊天的方式讲一遍。这篇 paper 本质上就是在说一件事: **与其费劲去真实世界采数据, 不如训练一个能"做梦"的模型, 让它批量梦见机器人干活的画面, 然后把这些梦当成训练数据**。

但做梦有个问题 — 梦里东西可能飘起来, 物理不对, 或者机械臂长得不对。所以这套系统分成两大块: 一块负责"把画面做得逼真" (video generation), 一块负责"让物理和几何不出错" (3D + physics)。两者一拼, 你拿到的就是既好看又符合物理的机器人数据, 拿去训练 VLA policy。

---

## 为什么这事值得干

机器人采数据这事有多痛苦, 你肯定比我清楚。teoperation 一个人盯着机械臂抓杯子, 一天采几百条, 贵得要死, 还容易把杯子摔了。simulator 倒是便宜, 但 simulator 渲染出来的画面跟真实世界永远有 gap — texture 假, lighting 假, 机械臂表面看起来像塑料。

GigaWorld-0 的想法很直接: 既然现在 video generation model (Sora, Wan, HunyuanVideo 这些) 已经能在 internet video 上学到 photorealistic 的 visual distribution, 那**直接用它来生成机器人场景视频就好了**。text prompt 控制 "做什么任务", camera 参数控制 "从哪个角度看", 物理参数控制 "arm 的 friction 和 damping"。

最终 target 就是一个 data factory: 输入任务描述, 输出一堆 (video, action) pair, 喂给 VLA policy 训练。整个 pipeline 不需要任何真实机器人采集 — 这是核心卖点。

参考资料:
- Cosmos (NVIDIA 的 physical AI world model): https://arxiv.org/abs/2501.03575
- DreamGen (类似 idea): https://arxiv.org/abs/2505.12705

---

## GigaWorld-0-Video: "做梦" 的那部分

### Foundation model: GigaWorld-0-Video-Dreamer

这是整套系统的底座, 一个 image-text-to-video (IT2V) 的 generative model。给它一张初始帧 + 一句 text prompt ("put the gray plate into the sink"), 它给你生成一段机械臂抓盘子的视频。

**架构上几个关键选择**:

#### Flow matching 而不是 DDPM

公式长这样:

$$
\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c})
$$

人话: $\mathbf{z}_t$ 是当前时刻 $t$ 的 latent state (一张被 noise 污染的图), $t$ 从 0 走到 1, $t=0$ 是纯 noise, $t=1$ 是 clean video frame。$\mathbf{c}$ 是条件 (text + image embedding)。$\mathbf{v}_\theta$ 是网络学的 "velocity field" — 告诉你从当前 latent 应该往哪个方向走, 才能走到最终 data。

相比 DDPM 那种加噪 - 去噪的马尔科夫链, flow matching 学的是一个 ODE 的 vector field, trajectory 可以走直线 (optimal transport), 训练更稳, sample 步数更少。对 video 这种高维 latent 是实打实的 advantage。

参考资料: Flow matching 原始 paper https://arxiv.org/abs/2210.02747

#### 3D-VAE 压缩 + 3D-RoPE 编码位置

video 太大, 直接在 pixel space 训练不现实。3D-VAE 把 (T, H, W) = (4, 8, 8) 倍压缩 — 也就是时间压 4 倍, 空间压 8 倍 × 8 倍, 输出 16-channel latent。再叠一层 $1 \times 2 \times 2$ 的 patchify 进一步降维。

位置编码用 **3D-RoPE** — 在 (T, H, W) 三个 axis 上都加 rotary embedding。这事的重要性在于: 标准 1D RoPE 只能编码 sequence order, 但 video 是 3D grid, 你希望 attention 能感知 "这个 pixel 在时间上离我多远, 在空间上离我多远"。3D-RoPE 把这个 locality inductive bias 直接 baked 进 positional encoding。

参考资料: RoPE 原始 paper https://arxiv.org/abs/2104.09864

#### Sparse Attention + MoE: 不想全连接

attention 用的是 NATTEN (Neighborhood Attention), 每个 token 只 attend 局部 neighborhood, 不做 full attention。video 这种局部相关性强的数据, full attention 是浪费。

FFN 换成 **MoE**, 4 个 expert, 每个 token 只激活 2 个。公式:

$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{4} g_{i,t} \mathrm{FFN}_i(\mathbf{u}_t)
$$

人话: 每个 token 进来, 用 router 算一下它跟 4 个 expert 的 affinity, 选 top-2 高的 expert 激活, 其他直接 zero out。$g_{i,t}$ 就是 sparse gate, 大部分位置是 0。

**Critical 决策**: 这里**不使用 shared expert** (DeepSeek-V2 有, 这篇没有)。理由是 video 语义太杂 — rigid body, deformable object, background, foreground, lighting — 你强制 share 一部分 capacity, 反而限制每个 expert specialization。4 expert / 2 active 是非常 sparse 的配置, 等效参数 ~2B, 但 total 参数接近 4B, 推理时算力只花在 2B 上。

### Load balance loss: 防 routing collapse

如果不加约束, router 容易塌缩到 "永远只用 expert 1 和 2", expert 3 和 4 饿死。DeepSeek-V3 的 complementary balance loss:

$$
\mathcal{L}_{\mathrm{Load}} = \alpha \sum_{i=1}^{4} f_i P_i
$$

其中 $f_i$ 是 expert $i$ 被 routed 到的 frequency (硬计数), $P_i$ 是 average normalized affinity (软分数)。两者相乘, 你要 collapse, 这个 loss 就会变大。$\alpha = 0.01$ 是个小权重, 防止 balance loss 盖过 generation loss。

参考资料:
- DeepSeek-V2 MoE 架构: https://arxiv.org/abs/2405.04434
- DeepSeek-V3 balance loss: https://arxiv.org/abs/2412.19437
- NATTEN (Neighborhood Attention): https://github.com/SHI-Labs/NATTEN

---

### Dreamer 怎么当 Data Engine: IDM 这一步

Dreamer 给你 video, 但 VLA policy 要的是 (video, action) pair。怎么办? 训一个 Inverse Dynamics Model (IDM):

$$
\boldsymbol{\theta}_{1:T} = f_{\mathrm{IDM}}(\mathbf{V})
$$

输入 video $\mathbf{V} = \{\mathbf{v}_1, \ldots, \mathbf{v}_T\}$, 输出每个 timestep $t$ 的 joint angle trajectory $\boldsymbol{\theta}_t = [\theta_t^{(1)}, \ldots, \theta_t^{(D)}]^\top$, 其中 $D$ 是 arm 的 joint 数量 (G1 humanoid 是 12 arm joint + 2 gripper DOF)。

**关键 trick: masked training**。用 SAM2 把 robotic arm 从 video 里 segment 出来, 只把 arm region 喂给 IDM。这避免了 IDM 跑去学 background 的 spurious correlation — 你要它学的是 "arm 长这样 → action 应该是这样", 而不是 "背景是厨房 → action 是抓盘子"。

Figure 3 给了可视化: predicted joint trajectory 跟 ground truth 在 12 arm joint + 2 gripper DOF 上几乎完全重合。这步验证很重要 — 说明 IDM 能 reliable 地从 generated video recover action, 不然 VLA 训出来全是 noise。

**这套设计的 elegance 在于**: Dreamer 不需要懂 robot kinematics, 它只管 "生成 plausible visual future"; IDM 不需要懂 texture, 它只管 "看 arm pose 反推 joint angle"。两个 model 各自干擅长的事, 解耦得干净。

参考资料: SAM2 paper https://arxiv.org/abs/2408.00714

---

### 三个 post-training 变体: 拓展 Dreamer 的能力

Dreamer 是底座, 但 paper 还需要三个变体来 cover 不同维度 augmentation。

#### AppearanceTransfer: 改 texture / material / lighting

你采了一批真实数据, 但都是白天办公室场景。你想让 VLA 泛化到夜晚、户外、不同材质。appearance transfer 就干这个 — text prompt 说 "make it night scene with metal texture", 它给你生成对应变体, geometry 和 motion 不变。

**架构选择**: 这里 paper 明确**不用 ControlNet**。理由是 ControlNet 要复制一份 encoder, 在 MoE backbone 下 duplication 会让 parameter 翻倍, 太重了。换成更轻的方案:
- depth map + normal map 用 3D-VAE encode 成 latent
- 跟 noise latent channel-wise concat
- 几层 channel-compress MLP 降维
- 直接喂 transformer

depth 用 VideoDepthAnything, normal 用 LOTUS。这个设计本质是: 你的 conditional signal (depth/normal) 本身就是 dense 3D tensor, 形状跟 noise latent 兼容, 直接拼一起比 ControlNet 复制 encoder 高效得多。

参考资料:
- ControlNet: https://arxiv.org/abs/2302.05543
- VideoDepthAnything: https://arxiv.org/abs/2501.12375 (CVPR 2025)
- LOTUS: https://arxiv.org/abs/2409.18124

#### ViewTransfer: 改 camera viewpoint

这个数学最 elegant。你在 world frame $\mathcal{W}_A$ 采了 video, 想合成从另一个 world frame $\mathcal{W}_B$ 看的样子, 其中 robot base 已经 move 过了。

约束: **end-effector 在 world 中的绝对位置不能变**, 否则任务语义就变了 (抓杯子的轨迹变成抓空气)。

公式 (Eq.10):

$$
\mathbf{T}_t^{\mathrm{ee \leftrightarrow \mathcal{W}}} = \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_A}} \cdot \mathbf{T}_t^{\mathrm{ee \to base}} = \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_B}} \cdot \mathbf{K}_t
$$

人话: 在 frame A 下, ee 的世界坐标 = base 在 A 中的位置 × ee 相对 base 的位置。在新 frame B 下, 同一个世界坐标 = base 在 B 中的位置 × 新的 ee 相对 base 的位置 $\mathbf{K}_t$。两边相等, 反解 $\mathbf{K}_t$ (Eq.11):

$$
\mathbf{K}_t = (\mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_B}})^{-1} \cdot \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_A}} \cdot \mathbf{T}_t^{\mathrm{ee \to base}}
$$

这套 $\mathbf{K} = \{\mathbf{K}_t\}$ 就是新视角下保持任务一致性的 action sequence。

**训练时的麻烦**: 没有 paired multi-view real data, 怎么 self-supervised? 双重 reprojection trick:
1. 用 MoGe 估计 $\mathcal{W}_A$ 的 scaled depth
2. 把 video warp 到 target view $\mathcal{W}_B$, 再 reproject 回 $\mathcal{W}_A$
3. reprojected video 当 input, 原始 video 当 ground truth
4. robot arm 用 SAM2 mask 掉, 单独用 SAPIEN simulator 渲染 transformed action $\mathbf{K}$ 对应的 arm-only video, 提供 3D guidance

这本质是把 NeRF novel view synthesis 的 warp-and-refine 思想, 塞进 video diffusion 的 conditioning 里。Warp 提供 geometry prior, diffusion 负责 hallucinate warp 不准的部分。

参考资料:
- MoGe (monocular geometry estimation): https://wangrc.site/MoGePage/
- SAPIEN simulator: https://sapien.ucsd.edu/

#### MimicTransfer: 人手变机械臂

YouTube 上第一人称 human hand manipulation video 一抓一大把, 几乎零成本。但你想拿来训机器人, 中间隔着 embodiment gap — 人手长得跟机械臂完全不一样。MimicTransfer 就是把人手 video 翻译成机械臂 video。

**训练时的巧思**: 由于缺 paired (人手, 机械臂) 视频, 直接用机械臂 video 训。两个 condition:
- condition-1: mask 掉机械臂, 留 background
- condition-2: 用原 arm 的 motion 在 simulator 中驱动虚拟 arm, 生成 synthetic "human-like manipulation" video
- target: 原始 unmasked 机械臂 video

**inference 时** (用真 human hand video):
- mask 人手 → condition-1 (留 scene)
- 用 IK 解 human hand end-effector pose 对应的 robot joint angle, simulator render → condition-2
- 两个 condition 喂进去 → 输出机械臂 video

IK 在这里起 cross-embodiment bridge 的作用 — 人手末端位置对应到机械臂关节角度, 实现从 biological hand 到 mechanical arm 的 motion transfer。

参考资料: MimicDreamer 类似工作 https://arxiv.org/abs/2509.22199

---

### Multi-view 怎么搞: 简单粗暴但有效

paper 没搞复杂 cross-attention 架构, 直接把多视角图像**沿 width 拼成 panoramic**, 喂同一个 diffusion model。fine-tune 一下, model 靠 self-attention 在 view 之间 cross-attend, implicit 学会 cross-view geometry。

这跟 video diffusion 中 "把时间当成空间维度" 的 trick 同源 — 不新增 axis, 复用已有架构的 in-context capacity。Figure 7 可视化效果不错, 多视角间几何一致。

参考资料: Emma (Dong et al., 2025) https://arxiv.org/abs/2509.22407

---

## GigaWorld-0-3D: "保证物理和几何不崩" 的那部分

光靠 video generation 不够 — diffusion model 学到的是 visual plausibility, **不保证 physical plausibility**。生成的视频里杯子可能飘起来, 机械臂可能穿过桌子。所以需要一套 3D + physics 模块, 在 geometry 和 dynamics 层面做硬约束。

### 3D-FG: 生成 foreground 物体

基于 Trellis (image-to-3D), 但 Trellis 单独有几个问题: texture 容易 over-saturate (specular highlight 太强), 生成的 asset 没有 real-world scale, 没有物理属性, 没法塞进 physics simulator。

所以 3D-FG 是一个**质量 gated 的 pipeline**, 不是单一 model:
1. **Aesthetic-Checker**: 评估 input image texture richness, 太差的不要
2. **ImageSegChecker** (GPT-4o 驱动): 评估 segmentation 质量, 失败就 retry
3. **MeshGeoChecker**: 从 4 个正交视角渲染生成的 3D asset, 评估 geometric completeness, 不过关就 resubmit
4. 最终 asset export 成 **URDF** format, 兼容 MuJoCo / SAPIEN / Orbit 等 simulator

这套 gating 机制本质上是在做"生成的 asset quality control", 这在 embodied AI 场景下很重要 — 你要的不只是好看的 asset, 还要能塞进 simulator 算 collision 的 asset。

参考资料:
- Trellis: https://trellis3d.github.io/
- Hunyuan3D (alternative): https://arxiv.org/abs/2501.12202

### 3D-BG: 重建背景

3DGS (3D Gaussian Splatting) 是当前主流场景重建方法, 但传统 3DGS 假设 pinhole camera, 在 embodied setting 下的 rolling shutter / distortion camera 不友好。

paper 用 **3DGRUT** (Wu et al., 2025): 每个 3D Gaussian 关联 7 个 representative points (1 中心 + 6 边界), 精确建模 non-pinhole camera。这对真实机器人摄像头至关重要。

但 sparse-view 输入下 NVS 会有 artifact, paper 加一步 **view restoration** — 训一个 model hallucinate 中间视角, 拿去喂 dense 3DGS reconstruction, 最后用 Poisson Surface Reconstruction 转 watertight mesh。

整体策略: generative 补全 sparse-view 的 under-constrained 问题 + analytical reconstruction 保证 geometric accuracy。

参考资料:
- 3DGRUT: https://arxiv.org/abs/2411.11006
- Recondreamer (类似 idea): https://arxiv.org/abs/2411.19548

### 3D-Phys: 可微物理标定 (重点!)

这是我觉得技术上最有意思的一块。

机械臂的 joint friction, PD controller gains 这些参数, 现实中很难直接测量。传统 system identification 用 simulated annealing — 暴力搜参数空间, 慢得要死, labor-intensive。

paper 用 **PINN-based differentiable physics** 三阶段:

**Stage 1**: 真实轨迹 $(\mathbf{a}_{t-1}, \mathbf{s}_{t-1})$ 配合随机采样的物理参数 $(f, p, d)$ — friction, stiffness, damping — 在 simulator 中 rollout, 拿到 simulated state sequence。

**Stage 2**: 训一个 surrogate model $\mathcal{M}_{f,p,d}$, 学会 mimic simulator 的 dynamics。loss 是 MSE:

$$
\mathcal{L}_{\mathrm{surrogate}} = \|\mathcal{M}_{f,p,d}(\mathbf{a}_{t-1}, \mathbf{s}_{t-1}) - \mathbf{s}_t^{\mathrm{sim}}\|^2
$$

**Stage 3**: 固定 surrogate, 用 gradient descent 优化物理参数, 让 surrogate 在真实轨迹上的预测跟真实 state 对齐:

$$
(f^*, p^*, d^*) = \arg\min_{f,p,d} \|\mathcal{M}_{f,p,d}(\mathbf{a}_{1:T-1}, \mathbf{s}_{0:T-1}) - \mathbf{s}_{1:T}^{\mathrm{real}}\|^2
$$

**为什么不能直接对 simulator 做 gradient descent?** 因为大多数 simulator (MuJoCo, SAPIEN) 不可微。PINN surrogate 提供 differentiable pathway — 它学 simulator 的 dynamics, 一旦学会, 就能在 surrogate 上 backprop 优化物理参数。这是 "neural simulator distillation" 的思路, 比 Brax / MJX 这种重写可微 simulator 更通用。

对 manipulated object, 用 **Qwen3-VL agent** 从正交视角图推断 mass, friction, scale 等。

对 deformable object (毛巾, 衣服), 把 spring-mass system 绑到 3DGS particles 上 (PhysTwin 思路), 但 paper 探索 feedforward 推断 spring-mass 参数的 generalizable 版本, 而非 PhysTwin 的 per-scenario optimization。

参考资料:
- PhysTwin: https://arxiv.org/abs/2503.17973
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL

### 3D-Act: 生成 action trajectory

两层 pipeline:

**Simple scenario**: 少量 teleoperation demo → MimicGen 系统性扩展到新 object pose 和 scene layout。MimicGen 是个挺巧的工具, 给它几条 human demo, 它能通过 geometric transform 生成大量变体, 不需要新 human supervision。

**Complex scenario** (contact-rich, multi-step reasoning): teleoperation 数据当 cold-start → RLPD (Ball et al., 2023) fast online RL bootstrap → converged policy 部署生成大规模 trajectory。

**Intuition**: 简单任务靠 demo augmentation 就够, 复杂任务必须上 RL, 但 RL cold-start 是问题, demo 解决 exploration。

参考资料:
- MimicGen: https://arxiv.org/abs/2310.17596
- RLPD: https://arxiv.org/abs/2306.01664

---

## GigaTrain: 训练 infrastructure

训练 video foundation model 算力消耗巨大, paper 给的 Table 2 非常实用, 列了 8×H20 GPU, batch size 32 下各种配置的 time/memory:

| 配置 | Time (s/step) | Memory (MB) |
|---|---|---|
| DeepSpeed-ZeRO2 baseline | 32.84 | 95241 |
| FSDP-2 baseline | 33.19 | 89355 |
| FSDP-2 + FP8 | 29.53 | 71857 |
| FSDP-2 + FP8 + sparse attn | 25.38 | 73131 |
| FSDP-2 + FP8 + sparse + MoE + act ckpt | - | feasible |

几个 takeaway:
1. **FSDP-2 比 DeepSpeed-ZeRO-2 省内存** (~5GB), latency 相当
2. **FP8 在所有 framework 上 consistent 减 memory 和 time** 约 10-15%
3. **Sparse attention (NATTEN) 比 SageAttention 速度快**, 但需要 fine-tuning adaptation (直接 plug-and-play 性能会掉)
4. **MoE 增加参数量, 必须 activation checkpoint** (特别在 FFN 上), 否则 OOM

训练 resolution 是 480×768, 61 frames — 匹配主流 VLA (π0, GigaBrain-0) 的 480p input。

**inference 加速**: denoising step distillation (Yin et al., 2024) 把采样从几十步压到 1 步, 加上 FP8 inference, 总 speedup > 50×。这对大规模 data generation 很关键 — 你不可能跑 50 步 diffusion 生成每一条训练数据。

参考资料:
- GigaTrain (开源): https://github.com/open-gigaai/giga-train
- DMD distillation: https://arxiv.org/abs/2405.14885
- SageAttention: https://arxiv.org/abs/2410.02367

---

## Experiments: 几个关键数字

### PBench Robot Set (Table 3)

GigaWorld-0-Video-Dreamer (2B activated) 跟 14B 的 Cosmos-Predict2 和 Wan2.2 比, 8 个 quality metric + Domain Score + Overall Score:

- GigaWorld-0-Video-Dreamer (2B act.): Overall **82.07**
- Cosmos-Predict2 (14B): 79.88
- Wan2.2 (14B): 78.85
- Wan2.2 (5B): 77.15
- Cosmos-Predict2.5 (2B): 79.95

**用 1/7 的 activated parameter 击败 14B**, 这是 MoE + sparse attention + FP8 pipeline 的 efficiency advantage。

但 caveat: paper 没披露 pretraining data scale, 不知道是否完全 fair comparison。

### DreamGen Bench (Table 4)

在 GR1 dataset 上 fine-tune, 三个 subset (GR1-Env, GR1-Object, GR1-Behavior), 评估 instruction following (Qwen-IF, GPT-IF) + physical actionability (PA, average of PA I 和 PA II from VideoPhy):

- GR1-Env: GigaWorld-0 Qwen-IF 0.966, PA 0.529
- GR1-Object: GigaWorld-0 Qwen-IF 0.920, PA 0.481
- GR1-Behavior: GigaWorld-0 Qwen-IF 0.894, PA 0.446

instruction following 都领先或持平, 但 PA 略低。**这恰好说明 pure video generation 的 limitation** — 它学 visual plausibility, 不学 physical dynamics。这正是 GigaWorld-0-3D 存在的 reason, 用 3D + physics 补这块短板。

### Downstream: GigaBrain-0 真机部署

最关键验证: 用 GigaWorld-0 生成的 data 训练 GigaBrain-0 VLA model, 部署到真实机器人。任务覆盖:

- **Deformable**: Laundry Folding (G1 humanoid, Fig. 18), 衣服这种 deformable object 是 sim2real 难点
- **Rigid**: Paper Towel Preparation (PiPER arm, Fig. 19)
- **Long-horizon**: Table Bussing (PiPER, Fig. 20) 多步任务, Juice Preparation (G1, Fig. 21)
- **Mobile manipulation**: Boxes Moving, Laundry Baskets Moving (G1, Fig. 22-23)

**所有任务都不需要 real-world interaction training**。这是 paper 的 "money shot" — synthetic data 直接 deploy 真机, 涵盖四大类任务。详细 quantitative 在 GigaBrain-0 paper。

参考资料:
- GigaBrain-0 paper: https://arxiv.org/abs/2510.19430
- DreamGen bench: https://arxiv.org/abs/2505.12705
- PBench: https://arxiv.org/abs/2511.00062

---

## 我对这 paper 的几点 concern

诚实地讲, 几个地方我觉得有疑问:

**1. IDM 误差传播没量化**: Dreamer 生成的 video 跟真实 video 有 distribution gap, IDM 在 generated video 上 accuracy 跟 real video 上差多少, paper 没明确量化。如果 IDM 有 systematic bias, 这个 bias 直接灌进 VLA policy, 训出来 policy 永远带这个 bias。

**2. Multi-view 一致性存疑**: Table 3 的 sub-con (subject consistency) 只有 12.6, 比 bg-con 的 66.8 低一个数量级, 说明 frame-to-frame / view-to-view 一致性可能不够好, VLA 训练时可能学到 noise。

**3. 3D-Phys 的 OOD 问题**: PINN surrogate 在 in-distribution 物理参数上能 fit, 但 OOD 参数 (e.g., friction 极低, 或 collision pattern 没见过) 可能 fail。surrogate-based differentiable physics 的通病。

**4. 3D-Act 的 cold-start bottleneck**: RLPD 在 complex scenario 还是靠 teleoperation demo cold-start, 这把 data bottleneck 从 "采大量数据" 变成 "采少量 demo", 改善了但没消除。

**5. 评估缺少 ablation**: paper 整体 pipeline 很复杂, 但缺少 ablation study 说明每个组件 (MoE vs dense, FP8 vs FP16, 3D-Phys vs 不用, AppearanceTransfer 贡献多少) 各自的 marginal contribution。

---

## Broader picture 和 future

这篇 paper 背后的 paradigm shift 很值得注意 — **world model 从 "未来预测器" 变成 "数据生成器"**。

之前 world model (Dreamer 系列, Ha & Schmidhuber 那条线) 主要是 model-based RL 的 imagination engine — agent 在 world model 里 rollout, 学 policy。那是 closed-loop 的, agent 跟 world model 实时交互。

GigaWorld-0 是 **open-loop data generator** — 离线生成大量 data, 喂给 VLA 训练, 没有实时交互。这是一种实用主义的转向, 优先解决当下最痛的 data scarcity 问题。

但 paper conclusion 提到三个 future direction, 我特别看好第一个: **World Model as Interactive Policy Environment**。如果能把 GigaWorld-0 从 open-loop generator 转成 closed-loop interactive environment, agent 可以在里面做 imagination-based planning, 类似 DreamerV3, sample efficiency 会比 offline data generation 高一个数量级。

挑战在于: video generation model 条件于整个 history, 是 non-Markovian 的, 怎么 turn 成 Markovian transition model? 一种可能: 把 world model 拆成 (a) latent state encoder, (b) latent dynamics predictor, (c) reward predictor, 用 video diffusion 当 image decoder。这样 agent 在 latent space 做 planning, 跟 DreamerV3 架构兼容。

参考资料:
- DreamerV3: https://arxiv.org/abs/2301.04104
- World Models (Ha & Schmidhuber 2018): https://worldmodels.github.io/

---

## 一句话总结

GigaWorld-0 把 world model 当 data factory 用: video generation 解 texture, 3DGS + physics 解 geometry 和 dynamics, MoE + FP8 让训练可行, IDM 把 video 转成 (video, action) pair 喂 VLA。最终在真机上 deploy 成功, 涵盖 deformable / rigid / long-horizon / mobile 任务, 不靠任何 real-world interaction training。

它的哲学 hybrid 很 elegant — **neural network 解决 perception-heavy 任务 (texture, appearance), analytical model 解决 constraint-heavy 任务 (collision, friction, gravity)**。这是经典 engineering 思路在 embodied AI 上的最新体现。

Project page: https://giga-world-0.github.io  
GigaBrain-0 (downstream VLA): https://arxiv.org/abs/2510.19430  
GigaTrain (开源): https://github.com/open-gigaai/giga-train

如果你对其中某块 (e.g., PINN surrogate 的具体训练细节, IDM 在 generated video 上的 accuracy, MoE routing 在 video 上的 specialization pattern) 想再展开, 我可以深挖。

---

# GigaWorld-0: 作为 Data Engine 的 World Model 深度解读

Andrej 你好, 这篇 paper 涉及一个我特别感兴趣的 paradigm shift — 把 world model 从 "未来预测器" 重新定位成 "数据生成引擎"。我会从 design philosophy 开始, 逐层下钻到 architecture, 公式, training infrastructure, 最后讨论 experimental results 和我对这个方向的 broader intuition。

---

## 1. High-Level Design Philosophy: 为什么 World Model 当 Data Engine

Embodied AI 的根本 bottleneck 是 data collection cost。真实世界机器人数据采集面临硬件可用性, 安全性, labor cost 三重约束。传统解法是 simulation (MuJoCo, Isaac Sim, SAPIEN), 但 simulation 永远存在 sim2real gap — texture 不真实, lighting 不自然, dynamic 不准。

GigaWorld-0 的核心 insight 在于: **video generation model 在大规模 internet data 上 pretrained 之后, 已经 implicitly 学到了 photorealistic appearance distribution, 通过 conditioning 机制可以把它"注入"到 embodied scenario 中**。而 3DGS + differentiable physics 保证 geometry 和 dynamics 的 consistency。两者 coupling 之后, 你拿到的是一个 texture-rich *and* geometrically consistent *and* physically plausible 的 data factory。

这跟纯 NeRF/3DGS reconstruction 路线或者纯 generative video 路线都不同 — 它是 **generative prior (解 texture) + analytical constraint (解 geometry/physics)** 的 hybrid。

参考资料:
- Cosmos World Foundation Model: https://arxiv.org/abs/2501.03575
- V-JEPA 2 (self-supervised world model): https://arxiv.org/abs/2506.09985
- Sora survey (world model综述): https://arxiv.org/abs/2405.03520

---

## 2. GigaWorld-0-Video: 视频生成分支

### 2.1 GigaWorld-0-Video-Dreamer: Foundation Model

#### 2.1.1 Flow Matching Formulation

paper 采用 flow matching (Lipman et al., 2022) 而非 traditional DDPM/DDIM:

$$
\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t, \mathbf{c}) \tag{1}
$$

变量含义:
- $\mathbf{z}_t$: latent state at flow time $t \in [0, 1]$, $t=0$ 对应 noise, $t=1$ 对应 data
- $\mathbf{c}$: conditioning signal (text embedding from T5 + image embedding)
- $\mathbf{v}_\theta$: learnable velocity field, parameterized by DiT with MoE

**Intuition**: flow matching 学习一个 ODE 的 vector field, 把 simple prior (Gaussian) 连续变换成 data distribution。相比 DDPM, flow matching 的 trajectory 可以是直线 (optimal transport), 训练更稳定, inference 时 ODE solver 步数更少。这对 video 这种高维 latent 至关重要。

参考资料: https://arxiv.org/abs/2210.02747

#### 2.1.2 Latent Representation: 3D-VAE + Patchification

- 3D-VAE (Wang et al., 2025) 做 spatial-temporal compression, ratio = (4, 8, 8) for (T, H, W)
- 输出 16-channel video latents
- 之上再 patchify $1 \times 2 \times 2$ 进一步压缩
- 位置编码用 3D-RoPE (Su et al., 2023), 在 (T, H, W) 三个 axis 上都加 rotation

**Intuition**: 3D-RoPE 让 attention 能感知 spatiotemporal 邻近关系, 而 1D RoPE 只能解 1D 序列。这对 video 这种 grid-structured data 的 extrapolation 和 locality inductive bias 都很关键。

参考资料: https://arxiv.org/abs/2104.09864

#### 2.1.3 Sparse Attention + MoE Backbone

backbone 是 DiT, attention 用 NATTEN (Hassani et al., 2023) 的 neighborhood attention, FFN 替换成 MoE。MoE 公式:

$$
\mathbf{h}_t' = \mathbf{u}_t + \sum_{i=1}^{N_r} g_{i,t} \mathrm{FFN}_i(\mathbf{u}_t) \tag{2}
$$

$$
g_{i,t}' = \begin{cases} s_{i,t}, & \text{if } s_{i,t} \in \mathrm{Topk}(\{s_{j,t} | 1 \leq j \leq N_r\}, K_r) \\ 0, & \text{otherwise} \end{cases} \tag{3}
$$

$$
s_{i,t} = \mathrm{softmax}(\mathbf{u}_t^\top \mathbf{e}_i) \tag{4}
$$

变量:
- $\mathbf{u}_t$: 第 $t$ 个 token 的 FFN input
- $N_r = 4$: routed experts 总数
- $K_r = 2$: 每个 token 激活的 expert 数
- $\mathbf{e}_i \in \mathbb{R}^d$: 第 $i$ 个 expert 的 learnable routing embedding
- $s_{i,t}$: token $t$ 对 expert $i$ 的 affinity score
- $g_{i,t}$: gate value (sparse, 大部分为 0)

**Critical design choice**: 与 DeepSeek-V2 不同, 这里 **不使用 shared expert**。理由是 shared expert 会 forced-share 一部分 capacity, 在 video 这种多模态 semantic (rigid body, deformable, background, foreground) 上反而限制 specialization。4 expert / 2 active 是非常 sparse 的配置 — 等效参数量像 2B, 但 total parameter 接近 4B 左右 (取决于 FFN 占比)。

#### 2.1.4 Load Balance Loss (DeepSeek-V3 style)

为了避免 routing collapse (所有 token 都涌向同一 expert), 用 complementary balance loss:

$$
\mathcal{L}_{\mathrm{Load}} = \alpha \sum_{i=1}^{N_r} f_i P_i \tag{5}
$$

$$
f_i = \frac{N_r}{K_r T} \sum_{t=1}^T \mathbb{1}(s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}, K_r)) \tag{6}
$$

$$
s_{i,t}' = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}} \tag{7}
$$

$$
P_i = \frac{1}{T} \sum_{t=1}^T s_{i,t}' \tag{8}
$$

变量:
- $\alpha = 0.01$: balance factor
- $T$: sequence 中 token 数量
- $f_i$: expert $i$ 被 routed 到的 frequency (在 batch 内平均)
- $P_i$: expert $i$ 的 average normalized affinity
- $\mathbb{1}(\cdot)$: indicator function

**Intuition**: $f_i$ 是 hard count (是否被选中), $P_i$ 是 soft score (被选中的强度)。两者相乘 — 如果某个 expert 选中率高但平均 affinity 低, 或者 affinity 高但选中率低, loss 都会变大。这种 multiplicative form 比 DeepSeek-V2 的 additive form 更 aggressive, 更适合 expert 数量少的场景。

参考资料:
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- NATTEN: https://github.com/SHI-Labs/NATTEN

### 2.2 GigaWorld-0-Video-Dreamer 作为 Data Engine: IDM

paper 的关键 trick: 用 video generation 产 data, 再用 Inverse Dynamics Model (IDM) 把 video "解码"成 action:

$$
\boldsymbol{\theta}_{1:T} = f_{\mathrm{IDM}}(\mathbf{V}) \tag{9}
$$

变量:
- $\mathbf{V} = \{\mathbf{v}_1, \ldots, \mathbf{v}_T\}$: generated video, $\mathbf{v}_t \in \mathbb{R}^{H \times W \times 3}$
- $\boldsymbol{\theta}_t = [\theta_t^{(1)}, \ldots, \theta_t^{(D)}]^\top \in \mathbb{R}^D$: $D$ 个 joint 的 rotation angle at timestep $t$
- $f_{\mathrm{IDM}}$: inverse dynamics network

**Masked Training**: paper 用 SAM2 (Ravi et al., 2024) 把 robotic arm 从背景中 segment 出来, 只把 arm region 喂给 IDM。这避免了 cluttered background 引入的 spurious correlation, 让 IDM focus 在 arm pose estimation 上。

**Intuition**: 这一步非常关键 — 它把"未来视频"解耦成"未来视觉"和"未来动作"两个 supervision signal。前者由 Dreamer 生成 (善于 appearance modeling), 后者由 IDM 推断 (善于 inverse kinematics)。Dreamer 不需要懂 robot kinematics, IDM 不需要懂 texture, 各司其职。

参考资料: 
- SAM2: https://arxiv.org/abs/2408.00714
- 类似 IDM 思路: DreamGen (Jang et al., 2025) https://arxiv.org/abs/2505.12705

---

### 2.3 GigaWorld-0-Video-AppearanceTransfer: 外观迁移

目标: 用 text prompt 修改 video 的 texture / material / lighting, 保持 geometry 和 motion。

**Architecture choice**: paper 明确拒绝 ControlNet, 理由是 MoE backbone 下 ControlNet duplication 会让 parameter count 爆炸。替代方案是 **channel-concat + MLP compression**:

1. depth map 和 normal map 用 3D-VAE encode 成 latent
2. 与 noise latent channel-wise concat
3. 几层 channel-compress MLP 输出最终 latent, 喂给 transformer

depth 来源: VideoDepthAnything (Chen et al., 2025)
normal 来源: LOTUS (He et al., 2024)

**Intuition**: ControlNet 复制 encoder, 适合 dense conditional control (e.g., pose, canny edge)。但 video 的 conditioning 信号 (depth, normal) 本身就是 dense 3D tensor, 跟 noise latent 形状兼容, 直接 concat + compression 更高效, 也避免 train-from-scratch ControlNet 的开销。

参考资料:
- ControlNet: https://arxiv.org/abs/2302.05543
- VideoDepthAnything: https://github.com/DepthAnything/Video-Depth-Anything
- LOTUS: https://arxiv.org/abs/2409.18124

---

### 2.4 GigaWorld-0-Video-ViewTransfer: 视角迁移

这是 paper 中数学最 elegant 的部分。给定 robot 在 world frame $\mathcal{W}_A$ 操作, end-effector pose 为 $\{\mathbf{T}_t^{\mathrm{ee \to base}}\}$, 目标是合成从 $\mathcal{W}_B$ 看到的 video, 其中 robot base 已 relocate。

约束: **world frame 中 end-effector 绝对位置不变** (任务语义保持):

$$
\mathbf{T}_t^{\mathrm{ee \leftrightarrow \mathcal{W}}} = \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_A}} \cdot \mathbf{T}_t^{\mathrm{ee \to base}} = \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_B}} \cdot \mathbf{K}_t \tag{10}
$$

求解 new action $\mathbf{K}_t$:

$$
\mathbf{K}_t = (\mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_B}})^{-1} \cdot \mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_A}} \cdot \mathbf{T}_t^{\mathrm{ee \to base}} \tag{11}
$$

变量:
- $\mathbf{T}^{\mathrm{base \leftrightarrow \mathcal{W}_A}}$: 在原 world frame 下 robot base 的 pose (6-DoF rigid transform)
- $\mathbf{T}_t^{\mathrm{ee \to base}}$: timestep $t$ 时 end-effector 相对 base 的 transform
- $\mathbf{K}_t$: 在新 world frame $\mathcal{W}_B$ 下, end-effector 相对 relocated base 的 transform
- 上标 $\mathrm{ee}, \mathrm{base}, \mathcal{W}$: end-effector, robot base, world frame

**Self-supervised training (double reprojection)**: 由于没有 paired multi-view real data, paper 用巧妙 self-supervision:
- 用 MoGe (Wang et al., 2025) 估计 $\mathcal{W}_A$ 中的 scaled depth
- warp $\mathbf{V}_A$ 到 target view $\mathcal{W}_B$, 再 reproject 回 $\mathcal{W}_A$
- reprojected video 作为 input, $\mathbf{V}_A$ 作为 ground truth
- 把 robot arm 用 SAM2 mask 掉, 单独用 simulator (SAPIEN, Xiang et al., 2020) 渲染 transformed action $\mathbf{K}$ 对应的 arm-only video

**Intuition**: 这是把 NeRF-style novel view synthesis 思想注入 video diffusion 的尝试 — 用 3D-aware warping 提供 geometry prior, 但不强求 3D reconstruction 完全准确 (那是 Recondreamer 路线), 而是让 diffusion model hallucinate 缺失的部分, 同时保持 action consistency。

参考资料:
- MoGe: https://wangrc.site/MoGePage/
- SAPIEN: https://sapien.ucsd.edu/
- 类似工作 EgoDemoGen: https://arxiv.org/abs/2509.22578

---

### 2.5 GigaWorld-0-Video-MimicTransfer: 人手到机械臂

解决 cross-embodiment gap — 把 first-person human hand manipulation video 翻译成 robot arm manipulation video。

**Training trick**: 由于缺少 paired (human hand, robot arm) 视频, 训练时 **只用 robotic arm video**, 通过:
- video condition-1: 用 SAM2 mask 掉 arm, 保留 background
- video condition-2: 用原 arm 的 motion trajectory 在 simulator 中驱动一个虚拟 arm, 生成 synthetic human-like manipulation video
- target: 原始 unmasked robotic arm video

**Inference 时**: input 是 human hand video
- mask 掉 human hand → condition-1
- 用 IK 解 human hand end-effector pose 对应的 robot joint angle, 在 simulator 中 render → condition-2
- 两个 condition 喂给 model → 输出 robotic arm manipulation video

**Intuition**: 这是一个 **decompose-and-synthesize** 的策略 — 把 "human hand manipulation" 解耦成 "scene context" + "motion trajectory", 再让 model 在 robot embodiment 上 re-compose。这里 IK 起到 cross-embodiment bridge 的作用。

参考资料: MimicDreamer (Li et al., 2025) https://arxiv.org/abs/2509.22199

---

### 2.6 Multi-View Generation 的简洁设计

paper 提到 multi-view 通过 **width-dimension concatenation** 实现 — 把多视角图像沿 width 拼成 panoramic input, 复用原 diffusion 架构, 借助 in-context learning 能力在 multi-view data 上 fine-tune。

**Intuition**: 这跟 video diffusion 中 "时间维当成另一维空间" 的 trick 同源。multi-view 不是新 axis, 而是 width 维的 "spatial in-context" — 让 self-attention 在 view 之间 cross-attend, implicit 学会 cross-view geometry。

参考资料:
- Emma (Dong et al., 2025): https://arxiv.org/abs/2509.22407
- RobotTransfer (Liu et al., 2025): https://arxiv.org/abs/2505.23171

---

## 3. GigaWorld-0-3D: 几何与物理分支

### 3.1 GigaWorld-0-3D-FG: Foreground 资产生成

基于 Trellis (Xiang et al., 2025), 但 augmented with:
- **Aesthetic-Checker** (Ma et al., 2025): 评估 input image 的 texture richness
- **ImageSegChecker** (GPT-4o powered): 评估 segmentation 质量
- **MeshGeoChecker**: 从 4 个正交视角渲染, 评估 geometric completeness

如果 check 失败, 自动 retry (修改参数或重新生成 input)。

**Output**: URDF format, 兼容 physics simulator (SAPIEN, MuJoCo, Orbit)。

**Intuition**: Trellis 这类 image-to-3D model 在 internet image 上训练, 生成 asset 没有 real-world scale, 缺 material property, specular highlight 也会 over-saturate。GigaWorld-0-3D-FG 是一个 quality-gated pipeline, 而非单一 model — 这点很重要。

参考资料:
- Trellis: https://trellis3d.github.io/
- Hunyuan3D: https://arxiv.org/abs/2501.12202
- Orbit simulator: https://github.com/NVIDIA-Omniverse/Orbit

### 3.2 GigaWorld-0-3D-BG: 背景重建

核心是 **3DGRUT** (Wu et al., 2025) — 每个 3D Gaussian 关联 7 个 representative points (1 中心 + 6 边界), 支持非 pinhole camera (rolling shutter), 这对 real-world embodied setting 中的 distortion camera 很关键。

sparse-view 输入导致 NVS 有 artifact, paper 用 **view restoration model** (Ni et al., 2024 启发) 先 hallucinate intermediate views, 再做 dense 3DGS, 最后用 Poisson Surface Reconstruction 转成 watertight mesh。

**Intuition**: 这是 "generative 补全 + 重建" 两阶段 — generative 解决 sparse-view 的 under-constrained 问题, 重建提供 geometric accuracy。两阶段耦合比单一 3DGS 在 sparse 场景下更 robust。

参考资料:
- 3DGRUT: https://arxiv.org/abs/2411.11006 (CVPR 2025)
- Recondreamer: https://arxiv.org/abs/2411.19548

### 3.3 GigaWorld-0-3D-Phys: 可微物理标定

这部分我特别感兴趣。robotic arm 的 joint friction, PD controller gains 这些参数现实中很难直接测量, 传统 system identification 用 simulated annealing, 慢且 labor-intensive。

paper 的方案是 **PINN-based differentiable physics**:

1. **Stage 1**: 真实轨迹 $(\mathbf{a}_{t-1}, \mathbf{s}_{t-1})$ 配合随机采样物理参数 $(f, p, d)$ (friction, stiffness, damping) 在 simulator 中 rollout
2. **Stage 2**: 训练 surrogate model $\mathcal{M}_{f,p,d}$ 拟合 simulator dynamics, 用 MSE loss:
$$
\mathcal{L}_{\mathrm{surrogate}} = \|\mathcal{M}_{f,p,d}(\mathbf{a}_{t-1}, \mathbf{s}_{t-1}) - \mathbf{s}_t^{\mathrm{sim}}\|^2
$$
3. **Stage 3**: 固定 surrogate, 用 gradient descent 优化 $(f, p, d)$ 使得:
$$
(f^*, p^*, d^*) = \arg\min_{f,p,d} \|\mathcal{M}_{f,p,d}(\mathbf{a}_{1:T-1}, \mathbf{s}_{0:T-1}) - \mathbf{s}_{1:T}^{\mathrm{real}}\|^2
$$

变量:
- $\mathbf{a}_t$: action at timestep $t$
- $\mathbf{s}_t$: state at timestep $t$
- $f, p, d$: friction, stiffness, damping (待标定参数)
- $\mathcal{M}_{f,p,d}$: parameter-conditioned dynamics surrogate

**Intuition**: 直接对 simulator 做 gradient descent 不可行 (simulator 通常不可微)。PINN surrogate 提供 differentiable pathway — 一旦 surrogate 学会 mimic simulator, 就能在 surrogate 上 backprop 优化物理参数。这是一种 "neural simulator distillation" 的思路, 比 differentiable simulator (e.g., Brax, MJX) 更通用, 因为不需要重写 simulator。

对于 manipulated object, 用 **Qwen3-VL** agent 从正交视角图推断 mass, friction, scale 等。

对于 deformable object, 用 spring-mass system 绑定到 3DGS particles (PhysTwin 思路, Jiang et al., 2025), 但探索 feedforward 推断 spring-mass 参数的 generalizable 版本。

参考资料:
- PhysTwin: https://arxiv.org/abs/2503.17973
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL

### 3.4 GigaWorld-0-3D-Act: 动作生成

两层 pipeline:

**Simple scenario**: 少量 teleoperation demonstration → MimicGen (Mandlekar et al., 2023) 系统性扩展到新 object pose 和 scene layout。

**Complex scenario**: teleoperation 数据作为 cold-start → RLPD (Ball et al., 2023) 快速 online RL bootstrap → converged policy 部署生成大规模 trajectory。

**Intuition**: 这是 hybrid demonstration-augmented RL — 早期靠 demo 探索 efficient region, 后期靠 RL refine。MimicGen 对 rigid-body 简单任务高效, RLPD 对 contact-rich 任务必要。

参考资料:
- MimicGen: https://arxiv.org/abs/2310.17596
- RLPD: https://arxiv.org/abs/2306.01664

---

## 4. GigaTrain: Training Infrastructure

paper 提供了一份非常实用的 training efficiency table (Table 2), 8×H20 GPU, batch size 32:

| Framework | FP8 | Sparse Attn | MoE | Time (s/step) | Memory (MB) |
|---|---|---|---|---|---|
| DeepSpeed-Zero2 | - | - | - | 32.84 | 95241 |
| FSDP-2 | - | - | - | 33.19 | 89355 |
| DeepSpeed-Zero2 | ✓ | - | - | 29.75 | 76419 |
| FSDP-2 | ✓ | - | - | 29.53 | 71857 |
| FSDP-2 | ✓ | ✓ | - | 25.38 | 73131 |
| FSDP-2 | ✓ | ✓ | ✓ (w/ act ckpt) | - | feasible |

**Key takeaways**:
1. FSDP-2 比 DeepSpeed-ZeRO-2 memory 更省 (少 ~5GB), latency 相当
2. FP8 在所有 framework 上 consistent 减少 memory 和 time (~10-15%)
3. Sparse attention (NATTEN) 比 SageAttention (Zhang et al., 2024) 速度优势明显, 但需要 fine-tuning adaptation
4. MoE 增加参数量, 必须 activation checkpoint (特别在 FFN 上) 才能跑得动

**Resolution choice**: 480×768, 61-frame — 匹配主流 VLA model (π0, GigaBrain-0) 的 480p input

**Acceleration**: denoising step distillation (Yin et al., 2024) 把采样从几十步压到 1 步, 加上 FP8 inference, 总 speedup > 50×。

参考资料:
- GigaTrain: https://github.com/open-gigaai/giga-train
- SageAttention: https://arxiv.org/abs/2410.02367
- DMD distillation: https://arxiv.org/abs/2405.14885

---

## 5. Experimental Analysis

### 5.1 PBench Robot Set (Table 3)

GigaWorld-0-Video-Dreamer (2B activated) vs:
- Cosmos-Predict2 14B
- Wan2.2 14B / 5B
- Cosmos-Predict2.5 2B

在 8 个 quality metric (i2v-bg, i2v-s, aes, img, bg-con, mot, sub-con, o-con) + Domain Score + Overall Score 上, GigaWorld-0 以 2B activated 拿到 Overall 82.07, 击败 14B 的 Wan2.2 (78.85) 和 Cosmos-Predict2 (79.88)。

**Intuition**: activated parameter < 1/7 的情况下 outperform, 印证 MoE + sparse attention + FP8 pipeline 在 embodied video 上的 efficiency 优势。但要注意 paper 没披露 pretraining data scale, 不知道是否 fair comparison。

参考资料: 
- PBench: https://arxiv.org/abs/2511.00062
- Wan 2.2: https://arxiv.org/abs/2503.20314

### 5.2 DreamGen Bench (Table 4)

在 GR1 dataset 上 fine-tune, 三个 subset: GR1-Env (29), GR1-Object (50), GR1-Behavior (47)。

metric: Qwen-IF (instruction following via Qwen), GPT-IF (via GPT), PA (Physical Actionability, average of PA I 和 PA II from VideoPhy)

GigaWorld-0-Video-Dreamer 2B 在所有三个 subset 的 Qwen-IF 上都领先或持平 Cosmos-Predict2.5-2B, 但 PA score 略低 (GR1-Behavior 0.446 vs 0.471)。

**Intuition**: PA 评估 physical commonsense, 这正是 pure video generation model 的弱项 — 它学到 visual plausibility, 不一定学 physical dynamics。这正是 GigaWorld-0-3D 存在的理由 — 把 video 生成不能保证的物理 consistency 用 3D + physics 补上。

参考资料: DreamGen (Jang et al., 2025) https://arxiv.org/abs/2505.12705

### 5.3 Downstream: GigaBrain-0 Real-World Deployment

最关键的 validation: 用 GigaWorld-0 生成的 data 训练 GigaBrain-0 VLA model, 部署到真实机器人, 任务包括:
- Laundry Folding (G1 humanoid, Fig. 18) — deformable
- Paper Towel Preparation (PiPER arm, Fig. 19) — rigid
- Table Bussing (PiPER, Fig. 20) — long-horizon mobile manipulation
- Juice Preparation (G1, Fig. 21) — multi-step
- Boxes Moving / Laundry Baskets Moving (G1, Fig. 22-23) — dynamic mobile

**Intuition**: 这是 paper 的 "money shot" — 不靠 real-world interaction 训练就能 deploy 到真实硬件, 涵盖 deformable, rigid, long-horizon, mobile 四大类任务。详细 quantitative 在 GigaBrain-0 paper (https://arxiv.org/abs/2510.19430)。

---

## 6. Broader Intuition 和 Open Questions

### 6.1 World Model 作为 Data Engine 的 paradigm 优势

1. **Decoupling**: VLA model 不需要知道 world dynamics, 只需要 imitation learn from (obs, action) pair。World model 承担 dynamics 学习, VLA 承担 policy 学习。
2. **Compositionality**: GigaWorld-0-Video 和 GigaWorld-0-3D 互补 — 一个解 texture, 一个解 geometry/physics。两者 failure mode 不重叠。
3. **Controllability**: text prompt + camera extrinsics + physics parameter 全部 explicit controllable, 比 random data collection 高效几个数量级。

### 6.2 我对 paper 的几点 concern

1. **IDM 误差累积**: 如果 GigaWorld-0-IDM 在 generated video 上有 systematic bias, 这个 bias 会直接传到 VLA policy。paper 没有量化 IDM 在 generated video (vs real video) 上的 accuracy 差距。
2. **Multi-view 一致性评估缺失**: paper claim multi-view generation 一致, 但 Table 3 的 sub-con (subject consistency) 分数只有 12.6, 说明 view 间/帧间一致性可能仍不够好。
3. **3D-Phys 的 generalization**: PINN surrogate 在 in-distribution 物理参数上能 fit, 但 OOD 参数 (e.g., 极低 friction) 可能 fail。这是 surrogate-based differentiable physics 的通病。
4. **Cold-start for RL in 3D-Act**: RLPD 在 complex scenario 中需要 teleoperation demo cold-start, 这把 bottleneck 转移到 teleoperation, 没有 fully 解决 data scaling。

### 6.3 与相关工作的 positioning

- vs **Cosmos (NVIDIA)**: Cosmos 是 general physical AI world foundation model, GigaWorld-0 更专注 embodied manipulation, 在 PBench Robot Set 上 outperform。
- vs **V-JEPA 2 (Meta)**: V-JEPA 2 用 self-supervised joint-embedding 预测, 不生成 pixel, 更适合 representation learning; GigaWorld-0 是 generative, 直接产 pixel + action。
- vs **Genie / DreamGen**: DreamGen 也用 video world model + IDM 思路, GigaWorld-0 在此基础上加了 3D-Phys + 3D-Act, 把 physics consistency 显式 modeling。
- vs **PhysTwin**: PhysTwin 用 3DGS + spring-mass 做 digital twin, 但是 per-scenario optimization; GigaWorld-0-3D-Phys 探索 feedforward 推断物理参数, 更 generalizable。

### 6.4 Future Direction: World Model as Policy Environment

paper 在 conclusion 提到三个方向:
1. **World model as interactive policy environment** for model-based RL — 让 agent 在 world model 里 safe explore, plan, refine
2. **World model as policy co-designer** — 主动 propose plausible action sequence 或 decompose complex task into subgoals
3. **Closed-loop self-improvement** — real rollout → improve world model → generate better data → improve policy

我特别看好方向 1。当前 GigaWorld-0 是 **open-loop data generator** (生成 offline data, 不与 policy 交互)。如果转成 **closed-loop interactive environment**, 它就变成一个 learnable simulator — agent 可以在 world model 内做 imagination-based planning (类似 Dreamer, Ha & Schmidhuber, 2018), 这会比 offline data generation 更 sample-efficient。

但挑战在于: video generation model 是非 Markovian 的 (条件于整个 history), 如何 turn 成 Markovian transition model 是 open problem。一种可能思路: 把 world model 拆成 (a) latent state encoder 和 (b) latent dynamics + reward predictor, 类似 DreamerV3, 但用 video diffusion 作为 image decoder。

参考资料:
- DreamerV3: https://arxiv.org/abs/2301.04104
- World Models (Ha & Schmidhuber): https://worldmodels.github.io/

---

## 7. 总结

GigaWorld-0 的 contribution 在于把 world model 从 "未来预测器" 重新定义成 "embodied AI data engine", 通过 GigaWorld-0-Video (texture-rich video generation) 和 GigaWorld-0-3D (geometrically consistent + physically grounded scene) 两个互补 stream, 配合 GigaTrain (FP8 + sparse attention + MoE) 让大规模训练可行。最重要的是 downstream validation: VLA policy 在纯合成 data 上训练就能 deploy 到真实机器人, 涵盖 deformable / rigid / long-horizon / mobile 多类任务。

我对其 hybrid generative-analytical philosophy 印象深刻 — 这其实是 "neural network 解决 perception-heavy 任务 + analytical model 解决 constraint-heavy 任务" 这一经典思路在 embodied AI 上的最新体现。后续 closed-loop interactive world model 方向值得 community 共同探索。

paper 的 project page: https://giga-world-0.github.io
GigaBrain-0 (downstream VLA): https://arxiv.org/abs/2510.19430
GigaTrain framework: https://github.com/open-gigaai/giga-train

如果你 (Andrej) 对其中某个 sub-component (e.g., differentiable physics surrogate, IDM masked training, MoE routing in video diffusion) 想更深入讨论, 我可以再展开。
