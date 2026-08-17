---
source_pdf: RoboWM-Bench A Benchmark for Evaluating.pdf
paper_sha256: 23ecf5c929d1dba98a8baa8573ac388f1207ec57c1d0d24aba0bca9b95c1e37c
processed_at: '2026-08-12T02:11:01-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 RoboWM-Bench

## 一句话版本

现在那些能生成视频的 AI(比如 Sora、Wan、Veo),生成的视频看起来很真,但让机器人照着做,基本做不了。这篇 paper 就是搞了个考场,专门测这件事,然后发现:最厉害的模型在简单任务上也就 50-80%,稍微难点的任务直接挂零。

## 为什么要搞这个 benchmark

现在 video world model 火得一塌糊涂。你给它一张桌面照片加一句"把杯子放到盘子上",它能给你生成一段视频,看起来手(或者机械臂)确实把杯子拿起来放好了。

问题来了:**看起来对,跟真的对,完全是两码事**。

举个例子,Figure 3 里有个 case,Veo 生成的视频里,手指头只是轻轻碰了一下杯子,根本没夹住,结果视频里杯子居然自己飞起来了——视频模型学到了"手接近 → 物体上升"这个 visual pattern,但它不懂"物体上升是因为夹持力大于重力"这个物理道理。

所以你拿这种视频去训练机器人 policy,或者让机器人照着执行,机器人一夹,杯子掉了,任务失败。

之前有没有人测这事?有,但都不太够:
- VBench、PAI-Bench 这类:只看视频"好不好看",打分都很高,但实际上没用
- LVP 那篇拿真机器人跑:靠谱但贵,而且别人复现不了

RoboWM-Bench 的思路是:**别光看视频好看不好看,把它翻译成机器人动作,在仿真里真跑一遍,看任务成不成**。

## 整个流程怎么跑的

想象一个 pipeline,分四步:

**第一步**:给 world model 一张初始场景图 + 任务描述,它生成一段 manipulation 视频。

**第二步**:把这段视频"翻译"成机器人能执行的动作序列。这里分两种情况:
- 如果视频里是人手:用 HaMeR 估计 3D 手部姿态,然后 retarget 成机械臂末端位姿
- 如果视频里是机械臂:用 inverse dynamics model,输入连续两帧,输出中间的 joint action

**第三步**:把真实场景重建到仿真器里。背景用 4D Gaussian splatting,刚体物体用 3D 重建,铰接体和可变形物体直接用 asset 库里的对应物,物体初始位姿用 Megapose 估,相机位姿用 FEEPE 校准。

**第四步**:在重建的仿真环境里执行动作,用 step-level checker 和 task-level success rate 评估。

这套流程跑下来,你就能得到一个干净的结论:**这个 world model 生成的视频,有多少能真正让机器人完成任务**。

## 两个关键技术细节值得说说

### 人手 retargeting 的小改进

之前 Phantom 那篇工作是这么干的:thumb 和 index 指尖中点作为 gripper 位置,thumb 到 index 的方向作为 x 轴,global 手指 configuration 算出 z 轴。

问题是 z 轴这么算会不稳定,gripper 容易歪。

RoboWM-Bench 改成:**先对 thumb 和 index 的 keypoints 拟合一个平面,平面法向作为 z 轴,然后把两个指尖投影到这个平面上,投影点连线作为 x 轴**。这样定义的坐标系紧贴 contact 几何,稳定得多。

gripper 开度也改了:之前用 thumb-index 距离,但有时候 index 不是主接触点,所以改成 **thumb 指尖到其他所有指尖的最小距离**。这个细节很 practical。

### IDM 的两阶段训练

inverse dynamics model 要解决的问题:给两帧图,推断中间机械臂怎么动。

直接用真实数据训?数据少,学不好。直接用仿真数据训?sim 和 real 视觉差异大,迁移差。

他们的方案:**仿真预训练 + 真实微调,而且仿真预训练时把背景 mask 掉,只留机械臂**。

为什么 mask 背景管用?因为 IDM 关心的是"机械臂从 A 到 B 怎么动",背景是干扰项。仿真背景和真实背景纹理、光照完全不同,但机械臂长得一样。把背景 mask 掉,等于强迫 model 学 domain-invariant 特征。

效果:IDM_Real 平均 71.4%,IDM_Sim+Real 平均 95.7%。24 个百分点的提升,直接证明这招管用。

## 实验结果说了什么

### Table 1 的核心数据

人手任务,Wan 2.6 最强,Pick Object 83%,Push Button 100%,但 Fold Towel 只有 40%。Cosmos 最弱,Pour Water 直接 0%。

机械臂任务更惨,大部分模型在 0-20% 徘徊。但 **Cosmos-Finetune(只用 50 条/task 微调)大幅提升**,Close Drawer 从 0% 飙到 90%。

### 三个 trend

**第一,人手视频比机械臂视频执行成功率高很多**。原因有两个:网上人手视频海量,机械臂视频稀少;人手 5 根手指 + 手掌自带丰富几何约束,生成时不容易形变,机械臂 link 容易被扭曲。

**第二,任务越复杂成功率越低**。短 horizon 的 Push Button 还行,长 horizon 的 Put in Drawer 就崩了。这是 autoregressive generation 的老毛病——error 累积。

**第三,微调有用但不彻底**。Cosmos-FT 在大部分任务上大幅提升,但 spatial reasoning 仍然差,经常物体定位不准导致 grasp 失败。说明视频模型的 3D understanding 是 representation bottleneck,光靠数据喂不出来。

### Figure 4 的核心 insight

把 PAI-Bench 的 quality score 和 RoboWM-Bench 的 execution accuracy 画成 scatter plot,你会发现 PAI-Bench 分数都挤在 0.78 附近,execution accuracy 却从 0 散到 100。

**这意味着现有的 video benchmark 已经饱和,无法区分模型的物理能力**。下一步 evaluation 必须走 execution-grounded 路线。

## 这篇 paper 真正想说的

我觉得这篇 paper 在更深层面上揭示了一件事:**当前 video world model 学到的是 pixel correlations,不是 dynamics**。

它能生成"手拿杯子"的视频,是因为它在训练数据里见过无数"手接近 → 物体上升"的 visual pattern。但它不知道物体为什么上升——是 grasp 力?是磁铁?是绳子吊着?这些 physics 它一概不懂。

这跟 LLM 的情况很像:LLM 能生成看起来合理的推理,但它不懂 why。video model 能生成看起来合理的 manipulation,但它不懂 physics。

**视觉真实 ≠ 物理可行**,这是 paper 的核心 thesis,也是给社区的 wake-up call。

## 几个有意思的细节

### Appendix E 的 negative result

他们试过用 Video Depth Anything 给生成视频补深度(因为 retargeting 需要 depth),结果发现:直接用预测的绝对深度,误差很大;用相对深度对齐第一帧 ground truth,改善但仍有误差。最后发现**加了深度估计反而让 pose tracking 变差**,所以干脆不用。

这说明当前 monocular video depth estimation 精度还不够支持 fine-grained manipulation。一个值得深挖的方向。

### Bimanual 任务

Appendix B 加了 Cook(左手拿铲右手拿锅)和 Lift Large Box(双手抬箱子)。Wan 2.6 Cook 50%,Lift Box 60%。双臂协调是 single-arm prior 无法直接 generalize 的,这个数据点很有参考价值。

### Real-to-sim consistency 的 sanity check

10 条成功 trajectory 和 10 条失败 trajectory,replay 到重建仿真里,outcome 100% 一致。这个实验很关键——它证明仿真重建的物理 fidelity 足够高,所有 success/failure 都归因于 world model 视频质量,不是仿真 artifact。把 confounding factor 隔离干净了。

## 我的几点延伸联想

**第一,这条 evaluation 路线跟 LLM 评估的演进是平行的**。LLM 从 perplexity → MMLU → agent task completion,video model 现在从 FID/FVD → VBench → execution success。下一步的 benchmark 会越来越 downstream、越来越 task-grounded。

**第二,这暗示了一个研究方向**:纯 pixel prediction 可能不是通往真正 world model 的路。Yann LeCun 一直说该在 latent space 预测,V-JEPA 2 就在这条路上。也许 manipulation world model 应该是 latent prediction + explicit physics constraint 的 hybrid。

**第三,contact 是 manipulation 的第一性变量**。paper 反复提到 unstable contact、grasp failure。未来 model 可能需要显式的 contact representation——把 contact state 作为 latent variable 注入 generation process,而不是期望 model 从 pixel 里隐式学到 contact dynamics。

**第四,benchmark 本身的 limitation**:IDM 有 ~5% residual error,retargeting 是 greedy 的没考虑 robot workspace 限制,LeHome deformable physics 跨 engine 一致性未验证,task 覆盖偏 tabletop 家用场景。这些都是未来可以改进的点。

**第五,从 Karpathy 你的视角看**,这其实是一个"software 2.0" vs "software 1.0"的问题。纯 data-driven 的 video generation 是 software 2.0 的极致,但物理约束是 software 1.0 的 explicit knowledge。两者如何融合——是 architecture 里加 physics module,还是 loss 里加 physics constraint,还是 data 里加 interaction signal——会是接下来几年 embodied AI 的核心问题。

## 一句话总结

RoboWM-Bench 给社区立了一面镜子:你生成的视频再好看,让机器人跑一遍就露馅。当前 SOTA 在简单 task 上勉强及格,在 deformable、long-horizon、bimanual 上基本全挂。但微调确实有用,说明这条路有希望。benchmark 本身设计得很 solid,real-to-sim 重建 + IDM + retargeting 三件套把"可执行性"变成了可测量的东西,后续 work 可以在这个 infrastructure 上持续 measure progress。

参考链接:
- [RoboWM-Bench 项目主页](https://robowm-bench.github.io/RoboWM-Bench/)
- [Sora Technical Report](https://openai.com/research/video-generation-models-as-world-simulators)
- [Wan 2.1 arXiv](https://arxiv.org/abs/2503.20314)
- [Veo 3 Technical Report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf)
- [Cosmos arXiv](https://arxiv.org/abs/2501.03575)
- [DreamGen arXiv](https://arxiv.org/abs/2505.12705)
- [Phantom arXiv](https://arxiv.org/abs/2503.00779)
- [HaMeR (CVPR 2024)](https://arxiv.org/abs/2404.10811)
- [V-JEPA 2 (Meta FAIR)](https://arxiv.org/abs/2506.09985)
- [PAI-Bench arXiv](https://arxiv.org/abs/2512.01989)
- [VBench arXiv](https://arxiv.org/abs/2311.15535)
- [LVP arXiv](https://arxiv.org/abs/2512.15840)
- [EnerVerse arXiv](https://arxiv.org/abs/2501.01895)
- [WoW arXiv](https://arxiv.org/abs/2509.22642)
- [GR00T N1 arXiv](https://arxiv.org/abs/2503.14734)
- [GigaWorld-0 arXiv](https://arxiv.org/abs/2511.19861)
- [Video Depth Anything](https://arxiv.org/abs/2503.11504)
- [MegaPose arXiv](https://arxiv.org/abs/2212.06870)
- [FoundationPose (CVPR 2024)](https://arxiv.org/abs/2403.08051)

---

# RoboWM-Bench: 从视觉真实到物理执行的范式转变

## 1. 这篇 paper 的核心定位与 motivation

RoboWM-Bench 由 Peking University 团队提出，论文核心 motivation 可以一句话概括：**video world models 生成的视频即使视觉上再真实，也无法直接说明它们对 robotic control 有用——只有把生成的 behavior 真正"跑起来"，才能验证 video world model 是否学到了 physical dynamics**。

这其实触及了当前 video generation 领域一个深层的认识论问题：当前 Sora、Wan、Veo 这类 large-scale video diffusion models 学到的，本质上是 **pixel-level 的 spatiotemporal correlations**，而**物理可执行性（physical executability）是一种隐含在 contact、force、rigidity 等约束下的 latent structure**，二者并不等价。RoboWM-Bench 就是来量化这二者之间的 gap。

参考：[Sora Tech Report (OpenAI)](https://openai.com/research/video-generation-models-as-world-simulators) ｜ [Wan 2.1 arXiv](https://arxiv.org/abs/2503.20314) ｜ [Veo 3 Tech Report (DeepMind)](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf) ｜ [Cosmos arXiv](https://arxiv.org/abs/2501.03575)

## 2. Benchmark 的整体 pipeline（Figure 2 解析）

整个 RoboWM-Bench 的 evaluation pipeline 可以分解成 4 个核心阶段：

```
Initial scene observation + task description
              │
              ▼
    [Video World Model]  → predicted manipulation video
              │
              ├──────────────┬──────────────────────────┐
              ▼              ▼                          ▼
   Human-centric        Robot-centric            Real-to-sim
   retargeting          inverse dynamics         reconstruction
              │              │                          │
              └──────► action sequence ◄──────────────┘
                              │
                              ▼
                   Execute in high-fidelity simulation
                              │
                              ▼
              Step-level checkers + Task-level success
```

这个 pipeline 的设计哲学在于**把"看起来对"翻译为"做起来对"**：从 video pixel 一直贯通到物理引擎里的 rigid body / deformable body simulation，从而任何视觉上看似合理但物理上不可行的细节都会在执行阶段暴露。

参考：[RoboWM-Bench 项目主页](https://robowm-bench.github.io/RoboWM-Bench/)

## 3. 关键技术 1：Human-Centric Retargeting 的形式化

人手视频到 robot end-effector action 的转换是 benchmark 评估的关键模块之一。这里 paper 做了几个非平凡的改进，我用公式形式化一下：

### 3.1 End-effector position

设 thumb fingertip 3D 位置为 $\mathbf{p}_t \in \mathbb{R}^3$，index fingertip 位置为 $\mathbf{p}_i \in \mathbb{R}^3$（由 HaMeR [39] 重建得到），则 gripper 中心位置取：

$$\mathbf{p}_g = \frac{1}{2}(\mathbf{p}_t + \mathbf{p}_i)$$

变量含义：$\mathbf{p}_g$ 是机器人末端夹爪的目标位置。

### 3.2 End-effector orientation（改进的核心）

Prior work（如 Phantom [28]）直接基于 global finger configuration 定义 orientation，会产生不稳定或倾斜的 end-effector pose。RoboWM-Bench 的改进思路是：**用 contact-relevant 几何而非 global 手指姿态来定义坐标系**。

具体步骤：

1. 对 thumb 和 index finger 上的多个 keypoints 拟合一个 plane $\Pi$，得到 plane normal $\mathbf{n} \in \mathbb{R}^3$（即 z-axis）。
2. 把 $\mathbf{p}_t, \mathbf{p}_i$ 投影到 $\Pi$ 上：

$$\mathbf{p}_t' = \mathbf{p}_t - \big((\mathbf{p}_t - \mathbf{p}_0)\cdot \mathbf{n}\big)\,\mathbf{n}$$

其中 $\mathbf{p}_0$ 是 plane $\Pi$ 上任一参考点，$(\cdot)$ 是内积，目的是消除沿 normal 方向的分量，只保留 plane 内的几何关系。

3. 定义坐标系三轴：

$$\mathbf{x} = \frac{\mathbf{p}_i' - \mathbf{p}_t'}{\|\mathbf{p}_i' - \mathbf{p}_t'\|}, \quad \mathbf{z} = \mathbf{n}, \quad \mathbf{y} = \mathbf{z} \times \mathbf{x}$$

这里 $\mathbf{x}$ 指 thumb 指向 index 的方向（contact-relevant direction），$\mathbf{z}$ 是 contact plane 的法向，$\mathbf{y}$ 由右手法则补全。这样定义的好处是坐标系稳定、贴近 contact geometry。

### 3.3 Gripper opening（也是改进点）

Prior work 用 $\|\mathbf{p}_t - \mathbf{p}_i\|$ 作为 gripper 开度，但在 power grasp、lateral grasp 等场景中，index fingertip 不一定是 primary contact point。paper 改为：

$$\delta_{\text{grip}} = \min_{j \in \mathcal{F}\setminus\{t\}} \|\mathbf{p}_t - \mathbf{p}_j\|$$

其中 $\mathcal{F}$ 是所有 5 个 fingertips 的集合，$j$ 遍历除 thumb 之外的所有 fingertips。$\delta_{\text{grip}}$ 表示 gripper 开度。

### 3.4 最后的轨迹平滑

应用 trajectory smoothing + temporal denoising 来稳定 retargeted motion signal——这在 signal processing 意义上等价于一个 low-pass filter，去除 HaMeR 输出的 high-frequency jitter。

参考：[HaMeR (CVPR 2024)](https://arxiv.org/abs/2404.10811) ｜ [Phantom arXiv](https://arxiv.org/abs/2503.00779) ｜ [Masquerade arXiv](https://arxiv.org/abs/2508.09976)

## 4. 关键技术 2：Robot-Centric Inverse Dynamics Model (IDM)

对于 robot manipulation videos，paper 用 IDM 把预测视频帧转回 joint-space action chunk。形式化：

$$\mathbf{a}_t = f_\theta(I_t, I_{t+1})$$

- $I_t, I_{t+1} \in \mathbb{R}^{H\times W\times 3}$：连续两帧 RGB image
- $\mathbf{a}_t \in \mathbb{R}^7$：Franka Panda 7-DoF joint action（不包括 gripper，gripper 通常单独处理）
- $f_\theta$：IDM 网络，architecturally 跟随 DreamGen [23]

这个公式背后的思想是：**两帧之间的视觉差异编码了"发生了什么动作"**——这本质上是一个 video-conditioned inverse problem，对应 forward model $I_{t+1} = g(I_t, \mathbf{a}_t)$ 的逆。

### 4.1 两阶段 IDM 训练策略（关键 insight）

paper 提出了一个 **sim-pretrain + real-finetune** 的两阶段策略，这是从 RL 的 sim-to-real 文献中借鉴的经典 recipe：

**Stage 1 — Simulation pretraining with background masking**：
- 在 LeHome/Isaac Sim 中跑大量随机 Franka trajectories
- 记录 paired (RGB, joint action) 数据
- 关键 trick：**对 background 做 masking**，只保留 robot arm 区域

为什么 background masking 重要？它缓解了 sim-to-real 的 **visual domain gap**——sim 的 background（合成纹理、光照）和 real world 差异大，但 robot arm 几何相似。IDM 只需要学到"frame A → frame B 之间 robot 怎么动"这个 mapping，background 是 nuisance variable，masking 掉等价于 **domain-invariant feature learning**。

**Stage 2 — Real-world finetuning without masking**：
- 收集少量（50 trajectories/task）real Franka 数据
- 不再 mask background，让 IDM 适应真实视觉分布

**Stage 3 — Apply IDM on world-model-generated videos**：因为 generated videos 视觉分布接近 real videos，stage 2 finetune 后的 IDM 可以直接迁移。

### 4.2 IDM 性能验证（Table 2 数据）

| Method | Pick | Pull | Push | Discard | Close | Put-on-Plate | Put-in-Drawer | Avg |
|---|---|---|---|---|---|---|---|---|
| IDM_Real | 70% | 70% | 80% | 70% | 90% | 70% | 50% | 71.4% |
| IDM_Sim+Real | **100%** | 90% | **100%** | 90% | **100%** | **100%** | 90% | **95.7%** |

这 24 个百分点的提升直接证明：**simulation pretraining 提供了 motion priors，real finetuning 弥合 visual gap**。这个结果本身也佐证了 RoboWM-Bench pipeline 的可靠性——即使有 5% 的 residual error，它也主要发生在 shallow grasp 的边缘场景，不影响 benchmark 整体结论的 validity。

参考：[DreamGen arXiv](https://arxiv.org/abs/2505.12705) ｜ [VPT (Baker et al., NeurIPS 2022)](https://arxiv.org/abs/2206.09586) ｜ [Predictive IDM (Tian et al.)](https://arxiv.org/abs/2412.15109) ｜ [LeHome workshop paper](https://robowm-bench.github.io/RoboWM-Bench/)

## 5. 关键技术 3：Real-to-Sim High-Fidelity Reconstruction

这是 RoboWM-Bench 能做到 reproducible 的根本。Pipeline 模块化分解：

```
Real scene  ──►  [4D Gaussian Splatting]  ──►  Background (visual fidelity)
            ──►  [SAM 3D / 3D recon]     ──►  Rigid objects (geometry)
            ──►  [Deformable asset lib]   ──►  Towels, clothes (deformable)
            ──►  [Megapose/FoundationPose] ─► Object initial 6D pose
            ──►  [FEEPE]                   ─► Camera extrinsic calibration
                                  │
                                  ▼
              Reconstructed LeHome scene (physically + visually faithful)
```

### 5.1 各模块的技术选择和理由

- **Background**：4D Gaussian representations [52]，相比 NeRF 在 dynamic 一致性和渲染速度上有优势（参考 World Labs Marble [webpage](https://www.worldlabs.ai/blog/marble-world-model)）
- **Rigid objects**：SAM 3D [11] 做 3D segmentation + reconstruction
- **Articulated / deformable objects**：直接用 [30] 中的 asset pairs，避免从零重建 deformable geometry 这个 hard problem
- **Object 6D pose**：Megapose [26] 和 FoundationPose [51]，前者基于 render & compare，后者 unified 6D pose + tracking
- **Camera pose**：FEEPE [54]，marker-free and learning-free，多次运行取 average

### 5.2 Real-to-sim fidelity 验证（Figure 5）

paper 设计了一个巧妙的 sanity check：拿 10 条 real-world 成功 trajectory 和 10 条失败 trajectory，把它们 replay 到 reconstructed simulation 中，看 success/failure outcome 是否一致。

结果：**所有 7 个 task 都达到 10/10 success consistency 和 10/10 failure consistency**。

这个数字的含义非常重要——它意味着 **real-to-sim 的物理 fidelity 已经高到足以保留 task-critical dynamics**，于是所有在 sim 中测出的 success/failure 都可以归因于 world model 生成视频的质量，而非 simulation 重建的 artifact。这就把 confounding factor 隔离开来了。

参考：[MegaPose arXiv](https://arxiv.org/abs/2212.06870) ｜ [FoundationPose (CVPR 2024)](https://arxiv.org/abs/2403.08051) ｜ [SAM 3D arXiv](https://arxiv.org/abs/2511.16624)

## 6. Task Suite 的设计哲学（Section 3.4）

RoboWM-Bench 的 task suite 不是随便选的，它沿几个 orthogonal 维度系统化设计：

| 维度 | 取值 | 评估的能力 |
|---|---|---|
| Object type | rigid / articulated / deformable | 不同 dynamics 建模 |
| Horizon | short / long | temporal consistency & planning |
| Embodiment | single-arm / bimanual / human-hand | embodiment generalization |
| Action type | pick, push, pour, fold, assemble, ... | contact diversity |

具体任务列表：
- **Rigid + short**：Pick Object, Push Button, Put on Plate, Discard Trash
- **Articulated**：Open/Close Drawer, Turn Off Faucet（要求理解 kinematic chain）
- **Deformable**：Fold Towel, Fold Clothes（要求 non-rigid dynamics）
- **Long-horizon compositional**：Assemble Burger, Put in Drawer（multi-stage planning）
- **Bimanual**：Cook, Lift Large Box, Collaborative Towel Fold（coordination constraints）

这种设计的 intuition 是：**模型的能力不应该被一个 aggregate metric 抹平**。如果一个 model 在 rigid + short 上 80% 但 deformable 上 0%，aggregate 分数会掩盖这个 critical weakness。Step-level + task-level + per-task 三层评估就是为了把这个 capability surface 完整 expose 出来。

## 7. 评估协议：step-level + task-level 双层（Section 3.5）

形式化定义：

设一个 task 有 $K$ 个 predefined key action nodes $\{t_1, t_2, ..., t_K\}$，每个 node 对应一个 semantically meaningful stage（e.g., contact, lift, place, above-drawer, in-drawer, close-drawer）。每个 node 关联一个 predicate $\phi_k(s_{t_k})$，依赖当时的 simulation state $s_{t_k}$。

**Step-level check at node $k$**：

$$\text{StepPass}_k = \mathbb{1}[\phi_k(s_{t_k}) = \text{True}]$$

**Task-level success**：

$$\text{TaskSuccess} = \left(\bigwedge_{k=1}^{K} \text{StepPass}_k\right) \wedge \mathbb{1}[\text{TaskObjective}(s_T) = \text{True}]$$

其中 $T$ 是 trajectory 末时刻，$\text{TaskObjective}$ 是最终任务目标判定（如 object 在 plate 上、drawer 关闭、towel 折好）。

**为什么这个 protocol 有 information gain**：

它把"任务失败"细分为"在哪一步失败"。比如 Open Drawer 任务失败可能是：
1. contact fail：gripper 根本没碰到 drawer handle
2. grasp fail：碰到了但没形成 stable grasp
3. pull fail：grasp 了但 pull direction 不对
4. drawer 没真正打开

每种 failure mode 指向不同的 model deficiency，对 future model design 有 actionable insight。

## 8. 实验结果深度解读（Section 4.2，Table 1）

### 8.1 Human-hand tasks（task level）

| Method | Pick | Push Btn | Put Plate | Pour | Stack | Open Drw | Put Drw | Fold Twl |
|---|---|---|---|---|---|---|---|---|
| Cosmos | 23% | 40% | 15% | 0% | 10% | 10% | 10% | 0% |
| Wan 2.2 | 57% | 80% | 55% | 60% | 40% | 0% | 20% | 0% |
| **Wan 2.6** | **83%** | 100% | **70%** | **80%** | **80%** | **80%** | **80%** | **40%** |
| Veo 3.1 | 73% | 100% | 30% | 60% | 20% | 20% | 60% | 0% |
| LVP | 40% | 70% | 40% | 20% | - | - | - | - |

**关键观察**：

1. **Wan 2.6 是最强 model**（在 paper 后续 qualitative 分析中也是主要 baseline）。Veo 3.1 虽然在 Push Button 上 perfect，但在多数 task 上反而不如 Wan 2.6，这说明 closed-source ≠ 必然更强。
2. **Fold Towel 是最难任务**：最好的 Wan 2.6 也只有 40%。deformable object manipulation 是当前 video world models 的 fundamental bottleneck，因为 cloth 的 state space 是 infinite-dimensional，pixel-level generation 无法捕捉 folding 的力学约束。
3. **Pour Water 在 Cosmos 上是 0%**：流体行为本质上是 particle simulation，video model 完全没学到。

### 8.2 Robot tasks（task level，Cosmos-Finetune 显著提升）

| Method | Close Drw | Pick | Push | Push Btn | Put Plate | Discard | Pull | Put Drw |
|---|---|---|---|---|---|---|---|---|
| Cosmos | 0% | 10% | 10% | 10% | 10% | 0% | 0% | 0% |
| Wan 2.2 | 30% | 10% | 0% | 0% | 0% | 0% | 0% | 0% |
| Wan 2.6 | 50% | 20% | 40% | 40% | 20% | 10% | 0% | 0% |
| Veo 3.1 | 20% | 20% | 10% | 20% | 10% | 0% | 0% | 0% |
| **Cosmos-FT** | **90%** | **50%** | **50%** | **60%** | **40%** | **30%** | **40%** | **20%** |

**Cosmos-FT 的提升幅度惊人**——例如 Close Drawer 从 0% → 90%。这暗示：

- **video world model 的物理一致性是可以通过 data 显著改善的**，即使只用 50 trajectories/task 的小数据
- 但仍未达到 perfect，说明单靠 supervised finetuning 不够，可能需要 reinforcement / interaction-based training

### 8.3 三个 cross-cutting trends

paper 总结的三个 trends（这些是真正 build intuition 的部分）：

**Trend 1: Human-hand > Robot manipulation**

为什么？两个 hypothesis：
1. **Pretraining data bias**：internet 视频里人手操作海量，robot arm 视频稀少。video model 学到的人手 prior 远比 robot arm prior 强。
2. **Geometric stability**：人手在生成视频中保持 stable geometry 的能力更强（5 个 fingers + palm 提供丰富 self-consistency 约束）；而 robot arm 容易出现 link distortion、gripper 形变。

这个 observation 实际上指向一个 deeper 的问题：**world model 的泛化能力受限于 pretraining data 中 embodiment 的分布**。这也是 GR00T、GigaBrain-0 等 VLA model 想通过大规模 robot data pretraining 解决的方向。

参考：[GR00T N1 arXiv](https://arxiv.org/abs/2503.14734) ｜ [GigaBrain-0 arXiv](https://arxiv.org/abs/2510.19430) ｜ [GigaWorld-0 arXiv](https://arxiv.org/abs/2511.19861)

**Trend 2: Success rate ↓ as task complexity ↑**

从 Push Button（短 horizon、单 contact）→ Put in Drawer（多 stage、需要 stable grasp + place + push）成功率显著下降。这反映 **error accumulation in long-horizon prediction**——video model 在每一步都有 small error，long-horizon 下 error 累积成 catastrophic failure。

这是 autoregressive generation 的 fundamental issue，与 LLM 中 long context reasoning 的 difficulty 类比。

**Trend 3: Fine-tuning helps but doesn't solve everything**

Cosmos-FT 显著提升，但 residual physical inconsistency 仍在——尤其是 3D spatial reasoning。说明 video model 的 3D understanding 不是简单的数据问题，而是 **representation bottleneck**：2D pixel 不足以推断 3D 物理交互。

## 9. Perceptual Plausibility vs. Embodied Executability（Section 4.3，关键 insight）

这是 paper 最 conceptual 的一节。Figure 4 的 scatter plot 显示：

- PAI-Bench 的 quality scores 都聚在 ~0.78 附近，几乎 saturate
- RoboWM-Bench 的 execution accuracy 跨度从 0% 到 100%

这给出一个**强结论**：**当前 video world model benchmark 的 perceptual metrics 已经 saturate，无法 discriminative 地反映 model 的物理能力**。

paper 还给出了 PAI-Bench 详细的 quality metrics（Tables 5-8）：Subject Consistency ~95-97, Background Consistency ~93-96, Motion Smoothness ~99, Aesthetic Quality 36-45（比较低！）, Imaging Quality 64-77, Overall Consistency 22-26（也很低）, I2V Subject ~94-98, I2V Background 92-98。

注意 Aesthetic Quality 和 Overall Consistency 这两项数值很低，说明 PAI-Bench 的"高分 saturation"也不是绝对的——只是与物理执行能力相比，这些 metrics 仍然无法 predictive。

**Conceptually，这意味着**：

video generation 评估进入了一个新阶段——**"looks right" 阶段已经基本被攻克，"works right" 阶段刚刚开始**。这与 Yann LeCun 长期主张的"predict in latent space, not pixel space"思路呼应：pixel-level objective 会过拟合到 texture、lighting 等 nuisance，而物理一致性需要更 abstract 的 prediction target。

参考：[V-JEPA 2 (Meta FAIR)](https://arxiv.org/abs/2506.09985) ｜ [VBench arXiv](https://arxiv.org/abs/2311.15535) ｜ [VBench-2.0 arXiv](https://arxiv.org/abs/2503.21755) ｜ [PAI-Bench arXiv](https://arxiv.org/abs/2512.01989) ｜ [WorldModelBench arXiv](https://arxiv.org/abs/2502.20694) ｜ [WorldArena arXiv](https://arxiv.org/abs/2602.08971) ｜ [EWM Bench arXiv](https://arxiv.org/abs/2505.09694)

## 10. 定性 failure mode 分析（Figure 3）

paper 给出几个非常有教育意义的 failure 案例：

### 10.1 "Touch without grasp" failure

在 Put on Plate 任务中，Veo 3.1 生成的视频里 fingers 只是"touch"了 object，没有形成 stable grasp，但视频中 object 居然被 lift 起来了——这是 visual model 在"伪造"物理结果，没有 grasp 力学约束。

这种 failure mode 的本质是：**video model 学到的是"在 pick-and-place 任务中，hand 接近 → 物体上升 → hand 离开"的 visual correlation，但没学到"上升"必须由 grasp 力学支撑**。

### 10.2 "Close instead of open" failure

Open Drawer 任务中，模型生成的 motion 实际上是 close drawer 的方向，但视觉上看起来像在操作 drawer。执行时 simulation 忠实地反映了这个错误——drawer 反而被关上了。

这说明 video model 缺乏 **action semantics grounding**——它能生成"在 drawer 附近的手部运动"，但分不清是 open 还是 close。

### 10.3 Robot structural distortion

robot arm 视频中经常出现 link 扭曲、gripper 形变等 geometric inconsistency。这是因为：
- robot arm 在 pretraining data 中样本稀疏
- robot arm 的 rigid structure 没有作为 explicit constraint 编码进 model
- diffusion model 的 generator 容易破坏 rigid 几何

## 11. RoboWM-Bench 与相关工作的差异化定位

paper 在 Section 2.3 区分了几类 benchmark：

| Benchmark 类别 | 代表 | 评估什么 | 局限 |
|---|---|---|---|
| **Perception-only** | VBench, EvalCrafter, T2VEval | 视觉 fidelity, temporal coherence | 完全不评估物理 |
| **Physical-AI oriented** | PAI-Bench, VBench-2.0, WorldModelBench, WorldArena, EWM-Bench | 加入 physical reasoning VQA | 仍是 perception/diagnostic，无 execution |
| **Real-robot validation** | LVP, "Wow, wo, val!" | 真实 robot 执行 | 难 reproducible，覆盖窄 |
| **RoboWM-Bench** | （本 paper） | real-to-sim + 执行 | 综合、可复现、manipulation-centric |

RoboWM-Bench 的 unique value proposition 在于：**用 real-to-sim 把"physical executability"变成一个 measurable quantity，同时保留 reproducibility**。这是一个 sweet spot——比 real-robot 评估 scalable，比 perception benchmark 严格。

参考：["Wow, wo, val!" arXiv](https://arxiv.org/abs/2601.04137) ｜ [LVP arXiv](https://arxiv.org/abs/2512.15840) ｜ [EnerVerse arXiv](https://arxiv.org/abs/2501.01895)

## 12. Section 4.4 Robustness 验证的两个核心点

paper 用两块实验证明 RoboWM-Bench 本身是可靠的，这一步对 benchmark paper 来说至关重要。

### 12.1 Action extraction accuracy（Table 2）

Human retargeting pipeline 在 real trajectories 上达到 97.1% 平均成功率——证明 HaMeR + retargeting 足以可靠地把真实人手视频转成可执行的 robot action。

Robot IDM 用 Sim+Real 两阶段训练后达到 95.7%——证明 IDM 足以可靠地从真实 robot video 还原 action。

这两点合起来意味着：**如果 world model 生成的视频质量接近 real video，我们的 pipeline 就能 reliable 地从 video 中提取 action**。所以 benchmark 测出的 success rate 反映的是 world model 视频质量，而非 action extraction 噪声。

### 12.2 Real-to-sim consistency（Figure 5）

10 成功 + 10 失败 trajectory 在 real 和 reconstructed sim 中 outcome 100% 一致。

这证明：**sim 的物理 fidelity 足够高，不会引入 spurious success/failure**。

合起来这两个 robustness check 把 benchmark 的 measurement noise 隔离到可控范围（< 5%），保证了 paper 后续结论的 statistical validity。

## 13. Appendix 关键补充材料

### 13.1 Appendix A: Purely simulated robot tasks（Table 3）

完全在 sim 里的 6 个 task：Close Drawer, Push Button, Cut Sausage, Turn Off Faucet, Assemble Burger, Fold Clothes。

观察：
- Fold Clothes 和 Assemble Burger 几乎所有 model 都 0%——long-horizon compositional + deformable 是双 hard
- Turn Off Faucet 全部 0%——articulated object 的 rotation 动作难
- Wan 2.6 在 Close Drawer 上 30%，在 Cut Sausage 上 40%，相对最好

### 13.2 Appendix B: Bimanual human tasks（Table 4）

Cook（左手 spatula + 右手 pan）和 Lift Large Box。Wan 2.6 仍最强（Cook 50%, Lift Box 60%）。

Bimanual 协调的 challenge 在于：两个 hand 必须 spatially 和 temporally coordinated，单 hand model 的 prior 不足以 generalize 到 bimanual。

### 13.3 Appendix E: 关于 depth estimation 的 negative result

这是一个有意思的细节：paper 尝试用 Video Depth Anything [10] 给人手视频补深度（因为 Phantom [28] 需要 ground-truth depth，但生成视频只有第一帧 depth）。

两种 strategy：
1. 直接用 predicted absolute depth → 与 ground truth 差距大（Figure 9a）
2. 用 predicted relative depth + 第一帧 GT depth 对齐 → 改善但仍有非平凡 error（Figure 9b）

**结论：加入 estimated depth 不改善 pose tracking 性能**——所以最终 pipeline 不用 depth。

这个 negative result 本身揭示了一个 insight：**当前 monocular video depth estimation 的精度还不足以支持 fine-grained manipulation tracking**。这是一个值得深挖的研究方向。

参考：[Video Depth Anything](https://arxiv.org/abs/2503.11504) ｜ [Depth Anything (CVPR 2024)](https://arxiv.org/abs/2401.10891)

## 14. 我的延伸思考：这 paper 暴露的根本问题

### 14.1 Video model 是 "correlation engine"，不是 "dynamics engine"

paper 的实验数据本质上证明：当前 video world models 学到的是 spatiotemporal pixel correlations，而不是物理动力学。它们能模仿"pick 任务中 hand 上升时物体也上升"的视觉模式，但不知道**为什么**物体会上升（因为 grasp 力 > 重力）。

### 14.2 这指向几个可能的 future direction

**Direction 1: Joint latent + pixel prediction**

V-JEPA 2 [3] 这条路线——在 latent space 预测 + 跳过 pixel decoding——可能更适合学物理。因为 latent representation 可以丢弃 texture、lighting 等 nuisance，专注 dynamics。

**Direction 2: Hybrid world model + physics engine**

EnerVerse [21] 提出"embodied future-space"——结合 generative prior 和 explicit physics constraints。这种 hybrid 架构可能比纯 generative 更适合 manipulation。

**Direction 3: Interaction-grounded training**

WoW [12] 强调用大规模 embodied interaction data 训练。DreamGen [23] 用 video + IDM 联合训练。这些都指向：**passive video 不够，需要 active interaction data**。这与 RL 中 model-based 的 dream-and-act loop 思想一脉相承。

**Direction 4: Contact-aware representation**

paper 反复提到的 failure mode（contact 不稳定、unstable grasp）暗示：**contact 是 manipulation 的第一性变量**，未来 model 可能需要显式的 contact representation，比如把 contact state 作为 latent variable 注入到 generation process。

### 14.3 Benchmark 本身的更深含义

RoboWM-Bench 实际上提出了一个更 general 的 evaluation philosophy：**对于生成式 AI 的下游 task usefulness，应该通过 downstream task metric 评估，而非 generation metric**。

这与 LLM 评估的演进路径高度类似——从 perplexity（generation metric）→ MMLU（capability proxy）→ real-world agent task completion（true task metric）。

video world model 的评估正在经历同样的演进：FID/FVD → VBench perceptual scores → RoboWM-Bench execution success。

参考：[WoW arXiv](https://arxiv.org/abs/2509.22642) ｜ [DreamGen arXiv](https://arxiv.org/abs/2505.12705) ｜ [Latent Action Pretraining (Ye et al.)](https://arxiv.org/abs/2410.11758) ｜ [R3M arXiv](https://arxiv.org/abs/2203.12601) ｜ [VIP arXiv](https://arxiv.org/abs/2210.00030) ｜ [Ego4D (CVPR 2022)](https://arxiv.org/abs/2110.07058) ｜ [Gen2act arXiv](https://arxiv.org/abs/2409.16283) ｜ [Learning by watching (IROS 2021)](https://arxiv.org/abs/2010.04613) ｜ [DexMV (ECCV 2022)](https://arxiv.org/abs/2108.05877) ｜ [Gemini Robotics Veo Simulator arXiv](https://arxiv.org/abs/2512.10675) ｜ [Grounding Video Models to Actions arXiv](https://arxiv.org/abs/2411.07223) ｜ [Universal Policy via Text-Guided Video Gen (NeurIPS 2023)](https://arxiv.org/abs/2310.03509) ｜ [RoboDreamer arXiv](https://arxiv.org/abs/2404.12377) ｜ [Learning Interactive Real-World Simulators arXiv](https://arxiv.org/abs/2310.06114) ｜ [CogVideo arXiv](https://arxiv.org/abs/2205.15868) ｜ [LaVie (IJCV 2025)](https://arxiv.org/abs/2309.15183) ｜ [LTX-Video arXiv](https://arxiv.org/abs/2501.00103) ｜ [Hunyuan Video arXiv](https://arxiv.org/abs/2511.18870) ｜ [Kling Avatar 2.0 arXiv](https://arxiv.org/abs/2512.13313) ｜ [Seedance 1.0 arXiv](https://arxiv.org/abs/2506.09113) ｜ [TC-Bench arXiv](https://arxiv.org/abs/2406.08656) ｜ [EvalCrafter (CVPR 2024)](https://arxiv.org/abs/2406.09104) ｜ [T2VEval arXiv](https://arxiv.org/abs/2406.09104) ｜ [VMBench arXiv](https://arxiv.org/abs/2410.08953) ｜ [EgoBridge workshop](https://arxiv.org/abs/2505.09700) ｜ [Qwen3-VL arXiv](https://arxiv.org/abs/2511.21631) ｜ [MegaSAM arXiv](https://arxiv.org/abs/2406.12336) ｜ [Emergence of Human-to-Robot Transfer in VLA arXiv](https://arxiv.org/abs/2512.22414) ｜ [Towards Generalist Robot Learning from Internet Video survey](https://arxiv.org/abs/2404.12764) ｜ [Isaac Sim (NVIDIA)](https://developer.nvidia.com/isaac-sim) ｜ [World Labs Marble blog](https://www.worldlabs.ai/blog/marble-world-model)

## 15. 局限性与潜在批评

为平衡起见，几个可能的局限：

1. **Inverse dynamics 的依赖**：IDM 本身是 imperfect predictor（~95% 上限），residual error 会 propagate 到 evaluation 结果。如果 future world model 生成视频质量极高，IDM 的 5% error 可能成为 bottleneck。
2. **Retargeting 是 greedy 的**：human retargeting 没有考虑 robot workspace 限制、self-collision 等。如果生成的 motion 在物理上可行但 retarget 后超出 robot joint limit，会算成 fail，这可能高估了 model 的失败率。
3. **LeHome 的 deformable physics**：deformable object simulation 在不同 engine 中结果可能不一致。即使 RoboWM-Bench 验证了 internal consistency，跨 engine 的 external consistency 仍需更多验证。
4. **任务覆盖仍偏家用 / tabletop**：没有 legged locomotion、whole-body manipulation、tool use with multi-step reasoning 等更复杂场景。
5. **评估的是 "video → action → sim execution" 闭环，但没有评估 "video → policy training" 路径**：后者是 video world model 的另一个主要下游用途（如 DreamGen），未来 benchmark 可扩展到这条路径。

## 16. 总结：RoboWM-Bench 给社区的核心 message

最后我用一个层次化的总结来收束：

**Tactical level（具体技术）**：
- Real-to-sim + IDM + retargeting 组成的 evaluation pipeline 把 video world model 的"可执行性"变成 measurable quantity
- Sim+Real 两阶段 IDM 训练策略值得借鉴
- Human retargeting 的 contact-relevant orientation 定义是一个小的 but meaningful 改进

**Strategic level（领域方向）**：
- Video generation 的 perceptual 评估已 saturate，下一阶段是 execution-grounded 评估
- 当前 SOTA video world models 的物理一致性远未达到 robot control 的要求
- Fine-tuning on manipulation data 显著有效，但仍未解决 representation bottleneck

**Paradigmatic level（认识论）**：
- "Visual realism" 与 "physical executability" 是两个 fundamental 不同的 axes
- 真正的 world model 需要从"pixel correlations"进化到"dynamics understanding"
- 这条 path 可能要融合：generative prior + latent prediction + explicit physics + interaction data

RoboWM-Bench 给这条 path 提供了一个清晰、可复现、可扩展的 measurement infrastructure——这本身就是一个 community-level 的 contribution，因为它让未来的 progress 变得 verifiable。

参考：[RoboWM-Bench 项目主页](https://robowm-bench.github.io/RoboWM-Bench/) ｜ [Paper PDF (arXiv 即将上线后可搜)](https://arxiv.org/abs/)
