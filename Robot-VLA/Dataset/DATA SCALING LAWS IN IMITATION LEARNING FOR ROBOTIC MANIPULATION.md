---
source_pdf: DATA SCALING LAWS IN IMITATION LEARNING FOR ROBOTIC MANIPULATION.pdf
paper_sha256: d4766656dafced91f7e29acfd972dfb049c2f9f15dccda0af4758a2aac8052d9
processed_at: '2026-08-03T18:16:56-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲这篇 paper

## 这群人在搞什么？

简单说：**机器人学倒水、摆鼠标这种活儿，到底要喂多少数据才能"举一反三"？**

你想啊，现在教机器人干活，通常是在实验室里用同一个杯子、同一张桌子反复演示几百遍，机器人学会了。但你换个杯子、换个厨房，它就懵了。这帮人想搞清楚：**要收集多少种不同场景、多少种不同物体的数据，机器人才能在从没见过的新地方、面对从没见过的新物体也能把活干好？**

这个能力叫 zero-shot generalization，就是"零样本泛化"——不用再教，直接上手就能干。

## 怎么教机器人的？

他们用了个叫 UMI 的玩意儿，就是一个手持的夹爪，长得像随便一个抓东西的工具。人握着它去倒水、摆鼠标，它会把人的动作记录下来。好处是**不需要真的机器人参与**，人拿着到处跑就能采集数据，比传统 teleoperation 快多了。

然后把采集来的数据喂给 Diffusion Policy 训练——就是现在比较流行的用扩散模型来预测机器人动作序列的方法。他们还加了个 DINOv2 视觉编码器，让机器人能更好地"看懂"场景。

## 三个核心发现

### 发现一：多样化比量大管用得多

这是最重要的结论。他们做了个实验：

- 固定一个环境，换 32 种不同物体训练
- 固定一个物体，换 32 种不同环境训练
- 同时换环境和物体训练

结果发现：**增加环境种类和物体种类，性能提升符合 power law（幂律）**——就是那种"越多越好，但边际收益递减"的曲线。但**增加每个场景的演示数量，到了一定 threshold 就没用了**。

打个比方：教小孩认猫，给他看 32 种不同的猫（橘猫、黑猫、布偶、暹罗……各看几张照片），比给他看同一只橘猫的 1000 张照片强多了。多样性远比重复量重要。

### 发现二：32 个场景就够 train 出能打 90 分的 policy

他们具体测出来：**收集 32 对"环境-物体"组合，每个组合录 50 段演示，总共 1600 段，就能训出 zero-shot 部署的 policy**。

具体怎么算出来的：

- 8 对时 400 段就饱和
- 16 对时 800 段饱和
- 32 对时 1600 段饱和
- 每对约 50 段刚好

这个结论特别 actionable。之前大家搞 robotic 数据集都是奔着百万级去的（Open X-Embodiment 一百万条轨迹），但那是为了 cross-robot transfer。如果只做 single-task，**32 个场景、1600 段演示就够你用了**。

### 发现三：环境泛化比物体泛化难

同样是 power-law 上升，但环境曲线的斜率更平。**换厨房比换杯子难**。这跟直觉一致——环境变化包含光照、背景、干扰物等一堆 factor 同时变化，物体变化相对 isolated。

但他们还发现个有意思的现象：**当你同时换环境和物体时，反而比单独换环境更快达到性能饱和**。因为多样性更高，policy 学得更高效，对每对演示数量的依赖反而更弱。这就是 "diversity is all you need" 的实证支持。

## 他们怎么验证这套 recipe 的？

拿两个新任务验证：叠毛巾（deformable object，软的难搞）和拔充电器（需要 force 和 speed）。按 recipe 收 32 对场景、每对 50 段，**4 个人一下午采集完**。

结果：

- 倒水 85% 成功率
- 摆鼠标 92.5%
- 叠毛巾 87.5%
- 拔充电器 90%

都是在 8 个从没见过的环境、每个环境 2 个从没见过的物体上测的。**一下午的活，换来了 90% 的 zero-shot 泛化**。这在 robotics 里是相当惊艳的数字。

## 模型那边的额外发现

他们还顺手探索了 model scaling：

- **视觉编码器越大越好**：ViT-S 0.66 分 → ViT-B 0.81 → ViT-L 0.90
- **但 action diffusion 的 U-Net 放大没用**：small 0.88，base 0.90，large 反而降到 0.83

直觉解释：机器人感知图像是高维复杂输入，需要大模型抽特征；但输出动作就 8 维左右（7 关节 + 1 夹爪），小网络已经够表达。**瓶颈在"看"，不在"动"**。

还有个反直觉的点：DINOv2 必须 full fine-tune，**frozen 直接用完全失败（0 分），LoRA 也只有 0.72**。说明机器人控制需要改动视觉模型内部很广泛的权重，不能只调表面。

## 对你的 intuition 帮助

如果把 robotics scaling 跟 LLM scaling 类比：

- LLM 的 scaling law 是参数量、数据量、计算量三轴都遵循 power law
- Robotics 的 scaling law 是**多样性轴遵循 power law，重复量轴很快饱和**

这意味着 robotics 现在还处在 LLM 2018-2020 那个阶段——power-law 已经显现，指数还比较大（说明收益还在快速增长期），scaling 性价比高。但 robotics 的 scaling driver 不是"堆算力堆参数"，而是"堆场景堆物体"。这是个范式性的 insight。

实际意义：**每个任务的数据收集成本是 bounded 的**。不是要无限砸钱砸数据，而是要聪明地投资多样性。32 个场景一下午搞定，90% 泛化，这对 industry 部署是实打实的 cost-effective recipe。

参考链接：
- Paper project: https://data-scaling-laws.github.io/
- UMI: https://umi-gripper.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DINOv2: https://dinov2.metademolab.com/

---

# Data Scaling Laws in Imitation Learning for Robotic Manipulation 深度解读

## 一、Paper 的核心 question 与 motivation

这篇 paper 来自 Tsinghua University + Shanghai Qi Zhi Institute + Shanghai AI Lab，作者 Fanqi Lin, Yingdong Hu 等。核心问题非常直接：**robotic manipulation 中是否存在类似 NLP/CV 那样的 data scaling laws？能否通过适当的数据 scaling，让 single-task policy zero-shot 部署到任意同 category 物体、任意环境？**

这里的 framing 很关键：他们**不**追求 task-level generalization（那需要上千个 task），**也不**追求 cross-embodiment generalization（那是 Open X-Embodiment 的方向）。他们聚焦在 single-task 的两个 generalization axis：

- **Environment generalization**：unseen 场景，含 lighting、distractor、background 变化
- **Object generalization**：同 category 但 color/size/geometry 不同的物体

这种 decomposition 很 clean，而且直接对应 real-world deployment 中真正会遇到的两类 variation。

Project page: https://data-scaling-laws.github.io/

## 二、方法栈拆解

### 2.1 Data source — UMI hand-held gripper

他们用 UMI (Universal Manipulation Interface, Chi et al. 2024) 收集 demonstration，这是一个手持式夹爪，通过 SLAM 估计 end-effector pose，从而避免了 embodiment gap（直接用人的手演示，没有 robot 介入）。这点很重要：teleoperation 数据 collection 慢且需要真实 robot，learning from YouTube video 缺 action label，hand-held gripper 是介于两者之间的 sweet spot。

UMI 的 SLAM 依赖要求 environment 有足够 visual feature，这是它的 limitation（dark room、blank wall 会失败，~90% demo 有效）。作者的经验 tip（Appendix B.2）值得记一下：

1. **Random initial pose is crucial** — gripper 的 height/orientation 随机化，否则 policy 过拟合到 specific initial pose。物体初始位置也要尽可能散。
2. **Environment 要有 rich visual features** — 用 Pangolin 可视化工具验证；可加 distractor 或 texture 来同时增加 SLAM feature 和 data augmentation 效果。
3. **Object size 要适中** — 大物体会挡 camera 导致 SLAM 误判 camera 静止。
4. **标准化 behavior** — 不同 collector 的动作模式和时间要一致，减少 multimodality。
5. **Close gripper 时加一点力** — 引入轻微形变。

UMI: https://umi-gripper.github.io/

### 2.2 Policy learning — Diffusion Policy + DINOv2 + Temporal Ensemble

Policy 主体是 Diffusion Policy (Chi et al. 2023)，用 1D CNN U-Net 作为 noise prediction network，DDIM 加速推理。两个关键改进：

**(1) DINOv2 ViT-L/14 visual encoder（full fine-tune）**

这个选择是 paper 中 model scaling 部分最有意思的发现之一。Table 2a 显示：

| Training Strategy | Score |
|---|---|
| DINOv2 ViT-L/14 full fine-tune | 0.90 |
| LfS ViT-L/14 (learning from scratch) | 0.03 |
| Frozen DINOv2 | 0.00 |
| LoRA DINOv2 (rank=8) | 0.72 |

**Pretraining + full fine-tuning 两者缺一不可**。Frozen DINOv2 完全失败（score=0）是 surprising 的——这意味着 DINOv2 的 feature 虽然对 dense prediction 任务很有用，但直接 frozen 接 action head 完全不能 motor control；必须让 gradient backprop 进 ViT。LfS 也几乎失败（0.03），说明 pretraining 提供的 inductive bias 是必要的。LoRA 的 0.72 比 full fine-tune 0.90 低很多，说明 motor control 需要修改 ViT 内部很 broad 的 weights，low-rank 不足以表达这种 modification。

这跟 Hu et al. 2023b "For pre-trained vision models in motor control, not all policy learning methods are created equal" 的结论一致。

Table 2b 显示 visual encoder scaling 有用：

| Encoder | Score |
|---|---|
| DINOv2 ViT-S/14 | 0.66 |
| DINOv2 ViT-B/14 | 0.81 |
| DINOv2 ViT-L/14 | 0.90 |

但 Table 2c 显示 action diffusion U-Net scaling **没用**：

| U-Net size | Score |
|---|---|
| small | 0.88 |
| base | 0.90 |
| large | 0.83 |

这暗示 action distribution 在 single-task 下相对 simple，small U-Net 已经过 parametrization；或者是当前架构不 scale。我个人倾向于后者——diffusion 在低维 action space 上的 capacity bottleneck 在 visual encoder 而不在 denoiser，跟 LLM 中 decoder 才是 bottleneck 完全反过来。

**(2) Temporal Ensemble**

Diffusion Policy 每 $T_1$ 步预测一个长度 $T_2$ 的 action chunk（$T_2 > T_1$），只执行前 $T_1$ 步。chunk 切换处会有 discontinuity 导致 jerky motion。他们借 ACT (Zhao et al. 2023) 的 temporal ensemble：每个 timestep 都 predict 一次，多个重叠预测用 exponential weighting 平均：

$$a_t = \sum_i w_i \cdot a_t^{(i)}, \quad w_i \propto \exp(-\lambda \cdot i)$$

其中 $a_t^{(i)}$ 是第 $i$ 次 predict 在 timestep $t$ 的 action，$\lambda$ 是 adaptation rate（Table 3 中为 -0.01）。这把 chunk 之间的不连续平滑掉。

Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
DINOv2: https://dinov2.metademolab.com/
ACT / ALOHA: https://tonyzhaozh.github.io/aloha/

### 2.3 Evaluation protocol — 严格得有点过分

这部分是 paper 的 methodological highlight：

1. **只在 unseen environment / unseen object 上 test**，不做 in-distribution eval。
2. **Tester-assigned normalized score** 替代 success rate（太 sparse）和 MSE on validation（经常不 correlate，见 Appendix E.1）。任务被拆成 2-3 个 step，每 step 最多 3 分，归一化：

$$\text{Normalized score} = \frac{\text{Total test score}}{3 \times \text{Number of steps}}$$

最大值 1。这个 metric 给出更细的 granularity，能区分"接近成功"和"完全失败"的 policy。

3. **Blind test with shuffle**：21 个 policy 训完后，在一个 environment 里 set 一个 initial pose，随机 shuffle policy 顺序测试；换 pose 再 shuffle 测试。同一 batch 内可比，跨 batch 不可比（因为 environment/initial pose 不同）。这避免了 tester 的 subconscious bias（比如倾向给后训的 policy 打高分）。

总共 15,000+ real-world rollout，40,000+ demonstration——这种规模在 robotics paper 里是 rare 的。

## 三、Scaling 实验设计与发现

### 3.1 形式化

数据 setup：

- $M$ 个 environments: $E_1, E_2, \ldots, E_M$
- $N$ 个同 category objects: $O_1, O_2, \ldots, O_N$
- 每个 environment-object pair 收 $K$ 条 demonstrations: $D_{ij1}, D_{ij2}, \ldots, D_{ijK}$
- Test score $S$ 在 unseen env/obj 上测

要刻画 $S$ 关于 $M, N, K$ 的依赖关系。

### 3.2 Object generalization 实验

固定 1 个 environment，32 个 object，每 object 120 demo。从 32 个 object 里选 $2^m$ 个（$m=0,1,2,3,4,5$，即 1,2,4,8,16,32），每个 object 取 $2^n$ fraction demo（$n=0,-1,-2,-3,-4,-5$，即 100%, 50%, ..., 3.125%）。21 个 policy，每个在 8 个 unseen object 上测 5 trial。

Fig. 2 的关键观察：
- Object 数从 1→8 时 score 飙升（>0.8），从 8→32 提升变缓（>0.9）。
- **Object 多时，per-object demo 数减少影响变小**。8 object 时 12.5% vs 100% 差距明显；32 object 时几乎无差。

### 3.3 Environment generalization 实验

对称地，固定 1 个 object，32 个 environment。Fig. 3：
- 同样 power-law 上升，但**比 object generalization 更难**。小数量时，增加 environment 数带来的增益比增加 object 数小（曲线 slope 更平）。
- Demo fraction 在 50% 和 100% 之间已经 overlap，diminishing return 很早。

### 3.4 Environment + Object 同时 generalize

32 个 environment-object pair（每个 env 一个 unique object）。Fig. 4：
- 同样 power-law。
- **有意思的是**：25% 和 100% demo 的曲线很快 overlap——同时变化 env 和 obj 增加了 data diversity，让 policy 更快 saturate，对 per-pair demo 数的依赖反而更弱。

这是 "diversity is all you need" 的核心 evidence。

### 3.5 Power-law fitting

他们用：

$$Y = \beta \cdot X^{\alpha}$$

取 log：

$$\log Y = \alpha \log X + \log \beta$$

其中：
- $Y$ = optimality gap = $1 - \text{Normalized Score}$（即距离满分 1 的差距）
- $X$ = 训练用 environment 数 / object 数 / environment-object pair 数
- $\alpha$ = scaling exponent（负数，绝对值越大 scaling 越 steep）
- $\beta$ = prefactor（不可约的 base error rate，类似 irreducible loss）

Fig. 5 显示 fit 良好，correlation coefficient $r$ 都比较高。比如 Mouse Arrangement 的 env-object pair scaling 方程给出：要达到 normalized score 0.99，需要约 1191 个 env-object pair。这是个**可外推的 quantitative prediction**，类似 Kaplan et al. 2020 在 LLM 上做的事。

注意：Demo 数 $K$ vs 性能**没有明显 power-law**，而是快速 plateau（Fig. 7 左）。相关系数只有 -0.62 / -0.79。这是跟 LLM scaling 的关键不同——LLM 是 token 数越多 loss 越低，robotics 中 demo 数一旦达到 threshold 就 saturate。

Kaplan scaling law: https://arxiv.org/abs/2001.08361

## 四、Efficient Data Collection Strategy

这是 paper 最 actionable 的部分。

### 4.1 选择 $M$ 和 $N$ 的形式

实际场景下 $N$ 是 $M$ 的 multiple（每 env 多 object）。他们实验 $M=16, N=64$（每 env 4 object）。Fig. 6 heatmap 显示：

- Env 数小时，per-env 多 object 提升 performance。
- **Env 数 ≥ 16 后，per-env 多 object 与 1 object 性能 gap 几乎消失**。

Recommendation：**收集尽可能多 diverse environment，每个 env 只放一个 unique object**。当 env-object pair 数达 32，一般足够 train 出能 generalize 到 novel env + unseen object 的 policy。

### 4.2 选择 $K$（demonstration 数）

Fig. 7：
- $M=16, N=64$ 时，performance 在 total demo = 800 时 plateau。
- $M=N=8$ 时，400 demo plateau；$M=N=16$ 时 800 plateau；$M=N=32$ 时 1600 plateau。

倒推：**per env-object pair 50 demo** 就够（$K=50$）。32 env-object pair × 50 demo = 1600 total。

### 4.3 Strategy 验证

应用到两个新 task：Fold Towels（deformable）和 Unplug Charger（force-intensive）。32 env-object pair，每 pair 50 demo，**4 个 collector 一下午**搞定。

Table 1 结果：

| Task | Score | Success Rate |
|---|---|---|
| Pour Water | 0.922 ± 0.075 | 85.0 ± 19.4% |
| Mouse Arrangement | 0.933 ± 0.088 | 92.5 ± 9.7% |
| Fold Towels | 0.95 ± 0.062 | 87.5 ± 17.1% |
| Unplug Charger | 0.887 ± 0.14 | 90.0 ± 14.1% |

~90% success rate across 8 unseen env × 2 unseen obj。这是一个**惊人的 efficient 数据收集 recipe**。

## 五、Intuition building — 我的几点联想

### 5.1 为什么 power-law 在 diversity 而不在 demo 数？

我的理解：在 imitation learning 中，policy 的 generalization error 主要来自 train/test distribution shift，而非样本估计噪声。增加 demo 数减少的是后者（IRL 类似 sample complexity），但 motor control 任务中每条 demo 信息量已经很高（连续 action trajectory），少量 demo 就能 fit 单模态分布。

增加 env/obj 数减少的是前者——distribution 覆盖。这跟 supervised learning 中 "more data" 主要是覆盖 input distribution 是同质的。power-law 反映的是 "新 env/obj 给 policy 带来的新 information" 随已见 env/obj 数 sublinearly 衰减（类似 LLM 中每个新 token 的 information 衰减）。

### 5.2 为什么 visual encoder scaling 有用，action diffusion U-Net scaling 没用？

这跟 robotic manipulation 的 input/output structure 有关。Input 是高维 image，需要 rich perceptual abstraction；output 是低维 action（7-DoF arm + 1-DoF gripper ≈ 8 维）。**bottleneck 在 perception 而不在 action decoding**。

这跟 LLM 相反——LLM input 是离散 token embedding（已经被 tokenize 抽象过），bottleneck 在 next-token decoder 的容量。Robotics 中 visual encoder 扮演了 "input tokenizer + feature extractor" 的双重角色，scaling 它有 prompt 返回。

这跟近期 vision-language-action model（如 OpenVLA, RT-2）的方向一致——用大 vision-language backbone 作 encoder，small action head 作 decoder。

OpenVLA: https://openvla.github.io/
RT-2: https://robotics-transformer2.github.io/

### 5.3 为什么 single-task 90% zero-shot generalize 是 big deal？

之前 robotic manipulation 的 narrative 是 "policy 只在训练 environment 工作，换 env 要 fine-tune 或 meta-learn"。Open X-Embodiment 是 cross-embodiment 但仍需 fine-tune 到新 env。这篇 paper 证明：**只要数据收集 strategy 正确（diversity over quantity），single-task 也能 zero-shot 跨 env 和 object**。

这对 industry 部署意义重大——意味着每个 task 的数据收集成本是 bounded 的（一下午 + 4 人），而不是需要无限 scaling。

### 5.4 与 Mobile ALOHA、ALOHA Unleashed 的关系

同期工作 ALOHA Unleashed (Zhao et al. 2024) 也用 bimanual teleop + 大量 demo 训 single-task，达到 dexterous 任务高 success rate。区别是 ALOHA 用 puppeting device（仍需 robot 在 loop），UMI 是纯手持无 robot。两者都说明：**single-task + 大量 diverse demo 是当前 path of least resistance**。

Mobile ALOHA: https://mobile-aloha.github.io/
ALOHA Unleashed: https://openreview.net/forum?id=gvdXE7ikHI

### 5.5 Limitation 与 future direction

作者明确指出：
- 只做 single-task，没探索 task-level generalization（需要 language-conditioned + 千 task）。
- 只做 imitation learning，RL 可能进一步提升 capability。
- UMI 有 SLAM 误差，data quality 影响 scaling law。
- 只在 4 个 task 验证，更复杂 dexterous task 可能需要更多 demo。

我的额外 intuition：**这条 power-law 的 exponent $\alpha$ 应该是 task-dependent 的**。Pour Water 和 Mouse Arrangement 的 $\alpha$ 不同，意味着不同 task 的 generalization 难度有结构性差异。如果 build 一个 task taxonomy，把 $\alpha$ 与 task 的 visual complexity / action precision / contact dynamics 关联起来，可能预测新 task 所需数据规模。

### 5.6 跟你的 Neural Scaling Laws 直觉的连接

Kaplan 2020 的 LLM scaling law 是 $L(N) = (N_c/N)^{\alpha_N}$，其中 $N$ 是 parameter 数，$\alpha_N \approx 0.076$。这里 robotics 的 exponent 在 Fig. 5 legend 里能看到，比如 Pour Water env-object pair 是 $Y = 0.276 \cdot X^{-0.396}$，$\alpha = -0.396$。

注意 $|\alpha|$ 比 LLM 大很多——意味着 robotics 中**增加 diversity 的边际收益衰减比 LLM 中增加 parameter 慢**。这跟我们的直觉一致：robotics 还没到 diminishing return 阶段，scaling 还在 "linear-ish" 区间，投资 diversity 性价比高。

## 六、Reference & Further Reading

- Paper project page: https://data-scaling-laws.github.io/
- Diffusion Policy (Chi et al. 2023): https://diffusion-policy.cs.columbia.edu/
- UMI (Chi et al. 2024): https://umi-gripper.github.io/
- DINOv2 (Oquab et al. 2023): https://dinov2.metademolab.com/
- ACT / ALOHA (Zhao et al. 2023): https://tonyzhaozh.github.io/aloha/
- ALOHA Unleashed: https://openreview.net/forum?id=gvdXE7ikHI
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- OpenVLA: https://openvla.github.io/
- RT-2: https://robotics-transformer2.github.io/
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla (Hoffmann et al. 2022): https://arxiv.org/abs/2203.15556
- Mobile ALOHA: https://mobile-aloha.github.io/
- Hu et al. 2023b "Pre-trained vision models in motor control": https://arxiv.org/abs/2302.00049
- RUMs (Etukuru et al. 2024): concurrent zero-shot deployment work

---

如果你想 build intuition 的话，我建议你 mental model 这样：**robotics 当前正处在 LLM 2018-2020 的位置——power-law 已经显现，exponent 还大，scaling 收益还在快速期；但 robotics 的 scaling 不是 parameter-driven，而是 diversity-driven**。这篇 paper 量化了 diversity 在两个 axis（env, obj）上的 exponent，给出了 actionable recipe，让 single-task zero-shot 部署从 "Holy Grail" 变成 "一下午的事"。这点跟 LLM 当年 "more compute = better" 的简单叙事一样具有范式 shifting 的味道。
