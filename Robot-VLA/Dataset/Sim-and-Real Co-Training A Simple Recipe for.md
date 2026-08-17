---
source_pdf: Sim-and-Real Co-Training A Simple Recipe for.pdf
paper_sha256: fcaf870c0e325ae898154af0b093faec4ead174add91a6db60e9a21287f3e292
processed_at: '2026-08-12T06:09:33-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，我换个腔调，不搞 academic summary 了，用大白话讲讲这帮人到底干了啥、为啥这么干、发现了啥让我意外的东西。

---

## 这帮人到底在纠结啥

robot learning 这两年有个挺尴尬的局面：

- 大家都知道 **more data = better policy**，π₀ (https://arxiv.org/abs/2410.24164)、OpenVLA (https://arxiv.org/abs/2406.09246)、GR00T N1 (https://arxiv.org/abs/2503.14734) 这些 foundation model 都证明了这一点。
- 但 real robot data 贵得离谱——50 条 human demo 就要一个 grad student 抓 SpaceMouse 抓一天，还要担心 robot safety、hardware wear、reset 时间。要 scale 到 10k 条就得搞 Open X-Embodiment 这种十几个 lab 联合的大工程 (https://arxiv.org/abs/2310.08864)。
- Sim 里 data 倒是便宜，MimicGen (https://mimicgen.github.io) / DexMimicGen (https://dexmimicgen.github.io) 这种工具能从 10 条 human demo 自动生成 1000 条。但 sim 长得跟 real 不一样，physics 也不准，直接拿 sim policy 去 real 跑经常翻车。

传统的两条路都不好用：
- **Domain randomization** (https://arxiv.org/abs/1703.06907): 在 sim 里疯狂 randomize 光照、texture、friction，希望 real 落在这个 random 分布里。问题是 randomize 范围得人肉 tune，太大 policy 学不会，太小 transfer 不过去。每个 task 都要重新 tune 一遍。
- **System ID / Digital Twin** (https://arxiv.org/abs/2204.02811): 把 sim 调得跟 real 一模一样，3D asset 扫描、物理参数 identification、texture 匹配。问题是每个 task 都要做一遍，不可 scale。

这帮人想：**既然 sim 单独用不行、real 单独又少，那为啥不俩混一起 train？** 这个 idea 其实不算新——RoboCasa (https://robocasa.ai)、RT-X (https://arxiv.org/abs/2310.08864) 这些工作都零星试过。但没人系统研究过：**到底怎么混、混多少、sim 要不要 align real、align 哪些东西**。这篇 paper 就是干这事的。

---

## 实验设置：简单到令人发指

formula 就一行，是 behavioral cloning 的 weighted sum：

$$
\mathcal{L}_{\text{total}}(\theta) = \alpha \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{sim}}) + (1 - \alpha) \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{real}})
$$

人话翻译：policy 网络 θ（就是个 Diffusion Policy，https://diffusion-policy.cs.columbia.edu）要学从 image + proprioception 预测 action。Loss 是 sim data 的 BC loss 乘 α，加上 real data 的 BC loss 乘 (1-α)。

变量含义：
- **θ**: Diffusion Policy 的参数（Panda 用 transformer + ResNet；GR-1 用 ViT + UNet）
- **D_sim**: sim 里用 MimicGen 生成的几千到几万条 trajectory
- **D_real**: 人用 SpaceMouse / MANUS glove teleop 收的 20-50 条 demo
- **α**: sim data 占的比例，这是 paper 里最关键的一个 hyperparameter

实现上 α 不是直接加权 loss，而是当 "minibatch 里从 sim 采样的概率"：
$$
P[(o_i, a_i) \in \mathcal{D}_{\text{sim}}] = \alpha
$$
这两种写法 expectation 下等价，但 sampling 形式更好实现。

就这么简单。没有什么 domain randomization、没有 CycleGAN、没有 teacher-student distillation、没有 progressive training。就是把两堆 data 扔一个 dataloader 里，按 α 控制比例，train 完直接 deploy 到 real robot。

---

## 两堆 sim data 的区别

这是 paper 的 conceptual 核心。他们把 sim data 分成两类：

### Digital Cousin (DC) —— "亲戚"，不是双胞胎

"Digital cousin" 这个词是 Dai et al. (https://arxiv.org/abs/2410.07408) 提出的，意思是 sim 环境 **跟 real 有点像但不必一模一样**。paper 给了精确定义，DC 必须保留 4 个东西：

1. **同一个 robot**（Panda 还是 Panda，GR-1 还是 GR-1）
2. **同一个 task goal**（success check 一致，language instruction 一致）
3. **同一类 object**（cup 都是 cup，但 sim 里的 cup 可以是不同 3D model、不同 texture）
4. **同一类 fixture**（kitchen counter 还是 kitchen counter，cabinet door 还是 cabinet door）

**不**要求的：
- 3D asset 一模一样（不用扫描真实物体）
- physics 参数一样（friction、mass 不用调）
- camera intrinsics 一样（real humanoid camera 有 fisheye，sim 里没建模，照样 work）
- texture / lighting 一样

构建成本：paper 说每个 DC 大概是 100 条 human source demo + MimicGen 放大到 10k（Panda），或者 10 条 source + DexMimicGen 放大到 1k（GR-1）。这是中等 effort，但远比 digital twin 便宜。

### Prior —— "路人甲"，跟 real task 完全无关

定义：在 real task 设计之前就已经存在的 sim dataset，直接拿来用。Paper 用的是 RoboCasa（https://robocasa.ai）里的 60k 条 Panda demo（20 个 task × 3k demos）和自建的 10k GR-1 demo（10 个 task × 1k）。

这些 task 跟 real task 完全不同：
- Real 是 "counter 到 sink"，Prior 里有 "stove 到 counter"、"microwave 到 counter"、"open drawer" 等等
- Object category overlap 很小（real 9 类，prior 66 类，重叠几类而已）

唯一做的 post-processing 是 **把 Prior 的 camera re-render 一下**，让 viewpoint 大致对齐 real camera。这个动作看似 minor，但实验证明非常关键。

---

## 主结果：惊艳到什么程度

| Data | C2SPnP | C2CPnP | CloseDoor | CupPnP | MilkPnP | Pouring | Avg |
|---|---|---|---|---|---|---|---|
| Real only | 44% | 38% | 10% | 65% | 50% | 65% | **45.3%** |
| Real + DC | 67% | 72% | 100% | 95% | 70% | 85% | **81.1%** |
| Real + Prior | 58% | 53% | 100% | 80% | 80% | 70% | **76.8%** |
| Real + DC + Prior | 72% | 72% | 100% | 85% | 80% | 90% | **83.2%** |

（C2SPnP = CounterToSinkPnP，C2CPnP = CounterToCabPnP）

几个让我 "wow" 的点：

1. **Real only 平均 45.3%，加上 DC 直接飙到 81.1%，涨 36 个点**。这种幅度在 robotics paper 里很少见，通常能涨 5-10% 就值得发 paper 了。

2. **Prior 完全不 align 也涨 31 个点**。这是最反直觉的。Prior 的 task 跟 real task 完全不同，object category 也不 overlap，只是同一个 robot、同一个厨房场景的大背景，结果 co-train 也能涨这么多。说明 sim data 起作用的机制不是 "提供 task-specific supervision"，而是 **"提供 broad robot behavior pretraining"**——就像 LLM 在 web text 上 pretrain，下游 task 再 fine-tune。

3. **CloseDoor 从 10% 直接到 100%**。这个 gap 大得吓人。Appendix VIII-L 说即使把 real demo 翻倍到 100 条，Real-only 也只有 80%。说明 50 条 real demo 根本学不会 close door 这个 motion，sim data 提供了 real data 缺失的关键 behavior prior。

4. **DC + Prior 最好，但 DC 单独已经接近天花板**。Prior 在 DC 之上的 marginal 收益其实不大（81 → 83）。这意味着如果你有精力 build DC，Prior 的额外价值就有限了。但如果 DC 太麻烦，Prior 也能拿到大部分收益。

---

## 最反直觉的发现：α 要很大

这个我必须详细讲，因为这是 paper 最 actionable 也很容易被忽略的发现。

α 是 sim data 占的比例。直觉上你会想："real data 才是 ground truth啊，sim data 是 noisy 补充，应该 real 占大头，α 最多 0.3-0.5 吧？"

**完全错。** Figure 5 在 GR-1 CupPnP（20 real + 1k sim DC）上的 sweep：

| α | 成功率 |
|---|---|
| 0.50 | 差 |
| 0.90 | 好 |
| 0.99 | **最佳（95%）** |
| 0.995 | 开始降，~80% |
| 0.999 | 60% |

最优 α = 0.99，意思是 **每个 minibatch 99% 来自 sim，只有 1% 来自 real**。

build intuition 一下：想象 policy 在拟合 π(a|o)。Real data 是 20 条高保真但稀疏的样本，sim data 是 1000 条低保真但稠密的样本。如果 α=0.5，每个 real sample 被采样到的次数是 sim sample 的 50 倍（因为 sim 比 real 多 50 倍）。这意味着 real data 被"过度复用"，policy 会 overfit 到那 20 条 real 的 specific visual feature 和 action pattern，而在 sim 的 diverse 场景上欠拟合。

把 α 调到 0.99，相当于把 real data "稀释"到和 sim data 量级匹配的浓度。Policy 每个 epoch 看到的 real sample 数量跟 α=0.5 时差不多（因为 real dataset 小，总 sample 数被 dataloader 控制了），但 **sim data 的 gradient 信号终于能压过 real data 的过拟合倾向**。

但 α 太高（0.999）又不行，因为 real 的 ground truth 信号被彻底淹没，policy 开始偏向 sim 的 systematic bias（比如 sim 里物体没有质量分布误差、摩擦固定）。

这个 trade-off 跟 LLM 里 data mixing 是同一个故事（Re-Mix https://arxiv.org/abs/2408.14037、data mixing laws https://arxiv.org/abs/2403.08540）。Robotics 社区第一次把它显化出来。

**practitioner take-away**: 别凭直觉设 α，必须 sweep。起步建议 α = 0.9-0.99，data 越不平衡（sim/real ratio 越大）α 越往大调。

---

## Generalization：sim data 不只是 "in-domain 重复"

这个实验设计得很巧妙。问题是：sim data 是不是只是让 policy 在 real demo 覆盖的场景上更准？还是能让 policy generalize 到 real demo 没见过的场景？

### Unseen objects

- Panda C2SPnP: 训练用 9 类 object，测试换 8 类全新 object（carrot、ladle、lime、apple、orange、sponge、cucumber、banana）+ 同类不同 size/color 的 instance
- GR-1 CupPnP: 训练用红 cup，测试换不同颜色 cup + 新 object

| | Panda | GR-1 |
|---|---|---|
| Real | 33% | 10% |
| Real + DC | 50% | 80% |

### Unseen positions

更狠：训练时 real demo 只把 object 放在 workspace 边界/角落（人 teleop 时无意识偏好的位置），sim demo 仍 uniform 覆盖整个区域。测试时把 object 放在中心（real demo 完全没见过的位置）。

| | Panda | GR-1 |
|---|---|---|
| Real | 11% | 43% |
| Real + DC | 28% | **100%** |

GR-1 从 43% → 100%，这简直是魔法。但仔细想想很合理：

**Real demo 50 条（GR-1 只有 20 条）是人 teleop 收的，人会有 sampling bias**——容易把 object 放在 easy-to-reach 的位置，motion pattern 也倾向于走熟悉的轨迹。Policy 学到的是这个 biased distribution。

**Sim data 是 MimicGen 自动生成的，object 位置 uniform random，motion 也更多样**。Co-train 后 policy 看到了更 "complete" 的 conditional distribution π(a|o)，自然 generalize 到 unseen position。

这个发现对 robotics practitioner 意义重大：**sim data 最大的价值可能不是 "in-domain 加成"，而是 "填补 real data 的 distribution hole"**。Real data 永远是 biased 的（人 teleop 就是这样），sim data 是 unbiased 的（randomization 免费），混一起就得到更 robust 的 policy。

---

## Camera alignment：必须 align，但不用完美

这个实验很有意思。DC 默认 camera 跟 real camera 有偏差（Appendix VIII-M 给了具体 delta）：

- Panda third-person camera: position 差 37cm，orientation 差 20°
- Panda wrist camera: position 差 9cm，orientation 差 **180°**（完全反了）
- GR-1 camera: position 差 36cm，orientation 差 60°

paper 把 DC camera re-render 对齐 real 后 vs. 不对齐的 default：

| 任务 | aligned DC | unaligned DC |
|---|---|---|
| Panda C2SPnP | 67% | 56% |
| GR-1 CupPnP | 95% | 70% |

**差 11-25 个点，camera alignment 很关键**。但注意：aligned 也不是 perfect——GR-1 real camera 有 fisheye distortion，sim 里没建模；Panda wrist camera 180° orientation delta 被接受。所以 **"approximate alignment" 就够，不需要 digital twin 级别的精确**。

build intuition：Diffusion Policy 是从 pixel 直接预测 action 的 visuomotor policy，camera viewpoint 错太多的话，sim 里 "object 在画面左侧" 对应的 action，到了 real 里 "object 在画面右侧"，policy 直接学错。视觉 alignment 是 co-training 能 work 的 baseline 要求。

但 camera intrinsics（fisheye distortion）差一点没事，因为 policy 主要学的是 **"画面里 object 位置 → action"** 的 mapping，radial distortion 只是次级 visual feature，policy 能 robust 到。

---

## Dynamics alignment：居然不重要

FAQ 里一句话让我愣了一下：在 GR-1 CupPnP 上做了 dynamics tuning（调 friction、mass 等 physics 参数让 sim open-loop rollout 跟 real 对齐），结果 **success rate 无差别（都 95%）**。

build intuition：PnP、close door 这种 quasi-static manipulation task，robot move-then-grasp-then-place，速度慢、contact 短暂，dynamics 误差积累不严重。Policy 主要靠 visual feedback 闭环纠正，open-loop dynamics 准不准无所谓。

但 paper 在 limitations 里承认：**high-speed task（fast throwing、peg insertion）或 contact-rich task（pouring liquid、deformable object）可能 dynamics 很重要**。这是 paper 没覆盖的边界，practitioner 别盲目推广。

---

## Vid2Vid：锦上添花，不是 game changer

Appendix VIII-K 有个 bonus experiment，用 CogVideo-X (https://arxiv.org/abs/2408.06072) 在 real CupPnP video 上 fine-tune，然后做 sim video → realistic video 的 style transfer。Trick 是从 sim video 加噪到 noise strength = 0.6 作为初始化，再让 diffusion denoise，这样既保留 object position（action label 还能用），又让 visual 更 realistic。

| Real \ Sim | 20 sim | 100 sim | 1000 sim |
|---|---|---|---|
| Real + DC | 48% | 73% | 95% |
| Real + DC w/ V2V | 70% | 80% | 95% |

**Low-data regime（sim 20-100）时 V2V 收益大（+22%、+7%），data-rich 时（sim 1000）几乎无收益**。

build intuition：visual gap 在 co-training 里是 "次级问题"。当 sim data 量足够、覆盖足够广时，policy 自己能学会 robust 到 visual gap（因为 real data 在 anchor visual statistics）。Vid2Vid 只在 sim data 也少、policy 没 "看够" 时才需要。这跟 domain randomization / domain adaptation 文献里 "visual gap 是主要 bottleneck" 的结论很不一样，原因是 co-training 有 real data anchor，policy 不需要单独从 sim 推断 real visual。

---

## Paper 给 practitioner 的 recipe

人话版：

1. **如果有精力 build DC，先 build DC**。DC 是收益最大的 sim data。保留 task definition、object category、fixture category、approximate camera viewpoint 就行。不用扫 3D asset、不用调 physics。

2. **如果没精力 build DC，直接用现成 Prior 也 work**。只要 same robot + same action space，task 不一样也能涨 30 个点。

3. **Sim data 量要够**。500 条 sim 不够，1k-10k 起步。orders of magnitude > real data 是理想状态。

4. **α 必须 sweep**。起步 0.9-0.99，sim/real ratio 越大 α 越大。别凭直觉设 0.5。

5. **Camera viewpoint 要大致 align**，intrinsics 不用完美。Re-render sim data 调 extrinsics 是 high ROI 操作。

6. **Dynamics 不用 align**（至少对 quasi-static task）。

7. **Object diversity + position randomization 是免费午餐**，直接提升 generalization。

---

## 这篇 paper 让我想到的更大的 picture

Karpathy 你可能会对这几个方向感兴趣：

**1. Robotics 的 "data mixing laws"**

NLP 里有 Chinchilla scaling laws (https://arxiv.org/abs/2203.15556)、data mixing laws (https://arxiv.org/abs/2403.08540)。Robotics 一直缺这种 systematic study，这篇 paper 是 early attempt。问题是：**policy performance = f(real_data_size, sim_data_size, α, alignment_level)**，这个 f 是啥形状？paper 给了 discrete 的 data points，但没拟合 continuous law。如果能拟合出来，practitioner 就能 predict "给我 100 条 real + 10k sim + α=0.99，我大概能拿多少 success rate"，不用每次都跑实验。

**2. VLA 时代的 co-training**

paper 用 Diffusion Policy（specialist，per-task training）。现在 trend 是 VLA（OpenVLA、π₀、GR00T N1），已经在 web data + multi-robot data 上 pretrain 过。Sim data 在 VLA 上的 marginal 价值会怎样？

我的直觉：VLA 已经有 broad visual-language pretraining，sim data 的 "broad behavior pretraining" 价值会下降，但 **task-specific DC 仍有价值**（因为 VLA 没见过你这个具体 task）。α 可能需要重新 sweep，因为 VLA 本身就是 "已经 pretrain 过的 policy"，sim data 的角色从 "pretraining" 变成 "fine-tuning 时的 regularizer"。

**3. World model 替代 physics sim**

paper 在 conclusion 提到 Cosmos (https://build.nvidia.com/cosmos)、Genie (https://arxiv.org/abs/2402.15391)、GAIA-1 (https://arxiv.org/abs/2309.17068)。如果 sim data 来自 learned world model 而不是 physics simulator，dynamics gap 自动消失（world model 是从 real video 学的），visual gap 也消失（生成的 video 就是 realistic 的）。Co-training recipe 可能直接 apply，但 data quality、action label 准确性是新问题。

**4. Negative transfer 的边界**

Appendix VIII-M 里提到一个 bimanual task 实验：用单臂 prior data co-train GR-1 bimanual task，policy 学出了单臂行为，success rate ~0%。这是 negative transfer 的典型 case——sim data 的 behavior distribution 跟 real task **冲突** 时，co-training 反而有害。

系统化研究 "什么样的 mismatch 导致 negative transfer" 是个重要 open problem。我的猜想：当 sim data 教的 **action distribution** 跟 real task 的 action distribution 有 conflict（比如 sim 是单臂抓取，real 要双臂协调），policy 会 prefer sim 的 dominant pattern，real data 被淹没。这跟 α 太大的 failure mode 是同一个机制。

**5. Bimanual / long-horizon 的挑战**

paper 主要做 pick-and-place，bimanual、long-horizon 只在 appendix 提了一句（50 real + 1k DC，15% → 50%）。这些 task 的 sim data 生成更难（DexMimicGen 需要更复杂的 segmentation），co-training recipe 是否 still work 未知。我感觉 long-horizon 是 sim data 最有价值的地方——real demo 收集成本随 horizon 线性增长，sim demo 生成成本几乎不变。

---

## 我对这篇 paper 的 overall 评价

**它做对了的事**：
- 在 robotics 社区第一次系统化研究 co-training，扫了多个 factor（data quantity、α、camera alignment、dynamics）
- 两个 domain（Panda + GR-1 humanoid）+ 6 个 task，实验广度够
- Recipe 部分 actionable，practitioner 可以照搬
- Generalization 实验把 co-training 价值从 "in-domain 重复" 提升到 "distribution hole filling"
- Honest about limitations：没说这是 silver bullet，明确 quasi-static task 的适用边界

**它没做好的事**：
- α sweep 只在 CupPnP 上做了，其他 task 的 α curve 未知
- CloseDoor 10% → 100% 这个 dramatic gap 没深入解释
- 没有跟 domain randomization baseline 直接对比（只在 FAQ 说 "complementary"）
- 没理论分析（robotics empirical paper 通病，但 α 的最优值如果能有个理论解释会更有说服力）

**对你的直觉 validate 的部分**：

你一直在各种 talk 里强调 "more data is all you need"、"the bitter lesson"。这篇 paper 在 robotics 上再次验证了这一点——**但 data 不必全部来自 real**。Sim data 是 real data 的 "data augmentation in a much bigger sample space"，只要 mixing ratio 对、alignment 程度够，policy 就能 extract 信号。

这跟 LLM 里 web data + curated data 混合训练是同一个 paradigm，只不过 robotics 版本多了 action space 和 embodiment 这个 dimension。

下一个阶段 robotics foundation model 的核心问题，可能不是 "如何 sim-to-real transfer"（那是 specialist policy 时代的问题），而是 **"如何在 mixed data distribution 上训练 generalist，让它 inference 时自动 focus on target domain"**。Co-training 是这个方向的 empirical 基础，这篇 paper 是第一块砖。

---

# Sim-and-Real Co-Training: 一篇关于 robot manipulation 中混合 sim/real 数据训练的 systematic study

这篇 paper 来自 UT Austin + NVIDIA + UC Berkeley 的联合工作（Nasiriany, Maddukuri, Chen, Zhu, Fan, Mandlekar 等），是 2024-2025 年 robotics 社区对 "co-training" 这个 recipe 的第一次系统化研究。在我看，这篇 paper 的价值不在提出 new method，而在于把 community 已经隐约知道的一些现象（"sim data + real data 一起 train 会有帮助"）做成了 controlled experiments，并提炼出可执行的 recipe。

项目主页: https://co-training.github.io
RoboCasa (主使用的 prior dataset): https://robocasa.ai
MimicGen (sim data 生成): https://mimicgen.github.io
DexMimicGen (bimanual/humanoid 版本): https://dexmimicgen.github.io
Diffusion Policy (基线 policy): https://diffusion-policy.cs.columbia.edu
DROID (Panda Kitchen 真实数据硬件参考): https://droid-dataset.github.io

---

## 1. 高层 motivation 与 positioning

这篇 paper 处于三个 trend 的交汇点：

1. **Generalist robot models 的兴起**: π₀ (https://arxiv.org/abs/2410.24164)、OpenVLA (https://arxiv.org/abs/2406.09246)、GR00T N1 (https://arxiv.org/abs/2503.14734)、RT-2 (https://arxiv.org/abs/2307.15818) 等工作展示了在 web-scale vision-language data + robot data 上训练 generalist VLA 的可行性。
2. **真实机器人数据采集的瓶颈**: 收集一个 task 50 条 human demo 已经是个相当 expensive 的事，scaling 到 100k 级别需要 Open X-Embodiment 这种级别的协作 (https://arxiv.org/abs/2310.08864)。
3. **Sim 中可规模化生成数据**: MimicGen / DexMimicGen / RoboCasa 这套 stack 允许从少量 human demo 自动放大到 10k+ trajectories，且 generative AI 工具 (asset/texture 生成) 让 sim 场景的多样性也大幅提升。

传统 sim-to-real 思路有两个 extreme：
- **Domain randomization** (Tobin et al. IROS 2017, https://arxiv.org/abs/1703.06907): 在 sim 中 randomize 大量参数（光照、texture、dynamics），希望 real 落在 randomize 后的分布里。问题：randomize 范围需要精心 tune，太大 policy 学不好，太小 transfer 不过去。
- **System identification / Digital Twin** (Real2Sim2Real, Ditto https://arxiv.org/abs/2204.02811): 把 sim 精细 align 到 real。问题：需要精细 3D asset、物理参数 identification，对每个 task 都要做一次，不可 scale。

Co-training 是第三条路：**不要求 sim "替代" real，而是用 sim data 作为 real data 的"放大器"**。但这条路一直没有被系统化研究过——什么类型的 sim data 有用？需要多 aligned？mixing ratio 怎么定？camera 要不要 align？这正是这篇 paper 要回答的。

---

## 2. Method 的核心 formulation

paper 用的是 supervised behavioral cloning (BC) 框架，policy π_θ 直接学习从 observation o_i 预测 action a_i。

### 2.1 Co-training loss

公式 (1):

$$
\mathcal{L}_{\text{total}}(\theta; \mathcal{D}_{\text{real}}, \mathcal{D}_{\text{sim}}) = \alpha \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{sim}}) + (1 - \alpha) \cdot \mathcal{L}(\theta; \mathcal{D}_{\text{real}})
$$

其中：

$$
\mathcal{L}(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{(o_i, a_i) \in \mathcal{D}} -\log \pi_\theta(a_i \mid o_i)
$$

变量含义：
- **θ**: policy 网络参数（Diffusion Policy 的 transformer/UNet 参数）
- **D_real = {ξ_i}^N**: 真实机器人收集的 N 条 trajectory demos
- **D_sim = {ξ_i}^M**: sim 中生成的 M 条 trajectory demos（通常 M >> N）
- **o_i**: observation，包括 RGB images + proprioception
- **a_i**: action（Panda 是 7-DoF delta end-effector + gripper；GR-1 是 arm + hand joint positions）
- **α ∈ [0, 1]**: co-training ratio，控制 sim data 的相对权重

### 2.2 实际实现：sampling 形式

paper 强调他们用 **equivalent formulation of α**：把 α 解释为 "每个 minibatch 中从 sim dataset 采样的概率"，而不是直接对 loss 加权：

$$
P[(o_i, a_i) \in \mathcal{D}_{\text{sim}}] = \alpha, \quad P[(o_i, a_i) \in \mathcal{D}_{\text{real}}] = 1 - \alpha
$$

实现上是先按 dataset size 归一化每个 sample 的 weight，再乘 α（sim）或 (1-α)（real）。这两种 formulation 在 expectation 下等价，但 sampling 形式更易与现有 dataloader 兼容，也避免了 gradient magnitude scaling 的 trickiness。

### 2.3 Build intuition：为什么 α 通常要很大（0.9 ~ 0.99）

这是这篇 paper 最反直觉也最重要的发现之一。我们本能会想："real data 才是 ground truth，sim data 是 noisy supplement，应该 real 占主导"。但实验表明 **sim 占 90%~99% 反而最好**，α=0.5 是 suboptimal，α=0.995/0.999 又开始 degrade。

build intuition 的方式：
- 想象 policy training 是在拟合一个 conditional distribution π(a|o)。Real data 提供的是 high-fidelity 但 sparse 的样本，sim data 提供的是 lower-fidelity 但 dense 的样本。
- 当 sim data 是 real data 的 ~100x 时（10k sim vs 50~100 real），如果 α=0.5，每个 real sample 被"复用"的次数比 sim sample 多 ~100x，policy 很快 overfit 到那 50 条 real，并在 sim 的 diverse 场景上欠拟合。
- 把 α 调到 0.99 等价于把 real data "稀释"到和 sim data 相对的合适浓度，让 policy 既能学到 sim 的 diverse priors，又能在 real 上 anchor 住关键 statistics。
- 但 α 太高（0.999+）就几乎不 train real 了，real 的 ground truth 信号被淹没，policy 开始偏向 sim 的 bias。

这个 trade-off 在 NLP/CL 社区的 **data mixing** 研究里也是常见现象（例如 Re-Mix https://arxiv.org/abs/2408.14037、DoReMi、data mixing laws）。Robot learning 这里第一次把它显化。

---

## 3. Data composition factors 的分解

paper 的核心 conceptual 贡献是把 "sim data 和 real data 的 alignment" 拆成了 6 个可量化的 factors（borrowing notation from MimicLabs, https://arxiv.org/abs/2501.15020 大致这一系列工作）：

每个 dataset 被视为一组 factor 分布 $\{\mathcal{Z}^{(1)}, \mathcal{Z}^{(2)}, \dots, \mathcal{Z}^{(K)}\}$，并 explicitly 承认 $\mathcal{Z}^{(i)}_{\text{sim}} \neq \mathcal{Z}^{(i)}_{\text{real}}$，不要求 perfect alignment。

| Factor | 内容 | alignment 重要性 |
|---|---|---|
| Task composition | 哪些 tasks、subtask ordering、motion patterns | 高（DC 的核心） |
| Scene composition | scene 数量、fixtures、articulation、lighting、background textures | 中（prior 提供多样 scene 有利） |
| Object composition | object categories + instances per category | 中-高（同 category 不同 instance 是 DC 的关键） |
| Initialization distribution | 初始 robot pose、joint config、object placement 分布 | 高（DC align 这个 factor） |
| Camera parameters | intrinsics + extrinsics | **高**（实验显示 alignment 带来 11~25% 提升） |
| Dynamics parameters | friction、mass、inertia、controller gains | **低**（FAQ 里提到 dynamics alignment 在 CupPnP 上无差别） |

这个 factor 分解非常实用——它给了 practitioner 一个 checklist：要 build DC 应该先 align 哪几个 factor，哪些可以放手。

---

## 4. 两类 sim data 的对比

这是 paper 的 conceptual framework：

### 4.1 Task-Aware Digital Cousins (DC)

paper 借用并 refine 了 Dai et al. "Automated Creation of Digital Cousins" (https://arxiv.org/abs/2410.07408) 的概念。一个 task-aware digital cousin 必须保留 4 个 elements：

1. **Same robot & action space**（如同样的 Franka 或 GR-1，same DoF 控制）
2. **Same task goal**：success check 一致，language instruction 一致
3. **Same object categories**（但 individual instance 可以 geometry/texture 不同）
4. **Same environmental fixture categories**（kitchen counter、cabinet door 等）

注意它 **不** 要求：
- 一样的 3D asset（不用扫描真实物体）
- 一样的物理参数（friction、mass 可不调）
- 一样的 camera intrinsics（fisheye 等 distortion 可不建模）
- 一样的 texture / lighting

paper 用 MimicGen（Panda）和 DexMimicGen（GR-1 humanoid/bimanual）从 10-100 条 human source demos 自动生成 1k-10k 条 sim demos。生成机制是 object-centric segmentation + linear transform 拼接（详见 https://mimicgen.github.io）。

### 4.2 Task-Agnostic Prior (Prior)

定义：在 real-world task 设计之前就存在的 sim dataset，直接 out-of-the-box 拿来用。本文用的是：

- **Panda Kitchen**: RoboCasa dataset，72k demos over 24 tasks × 100 scenes，每 task 由 50 source human demos 经 MimicGen 放大到 3k。Paper 用了 60k（排除 4 个被 camera alignment 遮挡的 task）。
- **Humanoid Tabletop**: 自建 10-task RoboCasa-based dataset，每 task 1k demos via DexMimicGen，共 10k。

paper 在 Prior data 上做的唯一 post-processing 是 **重新 render camera view 让它大致对齐 real camera**（re-rendering with adjusted extrinsics）。这个动作看似 minor，但实验证明它非常关键。

### 4.3 对比表（基于 Table III, IV 的简化）

| 维度 | Real | DC | Prior |
|---|---|---|---|
| 数据量 | 20~50 / task | 1k~10k / task | 10k~60k total |
| Task 语义 | target task | 与 real 对齐 | 与 real 不同 |
| Object category | 9 类 | 10 类（9/10 overlap） | 66~70 类（大不 overlap） |
| Scene 数量 | 1 | 100 | 100 |
| Human effort | 高 | 中 | 0（用现成） |

---

## 5. Study setup：两个 domain

### 5.1 Panda Kitchen

- Robot: Franka Emika Panda + Deoxys OSC controller (https://github.com/UT-Austin-RPL/deoxys_control)
- Cameras: 2 个 third-person side view + 1 个 eye-in-hand wrist camera
- Tasks:
  - **CounterToSinkPnP (C2SPnP)**: 9 类 object（can、cup、coffee cup、water bottle、lemon、garlic、bowl、granola bar、pear）从 counter 到 sink basin
  - **CounterToCabPnP (C2CPnP)**: 8 类 object 到 cabinet
  - **CloseDoor**: 关 cabinet door，初始角度 [85°, 115°]
- Real demo 数：50 / task
- DC demo 数：10k / task（从 100 source demos 放大）
- Prior demo 数：60k total

### 5.2 Humanoid Tabletop (Fourier GR-1)

- Robot: GR-1，6-DoF dexterous hand，IK controller 用 mink (https://github.com/kevinzakka/mink)
- Camera: head-mounted OAK-D（只用单目 RGB，不用 depth）
- Teleop: MANUS gloves + VIVE tracker
- Tasks:
  - **CupPnP**: cup 从 plate 到 table
  - **MilkPnP**: milk box 从 table 到 shelf 第二层
  - **Pouring**: cup 里有 ping-pong ball，pick up 并 pour 到 bowl
- Real demo 数：20 / task
- DC demo 数：1k / task（从 10 source demos 放大）
- Prior demo 数：10k total

### 5.3 Policy 架构

两边都用 **Diffusion Policy** (https://diffusion-policy.cs.columbia.edu)，但 backbone 不同：

- **Panda**: Transformer-based DP，ResNet visual encoder，3 个 128×170 image view 输入 + proprioception（end-effector pose + gripper）→ 7-DoF delta action。Batch size 256。加 CLIP language conditioning + FiLM layers (https://arxiv.org/abs/1709.07871) 以支持 multi-task。
- **Humanoid**: 用 UMI 的 DP 实现，ViT (https://arxiv.org/abs/2010.11929) vision encoder + UNet (https://arxiv.org/abs/1505.04597) diffusion backbone。Single-task training，无 language。

训练时 default co-training ratio:
- Panda: α = 0.90（real 0.10）
- Humanoid: α = 0.99（real 0.01）

这个差异可能来自 real/sim data 数量比，humanoid sim/real 比 = 50:1，Panda = 200:1（per task DC 数），但 Prior 总量大。

---

## 6. 主结果（Table I 复现）

| Data Composition | C2SPnP | C2CPnP | CloseDoor | CupPnP | MilkPnP | Pouring | Average |
|---|---|---|---|---|---|---|---|
| Real | 44% | 38% | 10% | 65% | 50% | 65% | 45.3% |
| Real + DC | 67% | 72% | **100%** | 95% | 70% | 85% | 81.1% |
| Real + Prior | 58% | 53% | **100%** | 80% | 80% | 70% | 76.8% |
| Real + DC + Prior | 72% | 72% | **100%** | 85% | 80% | 90% | **83.2%** |

**关键 observations**：

1. **Average 提升 38%**（从 45.3% → 83.2%），这是相当 substantial 的 gap，证明 co-training recipe 的有效性。
2. **Prior 单独也有 +31.5%** 提升——这是 paper 最 surprising 的结论。即使 Prior 数据完全没有为 real task 量身定制（task 不同、object category 66 类大不 overlap），co-training 依然 work。
3. **DC + Prior 联合最好**，说明两者是互补的：DC 提供 task-aligned 的 dense supervision，Prior 提供 task-agnostic 的 broad pretraining。
4. **CloseDoor 的极端 case**：Real 只有 10%，co-training 直接拉到 100%。Appendix VIII-L 显示即使把 real demo 翻倍到 100，Real-only 也只有 80%，说明 sim data 在某些 task 上不只是 "数据补充"，而是提供了 real data 缺失的关键 behavior（比如 close door 所需的 specific motion pattern）。

### Build intuition: 为什么 Prior 不 aligned 也有用？

可以这样想：Diffusion Policy 学的是 π(a | o) 的 conditional distribution。即使 Prior 的 task 不同，它的视觉 concepts（gripper、object edge、counter surface、cabinet handle）和低层 motion primitives（reach、grasp、move-to-target）是 task-agnostic 的。Prior data 等价于让 policy 在"广大 robot behavior manifold"上做了 pretraining，real data 再 fine-tune 到具体 task。这与 NLP 里 general pretrain + task fine-tune 的 paradigm 异曲同工，只不过这里 "pretraining" 是在同 modality 的 robot data 上做的，而不是 text。

---

## 7. Generalization 实验（Table II）

paper 进一步问：sim data 不只是 "in-domain 重复"，能不能让 policy generalize 到 real data 没覆盖的场景？

### 7.1 Unseen objects

- Panda C2SPnP: 8 个新 object category（carrot、ladle、lime、apple、orange、sponge、cucumber、banana）+ 同 category 不同 size/color 的 instance
- GR-1 CupPnP: 换不同颜色 cup + 新 objects

| | Panda unseen obj | GR-1 unseen obj |
|---|---|---|
| Real | 33% | 10% |
| Real + DC | 50% | 80% |

### 7.2 Unseen positions

训练时 real demo 只把 object 放在 workspace 边界/角落，sim demo 仍 uniform 覆盖整个矩形。测试时把 object 放在中心（unseen position）。

| | Panda unseen pos | GR-1 unseen pos |
|---|---|---|
| Real | 11% | 43% |
| Real + DC | 28% | **100%** |

**Build intuition**: 这个结果非常 compelling——sim data 在 generalization axis 上的提升比 in-domain 还显著。原因是 sim data 提供了 real data 缺失的"覆盖"。Real data 由于采集成本，往往是 biased sample（人容易把 object 放在 easy-to-reach 的位置），sim data 通过 randomization 填补了 distribution 的 hole。Policy 学到的是更"complete"的 conditional distribution。

GR-1 CupPnP 的 10% → 80% (unseen obj) 和 43% → 100% (unseen pos) 提升幅度比 Panda 大，因为 GR-1 real data 更少（20 vs 50）且 object 多样性更低（fixed 红杯），sim 的 marginal benefit 更大。

---

## 8. 关键因素 ablation

### 8.1 Real demo 数量 scaling（Figure 4, MultiTaskPnP）

- 4 个 multi-task subtask，固定 4k DC demos，real demo 数从 40 → 400 变化
- 即使 real = 400，Real + DC 仍 > Real only，说明 sim data 在 data-rich setting 仍 beneficial
- 这反驳了"co-training 只在 low-data regime 有效"的假设

### 8.2 Sim demo 数量

- Panda C2SPnP: 10k → 500 sim，success 67% → 53%
- GR-1 CupPnP: 1k → 100 sim，success 95% → 75%
- 说明 **sim data 的绝对量很重要**——不是"加一点就有用"，需要达到一定规模

### 8.3 Co-training ratio α（Figure 5）

GR-1 CupPnP, 20 real + 1k sim DC：
- α = 0.50: 较差
- α = 0.90: 好
- α = 0.99: 最佳（95%）
- α = 0.995: 下降到 ~80%
- α = 0.999: 60%

这个曲线很关键，它说明 α 有一个 **窄的最优区间**。Practitioner 必须做 sweep，不能 default 设置了就跑。

### 8.4 Camera alignment（Figure 6）

DC 默认 camera 不 align real camera 时：
- Panda C2SPnP: 67% → 56%（降 11%）
- GR-1 CupPnP: 95% → 70%（降 25%）

但 "aligned" 也不要求 perfect——real humanoid camera 有 fisheye distortion，sim 里没建模，依然 work。所以 **"approximate alignment" 就足够**，这是一个非常 practitioner-friendly 的结论。

Camera pose delta（Appendix VIII-M）：
- Panda third-person: 37cm position, 20° orientation
- Panda wrist: 9cm position, 180° orientation（！）
- GR-1: 36cm position, 60° orientation

可见容忍度相当大。

### 8.5 Dynamics alignment（FAQ）

paper 明确说在 GR-1 CupPnP 上做了 dynamics tuning（调 friction 等），结果**无差别**（都 95%）。这暗示对于 quasi-static manipulation task（PnP、close door 这种），dynamics 不是 bottleneck，**visual + action distribution** 才是。这对 high-speed 或 contact-rich task（insertion、pouring liquid）可能不成立，paper 在 limitations 里也承认了。

---

## 9. Vid2Vid 实验（Appendix VIII-K，Table VI）

这是 paper 的一个 bonus experiment：用 CogVideo-X (https://arxiv.org/abs/2408.06072) 在 real CupPnP video 上 fine-tune，然后做 **sim video → realistic video** 的 style transfer。关键 trick：从 sim video 加噪到 noise strength = 0.6 作为初始化，再让 diffusion denoise，这样既保留 object position（action label 仍可用），又增加 visual realism。

结果（CupPnP, Real + DC）:

| Real \ Sim | 20 | 100 | 200 | 500 | 1000 |
|---|---|---|---|---|---|
| Real + DC | 48 | 73 | 85 | 88 | 95 |
| Real + DC w/ V2V | 70 | 80 | 88 | 93 | 95 |

观察：
- **Low-data regime（sim 20-100 或 real 1-10）时 V2V 收益最大**（+22%, +20%）
- **Data-rich 时 V2V 收益 marginal**（sim 1000 或 real 20+ 时几乎无差）
- 暗示 visual gap 在 co-training 里是 "次级问题"——当 sim data 量足够、覆盖足够广时，policy 自己能学会 robust 到 visual gap

这与 domain randomization / domain adaptation 文献里 "visual gap 是主要 bottleneck" 的结论形成对比，原因是 co-training 同时有 real data anchor，policy 不需要单独从 sim "推断" real visual。

---

## 10. Paper 提炼的 Recipe（Section V-E）

按重要性排序：

1. **Task & scene composition**: 用 DC 最好，但 Prior 也 work。组合两者最佳。
2. **Object composition & initialization**: sim 中要 diverse objects + varied placements，这直接决定 generalization 能力。
3. **DC alignment requirements**:
   - 必须：same task definition + same success criteria
   - 应该：similar camera viewpoints（不需 perfect）
   - 可选：dynamics alignment（quasi-static task 不需要）
4. **Co-training hyperparameters**:
   - Sim data 量：orders of magnitude > real data（10x-200x）
   - α：必须 sweep，建议从 0.9-0.99 起步，data 越不平衡 α 越大

---

## 11. Limitations 与未来方向

paper 自己列的 limitations：
- 主要 pick-and-place，未覆盖 high-precision insertion、long-horizon
- Policy 性能仍非 perfect（即使 co-training 后平均 83%）
- Deformable object / liquid 难 sim，co-training 受限

我会补充几个 paper 没明说但我觉得是 open question 的：

1. **Action space 跨 embodiment**：paper 强调 same robot & action space 是 DC 的要素。如果 sim 是 Panda，real 是 GR-1，能不能 co-train？这需要 action embedding 的对齐，可能是 future foundation model 的方向。
2. **α 的自动 tuning**：目前需要 sweep，对每个 task 都重做。能不能用 dataset size ratio 自动设置 α？NLP 里有 data mixing laws 的工作可借鉴（https://arxiv.org/abs/2403.08540）。
3. **World model 路径**：paper 在 conclusion 提到 Cosmos (https://build.nvidia.com/cosmos)、Genie (https://arxiv.org/abs/2402.15391)、GAIA-1 (https://arxiv.org/abs/2309.17068) 作为 sim 的替代。这是 NVIDIA 自己也在推的方向——用 learned world model 替代 physics simulator，可能 dynamics gap 自动消失。
4. **VLA 时代的 co-training**：paper 用的是 Diffusion Policy 这种 specialist policy。当 backbone 换成 VLA（OpenVLA、π₀、GR00T N1），co-training recipe 怎么变？特别是 VLA 已经有 web pretraining，sim data 的 marginal 价值会不会下降？这是个关键的 next-step 问题。
5. **Negative transfer 的边界**：FAQ 里提到 bimanual task 用单臂 prior data co-train 反而学出单臂行为，success rate ~0%。这说明 Prior data 的 behavior distribution 必须与 real 兼容。系统化研究 "什么样的 mismatch 会导致 negative transfer" 是个重要方向。

---

## 12. 我对这篇 paper 的整体评价

**Strengths**：
- Systematic study 而非 single method paper，这种 paper 在 robotics 社区很稀缺
- 两个 domain（Panda + GR-1 humanoid）+ 6 个 task，实验广度足够
- Recipe 部分直接 actionable，practitioner 可以照搬
- Generalization 实验（unseen object/position）是亮点，把 co-training 的价值从 "in-domain 重复" 提升到 "out-of-distribution generalization"

**Weaknesses**：
- 没有理论分析（不过 robotics empirical paper 普遍如此）
- α 的 sweep 只在 CupPnP 上做了，其他 task 是否同 curve 未知
- CloseDoor 的 10% → 100% gap 没有深入解释（是 real data 50 条不够？还是 task 本身需要 sim 提供 motion prior？）
- 没有与 domain randomization baseline 直接对比（只在 FAQ 提到 complementary）

**对我（Andrej Karpathy）的 take-away**：

这篇 paper 印证了我一直强调的一个观点——**"more data is all you need" 在 robotics 上同样成立，但 data 不必全部来自 real**。关键是把 sim data 视为 real data 的 "data augmentation in a much bigger sample space"，而不是需要完美对齐的替代品。这与 LLM 里 web data + curated data 的混合训练是同一个 paradigm。

下一个阶段 robotics foundation model 的关键问题，可能不是 "如何 sim-to-real transfer"，而是 **"如何在混合 data distribution 上训练一个 robust 的 generalist，让它 infer 时自动 focus on target domain"**。Co-training 是这个方向的 early empirical work。

---

## 13. 一些可能你（Karpathy）会感兴趣的延伸阅读

- **MimicLabs** (Saxena et al. CoRL 2024 workshop, https://arxiv.org/abs/2501.15020 大致这一系列): 同样研究 dataset composition 对 policy 的影响
- **RE-MIX** (Hejna et al. 2024, https://arxiv.org/abs/2408.14037): imitation learning 的 data mixture optimization
- **The Colosseum** (https://arxiv.org/abs/2402.08191): robot manipulation generalization benchmark
- **Decomposing the Generalization Gap** (Xie et al. ICRA 2024, https://arxiv.org/abs/2403.05110): 分析 IL 中 generalization gap 的来源
- **RoboGen** (https://arxiv.org/abs/2311.01455): generative sim 自动生成 task
- **GR00T N1** (https://arxiv.org/abs/2503.14734): NVIDIA humanoid foundation model，本身就是 sim + real co-training 思路的 industrial-scale 实践
- **π₀** (https://arxiv.org/abs/2410.24164): Physical Intelligence 的 VLA flow model，可能也在用类似 data mixing

如果要从这篇 paper 出发思考下一个工作，我觉得最 promising 的方向是 **把 co-training 从 specialist policy（Diffusion Policy）推到 VLA foundation model 上**，研究在 VLA 已有 web pretraining 的情况下，sim data 的 marginal 价值曲线，以及 α 如何随 model scale / data scale 变化。这相当于 robotics 版的 "Chinchilla data mixing laws"。
