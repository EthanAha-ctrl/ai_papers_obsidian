---
source_pdf: CONTACTMIMIC.pdf
paper_sha256: 711c8fb796723c60412a56dd50f46dd71c4ccffe091bec572c08e76b0496e13d
processed_at: '2026-08-18T04:02:24-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 CONTACTMIMIC

## 这篇 paper 在干嘛

一句话：教人形机器人怎么"按命令"去碰或者不碰东西。

## 问题出在哪

想象你在训练机器人擦白板。你给它一段人类擦白板的动作，让它跟着学——具体就是让机器人的手腕关键点跟着人的手腕走。机器人学得挺好，手腕轨迹复刻得几乎完美，MPJPE 接近 0。

但问题是：机器人的手可能在白板表面前 5mm 处完美复刻这个轨迹。几何上完全对，任务上完全失败——根本没擦到。

类似的坑到处都是：坐椅子 vs 在椅子前蹲下、推椅子 vs 在椅子前空推、靠椅背 vs 在椅背前挺直。这些成对动作的 keypoint 几乎一样，区别只有一个——**有没有真的接触**。

所以光告诉机器人"去哪"是不够的，还得告诉它"要不要碰"。

## 作者的想法

给 policy 多塞一个输入：一个 binary 的接触开关。每一步、每个身体部位都有一个 0/1 命令——这只手要不要碰白板、这个屁股要不要坐椅子、这个背要不要靠椅背。

这个开关同时是训练目标，也是部署时的用户指令。你想让机器人坐但不靠背，就把"靠背接触"那个开关关掉，其他保持。

## 真正的难点

读到这儿你可能觉得挺简单——加个 input、加个 reward 不就行了。问题在于训练数据。

人类动作数据里，keypoint 和 contact 是绑死的。人坐椅子时屁股一定贴着椅子，contact 一定是 1。人不坐时屁股一定远离椅子，contact 一定是 0。这两个信号高度相关。

policy 一看这数据，马上学会一个 shortcut："屁股在椅子附近 → contact 应该是 1"。它压根不看你的 contact 命令，直接从 keypoint 位置推断出来。你 toggle 接触开关，机器人完全没反应。

这是经典的 spurious correlation / shortcut learning 问题。在 image classification 里很常见（牛总在草地上，分类器学到的是草地不是牛）。在 robot learning 里相对少见，因为之前很少有"两个 input 互相冗余"的 setup。

## 作者的解法：造 counterfactual 数据

既然自然数据里 keypoint 和 contact 绑死了，那就人为造一些把它们解绑的样本。三种 augmentation：

1. **Contact label flipping**：保持动作轨迹不变，把接触命令反过来。同一段坐椅子的动作，告诉机器人"坐下但别真坐"。机器人在 sim 里还是按原轨迹走，但被惩罚真接触椅子。
   
2. **Object removal**：把椅子从场景里拿走，但动作轨迹完全保留。机器人得学会在原位置 hover——手放桌面位置但桌子没了，得悬空。
   
3. **Inflated geometry**：retargeting 阶段把椅子的 collision geometry 膨胀 5-10cm，让 retargeter 自动绕开，得到一条"远一点"的轨迹。这个轨迹的 contact label 全部置 0。

这些 augmentation 可以组合。比如 inflated geometry + contact flipping，就得到"远 keypoint + 接触命令 on"——模拟"keypoint 给得有点偏但用户还是希望接触"的场景。

训练时每个 episode 随机抽一种 augmentation，policy 既能看到"keypoint near + contact on"的常规样本，也能看到"keypoint near + contact off"、"keypoint far + contact on"等反常样本。逼着它学到"contact 命令是个独立信号，必须看"。

这个 trick 是整篇 paper 最有价值的地方。没有这个 data curation，光改 reward 和 policy input，policy 照样学不会 toggle 接触。Table 9 的 ablation 证明了这点——no augmentation 时 sit and squat 的接触 metrics 在 contact on/off 命令下几乎一样，policy 完全无视命令。

## reward 怎么设计

总 reward 是三部分：常规的 keypoint tracking + contact-aware + regularization。

contact-aware 有两个 term：

**Label matching**：奖励"该接触的真的接触了"和"不该接触的真的没接触"。用 balanced accuracy $\frac{1}{2}(\text{TPR} + \text{TNR})$，避免大多数 pair 都不该接触时 TNR saturate 的坑。对于擦白板这种 contact 很稀疏的动作，换成 TP − FP，给 false positive 更强惩罚。

**Contact distance**：这个是 dense shaping。即使还没真接触，只要该接触的部位在往目标表面靠近就给 reward（高斯核，σ=0.2m）。反过来，不该接触的部位进入 5cm 内就罚。这给 policy 一个连续的 gradient，而不是干等接触发生那一刻才有信号。

contact reward 的权重（4.0 和 3.0）比 keypoint tracking（0.5-1.0）大很多，暗示 contact 在最终行为里占主导。

## 实验设计最漂亮的地方

怎么验证 policy 真的听 contact 命令、不是又走了某个 shortcut？

作者构造了 4 种测试条件：near/far keypoint × contact on/off。

- $\mathcal{T}_{\text{near}}^\checkmark$: 原始轨迹 + 接触开
- $\mathcal{T}_{\text{near}}^\times$: 原始轨迹 + 接触关（关键的反常 case）
- $\mathcal{T}_{\text{far}}^\checkmark$: 远轨迹 + 接触开
- $\mathcal{T}_{\text{far}}^\times$: 远轨迹 + 接触关

如果 policy 真听话，contact metrics 应该跟着 contact label 走，不跟 keypoint variant 走。比如 $\mathcal{T}_{\text{near}}^\checkmark$ 和 $\mathcal{T}_{\text{near}}^\times$ keypoint 完全一样，只有 contact 命令不同，metrics 应该明显不同。

最 decisive 的例子是 **pick up box**：原始抓箱子的轨迹，但 contact 命令关掉，机器人就不抓，箱子几乎不动（位移 0.20m）。同一段轨迹，contact 命令打开，箱子被搬起 0.49m。

keypoint 几何完全一样，行为完全不同。这证明 policy 不是在模仿几何，而是在执行接触意图。

## real world 上 G1 怎么样

5 个动作 × 接触开关 on/off 各跑多次，成功率接近 95%。同一段录好的参考动作，按下按钮让机器人"接触"——它就真贴上去；按按钮让它"别接触"——它就在表面前 hover 出同样姿势。

部署时没用接触传感器，只用 onboard proprioception（关节角度、速度、力矩）。作者用 linear probing 验证：policy 的 Layer 2 representation 里线性可解码出当前接触状态，F1 普遍 > 0.93。说明 policy 内部已经学了个 implicit contact observer，从关节反应推断"碰没碰"。

这其实有点意外。机器人脚踩到地面和悬空时，踝关节力矩模式确实不同；手压到白板和悬空时，手腕力矩也不同。policy 学会读这些 dynamic cues 来推断接触状态。sim2real 时这个 estimator 仍然 work，是 sim2real 干净的关键之一。

## 这工作有意思在哪

我觉得最有意思的点是它把 contact 提升成了一个 first-class 的 controllable modality。

之前所有 humanoid tracking 工作（PHC、ExBody2、OmniH2O、BeyondMimic、HOVER、MaskedMimic）都把接触当副产品——你跟好 keypoint，碰不碰随缘。这些方法在低摩擦仿真下碰巧接触，到真实机器人上 contact 就丢了。BeyondMimic 的对比数据特别明显：MPJPE 跟 CONTACTMIMIC 差不多，但 contact bodies 和 impulse 差好几倍——几何形态学到了，物理接触没发生。

CONTACTMIMIC 的 framing 一旦成立，所有后续 design 都自然 follow：数据需要 break correlation，reward 需要 contact-aware，实验需要 controlled toggle。

更深层看，这个工作其实在做 **classifier-free guidance 的 imitation learning 版本**。Diffusion 里 classifier-free guidance 通过训练时随机 drop conditioning，强制 model 既学 conditional 又学 unconditional，inference 时用差值放大 conditioning signal。CONTACTMIMIC 通过造 counterfactual data，强制 policy 既学 keypoint→motion 又学 keypoint+contact→motion，inference 时 contact signal 不会被 keypoint shortcut 掉。

这个思路可以 generalize 到其他 conditional policy 问题。比如 language-conditioned manipulation——如果 instruction 和 state 高度相关，policy 会 ignore instruction。同样需要 data augmentation 来 break correlation。

## 局限和可能的下一步

几个我觉得可以挑战的点：

**Per-motion policy**：每个动作训一个 policy，没 scale 到 universal tracker。要做 humanoid foundation model，得 multi-motion joint training，可能上 transformer policy，类似 MaskedMimic 那种 masked motion inpainting 框架。

**Binary contact 太粗**：真实 manipulation 经常需要 fine-grained 接触位置（抓杯子把手 vs 杯身）、力大小（轻擦 vs 重擦）、接触类型（滑动 vs 固定抓握）。binary per-body label 抓不住。可以 extend 到 continuous vector——desired force profile、contact duration curve 等。FALCON 那种 force-adaptive 工作是 complementary direction。

**没视觉闭环**：部署时 policy 不看摄像头，object 位置 reset 时手测一次固定。如果 object 在 task 中移动（除了 kick chair），policy 不知道。对未来 generalization 是大限制。

**数据依赖 HUMOTO**：高质量 mocap dataset 多样性有限。要 in-the-wild video 数据，需要 3D 重建 + 接触自动标注，contact label noise 会比 mocap 大很多，augmentation 策略得 robustify。

**train/inference asymmetry**：训练时 contact state $c_{t,b,p}$ 是仿真 ground truth，用来算 reward。inference 时 policy 没这个 input。这跟 privileged critic 的思路类似，但 reward 端的 asymmetry 没显式处理。一个可能的改进是 teacher-student distillation：teacher 看接触状态，student 不看，蒸馏过去。

## 一句话总结

CONTACTMIMIC 把"接触"从模仿学习的副产品升级成一个用户可以直接 toggle 的开关。核心 trick 是用 data augmentation 打破 keypoint 和 contact 之间的 spurious correlation——光改 reward 和 policy input 没用，必须主动造 counterfactual 样本逼 policy 学会听命令。

整篇 paper 最值得 build intuition 的点是：**conditional policy 训练时，如果 conditioning signal 和其他 input 高度相关，policy 会 shortcut 掉 conditioning signal。要让它真听话，得人为制造 conditioning 与其他 input 不一致的数据。**

这个 insight 跨领域通用。language-conditioned robot learning、instruction-tuned LLM、多模态 foundation model 都会碰到类似问题。CONTACTMIMIC 给 humanoid motion tracking 这个具体场景提供了一个干净的解法示范。

参考链接：
- Project page: https://lixinyao11.github.io/contactmimic-page/
- BeyondMimic (baseline): https://arxiv.org/abs/2508.08241
- HUMOTO dataset: https://humoto.cs.cmu.edu/
- ResMimic (closest prior): https://arxiv.org/abs/2510.05070
- Classifier-free guidance (类比思路来源): https://arxiv.org/abs/2207.12598
- Unitree G1: https://www.unitree.com/g1/

---

# CONTACTMIMIC 深度讲解

## 一、Motivation: 从 "where" 到 "whether touching"

先 build intuition。想象你训练一个 humanoid 去"擦白板"，loss 只监督手腕 keypoint 的 3D 位置。最优解是什么？机器人可以在白板表面前 5mm 处完美复刻擦写轨迹——pose 完全对，但根本没擦到。MPJPE 接近 0，task 完全失败。

这就是 paper 的核心 motivation：**keypoint trajectory 是一个 incomplete specification**。一个 useful 的 loco-manipulation task，success criteria 不是 keypoint 而是哪个 body part 接触了哪个 object part、什么时候接触。擦白板 vs 在白板前挥手、坐椅子 vs 蹲下、推椅子 vs 在椅子前空推——这些 task pair 的 keypoint 几乎一样，但 semantic 完全不同。

作者提出的 idea：把 per-time-step、per-body-part 的 binary contact label $\bar{\mathbf{c}}_t \in \{0,1\}^{|B|}$ 当作 **runtime-controllable knob** 注入到 policy 里。$|B|$ 是 contact-capable bodies 数量（pelvis、torso、hips、knees、ankles、shoulders、wrists）。

这个 framing 很关键。区别于 prior work 如 ResMimic [23]，后者把 contact 当 training reward 但不 expose 给 user；CONTACTMIMIC 把 contact label 同时用作 **training target** 和 **test-time command**。

Project page: https://lixinyao11.github.io/contactmimic-page/

---

## 二、Policy 形式

$$
\pi_\theta(\mathbf{a}_t \mid \mathbf{p}_t, \bar{\mathbf{k}}_t, \bar{\mathbf{c}}_t)
$$

变量解释：
- $\mathbf{a}_t$: action（target joint angles），后续通过 PD controller 转 torque
- $\mathbf{p}_t$: proprioception（joint positions、joint velocities、base angular velocity、projected gravity）
- $\bar{\mathbf{k}}_t$: reference keypoint positions，由 reference configuration $\bar{\mathbf{q}}_t$ 通过 forward kinematics 得到，expressed 在 robot's local frame
- $\bar{\mathbf{c}}_t \in \{0,1\}^{|B|}$: per-body binary contact command，其中 $\bar{c}_{t,b} = \max_p \bar{c}_{t,b,p}$（对 object parts p 取 max，因为一个 robot body 可能接触多个 object semantic parts，取并集）

Architecture 简单：actor 和 critic 都是 MLP，hidden dims `[512, 256, 128]`，ELU activation。critic 拿到 actor obs 的 noise-free 版本 + base linear velocity（作为 privileged info）。PPO，4096 parallel envs，50Hz，30k iterations，GAE λ=0.95，clip 0.2，KL target 0.01。

注意一个细节：**policy 不接收 runtime contact state $c_t$ 作为 input**。这是 deliberate 的选择，作者后面用 linear probing 验证：proprioception 本身就 linearly decodable 出 contact state（F1 通常 >0.93），不需要 contact sensors。这与 HOMIE [16]、HOMIE 等 teleop 工作不同。

---

## 三、Reward 设计——把 contact 变成 differentiable signal

公式 (1) total reward：

$$
r_t = \underbrace{r_t^{\text{track}}}_{\text{tracking}} + \underbrace{w_{\text{lm}} r_t^{\text{lm}} + w_{\text{cd}} r_t^{\text{cd}}}_{\text{contact-aware (ours)}} + \underbrace{r_t^{\text{reg}}}_{\text{regularization}}
$$

下标含义：`track` 跟踪 keypoint（来自 BeyondMimic [1]），`lm` = label matching，`cd` = contact distance，`reg` = regularization（action rate、joint limits、undesired contacts）。

### 3.1 Label matching reward $r_t^{\text{lm}}$

定义两个集合：
- $S_+ = \{(b,p): \bar{c}_{t,b,p}=1\}$: 应该接触的 (robot body, object part) pairs
- $S_- = \{(b,p): \bar{c}_{t,b,p}=0\}$: 不应该接触的 pairs

公式 (2)：

$$
\text{TPR} = \frac{1}{|S_+|}\sum_{(b,p)\in S_+} c_{t,b,p}, \quad \text{TNR} = \frac{1}{|S_-|}\sum_{(b,p)\in S_-}(1-c_{t,b,p}), \quad \text{FPR} = 1 - \text{TNR}
$$

变量含义：
- $c_{t,b,p} \in \{0,1\}$: 实际 contact state，由仿真判定（force threshold 1N）
- TPR: 应该接触的中，实际接触了的比例（sensitivity / recall）
- TNR: 不该接触的中，实际没接触的比例（specificity）
- FPR = 1 - TNR: false positive rate

两种形式：

**(a) Balanced accuracy（default）**: $r_t^{\text{lm}} = \frac{1}{2}(\text{TPR} + \text{TNR})$

intuition：当 $|S_+| \ll |S_-|$（绝大多数 pair 不该接触），naive accuracy 会被 TNR saturate——只要不动就接近满分。balanced accuracy 用 mean 来 force 信号从两边都来。

**(b) TP − FP（for sparse-contact motions）**: $r_t^{\text{lm}} = \text{TPR} - \lambda \cdot \text{FPR}$, $\lambda = 1.0$

intuition：擦白板这种 motion，contact 几乎是 instantaneous event，绝大多数 timestep $|S_+|=0$。balanced accuracy 在这些时刻 trivially 满分，policy 学不到东西。TP−FP 直接惩罚 false positive，给 sparse contact 更强 gradient。Table 7 显示 wipe whiteboard、kick chair、pick up box 用 TP-FP，其他用 balanced accuracy。

### 3.2 Contact distance reward $r_t^{\text{cd}}$

公式 (3)：

$$
r_t^{\text{cd},+} = \frac{1}{|S_+|}\sum_{(b,p)\in S_+} \exp\left(\frac{-d(b,p)^2}{2\sigma^2}\right), \quad r_t^{\text{cd},-} = \frac{-1}{|S_-|}\sum_{(b,p)\in S_-} \mathbf{1}[d(b,p) < \delta]
$$

变量含义：
- $d(b,p)$: robot body $b$ 的 origin 到 object part $p$ 表面的距离
- $\sigma = 0.2\text{m}$: Gaussian kernel bandwidth，控制 reward 的"软度"
- $\delta = 0.05\text{m}$: 硬阈值，进入这个距离就罚
- $\mathbf{1}[\cdot]$: indicator function

intuition：
- $r_t^{\text{cd},+}$ 是 **dense shaping signal**：即使 $c_{t,b,p}=0$（没真正接触），只要接近就给 reward，提供 gradient 让 policy 学会靠近
- $r_t^{\text{cd},-}$ 是 **avoidance penalty**：不该接触的 pair 进入 5cm 内就罚，避免 accidentally 接触到错误部位
- 这两个 reward 互补：`+` 拉近应该接触的，`-` 推远不该接触的

Table 6 显示 $w_{\text{lm}}=4.0$、$w_{\text{cd}}=3.0$，比 tracking reward 的 weight (0.5-1.0) 大很多。这暗示 contact-aware signal 在最终 policy 行为中占主导，与 tracking 形成平衡。

---

## 四、最关键 contribution: Trajectory augmentation 打破 spurious correlation

这一节是 paper 的核心 insight，build intuition 是关键。

### 4.1 为什么需要 augmentation

考虑自然 human data：当人坐椅子时，pelvis keypoint 几乎一定贴着 chair seat，且 contact label = 1。当人挥手不打招呼地走过椅子时，pelvis keypoint 远离 chair seat，contact label = 0。**keypoint 位置和 contact label 几乎完美 correlated**。

policy 在这种数据上训练，会学到 shortcut："看到 keypoint 在椅子附近 → contact label 应该是 1"。于是 contact command 变成冗余信号——policy 直接从 $\bar{\mathbf{k}}_t$ infer 出 $\bar{\mathbf{c}}_t$，根本不看 contact input。

这就是为什么 naïve 训练会 fail（paper §4.5 ablation 验证）。

这是一个 **spurious correlation / shortcut learning** 问题，在 image classification 里很经典（比如牛总是在草地上 → 学背景）。在 robot learning 里出现得相对少，因为之前很少有这种"input 之间 redundant"的 setup。

### 4.2 三种 augmentation 策略

paper 提出 3 种 augmentation，可以 compose：

**❶ Contact-label flipping**: 保留原 trajectory 不变，但 flip task-relevant contact labels。Object 还在 sim 里，但 robot 被 Penalize for making contact。这创造了一种 **counterfactual**：相同的 keypoint 但 contact command 反转。

intuition：给 policy 看"坐着但别真坐下"的训练样本，强迫它学到 contact label 不是 keypoint 的 trivial function。

**❷ Object removal**: 把 object 从 scene 中拿走，所有 target contact labels 强制为 0。Keypoint trajectory 完全保留。

intuition：如果 keypoint 让手放在桌面位置，但桌子没了，policy 必须学会 "hover" 在原位置。这强化了 "contact off" 状态的几何表达。

**❸ Inflated geometry**: 在 retargeting 阶段把 target object parts 的 collision geometry isotropically inflate $\delta_{\text{infl}} \in [5, 10]$ cm。这迫使 retargeter 把相关 robot body 绕开更远，得到一条 **perturbed keypoint trajectory**，目标 contact 不再发生，labels 置零。

intuition：这给"远 keypoint + 无 contact"pair 提供数据。结合 flipping 可以得到"远 keypoint + 有 contact"——模拟 keypoints slightly misspecified 但用户仍希望接触的情形。

### 4.3 数据组合后的训练

每个 episode reset 时随机选：default trajectory / 某个 augmentation / composition。Table 7 列出 per-motion 的 augmentation 配置。例如 sit on table 不用 object removal（因为 table 没了 robot 会直接掉下去），sit and squat 不用 inflated geometry（直接 remove chair 即可）。

这个 setup 很聪明：**用 data curation 解决 conditional generation问题**，而不是用复杂的 latent variable model 或者 classifier-free guidance。类似 GAN 的 paired data 思路，但应用到 motion 上。

---

## 五、实验设计：4 个 trajectory set 测试 contact controllability

这个实验设计非常 elegant。给定一个 motion 的 keypoint trajectory $\tau$，命令 contact on ($\tau^\checkmark$) 或 contact off ($\tau^\times$)。同时有两种 keypoint variant：near（原始）和 far（用 inflated geometry perturbed）。两两组合得到 4 个 trajectory set：

- $\mathcal{T}_{\text{near}}^\checkmark$: near keypoint + contact on （自然 case）
- $\mathcal{T}_{\text{far}}^\checkmark$: far keypoint + contact on （keypoints 错了但仍想接触）
- $\mathcal{T}_{\text{near}}^\times$: near keypoint + contact off （keypoints 接触位置但不想接触）
- $\mathcal{T}_{\text{far}}^\times$: far keypoint + contact off （双重一致 off）

如果 policy 真的听 contact command，应该看到：
- $\mathcal{T}_{\text{near}}^\checkmark \to \mathcal{T}_{\text{near}}^\times$: contact metrics 下降（contact off）
- $\mathcal{T}_{\text{far}}^\checkmark \to \mathcal{T}_{\text{far}}^\times$: contact metrics 下降

如果 policy 只是 shortcut，metrics 会跟着 keypoint variant 走而不是 contact label。

Table 8 / Fig 4 显示，contact bodies 和 impulse 都 track contact label，不 track keypoint variant。这是非常 clean 的 controlled experiment，比单纯报告 success rate 有说服力得多。

特别有意思的发现：**pick up box** 在 $\mathcal{T}_{\text{near}}^\times$ 条件下，box 位移只有 0.20m（基本没动），即使 keypoint trajectory 是原始的 pickup motion。这意味着 policy 完全 ignore 了"应该抓起 box"的 keypoint signal，转而执行 hover 动作。这是 contact conditioning 真正控制行为的最强证据。

### 5.1 与 BeyondMimic 对比 (Table 3)

BeyondMimic [1] 是 SOTA keypoint-only tracker。在 contact bodies 和 impulse 两个 metrics 上：
- Wipe whiteboard: BM 0.01±0.09 contact bodies vs Ours 0.65±0.45
- Lean on backrest I: BM 0.12±0.24 vs Ours 1.38±1.27
- Pick up box: BM obj disp 0.03m vs Ours 0.49m

但 MPJPE 类似（BM 3.9cm vs Ours 3.6cm for wipe whiteboard）。这说明 keypoint tracking 本身没坏，contact 只是没发生。

intuition：BeyondMimic 的 policy 学到的是"几何形态"，contact 是 incidental byproduct。这种 policy 在低摩擦、低精度仿真下可能碰巧接触，但在真实 G1 上，加上 PD controller 误差、sim2real gap，contact 就丢了。

### 5.2 Real-world sim2real (Table 2)

5 个 motion × 2 contact condition（on/off）× 多次 trial。成功率基本都在 4/5 以上，total 接近 95%。sim2real transfer 这么干净，主要靠：
- Domain randomization on link masses、joint friction、object friction and mass
- 50Hz policy + 1000Hz onboard PD
- 无需 external mocap 或 vision（用 pre-recorded reference）
- proprioception alone 足够 inference contact state

特别优雅的 design：不用 contact sensor 在 deployment 时，因为 Table 4 显示 Layer 2 representation 的 linear probe F1 普遍 >0.93，远高于 chance。这意味着 policy 内部已经 learned 一个 implicit contact estimator，sim2real 时这个 estimator 仍然 work。

---

## 六、Ablation 验证 augmentation 必要性 (Table 9 / Fig 6)

这是 paper 最 decisive 的实验。Without augmentation 时：
- Sit and squat: $\mathcal{T}_{\text{near}}^\checkmark$ 0.47 bodies / 1.57 N·s → $\mathcal{T}_{\text{near}}^\times$ 0.46 bodies / 1.17 N·s（几乎没变，policy ignore contact command）
- With augmentation: 0.53/2.18 → 0.08/0.05（drop 6-23×）

Pick up box without aug: $\mathcal{T}_{\text{near}}^\times$ 还在 grab box（1.95 bodies, 1.85 N·s）——policy 完全 ignore 了 "don't touch" command。With aug: 0.29/0.17，干净地停手。

Step foot on chair 的 ablation 几乎无差。intuition：这个 motion 的 keypoint 本身 force foot contact（单腿平衡 impossible without chair），所以即使没 augmentation 也没 room 让 policy 学 decoupling。

这个 ablation 的 takeaway：**contact-conditioned policy 不是 reward engineering 就能学出来的，需要 data curation 主动 break spurious correlation**。这跟 image classification 里 "shortcut learning" 文献的结论一致。

---

## 七、与相关工作的位置

Paper 在 three 个 lineage 的交叉点：

1. **Humanoid whole-body tracking**: PHC [8]、ExBody2 [2]、OmniH2O [3]、HumanPlus [4]、ASAP [5]、BeyondMimic [1]、MaskedMimic [6]、HOMIE [16]、HOVER [7] 等。所有这些 method 都把 contact 当 incidental byproduct。CONTACTMIMIC 是 first to make contact explicit controllable。

2. **HOI data & retargeting**: AMASS [24]、GRAB [25]、BEHAVE [26]、HUMOTO [28]、OmniRetarget [29]、GMR [30]。Prior pipeline 输出 single canonical trajectory per clip。CONTACTMIMIC 在 retargeting 基础上做 augmentation 来制造 motion pairs。

3. **Contact-aware motion synthesis**: classical trajectory optimization with complementarity (Mordatch [31]、Posa [32]、Tassa [33])、physics-based character animation (DeepMimic [34]、Neural State Machine [35]、SAM [36])、learning-based HOI synthesis (InterDiff [37]、HOI-Diff [38]、CHOIS [39]、InterPrior [41]、TokenHSI [42])。这些都把 contact 当 intermediate representation / affordance，**不是 user-exposed knob**。CONTACTMIMIC 的差异在于 contact 是 runtime input。

最接近的 prior work 是 ResMimic [23]（同样在 HOI tracker 上加 contact-tracking reward），但它 condition only on motion + object trajectory，没有 fine-grained controllability。CONTACTMIMIC 的 trajectory augmentation 是它的关键不同点。

参考链接：
- HUMOTO: https://humoto.cs.cmu.edu/
- BeyondMimic: https://arxiv.org/abs/2508.08241
- OmniRetarget: https://arxiv.org/abs/2509.26633
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Unitree G1: https://www.unitree.com/g1/
- PPO: https://arxiv.org/abs/1707.06347
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
- ResMimic: https://arxiv.org/abs/2510.05070

---

## 八、Intuitive 框架与可能联想

### 8.1 把这个工作类比成什么

我把它类比为 **"classifier-free guidance for contact"**。Diffusion model 里 classifier-free guidance 之所以 work，是因为训练时随机 drop conditioning label，让 model 既学 conditional $p(x|y)$ 又学 unconditional $p(x)$，inference 时用两者差值 $(1+w)\cdot p(x|y) - w\cdot p(x)$ 放大 conditioning signal。

CONTACTMIMIC 的 trajectory augmentation 在做类似事情：通过人为创造 "label flipped" 和 "object removed" 的样本，强制 policy 既学 keypoint→motion 的 unconditional mapping，又学 contact-conditioned mapping。这才能保证 contact command 在 inference 时不被 keypoint shortcut 掉。

这是一个 deep 的相似性。Imitation learning 里 conditional policy 的 spurious correlation 问题其实普遍存在（language conditioned policies 也有类似问题：if instruction 和 state correlated，policy ignores instruction）。CONTACTMIMIC 的 augmentation 思路可以 generalize。

### 8.2 与 model-based contact planning 的关系

Classical contact-aware trajectory optimization（Posa、Tassa、Mordatch）的痛点是 contact schedule 必须 pre-specified。CONTACTMIMIC 用 learning + data augmentation 绕开了这个——schedule 变成 runtime command，policy 学会处理任意 schedule。

可能的下一步：把 CONTACTMIMIC policy 当作 differentiable contact executor，外面套一层 trajectory optimizer 来 emit contact schedule。这样得到 model-based planning 的 generality + learning-based policy 的 robustness。

### 8.3 与 humanoid foundation models 的关系

目前 humanoid foundation model 的工作（HOMIE、H2O、BeyondMimic、H1/H2 等）都 build 在 keypoint tracking 之上。但真实 task（sit、grasp、push、wipe、lean）都是 contact-defined 的。CONTACTMIMIC 提供了一个 possible 接口：foundation model 应该 emit $(\bar{\mathbf{k}}_t, \bar{\mathbf{c}}_t)$ pair，而不是只 emit $\bar{\mathbf{k}}_t$。

这跟 VLM 的 instruction following 问题很像：用户说 "sit down and lean back" vs "sit down without leaning"——语义差别在 contact，不在 keypoint。如果未来的 humanoid policy 直接从 language 生成 motion，它必须先生成 contact schedule，再生成 keypoint conditioned on contact schedule。

### 8.4 Per-body contact label 的 expressivity

当前 $\bar{\mathbf{c}}_t \in \{0,1\}^{|B|}$ 是 binary。但真实 task 可能有更丰富的 contact semantics：
- Force magnitude（"轻擦" vs "重擦"）
- Contact type（frictional sliding vs fixed grasp）
- Contact duration profile
- 多 body 同步接触（坐+手放桌上）

这些都可以 extend 到 continuous vector space。Paper 的 binary 设计是 minimal viable interface，证明 concept work 之后可以 generalize。

### 8.5 "Implicit contact sensing from proprioception" 的 implications

Table 4 显示 proprioception 已经 encodes contact state。这其实是一个相当 deep 的 finding。可能的解释：
- 当 hand 接触白板，arm joint torque 会有 reaction pattern
- 当 pelvis 坐下，hip joint 角速度会突然 drop
- Contact 改变 robot 的 dynamic mode（free space vs constrained）

这意味着 policy 的 Layer 2 representation 类似一个 **implicit contact observer**。可能 future work 可以做：让 policy output 一个 explicit contact estimate head，作为 self-supervised auxiliary task，进一步增强 sim2real 鲁棒性。

### 8.6 与 force control 的关系

CONTACTMIMIC 处理 binary contact，但没处理 force magnitude。对于 fragile object、precise assembly、human-robot handover 这些 task，force profile 很关键。FALCON [20] 之类的 force-adaptive loco-manipulation 工作可以视作这个方向的 complementary work。把 binary contact command extend 到 desired force profile 是 natural next step。

### 8.7 关于 motion 数据依赖

Paper 用 HUMOTO，这是 high-quality MoCap dataset with object interaction。Limitation 是 diversity 有限。下一步是用 in-the-wild video 数据。这就涉及：
- Video 到 3D motion reconstruction (TRAM, SLAHMR, etc.)
- Object 3D reconstruction
- Contact label 自动标注（用 some contact detection network）

这些都有 prior work 可以 plug in。但 in-the-wild 数据的 contact label noise 可能比 MoCap 大很多，augmentation strategy 需要 robustify。

### 8.8 "Single policy per motion" 限制的解决思路

Paper 训 per-motion policy。要达到 universal contact-conditioned tracker，需要：
- Multi-motion joint training with motion embedding / conditioning
- 可能需要更大 capacity backbone（transformer policy）
- Curriculum learning to handle diverse contact patterns
- 类似 MaskedMimic [6] 的 masked motion inpainting 框架，把 contact label 当 mask 的一种

这跟 language model 的 generalization 路径相似：从 per-task fine-tune → multi-task instruction tuning → general instruction following。CONTACTMIMIC 目前是 "per-task instruction following" stage，证明 instruction interface works；下一步是 scaling up 到 multi-task。

### 8.9 Reward shaping 的隐含 bias

Table 8 显示一些 motion 在 $\mathcal{T}_{\text{near}}^\times$ 下仍有 residual contact（如 wipe whiteboard 0.52 bodies）。作者解释为 keypoint tracking error cost > contact penalty，policy settle 在 trade-off。

这个 trade-off 由 reward weight 决定（$w_{\text{lm}}=4.0, w_{\text{cd}}=3.0$）。如果用户要 stricter contact suppression，应该提高 contact penalty weight，但会牺牲 keypoint tracking 精度。这是一个 Pareto frontier。可以联想 RLHF 里的 KL divergence penalty——trade-off between reward following 和 reference policy matching。

可能的改进：用 **constraint-based formulation**（Lagrangian PPO 或 CPO）把 contact command 当 hard constraint，而不是 soft reward。这样 contact on/off 是 guaranteed，keypoint tracking 是 best-effort。

---

## 九、Critique 与开放问题

最后列几个我作为 reader 觉得可以 challenge 的点：

1. **Per-part binary label 的 coarse-grained nature**: 真实 manipulation 经常需要 fine-grained contact location（比如抓杯子的 handle vs body）。Binary per-body label 抓不住这个。

2. **Reward weight 的 motion-specific tuning**: Table 6 是 default，但 Table 7 显示 mode (Bal. vs TP-FP) 是 per-motion 选的。这说明 reward design 仍有 task-specific tuning。能不能用一个统一 reward across all motions？

3. **Sim2real gap in contact dynamics**: PhysX rigid body 仿真的 contact stiffness/damping 和真实 G1 的 rubber feet 不一样。Paper 没详细 report 这个 gap，但 Table 2 的成功率暗示 it works。可能是因为 contact binary 化降低了对 force fidelity 的要求——只要接触发生了，policy 就 satisfied。

4. **没有 closed-loop visual feedback**: Policy 不看 camera，只在 reset 时测 object pose。如果 object 在 task 中移动（kick chair 之外），policy 不知道。这对未来 generalization 是大限制。

5. **Reward 的 contact state $c_{t,b,p}$ 来自仿真 ground truth**: 训练时 $c_{t,b,p}$ 是仿真判定，但 inference 时没有。这是 train/inference asymmetry——训练时 policy 在 privileged contact state 监督下学习，inference 时却没有这个 input。这跟 prior sim2real 工作（learning privileged critic）思路类似，但 reward 端的 asymmetry 没有显式处理。一个可能改进：用 teacher-student distillation，teacher 看 contact state，student 不看。

---

## 十、Conclusion 的一句话总结

CONTACTMIMIC 把 contact label 从 incidental byproduct 提升为 first-class controllable variable，用 trajectory augmentation 打破 keypoint-contact shortcut，用 contact-aware reward 把 binary command differentiably 注入 RL training。结果是一个**在相同 keypoint trajectory 下可以 toggle 物理接触**的 humanoid policy，并 sim2real 到 Unitree G1。

它的核心贡献不在 reward engineering（虽然 reward 设计也不错），而在 framing：**contact 是一个独立于 keypoint 的 modality，需要被 explicitly condition on**。这个 framing 一旦成立，后面所有的 data curation、reward design、experiment design 都自然 follow。

如果你 (Andrej) 要进一步推这个 line of work，我会建议：把它 scale 到 universal multi-motion policy；用 latent variable model（VAE / diffusion）替代 per-motion policy；把 contact label 从 binary extend 到 continuous force/time profile；接入 VLM 让 high-level language 直接 emit contact schedule。这条 path 通往真正的 humanoid foundation model——不仅仅 replicate 人的 motion，而是 purposefully interact with the world。

参考文献的 web links 整理：
- Project page: https://lixinyao11.github.io/contactmimic-page/
- BeyondMimic (baseline): https://arxiv.org/abs/2508.08241
- HUMOTO dataset: https://humoto.cs.cmu.edu/
- OmniRetarget: https://arxiv.org/abs/2509.26633
- ResMimic (closest prior): https://arxiv.org/abs/2510.05070
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Unitree G1: https://www.unitree.com/g1/
- PPO: https://arxiv.org/abs/1707.06347
- AMASS (related MoCap): https://amass.is.tue.mpg.de/
- BEHAVE dataset: https://virtualhumans.mpi-inf.mpg.de/behave/
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
