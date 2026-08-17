---
source_pdf: Turning Video Models into Generalist Robot Policies.pdf
paper_sha256: c2c0d12b2d66b0c6abea453c8478cc2f2061b328e49cd5869f70fa8fc1f8129f
processed_at: '2026-08-12T18:23:57-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VERA 人话版：让机器人学会"做梦"和"翻译"

好，我重新用大白话讲一遍，重点放在"为什么要这么做"和"妙在哪"，公式还是给但用通俗方式拆解。

---

## 一、整个field在吵什么

robot learning现在是个很尴尬的局面。你看NLP那边，train个LLM，web上text管够，scale up就完事了。robot这边呢？你想让robot学会抓东西，你得有(image, action)配对data——但真实robot action data稀缺得要命。DROID [1]算大的了，也就几万条轨迹。对比LLM吃的trillion token，差了6个数量级。

所以field里分了三派：

**第一派：VLA (Vision-Language-Action)**——RT-2 [2]、OpenVLA [3]、π_0 [4]、π_0.5 [5]。思路是拿个VLM backbone，加个action head，在robot data上finetune。想法很美：VLM已经懂世界了，接个head就能control。现实很骨感：action head把VLM的reasoning dilute掉了。Paper引用的[6]就是评估π在wild中各种翻车的report。

**第二派：WAM (World-Action Model)**——GR-2 [7]、DreamZero [8]、LingBot-VA [9]。思路是joint predict (future frame, action)，让video prediction和action prediction互相supervise。听起来更principled，但data要求更狠：要paired (video, action, text)。比VLA还稀缺。

**第三派：Video + IDM**——UniPi [10]、LVP [11]、VERA（本文）、Rhoda AI [12]。思路是：**我凭什么要让video model吐action？** video model就负责"想象成功会长什么样"，action的事让另一个小模块去翻译。这两个可以分开train，data要求大幅降低。

VERA就是第三派里目前最完整的system。它的核心claim是：**如果你把"翻译"这一步做对了，video model本身已经足够强，可以zero-shot控制各种robot**。

---

## 二、VERA的核心idea，一句话版

> **让video model"做梦"——想象task完成的visual sequence；让Jacobian IDM"翻译"——把pixel motion倒推回action。两者decoupled，各训各的。**

为什么decoupling好？三个理由，都是practical的：

1. **Video model不需要action data**。Wan-14B在web video上pretrain就够了，robot video只用来post-training adapt场景。
2. **同一个video planner能serve多个embodiment**。Panda arm和Allegro hand用同一个planner，只要换不同的IDM。
3. **Video model升级时IDM不用重训**。明天Sora-2出了，plug in就行。

这第三点其实是Karpathy你自己说过的"software 2.0"思路的延伸——video model是software 2.0的部分，IDM是software 1.5的classical structure。两者结合，各取所长。

---

## 三、Jacobian IDM——整篇paper的灵魂

这部分我重点讲，因为这是VERA和UniPi等前作的关键区别。

### 3.1 先说问题：为什么"翻译"难？

假设video model给你两帧$(\hat{o}_t, \hat{o}_{t+1})$——它"梦到"robot应该从状态A走到状态B。你要recover出action $a_t$让robot真的这么走。

最naive的做法（UniPi干的）：训个net $f_\theta(o_t, o_{t+1}) \to a_t$，直接regress。叫**Direct IDM (D-IDM)**。

听起来挺好，但paper Sec. 4.2做了controlled experiment发现两个失败模式：

**失败模式1：data有限时OOD崩**。D-IDM学的map是$R^{2HW} \to R^n$，input维度巨大。每对new image就是新input，data不够时interpolate不出去。

**失败模式2：随DoF增长迅速劣化**。Panda是7-DoF，Allegro是16-DoF。action维度$n$一涨，output space指数膨胀，同样data budget下每个dim分到的supervision变少。Table 2里UniPi*在Allegro-Sim上MSE=0.063，PushT(2-DoF)上才0.071——维度涨了8倍，误差只小一点点，但其实是因为Allegro的action range更小所以数值看起来差不多。看Table 1的success rate更扎心：UniPi*在Allegro和Panda-Sim上直接0%。

### 3.2 VERA的solution：别regress action，regress Jacobian

回到经典robotics。任何一个robot，在某个state下，action的微小变化$\delta a$和end-effector位置$\delta x$的关系是**局部线性的**：

$$\delta x = J(a) \, \delta a$$

这个$J$就是Jacobian，$3 \times n$矩阵（3D空间，n维action）。这是Robotics 101的内容，本科教材[13]里就有。

但VERA有个twist：**3D位置观测不到**，robot只看RGB。所以把Jacobian搬到image space——对每个pixel $p$，预测一个$2 \times n$矩阵$J_\theta(p, o)$，表示"action第j维变化单位1，pixel $p$会沿(x,y)方向移动多少"。

这就是Eq. 3的image-space Jacobian field。

**妙在哪？** 原来"从image pair预测action"是个高度nonlinear的inverse problem。现在被拆成两步：
- **Forward**：$J_\theta$预测Jacobian field——这是image的smooth function，well-posed
- **Inverse**：给定pixel flow，用closed-form ridge regression求action——**没有learnable parameter，不会OOD fail**

所有nonlinearity被push到forward Jacobian的prediction上，inverse是纯线性代数。这就是把inductive bias塞进architecture的玩法。

### 3.3 训练loss——joint forward + inverse

Eq. 5是核心，我拆开讲：

$$\mathcal{L} = \underbrace{\sum_p \rho(J_\theta(p, o_t)\delta a_t - v_t(p))}_{\text{forward term}} + w_a \underbrace{\sum_p \|\delta a_t - J_\theta^{\dagger,\lambda}(p, o_t) v_t(p)\|_2^2}_{\text{inverse term}}$$

逐个变量说：
- $o_t$：当前帧RGB
- $\delta a_t$：当前时刻action increment（data里已标注）
- $v_t \in R^{H \times W \times 2}$：从$(o_t, o_{t+1})$用RAFT [14]提取的optical flow——每个pixel的2D displacement
- $\rho(x) = \sqrt{x^2 + \varepsilon^2}$：Charbonnier loss，smooth L1的smooth版，对outlier鲁棒
- $J_\theta^{\dagger,\lambda}$：ridge-regularized pseudo-inverse，closed form是$(JJ^\top + \lambda I)^{-1}J^\top$（Eq. 8）
- $w_a = 0.3$：inverse term的权重

**Forward term的intuition**：给我一个action $\delta a_t$，我predict的Jacobian field乘上这个action，应该能reproduce观测到的optical flow。这是"我的Jacobian对了"的直接监督。

**Inverse term的intuition**：给我观测到的flow，我用Jacobian的pseudo-inverse反解出来的action，应该等于ground truth action。这是"我的Jacobian不仅forward对，inverse也要well-conditioned"的监督。

**为什么joint training关键**？如果只训forward，Jacobian可能在某些action dimension上接近零（那些dimension不引起pixel motion），inverse就explode。Inverse term直接监督"inverse也要work"，强迫Jacobian保持数值well-conditioned。这是个很elegant的设计——用inverse reconstruction作为regularizer。

### 3.4 Deployment：ridge regression求action

Inference时（Eq. 6）拿到video model生成的相邻帧$(\hat{o}_t, \hat{o}_{t+1})$：

1. 提flow：$v_t = \text{RAFT}(\hat{o}_t, \hat{o}_{t+1})$
2. Predict Jacobian field：$J_\theta(\cdot, \hat{o}_t)$，shape是$H \times W \times 2 \times n$
3. Ridge regression求action：

$$\hat{\delta a}_t = \arg\min_{\delta a} \sum_p \|J_\theta(p, \hat{o}_t)\delta a - v_t(p)\|_2^2 + \lambda\|\delta a\|_2^2$$

这是least squares with L2 regularization。HW个pixel，每个给2个方程（x方向、y方向），叠起来是$2HW$个方程$n$个未知数。$HW \gg n$时over-determined，ridge regression给出robust解。

λ的作用：当Jacobian某些列接近零（action dimension不引起可观测motion），unregularized least squares会blow up。λ把solution拉向零，避免飞掉。

### 3.5 这个设计为什么scale到高DoF

Figure 5的toy finger experiment讲得最清楚。2D平面finger，sweep DoF从1到5，sweep data从少到多。

观察：
- DoF=1时D-IDM和J-IDM都行
- DoF=5时D-IDM崩，J-IDM保持准确
- 固定DoF=5，J-IDM比D-IDM数据效率高约2x

为什么？Jacobian field本质是$n$个独立的"action channel $j$ 如何move每个pixel"的field。每个channel的supervision都来自全pixel的forward loss，可以share backbone features。D-IDM是直接学$R^{2HW} \to R^n$的end-to-end map，n大时output space指数膨胀。

用Karpathy你自己的话说：**Jacobian structure是个很重的prior，把"什么该学"和"什么不该学"分开了**。该学的是forward Jacobian（image的smooth function，可以share features across action dims）；不该学的是inverse map（用closed-form解就行）。

---

## 四、Video Model部分——其实是"小事"

相比之下video model这边没什么新东西，就是工程：

**Backbone**：Wan-14B [15]，open-weight video diffusion transformer。主实验用Large Video Planner (LVP) [11] warm-start，因为LVP已经在robot video上pretrain过。

**Training objective** (Eq. 7)：
$$\mathcal{L}_{vid}(\theta) = \mathbb{E}\left[\sum_{i=1}^M \|\hat{v}_{\theta,i} - v_i\|_2^2\right]$$

注意这里$v_i$是diffusion forcing [16]里的linear flow target（noise schedule的velocity），**不是**optical flow——同一个符号两种含义，容易搞混。

**关键**：training里**完全没有action**。Video model只学"future frame长什么样"，action信息从来不入video model的梯度。这就是"action-free post-training"。

**Diffusion Forcing** [16]：让不同frame可以assign不同noise level。所以能condition on clean history + denoise future。比vanilla video diffusion更适合closed-loop rollout——history frame不会被quantization noise污染。

**Multi-view**：Panda有3个相机（2 external + 1 wrist），直接在pixel space concat成128×576的wide image。简单粗暴但work。Cross-embodiment训练时missing view用blank frame padding。

**Initialization ablation** (Table 4)：
- Random init 14B: val MSE = 0.40
- Wan pretrained 14B: 0.13
- LVP warm-start 14B: 0.10

Video pretraining的prior巨大（0.40→0.13是3倍降误差）。这印证了核心thesis——video model的"世界知识"是从web video来的，robot video只是adaptation。

---

## 五、Closed-loop怎么跑

Algorithm 1的伪代码，我用大白话翻译：

```
loop:
    看最近N=6帧 (context)
    video model做梦：生成M=4帧future
    只commit前K=1帧 (保守，频繁replan)
    对这一对 (当前帧, 梦到的下一帧)：
        1. RAFT提取optical flow
        2. J-IDM预测Jacobian field
        3. Ridge regression求action
    执行这个action chunk
    观察新state，加入history，replan
```

**K=1很激进**——每次只执行一个latent transition对应的action就replan。Paper说这是为了contact-rich manipulation的频繁feedback。Figure 8 ablation显示K太大会因为accumulated error翻车，K太小又过度replan浪费compute。

Wan tokenizer有temporal compression：1 latent frame = 4 RGB frames。所以K=1 latent实际是4个RGB transitions，对应4个低层action command（取决于control rate）。

Canonical constants（Table 6）：N=6 context, M=4 look-ahead, K=1 commit, 40 UniPC denoising steps。

---

## 六、实验结果——重点讲insight

### 6.1 J-IDM vs D-IDM (Table 1)

| | Allegro-Sim (16-DoF) | Panda-Sim (7-DoF) | PushT-Sim (2-DoF) |
|---|---|---|---|
| UniPi* (D-IDM) | 0% / 0% | 0% / 0% | 74% / 85% |
| J-IDM (VERA) | **70% / 70%** | **94% / 94%** | **93% / 96%** |

D-IDM在16-DoF和7-DoF上完全失败。这就是Eq. 3.1讲的"随DoF崩溃"。J-IDM保持高success rate。

### 6.2 Action reconstruction MSE (Table 2)

| | Allegro-Sim | Panda-Sim | PushT-Sim | 5-joint finger |
|---|---|---|---|---|
| UniPi* | 0.063 | 0.38 | 0.071 | 0.047 |
| D-IDM + Flow | 0.044 | **0.09** | 0.059 | 0.030 |
| J-IDM | **0.031** | 0.19 | **0.046** | **0.017** |

注意Panda-Sim上D-IDM+Flow的MSE比J-IDM低（0.09 vs 0.19），但closed-loop success J-IDM仍胜（94% vs D-IDM低）。**这值得追问**——为什么reconstruction更准的model closed-loop反而差？

我的猜测：D-IDM+Flow可能overfit到training distribution的action pattern，reconstruction MSE低但OOD robustness差。J-IDM的ridge regression天然带regularization，避免action overshoot，closed-loop鲁棒性更高。MSE不是唯一指标，这点paper没明说但Figure 6的reasoning task结果间接支持。

### 6.3 vs π_0.5 和 DreamZero (Figure 6)

在Panda-Real上对比state-of-the-art VLA/WAM：

**Basic task ("push A", "pick B")**：
- DreamZero 90%
- VERA 60%
- π_0.5 30%

VERA在basic上落后DreamZero。Paper坦诚承认："VERA的失败一般来自video-to-action翻译步——failing rollouts的dreamed future其实完成了task，但翻译成action时loss fidelity"。这是J-IDM还有提升空间。

**Reasoning task (location-based, semantic-based prompt)**：
- DreamZero差
- VERA **明显胜**
- π_0.5差

例：prompt是"push the button matching the wrench's color"——需要visual grounding。π_0.5和DreamZero经常抓错物体。VERA靠video model的reasoning能力做对。

**Hidden button (occlusion)**：
- DreamZero失败
- VERA **成功**
- π_0.5失败

Button藏在墙后，3个相机只有1个看得到，还有distractor。π_0.5和DreamZero连locate都做不到。VERA能find并press。

**Paper的论点**：end-to-end VLA/WAM把action head塞进backbone会dilute video model的reasoning。VERA的decoupling保留video model原汁原味的reasoning能力。这个result是paper最强的证据。

---

## 七、Cross-embodiment demonstration

同一个video planner（在DROID + sim/real Allegro上混训）配上不同J-IDM，能控制Panda arm做manipulation，也能控制16-DoF Allegro hand做in-hand cube reorientation（Figure 4）。

这个result的意义：**video planner是embodiment-agnostic的**，因为它只predict pixel motion。Embodiment信息全在J-IDM里。未来换更强video backbone（Sora-class）J-IDM不用重训。

---

## 八、Limitations和我的追问

Paper自承：
1. 仍需robot-specific video做planner post-training
2. J-IDM依赖off-the-shelf optical flow [14, 17, 18]——textureless region、large displacement时不准
3. 无法处理force-based control（RGB-only）
4. Predicted transition的pixel motion很小时inverse conditioning变差

我自己追问几个：

**Q1：Jacobian的linear假设在contact-rich场景下成立吗？**  
Paper用K=1激进replanning来mitigate——每次action increment很小，linear近似就准。但本质上是把nonlinearity推给"频繁replan"。如果想用更长chunk省compute，linear假设会break吗？

**Q2：Action space里的discrete dimension怎么处理？**  
Panda action有gripper state（开/合），是discrete。Jacobian是连续map，对discrete dimension不适用。Paper没明说怎么处理——可能把gripper单独bin分类，或者用continuous relaxation。DROID data里确实有gripper dim，这是个implementation detail但paper没讲清。

**Q3：Video model的"幻觉"怎么办？**  
Video generative model有时generate physically impossible motion（物体穿透、违反重力）。J-IDM会faithfully翻译这种坏plan，可能产生危险action。Paper没讨论safety filtering。Real deployment这是个must-fix。

**Q4：Closed-loop latency够real-time吗？**  
每次planner call：40 UniPC steps + flow estimation + J-IDM forward + ridge solve。粗估video diffusion sampling几秒级，远达不到10Hz control。Paper没给wall-clock numbers。Rhoda AI [12]的industrial version可能更optimized。

**Q5：Multi-view consistency怎么保证？**  
J-IDM在不同view下predict的Jacobian是否物理consistent？如果不一致，inverse时不同pixel的constraint互相打架。Ridge regression自然average掉，但理论上怎么保证？作者组Nature paper [19]用3D NeRF field来enforce consistency，VERA丢掉这个，trade-off了什么？

---

## 九、几个有趣的联想

### 9.1 和Dreamer的关系
VERA本质是**decoupled Dreamer** [20]——video model是world model，J-IDM是policy。但Dreamer在latent space rollout + backprop through differentiable dynamics；VERA在pixel space rollout + 用Jacobian做inverse。Dreamer需要differentiable dynamics，VERA不需要——能用black-box video diffusion。这是把"可微simulator"换成"generative video model"。

### 9.2 和Diffusion Policy的对比
Diffusion Policy [21]直接在action space做diffusion。VERA在pixel space做diffusion，再Jacobian lift到action。VERA好处是pixel space的diffusion prior可以从web video pretrain；action space diffusion需要robot action data。

### 9.3 和Differentiable Simulation的呼应
VERA的Jacobian field本质是**learnable differentiable simulation**——学了"action → pixel motion"的local linearization。和classical differentiable simulation [22]（Brax、Taichi、MJX）的区别是：VERA不假设rigid body / URDF / contact model，直接从data学。Trade-off是失物理精确性，gain从raw video learn的能力。

### 9.4 "做梦的robot"这个metaphor
VERA的video model确实在做"dreaming"——不execute action，只generate"如果成功了会怎样"的visual hallucination。这和人类motor control的forward model假说有echo：人脑plan movement时会simulate sensory consequences [23]。VERA把forward simulation外包给video diffusion，inverse（从desired sensory consequence到motor command）外包给J-IDM。这个factorization在cognitive science里有讨论。

### 9.5 Scaling law猜想
Paper没给J-IDM的scaling law。Figure 5c暗示DoF=5时J-IDM比D-IDM数据效率好约2x。如果DoF=50（humanoid），gap会继续扩大吗？还是Jacobian field本身representation bottleneck饱和？作者组Nature paper [19]似乎在往这方向push。

### 9.6 和Karpathy你的"software 2.0"框架的连接
VERA是个hybrid：video model是software 2.0（learned from data, end-to-end），Jacobian IDM是software 1.5（classical structure + learned parameters）。两者各司其职。这和你当年提的"software 2.0 vs 1.5"张力很有共鸣——纯end-to-end未必是最优，加classical structure的inductive bias在小data regime下可能更practical。

---

## 十、总结：VERA的positioning

VERA代表一个**counter-trend**：整个field在往"end-to-end VLA with bigger backbone + more action data"卷的时候，VERA说"等一下，video model本身已经够强，我们只要解决好translation就好"。

赌注是：
1. Video model会继续变强（Sora、Wan、Veo升级），VERA自动受益
2. Action data永远稀缺，avoid action-labeled pretraining有structural advantage
3. Inductive bias在小data regime下beats end-to-end

J-IDM的下一步可能：和force/tactile sensing结合（处理pixel motion小但force大的contact），或往embodiment-conditioned J-IDM走（一个J-IDM serve multiple morphologies——学conditional Jacobian $J_\theta(p, o, \text{embodiment\_id})$）。

---

## References

[1] DROID: https://arxiv.org/abs/2403.12945  
[2] RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html  
[3] OpenVLA: https://proceedings.mlr.press/v270/kim25c.html  
[4] π_0: https://www.roboticsproceedings.org/rss21/p010.html  
[5] π_0.5: https://proceedings.mlr.press/v305/black25a.html  
[6] π in the wild: https://arxiv.org/abs/2507.09061  
[7] GR-2: https://arxiv.org/abs/2410.06158  
[8] DreamZero / WAM: https://arxiv.org/abs/2602.15922  
[9] LingBot-VA: https://arxiv.org/abs/2601.21998  
[10] UniPi: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html  
[11] LVP: https://arxiv.org/abs/2512.15840  
[12] Rhoda AI: https://www.rhoda.ai/research/direct-video-action  
[13] Modern Robotics (Lynch & Park): http://hades.mech.northwestern.edu/index.php/Modern_Robotics  
[14] RAFT: https://arxiv.org/abs/2003.12039  
[15] WAN: https://arxiv.org/abs/2503.20314  
[16] Diffusion Forcing: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html  
[17] AllTracker: https://arxiv.org/abs/2603.25739  
[18] TAPIR: https://arxiv.org/abs/2306.08637  
[19] Jacobian Fields (Nature): https://www.nature.com/articles/s41586-025-09170-0  
[20] Dreamer: https://dreamrl.github.io/  
[21] Diffusion Policy: https://journals.sagepub.com/doi/10.1177/02783649241273668  
[22] Brax: https://github.com/google/brax  
[23] Wolpert & Flanagan, Nature Neuroscience 2001: https://www.nature.com/articles/nn1201-729

想继续深挖哪块都行——比如VGGT init怎么对接J-IDM、Diffusion Forcing的noise schedule细节、或者DROID post-training的具体recipe。

---

# VERA: Video-to-Embodied Robot Action Model 深度解析

## 一句话intuition

VERA把robot control拆成两个decoupled模块：一个**action-free的video world model负责"做梦"**（生成visual plan），一个**embodiment-specific的Jacobian inverse dynamics model负责"翻译"**（把pixel motion倒推回action）。关键insight是：**不要让video model直接吐action**，否则会dilute它的visual reasoning能力；同时**不要让IDM直接regress action**，否则在高DoF下会fail——而要让它预测一个image-space Jacobian field，用ridge-regressed pseudo-inverse来recover action。

paper link: https://arxiv.org/abs/2512.15840 (LVP), 项目网站在paper footnote里。相关背景paper我会穿插给link。

---

## 一、大背景：Robot Foundation Model的三条路线

目前用large pre-trained model做robot control主要有三条technical route，每条对data的要求不同：

| 路线 | 代表 | 训练data要求 | Action如何产生 |
|---|---|---|---|
| **VLA** (Vision-Language-Action) | RT-2 [1], OpenVLA [2], Octo [3], π_0 / π_0.5 [4,5] | web-scale paired (image, text, action) | end-to-end head直接regress action |
| **WAM** (World-Action Model) | GR-2 [6], DreamZero [7], LingBot-VA [8] | paired (video, action, text) | joint predict (future frame, action) |
| **Video + IDM** (本文VERA) | UniPi [9], LVP [10], VERA | video-only + (video, action)分别训练 | video model生成visual plan, IDM翻译 |

关键trade-off：
- VLA要求**web-scale paired robot action data**——但现实里这种data非常稀缺，所以VLA的cross-embodiment generalization很难做出来（paper Sec.1引用了[11]的evaluation指出π在wild中问题很多）。
- WAM要求**paired video+action+text data**——更稀缺。
- **Video + IDM只要求video data (海量) + 少量task-agnostic的 action data (可self-play生成)**。这是VERA选择这条路的核心动机。

paper Sec.2里有详细的related work讨论。Rhoda AI的Direct Video-Action system [12]是concurrent industrial work，证明这条decoupling路线可以scale到real bimanual deployment。

---

## 二、VERA的系统架构

整个pipeline是closed-loop receding horizon controller：

```
┌─────────────────────────────────────────────────────────┐
│  Observation history o_{≤t}  +  Goal g (text)            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Video World Model     │  Wan-14B + LVP init
              │  π_vid(· | o_{≤t}, g)   │  Diffusion Forcing
              └────────────────────────┘
                          │
                          ▼  samples M=4 future latent frames
              ┌────────────────────────┐
              │  Committed prefix K=1  │  decode to RGB
              └────────────────────────┘
                          │
                          ▼  pair (ô_t, ô_{t+1})
              ┌────────────────────────┐
              │  Optical Flow estimator│  RAFT / AllTracker
              │  v_t = flow(ô_t,ô_{t+1})│
              └────────────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Jacobian IDM G_φ      │  VGGT init
              │  J_θ(p, ô_t) ∈ R^{2×n}  │  per-pixel Jacobian field
              └────────────────────────┘
                          │
                          ▼  ridge pseudo-inverse
              δâ_t = J_θ^{†,λ}(ô_t) · v_t
                          │
                          ▼
              Execute action chunk, observe, replan
```

paper Figure 2画了三步：(a) video rollout, (b) J-IDM per-step inversion, (c) closed-loop execute + replan。

---

## 三、Jacobian IDM——本文的核心contribution

这是整篇paper最值得深挖的地方。我重点讲为什么这个structure比D-IDM好。

### 3.1 经典robotics里的embodiment Jacobian

机器人学里，给定state o和action a ∈ R^n，body上某点x_i ∈ R^3的微小运动可以用**embodiment Jacobian**线性化：

$$
\mathbf{J}_i(\mathbf{o}) = \left.\frac{\partial \mathbf{x}_i}{\partial \mathbf{a}}\right|_{\mathbf{o}} \in \mathbb{R}^{3 \times n} \tag{Eq. 2}
$$

含义：$\mathbf{J}_i$的第j列就是第j个action channel单独作用时，body point $i$在3D空间里的瞬时速度方向。线性近似 $\delta \mathbf{x}_i \approx \mathbf{J}_i(\mathbf{o}) \delta \mathbf{a}$ 在小action increment下很准。

**问题**：3D point $\mathbf{x}_i$从来不会被directly观测——robot只能看到RGB。所以不能直接学这个3D Jacobian。

### 3.2 Image-space Jacobian field

VERA的solution：**直接在image space预测Jacobian**。给定单张图o ∈ R^{H×W×3}，一个image-conditioned transformer $\mathbf{J}_\theta$输出dense field：

$$
\mathbf{J}_\theta(\cdot, \mathbf{o}): \{1,\ldots,H\} \times \{1,\ldots,W\} \longrightarrow \mathbb{R}^{2 \times n} \tag{Eq. 3}
$$

含义：对每个pixel $\mathbf{p} = (h, w)$，$\mathbf{J}_\theta(\mathbf{p}, \mathbf{o})$ 是一个 $2 \times n$ 矩阵——第j列表示"如果action的第j维变化一个单位，pixel $\mathbf{p}$会沿(x, y)方向移动多少"。

forward model (Eq. 4)：
$$
\delta \mathbf{p} = \mathbf{J}_\theta(\mathbf{p}, \mathbf{o}) \, \delta \mathbf{a}
$$

这里 $\delta \mathbf{p}$ 就是optical flow（pixel在两帧之间的displacement），用现成的RAFT [13] / AllTracker [14] / TAPIR [15] / MegaFlow [16]估计。

注意：这其实是作者组之前Nature paper [17]的follow-up——那篇用NeRF-style 3D field + volume rendering来参数化Jacobian，VERA把3D scene representation去掉了，让J_θ就是一个pure image-conditioned transformer，所以可以scale up用VGGT [18]。

### 3.3 Joint forward-inverse training loss

这是关键设计。训练data是 $(\mathbf{o}_t, \delta \mathbf{a}_t, \mathbf{o}_{t+1})$ tuples。loss有两部分：

$$
\mathcal{L} = \underbrace{\sum_{\mathbf{p}} \rho\!\left(\mathbf{J}_\theta(\mathbf{p}, \mathbf{o}_t)\,\delta \mathbf{a}_t - \mathbf{v}_t(\mathbf{p})\right)}_{\text{forward (pixel-flow)}} + w_{\mathbf{a}} \underbrace{\sum_{\mathbf{p}} \left\| \delta \mathbf{a}_t - \mathbf{J}_\theta^{\dagger,\lambda}(\mathbf{p}, \mathbf{o}_t)\,\mathbf{v}_t(\mathbf{p}) \right\|_2^2}_{\text{inverse (action reconstruction)}} \tag{Eq. 5}
$$

变量含义：
- $\rho(x) = \sqrt{x^2 + \varepsilon^2}$ 是 **Charbonnier loss**（smooth L1的smooth版），比L2对outlier鲁棒。
- $\mathbf{v}_t \in \mathbb{R}^{H \times W \times 2}$ 是从 $\mathbf{o}_t, \mathbf{o}_{t+1}$ 提取的dense optical flow。
- $\mathbf{J}_\theta^{\dagger,\lambda}$ 是 λ-regularized pseudo-inverse，closed form为 $\tilde{\mathbf{J}}_\theta^{-1} = (\mathbf{J}_\theta \mathbf{J}_\theta^\top + \lambda \mathbf{I})^{-1} \mathbf{J}_\theta^\top$ (Eq. 8)。
- $w_{\mathbf{a}} = 0.3$。

**为什么joint training重要**？纯forward loss只保证"给我action能predict pixel motion"——但deployment时是反过来的：我有pixel motion（来自video model的flow），要recover action。如果只训forward，inverse problem可能ill-conditioned。Inverse term直接监督"inverse也要work"，相当于让J_θ学一个**numerically well-conditioned**的Jacobian field。

### 3.4 Inference: ridge-regressed pseudo-inverse

Deployment时拿到相邻预测帧 $(\hat{\mathbf{o}}_t, \hat{\mathbf{o}}_{t+1})$：

1. 提取flow: $\mathbf{v}_t = \text{flow}(\hat{\mathbf{o}}_t, \hat{\mathbf{o}}_{t+1})$
2. 用当前observation $\hat{\mathbf{o}}_t$ 作为conditioning，predict Jacobian field
3. Ridge regression求解action：

$$
\widehat{\delta \mathbf{a}}_t = \mathbf{J}_\theta^{\dagger,\lambda}(\hat{\mathbf{o}}_t)\,\mathbf{v}_t \;\triangleq\; \underset{\delta \mathbf{a} \in \mathbb{R}^n}{\operatorname{argmin}} \sum_{\mathbf{p}} \left\| \mathbf{J}_\theta(\mathbf{p}, \hat{\mathbf{o}}_t)\,\delta \mathbf{a} - \mathbf{v}_t(\mathbf{p}) \right\|_2^2 + \lambda \|\delta \mathbf{a}\|_2^2 \tag{Eq. 6}
$$

这是 per-pixel least-squares——把HW个pixel的2D flow constraint堆起来求解n维action。λ起regularization作用，避免J_θ的某些列接近0时导致solution blow up。

### 3.5 为什么Jacobian structure beats D-IDM——intuition

paper Sec. 4.2的controlled toy experiment (Figure 5) 给了清晰的picture。设置是2D "toy finger"，sweep DoF (1→5) 和 data quantity。

**D-IDM (Direct IDM) 的失败模式**：直接拿一对image (或 image + flow) 训个net regress action $\hat{\mathbf{a}} = f_\theta(\mathbf{o}_t, \mathbf{o}_{t+1})$。问题：
- **out-of-distribution generalization差**：每对新的image对应一个全新的regression target，net需要在 R^{2HW} → R^n 的map上做interpolation，data有限时generalize不出去。
- **随DoF增长迅速degrade**：n增大时output space指数级膨胀，相同data budget下每个action dimension分到的有效supervision变少。Table 2里 UniPi* 在16-DoF Allegro-Sim上的MSE是0.063，J-IDM是0.031——一半。

**J-IDM 的优势**：J_θ本质上学的是**局部linearization**，是一个**structural prior**：
- 模型只需要学"action channel j 如何 move每个pixel"——这是一个**smooth function of image**，因为同一物体在同一view下，物理motion对action的响应是局部连续可微的。
- 不同action dimension是**additive**的（线性叠加），所以n维action的Jacobian field = n个独立的flow field，每个的supervision都可以用全pixel的forward loss。
- Inverse时是**closed-form ridge regression**，没有learnable parameter——所以inverse本身不会OOD fail，只要forward Jacobian预测准。

**这个intuition用一句话总结**：D-IDM学的是一个高度nonlinear的inverse map，而J-IDM把inverse map固定成"linear ridge regression"，所有nonlinearity都被push到forward Jacobian的prediction上，而forward是well-posed的（forward Jacobian是image的smooth function）。

这是典型的"把inductive bias塞进architecture"的玩法——和NeRF把scene表示塞进positional encoding、和diffusion把score function塞进Gaussian noise schedule是一类思路。

---

## 四、Video World Model作为planner

### 4.1 Backbone

- **Wan-14B** (主实验) / Wan-1.3B (ablation)——open-weight video diffusion transformer [19]
- 主实验用**Large Video Planner (LVP) [10] warm-start**，因为LVP已经在robot video上pretrain过一遍
- 1 H200 node, 8 GPUs训练14B

### 4.2 Action-free post-training

VERA**不**给video model加action head，training objective就是纯generative video prediction（Eq. 7）：

$$
\mathcal{L}_{\text{vid}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{M} \|\hat{\mathbf{v}}_{\theta,i} - \mathbf{v}_i\|_2^2\right]
$$

这里 $\mathbf{v}_i$ 是diffusion forcing [20]里的linear-flow target，不是optical flow（注意区分——同一个符号v在不同地方含义不同）。

Diffusion Forcing的关键特性：不同frame可以assign不同noise level，所以能condition on clean/partially noised history + denoise future。这比vanilla video diffusion更适合closed-loop rollout，因为planner不会因为history frame的quantization/noise accumulate而drift。

History-guided video diffusion [21]是这个方法的延伸，论文也引用了。

### 4.3 Multi-view formatting

简单粗暴但有效：把多个camera view在pixel space concatenate成一个wide image（Panda triview就是 128×576 = 128×192×3）。这样同一个backbone就能handle单view / dual-view / triview，cross-embodiment训练时missing view用blank frame padding。

Table 5的camera ablation有个有意思的发现：**two external views的validation loss (0.11) < triview (0.13) < wrist-only (0.61)**。Wrist-only非常差，因为视角太窄、移动剧烈，video model很难predict整个scene。Triview虽然loss稍高但policy用起来更好，因为wrist view提供contact附近的local info。

### 4.4 Initialization ablation (Table 4)

| Init | Val MSE |
|---|---|
| Random init 14B | 0.40 |
| Wan pretrained 14B | 0.13 |
| LVP warm-start 14B | **0.10** |

预训练video prior的增益巨大（0.40→0.13），LVP进一步降到0.10。这印证了paper Sec.1的论点：video model的"visuospatial reasoning, prompt-following, embodiment generalization"是通过pretraining获得的，robot video post-training只是adaptation。Appendix C提到from-scratch训练根本达不到rollout stability。

---

## 五、Closed-loop系统细节

Algorithm 1的伪代码：

```
while episode not terminated:
    z_{t-N+1:t} ← E_tok(o_{t-N+1:t})           # encode context (N=6)
    ẑ_{t+1:t+M} ~ F_θ(· | z_{t-N+1:t}, c)       # sample M=4 future latents
    ô_{t+1:t+K} ← D_tok(ẑ_{t+1:t+K})           # decode K=1 committed prefix
    τ̂^vid = (o_t, ô_{t+1}, ..., ô_{t+K})
    τ̂^a ← G_φ(τ̂^vid)                          # J-IDM produce action chunk
    execute τ̂^a, append new obs, replan
```

Canonical operating point (Table 6)：
- N=6 context, M=4 look-ahead, K=1 committed, 40 UniPC steps

**K=1 是激进的replanning**——每次只执行一个latent transition对应的action chunk就replan。Paper Sec. 4说这是为了"contact-rich manipulation的频繁feedback"。Figure 8 ablation显示 K 增大会先提升后下降：太大失去feedback、太小过度replanning。

注意Wan tokenizer的causal temporal compression: N latent = 4N-3 RGB frames, M latent = 4M RGB frames。所以实际 K=1 latent = 4 RGB frames = 4 action transitions (取决于control rate)。

---

## 六、实验结果分析

### 6.1 Main closed-loop (Table 1)

| Model | Allegro-Sim (16-DoF) | Panda-Sim (7-DoF) | PushT-Sim (2-DoF) |
|---|---|---|---|
| UniPi* (D-IDM) | 0.0 / 0.0 | 0.0 / 0.0 | 74.4 / 84.8 |
| J-IDM (ours) | **70.0 / 70.0** | **94.0 / 94.0** | **92.5 / 95.5** |

格式是 success rate / task progress (%)。UniPi*在Allegro和Panda-Sim上完全失败（0%）——印证了D-IDM在高DoF下的崩溃。PushT的2-DoF下还能work，但J-IDM更好。

### 6.2 Action reconstruction MSE (Table 2)

| Model | Allegro-Sim | Panda-Sim | PushT-Sim | 5-joint finger |
|---|---|---|---|---|
| UniPi* | 0.063 | 0.38 | 0.071 | 0.047 |
| D-IDM + Flow | 0.044 | 0.09 | 0.059 | 0.030 |
| J-IDM (ours) | **0.031** | 0.19 | **0.046** | **0.017** |

注意Panda-Sim上D-IDM+Flow最好（0.09 vs J-IDM 0.19）——但closed-loop success J-IDM仍胜（94% vs D-IDM低）。这说明reconstruction MSE不是唯一指标，J-IDM可能在Panda上略有gap但closed-loop鲁棒性更强。这值得追问——可能的解释是ridge regression提供天然regularization，避免action overshoot。

### 6.3 vs VLA / WAM baselines (Figure 6)

在Panda-Real上对比 $\pi_{0.5}$ [5] 和 DreamZero [7]：

| 任务类型 | DreamZero | VERA | π_0.5 |
|---|---|---|---|
| Basic "push A / pick B" | 90% | 60% | 30% |
| Reasoning prompts (location-based, semantic-based) | 差 | **好** | 差 |
| Hidden button (occlusion) | 失败 | **成功** | 失败 |

**关键观察**：在basic任务上VERA反而落后DreamZero——paper解释是"VERA的失败一般来自video-to-action翻译步：failing rollouts的dreamed future其实是对的，但翻译成action时loss fidelity"。这是J-IDM还有提升空间。

但在**reasoning-heavy任务**上VERA明显胜出。paper的论点是：end-to-end VLA/WAM把action head塞进backbone会"dilute"video model的reasoning ability；VERA的decoupling保留video model原汁原味的reasoning。Hidden button任务（button藏在墙后，3个camera只有1个看得到，还有distractor）特别illuminating——$\pi_{0.5}$和DreamZero都fail to locate button，VERA不仅能find还能press。

---

## 七、Cross-embodiment demonstration

Paper Sec. 4.1的multi-embodiment claim：同一个video planner（在DROID + sim/real Allegro hands上混训）能generate两种embodiment的plan，配上不同J-IDM就能控制Panda arm和16-DoF Allegro hand。Figure 4展示Allegro在RGB-only下做in-hand cube reorientation——这是contact-rich、高DoF、多阶段的difficult task，传统method都很难。

这个结果的意义：**video planner是embodiment-agnostic的**，因为它只predict pixel motion。Embodiment-specific的信息全在J-IDM里。所以未来如果swap in更强的video backbone（比如新的Sora-class model），J-IDM不用重训。这是decoupling最大的practical benefit。

---

## 八、Limitations和open questions

Paper Sec. 5自承的limitations：
1. 仍需robot-specific video做planner post-training（不能纯靠generic video pretraining）
2. J-IDM supervision依赖off-the-shelf optical flow estimator [13, 14]——flow在textureless region、large displacement下不准
3. 无法处理force-based control（RGB-only，没有tactile）
4. 当predicted transition的pixel motion很小时，J-IDM的inverse conditioning变差（ridge regression的solution对noise敏感）

我自己补充几个追问：
- **J-IDM的linear approximation在小action increment下OK，但contact-rich场景下delta_a可能不"small"**——paper用K=1激进replanning来mitigate，但本质上是把nonlinearity推给"频繁replan + 短chunk"。如果未来想用更长chunk省compute，这个linear假设会break吗？
- **Action space的modality**：paper里action都是end-effector delta或joint position target。如果action space包含gripper state (discrete)、suction on/off这种categorical维度，Jacobian的连续性假设怎么处理？DROID里Panda action确实有gripper dim——paper没明说怎么处理。
- **Video model的"幻觉"**：video generative model有时会generate physically impossible motion（比如物体穿透）。J-IDM会faithfully翻译这种"坏plan"——可能产生危险action。Paper没讨论safety filtering。
- **Closed-loop latency**：每次planner call要40 UniPC steps + flow estimation + J-IDM forward + ridge solve。对real-time control（>10Hz）这太慢。Paper没给wall-clock numbers。Rhoda AI的Direct Video-Action [12]在industrial deployment可能更optimized。
- **Multi-view consistency**：J-IDM在不同view下predict的Jacobian是否consistent？如果不一致，inverse时不同pixel的constraint会互相打架。Paper的ridge regression自然average掉，但理论上这个multi-view fusion怎么保证物理一致性是个open question——作者组的Nature paper [17]用3D NeRF field来enforce consistency，VERA丢掉了这个，是不是trade-off了什么？

---

## 九、几个有趣的联想

### 9.1 和World Models / Dreamer的对比
VERA本质是**model-based RL的"decoupled dreamer"**：video model是world model，J-IDM是policy。但和Dreamer [22]的区别是：Dreamer在latent space rollout + backprop through dynamics；VERA在pixel space rollout + 用Jacobian做inverse。Dreamer需要differentiable dynamics，VERA不需要——所以能用black-box video diffusion。这是把"不同iablesimulator"换成"generative video model"的思路。

### 9.2 和Diffusion Policy的对比
Diffusion Policy [23]直接在action space做diffusion。VERA在pixel space做diffusion，然后通过Jacobian lift到action。VERA的好处是pixel space的diffusion prior可以从web video pretrain；action space diffusion需要robot action data。

### 9.3 和VGGT / DINOv2的连接
J-IDM用VGGT [18] init很有意思——VGGT是multi-view geometry grounded transformer，本身学过3D structure prior。J_θ预测的Jacobian field本质上隐含了scene的3D geometry（pixel motion ↔ 3D motion ↔ camera projection）。所以VGGT init可能给J_θ一个好的geometric inductive bias，让它不用从zero学projection geometry。

### 9.4 和SRC (Structured Robot Control) / Differentiable Simulation的呼应
VERA的Jacobian field本质是**learnable differentiable simulation**——学了"action → pixel motion"的local linearization。这和classical differentiable simulation [24]的区别是：VERA不假设rigid body / URDF / contact model，直接从data学。Trade-off是失去了物理精确性，gain了从raw video learn的能力。

### 9.5 "做梦的robot"这个metaphor
VERA的video model确实在做"dreaming"——它不execute action，只generate"如果成功了会怎么样"的visual hallucination。这和人类motor control的forward model假说有echo：人脑在plan movement时会simulate sensory consequences。VERA把这个forward simulation外包给video diffusion，把inverse (从desired sensory consequence到motor command) outsourcing给J-IDM。这个factorization在cognitive science里也有讨论 [25]。

### 9.6 Scaling laws的可能性
Paper没给J-IDM的scaling law，但Figure 5c暗示J-IDM的data-efficiency在DoF=5时比D-IDM好~2x。如果DoF更高（比如humanoid的50+ DoF），gap会不会继续扩大？还是Jacobian field本身的representation bottleneck会饱和？这是future work的重要方向——作者组的Nature paper [17]似乎已经在往这个方向push。

### 9.7 Failure mode: 当pixel motion很小时
Paper Sec. 5承认"faithful action recovery在predicted transition的observable pixel motion很小时degrade"。这其实是Jacobian inverse的**observability**问题——如果某action dimension不引起任何pixel变化（比如gripper close但object不在gripper内），Jacobian那一列接近0，ridge inverse会dominated by regularization。这和robotics里的"kinematic singularity"是同一个现象。可能的fix：用active sensing或者force/tactile input补充observable channel。

---

## 十、总结：VERA的positioning

VERA代表了一个**counter-trend**：当整个field在往"end-to-end VLA with bigger backbone + more action data"卷的时候，VERA说"等一下，video model本身已经很强了，我们只要解决好'从pixel motion到action'这个最后的translation就好"。

这个thesis的赌注是：
1. **Video model会继续变强**（Sora、Wan、Veo一路升级），如果VERA成立，robot policy自动受益
2. **Action data永远稀缺**，所以能avoid action-labeled pretraining的方案有structural advantage
3. **Inductive bias（Jacobian）在小data regime下 beats end-to-end learning**

这个赌注和Karpathy你自己提过的"software 2.0 vs software 1.5"的张力很有共鸣——VERA是用software 1.5的classical Jacobian structure来constrain software 2.0的video model，让两者各司其职。这种"hybrid explicit-implicit"路线在robotics里可能比纯end-to-end更practical。

我个人觉得J-IDM的下一步可能会和**force/tactile sensing**结合——把Jacobian field从pixel space扩展到multi-modal sensory space，这样能处理contact-rich任务里pixel motion很小但force很大的情况。或者往**embodiment-conditioned J-IDM**走（paper Future Work提到），让一个J-IDM serve multiple morphologies——这本质是学一个conditional Jacobian $\mathbf{J}_\theta(\mathbf{p}, \mathbf{o}, \text{embodiment\_id})$。

---

## References (with links)

[1] RT-2: https://proceedings.mlr.press/v229/zitkovich23a.html  
[2] OpenVLA: https://proceedings.mlr.press/v270/kim25c.html  
[3] Octo: https://octo-models.github.io/  
[4] π_0: https://www.roboticsproceedings.org/rss21/p010.html  
[5] π_0.5: https://proceedings.mlr.press/v305/black25a.html  
[6] GR-2: https://arxiv.org/abs/2410.06158  
[7] DreamZero / World Action Models: https://arxiv.org/abs/2602.15922  
[8] LingBot-VA: https://arxiv.org/abs/2601.21998  
[9] UniPi: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1d5b9233ad716a43be5c0d3023cb82d0-Abstract-Conference.html  
[10] Large Video Planner (LVP): https://arxiv.org/abs/2512.15840  
[11] π in the wild evaluation: https://arxiv.org/abs/2507.09061 (Wang et al. 2025)  
[12] Rhoda AI Direct Video-Action: https://www.rhoda.ai/research/direct-video-action  
[13] RAFT: https://arxiv.org/abs/2003.12039  
[14] AllTracker: https://arxiv.org/abs/2603.25739 (Harley et al. ICCV 2025)  
[15] TAPIR: https://arxiv.org/abs/2306.08637  
[16] MegaFlow: https://arxiv.org/abs/2603.25739  
[17] Controlling diverse robots by inferring jacobian fields (Nature): https://www.nature.com/articles/s41586-025-09170-0  
[18] VGGT: https://arxiv.org/abs/2503.11651  
[19] WAN: https://arxiv.org/abs/2503.20314  
[20] Diffusion Forcing: https://proceedings.neurips.cc/paper_files/paper/2024/hash/2aee1c4159e48407d68fe16ae8e6e49e-Abstract-Conference.html  
[21] History-guided video diffusion: https://proceedings.mlr.press/v267/song25b.html  
[22] Dreamer (Hafner et al.): https://dreamrl.github.io/  
[23] Diffusion Policy: https://journals.sagepub.com/doi/10.1177/02783649241273668  
[24] Differentiable Physics (Brax, Taichi, MJX): https://github.com/google/brax  
[25] Wolpert & Flanagan, "Motor control and prediction in the brain" (Nature Neuroscience, 2001): https://www.nature.com/articles/nn1201-729

如果你想我深挖哪一块——比如J-IDM的VGGT init具体怎么对接、Diffusion Forcing的noise schedule细节、或者VERA在DROID上post-training的具体recipe——可以继续问。
