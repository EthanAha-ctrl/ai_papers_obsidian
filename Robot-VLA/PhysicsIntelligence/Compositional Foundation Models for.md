---
source_pdf: Compositional Foundation Models for.pdf
paper_sha256: b8b083d804a6fc63349e100ab9600f5ce97384adce75b0265a6fa4cd0f7b11be
processed_at: '2026-08-03T16:44:26-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HiP 用人话版：三个experts拼成一个planner

## 核心idea一句话版本

**别想着训练一个什么都会的giant robot brain，而是找三个分别在text、video、robot action上各练各的expert，然后用一个"开会协调"的机制让它们达成consensus，输出一个long-horizon plan。**

---

## 1. 为什么不能just train one big model？

做robotics long-horizon task的traditional approach是：收集一堆 `(language instruction, camera video, robot action)` 三元组paired data，训练一个end-to-end network。比如 [RT-1](https://arxiv.org/abs/2212.06817), [Gato](https://www.deepmind.com/publications/a-generalist-agent), [PaLM-E](https://palm-e.github.io/)。

问题在哪？

1. **Paired data贵得离谱**。要让robot拿着kettle演示泡茶，hire人操作机械臂录video——一条trajectory可能要半小时人功。而互联网上纯text、纯video、纯robot demo单独都是ocean of data，但三者aligned的几乎没有。

2. **Closed-source model fine-tune不了**。你想在GPT-4上面接个robot head？OpenAI不给你weights。你只有API access。所以RT-2那种"把VLM改造成VLA"的路线对于普通人不可复制。

3. **Long-horizon是组合爆炸**。"收拾厨房"可能有上千种decomposition，每种都需要execution robustness。一个monolithic model要同时学会decomposition和execution，capacity竞争激烈。

HiP的insight：**既然internet上有海量独立modality的data，能不能让它们各练各的，最后拼起来？** 这就是"Compositional Foundation Models"的字面意思。

---

## 2. 三个expert各干各的

HiP把"看到goal → 做出action"这件事拆成三层，每层对应一个独立的foundation model：

### Layer 1: LLM当"项目经理"（Task Planning）

给一个language goal $g$，比如 "stack pink block on yellow block and place green block right of them"，LLM负责输出一串subgoal：

```
w1 = "pick white block, paint it pink in red bowl"
w2 = "pick another white block, paint it yellow in yellow bowl"
w3 = "pick pink block, place on yellow block"
w4 = "pick green block, place right of them"
```

LLM用 [GPT-3.5-turbo](https://arxiv.org/abs/2203.02155)，加几个few-shot examples prompt一下就行，**完全不动LLM的weights**。

**Intuition**：LLM在internet text上学了海量的"task decomposition prior"，比如泡茶要先烧水，打包行李要先找箱子。这个prior几乎free，因为text data已经在LLM里了。

### Layer 2: Video diffusion当"视觉想象师"（Visual Planning）

LLM给出subgoal $w_i$ 和当前observation $x_{i,1}$（camera拍的一张图），video diffusion负责"脑补"一段50帧的video，展示从 $x_{i,1}$ 到完成 $w_i$ 的visual过程。

模型是 [PVDM](https://arxiv.org/abs/2302.07685) 架构，先在 [Ego4D](https://ego4d-data.org/) 上pretrain（344k段first-person video），再在task-specific的100k trajectories上finetune。

**Intuition**：video model在Ego4D上学到了"物理世界things怎么动"的prior——block掉下来会bounce，drawer拉开有特定角度，kettle被拿起来会有手部occlusion。这个prior对任何tabletop manipulation task都有用，transfer成本低。

### Layer 3: Inverse dynamics当"动作执行员"（Action Planning）

video model给了50帧image sequence $\tau_x = \{x_1, x_2, ..., x_{50}\}$，inverse dynamics负责：对于每一对相邻frames $(x_t, x_{t+1})$，推断"要做什么action才能从 $x_t$ 变到 $x_{t+1}$"。

用 [VC-1](https://arxiv.org/abs/2303.18240) pretrained的ViT-B作为视觉backbone，linear head输出7维robot state（6个joint angle + 1个gripper）。Action就是相邻state的差。

**Intuition**：inverse dynamics最妙的一点是**它不需要action label**。只要你能从image推断robot state，action自动算出来。这让它能leverage任意ego-centric video data（不需要知道当时是哪个robot在动）。VC-1在ego-centric images上的pretrain让1K条robot trajectories就能达到10K的训练效果。

---

## 3. 为什么不能simple把它们串起来？

如果naive地把三个model接起来——LLM出subgoal → video model生成video → inverse dynamics出action——会爆炸。

**Example**：goal是"把mouse、sneakers、pepsi box、train装进brown box"，但mouse已经在box里了。LLM不知道这个事实，可能还是输出subgoal "put mouse in brown box"。video model收到这个subgoal和当前image（mouse已经在box里），只能生成"什么都不做"的video，但这看起来很weird。inverse dynamics看到两帧几乎一样的image，输出zero action。整个pipeline collapse。

更bad的情况：LLM说"open cabinet to find kettle"，但环境中根本没有cabinet。video model试图生成"cabinet opening"的video，但因为initial frame里没cabinet，只能生成"凭空冒出一个cabinet"的video——hallucination。Inverse dynamics看到cabinet突然出现，推断出物理impossible的action。

**根本问题**：三个model的joint distribution不consistent。它们各自max-likelihood的mode不在同一处。需要sample一个 $(W, \tau_x, \tau_a)$ triple使得三者**同时**高likelihood。

---

## 4. Iterative Refinement：让三个model"开会"

HiP的解法：用iterative refinement，让downstream model给upstream model反馈，upstream根据反馈调整自己的输出。

### Task Planning refinement：让LLM看着observation做决定

LLM先sample $M$ 个candidate subgoals $\{w_1, w_2, ..., w_M\}$（paper里 $M=6$ 或5或4）。然后训练一个小classifier $f_\phi$，输入 `(current observation image, goal, all candidate subgoals)`，输出"哪个subgoal是当前observation下最合理的下一步"。

这个classifier本质上是个density ratio estimator，approximate了"在已知当前observation的情况下，每个subgoal的posterior probability"。

**人话**：LLM先给6个candidate方案，然后一个vision-text classifier看一眼当前camera画面说"我觉得方案3最合理，因为mouse已经在box里了，方案1 redundant"。

**为什么不用VLM？** Paper试了用frozen [MiniGPT-4](https://arxiv.org/abs/2304.10592) 做这个classifier。在简单的paint-block domain上能match learned classifier，但在更复杂的object-arrange domain上fail了——VLM不认识 uncommon objects（如hard drive），还会hallucinate物体存在。所以learned classifier在小domain更可靠。但作者也说，未来VLM更强了可能可以替代。

### Visual Planning refinement：让video model尊重action feasibility

Video diffusion生成50帧video的过程是K步denoising（从pure noise逐渐refine到clean video）。HiP在每一步denoising加一个额外的guidance term。

具体做法：训练一个binary classifier $g_\psi$，输入一段video trajectory，输出"这段video能否对应一个physical feasible的action sequence"。

- Positive samples：真实trajectory的latent encoding
- Negative samples：把真实trajectory的frames随机shuffle后（局部物理impossible）的latent encoding

在denoising step $k$，modified noise prediction是：

```
ε̂ = ε_unconditional + ω × (ε_conditional - ε_unconditional) - ω' × ∇log g_ψ(1|trajectory)
```

第一项和第二项是standard classifier-free guidance（让video model按subgoal condition）。第三项是classifier guidance——推denoising朝"action-feasible trajectory"的方向走。

**人话**：video model每denoise一步，都要问action model"哥们，我目前imagine的这段trajectory你能不能执行？" action model回个分数，video model就稍微调整一下imagination让它更executable。

---

## 5. 整个pipeline长什么样

Algorithm 1的pseudocode翻译成人话：

```
1. 收到language goal g 和当前camera image x_t
2. LLM生成6个candidate subgoals {w_1...w_6}
3. classifier f_φ看一眼x_t和g，挑出最合理的w*
4. 用w*和x_t作为condition，video diffusion开始denoise：
   - 每一步denoise，先用standard classifier-free guidance按w* condition
   - 再加一个classifier guidance term，让trajectory对action model友好
5. 最终得到一段50帧video
6. 把video的相邻frame pair (x_t, x_{t+1})喂给inverse dynamics，得到action a_t
7. Robot执行第一个action，得到新observation
8. 如果还没完成整个subgoal（50帧没走完），继续执行下一个action
9. 完成一个subgoal后，回到step 1，重新规划下一个subgoal
```

**注意一个细节**：HiP在每个subgoal内是open-loop执行——生成50帧video后，所有action都是从这50帧推断的，执行过程中不重新看observation。只有完成一个subgoal后才看一次observation重新plan。这是limitation之一，但也是runtime能接受的原因。

---

## 6. 实验结果为什么这么强

看 [Table 1](https://hierarchical-planning-foundation-model.github.io/) 的数字：

HiP在没有oracle subgoals的情况下，paint-block unseen 72.8%，object-arrange unseen 75.4%，kitchen-tasks unseen 83.5%。

第二名通常是UniPi（也是Du et al.的工作，HiP的前身）或Gato（oracle subgoals）。

**为什么HiP这么强？**

1. **Hierarchy很重要**。HiP > UniPi，因为UniPi没LLM做high-level decomposition，让video model直接imagine整个long-horizon task，太coarse了，video model无法sustain那么长的coherent trajectory。HiP把long-horizon拆成4-6个short subgoal，每个subgoal生成50帧video，video model只在short horizon内imagine，更容易做好。

2. **Visual planning很重要**。HiP > Trajectory Transformer / Action Diffuser。这些baseline直接在action space用diffusion或transformer，没有visual imagination这一层。它们的pretrain data只有task-specific robot demo（几千条），而HiP的video model能pretrain on Ego4D（344k videos）。Internet-scale data的prior威力巨大。

3. **Task planning refinement很重要**。去掉 $f_\phi$ classifier后（即randomly pick subgoal from candidates），性能暴跌。这是HiP最critical的component。LLM给的candidates包含了正确答案，但LLM不知道哪个是当前observation下最合理的——必须靠vision-grounded classifier来选。

4. **Video model > RSSM**。换成DreamerV3的RSSM做visual planner后，kitchen-tasks从83%掉到46%。Video diffusion在visual richness上碾压recurrent state space model。

5. **VC-1 pretrain极有效**。1K trajectories + VC-1 init = 10K trajectories from scratch。这意味着即使你只有少量robot data，只要visual representation pretrain得好，也能train出不错的inverse dynamics。

6. **Combinatorial generalization**。unseen combination的performance几乎不掉。这是compositionality的胜利——LLM能compositional地decompose新task，video model能compositional地imagine新object组合，inverse dynamics是object-agnostic的。

---

## 7. 几个我个人的intuition和critique

### 7.1 Open-loop execution是最大limitation

每个subgoal生成50帧video，然后open-loop执行50个action。这在real world会fail——50步里任何一步有小的execution error，后面49步就跑偏了。

Paper用simulation，所以open-loop还行。Real world deployment必须每几步replan。但replan需要online video inpainting或partial denoising，不是从头generate。这是未来工作的明显方向。

### 7.2 Classifier guidance是approximation

$f_\phi$ classifier是approximate $p(x_{i,1}|w_i, g)/p(x_{i,1}|g)$，$g_\psi$ classifier是approximate $\prod_t p_\psi(a_{i,t}|x_{i,t}, x_{i,t+1})$。两者都是coarse approximation。

更principled的做法是用energy-based model直接estimate joint energy，然后用Langevin dynamics或HMC sample。但那会很慢。Classifier guidance是计算和accuracy的sweet spot。

### 7.3 Markov假设很强

Appendix A的factorization假设 $p_\theta(w_i|g, x_{i,1}, ...) \approx p_\theta(w_i|g)$。意思是LLM不直接看observation，只看goal给subgoal。这个假设让LLM和vision解耦，pretrain可以独立做。

但future VLM（GPT-4V, Gemini, Claude Vision）可以直接看image，这个假设就过时了。HiP framework可以adapt成：让VLM直接看image出subgoal，省掉 $f_\phi$ classifier。但需要VLM的visual reasoning足够强（目前还不行，paper的MiniGPT-4 ablation就fail了）。

### 7.4 三个model的scale不匹配

LLM是几百B参数（GPT-3.5），video diffusion可能几B（PVDM规模不大），inverse dynamics是ViT-B几百M。这种scale不匹配意味着LLM的prior很rich，但video model的prior相对weak。未来如果用Sora级video model，整个系统的physical reasoning能力会质变。但classifier guidance也会变贵——你需要对frozen 100B video model做gradient backprop，可能需要 [IP-Adapter](https://ip-adapter.github.io/) 之类的efficient adapter。

### 7.5 为什么不直接用VLA model？

[RT-2](https://robotics-transformer2.github.io/), [OpenVLA](https://openvla.github.io/), [Octo](https://octo-models.github.io/) 等 VLA model 走 monolithic 路线，把vision encoder + LLM + action head 放一起 end-to-end 训练。HiP 走 compositional 路线。

Trade-off：
- Monolithic：co-design representation，inference fast，但需要巨量paired data
- Compositional：每个component能leverage最大的internet data，但consistency enforcement是trick，inference slow

Long-horizon (>10 subgoals) 场景下，compositional 可能更有优势，因为 monolithic VLA 处理 long context horizon 时有 efficiency 问题，而 compositional 可以 hierarchical decompose。但 short-horizon 场景下 monolithic 更直接。

### 7.6 和 human cognition的类比

Paper的author Josh Tenenbaum和Leslie Kaelbling都是cognitive science背景。HiP的hierarchy其实对应human的dual-process theory：

- System 2（slow, deliberative）：LLM做task decomposition
- System 1（fast, automatic）：video imagination + motor execution

Iterative refinement对应"mental simulation"——你在做计划时会mentally simulate一下"如果这样做会怎样"，然后调整计划。Video diffusion就是这个mental simulator。

这个analogy不仅是表面上的，paper的framework确实实现了类似的functional structure。未来的AGI for embodied agents很可能长这样：一个fast generative world model做mental simulation，一个slow LLM做deliberative planning，一个low-level controller做execution。

### 7.7 关于subgoal granularity的sweet spot

Table 2 显示 "1 pick+place" per subgoal是最佳的。"2 pick+place" per subgoal性能掉（visual planner承担太多），"0.5 pick+place" per subgoal也没有额外提升（planning开销增加）。

**Intuition**：hierarchy的每一层应该承担roughly equal的complexity。如果high-level太coarse，low-level overload；如果high-level太fine，high-level overload。Pick+place是tabletop manipulation的"natural unit"，类似language的word——sub-word没意义，多-word phrase太coarse。

这个insight对hierarchical RL community早就有（[options framework](https://arxiv.org/abs/1606.02138) 的option length设计），但HiP在foundation model context下重新验证了它。

### 7.8 和 SayCan 的本质区别

[SayCan](https://say-can.github.io/) 也是LLM + low-level skills。但SayCan需要predefined skill library（每个skill是一个trained policy + 一个affordance function）。新skill不能on-the-fly合成。

HiP用video planner替代skill library。任何能用language描述的subgoal，video model都能尝试imagine出来。这是从"fixed primitives"到"generative skills"的shift，极大扩展了generalization。

但代价是：SayCan的affordance function直接对应physical executability（because skill是trained policy），而HiP需要额外的 $g_\psi$ classifier来approximate executability。Generative flexibility换来的是consistency的difficulty。

### 7.9 为什么不直接让VLM生成subgoal sequence？

Paper试过让MiniGPT-4直接生成next subgoal given image and goal，完全fail了（Appendix C.1）。原因是VLM的in-context learning能力不够强，给5个few-shot examples都不能让它学到"在这种observation下应该output什么subgoal"。

这其实揭示了一个interesting fact：当前的VLM在"vision-grounded reasoning"上还不够强，需要learned classifier这种task-specific component做bridge。但这个gap应该会fast close——未来的GPT-5V级别model可能直接做这个事，HiP framework就可以简化。

### 7.10 Ego4D pretrain的hidden value

Paper的Figure 5 显示，即使data减少到50%，有Ego4D pretrain的video model依然显著好于no pretrain。这说明Ego4D里capture的"first-person physical world prior"是robotics的strong inductive bias。

**Intuition**：Ego4D是人类拿着相机录的第一人称video，包含了大量"手拿东西、放东西、开门"的visual patterns。虽然不是robot演示，但visual motion pattern是transferable的。这和 [R3M](https://arxiv.org/abs/2203.12601) 的insight一致——ego-centric human video是robotics的rich pretrain source。

未来如果有大规模的human hand manipulation video dataset（如 [EgoExo4D](https://egoexo4d-data.org/)），这个effect会更显著。

---

## 8. 对整个robotics foundation model field的影响

HiP代表了"compositional foundation model" paradigm的一个milestone。它证明了：

1. **不需要paired data也能做long-horizon planning**。只要每个modality的independent data足够，compositional approach能work。
2. **Iterative refinement是compositional consistency的principled解法**。不是简单的pipeline，而是有feedback loop的joint inference。
3. **Video generation > Action generation for planning**。在observation space做generative planning比在action space做更transferable，因为observation是embodiment-agnostic的。

未来几个可能的evolution：
- 把video model换成Sora-level foundation model
- 把LLM换成multimodal VLM，省掉 $f_\phi$ classifier
- 加online replan，close the open-loop gap
- 加touch或audio modality，扩展sensing hierarchy
- 把inverse dynamics换成 [Vision-Language-Action model](https://robotics-transformer2.github.io/)，让action execution也享受LLM prior

HiP是"LEGO积木式AI系统"的示范——每个component可以是独立的SOTA model，通过精心设计的interface组合起来。这个paradigm在LLM era会越来越重要，因为单一giant model的training cost prohibitive，而compositional approach让small labs也能build SOTA系统。

---

## 9. 几个可能的批判性看法

虽然HiP很elegant，但有一些值得skeptical的点：

### 9.1 Simplicity vs elegance的trade-off

HiP有3个model + 2个classifier + 2个guidance scale + 复杂的training pipeline。对比之下，[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 一个model就work。HiP的complexity是否justified？

Paper的argument是compositional generalization和data efficiency。但如果你有足够paired data，monolithic approach可能更简单且性能更好。HiP的优势在"data scarce + long horizon"的corner case。

### 9.2 三个domain的specifics

paint-block, object-arrange, kitchen-tasks 都是 tabletop manipulation with 4-6 subgoals。这对real-world的long-horizon（比如"做一顿饭"需要50+ subgoals）还差得远。HiP能否scale到真正long-horizon是open question。

### 9.3 Open-loop execution在real world的失败

50步open-loop在simulator work，在real world必然fail。这个gap是deployment的核心障碍。

### 9.4 Classifier guidance的hyperparameter sensitivity

Table 4 显示 $\omega'$ 从0.5到2.0，性能从71%波动到68%，相对稳定但需要tune。如果domain变了，guidance scale是否需要retune？这影响deployment的robustness。

### 9.5 LLM的role可能被overrated

HiP的LLM只做"output candidate subgoals"，真正的decision在 $f_\phi$ classifier。如果LLM的candidate quality很差，classifier也救不回来。Paper用了GPT-3.5，quality还不错，但如果换成更弱的LLM（如Llama-7B），可能candidate set质量下降明显。

---

## 10. 给Andrej的final take-away

如果你要给HiP一个elevator pitch：

**HiP是一个用LLM做项目经理、video diffusion做视觉想象、inverse dynamics做执行的hierarchical robot planner。三个model独立pretrain，通过classifier guidance做iterative consensus来保证三者output compatible。它证明了compositional foundation model approach在long-horizon planning上能beat monolithic model，尤其是在paired data稀缺的场景。**

它的intellectual贡献主要有三：
1. **把long-horizon planning factorize成三个independent learnable subproblem**，让每个subproblem都能leverage internet-scale data
2. **用classifier guidance实现cross-model consistency**，principled且efficient
3. **验证video generation是planning的good interface**，embodiment-agnostic且physical informative

未来的 embodied AGI 很可能就是 HiP 这种 structure：large multimodal reasoner + generative world model + low-level controller，三者通过某种 probabilistic interface 协作。HiP 给这个 paradigm 提供了一个 concrete instantiation。

---

## Web References

1. HiP 项目主页: https://hierarchical-planning-foundation-model.github.io/
2. UniPi (HiP 前身): https://arxiv.org/abs/2302.00112
3. SayCan (对比 baseline): https://say-can.github.io/
4. Ego4D (video pretrain): https://ego4d-data.org/
5. VC-1 (inverse dynamics 初始化): https://arxiv.org/abs/2303.18240
6. PVDM (video diffusion 架构): https://arxiv.org/abs/2302.07685
7. Latent Diffusion Models: https://arxiv.org/abs/2112.10752
8. Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
9. Gato (baseline): https://www.deepmind.com/publications/a-generalist-agent
10. RT-1: https://arxiv.org/abs/2212.06817
11. RT-2 (VLA model): https://robotics-transformer2.github.io/
12. PaLM-E: https://palm-e.github.io/
13. OpenVLA: https://openvla.github.io/
14. Octo: https://octo-models.github.io/
15. Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
16. R3M (ego-centric pretrain): https://arxiv.org/abs/2203.12601
17. MAE (VC-1 训练目标): https://arxiv.org/abs/2111.06377
18. CLIPort (SayCan 的 skills): https://cliport.github.io/
19. Flan-T5: https://arxiv.org/abs/2210.11416
20. MiniGPT-4 (VLM baseline): https://arxiv.org/abs/2304.10592
21. Socratic Models: https://socraticmodels.github.io/
22. Pre-trained LMs for Decision Making (Du & Li): https://arxiv.org/abs/2210.03710
23. Density Ratio Estimation: https://arxiv.org/abs/2305.00869
24. Compositional Visual Generation with EBMs: https://arxiv.org/abs/2006.06051
25. Diffuser: https://arxiv.org/abs/2205.09991
26. Trajectory Transformer: https://arxiv.org/abs/2106.01339
27. Foundation Models for Decision Making Survey: https://arxiv.org/abs/2303.04129
28. PDSketch (paint-block 灵感): https://arxiv.org/abs/2210.02328
29. KitchenShift: https://openreview.net/forum?id=DdglKo8hBq0
30. IP-Adapter (efficient adapter): https://ip-adapter.github.io/
31. EgoExo4D: https://egoexo4d-data.org/
32. Options Framework (hierarchical RL): https://arxiv.org/abs/1606.02138

---

# Compositional Foundation Models for Hierarchical Planning (HiP) 深度解析

## 1. 核心Motivation与Intuition

这篇paper来自MIT Improbable AI Lab（Pulkit Agrawal组）和MIT-IBM Watson AI Lab，作者Anurag Ajay, Seungwook Han, Yilun Du等。核心problem非常深刻：**long-horizon decision making需要跨越spatial和temporal scales的hierarchical reasoning**，而我们当前手头的foundation models分别擅长不同的modality——LLM擅长semantic planning，video diffusion擅长visual imagination，inverse dynamics擅长ego-centric control。

传统approach有两种：
1. **Monolithic foundation model**（如RT-1, Gato）：收集paired language-vision-action data训练一个giant network。问题：data collection极其expensive，无法scale，且fine-tune closed-source models like GPT-4/PaLM impossible。
2. **Fine-tune LLM with vision+action**（如PaLM-E）：依然需要大量paired data。

HiP的核心insight：**能否把分别在不同modality上独立训练的expert foundation models组合起来，使得它们的joint distribution对应于long-horizon planning的solution？**

这背后的key assumption是：互联网上有海量的纯text、纯video、纯ego-centric robot data，但三者paired的数据稀缺。HiP把这三个独立pretrain的model当作different expert，然后通过iterative refinement来enforce cross-level consistency。

Project page: https://hierarchical-planning-foundation-model.github.io/

---

## 2. Hierarchical Factorization的数学结构

### 2.1 Markov假设下的factorization

paper的核心公式(1)：

$$
p_\Theta(W, \{\tau_x^i\}, \{\tau_a^i\} | g, x_{1,1}) = \underbrace{\left(\prod_{i=1}^{N} p_\theta(w_i|g)\right)}_{\text{task planning}} \underbrace{\left(\prod_{i=1}^{N} p_\phi(\tau_x^i|w_i, x_{i,1})\right)}_{\text{visual planning}} \underbrace{\left(\prod_{i=1}^{N}\prod_{t=1}^{T-1} p_\psi(a_{i,t}|x_{i,t}, x_{i,t+1})\right)}_{\text{action planning}}
$$

**变量含义**：
- $W = \{w_1, w_2, ..., w_N\}$：subgoal序列（language tokens），$N$为subgoal数量
- $\tau_x^i = \{x_{i,1}, x_{i,2}, ..., x_{i,T}\}$：第$i$个subgoal对应的image trajectory（video），$T=50$为时间horizon
- $\tau_a^i = \{a_{i,1}, ..., a_{i,T-1}\}$：第$i$个subgoal对应的action trajectory
- $g$：language goal（e.g. "stack pink block on yellow block and place green block right of them"）
- $x_{1,1}$：初始observation（agent的第一个observation）
- $x_{i,1}$：第$i$个subgoal阶段的初始observation（等于上一个阶段的final observation）
- $\theta$：LLM参数，$\phi$：video diffusion参数，$\psi$：inverse dynamics参数

### 2.2 Markov假设的derivation（Appendix A）

完整factorization其实是：
$$
p_\Theta(\cdot) = \prod_{i=1}^{N} p_\theta(w_i | g, x_{i,1}, w_{<i}, \tau_x^{<i}, \tau_a^{<i}) \prod_{i=1}^{N} p_\phi(\tau_x^i | w_{\le i}, x_{i,1}, g, \tau_x^{<i}, \tau_a^{<i}) \prod_{i=1}^{N} p_\psi(\tau_a^i | \tau_x^i, w_i, x_{i,1}, g)
$$

通过三个Markov假设简化到公式(1)：
1. **LLM independent of observation**：$p_\theta(w_i|g, x_{i,1}, ...) \approx p_\theta(w_i|g)$（因为LLM是purely text-trained的）
2. **Video conditionally independent of $g$ given $w_i, x_{i,1}$**：因为visual trajectory只取决于subgoal和当前state
3. **Action只取决于 $(x_{i,t}, x_{i,t+1})$**：inverse dynamics的Markov property

**Intuition**：这个factorization本质上把"看到goal→想象整段视频→执行"这个human reasoning process拆成三个可独立学习的sub-problem，每个sub-problem对应一种internet上abundant的数据类型。

---

## 3. Iterative Refinement：让独立训练的models达成consensus

### 3.1 问题：naive max-likelihood sampling导致inconsistency

如果naive地取每个level的max-likelihood输出：
- LLM可能说"找kettle in cabinet"——但环境中根本没有cabinet
- Video diffusion可能生成visual上plausible但物理上unachievable的trajectory
- Inverse dynamics可能从impossible image pair推断出invalid action

**核心问题**：$p_\theta, p_\phi, p_\psi$是独立训练的，它们的joint distribution可能concentrate在完全不同的region。我们需要sample $(W, \{\tau_x^i\}, \{\tau_a^i\})$使得三者likelihood同时高。

### 3.2 Task Planning的一致性refinement

目标公式(2)：
$$
w_i^* = \arg\max_{w_i} p_{LLM}(w_i|g) \cdot p_\phi(\tau_x^i|w_i, x_{i,1})
$$

但 $p_\phi(\tau_x^i|w_i, x_{i,1})$ 需要 generate video 来评估，太expensive。

**Key trick**：用Bayes rule改写。要maximize的其实是 $p(w_i | g, x_{i,1})$（公式3），展开：

$$
w_i^* = \arg\max_{w_i} \left[\log p_{LLM}(w_i|g) + \log\frac{p(x_{i,1}|w_i, g)}{p(x_{i,1}|g)}\right]
$$

**变量解释**：
- 第一项 $\log p_{LLM}(w_i|g)$：LLM给的language prior
- 第二项 $\log\frac{p(x_{i,1}|w_i, g)}{p(x_{i,1}|g)}$：当前observation $x_{i,1}$ 在 "已知subgoal $w_i$" 条件下的likelihood ratio——即"如果subgoal是 $w_i$，当前observation有多plausible"

**Density ratio estimation via classifier**：

paper用multi-class classifier $f_\phi(x_{i,1}, \{w_j\}_{j=1}^M, g)$来估计这个ratio。理论基础是[Srivastava et al. 2023](https://arxiv.org/abs/2305.00869)的multinomial logistic regression density ratio estimation：

$$
\log r(x) = \hat{h}_p(x) - \hat{h}_q(x)
$$

其中 $\hat{h}_i(x)$ 是classifier第 $i$ 类的unnormalized log probability。training data $\mathcal{D}_{classify} := \{x_{i,1}, g, \{w_j\}_{j=1}^M, i^*\}$，$i^*$是正确subgoal label。

**Algorithm 1中的实现**：
```
Generate M subgoals w_i ~ p_LLM(w_i | g)
W ← {w_i}_{i=1}^M
w ← argmax_w f_φ(x_t, W, g)
```

### 3.3 Visual Planning的一致性refinement

目标公式(4)：
$$
(\tau_x^i)^* = \arg\max_{\tau_x^i} p_\phi(\tau_x^i|w_i, x_{i,1}) \prod_{t=1}^{T-1} p_\psi(a_{i,t}|x_{i,t}, x_{i,t+1})
$$

直接计算 $\prod_t p_\psi(a_{i,t}|x_{i,t}, x_{i,t+1})$ 在每个diffusion denoising step都昂贵。

**Approximation**：用binary classifier $g_\psi(\tau_x^i)$ 来判断 "这个trajectory是否lead to high-likelihood action trajectory"。

**Training data for $g_\psi$**：
- Positive samples ($y=1$)：从 $\mathcal{D}_{video}$ 真实trajectory的latent编码 $\tau_z^i = v_{enc}(\tau_x^i)$
- Negative samples ($y=0$)：随机shuffle frames后的trajectory $(\tau_z^i)' = v_{enc}(\sigma(\tau_x^i))$

**Shuffling策略细节**：trajectory长度50，随机选5个frame，每个与其相邻frame（前一个或后一个）交换。这制造了"局部物理不合理"的负样本。

### 3.4 Modified diffusion denoising with dual guidance

Algorithm 1中的核心update：

$$
\hat{\epsilon} := \epsilon_\phi((\tau_x)_k, x_t, k) + \omega\left(\epsilon_\phi((\tau_x)_k, x_t, w, k) - \epsilon_\phi((\tau_x)_k, x_t, k)\right) - \omega' \nabla_{(\tau_x)_k} \log g_\psi(1|(\tau_x)_k)
$$

**逐项解析**：
- $\epsilon_\phi((\tau_x)_k, x_t, k)$：unconditional noise prediction（classifier-free guidance的baseline）
- $\omega(\epsilon_\phi(\cdot, w, k) - \epsilon_\phi(\cdot, k))$：**classifier-free guidance** term，$\omega$ 控制subgoal conditioning的强度（[Ho & Salimans 2022](https://arxiv.org/abs/2207.12598)）
- $-\omega' \nabla_{(\tau_x)_k} \log g_\psi(1|(\tau_x)_k)$：**classifier guidance** term，$\omega'$ 控制"action feasibility"的引导强度。负号因为我们要maximize $\log g_\psi(1|\cdot)$，所以沿其gradient ascent

注意：classifier $g_\psi$ 在latent space操作，$\nabla_{(\tau_x)_k}$ 实际上是 $\nabla_{(\tau_z)_k}$，但论文notation统一用 $\tau_x$。

**关键hyperparameter**：$\omega=4, \omega'=1$（或1.25，见Table 4 ablation）。

---

## 4. 架构细节

### 4.1 Visual Planning: PVDM-based Latent Video Diffusion

paper borrow [PVDM (Yu et al. 2023)](https://arxiv.org/abs/2302.07685) 的架构：

**Tri-plane latent representation**：
$$
\tau_z = [\tau_z^T, \tau_z^H, \tau_z^W]
$$
- $\tau_z^T \in \mathbb{R}^{C \times H' \times W'}$：捕捉时间维度的variation
- $\tau_z^H \in \mathbb{R}^{C \times T \times W'}$：捕捉height维度的variation  
- $\tau_z^W \in \mathbb{R}^{C \times H' \times T}$：捕捉width维度的variation

**Dimension具体值**：
- $T=50$（time horizon）
- $H=48, W=64$（video spatial dimensions）
- $C=4$（latent codebook dim）
- $H'=12, W'=8$（latent spatial dims）

**Autoencoder $v_{enc}$**：用transformer把 $\tau_x \in \mathbb{R}^{T \times H \times W}$ project到tri-plane。

**Noise model $\epsilon_\phi$**：2D UNet (from [Latent Diffusion Model, Rombach et al. 2022](https://arxiv.org/abs/2112.10752)) with cross-attention注入subgoal的text embedding。

**Pretraining pipeline**：
1. Pretrain $v_{enc}$ + $\epsilon_\phi$ on [Ego4D](https://ego4d-data.org/)（344k clips, each 50 frames resized to 48×64）
2. Finetune on $\mathcal{D}_{video}$（~100k observation trajectories + subgoal text）

**Loss for autoencoder**：pixel-level L1 reconstruction + negative perceptual similarity（无adversarial loss，简化自PVDM）

**Loss for noise model**：
$$
\mathbb{E}_{k \sim [1,K], \tau_z, w, x \sim \mathcal{D}, \epsilon \sim \mathcal{N}(0, I)} \left[\|\epsilon - \epsilon_\phi((\tau_z)_k, l_{enc}(w), v_{enc}(x), k)\|^2\right]
$$

w以10%概率替换为null token，实现classifier-free guidance的conditional+unconditional联合训练。

### 4.2 Action Planning: VC-1 initialized Inverse Dynamics

**Architecture**：
- ViT-B backbone，用 [VC-1 (Majumdar et al. 2023)](https://arxiv.org/abs/2303.18240) initialization
- VC-1是在ego-centric images上用 [MAE (He et al. 2022)](https://arxiv.org/abs/2111.06377) 训练的ViT-B
- 输入 $x_{i,t} \in \mathbb{R}^{48 \times 64 \times 3}$ → ViT-B → 768-dim latent → Linear → 7-dim robot state $s_{i,t}$

**7-dim robot state**：
- 前6维：joint angles
- 第7维：gripper state (open/closed)

**Action representation**：
$$
a_{i,t}[:6] = s_{i,t+1}[:6] - s_{i,t}[:6] \quad \text{(joint angle differences)}
$$
$$
a_{i,t}[-1] = s_{i,t+1}[-1] \quad \text{(absolute gripper state)}
$$

**Training**：MSE loss on $s_{i,t}$，数据集 $\mathcal{D}_{inv}$ 仅需1K-3.5K trajectories（得益于VC-1 pretraining）。

**Key insight**：inverse dynamics不需要action label！只要从 $(x_t, x_{t+1})$ 推出 $s_t, s_{t+1}$，action是state difference自动算出。这避免了action space异构问题。

### 4.3 Subgoal Classifier $f_\phi$

- Vision encoder：[ResNet-18](https://arxiv.org/abs/1512.03385) (ImageNet pretrained)
- Language encoder：[Flan-T5-Base](https://arxiv.org/abs/2210.11416)（frozen）
- Architecture：concatenate(encoded image, encoded subgoals, encoded goal) → MLP with hidden layers [512, 256, 128] → M classes

**M classes per domain**：
- paint-block: 6 classes
- object-arrange: 5 classes  
- kitchen-tasks: 4 classes

### 4.4 Action Feasibility Classifier $g_\psi$

- ResNet-9 encoder分别处理 $\tau_z^T, \tau_z^H, \tau_z^W$
- Concatenate → MLP [256, 128] → 1-dim output (binary)
- Binary cross-entropy loss

---

## 5. 实验环境与数据生成

### 5.1 三个Long-Horizon Environments

| Domain | 来源inspiration | 任务特点 | Subgoal数量 |
|--------|----------------|---------|------------|
| paint-block | [Mao et al. PDSketch](https://arxiv.org/abs/2210.02328) | 涂色blocks然后stack/place | 4-6 |
| object-arrange | [CLIPort, Shridhar et al.](https://arxiv.org/abs/2109.12098) | 把特定objects放入box（含脏物需先清洗） | 3-5 |
| kitchen-tasks | [KitchenShift, Xing et al.](https://openreview.net/forum?id=DdglKo8hBq0) | 7种kitchen subtask随机组合 | 3-4 |

**Combinatorial generalization设计**：
- paint-block unseen：novel combinations of object colors
- object-arrange unseen：novel combinations of object categories（55个objects中sample 7，含3个distractor，1个dirty）
- kitchen-tasks unseen：novel combinations of subtasks（7个subtask中选4，含50%概率1个已完成）

### 5.2 Baselines

1. **Transformer BC**（[Janner et al. 2021](https://arxiv.org/abs/2106.01339)）：goal-conditioned transformer，oracle subgoals
2. **Gato**（[Reed et al. 2022](https://arxiv.org/abs/2205.06175)）：generalist agent transformer
3. **Trajectory Transformer**：sequence model
4. **Action Diffuser**（[Diffuser, Janner et al. 2022](https://arxiv.org/abs/2205.09991)）：diffusion over actions
5. **UniPi**（[Du et al. 2023](https://arxiv.org/abs/2302.00112)）：video diffusion直接生成整段plan，跳过task planning
6. **SayCan**（[Ahn et al. 2022](https://arxiv.org/abs/2204.01691)）：LLM + CLIPort affordance skills

### 5.3 主实验结果（Table 1）

| Model | Paint-block Seen | Paint-block Unseen | Object-arrange Seen | Object-arrange Unseen | Kitchen Seen | Kitchen Unseen |
|-------|------------------|--------------------|--------------------|----------------------|--------------|----------------|
| Transformer BC (oracle) | 8.3±1.9 | 5.1±1.6 | 10.2±2.9 | 7.3±1.7 | 48.4±21.6 | 32.1±24.2 |
| Gato (oracle) | 31.2±2.4 | 28.6±2.9 | 37.9±3.3 | 36.5±3.2 | 70.2±10.8 | 66.8±12.2 |
| Trajectory Transformer (oracle) | 22.1±2.1 | 22.3±2.5 | 30.5±2.3 | 29.8±2.9 | 66.4±20.7 | 52.1±22.3 |
| Action Diffuser (oracle) | 21.6±2.6 | 18.2±2.3 | 29.2±2.4 | 27.6±2.1 | 65.9±23.2 | 55.1±22.8 |
| **HiP (oracle subgoals)** | **81.2±1.8** | **79.6±1.9** | **91.8±2.9** | **92.3±2.3** | **92.8±7.1** | **89.8±7.6** |
| UniPi | 37.2±3.8 | 35.3±3.2 | 44.1±3.1 | 44.2±2.9 | 74.6±14.8 | 73.4±11.2 |
| SayCan | 67.2±3.3 | 62.8±3.7 | 70.3±2.6 | 66.9±2.8 | — | — |
| **HiP (Ours)** | **74.3±1.9** | **72.8±1.7** | **75±2.8** | **75.4±2.6** | **85.8±9.4** | **83.5±10.2** |

**关键观察**：
1. HiP在没有oracle subgoals的情况下，依然大幅超过有oracle subgoals的baselines（除了kitchen-tasks上UniPi接近）
2. HiP在unseen combination上performance保持intact（combinatorial generalization）
3. kitchen-tasks上SayCan无法运行——因为缺乏clear-cut primitives，体现HiP的video planner泛化优势

---

## 6. 关键Ablation Studies

### 6.1 Pretraining Effects

**Video diffusion pretraining (Figure 5)**：
- 在dataset大小reduced到75%、50%时，Ego4D pretraining依然保持lower FVD (Fréchet Video Distance) 和 higher success rate
- 随domain data减少，pretrain vs no-pretrain的gap**扩大**

**Inverse dynamics pretraining (Figure 6)**：
- VC-1 init + 1K trajectories ≈ no-init + 10K trajectories（paint-block和object-arrange）
- VC-1 init + 3.5K trajectories ≈ no-init + 10K trajectories（kitchen-tasks）
- ResNet-18需要2.5K-6K trajectories才能接近VC-1 init的水平
- **Intuition**：ego-centric internet images蕴含的visual prior极大降低robotic data需求

### 6.2 Iterative Refinement重要性（Figure 7）

- Task plan refinement（$f_\phi$ classifier）：**critical**，去掉后performance暴跌
- Visual plan refinement（$g_\psi$ classifier）：improvement较小但consistent
- Frozen VLM (MiniGPT-4) as classifier：paint-block上接近learned classifier，但object-arrange上明显劣化

**VLM failure modes**（Figure 8）：
1. 无法识别uncommon objects（e.g. computer hard drives）
2. Hallucination of object presence

### 6.3 Subgoal Granularity（Table 2）

| Model | Paint-block Seen | Paint-block Unseen |
|-------|------------------|--------------------|
| HiP (more granular) | 74.5±1.8 | 73.1±2.1 |
| HiP (Standard) | 74.3±1.9 | 72.8±1.7 |
| HiP (less granular) | 61.8±3.1 | 58.2±3.4 |
| UniPi | 37.2±3.8 | 35.3±3.2 |

**Intuition**：subgoal granularity影响visual planner的performance。太coarse（如UniPi直接imagine整段）则video diffusion失败；太fine也没有额外好处。**Standard granularity（一个pick+place operation）是sweet spot**。

### 6.4 Visual Planner Architecture Choice（Table 3）

| Model | Paint-block Seen | Paint-block Unseen | Object-arrange Seen | Object-arrange Unseen | Kitchen Seen | Kitchen Unseen |
|-------|------------------|--------------------|---------------------|----------------------|--------------|----------------|
| HiP (RSSM) | 70.2±2.4 | 69.5±1.6 | 59.6±3.8 | 59.2±3.9 | 50.6±16.2 | 46.8±19.4 |
| HiP | 74.3±1.9 | 72.8±1.7 | 75±2.8 | 75.4±2.6 | 85.8±9.4 | 83.5±10.2 |

用DreamerV3的RSSM作为visual model，在visually complex domain（kitchen）上performance严重退化。**Video diffusion在复杂visual generation上有本质优势**。

### 6.5 Guidance Scale Sensitivity（Table 4）

$\omega'$ 在 $\{1, 1.25\}$ 时最优，过大（≥1.5）则performance退化——过度强调action feasibility会扭曲video distribution。

---

## 7. Runtime Analysis（Table 5）

| Domain | Subgoal Candidate Gen | Subgoal Classification | Visual Planning/subgoal | Action Planning/subgoal | Action Execution/subgoal | Episodic Total |
|--------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------|
| Paint-block | 1.85s | 0.41s | 7.32s | 0.91s | 6.35s | 80.61s |
| Object-arrange | 1.9s | 0.43s | 7.39s | 0.89s | 9.57s | 78.71s |
| Kitchen-tasks | 1.81s | 0.41s | 7.35s | 0.98s | 1.28s | 40.37s |

**Bottleneck**：Visual planning（~7.3s/subgoal × ~6 subgoals ≈ 44s占episodic一半）。未来可用diffusion加速方法（如[DDIM, Consistency Models, Operator Learning](https://arxiv.org/abs/2202.00512)）。

---

## 8. 与相关工作的联系和Intuition

### 8.1 思想渊源

1. **[Du et al. 2020 "Compositional Visual Generation with EBMs"](https://arxiv.org/abs/2006.06051)**：通过取product of distributions实现compositionality。HiP的joint distribution本质上是三个energy product。

2. **[Li et al. 2022 "Pre-trained LMs for Interactive Decision-Making"](https://arxiv.org/abs/2210.03710)**（Yilun Du和Shuang Li）：iterative consensus procedure用于组合foundation models。HiP直接继承这个framework。

3. **[UniPi, Du et al. 2023](https://arxiv.org/abs/2302.00112)**：用video diffusion做image-space planning。HiP把UniPi作为visual planner的building block，并通过LLM layer扩展到long-horizon。

4. **[SayCan, Ahn et al. 2022](https://say-can.github.io/)**：LLM作为high-level policy + skill value functions。HiP用video planner替代fixed skill set，gain generalization。

5. **[Socratic Models, Zeng et al. 2022](https://arxiv.org/abs/2204.00598)**：zero-shot组合multimodal models。HiP更结构化（hierarchical + iterative refinement）。

6. **[Density Ratio Estimation, Gutmann & Hyvärinen 2012](https://jmlr.org/papers/v13/gutmann12a.html), [Srivastava et al. 2023](https://arxiv.org/abs/2305.00869)**：用classifier估计density ratio——这是$f_\phi$的理论基础。

### 8.2 与foundation model for decision making literature的关系

[Foundation Models for Decision Making survey, Yang et al. 2023](https://arxiv.org/abs/2303.04129) 指出两条路线：
- **Monolithic**：一个giant model吃所有modality（RT-1, Gato, PaLM-E）
- **Compositional**：组合多个expert models（SayCan, Socratic, HiP）

HiP在compositional路线上推进了三件事：
1. 用video diffusion代替skill primitives（解决SayCan的fixed skill set限制）
2. 用iterative refinement解决跨model consistency（解决Socratic Models的zero-shot limitation）
3. 三层hierarchy而非flat composition

### 8.3 与hierarchical RL的对比

传统hierarchical RL（如option-critic, feudal networks）需要：
1. 在同一env中训练high-level和low-level policy
2. 定义subgoal space或option space

HiP的优势：
- Subgoal space是natural language（LLM直接给出）
- High-level (LLM) 和 low-level (video+action) 完全独立训练
- 通过classifier guidance实现"软"hierarchical consistency

### 8.4 与Diffusion Planner的关系

[Diffuser, Janner et al. 2022](https://arxiv.org/abs/2205.09991) 和 [Planning with Diffusion](https://arxiv.org/abs/2205.09991) 在action space用diffusion。HiP选择在**observation space**（video）diffuse，理由是：
1. Video diffusion可以pretrain on internet video（action-agnostic）
2. Observation trajectory比action trajectory更容易transfer cross-embodiment
3. Inverse dynamics作为最后一步translate image plan到action，避免action space heterogeneity

---

## 9. Limitations与Future Directions

paper承认的limitation：
1. 视觉sequence prediction和action generation的真正foundation model尚未存在——HiP当前用small-scale simulation-trained model作为proxy
2. Joint distribution sampling用approximation（classifier guidance）——未来可探索更精确高效的consistency enforcement

潜在extension（paper提到）：
1. 引入touch和sound modality的foundation model
2. 用large video foundation model（如未来的Sora-class model）guidesmaller video model（参考 [Yang et al. 2023](https://arxiv.org/abs/2306.01872) Probabilistic Adaptation）
3. 用diffusion加速方法减少visual planning latency

我个人的几个联想：

**关于Open-loop execution的limitation**：HiP在每个subgoal内是open-loop执行（生成50帧video → inverse dynamics提取action → 直接执行）。这在long-horizon中容易积累execution error。一个natural extension是每几步replan，但需要online video inpainting而不是从头generate。

**关于classifier guidance的替代**：当前的binary classifier $g_\psi$只判断整体feasibility，更principled的做法是用energy-based model直接估计 $\prod_t p_\psi(a_{i,t}|x_{i,t},x_{i,t+1})$。可能的方法：score matching or flow matching over trajectories。

**关于LLM不依赖observation的假设**：Appendix A中 $p_\theta(w_i|g, x_{i,1}, ...) \approx p_\theta(w_i|g)$ 的Markov假设很强。未来如果LLM能multimodal grounding（如GPT-4V, Gemini），这个假设可以放松，直接让LLM看 $x_{i,1}$ 推subgoal。

**关于action representation的embodiment transfer**：当前action是7-dim joint state difference。如果换embodiment（different robot, human hand），inverse dynamics需要retrain。但video plan保持不变——这是HiP的modularity优势。这与 [RT-2](https://robotics-transformer2.github.io/) 的vision-action model形成对比。

**关于semantic和geometric grounding的层次**：HiP的层次其实对应cognitive science中的"Dual System"——System 1 (visual imagination + motor control) fast and automatic，System 2 (LLM planning) slow and deliberative。这与Josh Tenenbaum和Leslie Kaelbling的research program一脉相承。

**关于compositional generalization的来源**：HiP在unseen combination上performance保持的关键是每个component的compositional structure：
- LLM本身compositional（novel subgoal sequences）
- Video diffusion预训练在Ego4D上看到大量compositional visual concepts
- VC-1 pretraining in ego-centric images提供compositional visual features

**关于inverse dynamics的VC-1 vs R3M vs MVP对比**：paper用VC-1，但[R3M (Nair et al. 2022)](https://arxiv.org/abs/2203.12601)和[MVP (Xiao et al. 2022)](https://arxiv.org/abs/2210.03110)都是ego-centric visual representation candidates。VC-1的选择可能基于 [Majumdar et al. 2023](https://arxiv.org/abs/2303.18240) 的comprehensive benchmark显示VC-1在多个robotic tasks上表现最佳。

**关于HiP与最近的VLM-based robotics的关系**：[RT-2](https://arxiv.org/abs/2307.15818), [PaLM-E](https://palm-e.github.io/), [Octo](https://octo-models.github.io/)走monolithic VLA路线，HiP走compositional路线。两者trade-off：
- Monolithic：end-to-end训练，co-design representation，但data expensive
- Compositional：modular，each component can leverage largest internet data，但consistency enforcement需要trick

未来trend可能是hybrid：large VLM作为high-level reasoner + small diffusion policy作为low-level executor + 精心设计的interface（如HiP的classifier guidance）。

**关于text-to-video model的scale effect**：当前HiP用small-scale video diffusion（48×64 resolution, 50 frames）。如果换成Sora-level model，visual planner的physical reasoning能力会质变。但classifier guidance的cost也会scale。需要efficient adapter方法（如[IP-Adapter](https://ip-adapter.github.io/)）来leverage frozen large video model。

---

## 10. 总结：HiP的Core Insight

HiP给我的最大intuition是：**long-horizon planning的本质难题是representation granularity mismatch**——language太abstract，action太concrete，需要中间的visual layer作为bridge。这个bridge不是预先定义的fixed representation（如skill library），而是通过video diffusion这个generative model动态合成的。

三个foundation model对应三种internet data distribution，各自capture一种prior：
- LLM：semantic/task prior（"泡茶需要先烧水"）
- Video diffusion：geometric/physical prior（"水壶从柜子里拿出来的视觉trajectory"）
- Inverse dynamics：embodiment prior（"我的手臂如何执行这个visual change"）

iterative refinement的physical意义：当三个model独立预训练时，它们implicit的world model不一致。通过classifier guidance做message passing，让每一层"知道"下一层的constraint，最终收敛到三者的intersection——这就是physically executable、semantically meaningful、visually plausible的plan。

这是compositional foundation model paradigm的一个convincing demonstration，showing that properly composing independently-trained experts can rival or surpass monolithic models trained on paired data。随着video foundation model和embodied foundation model的成熟，HiP-style compositional approach可能成为long-horizon decision making的主导paradigm。

---

## References

1. HiP Project Page: https://hierarchical-planning-foundation-model.github.io/
2. UniPi (Du et al. 2023): https://arxiv.org/abs/2302.00112
3. SayCan (Ahn et al. 2022): https://say-can.github.io/
4. Ego4D: https://ego4d-data.org/
5. VC-1: https://arxiv.org/abs/2303.18240
6. PVDM (Yu et al. 2023): https://arxiv.org/abs/2302.07685
7. Latent Diffusion Models: https://arxiv.org/abs/2112.10752
8. Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
9. CLIPort: https://cliport.github.io/
10. Gato: https://www.deepmind.com/publications/a-generalist-agent
11. Flan-T5: https://arxiv.org/abs/2210.11416
12. RT-1: https://arxiv.org/abs/2212.06817
13. RT-2: https://robotics-transformer2.github.io/
14. PaLM-E: https://palm-e.github.io/
15. Socratic Models: https://socraticmodels.github.io/
16. Density Ratio Estimation (Srivastava et al.): https://arxiv.org/abs/2305.00869
17. Compositional Visual Generation with EBMs: https://arxiv.org/abs/2006.06051
18. Pre-trained LMs for Interactive Decision-Making: https://arxiv.org/abs/2210.03710
19. Diffuser: https://arxiv.org/abs/2205.09991
20. R3M: https://arxiv.org/abs/2203.12601
21. MAE: https://arxiv.org/abs/2111.06377
22. PDSketch: https://arxiv.org/abs/2210.02328
23. KitchenShift: https://openreview.net/forum?id=DdglKo8hBq0
24. Socratic Models paper: https://arxiv.org/abs/2204.00598
25. Foundation Models for Decision Making Survey: https://arxiv.org/abs/2303.04129
26. MiniGPT-4: https://arxiv.org/abs/2304.10592
27. Probabilistic Adaptation of Text-to-Video: https://arxiv.org/abs/2306.01872
28. Progressive Distillation for Diffusion: https://arxiv.org/abs/2202.00512
29. ResNet: https://arxiv.org/abs/1512.03385
30. ViT: https://arxiv.org/abs/2010.11929
31. GPT-3.5/InstructGPT: https://arxiv.org/abs/2203.02155
32. PaLM: https://arxiv.org/abs/2204.02311
33. Trajectory Transformer: https://arxiv.org/abs/2106.01339
34. DDPM: https://arxiv.org/abs/2006.11239
35. Noise-Contrastive Estimation: https://jmlr.org/papers/v13/gutmann12a.html
