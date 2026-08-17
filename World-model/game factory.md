---
source_pdf: game factory.pdf
paper_sha256: 0c908064f5a8ee807c6a6fab938a29023d72ef663616140abf1bdcaf85bd607f
processed_at: '2026-08-04T11:58:27-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GameFactory 大白话版

## 一句话说清楚

**GameFactory 解决的问题是：怎么让一个会生成视频的 AI，学会听你的键盘鼠标指令来生成视频，同时还能保留它原本"什么场景都能画"的本事。**

## 为什么这事难

想象你有个画师，啥都会画——Renaissance palace、赛博朋克街道、沙漠日落——画得都贼好。现在你想教他一个新技能：**根据你按的按键来画第一人称视角的画面**。按 W 画面往前走，按鼠标画面转视角。

最直觉的做法就是：给他看一堆 Minecraft 的游戏录像，每个录像都标好了"这一帧按了什么键"。让他看着学。

问题来了——**他学完之后，只会画 Minecraft 了。** 你让他画 Renaissance palace 里按 W 前进，他画出来的是 Minecraft 方块风格的宫殿。风格被"污染"了。

这就是所有现有方法（GameNGen、Oasis、DIAMOND）的问题：**它们把"学风格"和"学控制"这两件事搅在一起了。**

## GameFactory 的核心思路

用一句人话说：**把"学风格"和"学动作"拆开，分别交给两个不同的模块学，最后用的时候只留"动作模块"，把"风格模块"扔掉。**

具体来说：

### 第一步：先让 LoRA 学 Minecraft 的"画风"

LoRA 你可以理解成一个"风格滤镜插件"。给它看 Minecraft 视频，它学会了：哦，Minecraft 的画面是方块状的，颜色是这样的，纹理是那样的。

这时候动作控制模块还没接入，LoRA 只管把 Minecraft 画风学好。

### 第二步：冻住 LoRA，让动作模块专心学"按键→画面变化"的因果关系

关键来了。LoRA 已经把 Minecraft 画风"搞定"了，也就是说 diffusion loss 里关于"画风"的部分已经被 LoRA 扛走了。那动作控制模块的 gradient 还能优化什么？**只能去学"按 W 键 → 画面怎么往前移动"这种纯粹的因果关系。**

它学到的不是"Minecraft 里按 W 长什么样"，而是"在任何场景里，往前移动这个动作会让画面产生什么样的 optical flow 变化"。

### 第三步：推理的时候，把 LoRA 拔掉

LoRA 是个插件，拔掉就拔掉。拔掉之后，pretrained model 的 open-domain 能力恢复了——又能画 Renaissance palace 了。但动作控制模块还留着，它学到的是 style-agnostic 的。

所以最终效果：**你输入 Renaissance palace 的 prompt + 按 W 键，model 画出 palace 风格的、画面往前移动的视频。**

这就是 Fig. 1 里展示的效果。

## 为什么数据也很关键

这里有个很实际的 insight：**人类玩家的按键习惯是有 bias 的。**

你玩 Minecraft 的时候，90% 的时间在按 W 往前走，几乎从来不按 S 往后退。如果用人类游戏录像训练，model 根本没见过"后退"长什么样，你让它后退它就懵了。

GameFactory 的做法很 simple 但很 effective：**不用人类录像，用程序生成随机按键序列。** W/A/S/D/Space/Shift/Ctrl 每个键出现频率都差不多 13-15%，mouse movement 也随机。

这就像教小孩学开车，你不能只让他看别人正常开车，得让他专门练急刹车、倒车、原地打方向这些"罕见操作"。

Table 4 的数据很直观：用 GF-Minecraft 训练的 model，Cam metric 是 0.0839；用 VPT（人类录像）训练的是 0.1324。差距巨大。

## Action Control 的设计细节

### 为什么 mouse 和 keyboard 要用不同的 fusion 方式

Keyboard 是离散信号——你按了 W 就是按了，没按就是没按，跟 text token 很像。所以用 cross-attention，跟处理 text prompt 一样，model 通过"相似度匹配"来决定怎么响应。

Mouse 是连续信号——鼠标移动了 0.5 和移动了 2.3，magnitude 完全不同，这个数值大小直接决定了画面转多快。如果用 cross-attention，softmax 会把 magnitude 信息抹掉（因为 attention 本质是算相似度，不管绝对值）。所以用 concatenation，把 raw 数值直接拼进去，保留 magnitude。

这个 insight 其实挺 generalizable 的：**离散控制信号用 attention，连续控制信号用 concatenation。** 以后做类似的 controllable generation 都可以参考。

### Sliding Window 为什么必要

两个原因：

1. **Granularity 对齐**：VAE 把 4 帧压成 1 个 latent，所以 action 序列长度是 latent 数量的 4 倍。不对齐没法 fuse。

2. **Delayed effect**：你按了 jump 键，角色不是只跳一帧，而是腾空、上升、下落，影响后面好几帧。Window size=3 意味着当前 latent 能"看到"前面 12 帧（3×4）的 actions，这样 jump 的 delayed effect 就被 capture 了。

## Long Video 怎么做

标准 diffusion 要求所有 frame 用同一 noise level——所有 frame 同时 denoise。这没法 autoregressive。

GameFactory 的做法：**前 k+1 帧不加 noise（当作已经生成好的 condition），后面 N-k 帧加 noise 让 model 去 predict。** 而且算 loss 的时候只算后面 N-k 帧的 loss，不算前面 condition 帧的。

为什么不算 condition 帧的 loss？因为那些帧本来就是 clean 的，你让 model 去 "denoise clean frame" 等于让 model 学一堆 irrelevant 的 noise pattern，反而有害。Table 6 证明了这个：只算 predicted frames 的 loss，FVD 从 1592 降到 1154。

推理的时候就是滑窗式生成：用最新的 k+1 帧做 condition，生成新的 N-k 帧，循环往复，理论上无限长。

## 最 Amazing 的发现：Racing Game 的自动迁移

Fig. 13 这个例子特别有意思。model 只在 first-person Minecraft 上训练过，结果你给它一个 racing game 的 prompt，它居然能：

- **鼠标 yaw 控制 → 自动变成方向盘控制**。因为在 Minecraft 里 yaw 就是"左右转头"，在赛车场景里这个概念自然映射成"左右转向"。
- **后退/左右平移 → 自动减弱**。因为赛车场景下这些 action 没意义，model 自动 suppress 了。

这说明 model 学到的不是"Minecraft 的按键规则"，而是一个更抽象的 **"navigation semantics"**。这种 emergent generalization 是最 exciting 的——你不知道它还能迁移到什么场景。

## 这篇 Paper 的真正贡献

抛开技术细节，我觉得 GameFactory 最大的 contribution 是一个 conceptual shift：

**Generative game engine 不需要从零学一个特定游戏的 dynamics。你可以 "borrow" pretrained video model 的 open-domain priors，然后只学一个轻量的、style-agnostic 的 action control module，插上去就能用。**

这个思路跟 LoRA 之于 LLM 的关系很像——你不需要 finetune 整个 model 来学新能力，只需要学一个小的 adapter。

往大了想，这就是 **Generalizable World Model** 的雏形。Paper Appendix C 提到这个 vision：同样的 framework 可以用来做 autonomous driving 的 data generation（用少量 labeled driving data + open-domain video prior）、robotics 的 sim-to-real（用少量 robot action data + open-domain prior 生成无限训练场景）。

核心 pattern 是：**小规模 labeled data 提供的"action knowledge" + 大规模 unlabeled data 提供的"scene knowledge" = 通用 world model。**

这个 pattern 如果能 scale up，可能比单纯的"收集更多 labeled data"更有前途。

## References

- [GameFactory Project Page](https://yujiwen.github.io/gamefactory/)
- [GameNGen - Diffusion Models are Real-Time Game Engines](https://arxiv.org/abs/2408.14837)
- [Oasis - Etched/Decart](https://oasis-model.github.io/)
- [Genie 2 - Google DeepMind](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [DIAMOND - Diffusion for World Modeling](https://diamond-wm.github.io/)
- [VPT - Video PreTraining from OpenAI](https://openai.com/research/vpt)
- [LoRA - Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Diffusion Forcing - Chen et al.](https://boyuan.space/diffusion-forcing/)
- [MineDojo - Fan et al.](https://minedojo.org/)
- [Sora - OpenAI](https://openai.com/sora/)

---

# GameFactory 深度解析

Andrej，这篇paper的核心insight非常elegant，让我一层层剥开来讲。

## 1. Core Problem & Motivation

当前 generative game engine 的根本 bottleneck 是 **scene generalization**。GameNGen 只能生成 DOOM，Oasis 只能生成 Minecraft，DIAMOND 只能生成 Atari——这些 model 都 overfit 到特定 game 的 visual style 和 dynamics。如果你想让 model 生成一个"Renaissance palace 里的 first-person exploration game"，这些方法全挂。

GameFactory 提出的问题非常关键：**能不能用 open-domain pretrained video model 的 rich generative priors，加上小规模 action-annotated game data，实现 scene-generalizable 的 action-controllable video generation？**

这个问题的 nontrivial之处在于：直接 finetune pretrained model 会让 model 同时学到 Minecraft 的 visual style 和 action control，导致 style-control entanglement。生成结果会带 Minecraft 的 blocky 风格，丢失 open-domain generalization。

## 2. GF-Minecraft Dataset：消除 Human Bias

### 2.1 为什么 VPT 数据不行

VPT dataset 是从 human gameplay 录制的，存在严重的 action distribution skew。看 Table 5 的数据：

| Key | VPT | GF-Minecraft |
|-----|-----|--------------|
| W (forward) | 50.11% | 13.56% |
| S (back) | 0.32% | 13.56% |
| Space (jump) | 20.37% | 15.25% |

W 键出现频率是 S 键的 **156倍**。这种 skew 导致 model 从未见过 "backward movement" 的 training signal，自然无法生成 backward 的 video。

### 2.2 Data Collection 策略

GameFactory 的做法是 **decompose 成 atomic actions 并保证 uniform distribution**：

- 用 MineDojo 的 API 执行 predefined random action sequences
- 3 biomes (forest/plains/desert) × 3 weather × 6 times of day = 54 scene configurations
- 70 hours，2000 clips × 2000 frames
- 每个 atomic action 的 duration 随机化，避免 temporal bias
- Text annotation 用 MiniCPM-V

关键 insight：**data distribution 的 uniformity 比 data volume 更重要**。Table 4 显示 GF-Minecraft 训练的 model 在 Cam metric 上 0.0839 vs VPT 的 0.1324，提升巨大。

## 3. Action Control Module：解决 Granularity Mismatch

### 3.1 核心挑战

Temporal compression ratio r=4 意味着：4 个 video frames → 1 个 latent frame。所以 action sequence 长度 rn 和 latent feature 数量 (n+1) 不匹配。直接 fuse 会出问题。

### 3.2 Sliding Window Grouping

这是 paper 里一个很精妙的设计。对第 i 个 latent feature f^i，考虑 action window：

$$[a^{r \times (i-w+1)}, ..., a^{ri}]$$

其中 w=3 是 window size。这个 design 有两个 purpose：

1. **Alignment**：把 rn 个 actions group 成 (n+1) 组，每组 rw 个 actions
2. **Delayed effect modeling**：jump 这个 action 会影响后续多帧的 physics（角色腾空、落地），window 让 current latent 能 "看到" 之前几帧的 actions

对于 boundary indices，用 boundary actions 做 padding。

Grouping 后：
- Mouse: $M_{group} \in \mathbb{R}^{(n+1) \times rw \times d_1}$
- Keyboard: $K_{group} \in \mathbb{R}^{(n+1) \times rw \times c}$（先 embedding + positional encoding 再 group）

### 3.3 Mouse vs Keyboard 的不同 Fusion 策略

这是 Table 2 ablation 的核心发现：

**Mouse（continuous）→ Concatenation**
- Reshape $M_{group}$ 从 $\mathbb{R}^{(n+1) \times rw \times d_1}$ 到 $\mathbb{R}^{(n+1) \times 1 \times rwd_1}$
- Repeat 到 token length：$M_{repeat} \in \mathbb{R}^{(n+1) \times l \times rwd_1}$
- Concat with F：$F_{fused} \in \mathbb{R}^{(n+1) \times l \times (c + rwd_1)}$
- 再过 MLP + temporal self-attention

**Keyboard（discrete）→ Cross-Attention**
- $K_{group}$ 作为 key/value，F 作为 query
- 类似 text prompt 的 cross-attention 机制

**为什么这样设计？** Intuition 在于：

- Mouse movement 是 continuous 数值，magnitude 重要。Cross-attention 的 softmax 会 normalize 掉 magnitude 信息（similarity computation 倾向于 reduce magnitude 的影响）。Concat 保留了 raw 数值。
- Keyboard 是 discrete category，本质是 "是否按下某键" 的 binary signal。Cross-attention 的 similarity-based matching 更适合这种 "category matching" 的 logic，类似 text token 的处理。

Table 2 数据验证：Cross-Attn + Concat 组合在 only-key 上 Flow 7.79（最优），mouse-large 上 Cam 0.1021（最优）。

## 4. Autoregressive Long Video Generation

### 4.1 为什么不能直接用标准 diffusion

Standard video diffusion 要求所有 frames 用同一 noise level。这限制了 autoregressive generation，因为你想让前面的 frames "确定下来" 再生成后面的。

### 4.2 Diffusion Forcing 的变体

GameFactory 借鉴 Diffusion Forcing 的思想，允许 **不同 frames 有不同 noise level**：

**Training（Fig. 5a）**：
- N+1 frame latents（index 0 到 N）
- 随机选 k+1 个 frames 作为 condition（不加 noise）
- 剩余 N-k 个 frames 加 noise
- Loss **只计算** N-k 个 predicted frames

公式上，standard loss 是：
$$\mathcal{L}_a(\phi) = \mathbb{E}[||\epsilon_\phi(Z_t, p, A, t) - \epsilon||_2^2]$$

修改后只对 predicted frames 计算 loss。Table 6 显示这个改动很关键：Flow 85.45 vs 148.73，FVD 1154.45 vs 1592.43。

**Intuition**：如果对 condition frames 也算 loss，model 会学习去 "denoise 已经 clean 的 frames"，这是 irrelevant 的 noise，反而干扰 learning。

**Inference（Fig. 5b）**：
1. 先 full-sequence 生成前 N+1 frames
2. 取最近 k+1 frames 作为 condition
3. 生成新的 N-k frames
4. Merge 进 history latents
5. Repeat → 无限长度

优势：每步生成 multiple frames，比 next-frame prediction 快很多。

## 5. Style-Action Decoupling（核心创新）

这是整篇 paper 的 soul。让我详细讲清楚为什么这个 design 能 work。

### 5.1 问题本质

当你 finetune pretrained T2V model with Minecraft data，diffusion loss 同时优化两件事：
1. **Visual style adaptation**：学习 Minecraft 的 blocky textures、pixelated 风格
2. **Action control learning**：学习 "按 W 键 → 前进" 的 mapping

这两个 learning signal 在 standard fine-tuning 中是 entangled 的，因为它们共享同一组 model parameters。

### 5.2 Decoupling Strategy

核心 insight：**用不同的 parameter subset 分别承担这两个 learning task**。

- **Domain Adapter（LoRA）**：学习 Minecraft style
  - rank=128，插入到 transformer layers
  - LoRA 的 low-rank 结构天然适合学习 style 这种 "low-dimensional" 的 visual pattern
  
- **Action Control Module**：学习 action → dynamics mapping
  - 独立的 module，不修改 original parameters

### 5.3 Multi-Phase Training（Fig. 6）

**Phase #0**：Pretrain on open-domain
- 获得强大的 scene generation prior

**Phase #1**：Train LoRA only（lr=1e-4）
- 目标：让 model 能 generate Minecraft-style videos
- 这时候 **action control module 还没接入**
- LoRA 充分吸收 Minecraft visual style

**Phase #2**：Freeze LoRA + pretrained params，train action control module only（lr=1e-5）
- 关键：此时 Minecraft style 已经被 LoRA "lock 住"
- Action control module 的 gradient 只能往 "学习 action-dynamics mapping" 的方向走
- Style learning 的 loss 已经被 Phase #1 minimize 了，所以 Phase #2 的 loss 下降主要来自 action control 的改进
- 这实现了真正的 **functional decoupling**

**Phase #3**：Inference
- **Remove LoRA**！只保留 action control module
- Pretrained model 的 open-domain prior 恢复
- Action control module 是 style-agnostic 的（因为 Phase #2 学的是 pure action-dynamics）
- 结果：open-domain scene + action control

### 5.4 为什么这个能 generalize

Table 3 的数据很说明问题：

| Strategy | Domain | Cam↓ | Flow↓ | FID↓ | FVD↓ |
|----------|--------|------|-------|------|------|
| Multi-Phase | In-domain | 0.0839 | 43.48 | - | - |
| Multi-Phase | Open-domain | 0.0997 | 54.13 | 121.18 | 1256.94 |
| One-Phase | Open-domain | 0.1134 | 76.02 | 167.79 | 1323.58 |

Multi-phase 在 open-domain 上 Cam 0.0997，非常接近 in-domain 的 0.0839。而 one-phase 的 0.1134 明显更差。FID 差距更大：121.18 vs 167.79。

**Dom metric**（衡量 finetuned model 与 original model 在 CLIP space 的相似度）也验证了：multi-phase 保持了 original model 的 domain，没有 style leakage。

## 6. Architecture Details

### 6.1 Backbone

- 1B parameter transformer-based T2V diffusion model
- Distilled from 更大的 pretrained model
- Resolution: 360×640
- VAE temporal compression: r=4
- Latent frames: (1+n)，实际 video frames: (1+rn)

### 6.2 Training Config

- 8×A100 GPU
- Batch size: 64
- Phase #1 LoRA: rank=128, lr=1e-4
- Phase #2 Action module: lr=1e-5
- 每个phase 2-4 days
- DDIM sampling, 50 steps
- Classifier-free guidance only on text prompt

### 6.3 Action Space

Table 7 的完整 action space：

| Behavior | Signal | Interface |
|----------|--------|-----------|
| forward | W | Interface1 |
| back | S | Interface1 |
| left | A | Interface2 |
| right | D | Interface2 |
| jump | Space | Interface3 |
| sneak | Shift | Interface3 |
| sprint | Ctrl | Interface3 |
| vertical look | mouse yaw | Interface4 |
| horizontal look | mouse pitch | Interface5 |

Mutually exclusive actions（forward/back）分到同一 interface。

## 7. Fascinating Generalization Phenomena

### 7.1 Racing Game Transfer（Fig. 13）

这是最 amazing 的 finding。用 first-person Minecraft 数据训练的 model，给一个 racing game prompt：

> "On a racing track, from a first person perspective, one can see holding a steering wheel."

结果：
- **Mouse yaw control → Steering control**：seamless transfer！
- **Backward/left/right movement → 自动 diminish**：因为这些 action 在 racing context 下 meaningless

这说明 model 学到的不是 "Minecraft 的 action mapping"，而是更抽象的 **"navigation action → camera/world dynamics"** 的通用 mapping。

### 7.2 Collision Detection（Fig. 12）

Training data 里自然包含 collision 案例（random scene generation）。Model 学会了：即使输入 "forward" action，如果前面有 wall，agent 应该 stationary。这是一种 emergent physical understanding。

### 7.3 Implicit Physics Learning

这呼应了 Sora 等 video model 展现的 "world simulation" 能力。Diffusion model 在生成过程中隐式学习了：
- Object permanence
- Collision physics
- Camera-action dynamics
- Temporal consistency

## 8. Evaluation Metrics 解析

### 8.1 Action Following Metrics

- **Flow**：计算 generated video 的 optical flow，与 reference video 的 optical flow 做 MSE。反映 action-following 的动态准确性。
- **Cam**：用 GLOMAP 提取 camera pose，计算 predicted vs reference 的 Euclidean distance。直接衡量 mouse control 的效果。

### 8.2 Generation Quality Metrics

- **CLIP**：text-video semantic alignment
- **FID**：frame-level distribution distance
- **FVD**：video-level distribution distance（考虑 temporal）

### 8.3 Domain Preservation

- **Dom**：finetuned model vs original model 在 CLIP space 的相似度。衡量 style leakage。

## 9. Comparison with Related Works

Table 1 很 informative。关键对比：

| Method | Scene Gen | Action Space | Paper |
|--------|-----------|--------------|-------|
| GameNGen | ✗ | Key | ✓ |
| Oasis | ✗ | Key+Mouse | ✗ |
| Matrix | ✓ | 4 Keys | ✓ |
| Genie 2 | ✓ | Key+Mouse | ✗ |
| **GameFactory** | **✓** | **7 Keys+Mouse** | **✓** |

GameFactory 是唯一同时满足：technical paper + complex action space + scene generalizable 的 method。

## 10. Limitations & Future Directions

Paper 自己提到的：
- Level design and gameplay diversity
- Player feedback systems
- In-game object manipulation
- Long-context memory
- Real-time generation（目前 50 step DDIM 太慢）

我补充几个观察：
- **第一人称 bias**：racing game 的成功 transfer 依赖 first-person perspective 的 similarity。第三人称 game 能否 generalize 存疑。
- **Action space 局限**：只有 navigation actions，没有 combat、interaction、inventory management
- **Determinism**：diffusion 的 stochasticity 可能导致相同 action sequence 产生不同结果，这对 game logic 是问题
- **Real-time constraint**：50 step DDIM 无法 real-time，需要 distillation 或 consistency model

## 11. Broader Implications：Generalizable World Model

Paper Appendix C 提出一个很有 vision 的观点：GameFactory 本质是一个 **Generalizable World Model**。

两个应用方向：

1. **Data Producer**：用小规模 labeled data 的 knowledge，generate 无限的 open-domain action-annotated data。对 autonomous driving、robotics 等 data-hungry 领域意义重大。

2. **Simulator**：直接训练 agent。能 generate extreme scenarios（碰撞、极端天气），解决 sim-to-real transfer 的 long-tail problem。

这让我想到 Sora、Genie 2、Vista 等 work 都在往这个方向走。Video model 作为 world simulator 的 paradigm 正在 converge。

## 12. Personal Takeaways

### 12.1 Decoupling is Key

Style-action decoupling 的思想非常 generalizable。任何 "finetune 大 model 学习新能力但保持原能力" 的场景都可以用：
- Robot learning：decouple embodiment style from control policy
- Style transfer：decouple content from style
- Multi-task learning：decouple task-specific from shared representations

### 12.2 Data Distribution > Data Volume

GF-Minecraft 只有 70 hours，但 uniform action distribution 让它 beat 了更大规模的 VPT。这对 data collection 有重要指导意义。

### 12.3 Emergent Generalization

Racing game 的 transfer 是 emergent 的，不是 designed 的。这说明 model 学到了比 "Minecraft action mapping" 更 abstract 的东西。这种 emergent capability 是 scaling 和 good inductive bias 的结果。

### 12.4 Diffusion as Game Engine

这个 paradigm 的核心 assumption 是：game = interactive video sequence。如果这个 assumption 成立，video diffusion model 就是天然的 game engine。但 game 还有 logic、state、determinism 等 video 不 capture 的维度。这是未来要解决的。

## References

- [GameFactory Project Page](https://yujiwen.github.io/gamefactory/)
- [Diffusion Forcing (Chen et al.)](https://boyuan.space/diffusion-forcing/)
- [GameNGen (Valevski et al., 2024)](https://arxiv.org/abs/2408.14837)
- [Oasis (Etched/Decart)](https://oasis-model.github.io/)
- [Genie 2 (Google DeepMind)](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- [DIAMOND (Alonso et al., NeurIPS 2024)](https://diamond-wm.github.io/)
- [Matrix (Feng et al., 2024)](https://arxiv.org/abs/2412.03568)
- [VPT (Baker et al., NeurIPS 2022)](https://openai.com/research/vpt)
- [MineDojo (Fan et al., NeurIPS 2022)](https://minedojo.org/)
- [LoRA (Hu et al., ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [Sora (OpenAI)](https://openai.com/sora/)
- [GLOMAP (Pan et al., ECCV 2024)](https://arxiv.org/abs/2407.20214)
- [MiniCPM-V (Yao et al., 2024)](https://arxiv.org/abs/2408.01800)
- [DDIM (Song et al., 2020)](https://arxiv.org/abs/2010.02502)
- [DiT (Peebles & Xie, ICCV 2023)](https://arxiv.org/abs/2212.09748)
