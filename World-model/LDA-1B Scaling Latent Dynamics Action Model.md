---
source_pdf: LDA-1B Scaling Latent Dynamics Action Model.pdf
paper_sha256: 7e3619baa7129f509e21c842b4bc5aca08808bbb30b85ca9b240a779ebf2264b
processed_at: '2026-08-05T12:27:56-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LDA-1B 人话版：一个 robot 怎么学会"想象"未来

Hey Andrej，我来换个角度，用大白话把这篇 paper 重新讲一遍。上次的版本太像 paper review 了，这次我把它当成跟同事 coffee chat 的语气来聊。

---

## 一、从一个尴尬的事实说起

你有没有想过一个奇怪的事：我们现在的 robot foundation model，像 π0、GR00T、RDT 这些，训练数据里**每一条 trajectory 都同时带着 "我看到了什么" 和 "我做了什么" 两个信号**，但 BC 范式只用了后者来监督 action prediction，前者只是个 conditioning input。

这就好比你有一本日记，每一页都写着 "今天天气晴，我去了公园"，你只用它学 "晴天→去公园" 这个 mapping，却从来没想过 "公园本身长什么样、公园里有滑梯、滑梯怎么玩" 这些信息也都在日记里。

更尴尬的是，人类有一堆 ego-centric video（Ego4D、Epic-Kitchens 这些），里面只有 "我看到了什么"，没有 "我做了什么" 的精确 label。BC 范式直接把这些数据扔了——因为没 action label 没法 supervised action prediction。但这些 video 里**藏着巨量的物理常识**：杯子掉地上会碎、抽屉拉开需要先往外再往上、面包可以用 spatula 翻面。这些 dynamics knowledge 全浪费了。

LDA 的核心 motivation 就一句话：**别浪费 data，让每条 data 都贡献它能贡献的东西**。

参考：Ego4D https://arxiv.org/abs/2110.07058 ; Epic-Kitchens https://arxiv.org/abs/2008.00460

---

## 二、用做菜类比解释 4 个训练目标

LDA 同时训练 4 个 objective，听起来很复杂，但其实就像教一个学徒做菜：

| Objective | 人话 | 类比 |
|-----------|------|------|
| **Policy** | 看到食材，知道下一步该切还是炒 | 学徒看菜谱 "看到 X 就做 Y" |
| **Forward Dynamics** | 知道如果我把鸡蛋打进去，锅里会发生什么 | 学徒脑补 "打蛋后锅会冒泡变黄" |
| **Inverse Dynamics** | 看到前后两个画面，猜中间发生了什么动作 | 学徒看 before/after 图反推 "这之间切了葱花" |
| **Visual Forecasting** | 即使没动作，世界本身也在变（水开了、冰化了） | 学徒知道 "放着不管，水也会开" |

BC 只教第一个，学徒变成只会照菜谱机械执行的工具人。LDA 四个一起教，学徒开始有 "物理直觉"——知道自己的动作会让世界变成什么样，知道世界本身也会自顾自地变。

**为什么这很重要**？因为 long-horizon 任务（比如 "打扫垃圾"：拿扫把→扫纸团进 dustpan→dustpan 倒进 trash bin）每一步都可能出小偏差，BC policy 没有内部世界模型，不知道自己下一步动作的后果，偏差累积就崩。LDA 知道 "我这一扫，纸团会飞到哪里"，所以能 correction。

paper 里 Clean Rubbish 任务的数据：**LDA 35%，GR00T 和 π0.5 都是 0%**。这不是随机噪声，是 BC 范式结构性缺陷的直接体现。

---

## 三、为什么选 DINO latent 而不是 pixel/VAE

这是整篇 paper 最 subtle 但最关键的决定。我用一个 thought experiment 解释。

假设你在学 "forward dynamics"：给定 $o_t$（当前画面）和 $a_t$（动作），预测 $o_{t+1}$（下一帧画面）。

**场景 A：pixel space 预测**
你训练 loss 是 $\|o_{t+1}^{\text{pred}} - o_{t+1}^{\text{gt}}\|^2$ 在 RGB pixel 上。问题是：如果光照变了一点点、相机晃了一点点、背景里有人走过，pixel 全变了，但这些变化跟你的 action 一点关系都没有。你的 model 90% 的 capacity 在学 "怎么画 realistic 的图"，10% 在学 "action 让物体怎么动"。

**场景 B：VAE latent space**
VAE 是为 pixel reconstruction 训练的，所以它的 latent 仍然 encoding 大量 appearance 信息。问题同上，只是从 pixel 噪声变成 latent 噪声。

**场景 C：DINO latent space**
DINO 是 self-supervised 训练的，学的是 "这张图里有什么 object、object 在什么位置、object 之间什么关系"。光照变、背景变，DINO feature 几乎不变。物体位置变了，DINO feature 立刻变。

所以 DINO latent 就是**专门为你想要的 supervision signal 量身定制的**——它 encode 的是 "task-relevant 的物体状态"，不是 "pixel appearance"。

paper Table II 的 ablation 直接给了铁证：UWM 用 VAE latent，1B 参数，20% success rate。LDA 换成 DINO latent，其他全一样，48.9%。**换一个 representation，涨 28.9 个点**。这比任何 architecture innovation 都 dramatic。

这跟 DINO-WM (Zhou et al. 2024) 的 insight 完全一致：他们早就发现 world model 在 DINO feature 上 planning 比 pixel space 强太多。LDA 把这个 idea 从 academic demo scale 到 1B foundation model。

参考：DINO-WM https://arxiv.org/abs/2411.04983 ; DINOv3 https://arxiv.org/abs/2508.10104

---

## 四、Universal Data Ingestion：别挑食

LDA 的数据策略可以总结成一个 metaphor：

**BC 范式**：米其林餐厅只用最顶级的食材，稍微差一点的食材直接扔垃圾桶。
**LDA 范式**：家庭厨房，好食材做主菜，差食材熬汤底，没 action label 的 video 当调味料。

具体分配：

| 数据类型 | 例子 | 在 LDA 里干什么 |
|----------|------|-----------------|
| 高质量 robot demo | Open X-Embodiment, Agibot World | 4 个 objective 全监督 |
| 低质量 trajectory | RoboCOIN 这种 noisy 数据 | 只监督 forward dynamics + visual forecasting（action 不准，但 state transition 合法）|
| 无 action 的 human video | Egocentric-10k | 只监督 visual forecasting（学世界自主演化）|

paper Table IV 有个实验特别 striking：

Pen-into-Box 任务，63 条 high-quality + 37 条 low-quality：
- π0.5：60% → 40%（**掉 20 个点**，被低质量数据污染）
- LDA：70% → 80%（**涨 10 个点**，把低质量数据变废为宝）

这个结果对实际部署意义重大。teleoperation 采集数据天然 noisy——操作员会 pause、retry、手抖。BC 方法需要花人力 filter expert subset。LDA 直接用 raw data，省掉 filter 成本。

**为什么 low-quality data 对 LDA 是正向的**？因为它的 state transition 在物理上还是合法的。"我手抖了一下然后 pause" 这个轨迹，action 是 suboptimal 的，但 "手抖然后 pause" 这个 state→state' 的 transition 是真实发生过的物理过程。LDA 的 forward dynamics 从这个 transition 里学到了 "手抖会导致物体怎样"，这个 knowledge 是有用 prior。

类比：你学开车，师傅开车的过程是 expert demo，但你在副驾看师傅偶尔刹车避让行人，这个 "刹车避让" 的 visual 后果（行人停下、车减速）也是有用的 dynamics knowledge，即使你不知道师傅当时踩了多少刹车踏板。

---

## 五、MM-DiT 架构：一条流水线处理两种东西

架构上最直观的理解：LDA 把 action 和 visual feature 两条 token stream 塞进**同一个 transformer**，让它们在 self-attention 层互相 attend。

为什么这么做？因为 action 和 visual state 是因果耦合的——"推杯子" 这个 action 会让 "杯子的 DINO feature" 变化，而 "杯子当前在桌子边缘" 这个 visual state 决定了 "推一下它会掉下去" 这个 action consequence。如果用两个独立 model 分别处理，cross-modal interaction 只能在 output 层 happen，太晚了。共享 self-attention 让 interaction 在每一层都 happen。

具体到 MM-DiT block 的设计（参考 Stable Diffusion 3 / FLUX）：
- **QKV projection 是 modality-specific 的**：action token 用自己的 W_q,W_k,W_v，visual token 用自己的。这保留了各自的 inductive bias。
- **Self-attention 是 shared 的**：action token 可以 attend 到 visual token，反之亦然。这是 cross-modal interaction 的关键。
- **FFN 是 modality-specific 的**：processing 各自 refine。
- **AdaLN 注入 conditioning**：language、diffusion timestep、task embedding 通过 adaptive layer norm 注入。

这个设计有点像你 nanoGPT 里那种 "keep it simple, just transformer" 的哲学——没有花哨的 cross-attention module、没有 separate decoder，就是 shared self-attention 加 modality-specific 的 QKV/FFN。但它在 1B scale 上稳定 work。

参考：DiT https://arxiv.org/abs/2212.09748 ; Stable Diffusion 3 https://arxiv.org/abs/2303.09533

---

## 六、Register Token：一个 model 玩多种 task 的 trick

这个设计很 elegant，我多说几句。

LDA 要用一个 model 支持 4 种 inference mode：policy / forward dyn / inverse dyn / visual forecasting。每种 mode 的 input/output 不一样：
- Policy：input 是 obs，output 是 action
- Forward dyn：input 是 obs + action，output 是 future obs
- Visual forecasting：input 是 obs，output 是 future obs（无 action）

最朴素的方案是给每个 task 一个单独的 model head。但这样参数浪费，而且不同 head 之间不 share knowledge。

LDA 的 trick：**用 learnable register token 占位**。

训练 policy 时，input 是 [noised action tokens] + [visual register token]。register token 是个 placeholder，告诉 model "future obs 我没有，你别去 predict visual"。

训练 visual forecasting 时，input 是 [action register token] + [noised visual tokens]。同理。

这样 model 拓扑完全不变，只是 input 里哪个 modality 是 real noised token、哪个是 register token 在变。一个 1B model 灵活支持 4 种 mode。

这让我想到 BERT 的 [MASK] token——一个占位符 trick 让一个 model 支持 masked LM、NSP 等多种 task。LDA 的 register token 是这个 idea 在 diffusion + multi-modal 上的延伸。

---

## 七、异步采样：3Hz visual vs 10Hz action

这是个被低估但很重要的设计细节。

物理世界里，action 是连续控制信号，变化快。visual observation 是被动观测，相邻帧高度相关（你手没动的时候，相机画面几乎不变）。

如果 visual 和 action 都用 10Hz 采样，visual 这边 90% 的相邻帧几乎 redundant，浪费 compute。如果都用 3Hz，action 这边丢失 fine-grained control 细节。

LDA 的折中：**visual 3Hz，action 10Hz**。两条 stream 在 transformer 里通过 positional encoding 对齐时间。这相当于 action 给 visual "填帧"——model 知道 "在这两个 visual frame 之间，我执行了这些 action"。

类比：你看电影是 24fps，但游戏引擎的 physics 是 60fps。你不需要 60fps 的视觉反馈，但你需要 60fps 的 physics 计算，否则碰撞检测不准。LDA 同理。

这个设计还有一个隐性 benefit：它让 **action chunk 和 visual chunk 的长度不同**，action chunk 是 10 个 timestep（1 秒），visual chunk 是 3 个 timestep（1 秒）。这正好对应 UWM paper 里强调的 "predict at natural frequency of each modality"。

---

## 八、EI-30k 数据集：30k 小时是怎么凑出来的

这个数据集的组成其实揭示了 LDA 的野心：

| 类别 | Hours | 占比 | 角色 |
|------|-------|------|------|
| Real robot | 7.6k | 25% | 4 个 objective 全监督 |
| Sim robot | 9.6k | 32% | 4 个 objective 全监督（sim 数据 action 完美准确）|
| Ego human w/ action | 6.3k | 21% | 4 个 objective 全监督（人手数据）|
| Ego human actionless | 10.3k | 34% | 只 visual forecasting |

注意 **actionless video 占了 1/3**。这是 BC 范式完全没法用的数据，但 LDA 把它变成了 visual forecasting 的 supervision。Fig. 10 的 scaling curve 显示，**即使所有 action-labeled data 都用完了，加这 10k actionless video 还能继续降 prediction error**。

数据集的工程细节也很务实：
1. 全部统一到 LeRobot 2.1 format——HuggingFace 的开源 robot data format，避免反复造轮子
2. Action 全部对齐到 hand-centric coordinate frame。意思是 robot gripper、22-DoF Sharpa hand、10-DoF BrainCo hand、人手 MANO，全部用 "delta wrist pose + finger config" 这套统一表示
3. 每条 trajectory 标 quality label，低质量数据保留不删

数据对齐到 hand-centric frame 是个关键决定。这意味着同一段 "delta wrist pose = [0.1, 0, 0, 0, 0, 0], finger config = [0.5]" 在不同 embodiment 上语义是一致的——都是 "手腕往前移 10cm，手指半开"。这让 model 能跨 embodiment transfer knowledge。

参考：LeRobot https://github.com/huggingface/lerobot ; MANO hand model https://arxiv.org/abs/1904.01760

---

## 九、几个实验结果的 intuition

### 9.1 RoboCasa 的 "Close" 类任务优势
LDA 在 "PnP Can To Drawer Close" 上 71% vs GR00T 13%。这种任务需要 "把物体放进抽屉后关抽屉"。关抽屉这个动作有个陷阱：如果物体没放到位，关抽屉会夹住物体，抽屉关不上。

GR00T 没有 forward dynamics，它学到的是 "放完物体→关抽屉" 这个 action sequence，但不知道 "我这个关抽屉动作会让物体怎样"。如果物体稍微凸出来一点，GR00T 还是机械地关，结果夹住失败。

LDA 有 forward dynamics，它 anticipate "如果我现在关抽屉，物体会被夹住"，所以会先调整物体位置再关。这就是 dynamics awareness 的价值。

### 9.2 Dexterous 任务的优势
Flip Bread 这个任务，LDA 90% vs π0.5 10%。这个任务需要：
1. Spatula 插到面包底下（finger 协调）
2. 持续接触面包往上翻（wrist + finger 联动）
3. 面包翻面后稳定接住

这是个 22-DoF + continuous contact 的任务。BC policy 学到的是 "spatula 轨迹 + finger 角度" 的 mapping，但这个 mapping 对接触位置极其 sensitive——spatula 插深一点、浅一点，后续 action 完全不一样。

LDA 的 forward dynamics 让它知道 "我这样插 spatula，面包会翘起多少"，然后根据这个 anticipation 调整后续 action。这就是 contact-rich 任务为什么需要 dynamics model 的根本原因。

### 9.3 Long-horizon 的 0% vs 35%
Clean Rubbish：拿 dustpan→扫纸团→dustpan 倒进 trash bin。每一步都依赖上一步的成功。BC policy 在 step 1 失败后，step 2 的 obs distribution 跟训练时见过的不一样（covariate shift），直接崩。

LDA 即使 step 1 出了小偏差，forward dynamics 让它知道 "现在这个 state，我下一步 action 会导致什么"，所以能在新 state 下继续规划。这是 model-based 方法 vs model-free 方法的经典优势。

---

## 十、Gradient Analysis 的直觉解释

Fig. 12 的 gradient cosine similarity 其实讲了一个很有意思的故事。

训练 10k iteration 时：
- Policy 和 inverse dynamics 的 gradient 高度 aligned（都预测 action）
- Forward dynamics 和 visual forecasting 的 gradient 高度 aligned（都预测 DINO feature）
- Policy 和 forward dynamics 的 gradient **weak negative**（互相竞争）

这很 intuitive——早期 model 还没学会 "action 和 state transition 是怎么耦合的"，所以 "预测 action" 和 "预测 state" 这两个目标在抢 capacity。

训练 400k iteration 时：
- 所有 pair 的 gradient 都变 **positive aligned**

这说明 model 学到了 "action 预测和 state 预测是同一件事的两面"——知道 action 能帮你预测 state，知道 state transition 能帮你反推 action。这就是 model-based control 的精髓，LDA 通过 multi-task co-training 自动 emergent 出了这个 structure。

这个结果对 multi-task learning 是个乐观信号：**只要训练够久，multi-objective 的 gradient conflict 会自动消解**。这意味着 scale up task 数量不会无限加 conflict。

---

## 十一、跟你以前 work 的 connection

既然你是 Andrej，我忍不住联想几个点：

### 11.1 跟 "Software 2.0" 的延伸
你的 Software 2.0 essay 说 "用 data 替代 code 定义 behavior"。LDA 进一步说："data 的不同部分定义 behavior 的不同 aspect"。High-quality data 定义 action policy，low-quality data 定义 dynamics understanding，actionless data 定义 world priors。这是 Software 2.0 的精细化版本——**data 的 quality distribution 本身就是 architecture 的一部分**。

参考：Software 2.0 https://karpathy.medium.com/software-2-0-a6eb4165e0b5

### 11.2 跟 Tesla world model 路线
Tesla 在 2024 宣布 world model approach，思路跟 LDA 高度相似：用大量无 action 监督的 driving video 学 world dynamics，再用少量 action-labeled 数据 fine-tune policy。LDA 就是这个思路在 manipulation 上的 instance。Tesla 用 "预测未来 frame" 学 driving dynamics，LDA 用 "预测未来 DINO feature" 学 manipulation dynamics。

### 11.3 跟 "micrograd" / "nanoGPT" 的 simplicity 哲学
LDA 的架构其实挺 simple 的——就是 MM-DiT 加 register token。没有复杂的 cross-attention module、没有 separate policy head 和 dynamics head、没有 RL 的 reward shaping。所有 complexity 都在 data 端（role assignment）和 representation 端（DINO latent）。这跟你一直倡导的 "keep architecture simple, let data do the work" 哲学一致。

### 11.4 跟你最近谈的 "robotics 需要多少 data" 的话题
你之前在几个 podcast 里聊过 robot data 的稀缺性。LDA 的回答是：**别只盯着 robot data，把 human video 也算进来**。30k hours 里只有 16k 是 robot data，剩下 14k 是 human video。LDA 证明 human video 对 robot policy 有正向 transfer——通过 dynamics 和 visual forecasting 这两个间接 channel。

---

## 十二、我的几个 skepticism

虽然 paper 很 strong，但我有几个疑问：

1. **DINO latent 的精度上限**：DINO feature 是 patch-level 的（14×14 grid），每个 patch 是 384 维。如果任务需要 sub-pixel 精度（比如插 USB 线），DINO latent 可能不够。Paper 里没测这种 fine-grained 任务。

2. **Forward dynamics 的 long-range 准确性**：paper 只 visualize 了未来几帧的 DINO feature 预测。如果 predict 10 秒后的 visual state，error 会累积多少？这决定了 LDA 能做多 long-horizon 的 planning。

3. **Action space 的统一性**：hand-centric frame 对 manipulation 好，但对 whole-body locomotion 不够。如果 robot 需要边走边 manipulate，action 表示怎么扩展？paper 没碰这个。

4. **Reward signal 缺失**：LDA 全是 imitation + dynamics，没有 reward。如果 expert demo 本身是 suboptimal 的（人也会用笨办法做事），LDA 会学笨办法。加一个 RL fine-tuning 阶段可能突破这个 ceiling，但 paper 没做。

5. **Sim-to-real 的 gap**：9.6k hours 是 sim data。虽然 DINO latent 对 appearance variation robust，但 sim 的 physics 可能跟 real 有 gap。Paper 的 real-world 实验 fine-tune 了 100 trajectories/task，这个 sim-to-real 的功劳是 fine-tune 还是 pretrain sim dynamics？

---

## 十三、一句话总结

如果让我用一句话跟别人介绍 LDA：

**"Robot learning 之前只用 high-quality action label 这一种 supervision，LDA 把 dynamics 作为一个新的 supervision channel，让 low-quality 数据和无 action video 也能贡献训练；关键是把预测目标放在 DINO latent space，避免 pixel-level appearance entanglement，让 dynamics learning 真正 scale 起来。"**

或者更短：

**"让 robot 学会想象未来，而不仅仅是模仿过去。"**

---

## 参考 links

- LDA 项目主页：https://pku-epic.github.io/LDA
- UWM (LDA 的前作)：https://arxiv.org/abs/2504.02792
- π0.5：https://arxiv.org/abs/2504.16054
- GR00T N1：https://arxiv.org/abs/2503.14734
- DINOv3：https://arxiv.org/abs/2508.10104
- DINO-WM (DINO latent world model 的早期工作)：https://arxiv.org/abs/2411.04983
- R3M (robot representation learning)：https://arxiv.org/abs/2203.12601
- VIP (visual implicit pretraining)：https://arxiv.org/abs/2210.00030
- Open X-Embodiment：https://arxiv.org/abs/2310.08864
- Ego4D：https://arxiv.org/abs/2110.07058
- LeRobot (数据 format)：https://github.com/huggingface/lerobot
- DiT (Diffusion Transformer)：https://arxiv.org/abs/2212.09748
- Stable Diffusion 3 (MM-DiT 架构)：https://arxiv.org/abs/2303.09533
- Software 2.0 (Karpathy essay)：https://karpathy.medium.com/software-2-0-a6eb4165e0b5
- RoboCasa benchmark：https://arxiv.org/abs/2406.02523

---

希望这个 version 更对你胃口。核心 intuition 就三件事：(1) dynamics 是被浪费的 supervision channel，(2) DINO latent 让 dynamics learning 不被 appearance 干扰，(3) heterogeneous data 按 quality 分角色而非一刀切。剩下都是 implementation detail。

---

# LDA-1B: Scaling Latent Dynamics Action Model 深度讲解

Hey Andrej，这篇 paper 我觉得抓住了一个 robotics foundation model scaling 的关键瓶颈——**dynamics knowledge 的浪费**。让我把整个故事线、数学公式、架构细节和数据组成都拆开讲，帮你 build intuition。

---

## 1. 高层直觉：为什么 BC-centric 范式撞墙了

当前 robot foundation model 的主流路径是 behavior cloning (BC)，像 π0、RDT、InternVLA、GR00T 这些工作本质上是在做 large-scale imitation of expert actions。这路径有个根本问题：**它把 embodied data 当成 "state→action" 监督信号用，而忽略了 state→state' (dynamics) 这条 supervisory channel**。

举一个具体场景：一段 egocentric video 里，人手把杯子从 A 桌面移到 B 抽屉里。BC 视角下，如果没有 action label，这帧数据就废了；但如果从 dynamics 视角看，这帧数据告诉你 "杯子 + 抽屉 + 这种手腕轨迹 → 杯子最终状态"，这是个非常强的物理 prior。

LDA 的核心 insight 就是：**heterogeneous data 不是按 quality 分等级丢弃，而是按 supervision channel 分配角色**。

参考：π0 paper https://arxiv.org/abs/2410.24164 ; RDT-1B https://arxiv.org/abs/2410.07864 ; GR00T N1 https://arxiv.org/abs/2503.14734

---

## 2. UWM 框架的 4 个联合目标

LDA 是 UWM (Unified World Model, Zhu et al. 2025) 的升级版。UWM 的核心是把 4 个 conditional distribution 塞进一个 diffusion model：

| 目标 | Conditional Distribution | 物理含义 |
|------|--------------------------|----------|
| **Policy** | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t)$ | 给当前观测，预测未来动作 |
| **Forward Dynamics** | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t, \mathbf{a}_{t+1:t+k})$ | 给观测+动作，预测未来状态 |
| **Inverse Dynamics** | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_{t:t+k})$ | 给观测序列，反推动作 |
| **Visual Forecasting** | $p(\mathbf{o}_{t+1:t+k} \mid \mathbf{o}_t)$ | 给当前观测，预测未来视觉 |

这 4 个目标为什么必须联合训练？因为它们在因果上是耦合的：
- Policy 学的是 "我想做什么"
- Forward dynamics 学的是 "如果我做了 a，世界会怎样"
- Inverse dynamics 学的是 "要让世界从 o_t 变到 o_{t+k}，需要什么 a"
- Visual forecasting 学的是 "世界本身如何自主演化"

一个 robot 如果只学 policy，它在 long-horizon 任务里就会因 compounding error 失败——因为它没有 "我的动作会对未来产生什么后果" 的内部 model。LDA 的实验数据（Clean Rubbish 任务：LDA 35% vs GR00T/π0.5 0%）就是这个直觉的硬证据。

参考：UWM https://arxiv.org/abs/2504.02792

---

## 3. 公式 (1) 的逐项拆解

论文给出的 flow-matching 目标如下：

$$
l_{\text{action}}^{\theta} = \mathbb{E}_{(o_{t:t+k}, a_{t+1:t+k}, \ell) \sim \mathcal{D}} \left\| v_a^{\theta} - (\epsilon_a - \mathbf{a}_{t+1:t+k}) \right\|_2^2
$$

逐个变量解释：

- **$l_{\text{action}}^{\theta}$**：上标 $\theta$ 表示这是参数为 $\theta$ 的模型的 action flow-matching loss。下标 "action" 区分于 obs loss。
- **$\mathbb{E}_{(\cdot) \sim \mathcal{D}}$**：对数据集 $\mathcal{D}$ 里的样本 $(o_{t:t+k}, a_{t+1:t+k}, \ell)$ 取期望。$\ell$ 是 language instruction。
- **$o_{t:t+k}$**：从时刻 $t$ 到 $t+k$ 的观测序列。下标范围 $[t, t+k]$。
- **$a_{t+1:t+k}$**：从 $t+1$ 到 $t+k$ 的 action chunk。注意这里 action 比 obs 错位一步——因为 action 是 "在 obs 之后采取" 的。
- **$v_a^{\theta}$**：模型预测的 action velocity field。Flow matching 的核心是学习一个向量场，把 noise distribution 拉到 data distribution。这里 $v_a^{\theta} \in \mathbb{R}^{|a|}$，维度等于 action chunk 维度。
- **$\epsilon_a$**：从标准高斯 $\mathcal{N}(\mathbf{0}, I)$ 采样的 noise。下标 $a$ 表示这是 action 路径的 noise。
- **$\tau_a \sim \mathcal{U}(0, T_\tau)$**：action 路径的 diffusion timestep，从 $[0, T_\tau]$ 均匀采样。$T_\tau$ 是最大 timestep。
- **$(\epsilon_a - \mathbf{a}_{t+1:t+k})$**：这是 flow matching 的 target velocity field——从 noise $\epsilon_a$ 到 data $a$ 的直线方向（rectified flow 风格）。

Obs loss 形式完全对称：

$$
l_{\text{obs}}^{\theta} = \mathbb{E} \left\| v_o^{\theta} - (\epsilon_o - o_{t+1:t+k}) \right\|_2^2
$$

最终 total loss：

$$
l^{\theta} = l_{\text{action}}^{\theta} + l_{\text{obs}}^{\theta}
$$

**关键细节**：$\tau_a$ 和 $\tau_o$ 是独立采样的。也就是说 action 和 obs 在不同的 noise level 上被 denoise。这是个非常重要的设计——它允许 action stream 和 visual stream 以不同速度去噪，对应物理世界中 action 变化快（10Hz）、视觉变化慢（3Hz）的事实。

参考：DDPM https://arxiv.org/abs/2006.11239 ; Flow Matching https://arxiv.org/abs/2210.02747

---

## 4. UWM vs LDA 的本质差异：Representation Space

这是 paper 里最 subtle 也最重要的设计选择。看 Table II 的 ablation：

| Model | Vis. Rep. | MM-DiT | Success Rate |
|-------|-----------|--------|--------------|
| UWM-0.1B | VAE | ✗ | 14.2 |
| UWM-1B | VAE | ✗ | 19.3 |
| UWM (MM-DiT) | VAE | ✓ | 20.0 |
| LDA (DiT) | DINO | ✗ | 48.9 |
| LDA-0.5B | DINO | ✓ | 50.7 |
| LDA-1B | DINO | ✓ | **55.4** |

**20.0 → 48.9 这 28.9 个点的跃升，纯粹来自把 VAE latent 换成 DINO latent**。这个 ablation 极其说明问题。

为什么 VAE 不行？VAE latent space 是为了重建 pixel 而训练的，它**纠缠 (entangle) 了 appearance、geometry、dynamics**。模型在学 forward dynamics $p(o_{t+1} \mid o_t, a_t)$ 时，loss 会被 pixel-level 的 appearance 变化主导——光照变了、纹理变了、背景杂物变了，VAE latent 全跟着变，但这些都不是 action 引起的。

DINO latent (DINOv3 https://arxiv.org/abs/2508.10104) 是 self-supervised 训练出来的，它 encode 的是 **object-level semantic + spatial structure**，而 background clutter、illumination variation 在 DINO feature space 里是被压制的。这就意味着 forward dynamics loss 只需要关心 "action 让 object 怎么动了"，而不是 "action 让 pixel 怎么变了"。

这个 insight 跟 DINO-WM (Zhou et al. 2024) https://arxiv.org/abs/2411.04983 是一脉相承的——他们的工作早就证明在 DINO feature 上做 world model planning 比 pixel space 强得多。LDA 把这个 idea scale 到了 1B foundation model 级别。

---

## 5. MM-DiT 架构：异步流 + 共享 attention

Fig. 2 的架构图可以拆成 4 个 component：

### 5.1 Conditioning Inputs
- **Current observation $o_t$**：通过 frozen Qwen3-VL-4B encode 成 conditioning tokens
- **Language instruction $\ell$**：通过同一个 Qwen3-VL encode
- **Diffusion timestep**：sinusoidal embedding
- **Task embedding**：4 个 learnable embedding，分别对应 policy / forward dyn / inverse dyn / visual forecasting
- **History (2 timesteps)**：past DINO obs + past actions

所有 conditioning 通过 **AdaLN** (Adaptive Layer Normalization, Peebles & Xie 2023) 注入每个 transformer block。

### 5.2 Noised Inputs
两条 token stream 并行 noised：
- **Action tokens**：action chunk $\tilde{a}_{t_a}$，noised 到 timestep $\tau_a$
- **Visual tokens**：future DINO features $\tilde{o}_{t_o'}$，noised 到 timestep $\tau_o'$

### 5.3 MM-DiT Block
每个 block 的核心 operation：

```
[Action tokens | Visual tokens] → modality-specific QKV projection
                              → shared self-attention
                              → modality-specific FFN
                              → AdaLN conditioning
```

这是借鉴 Stable Diffusion 3 / FLUX 那套 MM-DiT 设计 https://arxiv.org/abs/2303.09533 。**关键点**：QKV projection 和 FFN 是 modality-specific 的（保留 inductive bias），但 attention 是 shared 的（让 action 和 visual token 互相 attend，实现 cross-modal interaction）。

### 5.4 Register Token：单模型多任务的 trick
这是个非常聪明的设计。论文说：

> "two learnable register tokens, one for action and one for visual state, serve as placeholders for modalities that are absent in a given task"

具体来说：
- 训练 policy 时：输入是 noised action tokens + **visual register token**（一个占位符，表示"未来 obs 未知"）
- 训练 visual forecasting 时：输入是 noised visual tokens + **action register token**（表示"无 action 监督"）

这样网络拓扑不变，只是输入 token 的 modality flag 变化。这避免了为每个 task 维护单独的 model head，让一个 1B model 灵活支持 4 种 inference mode。

---

## 6. Universal Data Ingestion：数据按 quality 分角色

这是 LDA 的数据策略核心。看 Fig. 1 的示意：

| Data Type | Role Assigned | Hours |
|-----------|---------------|-------|
| High-quality robot demos | Policy + Forward Dyn + Inverse Dyn + Visual Forecast | ~16k |
| Low-quality trajectories | Forward Dyn + Visual Forecast (no policy) | some |
| Actionless human videos | Visual Forecasting only | 10k |

直觉上：
- **High-quality 数据** 有准确的 action label，所以 4 个目标都能监督——这是 BC 的传统地盘。
- **Low-quality 数据** action 不准（pause、retry、inefficient motion），如果用来训练 policy 会让模型学到 suboptimal behavior。但它的 state transition $o_t \to o_{t+1}$ 是物理上合法的，所以可以用来监督 forward dynamics。
- **Actionless video** 完全没有 action，只能监督 visual forecasting——但 visual forecasting 本质是学 "世界自主演化" 的 prior，这不需要 action。

**这个设计的 scaling 意义**：传统 BC 在 data scale 到一定程度后会 plateau，因为 high-quality data 是稀缺的。LDA 通过让 low-quality 和 actionless data 也参与训练，把可利用的 supervision 信号从 ~16k hours 扩展到 30k+ hours。Fig. 10 的 scaling curve 显示，**即使所有 action-labeled data 都用完，加 10k actionless video 还能继续降 prediction error**——这是 paper 最强的 scaling 论证。

参考：R3M https://arxiv.org/abs/2203.12601 ; VIP https://arxiv.org/abs/2210.00030

---

## 7. EI-30k 数据集组成

从 Table IX 拆出来：

| 类别 | 子数据集 | Hours |
|------|----------|-------|
| **Real Robot** | Open X-Embodiment | 3000 |
|  | Agibot World | 3276 |
|  | RoboMIND | 305 |
|  | Humanoid Everyday | 30 |
|  | RoboCOIN | 500 |
|  | Galaxea | 500 |
| **Sim Robot** | LET | 1000 |
|  | InternData-A1 | 7433 |
|  | Behavior-1k | 1200 |
| **Ego Human w/ Action** | Ego4D | 3670 |
|  | Epic-Kitchens | 100 |
|  | Ego-Exo4D | 1286 |
|  | SSV2 | 240 |
|  | EgoDex | 830 |
|  | HOT3D | 16 |
|  | HoloAssist | 166 |
|  | OAKINK2 | 6.5 |
|  | TACO | 3.2 |
|  | HOI4D | 7.6 |
| **Ego Human Actionless** | ARCTIC | 2.3 |
|  | Egocentric-10k | 10000 |
|  | RH20T-human | 100 |
|  | EgoMe | 80 |
|  | Taste-Rob | 130 |
| **Total** | | **30k+** |

**关键工程细节**：
1. 全部统一到 **LeRobot 2.1 format** https://github.com/huggingface/lerobot ——这是个非常务实的决定，避免反复造 format 轮子。
2. Action 全部对齐到 **hand-centric coordinate frame**。Fig. 3 显示了不同 embodiment 的 wrist frame 对齐方式。这意味着同一个 "delta wrist pose + finger config" 表示在 robot gripper、22-DoF Sharpa hand、10-DoF BrainCo hand、人手 MANO 上都能用。
3. Visual 3Hz, Action 10Hz 异步采样——避免相邻 obs 帧冗余，同时保留 fine-grained action dynamics。

参考：Open X-Embodiment https://arxiv.org/abs/2310.08864 ; Ego4D https://arxiv.org/abs/2110.07058 ; Agibot World https://arxiv.org/abs/2510.08807

---

## 8. 实验：RoboCasa-GR1 Simulation

RoboCasa-GR1 是 24 个 tabletop 任务，用 GR-1 humanoid + Fourier dexterous hand，egocentric RGB。每任务 1000 demos fine-tune，51 trials 评估。

Table VI 的 per-task 数据非常 informative。挑几个关键对比：

| Task | GR00T | LDA-1B | Gap |
|------|-------|--------|-----|
| PnP Can To Drawer Close | 13 | 71 | **+58** |
| PnP Milk To Microwave Close | 14 | 52 | **+38** |
| PnP Bottle To Cabinet Close | 51.5 | 76 | **+24.5** |
| PnP Novel From Cuttingboard To Cardboardbox | 46.5 | 69 | +22.5 |
| PnP Novel From Placemat To Tieredshelf | 28.5 | 24 | -4.5 |

注意 LDA 不是在所有任务上都赢——PnP From Placemat To Tieredshelf 这个 GR00T 反而略胜。但**整体 mean 55.4 vs 47.6**，且 advantage 集中在 **"Close" 类任务**（即需要 close drawer/cabinet/microwave 这类 contact-rich 操作）。

Fig. 13 的 qualitative comparison 给了 intuition：GR00T 在 "把物体放进 container 之后撤手" 这个动作上经常把刚放好的物体碰倒，因为它没有 forward dynamics  anticipate 自己撤手轨迹会撞到物体。LDA 由于显式建模了 "如果我执行这个撤手 action，未来 state 会怎样"，能预先规划避碰轨迹。

---

## 9. Real-World 实验：Gripper + Dexterous

### 9.1 Gripper Manipulation (Fig. 6)
8 个任务分 4 类：

| Category | Tasks | LDA 表现 |
|----------|-------|----------|
| Pick & Place | Beat Block, Handover, Pick Pepper | 80-90% |
| Contact-rich | Flip Box | 60% (vs GR00T 20%) |
| Fine | Water Flower, Wipe Board | 80% |
| Long-horizon | Clean Rubbish, Sweep Table | 35% (vs GR00T/π0.5 0%) |

**Long-horizon 的 0% vs 35% gap 是最戏剧化的**。Clean Rubbish 需要：dual-arm coordination + tool use (dustpan) + sequential transfer to trash bin。任何 intermediate step 失败整个任务就失败。BC policy 没有 "我的下一步 action 会让 state 变成什么" 的内部 model，所以一旦 intermediate 偏差就崩。LDA 的 forward dynamics 让它能 anticipate consequence 并 recover。

### 9.2 Dexterous Manipulation (Fig. 7)
3 个 low-DoF (BrainCo 10-DoF) + 2 个 high-DoF (Sharpa 22-DoF) 任务：

| Task | DoF | π0.5 | GR00T | LDA |
|------|-----|------|-------|-----|
| Pick Bottle | 10 | 20% | 75% | **90%** |
| Open MacBook | 10 | similar | similar | similar |
| Pull Nail | 10 | 0% | 40% | **80%** |
| Pick Bread | 22 | 10% | 20% | **70%** |
| Flip Bread | 22 | 10% | 10% | **90%** |

Pull Nail 这个任务特别有意思——需要 claw hammer 精确对准 nail head 并施加稳定 pull force。这是 contact-rich + force-sensitive 的极致场景。π0.5 完全失败说明 BC policy 学不到这种 closed-loop force interaction。LDA 80% 说明 forward dynamics 让它 anticipate hammer-nail contact 的物理后果。

Flip Bread 90% vs 10% 的 gap 也很震撼。这个任务需要 wrist + fingers 协调，spatula 持续接触 bread 并 flip。Action space 是 22-DoF，contact 是 continuous。这恰好是 forward dynamics 最有价值的场景。

---

## 10. Mixed-Quality Fine-Tuning 实验

Table IV 是 LDA 数据哲学的最强 evidence：

| Method | Pen Into Box (63H) | Pen Into Box (63H+37L) | Δ |
|--------|---------------------|--------------------------|---|
| π0.5 | 60% | 40% | **-20** |
| LDA | 70% | 80% | **+10** |

π0.5 加 37% low-quality 数据后掉 20 个点——这是 BC 的典型表现，suboptimal action 污染了 expert distribution。LDA 加同样的 low-quality 数据反而涨 10 个点——因为这些数据虽然 action 不准，但 state transition 是合法的，feed 进 forward dynamics 让模型更好理解物理。

**这个结果对实际部署意义重大**：teleoperation 数据收集天然 noisy，BC 方法需要人工 filter expert subset，成本高。LDA 可以直接用 raw teleop data，省 filter 成本。

---

## 11. Scaling Analysis (Fig. 10)

Fig. 10 是 paper 的 scaling 故事核心。两个子图：

### Top: Data Scaling
- x 轴：训练数据 hours (从少到多)
- y 轴：action prediction L1 error (越低越好)
- 4 条线：Policy Only / Policy + Visual Forecast / Policy + Forward & Inverse Dyn / Full Co-training (Ours)

关键观察：
1. **Policy Only (grey)** 在加 low-quality data 后开始 degrade——这是 BC 的 plateau。
2. **Partial co-training (green/brown)** 比 Policy Only 好但 saturate。
3. **Full co-training (blue)** 持续下降，**即使在 action-labeled data 用完后，加 10k actionless video 还能继续降 error**。

### Bottom: Model Scaling
- x 轴：data hours
- y 轴：L1 error
- 3 条线：UWM vs LDA at 0.1B / 0.5B / 1B

关键观察：
1. **UWM 快速 saturate**，无论怎么 scale model 或 data 都没用——这是 VAE latent 的 bottleneck。
2. **LDA 1B < LDA 0.5B < LDA 0.1B**，monotonic improvement——这说明 DINO latent + MM-DiT 的组合是 scalable architecture。

---

## 12. Qualitative 分析：Attention Heatmap

Fig. 11 是 paper 里我最喜欢的 intuition-building figure。做法：

给定观测 $o$，分别 forward 两次：
1. Action embedding = "Push Right" → 得到 attention map $A_1$
2. Action embedding = "No-Op" (static) → 得到 attention map $A_2$
3. 计算 $|\Delta A| = |A_1 - A_2|$

这个 difference map **isolates action-induced attention**——纯视觉 saliency 在 No-Op 时也会出现，subtraction 后被滤掉。

结果：
- **Push Right**：attention 集中在 mug 的 leading edge + 预期移动方向
- **Push Close**：attention 集中在 drawer contact surface

这说明 LDA 不是在学 "这个物体长什么样"，而是在学 "如果我对这个物体施加 action A，哪些 region 会被 causally affect"。这是 dynamics-aware representation 的直接 evidence。

---

## 13. Gradient Similarity Analysis (Fig. 12)

这是 paper 的另一个隐藏 gem。在 10k 和 400k iteration 测 4 个目标之间的 gradient cosine similarity：

- **Policy ↔ Inverse Dynamics**：早期就高 aligned (都预测 action)，后期更高
- **Forward Dyn ↔ Visual Forecast**：早期就高 aligned (都预测 DINO feature)，后期更高
- **Policy ↔ Forward Dyn**：早期 weak negative (competition)，后期转正
- **所有 pair 在 400k iteration 都变正**

这告诉我们：**multi-task co-training 早期有 gradient competition，但训练充分后不同目标会 converge 到 compatible feature subspace**。这是个非常乐观的 scaling 信号——意味着加更多 task 不会无限加 gradient conflict。

---

## 14. 跟 Karpathy 自己工作的联想

既然你是 Andrej，我猜你会对几个 connection 敏感：

### 14.1 跟 "micrograd" / "nanoGPT" 的 philosophy 对比
LDA 的核心是 "let different data play different roles"。这跟 nanoGPT 那种 "one clean objective, scale it up" 的 philosophy 不一样。LDA 是 multi-objective co-training。但有意思的是，LDA 的 gradient analysis (Fig. 12) 显示 multi-objective 在训练充分后会 align——这某种程度上 redeems multi-task 的复杂性。

### 14.2 跟 Tesla FSD / World Model 路线
Tesla 在 2024 宣布 world model 路线，思路跟 LDA 高度类似：用大量无 action 监督的 driving video 学 world dynamics，再用少量 action-labeled 数据 fine-tune policy。LDA 的 universal data ingestion 就是这个思路在 manipulation 上的 instance。参考：Tesla World Model https://www.youtube.com/watch?v=6x4H4nGgXjw

### 14.3 跟 "Software 2.0" essay 的 connection
你之前写过 Software 2.0 是 "用 data 定义 behavior"。LDA 更进一步——**用不同 quality 的 data 定义不同 aspect 的 behavior**。High-quality data 定义 action policy，low-quality data 定义 dynamics understanding。这是 Software 2.0 的细化版本。

### 14.4 跟 "Attention is All You Need" 的 connection
LDA 的 MM-DiT 把 action token 和 visual token 放在 shared self-attention 里。这跟你一直强调的 "transformer 是通用 differentiable computer" 的哲学一致——不同 modality 不需要专门 module，只需要在 attention 层面交互。

---

## 15. Limitations & 可能的 extension

Paper 自己承认的：
1. **Fixed DINO feature**：DINOv3 是 frozen 的，可能限制了 cross-viewpoint generalization。Future direction: jointly learn visual representation + dynamics。
2. **Egocentric only**：当前只用 head camera。Multi-view 应该会更强。
3. **Manual data role assignment**：哪些数据用于哪些 objective 是人工定义的。Future: 自动学习 data role。

我（作为讲解者）额外的几个想法：
1. **DINO vs VAE 的 trade-off**：DINO 丢了 pixel-level detail。如果任务需要 sub-millimeter 视觉精度（比如 PCB assembly），DINO latent 可能不够。可能需要 hierarchical representation：DINO for semantics + VAE for detail。
2. **Action space 设计**：hand-centric coordinate 是个好 abstraction，但对 whole-body locomotion + manipulation 可能不够。需要扩展到 full-body joint space。
3. **Reward signal**：LDA 完全是 imitation + dynamics，没有 reward。如果叠一个 RL fine-tuning 阶段，可能突破 imitation 的 ceiling。

---

## 16. 总结：LDA 给 robotics foundation model 的几个 takeaway

1. **Dynamics 是被低估的 supervision channel**。BC 只用了 "state→action" 这条 channel，丢了 "state→state'" 这条 channel。LDA 把两条 channel 都用上。
2. **Latent space 选择决定 scalability**。Pixel space / VAE latent 让 dynamics learning 被 appearance entangle，scale 不动。DINO latent 解耦了 appearance 和 dynamics，让 1B scale 稳定。
3. **Heterogeneous data 不是 noise，是不同 channel 的 supervision**。Low-quality 数据对 policy 是 noise，对 dynamics 是 signal。
4. **MM-DiT + register token 是处理 multi-task / multi-modality 的优雅方案**。一个 model 支持多种 inference mode，无需架构改动。
5. **Long-horizon 任务需要 forward dynamics**。BC policy 在 long-horizon 因 compounding error 必然崩。Forward dynamics 让 policy 能 "预见" 后果并规划。

---

## 参考 web links 汇总

- LDA 项目页：https://pku-epic.github.io/LDA
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 paper: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- RDT-1B: https://arxiv.org/abs/2410.07864
- UWM: https://arxiv.org/abs/2504.02792
- UVA: https://arxiv.org/abs/2503.00200
- DINOv3: https://arxiv.org/abs/2508.10104
- DINO-WM: https://arxiv.org/abs/2411.04983
- R3M: https://arxiv.org/abs/2203.12601
- VIP: https://arxiv.org/abs/2210.00030
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- Ego4D: https://arxiv.org/abs/2110.07058
- Agibot World: https://arxiv.org/abs/2510.08807
- LeRobot: https://github.com/huggingface/lerobot
- DDPM: https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- RoboCasa: https://arxiv.org/abs/2406.02523
- DreamVLA: https://arxiv.org/abs/2507.04447
- Being-H0: https://arxiv.org/abs/2507.15597
- UniVLA: https://arxiv.org/abs/2505.06111

---

希望这个 deep dive 帮你 build 出 LDA 的 intuition——它的核心 contribution 不是某个单点创新，而是把 **latent representation + multi-objective co-training + heterogeneous data role assignment + MM-DiT architecture** 这 4 个 piece 拼成一个能 scale 到 1B 的完整 picture。最后一个 thought：这篇 paper 让我想到你之前说的 "data is the new code"——LDA 进一步说，"data quality distribution is the new architecture"。Architecture 决定 model 能学什么，data role assignment 决定 model 从什么里学。
