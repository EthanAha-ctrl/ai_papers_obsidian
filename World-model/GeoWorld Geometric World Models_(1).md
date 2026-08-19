---
source_pdf: GeoWorld Geometric World Models.pdf
paper_sha256: 6971c5cfdd4b488c0fe6c7fdbd5140634b6f6758f5f5dbe2845403a7c2adcd56
processed_at: '2026-08-19T09:33:11-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲GeoWorld

Andrej，好嘞，我把那些公式和jargon先放一边，咱就聊intuition，聊到最后再点几处技术细节。

---

## 1. 这篇paper到底在干嘛——一句话版本

**现在的video world model预测未来的时候，是在一个"平地"上想象，但未来其实长成一棵树；把这块"平地"换成一个"碗"，AI的想象力立刻就有了层次感，长程规划也就不崩了。**

就这么个事儿。下面我慢慢拆。

---

## 2. 先把背景摆清楚：什么是"world model"，什么是JEPA

### 2.1 World model就是AI的"脑内沙盘"

你小时候玩super mario，闭上眼也能想象下一帧水管工会从哪儿冒出来——那就是你脑子里的world model。AI也想要这么个东西：给它一段视频、一个目标，它能"想象"出接下来几步会发生什么，然后挑一条最优路径去走。

现在做world model有两条路：

**Generative world model**（比如Sora、VideoWorld那种）：直接画出来。"我预测下一帧会长这样"——pixel by pixel。好处是直观，坏处是**每一步都要画图，画错一点下一步全崩**，而且画图本身很贵，跟你做planning这件事关系不大。你想要的是"接下来该干啥"，它给你的是"接下来画面长啥样"，中间还得多加一个inverse dynamics model去反推action。

**Predictive world model**（LeCun的JEPA路线）：不画图，只预测"抽象表示"。把当前画面encode成一个向量$s_t$，目标画面encode成$s_{1+T}$，然后学一个predictor $P$去想象"如果我执行action $a$，latent会怎么变"。Planning就变成：找一串action，让imagined latent尽量靠近goal latent。

这个路线的好处是：不浪费算力画没用的pixel，直接在"语义层"做规划。LeCun搞的[V-JEPA 2](https://arxiv.org/abs/2506.09985)就是Meta 2025年的代表作，在百万小时无标签视频上pretrain，然后action-conditioned post-training，planning效果吊打一堆LLM-based方法。

### 2.2 V-JEPA 2的两个毛病

但V-JEPA 2有两个问题，这篇paper就是冲着这俩去的：

**毛病一：latent space是个"没结构的平地"**

V-JEPA 2的latent就是$\mathbb{R}^n$里的向量，用L1距离度量。听起来没啥问题，但你想想——所有的state都"平等"地躺在一个flat space里，没有hierarchy，没有"远近亲疏"的几何含义。两个state的distance就是个数字，不能告诉你"它俩是不是在同一条subtask上"。

**毛病二：长horizon崩盘**

V-JEPA 2-AC主要在one-step transition上训练，到了planning阶段rollout多了，误差一路累积。你看paper里Table 5那个数字：T=3时SR=50.16，T=8时SR=4.95。从一半猜对，到二十分之一猜对，**衰减了10倍**。这就是autoregressive rollout的宿命——每一步错一点点，越往后越离谱。

---

## 3. 为什么"平地"不对——Future是棵树

这是整篇paper最核心的intuition，我得讲透。

### 3.1 想象你站在现在往未来看

假设你现在有个状态$s_t$，有$B$个可能的action。一步future就有$B$种可能，两步有$B^2$种，$d$步有$B^d$种。

这是个**指数爆炸的树**。深度=预测horizon，branching factor=action space大小。

那这种树状结构用什么几何空间表示最natural？答案早就有了——**hyperbolic space**。

### 3.2 为什么是hyperbolic——一个形象的比喻

Euclidean space像一张无限大的纸，你往四面八方走，"能容纳的东西"按$r^2$增长（面积），$r^3$增长（体积），总之是**多项式增长**。

Hyperbolic space像一个**碗**（或者 saddle），你从中心往外走，越往边缘空间越"膨胀"，能装下的点按$e^r$增长——**指数增长**。

这个指数增长恰好匹配树的叶子节点数。所以你把一棵树"塞"进hyperbolic space，叶子能舒舒服服摊开；塞进Euclidean space，就憋屈了——要么严重distortion，要么维度爆炸。

这是[Nickel & Kiela 2017](https://arxiv.org/abs/1705.08039)那篇Poincaré Embeddings的老insight，本来是给WordNet的hierarchy用的，后来被用到了graph、用到了vision-language。GeoWorld就是把它用到world model的latent space上。

### 3.3 一个更直接的图像

想象V-JEPA 2的latent space是一个parking lot，所有车（state）都停在一个平面上，车和车的距离就是直线距离。你问"这俩state是不是在同一个subtask里"，模型回答不了——平面上看不出hierarchy。

GeoWorld把parking lot换成一个**碗状的parking lot**：碗中心是"宏观目标"（比如"换个内存条"），碗边缘是"微观操作"（比如"拧第3颗螺丝"）。同一个subtask的细操作在碗边缘扎堆，宏观到微观的路径是碗中心到边缘的"弧线"。这种结构天然encode了hierarchy。

---

## 4. 怎么把"平地"换成"碗"——Poincaré Ball

技术上这个"碗"叫Poincaré ball model。

### 4.1 直觉版本

Poincaré ball就是欧氏空间里一个open ball（半径$1/\sqrt{c}$的球，内部但不到边界）：

$$\mathbb{B}_c^n = \{z \in \mathbb{R}^n : c\|z\|^2 < 1\}$$

- $z$：ball里的一个点，是个$n$维向量
- $c$：curvature参数，控制碗的"曲率"。$c=0$退化成Euclidean，$c$越大碗越弯
- $\|z\|$：到origin的Euclidean距离，**越接近$1/\sqrt{c}$越接近boundary，越接近boundary在hyperbolic意义上就"越远"**

关键魔法在于：**ball内部的Euclidean距离不等于hyperbolic距离**。两个点Euclidean上看着近，但都在boundary附近时，hyperbolic距离可以是无穷大。这就是"碗"的体现——靠近边缘的地方"距离"被无限拉伸。

### 4.2 Hyperbolic distance公式

$$d_{\mathbb{H}}(u, v) = \frac{1}{\sqrt{c}} \text{arcosh}\left(1 + 2c \frac{\|u - v\|^2}{(1 - c\|u\|^2)(1 - c\|v\|^2)}\right)$$

- $u, v$：ball里两个点
- $\|u - v\|$：它俩的Euclidean距离
- $1 - c\|u\|^2$和$1 - c\|v\|^2$：到boundary的"剩余空间"，越靠近边界这个值越小
- 整个分母越小，整个distance越大——边界附近的点彼此"hyperbolically远"

直白说：**碗中心两个点，Euclidean距离≈hyperbolic距离；碗边缘两个点，Euclidean看着近，hyperbolic其实远得离谱**。这就是hierarchy被encode的机制。

### 4.3 Exponential Map——怎么把Euclidean embedding"塞进"碗里

V-JEPA 2的encoder输出的是$\mathbb{R}^n$的向量，怎么变成ball里的点？

**办法**：把这个Euclidean向量看成"在ball的origin处、tangent space里的一个切向量"，然后用exponential map"走"到ball内部。

$$s_{t, \mathbb{H}}^x = \exp_0(s_t^x) = \tanh(\sqrt{c}\|s_t^x\|) \frac{s_t^x}{\sqrt{c}\|s_t^x\|}$$

- $s_t^x = E_\theta(x_t)$：encoder输出的Euclidean embedding
- $\tanh$：核心魔法。tanh把任何正数压到$[0, 1)$，所以无论$s_t^x$的norm多大，$\tanh(\cdot)$保证最后结果在ball内
- $\sqrt{c}\|s_t^x\|$：curvature-scaled的"走多远"
- 最后乘上$\frac{s_t^x}{\sqrt{c}\|s_t^x\|}$：保留方向

**直觉**：encoder说"这个state应该有多远"，tanh说"再远也不能出碗"，方向不变。这就是一个简单的、可微分的、把$\mathbb{R}^n$映射到$\mathbb{B}_c^n$的层。代码上就是一个non-linear layer，几行PyTorch搞定。

---

## 5. Hyperbolic JEPA：把predictor搬到碗里

### 5.1 改了什么

V-JEPA 2-AC的training loss是：
$$\mathcal{L} = \|P(s_t^x, a_t) - s_{t+1}^x\|_1$$

L1 distance in Euclidean space。

GeoWorld改成：
$$\mathcal{L} = d_{\mathbb{H}}(P(\exp_0(s_t^x), a_t), \exp_0(s_{t+1}^x))$$

Hyperbolic geodesic distance。

**就这一行改动**。Predictor网络本身还是标准Transformer，输入输出都还是tensor，只是loss function从L1换成了hyperbolic distance。Curvature $c$作为learnable scalar一起optimize。

### 5.2 加了个2-step rollout loss

光one-step不够，paper还加了：
$$\mathcal{L}_{\text{rollout}} = d_{\mathbb{H}}(P(\exp_0(s_t^x), a_t, a_{t+1}), \exp_0(s_{t+2}^x))$$

给predictor两个action，让它直接跳两步。这避免autoregressive rollout的数值不稳定，又给了"多步一致性"的supervision。Ablation显示$\lambda = 0.5$（一步两步等权）最好。

### 5.3 一个微妙但重要的点

注意predictor $P$本身**没有改成hyperbolic operation**。它还是标准attention + MLP。它只是把input output都**解释成**hyperbolic manifold上的点，然后用$d_{\mathbb{H}}$做loss。

理论上更"纯"的做法是用[Möbius operation](https://arxiv.org/abs/1805.09112)、[Fractal RNN](https://arxiv.org/abs/2106.04216)这种真正在hyperbolic space做arithmetic的网络。但工程上很难scale，所以这篇paper走了个折中——网络结构保持Euclidean-friendly，loss换成hyperbolic。这是一种实用主义选择。

---

## 6. GRL：让predictor自己给自己当教练

这是paper最clever的部分。

### 6.1 一句话版本

**把prediction error重新解释成RL的reward，然后用value-based RL去refine predictor。不训练额外的policy，不训练额外的reward model，就是让world model自己优化自己。**

### 6.2 怎么把prediction变成RL

定义energy cost（就是hyperbolic distance）：
$$c_t = d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)$$

Reward就是负的energy cost：
$$r_t = -c_t$$

预测越准，reward越高。**预测本身就是reward**。

Path value function（从start到goal的累积reward）：
$$V = \mathbb{E}\left[\sum_{t=1}^T \gamma^{t-1} r_t\right]$$

- $\gamma$：discount factor，0.99最好。越大越重视长horizon
- $r_t$：每步reward = -hyperbolic distance

Optimal value = maximize return = minimize total hyperbolic distance：
$$V^* = \max_\phi \mathbb{E}\left[\sum_t \gamma^{t-1} r_t\right] = \min_\phi \mathbb{E}\left[\sum_t \gamma^{t-1} d_{\mathbb{H}}(\hat{s}_{t+1}, s_{t+1})\right]$$

**最大化reward = 最小化预测误差**。两个目标完全等价，但RL的formulation给了我们一个machinery去优化长horizon的cumulative objective，而不只是单步。

### 6.3 Triangle Inequality Regularization——这个是真有idea

Hyperbolic distance满足triangle inequality：从A到C的距离 ≤ A到B + B到C。

如果predictor想象出来的trajectory确实走在geodesic上（碗里的最短弧线），那么A到C应该 = A到B + B到C（等号成立）。

如果predictor想象的中间state跑偏了（不在geodesic上），那么A到C < A到B + B到C（严格小于）。

Paper定义：
$$\mathcal{L}_\Delta = \frac{1}{T-2} \sum_t \left[d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+2}) - d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+1}) - d_{\mathbb{H}}(\hat{s}_{t+1}, \hat{s}_{t+2})\right]$$

这个量理论上**永远≥0**。最小化它就是在逼predictor"想象中间state时别绕弯路，走geodesic"。

直觉版本：你让AI想象从家到机场的路线，它可能想象成"家→超市→机场"，绕了一下。Triangle inequality regularizer就是在告诉AI："你想象的家→机场的距离，应该等于家→超市 + 超市→机场的距离。如果你想象的中间步骤有用，那绕一段也行，但别瞎绕。"

[Pitis et al. 2020](https://openreview.net/forum?id=rJzUxRktwS)和[Cetin et al. 2023](https://openreview.net/forum?id=hyUcC3T8mI)探索过类似的inductive bias，GeoWorld把它具体化成hyperbolic space里的regularizer。

### 6.4 最终GRL loss

$$\mathcal{L}_{\text{GRL}} = \mathbb{E}\left[\sum_t \gamma^{t-1} d_{\mathbb{H}}(\hat{s}_{t+1}, s_{t+1})\right] + \beta \mathcal{L}_\Delta$$

- $\beta = 0.1$：regularization weight，ablation显示0.1最好，0.2反而差一点（约束太强）
- $\gamma = 0.99$：让远期reward也重要

---

## 7. Planning怎么干——CEM扔飞镖

训练完了，到了test time，给你current frame $x_1$和goal frame $x_{1+T}$，怎么找出action sequence？

### 7.1 老办法：Cross-Entropy Method

CEM是个经典的zero-order optimization，[De Boer et al. 2005](https://link.springer.com/article/10.1007/s10479-005-5729-z)有tutorial。流程：

1. 初始化一个Gaussian distribution over action sequences（mean $\mu_0$, covariance $\Sigma_0$）
2. Sample $N = 800$条candidate action sequences
3. 对每条，用predictor算predicted latent，跟goal latent算hyperbolic distance（这就是energy cost）
4. 选cost最低的$K = 80$条作为"elite"
5. 用elite重新fit Gaussian（更新$\mu, \Sigma$）
6. 重复$I = 10$次
7. 最后cost最低那条就是答案

直觉版本：**你蒙着眼睛扔800个飞镖，看哪些扎在靶心附近，记录那片区域的中心，下次往那扔，扔10轮，越扔越准**。

### 7.2 为什么在hyperbolic space里CEM更好用

在Euclidean space里，energy landscape是个flat的碗状抛物面，CEM的搜索方向没有特别强的guidance，所有方向"长得都差不多"。

在hyperbolic space里，energy landscape有**结构化的directional variation**——沿着geodesic方向energy下降快，偏离geodesic方向energy指数增长。Paper的Figure 2可视化了这点：Euclidean landscape是接近对称的paraboloid，hyperbolic landscape是sharp的、directional的basin。

CEM在这种结构化landscape上自然更efficient——sample更容易往geodesic方向集中。

---

## 8. 实验结果——重点看长horizon

### 8.1 主结果速读

CrossTask Visual Planning with Videos，关键几个数字：

| Method | T=3 SR | T=4 SR |
|--------|--------|--------|
| V-JEPA 2 ViT-g384 | 50.16 | 35.01 |
| GPT-5 (zero-shot) | 50.03 | 30.20 |
| Gemini 2.5 Pro | 48.91 | 31.53 |
| GeoWorld ViT-g384 | 51.71 | 37.04 |

T=3提升约1.5个点，T=4提升约2个点。**T=4时GeoWorld甚至超过GPT-5**——一个700M级别的predictive world model在长horizon planning上击败了万unknowing的LLM，挺有意思的。

### 8.2 长horizon是核心证据

Table 5是真正的高光：

| Method | T=3 | T=5 | T=7 | T=8 |
|--------|-----|-----|-----|-----|
| V-JEPA 2 | 50.16 | 23.17 | 8.26 | 4.95 |
| SFT (Hyperbolic only) | 50.42 | 23.64 | 14.88 | 11.51 |
| GRL (Euclidean only) | 50.26 | 23.85 | 15.12 | 12.74 |
| GRL (Hyperbolic only) | 51.04 | 24.05 | 15.54 | 13.10 |
| SFT + GRL (full) | 51.71 | 24.83 | 16.09 | 13.81 |

**关键观察**：
- T=8时，V-JEPA 2崩到4.95，GeoWorld到13.81，**几乎3倍**
- Hyperbolic geometry alone就贡献了大部分long-horizon gain（4.95 → 11.51）
- GRL在hyperbolic上额外又拉了一截（11.51 → 13.81）
- 两个component是complementary的

这就是paper的claim：**geometry structures the landscape, GRL exploits that structure**。

### 8.3 Curvature学到了什么

Curvature $c$初始化为1，训练中逐渐降到约0.3，然后稳定。

这意味着model学到了一个**"比较平但仍是hyperbolic"**的latent geometry。完全Euclidean（$c \to 0$）丢失hierarchy，太curved（$c$大）数值不稳定，$c \approx 0.3$是个data-driven的sweet spot。

挺有意思——model自己发现"我不需要太弯的碗，稍微弯一点就够了"。

### 8.4 Gromov δ-Hyperbolicity

这是衡量metric space有多"tree-like"的指标。$\delta = 0$就是完美tree，$\delta \to \infty$就是Euclidean。

Paper在CrossTask上sample latent quadruples，算每个model latent space的$\delta$分布。GeoWorld的$\delta$集中在near-zero，V-JEPA 2的$\delta$分布更散。**直接证明了GeoWorld的latent space更tree-like**。

[Gromov δ-hyperbolic space wiki](https://en.wikipedia.org/wiki/Gromov_%CE%B4-hyperbolic_space)有formal定义。

---

## 9. 我对这篇paper的吐槽和喜欢

### 9.1 我喜欢的

**Geometry作为inductive prior用对了地方**。Hyperbolic space匹配exponential branching tree这个motivation非常natural，不是硬凑的。同样的idea在WordNet上work、在knowledge graph上work，现在在world model上也work，是个principled的迁移。

**GRL的"self-reward"设计很clever**。把prediction error当reward，让world model自己refine自己，不需要额外的reward model或policy network，工程上简洁，理论上self-consistent。

**Ablation做得到位**。分别消融hyperbolic和GRL，证明两者complementary。Long-horizon那个Table 5是最好的证据——你能清楚看到每个component在T=8时的贡献。

### 9.2 我觉得有问题的

**绝对提升数字不大**。T=3提升3%，T=4提升2%。用32个H100训练出来的结果，相对V-JEPA 2的提升不算大。性价比存疑。

**训练时只用2-step rollout，测试时planning到T=8**。这个gap有点大。T=8的SR=13.81全靠generalization，model从来没在训练时见过8步的rollout。这能work有点magic，但也说明potential没被挖完——如果训练时也用更长rollout，可能能再拉一截。

**Hierarchical structure的motivation有点闪烁**。Paper一会儿说是subtask hierarchy（像"做饭→切菜→切土豆"），一会儿在Section 7又承认"我们的hierarchy来自future expansion，不是来自subtask decomposition"。这俩是不同的东西，[Hierarchical-JEPA](https://openreview.net/forum?id=Bibir5NcN5)原本的intuition是前者，但GeoWorld实际实现的是后者。这个gap作者承认了但没解决。

**没在continuous action上试**。所有实验都是discrete action的procedural planning（"切菜"、"开盖子"这种）。机器人continuous control上能不能work不知道。我个人最想看的是把这套用到[Droid dataset](https://princeton-vl.github.io/droid_website)的机械臂上，看long-horizon manipulation能不能也拿到类似gain。这才是真正能落地的方向。

**缺一个最直接的baseline**：V-JEPA 2 + multi-step rollout loss in Euclidean space（不加hyperbolic）。这样能更好isolate hyperbolic geometry本身的贡献，而把multi-step supervision的功劳分出来。

### 9.3 让我联想到的

1. **[DreamerV3](https://danijar.com/project/dreamerv3/)**：也是latent world model做planning，但是generative + RSSM。GeoWorld的predictive approach理论上更省，但Dreamer在continuous control上work得很好，是个好的对比点
2. **[Ha & Schmidhuber World Models 2018](https://worldmodels.github.io/)**：latent imagination的开山之作，VAE + MDN-RNN + Controller。GeoWorld某种意义上是这个思路的modern、geometry-aware版本
3. **[π0 / π0.5](https://arxiv.org/abs/2410.24164)**：Physical Intelligence的VLA model，generative action prediction。和GeoWorld的predictive planning是两条路线的对比
4. **[Hyperbolic Image-Text Representations (Desai et al.)](https://arxiv.org/abs/2212.14075)**：MERLOT-ResNet风格的hyperbolic vision representation，可以看作"hyperbolic vision"这条线上的姊妹工作
5. **[Lorentz Model](https://arxiv.org/abs/2003.09122)**：numerical更稳定的hyperbolic model，作者没用，可能是工程trade-off
6. **[E3P](https://arxiv.org/abs/2308.07231)**和[PDPP](https://arxiv.org/abs/2303.01076)：CrossTask上的generative baselines，被V-JEPA 2和GeoWorld吊打，证明predictive paradigm有结构性优势
7. **[Self-forcing / Free noises](https://arxiv.org/abs/2506.08009)**：autoregressive video generation里的rollout consistency trick，思想有共通——"训练时见过rollout才能在inference时稳定"
8. **[VideoWorld (Ren et al. CVPR 2025)](https://arxiv.org/abs/2503.13235)**：直接对比的generative baseline，performance gap证明predictive的优势
9. **[V-JEPA 2 paper](https://arxiv.org/abs/2506.09985)**：当然是必读，GeoWorld的所有baseline都是它
10. **[I-JEPA / V-JEPA lineage](https://arxiv.org/abs/2301.08243)**：LeCun的JEPA从image到video到action-conditioned的演进路线，GeoWorld是给这条线加上geometric prior

---

## 10. Build Intuition的最终收尾

如果让我用一段话给Andrej总结这篇paper的核心insight：

**World model的latent space不是中性的容器，它本身就有几何。如果你把latent放在一个跟problem structure匹配的manifold上，很多好的property会自然涌现——hierarchical planning、long-horizon stability、geodesic-consistent rollouts。Hyperbolic space跟"未来是指数branching tree"这个structure匹配，所以work。GRL做的事情是让predictor在training阶段就explicitly优化"沿着geodesic走"这个objective，而不是被动地学one-step prediction然后指望它generalize到long-horizon。**

更哲学一点：**inductive bias不是约束，是leverage**。你给model一个正确的几何先验，它就用更少的data、更稳定的training、更好的generalization学到你要的东西。这个lesson其实跟LeCun当年推广JEPA的motivation是一脉相承的——"don't predict pixels, predict abstractions"。GeoWorld往前走了一步："don't predict in flat space, predict in the right curved space"。

---

## References

- [GeoWorld Project Page](https://steve-zeyu-zhang.github.io/GeoWorld)
- [V-JEPA 2 (Meta AI 2025)](https://arxiv.org/abs/2506.09985)
- [JEPA (LeCun 2022)](https://openreview.net/forum?id=Bibir5NcN5)
- [I-JEPA (Assran et al. 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA (Bardes et al. 2024)](https://arxiv.org/abs/2304.08870)
- [Poincaré Embeddings (Nickel & Kiela 2017)](https://arxiv.org/abs/1705.08039)
- [Hyperbolic Neural Networks (Ganea et al. 2018)](https://arxiv.org/abs/1805.09112)
- [Hyperbolic GCN (Chami et al. 2019)](https://arxiv.org/abs/1910.12892)
- [Hyperbolic Deep RL (Cetin et al. 2023)](https://openreview.net/forum?id=hyUcC3T8mI)
- [An Inductive Bias for Distances (Pitis et al. 2020)](https://openreview.net/forum?id=rJzUxRktwS)
- [CEM Tutorial (De Boer et al. 2005)](https://link.springer.com/article/10.1007/s10479-005-5729-z)
- [CrossTask (Zhukov et al. 2019)](https://arxiv.org/abs/1812.02757)
- [COIN (Tang et al. 2019)](https://arxiv.org/abs/1903.10882)
- [DreamerV3](https://danijar.com/project/dreamerv3/)
- [Ha & Schmidhuber World Models](https://worldmodels.github.io/)
- [π0 (Physical Intelligence)](https://arxiv.org/abs/2410.24164)
- [Droid Dataset](https://princeton-vl.github.io/droid_website)
- [VideoWorld (Ren et al. CVPR 2025)](https://arxiv.org/abs/2503.13235)
- [Self-forcing (Sun et al. 2025)](https://arxiv.org/abs/2506.08009)
- [Lorentz Embeddings](https://arxiv.org/abs/2003.09122)
- [Gromov δ-hyperbolic space](https://en.wikipedia.org/wiki/Gromov_%CE%B4-hyperbolic_space)
- [AdamW (Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101)

---

# GeoWorld: Geometric World Models 深度解析

Andrej，这篇paper我觉得非常对你的胃口——它本质上是把LeCun的JEPA framework从Euclidean latent space搬到hyperbolic manifold上，然后通过geometry-aware RL去refine predictor的能量landscape。让我从intuition开始，逐步build up到技术细节。

---

## 1. 核心Intuition：为什么Hyperbolic Space适合World Model

### 1.1 状态转移的指数分支结构

考虑一个MDP，状态$s_t$，离散action set $\mathcal{A}$大小为$B$。预测$d$步future时，每个action choice产生一个不同的future trajectory，total futures是$N_d = B^d$。这本质上是**指数增长的tree**：depth对应prediction horizon，branching factor对应action space。

这种exponentially branching tree在Euclidean space里表示是非常inefficient的——欧氏空间的volume只按$r^n$增长（多项式），而tree的叶子节点按$B^d$增长（指数）。Hyperbolic space $\mathbb{H}^n$的关键性质是**volume随radius指数增长**，恰好匹配tree的几何特性。这就是Nickel & Kiela在Poincaré Embeddings (NeurIPS 2017)中提出的经典insight：
- Euclidean：$V(r) \propto r^n$
- Hyperbolic：$V(r) \propto \sinh(r)^{n-1} \approx \frac{1}{2^{n-1}} e^{(n-1)r}$（大$r$时）

Reference: [Poincaré Embeddings](https://arxiv.org/abs/1705.08039) | [Hyperbolic Neural Networks (Ganea et al. 2018)](https://arxiv.org/abs/1805.09112)

### 1.2 为什么V-JEPA 2的Euclidean Latent Space有问题

V-JEPA 2是Meta在2025年发布的predictive world model，基于Joint-Embedding Predictive Architecture (JEPA)——LeCun提出的非生成式latent prediction framework。V-JEPA 2-AC的energy cost是：

$$C((\hat{a}_t); s_1^e, s_1^x, s_{1+T}^x) = \|P((\hat{a}_t); s_1^e, s_1^x) - s_{1+T}^x\|_1$$

这里所有latent state都在$\mathbb{R}^n$中。问题在于：
1. **Geometric neglect**：energy landscape无法capture states之间的hierarchical关系，距离度量和geodesic structure脱节
2. **Multi-step shortcoming**：因为主要在one-step transition上训练，rollout越长误差累积越严重（T=8时V-JEPA 2的SR从50.16%暴跌到4.95%）

Reference: [V-JEPA 2](https://arxiv.org/abs/2506.09985) | [LeCun JEPA paper](https://openreview.net/forum?id=Bibir5NcN5) | [I-JEPA](https://arxiv.org/abs/2301.08243) | [V-JEPA](https://arxiv.org/abs/2304.08870)

---

## 2. H-JEPA：从$\mathbb{R}^n$到Poincaré Ball $\mathbb{B}_c^n$

### 2.1 Poincaré Ball Model回顾

Poincaré ball是hyperbolic space的conformal model：
$$\mathbb{B}_c^n = \{z \in \mathbb{R}^n : c\|z\|^2 < 1\}$$

其中$c > 0$是curvature参数（$K = -c$），ball半径为$1/\sqrt{c}$。当$c \to 0$时趋近Euclidean；$c \to \infty$时becomes更curved。

**关键公式**：Poincaré ball上两点$u, v$的hyperbolic geodesic distance：
$$d_{\mathbb{H}}(u, v) = \frac{1}{\sqrt{c}} \text{arcosh}\left(1 + 2c \frac{\|u - v\|^2}{(1 - c\|u\|^2)(1 - c\|v\|^2)}\right)$$

变量含义：
- $u, v \in \mathbb{B}_c^n$：manifold上的两个点
- $c$：curvature，控制负曲率的程度
- $\|u\|, \|v\|$：到origin的Euclidean距离，越大越接近boundary（"无限远"）
- $\text{arcosh}$：inverse hyperbolic cosine
- 关键性质：当$u, v$靠近boundary时，distance急剧增长（指数）

### 2.2 Exponential Map at Origin

H-JEPA的核心操作是把encoder的Euclidean output $s_t^x \in \mathbb{R}^n$视为tangent space $\mathbf{T}_0 \mathbb{H}^n$ at origin的tangent vector，然后通过exponential map投影到manifold：

$$s_{t, \mathbb{H}}^x = \exp_0(s_t^x) = \tanh(\sqrt{c}\|s_t^x\|) \frac{s_t^x}{\sqrt{c}\|s_t^x\|}$$

变量含义：
- $s_t^x = E_\theta(x_t) \in \mathbb{R}^n$：encoder输出的Euclidean embedding
- $\exp_0$：从origin出发的exponential map
- $\tanh$：把任何magnitude压缩到$[0, 1)$，保证结果落在ball内
- $\sqrt{c}\|s_t^x\|$：curvature-scaled的"行进距离"

**Intuition**：在origin处，tangent space恰好和Euclidean space对齐。tanh的饱和特性自动把任意norm的vector"压"进unit ball内部——norm越大，越接近boundary（在hyperbolic几何中相当于"更深层次的祖先"）。

### 2.3 Predictor在Hyperbolic Space中工作

Action-conditioned predictor $P_\phi$接收hyperbolic latent states序列和actions序列：

$$(\hat{s}_{t+1, \mathbb{H}}^x)_{t=1}^T = P_\phi\left((s_{t, \mathbb{H}}^x, a_t)_{t=1}^T\right)$$

注意：predictor network本身仍然是标准Transformer（300M参数，24层，16 heads，hidden size 1024，GELU），它只是把输入output视为hyperbolic manifold上的点。Loss function用hyperbolic distance而不是Euclidean distance，这才是关键。

---

## 3. Training Objective：把Geodesic作为Optimization Target

### 3.1 Teacher Forcing Loss（一步预测）

$$\mathcal{L}_{\text{TF}}(\theta, \phi) = \frac{1}{T} \sum_{t=1}^T d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)$$

其中$\hat{s}_{t+1, \mathbb{H}}^x = P_\phi(\exp_0(E_\theta(x_t)), a_t)$是predicted state，$s_{t+1, \mathbb{H}}^x = \exp_0(E_\theta(x_{t+1}))$是ground-truth state。

**对比V-JEPA 2**：V-JEPA 2用$\|\cdot\|_1$（L1 Euclidean distance），GeoWorld用$d_{\mathbb{H}}$（hyperbolic geodesic distance）。差异在于：在hyperbolic space中，处于不同hierarchical depth的states之间的distance会被指数放大，这会逼predictor学到geometrically meaningful的intermediate states。

### 3.2 Rollout Loss（两步预测）

$$\mathcal{L}_{\text{rollout}}(\theta, \phi) = \frac{1}{T} \sum_{t=1}^T d_{\mathbb{H}}\left(P_\phi(\exp_0(E_\theta(x_t)), a_t, a_{t+1}), \exp_0(E_\theta(x_{t+2}))\right)$$

这里把$a_t$和$a_{t+1}$都输入predictor，让它预测两步后的state。注意不是autoregressive rollout（即不feed自己的output作为input），而是直接给两个actions。这避免了exponential map的复合数值不稳定。

### 3.3 Total SFT Loss

$$\mathcal{L}_{\text{SFT}}(\theta, \phi) = \lambda \mathcal{L}_{\text{TF}} + (1-\lambda)\mathcal{L}_{\text{rollout}}$$

Ablation study（Table 3）显示$\lambda = 0.5$（等权）效果最好。$\lambda = 1$（纯teacher forcing）在T=4时SR=34.65，$\lambda = 0.5$时SR=35.92。说明multi-step supervision对long-horizon planning至关重要。

---

## 4. Geometric Reinforcement Learning (GRL)：Energy-Based Value Optimization

这是这篇paper最有意思的部分——它**不是训练一个separate policy或reward model**，而是直接通过RL objective refine predictor本身。

### 4.1 Energy Cost和Reward

定义从$s_{t, \mathbb{H}}^x$到$s_{t+1, \mathbb{H}}^x$的energy cost：
$$c_t(s_{t, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x) = d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)$$

Reward就是负的energy cost：
$$r_t = -c_t = -d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)$$

**Intuition**：在标准RL中reward是external signal，这里reward = "预测越准reward越高"。这本质上是把self-supervised prediction error重新解释成RL reward，从而可以用RL的value-based optimization machinery。

### 4.2 Path Value Function

$$V(s_{1, \mathbb{H}}^x, s_{1+T, \mathbb{H}}^x) = \mathbb{E}_{a_{1:T} \sim \phi}\left[\sum_{t=1}^T \gamma^{t-1} r_t\right]$$

变量含义：
- $V$：path value function，衡量从start state到goal state的cumulative reward
- $\gamma \in [0, 1)$：discount factor，ablation显示$\gamma = 0.99$最好
- $a_{1:T} \sim \phi$：actions从predictor参数化的distribution中采样
- $r_t$：单步reward = -hyperbolic distance

**Optimal Value**：最大化return = 最小化total hyperbolic distance：
$$V^* = \max_\phi \mathbb{E}\left[\sum_t \gamma^{t-1} r_t\right] = \min_\phi \mathbb{E}\left[\sum_t \gamma^{t-1} d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)\right]$$

### 4.3 Triangle Inequality Regularization

这是GRL的另一个关键insight。Hyperbolic geodesic distance满足triangle inequality：
$$d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+2}) \leq d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+1}) + d_{\mathbb{H}}(\hat{s}_{t+1}, \hat{s}_{t+2})$$

如果predicted trajectory在geodesic上，那么应该等号成立（geodesic上经过的点，距离之和等于total distance）。Regularization loss：

$$\mathcal{L}_\Delta = \frac{1}{T-2} \sum_{t=1}^{T-2} \left[d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+2}) - d_{\mathbb{H}}(\hat{s}_t, \hat{s}_{t+1}) - d_{\mathbb{H}}(\hat{s}_{t+1}, \hat{s}_{t+2})\right]$$

注意：这个loss **理论上应该≥0**（triangle inequality），所以最小化它意味着让predicted trajectory尽量接近geodesic。如果某段trajectory偏离geodesic（中间点不在最短path上），这个loss就会大于0。

**Connection to prior work**：Pitis et al. 2020的[An Inductive Bias for Distances](https://openreview.net/forum?id=rJzUxRktwS)和Cetin et al.的[Hyperbolic Deep RL](https://openreview.net/forum?id=hyUcC3T8mI)都探索过让neural network尊重triangle inequality。这里把它用作regularizer。

### 4.4 Total GRL Loss

$$\mathcal{L}_{\text{GRL}}(\phi) = \mathbb{E}_{a_{1:T} \sim \phi}\left[\sum_{t=1}^T \gamma^{t-1} d_{\mathbb{H}}(\hat{s}_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^x)\right] + \beta \mathcal{L}_\Delta$$

其中$\beta$是regularization weight，ablation显示$\beta = 0.1$配合$\gamma = 0.99$效果最好。

---

## 5. Energy-Based Planning with CEM

训练完成后，给定当前observation $x_1$和goal observation $x_{1+T}$：

1. **Encode**：$s_{1, \mathbb{H}}^x = \exp_0(E(x_1))$，$s_{1+T, \mathbb{H}}^x = \exp_0(E(x_{1+T}))$
2. **Define energy cost**：
$$C((\hat{a}_t); s_{1, \mathbb{H}}^x, s_{1+T, \mathbb{H}}^x) = d_{\mathbb{H}}\left(P((\hat{a}_t); s_{1, \mathbb{H}}^x), s_{1+T, \mathbb{H}}^x\right)$$
3. **CEM优化**：
$$(a_t^*) = \arg\min_{(\hat{a}_t)} d_{\mathbb{H}}\left(P((\hat{a}_t); s_{1, \mathbb{H}}^x), s_{1+T, \mathbb{H}}^x\right)$$

CEM参数：$N = 800$ samples，$K = 80$ elites，$I = 10$ iterations。这是一种zero-order optimization，sample action sequences from Gaussian，保留cost最低的top-K，re-fit Gaussian，迭代。

Reference: [CEM Tutorial (De Boer et al. 2005)](https://link.springer.com/article/10.1007/s10479-005-5729-z)

**Why CEM works better in hyperbolic space**：在Euclidean space中，CEM在energy landscape上做local search，但这个landscape没有结构化几何。在hyperbolic space中，energy landscape有明确的geodesic structure——沿着geodesic走energy最小，偏离geodesic走energy指数增长。这让CEM的搜索方向有更明确的geometry guidance。

---

## 6. 实验结果分析

### 6.1 主结果（Table 1和Table 2）

**CrossTask Procedural Planning**：
- V-JEPA 2 ViT-g384: T=3 SR=45.58%, T=4 SR=31.36%
- GeoWorld ViT-g384: T=3 SR=47.47%, T=4 SR=31.48%

**CrossTask Visual Planning with Videos**：
- V-JEPA 2 ViT-g384: T=3 SR=50.16%, T=4 SR=35.01%
- GeoWorld ViT-g384: T=3 SR=51.71%, T=4 SR=37.04%

**vs GPT-5**：GPT-5在T=3时SR=50.03%，T=4时SR=30.20%。GeoWorld在T=4上甚至超过了GPT-5！这说明predictive world model在长horizon planning上有结构性优势。

### 6.2 Long-Horizon Planning（Table 5，关键证据）

| Method | T=3 | T=4 | T=5 | T=6 | T=7 | T=8 |
|--------|-----|-----|-----|-----|-----|-----|
| V-JEPA 2 ViT-g384 | 50.16 | 35.01 | 23.17 | 16.88 | 8.26 | 4.95 |
| SFT (Hyperbolic) | 50.42 | 35.92 | 23.64 | 16.97 | 14.88 | 11.51 |
| GRL (Euclidean) | 50.26 | 35.47 | 23.85 | 17.03 | 15.12 | 12.74 |
| GRL (Hyperbolic) | 51.04 | 36.33 | 24.05 | 17.82 | 15.54 | 13.10 |
| SFT + GRL | 51.71 | 37.04 | 24.83 | 18.26 | 16.09 | 13.81 |

**关键观察**：
1. **Error accumulation**是V-JEPA 2的主要瓶颈——T从3到8，SR从50.16跌到4.95，几乎10倍衰减
2. **Hyperbolic geometry alone**（SFT）就能在T=7,8时把SR从8.26/4.95提升到14.88/11.51
3. **GRL alone in Euclidean**也有效，但比hyperbolic GRL弱
4. **Hyperbolic + GRL组合**最好，T=8时SR=13.81，相对V-JEPA 2提升约2.8x

### 6.3 Curvature Dynamics

Curvature $c$初始化为1，训练中逐渐decrease到约0.3。这意味着model学到了一个**较平但仍hyperbolic**的latent geometry。直觉是：完全Euclidean（$c \to 0$）丢失hierarchy，太curved（$c$大）数值不稳定。$c \approx 0.3$是一个data-driven的sweet spot。

Curvature clamp到$[0.1, 10.0]$防止训练不稳定，optimize log(c)保证$c > 0$。这个trick来自Chami et al.的[Hyperbolic GCN](https://arxiv.org/abs/1910.12892)。

### 6.4 Energy Landscape Visualization（Appendix 4）

这是一个非常insightful的分析。在COIN的"Replace Memory Chip"任务上：
- **V-JEPA 2 Euclidean landscape**：近对称的paraboloid，weak directional structure——perturbation被homogeneously treated
- **GeoWorld Hyperbolic landscape**：sharp, curvature-aware basin，strong directional variation——hierarchical structure encoded

公式上，可视化时sweep两个orthonormal tangent space方向$(\Delta x, \Delta y)$：
$$s_{t+1, \mathbb{H}}^{\text{hyp}} = \exp_0(s_t^x + \Delta x \cdot u_1 + \Delta y \cdot u_2)$$
$$\text{Energy}_{\mathbb{H}}(\Delta x, \Delta y) = d_{\mathbb{H}}(s_{t+1, \mathbb{H}}^x, s_{t+1, \mathbb{H}}^{\text{hyp}})$$

其中$u_1$取current state到goal state的方向，$u_2$是orthogonal direction。这个对比清晰展示了hyperbolic geometry如何induce结构化的energy landscape。

### 6.5 Gromov $\delta$-Hyperbolicity（Appendix Fig. 1）

Gromov $\delta$-hyperbolicity衡量metric space有多"tree-like"。对四个点$x_1, x_2, x_3, x_4$，定义：
$$\delta(x_1, x_2, x_3, x_4) = \frac{1}{2}\left|d(x_1, x_2) + d(x_3, x_4) - \max(d(x_1, x_3) + d(x_2, x_4), d(x_1, x_4) + d(x_2, x_3))\right|$$

在tree上$\delta = 0$，在Euclidean space上$\delta \to \infty$。GeoWorld的latent space的$\delta$分布明显集中在near-zero，V-JEPA 2的$\delta$分布更分散——证明GeoWorld学到了更tree-like的geometry。

Reference: [Gromov δ-hyperbolic space (Wikipedia)](https://en.wikipedia.org/wiki/Gromov_%CE%B4-hyperbolic_space)

---

## 7. Implementation细节

### 7.1 Architecture
- Encoder: V-JEPA 2 frozen ViT（ViT-L/H/g/g384多个scale）
- Exponential map layer: 可微分projection，curvature $c$为learnable parameter
- Predictor: 300M-param Transformer，24 layers, 16 heads, hidden size 1024, GELU activations

### 7.2 Training Schedule
**SFT Stage**:
- Optimizer: AdamW, weight decay 0.04
- LR schedule: warmup 4500 iters (7.5e-5 → 4.25e-4), constant 85500 iters, decay 4500 iters
- Batch size: 256

**GRL Stage**:
- 同样的AdamW，但更小LR和更短schedule
- LR: warmup 2000 iters (5.0e-5 → 2.0e-4), constant 18000 iters, decay 5000 iters
- Batch size: 128
- $\gamma = 0.99$, $\beta = 0.1$

### 7.3 Hardware
4 nodes × 8 NVIDIA H100 GPU = 32 H100s，48-core Intel Xeon Platinum 8469C CPU per node，230 GB RAM。Inference用单卡H100。

---

## 8. 我的Critical Analysis

### 8.1 Strengths

1. **Theory-Method Alignment好**：motivation（state transitions形成branching tree）和method（hyperbolic geometry for trees）的match很自然
2. **Ablation做得很彻底**：分别消融hyperbolic geometry和GRL，证明两者complementary
3. **Long-horizon优势明显**：T=7,8时几乎2x提升，证明geometric structure对error accumulation有structural mitigation
4. **No extra policy/reward model**：GRL直接refine predictor，工程上简洁

### 8.2 Potential Weaknesses

1. **绝对提升数字小**：T=3只提升约3%，T=4约2%。考虑到32 H100的训练cost，性价比是否值得商榷
2. **仍然只用T=2 rollout训练**：虽然叫long-horizon planning，但训练时只看到2-step rollout，T=8的planning全靠generalization
3. **Discrete action space**：实验只在discrete action的procedural planning上做。Continuous action（机器人）的extension未验证
4. **Comparison with simpler baselines缺失**：没有对比"V-JEPA 2 + multi-step rollout loss in Euclidean"这个最直接的baseline，难以isolate hyperbolic geometry的贡献
5. **Hierarchical structure来源争议**：Section 7作者承认hierarchical structure来自"multi-step future expansion"而不是"explicit high-level + low-level"——这个动机和Hierarchical-JEPA的原始insight有些gap

### 8.3 联想到的工作

1. **[DreamerV3](https://danijar.com/project/dreamerv3/)**：也是基于latent world model的planning，但是generative（reconstruct pixels）和RSSM-based。GeoWorld的predictive approach理论上更efficient
2. **[World Models (Ha & Schmidhuber 2018)](https://worldmodels.github.io/)**：latent imagination的先驱工作
3. **[Hyperbolic RL (Cetin et al. 2023)](https://openreview.net/forum?id=hyUcC3T8mI)**：在hyperbolic space做RL，是GRL的理论前驱
4. **[MERLOT (Minded et al.)](https://merlot.allenai.org/)**：multi-step temporal prediction for visual reasoning
5. **[PlaTe](https://arxiv.org/abs/2203.15028) and [PDPP](https://arxiv.org/abs/2303.01076)**：这些generative baselines在CrossTask上的表现和V-JEPA 2差很多，说明predictive paradigm确实有优势
6. **[π0 / π0.5 (Physical Intelligence)](https://arxiv.org/abs/2410.24164)**：VLA model，generative action prediction。和GeoWorld的predictive planning形成contrast
7. **[Dreamer-style latents in robotics](https://arxiv.org/abs/2304.10677)**：连续control setting下，看hyperbolic latent是否有用会是很好的future work
8. **[Nickel & Kiela 2017 Poincaré Embeddings](https://arxiv.org/abs/1705.08039)**：tree-like hierarchies in hyperbolic space的奠基工作
9. **[Lorentz Model](https://arxiv.org/abs/2003.09122)**：numerical更稳定的hyperbolic model，作者没采用Poincaré ball而没用Lorentz，可能是工程trade-off
10. **[VideoWorld (Ren et al. CVPR 2025)](https://arxiv.org/abs/2503.13235)**：直接comparison的generative baseline，performance gap证明predictive的优势

---

## 9. Summary：Build Intuition的几个Key Takeaways

1. **Hyperbolic geometry = inductive bias for tree-structured latents**：当你的problem domain有exponential branching structure（multi-step planning的action expansion），把latent space放到hyperbolic manifold上是一个principled的inductive bias，远好于在Euclidean空间硬学。

2. **Geodesic = natural plan**：在hyperbolic space中，"沿geodesic走"和"plan"等价。Energy minimization在geodesic上有明确几何意义，CEM搜索方向也有几何guidance。

3. **Self-supervised reward = -energy cost**：把prediction error重新解释为RL reward，让我们可以用RL的value-based optimization machinery（discount factor、triangle inequality regularization）去直接refine predictor，无需external reward。这本质上是**让world model变成自己的RL trainer**。

4. **Error accumulation的geometric视角**：长horizon planning的失败不只是error accumulation的"量"问题，更是"方向"问题——Euclidean space中误差drift方向无约束，hyperbolic space中geodesic约束把drift限制在hierarchy-meaningful directions。

5. **Frozen encoder + learnable curvature**：encoder保持frozen避免catastrophic forgetting，curvature作为单一learnable scalar控制整个manifold的"曲率"，这个参数efficiency很高——一个scalar就能改变entire latent geometry。

---

## References

- [GeoWorld Project Page](https://steve-zeyu-zhang.github.io/GeoWorld)
- [V-JEPA 2 (Meta AI 2025)](https://arxiv.org/abs/2506.09985)
- [JEPA (LeCun 2022)](https://openreview.net/forum?id=Bibir5NcN5)
- [I-JEPA (Assran et al. 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA (Bardes et al. 2024)](https://arxiv.org/abs/2304.08870)
- [Poincaré Embeddings (Nickel & Kiela 2017)](https://arxiv.org/abs/1705.08039)
- [Hyperbolic Neural Networks (Ganea et al. 2018)](https://arxiv.org/abs/1805.09112)
- [Hyperbolic GCN (Chami et al. 2019)](https://arxiv.org/abs/1910.12892)
- [Hyperbolic Deep RL (Cetin et al. 2023)](https://openreview.net/forum?id=hyUcC3T8mI)
- [An Inductive Bias for Distances (Pitis et al. 2020)](https://openreview.net/forum?id=rJzUxRktwS)
- [Cross-Entropy Method Tutorial (De Boer et al. 2005)](https://link.springer.com/article/10.1007/s10479-005-5729-z)
- [CrossTask (Zhukov et al. 2019)](https://arxiv.org/abs/1812.02757)
- [COIN (Tang et al. 2019)](https://arxiv.org/abs/1903.10882)
- [AdamW (Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101)
- [VideoWorld (Ren et al. CVPR 2025)](https://arxiv.org/abs/2503.13235)
- [π0 (Physical Intelligence)](https://arxiv.org/abs/2410.24164)
- [Droid Dataset](https://princeton-vl.github.io/droid_website)
- [Gromov δ-hyperbolic space](https://en.wikipedia.org/wiki/Gromov_%CE%B4-hyperbolic_space)
- [Metric Spaces of Non-Positive Curvature (Bridson & Haefliger)](https://link.springer.com/book/10.1007/978-3-0348-9025-5)

---

这篇paper给我的最大启发是：**geometry is a strong inductive prior for world models**。当LeCun的JEPA framework被放在正确的manifold上，很多"emergent property"（hierarchical planning、error stability）会自然涌现。下一步值得探索的方向：把这套hyperbolic latent ideas用到continuous control和robotics上，看geometric planning是否能scale到robot manipulation的long-horizon tasks。
