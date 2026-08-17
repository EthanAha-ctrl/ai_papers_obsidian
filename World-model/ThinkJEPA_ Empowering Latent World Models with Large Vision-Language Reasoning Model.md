---
source_pdf: ThinkJEPA_ Empowering Latent World Models with Large Vision-Language Reasoning
  Model.pdf
paper_sha256: ae5f3a389db026c54a3794ec4e1eb5363105d12704c6209d605530cf6e569c47
processed_at: '2026-08-12T15:46:38-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ThinkJEPA

## 一句话说清楚

V-JEPA 这种 latent world model 擅长预测"下一帧手会怎么动"，但它只盯着眼前一小段 dense window，不知道自己在看什么、在干什么任务。VLM 知道"这是在倒水、杯子要放到架子上"，但它没法输出精细的连续轨迹。**ThinkJEPA 就是让 VLM 在旁边"想"，把它的 thinking 状态用 FiLM 调制塞进 V-JEPA 的 predictor 里，帮它预测得更准、更稳。**

## V-JEPA 的问题

V-JEPA 用一个 dense 的小窗口（比如 32 帧）去看视频，然后预测未来 32 帧的 latent。问题是：

- **窗口太短，没大局观。** 它只知道"最近这几帧手在往右移"，但不知道"整个任务是把杯子从桌上拿到架子上"。所以它只能做 local extrapolation——最近怎么动，接下来就怎么动。这在长程上会 drift。
- **latent space 没语义。** 它的 representation 是 self-supervised 学出来的，对 motion 敏感，但不知道"那是杯子"、"这是螺丝刀"。换个没见过的 task 就容易崩。

这就像一个人只会盯着自己的脚看路面，不会抬头看路牌——短程走得稳，但不知道自己在去哪。

## VLM 的问题

VLM（这里用 Qwen3-VL Thinking）恰好互补：它能看 8 帧均匀采样的长程视频，告诉你"这段在倒水、手会从左边抓起杯子然后往右倾倒"。有语义、有长程、有 world knowledge。

但它不能直接当 predictor：

- **太 sparse。** 8 帧 vs V-JEPA 的 64 帧，高频运动细节（指尖接触、微滑动）全丢了；
- **language bottleneck。** 它的输出被压成 token，连续的物理状态（精确位置、速度）被离散化，metric-space 的精度不可能好；
- **小数据 finetune 会忘。** 在 EgoDex 这种小数据集上 fine-tune VLM 会 catastrophic forget 掉通用知识。

所以 VLM 适合当"军师"，不适合亲自下场。

## 怎么接

ThinkJEPA 的做法是 **dual-temporal + FiLM 注入**：

1. **同一段视频采样两次：**
   - Dense clip（64 帧）喂给 V-JEPA → 高频 motion 信号；
   - Uniform clip（8 帧）喂给 VLM → 长程语义 reasoning。

2. **从 VLM 里挖多层的 thinking trace：**
   不只拿最后一层——最后一层是"已经决定要说什么"，中间层才是"还在想"。从 layer {0, 4, 8, 12, 16, 20, 24, 27} 都挖出来，pool 成 guidance 特征。同时挖两条路：encoder tokens（视觉内容）+ AR tokens 的中间 hidden states（推理过程）。

3. **用 FiLM 注进 predictor：**
   guidance 特征经 MLP 映射成 per-block 的 $(\gamma, \beta)$，对 predictor 每一层的 feature 做 channel-wise 的 $\gamma \odot z + \beta$。VLM 不直接出轨迹，只出 modulation——相当于在 V-JEPA 耳边说"你在倒水，手应该往右上方走"，而不是替它走路。

4. **输出仍然是 latent：**
   predictor 的输出是 future latent tokens，接一个轻量 head 回归 3D hand trajectory。这保留了 JEPA 的 latent forecasting 接口，下游可以接任何 task head。

## 实验，重点是 rollout

主表上 ThinkJEPA 比 VLM-only 和 V-JEPA-only 都好，这个不意外。真正有意思的是 **recursive rollout**：

| Horizon | V-JEPA ADE | ThinkJEPA ADE |
|---|---|---|
| @4 | 0.121 | 0.071 |
| @16 | 0.134 | 0.092 |
| @32 | 0.142 | 0.111 |

VLM-only 在 rollout 下直接爆炸（@8 就 0.819 了）。V-JEPA 稳定但慢慢 drift。ThinkJEPA 在所有 horizon 都最低，而且 **horizon 越长优势越明显**。

这说明 VLM guidance 真正的价值不是一次性提升，是 **当 long-horizon anchor 抑制 error accumulation**。每一步 rollout 都被 VLM 的语义 prior 拉回 task manifold，防止 drift。这跟你让 LLM 边写边回看 prompt 的效果是一样的。

## 两个我觉得聪明的设计

**Pyramid extraction。** 别人用 VLM 通常只拿最后一层。这里从 8 个深度层都挖——实验里 last-layer 和 mid-layer 单独用效果差不多，但 all-layer 合起来 ADE 从 0.128 直接跳到 0.061。这说明 VLM 的"思考过程"分散在多层里，只看最终输出是在浪费信息。

**Encoder tokens + AR tokens 都要。** 单独用 encoder（视觉内容）或单独用 AR（推理 trace）效果都一般，两者合起来才有数量级跳跃。encoder 告诉你"画面里有杯子和手"，AR 告诉你"接下来手会去抓杯子"——一个回答 where，一个回答 what next。

## 跟其他工作的关系

- vs **V-JEPA2**：V-JEPA2 是纯视觉 latent world model，ThinkJEPA 在它上面加了 VLM 这个 semantic side channel；
- vs **VL-JEPA**：VL-JEPA 把 language 融进 JEPA 但输出偏 language understanding，ThinkJEPA 保留 latent forecasting 接口，VLM 只当 guidance；
- vs **Dreamer**：Dreamer 在 latent 里做 RL rollout + actor-critic，ThinkJEPA 是 observation-only forecasting + VLM prior，没 action、没 RL——但这是通往 action-conditioned world model 的自然前一步；
- vs **RT-2 / SayCan**：那些让 VLM 直接出 action token，ThinkJEPA 让 VLM 出 modulation 不出 action，避开了 language bottleneck 和 catastrophic forgetting；
- vs **diffusion policy**：在 EgoDex 上 ThinkJEPA 把 BC（0.077）打到 0.061，DDPM / Flow Matching 反而更差（0.11-0.17）。可能是 EgoDex 的长程、高维、metric-sensitive 场景下，latent forecasting + VLM semantic grounding 比 denoise-from-noise 更直接。

## 最重要的 takeaway

ThinkJEPA 的核心 insight 其实不是什么新架构，是 **"让擅长不同事情的模块各干各的，用一个轻量通道把它们的强项缝起来"**。VLM 不下场，V-JEPA 不抬头，FiLM 是它们之间的翻译层。rollout 实验证明这种缝合在长程上尤其值钱——这跟你直觉里 "system 2 给 system 1 提供长程 plan" 的故事完全对上。

个人觉得最自然的下一步是加 action conditioning，让 VLM 的 thinking 真正参与 decision 而不只是 trajectory forecasting。那时候 $(\gamma, \beta)$ 就不只是 modulation，是 "thought → action prior" 的桥。

---

# ThinkJEPA：让 latent world model 借 VLM 的"思考"来补语义与长程性

下面我尽量把这篇 paper 拆到能让你的 intuition 直接落地的颗粒度。我会按 "what → why → how → experiment → 在更大的 landscape 里它在哪"的顺序走，并在关键处把公式变量、上标下标讲清楚，最后给一些我作为 reader 会去追问的 open questions。

---

## 1. 一句话的 intuition

ThinkJEPA 想做的事情可以这么概括：**一个 dense、低层、motion-sensitive 的 JEPA latent predictor 负责把"未来会在哪、怎么动"预测准；一个 sparse、高层、semantic-rich 的 VLM（Qwen3-VL Thinking）作为 "thinker" 在旁边提供"这是什么物体、在做什么任务、大概会怎么发展"的长程语义 guidance**。两者通过 dual-temporal sampling 和 FiLM 调制接在一起，输出仍然是 latent token，下游接一个轻量 trajectory head 回归 3D hand pose。

用你熟悉的 system-1 / system-2 类比，这就像让一个 fast、reflexive 的 latent dynamics 模型（system 1，dense，metric-space accurate）在一个 slow、deliberative 的 VLM reasoner（system 2，semantic，long-horizon）的"思路提示"下做事——但 VLM 不直接产轨迹，只产 modulation 信号。这正好避开了 VLM 当 dense predictor 的三个老毛病：compute-driven sparsity、language-output bottleneck、data-regime mismatch。

---

## 2. 这篇 paper 想填的洞：现有 latent world model 的两个 failure mode

V-JEPA2 这类 JEPA-style latent world model 已经很能打，但作者指出两个结构性短板：

**Limitation 1 — temporal perspective 受限于 dense short window。**  
V-JEPA 用密集采样的小窗口去预测未来 latent。dense 对高频 motion（指尖接触、微滑动）友好，但窗口短 → temporal context 浅 → predictor 容易学成 "local extrapolator"：把最后一帧的速度方向延续下去。这跟你在 autoregressive LM 里看到的 "local n-gram bias" 是一类问题——上下文不够长，模型就只能拟合最近邻。

**Limitation 2 — semantic grounding 弱。**  
latent space 是 self-supervised masked prediction 学出来的，对 motion 敏感，但跟 open-vocabulary 概念、compositional knowledge 没对齐。predictor 知道"东西在动"，但不知道"那是杯子、要被放到架子上"，于是 out-of-distribution 的 task / scene 就容易崩。

VLM 恰好是另一极：semantic 强、长程强，但 dense 不行（quadratic attention + GPU memory）、language-output bottleneck（连续物理状态被压成 token）、小数据 finetune 会 catastrophic forgetting。**所以结论很自然：VLM 当 side-channel thinker，不当 predictor。**

---

## 3. Dual-temporal perception field：把"两件正交的事"交给两条路

核心设计是把同一段视频 $v = \{I_t\}_{t=1}^N$ 采样成两份：

### 3.1 Uniform clip 给 VLM thinker（大 temporal perception field）

$$v_u = \{I_{s_i}\}_{i=1}^{N_u}, \quad s_i = \left\lfloor 1 + (i-1)\cdot\frac{N-1}{N_u-1}\right\rfloor \tag{1}$$

变量含义：
- $N$ 是原 clip 总帧数；
- $N_u$ 是喂给 VLM 的采样帧数（小，比如 8）；
- $s_i$ 是第 $i$ 个采样帧在原 clip 里的 index，用 floor 取整；
- 这种 $\lfloor\cdot\rfloor$ 线性 spacing 保证首尾帧一定被采到，覆盖整个 horizon。

直觉：VLM 在 $N_u \ll N$ 下仍能看到"开头—中间—结尾"，做出 task-level 判断（"在倒水"、"在拧螺丝"）。代价是丢掉高频 motion——但 VLM 本来也不负责这个。

### 3.2 Dense clip 给 JEPA branch（小 temporal perception field，高 FPS）

$$v_d = \{I_t\}_{t=t_0}^{t_0+N_d-1} \tag{2}$$

变量含义：
- $t_0$ 是观察窗口起点；
- $N_d$ 是 dense 帧数（实验里 past/future split = 32/32，所以一个 window 64 帧）；
- 这里**不做 temporal subsampling**，所有帧都保留，保证 high-frequency cues。

直觉：dense window 是 JEPA 的"显微镜视野"——窄但分辨率高，能看见接触、滑动、指尖微小形变。

### 3.3 为什么这两条路不是 redundant

这其实是信号处理里的 multi-resolution 思路，跟 wavelet / coarse-to-fine 一脉相承。uniform sampling 给的是 **coarse, long-range envelope**（"这一段在干什么"），dense sampling 给的是 **fine, short-range detail**（"下一帧手会怎么转"）。把它们耦合成 dual-temporal，等价于在时间轴上同时拿到低频和高频成分。单独任何一条都会漏掉另一条的信息——VLM-only 把高频抹平，JEPA-only 把低频截断。

---

## 4. JEPA branch 内部：tokenization 与 recursive rollout

V-JEPA-L backbone（ViT-Large + RoPE）把 $v_d$ 编码成 per-frame patch tokens：

$$F \in \mathbb{R}^{B \times T \times P \times D}$$

- $B$：batch size；
- $T$：window 内帧数（这里 $T = 64$，含 past+future）；
- $P$：每帧 spatial token 数；
- $D$：backbone latent dim = 1024。

predictor 在 internal dim $D_p = 384$ 里跑，输出再投影回 $D = 1024$。对未来窗口超过单次 forward 的情况，用 recursive rollout：

$$\hat{F}_k^{\text{fut}} = g(F_k^{\text{past}}) \tag{3}$$
$$F_{k+1}^{\text{past}} \leftarrow \hat{F}_k^{\text{fut}} \tag{4}$$

- $k$ 是 rollout step index；
- $g(\cdot)$ 是 JEPA-style predictor；
- $F_k^{\text{past}}$ 是第 $k$ 步输入的 past latent tokens；
- $\hat{F}_k^{\text{fut}}$ 是该步预测的未来 latent；
- 下一步把预测的未来当新的过去，迭代下去。

**这就是 autoregressive latent rollout，和 LLM 的 token-by-token 生成是同构的**，所以它也继承了同样的毛病：error accumulation / exposure bias。每一步的小误差会在 $k$ 增大时 compounding，轨迹越长越崩。后面的 rollout 实验里 V-JEPA 在 $H=32$ 时 ADE 从 0.121 涨到 0.142，就是这个曲线的体现。

VLM guidance 的价值正是在这里：它像一个"长程 anchor"，在每个 rollout step 注入语义 prior，把 drifting 的 latent 拉回 task manifold。这跟你在 LLM 里看到的 "让 model 边想边写、中途 re-ground 到 prompt" 的效果很像。

---

## 5. VLM Thinker：Hierarchical Pyramid Representation Extraction

这是这篇 paper 我认为最值得关注的小创新点。

### 5.1 问题：用哪一层 VLM feature？

直觉反应是拿最后一层 hidden state。但作者指出：**深层越来越被 language-generation objective 拉偏**——spatial detail 被压进 token vocab 的方向，连续物理状态被离散化。中间层反而保留了更多 visual reasoning trace 和 spatial sensitivity。这跟你和很多 probing 工作看到的结论一致：LLM 的中间层是"还在 think 的地方"，最后一层是"已经决定要说什么"的地方。

### 5.2 方案：pyramid over depth

从 Qwen3-VL (Thinking) 的两层来源各取多深特征：

1. **Encoder tokens**：VLM 的 ViT visual tokenizer 输出（visual content summary，spatial 细节多）；
2. **AR tokens 的中间 hidden states**：language model 在 thinking / deepstack 过程里多层的 hidden state（reasoning trace）。

具体选了 layers $\mathcal{L} = \{0, 4, 8, 12, 16, 20, 24, 27\}$。把这些 depth 的特征 pool + project 成 guidance 特征 $\phi(v_u)$。

这相当于把 VLM 在时间维（uniform 采样）和深度维（多层 hidden）两个轴上同时"打开"，给 JEPA predictor 一个金字塔形的 reasoning context。

### 5.3 为什么 encoder + AR 两路都要

ablation（Tab. 2）很说明问题：

| Variant | ADE↓ | FDE↓ | Acc↑ |
|---|---|---|---|
| Encoder + V-JEPA | 0.128 | 0.129 | 0.100 |
| Encoder-only | 0.143 | 0.145 | 0.086 |
| AR + V-JEPA | 0.128 | 0.130 | 0.098 |
| AR-only | 0.142 | 0.144 | 0.086 |
| ThinkJEPA (both + dense JEPA) | **0.061** | **0.056** | **0.596** |

encoder tokens 单独、AR tokens 单独都只比 VLM-only 略好，但两个合起来 + dense JEPA 才有数量级提升。我的解读：encoder 提供 "what is where"（视觉内容），AR 提供 "what should happen next"（reasoning trace），两者合起来才是完整的 "think"。

---

## 6. Guidance 怎么注入：FiLM 调制

这是把 VLM 信号塞进 JEPA predictor 的关键 mechanics。作者用 feature-wise linear modulation (FiLM)：

$$\text{FiLM}(z; \gamma_\ell, \beta_\ell) = \gamma_\ell \odot z + \beta_\ell \tag{6}$$

变量含义：
- $z$ 是 predictor 第 $\ell$ 个 block 的输入 feature（shape 跟该 block 的 token 一致）；
- $\gamma_\ell, \beta_\ell$ 是 guidance 经一个轻量 MLP 从 $\phi(v_u)$ 映射出来的、per-block 的 affine 参数；
- $\odot$ 是 element-wise（channel-wise）乘法。

整条 conditioned 预测管线写出来是：

$$\hat{F}^{\text{fut}} = g\big(F^{\text{past}}(v_d);\ \phi(v_u),\ p\big) \tag{5}$$

- $F^{\text{past}}(v_d)$：dense clip 经 V-JEPA backbone 的 past latent tokens；
- $\phi(v_u)$：VLM guidance（pyramid 提取后）；
- $p$：text prompt（task name + scene description，帮 thinker focus）；
- $g(\cdot;\cdot)$：被 FiLM 调制的 JEPA predictor。

### 6.1 为什么 FiLM 而不是 cross-attention / AdaLN

Supplementary Tab. 8 给了对照：

| Conditioning | ADE↓ | FDE↓ | FD↓ |
|---|---|---|---|
| FiLM | 0.0706 | 0.064 | 73.878 |
| Cross-attn | 0.0707 | 0.066 | 73.965 |
| AdaLN | 0.0708 | 0.065 | 74.280 |

三者差不多，但作者选 FiLM 的理由（我觉得是对的）：
- FiLM 直接在 predictor latent space 上做 channel-wise affine，**modulate 的是 representation 本身**，跟"提升 latent prediction quality"的目标最对齐；
- cross-attention 会引入 token-token 交互，结构改动大，难把收益归因到 guidance；
- AdaLN 走 normalization 通路，控制更间接。

这跟你在 DiT 里看到的 AdaLN 用来 condition 是不同语境——DiT 是 condition 生成过程，这里是想 condition 一个预测的 latent 表征本身，FiLM 的"直接改 channel gain/bias"更贴合。

---

## 7. 下游：trajectory head

latent 预测完之后接一个轻量 head：
1. 每帧 spatial tokens 用 attention pooling（learnable query）聚成 per-frame representation；
2. temporal MLP 建模跨帧依赖；
3. stride-2 temporal downsampling（64→32）对齐预测 horizon；
4. linear projection 输出 $32 \times 52 \times 3$：32 future frames × 52 joints × 3 (xyz)。

这个 head 很"瘦"——说明 representation 本身已经够好，head 只做几何回归。这跟 V-JEPA2 的 downstream protocol 一致。

---

## 8. 实验数据：把直觉对齐到数字

### 8.1 主表（Tab. 1）

EgoDex（egocentric dexterous manipulation，3D hand pose）：

| Model | ADE↓ | FDE↓ | Acc↑ | FD↓ | SL1↓ | CD↓ |
|---|---|---|---|---|---|---|
| Qwen3-VL Thinking (VLM-only) | 0.142 | 0.144 | 0.084 | 99.538 | 1.656 | 0.615 |
| V-JEPA Predictor | 0.071 | 0.066 | 0.471 | 74.223 | 1.252 | 0.317 |
| ThinkJEPA | **0.061** | **0.056** | **0.596** | **74.032** | **1.248** | **0.315** |

读法：
- VLM-only 的 ADE 0.142、Acc 0.084 —— VLM 当 standalone dense predictor 确实崩，验证了 language-output bottleneck 的直觉；
- V-JEPA-only 的 ADE 0.071、Acc 0.471 —— dense dynamics 强但语义弱；
- ThinkJEPA 在 trajectory 和 latent metrics 上**两边都赢**，说明 guidance 不只是帮 head，是帮了 representation prediction 本身（FD/SL1/CD 都降）。

EgoExo4D 上同样的 pattern，但 ThinkJEPA 的 Acc 提升特别大（0.074 → 0.171）——更难、更多样的场景里，semantic guidance 的边际收益更大。

### 8.2 vs 任务专用 trajectory baselines（Tab. 3）

| Model | ADE↓ | FDE↓ |
|---|---|---|
| Decoder-only + BC | 0.0767 | 0.0818 |
| Decoder-only + DDPM | 0.1148 | 0.1238 |
| Decoder-only + Flow Matching | 0.1527 | 0.1574 |
| Encoder-decoder + BC | 0.0774 | 0.0924 |
| ThinkJEPA | **0.0610** | **0.0560** |

ThinkJEPA 把 BC（之前 SOTA-ish）从 0.077/0.082 拉到 0.061/0.056。DDPM 和 Flow Matching 反而更差——这个有点反直觉，因为 diffusion policy 在 robot manipulation 里通常很强。我的猜测：EgoDex 的 horizon 2 秒、关节多，diffusion 的 iterative denoise 在这种长程、高维、metric-sensitive 任务上没占到便宜，而 latent forecasting + 轻 head 反而更稳。这点值得在更广的 robot benchmark 上再验。

### 8.3 Long-horizon rollout（Tab. 5）—— 这张表最能说明问题

| Model | A@4 | A@8 | A@16 | A@32 | F@4 | F@8 | F@16 | F@32 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL Thinking | 0.140 | 0.819 | 1.375 | 1.026 | 0.143 | 2.850 | 0.286 | 1.092 |
| V-JEPA Predictor | 0.121 | 0.126 | 0.134 | 0.142 | 0.124 | 0.136 | 0.149 | 0.153 |
| ThinkJEPA | **0.071** | **0.078** | **0.092** | **0.111** | **0.073** | **0.090** | **0.118** | **0.136** |

- VLM-only 在 rollout 下直接爆炸（A@8 = 0.819，A@32 = 1.026），完全验证 "VLM 不能当 dense predictor"；
- V-JEPA 稳定但 drift（0.121 → 0.142）；
- ThinkJEPA 在所有 horizon 都最低，且 **horizon 越长相对优势越大**（@4 差 0.05，@32 差 0.031 但基数更小说明 proportion 更大）。

这跟 "guidance 当 long-horizon anchor 抑制 compounding error" 的直觉完全吻合。

### 8.4 Layer selection ablation（Tab. 4）

| Variant | ADE↓ | FD↓ |
|---|---|---|
| Last-layer | 0.128 | 78.858 |
| Mid-layer | 0.128 | 78.517 |
| All layers (ThinkJEPA) | **0.061** | **74.747** |

last-layer 略好 trajectory、mid-layer 略好 latent metrics，all-layer 把两者都拿到。这说明不同深度携带不同信息，金字塔聚合不是"锦上添花"，是结构性必要。

---

## 9. 这篇 paper 在 landscape 里的位置

把它放进你熟悉的几条线里看会更清楚：

**JEPA 系（LeCun 路线）**：I-JEPA → V-JEPA → V-JEPA2 → VL-JEPA。V-JEPA2 已经能做 understanding + prediction + planning，但还是纯视觉。VL-JEPA 把 language 信号融进 JEPA，但目的是 multimodal understanding，输出空间偏 language。ThinkJEPA 反过来——**保留 JEPA 的 latent forecasting 接口，把 VLM 降级为 thinker/guidance**。这是一个很务实的分工：让 VLM 干它最擅长的语义 reasoning，让 JEPA 干它最擅长的 dense latent dynamics。
- V-JEPA2: https://arxiv.org/abs/2506.09985
- VL-JEPA: https://arxiv.org/abs/2512.10942
- I-JEPA: https://arxiv.org/abs/2301.08243

**World model 系（Dreamer / Genie / Sora-like）**：Dreamer 在 latent 里 rollout + actor-critic，Genie 用 latent action model 做 controllable generation，Sora 走 pixel-space diffusion。ThinkJEPA 跟 Dreamer 最近（都 latent rollout），但 Dreamer 是 RL-centric、action-conditioned，ThinkJEPA 这里是 observation-only forecasting + VLM semantic prior。它更像 "Dreamer 的 representation 部分被 VLM 加持"。
- Dreamer: https://arxiv.org/abs/1912.01603
- Genie: https://arxiv.org/abs/2402.19427

**VLM-as-controller 系（RT-2 / VIMA / SayCan）**：这些是让 VLM 直接出 action token。ThinkJEPA 不走这条路——VLM 不出 action，只出 modulation。这避开了 "VLM 输出离散 token 无法表达连续 metric" 的 bottleneck，也避开了小数据 finetune 的 catastrophic forgetting（这里 VLM 是 frozen + cached）。
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691

**Diffusion / Flow-Matching policy 系**：DDPM、Flow Matching 在 robot manipulation 上很流行，但这里它们被 latent forecasting 超过。一个可能的解释是：diffusion 在"从噪声塑形到一条具体轨迹"时，对长程 task structure 的 grounding 弱；而 ThinkJEPA 把 task structure 直接 encode 进 latent prediction 的 prior 里。
- DDPM: https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747

**Thinking / o1-style reasoning**：Qwen3-VL (Thinking) 的 deepstack token + 中间 hidden state 被当成 reasoning trace 提取出来——这跟 o1 的 "let's think step by step" 内部化是一回事。ThinkJEPA 把这种 reasoning trace 蒸馏成 FiLM 参数，等于让 latent predictor "听见" VLM 的思考过程，而不是只听它的最终回答。这个 idea 我觉得可以推广得很远。
- Qwen3-VL: https://arxiv.org/abs/2511.21631

---

## 10. 我会去追问的 open questions

读完我会想这几个方向：

1. **VLM 能不能端到端被 finetune？** 现在 VLM 是 frozen + cached，guidance 是 one-shot 注入。如果让 VLM 的 thinking 随 JEPA 的 prediction loss 反向梯度（哪怕只调 LoRA），guidance 会不会变得更 "prediction-aligned"？这跟 RLHF 里 reward model 跟 policy 共训的 trade-off 类似。

2. **Guidance 的 ablation 还可以更细。** 现在 $\mathcal{L} = \{0,4,8,12,16,20,24,27\}$ 是手动选的。如果跑一个 learnable layer-weighting（soft pyramid），权重会不会自然集中在中层？这能验证 "中间层更 visual" 这个 claim 是不是 causal 而不只是 correlational。

3. **Action-conditioning 缺席。** 这篇是 observation-only forecasting，没 action。真正 robot control 需要 action-conditioned world model。把 ThinkJEPA 扩到 action-conditioned（比如把 action token 也塞进 FiLM）是自然的下一步，也是通往 planning 的必经之路。

4. **Rollout 误差曲线的形状。** Tab. 5 里 ThinkJEPA @32 = 0.111，V-JEPA @32 = 0.142。差 0.031 看着不大，但 baseline 已经在 0.14 量级，相对差 ~22%。更想知道的是：guidance 是把误差曲线整体下移，还是改变了曲线的斜率（即抑制 drift rate）？如果是后者，那 VLM guidance 在 100-step、1000-step rollout 上的价值会非线性放大。论文没给 >32 的数据，这是个遗憾。

5. **VLM 的 reasoning 真的在被用上吗？** 现在 prompt 是 task name + scene description。如果换成 random / wrong prompt，guidance 会不会退化？这个 sanity check 能证明 "thinking" 信号是不是真的在 drive 预测，还是只是给了一个 fixed prior。Supplementary 6.1 的 prompt-conditioned variant（ADE 0.069 vs 0.061）暗示 prompt 有用但不是主导，主导还是 visual thinking tokens。

6. **cross-dataset generalization。** 只在 EgoDex + EgoExo4D 上做，都是 egocentric manipulation。把 ThinkJEPA 拿到 third-person / outdoor / non-hand 场景，VLM 的 semantic guidance 是否还 transfer？这决定它是不是一个 "world model" 还是 "hand trajectory model"。

---

## 11. 一些值得记的 implementation 细节（Tab. 11）

- Backbone: V-JEPA-L (vit_large_rope), depth 24, dim 1024, Conv3d patch (kernel/stride = 2,16,16)；
- Predictor: dim 384, depth 12, heads 6, RoPE + 2 mask tokens；
- VLM: Qwen3-VL (Thinking) cached, token dim 2048, 8 cache clips, encoder length 480, AR length 15；
- Pyramid layers $\mathcal{L} = \{0,4,8,12,16,20,24,27\}$；
- Output: $32 \times 52 \times 3$（32 frames × 52 joints × xyz）；
- LR $10^{-3}$, predictor LR $10^{-4}$, batch 14 train / 6 eval, seed 42。

注意 predictor dim 384 远小于 backbone 1024——这是 JEPA 一贯的 "small predictor" 设计，逼 representation 自己承载信息，predictor 只学 dynamics。VLM guidance 注入到这个 384 维空间里，等于在"被压缩过的 dynamics 语言"上做 modulation。

---

## 12. 给你的 one-liner takeaway

ThinkJEPA = **V-JEPA2 的 latent rollout + Qwen3-VL 的多层 thinking trace 用 FiLM 调制 + dual-temporal sampling 让 dense dynamics 和 long-horizon semantics 各司其职**。它没让 VLM 干 VLM 不擅长的事（dense metric prediction），也没让 JEPA 干 JEPA 不擅长的事（semantic grounding），而是用一个轻量 modulation 通道把两者的强项缝起来。从 rollout 实验看，这种缝合在 long-horizon 上尤其值钱——这跟 "system 2 给 system 1 提供 long-range plan" 的直觉是完全一致的。

我个人最感兴趣的 extension 是把它推到 action-conditioned + RL planning，让 VLM 的 thinking 真正参与 decision，而不只是 trajectory forecasting。那时候 FiLM 的 $(\gamma_\ell, \beta_\ell)$ 就不只是 modulation 参数，而是 "thought → action prior" 的桥梁。这是个值得跟进的方向。

---

### References (with links)

- V-JEPA2 (Assran et al., 2025): https://arxiv.org/abs/2506.09985
- Qwen3-VL Technical Report: https://arxiv.org/abs/2511.21631
- I-JEPA (Assran et al., 2023): https://arxiv.org/abs/2301.08243
- VL-JEPA: https://arxiv.org/abs/2512.10942
- EgoDex: https://arxiv.org/abs/2505.11709
- EgoExo4D: https://arxiv.org/abs/2311.18258
- FiLM (Perez et al., 2018): https://arxiv.org/abs/1709.07871
- Dreamer (Hafner et al., 2020): https://arxiv.org/abs/1912.01603
- World Models (Ha & Schmidhuber): https://arxiv.org/abs/1803.10122
- DDPM: https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747
- CLIP: https://arxiv.org/abs/2103.00020
- LLaVA: https://arxiv.org/abs/2304.08485
- RT-2: https://arxiv.org/abs/2307.15818
- SayCan: https://arxiv.org/abs/2204.01691
- Genie: https://arxiv.org/abs/2402.19427
- LeCun "A path towards autonomous machine intelligence": https://openreview.net/pdf?id=BZ5a1r-kVsf
