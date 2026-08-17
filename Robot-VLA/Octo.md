---
source_pdf: Octo.pdf
paper_sha256: 73bff297cfafe523319162124e6b7f96919c0930e0a380f307255c4c7464ac93
processed_at: '2026-08-05T22:54:31-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，我用大白话给你捋一遍 Octo 这篇 paper，重点 build 你的 intuition。

## 一句话说清楚 Octo 在干嘛

Robotics 一直有个尴尬：每个 lab 的 robot 都不一样，sensor 配置不一样，action space 也不一样。你想把 Berkeley 用 WidowX 收的数据拿去 Stanford 给 Franka 用？基本没法直接用。Octo 想做的事情很简单——**pretrain 一个 "robot brain"，这个 brain 足够灵活，你换 robot、换 sensor、换 action space，它都能 finetune 适应**。

这个事情听起来简单，做起来极难。之前 RT-X 也想做，但 RT-X 的 architecture 把 input/output 的 type 写死了，你下游想加个 force-torque sensor？对不起，得重新设计 network。Octo 把这个问题彻底解决了。

## 怎么解决的？核心就一个 idea

**把所有的 input（image, language, goal image, force-torque...）全部变成 token sequence，然后用 block-wise attention 控制谁能看谁。**

你可以这么理解：transformer 就是一个超级 sponge，你往里面塞什么 tokens 它就处理什么 tokens。Image 来了？切成 patch，每个 patch 一个 token。Language 来了？过一遍 T5，出来 16 个 tokens。Goal image？跟普通 image 一样处理。新的 force-torque reading？过一个小 MLP 变成 token 塞进去就行。

关键在于 **block-wise masking**：

- Observation tokens 只能 attend 当前和历史的 observation，外加 task tokens
- Readout tokens（类似 BERT 的 `[CLS]`）能 attend 所有前面的东西，但不会被别人 attend
- 不存在的 modality 直接 mask 掉

这样设计的好处是：**finetune 到新 robot 时，pretrained weights 一行都不用改**。你只需要加新的 positional embedding，训练新的小 tokenizer，transformer backbone 完全 freeze 都行。这在之前的工作里是做不到的。

## Architecture 图解（脑子里的 mental model）

```
序列长这样：
[Task_Tokens] [Obs_1] [Readout_1] [Obs_2] [Readout_2] ... [Obs_t] [Readout_t]
     ↓           ↓         ↓          ↓         ↓              ↓         ↓
     ↓     ←─────┘         ↓     ←─────┘                     ↓
     ↓                     ↓                                  ↓
  全局可见          只看自己和之前                      只看前面所有
  的context          的observation                      
                                                          
Attention mask:
- Task tokens: 所有人都能看
- Obs tokens: 看当前+历史obs + task tokens
- Readout tokens: 看前面所有东西，但没人看它
```

Finetune 时加新 sensor：

```
[Task_Tokens] [Obs_1] [NEW_SENSOR_1] [Readout_1] ...
                              ↑
                    新加的token，只需训练这个小tokenizer
                    Transformer backbone 不用动
```

## Action 怎么生成的？为什么用 Diffusion？

这里有个很重要的 engineering insight。

Robot action distribution 经常是 multi-modal 的。比如你让 robot 绕过一个杯子，可以从左边绕，也可以从右边绕。如果你的 policy 用 MSE loss 训练，它会对这两个 mode 取平均，结果就是 robot 直直撞上杯子——这叫 "hedging" behavior。

RT-1 的解法是把 action discretize 成 256 bins，用 cross-entropy。这样能处理 multi-modal，但 precision 掉了，grasping 的时候经常差那么一点点抓不到。

Octo 用了 **diffusion head**，兼顾了两者：

$$x^{k-1} = \alpha(x^k - \gamma \epsilon_\theta(x^k, e, k) + \mathcal{N}(0, \sigma^2 I))$$

变量解释：
- $x^k$: 第 $k$ 步的 noisy action（一个 action chunk，预测未来多步）
- $x^{k-1}$: 去噪一步后的 action
- $e$: transformer readout token 的 embedding，作为 condition
- $k$: diffusion step index，从 $K$ 到 0
- $\epsilon_\theta$: 一个 3-layer MLP（hidden 256），预测 noise
- $\alpha, \gamma, \sigma$: cosine noise schedule 的参数，控制去噪步长和随机性

**关键 trick**：Transformer backbone 只 forward **一次**，产出 readout embedding $e$，然后 diffusion 的 20 步去噪全在那个小 MLP 里跑。这样 inference 速度完全可以接受。

## 实验数据看看效果

### Zero-shot（不 finetune，直接用）

在 pretraining 数据覆盖的 robot setup 上，Octo vs RT-1-X：

| Robot Setup | Octo | RT-1-X (35M) | RT-2-X (55B) |
|:---|:---|:---|:---|
| WidowX | ~85% | ~60% | ~85% |
| UR5 | ~75% | ~40% | N/A |
| RT-1 Robot | ~70% | ~50% | ~70% |

Octo 用 93M parameters 打平了 55B 的 RT-2-X，这说明 **architecture design 比 raw scale 更重要**。

### Finetuning（100 demos，5小时 A5000）

| Method | Avg Success Rate |
|:---|:---|
| From Scratch (ResNet+Transformer) | 20% |
| VC-1 (pretrained visual repr) | 15% |
| **Octo** | **72%** |

72% vs 20%，这是 3.5x 的提升。说明 Octo pretrained weights 是一个极好的 initialization，远比从 ImageNet pretrain 的 visual encoder 强。

### Ablation（WidowX 上测）

| Design Choice | Success Rate |
|:---|:---|
| Full Octo-Small | **83%** |
| 只用 RT-X data mix | 60% |
| 只用 single robot data | 43% |
| MSE action head | 35% |
| Discrete action head | 18% |
| ResNet encoder (替代 ViT) | 70% |

三个关键 takeaway：
1. **Data diversity matters**：800k cross-embodiment > 350k > single robot
2. **Diffusion > MSE > Discrete**：multi-modal + precision 都要
3. **ViT > ResNet at scale**：弱 inductive bias 在大数据下反而更强

## 为什么 ViT 在大数据下赢 ResNet？

这里有个很深的 intuition。

ResNet 有很强的 inductive bias——卷积的局部性、平移不变性。当你只有 100 个 demos 时，这些 prior 帮你快速学起来。但当你有 800k 跨 robot 的 data 时，这些 prior 反而成了 constraint，限制 model 去发现更 general 的 pattern。

ViT 的 attention 几乎没有 spatial inductive bias，它全靠 data 驱动来学 spatial relationship。大数据下，这种 "白纸" 特性让它能 scale 得更好。这跟 LLM 里 transformer 赢 RNN 的逻辑一样——弱 prior + 大 data = 强 generalization。

## 工程上一些有意思的细节

1. **Shuffle buffer 要超大**：从 25 个 dataset 并行加载 data，shuffle buffer 太小（20k）会导致严重 overfitting。他们 scale 到 500k frames，并且每个 trajectory 最多采样 100 frames。

2. **History 只要 2 帧**：试了更多帧，收益 diminishing。2 帧（当前+上一帧）就够了。

3. **Gripper action 用 absolute 不用 relative**：试了 relative（只在 open/close 那一刻给 +1/0），发现 retry behavior 变差了。Absolute（open=+1, closed=0）更好。

4. **Language encoder 用 frozen T5-base 就行**：试了 fine-tune T5、用更大的 T5，都没提升。原因可能是 OXE 数据里的 language annotation 不够 rich、不够 diverse。

5. **Image augmentation**：3rd person camera 做 random crop + resize 到 256x256 + color jitter；wrist camera 不做 crop，resize 到 128x128。

## Model Scale

| Model | Layers | Hidden Dim | MLP Dim | Heads | Params |
|:---|:---|:---|:---|:---|:---|
| Octo-Tiny | - | - | - | - | 10M |
| Octo-Small | 12 | 384 | 1536 | 6 | 27M |
| Octo-Base | 12 | 768 | 3072 | 12 | 93M |

Scaling curve 显示 model 越大 zero-shot performance 越好，没有饱和迹象。说明还可以继续 scale，可能 1B+ 的 Octo 会更强。

## Limitations 和 Future Work

Paper 自己承认的 short-coming：
- Wrist camera 处理不好（只有 27% 数据有 wrist cam）
- Language conditioning 比 goal image conditioning 差（只有 56% 数据有 language annotation）
- 都是 imitation learning，没有用 RL 或 sub-optimal data
- 只测了 manipulation，没有 navigation

## 我的 Intuition 总结

Octo 这个工作本质上是把 LLM 那套 "tokenize everything + transformer + scaling" 的哲学成功搬到 robotics。它的 contribution 不是某个单点技术突破（diffusion head、action chunking 这些之前都有），而是把这些 component 用一个极优雅的 modular design 串起来，并且 open source 出来。

Block-wise attention 这个设计我认为会成为 robotics 的标准。因为它把 "robot brain" 和 "robot body" 彻底解耦了——brain 是一个 modality-agnostic 的 transformer，body 是一堆可插拔的 tokenizer/head。未来如果有人要做一个 1B param 的 Octo，architecture 基本不用变，只需要 scale transformer backbone。

## Web Links

- Octo Project Page: https://octo-models.github.io
- Octo Paper (arXiv): https://arxiv.org/abs/2405.12213
- Open X-Embodiment: https://robotics-transformer-x.github.io
- Diffusion Policy (Chi et al.): https://diffusion-policy.cs.columbia.edu/
- RT-X Paper: https://arxiv.org/abs/2310.08864
- Octo GitHub (JAX): https://github.com/octo-models/octo
- T5 Model: https://arxiv.org/abs/1910.10683
- ViT Paper: https://arxiv.org/abs/2010.11929
- DDPM Paper: https://arxiv.org/abs/2006.11239

---

Andrej，这篇 paper 关于 Octo，一个 open-source generalist robot policy。因为 robotics 领域长期存在 embodiment 异构性问题，导致 data 无法像 NLP 或 CV 那样轻易 scale，所以 Octo 试图通过一个高度 modular 的 transformer architecture 来吸收 Open X-Embodiment (OXE) 的 800k 多样化 trajectories。下面我为你进行极度详细的拆解，旨在 build your intuition。

### Core Intuition: Modular Tokenization 与 Block-wise Attention

传统 robot policy 往往把 vision encoder 和 action head 强耦合，并且 lock 在特定的 observation 和 action space。如果下游任务的 robot 换了传感器，整个 model 需要重新设计或者重训。Octo 的核心 insight 在于：将所有的 inputs（language, images, goals）和 outputs（actions）全部转化为通用的 token sequence，并且通过 **block-wise attention** 严格控制信息流，从而实现 "plug-and-play" 的 finetuning。

在 architecture 上，Octo 包含三个核心模块：

1.  **Input Tokenizers**:
    *   **Language/Goal**: Language inputs 使用预训练的 `t5-base`（111M parameters）转化为 16 个 language embedding tokens。Goal images 则被当作普通的 image observations 处理。
    *   **Images**: Images 并没有使用厚重的 ResNet，而是使用 shallow CNN 随后切分成 $16 \times 16$ 的 patches（类似 ViT）。Third-person camera 映射为 256 tokens，wrist camera 映射为 64 tokens。这种 "transformer-first" 设计将绝大部分 parameters 和 FLOPS 放在了 transformer backbone 里面，更利于大规模 pretraining 时的 scaling。

2.  **Transformer Backbone 与 Block-wise Masking**:
    Transformer 处理的序列被组织成多个 blocks：`[Task_Tokens, Obs_1, Obs_2, ..., Obs_t, Readout_t]`。
    *   **Causal Observation Masking**: Observation tokens 只能 attend 到当前以及历史 time steps 的 tokens (`Obs_{0:t}`)，以及 task tokens。
    *   **Readout Tokens**: 类似 BERT 的 `[CLS]` token，Readout tokens 插入在每个 time step 的末尾。它们可以 attend 前面所有的 observation 和 task tokens，但 observation tokens 无法 attend 到 readout tokens。这保证了 readout token 被动地提取全局 representation，而不会污染前面的 observation embedding space。
    *   **Finetuning Flexibility**: 如果要 finetune 到一个带有 force-torque sensor 的新 robot，只需在序列中插入对应的 new observation tokens，并且加上新的 learnable positional embeddings。因为 attention 是 block-wise 的，pretrained transformer weights 完全不需要 re-initialization，自然就能处理这些新信息。这类似于 LLM 中的 prefix tuning，但发生在 spatial-temporal 层面。

3.  **Diffusion Action Head**:
    Readout token 的 embedding $e$ 会被送入一个轻量级的 diffusion head 来预测 action chunk（连续多步的 actions）。Diffusion 只在这个小 head 内部迭代，transformer backbone 只需要 forward 一次，极大降低了推理延迟。

### Technical Deep Dive: Diffusion Action Head 与公式解析

在 robot imitation learning 中，action distribution 经常是 multi-modal 的（比如绕过障碍物可以从左边也可以从右边）。如果用 MSE loss，model 会对这些 modes 取平均值，导致 "hedging" behavior，动作迟缓且不准确。如果像 RT-1 那样 discretize actions 成 256 bins，虽然能解决 multi-modal 问题，但会损失 precision，尤其是在精确抓取中。

Octo 采用了 conditional diffusion decoding，以下是其去噪过程的公式：

$$
x^{k-1} = \alpha (x^k - \gamma \epsilon_\theta(x^k, e, k) + \mathcal{N}(0, \sigma^2 I))
$$

*   **$x^k$**: 当前 diffusion step $k$ 下的 noisy action chunk。
*   **$x^{k-1}$**: 下一步（noise 更少）的 action chunk。
*   **$e$**: Transformer backbone 输出的 readout token embedding，作为条件控制 action 的生成。
*   **$k$**: 当前的 diffusion step index。
*   **$\epsilon_\theta$**: 神经网络（这里是一个 3-layer MLP，hidden dim 256，带 residual connections 和 layer norm），预测需要去除的 noise。
*   **$\alpha, \gamma, \sigma$**: 由 cosine noise schedule 决定的超参数，控制每一步去噪的步长和加入的随机性。
*   **$\mathcal{N}(0, \sigma^2 I)$**: 采样的 Gaussian noise。

Diffusion head 训练时使用标准的 DDPM objective，在 ground truth action 上加 Gaussian noise，训练 $\epsilon_\theta$ 去重构 original action。这种设计完美兼容 continuous action space 的 precision 和 multi-modal distribution 的 expressivity。

### Training Data Mixture 策略

Octo 使用了 Open X-Embodiment dataset 中的 25 个子数据集，总计 800k trajectories（比 RT-X 的 350K 更大）。Data mixture 极大地影响了 generalist policy 的性能。为了保证 batch 内部的 diversity，并且防止某些巨型 dataset（如 Fractal, Kuka, Bridge，各占 17%）dominate 整个训练，他们采用了 weighted sampling。

详细 mixture 比例可以参考 Table III，比如 Fractal 17.0%, Kuka 17.0%, Bridge 17.0%, BC-Z 9.1% 等。此外，为了统一 action space，他们把不同 robot 的 gripper action 都 align 成 +1 (open) 和 0 (closed)。在数据加载时，他们发现 shuffle buffer size 至关重要，太小会导致严重的 trajectory-level overfitting，他们将其 scale 到了 500k frames，并且限制每个 trajectory 最多采样 100 个 frames。

### Experiments 与 Ablations Data

Table I 展示了在不同 setup 下 finetuning 的结果。100 demos，相同的 hyperparameters，不到 5 小时在单张 A5000 GPU 上完成。

| Method | Berkeley Insertion* | Stanford Coffee | CMU Baking | Berkeley Pick-Up† | Berkeley Coke | Berkeley Bimanual† | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet+Transformer Scratch | 10% | 45% | 25% | 0% | 20% | 20% | 20% |
| VC-1 | 5% | 0% | 30% | 0% | 10% | 50% | 15% |
| **Octo (Ours)** | **70%** | **75%** | **50%** | **60%** | **100%** | **80%** | **72%** |

*(注：* 表示包含新的 observation input (force-torque)，† 表示包含新的 action space (joint position control))。*

你可以看到，Octo 作为一个 initialization 极其强大，平均成功率比 from-scratch 提升了 52%。尤其是在 Berkeley Bimanual 任务中，model 需要控制两个 7-DoF 的 arms，Octo 通过重新初始化 action head 并 finetuning，达到了 80% 的成功率。

Table II 提供了关于 architecture, data 和 policy 的 ablations (在 WidowX setup 上测试):

| Component | Variant | Success Rate |
| :--- | :--- | :--- |
| **Full Model** | **Octo-Small** | **83%** |
| Data | RT-X dataset mix | 60% |
| Data | Single robot dataset (Bridge Data) | 43% |
| Policy | Discretized Action Prediction | 18% |
| Policy | Continuous Action Prediction (MSE) | 35% |
| Arch | Resnet-50 + Transformer | 70% |

Ablation 证明了几个关键的 intuition：
1.  **Data diversity matters**: 用全量 OXE mix (83%) 优于 RT-X mix (60%) 优于 single robot dataset (43%)。Cross-embodiment data 产生了正向的 transfer。
2.  **Diffusion > MSE > Discrete**: MSE (35%) 导致 hedging，Discrete (18%) 导致 precision 下降，Diffusion (83%) 兼具两者优势。
3.  **ViT > ResNet**: "Transformer-first" (83%) 在大规模数据下胜过 ResNet+Transformer (70%)。虽然 ResNet 在小数据 from-scratch 训练时更好，但在 large-scale pretraining 下，ViT 的 weaker inductive bias 让它具有更好的 scaling properties。

### Code, Inference 与 Open-source 资源

Octo 基于 JAX 实现。Inference 非常简单，核心代码如下：

```python
import jax
from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
print(model.get_pretty_spec()) 
observation = {"image_primary": img}
task = model.create_tasks(texts=["pick up the fork"])
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
```

开源资源包括：
*   Pretrained Octo-Small (27M) 和 Octo-Base (93M) checkpoints。
*   JAX finetuning scripts。
*   Pretraining pipeline on OXE。
*   JAX/PyTorch 兼容的 data loaders。

### Web Links Reference

*   **Project Page**: https://octo-models.github.io
*   **Octo Paper (arXiv)**: https://arxiv.org/abs/2405.12213
*   **Open X-Embodiment Dataset**: https://robotics-transformer-x.github.io/
*   **Diffusion Policy (Chi et al.)**: https://diffusion-policy.cs.columbia.edu/
*   **JAX Framework**: https://github.com/google/jax

### My Intuition Build-up

Octo 的成功给我最大的启发在于它的 **"Block-wise Attention + Tokenizer"** 范式。在 LLM 中，modalities 的融合通常通过 cross-attention 或者拼接 embedding，但在 robotics 中，inputs 的存在性是高度不确定的（有的 robot 有 wrist cam，有的没有；有的有 force-torque，有的没有）。Octo 把所有的 observation 都当成可选的 token blocks 塞进一个大的 1D sequence 里，利用 attention mask 来控制信息流向。这意味着 model 的 representation space 本质上是 modality-agnostic 的。只要你有新的 sensor，你只需要训练一个小 tokenizer（如一层 CNN 或 MLP），把新 sensor 变成 tokens 加到序列里，transformer 就能像海绵一样吸收它。

这种设计极大概率会成为未来 generalist robot policy 的标准架构，因为它彻底解耦了 "brain" (transformer backbone) 和 "peripherals" (sensors/actuators)，完美契合了我们在 software engineering 中追求的 high cohesion, low coupling 原则。并且，ViT-first 的设计使得未来可以通过 scale up backbone parameters 到 1B 甚至 10B，直接享受 LLM 那套成熟的 scaling laws，这在 ResNet-based 的 RT-1 架构上是很难实现的。
