---
source_pdf: vla-eval A Unified Evaluation Harness for.pdf
paper_sha256: 8cc9983f82775598ad4a0c0144020437a019df14a607a70ca72e1c0fe87b3bf0
processed_at: '2026-08-13T02:44:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 vla-eval 这篇 paper

## 一句话总结

VLA 这个领域现在一团乱：每个人发 paper 都说自己 SOTA，但你想复现？想横向比较？基本没戏。这群人说："行了，我来搞个统一考试中心"，然后真搞出来了。

---

## 现状到底有多烂

想象一下你是 VLA researcher，你训了个新 model，想在 LIBERO 上跑一下。你去 LIBERO 的 repo 看——OK 需要 Python 3.8 + robosuite。装好了，跑通了。

然后你想顺便在 ManiSkill2 上也跑一下。哦，人家要 Python 3.10 + SAPIEN。你之前的 environment 直接挂了。

再来个 CALVIN？PyBullet，又一坨 dependency。

**三个 benchmark 跑下来，你的电脑上堆了三个互相打架的 Python 环境**，每个还自带一堆 scene files、textures、robot URDF，安装过程全靠 README 里那句 "hopefully this works"。

更恶心的是：你跑完了，出了分数，想跟别人的 paper 比。结果发现——

- 人家没写用了多少 seeds
- 人家没写跑了多少 episodes  
- 人家没写 observation 怎么 normalize 的
- 人家没写 physics settling 几步

**你根本不知道你跑的跟人家跑的是不是同一个东西。**

最后，一个 LIBERO evaluation 2000 episodes，sequential 跑要 **14 个小时**。你想做 ablation study 比较 5 个 variants？一个礼拜就过去了。

这就是 VLA evaluation 的现状。Paper 里原话叫 "fragmented"，我说这就是 **一坨**。

---

## 他们搞了什么

简单说就两件事：

### 事情一：搞了个统一框架

设计思路特别干净——**把 model 和 benchmark 拆开**。

```
Model Server (你的模型)  ←──WebSocket──→  Benchmark Docker (考试环境)
        ↑                              ↑
   在 host 上跑                    在 container 里跑
   自己的 Python 环境               自己的 Docker image
```

你只要写一个 `predict(obs)` 函数，你的 model 就能跑所有 benchmark。Benchmark 那边只要实现 4 个方法（reset, step, make_obs, get_step_result），就能被所有 model 跑。

这就是 **N×M 变成 N+M** 的经典操作。10 个 model × 13 个 benchmark，以前要写 130 套 glue code，现在只要写 10+13=23 个 adapter。

通讯用 WebSocket + msgpack。为什么？

- WebSocket 是 persistent connection，不像 HTTP 每次要 handshake。VLA 每个 episode 可能几百步 round-trip，省掉 handshake 很重要。
- msgpack 是 binary serialization，比 JSON 紧凑很多。你传一张 224×224 的 image，JSON 要 base64 编码再传，msgpack 直接 binary 过去。

### 事情二：搞了个 leaderboard

他们用 Claude Code（Anthropic 的 AI coding agent）配合 MCP tools 自动扫了 **2685 篇 paper**，从里面提取 evaluation results，搞了个 leaderboard：

🔗 https://allenai.github.io/vla-evaluation-harness/leaderboard

657 条 results，17 个 benchmark，509+ 个 model configuration。

---

## 最震撼的发现：81% 的 model 只在 1 个 benchmark 上测过

这个数据真的让我震惊。看 Fig. 5：

| 在几个 benchmark 上测过 | 占比 |
|----------------------|------|
| 1 个 | **81%** |
| 2 个 | ~12% |
| 3 个 | ~6% |
| 5 个以上 | **0.6%** |

大家都在喊 "generalist robot"，结果 **81% 的 model 只在一个考试里验证过**。你号称 generalist，但你只参加了一门考试，这叫什么 generalist？

这说明 cross-benchmark evaluation 现在太贵了、太麻烦了，没人愿意做。vla-eval 就是来解决这个问题的——让 cross-benchmark evaluation 变得便宜到人人都能做。

---

## 47× 加速是怎么做到的

这个值得细说。Sequential 跑 2000 episodes 要 14 小时，他们搞到 18 分钟，**47 倍加速**。

核心思路：**两条线并行**。

### 线 1：Environment 并行（Demand side）

把 2000 个 episodes 切成 50 份（shards），每份 40 episodes，丢到 50 个 Docker container 里并行跑。

$$\lambda(N) = \text{N 个 shards 的总 throughput}$$

实测：
- N=1（1 个 shard）：11.2 obs/s
- N=50（50 个 shards）：364.6 obs/s
- 加速 **32.6×**

为什么不是 50×？因为 Docker overhead、CPU contention、GPU rendering 抢资源。

### 线 2：Model inference 并行（Supply side）

Model server 那边用 batch inference。同时收 16 个 observation request，一起 forward pass。

$$\mu(B) = \text{batch size B 时的 model throughput}$$

实测：
- B=1：165.2 obs/s
- B=16：468.2 obs/s
- 加速 **2.8×**

为什么不是 16×？因为 VLA model 是 autoregressive generation，batch 内不同 sequence 长度不同，有 padding waste；而且 KV cache 随 batch size 线性增长，VRAM 有限。

### 怎么选 operating point

这是个 queuing theory 问题。Model server 是个 single server，N 个 environment shards 往它发 request。如果 arrival rate 超过 service rate，queue 就无限增长。

稳定性条件：
$$\lambda(N) < 0.8 \cdot \mu(B^*)$$

- $\lambda(N)$：environment 总 throughput（demand）
- $\mu(B^*)$：model server throughput at optimal batch size（supply）
- $0.8$：留 20% headroom，防止 burst 和 estimation error

他们的 operating point：
- N=50, B=16
- $\lambda(50) = 364.6$
- $0.8 \times \mu(16) = 0.8 \times 468.2 = 374.6$
- $364.6 < 374.6$ ✓ 稳定

利用率 364.6/468.2 = **77.9%**，留了 22% headroom。

### 不同 benchmark 瓶颈不同

| Benchmark | 瓶颈在哪 | 加速倍数 | 为什么 |
|-----------|---------|---------|--------|
| LIBERO | Model (GPU) | 47× | GPU 是瓶颈，batch inference 帮助大 |
| CALVIN | PyBullet CPU rendering | 16× | CPU-bound，加 shard 帮助有限 |
| SimplerEnv | SAPIEN GPU rendering | 12× | 每个 container 独占 GPU，加 shard 也没用 |

**Takeaway：没有 universal 最优策略，得根据 bottleneck 选 parallelism 方案。**

---

## 最有价值的发现：两个隐藏陷阱

他们拿 DB-CogACT 在三个 benchmark 上做 reproduction audit，结果发现——即使你自以为复现成功了，也可能被两个坑默默搞死。

### 坑 1：SimplerEnv 的 terminated flag 在骗你

Gymnasium API 有两个 flag：
- `terminated`：episode 自然结束
- `truncated`：达到 max steps 被截断

你看到 `terminated=True` 就停了对吧？

**但 SimplerEnv 的 terminated 实际意思是 "短暂成功了一下"**。比如 block 刚好 stacked 了一瞬间，terminated 就 True 了。但 robot 之后可能还会碰倒它。

如果你在 terminated=True 就停，你的 score 会 **虚高**。正确做法是一直跑到 `truncated=True`。

这个信息 **不在任何文档里**，你得去读 simulator source code 才知道。

### 坑 2：CALVIN 藏了 normalization stats

CALVIN 需要对 observation 做 normalization：39 个 dimension（15 robot-state + 24 scene-state）的 mean 和 std。

这些 stats 从 training dataset 算出来，**但 official evaluation documentation 里根本没写**。你得 trace 回 training codebase 去找这些数。

用错 stats 的话，observation preprocessing 就错了，model performance 会默默变差，你都不知道为什么。

**这两个坑的本质是：benchmark 的 critical implementation details 藏在 source code 里，paper 和文档都不会告诉你。**

---

## Reproduction 结果其实还不错

虽然有坑，但他们复现的结果跟 published values 差距在 ±3% 以内：

| Benchmark | 他们复现的 | Paper 报的 | 差距 |
|-----------|-----------|-----------|------|
| LIBERO Spatial | 95.2% | 93.8% | +1.4 |
| LIBERO Object | 98.6% | 97.8% | +0.8 |
| LIBERO Goal | 95.2% | 96.2% | -1.0 |
| LIBERO Long-Horizon | 89.6% | 91.8% | -2.2 |
| CALVIN Avg Len | 4.051 | 4.063 | -0.012 |
| SimplerEnv Avg SR | 72.22% | 69.45% | +2.77 |

注意差距方向不一致——有的高有的低，说明不是 systematic bias，而是 stochastic variation + 隐藏配置差异的混合效果。

---

## 工程上几个聪明的设计

### 1. PEP 723 解决 dependency 冲突

Model server 端，每个 model 的依赖写在脚本顶部的 comment 里：

```python
# /// script
# dependencies = ["transformers==4.40.1", "torch>=2.0"]
# ///
```

然后用 `uv run` 启动，uv 会自动创建 isolated venv。CogACT 要 `transformers==4.40.1`，X-VLA 要 `transformers>=4.44`，两个 model server 各跑各的，互不干扰。

参考：https://peps.python.org/pep-0723/

### 2. Action chunking 的 ensemble 策略

现代 VLA model（π0、CogACT）一次预测未来 H 步 action。但当前该执行哪个 action？有三种策略：

- **newest**：用最新一次预测的。响应快，但 jitter。
- **average**：把所有覆盖当前时刻的 prediction 求平均。平滑，但 lag。
- **EMA**：$a_t = \alpha \cdot \hat{a}_t + (1-\alpha) \cdot a_{t-1}$，其中 $\alpha$ 是 smoothing factor。折中方案。

vla-eval 把这个选择 expose 给用户，不同 benchmark 可以用不同策略。

### 3. 两条命令搞定一切

```bash
vla-eval serve --config model_server.yaml   # 启动模型
vla-eval run --config benchmark.yaml         # 跑 benchmark
```

Config 文件就是完整的 provenance——Docker image tag、seeds、episode counts 全在里面。任何人拿到这个 config 都能完全复现你的结果。

---

## 这篇 paper 的深层意义

### VLA 领域的 "lm-evaluation-harness 时刻"

LLM 领域 2020-2022 也经历过这种 fragmentation。每个 model repo 自己写 evaluation script，互相不一致。然后 EleutherAI 搞了 lm-evaluation-harness，大家终于有了统一标准。

VLA 现在就在这个转折点上。vla-eval 可能就是 VLA 领域的 lm-evaluation-harness。

参考：https://github.com/EleutherAI/lm-evaluation-harness

### AI agent 做 research curation

他们用 Claude Code + MCP 自动扫了 2685 篇 paper。这是未来 trend——literature review 和 meta-analysis 会越来越自动化。人只需要 review AI 提取的结果，处理 edge case。

### Infrastructure 就是 contribution

这篇 paper 没提出新 model、没提出新 algorithm。它搞的是 infrastructure。但好的 infrastructure 能 unlock 整个社区的进展。47× 加速意味着 researcher 可以做更多 ablation、更多 cross-benchmark comparison，最终推动更好的 model 出来。

---

## 局限性（他们自己承认的）

1. **只 audit 了一个 model**——DB-CogACT 在三个 benchmark 上。应该 audit 更多 model × benchmark 的 matrix。
2. **只支持 simulation**——real robot 评估不在范围内。sim-to-real gap 是 VLA 的核心问题，这个框架帮不了。
3. **Leaderboard 的数据从 paper 提取**——没有全部独立 re-verify。Paper 里的数字可能本身就是错的。
4. **只有 success rate**——缺少 motion quality、efficiency、safety 等维度。

---

## 最后的 takeaway

如果你做 VLA research，这篇 paper 告诉你三件事：

1. **别自己写 evaluation script 了**，用 vla-eval，省下来的时间做 research。
2. **别只在一个 benchmark 上测**，那不叫 generalist。用 vla-eval 在多个 benchmark 上跑，成本低得多。
3. **Reproducibility 不只是写清 seeds**，还有 termination semantics、normalization stats 这些藏在 source code 里的东西。vla-eval 通过 Docker + config 把这些全锁定了。

链接汇总：
- Paper repo: https://github.com/allenai/vla-evaluation-harness
- Leaderboard: https://allenai.github.io/vla-evaluation-harness/leaderboard
- Docker images: https://github.com/allenai/vla-evaluation-harness/pkgs/container/vla-evaluation-harness

希望这个人话版本够清楚，Andrej！

---

# vla-eval: Unified Evaluation Harness for VLA Models 技术讲解

## 1. Paper 的核心动机和问题域

VLA (Vision-Language-Action) models 在 robotic manipulation 领域正在爆炸式发展——OpenVLA、CogACT、π0、GR00T N1、X-VLA 等模型不断涌现，每个都声称在多个 simulation benchmarks 上达到 SOTA。但是 evaluation ecosystem 处于严重的碎片化状态，存在四个痛点：

**Pain Point 1: Duplicated Effort**
每个 model repository 独立维护一整套 benchmark-specific scripts。当上游 benchmark interface 变更时，下游 model repos 静默 diverge，导致同一个 benchmark 在不同 paper 中实际跑的是不同版本的东西。

**Pain Point 2: Dependency Hell**
这是一个非常实际的问题，让我列出三个 benchmark 的环境需求矛盾：
- LIBERO → Python 3.8 + robosuite
- ManiSkill2 → Python 3.10 + SAPIEN  
- CALVIN → PyBullet

没有单一的 Python 环境能同时满足这三个。更糟糕的是，每个 benchmark 还需要自己的 asset setup（scene files, textures, robot URDF descriptions），这些安装过程是 ad-hoc 的，没有标准化。

**Pain Point 3: Underspecified Protocols**
papers 经常省略：seeds、episode counts、normalization statistics、physics-settling steps。这些遗漏使得独立 reproduction 几乎不可能。

**Pain Point 4: Slow Evaluation**
单个 LIBERO evaluation（2000 episodes）sequential 跑 ~14 小时。如果你要比较 5 个 model variants × 4 个 benchmarks = 20 次评估，那就是 ~280 小时，routine comparative studies 根本不现实。

vla-eval 的设计哲学直接借鉴自 lm-evaluation-harness（EleutherAI 的 LLM 评估框架）：**"Models integrate once, benchmarks integrate once, the full cross-evaluation matrix works automatically."** 这是一个 N×M → N+M 的复杂度降低。

参考链接：
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- vla-eval GitHub: https://github.com/allenai/vla-evaluation-harness
- vla-eval Leaderboard: https://allenai.github.io/vla-evaluation-harness/leaderboard

---

## 2. System Architecture 深度解析

### 2.1 Client-Server 解耦设计

核心设计是 **decouple model inference from benchmark execution**。这是一个非常关键的工程决策，解决了很多根本性问题。

```
┌─────────────────────────────────────────────────────────────┐
│  Host Machine                                              │
│  ┌──────────────────────┐                                  │
│  │  Model Server         │  ← runs on host (GPU accessible) │
│  │  (Python + uv env)    │  ← PEP 723 inline metadata       │
│  │  WebSocket server     │                                  │
│  └──────────┬───────────┘                                  │
│             │ WebSocket + msgpack                           │
│             │ (binary serialization)                        │
│  ┌──────────▼───────────┐  ┌──────────────────────┐         │
│  │ SyncEpisodeRunner    │  │ Docker Container 1    │ ← shard 1
│  │ (orchestrator)       │  │ (Benchmark env)       │         │
│  │ observe→act→step    │  │ GPU (optional)        │         │
│  │ loop                 │  └──────────────────────┘         │
│  └──────────┬───────────┘  ┌──────────────────────┐         │
│             │              │ Docker Container 2    │ ← shard 2
│             │              │ (Benchmark env)       │         │
│             │              └──────────────────────┘         │
│             │              ┌──────────────────────┐         │
│             │              │ Docker Container N    │ ← shard N
│             │              │ (Benchmark env)       │         │
│             │              └──────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

为什么用 WebSocket + msgpack 而不是 REST + JSON？
- **WebSocket**：persistent connection，避免每步 HTTP handshake overhead。VLA evaluation 每个 episode 可能需要几百步，high-frequency round-trip 是常态。
- **msgpack**：binary serialization，比 JSON 紧凑得多。Observation 中包含 image（高维数组），msgpack 能显著减少 serialization/deserialization 开销。

每条消息的结构：
- `type`：observation | action | episode_start | episode_end
- `payload`：benchmark-specific data（灵活，不强制 schema）
- `sequence number`：用于 ordering 和 debugging
- `timestamp`：latency profiling

### 2.2 Model Server 层次设计

这是 OOP 继承的一个优雅设计：

**Layer 1: `ModelServer` ABC**
- Fully asynchronous interface
- 用于 advanced use cases（streaming、multi-modal async coordination）

**Layer 2: `PredictModelServer`**
- 提供 blocking `predict(obs, ctx)` 方法
- 典型集成只需 ~50 行代码（Listing 1 展示 OpenVLA 完整集成）
- 内置功能：
  - **Action chunking**：很多 VLA 模型（如 π0、CogACT）一次预测未来 T 步 actions，而非单步
  - **Ensemble strategies**：如何从 chunk 中选择当前 action
    - `newest`：使用最新预测的 action（高适应性，可能 jitter）
    - `average`：对 chunk 内所有 prediction 求平均（平滑，可能 lag）
    - `EMA`：指数加权移动平均（balance）
  - **Batched inference** via `max_batch_size`：关键优化

让我解析 Listing 1 的 OpenVLA 集成代码：

```python
class OpenVLAServer(PredictModelServer):
    def __init__(self, model_path, **kw):
        super().__init__(**kw)
        self.model_path = model_path
        self._model = self._proc = None  # lazy loading
    
    def _load_model(self):
        if self._model is not None:
            return  # idempotent
        self._proc = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq \
            .from_pretrained(self.model_path,
                torch_dtype=torch.bfloat16,  # 关键：bf16 减少 VRAM
                trust_remote_code=True).to("cuda")
    
    def predict(self, obs, ctx):
        self._load_model()  # lazy load
        img = Image.fromarray(
            next(iter(obs["images"].values())))  # 取第一张图
        prompt = f"In: What action should the robot" \
                 f" take to {obs['task_description']}?\nOut:"
        inp = self._proc(prompt, img).to("cuda", dtype=torch.bfloat16)
        act = self._model.predict_action(inp)
        return {"actions": act}
```

Key observations：
- Lazy loading：模型在第一次 `predict` 调用时才加载，避免 server startup 时间
- `bfloat16`：相比 float32 节省一半 VRAM，相比 float16 有更好的数值稳定性
- Prompt 格式：`"In: What action should the robot take to <task>?\nOut:"` —— OpenVLA 是基于 Prismatic VLM，使用 instruction-tuned chat format
- `predict_action` 而非 `generate`：OpenVLA 暴露了专用 API 处理 action tokenization + denormalization

### 2.3 Dependency Isolation 的妙用

Model server 端用 **PEP 723 inline metadata** + **uv run**。PEP 723 是 Python 的新标准，允许在脚本顶部用 comment 声明依赖：

```python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.40.1",
#   "torch>=2.0",
#   "pillow",
# ]
# ///
```

`uv run` 看到这个 metadata 会自动创建 isolated venv。这解决了 CogACT pinning `transformers==4.40.1` vs X-VLA requiring `transformers>=4.44` 的冲突——每个 model server 在自己的 venv 里运行，互不干扰。

Benchmark 端用 Docker isolation，每个 benchmark 一个 image，all assets bundled。

---

## 3. Parallel Evaluation 数学建模

这是 paper 最有技术含量的部分。让我详细讲解 demand/supply methodology。

### 3.1 问题形式化

在 VLA evaluation 中，有两个并行的 throughput 瓶颈：

**Demand side (Environment)**：
$$\lambda(N) = \text{environment throughput as function of shard count } N$$

每个 shard 是一个独立的 Docker container 跑 benchmark。增加 shards 可以并行执行多个 episodes。但有 diminishing returns：
- Docker overhead
- CPU contention
- GPU rendering contention（SAPIEN/PyBullet）

**Supply side (Model server)**：
$$\mu(B) = \text{model inference throughput as function of batch size } B$$

Batch inference 利用 GPU parallelism，但有上限：
- VRAM capacity
- Compute saturation

### 3.2 Queuing Theory 视角

这是一个经典的 **M/M/1 queue** 的稳定性条件。如果 arrival rate λ 超过 service rate μ，queue 会无限增长，latency → ∞。

为了保证 queue 不 buildup，需要：
$$\lambda(N) < \mu(B^*)$$

但 paper 用更保守的 **0.8 系数**：
$$\lambda(N) < 0.8 \cdot \mu(B^*)$$

为什么 0.8？这是工程实践中的 **headroom factor**：
- 留 20% 余量吸收 burst arrivals
- 防止 queue 因为 variance 而 buildup
- 容忍 throughput estimation 误差

### 3.3 实测数据（LIBERO + CogACT on H100）

**Demand side scaling**：
| N (shards) | λ(N) [obs/s] | Speedup vs N=1 |
|-----------|--------------|----------------|
| 1         | 11.2         | 1.0×           |
| 50        | 364.6        | 32.6×          |

Environment throughput 不是线性 scaling——50 shards 只得到 32.6× 而非 50×，说明 overhead 不可忽略。

**Supply side scaling**：
| B (batch size) | μ(B) [obs/s] | Speedup vs B=1 |
|----------------|--------------|----------------|
| 1              | 165.2        | 1.0×           |
| 16             | 468.2        | 2.8×           |

Batch inference 只得到 2.8× 而非 16×，这是因为：
- CogACT-7B 是 autoregressive generation，batch 内不同 sequence 长度不一
- Padding 浪费
- KV cache memory 随 batch size 线性增长

**Operating point selection**：
- N* = 50, B* = 16
- λ(50) = 364.6 obs/s
- 0.8 × μ(16) = 0.8 × 468.2 = 374.6 obs/s
- 364.6 / 468.2 = 77.9% utilization ✓

**Combined speedup**：
- Sequential: ~14 hours for 2000 episodes
- Parallel: ~18 minutes
- **47× wall-clock speedup**（不是 32.6 × 2.8 = 91× 因为有 Amdahl's law 和 overhead）

### 3.4 不同 benchmark 的瓶颈差异

这非常 insightful：

| Benchmark | Bottleneck | Speedup | Note |
|-----------|-----------|---------|------|
| LIBERO    | Model inference (GPU) | 47× | robosuite CPU 渲染快，GPU 是瓶颈 |
| CALVIN    | PyBullet CPU rendering | 16× | CPU-bound，更多 shards 帮助有限 |
| SimplerEnv | SAPIEN GPU rendering | 12× | per-container GPU 是瓶颈 |

这告诉我们：**没有 one-size-fits-all 的 parallelism 策略**。LIBERO 容易加速因为瓶颈是 model，可以 batch；CALVIN/SimplerEnv 难加速因为瓶颈是 renderer，每个 shard 独占资源。

---

## 4. Reproducibility Audit 实验设计

### 4.1 实验配置

他们选了 **DB-CogACT**（CogACT 的一个 reimplementation with modern base model）作为 audit target，跑三个 benchmark：

| Benchmark | Config | Shards | Episodes |
|-----------|--------|--------|----------|
| LIBERO    | 4 suites, 10 tasks × 50 episodes | 50 | 2000 |
| CALVIN    | ABC→D split, 1000 chained sequences | 16 | 1000 |
| SimplerEnv | 4 WidowX tasks, 24 episodes × 3 seeds | 16 | 288 |

固定 seeds，versioned Docker images from ghcr.io。

### 4.2 Reproduction Results（Table II 详解）

| Benchmark | Suite/Metric | Ours | Reference | ∆ |
|-----------|--------------|------|-----------|---|
| LIBERO    | Spatial      | 95.2 | 93.8 | +1.4 |
| LIBERO    | Object       | 98.6 | 97.8 | +0.8 |
| LIBERO    | Goal         | 95.2 | 96.2 | -1.0 |
| LIBERO    | Long-Horizon | 89.6 | 91.8 | -2.2 |
| LIBERO    | Avg Len (ABC→D) | 4.051 | 4.063 | -0.012 |
| CALVIN SimplerEnv | Avg SR | 72.22 | 69.45 | +2.77 |

**Key insight**: 所有结果在 ±3 percentage points 内。这看似 reproduction 成功了，但 **discrepancy 方向不一致**：LIBERO Spatial/Object 高于 reference，Goal/Long-Horizon 低于 reference。这说明不是简单的 systematic bias，而是真实存在的 stochastic variation + 隐藏配置差异。

### 4.3 两个隐藏陷阱（这是 paper 最有价值的发现）

**Trap 1: Ambiguous Termination Semantics (SimplerEnv)**

SimplerEnv 使用 Gymnasium API，其中有 `terminated` 和 `truncated` 两个 flag：
- `terminated`：episode 自然结束（success 或 failure）
- `truncated`：episode 因为 max steps 被截断

但 SimplerEnv 的 `terminated` flag 实际含义是 **transient success event**（比如 block 短暂 stacked），而非 episode 真的结束。如果在这个 flag 上 early-stop，会 **inflate scores** 因为 robot 之后可能 disturb object。

正确做法：run until `truncated` at `max_episode_steps`。这个语义 overloading **未在文档中说明**，需要读 simulator source code 才能发现。

**Trap 2: Hidden Normalization Statistics (CALVIN)**

CALVIN 需要 hardcoded observation normalization statistics：
- 15 个 robot-state 维度的 mean/std
- 24 个 scene-state 维度的 mean/std
- 总共 39 个 dimension 的 normalization stats

这些 stats 从 **training dataset** 计算得出，但 **不在 official evaluation documentation 中**。需要 trace 回 training codebase 才能找到。用错 stats 会让 observation preprocessing 产生错误输入，silently degrade model performance。

这两个 trap 揭示了 VLA evaluation 的一个深层问题：**benchmark 的 critical implementation details 隐藏在 simulator 源码和 training codebase 中，无法仅通过 paper 复现**。

---

## 5. VLA Leaderboard 和 Cross-Benchmark Analysis

### 5.1 Leaderboard 规模

- 657 results aggregated
- 17 benchmarks tracked
- 509+ model configurations
- 2,685 papers reviewed

### 5.2 Curation Pipeline

这是一个非常现代的 pipeline：

```
2,685 papers 
    ↓
Claude Code (Opus 4.6) via MCP tools
    (arXiv, Semantic Scholar, PDF reader)
    ↓
Extract & normalize results against canonical protocols
    ↓
Human operator review (resolve anomalies)
    ↓
Provenance tracking (curated_by field)
    ↓
Public display with protocol caveats
    ↓
Community contributions via PR + schema validation
```

MCP (Model Context Protocol) 是 Anthropic 推出的标准，让 Claude 能调用外部 tools。这里用 MCP 让 Claude 同时访问 arXiv API、Semantic Scholar API、PDF reader，自动化 paper review。

### 5.3 Cross-Benchmark Coverage 分布（Fig. 5）

| # Benchmarks per model | % of models |
|----------------------|------------|
| 1                    | 81%        |
| 2                    | ~12%       |
| 3                    | ~6%        |
| 4                    | ~0.4%      |
| 5+                   | 0.6%       |

**Shocking finding**: 81% 的 VLA 模型只在 **一个** benchmark 上评估。只有 0.6% 在 5+ benchmarks 上评估。

这意味着 VLA 领域的 **cross-benchmark generalization** 几乎没有被系统研究。很多模型声称 "generalist"，但实际只在 1-2 个 benchmark 上展示。这正是 vla-eval 试图解决的——让 cross-benchmark evaluation 变得 practical。

---

## 6. 关于 Action Chunking 和 Ensemble Strategies 的技术细节

Paper 简略提到了 action chunking ensemble strategies (newest, average, EMA)，但这是 VLA 的一个重要 implementation detail，值得展开。

### 6.1 Action Chunking 背景

很多现代 VLA 模型（π0、CogACT、OpenVLA-OFT）使用 action chunking：一次 forward pass 预测未来 H 步 actions $a_{t:t+H}$，而非单步 $a_t$。好处：
- 减少推理频率（每 H 步只 forward 一次）
- Temporal consistency（chunk 内 actions 相互协调）

但 chunking 引入一个问题：**当前应该执行哪个 action？**

### 6.2 Ensemble Strategies 数学

设第 $k$ 次 forward pass 在 time step $t_k$ 预测的 chunk 为 $\hat{a}_{t_k:t_k+H}^{(k)}$。当前 time step 为 $t$。

**Strategy 1: Newest**
$$a_t = \hat{a}_t^{(k^*)}, \quad k^* = \arg\max_k \{k : t_k \leq t\}$$
用最新一次预测中对应 $t$ 的 action。响应快，但可能有 jitter。

**Strategy 2: Average**
$$a_t = \frac{1}{|K_t|} \sum_{k \in K_t} \hat{a}_t^{(k)}$$
其中 $K_t = \{k : t \in [t_k, t_k+H)\}$ 是所有覆盖 $t$ 的 predictions。平滑，但 lag。

**Strategy 3: EMA (Exponential Moving Average)**
$$a_t = \alpha \cdot \hat{a}_t^{(k^*)} + (1-\alpha) \cdot a_{t-1}$$
其中 $\alpha \in [0,1]$ 是 smoothing factor。Balance 响应性和平滑性。

不同策略对不同 benchmark 表现不同。LIBERO 需要精确，可能 EMA 好；CALVIN 需要 long-horizon consistency，可能 average 好。vla-eval 把这个选择 expose 给用户。

---

## 7. Declarative Config 设计

两个 YAML configs 驱动整个 evaluation：

**Model server config (`model_server.yaml`)**：
```yaml
model_server:
  name: openvla
  class: OpenVLAServer
  model_path: /path/to/openvla
  max_batch_size: 16
  action_chunk_strategy: ema
  ema_alpha: 0.5
```

**Benchmark config (`benchmark.yaml`)**：
```yaml
benchmark:
  name: libero
  image: ghcr.io/allenai/vla-eval-libero:v1.0
  suites: [spatial, object, goal, long_horizon]
  episodes_per_task: 50
  num_shards: 50
  seed: 42
  gpu: true
```

这种设计的好处：
- **Reproducibility**：config 文件就是完整 provenance
- **Comparability**：不同 model 跑同一 config 直接比较
- **Versioning**：Docker image tag 锁定 benchmark 版本

两条命令完成 evaluation：
```bash
vla-eval serve --config model_server.yaml   # 启动 model server
vla-eval run --config benchmark.yaml         # 跑 benchmark
```

---

## 8. Limitations 和未来方向

Paper 诚实列出了 limitations：

1. **Audit scope narrow**: 只 audit 了一个 model (DB-CogACT) 在三个 benchmark 上。理想情况应该 audit 多个 model × 多个 benchmark 的 matrix。

2. **Simulation only**: real-robot evaluation out of scope。这是大局限，因为 sim-to-real gap 是 VLA 的核心挑战。

3. **Leaderboard not independently verified**: 657 results 是从 papers 提取的，没有全部 re-evaluate。可能存在 paper 中的错误。

4. **Metric limitation**: 只支持 task success rate。缺少：
   - Motion quality (smoothness, naturalness)
   - Efficiency (steps to complete, energy)
   - Safety (collision rate, force limits)
   - Generalization (OOD robustness)

---

## 9. 我的思考：这个工作的重要意义

从 Karpathy 的角度，我认为这个 paper 体现了几个重要趋势：

### 9.1 VLA 领域的 "lm-evaluation-harness 时刻"

LLM 领域在 2020-2022 经历了 fragmentation → standardization 的过程。EleutherAI 的 lm-evaluation-harness 是关键 infrastructure。现在 VLA 领域正在重演这个过程。vla-eval 可能成为 VLA 评估的标准。

### 9.2 Reproducibility 作为一等公民

Paper 不只 release 一个 framework，还做了一个 **reproducibility audit**，发现两个 hidden traps。这种 "audit" 文化在 ML 社区越来越重要。类似 work：
- MMMU: https://mmmu-benchmark.github.io/
- HELM: https://crfm.stanford.edu/helm/
- Reproducibility Project: https://reproducibility.cs.princeton.edu/

### 9.3 Evaluation Infrastructure 作为研究 contribution

传统 ML paper 贡献的是 model 或 algorithm。但 evaluation infrastructure 越来越被认可为 first-class contribution。这个 paper 在一个 workshop/short paper 格式里展示了 infrastructure + audit + leaderboard 的组合拳。

### 9.4 AI Agent 辅助 research curation

用 Claude Code + MCP 自动 review 2685 papers 是一个非常前瞻的做法。这预示着未来 literature review、meta-analysis 会越来越自动化。

---

## 10. 技术细节补充

### 10.1 关于 PEP 723

PEP 723 (Inline script metadata) 是 2024 年的 Python 标准，允许在单个 .py 文件顶部用 TOML comment 声明依赖：

```python
# /// script
# dependencies = ["requests", "rich"]
# ///
import requests
from rich import print
```

这让 single-file scripts 变成 self-contained。`uv run` 会自动解析并创建 isolated env。vla-eval 用这个解决 model server 的 dependency 冲突。

参考：
- PEP 723: https://peps.python.org/pep-0723/
- uv: https://github.com/astral-sh/uv

### 10.2 关于 msgpack vs alternatives

为什么不用 protobuf 或 flatbuffers？
- msgpack 是 dynamic schema，适合 research 中频繁变动的 observation format
- protobuf 需要 .proto file 和 codegen，太重
- flatbuffers 主要优势是 zero-copy，但 VLA observation 需要 decode 成 numpy array，zero-copy 帮助不大

msgpack-python 可以直接 serialize numpy array（通过 numpy array buffer protocol），效率很高。

### 10.3 Docker image 体积分析

Table I 的 Docker image size 差异巨大：

| Benchmark | Size | 原因 |
|-----------|------|------|
| RLBench   | 4.7 GB | 轻量， coppeliasim |
| SimplerEnv| 4.9 GB | SAPIEN + 少量 assets |
| LIBERO    | 6.0 GB | robosuite + MuJoCo |
| CALVIN    | 9.5 GB | PyBullet + 场景 |
| Kinetix   | 9.5 GB | JAX + physics |
| ManiSkill2| 9.8 GB | SAPIEN + 大量 assets |
| MIKASA-Robo| 10.1 GB | 复杂场景 |
| LIBERO-Mem| 11.3 GB | LIBERO + memory modules |
| VLABench  | 17.7 GB | 复杂 long-horizon tasks |
| RoboTwin 2.0| 28.6 GB | bimanual + 大量 assets |
| RoboCasa | 35.6 GB | 大规模 household scenes |

RoboCasa 的 35.6 GB 是因为大规模 household 场景包含大量 3D assets。这对 Docker registry 是负担，但保证了 reproducibility。

### 10.4 关于 LIBERO 的 suites

LIBERO 有 4 个 suites：
- **Spatial**: 空间关系推理（如 "pick up the red mug on the left"）
- **Object**: object interaction（如 "open the drawer"）
- **Goal**: multi-step goal completion
- **Long-Horizon**: 长序列任务（5-10 步）

每个 suite 10 tasks × 50 episodes = 500 episodes，4 suites = 2000 episodes total。这个规模刚好能在 ~18 分钟内跑完。

### 10.5 关于 CALVIN ABC→D

CALVIN 是 long-horizon language-conditioned manipulation benchmark。ABC→D 表示：
- **Training**: tasks A, B, C（每个 task 一定数量的 demos）
- **Evaluation**: task D（held-out）
- **Metric**: average chain length——连续完成多少个 D tasks 不失败

Chain length 4.051 vs 4.063 意味着平均连续完成 ~4 个任务，这已经是非常好的 long-horizon 性能。

### 10.6 SimplerEnv 的特殊性

SimplerEnv 是一个特殊的 benchmark——它 **在 simulation 中评估 real-world robot policies**。具体来说：
- 包含 WidowX 和 Google Robot 的 simulation 环境
- 评估在 real-world data 上训练的 policies
- 用 Sim-to-Real 的反向：Real policies evaluated in Sim

这使得 SimplerEnv 成为 sim-to-real 研究的重要工具，但它的 `terminated` 语义陷阱特别危险。

---

## 11. 相关 Work 和延伸阅读

### 11.1 类似的 evaluation harness 工作

- **lm-evaluation-harness** (EleutherAI): https://github.com/EleutherAI/lm-evaluation-harness
- **HELM** (Stanford): https://crfm.stanford.edu/helm/
- **OpenCompass**: https://github.com/open-compass/opencompass
- **BIG-bench**: https://github.com/google/BIG-bench

### 11.2 VLA models referenced

- **OpenVLA**: https://openvla.github.io/ — 开源 VLA，基于 Prismatic VLM
- **π0**: https://arxiv.org/abs/2410.24164 — Physical Intelligence 的 VLA flow model
- **CogACT**: https://arxiv.org/abs/2411.19650 — cognition + action synergistic model
- **GR00T N1**: https://arxiv.org/abs/2503.14734 — NVIDIA humanoid robot foundation model
- **X-VLA**: https://arxiv.org/abs/2510.10274 — cross-embodiment VLA

### 11.3 Benchmarks referenced

- **LIBERO**: https://libero-project.github.io/
- **CALVIN**: https://calvinrobot.github.io/
- **SimplerEnv**: https://simpler-env.github.io/
- **ManiSkill2**: https://maniskill2.github.io/
- **RoboCasa**: https://robocasa.ai/
- **RLBench**: https://sites.google.com/view/rlbench
- **Kinetix**: https://kinetix.github.io/

### 11.4 Reproducibility in ML

- **Reproducibility Project ML**: https://reproducibility.cs.princeton.edu/
- **Papers with Code**: https://paperswithcode.com/
- **ML Reproducibility Checklist**: https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf

---

## 12. 总结：这个 paper 的核心 takeaway

1. **VLA evaluation 的 fragmentation 是真实且严重的问题**——dependency conflicts、underspecified protocols、slow evaluation 阻碍了社区进展。

2. **Client-server architecture with WebSocket+msgpack+Docker** 是优雅的解法，让 model 和 benchmark 独立 evolve。

3. **Demand/supply methodology for parallelism tuning** 是实用的工程贡献。47× speedup 让 routine evaluation 变得可行。

4. **Reproducibility audit 发现了两个 hidden traps**——ambiguous termination semantics 和 hidden normalization statistics。这揭示了 benchmark documentation 的不足。

5. **VLA leaderboard 显示 81% 模型只在 1 个 benchmark 上评估**，cross-benchmark generalization 几乎没有被系统研究。这是 VLA 领域的盲点。

6. **Declarative config + versioned Docker images** 让 evaluation 结果 fully reproducible from a single config file。

7. **AI agent 辅助 research curation** 是未来趋势——Claude Code + MCP 自动 review 2685 papers。

这个 paper 体现了 ML infrastructure work 的价值：不只是写代码，而是通过好的 design 暴露 hidden problems，推动社区 standards 的形成。类似 lm-evaluation-harness 对 LLM 领域的影响，vla-eval 可能成为 VLA 领域的标准 evaluation framework。

从 build intuition 的角度，这个 paper 教会我：
- **Decoupling through protocol** > monolithic codebase
- **Demand/supply thinking** 适用于任何 producer-consumer system
- **Audit 是 infrastructure work 的一部分**——不 audit 就不知道 framework 是否正确
- **Declarative configs 是 reproducibility 的 foundation**

希望这个讲解对你 build intuition 有帮助，Andrej！
