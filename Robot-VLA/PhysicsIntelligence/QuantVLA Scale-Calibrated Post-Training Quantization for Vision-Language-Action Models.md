---
source_pdf: QuantVLA Scale-Calibrated Post-Training Quantization for Vision-Language-Action
  Models.pdf
paper_sha256: 021174bcd5ffe892b2a967e0a47380de088287afd5534fdc71002f711d433b4d
processed_at: '2026-08-06T07:57:41-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 QuantVLA

Andrej, 咱们把这 paper 拆成人话讲。

---

## 这 paper 在干啥

你有个 robot, 它脑子里跑一个 VLA model——看图、听指令、输出 action。现在大家用的大模型比如 π0.5、GR00T N1.5, 都是 language backbone + DiT action head 拼起来的, memory 动辄几个 GB。你要把它塞到 robot 上的 edge GPU 里, memory 紧得要死。

最自然的想法: quantize 它嘛, 把 weight 砍到 4-bit, activation 砍到 8-bit, memory 立省 70%。但是你拿现成的 PTQ 方法 (SmoothQuant, DuQuant) 直接套上去, robot 就傻了——LIBERO Long task 从 93.5% 掉到 50%, 基本没法用。

这 paper 就在回答一个问题: **为什么 VLA 量化这么脆, 以及怎么修**。

参考: https://arxiv.org/abs/2501.09930 (QuantVLA), https://arxiv.org/abs/2211.10438 (SmoothQuant), https://arxiv.org/abs/2406.01759 (DuQuant)

---

## 为啥 VLA 量化会崩

VLA 跟普通 LLM 不一样的地方在于它有个 DiT action head, 而且 language backbone 的输出直接喂给 DiT 当 condition。这就形成了一个 tight coupling:

```
RGB → vision encoder → visual tokens ─┐
                                       ├→ F_VL → DiT action head → action
language instruction → LLM → text tokens ─┘
```

你 quantize LLM 的时候, LLM 输出的 $F_{VL}$ 统计特性就漂了。这个漂移本身不大, 但是 DiT 一接收, 在 attention 里走一圈, 漂移就被 softmax 放大; 再走几十层 residual, 漂移就累积成灾难。

作者用 first-order Taylor 把这事算清楚了。设 teacher 输入 $X_T$, quantized 输入 $X_Q = X_T + \varepsilon_{up}$, $\varepsilon_{up}$ 是 upstream LLM 量化传导下来的 perturbation。

attention 的 logits 是:
$$L = \frac{Q K^\top}{\sqrt{d}}$$

其中 $d$ 是 head dimension, $\sqrt{d}$ 是标准 scaling factor, 作用是让 dot product 的方差跟 $d$ 无关, 让 softmax 梯度稳定。

一阶展开 $\Delta L = L_Q - L_T$:
$$\Delta L \approx \frac{1}{\sqrt{d}}\Big((\varepsilon_{up} W_q) K_T^\top + Q_T (\varepsilon_{up} W_k)^\top\Big) + \Delta L_{\text{local}}$$

这里 $\Delta L_{\text{local}}$ 是 quantized activation 本地的 rounding + scale mismatch。

**人话翻译**: quantization 让 Q 和 K 的方差变了, logits 的 std 就变了。softmax 对 std 极其敏感——std 大就 sharp (overconfident), std 小就 flat (uniform)。这等于悄悄改了 attention 的 temperature。

然后 value 路径也漂:
$$V_Q = V_T + \varepsilon_{up} W_v$$

output projection 一过, 整个 block 输出到 residual stream 的 energy 就偏了:
$$\Delta O \approx J_{\text{softmax}}(L_T)\Delta L \cdot V_T W_{o,T} + A_T \varepsilon_{up} W_v W_{o,T} + A_T V_T \delta W_o + \Delta O_{\text{local}}$$

四个 term: 第一个是 attention 路径扰动, 第二个是 value 路径扰动, 第三个是 output proj 权重扰动, 第四个是本地 rounding。

**人话**: 整个 DiT block 输出给 residual stream 的 vector, 它的 norm (energy) 偏离了 teacher。下游 LayerNorm 看到 norm 变了, operating point 就移位, 后面所有层都跟着偏。

---

## 关键 insight: 所有误差最后都收玫到两个乘积

这是 paper 最漂亮的地方。看 Appendix C 的推导。

integer tensor $\tilde Q, \tilde K, \tilde V$ 反量化时引入 scale:
$$\hat Q = s_q \tilde Q, \quad \hat K = s_k \tilde K, \quad \hat V = s_v \tilde V$$

$s_q, s_k, s_v$ 是 per-tensor 或 per-channel 的 dequantization scale, 就是 integer 值乘以这个 scale 还原成 float 近似值。

logits:
$$L = \frac{\hat Q \hat K^\top}{\sqrt{d}} = \frac{s_q s_k}{\sqrt{d}} \tilde Q \tilde K^\top$$

所以 effective temperature:
$$T_{\text{eff}} = \frac{\sqrt{d}}{s_q s_k}$$

**$s_q s_k$ 直接决定 attention sharpness**。

output projection $\hat W_o = s_o \tilde W_o$, block output:
$$Z = \text{Concat}(Y_h) \hat W_o = s_o \cdot \text{Concat}(Y_h) \tilde W_o$$

每个 head $Y_h = s_v A \tilde V$, 所以 residual stream energy 由 $s_v \cdot s_o$ 决定。

**人话总结**: 不管 quantization 误差从哪里来, 最后表现到 DiT attention 上就两件事:
1. $s_q s_k$ 偏了 → attention temperature 偏了
2. $s_v s_o$ 偏了 → residual energy 偏了

修这两件事就完事了。

---

## QuantVLA 的三招

### 第一招: Selective layout, 别全量化

策略:
- LLM 所有 linear layers: 全 quantize (LLM 容忍度高, 收益大)
- DiT 的 MLP / FFN: quantize
- DiT 的 attention projections $W_q, W_k, W_v, W_o$: **留 FP16**

为啥 attention proj 留 FP? 因为 attention proj 直接接 softmax, softmax 高度非线性, 对 input perturbation 放大最厉害。MLP 的 GELU/SiLU 虽然也非线性, 但通道间独立, 比 softmax 温和得多。同样的 quantization error 加在 attention proj 上后果严重得多。

Table 1 的 ablation 验证得明明白白 (关掉 ATM/OHB, 纯看 layout):

| π0.5 config | Spatial | Object | Goal | Long | Avg | Mem |
|---|---|---|---|---|---|---|
| FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27GB |
| W4A8 LLM only | 98.0 | 98.5 | 97.5 | 92.0 | 96.5 | 1.58GB |
| W4A8 DiT only | 81.5 | 94.5 | 71.5 | 39.0 | 71.6 | 3.85GB |
| W4A8 LLM+DiT full | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17GB |
| W4A8 LLM+DiT(MLP) | 98.0 | 97.0 | 94.5 | 92.0 | 95.4 | 1.28GB |

Long task 最敏感, DiT full quantization 直接掉到 39%。LLM+DiT(MLP) 几乎追平 FP16, memory 还省 70%。

### 第二招: ATM (Attention Temperature Matching)

每个 head 学一个 scalar $\alpha$, 让 student logits 的 std 等于 teacher logits 的 std:

$$\alpha_{\text{raw}} = \frac{\text{Std}(L_T)}{\text{Std}(L_Q) + 10^{-6}}$$

$L_T$ 是 teacher (FP16) 的 logits, $L_Q$ 是 quantized 的 logits, $10^{-6}$ 防 zero-division。

两道保险:
- clip 到 safe range: $\alpha \in [\alpha_{\min}, \alpha_{\max}]$, 实验取 $\log \alpha \in [-0.4, 0.4]$
- neutrality band: if $|\log \alpha| < \varepsilon$ then $\alpha = 1$ ($\varepsilon = 0.03$)

neutrality band 的直觉: 当 teacher/student 的 std 差距很小 (calibration noise 量级), 别瞎调, 保持 $\alpha = 1$ 才不引入新噪声。

inference 时:
$$L_Q = \frac{L_T}{\alpha}$$

(我怀疑这里 paper 有笔误, 应该是 $L_Q^{\text{corrected}} = \alpha \cdot L_Q^{\text{orig}}$ 才对, 因为 $\alpha = \text{Std}(L_T)/\text{Std}(L_Q) > 1$ 时 student std 比 teacher 小, 要放大 student。但语义上就是把 $\alpha$ fold 到 $s_q$ 或 $s_k$ 里, 效果一致。)

**人话**: 每个 head 量一下 teacher 的 logits 有多 sharp, student 的有多 sharp, 算个比值, 把 student 的 logits 整体缩放到跟 teacher 一样 sharp。一个 scalar 搞定。

### 第三招: OHB (Output Head Balancing)

每层学一个 scalar $\beta(l)$, 让 student 的 output RMS 等于 teacher 的 output RMS:

$$Z_l = \text{Concat}\{A_{l,h} V_{l,h}\} W_{o,l} + b_{o,l}$$

$$\beta_{\text{raw}}(l) = \frac{\text{RMS}(Z_{T,l})}{\text{RMS}(Z_{Q,l}) + 10^{-6}}$$

同样 clip + neutrality band:
$$\beta(l) = \text{clip}(\beta_{\text{raw}}(l), \beta_{\min}, \beta_{\max})$$
$$\text{if } |\log \beta(l)| < \varepsilon \text{ then } \beta(l) = 1$$

inference 时:
$$Z_Q = \frac{Z_l}{\beta(l)}$$

**人话**: 每层量一下 teacher 输出 vector 的 RMS 有多大, student 的有多大, 算个比值, 把 student 输出整体缩放回去。一个 scalar 搞定。

### 工程上的关键: 零 inference overhead

$\alpha$ fold 进 $s_q$ 或 $s_k$ (就是 dequant scale 乘以 $\alpha$), $\beta$ fold 进 $s_o$。都是 scalar 乘法, **integer GEMM 调用次数不变, operator schedule 不变, 不引入新 buffer**。calibration 一次性跑完 (128 steps, 最多 5 trials per task), inference 端零开销。

这点对部署极其关键。你想想, 如果 calibration 要在 inference 时跑个 extra forward pass, 那 latency 就废了。

---

## DuQuant 这个底座是啥

QuantVLA 建立在 DuQuant 的 reparameterization 上。DuQuant 对每个 linear layer 做三件事:

### Step 1: Per-channel smoothing

平衡 activation 和 weight 的量化难度。activation 有 outlier channel 很难量化, weight 通常好量化得多。用对角矩阵 $\Lambda$ 把 activation 的大幅 channel 缩小, weight 的对应行放大, 总乘积不变:

$$Y = (X\Lambda)(\Lambda^{-1}W) = X' W'$$

$$\Lambda_j = \frac{(\max|X_{:,j}|)^\alpha}{(\max|W_{j,:}|)^{1-\alpha}}, \quad \alpha \in [0,1]$$

$\Lambda_j$ 是对角矩阵第 $j$ 个对角元。$X_{:,j}$ 是 activation 第 $j$ 列的所有元素, $W_{j,:}$ 是 weight 第 $j$ 行的所有元素。$\alpha = 0.15$ 是实验取值, 偏向把更多 smoothing 给 activation。

**人话**: activation 有个 channel 数值特别大, 量化时它就把整个 grid 撑大了, 其他 channel 精度就烂。用 $\Lambda$ 把这个 channel 缩小, 同时把 weight 对应行放大, 乘积不变, 但 activation 好量化了。

### Step 2: Block-orthogonal rotation + zigzag permutation

$$Y = \underbrace{[(X\Lambda)\hat R_{(1)} P \hat R_{(2)}]}_{G} \underbrace{[\hat R_{(2)}^\top P^\top \hat R_{(1)}^\top (\Lambda^{-1}W)]}_{G^{-1}}$$

$\hat R_{(1)}, \hat R_{(2)}$ 是 block-orthogonal 矩阵 (block size 64), $P$ 是 zigzag permutation, 三者都正交所以 $G \cdot G^{-1} = I$。

**人话**: rotation 把 outlier channel 的能量打散到同 block 的其他 channel 上。原来一个 channel 数值特别大, 其他 channel 特别小, 量化 grid 被那一个 channel 主导。rotate 完所有 channel 数值差不多大, per-channel quantization grid 就均匀了。zigzag permutation 是把相邻 block 的 channel 交错重排, 进一步分散 outlier。

左边的 $G$ 作用在 activation 上 (inference 时 apply), 右边的 $G^{-1}$ fold 进 weight (离线算好存起来), 等价性保持。

---

## 实验结果

### 主结果 (Table 2)

| Model | Precision | Spatial | Object | Goal | Long | Avg | Mem | Saving |
|---|---|---|---|---|---|---|---|---|
| π0.5 | FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27GB | 0% |
| + DuQuant (LLM+DiT) | W4A8 | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17GB | 72.6% |
| + QuantVLA (LLM only) | W4A8 | 98.5 | 99.0 | 96.5 | 96.5 | 97.6 | 1.58GB | 63.0% |
| **+ QuantVLA (full)** | W4A8 | 98.5 | 98.0 | 98.0 | 96.0 | **97.6** | 1.28GB | **70.0%** |
| GR00T N1.5 | FP16 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.02GB | 0% |
| + DuQuant | W4A8 | 66.0 | 70.0 | 68.0 | 76.0 | 70.0 | 0.74GB | 63.4% |
| + QuantVLA (LLM only) | W4A8 | 96.0 | 94.0 | 92.0 | 66.0 | 87.0 | 1.25GB | 38.1% |
| **+ QuantVLA (full)** | W4A8 | 96.0 | 92.0 | 90.0 | 74.0 | **88.0** | 0.91GB | **55.0%** |

两个看点:
1. **QuantVLA 跑赢 FP16**: π0.5 上 97.6% > 97.1%, GR00T 上 88.0% > 86.5%。LLM PTQ 里也偶尔出现这种现象, 解释是 ATM/OHB 起了一点正则化作用, 把原来累积的轻微 scale 偏差也一并修了。
2. **DuQuant naive 全量化在 Long 上崩盘**: π0.5 Long 掉到 50%, GR00T Long 掉到 76%。Long suite 考的是 temporal decomposition + accumulated error, attention temperature drift 在长序列 rollout 里被指数放大。

### ATM / OHB 的视觉证据 (Figure 3)

- 左图: logits Std 跨 attention block 比较。no-calibration 曲线明显偏离 teacher, 加上 ATM 后几乎贴住 teacher, 深层 block 收敛最好。
- 右图: attention output RMS 跨 block 比较。OHB 把每层 output RMS 拉回 teacher 水平, 深层效果尤其明显。

这图是 paper 的 intuition building 核心——你能直接看到 quantization 怎么把 std 和 RMS 推偏, ATM 和 OHB 又怎么拉回来。

### 精度鲁棒性 (Table 3)

π0.5:
- FP16: 97.1%
- W4A8: 97.6%
- W4A4: 95.3% (Long 90.5%)

压到 W4A4 还能保持 95.3%, 说明 ATM/OHB 在更激进 bitwidth 下依然 work。

### Denoising step 鲁棒性 (Table 4)

GR00T N1.5:
- 8 steps: FP16 86.5% → QuantVLA 88.0%
- 16 steps: QuantVLA 88.5%

diffusion 步数变化时 QuantVLA 仍稳定。实际部署时工程师经常想用更少 step 换 latency, 这个结果说明量化跟 step reduction 是 orthogonal 的, 可以叠加用。

### Simpler Pick-and-Can (Table 6)

| Method | Precision | PickCan |
|---|---|---|
| GR00T | FP16 | 31/50 |
| + SmoothQuant | W4A8 | 16/50 |
| + QuantVLA | W4A8 | 27/50 |

SmoothQuant 在 aggressive 量化下直接腰斩, QuantVLA 几乎追上 FP16。这对比很 stark——SmoothQuant 的 channel-wise rescaling 对 VLA 这种 cross-module coupling 不够用。

### OpenVLA (non-DiT, Table 7)

OpenVLA 用 32-layer LLM + non-DiT action head, 架构 coupling 不同。ATM/OHB 是为 DiT 设计的, 但 QuantVLA 在 W8A16 下 Spatial 86.0% > FP16 84.7%。说明 selective layout 这个思路本身是通用的, 不限于 DiT。

---

## 为啥这套设计 work——几个 intuition angle

### Angle 1: Information theory

softmax 输出的 distribution 的 entropy 由 logits 的 std 决定 (mean-zero 假设下)。teacher 训练时模型已经学到了特定的 attention entropy schedule; quantization 改变 std 等于 shift 了 entropy, attention pattern 偏离训练分布上的最优解。ATM 把 std 拉回去, 本质上是恢复 teacher 的 attention entropy schedule。

### Angle 2: Residual stream geometry

Transformer 的 residual stream 是个高维向量空间, 每层加一个 update vector。LayerNorm 的 operating point (mean/std) 决定 update 的 effective magnitude。如果 $s_v s_o$ 偏了, 每层 update 的 norm 都偏, 相当于在 residual stream 上做了 systematic scaling, 几十层下来把 trajectory 推到训练分布外。OHB 对每层 update norm 归一化, 把 trajectory 拉回训练流形。

### Angle 3: 为啥 per-head / per-layer 而不是 global

$\alpha$ 用 per-head: 不同 head 学到不同 attention pattern, 有的 head 本来就 sharp (large logits std), 有的 flat (small std)。quantization 对每个 head 的 std perturbation 大小不同, 必须 per-head 修。

$\beta$ 用 per-layer: 不同深度层的 residual injection gain 不一样。浅层更多是 representation building, 深层更多是 task-specific refinement, 每层在 residual stream 里的能量贡献分布不同。

### Angle 4: 为啥 DiT attention proj 留 FP

attention proj 直接接 softmax, softmax 是高度非线性的, 小 input perturbation 被指数放大。MLP 的 GELU/SiLU 虽然也非线性, 但通道间独立, 没有 softmax 那种全局耦合。所以同样的 quantization error 加在 attention proj 上比加在 MLP 上后果严重得多。Table 1 验证了: DiT only quantization (包含 attention proj) Long 掉到 39%, DiT MLP only quantization Long 保持 92%。

---

## 一些可能的延伸和疑问

### W4A4 的进一步极限
现在 W4A4 在 Long 上 90.5%, 还能不能 push? 可能需要 per-timestep $\beta$ (不同 diffusion step 给不同 $\beta$), 因为不同 timestep 的 noise level 不同, 对 quantization error 的敏感度也不同。早期 step (高 noise) 可能容错高, 后期 step (低 noise) 可能需要更精细 calibration。

### Vision encoder 也 quantize 会怎样
paper 故意 keep vision frozen。vision encoder 是另一个 memory 大户, 特别是 SigLIP2 / DINOv2 这种大 backbone。如果 vision encoder 也 quantize, $F_{VL}$ 本身的分布也漂了, ATM/OHB 还够不够用? 可能需要在 vision encoder 输出端也加一个类似的 calibration。参考 DINOv2: https://arxiv.org/abs/2304.07193, SigLIP: https://arxiv.org/abs/2303.15343

### Real robot transfer
LIBERO 是 sim, sim 上 success rate 略涨不等于 real robot 上也涨。Real robot 有 sim-to-real gap, 有 sensor noise, 有 actuator dynamics, 量化引入的额外 error 在 real 上可能被放大。需要 real-world deployment 验证。

### 跟 KV cache quantization 的结合
VLA rollout 时 KV cache 也是 memory 大头, 特别是 long horizon 任务。QuantVLA 关注 weights + activations, 没碰 KV cache quantization。可以叠加上 KIVI (https://arxiv.org/abs/2402.02750) 这类 2-bit KV cache quantization, memory 收益更大。

### 跟 speculative decoding 的结合
DiT flow matching 是迭代 refinement, 每步都要跑一遍 network。如果可以做 speculative step (一次性预测多步然后 verify), 叠加 quantization 可以再省 inference latency。参考 Medusa: https://arxiv.org/abs/2401.10774, Eagle: https://arxiv.org/abs/2401.15077

### ATM 公式 (12) 的笔误嫌疑
$$L_Q = \frac{L_T}{\alpha}$$

按定义 $\alpha = \text{Std}(L_T)/\text{Std}(L_Q)$, 要让 student std = teacher std, 应该是 $L_Q^{\text{corrected}} = \alpha \cdot L_Q^{\text{orig}}$。写成 $L_Q = L_T / \alpha$ 语义上说不通 (你不知道 $L_T$ 在 inference 时是多少)。我猜作者实现里是把 $\alpha$ fold 到 $s_q$ 或 $s_k$ 上, 效果是把 student logits 乘以 $\alpha$, 最终一致, 但公式写得有歧义。

### 跟 rotation-based PTQ 的 broader context
QuaRot (https://arxiv.org/abs/2404.00456), SpinQuant (https://arxiv.org/abs/2405.16406), FlatQuant (https://arxiv.org/abs/2410.09426) 这条 rotation-based PTQ 线都在做类似的事: 用 orthogonal transform 把 outlier 分散, 让 per-channel quantization grid 更均匀。DuQuant 的特色是 block-orthogonal + zigzag permutation, 计算量比 full Hadamard rotation 小。QuantVLA 直接复用 DuQuant 作为底座, 在上面加 ATM/OHB 这两个 VLA-specific 的 calibration, 是个聪明的工程选择。

---

## 一句话 mental model

QuantVLA = **selective layout** (DiT attention proj 留 FP, 其余全 quantize) + **ATM** (per-head scalar $\alpha$ 修 attention temperature) + **OHB** (per-layer scalar $\beta$ 修 residual energy)。

三招都是 training-free, architecture-preserving, calibration-only, inference-time 零开销。在 LIBERO 上跑赢 FP16 + 70% memory saving, 给 VLA 在 robot 端的真实部署开了条路。

如果未来有人要 push 到 W2A4 或者 binary VLA, ATM/OHB 的 per-head/per-layer scalar 大概率不够, 需要 per-timestep、per-channel-group 的 finer-grained calibration。但作为 "VLA PTQ 的第一次正确打开方式", 这篇 paper 的 contribution 是清晰的。

参考链接汇总:
- QuantVLA: https://quantvla.github.io/
- DuQuant: https://arxiv.org/abs/2406.01759
- SmoothQuant: https://arxiv.org/abs/2211.10438
- QuaRot: https://arxiv.org/abs/2404.00456
- OpenPI π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1.5: https://arxiv.org/abs/2503.14734
- LIBERO: https://arxiv.org/abs/2306.03310
- Flow matching: https://arxiv.org/abs/2210.02747
- KIVI (KV cache quant): https://arxiv.org/abs/2402.02750
- DINOv2: https://arxiv.org/abs/2304.07193
- SigLIP: https://arxiv.org/abs/2303.15343
- OpenVLA: https://arxiv.org/abs/2406.09246
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- RDT-1B: https://arxiv.org/abs/2410.07864
- SVDQuant (DiT quant): https://arxiv.org/abs/2411.05007
- ViDiT-Q (DiT quant): https://arxiv.org/abs/2406.02540
- EfficientVLA: https://arxiv.org/abs/2506.10100
- VLA-Cache: https://arxiv.org/abs/2502.02175
- MoLe-VLA: https://arxiv.org/abs/2503.20384
- TinyVLA: https://arxiv.org/abs/2409.12514
- SmolVLA: https://arxiv.org/abs/2506.01844
- FAST (action tokenization): https://arxiv.org/abs/2501.09747

---

# QuantVLA 深度解读：让 VLA 模型在低比特下跑得稳

Andrej, 这篇 paper 直击一个实际但被忽视的痛点：VLA (Vision-Language-Action) 模型在真实机器人上部署时，memory 和 compute 都很紧，但以往的 efficiency 工作几乎都盯着 vision encoder 去做 token pruning / caching，**真正吃 memory 的是 language backbone + DiT (Diffusion Transformer) action head 这条 downstream stack**，却没人敢动它——因为 DiT action head 输出的是要直接喂给真电机的 action token，一点点 rounding error 都会变成控制误差。QuantVLA 第一次把 PTQ (Post-Training Quantization) 用到 VLA 上，并且不仅不掉点，反而在 LIBERO 上略微超过 FP16 baseline，同时把 memory 砍掉 70% 左右。

参考链接：
- Project homepage: https://quantvla.github.io/
- DuQuant (前身方法, NeurIPS 2024): https://arxiv.org/abs/2406.01759
- SmoothQuant (经典 PTQ baseline): https://arxiv.org/abs/2211.10438
- OpenPI π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1.5: https://arxiv.org/abs/2503.14734
- LIBERO benchmark: https://arxiv.org/abs/2306.03310
- Flow matching (DiT 训练目标): https://arxiv.org/abs/2210.02747

---

## 1. 问题本质：为什么不能直接拿 SmoothQuant / DuQuant 套上去

VLA 的 inference pipeline 形态长这样：RGB history → vision encoder → visual tokens；language instruction → language backbone → text tokens；两者在共享 transformer 空间里通过 attention 融合成 task-conditioned representation $F_{VL}$；然后 DiT action head 以 $F_{VL}$、robot proprioception、diffusion timestep $t$ 为条件，迭代执行
$$x_{t-1} = f_\theta(x_t, F_{VL}, t)$$
经过 $T$ 步 refinement，$x_0$ 解码成可执行 action。

关键 coupling 在于：$F_{VL}$ 是 DiT 的条件输入，language backbone 一旦被 quantize，喂给 DiT 的 $F_{VL}$ 的统计特性就会漂移，DiT 又要保证 action 精确，于是误差在 cross-module boundary 处被放大。这点是 unimodal LLM 量化里不会遇到的。

作者用 first-order Taylor 把这件事讲清楚了。设 teacher 输入 $X_T$，quantized 输入 $X_Q = X_T + \varepsilon_{up}$，其中 $\varepsilon_{up}$ 是从 upstream LLM 量化传导下来的 perturbation。

### 1.1 Attention logit 漂移
$$Q_Q = X_Q W_q = Q_T + \varepsilon_{up} W_q, \quad K_Q = X_Q W_k = K_T + \varepsilon_{up} W_k$$

logits 定义（$d$ 是 head dimension，$\sqrt{d}$ 是标准 scaling，作用是让 dot product 的方差不随 $d$ 爆炸，稳定 softmax 梯度）：
$$L_T = \frac{Q_T K_T^\top}{\sqrt{d}}, \quad L_Q = \frac{Q_Q K_Q^\top}{\sqrt{d}}$$

一阶展开后：
$$\Delta L \approx \frac{1}{\sqrt{d}}\Big((\varepsilon_{up} W_q) K_T^\top + Q_T (\varepsilon_{up} W_k)^\top\Big) + \Delta L_{\text{local}}$$

这里 $\Delta L_{\text{local}}$ 是 quantized activation 喂入 Q, K 时本地的 rounding + scale mismatch 累积。注意 $\varepsilon_{up} W_q$ 这种项是 cross-channel 相关的，没有理由在统计上相互抵消，所以 $\Delta L$ 系统性地改变了 logits 的 std。

softmax 的 Jacobian $J_{\text{softmax}}(L_T)$ 把 $\Delta L$ 映射成 attention 权重扰动：
$$A_Q \approx A_T + J_{\text{softmax}}(L_T)\,\Delta L$$

直观理解：logits 的 std 变了，相当于 softmax 的 effective temperature 变了——std 变大就是 "过冷" (overconfident, sharp)，std 变小就是 "过热" (flattened)。这件事在一层里不致命，但 deep DiT 里靠 residual + LayerNorm 一路累积，就会把 attention 分布越推越远。

### 1.2 Residual energy 漂移
Value path:
$$V_Q = X_Q W_v = V_T + \varepsilon_{up} W_v$$

Output head 路径（$\delta W_o$ 是 $W_o$ 量化引入的权重扰动）：
$$O_T = A_T V_T W_{o,T}, \quad O_Q = A_Q V_Q W_{o,Q}$$

一阶展开：
$$\Delta O \approx \underbrace{J_{\text{softmax}}(L_T)\Delta L\, V_T W_{o,T}}_{\text{attention 路径扰动}} + \underbrace{A_T \varepsilon_{up} W_v W_{o,T}}_{\text{value 路径扰动}} + \underbrace{A_T V_T \delta W_o}_{\text{output proj 权重扰动}} + \Delta O_{\text{local}}$$

四个 term 把上游传导的 $\varepsilon_{up}$ 在 value/output 通道上也表现出来了。$s_v \cdot s_o$ 这一对 dequant scale 共同决定 residual stream 的注入 energy，一旦它和 teacher 不一致，下游 LayerNorm 的 operating point 就偏了，等于把整个 DiT block 的归一化统计又移位一次。

### 1.3 量化 scale 怎么进入 DiT attention（Appendix C 的关键）
integer tensor $\tilde Q, \tilde K, \tilde V$ 反量化时引入 scale $s_q, s_k, s_v$：
$$\hat Q = s_q \tilde Q, \quad \hat K = s_k \tilde K, \quad \hat V = s_v \tilde V$$

logits:
$$L = \frac{\hat Q \hat K^\top}{\sqrt{d}} = \frac{s_q s_k}{\sqrt{d}}\, \tilde Q \tilde K^\top$$

所以 effective temperature $T_{\text{eff}} = \frac{\sqrt{d}}{s_q s_k}$，**$s_q s_k$ 直接控制 attention 的 sharpness**。output projection $\hat W_o = s_o \tilde W_o$，于是 block output
$$Z = \text{Concat}(Y_h)\hat W_o = s_o\, \text{Concat}(Y_h)\tilde W_o$$

而每个 head 的 $Y_h = s_v A \tilde V$，所以 $s_v s_o$ 共同决定 residual stream energy。

这两条公式是整篇 paper 的核心 insight：**所有量化误差，归根到底就是 $s_q s_k$ 偏离 teacher 的 logits scale，以及 $s_v s_o$ 偏离 teacher 的 residual energy**。QuantVLA 的两个 calibration 就是直接对这两条做 correction。

---

## 2. QuantVLA 框架：三个 scale-calibrated 组件

### 2.1 Selective Quantization Layout
不是一刀切 quantize 所有 linear。策略是：
- LLM 中所有 linear layers 全部 integerize（LLM 容忍度高，收益大）
- DiT 中只 quantize MLP / FFN blocks
- DiT 的 attention projections $W_q, W_k, W_v, W_o$ **保持 floating point**

这个设计直击前面分析的两个漂移源：attention proj 直接决定 $T_{\text{eff}}$ 和 residual energy，最敏感，留给 FP；MLP 是 channel-wise 大量乘加，quantize 收益最大且不直接扰动 softmax 分布。

Table 1 的 ablation 很说明问题（关掉 ATM/OHB，纯看 layout 选择）：

| π0.5 setting | Spatial | Object | Goal | Long | Avg | Mem (GB) |
|---|---|---|---|---|---|---|
| FP16 baseline | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27 |
| W4A8 LLM only | 98.0 | 98.5 | 97.5 | 92.0 | 96.5 | 1.58 |
| W4A8 DiT only | 81.5 | 94.5 | 71.5 | 39.0 | 71.6 | 3.85 |
| W4A8 LLM+DiT (full) | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17 |
| W4A8 LLM+DiT(MLP) | 98.0 | 97.0 | 94.5 | 92.0 | 95.4 | 1.28 |

Long task 最敏感——Long suite 考的是 temporal decomposition + accumulated error，DiT 量化全开时直接掉到 39%。这说明 long-horizon rollout 会把 attention temperature drift 累积放大。LLM+DiT(MLP) 几乎和 FP16 持平，又拿到接近 full quantization 的 memory 收益。

### 2.2 Attention Temperature Matching (ATM)
对每个 head 学一个 scalar $\alpha$，目标是把 quantized logits 的 std 拉回 teacher 的 std：

$$\alpha_{\text{raw}} = \frac{\text{Std}(L_T)}{\text{Std}(L_Q) + 10^{-6}}$$

这里 $10^{-6}$ 防 zero-division。然后两道保护：
- clip 到 safe range：$\alpha = \text{clip}(\alpha_{\text{raw}}, \alpha_{\min}, \alpha_{\max})$（实验里 $\alpha \in [-0.4, +0.4]$ 对应 $\log \alpha \in [-0.4, 0.4]$ 区间）
- neutrality band：if $|\log \alpha| < \varepsilon$ then $\alpha = 1$（$\varepsilon = 0.03$）

neutrality band 的直觉是：当 teacher/student 的 std 差距很小（calibration noise 量级）时不要瞎调，保持 $\alpha=1$ 才不引入额外噪声。

inference 时把 $\alpha$ fold 进 dequant scale（具体是把 $\alpha$ 当作 $s_q s_k$ 的乘性修正），不增加任何 kernel 调用。

### 2.3 Output Head Balancing (OHB)
对每层学一个 scalar $\beta(l)$，匹配 post-projection 的 RMS energy：
$$Z_l = \text{Concat}\{A_{l,h} V_{l,h}\} W_{o,l} + b_{o,l}$$

$$\beta_{\text{raw}}(l) = \frac{\text{RMS}(Z_{T,l})}{\text{RMS}(Z_{Q,l}) + 10^{-6}}$$

同样 clip + neutrality band：
$$\beta(l) = \text{clip}(\beta_{\text{raw}}(l), \beta_{\min}, \beta_{\max}), \quad \text{if } |\log \beta(l)| < \varepsilon \text{ then } \beta(l) = 1$$

inference 时：
$$Z_Q = \frac{Z_l}{\beta(l)}$$

也就是把 student 的 output 整体除以 $\beta(l)$ 让 RMS 和 teacher 对齐。**这一步直接修正 $s_v s_o$ 的能量偏移**，让下游 LayerNorm 的 operating point 回到 teacher 状态。

### 2.4 关键工程细节：零额外 inference cost
ATM 的 $\alpha$ fold 进 $s_q, s_k$，OHB 的 $\beta$ fold 进 $s_o$——都是 scalar 乘法，**整数 GEMM 调用次数不变、operator schedule 不变、不引入新 buffer**。calibration 一次性跑完，inference 端零开销。这点对部署很关键。

### 2.5 DuQuant reparameterization 作为底座
每层做 invertible reparameterization 让量化更稳：
1. Per-channel smoothing with diagonal $\Lambda$：
$$Y = (X\Lambda)(\Lambda^{-1}W) = X' W'$$
$$\Lambda_j = \frac{(\max|X_{:,j}|)^\alpha}{(\max|W_{j,:}|)^{1-\alpha}}, \quad \alpha \in [0,1]$$
（这里 $\Lambda_j$ 是对角矩阵第 $j$ 个对角元；$X_{:,j}$ 是 activation 第 $j$ 列；$W_{j,:}$ 是 weight 第 $j$ 行；$\alpha=0.15$ 是实验取值。直觉：把 "难量化" 的 activation 大幅 channel 通过 $\Lambda$ 平摊到 weight 侧，weight 量化容易得多）

2. Block-orthogonal rotation + zigzag permutation：
$$Y = \underbrace{[(X\Lambda)\hat R_{(1)} P \hat R_{(2)}]}_{G} \underbrace{[\hat R_{(2)}^\top P^\top \hat R_{(1)}^\top (\Lambda^{-1}W)]}_{G^{-1}}$$
$\hat R_{(1)}, \hat R_{(2)}$ 是 block-orthogonal（block size 64），$P$ 是 zigzag permutation，三者都正交所以 $G \cdot G^{-1} = I$。直觉：rotation 把 outlier channel 的能量打散到同 block 的其他 channel 上，让 per-channel quantization grid 更均匀。

---

## 3. 实验结果细节

### 3.1 主结果 (Table 2)

| 模型 | Precision | Spatial | Object | Goal | Long | Avg | Mem (GB) | 相对节省 |
|---|---|---|---|---|---|---|---|---|
| π0.5 | FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27 | 0% |
| + DuQuant (LLM+DiT) | W4A8 | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17 | 72.6% |
| + QuantVLA (LLM only) | W4A8 | 98.5 | 99.0 | 96.5 | 96.5 | 97.6 | 1.58 | 63.0% |
| **+ QuantVLA (full)** | W4A8 | 98.5 | 98.0 | 98.0 | 96.0 | **97.6** | 1.28 | **70.0%** |
| GR00T N1.5 | FP16 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.02 | 0% |
| + DuQuant | W4A8 | 66.0 | 70.0 | 68.0 | 76.0 | 70.0 | 0.74 | 63.4% |
| + QuantVLA | W4A8 | 96.0 | 92.0 | 90.0 | 74.0 | **88.0** | 0.91 | 55.0% |

两个看点：
1. **QuantVLA 不仅不掉点，π0.5 上 97.6% > FP16 的 97.1%**，GR00T 上 88.0% > 86.5%。这种 "量化反而更好" 现象在 LLM PTQ 里也偶尔出现，解释是 ATM/OHB 起到了一点正则化作用，把原来累积的轻微 scale 偏差也一并修了。
2. DuQuant (LLM+DiT) 这种 naive 全量化在 Long 上分别掉到 50.0% 和 76.0%，证实了 long-horizon rollout 对 attention temperature drift 的极端敏感性。

### 3.2 ATM / OHB 各自的视觉证据 (Figure 3)
- 左图：logits Std 跨 attention block 比较。QuantVLA-no-calibration 曲线明显偏离 teacher，加上 ATM 后曲线几乎贴住 teacher，尤其在深层 block 收敛最好。
- 右图：attention output RMS 跨 block 比较。OHB 把每层 output RMS 拉回 teacher 水平，深层效果尤其明显。

### 3.3 精度鲁棒性 (Table 3)
π0.5 上：FP16 97.1% → W4A8 97.6% → W4A4 95.3%。即使压到 W4A4，Long 还能保持 90.5%，说明 ATM/OHB 在更激进 bitwidth 下依然 work。

### 3.4 Denoising step 鲁棒性 (Table 4)
GR00T N1.5：
- 8 steps：FP16 86.5% → QuantVLA 88.0%
- 16 steps：QuantVLA 88.5%

diffusion 步数变化时 QuantVLA 仍稳定，这对实际部署很重要——工程师经常想用更少 step 换 latency。

### 3.5 跟 SmoothQuant 对比 (Table 5, Appendix E)
SmoothQuant 在 W8A8 下 96.6%，扩展到 LLM+DiT(MLP) 在 W8A8 下 97.0%，跟 QuantVLA W4A8 97.6% 接近。但 QuantVLA 用的 bitwidth 更激进，memory 收益更大。

### 3.6 Simpler Pick-and-Can (Table 6)
GR00T 在 Pick-and-Can 上：FP16 31/50 → SmoothQuant W4A8 16/50 → QuantVLA W4A8 27/50。SmoothQuant 在这种 aggressive 量化下直接腰斩，QuantVLA 几乎追上 FP16。

### 3.7 非 DiT 架构也适用 (Table 7, OpenVLA)
OpenVLA 用 32-layer LLM backbone + non-DiT action head，架构 coupling 不同。ATM/OHB 是为 DiT 设计的，但 QuantVLA 在 W8A16 下 Spatial 86.0% > FP16 84.7%，说明 selective layout 这个思路本身是通用的。

---

## 4. Intuition building：为什么这套设计是"对的"

我觉得这篇 paper 最漂亮的地方在于把 quantization error 的 propagation path 压缩到两个标量乘积 $s_q s_k$ 和 $s_v s_o$ 上，然后直接用两个 scalar $\alpha, \beta$ 把它们拉回去。这种做法之所以有效，可以从几个角度理解：

### 4.1 信息论角度
对 attention 来说，softmax 输出的 distribution 的 entropy 由 logits 的 std 决定（在 mean-zero 假设下）。teacher 训练时模型已经学到了特定的 attention entropy 分布；quantization 改变 std 等于 shift 了 entropy，attention pattern 就偏离了 data distribution 上训练出的最优解。ATM 把 std 拉回去，本质上是在恢复 teacher 的 attention entropy schedule。

### 4.2 残差流形几何角度
Transformer 的 residual stream 可以看作一个高维向量空间，每一层都是在这个空间里加一个 update vector。LayerNorm 的 operating point（即 mean/std）决定了 update 的 effective magnitude。如果 $s_v s_o$ 偏了，每层 update 的 norm 都偏，相当于在 residual stream 上做了一个 systematic scaling，几十层下来就把 trajectory 推到训练分布外的区域。OHB 直接对每层 update norm 做归一化，等于把 trajectory 拉回训练流形。

### 4.3 为什么是 per-head / per-layer 而不是 global
$\alpha$ 用 per-head 是因为不同 head 学到了不同的 attention pattern，有的 head 本来就 sharp（large logits std），有的 head 本来就 flat（small logits std）。quantization 对每个 head 的 std perturbation 大小不同，必须 per-head 修。

$\beta$ 用 per-layer 是因为不同深度层的 residual injection gain 不一样——浅层更多是 representation building，深层更多是 task-specific refinement，每层在 residual stream 里的能量贡献分布不同。

### 4.4 为什么 DiT attention proj 留 FP
attention proj 之所以敏感，是因为它直接接 softmax——softmax 是高度非线性的，对小 input perturbation 会放大。MLP 的 GELU/SiLU 虽然也非线性，但相对于 softmax 更温和，且 MLP 通道间独立，没有 softmax 那种全局耦合。所以同样的 quantization error 加在 attention proj 上比加在 MLP 上后果严重得多。这个 design choice 完全被 Table 1 验证了。

---

## 5. 一些可能的延伸和疑问

1. **W4A4 的进一步极限**：现在 W4A4 在 Long 上掉到 90.5%，还能不能 push 到 W4A4 + 更长 horizon？可能需要更精细的 per-timestep $\beta$（不同 diffusion step 给不同 $\beta$）。
2. **vision encoder 也 quantize 会怎样**：paper 故意 keep vision frozen。vision encoder 是另一个 memory大户，特别是 SigLIP2 / DINOv2 这种大 backbone。如果 vision encoder 也 quantize，$F_{VL}$ 本身的分布也漂了，ATM/OHB 还够不够用？
3. **real robot transfer**：LIBERO 是 sim，sim 上的 success rate 和 real robot 上的 success gap 往往不小。量化在 sim 里略涨点不等于 real 上也涨，需要 real-world deployment 验证。
4. **跟 KIVI / KV-quant 的结合**：VLA rollout 时 KV cache 也是 memory 大头，QuantVLA 关注 weights + activations，没碰 KV cache quantization，可以叠加上 KIVI (https://arxiv.org/abs/2402.02750) 这类方法。
5. **跟 Eagle / speculative decoding 的结合**：如果 DiT flow matching 可以做 speculative step（一次性预测多步），叠加 quantization 可以再省 inference latency。
6. **为什么 ATM 公式 (12) 写成 $L_Q = L_T / \alpha$**：这里我怀疑是 paper 的笔误，因为按定义 $\alpha = \text{Std}(L_T)/\text{Std}(L_Q)$，要让 student 的 std 等于 teacher 的 std，应该是 $L_Q^{\text{corrected}} = \alpha \cdot L_Q^{\text{orig}}$，而不是 $L_Q = L_T / \alpha$。但语义上理解就是用 $\alpha$ 折算 student logits 让其 std 对齐 teacher，作者实现里应该是把 $\alpha$ fold 到 $s_q$ 或 $s_k$ 上，最终效果一致。

---

## 6. 总结性的 mental model

把 QuantVLA 记成三件事就行：

1. **Layout 上做"半量化"**：DiT attention proj 留 FP，剩下能 integerize 的全 integerize。这是 "在刀刃上留 precision" 的极端版本。
2. **ATM 修 attention temperature**：每 head 一个 scalar $\alpha$，让 student logits 的 std = teacher logits 的 std。本质是把 quantization 引入的 $s_q s_k$ 偏差用 $\alpha$ 补偿。
3. **OHB 修 residual energy**：每层一个 scalar $\beta$，让 student output RMS = teacher output RMS。本质是把 quantization 引入的 $s_v s_o$ 偏差用 $\beta$ 补偿。

三个 component 都 training-free、architecture-preserving、calibration-only、inference-time 零开销。最终在 LIBERO 上跑赢 FP16 + 70% memory saving，给 VLA 在机器人端的真实部署打开了一条路。

未来如果有人要 push 到 W2A4 或者直接 binary VLA，这套 ATM/OHB 的思路大概率还要再升级——per-head/per-layer scalar 可能不够，需要 per-timestep、per-channel-group 的 finer-grained calibration。但作为 "VLA PTQ 的第一次正确打开方式"，这篇 paper 的 contribution 是清晰的。
