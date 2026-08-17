---
source_pdf: OSCAR Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization.pdf
paper_sha256: 71668cd74b2255834ff147a4639bbdf83e639a084a7b95042ca711f5c3d40d3b
processed_at: '2026-08-06T01:33:18-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 OSCAR

## 一句话

KV cache 压到 2 bit 只有 4 个 level, 直接压必崩. Hadamard rotation 能救一点但救不动, 因为它 "盲转". OSCAR 说: 你要问 attention 本人, 它到底在看 K 和 V 的哪些方向. 把那些方向标出来, 把 quantization error 推到 attention 看不见的角落里, 2 bit 就能用了.

---

## 为什么 KV cache 压 2 bit 这么难

每次 decode 一个 token, GPU 要把整个 KV cache 从 HBM 读一遍. 16 bit -> 2 bit 理论上 8× 省 memory + 8× 省 bandwidth, 这是 long-context serving 的圣杯配置.

但 2 bit 只有 4 个 level. KV activation 里有 channel-wise outlier——个别 channel 的 magnitude 是普通 channel 的几十倍. 量化 scale 被这个 outlier 拉到极大, 剩下 99% 的正常值全挤在 4 个 level 里, 几乎分辨不出来.

直接 round → 崩. Table 2 里 naive INT2 在所有 model 上 mean acc = 0.00.

## Hadamard rotation 为什么 4 bit 够用、2 bit 不够

Hadamard 是一个所有 entry 都是 ±1/√d 的正交矩阵. 把 K 乘一个 Hadamard, 等于把每个 channel 重新表示成 "所有原始 channel 的 ±1 加权和". 一个 outlier spike 会被打散成 d 个小贡献, 每个 ~1/√d. 这样 per-channel dynamic range 就均匀了, quantizer 看到的 distribution 就 friendly 了.

这套在 4 bit (16 level) 够用. 但到 2 bit (4 level), Table 2 显示 QuaRot 在 Qwen3-4B 上从 75.64 掉到 1.40——直接归零.

paper 的核心诊断是: **Hadamard 是 data-oblivious 的 "盲转"**. 它把 outlier 抹平了, 但是它不知道 attention 真正在意哪些方向. 4 level 的 budget 太紧, 必须把 error 精确推到 attention "视野盲区"里, 否则 logits 就会出大问题.

## Attention 到底在意什么

这是 paper 最关键的 insight, build intuition 的核心.

KV cache 的下游不是 "重建 K 和 V", 而是 "算 attention". Attention 用 K 是通过 $QK^\top$ 这个乘法——**K 的 error 被 $Q^\top Q$ 加权**. Attention 用 V 是通过 $SV$ 这个加权聚合——**V 的 error 被 $S^\top S$ 加权**.

所以真正该 minimize 的不是 $\|K - \mathcal{Q}(K)\|_F^2$, 而是

$$
\|QK^\top - Q\widehat{K}^\top\|_F^2 = \mathrm{tr}\big((K-\widehat{K})\, Q^\top Q\, (K-\widehat{K})^\top\big)
$$

这里 $Q^\top Q$ 才是 "度量张量". 在 $Q^\top Q$ 大的方向上, K 的 error 会被严重放大; 在 $Q^\top Q$ 小的方向上, K 量化得多烂都无所谓.

直觉: **query 在某些方向上 "问得狠", 那些方向上的 key 必须精确; query 几乎不问的方向, key 随便量化**.

V 同理, 但更绕一层——V 的下游是 $SV$, score-weighted aggregation, 所以度量是 $V^\top S^\top S V$, 这是 attention 已经聚合过的 value 的二阶矩.

## OSCAR 怎么把 error 推到盲区

离线跑一次 calibration (几千 token, 一次 forward), 对每一层 dump 出 Q, K, V, 算两个 matrix:

- Key 侧: $C_Q = \frac{1}{N}\sum_n q_n^\top q_n$ —— **query covariance**
- Value 侧: $C_S = \frac{1}{N} V^\top S^\top S V$ —— **score-weighted value covariance**

对 $C_Q$ 做 eigendecomposition $C_Q = U_Q \Lambda_Q U_Q^\top$. $\Lambda_Q$ 的特征值从大到小排: 大特征值方向是 "query 问得狠的方向", 小特征值方向是 "query 不关心的方向".

如果只做这一步: $R_K = U_Q$. 在 rotated basis 下, 第 j 个 channel 的 attention 重要性 = $\lambda_j$. 大 $\lambda$ channel 要保护好, 小 $\lambda$ channel 可以吃 error.

但 paper 发现只做 $U_Q$ 单独是不行的——因为 $Q^\top Q$ 和 $K^\top K$ 在真实 LLM 里 **几乎不共享 eigenbasis** (paper 实测 alignment 只有 0.05~0.15, 跟 random 的 0.09 一样). 也就是说, query 关心的方向和 K 自身能量大的方向是两套坐标系. 只对齐 $U_Q$, K 自身的 outlier 反而被 push 到低 $\lambda$ 方向——但 quantizer 看到的是 K 的 distribution, per-group dynamic range 反而变差, Table 10 显示 residual 涨 6 倍.

## 三个因子为什么缺一不可

OSCAR 最终的 rotation 是三个 orthogonal matrix 的乘积: $R_K = U_Q \cdot H_\mathrm{Had} \cdot P_\mathrm{br}$.

**$U_Q$**: 把 importance 度量 $Q^\top Q$ 对角化, 每个 channel 的 "attention 权重" 变成精确的 $\lambda_j$. 这是 "标出哪些方向重要" 的工作.

**$H_\mathrm{Had}$**: 两个作用. 
- 数学上: Lemma 1 给了个漂亮的恒等式—— 对任何对角矩阵 $\Lambda$, $H^\top \Lambda H$ 的对角线 **全是** $\mathrm{tr}(\Lambda)/d$. 也就是说, Hadamard 作用在已经对角化的 $C_Q$ 上, importance 对角线会精确均匀化, max/mean 从 46.9 压到 1.00. 
- 工程上: K 的 outlier 会被 ±1 signed sum 打散, per-token max 按 1/√d 缩.

**$P_\mathrm{br}$ (bit-reversal permutation)**: 因为 quantizer 是 per-group 的 (group size 64, 一个 head 128 维分 2 组). Hadamard 抹平了 marginal variance, 但 per-group dynamic range 还看 "这个 group 恰好抽到哪些 high-energy eigenchannel". Bit-reversal 是一个很巧的排列: 把 top-1 eigenvector 放位置 0, top-2 放位置 64, top-3 放位置 32, top-4 放位置 96... 这样对任何 power-of-two group size, top eigenvector 都会被均匀分到不同 group 里. Table 10 显示这步贡献了 OSCAR 对 $U_Q H_\mathrm{Had}$ 的主要增益 (residual 从 208 降到 169).

三个因子各自解决一个 failure mode, 互不干扰. 这是 paper 最 elegant 的部分.

## 系统层面: 怎么真的 deploy 到 SGLang

光有算法不够. OSCAR 还做了三件事让它能跑 production:

**1. 三段式 KV cache layout**: 
```
[sink, BF16] || [history, INT2 rotated] || [recent, BF16]
   前 64 个          中间全部             最近 256 个
```
sink tokens 是 attention sink, 丢精度必崩. recent 256 是新生成的, error 会立刻 propagate. 这两段保 BF16, 占 KV 总量的 0.24%, 几乎免费, 但精度回报巨大 (Table 5: (0,0) 配置精度全归零).

**2. Fused Triton kernel**: prefill 时把 rotate + clip + INT2 quantize 全 fuse 一次写完. decode 时三个 kernel: 一个处理 INT2 history, 一个处理 BF16 sink+recent, 一个 merge. 因为 BF16 段比 INT2 段小几个数量级, overhead 可忽略.

**3. Value rotation absorption**: 把 $R_V$ 直接吸进 model 的 $W_O$ projection weight 里. 这样 decode 时 V 不用再 rotate-back, 直接 dequant 就能用. 省一次矩阵乘.

## 实验数字到底有多好

Table 2 主结果:

| Model | BF16 mean | QuaRot-INT2 | OSCAR (2.28 BPE) |
|---|---|---|---|
| Qwen3-4B-Thinking | 75.64 | 1.40 (-74.2) | 71.86 (-3.78) |
| Qwen3-8B | 70.84 | 10.14 (-60.7) | 69.42 (-1.42) |
| Qwen3-32B | 74.19 | 7.90 (-66.3) | 74.17 (-0.02) |
| GLM-4.7 (358B) | 77.89 | 75.14 (-2.75) | 78.16 (+0.27) |

OSCAR 在 ~2 bit 下基本追平 BF16, 32B 和 358B 上甚至 tie 或略高. QuaRot 在小模型上直接归零.

long-context (RULER-NIAH 128K): OSCAR 在 128K context 还能拿 39~45 分 (Qwen3-4B/8B), QuaRot 归零. GLM-4.7 上 128K = 97.2, 跟 BF16 完全一样.

throughput: batch=1 decode 在 100K context 上比 BF16 快 3×; batch=32 在 GLM-4.7 上快 7.8×. memory 省 8×, 单 H100 能 scale 到 256 并发 (BF16 几十就 OOM).

## Ablation 里最震撼的一行

Table 4 有一个对比特别说明问题:

| Configuration | Mean Acc |
|---|---|
| Full OSCAR (attention-aware target $C_Q$) | 70.01 |
| OSCAR with $K^\top K$ / $V^\top V$ target (raw reconstruction) | 31.12 |

**同样是做 PCA + Hadamard + bit-reversal, 只是把 "对谁做 PCA" 从 $Q^\top Q$ 换成 $K^\top K$, 精度从 70 掉到 31**. 

这是对 paper 核心 thesis 的直接验证: 问题不在于 "要不要做 PCA", 而在于 "对哪个 matrix 做 PCA". 你要对的 matrix 必须是 attention 真正消费的那个, 不是 cache tensor 本身的二阶矩.

## 这个工作真正 "teach" 了什么

build my intuition 的话, OSCAR 教给我的最重要的 lesson 是:

**Quantization error minimization 是一个 weighted problem, 权重由下游计算决定.** 

你 minimize 的 objective 应该长成 $\mathrm{tr}(E \cdot M)$ 其中 $E$ 是 error covariance, $M$ 是 downstream-induced metric. 对 attention 来说, K 的 $M = Q^\top Q$, V 的 $M \approx V^\top S^\top S V$. Raw reconstruction 用 $M = I$ 是错的, 因为它假设所有方向同等重要——这跟 attention 的实际行为完全相反.

这个 insight 应该可以推广到很多地方: weight quantization 应该用 gradient/Hessian 作为 metric; activation quantization 应该用下一层 weight 的 covariance 作为 metric; 甚至 patch-based image compression 应该用 "human visual attention" 作为 metric. 本质上都是一件事: **error budget 要花在 downstream 看不见的地方**.

OSCAR 只是把这个 principle 在 KV cache + attention 这个特定场景下做对了, 做得很漂亮, 而且 deploy 到 production. 这就是一篇好 paper 的样子.

---

# OSCAR 深度解读：让 INT2 KV Cache 量化既准确又可部署

下面我从直觉到数学，从算法到系统，从实验到 ablation 逐步展开这篇工作。核心一句话概括：**OSCAR 不再追求 raw K/V tensor 的 reconstruction error 最小，而是直接对齐到 attention 真正消费的"目标 covariance"——也就是 query-side covariance 和 score-weighted value covariance——由此推导出固定的 rotation，让 INT2 (4 个 level) 的误差被推到 attention 几乎看不见的方向上**。

---

## 1. 问题背景：为什么 INT2 KV cache 这么难

在 autoregressive decoding 中，每生成一个 token 都要把整个 KV cache 从 GPU HBM 读一遍。cache 体积随 context length、batch size、depth 线性增长。BF16 一个 element 16 bit；INT2 只有 4 个 level，理论上能拿到 8× memory 和 bandwidth 收益，这是 long-context serving 的"圣杯"配置。

但 INT2 直接 round 几乎一定崩。原因在于 KV activations 有严重的 **channel-wise outliers**：少数 channel 的 magnitude 是其他 channel 的几十倍。量化 scale 被这些 outlier 拉到极大，绝大多数正常值被压在 4 个 level 里几乎分辨不出。

**Rotation 是经典的解决思路**：QuaRot [1] / QuIP# [2] / SpinQuant [3] 等用 Hadamard 或 learned orthogonal transform 把 outlier energy 在 d 个 channel 间 redistribute，让 dynamic range 均匀。但 paper 在 Table 2 显示：在 INT2 下，QuaRot 在 Qwen3-4B 上从 BF16 的 75.64 掉到 1.40（几乎归零），在 Qwen3-8B 上掉到 10.14。也就是说 **data-oblivious rotation 在 4-bit 够用，在 2-bit 直接崩**。

为什么崩？paper 的核心诊断是：

> Generic rotation 把 outlier 抹平了，但是它不知道 attention 真正在意哪些方向。INT2 只有 4 个 level，**必须把误差推到 attention 读不强烈的方向上**，否则 logits/output 就会出大问题。

---

## 2. 核心数学动机：为什么 raw reconstruction 是错的 objective

这是 paper 的最关键 insight，build intuition 必须从这里开始。

### 2.1 Key 的 downstream 是 logit，不是 K 本身

Attention logit 形式是 $QK^\top$。如果量化后得到 $\widehat{K}$，那么 logit distortion 的 Frobenius 范数是

$$
\|QK^\top - Q\widehat{K}^\top\|_F^2 = \mathrm{tr}\Big((K-\widehat{K})\,Q^\top Q\,(K-\widehat{K})^\top\Big)
$$

变量解释：
- $Q \in \mathbb{R}^{T \times d}$：query 矩阵，T 是 sequence length，d 是 head dim
- $K, \widehat{K} \in \mathbb{R}^{T \times d}$：原始和量化重建后的 key
- $Q^\top Q \in \mathbb{R}^{d \times d}$：**query covariance**——这才是 key error 真正被加权的"度量张量"
- $\mathrm{tr}(\cdot)$：矩阵 trace

注意 $(K-\widehat{K})$ 这个残差矩阵的 Frobenius 范数单独看是 $\|K-\widehat{K}\|_F^2 = \mathrm{tr}((K-\widehat{K})(K-\widehat{K})^\top)$——也就是说 raw reconstruction error 用的是 $I_d$ 作为度量。但 attention 真正用的度量是 $Q^\top Q$，**两个完全不同**。

直观上：query 在某些方向上有大能量（这些方向是 model 真正在"询问"的方向），key 在这些方向上的 error 会被放大；在 query 几乎没有能量的方向上，key 量化得再差也无所谓。

### 2.2 Value 的 downstream 是 score-weighted 聚合

Value 的下游 distortion：

$$
\|SV - S\widehat{V}\|_F^2 = \mathrm{tr}\Big((V-\widehat{V})^\top\, S^\top S\, (V-\widehat{V})\Big)
$$

变量解释：
- $S = \mathrm{softmax}_\mathrm{row}(QK^\top/\sqrt{d}) \in \mathbb{R}^{T \times T}$：attention score
- $S^\top S \in \mathbb{R}^{T \times T}$：score 的二阶统计
- 完整度量张量是 $V^\top S^\top S V$，这是 attention-weighted value 的二阶矩

也就是说，value 的"重要性方向"是 $C_S = V^\top S^\top S V$，而不是 raw value covariance $V^\top V$。

### 2.3 把这两个度量结合起来 → OSCAR 的 target covariance

- **Key 的 target covariance**: $C_Q = \frac{1}{N}\sum_{n=1}^N q_n^\top q_n$  
- **Value 的 target covariance**: $C_S = \frac{1}{N} V^\top S^\top S V$

这两个 matrix 才是 attention "真正看到的"二阶统计。OSCAR 不去 minimze $\|K-\mathcal{Q}(K)\|_F^2$，而是去 diagonalize $C_Q$ 和 $C_S$，把 rotation 对齐到它们的主方向。

Figure 2 用四个 subfigure 验证了这套 logic：raw K/V 的 MSE 上 OSCAR 不一定比 Hadamard-only 好多少，但 $QK^\top$ 的 MSE、$SV$ 的 MSE、attention output MSE、跨 layer hidden state MSE，OSCAR 全程领先。这直接对应 paper 的核心 claim："which covariance matrix one diagonalizes matters more than whether one diagonalizes at all."

---

## 3. OSCAR 的 rotation 结构：三个因子为什么是它们

OSCAR 的 rotation 是三个 orthogonal factor 的乘积：

$$
R_K = U_Q \cdot H_\mathrm{Had} \cdot P_\mathrm{br}
$$

同样地 $R_V = U_S \cdot H_\mathrm{Had} \cdot P_\mathrm{br}$。三个因子各自解决一个 failure mode，且彼此正交。Appendix A.4 给了非常细致的解释，我把它再展开讲清楚。

### 3.1 第一个因子 $U_Q$：把 importance 显式对角化

$C_Q = U_Q \Lambda_Q U_Q^\top$，其中 $\Lambda_Q = \mathrm{diag}(\lambda_1, \ldots, \lambda_d)$，$\lambda_1 \geq \cdots \geq \lambda_d \geq 0$。

设 $R_K = U_Q$（即只做这一步），那么 rotated importance metric 变成

$$
R_K^\top C_Q R_K = U_Q^\top C_Q U_Q = \Lambda_Q
$$

完全对角化。第 j 个 channel 的"attention 重要性权重"就是 $\lambda_j$ 本身。第 j 个 channel 上的 quantization error 会以 $\lambda_j$ 的权重 propagate 到 logits。**大 $\lambda$ channel 是必须保护好的方向，小 $\lambda$ channel 可以承担更多误差**。

但 paper 在 Appendix A.4 的"empirical aside"里给了一个非常重要的观察：**$C_Q = Q^\top Q$ 和 $\Sigma_K = K^\top K$ 在真实 LLM 中几乎不共享 eigenbasis**。Qwen3-8B 上测下来 top-8 eigenvector 的 self-alignment 只有 0.05~0.15，跟 random 的 $1/\sqrt{d} \approx 0.09$ 几乎一样。换句话说，"query 关注的方向"和"key 自身能量大的方向"是两个完全不相关的坐标系。

后果是什么？只做 $U_Q$ rotation，K 自身的 outlier 反而会被 push 到低 $\lambda$ 的特征方向里——但 min-max quantizer 看到的是 K 的 distribution，所以 per-group dynamic range 反而**变得更糟**。Table 10 里能看到：raw 的 per-group max-min 是 25.7，只做 $U_Q$ 后变成 22.4（略好），但 $\mathrm{tr}(E_K)$ 从 233 涨到 1354（爆了 6 倍）。

所以 $U_Q$ 单独用是不行的。它必须配 Hadamard。

### 3.2 第二个因子 $H_\mathrm{Had}$：让 importance 对角线"均匀化" + 抹平 K outlier

Walsh-Hadamard matrix 的每个 entry 都是 $\pm 1/\sqrt{d}$，正交。它有两个作用。

**作用 1（精确的）**：Lemma 1 给了一个非常漂亮的恒等式——

$$
\big(H_\mathrm{Had}^\top \Lambda H_\mathrm{Had}\big)_{ii} = \frac{1}{d}\,\mathrm{tr}(\Lambda), \quad \forall i
$$

证明就一行：Hadamard 每个元素 magnitude 是 $1/\sqrt{d}$，所以 $(H^\top \Lambda H)_{ii} = \sum_j H_{ji}^2 \Lambda_{jj} = \frac{1}{d}\sum_j \Lambda_{jj}$。

含义：在 $U_Q$ 把 $C_Q$ 对角化之后，再乘 $H_\mathrm{Had}$，importance metric 的对角线**完全均匀**，每个 channel 都有相同的 $\mathrm{tr}(C_Q)/d$。paper 里 Table 10 显示 max/mean 从 46.9 压到 1.00——这是**精确的代数恒等式**，不依赖任何分布假设。

**作用 2（heuristic 的）**：$K \cdot U_Q H_\mathrm{Had}$ 的每一行是 K 在 $U_Q$ eigenbasis 上投影后，再做一次 ±1 signed sum（每个系数 $\pm 1/\sqrt{d}$）。原本某个 principal direction 上的 spike（比如 29.77）会被打散成 128 个 $\pm 29.77/\sqrt{128} \approx \pm 2.6$ 的和，per-token max 大致按 $1/\sqrt{d}$ 缩小。Table 10 里 max $|\tilde{K}|$ 从 $U_Q$ 单独时的 29.4 降到 $U_Q H_\mathrm{Had}$ 时的 7.46。

这两个作用是**正交**的：Lemma 1 不要求 $\Sigma_K$ 和 $C_Q$ 共享 eigenbasis，只要 $\Lambda_Q$ 是对角就行；Hadamard 抹平 outlier 也不要求 $U_Q$ 对齐 $\Sigma_K$。这是为什么 paper 强调"compose without interference"。

但单独用 $H_\mathrm{Had}$（不带 $U_Q$，即 QuaRot 风格）会怎样？importance metric $C_Q$ 没有被对角化，Lemma 1 不适用。Table 10 显示 pure Hadamard 的 max/mean = 1.72，意味着某些 channel 的 importance 是平均的 1.72 倍——误差可能落在这些 channel 上。而 $U_Q H_\mathrm{Had}$ 的组合是 1.00，这是 OSCAR 对 QuaRot 的关键优势。

### 3.3 第三个因子 $P_\mathrm{br}$：bit-reversal permutation 让 group 间平衡

OSCAR 用 per-group INT2 quantization，group size $G_K = 64$（典型情况，即每个 128 维 head 分成 2 个 group）。每个 group 用自己的 min-max scale。

Hadamard 让 marginal channel variance 均匀了，但 per-group 的 dynamic range 还取决于"这个 group 里恰好抽到了哪些 high-energy eigenchannel"。如果某 group 偶然塞了多个 top-$\lambda$ 方向，它的 dynamic range 就会偏大。

**Bit-reversal permutation** 的 trick：把第 k 大 eigenvalue 放到位置 $\beta(k)$（k 的二进制 bit 反转）。例如 d=128 时：

- top-1 → 位置 0
- top-2 → 位置 64（不同 group）
- top-3 → 位置 32
- top-4 → 位置 96
- top-5~8 → 位置 16, 80, 48, 112

关键性质：**对任意 2 的幂 group size $G \mid d$，top-$d/G$ 个 eigenvector 会被均匀分到 $d/G$ 个不同 group 中**，每个 group 恰好分到一个。这种"递归平衡"对所有 power-of-two group size 都成立。

效果在 Table 10 里能直接看到：从 $U_Q H_\mathrm{Had}$ 到 $U_Q H_\mathrm{Had} P_\mathrm{br}$，$\mathrm{tr}(E_K)$ 从 208 降到 169（**1.23× 提升**，是 OSCAR 在这一 layer 上对 $U_Q H_\mathrm{Had}$ 的主要增益来源）。当 $G_K = d$ 时 PBR 是 no-op，所以 PBR 只在更细的 grouping 下有作用。

### 3.4 三个因子的"分工"总结

| 因子 | 作用 | 数学保证 |
|---|---|---|
| $U_Q$ | 把 importance 度量对角化，channel j 的 attention 权重 = $\lambda_j$ | 精确（特征分解） |
| $H_\mathrm{Had}$ | importance 对角线 → 常数 $\mathrm{tr}(C_Q)/d$；K outlier 按 $1/\sqrt{d}$ 缩 | Lemma 1 精确；outlier 抹平 heuristic |
| $P_\mathrm{br}$ | top eigenvector 跨 group 均匀分布 | 任意 power-of-two group size 都成立 |

这种"三因子分工"的设计是 paper 最 elegant 的部分之一。

---

## 4. Theorem 1：为什么 $U_Q$ 是"对的"rotation

paper 给了一个理论保证。简化陈述：

**定理 1 (informal)**：考虑 frozen-error surrogate $\tilde{\mathcal{L}}_K(R_k) = \mathrm{tr}(R_k^\top C_Q R_k E_K)$，约束 $R_k^\top R_k = I_d$。假设 frozen residual covariance $E_K = \mathrm{diag}(\mu_1, \ldots, \mu_d)$ 且 $\mu_1 \leq \cdots \leq \mu_d$（即残差能量在 ambient basis 上对角、递增）。那么 $R_k = U_Q$ 是 minimizer。

变量解释：
- $E_K$：量化残差 $\mathcal{Q}(k_j R_k) - k_j R_k$ 的二阶矩，假设"ambient basis diagonal"
- $\mu_j$：第 j 个 channel 上的残差能量，按递增排序
- $\lambda_j$：$C_Q$ 的特征值，按递减排序

证明的 key idea 是 **rearrangement inequality**：展开 $\tilde{\mathcal{L}}_K = \sum_{i,j} \mu_i \lambda_j z_{ji}^2$，其中 $Z$ 是 doubly stochastic（Birkhoff polytope 极点 = permutation matrix）。最小化时应该让大的 $\lambda$ 配小的 $\mu$，identity permutation 是 minimizer。

**这个定理有两个 caveat**，paper 在 Discussion 里也诚实地承认了：

1. "Ambient-basis diagonal residual" 是一个强假设——真实残差不一定在 channel basis 对角。但 paper 解释说这是"frozen-error surrogate"——即在固定 rotation 下估计残差，是一种"first-order approximation"。
2. Theorem 给的是 surrogate 的最优，不是端到端 autoregressive decoding 的最优。完整的理论保证是 future work。

即便如此，定理的价值是：**它告诉我们为什么应该 diagonalize $C_Q$ 而不是 $K^\top K$**——前者最小化的是 attention 真正关心的 weighted error，后者最小化的是 raw reconstruction error。

---

## 5. 系统设计：让 INT2 在 production SGLang 里跑起来

光有算法不够，OSCAR 还做了非常细致的系统工作。这是 paper 一个 underrated 的部分。

### 5.1 三段式 KV cache layout

```
[sink, BF16] || [history, INT2 with rotation] || [recent, BF16]
  位置 1~S₀        位置 S₀+1 ~ t-W              位置 t-W+1 ~ t
```

- $S_0 = 64$：attention sink tokens，前几十个 token 是 model 用来"建立 context"的，必须保护（StreamingLLM [4] 发现的现象）
- $W = 256$：recent window，最近生成的 token 也要保护，因为它们的 quantization error 会立刻 propagate
- 中间区域是 INT2 packed（4 个 2-bit value pack 进一个 byte）

额外 BF16 占比 $(S_0+W)/128\text{K} \approx 0.24\%$，几乎免费，但精度回报巨大（Table 5 显示 $(0,0)$ 配置下精度全归零）。

### 5.2 三个 fused Triton kernel

- **Prefill kernel**：把 $k_t R_K$、clip、INT2 quantize 全部 fuse 在一次写 cache 的 kernel 里。$k_t^+ = \mathcal{Q}_2^+(\mathrm{clip}(k_t R_K, \tau_t^{(K)}))$
- **Decode demote kernel**：当 recent window 满了，最老的 recent token 被"降级"成 INT2 history
- **Decode attention kernel**：OSCAR 在 Flash-Decoding [5] 的 two-kernel 结构（一个做 partial attention on KV segments，一个做 online softmax merge）基础上加第三个 kernel 处理 BF16 sink+recent 段，复用第二个 merge kernel。因为 BF16 段比 INT2 段小几个数量级，overhead 可忽略。

Table 8 的 profiling 数据：在 batch=32 下，OSCAR 的 attention 时间是 8.5 ms（占 54.6%），而 BF16 是 17.0 ms（占 72.7%）。OSCAR 在 attention 上节省一半多。Quantize overhead 只有 0.5 ms（占 3.2%）。

### 5.3 Value rotation 的 weight absorption trick

OSCAR 把 $R_V$ 吸收到 model 的 $W_O$ projection 里。这样 decode 时 V 不需要再做 rotation-back，直接读 INT2 dequant 就能用。这是 latency 优化的关键。

---

## 6. 实验：OSCAR 在 INT2 上接近 BF16

### 6.1 主结果 (Table 2)

| Model | Method | BPE | Mean Acc | Drop vs BF16 |
|---|---|---|---|---|
| Qwen3-4B-Thinking | BF16 | 16.00 | 75.64 | – |
| | QuaRot-INT2 | 2.25 | 1.40 | -74.24 |
| | **OSCAR** | 2.28 | 71.86 | **-3.78** |
| Qwen3-8B | BF16 | 16.00 | 70.84 | – |
| | QuaRot-INT2 | 2.25 | 10.14 | -60.70 |
| | **OSCAR** | 2.28 | 69.42 | **-1.42** |
| Qwen3-32B | BF16 | 16.00 | 74.19 | – |
| | QuaRot-INT2 | 2.25 | 7.90 | -66.29 |
| | **OSCAR** | 2.28 | 74.17 | **-0.02** |
| GLM-4.7-FP8 (358B) | BF16 | 16.00 | 77.89 | – |
| | QuaRot-INT2 | 2.25 | 75.14 | -2.75 |
| | **OSCAR** | 2.28 | 78.16 | **+0.27** |

观察：

1. **OSCAR 是唯一在 ~2 bit 下接近 BF16 的方法**。QuaRot-style rotation 在小模型上直接崩（4B 和 8B 上几乎归零），32B 上也掉 66 点。
2. **越大模型越 robust**。GLM-4.7 上 OSCAR 甚至比 BF16 高 0.27 点——这说明大模型对 KV 量化更不敏感，INT2 几乎免费。
3. **BPE 2.28 vs 2.25**：差异来自 OSCAR 用了 0.24% 的 BF16 sink/recent 保护。基本上是免费的。

### 6.2 Long-context (Table 3, RULER-NIAH)

OSCAR 在 128K context 下还能保留可观的 retrieval 精度：

- Qwen3-4B-Thinking：OSCAR 在 128k 还能拿 39.5（BF16 81.0），QuaRot 已经归零
- Qwen3-8B：OSCAR 在 128k 拿 45.0，QuaRot 归零
- GLM-4.7：OSCAR 128k = 97.2，BF16 = 97.2，**完全 on par**

这个结果验证了 paper 的核心论点：**long context 会把小的 KV 量化 error 累积放大**，所以"对齐 attention covariance"在 long context 下收益尤其大。Figure 3 的 KL 曲线显示 OSCAR 的 attention distribution 跟 FP16 几乎重合，QuaRot 在 16K 之后快速 drift。

### 6.3 Ablation: rotation decomposition (Table 4)

在 Qwen3-8B 上：

| Configuration | Mean Acc |
|---|---|
| Full OSCAR: $U \cdot H_\mathrm{Had} \cdot P_\mathrm{br}$ | 70.01 |
| w/o $P_\mathrm{br}$ | 68.00 (-2.0) |
| w/o $H_\mathrm{Had}$ (only $U \cdot P_\mathrm{br}$) | 51.74 (-18.3) |
| w/o $U$ (QuaRot + $P_\mathrm{br}$) | 32.82 (-37.2) |
| No rotation | 4.23 (-65.8) |
| OSCAR with $K^\top K / V^\top V$ PCA target | 31.12 (-38.9) |

最后那行特别重要：**如果用 raw-cache reconstruction target ($K^\top K$) 而不是 attention-aware target ($C_Q$)，精度直接掉到 31.12**。这是对核心 thesis 的直接验证——不是"做 PCA 就行"，而是"必须 PCA 对的 matrix"。

### 6.4 Sink + Recent window sweet spot (Table 5)

在 Qwen3-4B-Thinking 上：

| (S, R) | Mean Acc | Extra BF16 KV |
|---|---|---|
| (0, 0) | 0.00 | 0% |
| (32, 128) | 67.69 | 0.12% |
| **(64, 256)** | **71.86** | **0.24%** |
| (128, 512) | 72.96 | 0.49% |
| (256, 1024) | 73.08 | 0.98% |

明显的 knee 在 (64, 256)。更大窗口收益边际化但 BF16 占比翻 4 倍。

### 6.5 Calibration 数据敏感性 (Table 7)

只用 8K GPQA-Diamond tokens 做 calibration，效果就接近最优。WikiText 8k 也能用。MMLU 2k 略差但可接受。**这意味着 OSCAR 的部署成本极低**——一次 forward dump，per-layer Q/K/V 都拿到，然后离线算 rotation 就行。

### 6.6 系统 throughput (Figure 4, Table 9)

- **Pure decode speed** (batch=1, full prefix cache hit): OSCAR 在 100K context 上比 BF16 快 **3.08×**（Qwen3-4B），GLM-4.7 上 2.83×。这是因为 decode 是 memory-bandwidth bound，KV 从 16 bit 降到 2 bit 直接 8× 减少 traffic。
- **Batch scalability** (100K input): OSCAR 在单 H100 上能 scale 到 $2^8 = 256$ 并发请求（BF16 几十就 OOM），throughput 持续上升。
- **Job-level throughput**: GLM-4.7-FP8 上 BS=32 时 OSCAR 比 BF16 快 **7.83×**，per-GPU throughput 翻 7 倍多。
- **Prefix cache hit ratio**: OSCAR 在所有 cache hit ratio 下都接近效率 frontier，对 prefix cache 技术完全兼容。

---

## 7. 跟其他工作的关系

paper 的 Related Work 部分写得很细致，我把它整理一下：

### 7.1 KV cache compression 的几个 family

1. **Eviction / retention**：H2O [6]、SnapKV [7]、PyramidKV [8]、StreamingLLM [4]、ZipCache [9]——丢 token，不改 precision
2. **Low-rank / cross-layer sharing**：GEAR [10]、PALU [11]、xKV [12]、MatryoshkaKV [13]——用 SVD 之类压缩
3. **Low-bit quantization**：KIVI [14]、KVQuant [15]、WKVQuant [16]、SKVQ [17]、Kitty [18]、PM-KVQ [19]、RotateKV [20]——保持所有 token，降 precision

OSCAR 属于第 3 类，但跟其他 INT2 方法的区别是：**OSCAR 用 fixed token-wise transform + uniform paged layout**，而 KIVI/Kitty/SKVQ 等用了 channel-wise precision、residual buffer、progressive bit allocation、adaptive layout——这些都让集成到 paged KV-cache serving 和 fused kernel 变复杂。OSCAR 强调的是"serving-compatible"。

### 7.2 Rotation-based quantization

- QuaRot [1]：Hadamard transform 4-bit
- QuIP# [2]：Hadamard incoherence + lattice codebooks
- SpinQuant [3]：learned rotation，4-bit
- HALO [21]：Hadamard-assisted lower precision
- TurboQuant [22]：data-oblivious random rotation + Lloyd-Max + residual coding

OSCAR 跟它们的差异在于：**rotation target 是 attention-induced covariance，不是 raw-cache reconstruction**。Table 2 显示 TurboQuant (no mixed precision) 在 Qwen3-4B 上掉 43.9 点，OSCAR 只掉 3.78 点。

### 7.3 Covariance-aware calibration

- GPTQ [23]、QuIP [24]：用 Hessian 信息 guide weight rounding
- Drone [25]、ASVD [26]、SVD-LLM [27]、CorDA [28]：用 calibration activations guide SVD
- HaPPI [29]、CARE [30]、RecalKV [31]、CommonKV [32]：用 covariance-aware factorization 压缩 KV

OSCAR 的特殊性是：**calibration target 是 attention 诱导的 covariance ($Q^\top Q$ 和 $V^\top S^\top S V$)，而不是 raw activations 的二阶矩**。这是它跟 HaPPI 等工作的本质差异。

---

## 8. 局限与未解决问题

paper 在 Discussion 部分诚实地说了几点：

1. **理论保证是 surrogate 下的**：Theorem 1 是 frozen-error surrogate 的最优，不是完整 autoregressive decoding 的最优。Calibration 估计的 unbiased 性还没证。
2. **只探索 INT2**：没有探索 NVFP4、不同 weight 精度组合。
3. **Fixed bit-width**：没有 per-token bit allocation 或 thought-adaptive precision（ThinKV [33] 那条线）。
4. **硬件绑定 H100**：B200 等其他硬件需要重新 tuning kernel。
5. **跟 channel-wise 方法的正交性**：OSCAR 可以和 channel-wise / mixed-precision 方法组合，但组合后怎么保持 efficient serving layout 是 open problem。

---

## 9. 我对这个工作的几点直觉评论

- **"度量张量"的视角** 是这篇 paper 最 deep 的贡献。本质上 OSCAR 在说：quantization 误差最小化不是一个"几何问题"，而是一个"加权几何问题"，权重由下游计算决定。这个思路应该可以推广到其他 quantization 场景（weight quantization、activation quantization 都应该有对应的"downstream-induced metric"）。

- **三因子分解 ($U \cdot H \cdot P$) 是一个 elegant 的工程配方**。每个因子解决一个独立问题，互不干扰。Lemma 1 的代数恒等式（$H^\top \Lambda H$ 对角线恒等）是一个非常漂亮的小工具——它让 importance 度量精确均匀化，而不是依赖随机性。

- **Theorem 1 的"ambient diagonal residual"假设** 我觉得是 paper 最 weak 的地方。这个假设的真实性 paper 没有充分验证。但即使作为 surrogate，它解释了"为什么用 $C_Q$ 而不是 $K^\top K$"，这一点就够有价值了。

- **Value 的 target covariance $C_S = V^\top S^\top S V$ 是 heuristic 的**。paper 在 Appendix A.7 承认这一点：value 的下游 distortion 严格说是 $\sum_j e_j^{(V)} R_v^\top C_j R_v e_j^{(V) \top}$ 其中 $C_j = \sum_{i \leq j} S_{ij}^2$（position-dependent）。用 $C_S$ 替代是一种 mean-field 近似。这可能解释了为什么 value 的 target 在某些 layer 上效果略弱于 key。

- **Bit-reversal permutation 的使用是 small but clever trick**。它在 $G_K = d$ 时退化（no-op），但在 per-group INT2 下有 1.2× 左右的残差降低。这种"对 group structure 递归平衡"的思路可能可以推广到其他需要 fixed group 划分的场景（e.g., group convolution 的初始化）。

- **系统层面**的细节（fused kernel、weight absorption、三段式 layout）让 paper 从"算法 paper"变成"deployable system paper"。这一点是 OSCAR 跟很多纯算法工作最大的不同。Throughput 数字（7× on GLM-4.7 at BS=32）足以让 production team 严肃考虑部署。

---

## 10. References & 进一步阅读

主要参考论文：

- [1] QuaRot: https://arxiv.org/abs/2404.00456
- [2] QuIP#: https://arxiv.org/abs/2402.04396
- [3] SpinQuant: https://arxiv.org/abs/2405.16406
- [4] StreamingLLM / Attention Sinks: https://arxiv.org/abs/2309.17453
- [5] Flash-Decoding (PyTorch blog): https://pytorch.org/blog/flash-decoding/
- [6] H2O: https://arxiv.org/abs/2306.14048
- [7] SnapKV: https://arxiv.org/abs/2404.14469
- [8] PyramidKV: https://arxiv.org/abs/2406.02069
- [10] GEAR: https://arxiv.org/abs/2403.05527
- [11] PALU: https://arxiv.org/abs/2407.21118
- [12] xKV: https://arxiv.org/abs/2503.18893
- [14] KIVI: https://arxiv.org/abs/2402.02750
- [15] KVQuant: https://arxiv.org/abs/2402.09658
- [17] SKVQ: https://arxiv.org/abs/2405.06619 (CoLM 2024)
- [18] Kitty: https://arxiv.org/abs/2511.18643
- [20] RotateKV: https://arxiv.org/abs/2501.16383
- [22] TurboQuant: https://arxiv.org/abs/2504.19874
- [23] GPTQ: https://arxiv.org/abs/2210.17323
- [24] QuIP: https://arxiv.org/abs/2307.13305
- [25] Drone: https://arxiv.org/abs/2112.06398
- [26] ASVD: https://arxiv.org/abs/2312.05821
- [27] SVD-LLM: https://arxiv.org/abs/2403.07378
- [28] CorDA: https://arxiv.org/abs/2406.05223
- [30] CARE (ICLR 2026): https://openreview.net/forum?id=... (Covariance-Aware Rank-Enhanced, related to MLA)
- [33] ThinKV: https://arxiv.org/abs/2510.01290

Serving 框架相关：

- SGLang: https://arxiv.org/abs/2405.16514 (roughly), 项目主页 https://github.com/sgl-project/sglang
- PagedAttention / vLLM: https://arxiv.org/abs/2309.06180
- FlashAttention: https://arxiv.org/abs/2205.09866 (v1), https://arxiv.org/abs/2307.08691 (v2), https://arxiv.org/abs/2407.08608 (v3)
- Triton compiler: https:// Triton: https://dl.acm.org/doi/10.1145/3315508.3326490

经典理论参考：

- Rearrangement inequality: Hardy-Littlewood-Pólya, "Inequalities"
- Birkhoff-von Neumann theorem: doubly stochastic matrix polytope 的极点是 permutation matrix
- Ky Fan 1951 maximum principle (Proposition 1 in paper): https://www.pnas.org/doi/10.1073/pnas.37.11.760
- Walsh-Hadamard transform & FFT: Cooley-Tukey 1965 https://www.jstor.org/stable/2003357

OSCAR 代码（paper 提到 RotationZoo 项目）：见 paper Website/RotationZoo，但目前 arxiv version 没给直接 GitHub link，建议关注 https://github.com/sgl-project/sglang 的 PR 和 Together AI 官方 repo 跟进。

---

## 11. 一句话总结

OSCAR 的核心贡献可以浓缩成一句话：**KV cache 量化的目标不应该是最小化 $\|K - \mathcal{Q}(K)\|_F^2$，而应该是最小化 attention-induced metric 下的 distortion $\mathrm{tr}((K-\widehat{K}) Q^\top Q (K-\widehat{K})^\top)$——为此，离线 diagonalize $Q^\top Q$ 和 $V^\top S^\top S V$，组合 Hadamard 和 bit-reversal 得到固定 rotation，再配合 sink+recent BF16 保护，就能让 INT2 (2.28 BPE) 在 4B~358B 模型上几乎完全保留 BF16 精度，并在 SGLang production serving 上拿到 7× 吞吐**。这是一篇算法+系统协同设计的好范例。
