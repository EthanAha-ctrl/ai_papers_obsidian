---
source_pdf: EAGOR Embodied reAsoninG in Omni-diRection.pdf
paper_sha256: 9086824974d227d8308c6a1393b19d3bfda38c02096673a80f88654729aba954
processed_at: '2026-08-18T07:21:42-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EAGOR 用人话版

## 1. 一句话讲完它在干嘛

机器人身上挂个 360° 相机, 你问它 "我钥匙在哪", 它要回答 "钥匙在你左前方 30°"。以前大家都让 VLM 直接在一张被摊平的全景图上输出 pixel 坐标。EAGOR 说: 别这么干, 把"看图认东西"和"算它在哪"拆开, VLM 只负责说"这块区域可能是钥匙", 剩下的方向计算全部搬到球面上做, 用一套叫 spherical harmonics 的数学基底存"信念", 让机器人在转身、走动的时候, 这个信念能干净地跟着转、跟着累积, 不会因为图像重新采样而累积误差。

## 2. 以前做法为什么烂:橘子皮问题

360° 相机拍出来的是球面上的信息。大家习惯用 ERP 把球面摊成一张长方形全景图——这就像把橘子皮压平贴到墙上。压平必然带来三件事:

- 左右两边其实连在一起, 但在图上被剪开了 (seam)
- 上下靠近极点的地方被严重横向拉伸, 一个像素代表的方向被挤成一条线 (latitude distortion)
- 机器人一转身, 整张图要重新采样, 每采一次就引入一点插值误差, 走 10 步误差从 25° 涨到 90°

VLM 是在普通 perspective 图上 pretrain 的, 它看到 ERP 会犯傻: 它可能认出"那有个杯子", 但给的 pixel 坐标在几何上是不对的。而且每转一次身, 同一个杯子在 ERP 图上的 pixel 位置完全不一样, VLM 没法保证一致性。

paper 里 Fig. 2 画的就是这件事: 原本稳稳指向目标的方向, 走几步之后 centroid baseline 的预测就飘到天知道哪去了。

## 3. EAGOR 的核心思路:分两个部门

EAGOR 把任务拆成两个工种:

- **VLM 部门**: 只回答 "这张全景里, 哪些区域看起来像目标?" 输出一个 $[0,1]$ 的 response map, 像一张热力图。它可以说 "左边 0.7, 右下 0.3, 都可能", 允许多峰、允许模糊、允许错的。
- **几何部门**: 拿到这张热力图, lift 到球面上, 跟过去几帧的证据累加, 维护一个"目标在球面上每个方向的可能性"的连续场。机器人转身时把这个场一起转, 走到新位置时把新热力图叠进去。最后从这张场里读出一个方向作为答案。

VLM 完全 frozen, 不训练。它只提供 raw 证据。几何正确性全靠几何部门保证。这就是 training-free 的来源。

## 4. 为什么用球谐:球面上的"低音/中音/高音"

要在球面上存一张连续的概率场, 最朴素做法是存一个高分辨率 grid (比如 1024×512 的概率值)。但这有两个问题: 第一, 存储和计算量大; 第二, 旋转的时候又要插值, 又回到 ERP 的老毛病。

球谐 (spherical harmonics, SH) 是球面版的傅里叶基。就像一段声音可以用 "低音/中音/高音"几十个频段系数来描述一样, 球面上的任何标量场都可以用一组 SH 系数 $c_{\ell m}$ 来描述。paper 选 $L=7$, 一共 $(L+1)^2 = 64$ 个系数就够存一张 belief map。

这 64 个数字就是 agent 的"记忆状态"。比起存 50 万个像素, 压缩比 ~8000×。更重要的是, 后面所有操作都只在这 64 维上做, 不再碰 pixel grid。

公式长这样:

$$f_t(\omega) = \sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell} c_{\ell m}^{(t)}\, Y_\ell^m(\omega)$$

- $\omega$: 球面上一个方向 (单位向量)
- $f_t(\omega)$: t 时刻, "目标在 $\omega$ 方向"的 log-posterior 分数
- $\ell$: 阶, 决定角分辨率。$\ell=0$ 是整个球的平均值, $\ell=1$ 三个系数编码"这团 belief 整体偏向哪边", $\ell$ 越大越细
- $m$: 次, $-\ell \le m \le \ell$, 是绕 z 轴的角动量量子数
- $Y_\ell^m$: 球谐基函数, 球面上正交
- $c_{\ell m}^{(t)}$: 系数, 就是 agent 的 memory state

## 5. 转身怎么办:Wigner-D 的魔法

这是整篇 paper 最聪明的一步。

agent 从时刻 $t-1$ 走到 $t$, 转了一个旋转 $R_t$。belief 必须跟着转, 否则上一帧"目标在正前方"的信息就和新坐标系对不上了。

如果 belief 存在 pixel grid 上, 你要重新采样整张图, 每次重采样都引入插值误差——这就是 centroid baseline 飘的原因。

球谐有一个非常漂亮的定理: **对场做旋转, 等价于对每个 degree-$\ell$ 的系数块乘一个 Wigner-D 矩阵 $D^\ell(R_t)$**。

$$\tilde{c}_{\ell m}^{(t)} = \sum_{m'=-\ell}^{\ell} D^\ell_{mm'}(R_t)\, c_{\ell m'}^{(t-1)}$$

- $\tilde c_{\ell m}^{(t)}$: 转完之后, 新坐标系下的 belief 系数 (prior)
- $D^\ell_{mm'}(R_t)$: Wigner (real) D-matrix, 由旋转 $R_t$ 和阶数 $\ell$ 决定
- $c_{\ell m'}^{(t-1)}$: 上一帧的 posterior 系数

不同 $\ell$ 的 block 互相独立, 是个分块对角结构。平面导航时 $R_t$ 退化成 yaw 角度变化 $\Delta\psi_t$, Wigner-D 进一步退化成循环移位矩阵, 可以非常便宜地算。

打个比方: 你录一段音乐, 想把整个录音"挪个调", 你不需要重新演奏, 只要给每个频段做一个相位旋转就行。球谐也是一样的道理——你转 agent, 不需要重新算整张 belief, 只要在每个 $\ell$ block 上做一次矩阵乘法。

这是等变性 (equivariance) 的硬通货: 旋转后再做预测 = 先做预测再旋转, 数学上完全等价, 没有插值误差累积。

## 6. 多帧怎么累积:log 空间加法

每帧 VLM 给一张热力图, lift 到球面, 取 log, 投影到 SH 基底得到这一帧的 observation 系数 $b_{\ell m}^{(t)}$:

$$b_{\ell m}^{(t)} = \int_{S^2} r_t(\omega)\, Y_\ell^m(\omega)\, d\Omega$$

注意 $d\Omega = \cos\phi\, d\phi\, d\theta$, 这个 $\cos\phi$ 是球面面积元素, **它在积分时自动抵消了 ERP 在极点附近的过密采样**。这就是为什么 EAGOR 不需要 distortion-aware convolution——它在球面积分时天然就按正确几何加权了。

然后跟 propagated prior 直接相加:

$$c_{\ell m}^{(t)} = \tilde{c}_{\ell m}^{(t)} + b_{\ell m}^{(t)}$$

为什么能加? 因为我们在 log 空间, 概率相乘 = log 相加。这是 recursive Bayesian filter 的标准套路, 跟 Kalman filter 的 update step 同构, 只不过 Kalman 在欧氏空间用线性 transfer matrix, EAGOR 在 $S^2$ 上用 Wigner-D 做 predict、用 SH 系数加法做 update。

多帧累积带来的好处是: 单帧 VLM 错一次不要紧, 下一帧对的证据会加进来, 整体 belief 会收敛到正确方向。Table 1 显示 accuracy 从第 1 帧 8.5% 单调涨到 40.1% 就是这个证据累积的曲线。

## 7. 怎么读出答案:degree-1 球谐就是重心

更新完 belief 后, 要从场里取出一个方向。EAGOR 用球面 Fréchet mean:

$$\hat{\omega}_t^* = \frac{\mathbf{m}_t}{\|\mathbf{m}_t\|}, \quad \mathbf{m}_t = \int_{S^2} f_t(\omega)\, \omega\, d\Omega$$

- $\mathbf{m}_t \in \mathbb{R}^3$: belief 场的一阶矩 (加权向量均值)
- $\hat{\omega}_t^*$: 归一化后的单位方向, 就是答案

妙处在于: $S^2$ 上的 degree-1 球谐基 ($Y_1^{-1}, Y_1^0, Y_1^1$) 和 $\mathbb{R}^3$ 里的 vector 是同构的。所以 $\mathbf{m}_t$ 直接从 $c_{1,-1}, c_{1,0}, c_{1,1}$ 这三个系数解析算出, 不需要 grid search、不需要梯度上升。

只要 belief 是 unimodal 的, 这个 mean 就是 MAP 估计。如果是 multimodal (比如看到两把椅子), 这个 mean 会落在两个 mode 中间——这是 paper 自己承认的 multi-instance confusion 问题, Table 4 里 SR 33.3% 的来源。

resultant length $R_t = \|\mathbf{m}_t\| / \int f_t\, d\Omega \in [0,1]$ 顺手给了一个置信度度量, 类似 von Mises-Fisher 分布的 concentration parameter。$R_t$ 越接近 1, belief 越集中, agent 越自信。

## 8. 整个 pipeline 一图讲完

每个 timestep 做四件事:

1. **VLM 看 $I_t$** → response map $\ell_t(u,v)$ → lift 到球面 → log → SH 投影 → 得到 observation 系数 $\mathbf{b}^{(t)}$
2. **Predict**: 上一帧的 $\mathbf{c}^{(t-1)}$ 通过 Wigner-D 旋转, 得到 prior $\tilde{\mathbf{c}}^{(t)}$
3. **Update**: $\mathbf{c}^{(t)} = \tilde{\mathbf{c}}^{(t)} + \mathbf{b}^{(t)}$ (log 空间 Bayesian 加法)
4. **Decode**: 从 $c_{1,-1}, c_{1,0}, c_{1,1}$ 解析出 $\hat{\omega}_t^*$, 顺手得到置信度 $R_t$

这四步是一个完整的 recursive filter, 跟 Kalman filter 结构同构, 只不过状态空间从 $\mathbb{R}^n$ 换成了 $S^2$ 上的 SH 系数空间。

## 9. 效果用人话总结

- **HOS (Humanoid Object Search)**: Qwen2.5-VL-7B 加 EAGOR 从 27.24% 涨到 40.10%, 直接超过 12B 的 Gemma-3 裸模型。意思是: 64 个数字的几何外壳, 比把 backbone 翻一倍还赚。
- **OSR-Bench**: 同一个 7B 模型, 加 EAGOR 后 25.2%, 超过 10× 大小的 Qwen2.5-VL-72B 的 18.1%。
- **Map-Free Navigation**: 成功率从 82% 涨到 94%, seam 场景成功率从 19.6% 涨到 70.6% (3.6×)。因为球面上根本没有 seam。
- **Waypoint Following**: temporal consistency 从 7.3-19.2°/step 降到 2.6°/step (6.5× 改善)。这就是等变 propagation 的直接证据——agent 转身的时候 belief 平滑地跟着转, 不抖。
- **Real-world**: Unitree Go2 四足机器人 + Insta360 X3, training-free 直接部署, 静止/移动两种场景都能跟踪目标方向。

## 10. 它的边界在哪

paper 自己承认三种失败模式:

- **VLM 没认出目标** (rare target, SR 3.6%): 球面几何再正确也没用, semantic 源头就是 garbage in garbage out。改进方向是接 grounded detector 比如 Grounding DINO 补长尾。
- **Multi-instance confusion** (SR 33.3%): belief 是 multimodal, Fréchet mean 落在两个 mode 中间。改进方向是显式建模多 hypothesis, 或者用 mode-seeking 而非 mean。
- **OCR / fine-grained text** (SR 13.2%): VLM 本身读不准小字, 跟几何无关。

还有一个工程限制: 目前只对 rotation 等变, 没建模 translation 引起的 parallax。paper 用小步长把 parallax 压在单帧噪声范围内绕过去。大步长场景会崩, future work 说要做 translation-aware、多传感器融合版本。

## 11. 给你 build intuition 的几个比喻

- **ERP 像"压平橘子皮"**: 上下必然扯烂, 左右必然有缝, 转身必然重采样。
- **球谐系数像"EQ 旋钮"**: 64 个旋钮描述一张球面概率图。低阶旋钮 (小 $\ell$) 描述大尺度结构, 高阶旋钮 (大 $\ell$) 描述细节。
- **Wigner-D 像"挪调"**: 你转 agent, 等于把整段录音挪个调, 只需要给每个频段做相位旋转, 不需要重新演奏。
- **Log 加法像"投票"**: 每帧 VLM 投一次票, log 概率相加等价于概率相乘, 谁得票多谁赢, 一帧错不要紧, 多帧平均掉。
- **Degree-1 SH 像"重心"**: 这团 belief cloud 的重心在哪, 答案就在哪, 而且重心能从三个系数直接读出来。
- **跟 Kalman filter 同构**: predict 用 Wigner-D (球面版 state transition), update 用 SH 系数加法 (球面版 log-likelihood update)。状态空间从 $\mathbb{R}^n$ 换成 SH 系数空间。
- **跟 vMF filter 的关系**: vMF 是 unimodal 的, 用 mean + concentration 两参数表达。EAGOR 用 SH 场允许 multimodal, 代价是没有 closed-form, 要用 Wigner-D 做 predict。如果只做单目标追踪, vMF + EKF 可能更轻量; 如果要处理"看到三个候选"这种 ambiguous case, SH 才有价值。
- **跟 NeRF 的关系**: NeRF 在 3D 维护 radiance field, EAGOR 在 $S^2$ 维护 belief field。同样是"用 SH 表达场 + 用 Wigner-D 做等变操作"的 toolset。如果你想往后做 "agent 内部维护球面语义场", 这套数学直接可以搬。
- **跟 LLM chain-of-thought 的关系**: SH-BF 本质是一个结构化的 external memory。VLM 每帧只看一帧 (context = 1), 但 SH 系数跨帧持续更新, 等于一个"几何 scratchpad"。和让 LLM 把中间推理写在 context 里再读回来是同构思路, 只不过 scratchpad 是几何结构化的, 不是 free-form text。

## 12. 一句话总结

EAGOR 告诉你: **embodied 360° reasoning 的瓶颈不在 VLM 不够大, 在于你把几何状态塞进了 pixel 表示里**。把几何拎出来放到球谐系数上, 让 VLM 只管语义、让球面滤波器管几何, 64 个数字的外壳就能让 7B 模型干掉 72B。这是一篇"几何外壳套 VLM"的范本, 干净、数学扎实、training-free 还真 work。

进一步阅读:
- [Spherical harmonics - Wikipedia](https://en.wikipedia.org/wiki/Spherical_harmonics)
- [Wigner D-matrix - Wikipedia](https://en.wikipedia.org/wiki/Wigner_D-matrix)
- [Recursive Bayesian estimation - Wikipedia](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation)
- [Equirectangular projection - Wikipedia](https://en.wikipedia.org/wiki/Equirectangular_projection)
- [von Mises-Fisher distribution - Wikipedia](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)
- [SO(3) equivariant CNN (arXiv 1711.06721)](https://arxiv.org/abs/1711.06721)
- [Equiformer (arXiv 2206.11990)](https://arxiv.org/abs/2206.11990)
- [OSR-Bench (arXiv 2505.11907)](https://arxiv.org/abs/2505.11907)
- [HOS / Thinking in 360 (arXiv 2511.20351)](https://arxiv.org/abs/2511.20351)
- [Smooth SO(3) belief (arXiv 2006.01031)](https://arxiv.org/abs/2006.01031)
- [Unitree Go2](https://www.unitree.com/products/Go2)
- [Insta360 X3](https://www.insta360.com/product/insta360-x3)

---

# EAGOR：在球面上做 embodied directional reasoning

下面我把这篇paper拆开来聊，重点放在**为什么要在 sphere 上重新建模**、**SH-BF 的数学结构如何同时承载"加性证据累积"和"等变传播"两件事**、以及**这套几何先验为什么能让一个 7B 的 frozen VLM 超过 72B 的同族模型**。我会尽量把公式里每个符号讲清楚，方便你 build intuition。

---

## 1. 这篇 paper 真正想解决的问题：一个 representation gap

Embodied agent 拿 360° 相机，核心输出是**egocentric 的 agent-to-target direction** $\omega_t^* \in S^2$（一个单位向量，viewer-centric frame 下）。注意，这个量**不是 ERP image 里的 pixel 坐标**——pixel 坐标随相机姿态变化会发生 wrap、distortion、interpolation error。

**ERP（equirectangular projection）的根本毛病**有三个：
- **Seam discontinuity**：azimuth 在 $\theta = \pm\pi$ 处断开，左右边界像素其实指向相邻方向。
- **Latitude distortion**：靠近极点（$\phi \to \pm\pi/2$）的像素被严重横向拉伸，破坏 Euclidean convolution 假设。详见 [Equirectangular projection - Wikipedia](https://en.wikipedia.org/wiki/Equirectangular_projection)。
- **Interpolation drift under rotation**：agent 每转一次，pixel grid 重采样一次，方向估计误差逐步累积，Fig. 2 把这一点画得很清楚——centroid baseline 在 waypoint following 几段之后 MAE 飙到 90.9°。

VLM 的 backbone 是在 perspective image 上 pretrain 的，它对 ERP 输出语义响应没问题，但**它输出的 pixel-coordinate 不具备 rotation-equivariance**。这就是 paper 第 (I) 条 contribution 说的 representation gap：robotic action 要的是 motion-consistent 的 directional belief on $S^2$，而 VLM 给的是 ERP image-space 的 attention map。

**EAGOR 的哲学**：decouple "what to look for"（frozen VLM 干这个）和 "where the target lies"（SH-BF 在 sphere 上维护这个）。这是一个**training-free 的 two-stage pipeline**，把 VLM 当成 open-vocabulary 的方向证据发生器，把几何状态估计完全交给球面递推滤波器。

参考: [Equirectangular projection](https://en.wikipedia.org/wiki/Equirectangular_projection), [OSR-Bench (arXiv 2505.11907)](https://arxiv.org/abs/2505.11907), [Thinking in 360 / HOS (arXiv 2511.20351)](https://arxiv.org/abs/2511.20351)

---

## 2. 从 ERP response 到 sphere 上的 directional likelihood

每帧 $I_t$（360° panorama）+ target query 进 VLM，得到 target-conditioned response map $\ell_t(u,v) \in [0,1]$，width $W$、height $H$。这里 $(u,v)$ 是 ERP pixel，$u \in [0,W)$，$v \in [0,H)$。

像素到方向的映射（Eq. 1）：

$$\omega(u,v) = (\cos\phi\cos\theta,\; \cos\phi\sin\theta,\; \sin\phi)$$

变量解释：
- $\theta = 2\pi u/W - \pi$ 是 **azimuth**（方位角），范围 $[-\pi, \pi)$。$u=0$ 映射到 $-\pi$，$u=W$ 映射到 $\pi$，构成左右 seam。
- $\phi = \pi/2 - \pi v/H$ 是 **elevation**（仰角），范围 $[-\pi/2, \pi/2]$。$v=0$ 是天顶（$+\pi/2$），$v=H$ 是天底（$-\pi/2$）。
- $\omega \in S^2$ 是 viewer-centric frame 下的单位向量。

这一步本身没解决 distortion——它只是把 ERP pixel 通过解析映射 lift 到 sphere。**真正的纠正在后面 SH transform 里**：积分时乘 $\cos\phi$（spherical area element $d\Omega = \cos\phi\, d\phi\, d\theta$），抵消 ERP 在高纬度区的 oversampling。

为了跨帧累积证据，取 log：

$$r_t(\omega) = \log(\ell_t(\omega) + \epsilon)$$

$\epsilon$ 是数值稳定项。Log-space 的好处：**多帧证据可以加性累积**，等价于在原空间做 likelihood 乘积。这就是后面 Eq. 5 加法更新的合理性来源——这是 recursive Bayesian filter 在 log domain 的标准技巧，见 [Recursive Bayesian estimation - Wikipedia](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation)。

---

## 3. Spherical Harmonic Belief Field：核心数学结构

这是整篇 paper 最有意思的部分。**Belief** $f_t: S^2 \to \mathbb{R}$ 是一个标量场，物理含义是 "target 在方向 $\omega$ 上的累积 log-posterior score"：

$$f_t(\omega) \approx \log p(\omega^* = \omega \mid z_{1:t}, a_{1:t})$$

其中 $z_{1:t}$ 是历史观测、$a_{1:t}$ 是历史动作。把它在 real SH basis 上展开（Eq. 2）：

$$f_t(\omega) = \sum_{\ell=0}^{L} \sum_{m=-\ell}^{\ell} c_{\ell m}^{(t)}\, Y_\ell^m(\omega)$$

变量解释：
- $\ell$ 是 **degree（阶）**，决定角分辨率。$\ell=0$ 是 DC 分量（球的常数），$\ell=1$ 三个系数编码 mean direction（后面 decoding 用），$\ell$ 越大角分辨率越高。
- $m$ 是 **order（次）**，$m \in \{-\ell, \dots, \ell\}$，决定绕 z 轴的角动量模式。
- $Y_\ell^m(\omega)$ 是 real spherical harmonic basis function，在 $S^2$ 上正交。
- $c_{\ell m}^{(t)}$ 是 belief state 的系数，$(2L+1)^2 = 64$ 个数（$L=7$）就能编码整张 belief map。**这是 paper 选择 SH 的关键效率论点**——把一个连续场压缩成几十个 coefficient。
- $L=7$ 是 bandlimit，对应约 $180°/(L+1) \approx 22.5°$ 的角分辨率。Fig. 7 的 ablation 显示 $L$ 太大有 Gibbs ringing（球面版本的 Gibbs 现象，见 [Spherical harmonics - Wikipedia](https://en.wikipedia.org/wiki/Spherical_harmonics)），太小则定位过散。

**为什么 SH 是这个问题的"天然基底"？** 两个性质：

### 3.1 加性证据累积直接在系数空间做

SH 是线性基，所以两个场相加 $\Leftrightarrow$ 系数相加。这就是 Eq. 5：

$$c_{\ell m}^{(t)} = \tilde c_{\ell m}^{(t)} + b_{\ell m}^{(t)}$$

- $\tilde c_{\ell m}^{(t)}$：propagated prior（上一帧 posterior 经过 rotation 后的系数）
- $b_{\ell m}^{(t)}$：current observation 的 SH 系数

这正是 log-space Bayesian update。不需要重采样、不需要重投影，纯系数加法。

### 3.2 Rotation equivariance 通过 Wigner-D 矩阵实现

Agent 转了 $R_t \in SO(3)$，belief 要跟着转。SH 的关键定理：**对场做 $SO(3)$ rotation $\Leftrightarrow$ 对每个 degree-$\ell$ 的 coefficient block 乘一个 Wigner-D 矩阵 $D^\ell_{mm'}(R_t)$**。见 [Wigner D-matrix - Wikipedia](https://en.wikipedia.org/wiki/Wigner_D-matrix)。

Eq. 4：

$$\tilde c_{\ell m}^{(t)} = \sum_{m'=-\ell}^{\ell} D^\ell_{mm'}(R_t)\, c_{\ell m'}^{(t-1)}$$

- $D^\ell_{mm'}(R_t)$ 是 degree-$\ell$ 的 Wigner (real) D-matrix 元素，由 $R_t$ 决定。
- $m$ 是新坐标系下的 order，$m'$ 是旧坐标系下的 order。
- 不同 $\ell$ 的 block 互相独立——这是 SH 等变表示的"分块对角"特性，跟 [Cohen et al. SO(3) equivariant CNNs (arXiv 1711.06721)](https://arxiv.org/abs/1711.06721) 和 [Equiformer (arXiv 2206.11990)](https://arxiv.org/abs/2206.11990) 是同一类数学。

**对于平面导航**（agent 只绕垂直轴转），$R_t$ 退化成 yaw 变化 $\Delta\psi_t$，此时 Wigner-D 退化成每个 $\ell$ 一个 $(2\ell+1)\times(2\ell+1)$ 的循环移位矩阵——可以 very cheap 地用 FFT-style shift 实现。paper 没展开讲这个加速，但这就是工程上 training-free 还能跑得动的关键。

### 3.3 Observation 的 SH 投影（Eq. 3）

$$b_{\ell m}^{(t)} = \int_{S^2} r_t(\omega)\, Y_\ell^m(\omega)\, d\Omega = \int\int r_t(\theta, \phi)\, Y_\ell^m(\theta, \phi)\, \cos\phi\, d\phi\, d\theta$$

- $d\Omega = \cos\phi\, d\phi\, d\theta$ 是球面面积元素。
- $\cos\phi$ 这一项至关重要：它**正好抵消了 ERP 在极点附近过密的采样**。这是 SH representation 自带 "distortion-aware" 性质的根源——你不需要 distortion-aware convolution（如 [Tateno et al. (arXiv 1711.11449)](https://arxiv.org/abs/1711.11449) 那一套），因为你在积分时已经按球面几何正确加权了。
- 实际实现是离散求和，在 ERP grid 上算。

### 3.4 Decoding：Fréchet mean on $S^2$（Eq. 6）

$$\hat{\omega}_t^* = \frac{\mathbf{m}_t}{\|\mathbf{m}_t\|}, \quad \mathbf{m}_t = \int_{S^2} f_t(\omega)\, \omega\, d\Omega$$

- $\mathbf{m}_t \in \mathbb{R}^3$ 是 belief 场加权后的一阶矩（向量均值）。
- $\hat{\omega}_t^*$ 是归一化后的单位方向，即 Fréchet mean（球面 mean，见 [Fréchet mean - Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean)）。
- **妙处**：$\mathbf{m}_t$ 可以直接从 degree-1 SH 系数 $c_{1,-1}, c_{1,0}, c_{1,1}$ 解析算出，不需要 grid search 也不需要梯度上升。当 belief unimodal 时这退化成 MAP 估计。
- resultant length $R_t = \|\mathbf{m}_t\| / \int f_t d\Omega \in [0,1]$ 是一个置信度度量（类似 von Mises-Fisher 分布的 concentration parameter，见 [von Mises-Fisher distribution](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)）。这个 confidence signal 后续可以做 active sensing 决策。

**整个 filter 的循环**：
1. VLM 看 $I_t$ → response map → lift 到 sphere → log → SH 投影得 $\mathbf{b}^{(t)}$
2. 上一帧 $\mathbf{c}^{(t-1)}$ 经 Wigner-D 旋转得 prior $\tilde{\mathbf{c}}^{(t)}$
3. 加法更新 $\mathbf{c}^{(t)} = \tilde{\mathbf{c}}^{(t)} + \mathbf{b}^{(t)}$
4. Degree-1 系数解析出 $\hat{\omega}_t^*$

这是一个**结构上和 Kalman filter 高度对称**的设计：predict 步是 Wigner-D rotation（球面上的 "state transition"），update 步是 log-likelihood 加法。差别在于 Kalman 在 Euclidean 空间、用线性 transfer matrix；EAGOR 在 $S^2$ 上、用等变 Wigner-D。这跟 [Peretroukhin et al. RSS 2020 (arXiv 2006.01031)](https://arxiv.org/abs/2006.01031) 那篇用 Bingham distribution 在 $SO(3)$ 上做 rotation 不确定度的工作是一脉相承的思路，只是 EAGOR 把"分布"换成了更一般的 SH 场，能表达 multimodal belief（多个候选方向同时存在），这点在 active visual search 里很重要——同一帧 panorama 里可能看到多个椅子，belief 是 multimodal 的。

---

## 4. 与 related work 的对比，帮你定位 paper 在 landscape 里的位置

| 方向 | 代表工作 | 和 EAGOR 的差别 |
|---|---|---|
| ERP distortion-aware conv | [Tateno Distortion-aware (arXiv 1809.04)](https://arxiv.org/abs/1711.11449), [SphereNet (arXiv 1801.0)](https://arxiv.org/abs/1801.09200), [Tangent images (arXiv 1912.09)](https://arxiv.org/abs/1912.09390) | 它们在 feature 层处理 distortion，但仍输出 image-space 预测。EAGOR 把 geometry 提到 state estimation 层。 |
| Spherical positional encoding | [Bending Reality (arXiv 2203.01452)](https://arxiv.org/abs/2203.01452), [SphereUFormer (arXiv 2412.06968)](https://arxiv.org/abs/2412.06968) | 它们改 transformer 内部位置编码。EAGOR 完全不动 VLM 内部，外挂球面滤波器。 |
| SH 作为 learned feature | [HUSH (CVPR 2025)](https://arxiv.org/abs/2502.07499), [Equiformer](https://arxiv.org/abs/2206.11990) | 它们把 SH 当可学特征。EAGOR 把 SH 当 **belief representation**，连接 frozen VLM evidence 和 recursive Bayesian 估计。 |
| Probabilistic direction estimation | [Shao et al. (arXiv 2510.10912)](https://arxiv.org/abs/2510.10912), [Zangeneh et al. (arXiv 2301.02086)](https://arxiv.org/abs/2301.02086) | 它们在 2D planar image / 2D pose 空间做。EAGOR 直接在 $S^2$ 上做。 |
| VLM-based nav | [VLFM (arXiv 2312.03275)](https://arxiv.org/abs/2312.03275), [PanoNav (arXiv 2511.06840)](https://arxiv.org/abs/2511.06840), [UniNaVid (arXiv 2412.06224)](https://arxiv.org/abs/2412.06224) | 它们用 VLM 做 per-frame 决策 / frontier map。EAGOR 把 VLM 输出当成 likelihood observation，跨帧累积。 |
| MLLM omni-directional benchmark | [ODI-Bench (arXiv 2510.11549)](https://arxiv.org/abs/2510.11549), [OSR-Bench (arXiv 2505.11907)](https://arxiv.org/abs/2505.11907), [Spatial reasoning survey (arXiv 2511.15722)](https://arxiv.org/abs/2511.15722) | 这些 benchmark 揭示了 MLLM 在 frame transformation 下崩。EAGOR 不试图修 VLM，而是用几何外壳绕过这个 bug。 |

EAGOR 的定位很清楚：**它是一个 model-agnostic、training-free 的"几何外壳"**，套在任意 open-vocabulary VLM 外面，把 per-frame semantic response 转成球面上递推的、等变的方向 belief。这个定位决定了它的两个核心卖点——plug-and-play、几何正确。

---

## 5. 实验数据逐项解读

### 5.1 Active Visual Search（Table 1）

HOS（[Humanoid Object Search, arXiv 2511.20351](https://arxiv.org/abs/2511.20351)）上，重点看这两行：

- Qwen2.5-VL-7B standalone: **27.24%** overall
- Qwen2.5-VL-7B + EAGOR: **40.10%** overall（+12.86 pt，相对 +47%）

更狠的对比：**EAGOR + Qwen2.5-VL-7B (40.10%) > Gemma-3-12B standalone (25.67%)**，也 > Qwen2.5-VL-72B 在 OSR-Bench 上的 18.1%。也就是说，**用 1/10 参数量加一个球面滤波器，比单纯 scale backbone 还赚**。这是 paper 最强的 claim 之一，直接支持 [A-1]。

OSR-Bench（multi-target egocentric relative-direction reasoning，[arXiv 2505.11907](https://arxiv.org/abs/2505.11907)）：
- EAGOR + Qwen2.5-VL-7B: **25.2%**（standalone 16.8%，+50% 相对）
- EAGOR 在 OSR-Bench 上平均相对 gain +45.6%。

为什么 gain 这么大？OSR-Bench 的查询是 "object A 在 object B 的哪个方向"——纯 directional reasoning，正是 ERP 最容易崩、SH 最占便宜的场景。

### 5.2 Waypoint Following（Table 2）

这是验证**几何等变性**的关键实验。四个 trajectory segment L1-L4，转弯越来越急。

| Method | L1 | L2 | L3 | L4 | TC (°/step) |
|---|---|---|---|---|---|
| Centroid | 25.5 | 49.4 | 90.9 | 66.7 | 11.6 |
| Cent-Circ | 25.5 | 49.3 | 66.3 | 72.8 | 7.3 |
| Grid | 23.2 | 138.9 | 37.3 | 75.9 | 19.2 |
| EAGOR | **21.1** | **31.1** | **40.9** | **47.1** | **2.6** |

观察：
- L1 大家差不多，因为没有转动累积。
- L3 是大转弯：Centroid 90.9°、Grid 138.9°（直接崩了），EAGOR 40.9°。Grid 在 L3 这种剧烈变化下完全失控——它做了 temporal accumulation 但没做等变 propagation，所以一转就乱。
- **TC（temporal consistency）2.6°/step vs 7.3-19.2°/step**——这是 paper 强调的"6.5× 改善"。TC 衡量相邻两步预测的方向变化是否平滑，越小说明 belief 在 motion 下越稳定。这是等变 propagation 的直接证据。

### 5.3 Map-Free Navigation（Table 3）

| Method | SR↑ | SPL↑ | Steps↓ | MAE↓ | Seam↑ |
|---|---|---|---|---|---|
| Centroid | 82 | 54.4 | 61.0 | 45.9 | 19.6 |
| Cent-Circ | 62 | 41.5 | 77.8 | 46.5 | 33.5 |
| Grid | 82 | 41.4 | 63.6 | 54.2 | 15.8 |
| EAGOR | **94** | **56.8** | **50.2** | **33.8** | **70.6** |

几个有意思的点：
- **Seam success 70.6 vs 19.6**——3.6× 的 seam 鲁棒性提升。这是 SH 表示的最直接红利：球面没有 seam。
- **Cent-Circ 的 SR 反而比 Centroid 低（62 vs 82）**。这是 paper 想强调的一个反直觉点：**per-frame 的 wrap-around 修正不足以支撑 closed-loop 控制**，因为单帧修对了，跨帧的累积误差还在。EAGOR 不修单帧，它在 state 层做等变 propagation，所以跨帧一致。
- EAGOR SPL 56.8 比 Centroid 54.4 只高一点，但 SR 高 12pt——意味着 EAGOR 主要是把"差一点就成功"的 episode 救回来了，而不是把已成功的 episode 走得更短。结合 steps 50.2 vs 61.0，EAGOR 平均少走 18% 的步数。

### 5.4 Real-world（Unitree Go2 + Insta360 X3）

Fig. 5 是 real-world demo。两种场景：
- 静止 robot、移动 target（top row）：检验 belief 跟踪动态目标
- 移动 robot、静止 target（bottom row）：检验等变 propagation

real-world 的意义不在数字（paper 没给 quantitative real-world table），而在 **transferability**：training-free 方法在 sim 训都没训，直接 deploy 到有 sensor noise、motion blur、光照变化的真实 legged robot 上还能工作。这呼应 [A-3]。

参考硬件: [Unitree Go2](https://www.unitree.com/products/Go2), [Insta360 X3](https://www.insta360.com/product/insta360-x3)

---

## 6. 失败模式（Table 4）和 paper 自己承认的 limitation

Qwen2.5-VL-7B + EAGOR 在 HOS 上的 failure mode 分布：

| Failure Mode | Episodes 占比 | SR |
|---|---|---|
| OCR / fine-grained text | 18.5% | 13.2% |
| Rare target | 16.9% | 3.6% |
| Multi-instance confusion | 12.5% | 33.3% |
| VLM false detection | 11.9% | 13.6% |
| Overall | 59.9% | 40.1% |

几个 intuition：
- **Rare target (SR 3.6%)**：VLM 根本没认出目标，SH-BF 拿到的 likelihood 是 garbage。这暴露了 EAGOR 的根本依赖——它只是个 geometry shell，semantic ceiling 完全由 VLM 决定。改进方向是接 grounded detector 或开放词表 detector（如 [Grounding DINO](https://arxiv.org/abs/2303.05499)）补 VLM 的长尾。
- **Multi-instance confusion (SR 33.3%)**：belief 是 multimodal 的，Fréchet mean 落在两个 mode 之间——这是个已知问题。paper 在 Sec. 5 提到 "future work will incorporate grounded detection for multi-instance disambiguation"。
- **OCR / fine-grained text (SR 13.2%)**：VLM 读不准文字，纯 semantic 失败，和 geometry 无关。

EAGOR 自己承认的另两个 limitation：
1. **Translation 敏感**：现在只对 rotation 等变，平移引起的 parallax 没建模。paper 在 Sec. 4.1 说 "the resulting parallax remains within single-frame observation noise, avoiding explicit translational compensation"——意思是步长够小，平移近似 negligible。但大步长场景会崩。Future work: "translation-aware, multi-sensor formulations for parallax-aware reasoning"。
2. **Bio-inspired perception for efficiency**：引用了自己的 [LLMind (arXiv 2603.14882)](https://arxiv.org/abs/2603.14882) 和 [VL2Spike (arXiv 2606.15898)](https://arxiv.org/abs/2606.15898)，想做 spike-driven 低功耗版本。

---

## 7. 我自己的几点延伸联想（build your intuition）

**第一，这套框架本质上是 "von Mises-Fisher filter 的非参数推广"**。vMF 分布在 $S^2$ 上是 unimodal 的，可以用一个 mean direction $\mu$ 和一个 concentration $\kappa$ 参数化，filtering 时 closed-form。EAGOR 用 SH 场代替 vMF，等于允许 multimodal、非高斯型的 belief——代价是没有 closed-form prediction step（要用 Wigner-D 在 coefficient 空间做），收益是能表达 "看到两个椅子，都可能是 target" 这种情况。如果你以后想做更轻量的版本，可以先试 vMF mixture + EKF-style update，可能比 SH 更省。

**第二，Wigner-D propagation 等价于 "在 group Fourier domain 做 state transition"**。这是 [Cohen et al. (arXiv 1711.06721)](https://arxiv.org/abs/1711.06721) 的 S2CNN 的同款数学：rotation 在 spatial domain 是卷积，在 spectral domain 是 Wigner-D 乘法。EAGOR 巧妙地把它用到了 state estimation 而不是 feature learning 上。

**第三，degree-1 SH 解析解码这件事，暗示了一个更深的几何结构**。$S^2$ 上的 vector field（$\ell=1$ 的三个 basis $Y_1^{-1}, Y_1^0, Y_1^1$）和 $\mathbb{R}^3$ 里的 vector 是同构的。所以 $\mathbf{m}_t = \int f_t \omega d\Omega$ 其实就是把 belief 投影到 $\ell=1$ 子空间，再同构到 $\mathbb{R}^3$。如果 belief 是 vMF，$\ell=1$ 系数已经足够解码；如果不是，$\ell=1$ 给的是 best vector approximation。要恢复真正 MAP 还得在 sphere 上 grid search——paper 这里偷了个懒，但大多数情况够用。

**第四，整套思路和 classic SLAM 里的 pose graph / EKF 之争同构**。EAGOR 选了 "filter" 路线（每帧 update 一次 prior，不存历史），没选 "smoother" 路线（保留所有 observation，全局优化）。如果未来要做 loop closure 或全局一致，可能要换成 spherical pose graph optimization。

**第五，跟 NeRF/3DGS 的 connection**。EAGOR 在 sphere 上维护一个 "belief field"，这和 NeRF 在 3D 空间维护 radiance field 形式上很像。差别是 EAGOR 的 field 是 2-sphere 上的标量场（log-posterior），不是 3D 空间的 radiance + density。但 "用 SH 表示场 + 用 Wigner-D 做等变操作" 这套 toolset 在 [Plenoxels](https://arxiv.org/abs/2112.05131) 之类工作里也用过。如果你以后想做 "embodied agent 内部维持一个球面语义场"，这套数学直接可用。

**第六，和 LLM 的 "scratchpad / chain of thought" 的类比**。EAGOR 的 SH-BF 是一种 **external memory state**：VLM 每次只看一帧（context window = 1），但 SH coefficients 跨帧持续更新，相当于一个 "geometry scratchpad"。这跟让 LLM 把中间推理写在 context 里、再读回来是同构的思路，只不过 EAGOR 的 scratchpad 是几何结构化的（SH coefficients），不是 free-form text。这个 "结构化外部 memory" 的设计模式在 embodied AI 里会越来越常见。

**第七，关于 $L=7$ 的选择**。$(L+1)^2 = 64$ 个 coefficient 表达一张 belief map，相比 ERP grid 比如 $1024 \times 512 = 524288$ 像素，压缩比 ~8000×。这正是为什么 training-free 还能跑得动——你不需要在像素空间跑递推，在 64 维 coefficient 空间跑就行。代价是角分辨率 $180°/8 \approx 22.5°$，对于 "目标在房间哪个方向" 这种粗粒度任务够用，对于 "目标在书架第几格" 这种细粒度任务不够。要提高分辨率可以加 $L$，但 Gibbs ringing 会变严重，需要 windowing（类似 Hamming window 在球面版本的处理）。

---

## 8. 一句话总结

EAGOR 把 "embodied 360° directional reasoning" 从 "VLM 在 ERP image 上预测 pixel 坐标" 重新表述为 **"VLM 提供方向证据 + SH-BF 在 $S^2$ 上做等变递推 Bayesian filtering"**，用 Wigner-D 实现 rotation equivariance、用 degree-1 SH 实现 closed-form decoding、用 log-space 加法实现 multi-frame 证据累积。结果是 frozen 7B VLM 在 OSR-Bench 上打败 72B 同族、map-free navigation 在 seam 场景上 3.6× 鲁棒、real-world legged robot 上 training-free 直接 deploy。它的 ceiling 由 VLM 的 semantic 能力决定，它的 floor 由 SH 几何保证——这是一篇 "geometry shell around VLM" 的干净范例。

进一步阅读建议：
- 数学基础: [Spherical harmonics - Wikipedia](https://en.wikipedia.org/wiki/Spherical_harmonics), [Wigner D-matrix - Wikipedia](https://en.wikipedia.org/wiki/Wigner_D-matrix), [Recursive Bayesian estimation - Wikipedia](https://en.wikipedia.org/wiki/Recursive_Bayesian_estimation), [Fréchet mean - Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean), [von Mises-Fisher distribution - Wikipedia](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)
- 等变表示: [SO(3) equivariant CNN (arXiv 1711.06721)](https://arxiv.org/abs/1711.06721), [Equiformer (arXiv 2206.11990)](https://arxiv.org/abs/2206.11990), [Smooth SO(3) belief (arXiv 2006.01031)](https://arxiv.org/abs/2006.01031)
- ERP distortion 处理: [Distortion-aware conv (arXiv 1711.11449)](https://arxiv.org/abs/1711.11449), [SphereNet (arXiv 1801.09200)](https://arxiv.org/abs/1801.09200), [Tangent images (arXiv 1912.09390)](https://arxiv.org/abs/1912.09390), [Bending Reality (arXiv 2203.01452)](https://arxiv.org/abs/2203.01452), [HUSH CVPR 2025](https://arxiv.org/abs/2502.07499)
- Benchmark / VLM: [OSR-Bench (arXiv 2505.11907)](https://arxiv.org/abs/2505.11907), [HOS / Thinking in 360 (arXiv 2511.20351)](https://arxiv.org/abs/2511.20351), [ODI-Bench (arXiv 2510.11549)](https://arxiv.org/abs/2510.11549), [Qwen2.5-VL (arXiv 2309.16609)](https://arxiv.org/abs/2309.16609), [Gemma 3 (arXiv 2503.19786)](https://arxiv.org/abs/2503.19786)
- Embodied / Nav: [Habitat (arXiv 1904.01201)](https://arxiv.org/abs/1904.01201), [HM3D (arXiv 2109.08242)](https://arxiv.org/abs/2109.08242), [VLFM (arXiv 2312.03275)](https://arxiv.org/abs/2312.03275), [PanoNav (arXiv 2511.06840)](https://arxiv.org/abs/2511.06840), [UniNaVid (arXiv 2412.06224)](https://arxiv.org/abs/2412.06224), [Embodied QA (arXiv 1711.11543)](https://arxiv.org/abs/1711.11543), [VLN (arXiv 1711.07280)](https://arxiv.org/abs/1711.07280)
- 硬件: [Unitree Go2](https://www.unitree.com/products/Go2), [Insta360 X3](https://www.insta360.com/product/insta360-x3)
