---
source_pdf: A Closed-Loop CPR Training Glove with Integrated.pdf
paper_sha256: bde5d853063dfcc0fbdc08547932b5d0b2ece64bdfbd0a79c4a5751f8e3d3412
processed_at: '2026-07-17T09:44:39-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# CPR Training Glove: Closed-Loop Tactile Sensing & Haptic Feedback 深度解析

这篇 paper 来自 University of Washington 的 Yiyue Luo group（MIT CSAIL tactile sensing 背景）和 Hanyang University / GIST 的 Kyung-Joong Kim group 合作。第一作者 Jaeyoung Moon 和 Mingzhuo Ma 共同贡献。

---

## 1. Core Motivation: 为什么 closed-loop haptic 对 CPR 重要

CPR 是一个 high-stakes motor skill，quality 直接影响 cardiac arrest survival rate。American Heart Association (AHA) guideline 定义了三个 key metrics:

| Metric | Target | Clinical Rationale |
|--------|--------|-------------------|
| Compression rate | 100-120 cpm | 保证 adequate cardiac output |
| Compression depth | 5-6 cm (≈500-600 N on adult male) | 确保 effective chest compression |
| Hand pose | frontal, interlaced fingers lifted | 避免 rib fracture, force delivery efficiency |

**传统 self-training 工具的痛点**: Laerdal CPRmeter ($895)、TrueCPR ($2341) 等设备依赖 audio-visual feedback, 把 performer 的 gaze 从 patient/manikin 转移到 external screen。这打破了 CPR task 本身需要的 visual monitoring（检查 chest rise, airway, patient color）。

**Key insight**: CPR 是 sensorimotor task, 闭环应该走 proprioceptive channel, 而非 visual channel。Haptic feedback 保留了 visual attention 给 patient, 同时提供 task-compatible guidance。这和 surgical robotics 中的 haptic teleoperation 是同样的 design philosophy。

Reference: AHA 2020 Guidelines https://cpr.heart.org/en/resuscitation-science/cpr-and-ecc-guidelines

---

## 2. Hardware Architecture Deep Dive

### 2.1 Tactile Sensor: Velostat-based Resistive Array

**材料物理**: Velostat 是 3M 生产的 carbon-impregnated polyolefin film (0.1mm thick)。Carbon particles 分散在 polymer matrix 中, pressure 下 particle 间距缩小, tunneling current 增加, macroscopically 表现为 resistance 下降。这是 piezoresistive effect, 但 response 非linear, 且有 significant hysteresis。

**Sensor structure**:
- 两层 FPCB (flexible printed circuit board), polyimide substrate 上 pattern copper traces
- 正交排列: top layer traces 沿 x 方向, bottom layer 沿 y 方向
- Velostat sheet 夹在中间
- 每个 trace intersection 定义一个 sensing cell
- Palm array (13×14) + dorsum array (13×14) = 182 sensors total

**Readout circuit**: voltage divider

$$V_{out} = V_{cc} \times \frac{R_{velostat}}{R_{pullup} + R_{velostat}}$$

- $V_{cc}$ = supply voltage (3.3V from ESP32S2)
- $R_{velostat}$ = Velostat resistance at sensing cell (随 pressure 变化, 范围 ~10kΩ - 1MΩ)
- $R_{pullup}$ = 4.7 kΩ fixed resistor
- $V_{out}$ = ADC 读取的 voltage

当 pressure 增加, $R_{velostat}$ 下降, $V_{out}$ 下降。所以 raw signal 和 force 是 **inverse relationship**, 需要在 preprocessing 中 invert。

**为什么 palm + dorsum dual sensing**: 传统 tactile glove 只 sense palm side。Dorsum sensing 的作用:
- **Hand pose detection**: dorsum pressure pattern 可以 detect hand deviation (left/right skew), 因为 wrist extension/flexion 会把 pressure redistribute 到 dorsum
- **Force redistribution monitoring**: 不正确 pose 下 force 会 leak 到 dorsum
- **Redundancy**: 2× data points, 提升 model robustness

**Conductive strip width tradeoff** (这是 fabrication 的 key design parameter):

| Strip Width | $\Delta R / R_0$ (0-600N) | Hysteresis | 300-cycle Drift |
|-------------|---------------------------|------------|-----------------|
| 1 mm | ≈ 0.10 | 99.57% | - |
| **3 mm** | **≈ 0.85** | **56.04%** | **11.05%** |
| 6 mm | ≈ 0.03 | 22.39% | - |

**Intuition behind strip width**: 
- Strip width 决定 effective contact area 和 current density
- 1mm: contact area 小, current density 高, 但 carbon particle 的 stochastic contact 导致 high hysteresis (loading/unloading 路径差异大, 99.57% 意味着几乎完全 different path)
- 6mm: contact area 大, current density 低, sensitivity 极差 ($\Delta R / R_0$ 仅 0.03, 几乎不可用), 但 mechanical compliance 好, hysteresis 低
- 3mm: sweet spot, sensitivity 高 (0.85 = 6.7× dynamic range) 且 hysteresis 可通过 peak sampling mitigate

Hysteresis 56% 看起来高, 但对 CPR 这种 quasi-static application (每秒 ~2 compressions) acceptable。如果是高频 dynamic loading (e.g., typing detection at 10+ Hz), 这个 hysteresis 会 fatal。

Reference: Velostat product page https://www.adafruit.com/product/1361

### 2.2 Haptic Actuation: ERM Coin Motors

**ERM (Eccentric Rotating Mass)**: 一个 unbalanced mass 旋转产生 centrifugal force, 转化为 vibration。Frequency $f$ 和 amplitude $A$ 都和 driving voltage $V$ 正相关:

$$f \propto V, \quad A \propto V^2$$

- $f$ = vibration frequency (Hz)
- $A$ = vibration amplitude (displacement)
- $V$ = effective driving voltage

**为什么 ERM 而不是 LRA (Linear Resonant Actuator)**:
- ERM: $0.5/motor, 宽 frequency response, 但 startup/stop latency 大 (~50-100ms, 因为 rotor inertia)
- LRA: crisp response (~5ms), 更 focused frequency, 但需要 resonant frequency driving (~175Hz), cost 5-10×
- 对 CPR feedback, timing precision 要求不高 (compressions 本身 500-600ms), ERM 的 latency acceptable

**Placement**: upper wrist, 5 个 motors 对应 hand pose 的 spatial encoding。

**DRV2605 haptic driver**: Texas Instruments 的 dedicated haptic driver, 内置 waveform library (123 种 built-in effects), 支持 IIC control。这里用 raw PWM override, 因为需要 custom intensity control。

**TCA9548A IIC multiplexer**: 所有 5 个 DRV2605 共享同一个 IIC address (0x5A), 需要 multiplexer 来 individually address。TCA9548A 是 8-channel IIC switch, MCU 通过 3-bit select channel 来 communicate with specific motor。

Reference: DRV2605 datasheet https://www.ti.com/product/DRV2605 ; TCA9548A https://learn.adafruit.com/adafruit-tca9548a-1-to-8-i2c-multiplexer

### 2.3 Control Circuit & Data Flow

**MCU**: ESP32S2-Mini
- Single-core Xtensa LX7 (240 MHz) — 注意 S2 是 single-core, 比标准 ESP32 的 dual-core 弱, 但有 USB OTG 和 better security
- 13-bit ADC (effective ~11 ENOB after noise)
- Wi-Fi (SoftAP mode, 不需要 external router, glove 自己做 AP)
- USB-powered

**Data pipeline**:
```
[Sensor array 182ch] → [16:1 MUX scanning] → [ADC 13-bit] → [ESP32S2]
    → [UDP over Wi-Fi] → [Downstream PC: preprocessing + LDA]
    → [UDP command back] → [ESP32S2] → [TCA9548A] → [DRV2605 × 5] → [ERM motors]
```

**Cost breakdown** (total $64.195):
- Velostat sheet: ~$0.20
- FPCB fabrication (small batch): ~$20-30
- ESP32S2-Mini: ~$5
- ERM 1030 motors × 5: ~$2.5
- DRV2605 × 5: ~$5
- TCA9548A: ~$1
- Misc (resistors, connectors, polyimide): ~$5

vs. Laerdal CPRmeter $895, TrueCPR $2341。成本降低 **15-37×**。

---

## 3. Sensor Characterization 详解

### 3.1 Hysteresis 的物理来源

Velostat 的 hysteresis 来源于 carbon particle network 的 **percolation dynamics**:
- **Loading phase**: particles compress, new conductive paths form (tunneling distance < ~1nm 时 current 突增)
- **Unloading phase**: particles don't fully relax to original positions, some conductive paths persist
- 结果: unloading 时 resistance 比 loading 时低 (相同 force 下)

这个 hysteresis 是 **rate-dependent** — loading speed 越快, polymer matrix 来不及 relax, hysteresis 越大。

**Mitigation strategy**: Peak sampling (Section 4.3)。只取 compression peak 时刻的 reading, 因为 peak 总是对应 max loading, hysteresis effect 一致。

### 3.2 SNR Analysis

$$\text{SNR}_{dB} = 20 \log_{10}\left(\frac{\text{Signal}}{\text{Noise}}\right)$$

- $20 \log_{10}$ 是 **amplitude ratio** (如果是 power ratio 用 $10 \log_{10}$)
- Signal = median of average voltage drops across 300 cycles on **pressed cells**
- Local Noise = unwanted response on **surrounding unpressed cells** (mechanical coupling through Velostat sheet)
- Global Noise = unwanted signal on **all unpressed cells** (electrical crosstalk + baseline drift)

At 600N: global SNR = $18.90 \pm 2.41$ dB → Signal/Noise ≈ $10^{18.90/20} \approx 8.8$ → noise amplitude ≈ 11% of signal。

**为什么 global SNR > local SNR** (Figure 8): neighboring cells 通过 Velostat 的 mechanical conduction 受影响更大 (近场 coupling), 而 distant cells 主要受 baseline drift 影响 (远场, 更小)。

**Transient SNR drop at 100-200N**: 可能因为 sensor-manikin contact patch 在这个 force range 下 undergoing **contact area expansion** (从 point contact 到 area contact), contact dynamics 不稳定, signal fluctuation 大。这个现象在 Hertz contact mechanics 中有理论解释。

### 3.3 300-cycle Repeatability

11.05% drift over 300 cycles: Velostat 的 carbon particles 在 cyclic loading 下 **rearrange** (类似 plastic deformation 的 micro-version), 导致 baseline progressive shift。这是 Velostat 的 known limitation, long-term use 需要 periodic recalibration 或 adaptive baseline tracking algorithm。

**Alternative materials** 如果要 better long-term stability:
- Carbon-loaded silicone (custom formulation, higher cost)
- Conductive fabric (e.g., MedTex180, 但 sensitivity 低)
- Capacitive sensing (no material degradation, 但 fabrication 复杂)

Reference: Velostat sensor characterization https://www.nature.com/articles/s41598-020-66107-7

---

## 4. Modeling Pipeline Deep Dive

### 4.1 Task Formulation: Adaptive Force Thresholds

这里有一个 clever 的 design — **personalized force target based on body weight**:

$$[f_1, f_2] \approx [0.5w \times 9.8, \ 0.6w \times 9.8] \text{ N}$$

- $w$ = performer's body weight in kg
- $0.5w$ = coefficient corresponding to ~5cm compression depth
- $0.6w$ = coefficient corresponding to ~6cm compression depth
- $9.8$ = gravitational acceleration $g$ in m/s²
- $f_1$ = lower bound of correct force (N)
- $f_2$ = upper bound of correct force (N)

**Intuition**: adult chest 的 effective spring constant $k_{chest} \approx 50-60$ N/cm。要 compress 5cm, 根据 Hooke's law $F = k \cdot x$:

$$F_{5cm} = 50 \text{ N/cm} \times 5 \text{ cm} = 250 \text{ N}$$

但 heavier performer 可以 **leverage body weight** (让 gravity 做 work), lighter performer 需要 pure arm muscle force。这个 heuristic 把 force target personalized:

| Body Weight | Target Force Range | Rationale |
|-------------|-------------------|-----------|
| ≥ 90 kg | 500-600 N | Full AHA standard |
| 80 kg | 450-540 N | -10% |
| 70 kg | 405-486 N | -20% |
| 60 kg | 360-432 N | -30% |

每减 10kg, target 减 10%。这是 practical compromise — lighter people physically 难以达到 600N, 适当降低 target 让 training 更 achievable。

**Classification**:
- Force < $f_1$ → "too weak"
- $f_1$ ≤ Force ≤ $f_2$ → "correct"
- Force > $f_2$ → "too strong"

### 4.2 Data Collection Protocol

3-stage compression data 捕获三种 loading regime:

- **Stage 1** (Step pressing): 10N increments, 3s press + 1s release → quasi-static response curve, 用于 calibration
- **Stage 2** (Ramp): continuous ramp-up to max, then ramp-down → captures hysteresis loop
- **Stage 3** (Free-style): 20 random compressions → captures realistic distribution

**Ground truth**: SparkFun OpenScale + load cell amplifier (~10 Hz), reading in kgf, convert to N:

$$F_{N} = F_{kgf} \times 9.81 \text{ m/s}^2$$

- $F_{kgf}$ = load cell reading in kilogram-force
- $9.81$ = standard gravitational acceleration
- $F_N$ = force in Newton (SI unit)

**Sensor sampling**: 14.3 Hz, force ground truth aligned to closest tactile timestamp。这里有时序 mismatch (10 Hz vs 14.3 Hz), 通过 nearest-neighbor alignment 处理。

### 4.3 Preprocessing Pipeline

**Step 1: Offset correction**

$$V_{corr}(t) = V_{base} - V(t)$$

- $V_{base}$ = baseline signal (前 10s recording 的 per-channel average)
- $V(t)$ = raw voltage at time $t$
- $V_{corr}(t)$ = corrected signal

这个操作 dual purpose: (1) 去除 per-channel baseline variation (manufacturing tolerance 导致); (2) invert signal polarity (让 signal 和 force 正相关, 便于后续处理)。

**Step 2: Peak sampling**

Sliding window $n = 0.6$ s (对应一个 compression cycle at 100 cpm), 取 window 内 max 和 min index。

**Intuition**: Velostat 的 hysteresis 导致 loading 和 unloading path 不同。但 **peak 时刻** 总是对应 max loading, 此时 signal 最 robust。Peak sampling 相当于在 signal 的 most reliable point 采样, discards noisy transition regions。

Figure 6 展示了效果: without peak sampling, sensor values 按 scale 排序后 overlap 严重; with peak sampling, clear separation。

**Step 3: PCA dimensionality reduction**

182 features → 95% explained variance threshold。通常降到 ~20-30 dimensions。

**为什么 PCA 对 LDA critical**: LDA 假设 shared covariance matrix $\Sigma$ 可逆。当 feature dimension $p$ > sample size $n$, $\Sigma$ singular (rank-deficient), LDA 无法直接求解。PCA 把 $p$ 从 182 降到 ~30, 解决了 $n < p$ problem。

**Step 4: Normalization (pose only)**

Each 13×14 frame normalized to $[0, 1]$:

$$V_{norm}(t) = \frac{V_{corr}(t) - \min(V_{corr})}{\max(V_{corr}) - \min(V_{corr})}$$

- $\min(V_{corr})$, $\max(V_{corr})$ = per-frame min and max
- $V_{norm}(t)$ = normalized value in [0, 1]

这 highlight **spatial pattern** 而非 absolute magnitude, 让 model focus on pressure distribution shape (pose 的本质) 而非 force level。

### 4.4 LDA: 为什么 outperform Logistic Regression

**LDA (Linear Discriminant Analysis, Fisher 1936)** 的 decision function for class $k$:

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k$$

- $x$ = input feature vector (PCA-reduced tactile readings, ~30 dims)
- $\mu_k$ = class $k$ 的 mean vector (learned from training data)
- $\Sigma$ = shared (pooled) within-class covariance matrix (假设所有 class 共享)
- $\Sigma^{-1}$ = inverse of $\Sigma$
- $\pi_k$ = class $k$ 的 prior probability
- $\delta_k(x)$ = discriminant score for class $k$
- Prediction: $\hat{y} = \arg\max_k \delta_k(x)$

**Geometric intuition**: LDA 寻找 projection direction $w = \Sigma^{-1}(\mu_1 - \mu_2)$ that **maximizes between-class variance / within-class variance** (Fisher's criterion):

$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$

- $S_B$ = between-class scatter matrix
- $S_W$ = within-class scatter matrix
- $w$ = optimal projection direction

**Performance comparison**:

| Task | Logistic Regression | Ridge Regression | LDA |
|------|--------------------|-----------------|-----|
| Force (offline) | 74.7 ± 15.8% | 87.1 ± 15.5% | **96.1 ± 2.8%** |
| Pose (offline) | 93.3 ± 1.4% | 89.0 ± 3.0% | 92.1 ± 0.1% |
| Force (in-situ) | 72.8 ± 10.1% | 72.5 ± 9.1% | **79.8 ± 10.0%** |
| Pose (in-situ) | 92.5 ± 9.2% | 88.0 ± 10.7% | **95.2 ± 3.6%** |

**为什么 LDA 在 small-data regime 优势明显**:

1. **Generative vs Discriminative**: LDA 是 generative model (models $p(x|y)$ and $p(y)$, uses Bayes), Logistic regression 是 discriminative (directly models $p(y|x)$)。Ng & Jordan (2002) 证明: 在 $n$ small 时, generative model 的 sample complexity 更低, convergence 更快。

2. **Gaussian assumption fit**: Velostat 的 response 在 peak 附近 (after peak sampling) approximately Gaussian (central limit theorem on carbon particle aggregate), 所以 LDA 的 distributional assumption 合理。

3. **Parameter count**: LDA < 200 parameters (3 mean vectors × 30 dims + shared covariance)。Logistic regression with 30 features + 3 classes = 90 parameters, 但需要 regularization tuning。Ridge regression 类似。LDA 的 parameter efficiency 在 small data 下 favorable。

4. **Variance**: LDA 的 high bias (linear boundary, shared covariance) + low variance。Logistic regression 更 flexible 但 variance 高 (std 15.8% vs LDA 的 2.8%)。

**Inference time**: 0.004 ± 0.001ms (force), 0.001ms (pose)。LDA 本质是 matrix-vector multiplication + comparison, 在任何 MCU 上都 sub-millisecond。

Reference: ESL Chapter 4 https://hastie.su.domains/ElemStatLearn/ ; Fisher 1936 https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

### 4.5 In-situ Performance Degradation Analysis

Offline → in-situ: force 96.1% → 79.8% (drop 16.3%), pose 92.1% → 95.2% (gain 3.1%)。

**Force degrade 的原因**:
- **Glove-skin contact instability**: dynamic compression 中 glove 会 shift, sensor 和 skin 的 contact area 变化
- **Sweat**: 改变 Velostat 的 electrical properties (moisture increases conductivity)
- **Wrist motion artifacts**: arm movement 产生 inertial forces, 混入 tactile signal
- **Calibration drift**: 短 calibration (20 compressions) 不足以 capture full variability

**Pose 反而 improve 的可能原因**:
- User study 有 familiarization session, pose 更标准化
- Spatial pattern 对 magnitude noise robust (normalization 去除 magnitude)
- Pose classes 的 separation 主要靠 spatial location, 对 contact instability 较不敏感

---

## 5. System Latency Analysis

**End-to-end latency breakdown**:

| Component | Latency |
|-----------|---------|
| Sensor frame period | ~70 ms (14.3 Hz) |
| Data acquisition | 0.051 ± 0.003 ms |
| Modeling (LDA inference) | 0.005 ± 0.001 ms |
| Serial transmission (haptic command) | ~20 μs |
| **Total computation** | **~0.056 ms** |

**Key insight**: bottleneck 是 **sensor frame rate** (70ms), computation 完全不是 bottleneck。这和大多数 embedded ML 系统一致 — sensing I/O 往往比 inference 慢得多。

**14.3 Hz 是否足够**: CPR compression rate 100-120 cpm → 0.5-0.6s per compression → 14.3 Hz 给出 ~7-8 frames per compression。Nyquist 要求 >2× signal frequency, 这里 ~14× oversampling, 足够 peak detection 和 classification。

**Comparison with human perceptual thresholds**:

| Modality | Latency Threshold for "Real-time" |
|----------|----------------------------------|
| Visual | ~30 ms (below this feels instantaneous) |
| Audio | ~50 ms |
| Tactile | ~100 ms |

Total latency (70ms sensor + 0.056ms compute + ~50ms ERM startup) ≈ 120ms, 略超 tactile threshold, 但对 CPR 这种 2 Hz task, user 不会感知到明显 delay。

Reference: ESP32-S2 datasheet https://www.espressif.com/sites/default/files/documentation/esp32-s2_datasheet_en.pdf

---

## 6. Haptic Feedback Design: Co-design Workshop

### 6.1 Encoding Strategy

4 位 haptics experts (4+ years experience) 通过 co-design workshop 确定 mapping:

| CPR Metric | Haptic Dimension | Encoding |
|------------|-----------------|----------|
| Compression rate | **Pulse count** | 1 pulse = too slow, 2 = correct, 3 = too fast |
| Compression force | **Intensity** | Max = too weak, Mid = correct, Min = too strong |
| Hand pose | **Spatial position** | Left/Right/Center/Lower motor = corresponding error |

**Design rationale**: 利用 vibrotactile 的 3 个 **perceptually independent** dimensions。这是 Tactile Display 领域的 standard approach — 通过 multidimensional encoding 在有限 actuators 下 maximize information throughput。

Reference: Tan et al. (2020) haptic information transmission survey https://ieeexplore.ieee.org/document/9020244

### 6.2 Intensity Calibration Study

5 participants pilot study:
- **Low** (minimum perceivable): PWM 38-58 across users, mean = 50
- **High** (maximum tolerable): PWM 128 (all participants, full range)
- **Mid**: most distinguishable from high → PWM = 73

**Why PWM = 73 for mid, 而非 (50+128)/2 = 89**: 

因为 mid 和 high 在 static test 中容易 confuse (Figure 3)。选择离 high 最远的 mid value (73), 牺牲 mid-low discriminability 换取 mid-high discriminability。这是一个 asymmetric design choice — 因为 "too strong" (high) 和 "correct" (mid) 的 confusion 比 "too weak" (low) 和 "correct" (mid) 的 confusion 更 dangerous (over-compression 更 risky)。

### 6.3 User Study: Haptic Feedback 的 Perception Challenges

8 participants (6M/2F, 24.63 ± 4.96 years), between-subject design:
- 4 haptic group
- 4 audio-visual group (mobile app, Figure 9)

**Quantitative results** (Figure 10):

| Metric | Haptic | Audio-Visual | Direction |
|--------|--------|-------------|-----------|
| Physical workload | Lower | Higher | ↓ better |
| Mental workload | Higher | Lower | ↓ better |
| Ease of use | Lower | Higher | ↑ better |

**Haptic 的 paradox**: 减少了 physical workload (因为不需要转头看 screen), 但增加了 mental workload (因为需要 decode vibration patterns)。这个 trade-off 是 haptic interface design 的经典 challenge。

**Qualitative findings (thematic analysis, Braun & Clarke 2006)**:

**Theme 1: Visual distraction from audio-visual**
- P5: "I kept having my attention drawn to the app, I found it difficult to focus well on checking the patient's condition"
- P1: "For the sound feedback it is effective, but for the force feedback [the light] it is hard to follow"
- 2/4 audio-visual users explicitly mentioned visual distraction

**Theme 2: Motion-induced vibrotactile masking**
- P7: "sometimes I felt barely nothing during the CPR and sometimes I felt strong vibrations"
- 只有 max intensity (PWM 128) 在 dynamic compression 中可靠感知
- Low (PWM 50) 和 mid (PWM 73) 经常 imperceptible

**Theme 3: Cognitive load of pattern recall**
- P8: "hard to recall the meaning of vibration"
- P7: "The vibration patterns are too many to be remembered, and it is hard for me to realize their meanings during CPR"

**Root cause of masking**: CPR 中 arm 和 torso 的 active motion 产生 broadband mechanical noise, 激活 Pacinian corpuscles (负责 vibration perception, peak sensitivity ~250Hz)。这些 mechanoreceptors 被 motion noise saturate, 导致 narrowband vibration signal 被 masked。

Reference: Peeters et al. (2019) vibrotactile masking in cycling https://doi.org/10.1055/a-0895-5488 ; Sigrist et al. (2013) motor learning feedback review https://doi.org/10.3758/s13423-012-0333-8

---

## 7. Intuition Building: Core Design Principles

### 7.1 Closed-loop 的 Bandwidth Analysis

```
User compression → Tactile sensing (14.3 Hz) → LDA estimation (~0.005ms)
       ↑                                                    ↓
       ←←← Haptic feedback (ERM ~50ms startup) ←←← Error computation ←←
```

**Loop bandwidth** = min(sensing rate, feedback rate) ≈ 14 Hz。对于 CPR 这种 2 Hz task (100-120 cpm = 1.67-2 Hz), loop bandwidth 是 task frequency 的 ~7×, 远超 control theory 的 Nyquist requirement (2×)。这意味着 closed-loop 可以 track 每个 compression 的 performance 并给出 feedback。

**与 human motor control 的 comparison**: human proprioceptive feedback loop 的 latency ~100-150ms (spinal + cortical)。这个系统的 total latency ~120ms, 和 human proprioceptive loop comparable。所以 haptic feedback 感觉 "natural", 好像是自己 body 的 feedback。

### 7.2 为什么 Haptic 在 Dynamic Task 中 Fail

**Vibrotactile masking 的 mechanoreceptor 机制**:

Human skin 有 4 种 mechanoreceptors:
- **Pacinian corpuscles**: 快 adapting (FA II), peak sensitivity ~250 Hz, 负责 vibration perception, **最容易 masked**
- **Meissner corpuscles**: 快 adapting (FA I), peak sensitivity ~30-50 Hz, 负责 flutter
- **Merkel discs**: 慢 adapting (SA I), peak sensitivity ~5 Hz, 负责 pressure
- **Ruffini endings**: 慢 adapting (SA II), 负责 stretch

ERM motors 的 vibration frequency 在 PWM 50-128 下大约 100-200 Hz, 主要 activate Pacinian corpuscles。这些 receptors 在 body motion 时被 broadband mechanical noise saturate (motion 产生 10-1000 Hz 的 vibration noise), 导致 signal-to-noise ratio 在 perceptual level 下降。

**Design implication**: 
- Haptic feedback 应该 **encode information in spatial dimension** (which motor vibrates) 而非 intensity dimension (因为 intensity 被 masked)
- Temporal patterns (pulse count, rhythm) 对 masking 更 robust, 因为 cortical processing 可以 integrate over time
- 最好用 **lower frequency actuators** (e.g., voice coil at 30-50 Hz) activate Meissner corpuscles, 对 motion masking 更 robust

### 7.3 为什么 LDA 在 Small Data 下 Beat Deep Learning

**Bias-Variance tradeoff 的具体数字**:

LDA 的 variance: force estimation std = 2.8% (offline), 10.0% (in-situ)
Logistic regression 的 variance: std = 15.8% (offline), 10.1% (in-situ)

LDA 的 **low variance** 来自:
1. Shared covariance assumption (强制所有 class 共享 $\Sigma$, 大大减少 parameters)
2. Gaussian prior (regularize mean estimates)

**如果用 deep learning**: 182 → 64 → 64 → 3 的 MLP 有 ~12,000 parameters。在 ~200 training samples 下, overfitting 严重。需要 transfer learning 或 data augmentation, 增加 system complexity。

**On-device constraint**: ESP32S2 有 320KB SRAM, 4MB Flash。LDA model < 1KB, 可以轻松 fit。MLP 12KB 也可以, 但 inference 需要 matrix multiplications, 在 single-core 240MHz 上 ~1-5ms, 仍然 acceptable 但失去 LDA 的 simplicity advantage。

### 7.4 Velostat vs Alternative Tactile Sensing Technologies

| Technology | Sensitivity | Hysteresis | Cost | Fabrication | Durability |
|-----------|-------------|------------|------|-------------|------------|
| **Velostat (this paper)** | ΔR/R₀ ≈ 0.85 | 56% | $0.1/sheet | DIY/FPCB | ~300 cycles |
| Carbon-loaded silicone | Higher | ~20-30% | $5-10/sheet | Custom mold | ~1000+ cycles |
| Capacitive (MIT GelSight) | Very high | <5% | $50+ | Complex | High |
| Piezoelectric (PVDF) | Dynamic only | Low | $20+ | Specialized | High |
| Optical (FBG) | Very high | <1% | $100+ | Specialized | Very high |

Velostat 的 advantage 是 **cost-accessibility** — $0.1/sheet 让任何人都能 prototype。对于 research validation 和 education, 这是 ideal。对于 commercial product, 应该 upgrade 到 carbon-loaded silicone 或 capacitive。

Reference: MIT GelSight https://www.gelsight.com/ ; Tactile sensor review https://arxiv.org/abs/2205.11879

---

## 8. Limitations & Open Problems

### 8.1 Explicit Limitations

1. **N=8 pilot study**: statistical power 不足, 4 per group 无法做 rigorous hypothesis testing。需要 RCT with ≥30 participants per group。

2. **Motion masking 是 fundamental perceptual limitation**: 无论怎么 improve hardware, human vibrotactile perception 在 dynamic task 中就是 degraded。需要 **multimodal** (haptic + audio) 来 compensate。

3. **Calibration burden**: 每次使用需要 20+ compressions calibration。Barrier to adoption for casual users。需要:
   - Cross-user generalization (collect large dataset, train universal model)
   - Online adaptation (continuously update model during use)
   - Few-shot learning (1-3 calibration compressions)

4. **External computation dependency**: 当前依赖 downstream PC, 限制了 portability。需要:
   - On-device inference (ESP32 已经能跑 LDA, 但 preprocessing pipeline 需要 port)
   - Standalone app on phone (利用 smartphone 的 compute power)

5. **No direct depth measurement**: 系统 estimate force, 但 AHA guideline 关心 depth。Force-depth relationship 因 patient 而异 (chest stiffness varies), 这个 system 假设了 fixed mapping。

### 8.2 Future Directions (我的联想)

**方向 1: Physics-informed force-depth model**

Current system 用 linear heuristic $[0.5w \times 9.8, 0.6w \times 9.8]$。更 principled 的 approach:

$$F = k_{chest} \cdot x + c_{damping} \cdot \dot{x} + m_{effective} \cdot \ddot{x}$$

- $k_{chest}$ = chest stiffness (N/m), age/gender/body-mass dependent
- $x$ = compression depth (m)
- $c_{damping}$ = chest damping coefficient (viscoelastic behavior)
- $m_{effective}$ = effective mass being compressed
- $\dot{x}$, $\ddot{x}$ = velocity, acceleration

如果能从 tactile signal 估计 $\dot{x}$ 和 $\ddot{x}$ (通过 temporal derivatives), 可以 jointly estimate force 和 depth。Reference: Baubin et al. (1995) https://doi.org/10.1016/0300-9572(94)00861-B

**方向 2: Federated learning for cross-user generalization**

Collect data from many users, train shared model with privacy-preserving federated learning。每个 user 的 glove 只 upload model updates (gradients), 不 upload raw data。Server aggregates into universal model。这解决 calibration burden。

Reference: Google Federated Learning https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

**方向 3: Active haptic feedback (not just passive vibration)**

Current system 只用 vibration for **alerting** (告诉 user 错了)。更 advanced 的 approach: **active resistance** — glove 内置 exoskeleton 或 shape memory alloy, 在 user force 不足时 provide assistive force, 在 force 过大时 provide resistance。这是 **shared control** paradigm, 类似 surgical robot 的 haptic guidance。

**方向 4: VR/AR integration**

结合 VR/AR (e.g., Meta Quest, Apple Vision Pro) 提供 immersive CPR scenario:
- Visual: 虚拟 patient, 实时显示 compression depth, recoil
- Audio: 虚拟 heart sound, breath sound
- Haptic: glove 提供 force feedback

这创造 **multi-sensory closed-loop**, 可能比单一 modality 更 effective。Reference: Hou et al. (2021) AR CPR https://doi.org/10.1145/3508200

**方向 5: Long-term skill retention study**

CPR skill decay 是 known problem (Hamilton 2005)。这个 glove 如果能 enable **distributed practice** (每天 5 分钟, 而非集中 4 小时 course), 可能显著 improve retention。需要 longitudinal study (6-12 months) 测量 skill decay curve。

Reference: Hamilton (2005) CPR skill retention https://doi.org/10.1111/j.1365-2648.2005.03510.x

**方向 6: Tactile sensor 的 self-calibration**

利用 CPR 的 repetitive nature (每次 compression 是 similar waveform), 可以:
- Track baseline drift (exponential moving average of minima)
- Compensate hysteresis (learn loading/unloading curves online)
- Detect sensor failure (statistical outlier detection on individual channels)

这类似 IMU 的 zero-velocity update (ZUPT) — 利用 known "静止" 时刻 recalibrate。

---

## 9. 相关工作对比

### 9.1 CPR Feedback Devices

| Device | Cost | Metrics | Feedback Modality | Portability |
|--------|------|---------|-------------------|-------------|
| Laerdal CPRmeter | $895 | Depth, rate | Audio-visual | Manikin-mounted |
| TrueCPR | $2341 | Depth, rate | Audio-visual | Chest-mounted |
| CPR-Ezy | ~$200 | Force, rate | Audio-visual | Hand-held |
| iCPR (smartphone) | Free | Depth (accelerometer) | Audio | Phone on floor |
| **This glove** | **$64** | **Force, rate, pose** | **Haptic** | **Wearable** |

**Unique aspects of this glove**:
1. **Hand pose monitoring** — 大多数 device 忽略这个 metric
2. **Adaptive force target** — 考虑 body weight variability
3. **Haptic feedback** — 不分散 visual attention
4. **Wearable form factor** — 可以在实际 emergency 中也 wear (training transfer)

### 9.2 Tactile Sensing Gloves

| System | Sensor Count | Technology | Application | Cost |
|--------|-------------|------------|-------------|------|
| MIT Tactile Glove (Luo et al.) | 548 | Velostat + FPCB | Object recognition | ~$100 |
| GelSight Glove | ~15 | Optical (elastomer + camera) | Dense geometry capture | ~$500+ |
| Sensel Morph | 20,000+ | Capacitive | Multi-touch input | $249 |
| **This CPR glove** | **182** | **Velostat + FPCB** | **Force/pose estimation** | **$64** |

This glove 的 sensor count (182) 介于 MIT Tactile Glove (548, high-res for object recognition) 和简单 force sensors 之间。对 CPR 来说, 182 个 points 足以 capture pressure distribution pattern。

Reference: Luo et al. MIT Tactile Glove https://news.mit.edu/2023/mit-engineers-design-flexible-tactile-glove-0713

---

## 10. Summary: Build Intuition

这篇 paper 的核心 contribution 是把 **closed-loop control theory** 的思想应用到 CPR training:

1. **Sensing**: 182-channel Velostat array 提供 high-resolution pressure map, 14.3 Hz 足够 capture CPR dynamics
2. **Estimation**: LDA 在 small-data, low-latency constraint 下是 optimal choice (generative model + Gaussian assumption + <200 parameters)
3. **Feedback**: Haptic encoding 利用 3 个 perceptual dimensions (count, intensity, position), 但 dynamic masking 限制了 intensity dimension 的 effectiveness
4. **Closed-loop latency**: ~120ms total (sensing-limited), comparable to human proprioceptive loop

**Deep insight**: 这个系统的 bottleneck 不在 computation (0.056ms), 不在 actuation (50ms ERM startup), 而在 **human perception** — motion-induced masking 让 haptic feedback 在最需要的时刻 (active compression) 最 unreliable。这是 human-in-the-loop system 的 fundamental challenge: hardware 可以 fast, 但 human bandwidth limited。

**For Andrej**: 这个 work 体现了 embodied AI 的一个重要 direction — intelligence 不只在 model 里, 而在 sensor-actuator-feedback 的 complete loop 中。LDA 在这里 "enough" 是因为 task 的 structure (linearly separable after PCA + peak sampling), 不需要 deep learning。但 haptic interface design 需要 understanding human perception, 这是 ML 社区 typically overlooks 的 dimension。

如果要从 ML perspective improve: 
- **Online learning**: continuously adapt LDA model during use, track sensor drift
- **Multi-task learning**: jointly estimate force, pose, depth with shared representation
- **Reinforcement learning**: frame feedback selection as RL (which pattern to show to maximize learning), reward = user's improvement in next compression

但这些都是 over-engineering if the fundamental perceptual bottleneck (masking) 不解决。Hardware + perception co-design 是 key。
