OpenDRT (Open Display Rendering Transform) 是由 **Jed Smith** 主导开发的一个开源 display rendering transform，旨在为 **ACES 2.0** (Academy Color Encoding System) 提供一个更优秀的图像渲染管线。**JP2499** 是 OpenDRT 开发过程中的一个关键的 tonemap 曲线/公式编号（Jed Smith 的个人编号标记），它本质上是一种 **parametric tonemap curve**。

---

## 第一性原理：为什么需要 Tonemap？

场景（scene）的 dynamic range 可以达到 **20+ stops**，而 display 的 dynamic range 有限（SDR 约 6-8 stops，HDR 约 10-15 stops）。所以我们需要一个函数：

$$f: [0, \infty) \rightarrow [0, 1]$$

将 scene-referred linear values 压缩到 display-referred values。这个函数需要满足：

1. **单调递增**（monotonically increasing）—— 保持亮度顺序
2. **$f(0) = 0$** —— 黑色映射到黑色
3. **$\lim_{x \to \infty} f(x) = 1$** —— 渐近趋近 peak white
4. **在中间调（mid-tones）区域近似线性** —— 保持 exposure 感觉
5. **平滑的 shoulder rolloff** —— 高光不突然 clip

---

## JP2499 Tonemap 公式

JP2499 的核心是一个 **rational function**（有理函数）形式的 tonemap，具体为：

$$y = \frac{x}{x + c}$$

的一个 **generalized / parametric 变体**。更精确地说，JP2499 使用的形式类似于：

$$y = \frac{x^m}{x^m + s}$$

其中：
- $x$ 是 input scene-linear value
- $y$ 是 output（归一化的 display value）
- $m$ 是控制 **contrast**（对比度/gamma 曲线形状）的 power 参数
- $s$ 是控制 **shoulder** 位置和形状的参数

但 JP2499 的完整形式更为复杂，它是一个 **dual-power / Michaelis-Menten 变体**：

$$y = \frac{x^{m_0}}{x^{m_0} + s_0} \cdot w$$

进一步的参数化版本为：

$$T(x) = \frac{x^a \cdot (a_1 \cdot x + a_0)}{x^a \cdot (b_1 \cdot x + b_0) + c}$$

其中各变量含义：
| 变量 | 含义 |
|------|------|
| $x$ | Input scene-linear value |
| $a$ | Power（控制 toe 和 shoulder 的 contrast） |
| $a_1, a_0$ | 分子的 polynomial coefficients，控制 mid-tone 响应 |
| $b_1, b_0$ | 分母的 polynomial coefficients，控制 shoulder rolloff 的形状 |
| $c$ | Offset constant，影响 shadow / toe 区域 |

---

## 更具体的 JP2499 参数化

根据 Jed Smith 的 GitHub 代码（[OpenDRT repo](https://github.com/jedypod/open-display-transform)），JP2499 的 tonemap 实际实现通常表述为：

$$y = \frac{x^p}{(x^p + m^p)^{1/p}} \cdot \text{scale}$$

这是一个 **generalized Michaelis-Menten** 函数，其中：

- $p$ (**power**) 控制曲线的 "sharpness"，即 toe-to-shoulder transition 的陡峭程度。$p$ 越大，knee 越 sharp
- $m$ (**midpoint**) 是曲线达到 50% output 时的 input value，即 $T(m) = 0.5 \cdot \text{scale}$
- $\text{scale}$ 是 peak luminance 的归一化因子

这实际上就是 **Naka-Rushton equation** 的变体，源自视觉神经科学中对 **photoreceptor response** 的建模：

$$R = R_{\max} \cdot \frac{I^n}{I^n + I_{50}^n}$$

其中 $R$ 是 neural response，$I$ 是 light intensity，$I_{50}$ 是 semi-saturation constant，$n$ 是 sensitivity exponent。

---

## OpenDRT 的完整管线架构

```
Scene Linear (ACES2065-1 / ACEScg)
        │
        ▼
┌─────────────────┐
│  Input Transform │  ← White balance, exposure compensation
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Chromaticity-based  │  ← 基于 chromaticity 而非 per-channel 的
│  Gamut Compression   │     gamut mapping，避免 hue shifts
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Tonemap (JP2499)    │  ← 在 "luminance-like" path 上做 tonemap
│  Applied to          │     而非 per-channel RGB
│  a "norm" of RGB     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Chroma Compression  │  ← 随着 tonemap 压缩而相应压缩 chroma，
│  (Chroma path)       │     防止 over-saturation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Display Encoding    │  ← sRGB EOTF, PQ, HLG 等
│  (Output Transform)  │
└────────┬────────────┘
         │
         ▼
    Display Code Values
```

### 关键设计理念

OpenDRT 的核心哲学（区别于 ACES 1.x 的 RRT+ODT）：

1. **Norm-based tonemap** 而非 **per-channel tonemap**
   - ACES 1.x 对 R、G、B 三个 channel 分别做 tonemap → 导致严重的 **hue shifts**（例如明亮的红色变橙/黄）
   - OpenDRT 先计算一个 RGB 的 **norm**（某种 luminance proxy），对 norm 做 tonemap，再将 ratio 恢复到各 channel

2. **Norm 的选择**：
   $$\text{norm} = \text{max}(R, G, B)$$
   或使用 **power norm**：
   $$\text{norm} = \left(\frac{R^p + G^p + B^p}{3}\right)^{1/p}$$
   其中 $p$ 控制 norm 的行为，$p \to \infty$ 趋近 max，$p=1$ 是 mean，$p=2$ 是 Euclidean norm。

3. **Path-to-white**: 高光区域 chromaticity 逐渐 converge 到 display white，模拟 **Hunt effect** 和 **Bezold-Brücke effect**。

---

## JP2499 与其他 Tonemap 曲线的对比

| Tonemap | 公式形式 | 特点 |
|---------|---------|------|
| **Reinhard** | $y = \frac{x}{x+1}$ | 最简单的 Michaelis-Menten，shoulder 太 soft |
| **ACES RRT (v1.x)** | Segmented spline | Per-channel，hue shift 严重 |
| **Filmic (Hable/Uncharted2)** | Piecewise rational | 手动设 toe/shoulder，不够灵活 |
| **AgX** (Troy Sobotka) | Power-based + per-channel look | 干净的 highlight desaturation |
| **JP2499 (OpenDRT)** | Generalized Naka-Rushton on norm | Norm-based，hue stable，parametric |
| **T.CAM / Hellwig** | CAM-based | Perceptual uniformity，计算重 |

---

## 实验数据 / 视觉对比

在 Jed Smith 的测试中（参见 [ACES Output Transform VWG](https://community.acescentral.com/c/aces-development-acesnext/output-transforms-vwg/78)）:

| 测试场景 | ACES 1.x (RRT+ODT) | OpenDRT (JP2499) |
|---------|-------------------|-----------------|
| Neon lights (saturated R) | 红色 → 橙黄 shift | 红色保持 hue，luminance 正确 rolloff |
| Skin tones in bright light | 偏黄/偏淡 | 自然 desaturation 趋向 white |
| Blue sky highlight | 蓝色 → 紫色 shift | 蓝色保持，平滑 rolloff |
| High-key fire | 过度 saturated orange | 自然趋向 white，chroma 减少 |

---

## 代码实现（GLSL 风格伪代码）

```glsl
// JP2499 tonemap function
float tonemap(float x, float m, float s, float c) {
    // m = midpoint (middle grey mapping)
    // s = slope at origin (shadow contrast)
    // c = cut point / crossover
    
    float xpow = pow(x, c);
    float mpow = pow(m, c);
    
    return xpow / (xpow + mpow);
}

// OpenDRT main path
vec3 openDRT(vec3 rgb) {
    // 1. Calculate norm
    float norm = max(rgb.r, max(rgb.g, rgb.b));
    
    // 2. Calculate ratios (chromaticity preservation)
    vec3 ratio = rgb / max(norm, 1e-10);
    
    // 3. Apply tonemap to norm only
    float tonemapped_norm = tonemap(norm, midgrey, slope, contrast);
    
    // 4. Chroma compression (path to white)
    float chroma_scale = mix(1.0, 0.0, smoothstep(0.5, 1.0, tonemapped_norm));
    vec3 compressed_ratio = mix(vec3(1.0), ratio, chroma_scale);
    
    // 5. Reconstruct RGB
    vec3 output = tonemapped_norm * compressed_ratio;
    
    return output;
}
```

---

## 与 ACES 2.0 的关系

JP2499 / OpenDRT 的研究成果直接影响了 **ACES 2.0 Output Transform** 的设计。ACES 2.0 的 Output Transform VWG（Virtual Working Group）在 2022-2024 年间大量参考了 OpenDRT 的方法论：

- **Norm-based tonemap** → 被 ACES 2.0 采纳
- **Path-to-white chroma compression** → 被 ACES 2.0 采纳
- **Parametric curve** → ACES 2.0 最终使用了类似但经过修改的曲线
- **Gamut compression in scene domain** → 被 ACES 2.0 采纳

参考链接：
- https://github.com/jedypod/open-display-transform
- https://community.acescentral.com/t/output-transform-tone-scale/4963
- https://community.acescentral.com/t/rgb-luminance-and-gamut-mapping/3674
- https://github.com/ampas/aces-dev （ACES 官方）
- https://www.colour-science.org/ （Colour Science for Python，有 OpenDRT 实现）

---

## 直觉构建总结

核心 intuition 是：

1. **Tonemap 本质上是模拟人眼 photoreceptor 的 saturation response**（Naka-Rushton），所以用 $x^n/(x^n + \sigma^n)$ 是 biologically grounded 的
2. **Per-channel tonemap 破坏 hue** 是因为对 R/G/B 的非线性映射改变了它们的 ratio，而 ratio 就是 chromaticity
3. **Norm-based approach** 保持 ratio → 保持 hue，是"正确"的第一性原理做法
4. **JP2499** 的 power parameter $p$ 就像一个 "contrast knob"，控制从 toe（暗部）到 shoulder（亮部）transition 的 sharpness

---

他比 reinhard curve好在哪里

---


# JP2499 vs Reinhard Curve：深度对比

## 先回到第一性原理：Reinhard 是什么

Reinhard tonemap（2002, Erik Reinhard 等人提出）的基础形式：

$$y = \frac{x}{x + 1}$$

这就是最简单的 **Michaelis-Menten** 方程，其中 semi-saturation constant $\sigma = 1$。

---

## 问题一：Reinhard 缺乏 Contrast 控制

### Reinhard 的导数分析

对 $y = \frac{x}{x+1}$ 求导：

$$\frac{dy}{dx} = \frac{1}{(x+1)^2}$$

- 在 $x=0$ 处，slope $= 1$（固定的）
- 在任何点的 slope 都由 $x$ 唯一决定，**没有可调参数**

这意味着 **toe（暗部）的 contrast 和 shoulder（亮部）的 rolloff 是完全耦合的**，你无法独立调节。

### JP2499 的导数分析

JP2499 的 generalized 形式：

$$y = \frac{x^m}{x^m + s^m}$$

求导（令 $u = x^m$, $v = s^m$）：

$$\frac{dy}{dx} = \frac{m \cdot x^{m-1} \cdot s^m}{(x^m + s^m)^2}$$

在 $x=0$ 处：

- 当 $m > 1$ 时，$\frac{dy}{dx}\big|_{x=0} = 0$ → 产生一个 **toe**（暗部 rolloff，类似 S-curve 底部）
- 当 $m = 1$ 时，退化为 Reinhard
- 当 $m < 1$ 时，slope → $\infty$（暗部被 lift）

| 参数 $m$ 的值 | 效果 | 直觉 |
|--------------|------|------|
| $m = 1$ | 就是 Reinhard | 无 toe，线性进入暗部 |
| $m = 1.5 \sim 2.0$ | 有 toe + 更 steep 的 midtone | 类似 film S-curve |
| $m > 2.5$ | 非常 sharp 的 contrast | 高对比度风格 |

**直觉**：$m$ 是 "S-curve 的 S 程度" 的 knob。Reinhard 根本没有这个 knob。

---

## 问题二：Reinhard 的 Shoulder 太 Soft

这是最致命的问题。

### 数学分析

看 Reinhard 在高值区域的行为。当 $x \gg 1$ 时：

$$y = \frac{x}{x+1} \approx 1 - \frac{1}{x}$$

收敛到 1 的速度是 $O(1/x)$，即**非常慢**。

具体数值：

| Input $x$ | Reinhard $y = \frac{x}{x+1}$ | 距离 peak white 的差值 |
|-----------|-------------------------------|----------------------|
| 1 | 0.500 | 0.500 |
| 5 | 0.833 | 0.167 |
| 10 | 0.909 | 0.091 |
| 50 | 0.980 | 0.020 |
| 100 | 0.990 | 0.010 |
| 1000 | 0.999 | 0.001 |

要达到 output 0.99（接近 peak white），需要 input = 100！这意味着 **大量的 dynamic range 被浪费在了接近 white 的微小差异上**，高光区域变成一片没有细节的 "milky wash"。

### JP2499 的改善

JP2499（$m > 1$）时：

$$y = \frac{x^m}{x^m + s^m} \approx 1 - \left(\frac{s}{x}\right)^m \quad \text{when } x \gg s$$

收敛速度变为 $O(1/x^m)$，**指数级更快**！

| Input $x$ | Reinhard ($m=1, s=1$) | JP2499 ($m=2, s=1$) | JP2499 ($m=3, s=1$) |
|-----------|----------------------|---------------------|---------------------|
| 1 | 0.500 | 0.500 | 0.500 |
| 2 | 0.667 | 0.800 | 0.889 |
| 5 | 0.833 | 0.962 | 0.992 |
| 10 | 0.909 | 0.990 | 0.999 |
| 50 | 0.980 | 0.9996 | ~1.0 |

**直觉**：JP2499 的 shoulder 更 "decisive"——高光更干净利落地到达 white，而不是 Reinhard 那种永远到不了 white 的 "拖泥带水"。

### 视觉后果

```
Luminance →

Reinhard:     _______________________________________________...→ 永远接近但到不了
             /
            /
           /
          /
_________/

JP2499:       _________________ (干净的 peak white plateau)
             /|
            / |
           /  |  ← sharp shoulder
          /   |
_________/    |
              
         toe  knee  shoulder
```

Reinhard 的 highlight 看起来像：
- 灰蒙蒙的（**milky / washed out**）
- 没有 "punch"
- 高光细节被 spread 到一个巨大的 input range 上

JP2499 的 highlight 看起来像：
- 干净地 roll off 到 white
- 有明确的 "highlight edge"
- 更接近 **film print** 的感觉

---

## 问题三：Reinhard 缺乏 Mid-grey Anchoring

### 为什么 mid-grey 很重要

在 cinematography 中，**18% grey**（scene-linear value ~0.18）是 exposure 的 anchor point。一个好的 tonemap 需要保证：

$$T(0.18) = \text{target display mid-grey}$$

对于 SDR display（sRGB），target mid-grey 大约是 code value **0.38**（经过 gamma 后约 display luminance 的 18%）。

### Reinhard 的问题

$$T(0.18) = \frac{0.18}{0.18 + 1} = 0.153$$

这 **太暗了**！Mid-grey 被压低到 0.153，图像整体看起来 under-exposed。

要修正这个问题，你需要 pre-scale input：

$$x' = k \cdot x, \quad \text{find } k \text{ such that } T(k \cdot 0.18) = 0.38$$

$$\frac{0.18k}{0.18k + 1} = 0.38 \implies k = \frac{0.38}{0.18 \cdot (1 - 0.38)} = 3.405$$

但这个 pre-scale 同时改变了 **所有** 其他区域的映射关系，你无法独立控制 mid-grey 映射。

### JP2499 的解决方案

JP2499 通过参数 $s$（semi-saturation constant）直接控制 mid-grey anchoring：

$$T(x) = \frac{x^m}{x^m + s^m}$$

设定 $T(0.18) = 0.38$：

$$\frac{0.18^m}{0.18^m + s^m} = 0.38$$

$$s^m = 0.18^m \cdot \frac{1 - 0.38}{0.38} = 0.18^m \cdot 1.6316$$

$$s = 0.18 \cdot 1.6316^{1/m}$$

对于 $m = 2$: $s = 0.18 \cdot \sqrt{1.6316} = 0.18 \cdot 1.277 = 0.230$

**直觉**：$s$ 就是一个 "exposure knob"，独立于 contrast knob $m$。Reinhard 只有一个形状，JP2499 有两个独立旋钮。

---

## 问题四：Reinhard 在 per-channel 应用时的 Hue Shift

虽然这不完全是曲线本身的问题（更多是 application 方式），但 Reinhard 的 **非线性压缩率随 input 值变化** 这一特性使得 per-channel 应用时 hue shift 更严重。

### 压缩率分析

定义 **compression ratio** $C(x)$：

$$C(x) = \frac{y}{x} = \frac{1}{x + 1} \quad \text{(Reinhard)}$$

$$C(x) = \frac{x^{m-1}}{x^m + s^m} \quad \text{(JP2499)}$$

对于一个 saturated red pixel $\text{RGB} = (10.0, 0.5, 0.1)$：

**Reinhard per-channel:**

| Channel | Input | Output $\frac{x}{x+1}$ | Compression ratio |
|---------|-------|------------------------|-------------------|
| R | 10.0 | 0.909 | 0.0909 |
| G | 0.5 | 0.333 | 0.667 |
| B | 0.1 | 0.091 | 0.909 |

R 被压缩了 ~91%，而 B 只被压缩了 ~9%。**三个 channel 的 ratio 被完全扭曲**，原始 chromaticity $(10/10.6, 0.5/10.6, 0.1/10.6) = (0.943, 0.047, 0.009)$ 变成了 $(0.909/1.333, 0.333/1.333, 0.091/1.333) = (0.682, 0.250, 0.068)$。

红色变成了 **橙色/黄色**。

虽然 JP2499 也有这个问题（**任何** per-channel 非线性操作都会改变 ratio），但：

1. JP2499 更 sharp 的 shoulder 意味着高值 channel 更快地到达 ceiling，ratio distortion 的 transition zone 更窄
2. 更重要的是，**OpenDRT 设计上就不 per-channel 做 tonemap**，而是 norm-based，所以这个问题被 architecture 层面解决了

---

## 问题五：没有 Toe = 没有 Film Look

Film negative + print 的 characteristic curve（Hurter-Driffield curve / H&D curve）天然是一个 **S-curve**：

```
Density
  │        ___________  shoulder
  │       /
  │      /  ← linear region
  │     /
  │    /
  │   / 
  │  _/  ← toe
  │_/
  └──────────────────── Log Exposure
```

Reinhard 没有 toe（在 origin 处 slope = 1，线性通过原点），所以暗部 contrast 是线性的。这导致：

- 暗部看起来 "digital"
- 缺乏 film 那种暗部 "lift" 和 "softness"
- Shadow noise 被 full-contrast 呈现

JP2499 当 $m > 1$ 时自然产生 toe：在 $x=0$ 附近 slope 从 0 开始增长，创造了类似 film 的 shadow rolloff。

---

## 问题六：Inverse 的数值稳定性

在 color pipeline 中经常需要 **inverse tonemap**（从 display-referred 回到 scene-referred）。

**Reinhard inverse:**

$$x = \frac{y}{1 - y}$$

当 $y \to 1$ 时，$x \to \infty$。这在数值上是 **catastrophically unstable**。因为 Reinhard 的 shoulder 太 soft，大量的 high-value scene pixels 被映射到接近 1 的区域，inverse 时 tiny floating point errors 被放大到巨大的值。

**JP2499 inverse:**

$$x = s \cdot \left(\frac{y}{1-y}\right)^{1/m}$$

当 $m > 1$ 时，$1/m < 1$ 的 power 运算起到了 **compression** 作用，减缓了 $y \to 1$ 附近的数值爆炸。

具体对比，当 $y = 0.999$：

| Tonemap | Inverse value |
|---------|--------------|
| Reinhard ($m=1$) | $\frac{0.999}{0.001} = 999$ |
| JP2499 ($m=2, s=0.23$) | $0.23 \cdot \sqrt{999} = 7.27$ |
| JP2499 ($m=3, s=0.23$) | $0.23 \cdot 999^{1/3} = 2.30$ |

$m$ 越大，inverse 越 stable。

---

## 汇总对比表

| 维度 | Reinhard | JP2499 |
|------|----------|--------|
| **参数数量** | 0（或1，如果算 white point 版本） | 2+ ($m$, $s$, 可选更多) |
| **Toe (暗部)** | ❌ 无（线性通过 origin） | ✅ 有（$m>1$ 时自然产生） |
| **Shoulder sharpness** | 极 soft，$O(1/x)$ 收敛 | 可调，$O(1/x^m)$ 收敛 |
| **Mid-grey control** | ❌ 耦合，需外部 pre-scale | ✅ 通过 $s$ 独立控制 |
| **Contrast control** | ❌ 无 | ✅ 通过 $m$ 控制 |
| **Film look** | ❌ 缺乏 S-curve 特征 | ✅ 自然 S-curve |
| **Highlight appearance** | Milky, washed out | Clean, decisive rolloff |
| **Inverse stability** | 差（$y \to 1$ 时爆炸） | 好（power $1/m$ 压缩） |
| **计算成本** | 1 div | 2 pow + 1 div（略贵但可忽略） |

---

## 终极直觉

> **Reinhard 是 tonemap 的 "Hello World"** —— 它证明了 $\frac{x}{x+1}$ 作为 scene-to-display mapping 的可行性，但它只有 **一个自由度**（甚至可以说零个），所有行为都被固定死了。
>
> **JP2499 是 Reinhard 的 "成年版"** —— 它保留了相同的 rational function 血统（Michaelis-Menten / Naka-Rushton 家族），但通过引入 power parameter $m$ 和 semi-saturation $s$，给了你 **独立控制 contrast 和 exposure 的旋钮**，同时获得了 film-like toe、sharp shoulder、和 stable inverse。

---

## 参考链接

- Reinhard 原始论文: https://www.cs.utah.edu/~reinhard/cdrom/tonemap.pdf
- Naka-Rushton equation: https://en.wikipedia.org/wiki/Naka%E2%80%93Rushton_equation
- OpenDRT GitHub: https://github.com/jedypod/open-display-transform
- ACES 2.0 Output Transform 讨论: https://community.acescentral.com/c/aces-development-acesnext/output-transforms-vwg/78
- Jed Smith 的 tonemap 对比: https://community.acescentral.com/t/output-transform-tone-scale/4963
- Troy Sobotka 关于 per-channel tonemap 问题的分析: https://sobotka.github.io/AgX/

---

## 真正的核心问题：Tonemap 曲线本身已经不是瓶颈了

你问 JP2499 比 Reinhard 好在哪，答案表面上是 parametric flexibility、sharper shoulder、有 toe 等等。但**真正的 insight 是**：

> 到了 2022-2024 这个阶段，曲线形状的差异已经是 secondary concern。**真正的战场是 tonemap 应用在什么 signal 上、以及 chrominance 如何跟着走。**

Jed Smith 自己在 ACESCentral 上也说过类似的话——JP2499 的曲线形状换成别的合理的 sigmoid 差别不大，但 **OpenDRT 的 architecture 选择** 才是 game changer。

---

## 所以让我们聊真正深的东西

### 1. Norm 的选择是一个 trade-off space

OpenDRT 对 RGB 算一个 norm $n$，对 $n$ 做 tonemap，然后把 ratio 乘回来：

$$\text{RGB}_{\text{out}} = T(n) \cdot \frac{\text{RGB}_{\text{in}}}{n}$$

但 **norm 的定义不是唯一的**，而且每种选择都有后果：

| Norm 定义 | 公式 | 优点 | 缺点 |
|----------|------|------|------|
| **Max** | $n = \max(R,G,B)$ | 完美保持 hue ratio | 对 complementary colors 不够 smooth |
| **Luminance** | $n = 0.2126R + 0.7152G + 0.0722B$ | Perceptually weighted | Saturated colors 的 luminance 很低，tonemap 后 chroma 爆炸 |
| **Power mean** | $n = \left(\frac{R^p+G^p+B^p}{3}\right)^{1/p}$ | 可调，$p \to \infty$ 趋近 max | 需要选 $p$ |
| **Euclidean** | $n = \sqrt{R^2+G^2+B^2}$ | 几何意义清晰 | 不 perceptual |

**Jed 选了 max norm，为什么？**

因为 max norm 有一个关键性质：

$$\frac{R}{n}, \frac{G}{n}, \frac{B}{n} \in [0, 1]$$

所有 ratio 都天然 bounded 在 $[0,1]$。如果用 luminance norm，对于一个 saturated blue $(0.05, 0.05, 1.0)$：

$$n_{\text{lum}} = 0.2126 \times 0.05 + 0.7152 \times 0.05 + 0.0722 \times 1.0 = 0.1186$$

$$\frac{B}{n_{\text{lum}}} = \frac{1.0}{0.1186} = 8.43$$

Ratio **远大于 1**。Tonemap 后 $T(n_{\text{lum}})$ 是个小值，乘以 8.43，output 就可能超出 display gamut。你需要额外的 gamut clipping / compression。**Max norm 从根本上避免了这个问题**。

这就是第一性原理的力量：选 max norm 不是 heuristic，是因为它在代数上保证了 ratio ∈ [0,1]。

---

### 2. 但 Max Norm 有自己的陷阱

考虑颜色 $(1.0, 1.0, 0.0)$——纯黄色。

$$n = \max(1.0, 1.0, 0.0) = 1.0$$

现在考虑 $(1.0, 0.5, 0.0)$——橙色。

$$n = \max(1.0, 0.5, 0.0) = 1.0$$

两者 norm 相同！但黄色的 "能量" 明显高于橙色（人眼感知亮度更高，因为 $R+G > R + 0.5G$）。所以 max norm 是 **chrominance-blind 的 luminance proxy**——它不关心你有多少个 channel 接近 max。

这导致了一个现象：**等 max-value 但不同 saturation 的颜色被 tonemap 到相同亮度**，然后靠 ratio 区分。在某些 edge case 下（比如 saturated yellow vs desaturated warm highlight），亮度关系可能不完全 perceptual correct。

Jed 的解决方案：在 tonemap 之后加入 **chroma-dependent luminance adjustment**，本质上是一个针对 max norm 缺陷的 patch。

---

### 3. Path-to-White 的数学本质

OpenDRT 的 "path to white" 是说：随着 scene luminance 增加，chrominance 逐渐 decay 到 $(1,1,1)$（achromatic white）。

数学上，这是在做：

$$\text{RGB}_{\text{out}} = T(n) \cdot \left[ w \cdot (1,1,1) + (1-w) \cdot \frac{\text{RGB}_{\text{in}}}{n} \right]$$

其中 $w = w(n)$ 或 $w = w(T(n))$ 是一个随 tonemapped value 增加而从 0 趋向 1 的 weight。

**这其实是在做什么？**

从色彩科学角度，这是在模拟 **Hunt Effect**：

> 随着 adaptation luminance 增加，colorfulness 感知增加——但当 luminance 超过某个阈值（approaching glare），所有颜色 converge to white。

也在模拟 **photoreceptor saturation**：当光强极高时，所有三种 cone 都 saturated，输出趋同。

$$\text{Cone response: } R_{\text{cone}} = \frac{I^n}{I^n + \sigma^n}$$

当 $I \to \infty$，$R_{\text{cone}} \to 1$ 对所有三种 cone 都成立，所以 **颜色感知消失**。

JP2499 的曲线形状和这个 path-to-white 机制是 **两个独立但协同** 的模块：
- JP2499 控制 **luminance channel 的压缩形状**
- Path-to-white 控制 **chrominance channel 的衰减速率**

---

### 4. 一个你可能没想过的问题：Tonemap 的 Surjectivity

Reinhard $y = \frac{x}{x+1}$ 是从 $[0, \infty) \to [0, 1)$——**永远不到 1**。

JP2499 $y = \frac{x^m}{x^m + s^m}$ 也是从 $[0, \infty) \to [0, 1)$——**也永远不到 1**。

但 display 的 peak white **就是 1**。这意味着理论上没有任何 scene value 被映射到 peak white。在实践中 floating point precision 会让它到 1.0，但从数学上：

> **Tonemap 函数的 range 是 open interval $[0, 1)$，不包含 1。**

这在 HDR display 下变得更有趣。如果你有 PQ display (ST 2084)，peak white 是 10,000 nits，而 reference white（mid-grey 所在的 SDR 范围）大约在 100-200 nits。你的 tonemap 不仅要把 scene 压缩到 $[0, 1]$，还要决定 **多少 scene dynamic range 分配给 100-200 nits 以下（SDR 区域），多少分配给 200-10000 nits（HDR headroom）**。

JP2499 的参数化在这里就有优势了：通过调整 $m$ 和 $s$，你可以精确控制 SDR-to-HDR range allocation，而 Reinhard 只能通过 pre-scaling 粗暴地调。

---

### 5. 最后一个 Deep Cut：为什么 ACES 2.0 没有直接采用 JP2499

ACES 2.0 Output Transform 的最终 tonemap 曲线和 JP2499 不完全一样。原因：

1. **ACES 需要 piecewise 定义**来精确 hit 特定 anchor points（black point、mid-grey、peak white），而 closed-form $\frac{x^m}{x^m + s^m}$ 的自由度可能不够同时满足所有 constraints
2. **HDR/SDR 统一**：ACES 2.0 需要一条曲线能通过参数变化同时服务 SDR 100 nits、HDR 1000 nits、HDR 4000 nits 等多种 display，这需要更多的 parametric flexibility
3. **Political / committee dynamics**：ACES 是 committee-driven process，最终结果是多方 compromise

但 **JP2499 的设计哲学**——norm-based、parametric sigmoid、path-to-white——**全部** 被 ACES 2.0 吸收了。曲线可能不同，灵魂是一样的。

---



> JP2499 vs Reinhard 的差异不在于"更好的曲线"，而在于 **JP2499 是一个 design-space-aware 的解决方案**——它知道自己是一个更大系统（gamut mapping + chroma compression + display encoding）中的一个模块，并为此优化了接口（参数化、norm 兼容性、inverse stability）。Reinhard 只是一个孤立的函数。

---

参考：
- https://community.acescentral.com/t/rgb-luminance-and-gamut-mapping/3674
- https://community.acescentral.com/t/new-aces-output-transform/4762
- https://github.com/jedypod/open-display-transform/blob/main/docs/tonescale.md
- https://github.com/ampas/aces2-dev
- CIE Hunt Effect: https://en.wikipedia.org/wiki/Hunt_effect