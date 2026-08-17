---
source_pdf: 深入浅出讲解FOC算法与SVPWM技术.pdf
paper_sha256: 7fd5ebe4f507d37e4cdda45c4b70e100e82f36828ba98c39b6f4be9a7ad40412
processed_at: '2026-08-13T07:08:56-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用"人话"讲讲 FOC 和 SVPWM

好，让我把所有公式和术语都丢一边，咱们就像在白板前面喝咖啡一样，把这事儿从头聊一遍。我尽量用类比和直觉来构建你的 mental model，然后你再去读公式就会觉得 "哦，原来如此"。

---

## 一、电机本质上在干一件事：磁铁追磁铁

无刷电机里头，转子是一块永磁铁，定子是三组线圈（A、B、C）。你给线圈通电，线圈就变成电磁铁，会把转子的永磁铁"吸"过去。

**核心画面**：你手里拿着一块磁铁，在转子周围转圈跑，转子就像小狗一样追着你跑。你跑得快，转子就转得快；你停下来，转子就停下来对准你；你反向跑，转子就反向转。

所以电机控制的问题，本质是：**怎么让"我手里的磁铁"在空间中精确地、平滑地旋转到任何角度，并且幅值大小可控？**

更进一步说，我们想"pixel-level"地控制这个旋转磁场的大小和方向。这就是 FOC 想做的事。

参考资料：[Bose, *Modern Power Electronics and AC Drives*](https://www.wiley.com/en-us/Modern+Power+Electronics+and+AC+Drives-p-9780133875973) — 这本书把"磁场旋转"这个 mental model 讲得最透。

---

## 二、最朴素的玩法：六步换相（航模电调）

最简单的玩法：我有 3 个线圈，两两组合，可以产生 6 个不同方向的磁场。我按顺序切换这 6 个方向，转子就被牵引着转一圈，每个方向停留 60°。

这就是**航模电调（ESC）**干的事，叫**六步换相**（six-step commutation）。

问题来了：
- 只有 6 个方向，转子每转 60° 就要"顿一下"，有 **torque ripple**（力矩抖动）
- 你听到的航模电机那种"嗡嗡"高频噪音就是这么来的
- 低速的时候，6 个状态之间的"跳跃"非常明显，电机抖得厉害
- 换向靠感应反电动势过零点，低速反电动势太小，根本没法换向

所以六步换相只适合**高速 + 不在乎抖动**的场合，比如螺旋桨——螺旋桨本身就是个大飞轮，抖动被惯性平滑掉了，而且你也不在乎它低速性能。

参考：[ST AN1089 - Sensorless BLDC control](https://www.st.com/resource/en/application_note/cd00273858.pdf)

---

## 三、进阶玩法：正弦波驱动（SPWM）

既然方波太糙，那我就用**正弦波**驱动三相线圈，让合成的磁场平滑旋转，这就是 **SPWM**（Sinusoidal PWM）。

intuition 很简单：你看 PWM 把方波"平均"成直流电压的 trick——开 50% 关 50%，低通滤波后就是 0.5 倍电压。把这个 trick 用在时变信号上：你想输出一个正弦波，就让占空比按正弦变化，再让电机本身的电感做低通滤波，电流就是正弦的。三相各差 120°，合成的磁场就是匀速旋转的圆形磁场。

但 SPWM 有两个问题：

1. **电压利用率低**：母线电压 $U_{dc}$ 是固定的，正弦波的峰值顶天了也就 $U_{dc}/2$（每相相对中性点），线电压幅值是 $\sqrt{3} \cdot U_{dc}/2 \approx 0.866 U_{dc}$。理论上你能榨出的最大值是 $U_{dc}$，浪费了 15% 左右。
2. **没法直接控制力矩**：你给的是三相正弦电压，但力矩是电流产生的，电流响应有电感延迟，**电压到力矩是非线性动态系统**，控制器设计很麻烦。

第 2 点是更要命的。你想想，你给 $I_a, I_b, I_c$ 三个时变正弦量做反馈控制，等于在追踪一个**始终在变的参考值**，PI 控制器永远在"追"，永远有相位滞后。这就是为什么需要坐标变换——**把时变量变成常量**。

---

## 四、FOC 的核心魔法：把交流电机"骗"成直流电机

直流电机（有刷）为什么好控制？因为它的磁场结构和换向器保证了一件事：**电枢电流永远垂直于磁场，所以力矩 = 电流 × 常数**，一个标量控制就完事。

交流电机的麻烦在于，磁场在转，电流也在转，两者夹角一直在变，力矩 = $I \cdot B \cdot \sin\theta$，这个 $\theta$ 一直在变，根本没法直接控制。

**FOC 的绝招**：我让坐标系跟着转子一起转，在这个"旋转参考系"里看电流，电流就是常量了，磁场也是常量了，所有东西都变成直流电机的样子。

这就像你站在旋转木马上看旁边的小孩，那个小孩是"静止的"。从地面看他在跑，从木马上看他在原地。**换个参考系，时变量就变常量**。

这种"换参考系简化问题"的思想在物理学里到处都是：
- 拉格朗日力学：换到广义坐标，复杂约束变成独立自由度
- 量子力学：换到本征态基，含时薛定谔方程变成定态
- 信号处理：换到频域，时域卷积变成频域乘法

FOC 干的是同一件事，只不过是在电机控制里。

参考：[Blaschke 1971 原始论文](https://ieeexplore.ieee.org/document/5388269) — FOC 的鼻祖，他当时在 Siemens，把这个叫 "Transvector" 控制。

---

## 五、两次坐标变换的 intuition

### 5.1 Clarke 变换：三相 → 两相，去冗余

三相电流 $I_a, I_b, I_c$ 满足 $I_a + I_b + I_c = 0$（KCL），所以三个里有一个是冗余的。等于说你有三个变量，但实际只有两个自由度。

intuition：三个互差 120° 的轴，在 2D 平面上当然是冗余的，投影到正交的 $\alpha\text{-}\beta$ 轴就行。

```
        a轴 (0°)
         ↑
         |
  c轴 ↖  |  ↗ b轴
 (-120°)|  (120°)
```

投影计算（保持幅值不变的等幅值变换）：

$$
I_\alpha = I_a, \quad I_\beta = \frac{1}{\sqrt{3}}(I_a + 2I_b)
$$

或者矩阵形式：

$$
\begin{bmatrix} I_\alpha \\ I_\beta \end{bmatrix} = \frac{2}{3}\begin{bmatrix} 1 & -1/2 & -1/2 \\ 0 & \sqrt{3}/2 & -\sqrt{3}/2 \end{bmatrix}\begin{bmatrix} I_a \\ I_b \\ I_c \end{bmatrix}
$$

变量含义：
- $I_a, I_b, I_c$ 是三相瞬时电流（A、B、C 三个相线上的电流）
- $I_\alpha, I_\beta$ 是静止两相坐标系下的电流
- 系数 $\frac{2}{3}$ 是 "等幅值变换"，让变换前后电流矢量的幅值不变（如果用 $\sqrt{2/3}$ 那是等功率变换，需要电压电流一起变换）

变换后，电流还是个**旋转的正弦量**（$\alpha, \beta$ 两个正交分量都在正弦变化，相位差 90°），但是从 3 个变量变成 2 个变量了。这只是"减少变量"，还没"线性化"。

类比：相当于把 RGB 图像转成灰度图，去掉一个颜色通道冗余。信息没变，只是更紧凑。

### 5.2 Park 变换：静止 → 旋转，把交流变直流

关键一步来了。现在我有 $I_\alpha, I_\beta$ 两个正弦量在 $\alpha\text{-}\beta$ 平面上以电角速度 $\omega_e$ 旋转。我把这个坐标系**自己旋转 $\theta_e$**（转子当前角度），跟着转子一起跑。

$$
\begin{bmatrix} I_d \\ I_q \end{bmatrix} = \begin{bmatrix} \cos\theta_e & \sin\theta_e \\ -\sin\theta_e & \cos\theta_e \end{bmatrix}\begin{bmatrix} I_\alpha \\ I_\beta \end{bmatrix}
$$

变量：
- $\theta_e$ 是**电角度**（electrical angle），转子磁链相对 A 相的角度
- $I_d$ 是 **直轴电流**（direct-axis），和转子磁链方向一致
- $I_q$ 是 **交轴电流**（quadrature-axis），和转子磁链垂直

直觉：这就相当于把 $\alpha\text{-}\beta$ 平面"旋转" $\theta_e$，让 d 轴对准转子磁链方向。在这个新坐标系里，电流矢量是**静止的常量**，因为坐标系跟着电流矢量同步转。

物理意义：
- $I_d$ 是"和磁链同向"的电流，只影响磁场强度，不产生力矩（对于表贴式 PMSM）
- $I_q$ 是"和磁链垂直"的电流，产生力矩

所以 FOC 的目标就是：**$I_q$ 跟踪期望力矩对应的电流，$I_d$ 控制为 0**。两个 PI 控制器各自控制一个直流量的恒值，简单爆了。

类比：你原本要追一只奔跑的兔子（参考值在变），现在你骑上一匹和兔子同步奔跑的马（换参考系），兔子在你眼里就静止了，你慢慢走过去就行。

参考：[TI FOC App Note](https://www.ti.com/lit/an/bpra073/bpra073.pdf) — TI 这个文档把两次变换的物理意义讲得很清楚。

---

## 六、SVPWM：用 6 个方向拼出任意方向

### 6.1 问题：只有 8 个离散状态可用

三相逆变器 3 个桥臂，每个桥臂上下管互补导通，所以一共 $2^3 = 8$ 种状态：

- 6 个**非零矢量**：$U_1 \sim U_6$，每个对应一种"两相导通"的状态，方向互差 60°
- 2 个**零矢量**：$U_0(000), U_7(111)$，三相短接，不产生电压

但我想要的合成磁场方向是**任意角度**的，怎么办？

intuition：你想想 PWM 是怎么用"开"和"关"两个状态拼出任意平均电压的——通过占空比。SVPWM 是这个 trick 在 2D 平面上的推广：用 6 个固定方向的矢量，通过"时间加权平均"拼出任意方向的矢量。

### 6.2 伏秒平衡原则

核心方程（伏秒平衡）：

$$
\vec{U}_{ref} \cdot T = \vec{U}_x \cdot T_x + \vec{U}_y \cdot T_y + \vec{U}_0 \cdot T_0
$$

变量：
- $\vec{U}_{ref}$ 是我想要的目标电压矢量
- $T$ 是一个 PWM 周期（比如 100μs）
- $\vec{U}_x, \vec{U}_y$ 是目标所在扇区的相邻两个基本矢量（比如第一扇区用 $U_4, U_6$）
- $T_x, T_y$ 是它们各自的作用时间
- $\vec{U}_0$ 是零矢量，$T_0 = T - T_x - T_y$ 是零矢量时间

直觉：等式左边是"我想要的效果"，等式右边是"我用三种状态凑出来的效果"，只要在时间上"平均"对了，输出就等效了。和 PWM 完全同构，只不过 PWM 是 1D 的（开关两态），SVPWM 是 2D 的（六个方向 + 零矢量）。

### 6.3 时间怎么算

以第一扇区为例，目标矢量 $\vec{U}_{ref}$ 与 $U_4$ 方向夹角 $\theta$。用正弦定理：

$$
T_4 = m \cdot T \sin\left(\frac{\pi}{3} - \theta\right), \quad T_6 = m \cdot T \sin\theta
$$

其中调制比：

$$
m = \frac{\sqrt{3} |\vec{U}_{ref}|}{U_{dc}}
$$

变量：
- $m$ 是调制比，$m \le 1$ 在线性调制区，超过 1 就是过调制（会产生谐波）
- $|\vec{U}_{ref}|$ 是目标电压矢量幅值
- $U_{dc}$ 是母线电压

零矢量时间：

$$
T_0 = T_7 = \frac{1}{2}(T - T_4 - T_6)
$$

两个零矢量各占一半，对称配置，这是**七段式 SVPWM**。

### 6.4 切换顺序为什么要那么排

文章里给的切换顺序是：

```
0-4-6-7-7-6-4-0
```

每次切换只改变一个桥臂的状态。如果乱排，可能一次切换多个桥臂，开关损耗大。

intuition：你从 000 切到 110 要改两个桥臂，损耗是改一个桥臂的 2 倍。但如果走 000 → 100 → 110 → 111，每步只改一个，损耗最小。这就叫**最小开关损耗序列**。

而且对称排（4-6-7 在中间，0 在两边）让 PWM 是 center-aligned，谐波谱更干净，集中在载波频率附近，电机容易滤掉。

类比：玩 Rubik's cube，同样从状态 A 到状态 B，转动次数少的解法更优。SVPWM 的七段式就是"最少转动次数"的解。

参考：[Zhou & Wang, *SVPWM Theory and Application*](https://ieeexplore.ieee.org/document/1494985) — SVPWM 的经典综述。

### 6.5 为什么 SVPWM 比 SPWM 利用率高 15%

SPWM 用相电压峰值 $U_{dc}/\sqrt{3}$（线电压峰值 $U_{dc}$），但**只用了正弦波**，电压轨迹是个内切圆，圆的半径是 $\frac{U_{dc}}{\sqrt{3}} \approx 0.577 U_{dc}$（每相）。

SVPWM 不局限于正弦波，可以输出"含 3 次谐波"的波形（马鞍波），电压轨迹是六边形内切圆，半径是 $\frac{U_{dc}}{\sqrt{3}}$ 的 $\frac{2}{\sqrt{3}} \approx 1.155$ 倍。

具体数字：
- SPWM 线电压最大幅值 / $U_{dc}$ = $\frac{\sqrt{3}}{2} \approx 0.866$
- SVPWM 线电压最大幅值 / $U_{dc}$ = $1.0$
- 比值 $1 / 0.866 \approx 1.155$，即高 15.5%

实际意义：同样一块电池，SVPWM 能让电机多输出 15% 的最大力矩，或者同样力矩下转速上限高 15%。这就是为什么 FOC 几乎都用 SVPWM 而不是 SPWM。

---

## 七、三环 PID：层层委托，时间尺度分离

### 7.1 整体结构

```
位置指令 → [位置环 P] → 速度指令 → [速度环 PI] → 力矩指令 → [电流环 PI] → 电压 → SVPWM → 电机
              ↑                          ↑                         ↑
         位置反馈                    速度反馈                  电流反馈
        (编码器)                  (编码器差分)              (Shunt电阻)
```

intuition：每一层都把"上一层给的指令"当作目标，输出"下一层的指令"，最内层输出实际的 MOS 开关信号。

### 7.2 时间尺度分离原则

| 环路 | 典型带宽 | 采样周期 |
|------|---------|---------|
| 电流环 | 1–2 kHz | 50–100 μs |
| 速度环 | 100–200 Hz | 0.5–1 ms |
| 位置环 | 20–50 Hz | 1–5 ms |

**内层比外层快 5–10 倍**。为什么？因为外层看内层，希望内层"瞬时响应"，外层才能用简化模型设计。这是控制论里的 **time-scale separation**（时间尺度分离）原则。

类比：你做战略决策（位置，月级）、战术执行（速度，秒级）、肌肉动作（电流，毫秒级），三层各干各的，互不干扰。你不能让战略层每 10ms 调整一次，那系统永远在抖。

### 7.3 电流环为什么用 PI 不用 D

电流环被控对象是一阶惯性环节（电感电阻回路）：

$$
\frac{I_q(s)}{U_q(s)} = \frac{1}{L_q s + R_s}
$$

变量：
- $L_q$ 是交轴电感
- $R_s$ 是相电阻
- $s$ 是拉普拉斯算子

PI 控制器 $K_p + K_i/s$ 可以零极点对消：让 $K_p/K_i = L_q/R_s$，闭环就简化成一阶：

$$
G_{cl}(s) = \frac{\omega_c}{s + \omega_c}
$$

只剩一个参数 $\omega_c = K_p/L_q$（带宽）。**整定只需要选个带宽**，比如 1500 rad/s 对应电流环响应时间 ~0.7ms。

D 项没用，因为对象是一阶的，加 D 反而会放大电流采样的高频噪声。

### 7.4 速度环为什么也是 PI

速度环被控对象是积分环节（力矩 → 加速度 → 速度）：

$$
\frac{\omega(s)}{T_e(s)} = \frac{1}{Js + B} \approx \frac{1}{Js}
$$

变量：
- $J$ 是转动惯量
- $B$ 是粘性摩擦系数
- $T_e$ 是电磁转矩

PI 控制器加在这个积分环节上，构成 Type-2 系统，对**常值负载扰动稳态误差为零**。这是关键：电机带恒定负载时，速度不能有静差。

整定用**对称最优**（Symmetric Optimum）方法，三个参数（$K_p, K_i, J$）配成一个对称的极点分布，相位裕度 36.8°，动态响应最优。

参考：[Seung-Ki Sul, *Control of Electric Machine Drive Systems*](https://www.wiley.com/en-us/Control+of+Electric+Machine+Drive+Systems-p-9780470590797) — 这本书把三环整定讲得最系统。

---

## 八、几个容易踩坑的地方

### 8.1 死区补偿

MOS 开关不是瞬时的，需要插入死区时间防止上下桥臂直通。但死区会"丢失"一段电压，相当于在 PWM 上叠加了一个畸变。

在电流过零点附近，死区影响最大，会出现"狗骨头"形状的电流波形，导致 5 次、7 次谐波，反映到力矩上是 6 次波动。

解决方法：根据电流极性前馈补偿一个反向死区电压。TI 的 InstaSPIN 有现成的算法，[文档在这里](https://www.ti.com/lit/an/spracb1/spracb1.pdf)。

### 8.2 编码器误差

磁编码器（如 AS5047P）有非线性误差（±1° 左右），会直接进入 Park 变换的角度，导致电流环出现**周期性扰动**。

补偿方法：
- 离线标定 Look-Up Table（LUT）
- 在线自适应观测器（[MIT Cheetah 用这个](https://ieeexplore.ieee.org/document/8793510)）

### 8.3 电流采样时序

center-aligned PWM 的中点是电流"等于平均值"的位置，所以 ADC 要在中点触发。但单 Shunt 方案在零矢量期间母线电流为零，必须避开这些时段，算法复杂度高。

---

## 九、几个延伸的方向

### 9.1 无感 FOC

不用编码器，从电流和电压"反推"转子位置。主流方法：
- **滑模观测器 SMO**：低速不行，因为反电动势太小
- **高频注入 HFI**：利用 IPMSM 凸极效应，零速也能用，Tesla Model 3 就用这个
- **EKF**：精度高但算力贵

### 9.2 弱磁控制

高速时反电动势逼近母线电压，必须注入负 $I_d$ 削弱磁场。电动车高速巡航就是这个原理。

### 9.3 MTPA（最大转矩/电流比）

IPMSM 有凸极效应，注入少量负 $I_d$ 可以利用**磁阻转矩**增加总力矩。电动车效率优化的核心。

### 9.4 模型预测控制 MPC

不用 PI，直接枚举 8 个基本矢量，预测下一拍电流，选 cost 最低的。响应极快但开关频率不固定，谐波谱分散。C2000 和 STM32G4 都开始支持。

### 9.5 数据驱动控制

最近有 RL + FOC 的工作，用神经网络直接输出 $U_{ref}$ 端到端替代 PI 环。MIT 和 ETH 都有 paper，但目前还停留在实验室阶段。

---

## 十、最后一个 mental picture

把所有东西串起来给你一个完整画面：

> 你的机器人关节说"我要 5 N·m 力矩"。位置环说"OK，那要这么快转"，速度环说"OK，那需要这么大的电流"，电流环说"OK，$I_q$ 目标是这个值，$I_d$ 目标是 0"，电流采样说"现在 $I_q$ 是 4.8，$I_d$ 是 0.1"，PI 算"那就把 $U_q$ 加一点，$U_d$ 减一点"，反 Park 算"$U_\alpha, U_\beta$ 是这两个值"，SVPWM 算"这落在第 3 扇区，$T_2$ 这么多，$T_3$ 这么多，$T_0$ 这么多，按 0-2-3-7-7-3-2-0 顺序发"，三相逆变器按这个顺序开关 MOS，电机里产生磁场，转子被吸过去一点，5ms 后再来一遍。

整个过程以 10–20 kHz 的频率循环，每 50–100μs 完成一次闭环。在你眼里，电机就是"平滑转动、精确力控"的黑盒。

这就是 FOC。

---

## 推荐学习路径

1. **入门**：稚晖这篇博客 + [SimpleFOC 文档](https://docs.simplefoc.com/)（5 美元硬件就能跑通整个 pipeline）
2. **进阶**：王成元《现代电机控制技术》 + [ST MCSDK](https://www.st.com/en/embedded-software/x-cube-mcs.html)
3. **深入**：[Sul, *Control of Electric Machine Drive Systems*](https://www.wiley.com/en-us/Control+of+Electric+Machine+Drive+Systems-p-9780470590797)
4. **代码**：[VESC](https://github.com/vedderb/bldc) / [ODrive](https://github.com/madcowswe/ODrive) / [moteus](https://github.com/mjbots/moteus) 三个开源项目读源码
5. **前沿**：IEEE Trans. on Industrial Electronics / Power Electronics 近 5 年 paper

最后强烈推荐动手跑一遍 [SimpleFOC](https://github.com/simplefoc/Arduino-FOC)，用 10 美元买个 BLDC + 编码器 + SimpleFOC board，把整个 pipeline 跑通。读 100 篇 paper 不如自己写一次中断服务函数。

---

# FOC算法与SVPWM技术深度解析

非常好的文章，稚晖的这篇博客是国内FOC入门的标杆作品。我从控制理论、数学intuition、工程实现几个层面深入讲解，并补充一些文章没展开的技术细节。

## 一、把FOC放到电机控制的大图景里

电机控制的发展史大致经历了：

1. **开环控制 / 六步换相 (Six-step commutation)** — 航模ESC用的方式，把BLDC当作"电子换向的有刷电机"，每个60°电角度切换一次相序，输出**梯形波反电动势**对应**方波电流**。优点：算法极简，无需电流环；缺点：**torque ripple大**、低速性能差、噪声大。
2. **正弦波驱动 (SPWM)** — 用PWM合成正弦相电压，平滑度提升。母线电压利用率约 $\frac{\pi}{2\sqrt{3}} \approx 0.789$。
3. **磁场定向控制 (FOC / Vector Control)** — 1980s由Blaschke提出，核心思想是把三相交流量解耦到旋转坐标系下的直流量，使**交流电机可以像直流电机一样被控制**。
4. **直接转矩控制 (DTC)** — 由Depenbrock和Takahashi分别独立提出，跳过电流环直接控制磁链和转矩，有bang-bang特性，响应极快但torque ripple较大。
5. **模型预测控制 (MPC / FCS-MPC)** — 近年热门方向，利用电机模型预测下一拍状态，在有限控制集中优化cost function。ST、TI、Infineon的MCU都开始集成这类算法。

FOC是工业和机器人领域最主流的方案，它的"魔法"在于**通过两次坐标变换把时变正弦量变成时不变直流量**，从而可以使用经典线性控制理论。

参考链接：
- 矢量控制原始论文：Blaschke, F. "The principle of field orientation, as applied to the new transvector closed-loop control system for rotating field machines." *Siemens Review* 34 (1972): 217-220.
- TI FOC App Note: https://www.ti.com/lit/an/bpra073/bpra073.pdf
- ST Motor Control SDK: https://www.st.com/en/embedded-software/x-cube-mcs.html

---

## 二、Clarke变换的几何直觉

### 2.1 三相到两相的"投影降维"

三相绕组在空间上互差120°电角度，其电流矢量为：

$$
\vec{I}_{abc} = I_a \hat{a} + I_b \hat{b} + I_c \hat{c}
$$

其中 $\hat{a}, \hat{b}, \hat{c}$ 是空间中三个互差120°的单位矢量。由于三相绕组对称且中性点隔离时满足 $I_a + I_b + I_c = 0$，所以这个矢量只落在**2D平面**上，冗余一维。

Clarke变换就是把三相轴投影到正交的 $\alpha\text{-}\beta$ 轴上：

$$
\begin{bmatrix} I_\alpha \\ I_\beta \end{bmatrix} = \frac{2}{3} \begin{bmatrix} 1 & -\frac{1}{2} & -\frac{1}{2} \\ 0 & \frac{\sqrt{3}}{2} & -\frac{\sqrt{3}}{2} \end{bmatrix} \begin{bmatrix} I_a \\ I_b \\ I_c \end{bmatrix}
$$

变量说明：
- $I_a, I_b, I_c$ — 三相瞬时电流，**下标** a/b/c 标识相序
- $I_\alpha, I_\beta$ — 静止两相坐标系电流，**下标** α/β 是希腊字母对应空间正交轴
- 系数 $\frac{2}{3}$ 是**等幅值变换**，目的是让变换前后的电流矢量幅值保持一致（另一种常见是 $\sqrt{2/3}$ 系数，那是**等功率变换**，电压电流都需统一缩放）

文章里写的是没带 $\frac{2}{3}$ 的版本，实际工程实现要注意一致性：正反变换必须配套使用同一个系数。

### 2.2 等幅值 vs 等功率变换

| 变换类型 | 系数 | 特点 | 典型场景 |
|---------|------|------|---------|
| 等幅值 (Equal-amplitude) | $\frac{2}{3}$ | 电流矢量幅值在变换前后不变 | 电流采样直接对应相电流峰值 |
| 等功率 (Equal-power) | $\sqrt{\frac{2}{3}}$ | 变换前后功率不变，正交矩阵 | 理论分析、控制器设计 |

在嵌入式代码中，等幅值变换更直观，因为 $I_\alpha = I_a$ 直接对应，方便调试时对比示波器波形。

### 2.3 三相采样电阻数量

文章提到单/双/三Shunt方案。补充一点：

- **3-Shunt**：可同时采三相，无需"重建"第三相，适合高带宽场合，但要求3路同步ADC。
- **2-Shunt**：$I_c = -(I_a + I_b)$，最常用方案。STM32的ADC注入通道可以同步采样2路。
- **1-Shunt**：在DC bus负端串一个电阻，通过PWM不同时序窗口采样重建三相电流。**硬件最省**，但要求算法处理"采样窗口不可用"区段，是低端FOC驱动器（如SimpleFOC）的主流方案。

参考：
- ST AN4272: Three-shunt vs single-shunt current sensing https://www.st.com/resource/en/application_note/an4272.pdf

---

## 三、Park变换的"参考系魔法"

### 3.1 为什么这一步是FOC的灵魂

Park变换的核心是**让坐标系跟随转子一起旋转**。从物理上看，永磁转子产生一个以机械角速度 $\omega_m$ 旋转的磁链矢量 $\vec{\psi}_f$。电角速度 $\omega_e = p \cdot \omega_m$，其中 $p$ 是**极对数 (pole pairs)**。

如果我们在静止坐标系看电流矢量，它以 $\omega_e$ 旋转，是个**正弦量**，控制起来需要追踪时变信号；如果我们把参考系锁在转子上，电流矢量在d-q坐标系下就是**静止的恒值**，这就是直流电机的等效！

Park变换公式：

$$
\begin{bmatrix} I_d \\ I_q \end{bmatrix} = \begin{bmatrix} \cos\theta_e & \sin\theta_e \\ -\sin\theta_e & \cos\theta_e \end{bmatrix} \begin{bmatrix} I_\alpha \\ I_\beta \end{bmatrix}
$$

变量说明：
- $\theta_e$ — **电角度 (electrical angle)**，下标 e 表示电气量，区分于机械角度 $\theta_m = \theta_e / p$
- $I_d$ — **直轴电流 (direct-axis)**，与转子磁链方向一致，影响磁链
- $I_q$ — **交轴电流 (quadrature-axis)**，与转子磁链垂直，产生**转矩**

### 3.2 物理intuition：d-q轴的转矩方程

PMSM的电磁转矩方程在d-q坐标系下为：

$$
T_e = \frac{3}{2} p \left[ \psi_f I_q + (L_d - L_q) I_d I_q \right]
$$

变量说明：
- $T_e$ — 电磁转矩 (N·m)
- $\psi_f$ — 永磁体磁链 (Wb)，常量
- $L_d, L_q$ — 直轴/交轴电感 (H)
- $p$ — 极对数
- 系数 $\frac{3}{2}$ 来自Clarke等幅值变换，表示从两相到三相的功率等效

对于**表贴式 PMSM (SPMSM)**，$L_d = L_q$，第二项为零，转矩完全由 $I_q$ 决定，这就是文章里说"$I_d$ 希望控制为0"的原因。这就是**最大转矩/电流比 (MTPA)** 的trivial情况。

对于**内嵌式 PMSM (IPMSM)**，$L_d < L_q$，第二项产生**磁阻转矩 (reluctance torque)**。此时MTPA要 $I_d < 0$（弱磁），需要更复杂的优化算法。电动车驱动电机几乎都是IPMSM，这是为什么丰田Prius、Tesla Model 3的电机都有凸极比优化。

### 3.3 弱磁控制 (Field Weakening)

当电机转速升高到反电动势接近母线电压时，必须注入负的 $I_d$ 抵消永磁磁链，这就是**弱磁控制**：

$$
U_d = R_s I_d - \omega_e L_q I_q, \quad U_q = R_s I_q + \omega_e L_d I_d + \omega_e \psi_f
$$

电压极限圆：$U_d^2 + U_q^2 \leq U_{max}^2$

高速时 $\omega_e \psi_f$ 主导，必须 $I_d < 0$ 让 $-\omega_e L_d I_d$ 部分抵消反电动势。这是电动车高速巡航的关键技术。

参考：
- 电机控制经典教材：*Modern Power Electronics and AC Drives* by Bose
- 王成元《现代电机控制技术》
- 弱磁控制综述：https://ieeexplore.ieee.org/document/1459136

---

## 四、SVPWM的几何之美

### 4.1 八个基本电压矢量

三相逆变器3个桥臂每个有上/下两种状态，共 $2^3 = 8$ 种组合：

| 开关状态 $(S_a S_b S_c)$ | 电压矢量 | 复数表示 |
|------------------------|---------|---------|
| 000 | $U_0$ | 0 (零矢量) |
| 100 | $U_4$ | $\frac{2}{3}U_{dc}$ |
| 110 | $U_6$ | $\frac{2}{3}U_{dc} e^{j\pi/3}$ |
| 010 | $U_2$ | $\frac{2}{3}U_{dc} e^{j2\pi/3}$ |
| 011 | $U_3$ | $\frac{2}{3}U_{dc} e^{j\pi}$ |
| 001 | $U_1$ | $\frac{2}{3}U_{dc} e^{j4\pi/3}$ |
| 101 | $U_5$ | $\frac{2}{3}U_{dc} e^{j5\pi/3}$ |
| 111 | $U_7$ | 0 (零矢量) |

这6个非零矢量构成正六边形，**把空间分成6个扇区 (sector)**。$U_{dc}$ 是DC母线电压。

### 4.2 伏秒平衡原则 — SVPWM的核心intuition

SVPWM的intuition非常类似**信号处理中的向量分解**。我们想合成任意方向的 $\vec{U}_{ref}$，但只能"快速切换"6个离散矢量，类似**PWM用方波模拟正弦波**的思想在2D空间的推广。

伏秒平衡方程：

$$
\vec{U}_{ref} \cdot T = \vec{U}_x \cdot T_x + \vec{U}_y \cdot T_y + \vec{U}_0 \cdot T_0
$$

变量说明：
- $T$ — 一个PWM周期
- $T_x, T_y$ — 相邻两个非零矢量的作用时间
- $T_0$ — 零矢量作用时间，$T_0 = T - T_x - T_y$
- $\vec{U}_x, \vec{U}_y$ — 当前扇区相邻的两个基本电压矢量

以第一扇区为例，使用 $\vec{U}_4, \vec{U}_6$，由正弦定理：

$$
\frac{|U_{ref}|}{\sin\frac{2\pi}{3}} = \frac{T_6 \cdot |U_6| / T}{\sin\theta} = \frac{T_4 \cdot |U_4| / T}{\sin(\frac{\pi}{3} - \theta)}
$$

代入 $|U_4| = |U_6| = \frac{2}{3}U_{dc}$：

$$
T_4 = m \cdot T \sin\left(\frac{\pi}{3} - \theta\right), \quad T_6 = m \cdot T \sin\theta
$$

其中调制比：

$$
m = \frac{\sqrt{3} |U_{ref}|}{U_{dc}}
$$

线性调制区 $m \leq 1$，对应**最大线电压 $|U_{ref}|_{max} = \frac{U_{dc}}{\sqrt{3}}$**，电压利用率约 0.866 / 0.789 ≈ 1.155，即文章说的**比SPWM高15.5%**。

### 4.3 七段式 vs 五段式SVPWM

文章提到七段式 (S7-segment SVPWM)：

```
0-4-6-7-7-6-4-0
```

每个周期**对称分布**零矢量，使输出PWM是**center-aligned**，谐波最小，开关次数最少（每个周期7次切换）。

还有五段式：

```
4-6-7-6-4
```

只用一个零矢量，开关次数减半，谐波稍大但损耗低。

工业上还有：
- **DPWM (Discontinuous PWM)** — 在60°范围内钳位到某个零矢量，开关损耗降低33%，适合高频/高压场合
- **SHE-PWM (Selective Harmonic Elimination)** — 离线计算开关角，消除特定低次谐波，适合大功率

### 4.4 死区效应 (Dead-time Effect)

文章轻描淡写提到"死区问题"，实际工程中这是**FOC驱动器最难调的痛点之一**。

MOSFET/IGBT的开关不是瞬时的，存在 $t_{on}, t_{off}, t_d$（延迟）。为防止上下桥臂直通短路，必须插入死区时间 $t_{dead}$（典型 200ns ~ 1μs）。

死区会引入**电压畸变**，相当于在每个开关周期内"丢失"一段电压：

$$
\Delta U = \frac{t_{dead}}{T_{PWM}} \cdot U_{dc} \cdot \text{sign}(I)
$$

这导致：
1. **电流波形在过零点附近畸变**（"狗骨头"波形）
2. **6次谐波**主导，引起torque ripple
3. 低速时 $U_{ref}$ 很小，相对畸变更大

补偿方法：
- **基于电流极性的前馈补偿**（最常用）
- **基于电流重构的反馈补偿**
- **PWM载波移相 / 死区抖动**

参考资料：
- 死区补偿综述：https://ieeexplore.ieee.org/document/1621322
- TI InstaSPIN 自带dead-time compensation: https://www.ti.com/tool/MOTORWARE

---

## 五、PID控制环路的级联设计

### 5.1 三环级联的intuition

文章讲的三环结构：**位置环 → 速度环 → 电流环**，由外向内，**内环带宽约为外环的5-10倍**：

| 环路 | 典型带宽 | 采样周期 | 控制器 |
|------|---------|---------|--------|
| 电流环 | 1-2 kHz | 50-100 μs | PI |
| 速度环 | 100-200 Hz | 0.5-1 ms | PI |
| 位置环 | 20-50 Hz | 1-5 ms | P / PD / PID |

**intuition**: 内环必须远快于外环，这样从外环看内环相当于"瞬时响应"，才能用简化模型设计外环。这就是**时间尺度分离 (time-scale separation)** 思想，在机器人学中也是控制arm、leg的通用模式。

### 5.2 电流环传函分析（文章没展开的细节）

PMSM在d-q坐标系下的电气方程：

$$
\begin{cases}
U_d = R_s I_d + L_d \frac{dI_d}{dt} - \omega_e L_q I_q \\
U_q = R_s I_q + L_q \frac{dI_q}{dt} + \omega_e L_d I_d + \omega_e \psi_f
\end{cases}
$$

对q轴忽略交叉耦合（或用前馈解耦），传递函数：

$$
G_{I_q}(s) = \frac{I_q(s)}{U_q(s)} = \frac{1}{L_q s + R_s}
$$

这是一阶惯性环节，时间常数 $\tau = L_q / R_s$（典型几个ms）。

PI控制器：$C(s) = K_p + K_i/s$

闭环传函：

$$
G_{cl}(s) = \frac{K_p s + K_i}{L_q s^2 + (R_s + K_p) s + K_i}
$$

用**零极点对消**：令 $K_p / K_i = L_q / R_s$，则分子分母有公因子 $(s + R_s/L_q)$，闭环简化为：

$$
G_{cl}(s) = \frac{K_p / L_q}{s + K_p / L_q} = \frac{\omega_c}{s + \omega_c}
$$

其中 $\omega_c = K_p / L_q$ 是**电流环带宽**。这就是文章里说的"通过零极点对消简化，只控制一个参数即电流带宽"。

工程上 $\omega_c$ 取 1000-2000 rad/s，对应电流环响应时间 0.5-1ms。

### 5.3 速度环的PI整定

机械方程：

$$
J \frac{d\omega_m}{dt} = T_e - B\omega_m - T_L
$$

变量：
- $J$ — 转动惯量 (kg·m²)
- $B$ — 粘性摩擦系数
- $T_L$ — 负载转矩

如果内层电流环带宽足够高，近似为 $T_e \approx K_t I_q^{ref}$（$K_t$ 是转矩常数），速度环被控对象近似为积分环节 $\frac{1}{Js}$。

PI速度环：$C_\omega(s) = K_{p\omega} + K_{i\omega}/s$

开环传函：$L(s) = \frac{K_{p\omega}s + K_{i\omega}}{Js^2}$，这是**Type-2系统**，对常值负载扰动**稳态误差为零**。

整定方法常用**对称最优 (Symmetric Optimum)**：

$$
K_{p\omega} = \frac{J}{3 \omega_c \tau_{eq}^2}, \quad K_{i\omega} = \frac{K_{p\omega}}{3\tau_{eq}}
$$

其中 $\tau_{eq}$ 是内环等效时间常数。

参考：*Control of Electric Machine Drive Systems* by Seung-Ki Sul

---

## 六、无感FOC (Sensorless FOC)

文章末尾提到"无感控制是另一个话题"，我展开一下。

### 6.1 主要技术路线

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **滑模观测器 (SMO)** | 利用反电动势模型构造滑模面 | 实现简单，低速性能差 |
| **扩展卡尔曼滤波 (EKF)** | 把电机当作非线性系统状态估计 | 精度高但计算量大 |
| **Luenberger观测器** | 线性状态观测器 | 介于SMO和EKF之间 |
| **高频注入 (HFI)** | 利用凸极效应注入高频电压 | **零速可用**，仅适合IPMSM |
| **磁链积分法** | 直接积分反电动势估算磁链 | 简单但有积分漂移 |

### 6.2 SMO观测器

定义电流观测误差：$\hat{I}_\alpha - I_\alpha, \hat{I}_\beta - I_\beta$。

观测器动态：

$$
\frac{d\hat{I}_\alpha}{dt} = -\frac{R_s}{L_s}\hat{I}_\alpha + \frac{1}{L_s}(U_\alpha - \hat{E}_\alpha) + k \cdot \text{sgn}(\hat{I}_\alpha - I_\alpha)
$$

滑模面 $s = \hat{I}_\alpha - I_\alpha$，使 $s \dot{s} < 0$ 即可保证收敛。$\hat{E}_\alpha$ 通过LPF得到反电动势，再由 $\theta_e = \text{atan2}(-\hat{E}_\alpha, \hat{E}_\beta)$ 算出转子角度。

低速时反电动势小，信噪比低，**SMO在低速不可用**，需要配合HFI。

### 6.3 高频注入 (HFI)

利用IPMSM的凸极性 ($L_d \neq L_q$)，注入高频电压（如1-2kHz，远高于基波频率），高频电流响应中包含转子位置信息：

$$
I_{hd} \approx \frac{U_{inj}}{\omega_h L_d}, \quad I_{hq} \approx \frac{U_{inj}}{\omega_h L_q}
$$

估计误差 $\Delta\theta$ 会让高频电流出现特定包络，通过解调器提取。

**HFI是唯一能在零速获得转子位置的无感方法**，特斯拉Model 3的IPMSM驱动器就用了HFI + SMO的混合方案。

参考：
- 无感FOC综述：https://ieeexplore.ieee.org/document/1193664
- TI InstaSPIN-FOC 技术文档: https://www.ti.com/lit/an/spracb1/spracb1.pdf

---

## 七、现代扩展与前沿方向

### 7.1 模型预测控制 (MPC)

FCS-MPC (Finite Control Set MPC) 不用SVPWM，直接在8个基本矢量中**枚举选择**最优点：

$$
J = \sum_{k=1}^{N} \left\| I_q^{ref} - I_q(k+1|k) \right\|^2 + \lambda \left\| I_d(k+1|k) \right\|^2
$$

每个PWM周期枚举8个矢量，预测下一拍电流，选择cost最小的。**不需要PI整定**，动态响应极快，缺点是**开关频率不固定**，谐波谱分散。

### 7.2 自抗扰控制 (ADRC)

韩京清提出，用扩展状态观测器 (ESO) 估计**总扰动**（含参数变化、负载突变、外部扰动），前馈补偿。不依赖精确模型，鲁棒性强。

$$
\text{ESO}: \begin{cases} \dot{z}_1 = z_2 - \beta_1 z_1 + b_0 u \\ \dot{z}_2 = -\beta_2 z_1 \end{cases}
$$

$z_1$ 估计状态，$z_2$ 估计总扰动。ADRC在国产伺服（汇川、雷赛）开始应用。

### 7.3 数据驱动与神经网络控制

最近强化学习 + FOC的工作开始出现，比如用PPO训练一个policy直接输出 $U_{ref}$，端到端替代PID环。MIT和ETH都有相关paper。

- NeurIPS 2022: "Data-driven Control of PMSM using DNN"
- Berkeley: "Reinforcement Learning for Torque Control of BLDC"

### 7.4 硬件平台

主流FOC MCU：

| 厂商 | 型号 | 特点 |
|------|------|------|
| ST | STM32G4/H7 | Cortex-M4F/M7，CORDIC硬件加速三角函数，3路ADC |
| TI | TMS320F280049C | C2000系列，CLA协处理器，Piccolo架构 |
| Infineon | XMC4700 | ARM Cortex-M4，专用MATH单元 |
| NXP | S32K144 | Cortex-M4，集成电机控制库 |
| Espressif | ESP32-S3 | 双核，便宜，社区活跃 (SimpleFOC用) |

稚晖做的驱动器（看照片）应该用STM32G4系列，是当前DIY FOC的事实标准。

### 7.5 开源FOC生态

- **SimpleFOC** (https://github.com/simplefoc/Arduino-FOC) — 最易上手的开源FOC库，支持Arduino/ESP32/STM32
- **VESC** (https://github.com/vedderb/bldc) — Benjamin Vedder开发，电动车/电滑板主流开源controller
- **ODrive** (https://github.com/madcowswe/ODrive) — 机器人关节驱动器，支持CAN总线
- **moteus** (https://github.com/mjbots/moteus) — 仿生机器人专用，性能极强
- **ST MCSDK** — ST官方电机控制库，工业级
- **TI MotorWare** — TI官方参考实现

稚晖的驱动器跟moteus定位接近，都是追求"超小体积+高性能"的机器人关节方案。

---

## 八、工程实践中的"坑"

### 8.1 电流采样时序

PWM center-aligned时，电流在PWM周期中点采样最准确（瞬时电流=平均电流）。但**单Shunt方案**在零矢量期间母线电流为零，必须**在特定窗口采样**，复杂度高。

### 8.2 编码器误差补偿

磁编码器（如AS5047P）有非线性误差（约±1°），直接用会让电流环出现**周期性扰动**。补偿方法：
- 离线标定LUT (Look-Up Table)
- 在线自适应观测器
- Adept/MIT的"encoder self-calibration"算法

### 8.3 相序与极对数辨识

上电第一次启动电机，必须自动识别：
- **相序**：通过注入小电压看编码器方向
- **极对数**：施加阶跃电压，数编码器跳变次数
- **相电感/电阻**：施加阶跃电流看时间常数

这是TI InstaSPIN的"FAST"算法核心。

### 8.4 电流环延迟

数字控制系统的总延迟：

$$
T_{delay} = T_{ADC} + T_{calc} + T_{PWM\_update} + T_{dead}
$$

典型值 50-100μs，对电流环带宽有上限约束：

$$
\omega_c \leq \frac{1}{5 T_{delay}} \approx 2000-4000 \text{ rad/s}
$$

超过这个带宽会有相位裕度问题。这是为什么电流环很少超过3kHz的根本原因。

---

## 九、结语：FOC的本质是"解耦"

文章最后一段总结得好："所谓矢量控制其实就是在做解耦"。

更精确地说，FOC是三个层次的解耦：
1. **空间解耦**：Clarke变换把3个耦合的相电流降维到2个独立轴
2. **时间解耦**：Park变换把时变正弦量转为时不变直流量
3. **物理解耦**：d-q轴分别控制磁链和转矩（即"励磁"和"力矩"两个物理量）

这与信号处理中FFT的思想完全同构：把"难处理的时域信号"变换到"易处理的域"处理后再反变换回去。

学习FOC的intuition建议：
1. **画相量图**：每种工况下画 $\vec{U}_{ref}, \vec{I}_{ref}, \vec{\psi}$ 的相量关系
2. **手算一遍SVPWM**：自己推导扇区1的 $T_4, T_6$ 公式
3. **看波形对照**：用示波器看 $I_a, I_\alpha, I_q$ 的实际波形，对照理论
4. **跑一次SimpleFOC**：用5美元的硬件把整个pipeline跑通，比读100篇paper强

### 推荐学习路径

1. 入门：稚晖这篇博客 + SimpleFOC文档
2. 进阶：王成元《现代电机控制技术》 + TI/ST App Notes
3. 深入：Seung-Ki Sul《Control of Electric Machine Drive Systems》
4. 前沿：IEEE Trans. on Industrial Electronics / Power Electronics 近5年paper
5. 代码：ODrive / VESC / moteus 源码

### 关键参考资料链接

- 稚晖原文 (知乎): https://zhuanlan.zhihu.com/p/147659820
- SimpleFOC项目: https://docs.simplefoc.com/
- VESC开源项目: https://github.com/vedderb/bldc
- ODrive开源项目: https://github.com/madcowswe/ODrive
- moteus项目: https://github.com/mjbots/moteus
- ST MCSDK: https://www.st.com/en/embedded-software/x-cube-mcs.html
- TI MotorWare: https://www.ti.com/tool/MOTORWARE
- ST AN1078 (无感FOC): https://www.st.com/resource/en/application_note/an1078.pdf
- TI InstaSPIN User Guide: https://www.ti.com/lit/an/spraq91/spraq91.pdf
- IEEE FOC综述: https://ieeexplore.ieee.org/document/1193664
- Bose, *Modern Power Electronics and AC Drives* (书)
- Sul, *Control of Electric Machine Drive Systems* (书)

如果你对其中某个部分（比如SVPWM七段式实现细节、无感观测器、弱磁控制）感兴趣，可以继续展开讨论。
